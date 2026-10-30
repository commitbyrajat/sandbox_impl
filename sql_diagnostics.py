"""Schema-aware SQLGlot preflight and structured, domain-agnostic repair feedback."""

import hashlib
from collections.abc import Sequence
from typing import Any, Literal

import pyarrow as pa
import sqlglot
from datafusion import SessionContext, SQLOptions
from pydantic import BaseModel, Field
from sqlglot import exp
from sqlglot.optimizer.annotate_types import annotate_types
from sqlglot.optimizer.normalize_identifiers import normalize_identifiers
from sqlglot.optimizer.qualify import qualify
from sqlglot.optimizer.scope import Scope, traverse_scope

from sql_joins import JoinBinding, review_joins, validate_join_metadata

DIALECT = "postgres"  # Parsing conventions already used by DataSources, not a DataFusion dialect.


class SQLIssue(BaseModel):
    kind: str
    scope: str
    reference: str
    source_alias: str | None = None
    available_columns: list[str] = Field(default_factory=list)
    repair: str


class SQLDiagnostic(BaseModel):
    tool: Literal["run_sql"] = "run_sql"
    phase: str
    error: str
    failed_sql: str
    fingerprint: str
    issues: list[SQLIssue] = Field(default_factory=list)
    scope_inputs: list[dict[str, Any]] = Field(default_factory=list)
    referenced_base_column_types: dict[str, dict[str, str]] = Field(
        default_factory=dict
    )
    repair_instructions: list[str] = Field(default_factory=list)
    cacheable: bool = False
    repeated_failed_statement: bool = False
    attempt: int = 1
    join_context: list[JoinBinding] = Field(default_factory=list)


class SQLCheck(BaseModel):
    sqlglot_validated: bool = True
    datafusion_compiled: bool = True
    result_schema: list[dict[str, Any]]
    inferred_types: list[dict[str, str]] = Field(default_factory=list)
    warnings: list[str | dict[str, Any]] = Field(default_factory=list)
    reviewed_joins: list[dict[str, Any]] = Field(default_factory=list)
    all_join_keys_resolved: bool = True


class SQLPreflightError(ValueError):
    def __init__(self, diagnostic: SQLDiagnostic):
        self.diagnostic = diagnostic
        super().__init__(
            f"SQL preflight failed ({diagnostic.phase}): {diagnostic.error}"
        )


def sql_fingerprint(statement: str) -> str:
    """Ignore comments/formatting/unquoted case, preserving literals and quoted names."""
    try:
        if len(statement) > 20_000:
            raise ValueError("Oversized SQL")
        trees = sqlglot.parse(statement, read=DIALECT)
        normalized = "\0".join(
            tree.sql(dialect=DIALECT, normalize=True, comments=False)
            for tree in trees
            if tree is not None
        )
    except (sqlglot.errors.SqlglotError, ValueError, RecursionError):
        normalized = statement.strip()
    return hashlib.sha256(normalized.encode()).hexdigest()


def arrow_sql_type(kind: pa.DataType) -> str:
    """Translate schema types for SQLGlot analysis; DataFusion owns result precision."""
    if pa.types.is_dictionary(kind):
        return arrow_sql_type(kind.value_type)
    if pa.types.is_decimal(kind):
        return f"DECIMAL({kind.precision},{kind.scale})"
    if pa.types.is_integer(kind):
        return "BIGINT"
    if pa.types.is_floating(kind):
        return "DOUBLE"
    if pa.types.is_boolean(kind):
        return "BOOLEAN"
    if pa.types.is_date(kind):
        return "DATE"
    if pa.types.is_timestamp(kind):
        return "TIMESTAMPTZ" if kind.tz else "TIMESTAMP"
    if pa.types.is_time(kind):
        return "TIME"
    if (
        pa.types.is_string(kind)
        or pa.types.is_large_string(kind)
        or pa.types.is_string_view(kind)
    ):
        return "TEXT"
    if (
        pa.types.is_binary(kind)
        or pa.types.is_large_binary(kind)
        or pa.types.is_binary_view(kind)
    ):
        return "BINARY"
    if (
        pa.types.is_list(kind)
        or pa.types.is_large_list(kind)
        or pa.types.is_fixed_size_list(kind)
    ):
        return f"ARRAY<{arrow_sql_type(kind.value_type)}>"
    if pa.types.is_struct(kind):
        return (
            "STRUCT<"
            + ",".join(
                f"{exp.to_identifier(field.name, quoted=True).sql()} {arrow_sql_type(field.type)}"
                for field in kind
            )
            + ">"
        )
    return "UNKNOWN"


def _columns(source, schemas, seen=None):
    if isinstance(source, exp.Table):
        schema = schemas.get(source.name)
        return list(schema.names) if schema is not None else None
    if not isinstance(source, Scope):
        return None
    seen = set() if seen is None else seen
    if id(source) in seen:
        return None
    seen = seen | {id(source)}
    if source.outer_columns:
        return list(source.outer_columns)
    names = []
    for projection in source.expression.selects:
        if projection.is_star:
            inputs = source.selected_sources
            if isinstance(projection, exp.Column) and projection.table:
                inputs = {
                    name: value
                    for name, value in inputs.items()
                    if name == projection.table
                }
            for _, nested in inputs.values():
                expanded = _columns(nested, schemas, seen)
                if expanded is None:
                    return None
                names.extend(expanded)
        else:
            names.append(projection.alias_or_name)
    return names


def scope_details(statement: str, schemas: dict[str, pa.Schema]):
    """Explain missing projections across scopes, without rewriting the candidate."""
    tree = normalize_identifiers(
        sqlglot.parse_one(statement, read=DIALECT), dialect=DIALECT
    )
    issues, inputs, used_tables = [], [], set()
    for index, scope in enumerate(traverse_scope(tree)):
        parent = scope.expression.parent
        scope_name = (
            parent.alias_or_name
            if isinstance(parent, (exp.CTE, exp.Subquery))
            else f"query_{index + 1}"
        )
        selected = scope.selected_sources
        sources = {}
        for alias, (_, source) in selected.items():
            columns = _columns(source, schemas)
            sources[alias] = columns
            if isinstance(source, exp.Table):
                used_tables.add(source.name)
            inputs.append(
                {
                    "scope": scope_name,
                    "alias": alias,
                    "source_kind": "derived" if isinstance(source, Scope) else "base",
                    "available_columns": columns,
                }
            )
        for column in scope.columns:
            if column.is_star:
                continue
            alias, name = column.table, column.name
            source = selected.get(alias)
            if (
                alias
                and source is not None
                and sources[alias] is not None
                and name not in sources[alias]
            ):
                derived = isinstance(source[1], Scope)
                issues.append(
                    SQLIssue(
                        kind="missing_cte_projection" if derived else "unknown_column",
                        scope=scope_name,
                        reference=column.sql(),
                        source_alias=alias,
                        available_columns=sources[alias],
                        repair=(
                            "Carry the required field through the source CTE/subquery, or join back to the base fact using complete keys at the required grain."
                            if derived
                            else "Use an actual column from this base table; inspect its schema instead of inventing an identifier."
                        ),
                    )
                )
            elif alias and source is None and not scope.is_correlated_subquery:
                issues.append(
                    SQLIssue(
                        kind="unbound_source_alias",
                        scope=scope_name,
                        reference=column.sql(),
                        source_alias=alias,
                        repair="Bind this source in FROM/JOIN, or use an alias visible in this scope. Declaring a CTE does not join it into every SELECT.",
                    )
                )
            elif not alias and not scope.is_correlated_subquery:
                available = sorted(
                    {
                        field
                        for fields in sources.values()
                        if fields is not None
                        for field in fields
                    }
                )
                local_aliases = {
                    projection.alias
                    for projection in scope.expression.selects
                    if isinstance(projection, exp.Alias)
                }
                if (
                    sources
                    and all(fields is not None for fields in sources.values())
                    and name not in available
                    and name not in local_aliases
                ):
                    issues.append(
                        SQLIssue(
                            kind="unbound_column",
                            scope=scope_name,
                            reference=column.sql(),
                            available_columns=available,
                            repair="No input in this SELECT scope exposes this field. Use/alias an available source column, carry the field through the source CTE, or explicitly join its CTE into this scope. Declaring a lookup CTE does not make its columns globally available. Do not change unrelated filters to fix a missing projection.",
                        )
                    )
    referenced = {column.name for column in tree.find_all(exp.Column)}
    types = {
        name: {
            field.name: str(field.type)
            for field in schemas[name]
            if field.name in referenced
        }
        for name in sorted(used_tables)
        if name in schemas
    }
    unique = {(item.scope, item.reference, item.kind): item for item in issues}
    return list(unique.values()), inputs, types


def sql_diagnostic(
    statement: str,
    error: str,
    schemas: dict[str, pa.Schema],
    *,
    phase: str,
    cacheable: bool = False,
    bindings: Sequence[JoinBinding] = (),
) -> SQLDiagnostic:
    notes = [
        (
            "Revise the failed SQL and call run_sql again for the same pending step. Do not resubmit unchanged SQL."
            if cacheable
            else "This failure is not cached. Retry unchanged SQL only for a transient execution/infrastructure failure; otherwise correct the reported SQL defect."
        ),
        "Use only the owned workspace schema. Every CTE exposes only its projections; preserve all downstream measures and join keys.",
        "Keep the required observation grain. A latest-key/date lookup must be joined back using the complete keys to retrieve fact measures.",
        "Do not reference sibling SELECT aliases; move dependent expressions into the next CTE. Group non-aggregate projections correctly.",
        "SQLGlot analysis is advisory about types and relationships. DataFusion result_schema is authoritative; compilation does not prove business correctness.",
    ]
    if "Decimal" in error or "overflow" in error.lower():
        notes.append(
            "Use deliberate Decimal precision/scale. Cast compared expressions to compatible types and bound division results before subsequent products; final formatting cannot recover lost precision."
        )
    try:
        issues, inputs, types = scope_details(statement, schemas)
    except Exception:
        issues, inputs, types = [], [], {}
    if phase == "qualification":
        try:
            tree = normalize_identifiers(
                sqlglot.parse_one(statement, read=DIALECT), dialect=DIALECT
            )
            join_issues, _, _ = review_joins(tree, bindings)
            issues.extend(SQLIssue.model_validate(issue) for issue in join_issues)
        except Exception:
            pass  # Additional join analysis must not hide the original scope error.
    return SQLDiagnostic(
        phase=phase,
        error=error[:4000],
        failed_sql=statement,
        fingerprint=sql_fingerprint(statement),
        issues=issues,
        scope_inputs=inputs,
        referenced_base_column_types=types,
        repair_instructions=notes,
        cacheable=cacheable,
        join_context=[
            key
            for key in bindings
            if key.left_table in types and key.right_table in types
        ],
    )


def preflight_sql(
    statement: str,
    schemas: dict[str, pa.Schema],
    context: SessionContext,
    bindings: Sequence[JoinBinding] = (),
) -> SQLCheck:
    """Validate a copy in SQLGlot, then compile original SQL without fetching rows."""
    phase = "parse"
    try:
        if len(statement) > 20_000:
            raise ValueError("SQL must be at most 20,000 characters")
        trees = sqlglot.parse(statement, read=DIALECT)
        if len(trees) != 1 or not isinstance(trees[0], exp.Query):
            raise ValueError("Only one SELECT query (optionally with CTEs) is allowed")
        tree = normalize_identifiers(trees[0].copy(), dialect=DIALECT)
        phase = "scope"
        for scope in traverse_scope(tree):
            for source in scope.sources.values():
                if isinstance(source, exp.Table) and (
                    not isinstance(source.this, exp.Identifier)
                    or source.db
                    or source.catalog
                    or source.name not in schemas
                ):
                    raise ValueError(
                        "Query only tables belonging to the specified workspace"
                    )
        phase = "qualification"
        typed_schema = {
            table: {field.name: arrow_sql_type(field.type) for field in schema}
            for table, schema in schemas.items()
        }
        qualified = qualify(
            tree.copy(),
            dialect=DIALECT,
            schema=typed_schema,
            expand_alias_refs=False,
            validate_qualify_columns=True,
        )
    except (sqlglot.errors.SqlglotError, ValueError, RecursionError) as error:
        raise SQLPreflightError(
            sql_diagnostic(
                statement,
                str(error),
                schemas,
                phase=phase,
                cacheable=True,
                bindings=bindings,
            )
        ) from error

    try:
        validate_join_metadata(bindings, schemas)
    except ValueError as error:
        raise SQLPreflightError(
            sql_diagnostic(
                statement, str(error), schemas, phase="configuration", bindings=bindings
            )
        ) from error
    join_issues, reviewed, join_warnings = review_joins(qualified, bindings)
    if join_issues:
        diagnostic = sql_diagnostic(
            statement,
            "Join keys do not satisfy configured relationships",
            schemas,
            phase="join_keys",
            cacheable=True,
            bindings=bindings,
        )
        diagnostic.issues.extend(
            SQLIssue.model_validate(issue) for issue in join_issues
        )
        raise SQLPreflightError(diagnostic)

    warnings, inferred = list(join_warnings), []
    try:
        annotated = annotate_types(qualified, schema=typed_schema, dialect=DIALECT)
        inferred = [
            {
                "name": projection.alias_or_name,
                "type": projection.type.sql() if projection.type else "UNKNOWN",
            }
            for projection in annotated.selects
        ]
    except Exception as error:
        warnings.append(
            f"SQLGlot type inference unavailable ({type(error).__name__}); use the DataFusion result schema."
        )

    try:
        options = (
            SQLOptions()
            .with_allow_ddl(False)
            .with_allow_dml(False)
            .with_allow_statements(False)
        )
        result_schema = context.sql(statement, options=options).schema()
    except Exception as error:
        message = str(error)
        deterministic = any(
            marker in message.lower()
            for marker in (
                "schema error",
                "error during planning",
                "sql error",
                "parser error",
                "not implemented",
            )
        )
        raise SQLPreflightError(
            sql_diagnostic(
                statement,
                message,
                schemas,
                phase="datafusion_compile",
                cacheable=deterministic,
                bindings=bindings,
            )
        ) from error
    return SQLCheck(
        result_schema=[
            {
                "name": field.name,
                "arrow_type": str(field.type),
                "nullable": field.nullable,
            }
            for field in result_schema
        ],
        inferred_types=inferred,
        warnings=warnings,
        reviewed_joins=reviewed,
        all_join_keys_resolved=not any(
            isinstance(warning, dict)
            and warning.get("kind", "").startswith("unverified")
            for warning in warnings
        ),
    )
