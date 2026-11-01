"""Split read-only SQL at relational boundaries and materialize typed stages.

The compiler owns expressions and joins. This layer only substitutes complete,
uncorrelated relations, never fragments predicates or changes aggregation grain.
"""

from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import json
import os
from time import monotonic
from uuid import uuid4

import pyarrow as pa
from datafusion import SQLOptions
from pydantic import BaseModel, Field
import sqlglot
from sqlglot import exp
from sqlglot.optimizer.scope import traverse_scope

from runtime import trace
from sql_diagnostics import preflight_sql, SQLPreflightError


class SQLStagingSettings(BaseModel):
    enabled: bool = True
    max_chars: int = Field(default=6000, ge=100, le=20000)
    max_selects: int = Field(default=6, ge=1)
    max_joins: int = Field(default=3, ge=1)
    max_stages: int = Field(default=64, ge=2, le=256)
    max_rows: int = Field(default=100000, ge=1)
    max_bytes: int = Field(default=64000000, ge=1)

    @classmethod
    def from_env(cls):
        return cls.model_validate(
            {
                name: os.environ[key]
                for name in cls.model_fields
                if (key := "SQL_STAGING_" + name.upper()) in os.environ
            }
        )


@dataclass
class SQLStage:
    name: str
    statement: str
    dependencies: list[str] = field(default_factory=list)
    final: bool = False

    def receipt(self):
        return {
            "name": self.name,
            "statement": self.statement,
            "statement_sha256": hashlib.sha256(self.statement.encode()).hexdigest(),
            "dependencies": self.dependencies,
            "final": self.final,
        }


@dataclass
class StagedSQL:
    stages: list[SQLStage]
    settings: SQLStagingSettings
    complexity: dict
    source_statement: str = ""

    @property
    def staged(self):
        return len(self.stages) > 1


def plan_sql_stages(statement, settings=None):
    """Generate an ordered stage DAG using SQLGlot's lexical source resolution."""
    settings = settings or SQLStagingSettings.from_env()
    if len(statement) > 500000:
        raise ValueError("SQL exceeds the 500,000 character compilation limit")
    try:
        trees = sqlglot.parse(statement, read="postgres")
    except (sqlglot.errors.SqlglotError, RecursionError) as error:
        raise ValueError(f"Cannot generate SQL stages: {str(error)[:2000]}") from error
    if len(trees) != 1 or not isinstance(trees[0], exp.Query):
        raise ValueError("Only one SELECT query (optionally with CTEs) is allowed")
    tree = trees[0]
    complexity = {
        "characters": len(statement),
        "selects": len(list(tree.find_all(exp.Select))),
        "joins": len(list(tree.find_all(exp.Join))),
    }
    large = (
        complexity["characters"] > settings.max_chars
        or complexity["selects"] >= settings.max_selects
        or complexity["joins"] >= settings.max_joins
    )
    if not settings.enabled or not large:
        if len(statement) > 20000:
            raise ValueError(
                "SQL exceeds 20,000 characters; enable SQL staging or simplify the logical query"
            )
        return StagedSQL(
            [SQLStage("final", statement, final=True)], settings, complexity, statement
        )
    if any(with_.args.get("recursive") for with_ in tree.find_all(exp.With)):
        raise ValueError(
            "Recursive CTEs cannot be materialized as independent SQL stages"
        )
    for function in tree.find_all(exp.Func):
        name = (
            str(function.this).upper()
            if isinstance(function, exp.Anonymous)
            else function.sql_name().upper()
        )
        if name in {
            "RAND",
            "RANDOM",
            "UUID",
            "GENERATE_UUID",
            "CURRENT_TIMESTAMP",
            "CURRENT_DATE",
            "CURRENT_TIME",
            "NOW",
            "CLOCK_TIMESTAMP",
        }:
            raise ValueError(
                "Volatile SQL functions cannot be split into independent materialized stages"
            )
    scopes = list(traverse_scope(tree))
    reachable = set()

    def visit(scope):
        if id(scope) in reachable:
            return
        reachable.add(id(scope))
        for _, source in scope.selected_sources.values():
            if hasattr(source, "selected_sources"):
                visit(source)
        for child in [*scope.subquery_scopes, *scope.set_operation_scopes]:
            visit(child)

    visit(scopes[-1])
    prefix = "__agent_sql_" + uuid4().hex[:12]
    stages = []
    for scope in scopes:
        if id(scope) not in reachable:
            # Unused CTEs must not evaluate expressions or read rows merely
            # because a surrounding query crossed a staging threshold.
            if scope.is_cte and isinstance(scope.expression.parent, exp.CTE):
                parent = scope.expression.parent
                container = parent.parent
                parent.pop()
                if isinstance(container, exp.With) and not container.expressions:
                    container.pop()
            continue
        if not (scope.is_cte or scope.is_derived_table):
            continue
        # Lateral/outer references need the enclosing row and must stay together.
        if scope.is_correlated_subquery or any(c.table for c in scope.external_columns):
            continue
        parent = scope.expression.parent
        if not isinstance(parent, (exp.CTE, exp.Subquery)):
            continue
        # CTE column lists rename the relation's result columns; preserve them.
        columns = parent.args.get("alias")
        columns = columns.args.get("columns") if columns else None
        body = scope.expression.copy()
        if columns:
            inner = body.subquery("_renamed")
            inner.set(
                "alias",
                exp.TableAlias(
                    this=exp.to_identifier("_renamed"),
                    columns=[c.copy() for c in columns],
                ),
            )
            body = exp.select("*").from_(inner)
        name = prefix + "_" + str(len(stages) + 1)
        sql = body.sql(dialect="postgres")
        stages.append(SQLStage(name, sql))
        if scope.is_cte:
            for consumer in scopes:
                for alias, (node, source) in consumer.selected_sources.items():
                    if source is scope:
                        node.replace(exp.to_table(name).as_(alias, quoted=True))
            container = parent.parent
            parent.pop()
            if isinstance(container, exp.With) and not container.expressions:
                container.pop()
        else:
            replacement = exp.to_table(name)
            if parent.args.get("alias"):
                replacement.set("alias", parent.args["alias"].copy())
            parent.replace(replacement)
    stages.append(SQLStage("final", tree.sql(dialect="postgres"), final=True))
    if len(stages) > settings.max_stages:
        raise ValueError(
            f"SQL requires {len(stages)} stages; maximum is {settings.max_stages}"
        )
    produced = set()
    for stage in stages:
        if len(stage.statement) > 20000:
            raise ValueError(
                "A SQL stage still exceeds 20,000 characters. Simplify the logical query; correlated expressions cannot be split safely."
            )
        refs = {
            t.name
            for t in sqlglot.parse_one(stage.statement, read="postgres").find_all(
                exp.Table
            )
        }
        stage.dependencies = sorted(refs & produced)
        produced.add(stage.name)
    return StagedSQL(stages, settings, complexity, statement)


def read_options():
    return (
        SQLOptions()
        .with_allow_ddl(False)
        .with_allow_dml(False)
        .with_allow_statements(False)
    )


def validate_workspace_sql(statement, allowed):
    """Validate original lineage before private stage names enter the context."""
    tree = sqlglot.parse_one(statement, read="postgres")
    for scope in traverse_scope(tree):
        for source in scope.sources.values():
            if isinstance(source, exp.Table) and (
                not isinstance(source.this, exp.Identifier)
                or source.db
                or source.catalog
                or source.name.lower() not in allowed
            ):
                raise ValueError(
                    "Query only tables belonging to the specified workspace"
                )


@contextmanager
def staged_frame(context, plan, schemas, bindings=()):
    """Compile the entire DAG with empty schemas, then execute bounded stages.

    No stage data is read until all stages compile. Empty and populated temporary
    providers are host-owned and always removed, including on validation failure.
    """
    registered, checks, receipts = [], [], []
    stage_schemas = dict(schemas)
    total_bytes = 0
    try:
        # Review business joins with their full original lineage before replacing
        # those relations with private tables. DataFusion only compiles stages.
        lineage_check = preflight_sql(
            plan.source_statement,
            schemas,
            context,
            bindings,
            compile_datafusion=False,
            max_statement_chars=500000,
        )
        for stage_index, stage in enumerate(plan.stages):
            if not stage.final and context.table_exist(stage.name):
                raise ValueError("SQL stage table name collision")
            try:
                check = preflight_sql(
                    stage.statement,
                    stage_schemas,
                    context,
                    bindings,
                    review_relationships=False,
                )
            except SQLPreflightError as error:
                error.diagnostic.scope_inputs.insert(
                    0,
                    {
                        "stage_number": stage_index + 1,
                        "stage_count": len(plan.stages),
                        "stage_name": stage.name,
                        "dependencies": stage.dependencies,
                    },
                )
                error.diagnostic.repair_instructions.insert(
                    0,
                    "This stage was generated by the host compiler. Revise the logical query using the diagnostic; do not reference private stage tables in QueryIR.",
                )
                raise
            check.reviewed_joins = lineage_check.reviewed_joins
            check.all_join_keys_resolved = lineage_check.all_join_keys_resolved
            check.warnings.extend(lineage_check.warnings)
            checks.append(check)
            frame = context.sql(stage.statement, options=read_options())
            schema = frame.schema()
            if not stage.final:
                empty = pa.RecordBatch.from_arrays(
                    [pa.array([], type=f.type) for f in schema], schema=schema
                )
                context.register_record_batches(stage.name, [[empty]])
                registered.append(stage.name)
                stage_schemas[stage.name] = schema
        trace(
            "SQL_STAGING: "
            + json.dumps({**plan.complexity, "stages": len(plan.stages)})
        )
        for index, stage in enumerate(plan.stages):
            started = monotonic()
            trace(
                f"SQL_STAGE {index + 1}/{len(plan.stages)}: start; name={stage.name}; chars={len(stage.statement)}"
            )
            frame = context.sql(stage.statement, options=read_options())
            record = {**stage.receipt(), "schema": checks[index].result_schema}
            if stage.final:
                receipts.append(record)
                yield frame, checks[-1], receipts
                record["elapsed_seconds"] = round(monotonic() - started, 4)
                trace(f"SQL_STAGE {index + 1}/{len(plan.stages)}: completed; final")
                break
            batches, rows, size = [], 0, 0
            for batch in frame.execute_stream():
                arrow = batch.to_pyarrow()
                rows += arrow.num_rows
                size += arrow.nbytes
                if (
                    rows > plan.settings.max_rows
                    or total_bytes + size > plan.settings.max_bytes
                ):
                    raise ValueError(
                        f"SQL stage {index + 1} exceeds its intermediate row/memory limit; narrow or preaggregate the logical query"
                    )
                batches.append(arrow)
            if not batches:
                schema = frame.schema()
                batches = [
                    pa.RecordBatch.from_arrays(
                        [pa.array([], type=f.type) for f in schema], schema=schema
                    )
                ]
            total_bytes += size
            context.deregister_table(stage.name)
            context.register_record_batches(stage.name, [batches])
            record.update(
                rows=rows, bytes=size, elapsed_seconds=round(monotonic() - started, 4)
            )
            receipts.append(record)
            trace(
                f"SQL_STAGE {index + 1}/{len(plan.stages)}: completed; rows={rows}; bytes={size}"
            )
    finally:
        for name in reversed(registered):
            if context.table_exist(name):
                context.deregister_table(name)
