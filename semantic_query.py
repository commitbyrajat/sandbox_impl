"""Catalog-driven joins, grain planning and SQLGlot AST compilation.

There are no financial table names or business formulas in this module. Missing
semantics, unsafe fanout and ambiguous join paths fail closed before execution.
"""

from __future__ import annotations

import hashlib
import json
import re
from difflib import get_close_matches
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from datetime import datetime, timezone

from sqlglot import exp

from semantic_models import FieldRef, MetricDefinition, QueryIR, SemanticPlannerConfig


class SemanticQueryError(ValueError):
    def __init__(self, code, message, **details):
        self.diagnostic = {"error_code": code, "message": message, **details}
        super().__init__(json.dumps(self.diagnostic))


def fail(code, message, **details):
    raise SemanticQueryError(code, message, **details)


def identifier(name):
    if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]{0,95}", name):
        fail(
            "INVALID_IDENTIFIER",
            "Use simple catalog identifiers and output aliases",
            name=name,
        )
    return name


def col(name, table=None):
    return exp.column(
        identifier(name), table=identifier(table) if table else None, quoted=True
    )


def literal(value):
    if value is None:
        return exp.Null()
    if isinstance(value, bool):
        return exp.Boolean(this=value)
    if isinstance(value, (int, float, Decimal)):
        number = Decimal(str(value))
        if not number.is_finite():
            fail("INVALID_LITERAL", "Numeric literals must be finite")
        return exp.Literal.number(str(number))
    return exp.Literal.string(value)


def numeric(value):
    return exp.Cast(this=value, to=exp.DataType.build("DOUBLE"))


def composite_key_ast(values):
    """Injective typed identity encoding; separators inside values stay safe."""
    strings = [
        exp.Cast(this=value.copy(), to=exp.DataType.build("TEXT")) for value in values
    ]
    pieces = []
    for value in strings:
        pieces.extend(
            [
                exp.Cast(
                    this=exp.Length(this=value.copy()), to=exp.DataType.build("TEXT")
                ),
                literal(":"),
                value,
            ]
        )
    return (
        exp.Case()
        .when(
            exp.or_(
                *(exp.Is(this=value.copy(), expression=exp.Null()) for value in values)
            ),
            exp.Null(),
        )
        .else_(exp.Concat(expressions=pieces, safe=True, coalesce=False))
    )


def formula_ast(formula, resolve, depth=0):
    if depth > 20:
        fail("EXPRESSION_LIMIT", "Expressions must be at most 20 levels deep")
    if formula.op == "ref":
        return resolve(formula.name)
    if formula.op == "literal":
        return literal(formula.value)
    args = [formula_ast(arg, resolve, depth + 1) for arg in formula.args]
    if formula.op == "key":
        return composite_key_ast(args)
    if formula.op == "negate":
        return exp.Neg(this=args[0])
    if formula.op == "coalesce":
        return exp.Coalesce(this=args[0], expressions=[args[1]])
    if formula.op == "concat":
        return exp.Concat(
            expressions=[
                exp.Cast(this=value, to=exp.DataType.build("TEXT")) for value in args
            ],
            safe=True,
            coalesce=False,
        )
    if formula.op == "divide":
        return exp.Div(
            this=numeric(args[0]),
            expression=exp.Nullif(this=numeric(args[1]), expression=literal(0)),
        )
    cls = {"add": exp.Add, "subtract": exp.Sub, "multiply": exp.Mul}[formula.op]
    # Sums retain Arrow decimal money. Derived arithmetic uses bounded float64
    # intermediates to avoid Decimal128 precision growth/overflow; never call it exact.
    return cls(
        this=exp.Paren(this=numeric(args[0])),
        expression=exp.Paren(this=numeric(args[1])),
    )


@dataclass
class CompiledQuery:
    statement: str
    guards: list[tuple[str, str]]
    catalog_sha256: str
    relationships: list[str]
    preaggregated_tables: dict[str, list[str]]
    logical_plan: dict
    source_queries: dict[str, str] = field(default_factory=dict)
    leaf_metrics: dict[str, dict] = field(default_factory=dict)
    planner_receipt: dict = field(default_factory=lambda: {"backend": "catalog"})
    planner_artifacts: dict = field(default_factory=dict)

    def verify_data(self, context):
        """Reject inconsistent constants/keys; MAX must never hide corrupt facts."""
        for code, sql in self.guards:
            if context.sql(sql).limit(1).to_pylist():
                fail(
                    code,
                    "Semantic data consistency check failed; correct source data or metadata",
                    check_sql=sql,
                )


@dataclass
class SemanticCatalog:
    sources: dict
    profile: object
    planner: SemanticPlannerConfig = field(default_factory=SemanticPlannerConfig)
    columns: dict = field(init=False)
    tables: dict = field(init=False)
    relationships: dict = field(init=False)
    metrics: dict = field(init=False)

    def __post_init__(self):
        self.columns, self.tables, self.relationships = {}, {}, {}
        for source in self.sources.values():
            for table in source.tables:
                if table in self.columns:
                    fail(
                        "AMBIGUOUS_TABLE",
                        "Duplicate table across selected sources",
                        table=table,
                    )
                self.columns[table] = source.columns.get(table, {})
                if table in source.semantic_tables:
                    self.tables[table] = source.semantic_tables[table]
            for name, relationship in source.relationship_keys.items():
                if name in self.relationships:
                    fail(
                        "AMBIGUOUS_RELATIONSHIP",
                        "Duplicate relationship ID",
                        relationship_id=name,
                    )
                if {relationship.left_table, relationship.right_table} <= set(
                    self.columns
                ) | set(source.tables):
                    self.relationships[name] = relationship
        self.metrics = dict(self.profile.semantic_metrics)
        for table, semantics in self.tables.items():
            self.metrics.setdefault(
                f"{table}.__count",
                MetricDefinition(
                    description="Observation count at the table grain.",
                    table=table,
                    aggregation="count",
                ),
            )
            for column, policy in semantics.measures.items():
                self.metrics.setdefault(
                    f"{table}.{column}",
                    MetricDefinition(
                        description=policy.description or column,
                        table=table,
                        column=column,
                    ),
                )
        for table, spec in self.tables.items():
            self.check_field(table, None)
            for name in [
                *spec.grain,
                *spec.primary_key,
                *([spec.snapshot_column] if spec.snapshot_column else []),
                *spec.measures,
            ]:
                self.check_field(table, name)
            if not set(spec.primary_key) <= set(spec.grain):
                fail(
                    "INVALID_CATALOG",
                    "Primary key must be contained in declared grain",
                    table=table,
                )
            for name, policy in spec.measures.items():
                if (
                    set(policy.non_additive_over + policy.additive_over)
                    - self.columns[table].keys()
                ):
                    fail(
                        "INVALID_CATALOG",
                        "Measure additivity dimensions must exist",
                        table=table,
                        column=name,
                    )
            for rule in spec.reconciliations:
                for name in [*rule.group_by, rule.sum_column, rule.constant_column]:
                    self.check_field(table, name)
        for name, relation in self.relationships.items():
            for a, b in relation.column_pairs:
                self.check_field(relation.left_table, a)
                self.check_field(relation.right_table, b)
            meaning = self.profile.join_semantics.get(name)
            if meaning:
                for table, declared in (
                    (relation.left_table, meaning.left_grain),
                    (relation.right_table, meaning.right_grain),
                ):
                    if set(declared) != set(self.tables[table].grain):
                        fail(
                            "INCONSISTENT_BUSINESS_GRAIN",
                            "Business join grain must match the table semantic grain",
                            relationship_id=name,
                            table=table,
                        )
                left_unique = set(self.tables[relation.left_table].primary_key) <= {
                    a for a, _ in relation.column_pairs
                }
                right_unique = set(self.tables[relation.right_table].primary_key) <= {
                    b for _, b in relation.column_pairs
                }
                if (
                    meaning.cardinality in ("one_to_one", "one_to_many")
                    and not left_unique
                ) or (
                    meaning.cardinality in ("one_to_one", "many_to_one")
                    and not right_unique
                ):
                    fail(
                        "INCONSISTENT_BUSINESS_CARDINALITY",
                        "Configured business cardinality is not supported by the table primary keys",
                        relationship_id=name,
                    )
            for table, grain in relation.required_preaggregation.items():
                if (
                    table not in (relation.left_table, relation.right_table)
                    or not grain
                ):
                    fail(
                        "INVALID_CATALOG",
                        "Preaggregation must name a relationship table and keys",
                        relationship_id=name,
                    )
                for key in grain:
                    self.check_field(table, key)

    @property
    def fingerprint(self):
        return hashlib.sha256(
            json.dumps(
                {**self.describe(), "planner": self.planner.model_dump()},
                sort_keys=True,
            ).encode()
        ).hexdigest()

    def describe(self):
        return {
            "tables": {
                table: {
                    "columns": {
                        name: spec.model_dump()
                        for name, spec in self.columns[table].items()
                    },
                    **spec.model_dump(),
                }
                for table, spec in self.tables.items()
            },
            "relationships": {
                name: {
                    **rel.model_dump(),
                    "business": (
                        self.profile.join_semantics[name].model_dump()
                        if name in self.profile.join_semantics
                        else None
                    ),
                }
                for name, rel in self.relationships.items()
            },
            "metrics": {
                name: metric.model_dump() for name, metric in self.metrics.items()
            },
        }

    def check_field(self, table, column):
        if table not in self.columns or table not in self.tables:
            fail(
                "MISSING_TABLE_SEMANTICS",
                "Table must belong to this workspace and declare grain/keys",
                table=table,
            )
        if column is not None and column not in self.columns[table]:
            fail(
                "UNKNOWN_COLUMN",
                "Column does not exist in the scoped catalog",
                table=table,
                column=column,
            )


def compile_catalog_query(
    plan: QueryIR, catalog: SemanticCatalog, *, population=None
) -> CompiledQuery:
    """Compile selections; the LLM supplies neither SQL nor aggregation functions."""
    plan = plan.model_copy(deep=True)
    catalog.check_field(plan.base_table, None)
    if population is not None:
        catalog.check_field(population.table, population.source_key)
        plan = plan.model_copy(deep=True)
        plan.base_table = population.table
        identity_fields = {(population.table, population.source_key)}
        changed = True
        while changed:
            changed = False
            for relation in catalog.relationships.values():
                for left, right in relation.column_pairs:
                    a, b = (relation.left_table, left), (relation.right_table, right)
                    if a in identity_fields or b in identity_fields:
                        before = len(identity_fields)
                        identity_fields.update((a, b))
                        changed |= len(identity_fields) > before
        identity = next(
            (item for item in plan.dimensions if item.name == population.result_key),
            None,
        )
        if (
            identity is None
            or identity.transform != "identity"
            or (identity.field.table, identity.field.column) not in identity_fields
        ):
            fail(
                "POPULATION_IDENTITY_CONFLICT",
                "Project the population identity as a dimension using its registered key",
                table=population.table,
                column=population.source_key,
                alias=population.result_key,
            )
        identity.field = FieldRef(table=population.table, column=population.source_key)
    aliases = {table: f"s{index}" for index, table in enumerate(sorted(catalog.tables))}
    leaves, expanded = {}, {}

    def expand(name, stack=()):
        if name in stack or len(stack) > 20:
            fail(
                "METRIC_CYCLE",
                "Catalog metric definitions must be acyclic",
                metric=name,
            )
        if name in expanded:
            return
        if name not in catalog.metrics:
            fail(
                "UNDEFINED_METRIC",
                "Business metric has no configured definition; clarify or extend the catalog",
                metric=name,
                candidate_metric_ids=get_close_matches(
                    name, catalog.metrics, n=5, cutoff=0.3
                ),
            )
        metric = catalog.metrics[name]
        if metric.cumulative:
            fail(
                "UNSUPPORTED_CUMULATIVE_METRIC",
                "Cumulative metrics require the MetricFlow backend",
                metric=name,
            )
        if metric.horizon and name not in plan.metric_horizons:
            # A named metric's fixed horizon is catalog semantics, not missing
            # user intent. Resolve it once without another model request.
            plan.metric_horizons[name] = metric.horizon
        if metric.horizon and plan.metric_horizons.get(name) != metric.horizon:
            fail(
                "AMBIGUOUS_HORIZON",
                "The selected horizon conflicts with the named catalog metric",
                metric=name,
                required_horizon=metric.horizon,
                supplied_horizon=plan.metric_horizons.get(name),
            )
        if metric.formula:
            for node in walk_formula(metric.formula):
                if node.op == "ref":
                    expand(node.name, (*stack, name))
        else:
            catalog.check_field(metric.table, metric.column)
            policy = catalog.tables[metric.table].measures.get(metric.column)
            aggregation = metric.aggregation or (policy.aggregation if policy else None)
            if aggregation is None:
                fail(
                    "MISSING_MEASURE_POLICY",
                    "Metric needs a configured aggregation policy",
                    metric=name,
                )
            if (
                policy
                and metric.aggregation
                and metric.aggregation != policy.aggregation
                and metric.aggregation != "count_distinct"
            ):
                fail(
                    "INVALID_MEASURE_POLICY",
                    "Metric conflicts with its column's aggregation policy",
                    metric=name,
                )
            for predicate in metric.filters:
                if predicate.field.table != metric.table:
                    fail(
                        "INVALID_METRIC_FILTER",
                        "Conditional measures must filter their own fact table",
                        metric=name,
                    )
                catalog.check_field(predicate.field.table, predicate.field.column)
            leaves[name] = (metric, aggregation, policy)
        expanded[name] = metric

    for selection in plan.metrics:
        expand(selection.metric)
    needed = {plan.base_table, *(metric.table for metric, _, _ in leaves.values())}
    for dimension in plan.dimensions:
        catalog.check_field(dimension.field.table, dimension.field.column)
        needed.add(dimension.field.table)
        identifier(dimension.name)
    population_filters = [
        *plan.filters,
        *(
            predicate
            for metric in expanded.values()
            for predicate in metric.required_filters
        ),
    ]
    for predicate in population_filters:
        catalog.check_field(predicate.field.table, predicate.field.column)
        needed.add(predicate.field.table)
        if population is not None and predicate.field.table == plan.base_table:
            fail(
                "POPULATION_FILTER_CONFLICT",
                "A whole-population contract cannot filter away its registered identities",
                table=plan.base_table,
                column=predicate.field.column,
            )
    relations = catalog.relationships
    if plan.relationships:
        unknown = set(plan.relationships) - relations.keys()
        if unknown:
            fail(
                "UNKNOWN_RELATIONSHIP",
                "Select configured relationship IDs",
                relationships=sorted(unknown),
            )

    def shortest_paths(target, allowed):
        graph = {}
        for name in allowed:
            relation = relations[name]
            graph.setdefault(relation.left_table, []).append(
                (relation.right_table, name)
            )
            graph.setdefault(relation.right_table, []).append(
                (relation.left_table, name)
            )
        queue = [(plan.base_table, [], {plan.base_table})]
        paths, distance = [], None
        while queue:
            table, path, visited = queue.pop(0)
            if distance is not None and len(path) > distance:
                break
            if table == target:
                paths.append(path)
                distance = len(path)
                continue
            if len(path) < 10:
                queue.extend(
                    (neighbor, [*path, (table, neighbor, edge)], visited | {neighbor})
                    for neighbor, edge in sorted(graph.get(table, []))
                    if neighbor not in visited
                )
        return paths

    def path_to(target):
        # Relationship IDs disambiguate routes; they are not a replacement for
        # the catalog graph. A model may select just one edge of a valid path.
        paths = shortest_paths(target, plan.relationships or relations)
        if not paths and plan.relationships:
            paths = shortest_paths(target, relations)
            if len(paths) > 1:
                scores = [
                    sum(edge in plan.relationships for _, _, edge in path)
                    for path in paths
                ]
                best = max(scores)
                if best:
                    paths = [
                        path for path, score in zip(paths, scores) if score == best
                    ]
        if len(paths) != 1:
            fail(
                "AMBIGUOUS_JOIN_PATH" if paths else "NO_JOIN_PATH",
                (
                    "Select one complete candidate relationship path; these alternatives come from the full workspace catalog"
                    if paths
                    else "No configured relationship connects these tables; revise the selected metrics/dimensions or correct the catalog"
                ),
                base_table=plan.base_table,
                target=target,
                candidates=paths,
                candidate_relationship_ids=[
                    [edge for _, _, edge in path] for path in paths
                ],
                supplied_relationship_ids=plan.relationships,
                target_metrics=[
                    name
                    for name, (metric, _, _) in leaves.items()
                    if metric.table == target
                ],
                available_relationships={
                    name: {
                        "left_table": relation.left_table,
                        "right_table": relation.right_table,
                        "column_pairs": relation.column_pairs,
                    }
                    for name, relation in relations.items()
                },
            )
        return paths[0]

    joined, edges = {plan.base_table}, []
    for target in sorted(needed):
        for left, right, name in path_to(target):
            if right not in joined:
                joined.add(right)
                edges.append((left, right, name))
    collapse = {}
    for _, _, name in edges:
        for table, grain in relations[name].required_preaggregation.items():
            if table in collapse and collapse[table] != grain:
                fail(
                    "CONFLICTING_GRAINS",
                    "Relationships require incompatible preaggregation",
                    table=table,
                )
            collapse[table] = grain
    # A retained key on the preserved side of a LEFT join also retains a matched
    # fact's key on the nullable side. The converse is unsafe: missing right
    # rows could collapse multiple left periods into one NULL group.
    retained_fields = {
        (d.field.table, d.field.column)
        for d in plan.dimensions
        if d.transform == "identity"
    }
    if plan.time.selection in ("latest", "latest_common"):
        for table in plan.time.tables or [
            t for t in joined if catalog.tables[t].snapshot_column
        ]:
            if table in joined and catalog.tables[table].snapshot_column:
                retained_fields.add((table, catalog.tables[table].snapshot_column))
    changed = True
    while changed:
        changed = False
        for left, right, name in edges:
            relation = relations[name]
            for a, b in relation.column_pairs:
                left_key, right_key = (a, b) if left == relation.left_table else (b, a)
                if (left, left_key) in retained_fields and (
                    right,
                    right_key,
                ) not in retained_fields:
                    retained_fields.add((right, right_key))
                    changed = True
    # Repeated measures queried on their own still collapse the repeated axis.
    for metric, _, policy in leaves.values():
        if policy and policy.non_additive_over:
            dims = {
                column for table, column in retained_fields if table == metric.table
            }
            time_tables = plan.time.tables or [
                table for table in joined if catalog.tables[table].snapshot_column
            ]
            if (
                plan.time.selection in ("latest", "latest_common")
                and metric.table in time_tables
            ):
                dims.add(catalog.tables[metric.table].snapshot_column)
            unresolved = set(policy.non_additive_over) - dims
            if unresolved:
                if policy.consistency != "constant_per_preaggregation_grain":
                    fail(
                        "NON_ADDITIVE_MEASURE",
                        "Preserve non-additive dimensions or select a single snapshot; configure an explicit weighted metric for other reductions",
                        table=metric.table,
                        column=metric.column,
                        dimensions=sorted(unresolved),
                    )
                grain = [
                    x for x in catalog.tables[metric.table].grain if x not in unresolved
                ]
                if metric.table in collapse and collapse[metric.table] != grain:
                    fail(
                        "CONFLICTING_GRAINS",
                        "A repeated measure and relationship require different preaggregation grains",
                        table=metric.table,
                    )
                collapse[metric.table] = grain

    effective_grain = {
        table: collapse.get(table, catalog.tables[table].primary_key)
        for table in joined
    }
    for _, _, name in edges:
        relation = relations[name]
        for table, key_names in (
            (relation.left_table, [a for a, _ in relation.column_pairs]),
            (relation.right_table, [b for _, b in relation.column_pairs]),
        ):
            if table in collapse and not set(key_names) <= set(collapse[table]):
                fail(
                    "INVALID_PREAGGREGATION_KEYS",
                    "Preaggregation must retain every join key",
                    table=table,
                    relationship_id=name,
                )
    # Every additive fact must remain unique across all traversed joins.
    for name, (metric, aggregation, _) in leaves.items():
        if aggregation not in ("sum", "avg", "count"):
            continue
        queue, seen = [metric.table], set()
        while queue:
            table = queue.pop()
            seen.add(table)
            for a, b, edge in edges:
                if table not in (a, b):
                    continue
                neighbor = b if table == a else a
                if neighbor in seen:
                    continue
                relation = relations[edge]
                target_keys = [
                    right if neighbor == relation.right_table else left
                    for left, right in relation.column_pairs
                ]
                if not set(effective_grain[neighbor]) <= set(target_keys):
                    fail(
                        "UNSAFE_FANOUT",
                        "Additive measure would repeat after this join; define required preaggregation or split the query",
                        metric=name,
                        relationship_id=edge,
                        repeated_by=neighbor,
                    )
                queue.append(neighbor)

    def field_expr(ref):
        if ref.table in collapse and ref.column not in collapse[ref.table]:
            fail(
                "PREAGGREGATION_GRAIN_CONFLICT",
                "Requested field is removed by required preaggregation",
                table=ref.table,
                column=ref.column,
            )
        return col(ref.column, aliases[ref.table])

    def predicate_ast(predicate, resolve=field_expr):
        value = resolve(predicate.field)
        op = predicate.operator
        if op in ("is_null", "not_null"):
            node = exp.Is(this=value, expression=exp.Null())
            return exp.Not(this=node) if op == "not_null" else node
        if predicate.value is None:
            fail("INVALID_FILTER", "Use is_null/not_null for null comparisons")
        if op == "in":
            if not isinstance(predicate.value, list) or not predicate.value:
                fail("INVALID_FILTER", "in requires a nonempty list")
            return exp.In(this=value, expressions=[literal(v) for v in predicate.value])
        if isinstance(predicate.value, list):
            fail("INVALID_FILTER", "Comparison requires a scalar")
        # Dates are typed using registry types rather than string comparisons.
        dtype = catalog.columns[predicate.field.table][predicate.field.column].type
        right = literal(predicate.value)
        if dtype.startswith("date"):
            right = exp.Cast(this=right, to=exp.DataType.build("DATE"))
        elif dtype.startswith("timestamp"):
            right = exp.Cast(this=right, to=exp.DataType.build("TIMESTAMP"))
        cls = {
            "eq": exp.EQ,
            "ne": exp.NEQ,
            "lt": exp.LT,
            "le": exp.LTE,
            "gt": exp.GT,
            "ge": exp.GTE,
        }[op]
        return cls(this=value, expression=right)

    times = plan.time.tables or sorted(
        table for table in joined if catalog.tables[table].snapshot_column
    )
    if set(times) - joined:
        fail("INVALID_TIME_SCOPE", "Time selection tables must be part of the query")
    if plan.time.selection in ("latest", "latest_common"):
        conflicts = [
            p.model_dump()
            for p in population_filters
            if p.field.table in times
            and p.field.column == catalog.tables[p.field.table].snapshot_column
        ]
        if conflicts:
            fail(
                "CONFLICTING_SNAPSHOT_FILTER",
                "Latest snapshot selection cannot also hardcode a snapshot date. Remove the date predicate and use the selected snapshot, or use an explicit range/all selection for an intentional date filter. Do not guess month-end dates for monthly keys.",
                time_selection=plan.time.model_dump(),
                conflicting_filters=conflicts,
            )
    period = None
    if plan.time.selection == "range" and plan.time.start and plan.time.end:

        def instant(value):
            try:
                stamp = datetime.fromisoformat(value)
                return (
                    stamp.replace(tzinfo=timezone.utc)
                    if stamp.tzinfo is None
                    else stamp.astimezone(timezone.utc)
                )
            except (ValueError, TypeError):
                fail(
                    "INVALID_TIME_RANGE",
                    "Use ISO date/time range boundaries",
                    value=value,
                )

        start, end = instant(plan.time.start), instant(plan.time.end)
        if start >= end:
            fail("INVALID_TIME_RANGE", "Range start must precede its exclusive end")
        for predicate in population_filters:
            if (
                predicate.field.table not in times
                or predicate.field.column
                != catalog.tables[predicate.field.table].snapshot_column
                or predicate.operator not in ("eq", "lt", "le", "gt", "ge")
            ):
                continue
            boundary = instant(predicate.value)
            outside = (
                (predicate.operator in ("gt", "ge") and boundary < start)
                or (predicate.operator == "lt" and boundary > end)
                or (predicate.operator == "le" and boundary >= end)
                or (predicate.operator == "eq" and not start <= boundary < end)
            )
            if outside:
                fail(
                    "CONFLICTING_TIME_BOUNDARIES",
                    "A source date filter conflicts with the declared range. time.end is exclusive; align both representations with the requested period or use only the time selection.",
                    range=plan.time.model_dump(),
                    filter=predicate.model_dump(),
                )
    if plan.time.selection != "all":
        if not times or any(
            not catalog.tables[table].snapshot_column for table in times
        ):
            fail(
                "MISSING_TIME_SEMANTICS",
                "Time selection needs configured snapshot columns",
            )
        if plan.time.selection in ("latest", "latest_common"):
            if plan.time.selection == "latest" and len(times) != 1:
                fail(
                    "AMBIGUOUS_TIME",
                    "Use latest_common for multiple time series, or select one table",
                )
            periods = [
                exp.select(
                    col(catalog.tables[table].snapshot_column).as_("period")
                ).from_(exp.to_table(table))
                for table in times
            ]
            common = periods[0]
            for other in periods[1:]:
                common = exp.Intersect(this=common, expression=other, distinct=True)
            period = exp.select(exp.Max(this=col("period"))).from_(
                common.subquery("common_periods")
            )
        elif not plan.time.start or not plan.time.end:
            fail(
                "INVALID_TIME_RANGE", "Range requires inclusive start and exclusive end"
            )

    def time_filter(table, alias):
        if table not in times or plan.time.selection == "all":
            return []
        key = col(catalog.tables[table].snapshot_column, alias)
        if period is not None:
            return [exp.EQ(this=key, expression=period.copy().subquery())]
        return [
            exp.GTE(
                this=key.copy(),
                expression=exp.Cast(
                    this=literal(plan.time.start), to=exp.DataType.build("DATE")
                ),
            ),
            exp.LT(
                this=key,
                expression=exp.Cast(
                    this=literal(plan.time.end), to=exp.DataType.build("DATE")
                ),
            ),
        ]

    def source_filters(table, alias):
        return time_filter(table, alias) + (
            [
                predicate_ast(p, lambda ref: col(ref.column, alias))
                for p in population_filters
                if p.field.table == table
            ]
            if population is not None
            else []
        )

    def aggregate(aggregation, value):
        if aggregation == "count":
            return exp.Count(this=value)
        if aggregation == "count_distinct":
            return exp.Count(this=exp.Distinct(expressions=[value]))
        return {
            "sum": exp.Sum,
            "min": exp.Min,
            "max": exp.Max,
            "avg": exp.Avg,
            "constant": exp.Max,
        }[aggregation](this=value)

    def conditional(metric, alias):
        value = col(metric.column or catalog.tables[metric.table].primary_key[0], alias)
        if metric.filters:
            condition = exp.and_(
                *(
                    predicate_ast(p, lambda ref: col(ref.column, alias))
                    for p in metric.filters
                )
            )
            value = exp.Case(
                ifs=[exp.If(this=condition, true=value)], default=literal(0)
            )
        return value

    guards, ctes, leaf_values = [], [], {}
    if period is not None:
        empty_period = exp.select(literal(1)).where(
            exp.Is(this=period.copy().subquery(), expression=exp.Null())
        )
        guards.append(("NO_COMMON_PERIOD", empty_period.sql(dialect="postgres")))
    for table in sorted(joined):
        alias = aliases[table]
        keys = [col(key, alias) for key in catalog.tables[table].primary_key]
        duplicate = (
            exp.select(*keys)
            .from_(exp.to_table(table).as_(alias))
            .group_by(*keys)
            .having(exp.GT(this=exp.Count(this=exp.Star()), expression=literal(1)))
        )
        null_key = (
            exp.select(*keys)
            .from_(exp.to_table(table).as_(alias))
            .where(
                exp.or_(
                    *(exp.Is(this=key.copy(), expression=exp.Null()) for key in keys)
                )
            )
        )
        guards.extend(
            [
                ("DUPLICATE_SOURCE_GRAIN", duplicate.sql(dialect="postgres")),
                ("NULL_SOURCE_KEY", null_key.sql(dialect="postgres")),
            ]
        )
        for rule in catalog.tables[table].reconciliations:
            group_keys = [col(name, "reconcile") for name in rule.group_by]
            source = (
                exp.select(*group_keys)
                .from_(exp.to_table(table).as_("reconcile"))
                .group_by(*group_keys)
            )
            conditions = time_filter(table, "reconcile")
            if conditions:
                source = source.where(*conditions)
            repeated = col(rule.constant_column, "reconcile")
            guards.append(
                (
                    "INCONSISTENT_COHORT_ATTRIBUTE",
                    consistency_query(source.copy(), repeated.copy(), "reject").sql(
                        dialect="postgres"
                    ),
                )
            )
            total = exp.Sum(this=col(rule.sum_column, "reconcile"))
            difference = exp.Abs(
                this=exp.Sub(this=total, expression=exp.Max(this=repeated))
            )
            check = source.having(
                exp.GT(this=difference, expression=literal(rule.tolerance))
            )
            guards.append(
                ("GRAIN_RECONCILIATION_FAILED", check.sql(dialect="postgres"))
            )
        for name, (metric, _, policy) in leaves.items():
            if metric.table == table and policy and policy.null_policy == "reject":
                check = (
                    exp.select(literal(1))
                    .from_(exp.to_table(table).as_("checked"))
                    .where(
                        exp.Is(
                            this=col(metric.column, "checked"), expression=exp.Null()
                        ),
                        *time_filter(table, "checked"),
                    )
                )
                guards.append(("NULL_REQUIRED_MEASURE", check.sql(dialect="postgres")))
        if table not in collapse:
            continue
        grain = collapse[table]
        projected = [col(key, "raw") for key in grain]
        source = exp.to_table(table).as_("raw")
        query = exp.select(*projected).from_(source).group_by(*projected)
        conditions = source_filters(table, "raw")
        if conditions:
            query = query.where(*conditions)
        for name, (metric, aggregation, policy) in leaves.items():
            if metric.table != table:
                continue
            if aggregation in ("avg", "count_distinct", "count"):
                fail(
                    "NONCOMPOSABLE_AGGREGATE",
                    "Define a weighted/composable metric for required preaggregation",
                    metric=name,
                )
            if (
                metric.filters
                and policy
                and policy.non_additive_over
                and any(
                    predicate.field.column not in grain for predicate in metric.filters
                )
            ):
                fail(
                    "UNSAFE_CONDITIONAL_CONSTANT",
                    "Do not condition repeated-cohort attributes on a destination",
                    metric=name,
                )
            method = (
                "constant"
                if policy and policy.consistency == "constant_per_preaggregation_grain"
                else aggregation
            )
            if (
                policy
                and set(policy.non_additive_over) - set(grain)
                and method != "constant"
            ):
                fail(
                    "NON_ADDITIVE_MEASURE",
                    "Repeated measures require a constant consistency rule",
                    metric=name,
                )
            if method == "constant":
                consistency_source = (
                    exp.select(*projected).from_(source.copy()).group_by(*projected)
                )
                if conditions:
                    consistency_source = consistency_source.where(
                        *(condition.copy() for condition in conditions)
                    )
                check = consistency_query(
                    consistency_source,
                    col(metric.column, "raw"),
                    policy.null_policy if policy else "allow",
                )
                guards.append(
                    ("INCONSISTENT_COHORT_ATTRIBUTE", check.sql(dialect="postgres"))
                )
            inner_alias = f"m{len(leaf_values)}"
            query = query.select(
                aggregate(method, conditional(metric, "raw")).as_(inner_alias),
                append=True,
            )
            leaf_values[name] = col(inner_alias, alias)
        ctes.append((f"pre_{alias}", query))

    def source_table(table):
        if table in collapse:
            return exp.to_table(f"pre_{aliases[table]}").as_(aliases[table])
        predicates = source_filters(table, "time_source")
        if predicates:
            return (
                exp.select(exp.Star())
                .from_(exp.to_table(table).as_("time_source"))
                .where(*predicates)
                .subquery(aliases[table])
            )
        return exp.to_table(table).as_(aliases[table])

    query = exp.select().from_(source_table(plan.base_table))
    for left, right, name in edges:
        relation = relations[name]
        on = exp.and_(
            *(
                exp.EQ(
                    this=col(a, aliases[relation.left_table]),
                    expression=col(b, aliases[relation.right_table]),
                )
                for a, b in relation.column_pairs
            )
        )
        query = query.join(source_table(right), on=on, join_type="LEFT")
    conditions = (
        [] if population is not None else [predicate_ast(p) for p in population_filters]
    )
    if conditions:
        query = query.where(*conditions)
    groups, projections = [], []
    for dimension in plan.dimensions:
        value = field_expr(dimension.field)
        if dimension.transform != "identity":
            dtype = catalog.columns[dimension.field.table][dimension.field.column].type
            if not dtype.startswith(("date", "timestamp")):
                fail(
                    "INVALID_DIMENSION_TYPE",
                    "Date transforms require a date/timestamp column",
                )
            value = (
                exp.Cast(this=value, to=exp.DataType.build("DATE"))
                if dimension.transform == "day"
                else exp.DateTrunc(unit=literal(dimension.transform), this=value)
            )
        groups.append(value)
        projections.append(value.copy().as_(identifier(dimension.name), quoted=True))
    if groups and leaves:
        query = query.group_by(*groups)
    grouped_source = query.copy()

    def metric_ast(name):
        metric = expanded[name]
        if metric.formula:
            return formula_ast(metric.formula, metric_ast)
        _, aggregation, policy = leaves[name]
        value = (
            leaf_values[name].copy()
            if name in leaf_values
            else conditional(metric, aliases[metric.table])
        )
        if aggregation == "constant":
            check = query.copy().select(
                *(groups or [exp.Count(this=exp.Star())]), append=False
            )
            guards.append(
                (
                    "INCONSISTENT_RESULT_ATTRIBUTE",
                    consistency_query(
                        check, value.copy(), policy.null_policy if policy else "allow"
                    ).sql(dialect="postgres"),
                )
            )
        return aggregate(aggregation, value)

    for selection in plan.metrics:
        projections.append(
            metric_ast(selection.metric).as_(identifier(selection.name), quoted=True)
        )
    if not leaves:
        query = query.distinct()
    query = query.select(*projections, append=False)
    for name, cte in ctes:
        query = query.with_(name, as_=cte)
    available = {x.name for x in [*plan.dimensions, *plan.metrics]}
    for calculation in plan.calculations:

        def resolve(name):
            if name not in available:
                fail(
                    "UNKNOWN_EXPRESSION_REF",
                    "Calculations reference preceding output aliases only",
                    reference=name,
                )
            return col(name, "calculated")

        value = formula_ast(calculation.expression, resolve)
        query = exp.select(
            exp.Column(this=exp.Star(), table=exp.to_identifier("calculated")),
            value.as_(identifier(calculation.name), quoted=True),
        ).from_(query.subquery("calculated"))
        available.add(calculation.name)
    if plan.order_by:
        query = query.order_by(
            *(
                exp.Ordered(
                    this=col(item.name), desc=item.descending, nulls_first=False
                )
                for item in plan.order_by
            )
        )
    for metric_name, spec in expanded.items():
        bounds = []
        if spec.minimum is not None:
            bounds.append(
                exp.LT(
                    this=col("metric_value", "bounded"),
                    expression=literal(spec.minimum),
                )
            )
        if spec.maximum is not None:
            bounds.append(
                exp.GT(
                    this=col("metric_value", "bounded"),
                    expression=literal(spec.maximum),
                )
            )
        if bounds:
            checked_metric = grouped_source.copy().select(
                metric_ast(metric_name).as_("metric_value"), append=False
            )
            check = (
                exp.select(literal(1))
                .from_(checked_metric.subquery("bounded"))
                .where(exp.or_(*bounds))
            )
            guards.append(("METRIC_OUT_OF_BOUNDS", check.sql(dialect="postgres")))
    # Guards referring to collapsed relations need the same generated WITH clause.
    final_guards = []
    from sqlglot import parse_one

    for code, sql in guards:
        guard = parse_one(sql, read="postgres")
        existing_ctes = {cte.alias_or_name for cte in guard.find_all(exp.CTE)}
        for name, cte in ctes:
            if name not in existing_ctes:
                guard = guard.with_(name, as_=cte.copy())
        final_guards.append((code, guard.sql(dialect="postgres")))
    return CompiledQuery(
        query.sql(dialect="postgres"),
        final_guards,
        catalog.fingerprint,
        [edge for _, _, edge in edges],
        collapse,
        plan.model_dump(),
        source_queries={
            table: (
                dict(ctes)[f"pre_{aliases[table]}"].sql(dialect="postgres")
                if table in collapse
                else exp.select(exp.Star())
                .from_(source_table(table))
                .sql(dialect="postgres")
            )
            for table in sorted(joined)
        },
        leaf_metrics={
            name: {
                "table": metric.table,
                "expression": (
                    col(leaf_values[name].name)
                    if name in leaf_values
                    else conditional(metric, None)
                ).sql(dialect="postgres"),
                "aggregation": aggregation,
            }
            for name, (metric, aggregation, _) in leaves.items()
        },
    )


def compile_query(
    plan: QueryIR, catalog: SemanticCatalog, *, population=None
) -> CompiledQuery:
    if catalog.planner.backend == "metricflow":
        from metricflow_adapter import compile_metricflow_query

        return compile_metricflow_query(plan, catalog, population=population)
    return compile_catalog_query(plan, catalog, population=population)


def walk_formula(node):
    yield node
    for arg in node.args:
        yield from walk_formula(arg)


def consistency_query(query, value, null_policy):
    different = exp.GT(
        this=exp.Count(this=exp.Distinct(expressions=[value.copy()])),
        expression=literal(1),
    )
    count = exp.Count(this=value.copy())
    missing = exp.NEQ(this=count.copy(), expression=exp.Count(this=exp.Star()))
    if null_policy == "allow":
        missing = exp.and_(missing, exp.GT(this=count, expression=literal(0)))
    return query.having(exp.or_(different, missing))
