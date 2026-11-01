"""MetricFlow semantic-manifest adapter and compile-only DataFusion bridge.

Business definitions come from the scoped catalog. MetricFlow owns metric
aggregation, entity joins, ratio and cumulative dataflow planning. Source grain
checks and declared preaggregation remain mandatory host preparation steps.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import re
from dataclasses import replace
from datetime import date, datetime, timedelta
from threading import RLock

import sqlglot
from sqlglot import exp
from sqlglot.optimizer.scope import traverse_scope

from metricflow.engine.metricflow_engine import MetricFlowEngine, MetricFlowQueryRequest
from metricflow.protocols.sql_client import SqlEngine
from metricflow.sql.render.duckdb_renderer import DuckDbSqlPlanRenderer
from metricflow_semantic_interfaces.implementations.semantic_manifest import (
    PydanticSemanticManifest,
)
from metricflow_semantics.model.semantic_manifest_lookup import SemanticManifestLookup
from metricflow_semantics.model.dbt_manifest_parser import (
    transform_dbt_generated_manifest,
)
from metricflow_semantic_interfaces.validations.semantic_manifest_validator import (
    SemanticManifestValidator,
)

from semantic_models import Formula
from semantic_query import (
    SemanticQueryError,
    col,
    compile_catalog_query,
    fail,
    formula_ast,
    literal,
)

# MetricFlow resets generated node IDs. Serialize manifest/explain construction
# rather than sharing an engine or a cache between tenants / requests.
_COMPILE_LOCK = RLock()
_MANIFEST_VALIDATOR = SemanticManifestValidator()


class MetricFlowUnsupported(ValueError):
    pass


class CompileOnlyClient:
    sql_engine_type = SqlEngine.DUCKDB
    sql_plan_renderer = DuckDbSqlPlanRenderer()

    def query(self, *args, **kwargs):
        raise RuntimeError("MetricFlow may only compile; DataFusion owns execution")

    execute = query
    dry_run = query

    def close(self):
        pass


def mf_name(prefix, name):
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")[:65]
    return f"{prefix}_{slug}_{hashlib.sha256(name.encode()).hexdigest()[:8]}"


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str).encode()
    ).hexdigest()


def entity_expr(columns):
    if len(columns) == 1:
        return col(columns[0])
    # A length-prefixed tuple is injective, unlike delimiter-only CONCAT. Null
    # components remain NULL so they cannot create a spurious relationship.
    strings = [
        exp.Cast(this=col(name), to=exp.DataType.build("TEXT")) for name in columns
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
    return exp.Case(
        ifs=[
            exp.If(
                this=exp.or_(
                    *(exp.Is(this=col(name), expression=exp.Null()) for name in columns)
                ),
                true=exp.Null(),
            )
        ],
        default=exp.Concat(expressions=pieces, safe=True, coalesce=False),
    )


def _compile(plan, catalog, *, population=None):
    # Lower only cumulative references for the host's safety checks. The native
    # MetricFlow manifest below retains cumulative types and plans their windows.
    safety_catalog = copy.copy(catalog)
    safety_catalog.metrics = dict(catalog.metrics)
    for name, metric in catalog.metrics.items():
        if metric.cumulative:
            safety_catalog.metrics[name] = metric.model_copy(
                update={
                    "cumulative": None,
                    "formula": Formula(op="ref", name=metric.cumulative.input_metric),
                }
            )
    prepared = compile_catalog_query(plan, safety_catalog, population=population)
    effective = type(plan).model_validate(prepared.logical_plan)
    if population is not None:
        raise MetricFlowUnsupported(
            "Whole-population completion with zero-activity members uses the catalog compiler"
        )
    predicates = list(effective.filters)

    # Native metric dataflows aggregate sources independently. Cross-source WHERE
    # predicates and partially scoped snapshots can change the anchored LEFT-join
    # population (including phantom NULL groups). Retain catalog semantics until
    # these relational scopes can be represented in the semantic manifest.
    def required_filters(name, seen=None):
        seen = set() if seen is None else seen
        if name in seen or name not in catalog.metrics:
            return []
        seen.add(name)
        definition = catalog.metrics[name]
        result = list(definition.required_filters)
        if definition.formula:

            def refs(node):
                if node.op == "ref":
                    result.extend(required_filters(node.name, seen))
                for arg in node.args:
                    refs(arg)

            refs(definition.formula)
        if definition.cumulative:
            result.extend(required_filters(definition.cumulative.input_metric, seen))
        return result

    for selection in effective.metrics:
        predicates.extend(required_filters(selection.metric))
    if any(p.field.table != effective.base_table for p in predicates):
        raise MetricFlowUnsupported(
            "Cross-source population filters use the catalog compiler to preserve anchored outer-join row membership"
        )
    if effective.time.selection in ("latest", "latest_common"):
        clocks = set(
            effective.time.tables
            or [t for t in prepared.source_queries if catalog.tables[t].snapshot_column]
        )
        fact_clocks = {
            leaf["table"]
            for leaf in prepared.leaf_metrics.values()
            if catalog.tables[leaf["table"]].snapshot_column
        }
        if fact_clocks - clocks:
            raise MetricFlowUnsupported(
                "Partially scoped fact snapshots use the catalog compiler; select latest_common across all temporal fact inputs for native metric planning"
            )
    selected = set(prepared.leaf_metrics)

    def collect(name):
        if name not in catalog.metrics:
            fail(
                "UNDEFINED_METRIC",
                "Metric is not defined in this workspace",
                metric=name,
            )
        if name in selected:
            return
        selected.add(name)
        metric = catalog.metrics[name]
        if metric.cumulative:
            input_definition = catalog.metrics.get(metric.cumulative.input_metric)
            if (
                input_definition is None
                or input_definition.formula
                or input_definition.cumulative
            ):
                raise MetricFlowUnsupported(
                    "MetricFlow cumulative inputs must be base metrics; model cumulative components explicitly for derived or ratio metrics"
                )
            collect(metric.cumulative.input_metric)
        elif metric.formula:

            def walk(node):
                if node.op == "ref":
                    collect(node.name)
                for arg in node.args:
                    walk(arg)

            walk(metric.formula)

    for item in effective.metrics:
        collect(item.metric)
    if not effective.metrics:
        raise MetricFlowUnsupported(
            "Dimension-only lookups have no MetricFlow metric request"
        )
    cumulative = any(catalog.metrics[name].cumulative for name in selected)
    if cumulative:
        if (
            effective.time.selection != "range"
            or not effective.time.start
            or not effective.time.end
        ):
            raise MetricFlowUnsupported(
                "Cumulative metrics require an explicit bounded time range"
            )
        for predicate in effective.filters:
            if (
                predicate.field.column
                == catalog.tables[predicate.field.table].snapshot_column
            ):
                raise MetricFlowUnsupported(
                    "Use the time selection alone for cumulative time bounds; duplicate source date filters would truncate window history"
                )
        # Remove display bounds from the input sources; MetricFlow widens these
        # itself for a rolling window or a grain-to-date computation.
        safety_plan = effective.model_copy(
            update={"time": effective.time.model_copy(update={"selection": "all"})}
        )
        prepared = compile_catalog_query(
            safety_plan, safety_catalog, population=population
        )

    models, metrics, views = [], [], {}
    metric_ids = {name: mf_name("m", name) for name in selected}
    model_ids = {table: mf_name("s", table) for table in prepared.source_queries}
    dimension_ids = {}
    real_time = {}
    for table, sql in prepared.source_queries.items():
        source = sqlglot.parse_one(sql, read="postgres")
        keys = prepared.preaggregated_tables.get(
            table, catalog.tables[table].primary_key
        )
        columns = list(prepared.preaggregated_tables.get(table, catalog.columns[table]))
        # Preaggregation also exposes host-generated measure columns.
        for leaf in prepared.leaf_metrics.values():
            if leaf["table"] == table:
                for node in sqlglot.parse_one(
                    leaf["expression"], read="postgres"
                ).find_all(exp.Column):
                    if node.name not in columns:
                        columns.append(node.name)
        wrapper = exp.select(exp.Star()).from_(source.subquery("source_rows"))
        row_entity = mf_name("row", table)
        entities = [
            {
                "name": row_entity,
                "type": "primary",
                "expr": entity_expr(keys).sql(dialect="duckdb"),
            }
        ]
        for relationship_id in prepared.relationships:
            relationship = catalog.relationships[relationship_id]
            if table not in (relationship.left_table, relationship.right_table):
                continue
            names = [
                left if table == relationship.left_table else right
                for left, right in relationship.column_pairs
            ]
            if len(names) > 1 and any(
                catalog.columns[relationship.left_table][a].type
                != catalog.columns[relationship.right_table][b].type
                for a, b in relationship.column_pairs
            ):
                raise MetricFlowUnsupported(
                    "Compound MetricFlow entities require matching registered key types"
                )
            unique = set(keys) <= set(names)
            entities.append(
                {
                    "name": mf_name("e", relationship_id),
                    "type": "unique" if unique else "foreign",
                    "expr": entity_expr(names).sql(dialect="duckdb"),
                }
            )
        dimensions = []
        for column in columns:
            name = mf_name("d", f"{table}.{column}")
            dimension_ids[table, column, "identity"] = name
            dimensions.append(
                {
                    "name": name,
                    "type": "categorical",
                    "expr": col(column).sql(dialect="duckdb"),
                }
            )
        for item in effective.dimensions:
            if item.field.table != table or item.transform == "identity":
                continue
            key = table, item.field.column, item.transform
            if key in dimension_ids:
                continue
            name = mf_name("d", ".".join(key))
            value = col(item.field.column)
            value = (
                exp.Cast(this=value, to=exp.DataType.build("DATE"))
                if item.transform == "day"
                else exp.DateTrunc(unit=literal(item.transform), this=value)
            )
            dimensions.append(
                {
                    "name": name,
                    "type": "categorical",
                    "expr": value.sql(dialect="duckdb"),
                }
            )
            dimension_ids[key] = name
        temporal = catalog.tables[table].snapshot_column
        if temporal is None:
            temporal = next(
                (
                    name
                    for name, spec in catalog.columns[table].items()
                    if spec.type.startswith(("date", "timestamp"))
                ),
                None,
            )
        real_time[table] = temporal
        time_expr = (
            col(temporal)
            if temporal
            else exp.Cast(this=literal("1970-01-01"), to=exp.DataType.build("DATE"))
        )
        dimensions.append(
            {
                "name": "mf_event_time",
                "type": "time",
                "expr": time_expr.sql(dialect="duckdb"),
                "type_params": {"time_granularity": "day"},
            }
        )
        views[model_ids[table]] = wrapper
        models.append(
            {
                "name": model_ids[table],
                "node_relation": {
                    "schema_name": "metricflow",
                    "alias": model_ids[table],
                },
                "defaults": {"agg_time_dimension": "mf_event_time"},
                "entities": entities,
                "dimensions": dimensions,
            }
        )

    for name in sorted(selected):
        metric = catalog.metrics[name]
        spec = {"name": metric_ids[name], "description": metric.description}
        if metric.cumulative:
            if not all(
                real_time[leaf["table"]] for leaf in prepared.leaf_metrics.values()
            ):
                raise MetricFlowUnsupported(
                    "A static semantic model cannot be used as a cumulative time series"
                )
            params = {
                "metric": {"name": metric_ids[metric.cumulative.input_metric]},
                "period_agg": metric.cumulative.period_aggregation,
            }
            if metric.cumulative.window_count:
                params["window"] = {
                    "count": metric.cumulative.window_count,
                    "granularity": metric.cumulative.window_grain,
                }
            if metric.cumulative.grain_to_date:
                params["grain_to_date"] = metric.cumulative.grain_to_date
            spec.update(
                type="cumulative", type_params={"cumulative_type_params": params}
            )
        elif metric.formula:
            formula = metric.formula
            if formula.op == "divide" and all(arg.op == "ref" for arg in formula.args):
                spec.update(
                    type="ratio",
                    type_params={
                        "numerator": {"name": metric_ids[formula.args[0].name]},
                        "denominator": {"name": metric_ids[formula.args[1].name]},
                    },
                )
            else:
                references = set()

                def resolve(reference):
                    references.add(reference)
                    return col(metric_ids[reference])

                expr = formula_ast(formula, resolve).sql(dialect="duckdb")
                spec.update(
                    type="derived",
                    type_params={
                        "expr": expr,
                        "metrics": [
                            {"name": metric_ids[ref]} for ref in sorted(references)
                        ],
                    },
                )
        else:
            leaf = prepared.leaf_metrics[name]
            aggregation = {"constant": "max", "avg": "average"}.get(
                leaf["aggregation"], leaf["aggregation"]
            )
            spec.update(
                type="simple",
                type_params={
                    "expr": leaf["expression"],
                    "metric_aggregation_params": {
                        "semantic_model": model_ids[leaf["table"]],
                        "agg": aggregation,
                    },
                },
            )
        metrics.append(spec)
    manifest_data = {
        "semantic_models": models,
        "metrics": metrics,
        "project_configuration": {
            "time_spine_table_configurations": [
                {
                    "location": "metricflow.mf_time_spine",
                    "column_name": "date_day",
                    "grain": "day",
                }
            ]
        },
    }
    manifest = transform_dbt_generated_manifest(
        PydanticSemanticManifest.parse_obj(manifest_data)
    )
    _MANIFEST_VALIDATOR.checked_validations(manifest)
    resolved_manifest = json.loads(manifest.json())
    engine = MetricFlowEngine(SemanticManifestLookup(manifest), CompileOnlyClient())
    requested = list(
        dict.fromkeys(metric_ids[item.metric] for item in effective.metrics)
    )
    available = engine.simple_dimensions_for_metrics(requested)
    group_by, output_columns = [], {}
    for item in effective.dimensions:
        if (
            cumulative
            and item.field.column == catalog.tables[item.field.table].snapshot_column
        ):
            if item.transform not in ("day", "month", "year"):
                raise MetricFlowUnsupported(
                    "Cumulative metric_time grouping needs an explicit day/month/year transform"
                )
            dunder = f"metric_time__{item.transform}"
        else:
            dim_id = dimension_ids[item.field.table, item.field.column, item.transform]
            candidates = [dim for dim in available if dim.name == dim_id]
            if not candidates:
                raise MetricFlowUnsupported(
                    f"No common MetricFlow entity path for {item.field.table}.{item.field.column}"
                )
            dunder = min(
                candidates, key=lambda dim: (len(dim.entity_links), dim.dunder_name)
            ).dunder_name
        # QueryIR may expose the same dimension under several output aliases.
        # MetricFlow groups once; the outer projection preserves those aliases.
        if dunder not in group_by:
            group_by.append(dunder)
        output_columns[item.name] = dunder
    for item in effective.metrics:
        output_columns[item.name] = metric_ids[item.metric]
    # Source predicates keep their meaning in MetricFlow's post-join filter API.
    predicates = [
        *effective.filters,
        *(p for name in selected for p in catalog.metrics[name].required_filters),
    ]
    where = []
    for predicate in predicates:
        dim_id = dimension_ids[
            predicate.field.table, predicate.field.column, "identity"
        ]
        candidates = [dim for dim in available if dim.name == dim_id]
        if not candidates:
            raise MetricFlowUnsupported(
                f"No common MetricFlow filter path for {predicate.field.table}.{predicate.field.column}"
            )
        dunder = min(
            candidates, key=lambda dim: (len(dim.entity_links), dim.dunder_name)
        ).dunder_name
        ref = "{{ Dimension('" + dunder + "') }}"
        left = exp.Var(this=ref)
        if predicate.operator in ("is_null", "not_null"):
            node = exp.Is(this=left, expression=exp.Null())
            if predicate.operator == "not_null":
                node = exp.Not(this=node)
        elif predicate.operator == "in":
            node = exp.In(this=left, expressions=[literal(v) for v in predicate.value])
        else:
            cls = {
                "eq": exp.EQ,
                "ne": exp.NEQ,
                "lt": exp.LT,
                "le": exp.LTE,
                "gt": exp.GT,
                "ge": exp.GTE,
            }[predicate.operator]
            right = literal(predicate.value)
            dtype = catalog.columns[predicate.field.table][predicate.field.column].type
            if dtype.startswith(("date", "timestamp")):
                right = exp.Cast(
                    this=right,
                    to=exp.DataType.build(
                        "DATE" if dtype.startswith("date") else "TIMESTAMP"
                    ),
                )
            node = cls(this=left, expression=right)
        where.append(node.sql(dialect="duckdb"))
    start, end = None, None
    if cumulative:
        start, exclusive_end = datetime.fromisoformat(
            effective.time.start
        ), datetime.fromisoformat(effective.time.end)
        end = exclusive_end - timedelta(microseconds=1)
        spine_start, spine_end = date.fromisoformat(
            catalog.planner.time_spine_start
        ), date.fromisoformat(catalog.planner.time_spine_end)
        if start.date() < spine_start or exclusive_end.date() > spine_end:
            fail(
                "METRICFLOW_TIME_SPINE_RANGE",
                "Requested cumulative period is outside the configured time spine",
            )
    result = engine.explain(
        MetricFlowQueryRequest.create(
            metric_names=requested,
            group_by_names=group_by,
            where_constraints=where or None,
            time_constraint_start=start,
            time_constraint_end=end,
        )
    )
    if result.sql_statement.bind_parameter_set.param_items:
        fail(
            "METRICFLOW_BIND_PARAMETERS",
            "MetricFlow returned bound parameters that the DataFusion bridge does not support",
        )
    raw_sql = result.sql_statement.sql
    tree = sqlglot.parse_one(raw_sql, read="duckdb")
    used_views = set()
    base_tables = {
        id(source): source
        for scope in traverse_scope(tree)
        for source in scope.sources.values()
        if isinstance(source, exp.Table)
    }
    for table in base_tables.values():
        if table.db != "metricflow" or table.catalog:
            fail(
                "METRICFLOW_SOURCE_SCOPE",
                "MetricFlow SQL referenced an unrecognized relation",
                relation=table.sql(),
            )
        if table.name == "mf_time_spine":
            start_day = date.fromisoformat(catalog.planner.time_spine_start)
            end_day = date.fromisoformat(catalog.planner.time_spine_end)
            days = (end_day - start_day).days
            if not 0 < days <= 730:
                fail(
                    "METRICFLOW_TIME_SPINE_RANGE",
                    "Time spine must contain between 1 and 730 days",
                )
            tuples = [
                exp.Tuple(
                    expressions=[literal((start_day + timedelta(days=i)).isoformat())]
                )
                for i in range(days)
            ]
            values = exp.Values(
                expressions=tuples,
                alias=exp.TableAlias(
                    this=exp.to_identifier("calendar"),
                    columns=[exp.to_identifier("day_string")],
                ),
            )
            views["mf_time_spine"] = exp.select(
                exp.Cast(this=col("day_string"), to=exp.DataType.build("DATE")).as_(
                    "date_day"
                )
            ).from_(values)
        elif table.name not in views:
            fail(
                "METRICFLOW_SOURCE_SCOPE",
                "MetricFlow SQL escaped the scoped semantic sources",
                relation=table.sql(),
            )
        used_views.add(table.name)
        table.set("db", None)
    existing_ctes = tree.args.get("with_")
    tree.set(
        "with_",
        exp.With(
            expressions=[
                *(
                    exp.CTE(
                        this=views[key],
                        alias=exp.TableAlias(this=exp.to_identifier(key)),
                    )
                    for key in sorted(used_views)
                ),
                *(existing_ctes.expressions if existing_ctes else []),
            ]
        ),
    )
    # A parser/transpiler is a syntax bridge, never proof of DataFusion support.
    # Normalize ratio arithmetic to the pipeline's documented float64 boundary.
    for division in list(tree.find_all(exp.Div)):
        division.set(
            "this", exp.Cast(this=division.this, to=exp.DataType.build("DOUBLE"))
        )
        division.set(
            "expression",
            exp.Nullif(
                this=exp.Cast(
                    this=division.expression, to=exp.DataType.build("DOUBLE")
                ),
                expression=literal(0),
            ),
        )
    projections = []
    for target, source in output_columns.items():
        value = col(source, "mf_result")
        if cumulative and source.startswith("metric_time__"):
            value = exp.Cast(this=value, to=exp.DataType.build("DATE"))
        selection = next(
            (item for item in effective.metrics if item.name == target), None
        )
        if (
            selection
            and selection.metric in prepared.leaf_metrics
            and prepared.leaf_metrics[selection.metric]["aggregation"]
            in ("count", "count_distinct")
        ):
            value = exp.Coalesce(this=value, expressions=[literal(0)])
        projections.append(value.as_(target, quoted=True))
    query = exp.select(*projections).from_(tree.subquery("mf_result"))
    available_aliases = set(output_columns)
    for calculation in effective.calculations:

        def resolve(name):
            if name not in available_aliases:
                fail(
                    "UNKNOWN_EXPRESSION_REF",
                    "Use a preceding output alias",
                    reference=name,
                )
            return col(name, "calculated")

        value = formula_ast(calculation.expression, resolve)
        query = exp.select(
            exp.Column(this=exp.Star(), table=exp.to_identifier("calculated")),
            value.as_(calculation.name, quoted=True),
        ).from_(query.subquery("calculated"))
        available_aliases.add(calculation.name)
    if effective.order_by:
        query = query.order_by(
            *(
                exp.Ordered(
                    this=col(item.name), desc=item.descending, nulls_first=False
                )
                for item in effective.order_by
            )
        )
    text = result.dataflow_plan.structure_text()
    return replace(
        prepared,
        statement=query.sql(dialect="postgres", comments=False),
        logical_plan=effective.model_dump(),
        planner_receipt={
            "backend": "metricflow",
            "version": importlib.metadata.version("metricflow"),
            "renderer": "duckdb",
            "manifest_sha256": digest(resolved_manifest),
            "manifest_validated": True,
            "dataflow_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "dataflow_node_types": sorted(set(re.findall(r"<([A-Za-z]+Node)>", text))),
            "metric_bindings": metric_ids,
            "time_groupings": {
                item.name: {"grain": item.transform, "label": "period_start"}
                for item in effective.dimensions
                if item.transform in ("day", "month", "year")
            },
            "cumulative_period_aggregation": {
                item.name: catalog.metrics[item.metric].cumulative.period_aggregation
                for item in effective.metrics
                if catalog.metrics[item.metric].cumulative
            },
        },
        planner_artifacts={
            "semantic_manifest": resolved_manifest,
            "source_preparation_sql": prepared.source_queries,
            "dataflow_plan": text,
            "metricflow_sql": raw_sql,
            "datafusion_sql": query.sql(dialect="postgres", comments=False),
        },
    )


def compile_metricflow_query(plan, catalog, *, population=None):
    with _COMPILE_LOCK:
        try:
            return _compile(plan, catalog, population=population)
        except MetricFlowUnsupported as error:
            pending = [item.metric for item in plan.metrics]
            visited = set()
            has_cumulative = False
            while pending:
                name = pending.pop()
                if name in visited or name not in catalog.metrics:
                    continue
                visited.add(name)
                spec = catalog.metrics[name]
                has_cumulative |= spec.cumulative is not None
                if spec.formula:

                    def refs(node):
                        if node.op == "ref":
                            pending.append(node.name)
                        for arg in node.args:
                            refs(arg)

                    refs(spec.formula)
            if catalog.planner.unsupported != "catalog" or has_cumulative:
                fail("METRICFLOW_UNSUPPORTED", str(error))
            compiled = compile_catalog_query(plan, catalog, population=population)
            compiled.planner_receipt = {
                "backend": "catalog",
                "requested_backend": "metricflow",
                "fallback_reason": str(error),
            }
            return compiled
        except SemanticQueryError:
            raise
        except Exception as error:
            fail(
                "METRICFLOW_PLANNING_FAILED",
                str(error),
                error_type=type(error).__name__,
            )
