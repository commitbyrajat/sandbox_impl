"""Real MetricFlow planning and DataFusion numerical/tenant boundary checks."""

import hashlib
import json
import stat
import unittest
from datetime import date
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pyarrow as pa
from datafusion import SessionContext
from pydantic import ValidationError
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel
from test_agent import executable_plan, tool_returns
import test_semantic_query as fixtures
from semantic_fixtures import count_query, holdings_query

from data_access import DataSources
from datasource_mapping import SourceDefinition, resolve_scope
from domain_instructions import DomainProfile, get_domain_profile
from execution import execute_plan
from main import create_context
from metricflow_adapter import CompileOnlyClient, MetricFlowEngine, entity_expr, mf_name
from planning import ExecutionPlan
from semantic_models import (
    MetricDefinition,
    MetricSelection,
    QueryIR,
    SemanticPlannerConfig,
)
from semantic_query import SemanticCatalog, SemanticQueryError, compile_query
from sql_diagnostics import SQLPreflightError, preflight_sql
from sql_joins import configured_joins


def mf_catalog(tenant="1", *, fallback="error"):
    scope = resolve_scope(tenant, tenant)
    return SemanticCatalog(
        scope.workspace.sources,
        get_domain_profile(tenant, tenant_id=tenant),
        SemanticPlannerConfig(backend="metricflow", unsupported=fallback),
    )


def run_query(query, catalog, context):
    compiled = compile_query(query, catalog)
    schemas = {table: context.table(table).schema() for table in catalog.tables}
    preflight_sql(
        compiled.statement,
        schemas,
        context,
        configured_joins(catalog.sources, catalog.profile.join_semantics),
    )
    compiled.verify_data(context)
    return context.sql(compiled.statement).to_pylist(), compiled


class MetricFlowGoldenTests(unittest.TestCase):
    def test_metricflow_entity_join_matches_amc_holdings_and_distinct_counts(self):
        query = QueryIR.model_validate(holdings_query())
        query.metrics.extend(
            [
                MetricSelection(name="schemes", metric="scheme_count"),
                MetricSelection(name="folios", metric="folio_count"),
            ]
        )
        query = QueryIR.model_validate(query.model_dump())
        rows, compiled = run_query(
            query, mf_catalog("2"), create_context(source_ids=["smartwealth"])
        )
        self.assertEqual(len(rows), 6)
        self.assertEqual(sum(row["value"] for row in rows), Decimal("635082"))
        self.assertTrue(all(row["schemes"] == 3 and row["folios"] == 4 for row in rows))
        self.assertEqual(compiled.planner_receipt["backend"], "metricflow")
        self.assertIn(
            "JoinOnEntitiesNode", compiled.planner_receipt["dataflow_node_types"]
        )
        self.assertTrue(compiled.planner_artifacts["dataflow_plan"])
        self.assertNotIn("cc_transactions", json.dumps(compiled.planner_artifacts))

    def test_probability_by_variant_segment_uses_native_ratio_and_preaggregated_origins(
        self,
    ):
        query = fixtures.risk_query()
        query.dimensions = [
            d for d in query.dimensions if d.name in ("variant_id", "segment")
        ]
        query.metrics = [
            item for item in query.metrics if item.metric == "probability_of_default"
        ]
        # The metric definition itself excludes absorbing default origins.
        query.filters = []
        context = create_context(source_ids=["credit_cards"])
        rows, compiled = run_query(query, mf_catalog(), context)
        expected = context.sql("""WITH origins AS (
            SELECT variant_id,segment,from_bucket,MAX(cohort_opening_balance_inr) AS opening,
            SUM(CASE WHEN to_bucket='DEFAULT' THEN transitioned_balance_inr ELSE 0 END) AS defaults
            FROM cc_roll_rates_monthly WHERE month=DATE '2026-06-01' AND from_bucket<>'DEFAULT'
            GROUP BY variant_id,segment,from_bucket)
            SELECT variant_id,segment,CAST(SUM(defaults) AS DOUBLE)/CAST(SUM(opening) AS DOUBLE) AS probability_of_default
            FROM origins GROUP BY variant_id,segment""").to_pylist()
        key = lambda row: (row["variant_id"], row["segment"])
        self.assertEqual(len(rows), 12)
        for actual, golden in zip(sorted(rows, key=key), sorted(expected, key=key)):
            self.assertEqual(key(actual), key(golden))
            self.assertAlmostEqual(
                actual["probability_of_default"], golden["probability_of_default"]
            )
        manifest = compiled.planner_artifacts["semantic_manifest"]
        self.assertIn("ratio", {metric["type"] for metric in manifest["metrics"]})
        self.assertIn("MAX(", compiled.source_queries["cc_roll_rates_monthly"])
        self.assertEqual(
            compiled.preaggregated_tables["cc_roll_rates_monthly"],
            ["month", "variant_id", "segment", "currency", "from_bucket"],
        )

    def test_ratio_across_financial_and_card_models_preserves_distinct_denominator(
        self,
    ):
        query = QueryIR.model_validate(
            {
                "base_table": "cc_monthly_financials",
                "dimensions": [
                    {
                        "name": "customer_id",
                        "field": {"table": "cc_credit_cards", "column": "customer_id"},
                    }
                ],
                "metrics": [{"name": "spend_per_card", "metric": "spending_per_card"}],
                "relationships": ["financial_card"],
                "time": {"selection": "latest", "tables": ["cc_monthly_financials"]},
            }
        )
        context = create_context(source_ids=["credit_cards"])
        rows, compiled = run_query(query, mf_catalog(), context)
        expected = context.sql(
            "SELECT c.customer_id, CAST(SUM(f.approved_net_spend_inr) AS DOUBLE)/COUNT(DISTINCT c.card_id) AS spend_per_card FROM cc_credit_cards c LEFT JOIN cc_monthly_financials f ON c.card_id=f.card_id AND f.month=DATE '2026-06-01' GROUP BY c.customer_id"
        ).to_pylist()
        by_id = {row["customer_id"]: row for row in expected}
        self.assertEqual(len(rows), 20)
        for row in rows:
            self.assertAlmostEqual(
                row["spend_per_card"], by_id[row["customer_id"]]["spend_per_card"]
            )
        self.assertEqual(compiled.planner_receipt["backend"], "metricflow")
        self.assertIn(
            "CombineAggregatedOutputsNode",
            compiled.planner_receipt["dataflow_node_types"],
        )

    def test_full_risk_inputs_match_exposures_and_keep_null_rates(self):
        fixture = fixtures.SemanticGoldenTests()
        context, catalog, query = fixture.synthetic()
        catalog.planner = SemanticPlannerConfig(backend="metricflow")
        rows, compiled = run_query(query, catalog, context)
        actual = {row["id"]: row for row in rows}
        self.assertEqual(set(actual), set("abcde"))
        self.assertEqual(sum(row["exposure"] for row in rows), Decimal("420"))
        self.assertAlmostEqual(actual["a"]["probability"], 0.1)
        self.assertAlmostEqual(actual["b"]["probability"], 0.2)
        self.assertIsNone(actual["c"]["probability"])
        self.assertEqual(actual["d"]["probability"], 0)
        self.assertIsNone(actual["e"]["probability"])
        self.assertEqual(compiled.planner_receipt["backend"], "metricflow")
        context, catalog, query = fixture.synthetic(inconsistent=True)
        catalog.planner = SemanticPlannerConfig(backend="metricflow")
        with self.assertRaisesRegex(
            SemanticQueryError, "INCONSISTENT_COHORT_ATTRIBUTE"
        ):
            run_query(query, catalog, context)

    def test_month_end_cumulative_spend_matches_independent_running_sums(self):
        query = QueryIR.model_validate(
            {
                "base_table": "cc_transactions",
                "dimensions": [
                    {
                        "name": "month",
                        "field": {
                            "table": "cc_transactions",
                            "column": "transaction_timestamp",
                        },
                        "transform": "month",
                    }
                ],
                "metrics": [
                    {
                        "name": "running_spend",
                        "metric": "cumulative_approved_net_spending",
                    }
                ],
                "time": {
                    "selection": "range",
                    "start": "2026-01-01",
                    "end": "2026-07-01",
                },
                "order_by": [{"name": "month"}],
            }
        )
        context = create_context(source_ids=["credit_cards"])
        rows, compiled = run_query(query, mf_catalog(), context)
        monthly = context.sql(
            """SELECT CAST(date_trunc('month',transaction_timestamp) AS DATE) AS month,SUM(amount) AS spend FROM cc_transactions
            WHERE currency='INR' AND status='APPROVED' AND transaction_timestamp>=TIMESTAMP '2026-01-01' AND transaction_timestamp<TIMESTAMP '2026-07-01'
            GROUP BY month ORDER BY month"""
        ).to_pylist()
        running = Decimal(0)
        self.assertEqual(len(rows), 6)
        for actual, period in zip(rows, monthly):
            running += period["spend"]
            self.assertEqual(
                actual, {"month": period["month"], "running_spend": running}
            )
        self.assertEqual(rows[-1]["running_spend"], Decimal("2761194.60"))
        self.assertEqual(
            compiled.planner_receipt["time_groupings"],
            {"month": {"grain": "month", "label": "period_start"}},
        )
        self.assertEqual(
            compiled.planner_receipt["cumulative_period_aggregation"],
            {"running_spend": "last"},
        )
        self.assertIn(
            "JoinOverTimeRangeNode", compiled.planner_receipt["dataflow_node_types"]
        )
        self.assertIn(
            "WindowReaggregationNode", compiled.planner_receipt["dataflow_node_types"]
        )
        self.assertIn(
            "cumulative",
            {
                metric["type"]
                for metric in compiled.planner_artifacts["semantic_manifest"]["metrics"]
            },
        )

    def test_rolling_window_includes_history_before_display_start(self):
        context = SessionContext()
        table = pa.Table.from_pylist(
            [
                {"id": i, "day": date(2026, 1, i), "amount": Decimal(i * 10)}
                for i in [1, 2, 4, 5]
            ]
        )
        context.register_record_batches("events", [table.to_batches()])
        source = SourceDefinition.model_validate(
            {
                "kind": "parquet",
                "location": "unused",
                "description": "Golden window fixture",
                "tables": {"events": "unused.parquet"},
                "columns": {
                    "events": {
                        f.name: {"type": str(f.type), "nullable": f.nullable}
                        for f in table.schema
                    }
                },
                "relationships": [],
                "notes": [],
                "semantic_tables": {
                    "events": {
                        "grain": ["id"],
                        "primary_key": ["id"],
                        "snapshot_column": "day",
                        "measures": {"amount": {"aggregation": "sum"}},
                    }
                },
            }
        )
        profile = DomainProfile.model_validate(
            {
                "description": "Window fixture",
                "semantic_metrics": {
                    "amount": {
                        "description": "Event amount",
                        "table": "events",
                        "column": "amount",
                    },
                    "rolling": {
                        "description": "Three-day sum",
                        "cumulative": {
                            "input_metric": "amount",
                            "window_count": 3,
                            "window_grain": "day",
                        },
                    },
                },
            }
        )
        catalog = SemanticCatalog(
            {"fixture": source}, profile, SemanticPlannerConfig(backend="metricflow")
        )
        query = QueryIR.model_validate(
            {
                "base_table": "events",
                "dimensions": [
                    {
                        "name": "day",
                        "field": {"table": "events", "column": "day"},
                        "transform": "day",
                    }
                ],
                "metrics": [{"name": "amount", "metric": "rolling"}],
                "time": {
                    "selection": "range",
                    "start": "2026-01-03",
                    "end": "2026-01-06",
                },
                "order_by": [{"name": "day"}],
            }
        )
        rows, compiled = run_query(query, catalog, context)
        self.assertEqual(
            rows,
            [
                {"day": date(2026, 1, 3), "amount": Decimal(30)},
                {"day": date(2026, 1, 4), "amount": Decimal(60)},
                {"day": date(2026, 1, 5), "amount": Decimal(90)},
            ],
        )
        self.assertIn(
            "JoinOverTimeRangeNode", compiled.planner_receipt["dataflow_node_types"]
        )

    def test_workspace_two_cumulative_completed_purchases(self):
        query = QueryIR.model_validate(
            {
                "base_table": "sw_transactions",
                "dimensions": [
                    {
                        "name": "month",
                        "field": {
                            "table": "sw_transactions",
                            "column": "transaction_date",
                        },
                        "transform": "month",
                    }
                ],
                "metrics": [{"name": "amount", "metric": "cumulative_purchases"}],
                "time": {
                    "selection": "range",
                    "start": "2026-01-01",
                    "end": "2026-07-01",
                },
                "order_by": [{"name": "month"}],
            }
        )
        context = create_context(source_ids=["smartwealth"])
        rows, _ = run_query(query, mf_catalog("2"), context)
        total = context.sql(
            "SELECT SUM(amount) AS total FROM sw_transactions WHERE status='COMPLETED' AND transaction_type='PURCHASE'"
        ).to_pylist()[0]["total"]
        self.assertEqual(len(rows), 6)
        self.assertTrue(all(row["amount"] == total for row in rows))

    def test_empty_observation_count_is_zero_and_compilation_is_reproducible(self):
        catalog = mf_catalog()
        query = QueryIR.model_validate(count_query("cc_customers"))
        first = compile_query(query, catalog)
        second = compile_query(query, catalog)
        self.assertEqual(first.statement, second.statement)
        self.assertEqual(
            first.planner_receipt["manifest_sha256"],
            second.planner_receipt["manifest_sha256"],
        )
        context = create_context(source_ids=["credit_cards"])
        schema = context.table("cc_customers").schema()
        context.deregister_table("cc_customers")
        context.register_record_batches(
            "cc_customers",
            [
                [
                    pa.RecordBatch.from_arrays(
                        [pa.array([], type=f.type) for f in schema], schema=schema
                    )
                ]
            ],
        )
        rows, _ = run_query(query, catalog, context)
        self.assertEqual(rows, [{"n": 0}])


class MetricFlowBoundaryTests(unittest.TestCase):
    def test_population_fallback_is_explicit_and_strict_mode_rejects_it(self):
        from result_contracts import PopulationContract

        population = PopulationContract(
            table="cc_customers", source_key="customer_id", result_key="customer_id"
        )
        query = QueryIR.model_validate(
            {
                "base_table": "cc_transactions",
                "dimensions": [
                    {
                        "name": "customer_id",
                        "field": {"table": "cc_customers", "column": "customer_id"},
                    }
                ],
                "metrics": [{"name": "amount", "metric": "approved_net_spending"}],
            }
        )
        with self.assertRaisesRegex(SemanticQueryError, "METRICFLOW_UNSUPPORTED"):
            compile_query(query, mf_catalog(), population=population)
        compiled = compile_query(
            query, mf_catalog(fallback="catalog"), population=population
        )
        self.assertEqual(compiled.planner_receipt["backend"], "catalog")
        self.assertEqual(compiled.planner_receipt["requested_backend"], "metricflow")
        self.assertIn("population", compiled.planner_receipt["fallback_reason"])
        self.assertEqual(compiled.planner_artifacts, {})
        self.assertEqual(
            len(
                create_context(source_ids=["credit_cards"])
                .sql(compiled.statement)
                .to_pylist()
            ),
            20,
        )

    def test_scope_violations_and_native_errors_are_not_hidden_by_fallback(self):
        catalog = mf_catalog(fallback="catalog")
        with self.assertRaisesRegex(SemanticQueryError, "MISSING_TABLE_SEMANTICS"):
            compile_query(
                QueryIR(
                    base_table="sw_holdings",
                    metrics=[{"name": "x", "metric": "holding_value"}],
                ),
                catalog,
            )
        query = QueryIR.model_validate(count_query("cc_customers"))
        with patch.object(
            MetricFlowEngine,
            "explain",
            side_effect=RuntimeError("native planner failed"),
        ):
            with self.assertRaisesRegex(
                SemanticQueryError, "METRICFLOW_PLANNING_FAILED"
            ):
                compile_query(query, catalog)
        fake = SimpleNamespace(
            sql_statement=SimpleNamespace(
                sql="SELECT 1 FROM metricflow.foreign_workspace",
                bind_parameter_set=SimpleNamespace(param_items=()),
            ),
            dataflow_plan=SimpleNamespace(structure_text=lambda: "<DataflowPlan/>"),
        )
        with patch.object(MetricFlowEngine, "explain", return_value=fake):
            with self.assertRaisesRegex(SemanticQueryError, "METRICFLOW_SOURCE_SCOPE"):
                compile_query(query, catalog)

    def test_datafusion_gate_still_rejects_unsupported_generated_functions(self):
        catalog = mf_catalog(fallback="catalog")
        query = QueryIR.model_validate(count_query("cc_customers"))
        sql = f"SELECT no_such_function(1) AS {mf_name('m','cc_customers.__count')} FROM metricflow.{mf_name('s','cc_customers')}"
        fake = SimpleNamespace(
            sql_statement=SimpleNamespace(
                sql=sql, bind_parameter_set=SimpleNamespace(param_items=())
            ),
            dataflow_plan=SimpleNamespace(structure_text=lambda: "<DataflowPlan/>"),
        )
        with patch.object(MetricFlowEngine, "explain", return_value=fake):
            compiled = compile_query(query, catalog)
        self.assertEqual(compiled.planner_receipt["backend"], "metricflow")
        context = create_context(source_ids=["credit_cards"])
        with self.assertRaises(SQLPreflightError) as raised:
            preflight_sql(
                compiled.statement,
                {table: context.table(table).schema() for table in catalog.tables},
                context,
            )
        self.assertEqual(raised.exception.diagnostic.phase, "datafusion_compile")

    def test_unbound_native_parameters_are_rejected_without_fallback(self):
        fake = SimpleNamespace(
            sql_statement=SimpleNamespace(
                sql="SELECT :metric_parameter",
                bind_parameter_set=SimpleNamespace(param_items=(object(),)),
            )
        )
        with patch.object(MetricFlowEngine, "explain", return_value=fake):
            with self.assertRaisesRegex(
                SemanticQueryError, "METRICFLOW_BIND_PARAMETERS"
            ):
                compile_query(
                    QueryIR.model_validate(count_query("cc_customers")),
                    mf_catalog(fallback="catalog"),
                )

    def test_cumulative_cannot_fall_back_to_a_non_cumulative_answer(self):
        catalog = mf_catalog(fallback="catalog")
        query = QueryIR.model_validate(
            {
                "base_table": "cc_transactions",
                "metrics": [
                    {"name": "running", "metric": "cumulative_approved_net_spending"}
                ],
            }
        )
        with self.assertRaisesRegex(SemanticQueryError, "METRICFLOW_UNSUPPORTED"):
            compile_query(query, catalog)
        query.time.selection = "range"
        query.time.start = "2025-01-01"
        query.time.end = "2026-07-01"
        query.dimensions = []
        with self.assertRaisesRegex(SemanticQueryError, "METRICFLOW_TIME_SPINE_RANGE"):
            compile_query(query, catalog)

    def test_compound_entity_encoding_preserves_nulls_and_distinguishes_delimiters(
        self,
    ):
        expression = entity_expr(["a", "b"]).sql(dialect="postgres")
        rows = (
            SessionContext()
            .sql(
                f"SELECT {expression} AS key FROM (VALUES ('a|b','c'),('a','b|c'),('',''),(NULL,'c')) AS keys(a,b)"
            )
            .to_pylist()
        )
        self.assertEqual(len({row["key"] for row in rows if row["key"] is not None}), 3)
        self.assertIsNone(rows[-1]["key"])
        with self.assertRaises(ValidationError):
            SemanticPlannerConfig(
                time_spine_start="2027-01-01", time_spine_end="2026-01-01"
            )
        for method in ("query", "execute", "dry_run"):
            with self.assertRaisesRegex(RuntimeError, "DataFusion owns execution"):
                getattr(CompileOnlyClient(), method)("SELECT 1")


class MetricFlowPipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_native_plan_is_published_and_typed_rows_reach_real_monty(self):
        plan = ExecutionPlan.model_validate(
            {
                **executable_plan("1", "sql_python", declared=True),
                "query_language": "semantic",
            }
        )
        calls = 0

        def model(messages, info):
            nonlocal calls
            for part in tool_returns(messages):
                if part.tool_name == "run_sql":
                    payload = json.loads(part.content)
                    self.assertEqual(
                        payload["semantic_planner"]["backend"], "metricflow"
                    )
            calls += 1
            actions = [
                ("describe_sources", {"step_id": "s1"}),
                ("run_sql", {"step_id": "s2"}),
                (
                    "run_python",
                    {
                        "step_id": "s3",
                        "result_ids": ["query_1"],
                        "code": "{'count': data['query_1'][0]['n']}",
                    },
                ),
            ]
            if calls <= len(actions):
                name, args = actions[calls - 1]
                return ModelResponse(parts=[ToolCallPart(name, args)])
            return ModelResponse(parts=[TextPart("20 registered customers.")])

        result = await execute_plan(
            "Count customers",
            plan,
            DataSources(create_context(source_ids=["credit_cards"]), "1", "1"),
            FunctionModel(model),
        )
        self.assertEqual(result.python_runs[0]["output"], {"count": 20})
        self.assertTrue(result.python_runs[0]["executed"])
        receipt = result.validation_receipts[0]["planner"]
        self.assertEqual(receipt["backend"], "metricflow")
        path = Path(receipt["artifact_path"])
        self.assertIn("tenant-1/workspace-1", str(path))
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
        artifact = json.loads(path.read_text())
        self.assertEqual(artifact["datafusion_sql"], result.queries[0])
        self.assertEqual(
            artifact["statement_sha256"],
            hashlib.sha256(result.queries[0].encode()).hexdigest(),
        )
        self.assertTrue(artifact["dataflow_plan"])
        self.assertIn("semantic_models", artifact["semantic_manifest"])
        self.assertEqual(
            receipt["manifest_sha256"],
            hashlib.sha256(
                json.dumps(artifact["semantic_manifest"], sort_keys=True).encode()
            ).hexdigest(),
        )
        self.assertNotIn("sw_holdings", json.dumps(artifact["semantic_manifest"]))
        self.assertEqual(result.sql_metrics[0]["planner_backend"], "metricflow")

    async def test_compilation_publication_failure_stops_before_data_query(self):
        plan = ExecutionPlan.model_validate(
            {**executable_plan(declared=True), "query_language": "semantic"}
        )
        sources = DataSources(create_context(source_ids=["credit_cards"]), "1", "1")

        def model(messages, info):
            return ModelResponse(
                parts=(
                    [ToolCallPart("run_sql", {"step_id": "s2"})]
                    if tool_returns(messages)
                    else [ToolCallPart("describe_sources", {"step_id": "s1"})]
                )
            )

        with patch(
            "execution.publish_query_compilation",
            side_effect=OSError("artifact disk full"),
        ):
            with self.assertRaisesRegex(OSError, "artifact disk full"):
                await execute_plan("Count", plan, sources, FunctionModel(model))
        self.assertEqual(sources.queries, [])


if __name__ == "__main__":
    unittest.main()
