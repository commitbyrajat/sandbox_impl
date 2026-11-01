"""Stage DAG equivalence, type preservation, compilation gates and cleanup."""

from datetime import date
from decimal import Decimal
import copy
import unittest
from unittest.mock import patch

import pyarrow as pa
from datafusion import SessionContext

from data_access import DataSources
from sql_diagnostics import SQLPreflightError
from sql_staging import SQLStagingSettings, plan_sql_stages, staged_frame
from semantic_query import compile_query
from test_semantic_query import credit_catalog, risk_query
from main import create_context
from domain_instructions import get_domain_profile
from observation_planning import bind_observations
from test_agent import executable_plan


class SQLStagingTests(unittest.TestCase):
    def setUp(self):
        self.context = SessionContext()
        self.schema = pa.schema(
            [
                ("id", pa.int64()),
                ("amount", pa.decimal128(18, 2)),
                ("day", pa.date32()),
            ]
        )
        self.context.register_record_batches(
            "facts",
            [
                [
                    pa.RecordBatch.from_pylist(
                        [
                            {
                                "id": i,
                                "amount": Decimal("1.25") if i % 2 else None,
                                "day": date(2026, 6, 1),
                            }
                            for i in range(1000)
                        ],
                        schema=self.schema,
                    )
                ]
            ],
        )
        self.settings = SQLStagingSettings(max_selects=1)

    def execute(self, sql, **settings):
        original = self.context.sql(sql).to_pylist()
        plan = plan_sql_stages(sql, self.settings.model_copy(update=settings))
        with staged_frame(self.context, plan, {"facts": self.schema}) as (
            frame,
            checked,
            receipts,
        ):
            rows = frame.to_pylist()
            self.assertEqual(rows, original)
            self.assertTrue(checked.datafusion_compiled)
            self.assertEqual(len(receipts), len(plan.stages))
        for stage in plan.stages[:-1]:
            self.assertFalse(self.context.table_exist(stage.name))
        return rows, plan, receipts

    def test_small_query_remains_one_statement(self):
        plan = plan_sql_stages("SELECT id FROM facts", SQLStagingSettings())
        self.assertFalse(plan.staged)
        self.assertEqual(plan.stages[0].statement, "SELECT id FROM facts")

    def test_bad_sql_returns_a_repairable_stage_planning_error(self):
        with self.assertRaisesRegex(ValueError, "Cannot generate SQL stages"):
            plan_sql_stages("SELECT id FROM")

    def test_unused_ctes_are_not_materialized_and_volatile_sql_is_rejected(self):
        sql = "WITH unused AS (SELECT id / 0 AS broken FROM facts), a AS (SELECT id FROM facts) SELECT COUNT(*) AS n FROM a"
        plan = plan_sql_stages(sql, self.settings)
        self.assertEqual(len(plan.stages), 2)
        self.assertTrue(all("broken" not in stage.statement for stage in plan.stages))
        with staged_frame(self.context, plan, {"facts": self.schema}) as (frame, _, _):
            self.assertEqual(frame.to_pylist(), [{"n": 1000}])
        with self.assertRaisesRegex(ValueError, "Volatile SQL"):
            plan_sql_stages(
                "WITH a AS (SELECT RANDOM() AS x) SELECT * FROM a", self.settings
            )

    def test_cte_chain_keeps_all_intermediate_rows_and_arrow_types(self):
        sql = """WITH a AS (SELECT id, amount, day FROM facts),
            b AS (SELECT id, amount, day FROM a WHERE id >= 10)
            SELECT COUNT(*) AS n, SUM(amount) AS total, MAX(day) AS day FROM b"""
        rows, plan, receipts = self.execute(sql)
        self.assertEqual(
            rows[0], {"n": 990, "total": Decimal("618.75"), "day": date(2026, 6, 1)}
        )
        self.assertEqual(len(plan.stages), 3)
        self.assertEqual(receipts[0]["rows"], 1000)
        self.assertEqual(plan.stages[-1].dependencies, [plan.stages[1].name])

    def test_empty_materialization_preserves_decimal_and_date_schema(self):
        rows, _, _ = self.execute(
            "WITH a AS (SELECT * FROM facts WHERE id < 0) SELECT amount, day FROM a"
        )
        self.assertEqual(rows, [])

    def test_column_lists_and_shadowed_ctes_preserve_lexical_bindings(self):
        self.execute(
            "WITH a(k, v) AS (SELECT id, amount FROM facts) SELECT SUM(v) AS total FROM a"
        )
        sql = """WITH a AS (SELECT id FROM facts WHERE id < 20),
            b AS (WITH a AS (SELECT id FROM facts WHERE id > 900) SELECT COUNT(*) AS n FROM a)
            SELECT (SELECT COUNT(*) FROM a) AS outer_count, (SELECT n FROM b) AS inner_count"""
        plan = plan_sql_stages(sql, self.settings)
        with staged_frame(self.context, plan, {"facts": self.schema}) as (frame, _, _):
            self.assertEqual(
                frame.to_pylist(), [{"outer_count": 20, "inner_count": 99}]
            )

    def test_order_limit_window_and_correlated_subquery_keep_their_boundaries(self):
        self.execute(
            """WITH a AS (SELECT id, ROW_NUMBER() OVER (ORDER BY id DESC) AS rn FROM facts)
            SELECT id FROM a ORDER BY rn LIMIT 5"""
        )
        self.execute("""WITH a AS (SELECT id FROM facts WHERE id < 5)
            SELECT x.id FROM a x
            WHERE EXISTS (SELECT 1 FROM facts y WHERE y.id = x.id) ORDER BY x.id""")

    def test_entire_dag_compiles_before_any_stage_data_is_read(self):
        sql = "WITH a AS (SELECT id, amount FROM facts) SELECT id, SUM(amount) AS total FROM a"
        plan = plan_sql_stages(sql, self.settings)
        from datafusion import DataFrame

        with patch.object(
            DataFrame, "execute_stream", side_effect=AssertionError("must not execute")
        ):
            with self.assertRaises(SQLPreflightError):
                with staged_frame(self.context, plan, {"facts": self.schema}):
                    self.fail("Invalid aggregation was accepted")
        self.assertFalse(self.context.table_exist(plan.stages[0].name))

    def test_limits_fail_without_truncation_and_always_cleanup(self):
        for updates in ({"max_rows": 10}, {"max_bytes": 10}):
            plan = plan_sql_stages(
                "WITH a AS (SELECT * FROM facts) SELECT COUNT(*) FROM a",
                self.settings.model_copy(update=updates),
            )
            with self.assertRaisesRegex(ValueError, "intermediate row/memory limit"):
                with staged_frame(self.context, plan, {"facts": self.schema}):
                    self.fail("Oversized intermediate was accepted")
            self.assertFalse(self.context.table_exist(plan.stages[0].name))

    def test_request_scope_and_unknown_tables_cannot_be_hidden_in_ctes(self):
        source = DataSources(create_context(source_ids=["credit_cards"]), "1", "1")
        sql = (
            "WITH a AS (SELECT * FROM sw_amcs), b AS (SELECT * FROM a) SELECT * FROM b"
        )
        with self.assertRaisesRegex(ValueError, "specified workspace"):
            source.query(sql, stage_plan=plan_sql_stages(sql, self.settings))
        self.assertEqual(source.results, {})

    def test_real_risk_query_staged_and_single_statement_are_equivalent(self):
        compiled = compile_query(risk_query(), credit_catalog())
        source = DataSources(create_context(source_ids=["credit_cards"]), "1", "1")
        expected = source.context.sql(compiled.statement).to_pylist()
        result = source.query(compiled.statement)
        self.assertEqual(sorted(result["rows"], key=repr), sorted(expected, key=repr))
        self.assertEqual(len(result["rows"]), 36)
        self.assertEqual(len(result["sql_stages"]), 3)
        for receipt in result["sql_stages"][:-1]:
            self.assertFalse(source.context.table_exist(receipt["name"]))

    def test_size_and_stage_count_limits_are_explicit(self):
        sql = (
            "WITH a AS (SELECT id FROM facts), b AS (SELECT id FROM a) SELECT * FROM b"
        )
        with self.assertRaisesRegex(ValueError, "maximum is 2"):
            plan_sql_stages(sql, SQLStagingSettings(max_selects=1, max_stages=2))
        with self.assertRaisesRegex(ValueError, "enable SQL staging"):
            plan_sql_stages("SELECT 1" + " " * 21000, SQLStagingSettings(enabled=False))

    def test_large_statement_is_generated_and_executed_as_small_dependent_stages(self):
        ctes = []
        for index in range(35):
            source = "facts" if index == 0 else f"a{index-1}"
            ctes.append(
                f"a{index} AS (SELECT id, LENGTH('{'x' * 650}') AS n FROM {source})"
            )
        sql = "WITH " + ", ".join(ctes) + " SELECT COUNT(*) AS n FROM a34"
        self.assertGreater(len(sql), 20000)
        plan = plan_sql_stages(sql)
        self.assertTrue(plan.staged)
        self.assertLess(max(len(stage.statement) for stage in plan.stages), 1000)
        with staged_frame(self.context, plan, {"facts": self.schema}) as (frame, _, _):
            self.assertEqual(frame.to_pylist(), [{"n": 1000}])

    def test_native_metricflow_join_plan_retains_results_when_staged(self):
        from test_metricflow_adapter import mf_catalog

        query = risk_query()
        query.dimensions = [
            d for d in query.dimensions if d.name in ("variant_id", "segment")
        ]
        query.metrics = [
            m for m in query.metrics if m.metric == "probability_of_default"
        ]
        query.filters = []
        compiled = compile_query(query, mf_catalog())
        source = DataSources(create_context(source_ids=["credit_cards"]), "1", "1")
        expected = source.context.sql(compiled.statement).to_pylist()
        result = source.query(compiled.statement)
        self.assertEqual(sorted(result["rows"], key=repr), sorted(expected, key=repr))
        self.assertGreater(len(result["sql_stages"]), 1)

    def test_duplicate_identity_labels_group_once_in_metricflow(self):
        from test_metricflow_adapter import mf_catalog
        from semantic_models import Dimension, FieldRef, MetricSelection

        query = risk_query()
        query.filters = []
        query.dimensions = [
            Dimension(
                name="month",
                field=FieldRef(table="cc_delinquency_monthly", column="month"),
            ),
            Dimension(
                name="snapshot_label",
                field=FieldRef(table="cc_delinquency_monthly", column="month"),
            ),
        ]
        query.metrics = [
            MetricSelection(name="losses", metric="exposure_amount"),
            MetricSelection(name="exposure_copy", metric="exposure_amount"),
        ]
        compiled = compile_query(query, mf_catalog())
        source = DataSources(create_context(source_ids=["credit_cards"]), "1", "1")
        result = source.query(compiled.statement)
        self.assertEqual(len(result["rows"]), 1)
        row = result["rows"][0]
        self.assertEqual(row["month"], row["snapshot_label"])
        self.assertEqual(row["losses"], row["exposure_copy"])

    def test_required_month_alias_is_not_lost_to_a_synthetic_identity_label(self):
        profile = get_domain_profile("1", tenant_id="1")
        question = "What could next month's credit losses be? Show expected loss and a loss level exceeded in only 5% of scenarios."
        value = executable_plan("1", "sql_python", declared=True)
        query = risk_query().model_dump()
        query["dimensions"][0]["name"] = "cohort_key"
        step = value["steps"][1]
        step.update(
            logical_query=query,
            output_contract={
                "required_columns": {
                    "month": {"type": "date"},
                    "cohort_key": {"type": "any"},
                },
                "grain": ["month", "variant_id", "segment", "currency", "dpd_bucket"],
            },
        )
        original = copy.deepcopy(value)
        bound = bind_observations(
            value,
            credit_catalog(),
            profile.observation_policies(question),
            "uncertainty",
        )
        dimensions = {
            d["name"]: d["field"]
            for d in bound["steps"][1]["logical_query"]["dimensions"]
        }
        self.assertEqual(
            dimensions["month"], {"table": "cc_delinquency_monthly", "column": "month"}
        )
        self.assertNotIn("cohort_key", dimensions)
        self.assertIn(
            "cohort_key",
            {c["name"] for c in bound["steps"][1]["logical_query"]["calculations"]},
        )
        self.assertIn("month", bound["steps"][1]["output_contract"]["grain"])
        self.assertEqual(value, original)

    def test_declared_composite_identity_and_exact_catalog_output_are_completed(self):
        from planning import planner_output_type

        profile = get_domain_profile("1", tenant_id="1")
        question = "What could next month's credit losses be? Show expected loss and a loss level exceeded in only 5% of scenarios."
        value = executable_plan("1", "sql_python", declared=True)
        query = risk_query().model_dump()
        query["dimensions"][0]["name"] = "cohort_key"
        query["metrics"] = [
            m for m in query["metrics"] if m["metric"] != "probability_of_default"
        ]
        value["steps"][1].update(
            logical_query=query,
            output_contract={
                "required_columns": {
                    "cohort_key": {"type": "string"},
                    "exposure_amount": {"type": "numeric"},
                    "probability_of_default": {"type": "numeric"},
                    "loss_given_default": {"type": "numeric"},
                },
                "grain": ["cohort_key"],
            },
        )
        catalog = credit_catalog()
        bound = planner_output_type(
            "uncertainty", catalog, profile.observation_policies(question)
        ).model_validate(value)
        step = bound.steps[1]
        self.assertIn(
            "probability_of_default", {m.metric for m in step.logical_query.metrics}
        )
        key = next(c for c in step.logical_query.calculations if c.name == "cohort_key")
        self.assertEqual(key.expression.op, "key")
        self.assertEqual(len(key.expression.args), 5)
        compiled = compile_query(step.logical_query, catalog)
        source = DataSources(create_context(source_ids=["credit_cards"]), "1", "1")
        rows = source.query(compiled.statement, contract=step.output_contract)["rows"]
        self.assertEqual(len(rows), 36)
        self.assertEqual(len({r["cohort_key"] for r in rows}), 36)
        self.assertTrue(all(isinstance(r["cohort_key"], str) for r in rows))
        # The key may be requested in addition to an explicit tuple grain.
        value["steps"][1]["output_contract"]["grain"] = [
            "month",
            "variant_id",
            "segment",
            "currency",
            "dpd_bucket",
        ]
        tuple_bound = planner_output_type(
            "uncertainty", catalog, profile.observation_policies(question)
        ).model_validate(value)
        self.assertIn(
            "cohort_key",
            {c.name for c in tuple_bound.steps[1].logical_query.calculations},
        )

    def test_ordinary_alias_in_a_multi_column_grain_is_never_replaced_by_a_key(self):
        profile = get_domain_profile("1", tenant_id="1")
        question = "What could next month's credit losses be? Show expected loss and a loss level exceeded in only 5% of scenarios."
        value = executable_plan("1", "sql_python", declared=True)
        query = risk_query().model_dump()
        query["dimensions"][-1]["name"] = "from_bucket"
        value["steps"][1].update(
            logical_query=query,
            output_contract={
                "required_columns": {
                    d["name"]: {"type": "any"} for d in query["dimensions"]
                },
                "grain": [d["name"] for d in query["dimensions"]],
            },
        )
        bound = bind_observations(
            value,
            credit_catalog(),
            profile.observation_policies(question),
            "uncertainty",
        )
        result = bound["steps"][1]["logical_query"]
        self.assertEqual(result["calculations"], [])
        dimension = next(d for d in result["dimensions"] if d["name"] == "from_bucket")
        self.assertEqual(dimension["field"]["column"], "dpd_bucket")

    def test_composite_key_encoding_is_injective_and_keeps_null_keys_null(self):
        from semantic_models import Formula
        from semantic_query import formula_ast
        from sqlglot import exp

        expression = formula_ast(
            Formula.model_validate(
                {
                    "op": "key",
                    "args": [
                        {"op": "ref", "name": "a"},
                        {"op": "ref", "name": "b"},
                    ],
                }
            ),
            lambda name: exp.column(name),
        ).sql(dialect="postgres")
        sql = f"SELECT {expression} AS k FROM (VALUES ('a|b', 'c'), ('a', 'b|c'), (NULL, 'c')) AS x(a,b)"
        keys = [row["k"] for row in self.context.sql(sql).to_pylist()]
        self.assertNotEqual(keys[0], keys[1])
        self.assertIsNone(keys[2])


if __name__ == "__main__":
    unittest.main()
