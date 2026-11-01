"""Numerical golden queries and typed-tool integration; no model API required."""

import json
import unittest
from datetime import date
from decimal import Decimal

import pyarrow as pa
from datafusion import SessionContext
from pydantic import ValidationError
from pydantic_ai.messages import (
    ModelResponse,
    TextPart,
    ToolCallPart,
    ModelRequest,
    UserPromptPart,
)
from pydantic_ai.models.function import FunctionModel
from sqlglot import exp, parse_one
from test_agent import executable_plan, tool_returns, structured_response

from datasource_mapping import SourceDefinition, resolve_scope
from domain_instructions import DomainProfile, get_domain_profile
from semantic_models import QueryIR, GrainReconciliation
from semantic_query import SemanticCatalog, SemanticQueryError, compile_query
from sql_diagnostics import preflight_sql
from sql_joins import configured_joins
from main import create_context
from data_access import DataSources
from execution import execute_plan
from planning import ExecutionPlan, PlannerOutput, QuestionIntent, build_plan


def risk_query():
    return QueryIR.model_validate(
        {
            "base_table": "cc_delinquency_monthly",
            "dimensions": [
                {
                    "name": key,
                    "field": {"table": "cc_delinquency_monthly", "column": key},
                }
                for key in ["month", "variant_id", "segment", "currency", "dpd_bucket"]
            ],
            "metrics": [
                {"name": key, "metric": key}
                for key in [
                    "exposure_amount",
                    "probability_of_default",
                    "loss_given_default",
                ]
            ],
            "filters": [
                {
                    "field": {
                        "table": "cc_delinquency_monthly",
                        "column": "dpd_bucket",
                    },
                    "operator": "ne",
                    "value": "DEFAULT",
                }
            ],
            "time": {"selection": "latest_common"},
            "metric_horizons": {"probability_of_default": "one_month"},
        }
    )


def credit_catalog():
    return SemanticCatalog(
        resolve_scope("1", "1").workspace.sources,
        get_domain_profile("1", tenant_id="1"),
    )


class SemanticGoldenTests(unittest.TestCase):
    def replace_rows(self, context, name, rows):
        schema = context.table(name).schema()
        context.deregister_table(name)
        table = pa.Table.from_pylist(rows, schema=schema)
        context.register_record_batches(name, [table.to_batches()])

    def execute(self, query, catalog, context):
        compiled = compile_query(query, catalog)
        schemas = {table: context.table(table).schema() for table in catalog.tables}
        bindings = configured_joins(catalog.sources, catalog.profile.join_semantics)
        preflight_sql(compiled.statement, schemas, context, bindings)
        for _, guard in compiled.guards:
            preflight_sql(guard, schemas, context, bindings)
        compiled.verify_data(context)
        return context.sql(compiled.statement).to_pylist(), compiled

    def test_credit_risk_exact_grain_default_probability_and_exposure(self):
        context = create_context(source_ids=["credit_cards"])
        rows, compiled = self.execute(risk_query(), credit_catalog(), context)
        self.assertEqual(len(rows), 36)
        self.assertEqual({row["month"] for row in rows}, {date(2026, 6, 1)})
        expected = context.sql(
            """SELECT d.month,d.variant_id,d.segment,d.currency,d.dpd_bucket,d.balance_inr AS exposure_amount,d.loss_given_default,
            CAST(SUM(CASE WHEN r.to_bucket='DEFAULT' THEN r.transitioned_balance_inr ELSE 0 END) AS DOUBLE)/NULLIF(CAST(MAX(r.cohort_opening_balance_inr) AS DOUBLE),0) AS probability_of_default
            FROM cc_delinquency_monthly d JOIN cc_roll_rates_monthly r ON d.month=r.month AND d.variant_id=r.variant_id AND d.segment=r.segment AND d.currency=r.currency AND d.dpd_bucket=r.from_bucket
            WHERE d.month=DATE '2026-06-01' AND d.dpd_bucket<>'DEFAULT'
            GROUP BY d.month,d.variant_id,d.segment,d.currency,d.dpd_bucket,d.balance_inr,d.loss_given_default"""
        ).to_pylist()
        key = lambda row: (row["variant_id"], row["segment"], row["dpd_bucket"])
        for actual, golden in zip(sorted(rows, key=key), sorted(expected, key=key)):
            self.assertEqual(actual["exposure_amount"], golden["exposure_amount"])
            self.assertAlmostEqual(
                actual["probability_of_default"], golden["probability_of_default"]
            )
        tree = parse_one(compiled.statement, read="postgres")
        references_opening = lambda node: any(
            column.name == "cohort_opening_balance_inr"
            for column in node.find_all(exp.Column)
        )
        self.assertTrue(
            any(references_opening(node) for node in tree.find_all(exp.Max))
        )
        self.assertFalse(
            any(references_opening(node) for node in tree.find_all(exp.Sum))
        )
        self.assertEqual(compiled.relationships, ["risk_transition"])

    def test_customer_features_keep_population_and_distinct_card_denominator(self):
        query = QueryIR.model_validate(
            {
                "base_table": "cc_customers",
                "dimensions": [
                    {"name": key, "field": {"table": "cc_customers", "column": key}}
                    for key in ["customer_id", "full_name"]
                ],
                "metrics": [
                    {"name": x, "metric": x}
                    for x in [
                        "card_count",
                        "spending_per_card",
                        "profitability_per_card",
                        "opex_per_card",
                    ]
                ],
                "relationships": ["card_customer", "financial_card"],
                "time": {"selection": "latest", "tables": ["cc_monthly_financials"]},
            }
        )
        context = create_context(source_ids=["credit_cards"])
        rows, _ = self.execute(query, credit_catalog(), context)
        self.assertEqual(len(rows), 20)
        self.assertEqual(sum(row["card_count"] for row in rows), 40)
        expected = context.sql(
            """SELECT c.customer_id,COUNT(DISTINCT cc.card_id) AS cards,
            CAST(SUM(f.approved_net_spend_inr) AS DOUBLE)/NULLIF(COUNT(DISTINCT cc.card_id),0) AS spend
            FROM cc_customers c LEFT JOIN cc_credit_cards cc ON c.customer_id=cc.customer_id
            LEFT JOIN cc_monthly_financials f ON cc.card_id=f.card_id AND f.month=DATE '2026-06-01'
            GROUP BY c.customer_id"""
        ).to_pylist()
        by_id = {row["customer_id"]: row for row in expected}
        for row in rows:
            self.assertEqual(row["card_count"], by_id[row["customer_id"]]["cards"])
            self.assertAlmostEqual(
                row["spending_per_card"], by_id[row["customer_id"]]["spend"]
            )

    def test_smartwealth_amc_totals_and_distinct_counts(self):
        scope = resolve_scope("2", "2")
        catalog = SemanticCatalog(
            scope.workspace.sources, get_domain_profile("2", tenant_id="2")
        )
        query = QueryIR.model_validate(
            {
                "base_table": "sw_holdings",
                "dimensions": [
                    {
                        "name": "amc_name",
                        "field": {"table": "sw_amcs", "column": "amc_name"},
                    }
                ],
                "metrics": [
                    {"name": x, "metric": x}
                    for x in [
                        "holding_value",
                        "scheme_count",
                        "folio_count",
                        "valuation_date",
                    ]
                ],
            }
        )
        rows, _ = self.execute(
            query, catalog, create_context(source_ids=["smartwealth"])
        )
        self.assertEqual(len(rows), 6)
        self.assertEqual(sum(row["holding_value"] for row in rows), Decimal("635082"))
        self.assertTrue(
            all(row["scheme_count"] == 3 and row["folio_count"] == 4 for row in rows)
        )

    def test_snapshot_left_joins_keep_customers_without_cards_or_financial_facts(self):
        context = create_context(source_ids=["credit_cards"])
        cards = context.table("cc_credit_cards").to_pylist()
        ids = {row["card_id"] for row in cards if row["customer_id"] == "CUST0001"}
        self.replace_rows(
            context,
            "cc_credit_cards",
            [row for row in cards if row["customer_id"] != "CUST0002"],
        )
        facts = context.table("cc_monthly_financials").to_pylist()
        self.replace_rows(
            context,
            "cc_monthly_financials",
            [row for row in facts if row["card_id"] not in ids],
        )
        query = QueryIR.model_validate(
            {
                "base_table": "cc_customers",
                "dimensions": [
                    {
                        "name": "customer_id",
                        "field": {"table": "cc_customers", "column": "customer_id"},
                    }
                ],
                "metrics": [
                    {"name": name, "metric": name}
                    for name in ["card_count", "spending_per_card"]
                ],
                "relationships": ["card_customer", "financial_card"],
                "time": {"selection": "latest", "tables": ["cc_monthly_financials"]},
            }
        )
        rows, _ = self.execute(query, credit_catalog(), context)
        by_id = {row["customer_id"]: row for row in rows}
        self.assertEqual(len(rows), 20)
        self.assertGreater(by_id["CUST0001"]["card_count"], 0)
        self.assertIsNone(by_id["CUST0001"]["spending_per_card"])
        self.assertEqual(by_id["CUST0002"]["card_count"], 0)
        self.assertIsNone(by_id["CUST0002"]["spending_per_card"])

    def test_snapshot_measures_reject_cross_month_sums(self):
        catalog = credit_catalog()
        value = {
            "base_table": "cc_monthly_financials",
            "metrics": [
                {
                    "name": "anr",
                    "metric": "cc_monthly_financials.average_net_receivables_inr",
                }
            ],
        }
        with self.assertRaisesRegex(SemanticQueryError, "NON_ADDITIVE_MEASURE"):
            compile_query(QueryIR.model_validate(value), catalog)
        value["time"] = {"selection": "latest"}
        rows, _ = self.execute(
            QueryIR.model_validate(value),
            catalog,
            create_context(source_ids=["credit_cards"]),
        )
        self.assertGreater(rows[0]["anr"], 0)

    def synthetic(
        self, *, inconsistent=False, null=False, duplicate=False, no_common=False
    ):
        # Different domain and names demonstrate that the compiler has no risk-table branches.
        a = [
            {"period": date(2026, 1, 1), "id": key, "exposure": Decimal(value)}
            for key, value in [
                ("a", "100"),
                ("b", "200"),
                ("c", "0"),
                ("d", "50"),
                ("e", "70"),
            ]
        ]
        a.append(
            {"period": date(2026, 3, 1), "id": "later", "exposure": Decimal("999")}
        )
        b = [
            {
                "period": date(2026, 1, 1),
                "id": "a",
                "dest": "event",
                "opening": Decimal("100"),
                "moved": Decimal("10"),
            },
            {
                "period": date(2026, 1, 1),
                "id": "a",
                "dest": "other",
                "opening": Decimal("101" if inconsistent else "100"),
                "moved": Decimal("90"),
            },
            {
                "period": date(2026, 1, 1),
                "id": "b",
                "dest": "event",
                "opening": Decimal("200"),
                "moved": None if null else Decimal("40"),
            },
            {
                "period": date(2026, 1, 1),
                "id": "b",
                "dest": "other",
                "opening": Decimal("200"),
                "moved": Decimal("160"),
            },
            {
                "period": date(2026, 1, 1),
                "id": "c",
                "dest": "other",
                "opening": Decimal("0"),
                "moved": Decimal("0"),
            },
            {
                "period": date(2026, 1, 1),
                "id": "d",
                "dest": "other",
                "opening": Decimal("50"),
                "moved": Decimal("50"),
            },
            {
                "period": date(2026, 2, 1),
                "id": "later",
                "dest": "event",
                "opening": Decimal("800"),
                "moved": Decimal("800"),
            },
        ]
        if duplicate:
            b.append(dict(b[0]))
        if no_common:
            for row in b:
                row["period"] = date(2026, 2, 1)
        context = SessionContext()
        tables = {
            "balances": pa.Table.from_pylist(a),
            "changes": pa.Table.from_pylist(b),
        }
        for name, table in tables.items():
            context.register_record_batches(name, [table.to_batches()])
        columns = {
            table: {
                f.name: {"type": str(f.type), "nullable": f.nullable}
                for f in batch.schema
            }
            for table, batch in tables.items()
        }
        source = SourceDefinition.model_validate(
            {
                "kind": "parquet",
                "location": "unused",
                "description": "Synthetic golden dataset",
                "tables": {name: name + ".parquet" for name in tables},
                "columns": columns,
                "relationships": [],
                "notes": [],
                "semantic_tables": {
                    "balances": {
                        "grain": ["period", "id"],
                        "primary_key": ["period", "id"],
                        "snapshot_column": "period",
                        "measures": {"exposure": {"aggregation": "sum"}},
                    },
                    "changes": {
                        "grain": ["period", "id", "dest"],
                        "primary_key": ["period", "id", "dest"],
                        "snapshot_column": "period",
                        "measures": {
                            "opening": {
                                "aggregation": "sum",
                                "non_additive_over": ["dest"],
                                "consistency": "constant_per_preaggregation_grain",
                                "null_policy": "reject",
                            },
                            "moved": {"aggregation": "sum", "null_policy": "reject"},
                        },
                    },
                },
                "relationship_keys": {
                    "movement": {
                        "left_table": "balances",
                        "right_table": "changes",
                        "column_pairs": [["period", "period"], ["id", "id"]],
                        "required_preaggregation": {"changes": ["period", "id"]},
                    }
                },
            }
        )
        profile = DomainProfile.model_validate(
            {
                "description": "Other domain",
                "semantic_metrics": {
                    "exposure": {
                        "description": "Exposure",
                        "table": "balances",
                        "column": "exposure",
                    },
                    "opening": {
                        "description": "One origin balance",
                        "table": "changes",
                        "column": "opening",
                    },
                    "event": {
                        "description": "Event flow",
                        "table": "changes",
                        "column": "moved",
                        "filters": [
                            {
                                "field": {"table": "changes", "column": "dest"},
                                "operator": "eq",
                                "value": "event",
                            }
                        ],
                    },
                    "probability": {
                        "description": "Observed event rate",
                        "horizon": "one_period",
                        "formula": {
                            "op": "divide",
                            "args": [
                                {"op": "ref", "name": "event"},
                                {"op": "ref", "name": "opening"},
                            ],
                        },
                    },
                },
            }
        )
        query = QueryIR.model_validate(
            {
                "base_table": "balances",
                "dimensions": [
                    {"name": "id", "field": {"table": "balances", "column": "id"}}
                ],
                "metrics": [
                    {"name": name, "metric": name}
                    for name in ["exposure", "probability"]
                ],
                "metric_horizons": {"probability": "one_period"},
                "time": {"selection": "latest_common"},
            }
        )
        return context, SemanticCatalog({"test": source}, profile), query

    def test_missing_destination_zero_denominator_and_latest_common_intersection(self):
        context, catalog, query = self.synthetic()
        rows, _ = self.execute(query, catalog, context)
        actual = {row["id"]: row for row in rows}
        self.assertEqual(set(actual), set("abcde"))
        self.assertAlmostEqual(actual["a"]["probability"], 0.1)
        self.assertAlmostEqual(actual["b"]["probability"], 0.2)
        self.assertIsNone(actual["c"]["probability"])
        self.assertEqual(actual["d"]["probability"], 0)
        self.assertIsNone(actual["e"]["probability"])
        self.assertEqual(sum(row["exposure"] for row in rows), Decimal("420"))

    def test_catalog_reconciliation_and_probability_bounds_catch_bad_data(self):
        context, catalog, query = self.synthetic()
        catalog.tables["changes"].reconciliations = [
            GrainReconciliation(
                group_by=["period", "id"], sum_column="moved", constant_column="opening"
            )
        ]
        rows = context.table("changes").to_pylist()
        rows[0]["moved"] = Decimal("11")
        self.replace_rows(context, "changes", rows)
        with self.assertRaisesRegex(SemanticQueryError, "GRAIN_RECONCILIATION_FAILED"):
            self.execute(query, catalog, context)
        context, catalog, query = self.synthetic()
        catalog.metrics["probability"].minimum = 0
        catalog.metrics["probability"].maximum = 1
        rows = context.table("changes").to_pylist()
        rows[0]["moved"] = Decimal("110")
        self.replace_rows(context, "changes", rows)
        with self.assertRaisesRegex(SemanticQueryError, "METRIC_OUT_OF_BOUNDS"):
            self.execute(query, catalog, context)

    def test_data_guards_reject_inconsistent_constants_nulls_duplicate_grain_and_empty_intersection(
        self,
    ):
        for option, code in [
            ("inconsistent", "INCONSISTENT_COHORT_ATTRIBUTE"),
            ("null", "NULL_REQUIRED_MEASURE"),
            ("duplicate", "DUPLICATE_SOURCE_GRAIN"),
            ("no_common", "NO_COMMON_PERIOD"),
        ]:
            with self.subTest(option=option):
                context, catalog, query = self.synthetic(**{option: True})
                with self.assertRaisesRegex(SemanticQueryError, code):
                    self.execute(query, catalog, context)

    def test_missing_metrics_horizon_ambiguity_fanout_and_projection_are_rejected(self):
        catalog = credit_catalog()
        query = risk_query()
        query.metric_horizons = {"probability_of_default": "one_year"}
        with self.assertRaisesRegex(SemanticQueryError, "AMBIGUOUS_HORIZON"):
            compile_query(query, catalog)
        query = QueryIR.model_validate(
            {
                "base_table": "cc_customers",
                "metrics": [{"name": "x", "metric": "unknown_business_definition"}],
            }
        )
        with self.assertRaisesRegex(SemanticQueryError, "UNDEFINED_METRIC"):
            compile_query(query, catalog)
        query = QueryIR.model_validate(
            {
                "base_table": "cc_customers",
                "metrics": [{"name": "x", "metric": "spending"}],
            }
        )
        with self.assertRaisesRegex(SemanticQueryError, "AMBIGUOUS_JOIN_PATH"):
            compile_query(query, catalog)
        query = QueryIR.model_validate(
            {
                "base_table": "cc_credit_cards",
                "metrics": [
                    {"name": "x", "metric": "cc_credit_cards.credit_limit"},
                    {"name": "y", "metric": "spending"},
                ],
            }
        )
        # constant credit limit does not conceal differing values; a true additive parent is unsafe.
        parent = catalog.sources["credit_cards"].model_copy(deep=True)
        parent.semantic_tables["cc_credit_cards"].measures[
            "credit_limit"
        ].aggregation = "sum"
        with self.assertRaisesRegex(SemanticQueryError, "UNSAFE_FANOUT"):
            compile_query(
                query, SemanticCatalog({"credit_cards": parent}, catalog.profile)
            )
        query = risk_query()
        query = QueryIR.model_validate(
            {
                **query.model_dump(),
                "dimensions": [
                    {
                        "name": "dest",
                        "field": {
                            "table": "cc_roll_rates_monthly",
                            "column": "to_bucket",
                        },
                    }
                ],
            }
        )
        with self.assertRaisesRegex(
            SemanticQueryError, "PREAGGREGATION_GRAIN_CONFLICT"
        ):
            compile_query(query, catalog)

    def test_ir_cannot_override_aggregation_or_supply_sql_and_calculations_use_separate_scope(
        self,
    ):
        with self.assertRaises(ValidationError):
            QueryIR.model_validate(
                {
                    "base_table": "cc_customers",
                    "statement": "SELECT 1",
                    "metrics": [
                        {"name": "x", "metric": "customer_count", "aggregation": "sum"}
                    ],
                }
            )
        query = QueryIR.model_validate(
            {
                "base_table": "cc_customers",
                "metrics": [{"name": "n", "metric": "customer_count"}],
                "calculations": [
                    {
                        "name": "twice",
                        "expression": {
                            "op": "multiply",
                            "args": [
                                {"op": "ref", "name": "n"},
                                {"op": "literal", "value": 2},
                            ],
                        },
                    },
                    {
                        "name": "difference",
                        "expression": {
                            "op": "subtract",
                            "args": [
                                {"op": "ref", "name": "twice"},
                                {"op": "ref", "name": "n"},
                            ],
                        },
                    },
                ],
            }
        )
        rows, _ = self.execute(
            query, credit_catalog(), create_context(source_ids=["credit_cards"])
        )
        self.assertEqual(rows, [{"n": 20, "twice": 40.0, "difference": 20.0}])
        value = query.model_dump()
        value["calculations"][0]["expression"]["args"][0]["name"] = "not_projected"
        with self.assertRaisesRegex(SemanticQueryError, "UNKNOWN_EXPRESSION_REF"):
            compile_query(QueryIR.model_validate(value), credit_catalog())


class SemanticToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_structured_query_repair_reaches_monty_with_native_typed_data(self):
        plan = ExecutionPlan.model_validate(
            {
                **executable_plan("1", "sql_python", declared=True),
                "query_language": "semantic",
            }
        )
        plan.steps[1].logical_query = None  # Explicit manual semantic-query API.
        calls = 0

        def model(messages, info):
            nonlocal calls
            calls += 1
            if calls == 1:
                return ModelResponse(
                    parts=[ToolCallPart("describe_sources", {"step_id": "s1"})]
                )
            if calls in (2, 3):
                schema = next(
                    tool for tool in info.function_tools if tool.name == "run_sql"
                ).parameters_json_schema
                self.assertNotIn("statement", schema["properties"])
                self.assertIn("query", schema["required"])
                if calls == 3:
                    self.assertIn(
                        "UNDEFINED_METRIC", str(tool_returns(messages)[-1].content)
                    )
                    self.assertIn("UNDEFINED_METRIC", str(messages[-1].parts))
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            "run_sql",
                            {
                                "step_id": "s2",
                                "query": {
                                    "base_table": "cc_customers",
                                    "metrics": [
                                        {
                                            "name": "n",
                                            "metric": (
                                                "not_defined"
                                                if calls == 2
                                                else "customer_count"
                                            ),
                                        }
                                    ],
                                },
                            },
                        )
                    ]
                )
            if calls == 4:
                self.assertIn("HOST_DATA_OBJECT_JSON_BEGIN", info.instructions)
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            "run_python",
                            {
                                "step_id": "s3",
                                "result_ids": ["query_1"],
                                "code": "{'count': data['query_1'][0]['n']}",
                            },
                        )
                    ]
                )
            return ModelResponse(parts=[TextPart("20 customers.")])

        result = await execute_plan(
            "Count customers",
            plan,
            DataSources(create_context(source_ids=["credit_cards"]), "1", "1"),
            FunctionModel(model),
        )
        self.assertEqual(result.python_runs[0]["output"], {"count": 20})
        self.assertEqual(len(result.queries), 1)
        self.assertEqual(result.completed_steps, ["s1", "s2", "s3"])
        self.assertEqual(
            [item["logical_plan_valid"] for item in result.sql_metrics], [False, True]
        )
        self.assertTrue(result.sql_metrics[-1]["datafusion_compiled"])
        self.assertTrue(result.sql_metrics[-1]["output_contract_valid"])
        self.assertIn("catalog_sha256", result.validation_receipts[0])
        self.assertTrue(result.python_runs[0]["executed"])

    async def test_credit_loss_ir_to_monty_scipy_matches_independent_seeded_reference(
        self,
    ):
        import numpy as np
        from scipy import stats

        value = executable_plan("1", "sql_python", declared=True)
        query = risk_query()
        query.order_by = []
        value["query_language"] = "semantic"
        value["steps"][1]["logical_query"] = query.model_dump()
        value["steps"][1]["output_contract"] = {
            "required_columns": {
                key: {
                    "type": (
                        "date"
                        if key == "month"
                        else (
                            "numeric"
                            if key
                            in {
                                "exposure_amount",
                                "probability_of_default",
                                "loss_given_default",
                            }
                            else "string"
                        )
                    ),
                    "nullable": False,
                }
                for key in [
                    "month",
                    "variant_id",
                    "segment",
                    "currency",
                    "dpd_bucket",
                    "exposure_amount",
                    "probability_of_default",
                    "loss_given_default",
                ]
            },
            "grain": ["month", "variant_id", "segment", "currency", "dpd_bucket"],
            "min_rows": 36,
            "max_rows": 36,
        }
        plan = ExecutionPlan.model_validate(value)
        calls = 0
        code = """from decimal import Decimal
import scientific
rows = data['query_1']
probabilities = [float(row['probability_of_default']) for row in rows]
draws = await scientific.sample('bernoulli', {'p': probabilities}, size=[1000, len(rows)], seed=42)
losses = []
for scenario in draws['samples']:
    loss = Decimal('0')
    for i in range(len(rows)):
        if scenario[i]:
            loss += rows[i]['exposure_amount'] * rows[i]['loss_given_default']
    losses.append(loss)
ordered = sorted(losses)
{'expected_loss_inr': sum(losses) / 1000, 'loss_p95_inr': ordered[949], 'method': draws['method']}
"""

        def model(messages, info):
            nonlocal calls
            calls += 1
            if calls == 1:
                return ModelResponse(
                    parts=[ToolCallPart("describe_sources", {"step_id": "s1"})]
                )
            if calls == 2:
                return ModelResponse(parts=[ToolCallPart("run_sql", {"step_id": "s2"})])
            if calls == 3:
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            "run_python",
                            {"step_id": "s3", "result_ids": ["query_1"], "code": code},
                        )
                    ]
                )
            return ModelResponse(
                parts=[TextPart("Completed conditional block-default simulation.")]
            )

        result = await execute_plan(
            "Forecast conditional losses",
            plan,
            DataSources(create_context(source_ids=["credit_cards"]), "1", "1"),
            FunctionModel(model),
        )
        rows = result.sql_results["query_1"]["rows"]
        draws = stats.bernoulli.rvs(
            p=np.array([row["probability_of_default"] for row in rows]),
            size=(1000, len(rows)),
            random_state=np.random.default_rng(np.random.SeedSequence([42, 0])),
        )
        losses = [
            sum(
                (
                    rows[i]["exposure_amount"] * rows[i]["loss_given_default"]
                    for i in range(len(rows))
                    if scenario[i]
                ),
                Decimal(0),
            )
            for scenario in draws
        ]
        actual = result.python_runs[0]["output"]
        self.assertEqual(Decimal(str(actual["expected_loss_inr"])), sum(losses) / 1000)
        self.assertEqual(Decimal(str(actual["loss_p95_inr"])), sorted(losses)[949])
        self.assertTrue(result.python_runs[0]["executed"])
        self.assertEqual(len(result.python_runs[0]["scientific_calls"]), 1)
        self.assertTrue(result.sql_metrics[0]["first_attempt"])

    async def test_semantic_plan_feedback_supersedes_previous_schema_error(self):
        calls = 0

        def model(messages, info):
            nonlocal calls
            calls += 1
            value = executable_plan(declared=True)
            if calls == 1:
                value["steps"][1].pop("logical_query")
            elif calls == 2:
                value["steps"][1]["logical_query"]["metrics"][0][
                    "metric"
                ] = "spending"  # Known metric, but two equally short join paths.
            elif calls == 3:
                prompts = [
                    part.content
                    for message in messages
                    if isinstance(message, ModelRequest)
                    for part in message.parts
                    if isinstance(part, UserPromptPart)
                    and isinstance(part.content, str)
                    and part.content.startswith("Host validation repair handoff.")
                ]
                self.assertEqual(len(prompts), 1)
                self.assertIn("AMBIGUOUS_JOIN_PATH", prompts[0])
                self.assertIn("step_id='s2'", prompts[0])
                self.assertNotIn("Field required", prompts[0])
            return structured_response(info, value)

        result = await build_plan(
            "Count customers",
            QuestionIntent(intent="aggregation", objective="Count customers"),
            tenant_id="1",
            workspace_id="1",
            model=FunctionModel(model),
        )
        self.assertEqual(calls, 3)
        self.assertEqual(
            result.steps[1].logical_query.metrics[0].metric, "cc_customers.__count"
        )

    async def test_published_logical_query_executes_without_regeneration(self):
        plan = PlannerOutput.model_validate(executable_plan(declared=True))
        calls = 0

        def model(messages, info):
            nonlocal calls
            calls += 1
            if calls == 1:
                return ModelResponse(
                    parts=[ToolCallPart("describe_sources", {"step_id": "s1"})]
                )
            if calls == 2:
                schema = info.function_tools[0].parameters_json_schema
                self.assertEqual(set(schema["properties"]), {"step_id"})
                return ModelResponse(parts=[ToolCallPart("run_sql", {"step_id": "s2"})])
            self.assertEqual(
                json.loads(tool_returns(messages)[-1].content)["rows"], [{"n": 20}]
            )
            return ModelResponse(parts=[TextPart("20 customers.")])

        result = await execute_plan(
            "Count customers",
            plan,
            DataSources(create_context(source_ids=["credit_cards"]), "1", "1"),
            FunctionModel(model),
        )
        self.assertEqual(len(result.queries), 1)
        self.assertTrue(result.sql_metrics[0]["first_attempt"])
        self.assertEqual(
            result.validation_receipts[0]["query_ir"],
            plan.steps[1].logical_query.model_dump(),
        )

    def test_scalar_aggregate_contract_has_no_entity_population(self):
        value = executable_plan(declared=True)
        value["steps"][1]["output_contract"].update(
            grain=[],
            population={
                "table": "cc_customers",
                "source_key": "customer_id",
                "result_key": "n",
            },
        )
        plan = PlannerOutput.model_validate(value)
        contract = plan.steps[1].output_contract
        self.assertEqual(contract.grain, [])
        self.assertIsNone(contract.population)
        self.assertEqual((contract.min_rows, contract.max_rows), (1, 1))

    def test_host_orders_producers_before_consumers_and_adds_schema_without_changing_ir(
        self,
    ):
        value = executable_plan("1", "sql_python", declared=True)
        original = value["steps"][1]["logical_query"]
        value["steps"] = [value["steps"][2], value["steps"][1]]
        plan = PlannerOutput.model_validate(value)
        self.assertEqual(
            [step.action for step in plan.steps], ["schema", "sql", "python"]
        )
        self.assertEqual(
            plan.steps[1].logical_query.model_dump(exclude_defaults=True),
            QueryIR.model_validate(original).model_dump(exclude_defaults=True),
        )
        value["steps"][0]["input_steps"] = ["unknown_producer"]
        with self.assertRaises(ValidationError):
            PlannerOutput.model_validate(value)

    def test_live_planner_schema_requires_semantics_and_old_plans_remain_compatible(
        self,
    ):
        self.assertEqual(
            PlannerOutput.model_validate(executable_plan(declared=True)).query_language,
            "semantic",
        )
        self.assertEqual(
            ExecutionPlan.model_validate(executable_plan()).query_language, "legacy_sql"
        )
        with self.assertRaises(ValidationError):
            PlannerOutput.model_validate(
                {**executable_plan(declared=True), "query_language": "legacy_sql"}
            )


if __name__ == "__main__":
    unittest.main()
