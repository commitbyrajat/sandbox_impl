"""Accept complete SQL evidence, preserve pending steps, and bound input repair."""

import io
import json
import unittest
from contextlib import redirect_stderr
from unittest.mock import patch

from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel
from test_agent import executable_plan, tool_returns, structured_response

from data_access import DataSources
from datasource_mapping import resolve_scope, routing_catalog
from execution import execute_plan
from main import create_context
from planning import ExecutionPlan, QuestionIntent, build_plan
from result_contracts import ResultContractError, SQLResultContract
from runtime import TRACE_ENABLED

FEATURE_CONTRACT = {
    "required_columns": {
        "customer_id": {"type": "string", "nullable": False},
        "full_name": {"type": "string"},
        "month": {"type": "date"},
        "card_count": {"type": "integer", "nullable": False},
        "spending_per_card": {
            "type": "numeric",
            "meaning": "Latest-month approved net spending divided by distinct cards",
        },
        "profitability_per_card": {
            "type": "numeric",
            "meaning": "Latest-month profit divided by distinct cards",
        },
        "opex_per_card": {
            "type": "numeric",
            "meaning": "Latest-month operating cost divided by distinct cards",
        },
    },
    "grain": ["customer_id"],
    "population": {
        "table": "cc_customers",
        "source_key": "customer_id",
        "result_key": "customer_id",
    },
}
FEATURE_SQL = """SELECT c.customer_id,c.full_name,MAX(f.month) AS month,
COUNT(DISTINCT cc.card_id) AS card_count,
CAST(COALESCE(SUM(f.approved_net_spend_inr),0)/NULLIF(COUNT(DISTINCT cc.card_id),0) AS DECIMAL(22,6)) AS spending_per_card,
CAST(COALESCE(SUM(f.profit_inr),0)/NULLIF(COUNT(DISTINCT cc.card_id),0) AS DECIMAL(22,6)) AS profitability_per_card,
CAST(COALESCE(SUM(f.opex_inr),0)/NULLIF(COUNT(DISTINCT cc.card_id),0) AS DECIMAL(22,6)) AS opex_per_card
FROM cc_customers c LEFT JOIN cc_credit_cards cc ON cc.customer_id=c.customer_id
LEFT JOIN cc_monthly_financials f ON f.card_id=cc.card_id
AND f.month=(SELECT MAX(month) FROM cc_monthly_financials)
GROUP BY c.customer_id,c.full_name ORDER BY c.customer_id"""


def sources():
    return DataSources(create_context(source_ids=["credit_cards"]), "1", "1")


def action(name, **args):
    return ModelResponse(parts=[ToolCallPart(name, args)])


def count_plan():
    return ExecutionPlan.model_validate(
        executable_plan("1", "sql_python", declared=True)
    )


def state(info):
    return json.loads(
        info.instructions.split(
            "Current host SQL execution state (latest correction requirements):\n", 1
        )[1]
    )


class CatalogTests(unittest.TestCase):
    def test_both_workspace_catalogs_match_loaded_column_names_and_types(self):
        for tenant, workspace in (("1", "1"), ("2", "2")):
            with self.subTest(workspace=workspace):
                scope = resolve_scope(tenant, workspace)
                data = DataSources(
                    create_context(
                        source_ids=list(scope.workspace.sources),
                        source_definitions=scope.workspace.sources,
                    ),
                    tenant,
                    workspace,
                )
                data.validate_registry_schema()
                catalog = json.loads(routing_catalog(workspace, tenant_id=tenant))
                self.assertTrue(
                    all(source["columns"] for source in catalog["sources"].values())
                )
        fields = json.loads(routing_catalog("1", tenant_id="1"))["sources"][
            "credit_cards"
        ]["columns"]["cc_monthly_financials"]
        self.assertEqual(fields["month"]["type"], "date32[day]")
        self.assertTrue(fields["profit_inr"]["type"].startswith("decimal128("))

    def test_schema_drift_stops_generation_at_schema_inspection(self):
        data = sources()
        source = resolve_scope("1", "1").workspace.sources["credit_cards"]
        columns = {**source.columns, "cc_customers": {**source.columns["cc_customers"]}}
        columns["cc_customers"]["customer_id"] = source.columns["cc_customers"][
            "customer_id"
        ].model_copy(update={"type": "int64"})
        with patch.object(source, "columns", columns):
            with self.assertRaisesRegex(
                ValueError, "schema mismatch for cc_customers.customer_id"
            ):
                data.describe()
        self.assertEqual(data.queries, [])


class ContractTests(unittest.TestCase):
    def test_date_lookup_cannot_be_stored_as_customer_feature_evidence(self):
        data = sources()
        contract = SQLResultContract.model_validate(FEATURE_CONTRACT)
        with self.assertRaisesRegex(ResultContractError, "spending_per_card"):
            data.query(
                "SELECT MAX(month) AS latest_month FROM cc_monthly_financials",
                contract=contract,
            )
        self.assertEqual(data.results, {})
        self.assertEqual(data.queries, [])
        result = data.query(FEATURE_SQL, contract=contract)
        self.assertEqual(result["result_id"], "query_1")
        self.assertEqual(len(result["rows"]), 20)
        self.assertIn(
            "decimal128",
            next(
                f["type"]
                for f in result["column_types"]
                if f["name"] == "spending_per_card"
            ),
        )

    def test_wrong_grain_population_types_nulls_and_row_counts_are_rejected(self):
        cases = [
            (
                "SELECT customer_id FROM cc_credit_cards",
                {
                    "required_columns": {"customer_id": {"type": "string"}},
                    "grain": ["customer_id"],
                },
                "Duplicate observation",
            ),
            (
                "SELECT customer_id FROM cc_customers LIMIT 1",
                {
                    "required_columns": {"customer_id": {"type": "string"}},
                    "grain": ["customer_id"],
                    "population": FEATURE_CONTRACT["population"],
                },
                "Population mismatch",
            ),
            (
                "SELECT customer_id FROM cc_customers",
                {"required_columns": {"customer_id": {"type": "numeric"}}},
                "must be numeric",
            ),
            (
                "SELECT CAST(NULL AS INT) AS n",
                {"required_columns": {"n": {"type": "integer", "nullable": False}}},
                "unexpected nulls",
            ),
            (
                "SELECT customer_id FROM cc_customers",
                {"required_columns": {"customer_id": {}}, "max_rows": 1},
                "Row count",
            ),
        ]
        for sql, contract, error in cases:
            with self.subTest(error=error):
                data = sources()
                with self.assertRaisesRegex(ResultContractError, error):
                    data.query(sql, contract=SQLResultContract.model_validate(contract))
                self.assertEqual(data.results, {})

    def test_population_cannot_query_another_workspace(self):
        contract = SQLResultContract(
            required_columns={"id": {}},
            grain=["id"],
            population={
                "table": "sw_customers",
                "source_key": "customer_id",
                "result_key": "id",
            },
        )
        with self.assertRaisesRegex(ResultContractError, "unavailable workspace"):
            sources().query(
                "SELECT customer_id AS id FROM cc_customers", contract=contract
            )


class ContractPipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_planner_corrects_mismatched_population_entity_keys(self):
        calls = 0

        def respond(messages, info):
            nonlocal calls
            calls += 1
            value = executable_plan(
                "1",
                "sql_python",
                declared=True,
                output_contract=json.loads(json.dumps(FEATURE_CONTRACT)),
            )
            if calls == 1:
                value["steps"][1]["output_contract"]["population"] = {
                    "table": "cc_monthly_financials",
                    "source_key": "card_id",
                    "result_key": "customer_id",
                }
            else:
                self.assertIn(
                    "does not represent the configured identity customer_id",
                    str(messages[-1]),
                )
            return structured_response(info, value)

        plan = await build_plan(
            "Prepare customer features",
            QuestionIntent(
                intent="advanced_analysis", objective="Compare customer observations"
            ),
            tenant_id="1",
            workspace_id="1",
            model=FunctionModel(respond),
        )
        self.assertEqual(calls, 2)
        self.assertEqual(plan.steps[1].output_contract.population.table, "cc_customers")

    async def test_lookup_failure_keeps_sql_pending_then_real_monty_receives_complete_rows(
        self,
    ):
        data = sources()
        plan = ExecutionPlan.model_validate(
            executable_plan(
                "1", "sql_python", declared=True, output_contract=FEATURE_CONTRACT
            )
        )
        calls = 0

        def respond(messages, info):
            nonlocal calls
            calls += 1
            tools = {tool.name for tool in info.function_tools}
            if calls == 1:
                self.assertIn(
                    '"approved_net_spend_inr": {"type": "decimal128', info.instructions
                )
                self.assertEqual(tools, {"describe_sources"})
                return action("describe_sources", step_id="s1")
            if calls == 2:
                guidance = tool_returns(messages)[-1].content["_sql_guidance"]
                self.assertTrue(guidance["registry_verified"])
                return action(
                    "run_sql",
                    step_id="s2",
                    statement="SELECT MAX(month) AS latest_month FROM cc_monthly_financials",
                )
            if calls == 3:
                current = state(info)
                self.assertEqual(current["completed_steps"], ["s1"])
                self.assertEqual(current["sql_correction"]["phase"], "result_contract")
                self.assertEqual(tools, {"run_sql"})
                self.assertEqual(data.results, {})
                return action("run_sql", step_id="s2", statement=FEATURE_SQL)
            if calls == 4:
                current = state(info)
                self.assertEqual(current["sql_inputs"]["query_1"]["row_count"], 20)
                self.assertEqual(
                    current["sql_inputs"]["query_1"]["python_types"][
                        "spending_per_card"
                    ],
                    ["decimal.Decimal"],
                )
                self.assertEqual(tools, {"run_python", "request_sql_repair"})
                return action(
                    "run_python",
                    step_id="s3",
                    result_ids=["query_1"],
                    code="rows = data['query_1']\n{'observations': len(rows), 'spending_sum': sum(r['spending_per_card'] for r in rows)}",
                )
            self.assertEqual(tools, set())
            return ModelResponse(parts=[TextPart("Calculation complete")])

        token = TRACE_ENABLED.set(True)
        output = io.StringIO()
        try:
            with redirect_stderr(output):
                result = await execute_plan(
                    "Prepare per-customer features and calculate a summary",
                    plan,
                    data,
                    FunctionModel(respond),
                )
        finally:
            TRACE_ENABLED.reset(token)
        self.assertEqual(result.completed_steps, ["s1", "s2", "s3"])
        self.assertEqual(len(result.queries), 1)
        self.assertTrue(result.python_runs[0]["executed"])
        self.assertEqual(result.python_runs[0]["output"]["observations"], 20)
        self.assertIn('"phase": "result_contract"', output.getvalue())

    async def test_bounded_repair_invalidates_results_and_rejects_stale_python_ids(
        self,
    ):
        data = sources()
        calls = 0

        def respond(messages, info):
            nonlocal calls
            calls += 1
            if calls == 1:
                return action("describe_sources", step_id="s1")
            if calls == 2:
                return action(
                    "run_sql",
                    step_id="s2",
                    statement="SELECT COUNT(*) AS n FROM cc_customers WHERE customer_id='CUST0001'",
                )
            if calls == 3:
                return action(
                    "request_sql_repair",
                    sql_step_id="s2",
                    reason="The input counted only one customer; this calculation requires the entire registered population",
                )
            if calls == 4:
                self.assertEqual(state(info)["completed_steps"], ["s1"])
                self.assertNotIn("query_1", data.results)
                return action(
                    "run_sql",
                    step_id="s2",
                    statement="SELECT COUNT(*) AS n FROM cc_customers",
                )
            if calls == 5:
                self.assertEqual(set(state(info)["sql_inputs"]), {"query_2"})
                return action(
                    "run_python",
                    step_id="s3",
                    result_ids=["query_1"],
                    code="data['query_1'][0]['n']",
                )
            if calls == 6:
                self.assertIn(
                    "current SQL producer result IDs: ['query_2']",
                    str(tool_returns(messages)[-1].content),
                )
                return action(
                    "request_sql_repair",
                    sql_step_id="s2",
                    reason="Revalidate the population count before the final calculation",
                )
            if calls == 7:
                return action(
                    "run_sql",
                    step_id="s2",
                    statement="SELECT COUNT(*) AS n FROM cc_customers",
                )
            if calls == 8:
                self.assertEqual(state(info)["input_repairs_used"], 2)
                self.assertEqual(
                    {tool.name for tool in info.function_tools}, {"run_python"}
                )
                self.assertEqual(set(data.results), {"query_3"})
                return action(
                    "run_python",
                    step_id="s3",
                    result_ids=["query_3"],
                    code="{'customers': data['query_3'][0]['n']}",
                )
            return ModelResponse(parts=[TextPart("Calculation complete")])

        result = await execute_plan(
            "Count the complete population", count_plan(), data, FunctionModel(respond)
        )
        self.assertEqual(set(result.sql_results), {"query_3"})
        self.assertEqual(len(result.queries), 1)
        self.assertEqual(result.python_runs[0]["output"], {"customers": 20})

    async def test_planner_requires_contracts_before_publication(self):
        calls = 0

        def respond(messages, info):
            nonlocal calls
            calls += 1
            if calls > 1:
                self.assertIn("output_contract", str(messages[-1]))
                self.assertIn("Field required", str(messages[-1]))
            return structured_response(
                info, executable_plan("1", "sql_python", declared=calls > 1)
            )

        result = await build_plan(
            "Prepare a calculation",
            QuestionIntent(
                intent="advanced_analysis",
                objective="Calculate from prepared observations",
            ),
            tenant_id="1",
            workspace_id="1",
            model=FunctionModel(respond),
        )
        self.assertEqual(calls, 2)
        self.assertIsNotNone(result.steps[1].output_contract)
        self.assertEqual(result.steps[2].input_steps, ["s2"])


if __name__ == "__main__":
    unittest.main()
