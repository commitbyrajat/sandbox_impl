"""Projection feedback and population-preserving review-group regression tests."""

import json
import os
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError
from pydantic_ai import ModelRetry
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel
from test_agent import executable_plan, structured_response, tool_returns

from datasource_mapping import resolve_scope
from domain_instructions import get_domain_profile
from main import create_context
from pipeline import Pipeline, PipelineModels, PipelineRequest
from planning import PlannerDiagnostics, planner_output_type
from result_contracts import PopulationContract
from semantic_models import QueryIR
from semantic_query import SemanticCatalog, SemanticQueryError, compile_query

QUESTION = "Split customers into three review groups with roughly equal January–June 2026 approved INR net spending. Show members and totals."


def review_query():
    return {
        # The population contract, not this preferred fact root, owns retention.
        "base_table": "cc_transactions",
        "dimensions": [
            {
                "name": "customer_id",
                "field": {"table": "cc_transactions", "column": "customer_id"},
            },
            {
                "name": "full_name",
                "field": {"table": "cc_customers", "column": "full_name"},
            },
        ],
        "metrics": [{"name": "total_spend_inr", "metric": "approved_net_spending"}],
        "relationships": ["transaction_customer"],
        "filters": [
            {
                "field": {"table": "cc_transactions", "column": "status"},
                "operator": "eq",
                "value": "APPROVED",
            },
            {
                "field": {"table": "cc_transactions", "column": "currency"},
                "operator": "eq",
                "value": "INR",
            },
        ],
        "time": {"selection": "range", "start": "2026-01-01", "end": "2026-07-01"},
    }


def catalog():
    return SemanticCatalog(
        resolve_scope("1", "1").workspace.sources,
        get_domain_profile("1", tenant_id="1"),
    )


class ProjectionAliasTests(unittest.TestCase):
    def test_empty_output_is_not_reported_as_an_alias_collision(self):
        for calculations in (
            [],
            [{"name": "x", "expression": {"op": "literal", "value": 1}}],
        ):
            with (
                self.subTest(calculations=calculations),
                self.assertRaises(ValidationError) as raised,
            ):
                QueryIR.model_validate(
                    {"base_table": "cc_customers", "calculations": calculations}
                )
            error = raised.exception.errors(include_url=False)[0]
            self.assertEqual(error["type"], "query_outputs_empty")
            self.assertIn("both empty", error["msg"])

    def test_only_equivalent_projections_are_normalized(self):
        value = review_query()
        value["dimensions"].append(dict(value["dimensions"][0]))
        value["metrics"].append(dict(value["metrics"][0]))
        value["calculations"] = [
            {
                "name": "total_spend_inr",
                "expression": {"op": "ref", "name": "total_spend_inr"},
            }
        ]
        query = QueryIR.model_validate(value)
        self.assertEqual(len(query.dimensions), 2)
        self.assertEqual(len(query.metrics), 1)
        self.assertEqual(query.calculations, [])
        self.assertEqual(len(query._normalizations), 3)
        canonical = QueryIR.model_validate(query.model_dump())
        self.assertEqual(canonical.model_dump(), query.model_dump())
        self.assertNotIn("_normalizations", query.model_dump())

    def test_conflicting_expressions_keep_the_exact_alias_and_paths_in_error(self):
        cases = [
            (
                "metrics",
                {"name": "customer_id", "metric": "customer_count"},
                "dimensions[0]",
                "metrics[1]",
                "customer_id",
            ),
            (
                "metrics",
                {"name": "total_spend_inr", "metric": "card_count"},
                "metrics[0]",
                "metrics[1]",
                "total_spend_inr",
            ),
            (
                "calculations",
                {
                    "name": "total_spend_inr",
                    "expression": {
                        "op": "multiply",
                        "args": [
                            {"op": "ref", "name": "total_spend_inr"},
                            {"op": "literal", "value": 2},
                        ],
                    },
                },
                "metrics[0]",
                "calculations[0]",
                "total_spend_inr",
            ),
        ]
        for section, item, first, second, alias in cases:
            value = review_query()
            value.setdefault(section, []).append(item)
            with (
                self.subTest(section=section, alias=alias),
                self.assertRaises(ValidationError) as raised,
            ):
                QueryIR.model_validate(value)
            error = raised.exception.errors(include_url=False)[0]
            self.assertEqual(error["type"], "query_alias_collision")
            for part in (first, second, alias):
                self.assertIn(part, error["msg"])
            self.assertIn("conflicts", error["ctx"])

    def test_retry_feedback_includes_actual_aliases_and_output_contract(self):
        value = executable_plan(declared=True)
        value["steps"][1]["logical_query"]["calculations"] = [
            {"name": "n", "expression": {"op": "literal", "value": 1}}
        ]
        wire = planner_output_type(None, catalog())
        with self.assertRaises(ValidationError) as raised:
            wire.model_validate(value)
        diagnostics = PlannerDiagnostics()

        class Context:
            retry = 0

        with self.assertRaises(ModelRetry):
            diagnostics.on_output_validate_error(
                Context(),
                output_context=None,
                output=json.dumps(value),
                error=raised.exception,
            )
        feedback = diagnostics.latest_feedback
        details = feedback.authority["query_output_requirements"][0]
        self.assertEqual(details["step_id"], "s2")
        self.assertEqual(details["required_output_aliases"], ["n"])
        self.assertEqual(details["selected_aliases"]["calculations"], ["n"])
        self.assertIn("metrics[0]", feedback.diagnostic)
        self.assertIn("calculations[0]", feedback.diagnostic)
        self.assertEqual(feedback.retries_remaining, 3)

    def test_wire_restricts_relationships_and_metrics_to_scoped_catalog(self):
        wire = planner_output_type(None, catalog())
        value = executable_plan(declared=True)
        value["steps"][1]["logical_query"]["relationships"] = [
            "cc_transactions.customer_id = cc_customers.customer_id"
        ]
        with self.assertRaises(ValidationError) as raised:
            wire.model_validate(value)
        self.assertIn("transaction_customer", str(raised.exception))
        value["steps"][1]["logical_query"]["relationships"] = ["transaction_customer"]
        value["steps"][1]["logical_query"]["metrics"][0][
            "metric"
        ] = "sw_holdings.market_value"
        with self.assertRaises(ValidationError):
            wire.model_validate(value)

    def test_source_date_filter_cannot_silently_extend_exclusive_range(self):
        value = review_query()
        value["time"]["end"] = "2026-06-30"
        value["filters"].append(
            {
                "field": {
                    "table": "cc_transactions",
                    "column": "transaction_timestamp",
                },
                "operator": "le",
                "value": "2026-06-30T23:59:59Z",
            }
        )
        with self.assertRaisesRegex(SemanticQueryError, "CONFLICTING_TIME_BOUNDARIES"):
            compile_query(QueryIR.model_validate(value), catalog())
        for start, end in [("2026-07-01", "2026-01-01"), ("January", "June")]:
            value = review_query()
            value["time"].update(start=start, end=end)
            with (
                self.subTest(start=start),
                self.assertRaisesRegex(SemanticQueryError, "INVALID_TIME_RANGE"),
            ):
                compile_query(QueryIR.model_validate(value), catalog())

    def test_population_anchor_and_fact_filters_retain_zero_spend_members(self):
        population = PopulationContract(
            table="cc_customers", source_key="customer_id", result_key="customer_id"
        )
        query = QueryIR.model_validate(review_query())
        compiled = compile_query(query, catalog(), population=population)
        context = create_context(source_ids=["credit_cards"])
        compiled.verify_data(context)
        rows = context.sql(compiled.statement).to_pylist()
        self.assertEqual(len(rows), 20)
        self.assertGreater(sum(row["total_spend_inr"] == 0 for row in rows), 0)
        expected = context.sql(
            "SELECT SUM(amount) AS total FROM cc_transactions WHERE status='APPROVED' AND currency='INR' AND transaction_timestamp>=TIMESTAMP '2026-01-01' AND transaction_timestamp<TIMESTAMP '2026-07-01'"
        ).to_pylist()[0]["total"]
        self.assertEqual(sum(row["total_spend_inr"] for row in rows), expected)
        self.assertEqual(compiled.logical_plan["base_table"], "cc_customers")
        self.assertEqual(
            compiled.logical_plan["dimensions"][0]["field"]["table"], "cc_customers"
        )
        self.assertEqual(
            query.base_table, "cc_transactions"
        )  # Caller IR is immutable to compilation.
        value = review_query()
        value["filters"].append(
            {
                "field": {"table": "cc_customers", "column": "city"},
                "operator": "eq",
                "value": "Mumbai",
            }
        )
        with self.assertRaisesRegex(SemanticQueryError, "POPULATION_FILTER_CONFLICT"):
            compile_query(
                QueryIR.model_validate(value), catalog(), population=population
            )


class ReviewGroupPipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_duplicate_projections_run_once_through_sql_monty_and_summary(self):
        value = executable_plan("1", "sql_python", declared=True)
        query = review_query()
        query["metrics"].append(dict(query["metrics"][0]))
        query["calculations"] = [
            {
                "name": "total_spend_inr",
                "expression": {"op": "ref", "name": "total_spend_inr"},
            }
        ]
        value["steps"][1].update(
            logical_query=query,
            output_contract={
                "required_columns": {
                    "customer_id": {"type": "string", "nullable": False},
                    "full_name": {"type": "string", "nullable": False},
                    "total_spend_inr": {"type": "numeric", "nullable": False},
                },
                "grain": ["customer_id"],
                "min_rows": 20,
                "max_rows": 20,
                "population": {
                    "table": "cc_customers",
                    "source_key": "customer_id",
                    "result_key": "customer_id",
                },
            },
        )
        value["steps"][2].update(
            calculation={"kind": "balancing", "method": "largest_first"},
            result_contract={
                "required_fields": {
                    "groups": "list",
                    "groups.*.members": "list",
                    "groups.*.total": "numeric",
                },
                "invariants": [
                    {
                        "kind": "count_equals",
                        "output_path": "groups",
                        "expected_count": 3,
                    },
                    {
                        "kind": "members_equal_input",
                        "output_path": "groups.*.members.*.customer_id",
                        "input_step": "s2",
                        "input_column": "customer_id",
                    },
                    {
                        "kind": "sum_equals_input",
                        "output_path": "groups.*.total",
                        "input_step": "s2",
                        "input_column": "total_spend_inr",
                    },
                ],
            },
        )
        calls = {"plan": 0, "sql": 0, "python": 0, "summary": 0}

        def classifier(messages, info):
            return structured_response(
                info,
                {
                    "intent": "advanced_analysis",
                    "objective": QUESTION,
                    "analysis_kind": "balancing",
                },
            )

        def planner(messages, info):
            calls["plan"] += 1
            return structured_response(info, value)

        code = """from decimal import Decimal
rows = sorted(data['query_1'], key=lambda row: (-row['total_spend_inr'], row['customer_id']))
groups = [{'group_id': i + 1, 'members': [], 'total': Decimal('0')} for i in range(3)]
for row in rows:
    group = min(groups, key=lambda g: (g['total'], g['group_id']))
    group['members'].append({'customer_id': row['customer_id'], 'full_name': row['full_name']})
    group['total'] += row['total_spend_inr']
{'groups': groups, 'grand_total': sum(g['total'] for g in groups), 'gap': max(g['total'] for g in groups) - min(g['total'] for g in groups)}
"""

        def executor(messages, info):
            returns = tool_returns(messages)
            if not returns:
                return ModelResponse(
                    parts=[ToolCallPart("describe_sources", {"step_id": "s1"})]
                )
            if returns[-1].tool_name == "describe_sources":
                calls["sql"] += 1
                return ModelResponse(parts=[ToolCallPart("run_sql", {"step_id": "s2"})])
            if returns[-1].tool_name == "run_sql":
                calls["python"] += 1
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            "run_python",
                            {"step_id": "s3", "result_ids": ["query_1"], "code": code},
                        )
                    ]
                )
            return ModelResponse(parts=[TextPart("Review groups calculated.")])

        def summary(messages, info):
            calls["summary"] += 1
            payload = json.loads(messages[-1].parts[0].content)
            groups = payload["python_results"][0]["groups"]
            text = "\n".join(
                f"Group {group['group_id']}: INR {Decimal(group['total']):,.2f}; members "
                + ", ".join(member["customer_id"] for member in group["members"])
                for group in groups
            )
            return ModelResponse(parts=[TextPart(text)])

        with (
            tempfile.TemporaryDirectory() as folder,
            patch.dict(os.environ, {"AGENT_SUMMARIZATION_ENABLED": "true"}),
        ):
            context = await Pipeline().run(
                PipelineRequest(QUESTION, "1", "1", plan_dir=Path(folder)),
                PipelineModels(
                    default=FunctionModel(executor),
                    classifier=FunctionModel(classifier),
                    planner=FunctionModel(planner),
                    summarizer=FunctionModel(summary),
                ),
            )
        self.assertEqual(calls, {"plan": 1, "sql": 1, "python": 1, "summary": 1})
        result = context.execution.python_runs[0]["output"]
        self.assertEqual(len(result["groups"]), 3)
        members = [
            member["customer_id"]
            for group in result["groups"]
            for member in group["members"]
        ]
        self.assertEqual(len(members), len(set(members)))
        self.assertEqual(len(members), 20)
        self.assertEqual(result["grand_total"], Decimal("2761194.60"))
        self.assertIn("INR", context.answer)
        self.assertEqual(context.execution.sql_metrics[0]["attempt"], 1)
        self.assertTrue(context.execution.python_runs[0]["executed"])
        self.assertIn("summarize", context.completed_stages)


if __name__ == "__main__":
    unittest.main()
