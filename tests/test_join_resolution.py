"""Incomplete route hints, actionable planner feedback and exact group presentation."""

import copy
import json
import os
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.models.function import FunctionModel
from test_agent import executable_plan, structured_response, tool_returns
from test_projection_aliases import QUESTION, review_query

from datasource_mapping import resolve_scope
from domain_instructions import get_domain_profile
from main import create_context
from pipeline import Pipeline, PipelineModels, PipelineRequest
from result_contracts import PopulationContract
from semantic_models import QueryIR
from semantic_query import SemanticCatalog, SemanticQueryError, compile_query
from sql_diagnostics import preflight_sql
from sql_joins import configured_joins


def scoped_catalog():
    workspace = resolve_scope("1", "1").workspace
    return SemanticCatalog(
        workspace.sources,
        get_domain_profile("1", tenant_id="1"),
        workspace.semantic_planner,
    )


def financial_query(hints):
    query = review_query()
    query.update(
        base_table="cc_monthly_financials",
        metrics=[{"name": "total_spend_inr", "metric": "spending"}],
        filters=[],
        relationships=hints,
    )
    query["dimensions"][0]["field"]["table"] = "cc_customers"
    query["time"]["tables"] = ["cc_monthly_financials"]
    return QueryIR.model_validate(query)


POPULATION = PopulationContract(
    table="cc_customers", source_key="customer_id", result_key="customer_id"
)


class JoinResolutionTests(unittest.TestCase):
    def test_partial_hint_completes_financial_customer_route_without_changing_values(
        self,
    ):
        query = financial_query(["financial_card"])
        before = query.model_dump()
        compiled = compile_query(query, scoped_catalog(), population=POPULATION)
        self.assertEqual(
            set(compiled.relationships), {"financial_card", "card_customer"}
        )
        self.assertEqual(query.model_dump(), before)
        context = create_context(source_ids=["credit_cards"])
        preflight_sql(
            compiled.statement,
            {name: context.table(name).schema() for name in scoped_catalog().tables},
            context,
            configured_joins(
                scoped_catalog().sources, scoped_catalog().profile.join_semantics
            ),
        )
        compiled.verify_data(context)
        rows = context.sql(compiled.statement).to_pylist()
        expected = context.sql(
            """SELECT c.customer_id, COALESCE(SUM(t.amount),0) AS value
            FROM cc_customers c LEFT JOIN cc_transactions t ON c.customer_id=t.customer_id
            AND t.status='APPROVED' AND t.currency='INR'
            AND t.transaction_timestamp>=TIMESTAMP '2026-01-01'
            AND t.transaction_timestamp<TIMESTAMP '2026-07-01' GROUP BY c.customer_id"""
        ).to_pylist()
        self.assertEqual(
            {r["customer_id"]: r["total_spend_inr"] for r in rows},
            {r["customer_id"]: r["value"] for r in expected},
        )
        self.assertEqual(len(rows), 20)
        self.assertTrue(any(row["total_spend_inr"] == 0 for row in rows))

    def test_incomplete_unrelated_hint_reports_real_alternatives_instead_of_empty_paths(
        self,
    ):
        with self.assertRaises(SemanticQueryError) as raised:
            compile_query(
                financial_query(["transaction_customer"]),
                scoped_catalog(),
                population=POPULATION,
            )
        diagnostic = raised.exception.diagnostic
        self.assertEqual(diagnostic["error_code"], "AMBIGUOUS_JOIN_PATH")
        self.assertEqual(diagnostic["base_table"], "cc_customers")
        self.assertEqual(diagnostic["target_metrics"], ["spending"])
        self.assertEqual(
            {frozenset(path) for path in diagnostic["candidate_relationship_ids"]},
            {
                frozenset(("card_customer", "financial_card")),
                frozenset(("profile_customer", "financial_profile")),
            },
        )

    def test_explicit_business_route_is_preserved_and_ties_are_not_guessed(self):
        compiled = compile_query(
            financial_query(["profile_customer", "financial_profile"]),
            scoped_catalog(),
            population=POPULATION,
        )
        self.assertEqual(
            set(compiled.relationships), {"profile_customer", "financial_profile"}
        )
        with self.assertRaisesRegex(SemanticQueryError, "AMBIGUOUS_JOIN_PATH"):
            compile_query(
                financial_query(["financial_card", "financial_profile"]),
                scoped_catalog(),
                population=POPULATION,
            )

    def test_disconnected_sources_and_unknown_ids_still_fail_closed(self):
        query = QueryIR(
            base_table="cc_customers",
            metrics=[{"name": "exposure", "metric": "exposure_amount"}],
            relationships=["card_customer"],
        )
        with self.assertRaises(SemanticQueryError) as raised:
            compile_query(query, scoped_catalog())
        self.assertEqual(raised.exception.diagnostic["error_code"], "NO_JOIN_PATH")
        self.assertEqual(raised.exception.diagnostic["candidate_relationship_ids"], [])
        self.assertIn("exposure_amount", raised.exception.diagnostic["target_metrics"])
        self.assertNotIn("sw_holdings", json.dumps(raised.exception.diagnostic))
        with self.assertRaisesRegex(SemanticQueryError, "UNKNOWN_RELATIONSHIP"):
            compile_query(financial_query(["made_up_relationship"]), scoped_catalog())


class JoinRepairPipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_feedback_repairs_partial_route_then_sql_and_real_monty_execute_once(
        self,
    ):
        value = executable_plan("1", "sql_python", declared=True)
        value["steps"][1].update(
            logical_query=financial_query(["transaction_customer"]).model_dump(),
            output_contract={
                "required_columns": {
                    "customer_id": {"type": "string", "nullable": False},
                    "full_name": {"type": "string"},
                    "total_spend_inr": {"type": "numeric"},
                },
                "grain": ["customer_id"],
                "population": POPULATION.model_dump(),
                "min_rows": 20,
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
            proposal = copy.deepcopy(value)
            if calls["plan"] > 1:
                handoffs = [
                    part.content
                    for message in messages
                    if isinstance(message, ModelRequest)
                    for part in message.parts
                    if isinstance(part, UserPromptPart)
                    and isinstance(part.content, str)
                    and part.content.startswith("Host validation repair handoff.")
                ]
                self.assertEqual(len(handoffs), 1)
                self.assertIn("candidate_relationship_ids", handoffs[0])
                self.assertIn("financial_card", handoffs[0])
                self.assertIn("selected_metrics", handoffs[0])
                self.assertIn("approved_net_spend_inr", handoffs[0])
                proposal["steps"][1]["logical_query"]["relationships"] = [
                    "financial_card"
                ]
            return structured_response(info, proposal)

        code = """from decimal import Decimal
groups = [{'group':i+1,'members':[],'total':Decimal('0')} for i in range(3)]
for row in sorted(data['query_1'],key=lambda row:(-row['total_spend_inr'],row['customer_id'])):
    group = min(groups,key=lambda item:(item['total'],item['group']))
    group['members'].append({'customer_id':row['customer_id'],'full_name':row['full_name']})
    group['total'] += row['total_spend_inr']
{'groups':groups}
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
            # This draft is intentionally wrong; it must never replace retained membership.
            return ModelResponse(parts=[TextPart("Group 3: CUST0006–CUST0020")])

        def summary(messages, info):
            calls["summary"] += 1
            raise AssertionError("Summarization is disabled")

        with (
            tempfile.TemporaryDirectory() as folder,
            patch.dict(os.environ, {"AGENT_SUMMARIZATION_ENABLED": "false"}),
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
        self.assertEqual(calls, {"plan": 2, "sql": 1, "python": 1, "summary": 0})
        result = context.execution.python_runs[0]["output"]
        rendered = json.loads(
            context.answer.removeprefix("```json\n").removesuffix("\n```")
        )
        self.assertEqual(rendered, json.loads(json.dumps(result, default=str)))
        self.assertNotIn("CUST0006–CUST0020", context.answer)
        members = [m["customer_id"] for g in result["groups"] for m in g["members"]]
        self.assertEqual(len(members), 20)
        self.assertEqual(len(set(members)), 20)
        self.assertEqual(
            [g["total"] for g in result["groups"]],
            [Decimal("966250.70"), Decimal("946318.10"), Decimal("848625.80")],
        )
        self.assertTrue(context.execution.python_runs[0]["executed"])
        self.assertEqual(context.execution.sql_metrics[0]["attempt"], 1)
        self.assertIn("summarize", context.skipped_stages)


if __name__ == "__main__":
    unittest.main()
