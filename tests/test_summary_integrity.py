"""Regression coverage for unsupported totals and unrelated action disclaimers."""

import json
import unittest
from decimal import Decimal
from unittest.mock import patch

from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel
from test_agent import executable_plan, structured_response

from agent import classify_question
from data_access import DataSources
from domain_instructions import DOMAIN_PROFILES, domain_context
from execution import ExecutionResult
from main import create_context
from planning import ExecutionPlan
from summarization import summarize_result
from summary_evidence import prepare_evidence

QUESTION = "Show total market value in INR, distinct schemes and distinct folios by AMC, with the valuation date."
QUERY = """SELECT h.amc_id, a.amc_name, sum(h.market_value) AS total_market_value_inr,
count(DISTINCT h.scheme_id) AS distinct_schemes, count(DISTINCT h.folio_id) AS distinct_folios,
min(h.nav_date) AS valuation_date FROM sw_holdings h
JOIN sw_amcs a ON a.amc_id=h.amc_id GROUP BY h.amc_id,a.amc_name ORDER BY total_market_value_inr DESC"""


class SummaryIntegrityTests(unittest.IsolatedAsyncioTestCase):
    async def test_uncertainty_summary_retries_remove_unsupported_rounded_examples(
        self,
    ):
        plan = ExecutionPlan.model_validate(executable_plan("1", "sql_python"))
        execution = ExecutionResult(
            "",
            python_runs=[
                {
                    "output": {
                        "mean": Decimal("17.264"),
                        "lower": -1263.425,
                        "upper": 1008.67,
                        "customers": 20,
                    },
                }
            ],
        )
        calls = 0

        def respond(messages, info):
            nonlocal calls
            calls += 1
            self.assertIn("SQL\nrows are calculation inputs", info.instructions)
            if calls == 1:
                return ModelResponse(
                    parts=[
                        TextPart(
                            "Average INR 17.26; 95% range -INR 1,263.42 to INR 1,008.67. Customers lose over INR 6,000."
                        )
                    ]
                )
            feedback = str(messages[-1])
            self.assertIn("6,000", feedback)
            self.assertIn("Remove the entire unsupported claim", feedback)
            self.assertIn("another guessed or coarsely rounded value", feedback)
            return ModelResponse(
                parts=[
                    TextPart(
                        "Average INR 17.26; 95% range -INR 1,263.42 to INR 1,008.67 across 20 customers."
                    )
                ]
            )

        answer = await summarize_result(
            "How uncertain is average PAT per customer? Show the average and a 95% range.",
            plan,
            execution,
            FunctionModel(respond),
        )
        self.assertEqual(calls, 2)
        self.assertNotIn("6,000", answer)

    async def test_actual_query_total_and_disclaimer_are_checked(self):
        sources = DataSources(
            create_context(source_ids=["smartwealth"]), tenant_id="2", workspace_id="2"
        )
        rows = sources.query(QUERY)
        plan = ExecutionPlan.model_validate(executable_plan("2"))
        execution = ExecutionResult(
            "Incorrect draft total: INR 615,082; date from schemes table.",
            {rows["result_id"]: rows},
            sources.queries,
            [],
            ["s1", "s2"],
        )
        calls = []

        def model(messages, info):
            calls.append(1)
            data = (
                json.loads(messages[-1].parts[0].content) if len(calls) == 1 else None
            )
            if data:
                self.assertEqual(data["execution_draft"], "")
                self.assertEqual(data["verified_aggregates"][0]["sum"], "635082.000000")
                self.assertIn("min(h.nav_date)", data["executed_queries"][0])
                self.assertIn(
                    "valuation_date", data["verified_aggregates"][0]["partition"]
                )
                return ModelResponse(
                    parts=[
                        TextPart(
                            "Total market value is INR 615,082.00 as of 2026-06-30."
                        )
                    ]
                )
            if len(calls) == 2:
                return ModelResponse(
                    parts=[
                        TextPart(
                            "Total market value is INR 635,082.00. No next-best-action policy is configured."
                        )
                    ]
                )
            return ModelResponse(
                parts=[
                    TextPart(
                        "Across six AMCs, total holding value is INR 635,082.00 as of 2026-06-30."
                    )
                ]
            )

        answer = await summarize_result(QUESTION, plan, execution, FunctionModel(model))
        self.assertEqual(len(calls), 3)
        self.assertIn("635,082.00", answer)
        self.assertNotIn("615,082", answer)
        self.assertNotIn("next-best-action", answer)

    async def test_unsafe_totals_not_created_for_partial_or_different_dates(self):
        profile = DOMAIN_PROFILES["smartwealth"]
        execution = ExecutionResult(
            "",
            {
                "query_1": {
                    "truncated": True,
                    "rows": [
                        {"total_market_value_inr": Decimal("100")},
                        {"total_market_value_inr": Decimal("200")},
                    ],
                }
            },
        )
        evidence = prepare_evidence(execution, profile)
        self.assertEqual(evidence.aggregates, [])
        self.assertIn("300", evidence.unsupported_values("Total INR 300"))
        execution.sql_results["query_1"] = {
            "rows": [
                {
                    "total_market_value_inr": Decimal("100"),
                    "valuation_date": "2026-06-30",
                },
                {
                    "total_market_value_inr": Decimal("200"),
                    "valuation_date": "2026-05-31",
                },
            ]
        }
        evidence = prepare_evidence(execution, profile)
        self.assertEqual(len(evidence.aggregates), 2)
        self.assertIn("300", evidence.unsupported_values("Total INR 300"))
        self.assertIn("2026-07-30", evidence.unsupported_values("As of 2026-07-30"))

    async def test_configured_action_rules_are_received_without_false_absence_claim(
        self,
    ):
        profile = DOMAIN_PROFILES["smartwealth"].model_copy(deep=True)
        profile.action_policies = {
            "failed_transaction_review": [
                "When an AMC has failed transactions, flag it for transaction-failure review."
            ],
            "pending_followup": [
                "When transactions are pending, request an operations follow-up."
            ],
        }
        with patch.dict(DOMAIN_PROFILES, {"smartwealth": profile}):
            context = json.loads(domain_context("2", tenant_id="2", stage="classifier"))
            self.assertTrue(context["action_policies_configured"])
            self.assertEqual(len(context["action_policies"]), 2)

            def classifier(messages, info):
                self.assertIn("failed_transaction_review", info.instructions)
                self.assertIn("pending_followup", info.instructions)
                self.assertNotIn(
                    "No next-best-action policy or recommendation rules are configured.",
                    info.instructions,
                )
                return structured_response(
                    info,
                    {
                        "intent": "advanced_analysis",
                        "objective": "Apply the configured review policies",
                    },
                )

            intent = await classify_question(
                "Apply our review actions.",
                tenant_id="2",
                workspace_id="2",
                model=FunctionModel(classifier),
            )
            self.assertNotEqual(intent.intent, "clarification")

    async def test_no_actual_action_policies_are_silently_invented(self):
        profile = DOMAIN_PROFILES["smartwealth"].model_copy(deep=True)
        profile.action_policies = {}
        with patch.dict(DOMAIN_PROFILES, {"smartwealth": profile}):
            context = json.loads(domain_context("2", tenant_id="2", stage="summarizer"))
        self.assertEqual(context["action_policies"], {})
        self.assertFalse(context["action_policies_configured"])
        self.assertNotIn(
            "No next-best-action policy or recommendation rules are configured.",
            json.dumps(context),
        )

    async def test_summary_cannot_deny_supplied_policies(self):
        profile = DOMAIN_PROFILES["smartwealth"].model_copy(deep=True)
        profile.action_policies = {
            "failed_transaction_review": [
                "When an AMC has failed transactions, request a transaction-failure review."
            ]
        }
        plan = ExecutionPlan.model_validate(executable_plan("2"))
        evidence = ExecutionResult(
            "Apply the configured rule.",
            {
                "query_1": {
                    "rows": [
                        {
                            "amc_name": "Sample Horizon AMC",
                            "failed_transactions": 1,
                            "action": "Transaction-failure review",
                        }
                    ]
                }
            },
        )
        calls = []

        def model(messages, info):
            calls.append(1)
            if len(calls) == 1:
                return ModelResponse(
                    parts=[TextPart("No next-best-action policy is configured.")]
                )
            return ModelResponse(
                parts=[
                    TextPart(
                        "Request a transaction-failure review for Sample Horizon AMC."
                    )
                ]
            )

        with patch.dict(DOMAIN_PROFILES, {"smartwealth": profile}):
            answer = await summarize_result(
                "What next best actions follow our rules?",
                plan,
                evidence,
                FunctionModel(model),
            )
        self.assertEqual(len(calls), 2)
        self.assertIn("transaction-failure review", answer)
