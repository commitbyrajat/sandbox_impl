"""Regression tests for workspace vocabulary and partially answerable questions."""

import json
import unittest
from unittest.mock import patch

from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel
from test_agent import executable_plan, structured_response, tool_returns
from test_support import ask

from agent import classify_question
from datasource_mapping import classification_context
from planning import QuestionIntent

QUESTION = "Show holdings by AMC and what is their next best action?"
FOLLOW_UP = (
    "For next best action, do you mean portfolio review, customer outreach, "
    "rebalancing, or another defined policy?"
)


def compound_intent():
    return {
        "intent": "aggregation",
        "objective": "Show AMC holdings and clarify action policy",
        "objectives": [
            {"objective": "Show holdings by AMC", "status": "ready"},
            {
                "objective": "Determine next best action",
                "status": "needs_clarification",
                "clarification": FOLLOW_UP,
            },
        ],
    }


class CompoundQuestionTests(unittest.IsolatedAsyncioTestCase):
    async def test_context_contains_glossary_but_no_source_details(self):
        context = classification_context("2", tenant_id="2")
        self.assertIn("Asset Management Company", context)
        for forbidden in (
            "sw_holdings",
            "cc_transactions",
            "location",
            "password",
            "LITELLM",
        ):
            self.assertNotIn(forbidden, context)
        self.assertNotIn("AMC", classification_context("1", tenant_id="1"))

    async def test_answer_ready_part_and_preserve_pending_question(self):
        events = []

        def classifier(messages, info):
            events.append("classify")
            self.assertIn("Asset Management Company", info.instructions)
            self.assertIn("SmartWealth", info.instructions)
            self.assertNotIn("sw_holdings", info.instructions)
            return structured_response(info, compound_intent())

        def planner(messages, info):
            events.append("plan")
            intent = json.loads(messages[-1].parts[0].content)["classified_intent"]
            self.assertEqual(len(intent["objectives"]), 2)
            plan = executable_plan(
                "2",
                declared=True,
                output_contract={
                    "required_columns": {
                        "amc_name": {"type": "string"},
                        "market_value_inr": {"type": "numeric"},
                        "nav_date": {"type": "date"},
                    },
                    "grain": ["amc_name"],
                },
            )
            plan["steps"][1][
                "instruction"
            ] = "Aggregate holding value by AMC with NAV date"
            # A planner rephrase must not duplicate or replace the original question.
            plan["deferred_objectives"] = [
                {
                    "objective": "Determine next best action",
                    "clarification": "Which action policy?",
                }
            ]
            return structured_response(info, plan)

        def executor(messages, info):
            returns = tool_returns(messages)
            self.assertIn(FOLLOW_UP, info.instructions)
            if not returns:
                events.append("schema")
                return ModelResponse(
                    parts=[ToolCallPart("describe_sources", {"step_id": "s1"})]
                )
            if returns[-1].tool_name == "describe_sources":
                events.append("sql")
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            "run_sql",
                            {
                                "step_id": "s2",
                                "statement": "SELECT a.amc_name, sum(h.market_value) AS market_value_inr, "
                                "max(h.nav_date) AS nav_date FROM sw_holdings h "
                                "JOIN sw_amcs a ON h.amc_id=a.amc_id GROUP BY a.amc_name ORDER BY a.amc_name",
                            },
                        )
                    ]
                )
            events.append("answer")
            result = json.loads(returns[-1].content)
            self.assertEqual(len(result["rows"]), 6)
            self.assertEqual(
                sum(float(row["market_value_inr"]) for row in result["rows"]), 635082
            )
            if events.count("answer") == 1:
                return ModelResponse(
                    parts=[TextPart("Six AMCs hold INR 635,082. Which action policy?")]
                )
            return ModelResponse(
                parts=[TextPart("Six AMCs hold INR 635,082 as of 2026-06-30.")]
            )

        answer = await ask(
            QUESTION,
            tenant_id="2",
            workspace_id="2",
            classifier_model=FunctionModel(classifier),
            planner_model=FunctionModel(planner),
            model=FunctionModel(executor),
        )
        self.assertEqual(
            events, ["classify", "plan", "schema", "sql", "answer", "answer"]
        )
        self.assertIn("635,082", answer)
        self.assertIn(FOLLOW_UP, answer)
        self.assertEqual(answer.count(FOLLOW_UP), 1)
        self.assertNotIn("Which action policy?", answer)

    async def test_known_scope_and_glossary_clarifications_are_retried(self):
        for question in (
            "Which workspace/dataset should I use?",
            "What does AMC refer to?",
            'What does "AMC" refer to?',
        ):
            calls = []

            def classifier(messages, info):
                calls.append(1)
                if len(calls) == 1:
                    return structured_response(
                        info,
                        {
                            "intent": "clarification",
                            "objective": "Show holdings by AMC",
                            "clarification": question,
                        },
                    )
                return structured_response(
                    info, {"intent": "aggregation", "objective": "Show AMC holdings"}
                )

            with self.subTest(question=question):
                result = await classify_question(
                    QUESTION,
                    tenant_id="2",
                    workspace_id="2",
                    model=FunctionModel(classifier),
                )
                self.assertEqual(len(calls), 2)
                self.assertEqual(result.intent, "aggregation")

    async def test_wholly_unclear_request_does_not_open_data(self):
        def classifier(messages, info):
            return structured_response(
                info,
                {
                    "intent": "clarification",
                    "objective": "Determine next action",
                    "clarification": FOLLOW_UP,
                    "objectives": [
                        {
                            "objective": "Determine next action",
                            "status": "needs_clarification",
                            "clarification": FOLLOW_UP,
                        }
                    ],
                },
            )

        with patch("data_access.create_context") as loader:
            answer = await ask(
                "What is the next best action?",
                tenant_id="2",
                workspace_id="2",
                model=FunctionModel(classifier),
            )
            loader.assert_not_called()
        self.assertEqual(answer, FOLLOW_UP)

    async def test_mixed_request_cannot_be_marked_wholly_unclear(self):
        intent = compound_intent()
        intent.update(intent="clarification", clarification=FOLLOW_UP)
        with self.assertRaises(ValueError):
            QuestionIntent.model_validate(intent)
