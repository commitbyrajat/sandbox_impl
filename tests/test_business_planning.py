"""Business questions require no user-supplied implementation instructions."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path

from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel
from test_agent import executable_plan, structured_response, tool_returns
from test_support import ask
from test_tenants import answering_model

from agent import classify_question
from planning import ExecutionPlan


class BusinessPlanningTests(unittest.IsolatedAsyncioTestCase):
    async def test_business_question_runs_sql_and_python_and_publishes_choice(self):
        question = "Split AMCs into three review groups with roughly equal total holding value."
        events = []

        def classifier(messages, info):
            events.append("classify")
            return structured_response(
                info, {"intent": "advanced_analysis", "objective": question}
            )

        def planner(messages, info):
            events.append("plan")
            submitted_question = json.loads(messages[-1].parts[0].content)["question"]
            self.assertEqual(submitted_question, question)
            self.assertNotIn("python", submitted_question.lower())
            self.assertNotIn("sql", submitted_question.lower())
            self.assertIn("Choose the implementation automatically", info.instructions)
            self.assertIn("largest-first greedy assignment", info.instructions)
            plan = executable_plan(
                "2",
                "sql_python",
                declared=True,
                output_contract={
                    "required_columns": {
                        "amc_name": {"type": "string"},
                        "value": {"type": "numeric"},
                    },
                    "grain": ["amc_name"],
                },
            )
            plan["execution_reason"] = (
                "Balancing review groups requires repeated assignments after totaling AMC holdings."
            )
            plan["python_reason"] = (
                "Assign each AMC to the currently smallest group; the running totals change after every assignment."
            )
            return structured_response(info, plan)

        def executor(messages, info):
            returns = tool_returns(messages)
            if not returns:
                return ModelResponse(
                    parts=[ToolCallPart("describe_sources", {"step_id": "s1"})]
                )
            if returns[-1].tool_name == "describe_sources":
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            "run_sql",
                            {
                                "step_id": "s2",
                                "statement": "SELECT a.amc_name, sum(h.market_value) AS value "
                                "FROM sw_holdings h JOIN sw_amcs a ON a.amc_id=h.amc_id "
                                "GROUP BY a.amc_name ORDER BY value DESC, a.amc_name",
                            },
                        )
                    ]
                )
            if returns[-1].tool_name == "run_sql":
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            "run_python",
                            {
                                "step_id": "s3",
                                "result_ids": ["query_1"],
                                "code": """from decimal import Decimal
rows = sorted(data['query_1'], key=lambda r: (-r['value'], r['amc_name']))
groups = [{'id': i + 1, 'amcs': [], 'value': Decimal('0')} for i in range(3)]
for row in rows:
    target = min(groups, key=lambda g: (g['value'], g['id']))
    target['amcs'].append(row['amc_name'])
    target['value'] += row['value']
groups
""",
                            },
                        )
                    ]
                )
            groups = json.loads(returns[-1].content)["result"]
            self.assertEqual(len(groups), 3)
            self.assertEqual(sum(len(g["amcs"]) for g in groups), 6)
            self.assertEqual(
                [g["value"] for g in groups],
                ["222072.000000", "206145.000000", "206865.000000"],
            )
            events.append("answer")
            return ModelResponse(
                parts=[TextPart("Six AMCs assigned to three review groups.")]
            )

        output = io.StringIO()
        with tempfile.TemporaryDirectory() as directory, redirect_stderr(output):
            answer = await ask(
                question,
                tenant_id="2",
                workspace_id="2",
                plan_dir=Path(directory),
                model=FunctionModel(executor),
                classifier_model=FunctionModel(classifier),
                planner_model=FunctionModel(planner),
            )
            record = json.loads(next(Path(directory).rglob("*.json")).read_text())
        self.assertEqual(answer, "Six AMCs assigned to three review groups.")
        self.assertTrue(record["execution_summary"]["python_required"])
        self.assertEqual(record["plan"]["execution_mode"], "sql_python")
        self.assertIn("repeated assignments", record["plan"]["execution_reason"])
        self.assertIn("Python required: yes", output.getvalue())
        self.assertIn("Additional calculation:", output.getvalue())

    async def test_simple_business_question_publishes_sql_only_choice(self):
        events = []
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as directory, redirect_stderr(output):
            await ask(
                "How many customers do we have?",
                tenant_id="1",
                workspace_id="1",
                model=answering_model("1", "1", "cc_customers", 20, events),
                plan_dir=Path(directory),
            )
            record = json.loads(next(Path(directory).rglob("*.json")).read_text())
        self.assertFalse(record["execution_summary"]["python_required"])
        self.assertIn("Python required: no", output.getvalue())
        self.assertTrue(record["plan"]["execution_reason"])
        self.assertFalse(any(s["action"] == "python" for s in record["plan"]["steps"]))

    async def test_executable_plan_requires_explanation(self):
        plan = executable_plan()
        del plan["execution_reason"]
        with self.assertRaises(ValueError):
            ExecutionPlan.model_validate(plan)

    async def test_language_choice_clarification_is_rejected(self):
        calls = []

        def classifier(messages, info):
            calls.append(1)
            if len(calls) == 1:
                return structured_response(
                    info,
                    {
                        "intent": "clarification",
                        "objective": "Estimate uncertainty in daily spending",
                        "clarification": "Should I use SQL or Python?",
                    },
                )
            return structured_response(
                info,
                {
                    "intent": "advanced_analysis",
                    "objective": "Estimate uncertainty in daily spending",
                },
            )

        result = await classify_question(
            "How uncertain is our average daily spending?",
            tenant_id="1",
            workspace_id="1",
            model=FunctionModel(classifier),
        )
        self.assertEqual(result.intent, "advanced_analysis")
        self.assertEqual(len(calls), 2)
