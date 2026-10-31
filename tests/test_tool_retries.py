"""Exercise configurable SQL/Python correction budgets with actual tool failures."""

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import AsyncMock, patch

from pydantic_ai import ModelRetry
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel
from test_agent import executable_plan, tool_returns

from data_access import DataSources
from execution import execute_plan
from main import create_context
from planning import ExecutionPlan
from runtime import TRACE_ENABLED, tool_retry_limit


class ToolRetryConfigurationTests(unittest.TestCase):
    def test_cli_exposes_retry_setting_and_the_last_tool_failure(self):
        from agent import main

        error = UnexpectedModelBehavior(
            "Tool 'run_sql' exceeded max retries count of 3"
        )
        error.__cause__ = ModelRetry("Execute step s3 (python) next; use run_python")
        output = io.StringIO()
        with (
            patch(
                "sys.argv",
                [
                    "agent.py",
                    "--tenant-id",
                    "1",
                    "--workspace-id",
                    "1",
                    "Calculate a summary",
                ],
            ),
            patch("agent.Pipeline.run", new=AsyncMock(side_effect=error)),
            redirect_stderr(output),
            self.assertRaises(SystemExit) as raised,
        ):
            main()
        self.assertEqual(raised.exception.code, 1)
        self.assertIn("AGENT_TOOL_RETRIES", output.getvalue())
        self.assertIn("Last tool error: Execute step s3", output.getvalue())
        self.assertIn("use run_python", output.getvalue())

    def test_default_and_nonnegative_values(self):
        with patch("runtime.load_dotenv"), patch.dict(os.environ, {}, clear=True):
            self.assertEqual(tool_retry_limit(), 3)
        for value in (0, 1, 5):
            with (
                self.subTest(value=value),
                patch("runtime.load_dotenv"),
                patch.dict(os.environ, {"AGENT_TOOL_RETRIES": str(value)}),
            ):
                self.assertEqual(tool_retry_limit(), value)

    def test_invalid_values_are_actionable(self):
        for value in ("-1", "bad", "1.5", ""):
            with (
                self.subTest(value=value),
                patch("runtime.load_dotenv"),
                patch.dict(os.environ, {"AGENT_TOOL_RETRIES": value}),
            ):
                with self.assertRaisesRegex(ValueError, "AGENT_TOOL_RETRIES"):
                    tool_retry_limit()

    def test_dotenv_and_shell_precedence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".env").write_text("AGENT_TOOL_RETRIES=6\n")
            with patch("runtime.ROOT", root), patch.dict(os.environ, {}, clear=True):
                self.assertEqual(tool_retry_limit(), 6)
                os.environ["AGENT_TOOL_RETRIES"] = "2"
                self.assertEqual(tool_retry_limit(), 2)


class ToolRetryExecutionTests(unittest.IsolatedAsyncioTestCase):
    def sources(self):
        return DataSources(create_context(source_ids=["credit_cards"]), "1", "1")

    async def test_sql_and_python_each_get_the_configured_correction_budget(self):
        sources = self.sources()
        actions = (
            [("describe_sources", {"step_id": "s1"})]
            + [
                (
                    "run_sql",
                    {
                        "step_id": "s2",
                        "statement": "SELECT missing_column FROM cc_customers",
                    },
                )
            ]
            * 4
            + [
                (
                    "run_sql",
                    {
                        "step_id": "s2",
                        "statement": "SELECT COUNT(*) AS n FROM cc_customers",
                    },
                )
            ]
            + [
                (
                    "run_python",
                    {"step_id": "s3", "result_ids": ["query_1"], "code": code},
                )
                for code in (
                    "data = {'query_1': []}\ndata['query_1']",
                    "data['query_1'][0]['total_approved_net_spend_inr']",
                    "data['query_1'][0]['total_approved_net_spend_inr']",
                    "data['query_1'][0]['wrong_field']",
                    "{'customers': data['query_1'][0]['n']}",
                )
            ]
        )
        calls = []

        def model(messages, info):
            index = len(calls)
            calls.append(index)
            if 7 <= index <= 10:
                feedback = tool_returns(messages)[-1]
                self.assertEqual(feedback.tool_name, "run_python")
                repair = json.loads(
                    str(feedback.content).removeprefix("Python execution failed: ")
                )
                self.assertEqual(repair["authority"]["sql_columns"], {"query_1": ["n"]})
                self.assertEqual(repair["target"], "python_generator")
                self.assertIn("do not rerun SQL", str(feedback.content))
                if index in (8, 9):
                    self.assertIn("total_approved_net_spend_inr", str(feedback.content))
            if index < len(actions):
                name, arguments = actions[index]
                return ModelResponse(parts=[ToolCallPart(name, arguments)])
            return ModelResponse(parts=[TextPart("Calculation completed")])

        with patch.dict(os.environ, {"AGENT_TOOL_RETRIES": "5"}):
            result = await execute_plan(
                "Calculate a summary",
                ExecutionPlan.model_validate(executable_plan("1", "sql_python")),
                sources,
                FunctionModel(model),
            )
        self.assertEqual(len(calls), 12)
        self.assertEqual(result.completed_steps, ["s1", "s2", "s3"])
        self.assertEqual(len(result.queries), 1)
        self.assertEqual(len(result.python_runs), 1)
        self.assertEqual(result.python_runs[0]["output"], {"customers": 20})

    async def test_zero_retries_stops_at_the_first_failure_for_each_tool(self):
        for failing_tool in ("run_sql", "run_python"):
            calls = []

            def model(messages, info):
                index = len(calls)
                calls.append(1)
                if index == 0:
                    name, args = "describe_sources", {"step_id": "s1"}
                elif index == 1:
                    name, args = "run_sql", {
                        "step_id": "s2",
                        "statement": (
                            "SELECT missing_column FROM cc_customers"
                            if failing_tool == "run_sql"
                            else "SELECT COUNT(*) AS n FROM cc_customers"
                        ),
                    }
                else:
                    name, args = "run_python", {
                        "step_id": "s3",
                        "result_ids": ["query_1"],
                        "code": "data['query_1'][0]['missing']",
                    }
                return ModelResponse(parts=[ToolCallPart(name, args)])

            with (
                self.subTest(failing_tool=failing_tool),
                patch.dict(os.environ, {"AGENT_TOOL_RETRIES": "0"}),
                self.assertRaisesRegex(
                    UnexpectedModelBehavior,
                    f"Tool '{failing_tool}' exceeded max retries count of 0",
                ),
            ):
                await execute_plan(
                    "Calculate a summary",
                    ExecutionPlan.model_validate(executable_plan("1", "sql_python")),
                    self.sources(),
                    FunctionModel(model),
                )
            self.assertEqual(len(calls), 2 if failing_tool == "run_sql" else 3)

    async def test_tool_budget_does_not_increase_output_validation_retries(self):
        calls = []

        def model(messages, info):
            calls.append(1)
            return ModelResponse(parts=[TextPart("Premature answer")])

        with (
            patch.dict(os.environ, {"AGENT_TOOL_RETRIES": "5"}),
            self.assertRaisesRegex(
                UnexpectedModelBehavior, "maximum output retries.*3"
            ),
        ):
            await execute_plan(
                "Count customers",
                ExecutionPlan.model_validate(executable_plan()),
                self.sources(),
                FunctionModel(model),
            )
        self.assertEqual(len(calls), 4)

    async def test_out_of_order_sql_is_traced_as_a_rejected_call(self):
        calls = []
        output = io.StringIO()

        def model(messages, info):
            index = len(calls)
            calls.append(1)
            actions = [
                ("describe_sources", {"step_id": "s1"}),
                (
                    "run_sql",
                    {
                        "step_id": "s2",
                        "statement": "SELECT COUNT(*) AS n FROM cc_customers",
                    },
                ),
                (
                    "run_sql",
                    {
                        "step_id": "s2",
                        "statement": "SELECT COUNT(*) AS n FROM cc_customers",
                    },
                ),
                (
                    "run_python",
                    {
                        "step_id": "s3",
                        "result_ids": ["query_1"],
                        "code": "{'customers': data['query_1'][0]['n']}",
                    },
                ),
            ]
            if index == 3:
                self.assertIn("Use run_python", str(tool_returns(messages)[-1].content))
            if index < len(actions):
                name, args = actions[index]
                return ModelResponse(parts=[ToolCallPart(name, args)])
            return ModelResponse(parts=[TextPart("Completed")])

        token = TRACE_ENABLED.set(True)
        try:
            with (
                patch.dict(os.environ, {"AGENT_TOOL_RETRIES": "5"}),
                redirect_stderr(output),
            ):
                result = await execute_plan(
                    "Calculate a summary",
                    ExecutionPlan.model_validate(executable_plan("1", "sql_python")),
                    self.sources(),
                    FunctionModel(model),
                )
        finally:
            TRACE_ENABLED.reset(token)
        self.assertEqual(len(result.queries), 1)
        self.assertIn("EXECUTION tool retries: 5", output.getvalue())
        self.assertIn("TOOL run_sql: s2 rejected", output.getvalue())
