import unittest
from unittest.mock import AsyncMock, patch

from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel
from test_agent import executable_plan, tool_returns

from data_access import DataSources
from execution import execute_plan
from main import create_context
from planning import ExecutionPlan, consolidate_python_steps


class DedupTests(unittest.IsolatedAsyncioTestCase):
    def plan(self):
        value = executable_plan("1", "sql_python")
        value["steps"].append(
            {
                "step_id": "s4",
                "action": "python",
                "instruction": "Report mean also",
                "expected_output": "Mean",
            }
        )
        return ExecutionPlan.model_validate(value)

    async def test_consolidate_preserves_outputs_and_first_id(self):
        result = consolidate_python_steps(self.plan())
        self.assertEqual([s.step_id for s in result.steps], ["s1", "s2", "s3"])
        self.assertIn("Report mean also", result.steps[-1].instruction)
        self.assertIn("Mean", result.steps[-1].expected_output)

    async def test_repeated_code_uses_one_sandbox_call_per_request(self):
        session = AsyncMock()
        session.feed_run.return_value = {"total": 20}

        class Pool:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            def checkout(self, **kwargs):
                class Checkout:
                    async def __aenter__(self):
                        return session

                    async def __aexit__(self, *args):
                        pass

                return Checkout()

        def model(messages, info):
            returns = tool_returns(messages)
            calls = [
                ("describe_sources", {"step_id": "s1"}),
                (
                    "run_sql",
                    {
                        "step_id": "s2",
                        "statement": "SELECT count(*) AS n FROM cc_customers",
                    },
                ),
                (
                    "run_python",
                    {
                        "step_id": "s3",
                        "result_ids": ["query_1"],
                        "code": "{'total': data['query_1'][0]['n']}",
                    },
                ),
                (
                    "run_python",
                    {
                        "step_id": "s4",
                        "result_ids": ["query_1"],
                        "code": "# same calculation\n{'total': data['query_1'][0]['n']}",
                    },
                ),
            ]
            if len(returns) < len(calls):
                name, args = calls[len(returns)]
                return ModelResponse(parts=[ToolCallPart(name, args)])
            return ModelResponse(parts=[TextPart("Total 20")])

        with patch("execution.AsyncMonty", return_value=Pool()):
            for expected_calls in [1, 2]:
                sources = DataSources(
                    create_context(source_ids=["credit_cards"]),
                    tenant_id="1",
                    workspace_id="1",
                )
                result = await execute_plan(
                    "Count customers", self.plan(), sources, FunctionModel(model)
                )
                self.assertEqual(session.feed_run.await_count, expected_calls)
                self.assertTrue(result.python_runs[0]["executed"])
                self.assertFalse(result.python_runs[1]["executed"])
                self.assertEqual(result.python_runs[1]["reused_from_step_id"], "s3")
                self.assertNotIn("code", result.python_runs[1])
                self.assertEqual(result.completed_steps, ["s1", "s2", "s3", "s4"])
