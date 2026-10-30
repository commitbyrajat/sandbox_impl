"""Check structured outputs and function tools through the real OpenAI adapter."""

import io
import json
import os
import unittest
from contextlib import asynccontextmanager, redirect_stderr
from unittest.mock import AsyncMock, patch

import httpx2 as httpx
from openai import AsyncOpenAI
from pydantic_ai.exceptions import ModelHTTPError, ModelServerError
from pydantic_ai.providers.openai import OpenAIProvider
from test_agent import executable_plan

from agent import main, plan_question
from data_access import DataSources
from datasource_mapping import MODELS, ModelDefinition
from execution import execute_plan
from main import create_context
from model_config import configured_model
from planning import ExecutionPlan
from runtime import model_error_detail


def completion(body, *, content=None, call=None):
    message = {"role": "assistant", "content": content}
    if call is not None:
        name, arguments = call
        message["tool_calls"] = [
            {
                "id": "call",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }
        ]
    return httpx.Response(
        200,
        json={
            "id": "test",
            "object": "chat.completion",
            "created": 0,
            "model": body["model"],
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": "tool_calls" if call else "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )


@asynccontextmanager
async def mock_provider(respond):
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        client = AsyncOpenAI(
            api_key="test", base_url="http://test/v1", http_client=http
        )
        with (
            patch.dict(os.environ, {"LITELLM_MASTER_KEY": "test"}),
            patch(
                "model_config.OpenAIProvider",
                return_value=OpenAIProvider(openai_client=client),
            ),
        ):
            yield


class ModelProtocolTests(unittest.IsolatedAsyncioTestCase):
    async def test_qwen_uses_json_schema_and_retries_invalid_json_before_planning(self):
        requests = []

        def respond(request):
            body = json.loads(request.content)
            requests.append(body)
            self.assertEqual(body["model"], "qwen-coder")
            self.assertFalse(body.get("tools"))
            self.assertEqual(body["response_format"]["type"], "json_schema")
            properties = body["response_format"]["json_schema"]["schema"]["properties"]
            if len(requests) == 1:
                return completion(body, content='{"intent":')
            if "objective" in properties:
                payload = {
                    "intent": "advanced_analysis",
                    "objective": "Group customers into segments",
                }
            else:
                payload = executable_plan("1", "sql_python")
            return completion(body, content=json.dumps(payload))

        async with mock_provider(respond):
            with patch("data_access.create_context") as loader:
                intent, plan = await plan_question(
                    "Group customers into three segments with similar spending, profitability and operating cost per card. Show members and typical values.",
                    tenant_id="1",
                    workspace_id="1",
                )
                loader.assert_not_called()
        self.assertEqual(len(requests), 3)
        self.assertEqual(intent.intent, "advanced_analysis")
        self.assertEqual(plan.execution_mode, "sql_python")
        self.assertEqual([s.action for s in plan.steps], ["schema", "sql", "python"])

    async def test_ornith_retains_tool_output_mode(self):
        requests = []

        def respond(request):
            body = json.loads(request.content)
            requests.append(body)
            self.assertEqual(body["model"], "ornith-reasoner")
            self.assertNotEqual(
                body.get("response_format", {}).get("type"), "json_schema"
            )
            functions = [tool["function"] for tool in body["tools"]]
            self.assertEqual(len(functions), 1)
            function = functions[0]
            properties = function["parameters"]["properties"]
            payload = (
                {"intent": "aggregation", "objective": "Count customers"}
                if "objective" in properties
                else executable_plan("2")
            )
            return completion(body, call=(function["name"], payload))

        async with mock_provider(respond):
            _, plan = await plan_question(
                "How many customers?", tenant_id="2", workspace_id="2"
            )
        self.assertEqual(len(requests), 2)
        self.assertEqual(plan.sources, ["smartwealth"])

    async def test_native_profile_keeps_real_sql_and_monty_function_tools(self):
        requests = []
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
                "run_python",
                {
                    "step_id": "s3",
                    "result_ids": ["query_1"],
                    "code": "{'double_count': data['query_1'][0]['n'] * 2}",
                },
            ),
        ]

        def respond(request):
            body = json.loads(request.content)
            index = len(requests)
            requests.append(body)
            self.assertEqual(
                {tool["function"]["name"] for tool in body["tools"]},
                {"describe_sources", "run_sql", "run_python"},
            )
            self.assertNotEqual(
                body.get("response_format", {}).get("type"), "json_schema"
            )
            return (
                completion(body, call=actions[index])
                if index < len(actions)
                else completion(body, content="Calculation completed")
            )

        sources = DataSources(create_context(source_ids=["credit_cards"]), "1", "1")
        async with mock_provider(respond):
            result = await execute_plan(
                "Calculate double the customer count",
                ExecutionPlan.model_validate(executable_plan("1", "sql_python")),
                sources,
                configured_model("1", "1"),
            )
        self.assertEqual(len(requests), 4)
        self.assertEqual(len(result.queries), 1)
        self.assertEqual(result.python_runs[0]["output"], {"double_count": 40})
        self.assertTrue(result.python_runs[0]["executed"])


class ProviderErrorTests(unittest.TestCase):
    def test_custom_api_key_variable_is_also_redacted(self):
        custom = MODELS["qwen-coder"].model_copy(
            update={"api_key_env": "CUSTOM_CREDENTIAL"}
        )
        with (
            patch.dict(MODELS, {"qwen-coder": custom}),
            patch.dict(os.environ, {"CUSTOM_CREDENTIAL": "opaque-value"}),
        ):
            detail = model_error_detail(
                ModelServerError("qwen-coder", "Failed with opaque-value")
            )
        self.assertNotIn("opaque-value", detail)
        self.assertIn("[redacted]", detail)

    def test_cli_reports_provider_status_and_message_without_credentials(self):
        error = ModelHTTPError.for_category(
            ModelServerError,
            status_code=500,
            model_name="qwen-coder",
            body={
                "error": {
                    "message": "Ollama XML syntax error: element <parameter> closed by </function>; sk-secret; Bearer token-secret; opaque-secret"
                }
            },
        )
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
                    "Group customers",
                ],
            ),
            patch("agent.Pipeline.run", new=AsyncMock(side_effect=error)),
            patch.dict(os.environ, {"TEST_API_KEY": "opaque-secret"}),
            redirect_stderr(output),
            self.assertRaises(SystemExit) as raised,
        ):
            main()
        self.assertEqual(raised.exception.code, 1)
        text = output.getvalue()
        self.assertIn("qwen-coder (HTTP 500)", text)
        self.assertIn("element <parameter> closed by </function>", text)
        self.assertNotIn("database connection", text)
        for secret in ("sk-secret", "token-secret", "opaque-secret"):
            self.assertNotIn(secret, text)

    def test_non_http_provider_error_keeps_its_message(self):
        detail = model_error_detail(
            ModelServerError("qwen-coder", "Backend unavailable")
        )
        self.assertIn("Backend unavailable", detail)
        self.assertIn("ModelServerError", detail)

    def test_model_output_mode_is_validated_and_defaults_to_tool(self):
        self.assertEqual(
            ModelDefinition(model_name="example").structured_output_mode, "tool"
        )
        self.assertEqual(
            ModelDefinition(
                model_name="example", structured_output_mode="native"
            ).structured_output_mode,
            "native",
        )
        with self.assertRaises(ValueError):
            ModelDefinition(model_name="example", structured_output_mode="unknown")
