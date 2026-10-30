"""The planner schema must expose all step-shape errors in one retry."""

import io
import json
import unittest
from contextlib import redirect_stderr

from pydantic import ValidationError
from pydantic_ai.models.function import FunctionModel
from test_agent import executable_plan, structured_response
from test_model_protocol import completion, mock_provider

from model_config import configured_model
from planning import PlannerOutput, QuestionIntent, build_plan
from runtime import TRACE_ENABLED
from domain_instructions import get_domain_profile

CONTRACT = {
    "required_columns": {
        "customer_id": {"type": "string", "nullable": False},
        "pat_per_customer": {
            "type": "numeric",
            "meaning": "Selected-period PAT per customer",
        },
    },
    "grain": ["customer_id"],
    "population": {
        "table": "cc_customers",
        "source_key": "customer_id",
        "result_key": "customer_id",
    },
}


def proposal(*, invalid=False):
    value = executable_plan(
        "1",
        "sql_python",
        declared=True,
        output_contract=json.loads(json.dumps(CONTRACT)),
    )
    if invalid:
        value["steps"][1]["output_contract"]["population"] = {
            "table": "cc_monthly_financials",
            "source_key": "card_id",
            "result_key": "pat_per_customer",
        }
        value["steps"][2]["output_contract"] = {
            "required_columns": {"mean_pat": {"type": "numeric"}}
        }
    return value


class PlannerSchemaTests(unittest.IsolatedAsyncioTestCase):
    def test_focused_guidance_preserves_workspace_defaults_without_global_routing(self):
        profile = get_domain_profile("1", tenant_id="1")
        guidance = profile.focused_guidance(
            "How uncertain is average PAT per customer? Show the average and a 95% range.",
            "planner",
        )
        self.assertIn("latest available financial month", " ".join(guidance))
        self.assertIn("cc_customers.customer_id", " ".join(guidance))
        self.assertEqual(profile.focused_guidance("Count customers", "planner"), [])
        self.assertEqual(
            profile.focused_guidance(
                "How uncertain is average PAT per customer?", "classifier"
            ),
            [],
        )
        self.assertEqual(
            get_domain_profile("2", tenant_id="2").focused_guidance(
                "How uncertain is average PAT per customer?", "planner"
            ),
            [],
        )

    def test_python_cannot_declare_sql_contract_and_sql_contract_is_required(self):
        value = proposal()
        value["steps"][1].pop("output_contract")
        value["steps"][2]["output_contract"] = {
            "required_columns": {"mean": {"type": "numeric"}}
        }
        with self.assertRaises(ValidationError) as raised:
            PlannerOutput.model_validate(value)
        locations = [error["loc"] for error in raised.exception.errors()]
        self.assertIn(("steps", 1, "sql", "output_contract"), locations)
        self.assertIn(("steps", 2, "python", "output_contract"), locations)

    async def test_population_and_step_errors_are_corrected_in_one_retry_with_trace(
        self,
    ):
        calls = 0

        def respond(messages, info):
            nonlocal calls
            calls += 1
            if calls == 1:
                prompt = json.loads(messages[-1].parts[0].content)
                self.assertIn(
                    "latest available financial month",
                    " ".join(prompt["applicable_workspace_guidance"]),
                )
            if calls == 2:
                feedback = str(messages[-1])
                self.assertIn("grain=['customer_id']", feedback)
                self.assertIn("Input should be None", feedback)
            return structured_response(info, proposal(invalid=calls == 1))

        output = io.StringIO()
        token = TRACE_ENABLED.set(True)
        try:
            with redirect_stderr(output):
                plan = await build_plan(
                    "How uncertain is average PAT per customer? Show the average and a 95% range.",
                    QuestionIntent(
                        intent="advanced_analysis",
                        objective="Estimate uncertainty in average PAT per customer",
                    ),
                    tenant_id="1",
                    workspace_id="1",
                    model=FunctionModel(respond),
                )
        finally:
            TRACE_ENABLED.reset(token)
        self.assertEqual(calls, 2)
        self.assertIsNone(plan.steps[2].output_contract)
        self.assertEqual(
            plan.steps[1].output_contract.population.result_key, "customer_id"
        )
        diagnostics = [
            json.loads(line.removeprefix("PLAN_REJECTED: "))
            for line in output.getvalue().splitlines()
            if line.startswith("PLAN_REJECTED: ")
        ]
        self.assertEqual(len(diagnostics), 1)
        self.assertEqual(len(diagnostics[0]["errors"]), 2)
        self.assertTrue(all("input" not in error for error in diagnostics[0]["errors"]))

    async def test_real_openai_wire_uses_action_specific_schema_and_returns_both_errors(
        self,
    ):
        calls = 0

        def respond(request):
            nonlocal calls
            calls += 1
            body = json.loads(request.content)
            schema = body["response_format"]["json_schema"]["schema"]
            definitions = schema["$defs"]
            self.assertEqual(
                definitions["PythonPlanningStep"]["properties"]["output_contract"][
                    "type"
                ],
                "null",
            )
            self.assertIn("output_contract", definitions["SQLPlanningStep"]["required"])
            self.assertEqual(schema["properties"]["sql_recipe"]["type"], "null")
            if calls == 2:
                feedback = str(body["messages"][-1])
                self.assertIn("customer_id", feedback)
                self.assertIn("Input should be null", feedback)
            return completion(body, content=json.dumps(proposal(invalid=calls == 1)))

        async with mock_provider(respond):
            plan = await build_plan(
                "Estimate average PAT uncertainty",
                QuestionIntent(
                    intent="advanced_analysis",
                    objective="Estimate average PAT uncertainty",
                ),
                tenant_id="1",
                workspace_id="1",
                model=configured_model("1", "1"),
            )
        self.assertEqual(calls, 2)
        self.assertEqual(plan.execution_mode, "sql_python")


if __name__ == "__main__":
    unittest.main()
