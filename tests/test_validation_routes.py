"""Domain-independent checks and real model/tool correction routing."""

import io
import json
import os
import unittest
from contextlib import redirect_stderr
from decimal import Decimal
from unittest.mock import patch

from pydantic import ValidationError
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, UserPromptPart
from pydantic_ai.models.function import FunctionModel

from calculation_contracts import (
    CalculationValidationError,
    PythonResultContract,
    validate_execution_result,
    validate_python_result,
)
from data_access import DataSources
from execution import execute_plan
from main import create_context
from planning import (
    ExecutionPlan,
    PythonPlanningStep,
    QuestionIntent,
    build_plan,
    planner_output_type,
)
from python_validation import PythonPreflightError, preflight_python
from runtime import TRACE_ENABLED
from test_agent import executable_plan, structured_response, tool_returns


def group_contract():
    return PythonResultContract(
        required_fields={"groups": "list", "groups.*.total": "numeric"},
        invariants=[
            {"kind": "count_equals", "output_path": "groups.*", "expected_count": 3},
            {
                "kind": "sum_equals_input",
                "output_path": "groups.*.total",
                "input_step": "s2",
                "input_column": "value",
            },
            {
                "kind": "members_equal_input",
                "output_path": "groups.*.members.*",
                "input_step": "s2",
                "input_column": "id",
            },
        ],
    )


class PreflightTests(unittest.TestCase):
    def test_published_balancing_cannot_be_replaced_by_clustering_code(self):
        with self.assertRaisesRegex(
            PythonPreflightError, "does not implement.*balancing"
        ):
            preflight_python(
                "import scientific\nrows=data['q']\nawait scientific.kmeans([[float(row['value'])] for row in rows])",
                {"q": [{"value": 1}]},
                scientific=True,
                analysis_kind="balancing",
            )

    def check(self, code):
        return preflight_python(
            code, {"query_1": [{"id": "A", "value": Decimal("10")}]}, scientific=True
        )

    def test_numpy_and_scipy_are_rejected_before_native_execution(self):
        for module in ("numpy", "scipy", "pandas", "subprocess"):
            with (
                self.subTest(module=module),
                self.assertRaisesRegex(
                    PythonPreflightError, "Unsupported sandbox import"
                ),
            ):
                self.check(f"import {module}\nrows = data['query_1']\nrows")

    def test_actual_scientific_function_and_return_keys_are_enforced(self):
        with self.assertRaisesRegex(
            PythonPreflightError, "Unknown scientific function 'mean'"
        ):
            self.check(
                "import scientific as s\nrows = data['query_1']\nawait s.mean([1, 2])"
            )
        with self.assertRaisesRegex(
            PythonPreflightError, "convergence.*actual return keys.*converged"
        ):
            self.check(
                "from scientific import kmeans as cluster\nrows = data['query_1']\nresult = await cluster([[float(r['value'])] for r in rows])\nresult['convergence']"
            )
        self.check(
            "import scientific as s\nrows = data['query_1']\nr = await s.kmeans([[float(row['value'])] for row in rows])\nr['converged']"
        )
        self.check(
            "import scientific\nrows=data['query_1']\nresult=await scientific.kmeans([[1.0],[2.0]])\nresult: dict = {'total': len(rows)}\nresult['total']"
        )

    def test_sql_aliases_and_host_input_binding_are_checked(self):
        with self.assertRaisesRegex(PythonPreflightError, "actual columns"):
            self.check("rows = data['query_1']\n[row['invented'] for row in rows]")
        self.check("data = data['query_1']\n[row['value'] for row in data]")
        with self.assertRaisesRegex(PythonPreflightError, "actual columns"):
            self.check("data = data['query_1']\n[row['invented'] for row in data]")
        with self.assertRaisesRegex(PythonPreflightError, "SQL row list"):
            self.check("data = data['query_1']\ndata['query_1']")
        with self.assertRaisesRegex(PythonPreflightError, "Unknown SQL result"):
            self.check("rows = data['query_old']\nrows")


class CalculationContractTests(unittest.TestCase):
    def test_count_accepts_container_paths_and_wildcard_matches_without_ambiguity(self):
        for path in ("groups", "groups.*"):
            contract = PythonResultContract(
                required_fields={"groups": "list"},
                invariants=[
                    {"kind": "count_equals", "output_path": path, "expected_count": 3}
                ],
            )
            validate_python_result({"groups": [{}, {}, {}]}, contract)
            with self.assertRaises(CalculationValidationError):
                validate_python_result({"groups": [{}]}, contract)
        contract = PythonResultContract(
            required_fields={"groups": "list"},
            invariants=[
                {"kind": "count_equals", "output_path": "groups.*", "expected_count": 1}
            ],
        )
        validate_python_result({"groups": [{"a": 1, "b": 2}]}, contract)

    def test_jsonpath_style_is_normalized_and_invariants_require_their_own_fields(self):
        contract = PythonResultContract(
            required_fields={"$.groups[*].total": "numeric"},
            invariants=[
                {
                    "kind": "count_equals",
                    "output_path": "$.groups[*]",
                    "expected_count": 3,
                }
            ],
        )
        self.assertEqual(contract.invariants[0].output_path, "groups.*")
        self.assertIn("groups.*.total", contract.required_fields)
        extra = PythonResultContract(
            required_fields={"members": "list"},
            invariants=[
                {
                    "kind": "count_equals",
                    "output_path": "$.groups[*].members",
                    "expected_count": 3,
                }
            ],
        )
        validate_python_result({"members": [], "groups": [{"members": []}] * 3}, extra)
        with self.assertRaises(CalculationValidationError):
            validate_python_result({"members": []}, extra)
        root = PythonResultContract(
            required_fields={"$": "list"},
            invariants=[
                {"kind": "count_equals", "output_path": "$", "expected_count": 3}
            ],
        )
        validate_python_result([1, 2, 3], root)
        with self.assertRaisesRegex(ValidationError, "expressions and filters"):
            PythonResultContract(required_fields={"groups[?(@.value>1)]": "list"})

    def test_generation_schema_requires_method_contract_and_count_arguments(self):
        schema = planner_output_type("balancing").model_json_schema()
        self.assertEqual(
            schema["$defs"]["RequestedCalculation"]["properties"]["kind"]["const"],
            "balancing",
        )
        self.assertIn("expected_count", schema["$defs"]["CountInvariant"]["required"])
        self.assertIn("input_column", schema["$defs"]["SumInvariant"]["required"])
        self.assertTrue(
            {"calculation", "result_contract"}
            <= set(PythonPlanningStep.model_json_schema()["required"])
        )

    def test_coverage_counts_and_totals_are_independently_checked(self):
        inputs = {
            "q": [
                {"id": "A", "value": Decimal("10")},
                {"id": "B", "value": Decimal("20")},
            ]
        }
        output = {
            "groups": [
                {"members": ["A"], "total": Decimal("10")},
                {"members": ["B"], "total": Decimal("20")},
                {"members": [], "total": Decimal(0)},
            ]
        }
        validate_python_result(output, group_contract(), inputs, {"s2": "q"})
        for defect in ("members", "total", "count"):
            bad = {"groups": [dict(group) for group in output["groups"]]}
            if defect == "members":
                bad["groups"][1]["members"] = ["A"]
            elif defect == "total":
                bad["groups"][0]["total"] += 1
            else:
                bad["groups"].pop()
            with (
                self.subTest(defect=defect),
                self.assertRaises(CalculationValidationError),
            ):
                validate_python_result(bad, group_contract(), inputs, {"s2": "q"})

    def test_nonfinite_values_are_never_accepted_without_a_contract(self):
        for value in (float("nan"), float("inf"), Decimal("NaN")):
            with self.assertRaises(CalculationValidationError):
                validate_python_result({"nested": [value]})
        validate_python_result(Decimal("1e1000"))

    def test_kmeans_cannot_be_declared_as_a_balancing_algorithm(self):
        value = executable_plan("1", "sql_python", declared=True)
        value["steps"][-1]["calculation"] = {
            "kind": "balancing",
            "method": "scientific.kmeans",
        }
        with self.assertRaisesRegex(
            ValidationError, "clustering cannot balance totals"
        ):
            ExecutionPlan.model_validate(value)
        value["steps"][-1]["calculation"] = {
            "kind": "uncertainty",
            "method": "scientific.resample_indices",
        }
        ExecutionPlan.model_validate(value)


class RepairRouteTests(unittest.IsolatedAsyncioTestCase):
    async def test_input_repair_preserves_and_replays_unaffected_sql(self):
        value = executable_plan("1", "sql_python", declared=True)
        second_sql = dict(value["steps"][1], step_id="s3")
        python = dict(value["steps"][2], step_id="s4", input_steps=["s2", "s3"])
        value["steps"] = [*value["steps"][:2], second_sql, python]
        actions = [
            ("describe_sources", {"step_id": "s1"}),
            (
                "run_sql",
                {
                    "step_id": "s2",
                    "statement": "SELECT COUNT(*) AS n FROM cc_customers WHERE customer_id='CUST0001'",
                },
            ),
            (
                "run_sql",
                {
                    "step_id": "s3",
                    "statement": "SELECT COUNT(*) AS n FROM cc_customers",
                },
            ),
            (
                "request_sql_repair",
                {
                    "sql_step_id": "s2",
                    "reason": "Use the full registered population instead of one customer",
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
                "run_sql",
                {
                    "step_id": "s3",
                    "statement": "SELECT COUNT(*) AS n FROM cc_customers",
                },
            ),
            (
                "run_python",
                {
                    "step_id": "s4",
                    "result_ids": ["query_3", "query_2"],
                    "code": "{'total': sum(data[key][0]['n'] for key in data)}",
                },
            ),
        ]
        data = DataSources(create_context(source_ids=["credit_cards"]), "1", "1")

        def respond(messages, info):
            index = len(tool_returns(messages))
            if index == 4:
                self.assertNotIn("query_1", data.results)
                self.assertEqual(data.results["query_2"]["rows"], [{"n": 20}])
            if index == 6:
                replay = json.loads(tool_returns(messages)[-1].content)
                self.assertTrue(replay["reused"])
                self.assertEqual(replay["result_id"], "query_2")
            if index < len(actions):
                name, args = actions[index]
                return ModelResponse(parts=[ToolCallPart(name, args)])
            return ModelResponse(
                parts=[TextPart("Inputs repaired and calculation completed")]
            )

        with patch.object(data, "query", wraps=data.query) as query:
            result = await execute_plan(
                "Compare population totals",
                ExecutionPlan.model_validate(value),
                data,
                FunctionModel(respond),
            )
        self.assertEqual(query.call_count, 3)
        self.assertEqual(len(result.queries), 2)
        self.assertEqual(result.python_runs[-1]["output"], {"total": 40})
        self.assertEqual(result.repair_history[0].target, "sql_input_repair")

    async def test_unsuitable_method_returns_to_planner_instead_of_execution(self):
        calls = 0

        def respond(messages, info):
            nonlocal calls
            calls += 1
            value = executable_plan("1", "sql_python", declared=True)
            value["steps"][-1]["calculation"] = {
                "kind": "balancing",
                "method": "scientific.kmeans" if calls == 1 else "largest_first_greedy",
            }
            value["steps"][-1]["result_contract"] = group_contract().model_dump(
                mode="json"
            )
            if calls == 2:
                repair = json.loads(tool_returns(messages)[-1].content)
                self.assertEqual(repair["target"], "planner")
                self.assertIn("clustering cannot balance totals", repair["diagnostic"])
                self.assertGreaterEqual(repair["retries_remaining"], 0)
            return structured_response(info, value)

        result = await build_plan(
            "Split entities into three roughly equal-total review groups",
            QuestionIntent(
                intent="advanced_analysis",
                objective="Balance group totals",
                analysis_kind="balancing",
            ),
            tenant_id="1",
            workspace_id="1",
            model=FunctionModel(respond),
        )
        self.assertEqual(calls, 2)
        self.assertEqual(result.steps[-1].calculation.method, "largest_first_greedy")

    async def test_preflight_and_result_failures_repair_python_without_rerunning_sql(
        self,
    ):
        value = executable_plan(
            "1",
            "sql_python",
            declared=True,
            output_contract={
                "required_columns": {
                    "id": {"type": "string"},
                    "value": {"type": "numeric"},
                },
                "grain": ["id"],
            },
        )
        value["steps"][-1]["result_contract"] = group_contract().model_dump(mode="json")
        value["steps"][-1]["calculation"] = {
            "kind": "balancing",
            "method": "custom_assignment",
        }
        plan = ExecutionPlan.model_validate(value)
        base = "from decimal import Decimal\nrows = data['query_1']\ngroups = [{'members': [], 'total': Decimal('0')} for _ in range(3)]\nfor i, row in enumerate(rows):\n    groups[i % 3]['members'].append(row['id'])\n    groups[i % 3]['total'] += row['value']\n"
        actions = [
            ("describe_sources", {"step_id": "s1"}),
            (
                "run_sql",
                {
                    "step_id": "s2",
                    "statement": "SELECT customer_id AS id, 1 AS value FROM cc_customers ORDER BY customer_id",
                },
            ),
            (
                "run_python",
                {
                    "step_id": "s3",
                    "result_ids": ["query_1"],
                    "code": "import numpy as np\nrows = data['query_1']\nrows",
                },
            ),
            (
                "run_python",
                {
                    "step_id": "s3",
                    "result_ids": ["query_1"],
                    "code": "import scientific\nrows=data['query_1']\nanalysis=await scientific.kmeans([[float(row['value'])] for row in rows])\nanalysis['convergence']",
                },
            ),
            (
                "run_python",
                {
                    "step_id": "s3",
                    "result_ids": ["query_1"],
                    "code": base + "groups[0]['total'] += 1\n{'groups': groups}",
                },
            ),
            (
                "run_python",
                {
                    "step_id": "s3",
                    "result_ids": ["query_1"],
                    "code": base + "{'groups': groups}",
                },
            ),
        ]

        def respond(messages, info):
            index = len(tool_returns(messages))
            if index in (3, 4, 5):
                handoff = next(
                    part.content
                    for part in messages[-1].parts
                    if isinstance(part, UserPromptPart)
                    and isinstance(part.content, str)
                    and part.content.startswith("Host validation repair handoff")
                )
                self.assertIn("python_generator", handoff)
                self.assertIn("Call run_python", handoff)
                self.assertIn("Preserve completed SQL", handoff)
                repair = json.loads(
                    tool_returns(messages)[-1].content.removeprefix(
                        "Python execution failed: "
                    )
                )
                self.assertEqual(repair["target"], "python_generator")
                self.assertEqual(
                    repair["phase"],
                    "calculation_result" if index == 5 else "python_preflight",
                )
                self.assertEqual(
                    repair["authority"]["sql_columns"], {"query_1": ["id", "value"]}
                )
            if index < len(actions):
                name, arguments = actions[index]
                return ModelResponse(parts=[ToolCallPart(name, arguments)])
            return ModelResponse(parts=[TextPart("Groups calculated")])

        data = DataSources(create_context(source_ids=["credit_cards"]), "1", "1")
        output = io.StringIO()
        token = TRACE_ENABLED.set(True)
        try:
            with (
                patch.dict(os.environ, {"AGENT_TOOL_RETRIES": "3"}),
                redirect_stderr(output),
            ):
                result = await execute_plan(
                    "Balance totals", plan, data, FunctionModel(respond)
                )
        finally:
            TRACE_ENABLED.reset(token)
        self.assertEqual(len(result.queries), 1)
        self.assertEqual(len(result.python_runs), 1)
        self.assertEqual(output.getvalue().count("sandbox execution started"), 2)
        self.assertEqual(
            [item.target for item in result.repair_history], ["python_generator"] * 3
        )
        self.assertEqual(
            [item.retries_remaining for item in result.repair_history], [3, 2, 1]
        )
        self.assertEqual(result.completed_steps, ["s1", "s2", "s3"])
        validate_execution_result(plan, result)
        self.assertIn("result_contract", result.validation_receipts[-1]["checks"])
