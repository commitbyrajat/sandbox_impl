"""The generator sees the actual data binding; the sandbox keeps native values."""

import json
import unittest
from datetime import date
from decimal import Decimal

from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from data_access import DataSources
from execution import execute_plan
from main import create_context
from planning import ExecutionPlan
from python_input_context import python_data_context
from test_agent import executable_plan, tool_returns


class SnapshotTests(unittest.TestCase):
    def test_exact_json_shape_and_native_types_are_both_provided(self):
        row = {
            "amount": Decimal("123.4500"),
            "day": date(2026, 6, 1),
            "optional": None,
            "count": 2,
            "approved": True,
        }
        results = {
            "query_7": {
                "rows": [row],
                "columns": list(row),
                "truncated": False,
                "column_types": [
                    {"name": "amount", "type": "decimal128(18, 4)", "nullable": False}
                ],
            }
        }
        context = python_data_context(results, ["query_7"])
        self.assertEqual(
            json.loads(context["data_json"]),
            {
                "query_7": [
                    {
                        "amount": "123.4500",
                        "day": "2026-06-01",
                        "optional": None,
                        "count": 2,
                        "approved": True,
                    }
                ]
            },
        )
        types = context["results"]["query_7"]["columns"]
        self.assertEqual(types["amount"]["python_types"], ["decimal.Decimal"])
        self.assertEqual(types["day"]["python_types"], ["datetime.date"])
        self.assertEqual(types["optional"]["python_types"], ["builtins.NoneType"])
        self.assertEqual(types["amount"]["arrow_type"], "decimal128(18, 4)")
        self.assertIsInstance(row["amount"], Decimal)
        self.assertIsInstance(row["day"], date)
        self.assertTrue(context["snapshot_complete"])

    def test_only_current_result_ids_are_serialized(self):
        results = {
            key: {"rows": [{"value": value}], "columns": ["value"]}
            for key, value in (("query_old", 10), ("query_current", 20))
        }
        context = python_data_context(results, ["query_current"])
        self.assertEqual(
            json.loads(context["data_json"]), {"query_current": [{"value": 20}]}
        )
        self.assertNotIn("query_old", context["results"])

    def test_large_utf8_snapshots_are_explicit_bounded_previews(self):
        rows = [{"text": "₹" * 30, "id": i} for i in range(20)]
        results = {
            key: {"rows": rows, "columns": ["text", "id"]} for key in ("q1", "q2")
        }
        context = python_data_context(results, ["q1", "q2"], max_bytes=400)
        data = json.loads(context["data_json"])
        self.assertFalse(context["snapshot_complete"])
        self.assertLessEqual(len(context["data_json"].encode("utf-8")), 400)
        for key in ("q1", "q2"):
            self.assertGreater(len(data[key]), 0)
            self.assertEqual(context["results"][key]["stored_row_count"], 20)
            self.assertEqual(
                context["results"][key]["snapshot_row_count"], len(data[key])
            )
        self.assertEqual(len(results["q1"]["rows"]), 20)

    def test_unavailable_or_sql_truncated_rows_are_not_presented_as_usable(self):
        context = python_data_context(
            {"q": {"rows": [{"x": 1}], "columns": ["x"], "truncated": True}},
            ["q", "stale"],
        )
        self.assertEqual(json.loads(context["data_json"]), {})
        self.assertFalse(context["snapshot_complete"])
        self.assertFalse(context["results"]["q"]["usable_in_python"])
        self.assertEqual(len(context["input_issues"]), 2)


class GenerationContextTests(unittest.IsolatedAsyncioTestCase):
    async def test_json_binding_is_visible_before_python_generation_and_native_in_monty(
        self,
    ):
        plan = ExecutionPlan.model_validate(
            executable_plan(
                "1",
                "sql_python",
                declared=True,
                output_contract={
                    "required_columns": {
                        "amount": {"type": "numeric"},
                        "day": {"type": "date"},
                    }
                },
            )
        )
        actions = [
            ("describe_sources", {"step_id": "s1"}),
            (
                "run_sql",
                {
                    "step_id": "s2",
                    "statement": "SELECT CAST('1.25' AS DECIMAL(12,2)) AS amount, DATE '2026-06-01' AS day",
                },
            ),
            (
                "run_python",
                {
                    "step_id": "s3",
                    "result_ids": ["query_1"],
                    "code": "from decimal import Decimal\nrows = data['query_1']\n{'total': rows[0]['amount'] + Decimal('0.75'), 'year': rows[0]['day'].year}",
                },
            ),
        ]

        def respond(messages, info):
            index = len(tool_returns(messages))
            if index == 2:
                binding = json.loads(
                    info.instructions.split("HOST_DATA_OBJECT_JSON_BEGIN\n", 1)[
                        1
                    ].split("\nHOST_DATA_OBJECT_JSON_END", 1)[0]
                )
                self.assertEqual(
                    binding, {"query_1": [{"amount": "1.25", "day": "2026-06-01"}]}
                )
                self.assertIn("decimal.Decimal", info.instructions)
                self.assertIn("datetime.date", info.instructions)
                self.assertIn('"snapshot_complete": true', info.instructions)
            if index < len(actions):
                name, args = actions[index]
                return ModelResponse(parts=[ToolCallPart(name, args)])
            return ModelResponse(parts=[TextPart("Calculated")])

        result = await execute_plan(
            "Calculate from the SQL rows",
            plan,
            DataSources(create_context(source_ids=["credit_cards"]), "1", "1"),
            FunctionModel(respond),
        )
        self.assertEqual(
            result.python_runs[0]["output"], {"total": Decimal("2.00"), "year": 2026}
        )
