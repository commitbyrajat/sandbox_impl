"""Timeout budgets, stage diagnosis and bounded model context regressions."""

import os
import unittest
from decimal import Decimal
from unittest.mock import patch

from pydantic_ai.exceptions import ModelRequestTimeout, ModelTimeoutError
from test_pipeline import adapters

from domain_instructions import DOMAIN_PROFILES
from execution import ExecutionResult
from pipeline import Pipeline, PipelineModels, PipelineRequest
from runtime import PipelineModelTimeout, model_settings, result_preview
from summary_evidence import prepare_evidence


class RuntimeTests(unittest.IsolatedAsyncioTestCase):
    def test_shared_model_budget_and_validation(self):
        with patch("runtime.load_dotenv"), patch.dict(os.environ, {}, clear=True):
            settings = model_settings()
            self.assertEqual(settings["timeout"], 300)
            self.assertEqual(settings["request_timeout"], 330)
            self.assertEqual(settings["max_retries"], 0)
            self.assertEqual(settings["max_tokens"], 4096)
        with (
            patch("runtime.load_dotenv"),
            patch.dict(
                os.environ,
                {"LITELLM_TIMEOUT_SECONDS": "400", "LITELLM_MAX_OUTPUT_TOKENS": "8192"},
                clear=True,
            ),
        ):
            self.assertEqual(model_settings()["request_timeout"], 430)
            self.assertEqual(model_settings()["max_tokens"], 8192)
        for overrides in [
            {"LITELLM_TIMEOUT_SECONDS": "nan"},
            {"LITELLM_MAX_RETRIES": "-1"},
            {"LITELLM_REQUEST_DEADLINE_SECONDS": "10"},
        ]:
            with (
                patch("runtime.load_dotenv"),
                patch.dict(os.environ, overrides, clear=True),
            ):
                with self.assertRaises(ValueError):
                    model_settings()

    async def test_timeout_names_stage_and_stops_successors(self):
        for error in [
            ModelTimeoutError("qwen-coder", "read timeout"),
            ModelRequestTimeout("qwen-coder", 330),
        ]:
            events = []
            services = adapters(events)

            async def fail(*args, **kwargs):
                raise error

            services.classify = fail
            with self.assertRaises(PipelineModelTimeout) as raised:
                await Pipeline(services).run(
                    PipelineRequest("Count customers", "1", "1"),
                    PipelineModels(default=object()),
                )
            self.assertEqual(raised.exception.stage, "classify")
            self.assertEqual(raised.exception.model_name, "qwen-coder")
            self.assertIs(raised.exception.__cause__, error)
            self.assertEqual(events, [])

    def test_preview_keeps_complete_host_rows_and_truncation_semantics(self):
        original = {
            "rows": [{"amount": Decimal(i)} for i in range(163)],
            "truncated": False,
            "result_id": "query_1",
        }
        view = result_preview(original)
        self.assertEqual(len(view["rows"]), 12)
        self.assertEqual(view["stored_row_count"], 163)
        self.assertTrue(view["complete_rows_available_for_python"])
        self.assertFalse(view["truncated"])
        self.assertEqual(len(original["rows"]), 163)
        self.assertFalse(
            result_preview({**original, "truncated": True})[
                "complete_rows_available_for_python"
            ]
        )
        self.assertIs(result_preview({"rows": []}).get("rows_preview_only"), None)

    def test_requested_confidence_level_is_not_a_monetary_fact(self):
        evidence = prepare_evidence(
            ExecutionResult(
                "", python_runs=[{"output": {"mean_daily_inr": Decimal("100.25")}}]
            ),
            DOMAIN_PROFILES["credit_cards"],
            question="Give a 95% confidence range.",
        )
        self.assertEqual(evidence.unsupported_values("At 95%, mean INR 100.25."), [])
        self.assertIn("95", evidence.unsupported_values("Mean INR 95."))
        self.assertIn("99", evidence.unsupported_values("At 99%, mean INR 100.25."))

    def test_executed_method_parameters_do_not_authorize_monetary_claims(self):
        evidence = prepare_evidence(
            ExecutionResult(
                "",
                python_runs=[
                    {
                        "code": "random.seed(42)\nfor i in range(1000): pass\npercentiles = (2.5, 97.5)",
                        "output": {"mean": Decimal("100.25")},
                    }
                ],
            ),
            DOMAIN_PROFILES["credit_cards"],
            question="Give a 95% confidence range.",
        )
        self.assertEqual(
            evidence.unsupported_values(
                "95% confidence using 1,000 resamples, seed 42, 2.5th and 97.5th percentiles; mean INR 100.25."
            ),
            [],
        )
        self.assertIn("42", evidence.unsupported_values("Mean INR 42."))
        self.assertIn("999", evidence.unsupported_values("999 resamples."))
        self.assertIn("99.5", evidence.unsupported_values("99.5th percentile."))
