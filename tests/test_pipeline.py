"""Contract tests for replaceable handlers, evidence handoff and domain summaries."""

import json
import io
import os
import sys
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import Mock, patch

from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel
from pydantic_monty import AsyncMonty, MontyRuntimeError
from test_agent import executable_plan

from execution import ExecutionResult
from pipeline import (
    Pipeline,
    PipelineModels,
    PipelineRequest,
    PipelineServices,
    PipelineStage,
)
from planning import ExecutionPlan, QuestionIntent
from summarization import summarize_result
from calculation_contracts import CalculationValidationError


def adapters(events, *, no_execution=False):
    intent = QuestionIntent(intent="aggregation", objective="Count customers")
    plan = ExecutionPlan.model_validate(executable_plan())
    if no_execution:
        plan = ExecutionPlan(
            tenant_id="1",
            workspace_id="1",
            sources=[],
            execution_mode="none",
            steps=[],
            validation_checks=[],
            response="Which period do you mean?",
        )

    async def classify(*args, **kwargs):
        events.append("classify")
        return intent

    async def build(*args, **kwargs):
        events.append("plan")
        return plan

    def publish(*args, **kwargs):
        events.append("publish")
        return Path("/tmp/pipeline-test-plan.json")

    def load(*args):
        events.append("load")
        return object()

    async def execute(*args):
        events.append("execute")
        return ExecutionResult(
            "There are 20 customers.",
            {"query_1": {"rows": [{"n": 20}]}},
            ["SELECT count(*) FROM cc_customers"],
            [],
            ["s1", "s2"],
        )

    async def summarize(question, plan, evidence, model):
        events.append("summarize")
        if no_execution:
            assert evidence is None
            return plan.response
        assert evidence.sql_results["query_1"]["rows"] == [{"n": 20}]
        return "Your card workspace has 20 customers."

    return PipelineServices(
        classify=classify,
        plan=build,
        publish=publish,
        load=load,
        execute=execute,
        summarize=summarize,
    )


class PipelineTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        patcher = patch.dict(os.environ, {"AGENT_SUMMARIZATION_ENABLED": "true"})
        patcher.start()
        self.addCleanup(patcher.stop)

    async def test_graph_validates_evidence_before_summary(self):
        services = adapters([])
        services.validate = Mock(
            side_effect=CalculationValidationError("Incomplete evidence")
        )
        services.summarize = Mock()
        with self.assertRaisesRegex(CalculationValidationError, "Incomplete evidence"):
            await Pipeline(services).run(
                PipelineRequest("Count customers", "1", "1"),
                PipelineModels(default=object()),
            )
        services.validate.assert_called_once()
        services.summarize.assert_not_called()

    async def test_disabled_summary_returns_executor_report_without_summary_call(self):
        events = []
        output = io.StringIO()
        with (
            patch.dict(os.environ, {"AGENT_SUMMARIZATION_ENABLED": "false"}),
            redirect_stderr(output),
        ):
            context = await Pipeline(adapters(events)).run(
                PipelineRequest("How many customers?", "1", "1"),
                PipelineModels(default=object()),
            )
        self.assertEqual(events, ["classify", "plan", "publish", "load", "execute"])
        self.assertEqual(context.answer, "There are 20 customers.")
        self.assertEqual(context.skipped_stages, ["summarize"])
        self.assertNotIn("summarize", context.completed_stages)
        self.assertIn("STAGE summarize: end (skipped,", output.getvalue())
        self.assertEqual(context.execution.sql_results["query_1"]["rows"], [{"n": 20}])

    async def test_disabled_summary_preserves_clarifications_and_no_execution_response(
        self,
    ):
        from planning import DeferredObjective

        events = []
        services = adapters(events)
        original_plan = services.plan

        async def plan(*args, **kwargs):
            result = await original_plan(*args, **kwargs)
            result.deferred_objectives = [
                DeferredObjective(
                    objective="Compare periods", clarification="Which period?"
                )
            ]
            return result

        services.plan = plan
        with patch.dict(os.environ, {"AGENT_SUMMARIZATION_ENABLED": "false"}):
            context = await Pipeline(services).run(
                PipelineRequest("Count customers and compare periods", "1", "1"),
                PipelineModels(default=object()),
            )
            no_execution = await Pipeline(adapters([], no_execution=True)).run(
                PipelineRequest("Count customers in that period", "1", "1"),
                PipelineModels(default=object()),
            )
        self.assertEqual(context.answer, "There are 20 customers.\n\nWhich period?")
        self.assertEqual(no_execution.answer, "Which period do you mean?")
        self.assertIsNone(no_execution.execution)

    async def test_invalid_summary_toggle_fails_before_any_stage(self):
        services = adapters([])
        services.resolve = Mock()
        with patch.dict(os.environ, {"AGENT_SUMMARIZATION_ENABLED": "maybe"}):
            with self.assertRaisesRegex(ValueError, "AGENT_SUMMARIZATION_ENABLED"):
                await Pipeline(services).run(
                    PipelineRequest("Count customers", "1", "1")
                )
        services.resolve.assert_not_called()

    async def test_full_chain_and_final_summary(self):
        events = []
        pipeline = Pipeline(adapters(events))
        context = await pipeline.run(
            PipelineRequest("How many customers?", "1", "1"),
            PipelineModels(default=object()),
        )
        self.assertEqual(
            context.completed_stages,
            [
                "resolve",
                "classify",
                "plan",
                "publish",
                "load_sources",
                "execute",
                "validate_results",
                "summarize",
            ],
        )
        self.assertEqual(
            events, ["classify", "plan", "publish", "load", "execute", "summarize"]
        )
        self.assertEqual(context.answer, "Your card workspace has 20 customers.")
        self.assertEqual(context.publication, Path("/tmp/pipeline-test-plan.json"))
        for current, following in zip(pipeline.stages, pipeline.stages[1:]):
            self.assertIs(current.next_stage, following)

    async def test_previews_stop_at_the_requested_stage(self):
        for mode, expected in [
            ("classify", ["classify"]),
            ("plan", ["classify", "plan"]),
            ("publish_plan", ["classify", "plan", "publish"]),
        ]:
            events = []
            result = await Pipeline(adapters(events)).run(
                PipelineRequest("Count customers", "1", "1", mode=mode),
                PipelineModels(default=object()),
            )
            self.assertEqual(events, expected)
            self.assertTrue(result.stopped)

    async def test_no_execution_skips_data_but_reaches_summary(self):
        events = []
        result = await Pipeline(adapters(events, no_execution=True)).run(
            PipelineRequest("Count customers in that period", "1", "1"),
            PipelineModels(default=object()),
        )
        self.assertEqual(events, ["classify", "plan", "publish", "summarize"])
        self.assertEqual(result.answer, "Which period do you mean?")
        self.assertIsNone(result.sources)

    async def test_failure_prevents_successor_execution(self):
        events = []
        services = adapters(events)
        services.publish = Mock(side_effect=OSError("Publication failed"))
        with self.assertRaises(OSError):
            await Pipeline(services).run(
                PipelineRequest("Count customers", "1", "1"),
                PipelineModels(default=object()),
            )
        self.assertEqual(events, ["classify", "plan"])

    async def test_custom_handler_chain(self):
        class TagStage(PipelineStage):
            name = "tag"

            async def process(self, context):
                context.answer = (context.answer or "") + "tagged "

        services = PipelineServices()
        chain = Pipeline(services, stages=[TagStage(services), TagStage(services)])
        result = await chain.run(PipelineRequest("A request", "1", "1"))
        self.assertEqual(result.answer, "tagged tagged ")
        self.assertEqual(result.completed_stages, ["tag", "tag"])

    async def test_summarizer_uses_domain_tone_and_actual_evidence_without_tools(self):
        plan = ExecutionPlan.model_validate(executable_plan("2"))
        evidence = ExecutionResult(
            "Technical draft.",
            {
                "query_1": {
                    "rows": [
                        {
                            "amc_name": "Sample Horizon AMC",
                            "market_value": "70782.00",
                            "nav_date": "2026-06-30",
                        }
                    ]
                }
            },
        )

        def model(messages, info):
            self.assertEqual(info.function_tools, [])
            self.assertIn("mutual fund operations manager", info.instructions)
            self.assertIn("Asset Management Company", info.instructions)
            payload = json.loads(messages[-1].parts[0].content)
            self.assertEqual(payload["sql_results"], evidence.sql_results)
            return ModelResponse(
                parts=[
                    TextPart("Sample Horizon AMC holds INR 70,782 as of June 30, 2026.")
                ]
            )

        answer = await summarize_result(
            "Show holdings", plan, evidence, FunctionModel(model)
        )
        self.assertIn("INR 70,782", answer)
        self.assertNotIn("Technical draft", answer)

    async def test_monty_cannot_import_or_run_datafusion(self):
        async with AsyncMonty(
            binary_path=str(Path(sys.executable).parent / "monty"), max_processes=1
        ) as pool:
            async with pool.checkout() as session:
                with self.assertRaises(MontyRuntimeError):
                    await session.feed_run(
                        "import datafusion\ndatafusion.SessionContext()"
                    )
