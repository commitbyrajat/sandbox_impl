"""A plan must be published successfully before the agent opens its data."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

from test_agent import executable_plan
from test_support import ask
from test_tenants import answering_model

from main import create_context
from plan_publication import publish_plan
from planning import ExecutionPlan, QuestionIntent


class PlanPublicationTests(unittest.IsolatedAsyncioTestCase):
    async def test_publication_is_scoped_private_and_repeatable(self):
        intent = QuestionIntent(intent="aggregation", objective="Count customers")
        plan = ExecutionPlan.model_validate(executable_plan())
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as directory, redirect_stderr(output):
            first = publish_plan(
                "How many customers?", intent, plan, directory=Path(directory)
            )
            second = publish_plan(
                "How many customers?", intent, plan, directory=Path(directory)
            )
            self.assertNotEqual(first, second)
            self.assertEqual(first.parent, Path(directory) / "tenant-1" / "workspace-1")
            self.assertEqual(first.stat().st_mode & 0o777, 0o600)
            record = json.loads(first.read_text())
            self.assertEqual(record["status"], "planned")
            self.assertEqual(record["model_name"], "qwen-coder")
            self.assertEqual(record["plan"], plan.model_dump(mode="json"))
            self.assertEqual(record["intent"], intent.model_dump(mode="json"))
            self.assertIn("published_at", record)
            self.assertNotIn("LITELLM_MASTER_KEY", first.read_text())
        self.assertIn("s1 [schema]", output.getvalue())
        self.assertIn("s2 [sql]", output.getvalue())

    async def test_publication_precedes_source_loading_and_execution(self):
        events = []
        with tempfile.TemporaryDirectory() as directory:

            def publish(*args, **kwargs):
                events.append("publish")
                return publish_plan(*args, **kwargs)

            def load(*args, **kwargs):
                events.append("load")
                files = list(Path(directory).rglob("*.json"))
                self.assertEqual(len(files), 1)
                self.assertEqual(json.loads(files[0].read_text())["tenant_id"], "1")
                return create_context(*args, **kwargs)

            with (
                redirect_stderr(io.StringIO()),
                patch("pipeline.publish_plan", side_effect=publish),
                patch("data_access.create_context", side_effect=load),
            ):
                answer = await ask(
                    "How many customers?",
                    tenant_id="1",
                    workspace_id="1",
                    model=answering_model("1", "1", "cc_customers", 20, events),
                    plan_dir=Path(directory),
                )
        self.assertEqual(answer, "20 customers.")
        self.assertEqual(
            events, ["classify", "plan", "publish", "load", "schema", "sql", "answer"]
        )

    async def test_failed_publication_prevents_data_loading(self):
        events = []
        with (
            patch("pipeline.publish_plan", side_effect=OSError("Cannot save plan")),
            patch("data_access.create_context") as loader,
        ):
            with self.assertRaises(OSError):
                await ask(
                    "How many customers?",
                    tenant_id="1",
                    workspace_id="1",
                    model=answering_model("1", "1", "cc_customers", 20, events),
                )
            loader.assert_not_called()
        self.assertEqual(events, ["classify", "plan"])
