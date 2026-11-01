"""HTTP metadata, real pipeline SSE ordering, isolation and cancellation."""

import asyncio
import json
import unittest
from unittest.mock import Mock, patch

import httpx2 as httpx
from fastapi.testclient import TestClient

from api import QuestionRequest, create_app
from pipeline import Pipeline
from planning import QuestionIntent
from progress import PROGRESS_SINK, emit_progress
from runtime import trace
from test_pipeline import adapters


def events(response):
    return [
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]


def fake_pipeline():
    services = adapters([])
    services.model_factory = lambda *args: object()
    services.validate = lambda *args: None
    return Pipeline(services)


class APITests(unittest.TestCase):
    def setUp(self):
        self.factory = Mock(side_effect=fake_pipeline)
        self.client = TestClient(create_app(self.factory))
        self.body = {
            "tenant_id": "1",
            "workspace_id": "1",
            "question": "Count customers",
        }

    def test_metadata_is_scoped_and_does_not_expose_connections(self):
        result = self.client.get("/api/tenants")
        self.assertEqual(result.status_code, 200)
        self.assertEqual(len(result.json()["stages"]), 8)
        scopes = self.client.get("/api/tenants/1/workspaces").json()["workspaces"]
        self.assertEqual(scopes[0]["id"], "1")
        self.assertGreater(scopes[0]["table_count"], 0)
        self.assertNotIn("location", json.dumps(scopes))
        self.assertNotIn("api_key", result.text)
        self.assertEqual(
            self.client.get("/api/tenants/missing/workspaces").status_code, 404
        )
        self.factory.assert_not_called()

    def test_invalid_questions_and_cross_tenant_workspace_fail_before_model_calls(self):
        for changes, status in [
            ({"question": "  "}, 422),
            ({"max_tokens": 0}, 422),
            ({"max_tokens": True}, 422),
            ({"extra": 1}, 422),
            ({"workspace_id": "2"}, 404),
            ({"tenant_id": "missing"}, 404),
        ]:
            response = self.client.post(
                "/api/questions/stream", json={**self.body, **changes}
            )
            self.assertEqual(response.status_code, status, changes)
        self.factory.assert_not_called()

    def test_every_stage_has_an_outcome_before_the_final_answer(self):
        response = self.client.post("/api/questions/stream", json=self.body)
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/event-stream", response.headers["content-type"])
        data = events(response)
        stages = [e for e in data if e["kind"] == "stage"]
        self.assertEqual([e["status"] for e in stages], ["running", "completed"] * 8)
        complete = [e for e in stages if e["status"] == "completed"]
        self.assertEqual(
            [e["stage"] for e in complete], [s["id"] for s in data[0]["stages"]]
        )
        self.assertTrue(
            all(e["duration_seconds"] >= 0 and e["outcome"] for e in complete)
        )
        self.assertEqual(data[-1]["kind"], "result")
        self.assertEqual(data[-1]["answer"], "Your card workspace has 20 customers.")
        self.assertNotIn(".env", response.text)

    def test_failure_is_a_stage_outcome_and_never_a_final_answer(self):
        def fail():
            pipeline = fake_pipeline()

            async def classify(*args, **kwargs):
                raise ValueError("The selected period is unavailable.")

            pipeline.services.classify = classify
            return pipeline

        response = TestClient(create_app(fail)).post(
            "/api/questions/stream", json=self.body
        )
        data = events(response)
        self.assertEqual(data[-1]["kind"], "error")
        self.assertIn("selected period", data[-1]["error"]["message"])
        self.assertTrue(
            any(
                e.get("stage") == "classify" and e.get("status") == "failed"
                for e in data
            )
        )
        self.assertFalse(any(e["kind"] == "result" for e in data))

    def test_error_messages_redact_secrets_and_connection_credentials(self):
        def fail():
            pipeline = fake_pipeline()

            async def classify(*args, **kwargs):
                raise ValueError(
                    "secret-value at postgresql://analyst:db-password@localhost/db"
                )

            pipeline.services.classify = classify
            return pipeline

        with patch.dict("os.environ", {"EXAMPLE_API_KEY": "secret-value"}):
            response = TestClient(create_app(fail)).post(
                "/api/questions/stream", json=self.body
            )
        self.assertNotIn("secret-value", response.text)
        self.assertNotIn("db-password", response.text)
        self.assertIn("[redacted]", response.text)


class AsyncAPITests(unittest.IsolatedAsyncioTestCase):
    async def test_concurrent_streams_keep_question_events_isolated(self):
        def factory():
            pipeline = fake_pipeline()

            async def classify(question, **kwargs):
                await asyncio.sleep(0.01)
                await asyncio.to_thread(trace, "TOOL question: " + question)
                return QuestionIntent(intent="aggregation", objective=question)

            pipeline.services.classify = classify
            return pipeline

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_app(factory)),
            base_url="http://test",
        ) as client:
            first, second = await asyncio.gather(
                *[
                    client.post(
                        "/api/questions/stream",
                        json={"tenant_id": "1", "workspace_id": "1", "question": q},
                    )
                    for q in ("alpha-unique-question", "beta-unique-question")
                ]
            )
        self.assertIn("alpha-unique-question", first.text)
        self.assertNotIn("beta-unique-question", first.text)
        self.assertIn("beta-unique-question", second.text)
        self.assertNotIn("alpha-unique-question", second.text)
        self.assertEqual(events(first)[-1]["kind"], "result")
        self.assertIsNone(PROGRESS_SINK.get())

    async def test_closing_stream_cancels_the_request_task(self):
        cancelled = asyncio.Event()

        class SlowPipeline:
            async def run(self, request):
                emit_progress("stage", stage="classify", status="running")
                try:
                    await asyncio.Event().wait()
                finally:
                    cancelled.set()

        app = create_app(SlowPipeline)
        endpoint = next(
            route.endpoint
            for route in app.routes
            if getattr(route, "path", "") == "/api/questions/stream"
        )
        response = await endpoint(
            QuestionRequest(tenant_id="1", workspace_id="1", question="Wait")
        )
        stream = response.body_iterator
        self.assertIn("ready", await anext(stream))
        self.assertIn("running", await anext(stream))
        await stream.aclose()
        self.assertTrue(cancelled.is_set())
        self.assertIsNone(PROGRESS_SINK.get())


if __name__ == "__main__":
    unittest.main()
