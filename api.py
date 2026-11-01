"""Local HTTP API and live, request-scoped analysis event stream.

Run with: uv run uvicorn api:app --host 127.0.0.1 --port 8010
"""

import asyncio
from collections.abc import Callable
from contextlib import suppress
import json
import threading
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic_core import to_jsonable_python

from datasource_mapping import TENANTS, resolve_scope
from pipeline import Pipeline, PipelineRequest
from progress import PROGRESS_SINK, STAGES
from runtime import redact_sensitive


class QuestionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tenant_id: str = Field(min_length=1, max_length=64)
    workspace_id: str = Field(min_length=1, max_length=64)
    question: str = Field(min_length=1, max_length=8000)
    max_tokens: int = Field(default=8192, ge=1, le=32768, strict=True)

    @field_validator("question")
    @classmethod
    def question_not_blank(cls, value):
        if not value.strip():
            raise ValueError("Enter a question to begin your analysis.")
        return value.strip()


def create_app(pipeline_factory: Callable[[], Pipeline] = Pipeline) -> FastAPI:
    app = FastAPI(title="HDFC Insight API", version="1.0.0")

    @app.get("/api/health")
    async def health():
        return {"status": "ok", "service": "HDFC Insight"}

    @app.get("/api/tenants")
    async def tenants():
        return {
            "tenants": [
                {"id": key, "name": tenant.name} for key, tenant in TENANTS.items()
            ],
            "stages": STAGES,
        }

    @app.get("/api/tenants/{tenant_id}/workspaces")
    async def workspaces(tenant_id: str):
        if tenant_id not in TENANTS:
            raise HTTPException(404, "Tenant not found.")
        return {
            "workspaces": [
                {
                    "id": key,
                    "name": workspace.name,
                    "domain": workspace.domain_profile,
                    "sources": list(workspace.sources),
                    "table_count": sum(
                        len(source.tables) for source in workspace.sources.values()
                    ),
                }
                for key, workspace in TENANTS[tenant_id].workspaces.items()
            ]
        }

    @app.post("/api/questions/stream", response_class=StreamingResponse)
    async def question(body: QuestionRequest):
        try:
            resolve_scope(body.tenant_id, body.workspace_id)
        except ValueError as error:
            raise HTTPException(404, str(error)) from error

        async def stream():
            loop = asyncio.get_running_loop()
            loop_thread = threading.get_ident()
            queue: asyncio.Queue[dict | None] = asyncio.Queue()
            run_id = uuid4().hex
            activity_count = 0
            closed = False

            def enqueue(event):
                nonlocal activity_count
                # Lifecycle/results are never dropped. Keep a bounded activity
                # transcript even if a model enters a long correction loop.
                if event["kind"] == "activity":
                    activity_count += 1
                    if activity_count > 500:
                        return
                queue.put_nowait(event)

            def observe(event):
                if closed:
                    return
                if threading.get_ident() == loop_thread:
                    enqueue(event)
                else:
                    loop.call_soon_threadsafe(enqueue, event)

            async def run():
                token = PROGRESS_SINK.set(observe)
                try:
                    context = await pipeline_factory().run(
                        PipelineRequest(
                            body.question,
                            body.tenant_id,
                            body.workspace_id,
                            max_tokens=body.max_tokens,
                        )
                    )
                    queue.put_nowait(
                        {
                            "kind": "result",
                            "run_id": run_id,
                            "answer": context.answer,
                            "timings": context.stage_timings,
                            "completed_stages": context.completed_stages,
                            "skipped_stages": context.skipped_stages,
                        }
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    queue.put_nowait(
                        {
                            "kind": "error",
                            "run_id": run_id,
                            "error": {
                                "type": type(error).__name__,
                                "message": redact_sensitive(str(error))[:2000]
                                or "Analysis could not complete. Please try again.",
                            },
                        }
                    )
                finally:
                    PROGRESS_SINK.reset(token)
                    # Flush thread-safe callbacks before the end marker.
                    loop.call_soon(queue.put_nowait, None)

            task = asyncio.create_task(run(), name=f"analysis-{run_id}")
            sequence = 0
            try:
                yield "event: ready\ndata: " + json.dumps(
                    {"kind": "ready", "run_id": run_id, "stages": STAGES}
                ) + "\n\n"
                while True:
                    try:
                        event = await asyncio.wait_for(queue.get(), timeout=10)
                    except TimeoutError:
                        yield ": heartbeat\n\n"
                        continue
                    if event is None:
                        break
                    sequence += 1
                    payload = json.dumps(
                        to_jsonable_python(event, fallback=str), ensure_ascii=False
                    )
                    yield f'id: {sequence}\nevent: {event["kind"]}\ndata: {payload}\n\n'
            finally:
                # Closing the browser stream also cancels the model/sandbox task.
                closed = True
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "X-Accel-Buffering": "no",
            },
        )

    dist = Path(__file__).resolve().parent / "ui" / "dist"
    if dist.is_dir():
        app.mount("/", StaticFiles(directory=dist, html=True), name="ui")
    return app


app = create_app()
