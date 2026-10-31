"""Typed graph orchestration with injectable stages and a compatible custom chain."""

import asyncio
import json
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from time import monotonic
from typing import Any, Callable, Literal

from pydantic_ai.exceptions import ModelRequestTimeout, ModelTimeoutError
from pydantic_ai.models import Model

from data_access import DataSources, load_sources
from datasource_mapping import RequestScope, classification_context, resolve_scope
from domain_instructions import domain_context
from execution import ExecutionResult, execute_plan
from model_config import configured_model
from plan_publication import publish_plan
from planning import ExecutionPlan, QuestionIntent, build_plan, classify_intent
from runtime import (
    MAX_TOKENS_OVERRIDE,
    TRACE_ENABLED,
    PipelineModelTimeout,
    summarization_enabled,
    trace,
)
from summarization import append_clarifications, summarize_result
from calculation_contracts import validate_execution_result


@dataclass(frozen=True)
class PipelineRequest:
    question: str
    tenant_id: str | int
    workspace_id: str | int
    mode: Literal["answer", "classify", "plan", "publish_plan"] = "answer"
    trace: bool = False
    plan_dir: Path | None = None
    max_tokens: int | None = None


@dataclass(frozen=True)
class PipelineModels:
    default: Model | None = None
    classifier: Model | None = None
    planner: Model | None = None
    summarizer: Model | None = None


@dataclass
class PipelineContext:
    request: PipelineRequest
    models: PipelineModels
    scope: RequestScope | None = None
    model: Model | None = None
    intent: QuestionIntent | None = None
    plan: ExecutionPlan | None = None
    publication: Path | None = None
    sources: DataSources | None = None
    execution: ExecutionResult | None = None
    answer: str | None = None
    summarization_enabled: bool = True
    completed_stages: list[str] = field(default_factory=list)
    skipped_stages: list[str] = field(default_factory=list)
    stage_timings: dict[str, float] = field(default_factory=dict)
    stopped: bool = False


@dataclass
class PipelineServices:
    """Replace adapters without changing handlers or importing the CLI module."""

    resolve: Callable[..., Any] = field(default_factory=lambda: resolve_scope)
    model_factory: Callable[..., Any] = field(default_factory=lambda: configured_model)
    domain: Callable[..., Any] = field(default_factory=lambda: domain_context)
    classifier_context: Callable[..., Any] = field(
        default_factory=lambda: classification_context
    )
    classify: Callable[..., Any] = field(default_factory=lambda: classify_intent)
    plan: Callable[..., Any] = field(default_factory=lambda: build_plan)
    publish: Callable[..., Any] = field(default_factory=lambda: publish_plan)
    load: Callable[..., Any] = field(default_factory=lambda: load_sources)
    execute: Callable[..., Any] = field(default_factory=lambda: execute_plan)
    validate: Callable[..., Any] = field(
        default_factory=lambda: validate_execution_result
    )
    summarize: Callable[..., Any] = field(default_factory=lambda: summarize_result)


def require(value: Any, name: str) -> Any:
    if value is None:
        raise RuntimeError(f"Pipeline prerequisite missing: {name}")
    return value


class PipelineStage(ABC):
    """Each handler does one job and forwards the request to its successor."""

    name: str

    def __init__(self, services: PipelineServices):
        self.services = services
        self.next_stage: PipelineStage | None = None

    def set_next(self, successor: "PipelineStage") -> "PipelineStage":
        self.next_stage = successor
        return successor

    async def handle(
        self, context: PipelineContext, *, forward: bool = True
    ) -> PipelineContext:
        token = TRACE_ENABLED.set(context.request.trace)
        started = monotonic()
        print(f"STAGE {self.name}: start", file=sys.stderr, flush=True)
        status = "failed"
        try:
            await self.process(context)
        except (ModelTimeoutError, ModelRequestTimeout) as exc:
            status = "timeout"
            name = getattr(exc, "model_name", None) or getattr(
                context.model, "model_name", None
            )
            if not name and context.scope is not None:
                name = context.scope.model.model_name
            elapsed = monotonic() - started
            trace(f"STAGE {self.name}: model timeout after {elapsed:.1f}s")
            raise PipelineModelTimeout(self.name, name or "unknown", elapsed) from exc
        except asyncio.CancelledError:
            status = "cancelled"
            raise
        except Exception as exc:
            trace(
                f"STAGE {self.name}: failed ({type(exc).__name__}) after {monotonic()-started:.1f}s"
            )
            raise
        else:
            status = "skipped" if self.name in context.skipped_stages else "completed"
        finally:
            context.stage_timings[self.name] = monotonic() - started
            print(
                f"STAGE {self.name}: end ({status}, {context.stage_timings[self.name]:.1f}s)",
                file=sys.stderr,
                flush=True,
            )
            TRACE_ENABLED.reset(token)
        if status != "skipped":
            context.completed_stages.append(self.name)
        if forward and self.next_stage is not None and not context.stopped:
            return await self.next_stage.handle(context)
        return context

    @abstractmethod
    async def process(self, context: PipelineContext) -> None: ...


class ResolveStage(PipelineStage):
    name = "resolve"

    async def process(self, context: PipelineContext) -> None:
        request = context.request
        if request.mode not in ("answer", "classify", "plan", "publish_plan"):
            raise ValueError("Unknown pipeline operation")
        context.scope = self.services.resolve(request.tenant_id, request.workspace_id)
        # Validate the domain reference before constructing any model client.
        self.services.domain(
            request.workspace_id, tenant_id=request.tenant_id, stage="classifier"
        )
        context.model = context.models.default
        if context.model is None:
            context.model = self.services.model_factory(
                request.tenant_id, request.workspace_id
            )


class ClassifyStage(PipelineStage):
    name = "classify"

    async def process(self, context: PipelineContext) -> None:
        scope = require(context.scope, "scope")
        context.intent = await self.services.classify(
            context.request.question,
            model=context.models.classifier or require(context.model, "model"),
            workspace_context=self.services.classifier_context(
                scope.workspace_id, tenant_id=scope.tenant_id
            ),
        )
        context.stopped = context.request.mode == "classify"


class PlanStage(PipelineStage):
    name = "plan"

    async def process(self, context: PipelineContext) -> None:
        scope = require(context.scope, "scope")
        context.plan = await self.services.plan(
            context.request.question,
            require(context.intent, "intent"),
            tenant_id=scope.tenant_id,
            workspace_id=scope.workspace_id,
            model=context.models.planner or require(context.model, "model"),
        )
        if (context.plan.tenant_id, context.plan.workspace_id) != (
            scope.tenant_id,
            scope.workspace_id,
        ):
            raise ValueError(
                "Pipeline plan does not match the request tenant/workspace"
            )
        context.stopped = context.request.mode == "plan"


class PublishStage(PipelineStage):
    name = "publish"

    async def process(self, context: PipelineContext) -> None:
        context.publication = self.services.publish(
            context.request.question,
            require(context.intent, "intent"),
            require(context.plan, "plan"),
            directory=context.request.plan_dir,
        )
        if context.request.trace:
            scope = require(context.scope, "scope")
            print(
                "ROUTE: "
                + json.dumps(
                    {
                        "tenant_id": scope.tenant_id,
                        "workspace_id": scope.workspace_id,
                        "model_name": scope.model.model_name,
                    }
                ),
                file=sys.stderr,
            )
            print("INTENT: " + context.intent.model_dump_json(), file=sys.stderr)
            print("PLAN: " + context.plan.model_dump_json(), file=sys.stderr)
        context.stopped = context.request.mode == "publish_plan"


class LoadSourcesStage(PipelineStage):
    name = "load_sources"

    async def process(self, context: PipelineContext) -> None:
        if require(context.plan, "plan").execution_mode != "none":
            context.sources = await asyncio.to_thread(
                self.services.load,
                require(context.scope, "scope"),
                context.plan,
            )


class ExecuteStage(PipelineStage):
    name = "execute"

    async def process(self, context: PipelineContext) -> None:
        plan = require(context.plan, "plan")
        if plan.execution_mode == "none":
            return
        context.execution = await self.services.execute(
            context.request.question,
            plan,
            require(context.sources, "sources"),
            require(context.model, "model"),
        )
        if context.request.trace:
            print(
                "COMPLETED: " + json.dumps(context.execution.completed_steps),
                file=sys.stderr,
            )
            for query in context.execution.queries:
                print("SQL: " + query, file=sys.stderr)
            for run in context.execution.python_runs:
                label = (
                    "PYTHON REUSED: " if run.get("executed") is False else "PYTHON: "
                )
                print(label + json.dumps(run, default=str), file=sys.stderr)


class ValidateResultsStage(PipelineStage):
    name = "validate_results"

    async def process(self, context: PipelineContext) -> None:
        self.services.validate(require(context.plan, "plan"), context.execution)


class SummarizeStage(PipelineStage):
    name = "summarize"

    async def process(self, context: PipelineContext) -> None:
        plan = require(context.plan, "plan")
        if not context.summarization_enabled:
            context.skipped_stages.append(self.name)
            answer = (
                context.execution.draft
                if context.execution is not None
                else plan.response or ""
            )
            context.answer = append_clarifications(answer, plan)
            return
        answer = await self.services.summarize(
            context.request.question,
            plan,
            context.execution,
            context.models.summarizer or require(context.model, "model"),
        )
        context.answer = append_clarifications(answer, plan)


class Pipeline:
    def __init__(
        self,
        services: PipelineServices | None = None,
        *,
        stages: list[PipelineStage] | None = None,
    ):
        self.services = services or PipelineServices()
        self.use_graph = stages is None
        self.stages = (
            stages
            if stages is not None
            else [
                stage(self.services)
                for stage in (
                    ResolveStage,
                    ClassifyStage,
                    PlanStage,
                    PublishStage,
                    LoadSourcesStage,
                    ExecuteStage,
                    ValidateResultsStage,
                    SummarizeStage,
                )
            ]
        )
        if not self.stages:
            raise ValueError("A pipeline needs at least one stage")
        for current, successor in zip(self.stages, self.stages[1:]):
            current.set_next(successor)
        self.stages[-1].next_stage = None

    async def run(
        self, request: PipelineRequest, models: PipelineModels | None = None
    ) -> PipelineContext:
        if request.max_tokens is not None and (
            isinstance(request.max_tokens, bool)
            or not isinstance(request.max_tokens, int)
            or request.max_tokens < 1
        ):
            raise ValueError("max_tokens must be a positive integer")
        token = MAX_TOKENS_OVERRIDE.set(request.max_tokens)
        try:
            context = PipelineContext(
                request,
                models or PipelineModels(),
                summarization_enabled=summarization_enabled(),
            )
            if self.use_graph:
                from pipeline_graph import LIFECYCLE_GRAPH

                return await LIFECYCLE_GRAPH.run(state=context, deps=self)
            return await self.stages[0].handle(context)
        finally:
            MAX_TOKENS_OVERRIDE.reset(token)
