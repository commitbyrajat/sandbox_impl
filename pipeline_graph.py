"""Typed lifecycle graph. Stage adapters own their work and logging, not traversal."""

from pydantic_graph import BaseNode, End, GraphBuilder, GraphRunContext, StepContext

from pipeline import Pipeline, PipelineContext


async def run_stage(ctx, name):
    stage = next(stage for stage in ctx.deps.stages if stage.name == name)
    await stage.handle(ctx.state, forward=False)


class Resolve(BaseNode[PipelineContext, Pipeline]):
    async def run(self, ctx: GraphRunContext[PipelineContext, Pipeline]) -> "Classify":
        await run_stage(ctx, "resolve")
        return Classify()


class Classify(BaseNode[PipelineContext, Pipeline, PipelineContext]):
    async def run(
        self, ctx: GraphRunContext[PipelineContext, Pipeline]
    ) -> "Plan | End[PipelineContext]":
        await run_stage(ctx, "classify")
        return End(ctx.state) if ctx.state.stopped else Plan()


class Plan(BaseNode[PipelineContext, Pipeline, PipelineContext]):
    async def run(
        self, ctx: GraphRunContext[PipelineContext, Pipeline]
    ) -> "Publish | End[PipelineContext]":
        await run_stage(ctx, "plan")
        return End(ctx.state) if ctx.state.stopped else Publish()


class Publish(BaseNode[PipelineContext, Pipeline, PipelineContext]):
    async def run(
        self, ctx: GraphRunContext[PipelineContext, Pipeline]
    ) -> "LoadSources | End[PipelineContext]":
        await run_stage(ctx, "publish")
        return End(ctx.state) if ctx.state.stopped else LoadSources()


class LoadSources(BaseNode[PipelineContext, Pipeline]):
    async def run(self, ctx: GraphRunContext[PipelineContext, Pipeline]) -> "Execute":
        await run_stage(ctx, "load_sources")
        return Execute()


class Execute(BaseNode[PipelineContext, Pipeline]):
    async def run(
        self, ctx: GraphRunContext[PipelineContext, Pipeline]
    ) -> "ValidateResults":
        await run_stage(ctx, "execute")
        return ValidateResults()


class ValidateResults(BaseNode[PipelineContext, Pipeline]):
    async def run(self, ctx: GraphRunContext[PipelineContext, Pipeline]) -> "Summarize":
        await run_stage(ctx, "validate_results")
        return Summarize()


class Summarize(BaseNode[PipelineContext, Pipeline, PipelineContext]):
    async def run(
        self, ctx: GraphRunContext[PipelineContext, Pipeline]
    ) -> End[PipelineContext]:
        await run_stage(ctx, "summarize")
        return End(ctx.state)


def build_lifecycle_graph():
    builder = GraphBuilder(
        state_type=PipelineContext, deps_type=Pipeline, output_type=PipelineContext
    )

    @builder.step
    async def start(ctx: StepContext[PipelineContext, Pipeline, None]) -> Resolve:
        return Resolve()

    builder.add(builder.edge_from(builder.start_node).to(start))
    for node in (
        Resolve,
        Classify,
        Plan,
        Publish,
        LoadSources,
        Execute,
        ValidateResults,
        Summarize,
    ):
        builder.add(builder.node(node))
    return builder.build()


LIFECYCLE_GRAPH = build_lifecycle_graph()
