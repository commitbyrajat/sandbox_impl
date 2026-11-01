"""Request-local lifecycle events shared by the CLI pipeline and web transport."""

from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Callable

PROGRESS_SINK: ContextVar[Callable[[dict], None] | None] = ContextVar(
    "progress_sink", default=None
)
CURRENT_STAGE: ContextVar[str | None] = ContextVar("current_stage", default=None)

STAGES = [
    {
        "id": "resolve",
        "name": "Connect workspace",
        "description": "Resolve the tenant, workspace and model.",
    },
    {
        "id": "classify",
        "name": "Understand question",
        "description": "Identify the objective and constraints.",
    },
    {
        "id": "plan",
        "name": "Build analysis plan",
        "description": "Choose metrics, sources and calculations.",
    },
    {
        "id": "publish",
        "name": "Record plan",
        "description": "Save the execution plan for review.",
    },
    {
        "id": "load_sources",
        "name": "Connect data",
        "description": "Load the data sources for this workspace.",
    },
    {
        "id": "execute",
        "name": "Run analysis",
        "description": "Execute SQL stages and sandboxed calculations.",
    },
    {
        "id": "validate_results",
        "name": "Validate results",
        "description": "Check the evidence and output contracts.",
    },
    {
        "id": "summarize",
        "name": "Prepare answer",
        "description": "Write an answer grounded in the results.",
    },
]


def emit_progress(kind: str, **payload) -> None:
    sink = PROGRESS_SINK.get()
    if sink is not None:
        event = {
            "kind": kind,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **payload,
        }
        sink(event)


def stage_outcome(name, context) -> dict:
    """Expose evidence without provider credentials, connection strings or paths."""
    from runtime import calculation_preview

    if name == "resolve" and context.scope:
        scope = context.scope
        return {
            "tenant": scope.tenant.name,
            "workspace": scope.workspace.name,
            "model": scope.model.model_name,
            "sources": list(scope.workspace.sources),
        }
    if name == "classify" and context.intent:
        return context.intent.model_dump(mode="json")
    if name == "plan" and context.plan:
        return context.plan.model_dump(mode="json")
    if name == "publish" and context.publication:
        return {
            "publication": context.publication.name,
            "summary": "Execution plan recorded.",
        }
    if name == "load_sources" and context.scope:
        return {
            "connected": context.sources is not None,
            "tables": [
                table
                for source in context.scope.workspace.sources.values()
                for table in source.tables
            ],
        }
    if name == "execute" and context.execution:
        execution = context.execution
        return {
            "completed_steps": execution.completed_steps,
            "query_count": len(execution.queries),
            "queries": execution.queries,
            "datasets": [
                {
                    "result_id": key,
                    "row_count": len(value.get("rows", [])),
                    "columns": value.get("columns", []),
                    "column_types": value.get("column_types", []),
                    "preview": value.get("rows", [])[:5],
                    "truncated": value.get("truncated", False),
                    "sql_stages": [
                        {k: v for k, v in stage.items() if k != "statement"}
                        for stage in value.get("sql_stages", [])
                    ],
                }
                for key, value in execution.sql_results.items()
            ],
            "calculations": [
                {
                    "step_id": run.get("step_id"),
                    "executed": run.get("executed"),
                    "output": calculation_preview(run.get("output")),
                }
                for run in execution.python_runs
            ],
        }
    if name == "validate_results" and context.execution:
        return {
            "summary": "Result contracts and evidence checks passed.",
            "checks": [
                {
                    "step_id": r.get("step_id"),
                    "boundary": r.get("boundary"),
                    "checks": r.get("checks", []),
                }
                for r in context.execution.validation_receipts
            ],
        }
    if name == "summarize":
        return {"summary": "Answer prepared from validated evidence."}
    return {}
