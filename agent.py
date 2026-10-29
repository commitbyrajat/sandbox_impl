"""CLI and public entry points for the question-analysis pipeline."""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.models import Model

# Public analysis interfaces remain importable; implementations own no CLI state.
from data_access import DataSources
from execution import INSTRUCTIONS, Dependencies, create_agent
from model_config import configured_model
from pipeline import (Pipeline, PipelineModels, PipelineRequest,
                      PipelineServices)
from planning import ExecutionPlan, QuestionIntent
from runtime import PipelineModelTimeout
from summarization import append_clarifications


async def classify_question(
    question: str,
    *,
    tenant_id: str | int,
    workspace_id: str | int,
    model: Model | None = None,
) -> QuestionIntent:
    context = await Pipeline().run(
        PipelineRequest(question, tenant_id, workspace_id, mode="classify"),
        PipelineModels(default=model),
    )
    return context.intent


async def plan_question(
    question: str,
    *,
    tenant_id: str | int,
    workspace_id: str | int,
    model: Model | None = None,
    classifier_model: Model | None = None,
    planner_model: Model | None = None,
) -> tuple[QuestionIntent, ExecutionPlan]:
    context = await Pipeline().run(
        PipelineRequest(question, tenant_id, workspace_id, mode="plan"),
        PipelineModels(
            default=model, classifier=classifier_model, planner=planner_model
        ),
    )
    return context.intent, context.plan


async def ask(
    question: str,
    *,
    tenant_id: str | int,
    workspace_id: str | int,
    model: Model | None = None,
    classifier_model: Model | None = None,
    planner_model: Model | None = None,
    summarizer_model: Model | None = None,
    trace: bool = False,
    plan_dir: Path | None = None,
    services: PipelineServices | None = None,
    max_tokens: int | None = None,
) -> str:
    context = await Pipeline(services).run(
        PipelineRequest(
            question,
            tenant_id,
            workspace_id,
            trace=trace,
            plan_dir=plan_dir,
            max_tokens=max_tokens,
        ),
        PipelineModels(
            default=model,
            classifier=classifier_model,
            planner=planner_model,
            summarizer=summarizer_model,
        ),
    )
    return context.answer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", help="Question within the specified workspace")
    parser.add_argument(
        "--tenant-id", required=True, help="Tenant whose model and workspaces to use"
    )
    parser.add_argument(
        "--workspace-id", required=True, help="Workspace owned by the specified tenant"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        help="Maximum output tokens per model request; overrides LITELLM_MAX_OUTPUT_TOKENS",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Show tenant/model route, intent, plan, SQL and Python",
    )
    parser.add_argument(
        "--plan-dir", type=Path, help="Directory for published plan JSON files"
    )
    preview = parser.add_mutually_exclusive_group()
    preview.add_argument(
        "--classify-only", action="store_true", help="Identify the requested task only"
    )
    preview.add_argument(
        "--plan-only",
        action="store_true",
        help="Publish the plan without opening data sources",
    )
    args = parser.parse_args()
    if args.max_tokens is not None and args.max_tokens < 1:
        parser.error("--max-tokens must be a positive integer")
    mode = (
        "classify"
        if args.classify_only
        else "publish_plan" if args.plan_only else "answer"
    )
    try:
        context = asyncio.run(
            Pipeline().run(
                PipelineRequest(
                    args.question,
                    args.tenant_id,
                    args.workspace_id,
                    mode=mode,
                    trace=args.trace,
                    plan_dir=args.plan_dir,
                    max_tokens=args.max_tokens,
                )
            )
        )
        if args.classify_only:
            print(context.intent.model_dump_json(indent=2))
        elif args.plan_only:
            print(
                json.dumps(
                    {
                        "intent": context.intent.model_dump(),
                        "plan": context.plan.model_dump(),
                    },
                    indent=2,
                )
            )
        else:
            print(context.answer)
    except PipelineModelTimeout as exc:
        print(f"Agent failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
    except Exception as exc:
        detail = "Check the tenant/workspace IDs, mapped LiteLLM model and database connection."
        if isinstance(exc, (ValueError, UnexpectedModelBehavior)):
            detail = str(exc)
            if isinstance(exc, UnexpectedModelBehavior):
                detail += " Use --trace to see stage and tool correction failures."
                if "token limit" in str(exc).lower():
                    detail += " Set --max-tokens 8192 (or LITELLM_MAX_OUTPUT_TOKENS in .env) to raise the output budget."
        if "Invalid model name" in str(exc):
            detail = "LiteLLM does not expose the tenant-mapped model; check its /v1/models endpoint."
        print(f"Agent failed ({type(exc).__name__}). {detail}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
