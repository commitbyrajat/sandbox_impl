"""Publish reviewable execution plans locally before data access begins."""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote
from uuid import uuid4

from datasource_mapping import resolve_scope
from planning import ExecutionPlan, QuestionIntent

DEFAULT_PLAN_DIR = Path(__file__).resolve().parent / "artifacts" / "plans"


def publish_plan(
    question: str,
    intent: QuestionIntent,
    plan: ExecutionPlan,
    *,
    directory: Path | None = None,
) -> Path:
    """Save the plan and print its steps; failure prevents execution from starting."""
    scope = resolve_scope(plan.tenant_id, plan.workspace_id)
    publication_id = uuid4().hex
    root = DEFAULT_PLAN_DIR if directory is None else directory
    folder = (
        root
        / ("tenant-" + quote(scope.tenant_id, safe=""))
        / ("workspace-" + quote(scope.workspace_id, safe=""))
    )
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{publication_id}.json"
    record = {
        "version": 1,
        "publication_id": publication_id,
        "published_at": datetime.now(timezone.utc).isoformat(),
        "status": "planned",
        "tenant_id": scope.tenant_id,
        "workspace_id": scope.workspace_id,
        "model_id": scope.tenant.model_id,
        "model_name": scope.model.model_name,
        "question": question,
        "intent": intent.model_dump(mode="json"),
        "plan": plan.model_dump(mode="json"),
        "execution_summary": {
            "mode": plan.execution_mode,
            "python_required": plan.execution_mode == "sql_python",
            "reason": plan.execution_reason,
            "python_purpose": plan.python_reason,
        },
    }
    with os.fdopen(
        os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w"
    ) as handle:
        json.dump(record, handle, indent=2)
        handle.write("\n")

    print(f"Published plan: {path}", file=sys.stderr)
    print(
        f"Tenant {scope.tenant_id} / workspace {scope.workspace_id} / "
        f"{scope.model.model_name} / {plan.execution_mode}",
        file=sys.stderr,
    )
    approaches = {
        "sql": "Query the data (SQL)",
        "sql_python": "Query the data, then run an additional calculation (SQL + Python)",
        "none": "Clarification or unavailable information; no data execution",
    }
    print("Approach: " + approaches[plan.execution_mode], file=sys.stderr)
    if plan.sql_recipe is not None:
        print(
            "SQL recipe: "
            + plan.sql_recipe.recipe_id
            + " / parameters: "
            + json.dumps(plan.sql_recipe.parameters, default=str),
            file=sys.stderr,
        )
    print(
        "Python required: " + ("yes" if plan.execution_mode == "sql_python" else "no"),
        file=sys.stderr,
    )
    if plan.execution_reason:
        print("Why: " + plan.execution_reason, file=sys.stderr)
    if plan.execution_mode == "sql_python":
        print("Additional calculation: " + plan.python_reason, file=sys.stderr)
    for step in plan.steps:
        print(f"  {step.step_id} [{step.action}] {step.instruction}", file=sys.stderr)
        print(f"    Expected: {step.expected_output}", file=sys.stderr)
        if step.output_contract is not None:
            print(
                "    Required SQL columns: "
                + ", ".join(step.output_contract.required_columns),
                file=sys.stderr,
            )
            if step.output_contract.grain:
                print(
                    "    Observation grain: " + ", ".join(step.output_contract.grain),
                    file=sys.stderr,
                )
        if step.input_steps:
            print("    SQL inputs: " + ", ".join(step.input_steps), file=sys.stderr)
    for check in plan.validation_checks:
        print(f"  Check: {check}", file=sys.stderr)
    for objective in plan.deferred_objectives:
        print(f"  Deferred: {objective.objective}", file=sys.stderr)
    if plan.execution_mode == "none":
        print("  No data execution is planned.", file=sys.stderr)
    sys.stderr.flush()
    return path
