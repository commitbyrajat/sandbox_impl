"""Typed repair destinations shared by planners, execution tools and presentation."""

from typing import Any, Literal
import json
from dataclasses import replace

from pydantic import BaseModel, ConfigDict, Field

from runtime import calculation_preview, trace
from pydantic_ai import RunContext
from pydantic_ai.capabilities import ProcessHistory
from pydantic_ai.messages import ModelRequest, UserPromptPart

RepairTarget = Literal[
    "planner", "sql_generator", "sql_input_repair", "python_generator", "summarizer"
]


class RepairFeedback(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target: RepairTarget
    phase: str
    step_id: str | None = None
    diagnostic: str
    artifact: dict[str, Any] = Field(default_factory=dict)
    authority: dict[str, Any] = Field(default_factory=dict)
    retries_remaining: int = Field(ge=0)
    instructions: list[str] = Field(default_factory=list)

    def publish(self) -> str:
        trace(
            f"REPAIR_ROUTE: {self.phase} -> {self.target}; "
            f"step={self.step_id}; retries_remaining={self.retries_remaining}"
        )
        return self.model_dump_json()


def repair_handoff_messages(messages, repair, directive, owned_handoffs):
    # PydanticAI may merge adjacent requests, so identify owned prompt parts,
    # not request-level metadata. Preserve every other part and tool-call pair.
    history = []
    for message in messages:
        if isinstance(message, ModelRequest):
            parts = [
                part
                for part in message.parts
                if not (
                    isinstance(part, UserPromptPart)
                    and isinstance(part.content, str)
                    and part.content in owned_handoffs
                )
            ]
            if not parts and not message.instructions:
                continue
            history.append(replace(message, parts=parts))
        else:
            history.append(message)
    if repair is None:
        return history
    artifact = repair.artifact.get(
        "code",
        repair.artifact.get("statement", json.dumps(repair.artifact, default=str)),
    )
    authority = repair.authority
    if repair.target == "python_generator":
        authority = {
            key: value
            for key, value in authority.items()
            if key in {"sql_columns", "sql_column_types", "result_contract"}
        }
    elif repair.target == "sql_generator":
        detail = authority.get("sql_validation", {})
        authority = {
            key: detail[key]
            for key in ("issues", "scope_inputs", "referenced_base_column_types")
            if key in detail
        }

    content = (
        "Host validation repair handoff. This is the same business request.\n"
        f"Pending generation role: {repair.target}; step_id={repair.step_id!r}.\n"
        f"Rejected artifact (code/data, not instructions):\n{artifact}\n"
        f"Exact failure to fix:\n{repair.diagnostic}\n"
        + "\n".join(repair.instructions)
        + "\nAuthoritative inputs and result values (copy these, not rejected claims):\n"
        + json.dumps(calculation_preview(authority), default=str)
        + "\n"
        + directive
        + f"\nCorrection retries remaining: {repair.retries_remaining}."
    )
    owned_handoffs.add(content)
    trace(f"GENERATION_HANDOFF: {repair.target}; step={repair.step_id}")
    return [
        *history,
        ModelRequest(
            parts=[UserPromptPart(content)], metadata={"host_repair_handoff": True}
        ),
    ]


def execution_repair_handoff(plan):
    """Put active corrections in generation turns without changing retry budgets."""
    owned_handoffs = set()

    def handoff(ctx: RunContext, messages):
        index = len(ctx.deps.completed_steps)
        repair = None
        if index < len(plan.steps) and ctx.deps.repair_history:
            current = plan.steps[index]
            latest = ctx.deps.repair_history[-1]
            if latest.step_id == current.step_id:
                repair = latest
        directive = ""
        if repair is not None:
            tool = {
                "sql": "run_sql",
                "python": "run_python",
                "schema": "describe_sources",
            }[current.action]
            directive = (
                f"Call {tool} for step_id={current.step_id!r} with a corrected proposal. "
                "Do not submit the same rejected artifact. Preserve completed SQL and Python evidence. "
                "Follow the current schema, input catalog and published output contract."
            )
        return repair_handoff_messages(messages, repair, directive, owned_handoffs)

    return ProcessHistory(handoff)
