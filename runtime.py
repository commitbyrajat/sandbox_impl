"""Shared model budgets and request-local progress reporting."""

import math
import os
import re
import sys
from contextvars import ContextVar
from dataclasses import dataclass

from dotenv import load_dotenv
from pydantic_ai.exceptions import ModelAPIError

from datasource_mapping import MODELS
from main import ROOT

TRACE_ENABLED: ContextVar[bool] = ContextVar("pipeline_trace_enabled", default=False)
MAX_TOKENS_OVERRIDE: ContextVar[int | None] = ContextVar(
    "pipeline_max_tokens", default=None
)


@dataclass(frozen=True)
class ModelBudget:
    timeout_seconds: float = 300
    request_deadline_seconds: float = 330
    max_retries: int = 0
    max_output_tokens: int = 4096


def model_budget() -> ModelBudget:
    load_dotenv(ROOT / ".env")

    def positive(name: str, default: float) -> float:
        try:
            value = float(os.getenv(name, str(default)))
        except ValueError:
            raise ValueError(f"{name} must be a positive number") from None
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a positive number")
        return value

    timeout = positive("LITELLM_TIMEOUT_SECONDS", 300)
    deadline = positive("LITELLM_REQUEST_DEADLINE_SECONDS", max(330, timeout + 30))
    if deadline < timeout:
        raise ValueError(
            "LITELLM_REQUEST_DEADLINE_SECONDS must be >= LITELLM_TIMEOUT_SECONDS"
        )
    try:
        retries = int(os.getenv("LITELLM_MAX_RETRIES", "0"))
        tokens = int(os.getenv("LITELLM_MAX_OUTPUT_TOKENS", "4096"))
    except ValueError:
        raise ValueError(
            "LITELLM_MAX_RETRIES and LITELLM_MAX_OUTPUT_TOKENS must be integers"
        ) from None
    if retries < 0 or tokens < 1:
        raise ValueError(
            "LITELLM_MAX_RETRIES must be nonnegative and LITELLM_MAX_OUTPUT_TOKENS positive"
        )
    return ModelBudget(timeout, deadline, retries, MAX_TOKENS_OVERRIDE.get() or tokens)


def model_settings() -> dict:
    budget = model_budget()
    return {
        "temperature": 0,
        "timeout": budget.timeout_seconds,
        "request_timeout": budget.request_deadline_seconds,
        "max_retries": budget.max_retries,
        "max_tokens": budget.max_output_tokens,
    }


def tool_retry_limit() -> int:
    """Read the executor's per-tool correction budget, separately from HTTP retries."""
    load_dotenv(ROOT / ".env")
    try:
        value = int(os.getenv("AGENT_TOOL_RETRIES", "3"))
    except ValueError:
        raise ValueError("AGENT_TOOL_RETRIES must be a nonnegative integer") from None
    if value < 0:
        raise ValueError("AGENT_TOOL_RETRIES must be a nonnegative integer")
    return value


def model_error_detail(error: ModelAPIError) -> str:
    """Expose the provider's useful message without dumping headers or credentials."""
    body = error.body
    if isinstance(body, dict):
        body = body.get("error", body)
    if isinstance(body, dict):
        body = body.get("message")
    detail = body if isinstance(body, str) else error.message
    credential_names = {model.api_key_env for model in MODELS.values()}
    for name, value in os.environ.items():
        if (
            name in credential_names
            or name.endswith(
                ("API_KEY", "MASTER_KEY", "ACCESS_TOKEN", "PASSWORD", "SECRET")
            )
        ) and value:
            detail = detail.replace(value, "[redacted]")
    detail = re.sub(r"sk-[A-Za-z0-9_-]+", "[redacted]", detail)
    detail = re.sub(r"(?i)Bearer\s+\S+", "Bearer [redacted]", detail)
    status = getattr(error, "status_code", None)
    label = f"HTTP {status}" if status is not None else type(error).__name__
    return f"Model {error.model_name} ({label}): {detail[:1600]}"


def trace(message: str) -> None:
    if TRACE_ENABLED.get():
        print(message, file=sys.stderr, flush=True)


def result_preview(result: dict, limit: int = 12) -> dict:
    """Bound model context only; the host retains every stored row for Python."""
    rows = result.get("rows", [])
    if len(rows) <= limit:
        return result
    return {
        **result,
        "rows": rows[:limit],
        "stored_row_count": len(rows),
        "rows_preview_only": True,
        "complete_rows_available_for_python": not result.get("truncated", False),
    }


class PipelineModelTimeout(RuntimeError):
    def __init__(self, stage: str, model_name: str, elapsed: float):
        self.stage, self.model_name, self.elapsed = stage, model_name, elapsed
        super().__init__(
            f"Model {model_name} timed out during {stage} after {elapsed:.1f}s. "
            "Check the LiteLLM/Ollama response time and model queue. "
            "Model budgets: LITELLM_TIMEOUT_SECONDS and LITELLM_REQUEST_DEADLINE_SECONDS."
        )
