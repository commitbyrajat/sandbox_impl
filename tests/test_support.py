"""Deterministic presentation adapter for existing analysis-focused tests."""

import json
from functools import wraps

from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel

from agent import ask as pipeline_ask


def summary_echo(messages, info):
    payload = json.loads(messages[-1].parts[0].content)
    return ModelResponse(parts=[TextPart(payload["execution_draft"])])


@wraps(pipeline_ask)
async def ask(*args, **kwargs):
    kwargs.setdefault("summarizer_model", FunctionModel(summary_echo))
    return await pipeline_ask(*args, **kwargs)
