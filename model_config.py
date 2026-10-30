"""Construct the tenant-selected model without owning orchestration."""

import os

from dotenv import load_dotenv
from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.profiles.openai import OpenAIModelProfile
from pydantic_ai.providers.openai import OpenAIProvider

from datasource_mapping import resolve_scope
from main import ROOT


def configured_model(tenant_id: str | int, workspace_id: str | int) -> Model:
    scope = resolve_scope(tenant_id, workspace_id)
    load_dotenv(ROOT / ".env")
    key = os.getenv(scope.model.api_key_env)
    if not key:
        raise ValueError(f"Set {scope.model.api_key_env} in .env or your environment")
    profile = OpenAIModelProfile(
        default_structured_output_mode=scope.model.structured_output_mode
    )
    if scope.model.structured_output_mode == "native":
        profile.update(
            supports_json_schema_output=True,
            native_output_requires_schema_in_instructions=True,
        )
    return OpenAIChatModel(
        scope.model.model_name,
        profile=profile,
        provider=OpenAIProvider(
            base_url=os.getenv(scope.model.base_url_env, "http://localhost:4000/v1"),
            api_key=key,
        ),
    )
