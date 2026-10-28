"""Construct the tenant-selected model without owning orchestration."""

import os
from dotenv import load_dotenv
from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from datasource_mapping import resolve_scope
from main import ROOT


def configured_model(tenant_id: str | int, workspace_id: str | int) -> Model:
    scope = resolve_scope(tenant_id, workspace_id)
    load_dotenv(ROOT / '.env')
    key = os.getenv(scope.model.api_key_env)
    if not key:
        raise ValueError(f'Set {scope.model.api_key_env} in .env or your environment')
    return OpenAIChatModel(
        scope.model.model_name,
        provider=OpenAIProvider(
            base_url=os.getenv(scope.model.base_url_env, 'http://localhost:4000/v1'), api_key=key,
        ),
    )

