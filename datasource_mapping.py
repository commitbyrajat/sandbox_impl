"""Validated tenant -> model -> workspaces -> sources registry."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ROOT = Path(__file__).resolve().parent
SourceId = Literal['credit_cards', 'smartwealth']
TenantId = str
WorkspaceId = str


class SourceDefinition(BaseModel):
    model_config = ConfigDict(extra='forbid')
    kind: Literal['parquet', 'postgres']
    location: str
    description: str
    tables: dict[str, str]
    relationships: list[str]
    notes: list[str]


class WorkspaceDefinition(BaseModel):
    model_config = ConfigDict(extra='forbid')
    name: str
    domain_profile: str = Field(min_length=1)
    sources: dict[SourceId, SourceDefinition] = Field(min_length=1)


class ModelDefinition(BaseModel):
    model_config = ConfigDict(extra='forbid')
    model_name: str = Field(min_length=1)
    base_url_env: str = 'LITELLM_BASE_URL'
    api_key_env: str = 'LITELLM_MASTER_KEY'


class TenantDefinition(BaseModel):
    model_config = ConfigDict(extra='forbid')
    name: str
    model_id: str
    workspaces: dict[WorkspaceId, WorkspaceDefinition] = Field(min_length=1)


class TenantMapping(BaseModel):
    model_config = ConfigDict(extra='forbid')
    models: dict[str, ModelDefinition] = Field(min_length=1)
    tenants: dict[TenantId, TenantDefinition] = Field(min_length=1)

    @model_validator(mode='after')
    def validate_references(self):
        for tenant_id, tenant in self.tenants.items():
            if not tenant_id.strip() or tenant.model_id not in self.models:
                raise ValueError('Each tenant must have a nonempty ID and a configured model')
            if any(not key.strip() for key in tenant.workspaces):
                raise ValueError('Workspace IDs must be nonempty')
        return self


MAPPING = TenantMapping.model_validate_json((ROOT / 'datasource.json').read_text())
TENANTS = MAPPING.tenants
MODELS = MAPPING.models
# The standalone raw SQL demo defaults to the original sample sources. Agent
# loaders receive the exact tenant/workspace source definitions instead.
SOURCES = {source_id: source for tenant in TENANTS.values()
           for workspace in tenant.workspaces.values()
           for source_id, source in workspace.sources.items()}


@dataclass(frozen=True)
class RequestScope:
    tenant_id: TenantId
    workspace_id: WorkspaceId
    tenant: TenantDefinition
    workspace: WorkspaceDefinition
    model: ModelDefinition


def resolve_scope(tenant_id: str | int, workspace_id: str | int) -> RequestScope:
    """Reject unknown IDs/ownership before any model call or data connection."""
    tenant_key, workspace_key = str(tenant_id), str(workspace_id)
    if tenant_key not in TENANTS:
        raise ValueError('Unknown tenant ID')
    tenant = TENANTS[tenant_key]
    if workspace_key not in tenant.workspaces:
        raise ValueError('Workspace does not belong to the specified tenant')
    return RequestScope(tenant_key, workspace_key, tenant, tenant.workspaces[workspace_key],
                        MODELS[tenant.model_id])


def get_workspace(workspace_id: str | int, *, tenant_id: str | int) -> tuple[WorkspaceId, WorkspaceDefinition]:
    scope = resolve_scope(tenant_id, workspace_id)
    return scope.workspace_id, scope.workspace


def tables_for(
    source_ids: list[SourceId] | tuple[SourceId, ...],
    *, definitions: dict[SourceId, SourceDefinition] | None = None,
) -> tuple[str, ...]:
    sources = SOURCES if definitions is None else definitions
    return tuple(name for source_id in source_ids for name in sources[source_id].tables)


def routing_catalog(workspace_id: str | int, *, tenant_id: str | int) -> str:
    scope = resolve_scope(tenant_id, workspace_id)
    return json.dumps({'tenant_id': scope.tenant_id, 'workspace_id': scope.workspace_id,
                       **scope.workspace.model_dump()})


def classification_context(workspace_id: str | int, *, tenant_id: str | int) -> str:
    """Domain vocabulary only: no source names, schemas, locations or credentials."""
    from domain_instructions import domain_context

    return domain_context(workspace_id, tenant_id=tenant_id, stage='classifier')
