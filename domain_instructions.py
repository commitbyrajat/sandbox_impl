"""Load stage-specific domain guidance for the caller's authorized workspace."""

import json
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from datasource_mapping import resolve_scope

Stage = Literal['classifier', 'planner', 'executor', 'summarizer']


class TopicScope(BaseModel):
    model_config = ConfigDict(extra='forbid')
    request_pattern: str
    answer_pattern: str

    @field_validator('request_pattern', 'answer_pattern')
    @classmethod
    def valid_pattern(cls, value: str) -> str:
        re.compile(value)
        return value


class DomainProfile(BaseModel):
    model_config = ConfigDict(extra='forbid')
    description: str = Field(min_length=1)
    tone: str = 'Clear, concise and professional; explain results for a business user.'
    glossary: dict[str, str] = Field(default_factory=dict)
    common_instructions: list[str] = Field(default_factory=list)
    classifier_instructions: list[str] = Field(default_factory=list)
    planner_instructions: list[str] = Field(default_factory=list)
    executor_instructions: list[str] = Field(default_factory=list)
    summarizer_instructions: list[str] = Field(default_factory=list)
    action_policies: dict[str, list[str]] = Field(default_factory=dict)
    policy_absence_pattern: str | None = None
    topic_scopes: dict[str, TopicScope] = Field(default_factory=dict)
    summary_additive_fields: list[str] = Field(default_factory=list)
    summary_partition_fields: list[str] = Field(default_factory=list)

    @field_validator('policy_absence_pattern')
    @classmethod
    def valid_absence_pattern(cls, value: str | None) -> str | None:
        if value is not None:
            re.compile(value)
        return value

    @model_validator(mode='after')
    def valid_policies(self):
        for name, rules in self.action_policies.items():
            if not name.strip() or not rules or any(not rule.strip() for rule in rules):
                raise ValueError('Configured action policies need a name and nonempty rules')
        return self

    def for_stage(self, stage: Stage) -> list[str]:
        if stage not in ('classifier', 'planner', 'executor', 'summarizer'):
            raise ValueError('Unknown agent stage')
        return [*self.common_instructions, *getattr(self, f'{stage}_instructions')]


class DomainMapping(BaseModel):
    model_config = ConfigDict(extra='forbid')
    profiles: dict[str, DomainProfile] = Field(min_length=1)


DOMAIN_PROFILES = DomainMapping.model_validate_json(
    (Path(__file__).resolve().parent / 'domain_instructions.json').read_text()
).profiles


def domain_context(workspace_id: str | int, *, tenant_id: str | int, stage: Stage) -> str:
    scope = resolve_scope(tenant_id, workspace_id)
    profile_id = scope.workspace.domain_profile
    if profile_id not in DOMAIN_PROFILES:
        raise ValueError('Workspace references an unknown domain profile')
    profile = DOMAIN_PROFILES[profile_id]
    return json.dumps({
        'tenant_id': scope.tenant_id, 'workspace_id': scope.workspace_id,
        'workspace_name': scope.workspace.name, 'domain_profile': profile_id,
        'description': profile.description, 'glossary': profile.glossary, 'tone': profile.tone,
        'instructions': profile.for_stage(stage),
        'action_policies': profile.action_policies,
        'action_policies_configured': bool(profile.action_policies),
    })


def get_domain_profile(workspace_id: str | int, *, tenant_id: str | int) -> DomainProfile:
    scope = resolve_scope(tenant_id, workspace_id)
    profile_id = scope.workspace.domain_profile
    if profile_id not in DOMAIN_PROFILES:
        raise ValueError('Workspace references an unknown domain profile')
    return DOMAIN_PROFILES[profile_id]
