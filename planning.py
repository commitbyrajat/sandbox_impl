"""Separate intent classification from workspace-bound execution planning."""

import json
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_ai import Agent, ModelRetry, UsageLimits
from pydantic_ai.models import Model
from runtime import model_settings

from datasource_mapping import SourceId, TenantId, WorkspaceId, get_workspace, routing_catalog
from domain_instructions import domain_context


class Objective(BaseModel):
    model_config = ConfigDict(extra='forbid')
    objective: str = Field(min_length=1)
    status: Literal['ready', 'needs_clarification']
    clarification: str | None = None

    @model_validator(mode='after')
    def validate_status(self):
        if self.status == 'needs_clarification' and not self.clarification:
            raise ValueError('Deferred objectives need a specific clarification question')
        if self.status == 'ready' and self.clarification:
            raise ValueError('Ready objectives cannot require clarification')
        return self


class DeferredObjective(BaseModel):
    model_config = ConfigDict(extra='forbid')
    objective: str = Field(min_length=1)
    clarification: str = Field(min_length=1)


class QuestionIntent(BaseModel):
    """What the user wants; deliberately contains no routing or execution choices."""

    model_config = ConfigDict(extra='forbid')
    intent: Literal['lookup', 'aggregation', 'comparison', 'advanced_analysis', 'clarification']
    objective: str = Field(min_length=1)
    constraints: list[str] = Field(default_factory=list)
    clarification: str | None = None
    objectives: list[Objective] = Field(default_factory=list)

    @model_validator(mode='after')
    def check_clarification(self):
        if self.intent == 'clarification' and any(o.status == 'ready' for o in self.objectives):
            raise ValueError('Do not block ready objectives; use their intent and defer only unclear parts')
        if self.objectives and all(o.status == 'needs_clarification' for o in self.objectives):
            if self.intent != 'clarification':
                raise ValueError('A wholly unclear request requires clarification')
        if self.intent == 'clarification' and not self.clarification:
            raise ValueError('Provide a clarification question when intent is unclear')
        return self


class ExecutionStep(BaseModel):
    model_config = ConfigDict(extra='forbid')
    step_id: str = Field(min_length=1)
    action: Literal['schema', 'sql', 'python']
    instruction: str = Field(min_length=1)
    expected_output: str = Field(min_length=1)


class ExecutionPlan(BaseModel):
    """A concise operational plan, rather than a transcript of model deliberation."""

    model_config = ConfigDict(extra='forbid')
    tenant_id: TenantId
    workspace_id: WorkspaceId
    sources: list[SourceId]
    execution_mode: Literal['sql', 'sql_python', 'none']
    execution_reason: str | None = None
    steps: list[ExecutionStep] = Field(max_length=12)
    validation_checks: list[str]
    python_reason: str | None = None
    response: str | None = None
    deferred_objectives: list[DeferredObjective] = Field(default_factory=list)

    @model_validator(mode='after')
    def validate_plan(self):
        _, workspace = get_workspace(self.workspace_id, tenant_id=self.tenant_id)
        if set(self.sources) - workspace.sources.keys():
            raise ValueError('Selected sources must belong to the specified workspace')
        if len(self.sources) != len(set(self.sources)):
            raise ValueError('Source IDs must be unique')
        if self.execution_mode != 'sql_python' and self.python_reason:
            raise ValueError('A Python purpose belongs only to a plan that requires Python')
        if self.execution_mode == 'none':
            if self.sources or self.steps or not self.response:
                raise ValueError('No-execution plans need a response and no sources or steps')
            return self
        if self.response is not None:
            raise ValueError('Executable plans must answer from results, not a prewritten response')
        if not self.execution_reason or not self.execution_reason.strip():
            raise ValueError('Explain why the selected execution approach serves the business objective')
        if not self.sources or not self.steps or not self.validation_checks:
            raise ValueError('Executable plans need sources, steps and validation checks')
        ids = [step.step_id for step in self.steps]
        actions = [step.action for step in self.steps]
        if len(ids) != len(set(ids)):
            raise ValueError('Step IDs must be unique')
        if actions[0] != 'schema' or actions.count('schema') != 1 or 'sql' not in actions:
            raise ValueError('Inspect schema first, then obtain SQL evidence')
        if 'python' in actions and actions.index('python') < actions.index('sql'):
            raise ValueError('SQL must retrieve data before Python runs')
        if self.execution_mode == 'sql' and 'python' in actions:
            raise ValueError('SQL-only plans cannot include Python steps')
        if self.execution_mode == 'sql_python' and (
            'python' not in actions or not self.python_reason or not self.python_reason.strip()
        ):
            raise ValueError('Hybrid plans need a Python step and its non-SQL purpose')
        return self


CLASSIFIER_INSTRUCTIONS = """
Identify only WHAT the user asks to accomplish. Return intent, a short objective,
explicit constraints (filters, dates, groupings, output units, requested algorithms)
and a clarification question only when the request itself is ambiguous.
Intent is lookup, aggregation, comparison, advanced_analysis or clarification.
Do not choose data sources, tables, tools, SQL/Python, or execution steps. Do not
query data, decide source availability, or answer the data question. Do not add
assumptions or constraints that the user did not specify. Workspace selection is
fixed by the caller and is not your task. Interpret requests within that workspace
and do not ask the user to choose a dataset already fixed by the caller.
You receive trusted workspace context with its domain description and glossary.
Use that vocabulary to interpret acronyms. Never ask which tenant, workspace or
dataset to use: those IDs are already supplied. Do not ask the user to define a
term that the glossary resolves. Context is for interpretation, not source routing.
Split compound questions into objectives, preserving every requested part. Mark
each objective ready or needs_clarification. If ANY objective is ready, classify
the request by that objective, not clarification, and do not block the ready part.
Apply the supplied workspace domain guidance when interpreting terms and deciding
which objectives require missing user input. Do not invent domain rules or fill
missing decision criteria with assumptions.
Use overall clarification only if no part can proceed. Explicitly list objectives
even for a single-task question. Clarifications must target missing user intent,
not context already present in the supplied workspace.
Return a concise task description, without private deliberation.
The user is a business user. They need not mention a programming language,
query engine or implementation method. Do not ask them to choose SQL or Python,
or treat the absence of a technical method as ambiguity. Preserve the business
outcome and leave implementation decisions to the planner.
"""

PLANNER_INSTRUCTIONS = """
Create a concise, actionable execution plan from the question and classified intent.
Choose the implementation automatically from the business outcome, the available
data and the workspace guidance. Never require the user to say 'use SQL' or 'use
Python', select a language, or provide code. Missing technical preferences are not
missing business intent. Use documented workspace analysis defaults where they
apply and disclose the chosen method; clarify genuinely missing business criteria.
Output operational steps with IDs, actions, instructions and expected outputs;
do not output internal deliberation or a chain-of-thought transcript.
The caller's tenant and workspace are immutable. Return those exact tenant_id
and workspace_id values, and use only the
provided catalog. Never switch workspace or plan cross-workspace access. If any
requested data is unavailable, explain the limitation without inventing data.
For compound questions plan the supported ready objectives and defer the others.
Use execution_mode none, sources [] and steps [] only when no objective can be
served in this workspace. Preserve deferred_objectives and their clarification
questions. An unresolved objective must not block another supported ready objective.
Select the minimum required sources from this workspace. Do not answer before
execution. An executable plan must have response null.
For execution, the first step must inspect schema (action schema). Then use SQL
for retrieval, filtering, joins, aggregates, rankings, ratios, window functions
and supported statistics. Include all required calculations in the steps.
Plan only requested outputs and supported calculations. Apply the injected domain
guidance to choose valid measures, grouping levels and interpretation rules.
Use execution_mode sql unless an identified algorithm cannot reasonably be done
with SQL. Hybrid sql_python plans must first retrieve/pre-aggregate with SQL, then
include the specific Python calculation and python_reason. Use Python only for
an algorithm not reasonably supported by SQL; ordinary aggregation needs no Python.
For every executable plan, supply execution_reason in plain business language:
explain why querying the data is sufficient or why a separate calculation is
needed. For sql_python, python_reason must name the calculation, its inputs and
why SQL alone is insufficient. Publish the choice before execution, including
the Python step when needed. The planner owns this decision, not the user.
Use one schema step, at least one SQL step, at most 12 steps. Each step is completed
by one successful tool call, so split multiple queries into distinct SQL steps.
List concise validation_checks for the requested results, including completeness,
filters, units, dates and join cardinality as applicable.
The application publishes the validated plan before execution; do not add a
publication tool step. Plan only schema, SQL and Python actions.
Only catalog metadata is available now. SQL/code is generated during execution
after schema inspection. Respect the supplied domain guidance and catalog metadata.
Before schema inspection, describe required data concepts without asserting
unverified column names. Validation checks are checks to perform, not claims that
they have already passed. Use supplied action rules when applicable; do not assume
missing policies or request criteria that the user or workspace already provided.
Request clarification or report missing inputs rather than planning a calculation
that the available data does not support.
"""


async def classify_intent(question: str, *, model: Model, workspace_context: str) -> QuestionIntent:
    agent = Agent(model, output_type=QuestionIntent,
                  instructions=CLASSIFIER_INSTRUCTIONS + '\nTrusted workspace context:\n' + workspace_context,
                  retries=3, model_settings=model_settings())

    @agent.output_validator
    def clarify_only_missing_intent(intent: QuestionIntent) -> QuestionIntent:
        questions = [intent.clarification or ''] + [o.clarification or '' for o in intent.objectives]
        glossary = json.loads(workspace_context).get('glossary', {})
        for text in questions:
            if re.search(r'\b(?:sql\s*(?:or|/|versus|vs\.?)\s*python|'
                         r'python\s*(?:or|/|versus|vs\.?)\s*sql)\b', text, re.IGNORECASE):
                raise ModelRetry('The planner chooses SQL/Python; clarify only missing business requirements')
            if re.search(r'\b(?:which|what)\s+(?:(?:tenant|workspace|dataset)(?:\s+id)?|data\s+source)\b',
                         text, re.IGNORECASE):
                raise ModelRetry('Tenant, workspace and dataset are already fixed; clarify only missing task intent')
            for term, definition in glossary.items():
                quoted_term = r'''["'“”‘’]?''' + re.escape(term) + r'''["'“”‘’]?'''
                if re.search(r'\bwhat\s+(?:does\s+' + quoted_term
                             + r'\s+(?:mean|refer|stand)|is\s+(?:an?\s+)?'
                             + quoted_term + r'(?=\W|$))', text, re.IGNORECASE):
                    raise ModelRetry(f'The workspace glossary already defines {term}: {definition}')
        return intent

    return (await agent.run(question, usage_limits=UsageLimits(request_limit=4))).output


async def build_plan(
    question: str, intent: QuestionIntent, *, tenant_id: str | int, workspace_id: str | int, model: Model,
) -> ExecutionPlan:
    key, _ = get_workspace(workspace_id, tenant_id=tenant_id)
    tenant_key = str(tenant_id)
    deferred = [DeferredObjective(objective=o.objective, clarification=o.clarification)
                for o in intent.objectives if o.status == 'needs_clarification']
    if intent.intent == 'clarification':
        return ExecutionPlan(tenant_id=tenant_key, workspace_id=key, sources=[], execution_mode='none',
                             steps=[], validation_checks=[], response=intent.clarification,
                             deferred_objectives=deferred)
    agent = Agent(
        model, output_type=ExecutionPlan,
        instructions=(PLANNER_INSTRUCTIONS + '\nWorkspace domain guidance:\n'
                      + domain_context(key, tenant_id=tenant_key, stage='planner')
                      + '\nWorkspace catalog:\n' + routing_catalog(key, tenant_id=tenant_key)),
        retries=3, model_settings=model_settings(),
    )

    @agent.output_validator
    def enforce_workspace(plan: ExecutionPlan) -> ExecutionPlan:
        if (plan.tenant_id, plan.workspace_id) != (tenant_key, key):
            raise ModelRetry(f'Use only tenant {tenant_key}, workspace {key}')
        # Clarifications already identified by classification cannot be dropped
        # by the planner or silently replaced with invented decision rules.
        pending = {item.objective.strip().casefold(): item for item in plan.deferred_objectives}
        for item in deferred:
            pending[item.objective.strip().casefold()] = item
        plan.deferred_objectives = list(pending.values())
        return plan

    prompt = json.dumps({'question': question, 'classified_intent': intent.model_dump()})
    return (await agent.run(prompt, usage_limits=UsageLimits(request_limit=4))).output
