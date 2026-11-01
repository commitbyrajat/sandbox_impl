"""Separate intent classification from workspace-bound execution planning."""

import json
import re
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    create_model,
    model_validator,
)
from pydantic_ai import Agent, ModelRetry, RunContext, UsageLimits
from pydantic_ai.models import Model
from pydantic_ai.capabilities import AbstractCapability, ProcessHistory

from datasource_mapping import (
    SourceId,
    TenantId,
    WorkspaceId,
    get_workspace,
    routing_catalog,
)
from domain_instructions import domain_context, get_domain_profile
from query_recipes import SqlRecipeCall, match_sql_recipe
from runtime import model_settings, trace
from result_contracts import SQLResultContract, validate_contract_sources
from scientific_models import scientific_catalog
from calculation_contracts import AnalysisKind, CalculationMethod, PythonResultContract
from repair_feedback import RepairFeedback, repair_handoff_messages
from semantic_models import QueryIR, MetricSelection
from semantic_query import SemanticCatalog, SemanticQueryError, compile_query
from observation_planning import bind_observations


class Objective(BaseModel):
    model_config = ConfigDict(extra="forbid")
    objective: str = Field(min_length=1)
    status: Literal["ready", "needs_clarification"]
    clarification: str | None = None

    @model_validator(mode="after")
    def validate_status(self):
        if self.status == "needs_clarification" and not self.clarification:
            raise ValueError(
                "Deferred objectives need a specific clarification question"
            )
        if self.status == "ready" and self.clarification:
            raise ValueError("Ready objectives cannot require clarification")
        return self


class DeferredObjective(BaseModel):
    model_config = ConfigDict(extra="forbid")
    objective: str = Field(min_length=1)
    clarification: str = Field(min_length=1)


class QuestionIntent(BaseModel):
    """What the user wants; deliberately contains no routing or execution choices."""

    model_config = ConfigDict(extra="forbid")
    intent: Literal[
        "lookup", "aggregation", "comparison", "advanced_analysis", "clarification"
    ]
    objective: str = Field(min_length=1)
    constraints: list[str] = Field(default_factory=list)
    analysis_kind: AnalysisKind | None = None
    clarification: str | None = None
    objectives: list[Objective] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_clarification(self):
        if self.intent == "clarification" and any(
            o.status == "ready" for o in self.objectives
        ):
            raise ValueError(
                "Do not block ready objectives; use their intent and defer only unclear parts"
            )
        if self.objectives and all(
            o.status == "needs_clarification" for o in self.objectives
        ):
            if self.intent != "clarification":
                raise ValueError("A wholly unclear request requires clarification")
        if self.intent == "clarification" and not self.clarification:
            raise ValueError("Provide a clarification question when intent is unclear")
        return self


class ExecutionStep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    step_id: str = Field(min_length=1)
    action: Literal["schema", "sql", "python"]
    instruction: str = Field(min_length=1)
    expected_output: str = Field(min_length=1)
    output_contract: SQLResultContract | None = None
    input_steps: list[str] = Field(default_factory=list)
    calculation: CalculationMethod | None = None
    result_contract: PythonResultContract | None = None
    logical_query: QueryIR | None = None


class ExecutionPlan(BaseModel):
    """A concise operational plan, rather than a transcript of model deliberation."""

    model_config = ConfigDict(extra="forbid")
    tenant_id: TenantId
    workspace_id: WorkspaceId
    sources: list[SourceId]
    execution_mode: Literal["sql", "sql_python", "none"]
    execution_reason: str | None = None
    sql_recipe: SqlRecipeCall | None = None
    # Previously published/manual plans retain the original execution interface.
    query_language: Literal["legacy_sql", "semantic"] = "legacy_sql"
    steps: list[ExecutionStep] = Field(max_length=12)
    validation_checks: list[str]
    python_reason: str | None = None
    response: str | None = None
    deferred_objectives: list[DeferredObjective] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_plan(self):
        if self.sql_recipe is not None and (
            self.execution_mode != "sql"
            or [step.action for step in self.steps] != ["schema", "sql"]
        ):
            raise ValueError(
                "Configured SQL recipes require one schema step and one SQL step"
            )
        _, workspace = get_workspace(self.workspace_id, tenant_id=self.tenant_id)
        if set(self.sources) - workspace.sources.keys():
            raise ValueError("Selected sources must belong to the specified workspace")
        if len(self.sources) != len(set(self.sources)):
            raise ValueError("Source IDs must be unique")
        if self.execution_mode != "sql_python" and self.python_reason:
            raise ValueError(
                "A Python purpose belongs only to a plan that requires Python"
            )
        if self.execution_mode == "none":
            if self.sources or self.steps or not self.response:
                raise ValueError(
                    "No-execution plans need a response and no sources or steps"
                )
            return self
        if self.response is not None:
            raise ValueError(
                "Executable plans must answer from results, not a prewritten response"
            )
        if not self.execution_reason or not self.execution_reason.strip():
            raise ValueError(
                "Explain why the selected execution approach serves the business objective"
            )
        if not self.sources or not self.steps or not self.validation_checks:
            raise ValueError(
                "Executable plans need sources, steps and validation checks"
            )
        ids = [step.step_id for step in self.steps]
        actions = [step.action for step in self.steps]
        if len(ids) != len(set(ids)):
            raise ValueError("Step IDs must be unique")
        prior_sql = set()
        for step in self.steps:
            if step.action != "sql" and step.logical_query is not None:
                raise ValueError("Only SQL steps declare logical_query")
            if step.action != "python" and (step.calculation or step.result_contract):
                raise ValueError(
                    "Only Python steps declare calculation/result_contract"
                )
            if step.result_contract:
                if any(
                    rule.input_step and rule.input_step not in step.input_steps
                    for rule in step.result_contract.invariants
                ):
                    raise ValueError(
                        "Result invariants must reference this Python step's SQL input_steps"
                    )
            if step.calculation and step.calculation.method.startswith("scientific."):
                method = step.calculation.method.removeprefix("scientific.")
                operation = scientific_catalog()["operations"].get(method)
                if operation is None or step.calculation.kind not in operation.get(
                    "compatible_kinds", [operation["analysis_kind"], "custom"]
                ):
                    raise ValueError(
                        "Calculation method does not implement its declared analysis kind; clustering cannot balance totals"
                    )
            if step.output_contract is not None and step.action != "sql":
                raise ValueError("Only SQL steps declare output_contract")
            if step.input_steps and (
                step.action != "python" or set(step.input_steps) - prior_sql
            ):
                raise ValueError(
                    "Python input_steps must reference preceding SQL steps"
                )
            if step.action == "sql":
                prior_sql.add(step.step_id)
        if (
            actions[0] != "schema"
            or actions.count("schema") != 1
            or "sql" not in actions
        ):
            raise ValueError("Inspect schema first, then obtain SQL evidence")
        if "python" in actions and actions.index("python") < actions.index("sql"):
            raise ValueError("SQL must retrieve data before Python runs")
        if self.execution_mode == "sql" and "python" in actions:
            raise ValueError("SQL-only plans cannot include Python steps")
        if self.execution_mode == "sql_python" and (
            "python" not in actions
            or not self.python_reason
            or not self.python_reason.strip()
        ):
            raise ValueError("Hybrid plans need a Python step and its non-SQL purpose")
        return self


class SchemaPlanningStep(ExecutionStep):
    action: Literal["schema"]
    output_contract: None = None
    input_steps: list[str] = Field(default_factory=list, max_length=0)
    calculation: None = None
    result_contract: None = None
    logical_query: None = None


class SQLPlanningStep(ExecutionStep):
    action: Literal["sql"]
    output_contract: SQLResultContract
    input_steps: list[str] = Field(default_factory=list, max_length=0)
    calculation: None = None
    result_contract: None = None
    logical_query: QueryIR


class PythonPlanningStep(ExecutionStep):
    action: Literal["python"]
    output_contract: None = None
    input_steps: list[str] = Field(min_length=1)
    calculation: CalculationMethod
    result_contract: PythonResultContract
    logical_query: None = None


class PlannerOutput(ExecutionPlan):
    """LLM schema expresses step-specific constraints; stored plans stay compatible."""

    sql_recipe: None = None
    query_language: Literal["semantic"] = "semantic"
    steps: list[
        Annotated[
            SchemaPlanningStep | SQLPlanningStep | PythonPlanningStep,
            Field(discriminator="action"),
        ]
    ] = Field(max_length=12)

    @model_validator(mode="before")
    @classmethod
    def order_independent_stages(cls, value):
        """Lifecycle order is host-owned; do not ask an LLM to repair a sort.

        SQL nodes cannot consume Python outputs in this IR, so all SQL producers
        can precede their Python consumers. Unknown IDs and duplicate schemas
        still fail the normal dependency/shape validation.
        """
        if not isinstance(value, dict) or value.get("execution_mode") == "none":
            return value
        steps = value.get("steps")
        if not isinstance(steps, list) or not all(
            isinstance(step, dict) for step in steps
        ):
            return value
        steps = list(steps)
        for index, step in enumerate(steps):
            query = step.get("logical_query")
            contract = step.get("output_contract")
            if (
                step.get("action") == "sql"
                and isinstance(query, dict)
                and query.get("metrics")
                and not query.get("dimensions")
                and isinstance(contract, dict)
            ):
                # A scalar aggregate has one output row and no entity identity.
                # Population coverage is not an executable constraint on that row.
                steps[index] = {
                    **step,
                    "output_contract": {
                        **contract,
                        "grain": [],
                        "population": None,
                        "min_rows": 1,
                        "max_rows": 1,
                    },
                }
        if not any(step.get("action") == "schema" for step in steps):
            used = {step.get("step_id") for step in steps}
            key = "host_schema"
            while key in used:
                key += "_"
            steps.insert(
                0,
                {
                    "step_id": key,
                    "action": "schema",
                    "instruction": "Verify registered column names and types against the selected source schemas.",
                    "expected_output": "Verified scoped source metadata",
                },
            )
        return {
            **value,
            "steps": sorted(
                steps,
                key=lambda step: {"schema": 0, "sql": 1, "python": 2}.get(
                    step.get("action"), 3
                ),
            ),
        }


def planner_output_type(
    analysis_kind: AnalysisKind | None,
    catalog: SemanticCatalog | None = None,
    observation_policies=None,
):
    """Constrain a known business outcome in the wire schema, before generation."""
    sql_step_type = SQLPlanningStep
    if catalog is not None:
        selection = create_model(
            "CatalogMetricSelection",
            __base__=MetricSelection,
            metric=(
                Literal[tuple(sorted(catalog.metrics))],
                Field(
                    description="Select an exact metric ID from this workspace's catalog."
                ),
            ),
        )
        query_type = create_model(
            "CatalogQueryIR",
            __base__=QueryIR,
            base_table=(Literal[tuple(sorted(catalog.tables))], ...),
            metrics=(list[selection], Field(default_factory=list, max_length=30)),
            relationships=(
                (
                    list[Literal[tuple(sorted(catalog.relationships))]]
                    if catalog.relationships
                    else list[str]
                ),
                Field(
                    default_factory=list,
                    max_length=15 if catalog.relationships else 0,
                    description="Optional catalog relationship hints for ambiguous routes. The compiler completes missing edges; prefer [] when the path is unambiguous. Never SQL predicates or table.column expressions.",
                ),
            ),
        )
        sql_step_type = create_model(
            "SQLPlanningStep", __base__=SQLPlanningStep, logical_query=(query_type, ...)
        )
    base_type = PlannerOutput
    if catalog and observation_policies:

        def bind_policy(cls, value):
            return bind_observations(
                value, catalog, observation_policies, analysis_kind
            )

        base_type = create_model(
            "ObservationPlannerOutput",
            __base__=PlannerOutput,
            __validators__={
                "bind_observation_policy": model_validator(mode="before")(bind_policy)
            },
        )
    if analysis_kind is None and catalog is None:
        return PlannerOutput
    if analysis_kind is None:
        return create_model(
            "CatalogPlannerOutput",
            __base__=base_type,
            steps=(
                list[
                    Annotated[
                        SchemaPlanningStep | sql_step_type | PythonPlanningStep,
                        Field(discriminator="action"),
                    ]
                ],
                Field(max_length=12),
            ),
        )
    method_type = create_model(
        "RequestedCalculation",
        __base__=CalculationMethod,
        kind=(Literal[analysis_kind], ...),
    )
    step_type = create_model(
        "RequestedPythonStep",
        __base__=PythonPlanningStep,
        calculation=(method_type, ...),
    )
    return create_model(
        "RequestedPlannerOutput",
        __base__=base_type,
        steps=(
            list[
                Annotated[
                    SchemaPlanningStep | sql_step_type | step_type,
                    Field(discriminator="action"),
                ]
            ],
            Field(max_length=12),
        ),
    )


class PlannerDiagnostics(AbstractCapability):
    def __init__(self, catalog=None):
        self.catalog = catalog
        self.latest_feedback = None
        self.owned_handoffs = set()

    def handoff(self, ctx: RunContext, messages):
        return repair_handoff_messages(
            messages,
            self.latest_feedback,
            "Return a complete corrected execution plan using the configured output schema. Preserve the question, classified outcome and tenant/workspace; do not return a partial plan or a repair-feedback object.",
            self.owned_handoffs,
        )

    def on_output_validate_error(self, ctx, *, output_context, output, error):
        errors = (
            error.errors(include_input=False, include_context=True, include_url=False)
            if isinstance(error, ValidationError)
            else str(error)
        )
        query_context = self.query_validation_context(error, output)
        if self.catalog:
            for context in query_context:
                context["selected_metric_definitions"] = {
                    name: {
                        **self.catalog.metrics[name].model_dump(),
                        "source_grain": (
                            self.catalog.tables[self.catalog.metrics[name].table].grain
                            if self.catalog.metrics[name].table in self.catalog.tables
                            else None
                        ),
                    }
                    for name in context.pop("metric_ids", [])
                    if name in self.catalog.metrics
                }
        trace(
            "PLAN_REJECTED: "
            + json.dumps(
                {
                    "attempt": ctx.retry + 1,
                    "error_type": type(error).__name__,
                    "errors": errors,
                    "query_context": query_context,
                },
                default=str,
            )
        )
        self.latest_feedback = RepairFeedback(
            target="planner",
            phase="plan_validation",
            diagnostic=json.dumps(errors, default=str),
            artifact={"rejected_plan": output},
            authority={
                "schema": "PlannerOutput",
                "workspace_guidance": "datasource.json and domain_instructions.json",
                "query_output_requirements": query_context,
            },
            retries_remaining=max(0, 3 - ctx.retry),
            instructions=[
                "Correct all independent plan defects together; preserve the user's objective and workspace."
            ],
        )
        raise ModelRetry(self.latest_feedback.publish()) from error

    @staticmethod
    def query_validation_context(error, output):
        """Make alias/empty-projection errors actionable without dumping row data."""
        if not isinstance(error, ValidationError):
            return []
        if isinstance(output, str):
            try:
                output = json.loads(output)
            except ValueError:
                output = {}
        raw_steps = output.get("steps", []) if isinstance(output, dict) else []
        contexts = []
        for detail in error.errors(
            include_input=True, include_context=False, include_url=False
        ):
            location = detail["loc"]
            if (
                not any(key in location for key in ("logical_query", "output_contract"))
                or "steps" not in location
            ):
                continue
            index = location[location.index("steps") + 1]
            step = (
                raw_steps[index]
                if isinstance(index, int) and index < len(raw_steps)
                else {}
            )
            query = step.get("logical_query") or detail.get("input")
            if not isinstance(query, dict):
                continue
            contexts.append(
                {
                    "step_id": step.get("step_id"),
                    "location": list(location),
                    "required_output_aliases": list(
                        (step.get("output_contract") or {}).get("required_columns", {})
                    ),
                    "grain": (step.get("output_contract") or {}).get("grain", []),
                    "missing_grain_columns": sorted(
                        set((step.get("output_contract") or {}).get("grain", []))
                        - set(
                            (step.get("output_contract") or {}).get(
                                "required_columns", {}
                            )
                        )
                    ),
                    "missing_output_aliases": sorted(
                        set(
                            (step.get("output_contract") or {}).get(
                                "required_columns", {}
                            )
                        )
                        - {
                            item.get("name")
                            for section in ("dimensions", "metrics", "calculations")
                            for item in query.get(section, [])
                            if isinstance(item, dict)
                        }
                    ),
                    "metric_ids": [
                        item.get("metric")
                        for item in query.get("metrics", [])
                        if isinstance(item, dict)
                    ],
                    "population_remedy": "population is optional: set null for filtered/composite cohorts. Do not replace an invalid key with another balance, date or partial grouping key.",
                    "selected_aliases": {
                        section: [
                            item.get("name")
                            for item in query.get(section, [])
                            if isinstance(item, dict)
                        ]
                        for section in ("dimensions", "metrics", "calculations")
                    },
                }
            )
        return contexts


def consolidate_python_steps(plan: ExecutionPlan) -> ExecutionPlan:
    """Batch adjacent calculations while preserving every requested output."""
    steps = []
    for step in plan.steps:
        if (
            step.action == "python"
            and steps
            and steps[-1].action == "python"
            and not step.result_contract
            and not steps[-1].result_contract
            and not step.calculation
            and not steps[-1].calculation
        ):
            previous = steps[-1]
            steps[-1] = previous.model_copy(
                update={
                    "instruction": previous.instruction
                    + "\nAlso: "
                    + step.instruction
                    + "\nReturn all requested calculations together in one structured result; reuse intermediate values.",
                    "expected_output": previous.expected_output
                    + "\n"
                    + step.expected_output,
                    "input_steps": list(
                        dict.fromkeys([*previous.input_steps, *step.input_steps])
                    ),
                }
            )
        else:
            steps.append(step)
    return ExecutionPlan.model_validate({**plan.model_dump(), "steps": steps})


CLASSIFIER_INSTRUCTIONS = """
Identify only WHAT the user asks to accomplish. Return intent, a short objective,
explicit constraints (filters, dates, groupings, output units, requested algorithms)
and a clarification question only when the request itself is ambiguous.
Intent is lookup, aggregation, comparison, advanced_analysis or clarification.
For an analytical outcome, set analysis_kind to clustering (similar profiles),
balancing (roughly equal totals), uncertainty (interval around an estimate),
simulation, hypothesis_test or custom. This describes the requested outcome,
not an algorithm, language or implementation. Otherwise use null.
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
SQL execution uses a typed semantic QueryIR compiled by the host, never LLM SQL.
Describe business metrics and output grain using semantic_metrics, semantic_tables
and configured relationship meanings. The executor chooses catalog metric IDs,
dimensions, filters and time selection; the compiler owns joins and aggregation.
Do not prescribe SUM/MAX/GROUP BY or invent missing business metric definitions.
For temporal probability metrics, preserve their configured observation horizon.
For latest/latest_common selection, use the dynamic snapshot alone without a guessed
date predicate. The actual stored snapshot key is authoritative; a calendar month
does not imply a month-end date. Select a range for an explicitly bounded period.
When the configured backend is MetricFlow, cumulative metric IDs use the catalog's
cumulative definition. Group their source time field using an explicit day, month
or year transform, and use one bounded time selection without duplicate date
predicates. Use the input metric's own time column; a month grouping does not
require joining another table that happens to contain a month column. Select only
tables and relationship paths needed for the requested metrics, dimensions and
filters. Do not invent a cumulative formula or cumulative-window SQL.
Every SQL step must include logical_query as a QueryIR JSON object. Resolve its
dimensions and catalog metric IDs now from the workspace metadata. The host
validates and publishes this logical query, then executes it without asking for
SQL or generating the same query again. Include every required output alias.
Project each required alias exactly once: source identity/grouping fields belong
in dimensions, catalog aggregate values in metrics, and genuinely new derived
values in calculations. Do not repeat a selected column in calculations as a
pass-through reference. Metric IDs are catalog lookups; names are output aliases.
Preserve required canonical time and identity aliases. A composite observation
is identified by its full grain tuple; it does not require an invented cohort_key.
If a composite string identifier is needed, declare its concat calculation
explicitly while keeping all required grain projections. Renaming month to a
composite-key label does not construct a composite identity.
dimensions and metrics must not both be empty. In alias correction feedback, fix
the exact named locations; do not change amounts, grouping or filters to evade it.
Table observation counts use metric ID <table>.__count. For catalog formulas,
use their metric IDs rather than reconstructing their business logic in prose.
Population contracts compare against the whole source population. Do not declare
an unfiltered population for an intentionally filtered cohort analysis.
Set output_contract.population to null for filtered composite cohorts and ordinary
grouped reports. It is optional, not a request to choose any source column. Only
whole registered entity retention uses a single-key population contract. Dates,
balances, probabilities and a partial composite key are not entity identities.
For Python steps declare calculation.kind and calculation.method (an available
scientific.<function> or a named custom algorithm). Preserve the classified
analysis_kind: equal-total balancing and similarity clustering are different tasks.
Declare result_contract.required_fields as dot paths with expected coarse types
and result_contract.invariants when verifiable: count_equals (output_path with '*',
expected_count), sum_equals_input (output_path, input_step, input_column, tolerance),
members_equal_input (output_path, input_step, input_column). '*' traverses lists or
dict values. Input invariants name SQL steps in input_steps, never guessed result
IDs. For groups, validate group count, member coverage and total reconciliation.
For intervals, declare the estimate and lower/upper output fields. These contracts
must describe the final result, not a hardcoded implementation or answer.
Paths start at the final returned value. For a result with a groups wrapper,
use required_fields {"groups":"list", "groups.*.members":"list",
"groups.*.total":"numeric"}; a count_equals check on "groups.*" requires
expected_count equal to the requested group count. Never declare group row fields
as root fields while referring to a groups wrapper in invariants.
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
Leave sql_recipe null. Configured recipes are selected by the host before this
planning call; never invent a recipe ID or template hash. Plan generated SQL steps.
For execution, the first step must inspect schema (action schema). Then use SQL
for retrieval, filtering, joins, aggregates, rankings, ratios, window functions
and supported statistics. Include all required calculations in the steps.
Plan only requested outputs and supported calculations. Apply the injected domain
guidance to choose valid measures, grouping levels and interpretation rules.
When the user omits a period, preserve the workspace's period/as-of defaults in
the SQL instruction and returned date evidence. Do not expand a snapshot measure
across all historical periods unless the question or workspace rules require it.
Use execution_mode sql unless an identified algorithm cannot reasonably be done
with SQL. Hybrid sql_python plans must first retrieve/pre-aggregate with SQL, then
include the specific Python calculation and python_reason. Use Python only for
an algorithm not reasonably supported by SQL; ordinary aggregation needs no Python.
For every executable plan, supply execution_reason in plain business language:
explain why querying the data is sufficient or why a separate calculation is
needed. For sql_python, python_reason must name the calculation, its inputs and
why SQL alone is insufficient. Publish the choice before execution, including
the Python step when needed. The planner owns this decision, not the user.
The scientific module provides SciPy k-means, bootstrap, distribution sampling,
binomial/Bernoulli draws, resampling indices and Monte Carlo hypothesis tests.
The executor also has bounded async numpy, pandas, scipy and statistics facades
for implementation details. Plan business data and algorithms here; the executor
receives their exact function signatures before generating code.
Prefer these library operations to handwritten algorithms when they serve the
question. Publish initialization, confidence/quantile method, seed and library RNG
choices; NumPy seeded draws differ from Python random. Domain assumptions remain
in workspace guidance. Monte Carlo simulations use sampled outcomes; do not
substitute a Monte Carlo hypothesis test for an outcome distribution.
Batch related Python calculations on the same SQL inputs into one step and one
structured result. Reuse intermediate calculations; do not create separate steps
for a statistic already calculated by the main algorithm. Avoid redundant SQL
materializations: use a CTE/subquery inside the final retrieval where possible.
Every SQL step MUST declare output_contract with required_columns keyed by exact
output aliases, coarse types (numeric/string/integer/date/timestamp/boolean/any),
nullability, meaning, grain (unique non-null observation keys), and applicable
row-count bounds. Include population {table, source_key, result_key} when ALL
registered entities must be retained. Use configured registry columns and types.
For each Python step supply input_steps naming its SQL producers. Those producers
must return the complete feature/observation datasets, not preliminary date lookups.
Preserve requested per-unit denominators in the column meanings and SQL instructions.
Put snapshot lookup queries inside the feature query, or plan a separate SQL step
with a distinct output contract. Never consume a feature step with a date-only query.
Schema/Python steps use output_contract=null. Python results are described in
expected_output; output_contract validates SQL datasets only. A population's
result_key is an observation ID in grain, never the numerical measure being
analyzed. Its source_key must represent that same identity in a registry table.
Use one schema step, at least one SQL step, at most 12 steps. Each step is completed
by one successful tool call, so split multiple queries into distinct SQL steps.
List concise validation_checks for the requested results, including completeness,
filters, units, dates and join cardinality as applicable.
The application publishes the validated plan before execution; do not add a
publication tool step. Plan only schema, SQL and Python actions.
Only catalog metadata is available now. Generate logical_query using its registered
columns and metric definitions, without SQL text. Schema inspection verifies the
registry before the host compiles and executes SQL. Python code is generated after
complete SQL inputs are known. Describe required data concepts in step prose;
do not guess column names absent from the registry. Validation checks are checks to perform, not claims that
they have already passed. Use supplied action rules when applicable; do not assume
missing policies or request criteria that the user or workspace already provided.
Request clarification or report missing inputs rather than planning a calculation
that the available data does not support.
"""


async def classify_intent(
    question: str, *, model: Model, workspace_context: str
) -> QuestionIntent:
    agent = Agent(
        model,
        output_type=QuestionIntent,
        instructions=CLASSIFIER_INSTRUCTIONS
        + "\nTrusted workspace context:\n"
        + workspace_context,
        retries=3,
        model_settings=model_settings(),
    )

    @agent.output_validator
    def clarify_only_missing_intent(intent: QuestionIntent) -> QuestionIntent:
        questions = [intent.clarification or ""] + [
            o.clarification or "" for o in intent.objectives
        ]
        glossary = json.loads(workspace_context).get("glossary", {})
        for text in questions:
            if re.search(
                r"\b(?:sql\s*(?:or|/|versus|vs\.?)\s*python|"
                r"python\s*(?:or|/|versus|vs\.?)\s*sql)\b",
                text,
                re.IGNORECASE,
            ):
                raise ModelRetry(
                    "The planner chooses SQL/Python; clarify only missing business requirements"
                )
            if re.search(
                r"\b(?:which|what)\s+(?:(?:tenant|workspace|dataset)(?:\s+id)?|data\s+source)\b",
                text,
                re.IGNORECASE,
            ):
                raise ModelRetry(
                    "Tenant, workspace and dataset are already fixed; clarify only missing task intent"
                )
            for term, definition in glossary.items():
                quoted_term = r"""["'“”‘’]?""" + re.escape(term) + r"""["'“”‘’]?"""
                if re.search(
                    r"\bwhat\s+(?:does\s+"
                    + quoted_term
                    + r"\s+(?:mean|refer|stand)|is\s+(?:an?\s+)?"
                    + quoted_term
                    + r"(?=\W|$))",
                    text,
                    re.IGNORECASE,
                ):
                    raise ModelRetry(
                        f"The workspace glossary already defines {term}: {definition}"
                    )
        return intent

    return (await agent.run(question, usage_limits=UsageLimits(request_limit=4))).output


async def build_plan(
    question: str,
    intent: QuestionIntent,
    *,
    tenant_id: str | int,
    workspace_id: str | int,
    model: Model,
) -> ExecutionPlan:
    key, workspace = get_workspace(workspace_id, tenant_id=tenant_id)
    tenant_key = str(tenant_id)
    deferred = [
        DeferredObjective(objective=o.objective, clarification=o.clarification)
        for o in intent.objectives
        if o.status == "needs_clarification"
    ]
    if intent.intent == "clarification":
        return ExecutionPlan(
            tenant_id=tenant_key,
            workspace_id=key,
            sources=[],
            execution_mode="none",
            steps=[],
            validation_checks=[],
            response=intent.clarification,
            deferred_objectives=deferred,
        )
    if not deferred:
        profile = get_domain_profile(key, tenant_id=tenant_key)
        recipe = match_sql_recipe(question, profile.sql_recipes)
        if recipe is not None:
            spec = profile.sql_recipes[recipe.recipe_id]
            return ExecutionPlan(
                tenant_id=tenant_key,
                workspace_id=key,
                sources=spec.sources,
                execution_mode="sql",
                execution_reason=spec.reason,
                sql_recipe=recipe,
                steps=[
                    ExecutionStep(
                        step_id="schema",
                        action="schema",
                        instruction="Inspect the scoped source schemas required by the configured query.",
                        expected_output="Registered column names and types",
                    ),
                    ExecutionStep(
                        step_id="sql",
                        action="sql",
                        instruction="Execute configured SQL recipe "
                        + recipe.recipe_id
                        + " using published numeric parameters, without rewriting its formulas.",
                        expected_output=spec.expected_output,
                    ),
                ],
                validation_checks=spec.validation_checks,
            )
    diagnostics = PlannerDiagnostics(
        SemanticCatalog(
            workspace.sources,
            get_domain_profile(key, tenant_id=tenant_key),
            workspace.semantic_planner,
        )
    )
    agent = Agent(
        model,
        output_type=planner_output_type(
            intent.analysis_kind,
            SemanticCatalog(
                workspace.sources,
                get_domain_profile(key, tenant_id=tenant_key),
                workspace.semantic_planner,
            ),
            get_domain_profile(key, tenant_id=tenant_key).observation_policies(
                question
            ),
        ),
        capabilities=[diagnostics, ProcessHistory(diagnostics.handoff)],
        instructions=(
            PLANNER_INSTRUCTIONS
            + "\nWorkspace domain guidance:\n"
            + domain_context(key, tenant_id=tenant_key, stage="planner")
            + "\nWorkspace catalog:\n"
            + routing_catalog(key, tenant_id=tenant_key)
            + "\nConfigured observation policies for this request:\n"
            + json.dumps(
                {
                    name: policy.model_dump()
                    for name, policy in get_domain_profile(key, tenant_id=tenant_key)
                    .observation_policies(question)
                    .items()
                }
            )
            + "\nAvailable scientific capabilities:\n"
            + json.dumps(
                scientific_catalog(include_limits=True, include_external=False)
            )
        ),
        retries=3,
        model_settings=model_settings(),
    )

    @agent.output_validator
    def enforce_workspace(ctx: RunContext, plan: ExecutionPlan) -> ExecutionPlan:
        # A successful schema validation supersedes its old correction. Any
        # semantic validator rejection below must become the current feedback.
        diagnostics.latest_feedback = None
        for step in plan.steps:
            if step.logical_query is not None and step.logical_query._normalizations:
                trace(
                    "QUERY_IR_NORMALIZED: "
                    + json.dumps(
                        {
                            "step_id": step.step_id,
                            "changes": step.logical_query._normalizations,
                        }
                    )
                )
            if step.action == "python" and intent.analysis_kind:
                if (
                    not step.calculation
                    or step.calculation.kind != intent.analysis_kind
                    or not step.result_contract
                ):
                    raise ModelRetry(
                        RepairFeedback(
                            target="planner",
                            phase="method_selection",
                            step_id=step.step_id,
                            diagnostic=f"Expected analysis_kind={intent.analysis_kind!r}; received calculation={step.calculation!r}, result_contract present={step.result_contract is not None}. Declare the matching calculation and verifiable result_contract.",
                            authority={
                                "analysis_kind": intent.analysis_kind,
                                "scientific_operations": scientific_catalog()[
                                    "operations"
                                ],
                            },
                            artifact={"rejected_step": step.model_dump()},
                            retries_remaining=max(0, 3 - ctx.retry),
                            instructions=[
                                "Choose an algorithm for the requested business outcome. Similarity clustering does not balance group totals."
                            ],
                        ).publish()
                    )
        if (plan.tenant_id, plan.workspace_id) != (tenant_key, key):
            raise ModelRetry(f"Use only tenant {tenant_key}, workspace {key}")
        if plan.sql_recipe is not None:
            raise ModelRetry(
                "Set sql_recipe to null. Recipes are selected by the host from configuration before LLM planning; do not invent a recipe ID/hash. Return schema and generated SQL steps so the executor can validate and run the query."
            )
        for step in plan.steps:
            if step.action == "sql" and step.output_contract is None:
                raise ModelRetry(
                    f"SQL step {step.step_id} needs output_contract with required_columns, types, grain and applicable population checks before publication"
                )
            if step.action == "python" and not step.input_steps:
                raise ModelRetry(
                    f"Python step {step.step_id} needs input_steps referencing complete SQL datasets"
                )
            if step.output_contract is not None:
                _, workspace = get_workspace(key, tenant_id=tenant_key)
                try:
                    validate_contract_sources(
                        step.output_contract,
                        {name: workspace.sources[name] for name in plan.sources},
                    )
                except ValueError as error:
                    raise ModelRetry(
                        f"SQL step {step.step_id} output contract: {error}"
                    ) from error
                try:
                    catalog = SemanticCatalog(
                        {name: workspace.sources[name] for name in plan.sources},
                        get_domain_profile(key, tenant_id=tenant_key),
                        workspace.semantic_planner,
                    )
                    compiled_query = compile_query(
                        step.logical_query,
                        catalog,
                        population=step.output_contract.population,
                    )
                    step.logical_query = type(step.logical_query).model_validate(
                        compiled_query.logical_plan
                    )
                    aliases = {
                        item.name
                        for item in [
                            *step.logical_query.dimensions,
                            *step.logical_query.metrics,
                            *step.logical_query.calculations,
                        ]
                    }
                    missing = set(step.output_contract.required_columns) - aliases
                    if missing:
                        raise SemanticQueryError(
                            "MISSING_CONTRACT_OUTPUTS",
                            "Logical query must project every required output alias. "
                            "Preserve each requested identity dimension; do not rename one "
                            "grain column to stand in for a composite key. For a derived "
                            "identifier add an explicit calculation (such as concat of "
                            "the grain fields), or use the existing grain columns as "
                            "the observation identity and remove the unused synthetic key "
                            "from the output contract.",
                            missing=sorted(missing),
                        )
                except SemanticQueryError as error:
                    feedback = RepairFeedback(
                        target="planner",
                        phase="semantic_query",
                        step_id=step.step_id,
                        diagnostic=str(error),
                        artifact={"rejected_query": step.logical_query.model_dump()},
                        authority={
                            "output_contract": step.output_contract.model_dump(),
                            "compiler_diagnostic": error.diagnostic,
                            "selected_metrics": {
                                item.metric: catalog.metrics[item.metric].model_dump()
                                for item in step.logical_query.metrics
                                if item.metric in catalog.metrics
                            },
                        },
                        retries_remaining=max(0, 3 - ctx.retry),
                        instructions=[
                            "Correct the logical QueryIR using catalog metric definitions and valid relationship paths; do not generate SQL."
                        ],
                    )
                    diagnostics.latest_feedback = feedback
                    trace(
                        "PLAN_REJECTED: "
                        + json.dumps(
                            {
                                "attempt": ctx.retry + 1,
                                "error_type": "SemanticQueryError",
                                "step_id": step.step_id,
                                "diagnostic": error.diagnostic,
                                "rejected_query": step.logical_query.model_dump(),
                                "output_contract": step.output_contract.model_dump(),
                            }
                        )
                    )
                    raise ModelRetry(feedback.publish()) from error
        profile = get_domain_profile(key, tenant_id=tenant_key)
        if any(
            re.search(pattern, question, re.IGNORECASE)
            for pattern in profile.python_required_patterns
        ):
            if plan.execution_mode != "sql_python":
                raise ModelRetry(
                    "Workspace analysis guidance requires SQL preparation followed by Python for this question. Publish a sql_python plan with its specific algorithm, complete SQL inputs and python_reason."
                )
        # Clarifications already identified by classification cannot be dropped
        # by the planner or silently replaced with invented decision rules.
        pending = {
            item.objective.strip().casefold(): item for item in plan.deferred_objectives
        }
        for item in deferred:
            pending[item.objective.strip().casefold()] = item
        plan.deferred_objectives = list(pending.values())
        return consolidate_python_steps(plan)

    prompt = json.dumps(
        {
            "question": question,
            "classified_intent": intent.model_dump(),
            "applicable_workspace_guidance": get_domain_profile(
                key, tenant_id=tenant_key
            ).focused_guidance(question, "planner"),
        }
    )
    return (await agent.run(prompt, usage_limits=UsageLimits(request_limit=4))).output
