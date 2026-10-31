"""Declarative method and calculation invariants; never execute model expressions."""

import math
import re
from collections import Counter
from decimal import Decimal, InvalidOperation, localcontext
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

AnalysisKind = Literal[
    "clustering", "balancing", "uncertainty", "simulation", "hypothesis_test", "custom"
]


class CalculationMethod(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: AnalysisKind
    method: str = Field(min_length=1)


def normalize_output_path(path: str) -> str:
    if path == "$":
        return path
    path = path.removeprefix("$.").replace("[*]", ".*")
    if not re.fullmatch(
        r"(?:[A-Za-z_][A-Za-z_0-9]*|\*)(?:\.(?:[A-Za-z_][A-Za-z_0-9]*|\*))*", path
    ):
        raise ValueError(
            "Use a literal dot path with '*' wildcards, e.g. groups.*.total or $.groups[*].total; expressions and filters are unavailable"
        )
    return path


class ResultInvariant(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["sum_equals_input", "members_equal_input", "count_equals"]
    output_path: str = Field(min_length=1)
    input_step: str | None = None
    input_column: str | None = None
    expected_count: int | None = Field(default=None, ge=0)
    tolerance: Decimal = Field(default=Decimal("0.01"), ge=0, allow_inf_nan=False)

    @field_validator("output_path")
    @classmethod
    def path_syntax(cls, value):
        return normalize_output_path(value)

    @model_validator(mode="after")
    def required_arguments(self):
        if self.kind == "count_equals":
            if self.expected_count is None:
                raise ValueError("count_equals requires expected_count")
        elif not self.input_step or not self.input_column:
            raise ValueError("Input invariants require input_step and input_column")
        return self


class CountInvariant(ResultInvariant):
    kind: Literal["count_equals"]
    expected_count: int = Field(ge=0)


class SumInvariant(ResultInvariant):
    kind: Literal["sum_equals_input"]
    input_step: str = Field(min_length=1)
    input_column: str = Field(min_length=1)


class MembersInvariant(ResultInvariant):
    kind: Literal["members_equal_input"]
    input_step: str = Field(min_length=1)
    input_column: str = Field(min_length=1)


class PythonResultContract(BaseModel):
    model_config = ConfigDict(extra="forbid")
    required_fields: dict[
        str, Literal["numeric", "string", "list", "object", "boolean", "any"]
    ] = Field(default_factory=dict)
    invariants: list[
        Annotated[
            CountInvariant | SumInvariant | MembersInvariant,
            Field(discriminator="kind"),
        ]
    ] = Field(default_factory=list)

    @field_validator("required_fields")
    @classmethod
    def field_paths(cls, fields):
        return {normalize_output_path(path): kind for path, kind in fields.items()}


class CalculationValidationError(ValueError):
    pass


def select_values(value, path):
    """Dot paths with '*' over list items/dict values; no eval or arbitrary callbacks."""
    if path == "$":
        return [value]
    values = [value]
    for part in normalize_output_path(path).split("."):
        next_values = []
        for item in values:
            if part == "*" and isinstance(item, (list, tuple, dict)):
                next_values.extend(item.values() if isinstance(item, dict) else item)
            elif isinstance(item, dict) and part in item:
                next_values.append(item[part])
            else:
                raise CalculationValidationError(
                    f"Output path {path!r} is absent or has the wrong structure"
                )
        values = next_values
    return values


def decimal_number(value):
    try:
        if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
            raise ValueError()
        result = Decimal(str(value))
        if not result.is_finite():
            raise ValueError()
        return result
    except (ValueError, InvalidOperation):
        raise CalculationValidationError(
            f"Expected a finite numeric value, received {value!r}"
        ) from None


def validate_python_result(value, contract=None, inputs=None, producers=None):
    """Validate before a Python step is committed; failures keep its SQL evidence."""

    def finite(item):
        if (isinstance(item, Decimal) and not item.is_finite()) or (
            isinstance(item, float) and not math.isfinite(item)
        ):
            raise CalculationValidationError(
                "Python output contains a non-finite numerical result"
            )
        if isinstance(item, dict):
            for entry in item.values():
                finite(entry)
        elif isinstance(item, (list, tuple)):
            for entry in item:
                finite(entry)

    finite(value)
    if contract is None:
        return
    checks = {
        "numeric": lambda item: decimal_number(item) is not None,
        "string": lambda item: isinstance(item, str),
        "list": lambda item: isinstance(item, (list, tuple)),
        "object": lambda item: isinstance(item, dict),
        "boolean": lambda item: isinstance(item, bool),
        "any": lambda _: True,
    }
    for path, expected in contract.required_fields.items():
        for item in select_values(value, path):
            if not checks[expected](item):
                raise CalculationValidationError(f"Output {path!r} must be {expected}")
    for rule in contract.invariants:
        observed = select_values(value, rule.output_path)
        if rule.kind == "count_equals":
            count = len(observed)
            if (
                "*" not in rule.output_path.split(".")
                and len(observed) == 1
                and isinstance(observed[0], (list, tuple, dict))
            ):
                count = len(observed[0])
            if count != rule.expected_count:
                raise CalculationValidationError(
                    f"{rule.output_path}: expected {rule.expected_count} entries, got {count}"
                )
            continue
        result_id = (producers or {}).get(rule.input_step)
        if result_id not in (inputs or {}):
            raise CalculationValidationError(
                f"Invariant requires current SQL producer {rule.input_step!r}"
            )
        try:
            source = [row[rule.input_column] for row in inputs[result_id]]
        except KeyError:
            raise CalculationValidationError(
                f"Invariant input column {rule.input_column!r} is unavailable"
            ) from None
        if rule.kind == "sum_equals_input":
            with localcontext() as context:
                context.prec = 76
                difference = abs(
                    sum(map(decimal_number, observed), Decimal(0))
                    - sum(map(decimal_number, source), Decimal(0))
                )
            if difference > rule.tolerance:
                raise CalculationValidationError(
                    f"{rule.output_path}: totals do not reconcile with {rule.input_step}.{rule.input_column}; difference={difference}"
                )
        else:
            try:
                if Counter(observed) != Counter(source):
                    raise CalculationValidationError(
                        f"{rule.output_path}: members are missing, duplicated or invented relative to the SQL population"
                    )
            except TypeError:
                raise CalculationValidationError(
                    "Member identity paths must select scalar IDs"
                ) from None


def validate_execution_result(plan, execution):
    if plan.execution_mode == "none":
        return
    if execution is None or execution.completed_steps != [
        step.step_id for step in plan.steps
    ]:
        raise CalculationValidationError(
            "Result validation requires every published step to complete in order"
        )
    if not execution.queries or not execution.sql_results:
        raise CalculationValidationError(
            "Result validation requires stored SQL evidence"
        )
    for step in plan.steps:
        if step.action == "python":
            runs = [
                run for run in execution.python_runs if run["step_id"] == step.step_id
            ]
            if not runs or runs[-1].get("output") is None:
                raise CalculationValidationError(
                    f"Missing calculation evidence for Python step {step.step_id}"
                )
            validate_python_result(runs[-1]["output"])
            if step.result_contract:
                inputs = {
                    key: result["rows"] for key, result in execution.sql_results.items()
                }
                producer_ids = runs[-1].get("input_producers", {})
                validate_python_result(
                    runs[-1]["output"], step.result_contract, inputs, producer_ids
                )
