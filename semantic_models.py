"""Domain-independent query IR and machine-readable semantic catalog types."""

from __future__ import annotations

import math
from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator
from pydantic_core import PydanticCustomError


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SemanticPlannerConfig(StrictModel):
    backend: Literal["catalog", "metricflow"] = "catalog"
    # Compatibility is explicit and audited; a compiler error is never silently retried elsewhere.
    unsupported: Literal["error", "catalog"] = "error"
    time_spine_start: str = "2026-01-01"
    time_spine_end: str = "2027-01-01"

    @model_validator(mode="after")
    def valid_time_spine(self):
        try:
            days = (
                date.fromisoformat(self.time_spine_end)
                - date.fromisoformat(self.time_spine_start)
            ).days
        except ValueError:
            raise ValueError("Time spine boundaries must be ISO dates") from None
        if not 0 < days <= 730:
            raise ValueError("Time spine must contain between 1 and 730 days")
        return self


class CumulativeDefinition(StrictModel):
    input_metric: str
    period_aggregation: Literal["first", "last", "average"] = "last"
    window_count: int | None = Field(default=None, ge=1)
    window_grain: Literal["day", "week", "month", "quarter", "year"] | None = None
    grain_to_date: Literal["month", "quarter", "year"] | None = None

    @model_validator(mode="after")
    def valid_window(self):
        if (self.window_count is None) != (self.window_grain is None):
            raise ValueError("Cumulative window requires count and grain together")
        if self.window_count and self.grain_to_date:
            raise ValueError("Choose a rolling window or grain_to_date")
        return self


class MeasurePolicy(StrictModel):
    aggregation: Literal["sum", "min", "max", "avg", "count_distinct", "constant"]
    additive_over: list[str] = Field(default_factory=list)
    non_additive_over: list[str] = Field(default_factory=list)
    consistency: Literal["none", "constant_per_preaggregation_grain"] = "none"
    null_policy: Literal["allow", "reject"] = "allow"
    description: str = ""


class GrainReconciliation(StrictModel):
    group_by: list[str] = Field(min_length=1)
    sum_column: str
    constant_column: str
    tolerance: float = Field(default=0, ge=0, allow_inf_nan=False)


class TableSemantics(StrictModel):
    grain: list[str] = Field(min_length=1)
    primary_key: list[str] = Field(min_length=1)
    snapshot_column: str | None = None
    measures: dict[str, MeasurePolicy] = Field(default_factory=dict)
    reconciliations: list[GrainReconciliation] = Field(default_factory=list)


class FieldRef(StrictModel):
    table: str
    column: str


class Predicate(StrictModel):
    field: FieldRef
    operator: Literal["eq", "ne", "lt", "le", "gt", "ge", "in", "is_null", "not_null"]
    value: str | int | float | bool | list[str | int | float | bool] | None = None


class Formula(StrictModel):
    """Names reference metrics in catalog formulas, output aliases in calculations."""

    op: Literal[
        "ref",
        "literal",
        "add",
        "subtract",
        "multiply",
        "divide",
        "negate",
        "coalesce",
        "concat",
        "key",
    ]
    name: str | None = None
    value: str | int | float | bool | None = None
    args: list[Formula] = Field(default_factory=list, max_length=8)

    @model_validator(mode="after")
    def valid_shape(self):
        if self.op == "ref":
            if not self.name or self.args or self.value is not None:
                raise ValueError("ref requires only a name")
        elif self.op == "literal":
            if (
                self.name
                or self.args
                or isinstance(self.value, float)
                and not math.isfinite(self.value)
            ):
                raise ValueError("literal requires a finite scalar value")
        elif self.op in ("concat", "key"):
            if self.name or self.value is not None or not 2 <= len(self.args) <= 8:
                raise ValueError(f"{self.op} requires two to eight operands")
        elif (
            self.name
            or self.value is not None
            or len(self.args) != (1 if self.op == "negate" else 2)
        ):
            raise ValueError("Arithmetic requires exactly its declared operands")
        return self


class MetricDefinition(StrictModel):
    description: str
    table: str | None = None
    column: str | None = None
    # None means inherit the column's catalog aggregation; never supplied by LLM.
    aggregation: (
        Literal["sum", "min", "max", "avg", "count_distinct", "constant", "count"]
        | None
    ) = None
    filters: list[Predicate] = Field(default_factory=list)
    required_filters: list[Predicate] = Field(default_factory=list)
    formula: Formula | None = None
    cumulative: CumulativeDefinition | None = None
    horizon: str | None = None
    minimum: float | None = None
    maximum: float | None = None

    @model_validator(mode="after")
    def valid_definition(self):
        if (
            self.minimum is not None
            and self.maximum is not None
            and self.minimum > self.maximum
        ):
            raise ValueError("Metric minimum must not exceed maximum")
        if self.cumulative is not None:
            if (
                self.formula
                or self.table
                or self.column
                or self.aggregation
                or self.filters
            ):
                raise ValueError(
                    "Cumulative metrics reference an input metric, without a leaf or formula"
                )
        elif self.formula is not None:
            if self.table or self.column or self.aggregation or self.filters:
                raise ValueError("Derived metrics use only a catalog formula")
        elif not self.table or (not self.column and self.aggregation != "count"):
            raise ValueError("A base metric requires a table and column (or count)")
        return self


class Dimension(StrictModel):
    name: str = Field(
        description="Unique output alias for this grouping/source field. Do not also select it as a metric or calculation."
    )
    field: FieldRef
    transform: Literal["identity", "day", "month", "year"] = "identity"


class MetricSelection(StrictModel):
    name: str = Field(
        description="Output column alias, unique across all projections; use the alias from output_contract.required_columns."
    )
    metric: str = Field(
        description="Existing catalog metric ID. This is independent of the output alias."
    )


class Calculation(StrictModel):
    name: str = Field(
        description="A new output alias, different from every dimension, metric and earlier calculation."
    )
    expression: Formula


class TimeSelection(StrictModel):
    selection: Literal["all", "latest", "latest_common", "range"] = "all"
    # Latest-common means the maximum intersection, not the minimum of maxima.
    tables: list[str] = Field(default_factory=list)
    start: str | None = Field(
        default=None, description="Inclusive ISO date/time boundary."
    )
    end: str | None = Field(
        default=None,
        description="Exclusive ISO date/time boundary. To include a full month, use the first day of the following month.",
    )


class OrderBy(StrictModel):
    name: str
    descending: bool = False


class QueryIR(StrictModel):
    """Business selections, never SQL, join keys or aggregate function choices."""

    base_table: str
    dimensions: list[Dimension] = Field(
        default_factory=list,
        max_length=20,
        description="Selected source/grouping columns, including observation identities. Select each output alias once.",
    )
    metrics: list[MetricSelection] = Field(
        default_factory=list,
        max_length=30,
        description="Selected catalog aggregates. At least one dimension or metric is required; never reuse a dimension's alias.",
    )
    filters: list[Predicate] = Field(default_factory=list, max_length=30)
    time: TimeSelection = Field(default_factory=TimeSelection)
    calculations: list[Calculation] = Field(default_factory=list, max_length=30)
    order_by: list[OrderBy] = Field(default_factory=list, max_length=10)
    # Optional relationship hints disambiguate routes; the catalog supplies
    # missing edges and every join key. They do not replace the catalog graph.
    relationships: list[str] = Field(default_factory=list, max_length=15)
    metric_horizons: dict[str, str] = Field(
        default_factory=dict,
        description="Optional explicit horizon assertions. Fixed metric horizons are inherited from the catalog when omitted; contradictory assertions are rejected.",
    )
    _normalizations: list[str] = PrivateAttr(default_factory=list)

    @model_validator(mode="after")
    def valid_outputs(self):
        # A repeated identical projection and an identity reassignment do not
        # change the query's meaning. Never rename or merge different expressions.
        for section in ("dimensions", "metrics", "calculations"):
            retained = []
            for index, item in enumerate(getattr(self, section)):
                if any(item == prior for prior in retained):
                    self._normalizations.append(
                        f"Removed identical {section}[{index}] alias {item.name!r}"
                    )
                else:
                    retained.append(item)
            setattr(self, section, retained)
        selected = {item.name for item in [*self.dimensions, *self.metrics]}
        calculations = []
        for index, item in enumerate(self.calculations):
            if (
                item.name in selected
                and item.expression.op == "ref"
                and item.expression.name == item.name
            ):
                self._normalizations.append(
                    f"Removed identity calculations[{index}] alias {item.name!r}"
                )
                continue
            calculations.append(item)
            selected.add(item.name)
        self.calculations = calculations
        if not (self.dimensions or self.metrics):
            raise PydanticCustomError(
                "query_outputs_empty",
                "logical_query has no selected outputs: dimensions and metrics are both empty. "
                "Select source identity fields in dimensions and catalog metric IDs in metrics, "
                "using the output_contract.required_columns as output aliases. Calculations alone cannot select source observations.",
            )
        occurrences = {}
        for section in ("dimensions", "metrics", "calculations"):
            for index, item in enumerate(getattr(self, section)):
                occurrences.setdefault(item.name, []).append(f"{section}[{index}]")
        collisions = {
            name: paths for name, paths in occurrences.items() if len(paths) > 1
        }
        if collisions:
            raise PydanticCustomError(
                "query_alias_collision",
                "Output aliases must be unique across dimensions, metrics and calculations. "
                "Conflicts: {conflicts}. Keep each output_contract alias exactly once; remove "
                "an identical duplicate or give different expressions distinct aliases and update references.",
                {
                    "conflicts": "; ".join(
                        f"{name!r}: {', '.join(paths)}"
                        for name, paths in collisions.items()
                    )
                },
            )
        names = list(occurrences)
        if set(x.name for x in self.order_by) - set(names):
            raise ValueError("Order only by selected output aliases")
        return self
