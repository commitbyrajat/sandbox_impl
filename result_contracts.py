"""Domain-agnostic SQL output contracts, validated before accepting evidence."""

from typing import Literal

import pyarrow as pa
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlglot import exp


class ResultColumn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal[
        "any", "string", "numeric", "integer", "boolean", "date", "timestamp"
    ] = "any"
    nullable: bool = True
    meaning: str = ""


class PopulationContract(BaseModel):
    model_config = ConfigDict(extra="forbid")
    table: str
    source_key: str
    result_key: str


class SQLResultContract(BaseModel):
    model_config = ConfigDict(extra="forbid")
    required_columns: dict[str, ResultColumn] = Field(min_length=1)
    grain: list[str] = Field(default_factory=list)
    min_rows: int = Field(default=0, ge=0)
    max_rows: int | None = Field(default=None, ge=0)
    population: PopulationContract | None = None

    @model_validator(mode="after")
    def valid_keys(self):
        if set(self.grain) - self.required_columns.keys():
            raise ValueError("Every grain key must be a required output column")
        if self.max_rows is not None and self.max_rows < self.min_rows:
            raise ValueError("max_rows must be at least min_rows")
        if self.population and self.population.result_key not in self.grain:
            raise ValueError(
                f"Population result_key={self.population.result_key!r} must be an observation identity from grain={self.grain!r}. Set result_key to one of those grain fields, not a measure or its alias."
            )
        return self


class ResultContractError(ValueError):
    pass


def validate_contract_sources(contract, sources):
    """Reject unavailable population keys and mismatched known entity identities."""
    population = contract.population
    if population is None:
        return
    columns = {
        table: fields
        for source in sources.values()
        for table, fields in source.columns.items()
        if table in source.tables
    }
    if (
        population.table not in columns
        or population.source_key not in columns[population.table]
    ):
        raise ValueError(
            "Population table/source_key must exist in the selected datasource.json columns"
        )
    known_targets = {
        (table, population.result_key)
        for table, fields in columns.items()
        if population.result_key in fields
    }
    if not known_targets or population.source_key == population.result_key:
        return  # An explicitly aliased output is checked against actual keys at runtime.
    graph = {}
    for source in sources.values():
        for relation in source.relationship_keys.values():
            for left, right in relation.column_pairs:
                a, b = (relation.left_table, left), (relation.right_table, right)
                graph.setdefault(a, set()).add(b)
                graph.setdefault(b, set()).add(a)
    pending = [(population.table, population.source_key)]
    visited = set()
    while pending:
        current = pending.pop()
        if current in known_targets:
            return
        if current not in visited:
            visited.add(current)
            pending.extend(graph.get(current, set()) - visited)
    raise ValueError(
        f"Population key {population.table}.{population.source_key} does not represent the configured identity {population.result_key}. Use a registry population for that observation identity, e.g. {sorted(f'{table}.{key}' for table, key in known_targets)}. Do not compare different entity IDs."
    )


def type_matches(dtype: pa.DataType, expected: str) -> bool:
    if pa.types.is_dictionary(dtype):
        dtype = dtype.value_type
    checks = {
        "any": lambda _: True,
        "string": lambda t: pa.types.is_string(t)
        or pa.types.is_large_string(t)
        or pa.types.is_string_view(t),
        "numeric": lambda t: pa.types.is_integer(t)
        or pa.types.is_floating(t)
        or pa.types.is_decimal(t),
        "integer": pa.types.is_integer,
        "boolean": pa.types.is_boolean,
        "date": pa.types.is_date,
        "timestamp": pa.types.is_timestamp,
    }
    return checks[expected](dtype)


def validate_result_contract(contract, result, schema, context, owned_schemas):
    """Validate complete rows, aliases, Arrow types, key uniqueness and population."""
    if contract is None:  # Existing manually constructed/previously published plans.
        return
    problems = []
    missing = set(contract.required_columns) - set(result["columns"])
    if missing:
        problems.append(
            "Missing required output columns: " + ", ".join(sorted(missing))
        )
    rows = result["rows"]
    if result["truncated"]:
        problems.append(
            "Contract checks require complete rows; the result is truncated"
        )
    if len(rows) < contract.min_rows or (
        contract.max_rows is not None and len(rows) > contract.max_rows
    ):
        problems.append(
            f"Row count {len(rows)} is outside [{contract.min_rows}, {contract.max_rows}]"
        )
    for name, spec in contract.required_columns.items():
        if name in missing:
            continue
        if not type_matches(schema.field(name).type, spec.type):
            problems.append(
                f"Column {name} must be {spec.type}; received {schema.field(name).type}"
            )
        if not spec.nullable and any(row[name] is None for row in rows):
            problems.append(f"Column {name} contains unexpected nulls")
    if contract.grain and not (set(contract.grain) & missing):
        keys = [tuple(row[key] for key in contract.grain) for row in rows]
        if any(any(value is None for value in key) for key in keys):
            problems.append("Observation grain contains null keys")
        try:
            if len(keys) != len(set(keys)):
                problems.append(
                    "Duplicate observation keys at required grain: "
                    + ", ".join(contract.grain)
                )
        except TypeError:
            problems.append("Observation keys must be scalar values")
    population = contract.population
    if population:
        if (
            population.table not in owned_schemas
            or population.source_key not in owned_schemas[population.table].names
        ):
            raise ResultContractError(
                "Population contract references an unavailable workspace table/key"
            )
        if population.result_key not in missing:
            # Identifiers come from the owned registry, and are quoted with SQLGlot.
            query = (
                exp.select(exp.column(population.source_key, quoted=True))
                .distinct()
                .from_(exp.to_table(population.table, quoted=True))
            )
            expected = {
                row[population.source_key]
                for row in context.sql(query.sql(dialect="postgres")).to_pylist()
            }
            observed = {row[population.result_key] for row in rows}
            if expected != observed:
                problems.append(
                    f"Population mismatch: expected {len(expected)} distinct keys from {population.table}.{population.source_key}; got {len(observed)} (missing {len(expected-observed)}, unexpected {len(observed-expected)})"
                )
    if problems:
        raise ResultContractError("; ".join(problems))
