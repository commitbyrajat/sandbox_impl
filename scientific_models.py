"""Typed, domain-independent requests and resource limits for scientific calls."""

import json
import os
from decimal import Decimal
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, TypeAdapter

ROOT = Path(__file__).resolve().parent
Number = int | FiniteFloat | Decimal
Seed = Annotated[int, Field(strict=True, ge=0, le=2**64 - 1)]
Positive = Annotated[int, Field(strict=True, gt=0)]
Statistic = Literal["mean", "median", "sum", "std"]
Distribution = Literal["normal", "uniform", "binomial", "bernoulli"]


class ScientificSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    timeout_seconds: float = Field(default=30, gt=0, le=300)
    total_seconds: float = Field(default=120, gt=0, le=3600)
    memory_mb: int = Field(default=512, ge=64, le=65536)
    max_cells: int = Field(default=250_000, gt=0)
    max_work: int = Field(default=20_000_000, gt=0)
    max_observations: int = Field(default=10_000, gt=0)
    max_resamples: int = Field(default=10_000, gt=0)
    max_iterations: int = Field(default=500, gt=0)
    max_output_bytes: int = Field(default=4_000_000, gt=0)
    max_suspensions: int = Field(default=64, ge=1, le=4096)

    @classmethod
    def from_environment(cls):
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
        names = {
            "timeout_seconds": "SCIENTIFIC_TIMEOUT_SECONDS",
            "total_seconds": "SCIENTIFIC_TOTAL_SECONDS",
            "memory_mb": "SCIENTIFIC_MAX_MEMORY_MB",
            "max_cells": "SCIENTIFIC_MAX_CELLS",
            "max_work": "SCIENTIFIC_MAX_WORK",
            "max_observations": "SCIENTIFIC_MAX_OBSERVATIONS",
            "max_resamples": "SCIENTIFIC_MAX_RESAMPLES",
            "max_iterations": "SCIENTIFIC_MAX_ITERATIONS",
            "max_output_bytes": "SCIENTIFIC_MAX_OUTPUT_BYTES",
            "max_suspensions": "MONTY_MAX_SUSPENSIONS",
        }
        return cls.model_validate(
            {key: os.environ[name] for key, name in names.items() if name in os.environ}
        )


class Request(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    seed: Seed = 42
    stream: Seed = 0


class KMeansRequest(Request):
    operation: Literal["kmeans"] = "kmeans"
    observations: list[list[Number]]
    clusters: Positive = 3
    max_iterations: Positive = 50
    standardize: bool = True
    initialization: Literal["farthest_first", "kmeans++"] = "farthest_first"


class BootstrapRequest(Request):
    operation: Literal["bootstrap"] = "bootstrap"
    values: list[Number]
    statistic: Statistic = "mean"
    n_resamples: Positive = 1000
    confidence_level: Annotated[FiniteFloat, Field(gt=0, lt=1)] = 0.95
    method: Literal["percentile", "basic", "BCa"] = "percentile"
    quantile_method: Literal["nearest_rank", "linear"] = "nearest_rank"


class SampleRequest(Request):
    operation: Literal["sample"] = "sample"
    distribution: Distribution
    parameters: dict[str, Number | list[Number]]
    size: Positive | list[Positive] = 1000


class BinomialRequest(Request):
    operation: Literal["binomial"] = "binomial"
    n: Annotated[int, Field(strict=True, ge=0)]
    p: Annotated[FiniteFloat, Field(ge=0, le=1)]
    size: Positive = 1000


class BernoulliRequest(Request):
    operation: Literal["bernoulli"] = "bernoulli"
    p: Annotated[FiniteFloat, Field(ge=0, le=1)] = 0.5
    size: Positive = 1000


class ResampleRequest(Request):
    operation: Literal["resample_indices"] = "resample_indices"
    n_observations: Positive
    n_resamples: Positive = 1000


class MonteCarloTestRequest(Request):
    operation: Literal["monte_carlo_test"] = "monte_carlo_test"
    values: list[Number]
    distribution: Distribution = "normal"
    parameters: dict[str, Number | list[Number]] = Field(default_factory=dict)
    statistic: Statistic = "mean"
    n_resamples: Positive = 1000
    alternative: Literal["two-sided", "less", "greater"] = "two-sided"


ScientificRequest = Annotated[
    KMeansRequest
    | BootstrapRequest
    | SampleRequest
    | BinomialRequest
    | BernoulliRequest
    | ResampleRequest
    | MonteCarloTestRequest,
    Field(discriminator="operation"),
]
REQUEST_ADAPTER = TypeAdapter(ScientificRequest)
REQUEST_TYPES = {
    model.model_fields["operation"].default: model
    for model in (
        KMeansRequest,
        BootstrapRequest,
        SampleRequest,
        BinomialRequest,
        BernoulliRequest,
        ResampleRequest,
        MonteCarloTestRequest,
    )
}


def scientific_catalog(*, include_limits=False) -> dict:
    document = json.loads((ROOT / "scientific_capabilities.json").read_text())
    if set(document["operations"]) != set(REQUEST_TYPES):
        raise ValueError(
            "Scientific capability registry does not match the implemented allowlist"
        )
    if include_limits:
        document["effective_limits"] = (
            ScientificSettings.from_environment().model_dump()
        )
    return document


def scientific_stubs() -> str:
    return (
        "from typing import Any\nfrom decimal import Decimal\nNumber = int | float | Decimal\n"
        + "\n".join(
            "async def " + entry["signature"] + ": ..."
            for entry in scientific_catalog()["operations"].values()
        )
    )
