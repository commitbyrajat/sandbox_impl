"""Bounded library requests and the public, async Monty module contracts."""

import math
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue


class LibraryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    seed: int = Field(default=42, strict=True, ge=0, le=2**64 - 1)
    stream: int = Field(default=0, strict=True, ge=0, le=2**64 - 1)


class NumpyRequest(LibraryRequest):
    operation: Literal["numpy"] = "numpy"
    function: Literal[
        "array",
        "asarray",
        "reshape",
        "sum",
        "mean",
        "median",
        "std",
        "var",
        "min",
        "max",
        "percentile",
        "quantile",
        "sort",
        "sqrt",
        "abs",
        "add",
        "subtract",
        "multiply",
        "divide",
        "dot",
    ]
    a: Any
    parameters: dict[str, JsonValue] = Field(default_factory=dict)


class NumpyRandomRequest(LibraryRequest):
    operation: Literal["numpy_random"] = "numpy_random"
    function: Literal["binomial", "normal", "uniform", "random"]
    parameters: dict[str, JsonValue] = Field(default_factory=dict)
    size: int | list[int] = 1000


class PandasRequest(LibraryRequest):
    operation: Literal["pandas"] = "pandas"
    function: Literal[
        "DataFrame", "groupby", "merge", "sort_values", "dropna", "to_numeric"
    ]
    data: JsonValue
    right: JsonValue = None
    columns: list[str] | None = None
    by: str | list[str] | None = None
    agg: dict[
        str, Literal["sum", "mean", "min", "max", "count", "nunique", "median", "std"]
    ] = Field(default_factory=dict)
    ascending: bool = True
    dropna: bool = False
    how: Literal["inner", "left", "right", "outer"] = "inner"
    merge_validation: Literal[
        "one_to_one", "one_to_many", "many_to_one", "many_to_many"
    ] = "one_to_one"
    errors: Literal["raise", "coerce"] = "raise"


def encode_scalars(value):
    if type(value).__module__.startswith("pandas") and type(value).__name__ in (
        "NAType",
        "NaTType",
    ):
        return None
    if isinstance(value, Decimal):
        return {"__external_scalar__": "decimal", "value": str(value)}
    if isinstance(value, datetime):
        return {"__external_scalar__": "datetime", "value": value.isoformat()}
    if isinstance(value, date):
        return {"__external_scalar__": "date", "value": value.isoformat()}
    if isinstance(value, dict):
        if set(value) == {"__external_scalar__", "value"}:
            return {
                "__external_scalar__": "mapping",
                "value": [[k, encode_scalars(v)] for k, v in value.items()],
            }
        return {key: encode_scalars(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode_scalars(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    # NumPy/Pandas scalar values are converted in the numerical worker only.
    if type(value).__module__.startswith("numpy") and hasattr(value, "item"):
        return encode_scalars(value.item())
    raise ValueError(f"Unsupported tabular scalar type: {type(value).__name__}")


def decode_scalars(value):
    if isinstance(value, dict):
        if set(value) == {"__external_scalar__", "value"}:
            if value["__external_scalar__"] == "mapping":
                return {key: decode_scalars(item) for key, item in value["value"]}
            constructors = {
                "decimal": Decimal,
                "date": date.fromisoformat,
                "datetime": datetime.fromisoformat,
            }
            if value["__external_scalar__"] not in constructors:
                raise ValueError("Unsupported external scalar tag")
            return constructors[value["__external_scalar__"]](value["value"])
        return {key: decode_scalars(item) for key, item in value.items()}
    if isinstance(value, list):
        return [decode_scalars(item) for item in value]
    return value


def module_catalog(*, compact=False):
    functions = {}

    def add(path, signature, description, **extra):
        functions[path] = {
            "signature": signature,
            "description": description,
            "analysis_kind": "custom",
            "compatible_kinds": [
                "custom",
                "balancing",
                "clustering",
                "uncertainty",
                "simulation",
                "hypothesis_test",
            ],
            **extra,
        }

    for name in ("array", "asarray"):
        add(
            f"numpy.{name}",
            f"{name}(a: Any, dtype: str = 'float64') -> list",
            "Convert finite numbers to float64/int64/bool; returns nested Python lists, never ndarray. Use function-style reshape and reductions.",
        )
    add(
        "numpy.reshape",
        "reshape(a: Any, shape: list[int]) -> list",
        "Return reshaped nested lists; supports one inferred -1 dimension.",
    )
    for name in ("sum", "mean", "median", "min", "max", "sort"):
        add(
            f"numpy.{name}",
            f"{name}(a: Any, axis: int | None = None) -> Any",
            "NumPy float64 operation; returns native scalar/list. Empty statistical inputs are rejected.",
        )
    for name in ("std", "var"):
        add(
            f"numpy.{name}",
            f"{name}(a: Any, axis: int | None = None, ddof: int = 0) -> Any",
            "NumPy float64 dispersion; ddof=0 is population and ddof=1 is sample.",
        )
    for name in ("quantile", "percentile"):
        add(
            f"numpy.{name}",
            f"{name}(a: Any, q: float | list[float], axis: int | None = None, method: str = 'linear') -> Any",
            "Use method='inverted_cdf' for nearest-rank bounds; q is 0..1 for quantile and 0..100 for percentile.",
        )
    for name in ("sqrt", "abs"):
        add(
            f"numpy.{name}",
            f"{name}(a: Any) -> Any",
            "Finite elementwise NumPy operation returning scalar/list.",
        )
    for name in ("add", "subtract", "multiply", "divide", "dot"):
        add(
            f"numpy.{name}",
            f"{name}(a: Any, b: Any) -> Any",
            "Bounded float64 numeric operation, with finite output required.",
        )
    for name, params in (
        ("binomial", "n: int | list[int], p: float | list[float]"),
        ("normal", "loc: float = 0.0, scale: float = 1.0"),
        ("uniform", "low: float = 0.0, high: float = 1.0"),
        ("random", ""),
    ):
        signature = f"{name}({params + ', ' if params else ''}*, size: int | list[int] = 1000, seed: int = 42, stream: int = 0) -> list"
        add(
            f"numpy.random.{name}",
            signature,
            "Native NumPy Generator(PCG64), SeedSequence([seed,stream]); pass vector p and size=[scenario_count,cohort_count] for batched defaults.",
        )
    add(
        "pandas.DataFrame",
        "DataFrame(data: Any, columns: list[str] | None = None) -> list[dict[str, Any]]",
        "Build a real pandas DataFrame in the worker, return records. This is a records facade, not a live DataFrame. Use pandas function-style operations below; Decimal/date cells are preserved.",
    )
    add(
        "pandas.groupby",
        "groupby(data: list[dict[str, Any]], by: str | list[str], agg: dict[str, str], dropna: bool = False) -> list[dict[str, Any]]",
        "Group records with pandas. Aggregations: sum/mean/min/max/count/nunique/median/std. Decimal sums stay exact; statistics may return float64.",
    )
    add(
        "pandas.merge",
        "merge(left: list[dict[str, Any]], right: list[dict[str, Any]], on: str | list[str], how: str = 'inner', validate: str = 'one_to_one') -> list[dict[str, Any]]",
        "Merge supplied records; validates declared cardinality; no SQL/file/network input.",
    )
    add(
        "pandas.sort_values",
        "sort_values(data: list[dict[str, Any]], by: str | list[str], ascending: bool = True) -> list[dict[str, Any]]",
        "Stable record sorting with pandas.",
    )
    add(
        "pandas.dropna",
        "dropna(data: list[dict[str, Any]], subset: list[str] | None = None) -> list[dict[str, Any]]",
        "Remove missing rows for the specified columns.",
    )
    add(
        "pandas.to_numeric",
        "to_numeric(values: list[Any], errors: str = 'raise') -> list",
        "Explicit approximate numeric conversion; missing/non-finite cells return None.",
    )
    for name, params in (
        ("bernoulli", "p: float | list[float]"),
        ("binom", "n: int | list[int], p: float | list[float]"),
        ("norm", "loc: float = 0.0, scale: float = 1.0"),
        ("uniform", "loc: float = 0.0, scale: float = 1.0"),
    ):
        add(
            f"scipy.stats.{name}.rvs",
            f"rvs({params}, *, size: int | list[int] = 1000, seed: int = 42, stream: int = 0) -> list",
            "SciPy random samples returned as plain nested lists. Metadata is retained in the scientific call audit.",
        )
    add(
        "scipy.stats.bootstrap",
        "bootstrap(values: list[Any], *, statistic: str = 'mean', n_resamples: int = 1000, confidence_level: float = 0.95, method: str = 'percentile', quantile_method: str = 'nearest_rank', seed: int = 42, stream: int = 0) -> dict[str, Any]",
        "Named-statistic wrapper; same lower/upper/estimate/method dictionary as scientific.bootstrap, not a native BootstrapResult.",
        returns={
            "estimate": "float",
            "lower": "float",
            "upper": "float",
            "standard_error": "float",
            "bootstrap_distribution": "list[float]",
            "method": "dict",
        },
    )
    add(
        "scipy.cluster.vq.kmeans2",
        "kmeans2(data: list[list[float]], k: int, *, iter: int = 50, seed: int = 42, stream: int = 0) -> tuple[list[list[float]], list[int]]",
        "Bounded seeded SciPy kmeans2 with kmeans++ initialization; returns (centroids,labels). Use scientific.kmeans for automatic standardization and detailed metadata.",
        analysis_kind="clustering",
        compatible_kinds=["clustering", "custom"],
    )
    for name in ("mean", "fmean", "median", "pstdev", "stdev", "pvariance", "variance"):
        add(
            f"statistics.{name}",
            f"{name}(data: list[Any]) -> float",
            "Async worker reduction; mean/dispersion use NumPy float64. Use original Monty Decimal sums for exact accounting.",
        )
    if compact:
        functions = {
            path: {
                key: value
                for key, value in spec.items()
                if key in ("signature", "returns", "analysis_kind", "compatible_kinds")
            }
            for path, spec in functions.items()
        }
    return {
        "modules": ["numpy", "pandas", "scipy", "statistics"],
        "contract": "These are bounded ASYNC external-module facades. Await every library function. Arrays and DataFrames return lists/records; no ndarray/DataFrame methods, generators, file/network IO or arbitrary callbacks. Only listed dotted function paths exist. NumPy/SciPy float64 statistics are approximate; original SQL Decimal values remain available in Monty.",
        "functions": functions,
    }
