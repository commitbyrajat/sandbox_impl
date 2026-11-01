"""Finite, bounded operations on real NumPy and pandas; no arbitrary dispatch."""

import math
from decimal import Decimal

from external_module_models import (
    NumpyRequest,
    NumpyRandomRequest,
    PandasRequest,
    decode_scalars,
    encode_scalars,
)


def execute_library(req, settings, np, pd, rng):
    def bound(value, maximum, label):
        if value > maximum:
            raise ValueError(f"{label} {value} exceeds limit {maximum}")

    def array(value):
        result = np.asarray(value, dtype=np.float64)
        bound(result.size, settings.max_cells, "Array cells")
        if result.ndim:
            bound(result.shape[0], settings.max_observations, "Array observations")
        if result.ndim > 3 or not np.isfinite(result).all():
            raise ValueError(
                "Arrays must be finite numeric values with at most three dimensions"
            )
        return result

    if isinstance(req, NumpyRandomRequest):
        shape = tuple(req.size) if isinstance(req.size, list) else (req.size,)
        if (
            not shape
            or len(shape) > 3
            or any(type(x) is not int or x < 1 for x in shape)
        ):
            raise ValueError(
                "Random sample dimensions must be positive integers (at most three)"
            )
        bound(math.prod(shape), settings.max_cells, "Sample cells")
        args = dict(req.parameters)
        permitted = {
            "binomial": {"n", "p"},
            "normal": {"loc", "scale"},
            "uniform": {"low", "high"},
            "random": set(),
        }[req.function]
        if set(args) - permitted:
            raise ValueError("Unsupported random parameters")
        args = {k: array(v) for k, v in args.items()}
        if req.function == "binomial":
            if (
                not {"n", "p"} <= set(args)
                or np.any(args["n"] < 0)
                or np.any(args["n"] != np.floor(args["n"]))
                or np.any(args["n"] > 2**53 - 1)
                or np.any((args["p"] < 0) | (args["p"] > 1))
            ):
                raise ValueError("Binomial requires integer n>=0 and p in [0,1]")
            result = rng.binomial(args["n"].astype(np.int64), args["p"], size=shape)
        elif req.function == "normal":
            if np.any(args.get("scale", 1) <= 0):
                raise ValueError("Normal scale must be positive")
            result = rng.normal(args.get("loc", 0), args.get("scale", 1), size=shape)
        elif req.function == "uniform":
            if np.any(args.get("high", 1) <= args.get("low", 0)):
                raise ValueError("Uniform high must exceed low")
            result = rng.uniform(args.get("low", 0), args.get("high", 1), size=shape)
        else:
            result = rng.random(shape)
        if not np.isfinite(result).all():
            raise ValueError("Random sampling returned non-finite values")
        return result.tolist(), {
            "library_api": "numpy.random.Generator." + req.function,
            "shape": list(shape),
        }

    if isinstance(req, NumpyRequest):
        if (
            req.function in ("array", "asarray")
            and req.parameters.get("dtype") == "int64"
        ):
            if set(req.parameters) - {"dtype"}:
                raise ValueError("Unsupported NumPy parameters")
            original = np.asarray(req.a, dtype=object)
            bound(original.size, settings.max_cells, "Array cells")
            if original.ndim > 3:
                raise ValueError("Arrays support at most three dimensions")
            if original.ndim:
                bound(
                    original.shape[0], settings.max_observations, "Array observations"
                )
            for item in original.flat:
                n = Decimal(int(item)) if isinstance(item, bool) else Decimal(str(item))
                if not n.is_finite() or n < -(2**63) or n > 2**63 - 1:
                    raise ValueError("int64 conversion would overflow")
            value = np.asarray(req.a, dtype=np.int64)
            return value.tolist(), {
                "library_api": "numpy." + req.function,
                "rng": None,
                "seed": None,
                "stream": None,
                "numeric_dtype": "int64",
            }
        a = array(req.a)
        parameters = dict(req.parameters)
        allowed = {
            "array": {"dtype"},
            "asarray": {"dtype"},
            "reshape": {"shape"},
            "sum": {"axis"},
            "mean": {"axis"},
            "median": {"axis"},
            "min": {"axis"},
            "max": {"axis"},
            "sort": {"axis"},
            "std": {"axis", "ddof"},
            "var": {"axis", "ddof"},
            "quantile": {"q", "axis", "method"},
            "percentile": {"q", "axis", "method"},
            "sqrt": set(),
            "abs": set(),
            "add": {"b"},
            "subtract": {"b"},
            "multiply": {"b"},
            "divide": {"b"},
            "dot": {"b"},
        }[req.function]
        if set(parameters) - allowed:
            raise ValueError("Unsupported NumPy parameters")
        if (
            "axis" in parameters
            and parameters["axis"] is not None
            and type(parameters["axis"]) is not int
        ):
            raise ValueError("axis must be an integer or None")
        if "ddof" in parameters and (
            type(parameters["ddof"]) is not int or parameters["ddof"] < 0
        ):
            raise ValueError("ddof must be a nonnegative integer")
        if not a.size and req.function not in (
            "array",
            "asarray",
            "reshape",
            "sum",
            "sort",
        ):
            raise ValueError("At least one observation is required for this statistic")
        if req.function in ("array", "asarray"):
            dtype = parameters.get("dtype", "float64")
            if dtype not in ("float64", "int64", "bool"):
                raise ValueError("Supported dtypes: float64, int64, bool")
            if dtype == "int64" and (np.any(a < -(2**63)) or np.any(a >= 2**63)):
                raise ValueError("int64 conversion would overflow")
            result = a.astype(dtype)
        elif req.function == "reshape":
            shape = parameters["shape"]
            if (
                not isinstance(shape, list)
                or not 1 <= len(shape) <= 3
                or any(type(x) is not int or x < -1 for x in shape)
                or shape.count(-1) > 1
            ):
                raise ValueError(
                    "reshape requires up to three integer dimensions and at most one -1"
                )
            result = a.reshape(shape)
        elif req.function in ("add", "subtract", "multiply", "divide", "dot"):
            b = array(parameters["b"])
            if req.function == "dot":
                if a.ndim > 2 or b.ndim > 2:
                    raise ValueError("dot supports at most two dimensions")
                bound(
                    a.size * max(1, b.shape[-1] if b.ndim > 1 else 1),
                    settings.max_work,
                    "Dot work",
                )
                function = np.dot
            else:
                bound(
                    math.prod(np.broadcast_shapes(a.shape, b.shape)),
                    settings.max_cells,
                    "Broadcast cells",
                )
                function = {
                    "add": np.add,
                    "subtract": np.subtract,
                    "multiply": np.multiply,
                    "divide": np.divide,
                }[req.function]
            result = function(a, b)
        else:
            function = {
                "sum": np.sum,
                "mean": np.mean,
                "median": np.median,
                "min": np.min,
                "max": np.max,
                "sort": np.sort,
                "std": np.std,
                "var": np.var,
                "quantile": np.quantile,
                "percentile": np.percentile,
                "sqrt": np.sqrt,
                "abs": np.abs,
            }[req.function]
            if req.function in ("quantile", "percentile"):
                q = array(parameters.get("q"))
                bound(q.size * max(1, a.size), settings.max_work, "Quantile work")
                bound(
                    q.size * max(1, a.size),
                    settings.max_cells,
                    "Quantile output upper bound",
                )
                if parameters.get("method", "linear") not in (
                    "linear",
                    "inverted_cdf",
                    "nearest",
                    "lower",
                    "higher",
                    "midpoint",
                ):
                    raise ValueError("Unsupported quantile method")
            result = function(a, **parameters)
        value = np.asarray(result)
        bound(value.size, settings.max_cells, "Output cells")
        if not np.isfinite(value).all():
            raise ValueError(
                "NumPy returned non-finite values; check inputs and denominators"
            )
        return value.tolist(), {
            "library_api": "numpy." + req.function,
            "rng": None,
            "seed": None,
            "stream": None,
            "numeric_dtype": str(value.dtype),
        }

    if isinstance(req, PandasRequest):
        data = decode_scalars(req.data)

        def frame(value, columns=None):
            if not isinstance(value, (list, dict)):
                raise ValueError("DataFrame input must be records or column lists")
            rows = (
                len(value)
                if isinstance(value, list)
                else max((len(v) for v in value.values()), default=0)
            )
            width = (
                len(value)
                if isinstance(value, dict)
                else max((len(r) for r in value), default=0)
            )
            bound(rows, settings.max_observations, "Table rows")
            bound(rows * width, settings.max_cells, "Table cells")
            return pd.DataFrame(value, columns=columns)

        if req.function == "to_numeric":
            if not isinstance(data, list):
                raise ValueError("to_numeric requires a list")
            bound(len(data), settings.max_cells, "Numeric values")
            output = pd.to_numeric(pd.Series(data), errors=req.errors).tolist()
        else:
            df = frame(data, req.columns if req.function == "DataFrame" else None)
            if req.function == "groupby":
                if not req.by or not req.agg:
                    raise ValueError(
                        "groupby requires by keys and an explicit aggregation mapping"
                    )
                df = (
                    df.groupby(req.by, dropna=req.dropna, sort=True)
                    .agg(req.agg)
                    .reset_index()
                )
            elif req.function == "merge":
                if not req.by:
                    raise ValueError("merge requires explicit join keys")
                right = frame(decode_scalars(req.right))
                bound(len(df) * len(right), settings.max_work, "Merge work")
                df = df.merge(
                    right, on=req.by, how=req.how, validate=req.merge_validation
                )
            elif req.function == "sort_values":
                if not req.by:
                    raise ValueError("sort_values requires explicit keys")
                df = df.sort_values(req.by, ascending=req.ascending, kind="stable")
            elif req.function == "dropna":
                df = df.dropna(subset=req.columns)
            bound(len(df), settings.max_observations, "Output rows")
            bound(df.size, settings.max_cells, "Output table cells")
            output = df.to_dict("records")

        return encode_scalars(output), {
            "library_api": "pandas." + req.function,
            "pandas_version": pd.__version__,
            "rng": None,
            "seed": None,
            "stream": None,
            "numeric_dtype": "preserved object scalars / pandas statistical types",
        }

    raise ValueError("Unsupported library request")
