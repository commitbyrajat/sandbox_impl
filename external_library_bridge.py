"""Monty import facades; computation stays in the supervised numerical worker."""

from decimal import Decimal
from types import SimpleNamespace

from pydantic_monty import ExternalModule, ClassInstance

from external_module_models import decode_scalars, encode_scalars, module_catalog


def numeric_json(value):
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (list, tuple)):
        return [numeric_json(v) for v in value]
    if isinstance(value, dict):
        return {k: numeric_json(v) for k, v in value.items()}
    return value


def library_modules(bridge):
    callbacks = {}

    async def numpy(function, a, **parameters):
        result = await bridge.invoke(
            "numpy",
            function=function,
            a=numeric_json(a),
            parameters=numeric_json(parameters),
        )
        return result["value"]

    def array(function):
        async def call(a, dtype="float64"):
            return await numpy(function, a, dtype=dtype)

        return call

    for name in ("array", "asarray"):
        callbacks[f"numpy.{name}"] = array(name)

    async def reshape(a, shape):
        return await numpy("reshape", a, shape=list(shape))

    callbacks["numpy.reshape"] = reshape

    def reduction(function):
        async def call(a, axis=None):
            return await numpy(function, a, axis=axis)

        return call

    for name in ("mean", "sum", "median", "min", "max", "sort"):
        callbacks[f"numpy.{name}"] = reduction(name)

    def dispersion(function):
        async def call(a, axis=None, ddof=0):
            return await numpy(function, a, axis=axis, ddof=ddof)

        return call

    for name in ("std", "var"):
        callbacks[f"numpy.{name}"] = dispersion(name)

    def quantile(function):
        async def call(a, q, axis=None, method="linear"):
            return await numpy(function, a, q=q, axis=axis, method=method)

        return call

    for name in ("quantile", "percentile"):
        callbacks[f"numpy.{name}"] = quantile(name)

    def unary(function):
        async def call(a):
            return await numpy(function, a)

        return call

    for name in ("sqrt", "abs"):
        callbacks[f"numpy.{name}"] = unary(name)

    def binary(function):
        async def call(a, b):
            return await numpy(function, a, b=b)

        return call

    for name in ("add", "subtract", "multiply", "divide", "dot"):
        callbacks[f"numpy.{name}"] = binary(name)

    async def random_call(function, parameters, size, seed, stream):
        result = await bridge.invoke(
            "numpy_random",
            function=function,
            parameters=numeric_json(parameters),
            size=list(size) if isinstance(size, tuple) else size,
            seed=seed,
            stream=stream,
        )
        return result["value"]

    async def binomial(n, p, *, size=1000, seed=42, stream=0):
        return await random_call("binomial", {"n": n, "p": p}, size, seed, stream)

    async def normal(loc=0.0, scale=1.0, *, size=1000, seed=42, stream=0):
        return await random_call(
            "normal", {"loc": loc, "scale": scale}, size, seed, stream
        )

    async def uniform(low=0.0, high=1.0, *, size=1000, seed=42, stream=0):
        return await random_call(
            "uniform", {"low": low, "high": high}, size, seed, stream
        )

    async def random(*, size=1000, seed=42, stream=0):
        return await random_call("random", {}, size, seed, stream)

    for name, callback in {
        "binomial": binomial,
        "normal": normal,
        "uniform": uniform,
        "random": random,
    }.items():
        callbacks[f"numpy.random.{name}"] = callback

    async def table(function, data, **parameters):
        if "right" in parameters:
            parameters["right"] = encode_scalars(parameters["right"])
        result = await bridge.invoke(
            "pandas", function=function, data=encode_scalars(data), **parameters
        )
        return decode_scalars(result["value"])

    async def DataFrame(data, columns=None):
        return await table("DataFrame", data, columns=columns)

    async def groupby(data, by, agg, dropna=False):
        return await table("groupby", data, by=by, agg=agg, dropna=dropna)

    async def merge(left, right, on, how="inner", validate="one_to_one"):
        return await table(
            "merge", left, right=right, by=on, how=how, merge_validation=validate
        )

    async def sort_values(data, by, ascending=True):
        return await table("sort_values", data, by=by, ascending=ascending)

    async def dropna(data, subset=None):
        return await table("dropna", data, columns=subset)

    async def to_numeric(values, errors="raise"):
        return await table("to_numeric", values, errors=errors)

    for name, callback in {
        "DataFrame": DataFrame,
        "groupby": groupby,
        "merge": merge,
        "sort_values": sort_values,
        "dropna": dropna,
        "to_numeric": to_numeric,
    }.items():
        callbacks[f"pandas.{name}"] = callback

    async def scipy_sample(distribution, parameters, size, seed, stream):
        result = await bridge.invoke(
            "sample",
            distribution=distribution,
            parameters=parameters,
            size=list(size) if isinstance(size, tuple) else size,
            seed=seed,
            stream=stream,
        )
        return result["samples"]

    async def bernoulli(p, *, size=1000, seed=42, stream=0):
        return await scipy_sample("bernoulli", {"p": p}, size, seed, stream)

    async def binom(n, p, *, size=1000, seed=42, stream=0):
        return await scipy_sample("binomial", {"n": n, "p": p}, size, seed, stream)

    async def norm(loc=0.0, scale=1.0, *, size=1000, seed=42, stream=0):
        return await scipy_sample(
            "normal", {"loc": loc, "scale": scale}, size, seed, stream
        )

    async def scipy_uniform(loc=0.0, scale=1.0, *, size=1000, seed=42, stream=0):
        return await scipy_sample(
            "uniform", {"loc": loc, "scale": scale}, size, seed, stream
        )

    for name, callback in {
        "bernoulli": bernoulli,
        "binom": binom,
        "norm": norm,
        "uniform": scipy_uniform,
    }.items():
        callbacks[f"scipy.stats.{name}.rvs"] = callback

    async def bootstrap(values, **kwargs):
        return await bridge.invoke("bootstrap", values=values, **kwargs)

    callbacks["scipy.stats.bootstrap"] = bootstrap

    async def kmeans2(data, k, *, iter=50, seed=42, stream=0):
        result = await bridge.invoke(
            "kmeans",
            observations=data,
            clusters=k,
            max_iterations=iter,
            standardize=False,
            initialization="kmeans++",
            seed=seed,
            stream=stream,
        )
        return result["centroids"], result["labels"]

    callbacks["scipy.cluster.vq.kmeans2"] = kmeans2

    def statistic(name):
        async def call(data):
            fn = {
                "fmean": "mean",
                "pstdev": "std",
                "stdev": "std",
                "pvariance": "var",
                "variance": "var",
            }.get(name, name)
            return await numpy(
                fn,
                data,
                **(
                    {"ddof": 1 if name in ("stdev", "variance") else 0}
                    if fn in ("std", "var")
                    else {}
                ),
            )

        return call

    for name in ("mean", "fmean", "median", "pstdev", "stdev", "pvariance", "variance"):
        callbacks[f"statistics.{name}"] = statistic(name)

    catalog = module_catalog()
    if set(callbacks) != set(catalog["functions"]):
        raise ValueError(
            "External-module registry does not match callback implementations"
        )

    def build(prefix):
        attrs, definitions = {}, []
        paths = {
            path[len(prefix) + 1 :].split(".")[0]
            for path in callbacks
            if path.startswith(prefix + ".")
        }
        for key in sorted(paths):
            path = prefix + "." + key
            if path in callbacks:
                attrs[key] = callbacks[path]
                definitions.append(
                    "async def " + catalog["functions"][path]["signature"] + ": ..."
                )
            else:
                if path == "numpy.random":
                    # Monty reserves the bundled `random` module even in nested
                    # ExternalModules. Expose only this numeric namespace's methods.
                    methods = {
                        p.rsplit(".", 1)[1]: fn
                        for p, fn in callbacks.items()
                        if p.startswith(path + ".")
                    }
                    attrs[key] = ClassInstance(
                        SimpleNamespace(**methods), allowed_methods=set(methods)
                    )
                    definitions.append(
                        "class _Random:\n"
                        + "\n".join(
                            "    async def "
                            + catalog["functions"][path + "." + name][
                                "signature"
                            ].replace("(", "(self, ", 1)
                            + ": ..."
                            for name in methods
                        )
                        + "\nrandom: _Random"
                    )
                else:
                    attrs[key] = build(path)
                    definitions.append("from . import " + key + " as " + key)
        if prefix == "numpy":
            attrs.update(float64="float64", int64="int64", bool_="bool")
            definitions.append("float64: str\nint64: str\nbool_: str")
        return ExternalModule(
            attrs, stubs="from typing import Any\n" + "\n".join(definitions) + "\n"
        )

    return {name: build(name) for name in catalog["modules"]}
