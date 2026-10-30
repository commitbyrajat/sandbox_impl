"""Trusted numerical worker. JSON in/out only; never executes model-written code."""

import json
import math
import os
import sys
import warnings
from decimal import Decimal, ROUND_CEILING
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scientific_models import REQUEST_ADAPTER, ScientificSettings


def resource_limits(settings):
    if sys.platform != "win32":
        import resource

        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        if sys.platform.startswith("linux"):
            # Virtual address space includes native library mappings; RSS is separately
            # monitored by the parent on every supported platform.
            ceiling = max(1024**3, settings.memory_mb * 4 * 1024**2)
            resource.setrlimit(resource.RLIMIT_AS, (ceiling, ceiling))


def main():
    settings = ScientificSettings.model_validate_json(
        os.environ["SCIENTIFIC_WORKER_SETTINGS"]
    )
    resource_limits(settings)
    import numpy as np
    import scipy
    from scipy import stats
    from scipy.cluster.vq import kmeans2

    statistics = {"mean": np.mean, "median": np.median, "sum": np.sum, "std": np.std}
    distributions = {
        "normal": stats.norm,
        "uniform": stats.uniform,
        "binomial": stats.binom,
        "bernoulli": stats.bernoulli,
    }

    def bound(value, maximum, name):
        if value > maximum:
            raise ValueError(f"{name} {value} exceeds limit {maximum}")

    def vector(values):
        if not values:
            raise ValueError("At least one observation is required")
        bound(len(values), settings.max_observations, "Observations")
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 1 or not np.isfinite(array).all():
            raise ValueError("Observations must be finite numeric scalars")
        return array

    def distribution_args(name, parameters):
        allowed = {
            "normal": {"loc", "scale"},
            "uniform": {"loc", "scale"},
            "binomial": {"n", "p"},
            "bernoulli": {"p"},
        }[name]
        if set(parameters) - allowed:
            raise ValueError(
                f"Unsupported distribution parameters: {sorted(set(parameters)-allowed)}"
            )
        args = {
            key: np.asarray(value, dtype=np.float64)
            for key, value in parameters.items()
        }
        if name in ("normal", "uniform"):
            args.setdefault("loc", np.asarray(0.0))
            args.setdefault("scale", np.asarray(1.0))
        if any(not np.isfinite(value).all() for value in args.values()):
            raise ValueError("Distribution parameters must be finite")
        if name in ("normal", "uniform") and np.any(args.get("scale", 1) <= 0):
            raise ValueError("scale must be positive")
        if name in ("binomial", "bernoulli"):
            if "p" not in args or np.any((args["p"] < 0) | (args["p"] > 1)):
                raise ValueError("p must be in [0,1]")
        if name == "binomial":
            if (
                "n" not in args
                or np.any(args["n"] < 0)
                or np.any(args["n"] != np.floor(args["n"]))
                or np.any(args["n"] > 2**53 - 1)
            ):
                raise ValueError(
                    "n must contain nonnegative integer trial counts no larger than 2**53-1"
                )
        return args

    def execute(req):
        rng = np.random.default_rng(np.random.SeedSequence([req.seed, req.stream]))
        meta = {
            "operation": req.operation,
            "scipy_version": scipy.__version__,
            "numpy_version": np.__version__,
            "seed": req.seed,
            "stream": req.stream,
            "rng": "Generator(PCG64)/SeedSequence([seed, stream])",
            "numeric_dtype": "float64",
        }
        if req.operation == "kmeans":
            n = len(req.observations)
            if not n or not req.observations[0]:
                raise ValueError("A nonempty observation matrix is required")
            width = len(req.observations[0])
            bound(n, settings.max_observations, "Observations")
            bound(n * width, settings.max_cells, "Matrix cells")
            bound(req.max_iterations, settings.max_iterations, "Iterations")
            bound(
                n * width * req.clusters * req.max_iterations,
                settings.max_work,
                "K-means work",
            )
            if any(len(row) != width for row in req.observations):
                raise ValueError("Matrix must be rectangular")
            original = np.asarray(req.observations, dtype=np.float64)
            if not np.isfinite(original).all():
                raise ValueError("Features must be finite")
            means = original.mean(axis=0) if req.standardize else np.zeros(width)
            std = original.std(axis=0) if req.standardize else np.ones(width)
            scales = np.where(std == 0, 1, std)
            matrix = (original - means) / scales
            distinct = len(np.unique(matrix, axis=0))
            k = min(req.clusters, distinct)
            if req.initialization == "farthest_first":
                centers = [matrix[0]]
                for _ in range(1, k):
                    distances = np.min(
                        np.sum(
                            (matrix[:, None, :] - np.asarray(centers)[None, :, :]) ** 2,
                            axis=2,
                        ),
                        axis=1,
                    )
                    centers.append(matrix[int(np.argmax(distances))])
                centers = np.asarray(centers)
                previous = None
                first_iteration = 1
            else:
                # SciPy performs k-means++ initialization, then the usual update.
                centers, labels = kmeans2(matrix, k, iter=1, minit="++", rng=rng)
                previous = labels.copy()
                first_iteration = 2
            iteration = 1
            converged = False
            for iteration in range(first_iteration, req.max_iterations + 1):
                centers, labels = kmeans2(
                    matrix,
                    centers,
                    iter=1,
                    minit="matrix",
                    missing="warn",
                    check_finite=True,
                    rng=rng,
                )
                if previous is not None and np.array_equal(labels, previous):
                    converged = True
                    break
                previous = labels.copy()
            counts = np.bincount(labels, minlength=k)
            meta.update(
                {
                    "library_api": "scipy.cluster.vq.kmeans2",
                    "initialization": req.initialization,
                    "standardize": req.standardize,
                    "max_iterations": req.max_iterations,
                }
            )
            return {
                "labels": labels.tolist(),
                "centroids": (centers * scales + means).tolist(),
                "counts": counts.tolist(),
                "clusters_requested": req.clusters,
                "clusters_found": int(np.count_nonzero(counts)),
                "iterations": iteration,
                "converged": converged,
                "constant_features": np.flatnonzero(std == 0).tolist(),
                "method": meta,
            }
        if req.operation in ("bootstrap", "monte_carlo_test"):
            values = vector(req.values)
            bound(req.n_resamples, settings.max_resamples, "Resamples")
            extra = (
                len(values) ** 2
                if req.operation == "bootstrap" and req.method == "BCa"
                else 0
            )
            bound(
                len(values) * req.n_resamples + extra,
                settings.max_work,
                "Resampling work",
            )
            statistic = statistics[req.statistic]
            meta.update(
                {
                    "statistic": req.statistic,
                    "n_resamples": req.n_resamples,
                    "observations": len(values),
                }
            )
            if req.operation == "bootstrap":
                if len(values) < 2:
                    raise ValueError("Bootstrap requires at least two observations")
                if req.method != "percentile" and req.quantile_method != "linear":
                    raise ValueError(
                        "Nearest-rank quantiles require method='percentile'; use quantile_method='linear' for basic/BCa"
                    )
                result = stats.bootstrap(
                    (values,),
                    statistic,
                    n_resamples=req.n_resamples,
                    confidence_level=req.confidence_level,
                    method=req.method,
                    rng=rng,
                    batch=min(128, req.n_resamples),
                    vectorized=True,
                )
                low, high = result.confidence_interval
                if req.method == "percentile" and req.quantile_method == "nearest_rank":
                    alpha = (Decimal(1) - Decimal(str(req.confidence_level))) / 2
                    ordered = np.sort(result.bootstrap_distribution)
                    ranks = [
                        max(
                            0,
                            min(
                                req.n_resamples - 1,
                                int(
                                    (q * req.n_resamples).to_integral_value(
                                        rounding=ROUND_CEILING
                                    )
                                )
                                - 1,
                            ),
                        )
                        for q in (alpha, 1 - alpha)
                    ]
                    low, high = ordered[ranks]
                if not np.isfinite([low, high, result.standard_error]).all():
                    raise ValueError(
                        "Bootstrap returned non-finite bounds; inspect degenerate observations and method"
                    )
                meta.update(
                    {
                        "library_api": "scipy.stats.bootstrap",
                        "confidence_level": req.confidence_level,
                        "confidence_level_pct": req.confidence_level * 100,
                        "method": req.method,
                        "quantile_method": req.quantile_method,
                    }
                )
                return {
                    "estimate": float(statistic(values)),
                    "lower": float(low),
                    "upper": float(high),
                    "standard_error": float(result.standard_error),
                    "bootstrap_distribution": result.bootstrap_distribution.tolist(),
                    "method": meta,
                }
            args = distribution_args(req.distribution, req.parameters)
            result = stats.monte_carlo_test(
                values,
                lambda size: distributions[req.distribution].rvs(
                    **args, size=size, random_state=rng
                ),
                statistic,
                n_resamples=req.n_resamples,
                alternative=req.alternative,
                vectorized=True,
                batch=min(128, req.n_resamples),
            )
            meta.update(
                {
                    "library_api": "scipy.stats.monte_carlo_test",
                    "distribution": req.distribution,
                    "parameters": {key: value.tolist() for key, value in args.items()},
                    "alternative": req.alternative,
                }
            )
            return {
                "statistic": float(result.statistic),
                "pvalue": float(result.pvalue),
                "null_distribution": result.null_distribution.tolist(),
                "method": meta,
            }
        if req.operation == "resample_indices":
            bound(req.n_observations, settings.max_observations, "Observations")
            bound(req.n_resamples, settings.max_resamples, "Resamples")
            bound(
                req.n_observations * req.n_resamples,
                settings.max_cells,
                "Resample cells",
            )
            meta.update(
                {
                    "library_api": "numpy.random.Generator.integers",
                    "numeric_dtype": "int64",
                    "n_resamples": req.n_resamples,
                    "observations": req.n_observations,
                }
            )
            return {
                "indices": rng.integers(
                    0, req.n_observations, size=(req.n_resamples, req.n_observations)
                ).tolist(),
                "method": meta,
            }
        name = req.distribution if req.operation == "sample" else req.operation
        parameters = (
            req.parameters
            if req.operation == "sample"
            else ({"p": req.p, "n": req.n} if name == "binomial" else {"p": req.p})
        )
        shape = tuple(req.size) if isinstance(req.size, list) else (req.size,)
        if not shape or len(shape) > 3:
            raise ValueError("Sampling supports at most three dimensions")
        bound(math.prod(shape), settings.max_cells, "Sample cells")
        args = distribution_args(name, parameters)
        samples = np.asarray(
            distributions[name].rvs(**args, size=shape, random_state=rng)
        )
        if not np.isfinite(samples).all():
            raise ValueError("Sampling returned non-finite values")
        meta.update(
            {
                "library_api": "scipy.stats."
                + {
                    "normal": "norm",
                    "uniform": "uniform",
                    "binomial": "binom",
                    "bernoulli": "bernoulli",
                }[name]
                + ".rvs",
                "distribution": name,
                "parameters": {key: value.tolist() for key, value in args.items()},
                "shape": list(shape),
                "numeric_dtype": str(samples.dtype),
            }
        )
        result = {"samples": samples.tolist(), "shape": list(shape), "method": meta}
        if req.operation in ("binomial", "bernoulli"):
            n = req.n if req.operation == "binomial" else 1
            result.update(
                {
                    "theoretical_mean": n * req.p,
                    "theoretical_variance": n * req.p * (1 - req.p),
                }
            )
        return result

    while line := sys.stdin.buffer.readline(settings.max_output_bytes + 1):
        request_id = None
        try:
            if len(line) > settings.max_output_bytes:
                raise ValueError("Scientific request exceeds the wire-size limit")
            envelope = json.loads(line)
            request_id = envelope["id"]
            req = REQUEST_ADAPTER.validate_json(json.dumps(envelope["request"]))
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                output = execute(req)
            output["warnings"] = list(dict.fromkeys(str(w.message) for w in caught))
            response = json.dumps(
                {"id": request_id, "result": output},
                allow_nan=False,
                separators=(",", ":"),
            )
            if len(response.encode()) > settings.max_output_bytes:
                raise ValueError(
                    "Scientific output too large; reduce sample count or dimensions"
                )
        except Exception as exc:
            response = json.dumps(
                {"id": request_id, "error": f"{type(exc).__name__}: {str(exc)[:2000]}"}
            )
        print(response, flush=True)


if __name__ == "__main__":
    main()
