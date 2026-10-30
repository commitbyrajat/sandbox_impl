"""Allowlisted Monty modules backed by a resource-controlled CPython worker."""

import asyncio
import hashlib
import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path

import psutil
from pydantic_monty import ExternalModule

from scientific_models import REQUEST_ADAPTER, ScientificSettings, scientific_stubs

ROOT = Path(__file__).resolve().parent


class ScientificWorkerError(ValueError):
    pass


class ScientificWorker:
    """One lazy worker per request; no credentials or arbitrary code enter it."""

    def __init__(
        self, settings: ScientificSettings, *, worker_path: Path | None = None
    ):
        self.settings = settings
        self.worker_path = worker_path or ROOT / "scientific_worker.py"
        self.process = None
        self.stderr_task = None
        self.stderr_tail = b""
        self.lock = asyncio.Lock()
        self.sequence = 0

    async def _drain_stderr(self, process):
        while chunk := await process.stderr.read(2048):
            self.stderr_tail = (self.stderr_tail + chunk)[-4096:]

    async def _start(self):
        if self.process is not None and self.process.returncode is None:
            return
        await self.close()
        self.stderr_tail = b""
        environment = {
            "PATH": str(Path(sys.executable).parent),
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "SCIENTIFIC_WORKER_SETTINGS": self.settings.model_dump_json(),
        }
        if os.name == "nt" and "SYSTEMROOT" in os.environ:
            environment["SYSTEMROOT"] = os.environ["SYSTEMROOT"]
        self.process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-I",
            "-B",
            str(self.worker_path),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=environment,
            limit=self.settings.max_output_bytes + 16_384,
        )
        self.stderr_task = asyncio.create_task(self._drain_stderr(self.process))

    async def _memory_guard(self, process):
        while process.returncode is None:
            try:
                rss = psutil.Process(process.pid).memory_info().rss
            except psutil.NoSuchProcess:
                return
            if rss > self.settings.memory_mb * 1024**2:
                raise ScientificWorkerError(
                    f"Scientific worker exceeded RSS memory budget ({self.settings.memory_mb} MiB)"
                )
            await asyncio.sleep(0.025)

    async def execute(self, request: dict, *, timeout: float | None = None):
        async with self.lock:
            read_task = monitor = None
            try:
                async with asyncio.timeout(
                    timeout if timeout is not None else self.settings.timeout_seconds
                ):
                    await self._start()
                    process = self.process
                    self.sequence += 1
                    wire = (
                        json.dumps(
                            {"id": self.sequence, "request": request},
                            separators=(",", ":"),
                        ).encode()
                        + b"\n"
                    )
                    if len(wire) > self.settings.max_output_bytes:
                        raise ScientificWorkerError(
                            "Scientific input exceeds the wire-size limit"
                        )
                    process.stdin.write(wire)
                    await process.stdin.drain()
                    read_task = asyncio.create_task(process.stdout.readline())
                    monitor = asyncio.create_task(self._memory_guard(process))
                    done, _ = await asyncio.wait(
                        {read_task, monitor}, return_when=asyncio.FIRST_COMPLETED
                    )
                    if monitor in done:
                        monitor.result()
                    line = await read_task
                    if not line:
                        await process.wait()
                        raise ScientificWorkerError(
                            f"Scientific worker exited ({process.returncode}): {self.stderr_tail.decode(errors='replace')[-1000:]}"
                        )
                    if len(line) > self.settings.max_output_bytes:
                        raise ScientificWorkerError(
                            "Scientific output exceeds the wire-size limit"
                        )
                    response = json.loads(line)
                    if response.get("id") != self.sequence:
                        raise ScientificWorkerError(
                            "Scientific worker response ID mismatch"
                        )
                    if "error" in response:
                        raise ScientificWorkerError(response["error"])
                    return response["result"]
            except TimeoutError as exc:
                await self.close()
                raise ScientificWorkerError(
                    "Scientific worker timed out; the native process was terminated"
                ) from exc
            except BaseException:
                await self.close()
                raise
            finally:
                for task in (read_task, monitor):
                    if task is not None:
                        task.cancel()
                await asyncio.gather(
                    *(task for task in (read_task, monitor) if task is not None),
                    return_exceptions=True,
                )

    async def close(self):
        process, self.process = self.process, None
        if process is not None:
            if process.returncode is None:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
            await process.wait()
        if self.stderr_task is not None:
            await asyncio.gather(self.stderr_task, return_exceptions=True)
            self.stderr_task = None


class ScientificBridge:
    def __init__(self, settings: ScientificSettings | None = None, *, logger=None):
        self.settings = settings or ScientificSettings.from_environment()
        self.worker = ScientificWorker(self.settings)
        self.calls = []
        self.spent_seconds = 0.0
        self.lock = asyncio.Lock()
        self.logger = logger or (lambda message: None)
        self.versions = {
            name: importlib.metadata.version(name) for name in ("numpy", "scipy")
        }
        self.fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "versions": self.versions,
                    "settings": self.settings.model_dump(),
                    "stubs": scientific_stubs(),
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.worker.close()

    async def invoke(self, operation, **parameters):
        # This validation is also repeated inside the worker. No dispatch via eval,
        # arbitrary imports, attribute lookup or model-supplied callbacks is allowed.
        request = REQUEST_ADAPTER.validate_python(
            {"operation": operation, **parameters}
        )
        async with self.lock:
            remaining = self.settings.total_seconds - self.spent_seconds
            if remaining <= 0:
                raise ScientificWorkerError(
                    "Scientific execution time budget exhausted for this request"
                )
            started = time.monotonic()
            record = {
                "operation": operation,
                "seed": request.seed,
                "stream": request.stream,
                "status": "started",
            }
            self.calls.append(record)
            self.logger(
                f"SCIENTIFIC {operation}: start; seed={request.seed}; stream={request.stream}"
            )
            try:
                result = await self.worker.execute(
                    request.model_dump(mode="json"),
                    timeout=min(remaining, self.settings.timeout_seconds),
                )
                record.update(
                    {
                        "status": "completed",
                        "method": result["method"],
                        "warnings": result.get("warnings", []),
                    }
                )
                self.logger(
                    "SCIENTIFIC_RESULT: "
                    + json.dumps(
                        {
                            "operation": operation,
                            "method": result["method"],
                            "warnings": result.get("warnings", []),
                        }
                    )
                )
                return result
            except asyncio.CancelledError:
                record["status"] = "cancelled"
                raise
            except Exception as exc:
                record.update({"status": "failed", "error": str(exc)})
                self.logger(f"SCIENTIFIC {operation}: failed: {exc}")
                raise
            finally:
                elapsed = time.monotonic() - started
                record["elapsed_seconds"] = round(elapsed, 4)
                self.spent_seconds += elapsed

    def modules(self):
        async def kmeans(
            observations,
            *,
            clusters=3,
            max_iterations=50,
            standardize=True,
            initialization="farthest_first",
            seed=42,
            stream=0,
        ):
            return await self.invoke(
                "kmeans",
                observations=observations,
                clusters=clusters,
                max_iterations=max_iterations,
                standardize=standardize,
                initialization=initialization,
                seed=seed,
                stream=stream,
            )

        async def bootstrap(
            values,
            *,
            statistic="mean",
            n_resamples=1000,
            confidence_level=0.95,
            method="percentile",
            quantile_method="nearest_rank",
            seed=42,
            stream=0,
        ):
            return await self.invoke(
                "bootstrap",
                values=values,
                statistic=statistic,
                n_resamples=n_resamples,
                confidence_level=confidence_level,
                method=method,
                quantile_method=quantile_method,
                seed=seed,
                stream=stream,
            )

        async def sample(distribution, parameters, *, size=1000, seed=42, stream=0):
            return await self.invoke(
                "sample",
                distribution=distribution,
                parameters=parameters,
                size=size,
                seed=seed,
                stream=stream,
            )

        async def binomial(n, p, *, size=1000, seed=42, stream=0):
            return await self.invoke(
                "binomial", n=n, p=p, size=size, seed=seed, stream=stream
            )

        async def bernoulli(p=0.5, *, size=1000, seed=42, stream=0):
            return await self.invoke(
                "bernoulli", p=p, size=size, seed=seed, stream=stream
            )

        async def resample_indices(
            n_observations, *, n_resamples=1000, seed=42, stream=0
        ):
            return await self.invoke(
                "resample_indices",
                n_observations=n_observations,
                n_resamples=n_resamples,
                seed=seed,
                stream=stream,
            )

        async def monte_carlo_test(
            values,
            *,
            distribution="normal",
            parameters=None,
            statistic="mean",
            n_resamples=1000,
            alternative="two-sided",
            seed=42,
            stream=0,
        ):
            return await self.invoke(
                "monte_carlo_test",
                values=values,
                distribution=distribution,
                parameters=parameters or {},
                statistic=statistic,
                n_resamples=n_resamples,
                alternative=alternative,
                seed=seed,
                stream=stream,
            )

        return {
            "scientific": ExternalModule(
                {
                    "kmeans": kmeans,
                    "bootstrap": bootstrap,
                    "sample": sample,
                    "binomial": binomial,
                    "bernoulli": bernoulli,
                    "resample_indices": resample_indices,
                    "monte_carlo_test": monte_carlo_test,
                },
                stubs=scientific_stubs(),
            )
        }
