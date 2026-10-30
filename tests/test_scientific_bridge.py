"""Real Monty/CPython/SciPy integration, numerical parity and process limits."""

import asyncio
import importlib.metadata
import json
import os
import sys
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

import numpy as np
import psutil
from scipy import stats
from scipy.cluster.vq import kmeans2
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel
from pydantic_monty import AsyncMonty, MontyError
from test_agent import executable_plan, tool_returns

from data_access import DataSources
from execution import execute_plan, require_finished_values
from main import create_context
from planning import ExecutionPlan
from scientific_bridge import ScientificBridge, ScientificWorker, ScientificWorkerError
from scientific_models import ScientificSettings, scientific_catalog
from summary_evidence import prepare_evidence
from domain_instructions import DomainProfile
from execution import ExecutionResult


def rng(seed=42, stream=0):
    return np.random.default_rng(np.random.SeedSequence([seed, stream]))


class ScientificTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.bridge = await ScientificBridge().__aenter__()

    async def asyncTearDown(self):
        await self.bridge.__aexit__(None, None, None)

    async def test_kmeans_matches_scipy_and_ignores_constant_features(self):
        observations = [[0, 7], [1, 7], [9, 7], [10, 7]]
        output = await self.bridge.invoke(
            "kmeans", observations=observations, clusters=2
        )
        values = np.array(observations, dtype=float)
        means = values.mean(axis=0)
        std = values.std(axis=0)
        scales = np.where(std == 0, 1, std)
        normalized = (values - means) / scales
        centers, labels = kmeans2(
            normalized, normalized[[0, 3]], iter=50, minit="matrix", rng=rng()
        )
        np.testing.assert_array_equal(output["labels"], labels)
        np.testing.assert_allclose(output["centroids"], centers * scales + means)
        self.assertEqual(output["constant_features"], [1])
        self.assertTrue(output["converged"])
        self.assertEqual(
            output["method"]["scipy_version"], importlib.metadata.version("scipy")
        )

    async def test_identical_profiles_do_not_invent_three_clusters(self):
        result = await self.bridge.invoke(
            "kmeans", observations=[[1, 2]] * 4, clusters=3
        )
        self.assertEqual(result["clusters_found"], 1)
        self.assertEqual(result["labels"], [0, 0, 0, 0])
        self.assertEqual(result["counts"], [4])
        short = await self.bridge.invoke("kmeans", observations=[[1], [2]], clusters=3)
        self.assertEqual(short["clusters_found"], 2)

    async def test_kmeans_plus_plus_respects_iteration_budget(self):
        result = await self.bridge.invoke(
            "kmeans",
            observations=[[0], [1], [9], [10]],
            clusters=2,
            initialization="kmeans++",
            max_iterations=1,
        )
        self.assertEqual(result["iterations"], 1)

    async def test_bootstrap_matches_scipy_and_exact_nearest_rank_indices(self):
        values = [1, 3, 6, 8, 15, 19]
        result = await self.bridge.invoke("bootstrap", values=values, n_resamples=1000)
        expected = stats.bootstrap(
            (np.array(values),),
            np.mean,
            method="percentile",
            n_resamples=1000,
            rng=rng(),
            batch=128,
            vectorized=True,
        )
        np.testing.assert_array_equal(
            result["bootstrap_distribution"], expected.bootstrap_distribution
        )
        ordered = np.sort(expected.bootstrap_distribution)
        self.assertEqual(result["lower"], ordered[24])
        self.assertEqual(result["upper"], ordered[974])
        self.assertEqual(result["standard_error"], expected.standard_error)
        self.assertEqual(result["method"]["confidence_level_pct"], 95)

    async def test_scipy_basic_confidence_method_requires_explicit_linear_quantiles(
        self,
    ):
        with self.assertRaisesRegex(ValueError, "Nearest-rank"):
            await self.bridge.invoke("bootstrap", values=[1, 2, 3], method="basic")
        result = await self.bridge.invoke(
            "bootstrap",
            values=[1, 2, 3, 5],
            method="basic",
            quantile_method="linear",
            n_resamples=100,
        )
        expected = stats.bootstrap(
            (np.array([1, 2, 3, 5]),),
            np.mean,
            method="basic",
            n_resamples=100,
            rng=rng(),
            batch=100,
        )
        np.testing.assert_allclose(
            [result["lower"], result["upper"]], expected.confidence_interval
        )

    async def test_seeded_binomial_coin_flips_and_independent_streams(self):
        result = await self.bridge.invoke("binomial", n=10, p=0.3, size=100)
        expected = stats.binom.rvs(10, 0.3, size=(100,), random_state=rng())
        self.assertEqual(result["samples"], expected.tolist())
        again = await self.bridge.invoke("binomial", n=10, p=0.3, size=100)
        self.assertEqual(result["samples"], again["samples"])
        separate = await self.bridge.invoke("binomial", n=10, p=0.3, size=100, stream=1)
        self.assertNotEqual(result["samples"], separate["samples"])
        coins = await self.bridge.invoke("bernoulli", p=0.5, size=100)
        self.assertEqual(
            coins["samples"],
            stats.bernoulli.rvs(0.5, size=(100,), random_state=rng()).tolist(),
        )

    async def test_general_monte_carlo_batches_broadcast_probabilities(self):
        probabilities = [0, 0.2, 1]
        result = await self.bridge.invoke(
            "sample",
            distribution="bernoulli",
            parameters={"p": probabilities},
            size=[1000, 3],
        )
        expected = stats.bernoulli.rvs(
            probabilities, size=(1000, 3), random_state=rng()
        )
        self.assertEqual(result["samples"], expected.tolist())
        self.assertEqual(result["shape"], [1000, 3])
        self.assertTrue(all(row[0] == 0 and row[2] == 1 for row in result["samples"]))

    async def test_monte_carlo_hypothesis_test_matches_scipy(self):
        values = [0.2, 0.4, 0.8, -0.2]
        output = await self.bridge.invoke(
            "monte_carlo_test", values=values, n_resamples=100
        )
        generator = rng()
        expected = stats.monte_carlo_test(
            values,
            lambda size: stats.norm.rvs(size=size, random_state=generator),
            np.mean,
            n_resamples=100,
            batch=100,
            vectorized=True,
        )
        self.assertEqual(output["statistic"], expected.statistic)
        self.assertEqual(output["pvalue"], expected.pvalue)
        np.testing.assert_array_equal(
            output["null_distribution"], expected.null_distribution
        )

    async def test_monty_uses_scipy_and_keeps_original_money_as_decimal(self):
        rows = [
            {"value": Decimal("0.10")},
            {"value": Decimal("0.20")},
            {"value": Decimal("0.30")},
        ]
        code = """import scientific
from decimal import Decimal
rows = data['q']
draws = await scientific.resample_indices(len(rows), n_resamples=10, seed=42)
means = [sum((rows[i]['value'] for i in sample), Decimal('0')) / len(rows) for sample in draws['indices']]
{'mean': sum((row['value'] for row in rows), Decimal('0')) / len(rows), 'resampled_means': means, 'method': draws['method']}"""
        async with AsyncMonty(
            binary_path=str(Path(sys.executable).parent / "monty")
        ) as pool:
            async with pool.checkout(
                limits={"max_suspensions": 64}, external_modules=self.bridge.modules()
            ) as session:
                result = await session.feed_run(code, inputs={"data": {"q": rows}})
        self.assertEqual(result["mean"], Decimal("0.20"))
        self.assertTrue(
            all(isinstance(value, Decimal) for value in result["resampled_means"])
        )
        expected = rng().integers(0, 3, size=(10, 3))
        self.assertEqual(
            result["resampled_means"],
            [
                sum((rows[i]["value"] for i in indices), Decimal(0)) / 3
                for indices in expected
            ],
        )

    async def test_bridge_import_stubs_support_type_checking(self):
        async with AsyncMonty(
            binary_path=str(Path(sys.executable).parent / "monty")
        ) as pool:
            async with pool.checkout(
                type_check=True, external_modules=self.bridge.modules()
            ) as session:
                output = await session.feed_run(
                    "import scientific\n(await scientific.bernoulli(size=4))['samples']"
                )
        self.assertEqual(len(output), 4)

    async def test_unawaited_future_is_not_accepted_as_calculation_output(self):
        async with AsyncMonty(
            binary_path=str(Path(sys.executable).parent / "monty")
        ) as pool:
            async with pool.checkout(external_modules=self.bridge.modules()) as session:
                output = await session.feed_run(
                    "import scientific\nscientific.bernoulli(size=4)"
                )
                with self.assertRaisesRegex(ValueError, "Await scientific"):
                    require_finished_values({"unfinished": output})

    async def test_invalid_inputs_and_oversized_work_are_rejected(self):
        invalid = [
            ("binomial", {"n": -1, "p": 0.5}),
            ("bernoulli", {"p": 2}),
            ("kmeans", {"observations": [[1], [1, 2]], "clusters": 2}),
            ("bootstrap", {"values": [1]}),
            ("bootstrap", {"values": [1, float("nan")]}),
            ("sample", {"distribution": "normal", "parameters": {"scale": -1}}),
            (
                "sample",
                {"distribution": "normal", "parameters": {"filename": "/tmp/x"}},
            ),
            (
                "sample",
                {"distribution": "uniform", "parameters": {}, "size": 1_000_000},
            ),
            ("resample_indices", {"n_observations": 200, "n_resamples": 10_000}),
            ("arbitrary_eval", {"code": "print('x')"}),
        ]
        for operation, parameters in invalid:
            with self.subTest(operation=operation, parameters=parameters):
                with self.assertRaises(ValueError):
                    await self.bridge.invoke(operation, **parameters)

    async def test_output_size_and_total_execution_budgets_are_enforced(self):
        small = ScientificBridge(ScientificSettings(max_output_bytes=2000))
        async with small:
            with self.assertRaisesRegex(ValueError, "output too large"):
                await small.invoke(
                    "sample", distribution="normal", parameters={}, size=1000
                )
            small.spent_seconds = small.settings.total_seconds
            with self.assertRaisesRegex(ValueError, "budget exhausted"):
                await small.invoke("bernoulli", size=1)

    async def test_unlisted_modules_and_host_side_effects_are_unavailable(self):
        async with AsyncMonty(
            binary_path=str(Path(sys.executable).parent / "monty")
        ) as pool:
            for code in (
                "import scipy",
                "import numpy",
                "import scientific\nscientific.eval('1')",
                "import scientific\nscientific.open('/etc/passwd')",
            ):
                async with pool.checkout(
                    external_modules=self.bridge.modules()
                ) as session:
                    with self.assertRaises(MontyError):
                        await session.feed_run(code)


class WorkerResourceTests(unittest.IsolatedAsyncioTestCase):
    async def test_native_worker_timeout_kills_process_and_allows_fresh_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hang.py"
            path.write_text(
                "import sys,time\nfor line in sys.stdin:\n time.sleep(30)\n"
            )
            worker = ScientificWorker(
                ScientificSettings(timeout_seconds=0.2), worker_path=path
            )
            with self.assertRaisesRegex(ScientificWorkerError, "timed out"):
                await worker.execute({"operation": "bernoulli"})
            self.assertIsNone(worker.process)
            worker.worker_path = (
                Path(__file__).resolve().parents[1] / "scientific_worker.py"
            )
            result = await worker.execute(
                {"operation": "bernoulli", "size": 1}, timeout=30
            )
            self.assertEqual(len(result["samples"]), 1)
            pid = worker.process.pid
            await worker.close()
            self.assertFalse(psutil.pid_exists(pid))

    async def test_memory_supervisor_terminates_over_budget_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.py"
            path.write_text(
                "import sys,time\nfor line in sys.stdin:\n data=bytearray(100*1024*1024)\n time.sleep(30)\n"
            )
            worker = ScientificWorker(
                ScientificSettings(timeout_seconds=5, memory_mb=64), worker_path=path
            )
            with self.assertRaisesRegex(ScientificWorkerError, "RSS memory budget"):
                await worker.execute({"operation": "bernoulli"})
            self.assertIsNone(worker.process)

    async def test_credentials_are_not_forwarded_to_numeric_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "environment.py"
            path.write_text(
                "import sys,os,json\nfor line in sys.stdin:\n req=json.loads(line)\n print(json.dumps({'id':req['id'],'result':{'environment':dict(os.environ)}}),flush=True)\n"
            )
            worker = ScientificWorker(ScientificSettings(), worker_path=path)
            try:
                with patch.dict(
                    os.environ,
                    {
                        "LITELLM_MASTER_KEY": "private-test-key",
                        "SMARTWEALTH_DB_PASSWORD": "private-test-password",
                    },
                ):
                    result = await worker.execute({"operation": "bernoulli"})
                self.assertNotIn("LITELLM_MASTER_KEY", result["environment"])
                self.assertNotIn("SMARTWEALTH_DB_PASSWORD", result["environment"])
            finally:
                await worker.close()


class ScientificPipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_smartwealth_bootstrap_uses_complete_authorized_sql_rows(self):
        contract = {
            "required_columns": {
                "customer_id": {"type": "integer"},
                "holding_value": {"type": "numeric"},
            },
            "grain": ["customer_id"],
            "population": {
                "table": "sw_customers",
                "source_key": "customer_id",
                "result_key": "customer_id",
            },
        }
        plan = ExecutionPlan.model_validate(
            executable_plan("2", "sql_python", declared=True, output_contract=contract)
        )
        actions = [
            ("describe_sources", {"step_id": "s1"}),
            (
                "run_sql",
                {
                    "step_id": "s2",
                    "statement": "SELECT c.customer_id,COALESCE(SUM(h.market_value),0) AS holding_value FROM sw_customers c LEFT JOIN sw_holdings h ON h.customer_id=c.customer_id GROUP BY c.customer_id ORDER BY c.customer_id",
                },
            ),
            (
                "run_python",
                {
                    "step_id": "s3",
                    "result_ids": ["query_1"],
                    "code": "import scientific\nrows = data['query_1']\nvalues = [r['holding_value'] for r in rows]\nanalysis = await scientific.bootstrap(values, n_resamples=100)\n{'customers': len(rows), 'observed_mean': sum(values)/len(values), 'lower': analysis['lower'], 'upper': analysis['upper'], 'method': analysis['method']}",
                },
            ),
        ]

        def respond(messages, info):
            self.assertNotIn("cc_monthly_financials", info.instructions)
            turns = tool_returns(messages)
            if len(turns) < len(actions):
                name, args = actions[len(turns)]
                return ModelResponse(parts=[ToolCallPart(name, args)])
            return ModelResponse(parts=[TextPart("Analysis complete")])

        data = DataSources(create_context(source_ids=["smartwealth"]), "2", "2")
        result = await execute_plan(
            "How uncertain is average holding value per customer?",
            plan,
            data,
            FunctionModel(respond),
        )
        output = result.python_runs[0]["output"]
        self.assertEqual(output["customers"], 12)
        self.assertIsInstance(output["observed_mean"], Decimal)
        values = [
            float(row["holding_value"]) for row in result.sql_results["query_1"]["rows"]
        ]
        direct = stats.bootstrap(
            (np.array(values),),
            np.mean,
            method="percentile",
            n_resamples=100,
            rng=rng(),
            batch=100,
        )
        ordered = np.sort(direct.bootstrap_distribution)
        self.assertEqual([output["lower"], output["upper"]], [ordered[2], ordered[97]])

    async def test_unawaited_scientific_code_returns_feedback_and_can_be_corrected(
        self,
    ):
        actions = [
            ("describe_sources", {"step_id": "s1"}),
            (
                "run_sql",
                {
                    "step_id": "s2",
                    "statement": "SELECT COUNT(*) AS n FROM cc_customers",
                },
            ),
            (
                "run_python",
                {
                    "step_id": "s3",
                    "result_ids": ["query_1"],
                    "code": "import scientific\nn = data['query_1'][0]['n']\n{'result': scientific.binomial(n, 0.5, size=10)}",
                },
            ),
            (
                "run_python",
                {
                    "step_id": "s3",
                    "result_ids": ["query_1"],
                    "code": "import scientific\nn = data['query_1'][0]['n']\nout = await scientific.binomial(n, 0.5, size=10)\n{'total': sum(out['samples']), 'method': out['method']}",
                },
            ),
        ]

        def respond(messages, info):
            turns = tool_returns(messages)
            if len(turns) == 3:
                self.assertIn("Await scientific", str(turns[-1].content))
            if len(turns) < len(actions):
                name, args = actions[len(turns)]
                return ModelResponse(parts=[ToolCallPart(name, args)])
            return ModelResponse(parts=[TextPart("Analysis complete")])

        data = DataSources(create_context(source_ids=["credit_cards"]), "1", "1")
        with patch.dict(os.environ, {"AGENT_TOOL_RETRIES": "2"}):
            result = await execute_plan(
                "Simulate outcomes",
                ExecutionPlan.model_validate(
                    executable_plan("1", "sql_python", declared=True)
                ),
                data,
                FunctionModel(respond),
            )
        self.assertEqual(len(result.python_runs), 1)
        self.assertEqual(
            result.python_runs[0]["scientific_calls"][0]["status"], "completed"
        )
        self.assertEqual(len(result.queries), 1)

    async def test_real_pipeline_calls_scipy_from_generated_monty_code(self):
        actions = [
            ("describe_sources", {"step_id": "s1"}),
            (
                "run_sql",
                {
                    "step_id": "s2",
                    "statement": "SELECT COUNT(*) AS n FROM cc_customers",
                },
            ),
            (
                "run_python",
                {
                    "step_id": "s3",
                    "result_ids": ["query_1"],
                    "code": "import scientific\nn = data['query_1'][0]['n']\nout = await scientific.binomial(n, 0.5, size=10)\n{'trials': n, 'total_successes': sum(out['samples']), 'method': out['method']}",
                },
            ),
        ]

        def respond(messages, info):
            self.assertIn("Available scientific module", info.instructions)
            returns = tool_returns(messages)
            if len(returns) < len(actions):
                name, args = actions[len(returns)]
                return ModelResponse(parts=[ToolCallPart(name, args)])
            return ModelResponse(parts=[TextPart("Calculation completed")])

        sources = DataSources(create_context(source_ids=["credit_cards"]), "1", "1")
        result = await execute_plan(
            "Simulate outcomes",
            ExecutionPlan.model_validate(
                executable_plan("1", "sql_python", declared=True)
            ),
            sources,
            FunctionModel(respond),
        )
        self.assertEqual(result.completed_steps, ["s1", "s2", "s3"])
        run = result.python_runs[0]
        self.assertTrue(run["executed"])
        self.assertEqual(
            run["output"]["total_successes"],
            sum(stats.binom.rvs(20, 0.5, size=(10,), random_state=rng())),
        )
        self.assertEqual(run["scientific_calls"][0]["status"], "completed")
        self.assertEqual(
            run["scientific_calls"][0]["method"]["library_api"],
            "scipy.stats.binom.rvs",
        )

    def test_method_metadata_can_support_summary_without_inventing_money(self):
        call = {
            "status": "completed",
            "method": {"seed": 42, "n_resamples": 1000, "confidence_level_pct": 95},
        }
        evidence = prepare_evidence(
            ExecutionResult(
                "",
                python_runs=[
                    {"output": {"lower": 10, "upper": 20}, "scientific_calls": [call]}
                ],
            ),
            DomainProfile(description="Generic analysis"),
        )
        self.assertEqual(
            evidence.unsupported_values(
                "95% range 10 to 20, with 1000 resamples and seed 42."
            ),
            [],
        )
        self.assertIn("42", evidence.unsupported_values("INR 42"))
        execution = ExecutionResult(
            "",
            python_runs=[
                {
                    "output": {
                        "lower": 10,
                        "upper": 20,
                        "method": {
                            "operation": "bootstrap",
                            "library_api": "scipy.stats.bootstrap",
                            "scipy_version": "1.18.1",
                            "rng": "PCG64",
                            "seed": 42,
                        },
                    },
                    "scientific_calls": [call],
                }
            ],
        )
        self.assertIn(
            "42",
            prepare_evidence(
                execution, DomainProfile(description="Analysis")
            ).unsupported_values("INR 42"),
        )
