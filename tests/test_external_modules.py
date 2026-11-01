"""Real NumPy/pandas/SciPy Monty facades, snapshot guards and mutable repair tools."""

import ast
import copy
import json
import sys
import unittest
from datetime import date
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy import stats
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel
from pydantic_monty import AsyncMonty, MontyError
from test_agent import executable_plan, tool_returns
from test_semantic_query import risk_query
from semantic_fixtures import count_query

from data_access import DataSources
from datasource_mapping import resolve_scope
from domain_instructions import get_domain_profile
from execution import execute_plan
from external_module_models import decode_scalars, encode_scalars, module_catalog
from main import create_context
from planning import ExecutionPlan, planner_output_type, PlannerDiagnostics
from pydantic import ValidationError
from pydantic_ai import ModelRetry
from python_validation import preflight_python, PythonPreflightError
from scientific_bridge import ScientificBridge, ScientificWorkerError
from scientific_models import ScientificSettings
from semantic_models import QueryIR, Predicate
from semantic_query import SemanticCatalog, SemanticQueryError, compile_query


def catalog():
    scope = resolve_scope("1", "1")
    return SemanticCatalog(
        scope.workspace.sources,
        get_domain_profile("1", tenant_id="1"),
        scope.workspace.semantic_planner,
    )


class ModuleTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.bridge = await ScientificBridge().__aenter__()
        self.pool = await AsyncMonty(
            binary_path=str(Path(sys.executable).parent / "monty")
        ).__aenter__()

    async def asyncTearDown(self):
        await self.pool.__aexit__(None, None, None)
        await self.bridge.__aexit__(None, None, None)

    async def run_code(self, code, rows=None, **options):
        inputs = {
            "query_1": (
                rows
                if rows is not None
                else [
                    {"x": Decimal("1.25")},
                    {"x": Decimal("2.75")},
                    {"x": Decimal("5.00")},
                ]
            )
        }
        code = preflight_python(code, inputs, scientific=True)
        if options.get("type_check"):
            options["type_check_stubs"] = (
                "from typing import Any\ndata: dict[str, list[dict[str, Any]]]\n"
            )
        async with self.pool.checkout(
            external_modules=self.bridge.modules(),
            limits={"max_suspensions": 64},
            **options,
        ) as session:
            return await session.feed_run(code, inputs={"data": inputs})

    async def test_numpy_reductions_quantiles_and_reshape_match_native(self):
        result = await self.run_code("""import numpy as np
rows=data['query_1']
a=await np.asarray([r['x'] for r in rows],dtype=np.float64)
matrix=await np.reshape(a,[3,1])
{'mean':await np.mean(a),'std':await np.std(a,ddof=1),'bounds':await np.percentile(a,[2.5,97.5],method='inverted_cdf'),'sums':await np.sum(matrix,axis=0)}
""")
        native = np.array([1.25, 2.75, 5.0])
        self.assertEqual(result["mean"], np.mean(native))
        self.assertEqual(result["std"], np.std(native, ddof=1))
        self.assertEqual(
            result["bounds"],
            np.percentile(native, [2.5, 97.5], method="inverted_cdf").tolist(),
        )
        self.assertEqual(result["sums"], [9.0])

    async def test_numpy_seeded_vector_binomial_uses_actual_generator(self):
        code = """import numpy as np
rows=data['query_1']
await np.random.binomial(1,[0.0,0.2,1.0],size=[1000,len(rows)],seed=42,stream=3)
"""
        result = await self.run_code(code)
        expected = np.random.default_rng(np.random.SeedSequence([42, 3])).binomial(
            1, [0, 0.2, 1], size=(1000, 3)
        )
        self.assertEqual(result, expected.tolist())
        self.assertEqual(
            self.bridge.calls[0]["method"]["library_api"],
            "numpy.random.Generator.binomial",
        )
        self.assertEqual(self.bridge.calls[0]["method"]["stream"], 3)
        self.assertEqual(await self.run_code(code), result)

    async def test_explicit_int64_keeps_integer_precision(self):
        rows = [{"x": 2**60 + 1}, {"x": 2**63 - 1}]
        result = await self.run_code(
            "import numpy as np\nrows=data['query_1']\nawait np.array([r['x'] for r in rows],dtype=np.int64)",
            rows,
        )
        self.assertEqual(result, [r["x"] for r in rows])

    async def test_pandas_records_keep_decimals_dates_and_null_groups(self):
        rows = [
            {"g": "a", "amount": Decimal("0.10"), "day": date(2026, 6, 1)},
            {"g": "a", "amount": Decimal("0.20"), "day": date(2026, 6, 1)},
            {"g": None, "amount": Decimal("0.40"), "day": date(2026, 6, 2)},
        ]
        result = await self.run_code(
            """import pandas as pd
frame=await pd.DataFrame(data['query_1'])
grouped=await pd.groupby(frame,by='g',agg={'amount':'sum'},dropna=False)
{'frame':frame,'grouped':grouped}
""",
            rows,
        )
        self.assertEqual(result["frame"], rows)
        self.assertEqual(
            result["grouped"],
            [
                {"g": "a", "amount": Decimal(".30")},
                {"g": None, "amount": Decimal(".40")},
            ],
        )
        self.assertEqual(
            self.bridge.calls[-1]["method"]["pandas_version"], pd.__version__
        )

    async def test_pandas_merge_checks_cardinality_and_numeric_nulls(self):
        rows = [{"id": "a", "v": 2}, {"id": "a", "v": 3}]
        with self.assertRaisesRegex(MontyError, "not a one-to-one merge"):
            await self.run_code(
                """import pandas as pd
rows=data['query_1']
await pd.merge(rows,rows,on='id')
""",
                rows,
            )
        result = await self.run_code(
            """import pandas as pd
rows=data['query_1']
clean=await pd.dropna(rows,subset=['id'])
ordered=await pd.sort_values(clean,by='v',ascending=False)
numeric=await pd.to_numeric(['1.5','bad',None],errors='coerce')
{'records':ordered,'numeric':numeric}
""",
            rows,
        )
        self.assertEqual(result["records"], list(reversed(rows)))
        self.assertEqual(result["numeric"], [1.5, None, None])

    async def test_scipy_submodules_and_statistics_are_real_worker_calls(self):
        result = await self.run_code("""from scipy.stats import bernoulli
import statistics
rows=data['query_1']
draws=await bernoulli.rvs([0.0,0.2,1.0],size=[5,3],seed=42)
{'draws':draws,'mean':await statistics.mean([r['x'] for r in rows])}
""")
        native = stats.bernoulli.rvs(
            [0, 0.2, 1],
            size=(5, 3),
            random_state=np.random.default_rng(np.random.SeedSequence([42, 0])),
        )
        self.assertEqual(result["draws"], native.tolist())
        self.assertEqual(result["mean"], 3.0)
        self.assertTrue(
            self.bridge.calls[0]["method"]["library_api"].startswith("scipy.stats")
        )

    async def test_weighted_bernoulli_sums_one_total_per_scenario(self):
        probabilities = [0, 0.2, 1]
        weights = [10, 20, 30]
        result = await self.bridge.invoke(
            "bernoulli_sum",
            probabilities=probabilities,
            weights=weights,
            n_simulations=1000,
        )
        draws = stats.bernoulli.rvs(
            probabilities,
            size=(1000, 3),
            random_state=np.random.default_rng(np.random.SeedSequence([42, 0])),
        )
        totals = np.sum(draws * np.asarray(weights), axis=1)
        self.assertEqual(result["samples"], totals.tolist())
        self.assertEqual(len(result["samples"]), 1000)
        self.assertEqual(result["expected_value"], 34.0)
        self.assertEqual(result["simulated_mean"], float(totals.mean()))
        self.assertEqual(
            result["quantile"], float(np.quantile(totals, 0.95, method="inverted_cdf"))
        )
        self.assertEqual(result["method"]["quantile_level_pct"], 95)
        self.assertNotEqual(
            result["simulated_mean"], float(np.mean(np.sum(draws * weights, axis=0)))
        )
        with self.assertRaisesRegex(ScientificWorkerError, "same observation count"):
            await self.bridge.invoke(
                "bernoulli_sum", probabilities=[0.5], weights=[10, 20]
            )

    async def test_external_stubs_work_with_monty_type_checking(self):
        result = await self.run_code(
            """import numpy as np
import pandas as pd
from scipy import stats
rows=data['query_1']
values=await np.array([float(r['x']) for r in rows])
draws=await np.random.binomial(1,0.5,size=3)
frame=await pd.DataFrame(rows)
{'mean':await np.mean(values),'rows':len(frame),'draws':draws}
""",
            type_check=True,
        )
        self.assertEqual(result["mean"], 3.0)
        self.assertEqual(result["rows"], 3)

    async def test_limits_empty_statistics_and_io_rejections_remain_enforced(self):
        with self.assertRaisesRegex(MontyError, "At least one observation"):
            await self.run_code(
                "import numpy as np\nrows=data['query_1']\nawait np.mean([r['x'] for r in rows])",
                [],
            )
        for snippet in (
            'await np.load("/etc/passwd")',
            "np.mean([1])",
            "np.random.default_rng(42)",
        ):
            with self.assertRaises(PythonPreflightError):
                await self.run_code(
                    "import numpy as np\nrows=data['query_1']\n" + snippet
                )
        for snippet in (
            'import pandas as pd\nawait pd.read_csv("/etc/passwd")',
            "import os",
            "import numpy.random",
        ):
            with self.assertRaises(PythonPreflightError):
                await self.run_code("rows=data['query_1']\n" + snippet)
        async with ScientificBridge(ScientificSettings(max_cells=4)) as limited:
            with self.assertRaisesRegex(ScientificWorkerError, "Sample cells"):
                await limited.invoke(
                    "numpy_random",
                    function="binomial",
                    parameters={"n": 1, "p": 0.5},
                    size=[3, 3],
                )

    def test_scalar_wire_codec_does_not_confuse_business_keys_with_tags(self):
        value = [
            {"__external_scalar__": "decimal", "value": "12.30"},
            {"x": Decimal(".10"), "day": date(2026, 6, 1)},
        ]
        self.assertEqual(decode_scalars(encode_scalars(value)), value)
        for root in self.bridge.modules().values():
            if root.stubs:
                ast.parse(root.stubs)


class SnapshotAndRepairTests(unittest.IsolatedAsyncioTestCase):
    def test_configured_observation_unit_binds_full_grain_and_method_before_validation(
        self,
    ):
        question = "What could next month's credit losses be? Show expected loss and a loss level exceeded in only 5% of scenarios."
        profile = get_domain_profile("1", tenant_id="1")
        value = executable_plan("1", "sql_python", declared=True)
        raw = risk_query().model_dump()
        raw["dimensions"] = [
            d
            for d in raw["dimensions"]
            if d["name"] in ("variant_id", "segment", "currency")
        ]
        value["steps"][1].update(
            logical_query=raw,
            output_contract={
                "required_columns": {
                    name: {"type": "any"}
                    for name in (
                        "variant_id",
                        "segment",
                        "currency",
                        "exposure_amount",
                        "probability_of_default",
                        "loss_given_default",
                    )
                },
                "grain": ["variant_id", "segment", "currency"],
                "population": {
                    "table": "cc_delinquency_monthly",
                    "source_key": "balance_inr",
                    "result_key": "balance_inr",
                },
            },
        )
        original = copy.deepcopy(value)
        bound = planner_output_type(
            "uncertainty", catalog(), profile.observation_policies(question)
        ).model_validate(value)
        step = bound.steps[1]
        self.assertEqual(
            set(step.output_contract.grain),
            {"month", "variant_id", "segment", "currency", "dpd_bucket"},
        )
        self.assertIsNone(step.output_contract.population)
        self.assertEqual(step.output_contract.min_rows, 1)
        self.assertEqual(bound.steps[-1].calculation.method, "scientific.bernoulli_sum")
        self.assertEqual(value, original)
        self.assertEqual(step.logical_query.base_table, "cc_delinquency_monthly")
        self.assertEqual(step.logical_query.time.selection, "latest_common")
        self.assertEqual(
            [item.name for item in step.logical_query.order_by],
            step.output_contract.grain,
        )

    def test_output_contract_feedback_names_missing_aliases_and_optional_population(
        self,
    ):
        value = executable_plan(declared=True)
        value["steps"][1]["logical_query"]["dimensions"] = [
            {
                "name": "customer_id",
                "field": {"table": "cc_customers", "column": "customer_id"},
            }
        ]
        value["steps"][1]["output_contract"]["required_columns"]["customer_id"] = {
            "type": "string"
        }
        value["steps"][1]["output_contract"]["grain"] = ["missing_identity"]
        with self.assertRaises(ValidationError) as raised:
            planner_output_type(None, catalog()).model_validate(value)
        diagnostics = PlannerDiagnostics(catalog())
        with self.assertRaises(ModelRetry):
            diagnostics.on_output_validate_error(
                type("Ctx", (), {"retry": 0})(),
                output_context=None,
                output=json.dumps(value),
                error=raised.exception,
            )
        context = diagnostics.latest_feedback.authority["query_output_requirements"][0]
        self.assertEqual(context["missing_grain_columns"], ["missing_identity"])
        self.assertIn("null", context["population_remedy"])

    def test_preserved_join_period_can_group_a_nullable_fact_but_not_the_reverse(self):
        query = risk_query()
        query.base_table = "cc_roll_rates_monthly"
        raw = query.model_dump()
        for item in raw["dimensions"]:
            item["field"]["table"] = "cc_roll_rates_monthly"
            if item["field"]["column"] == "dpd_bucket":
                item["field"]["column"] = "from_bucket"
        raw["time"] = {"selection": "latest", "tables": ["cc_roll_rates_monthly"]}
        compiled = compile_query(QueryIR.model_validate(raw), catalog())
        context = create_context(source_ids=["credit_cards"])
        compiled.verify_data(context)
        rows = context.sql(compiled.statement).to_pylist()
        self.assertEqual(len(rows), 36)
        self.assertEqual({r["month"] for r in rows}, {date(2026, 6, 1)})
        self.assertEqual(compiled.planner_receipt["backend"], "catalog")
        unsafe = risk_query().model_dump()
        unsafe["time"] = {"selection": "all"}
        unsafe["dimensions"][0]["field"]["table"] = "cc_roll_rates_monthly"
        with self.assertRaisesRegex(SemanticQueryError, "NON_ADDITIVE_MEASURE"):
            compile_query(QueryIR.model_validate(unsafe), catalog())

    def test_latest_snapshot_rejects_guessed_month_end_before_execution(self):
        query = risk_query()
        query.filters.append(
            Predicate.model_validate(
                {
                    "field": {"table": "cc_delinquency_monthly", "column": "month"},
                    "operator": "eq",
                    "value": "2026-06-30",
                }
            )
        )
        query = QueryIR.model_validate(query.model_dump())
        with self.assertRaises(SemanticQueryError) as raised:
            compile_query(query, catalog())
        self.assertEqual(
            raised.exception.diagnostic["error_code"], "CONFLICTING_SNAPSHOT_FILTER"
        )
        query.filters = [p for p in query.filters if p.field.column != "month"]
        compiled = compile_query(query, catalog())
        context = create_context(source_ids=["credit_cards"])
        compiled.verify_data(context)
        rows = context.sql(compiled.statement).to_pylist()
        self.assertEqual(len(rows), 36)
        self.assertEqual({row["month"] for row in rows}, {date(2026, 6, 1)})

    def test_fixed_catalog_horizon_is_inherited_without_mutating_caller(self):
        query = risk_query()
        query.metric_horizons = {}
        compiled = compile_query(query, catalog())
        self.assertEqual(
            compiled.logical_plan["metric_horizons"],
            {"probability_of_default": "one_month"},
        )
        self.assertEqual(query.metric_horizons, {})

    async def test_reopening_a_semantic_input_requires_a_changed_query(self):
        value = executable_plan("1", "sql_python", declared=True)
        value["query_language"] = "semantic"
        bad = count_query("cc_customers")
        bad["filters"] = [
            {
                "field": {"table": "cc_customers", "column": "customer_id"},
                "operator": "eq",
                "value": "CUST0001",
            }
        ]
        value["steps"][1]["logical_query"] = bad
        plan = ExecutionPlan.model_validate(value)
        calls = 0
        data = DataSources(create_context(source_ids=["credit_cards"]), "1", "1")

        def respond(messages, info):
            nonlocal calls
            calls += 1
            if calls == 1:
                return ModelResponse(
                    parts=[ToolCallPart("describe_sources", {"step_id": "s1"})]
                )
            if calls == 2:
                return ModelResponse(parts=[ToolCallPart("run_sql", {"step_id": "s2"})])
            if calls == 3:
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            "request_sql_repair",
                            {
                                "sql_step_id": "s2",
                                "reason": "The input counted only one customer instead of the full population.",
                            },
                        )
                    ]
                )
            if calls in (4, 5):
                tool = next(t for t in info.function_tools if t.name == "run_sql")
                self.assertIn("query", tool.parameters_json_schema["required"])
                if calls == 5:
                    self.assertIn(
                        "SQL_REPAIR_UNCHANGED", tool_returns(messages)[-1].content
                    )
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            "run_sql",
                            {
                                "step_id": "s2",
                                "query": (
                                    bad if calls == 4 else count_query("cc_customers")
                                ),
                            },
                        )
                    ]
                )
            if calls == 6:
                self.assertNotIn("query_1", data.results)
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            "run_python",
                            {
                                "step_id": "s3",
                                "result_ids": ["query_2"],
                                "code": "rows=data['query_2']\n{'count':rows[0]['n']}",
                            },
                        )
                    ]
                )
            return ModelResponse(parts=[TextPart("20 customers")])

        with (
            patch.object(data, "query", wraps=data.query) as queries,
            patch.dict("os.environ", {"AGENT_TOOL_RETRIES": "2"}),
        ):
            result = await execute_plan(
                "Count all customers", plan, data, FunctionModel(respond)
            )
        self.assertEqual(queries.call_count, 2)
        self.assertEqual(result.python_runs[0]["output"], {"count": 20})
        self.assertEqual(len(result.queries), 1)

    async def test_credit_loss_sql_then_scipy_numpy_and_pandas_in_real_monty(self):
        value = executable_plan("1", "sql_python", declared=True)
        value["query_language"] = "semantic"
        query = risk_query()
        value["steps"][1].update(
            logical_query=query.model_dump(),
            output_contract={
                "required_columns": {
                    "month": {"type": "date"},
                    "variant_id": {"type": "string"},
                    "segment": {"type": "string"},
                    "currency": {"type": "string"},
                    "dpd_bucket": {"type": "string"},
                    "exposure_amount": {"type": "numeric", "nullable": False},
                    "probability_of_default": {"type": "numeric", "nullable": False},
                    "loss_given_default": {"type": "numeric", "nullable": False},
                },
                "grain": ["month", "variant_id", "segment", "currency", "dpd_bucket"],
                "min_rows": 1,
            },
        )
        value["steps"][2].update(
            calculation={"kind": "uncertainty", "method": "scientific.sample"},
            result_contract={
                "required_fields": {
                    "mean_loss": "numeric",
                    "loss_95th_percentile": "numeric",
                    "cohort_count": "numeric",
                },
                "invariants": [],
            },
        )
        plan = ExecutionPlan.model_validate(value)
        data = DataSources(create_context(source_ids=["credit_cards"]), "1", "1")
        calls = 0
        code = """import numpy as np
import pandas as pd
from scipy import stats
from decimal import Decimal
rows=data['query_1']
probabilities=[float(r['probability_of_default']) for r in rows]
draws=await stats.bernoulli.rvs(probabilities,size=[1000,len(rows)],seed=42)
losses=[sum((r['exposure_amount']*r['loss_given_default']*Decimal(draw) for r,draw in zip(rows,scenario)),Decimal('0')) for scenario in draws]
frame=await pd.DataFrame([{'loss':loss} for loss in losses])
values=[r['loss'] for r in frame]
{'mean_loss':await np.mean(values),'loss_95th_percentile':await np.percentile(values,95,method='inverted_cdf'),'cohort_count':len(rows)}
"""

        def respond(messages, info):
            nonlocal calls
            calls += 1
            actions = [
                ("describe_sources", {"step_id": "s1"}),
                ("run_sql", {"step_id": "s2"}),
                (
                    "run_python",
                    {"step_id": "s3", "result_ids": ["query_1"], "code": code},
                ),
            ]
            if calls <= 3:
                name, args = actions[calls - 1]
                return ModelResponse(parts=[ToolCallPart(name, args)])
            return ModelResponse(parts=[TextPart("Credit-loss scenarios computed")])

        result = await execute_plan(
            "Forecast credit losses", plan, data, FunctionModel(respond)
        )
        output = result.python_runs[0]["output"]
        rows = result.sql_results["query_1"]["rows"]
        draws = stats.bernoulli.rvs(
            [float(r["probability_of_default"]) for r in rows],
            size=(1000, len(rows)),
            random_state=np.random.default_rng(np.random.SeedSequence([42, 0])),
        )
        losses = [
            sum(
                (
                    r["exposure_amount"] * r["loss_given_default"] * int(d)
                    for r, d in zip(rows, s)
                ),
                Decimal(0),
            )
            for s in draws
        ]
        self.assertEqual(output["cohort_count"], 36)
        self.assertEqual(output["mean_loss"], np.mean([float(x) for x in losses]))
        self.assertEqual(output["loss_95th_percentile"], float(sorted(losses)[949]))
        self.assertTrue(result.python_runs[0]["executed"])
        self.assertEqual(
            [c["operation"] for c in result.python_runs[0]["scientific_calls"]],
            ["sample", "pandas", "numpy", "numpy"],
        )

    async def test_bound_cohort_simulation_rejects_a_handwritten_replacement(self):
        question = "What could next month's credit losses be? Show expected loss and a loss level exceeded in only 5% of scenarios."
        value = executable_plan("1", "sql_python", declared=True)
        raw = risk_query().model_dump()
        raw["dimensions"] = [
            d
            for d in raw["dimensions"]
            if d["name"] in ("variant_id", "segment", "currency")
        ]
        value["steps"][1].update(
            logical_query=raw,
            output_contract={
                "required_columns": {
                    name: {"type": "any"}
                    for name in (
                        "variant_id",
                        "segment",
                        "currency",
                        "exposure_amount",
                        "probability_of_default",
                        "loss_given_default",
                    )
                },
                "grain": ["variant_id", "segment", "currency"],
            },
        )
        value["steps"][-1].update(
            calculation={"kind": "uncertainty", "method": "manual_simulation"},
            result_contract={
                "required_fields": {
                    "expected_loss": "numeric",
                    "loss_95th_percentile": "numeric",
                },
                "invariants": [],
            },
        )
        plan = planner_output_type(
            "uncertainty",
            catalog(),
            get_domain_profile("1", tenant_id="1").observation_policies(question),
        ).model_validate(value)
        data = DataSources(create_context(source_ids=["credit_cards"]), "1", "1")
        correct = """import scientific
rows=data['query_1']
probabilities=[float(r['probability_of_default']) for r in rows]
weights=[r['exposure_amount']*r['loss_given_default'] for r in rows]
result=await scientific.bernoulli_sum(probabilities,weights,n_simulations=1000,quantile=0.95,seed=42)
{'expected_loss':result['expected_value'],'simulated_mean':result['simulated_mean'],'loss_95th_percentile':result['quantile'],'method':result['method']}
"""
        calls = 0

        def respond(messages, info):
            nonlocal calls
            calls += 1
            if calls in (3, 4):
                tool = next(t for t in info.function_tools if t.name == "run_python")
                description = tool.parameters_json_schema["properties"]["code"][
                    "description"
                ]
                self.assertIn(
                    "Preserve required output field names exactly", description
                )
                self.assertIn('"expected_loss":"numeric"', description)
            if calls == 1:
                return ModelResponse(
                    parts=[ToolCallPart("describe_sources", {"step_id": "s1"})]
                )
            if calls == 2:
                return ModelResponse(parts=[ToolCallPart("run_sql", {"step_id": "s2"})])
            if calls == 3:
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            "run_python",
                            {
                                "step_id": "s3",
                                "result_ids": ["query_1"],
                                "code": "rows=data['query_1']\n{'expected_loss':1,'loss_95th_percentile':2}",
                            },
                        )
                    ]
                )
            if calls == 4:
                self.assertIn(
                    "Execute the published calculation method scientific.bernoulli_sum",
                    tool_returns(messages)[-1].content,
                )
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            "run_python",
                            {
                                "step_id": "s3",
                                "result_ids": ["query_1"],
                                "code": correct,
                            },
                        )
                    ]
                )
            return ModelResponse(
                parts=[TextPart("Conditional loss simulation completed")]
            )

        with patch.object(data, "query", wraps=data.query) as queries:
            result = await execute_plan(question, plan, data, FunctionModel(respond))
        self.assertEqual(queries.call_count, 1)
        rows = result.sql_results["query_1"]["rows"]
        self.assertEqual(len(rows), 36)
        self.assertEqual(
            len(
                {
                    (
                        r["month"],
                        r["variant_id"],
                        r["segment"],
                        r["currency"],
                        r["dpd_bucket"],
                    )
                    for r in rows
                }
            ),
            36,
        )
        expected = sum(
            float(r["probability_of_default"])
            * float(r["exposure_amount"] * r["loss_given_default"])
            for r in rows
        )
        self.assertAlmostEqual(
            result.python_runs[0]["output"]["expected_loss"], expected
        )
        call = result.python_runs[0]["scientific_calls"][0]
        self.assertEqual(call["method"]["observations"], 36)
        self.assertEqual(call["method"]["n_simulations"], 1000)
        self.assertEqual(call["operation"], "bernoulli_sum")


if __name__ == "__main__":
    unittest.main()
