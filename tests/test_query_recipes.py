import unittest
from decimal import Decimal as D
from unittest.mock import patch

from pydantic import ValidationError
from pydantic_ai.models.function import FunctionModel

from data_access import DataSources
from domain_instructions import get_domain_profile
from execution import execute_plan
from main import create_context
from planning import QuestionIntent, build_plan
from query_recipes import match_sql_recipe, render_sql_recipe
from summary_evidence import prepare_evidence

QUESTION = "What is the combined impact on annualized portfolio net interest margin if COF rises 20 bps, revolve yield rises 25 bps, and transactor share rises 5 percentage points? Hold total ANR and funded balance fixed. Show the change in revolving ANR, component effects, the yield-mix interaction, and base versus adjusted NIM."


def forbidden_model(*args):
    raise AssertionError("Recipe must not ask a model to rewrite checked math")


class RecipeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.profile = get_domain_profile("1", tenant_id="1")
        self.model = FunctionModel(forbidden_model)

    async def plan(self, q=QUESTION):
        return await build_plan(
            q,
            QuestionIntent(intent="advanced_analysis", objective=q),
            tenant_id="1",
            workspace_id="1",
            model=self.model,
        )

    async def test_exact_request_one_correct_query_no_monty_or_executor_llm(self):
        plan = await self.plan()
        self.assertEqual([s.action for s in plan.steps], ["schema", "sql"])
        self.assertEqual(plan.execution_mode, "sql")
        self.assertEqual(
            plan.sql_recipe.parameters["transactor_share_pct_pt_change"], D(5)
        )
        sources = DataSources(
            create_context(source_ids=["credit_cards"]), tenant_id="1", workspace_id="1"
        )
        with patch(
            "execution.AsyncMonty",
            side_effect=AssertionError("No Monty for SQL recipe"),
        ):
            result = await execute_plan(QUESTION, plan, sources, self.model)
        self.assertEqual(len(result.queries), 1)
        rows = next(iter(result.sql_results.values()))["rows"]
        self.assertEqual(len(rows), 1)
        r = rows[0]
        self.assertEqual(str(r["month"]), "2026-06-01")
        self.assertEqual(r["change_revolving_anr_inr"], D("-160312.50"))
        self.assertEqual(
            r["adjusted_nii_inr"], r["base_nii_inr"] + r["combined_nii_impact_inr"]
        )
        self.assertLess(abs(r["nim_change_bps"] - D("-178.572991")), D(".00001"))
        self.assertEqual(r["invalid_input_rows"], 0)
        evidence = prepare_evidence(result, self.profile, question=QUESTION)
        self.assertEqual(
            evidence.unsupported_values(
                "COF +20 bps, yield +25 bps, share +0.05; revolving ANR change INR -160,312.50; NIM change -178.57 bps."
            ),
            [],
        )

    async def test_parameters_direction_and_unsupported_extra_filters(self):
        q = (
            QUESTION.replace("COF rises 20", "COF falls 10")
            .replace("yield rises 25", "yield decreases 15")
            .replace("share rises 5", "share falls 2")
        )
        call = match_sql_recipe(q, self.profile.sql_recipes)
        self.assertEqual(
            call.parameters,
            {
                "cof_bps_change": D(-10),
                "revolve_yield_bps_change": D(-15),
                "transactor_share_pct_pt_change": D(-2),
            },
        )
        self.assertIsNone(
            match_sql_recipe(
                QUESTION + " Use January 2026 only.", self.profile.sql_recipes
            )
        )
        self.assertIsNone(
            match_sql_recipe(
                QUESTION, get_domain_profile("2", tenant_id="2").sql_recipes
            )
        )

    async def test_recipe_source_and_parameter_validation(self):
        call = match_sql_recipe(QUESTION, self.profile.sql_recipes)
        with self.assertRaisesRegex(ValueError, "changed after planning"):
            render_sql_recipe(
                call.model_copy(update={"template_sha256": "wrong"}),
                self.profile.sql_recipes,
            )
        with self.assertRaisesRegex(ValueError, "not configured"):
            render_sql_recipe(call, {})
        with self.assertRaises(ValidationError):
            match_sql_recipe(
                QUESTION.replace("20 bps", "10000 bps"), self.profile.sql_recipes
            )

    async def test_invalid_share_blocks_adjusted_answer(self):
        q = QUESTION.replace("share rises 5", "share rises 70")
        plan = await self.plan(q)
        sources = DataSources(
            create_context(source_ids=["credit_cards"]), tenant_id="1", workspace_id="1"
        )
        with self.assertRaisesRegex(ValueError, "Invalid NIM scenario"):
            await execute_plan(q, plan, sources, self.model)

    async def test_enr_changes_do_not_change_anr_based_nim(self):
        from pathlib import Path

        import pyarrow as pa
        import pyarrow.parquet as pq
        from datafusion import SessionContext

        call = match_sql_recipe(QUESTION, self.profile.sql_recipes)
        sql = render_sql_recipe(call, self.profile.sql_recipes)
        data = pq.read_table(Path("data/parquet/monthly_financials.parquet"))

        def evaluate(table):
            ctx = SessionContext()
            ctx.register_record_batches("cc_monthly_financials", [table.to_batches()])
            return ctx.sql(sql).to_pylist()

        baseline = evaluate(data)
        for name in ["ending_net_receivables_inr", "revolving_enr_inr"]:
            index = data.schema.get_field_index(name)
            dtype = pa.decimal128(
                data.schema.field(index).type.precision + 1,
                data.schema.field(index).type.scale,
            )
            data = data.set_column(
                index,
                name,
                pa.array([x * 10 for x in data[name].to_pylist()], type=dtype),
            )
        self.assertEqual(baseline, evaluate(data))
