import unittest
from decimal import ROUND_HALF_UP
from decimal import Decimal as D
from pathlib import Path

from main import create_context

PAT = Path("queries/credit_cards/scenarios/10_flat_cost_pat.sql").read_text()
NIM = Path("queries/credit_cards/scenarios/11_portfolio_nim.sql").read_text()


def pat_query(x, y):
    return PAT.replace("'10' AS DECIMAL", f"'{x}' AS DECIMAL").replace(
        "'50' AS DECIMAL", f"'{y}' AS DECIMAL"
    )


def nim_query(cof, yield_, share):
    return (
        NIM.replace("'20' AS DECIMAL", f"'{cof}' AS DECIMAL")
        .replace("'25' AS DECIMAL", f"'{yield_}' AS DECIMAL")
        .replace("'5' AS DECIMAL", f"'{share}' AS DECIMAL")
    )


def money(x):
    return x.quantize(D(".01"), rounding=ROUND_HALF_UP)


class ScenarioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ctx = create_context(source_ids=["credit_cards"])
        cls.base = cls.ctx.sql(
            "SELECT * FROM cc_monthly_financials WHERE month=DATE '2026-06-01'"
        ).to_pylist()

    def row(self, sql):
        return self.ctx.sql(sql).to_pylist()[0]

    def test_base_balances_and_pat(self):
        for r in self.ctx.table("cc_monthly_financials").to_pylist():
            self.assertEqual(
                r["revolving_anr_inr"] + r["transactor_anr_inr"],
                r["average_net_receivables_inr"],
            )
            self.assertEqual(
                r["revolving_anr_inr"],
                r["average_net_receivables_inr"] * (1 - r["transactor_share"]),
            )
            self.assertEqual(r["pat_inr"], r["pbt_inr"] - r["tax_expense_inr"])
            self.assertEqual(r["tax_expense_inr"], money(r["pbt_inr"] * r["tax_rate"]))

    def test_zero_shocks_reproduce_recorded_results(self):
        p = self.row(pat_query(0, 0))
        self.assertEqual(p["base_pat_inr"], p["adjusted_pat_inr"])
        self.assertEqual(p["base_pbt_inr"], p["adjusted_pbt_inr"])
        n = self.row(nim_query(0, 0, 0))
        self.assertEqual(n["base_nim_rate"], n["adjusted_nim_rate"])
        self.assertEqual(n["base_revolving_anr_inr"], n["adjusted_revolving_anr_inr"])

    def test_flat_cost_pat_matches_independent_decimal_calculation(self):
        actual = self.row(pat_query(10, 50))
        expected = D(0)
        for r in self.base:
            t = D(r["days_in_period"]) / r["day_count_basis"]
            credit = money(
                r["ending_net_receivables_inr"]
                * D("1.10")
                * (r["annual_credit_cost_rate"] + D(".005"))
                * t
            )
            pbt = (
                r["pbt_inr"]
                + money(r["interest_income_inr"] * D("1.10"))
                - r["interest_income_inr"]
                + money(r["interchange_income_inr"] * D("1.10"))
                - r["interchange_income_inr"]
                - (credit - r["credit_cost_inr"])
            )
            expected += pbt - money(pbt * r["tax_rate"])
        self.assertEqual(actual["adjusted_pat_inr"], expected)
        for field in [
            "fee_income_inr",
            "opex_inr",
            "rewards_cost_inr",
            "funding_cost_inr",
            "network_cost_inr",
        ]:
            self.assertEqual(
                actual["held_flat_" + field], sum(r[field] for r in self.base)
            )

    def test_percentage_points_and_interaction(self):
        actual = self.row(nim_query(20, 25, 5))
        self.assertEqual(
            actual["base_revolving_anr_inr"] - actual["adjusted_revolving_anr_inr"],
            sum(r["average_net_receivables_inr"] for r in self.base) * D(".05"),
        )
        expected = sum(
            (
                (
                    r["revolving_anr_inr"] * D(".0025")
                    - r["average_net_receivables_inr"]
                    * D(".05")
                    * r["annual_revolve_yield"]
                    - r["average_net_receivables_inr"] * D(".05") * D(".0025")
                    - r["funded_balance_inr"] * D(".002")
                )
                * D(r["days_in_period"])
                / r["day_count_basis"]
                for r in self.base
            ),
            D(0),
        )
        self.assertLess(abs(actual["combined_nii_impact_inr"] - expected), D(".001"))
        self.assertEqual(
            actual["combined_nii_impact_inr"],
            sum(
                actual[k]
                for k in [
                    "cof_nii_impact_inr",
                    "yield_nii_impact_inr",
                    "mix_nii_impact_inr",
                    "yield_mix_interaction_inr",
                ]
            ),
        )
        self.assertLess(actual["yield_mix_interaction_inr"], 0)

    def test_direction_and_invalid_inputs(self):
        self.assertLess(self.row(nim_query(20, 0, 0))["nim_change_bps"], 0)
        self.assertGreater(self.row(nim_query(0, 25, 0))["nim_change_bps"], 0)
        self.assertLess(self.row(nim_query(0, 0, 5))["nim_change_bps"], 0)
        self.assertGreater(self.row(nim_query(0, 0, -5))["nim_change_bps"], 0)
        bad = self.row(
            nim_query(0, 0, 70)
        )  # Some cards invalid, even if portfolio-average share could pass.
        self.assertGreater(bad["invalid_input_rows"], 0)
        self.assertIsNone(bad["adjusted_nim_rate"])
        self.assertIsNone(self.row(pat_query(-101, 0))["adjusted_pat_inr"])
