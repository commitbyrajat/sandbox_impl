import unittest
from decimal import Decimal
from pathlib import Path

from data_access import DataSources
from main import create_context


class FinancialDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sources = DataSources(
            create_context(source_ids=["credit_cards"]), tenant_id="1", workspace_id="1"
        )

    def rows(self, query):
        return self.sources.query(query)["rows"]

    def test_financial_keys_and_reconciliation(self):
        self.assertEqual(
            self.rows("SELECT COUNT(*) AS n FROM cc_monthly_financials")[0]["n"], 240
        )
        self.assertEqual(
            self.rows(
                "SELECT COUNT(*) AS n FROM cc_monthly_financials WHERE profit_inr <> total_revenue_inr-total_cost_inr OR opex_inr <> fixed_opex_inr+variable_opex_inr"
            )[0]["n"],
            0,
        )
        self.assertEqual(
            self.rows(
                "SELECT COUNT(*) AS n FROM (SELECT month,card_id FROM cc_monthly_financials GROUP BY month,card_id HAVING COUNT(*)<>1)"
            )[0]["n"],
            0,
        )
        self.assertEqual(
            self.rows(
                "SELECT COUNT(*) AS n FROM cc_monthly_financials WHERE total_revenue_inr <> interest_income_inr+fee_income_inr+interchange_income_inr OR total_cost_inr <> funding_cost_inr+credit_cost_inr+fixed_opex_inr+variable_opex_inr+rewards_cost_inr+network_cost_inr"
            )[0]["n"],
            0,
        )
        self.assertEqual(
            self.rows(
                "SELECT SUM(approved_net_spend_inr) AS n FROM cc_monthly_financials"
            )[0]["n"],
            self.rows(
                "SELECT SUM(amount) AS n FROM cc_transactions WHERE status='APPROVED' AND currency='INR'"
            )[0]["n"],
        )
        mismatches = self.rows(
            """WITH tx AS (SELECT card_id,CAST(date_trunc('month',transaction_timestamp) AS DATE) AS month,SUM(amount) AS value FROM cc_transactions WHERE status='APPROVED' AND currency='INR' GROUP BY card_id,month)
        SELECT COUNT(*) AS n FROM cc_monthly_financials f LEFT JOIN tx t ON f.card_id=t.card_id AND f.month=t.month WHERE f.approved_net_spend_inr<>COALESCE(t.value,0)"""
        )
        self.assertEqual(mismatches[0]["n"], 0)

    def test_risk_cohorts_conserve_opening_balances(self):
        bad = self.rows(
            """SELECT COUNT(*) AS n FROM (SELECT month,variant_id,segment,from_bucket FROM cc_roll_rates_monthly GROUP BY month,variant_id,segment,from_bucket HAVING COUNT(*)<>4 OR MIN(cohort_opening_balance_inr)<>MAX(cohort_opening_balance_inr) OR SUM(transitioned_balance_inr)<>MAX(cohort_opening_balance_inr))"""
        )
        self.assertEqual(bad[0]["n"], 0)
        self.assertEqual(
            self.rows("SELECT SUM(balance_inr) AS n FROM cc_delinquency_monthly")[0][
                "n"
            ],
            self.rows(
                "SELECT SUM(average_receivables_inr) AS n FROM cc_monthly_financials"
            )[0]["n"],
        )

    def test_all_nine_question_queries_and_scenarios(self):
        results = {
            p.name: self.rows(p.read_text())
            for p in sorted(Path("queries/credit_cards").glob("*.sql"))
        }
        self.assertEqual(len(results), 9)
        self.assertEqual(
            results["01_variant_profitability.sql"][0]["variant_name"], "Diners Black"
        )
        self.assertTrue(results["02_diners_black_drivers.sql"][0]["profit_inr"] < 0)
        self.assertTrue(
            any(r["pareto_efficient"] for r in results["03_sourcing_comparison.sql"])
        )
        scenario = results["06_spend_credit_scenario.sql"][0]
        self.assertIsInstance(scenario["adjusted_profit_inr"], Decimal)
        self.assertEqual(
            scenario["adjusted_profit_inr"],
            scenario["base_profit_inr"]
            + scenario["spend_growth_impact_inr"]
            - scenario["additional_credit_cost_inr"],
        )
        funds = results["07_funding_sensitivity.sql"][0]
        self.assertTrue(
            funds["profit_if_rate_up_20bps_inr"]
            < funds["base_profit_inr"]
            < funds["profit_if_rate_down_20bps_inr"]
        )
        delta = self.rows(
            "SELECT SUM(CASE WHEN month=DATE '2026-06-01' THEN profit_inr WHEN month=DATE '2026-05-01' THEN -profit_inr ELSE 0 END) AS n FROM cc_monthly_financials"
        )[0]["n"]
        self.assertEqual(
            sum(
                r["profit_change_inr"] for r in results["08_monthly_profit_bridge.sql"]
            ),
            delta,
        )
        movement = self.rows(
            Path("queries/credit_cards/support/risk_movements.sql").read_text()
        )
        self.assertEqual(len(movement), 2)
        self.assertTrue(movement[-1]["default_roll_rate_change"] > 0)
        self.assertTrue(movement[-1]["delinquent_balance_change_inr"] > 0)
        forecast = results["09_credit_cost_forecast.sql"][0]
        self.assertEqual(str(forecast["forecast_month"]), "2026-07-01")
        self.assertTrue(forecast["expected_credit_loss_inr"] > 0)

    def test_loss_display_preserves_negative_evidence(self):
        from summary_evidence import SummaryEvidence

        evidence = SummaryEvidence()
        evidence.collect(Decimal("-27094.29"))
        self.assertEqual(
            evidence.unsupported_values("Profit -₹27,094.29; loss of INR 27,094.29."),
            [],
        )
        self.assertIn(
            "27,094.29", evidence.unsupported_values("Positive profit INR 27,094.29.")
        )
        self.assertIn(
            "99,999.00", evidence.unsupported_values("Loss of INR 99,999.00.")
        )
