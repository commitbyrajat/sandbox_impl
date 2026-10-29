"""Rebuild synthetic INR profitability/risk extensions without changing original Parquet files."""

from calendar import monthrange
from collections import defaultdict
from datetime import date
from decimal import ROUND_HALF_UP
from decimal import Decimal as D
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/parquet"
CENT = D(".01")


def money(x):
    return D(x).quantize(CENT)


def write(name, rows):
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, OUT / (name + ".parquet"), compression="snappy")
    print(name, table.num_rows)


def generate():
    cards = sorted(
        pq.read_table(OUT / "credit_cards.parquet").to_pylist(),
        key=lambda r: r["card_id"],
    )
    transactions = pq.read_table(OUT / "transactions.parquet").to_pylist()
    spending = defaultdict(lambda: D(0))
    for t in transactions:
        if t["status"] == "APPROVED" and t["currency"] == "INR":
            spending[t["card_id"], t["transaction_timestamp"].month] += t["amount"]
    names = ["Diners Black", "Platinum Plus", "Everyday Rewards", "Classic"]
    products = []
    for i, name in enumerate(names):
        products.append(
            dict(
                variant_id=f"V{i+1}",
                variant_name=name,
                annual_interest_rate=D([".15", ".24", ".27", ".29"][i]),
                annual_funding_rate=D(".075"),
                annual_credit_cost_rate=D([".12", ".025", ".035", ".04"][i]),
                monthly_fee_inr=D([100, 180, 80, 40][i]),
                fixed_opex_per_card_inr=D([1800, 220, 150, 120][i]),
                interchange_rate=D(".015"),
                rewards_rate=D([".035", ".008", ".012", ".003"][i]),
                variable_opex_rate=D(".002"),
                network_cost_rate=D(".001"),
                loss_given_default=D(".60"),
                transactor_share=D([".40", ".35", ".30", ".25"][i]),
                tax_rate=D(".25"),
            )
        )
    profiles = []
    facts = []
    groups = defaultdict(lambda: D(0))
    for i, c in enumerate(cards):
        product = products[i // 10]
        customer_num = int(c["customer_id"].removeprefix("CUST"))
        segment = ["SALARIED", "SELF_EMPLOYED", "AFFLUENT"][(customer_num - 1) % 3]
        profiles.append(
            dict(
                card_id=c["card_id"],
                customer_id=c["customer_id"],
                variant_id=product["variant_id"],
                segment=segment,
            )
        )
        risk_factor = {
            "SALARIED": D(".8"),
            "SELF_EMPLOYED": D("1.8"),
            "AFFLUENT": D(".6"),
        }[segment]
        for m in range(1, 7):
            days = monthrange(2026, m)[1]
            factor = D(days) / 365
            balance = money(D(45000 + (i % 9) * 7000) * (1 + D(m - 1) * D(".025")))
            spend = spending[c["card_id"], m]
            positive_spend = max(spend, D(0))
            r = dict(
                month=date(2026, m, 1),
                card_id=c["card_id"],
                currency="INR",
                days_in_period=days,
                day_count_basis=365,
                approved_net_spend_inr=spend,
                average_receivables_inr=balance,
                funded_balance_inr=balance,
                annual_funding_rate=product["annual_funding_rate"],
                annual_credit_cost_rate=product["annual_credit_cost_rate"]
                * risk_factor
                * (1 + D(m - 1) * D(".04")),
                interest_income_inr=money(
                    balance * product["annual_interest_rate"] * factor
                ),
                fee_income_inr=product["monthly_fee_inr"],
                interchange_income_inr=money(spend * product["interchange_rate"]),
                funding_cost_inr=money(
                    balance * product["annual_funding_rate"] * factor
                ),
                fixed_opex_inr=product["fixed_opex_per_card_inr"],
                variable_opex_inr=money(positive_spend * product["variable_opex_rate"]),
                rewards_cost_inr=money(positive_spend * product["rewards_rate"]),
                network_cost_inr=money(positive_spend * product["network_cost_rate"]),
            )
            r["credit_cost_inr"] = money(
                balance * r["annual_credit_cost_rate"] * factor
            )
            r["total_revenue_inr"] = sum(
                r[k]
                for k in [
                    "interest_income_inr",
                    "fee_income_inr",
                    "interchange_income_inr",
                ]
            )
            r["opex_inr"] = r["fixed_opex_inr"] + r["variable_opex_inr"]
            r["total_cost_inr"] = sum(
                r[k]
                for k in [
                    "funding_cost_inr",
                    "credit_cost_inr",
                    "opex_inr",
                    "rewards_cost_inr",
                    "network_cost_inr",
                ]
            )
            r["profit_inr"] = r["total_revenue_inr"] - r["total_cost_inr"]
            # This stationary-within-month fixture sets ENR equal to ANR.
            # The revolving yield is calibrated to preserve the existing interest/P&L.
            r["ending_net_receivables_inr"] = balance
            r["average_net_receivables_inr"] = balance
            r["transactor_share"] = product["transactor_share"]
            r["revolve_rate"] = D(1) - r["transactor_share"]
            r["revolving_enr_inr"] = balance * r["revolve_rate"]
            r["revolving_anr_inr"] = r["revolving_enr_inr"]
            r["transactor_anr_inr"] = balance - r["revolving_anr_inr"]
            r["annual_revolve_yield"] = (
                product["annual_interest_rate"] / r["revolve_rate"]
            ).quantize(D(".00000001"))
            r["tax_rate"] = product["tax_rate"]
            r["tax_loss_benefit_recognized"] = True
            r["pbt_inr"] = r["profit_inr"]
            r["tax_expense_inr"] = (r["pbt_inr"] * r["tax_rate"]).quantize(
                CENT, rounding=ROUND_HALF_UP
            )
            r["pat_inr"] = r["pbt_inr"] - r["tax_expense_inr"]
            r["spend_linked_revenue_inr"] = r["interchange_income_inr"]
            r["spend_linked_cost_inr"] = (
                r["variable_opex_inr"] + r["rewards_cost_inr"] + r["network_cost_inr"]
            )
            facts.append(r)
            groups[m, product["variant_id"], segment] += balance
    buckets = ["CURRENT", "DPD_1_29", "DPD_30_89", "DEFAULT"]
    balances = []
    transitions = []
    for (m, variant, segment), total in sorted(groups.items()):
        stress = D(m - 1) * D(".002") + (
            {"SALARIED": D("0"), "SELF_EMPLOYED": D(".03"), "AFFLUENT": D("-.01")}[
                segment
            ]
        )
        shares = [
            D(".88") - stress * 2,
            D(".07") + stress,
            D(".035") + stress,
            D(".015"),
        ]
        amounts = [money(total * s) for s in shares[:3]]
        amounts.append(total - sum(amounts))
        for b, amount in zip(buckets, amounts):
            balances.append(
                dict(
                    month=date(2026, m, 1),
                    variant_id=variant,
                    segment=segment,
                    currency="INR",
                    dpd_bucket=b,
                    balance_inr=amount,
                    loss_given_default=D(".60"),
                )
            )
            worsening = D(m - 1) * D(".001") + (
                D(".015") if segment == "SELF_EMPLOYED" else D(0)
            )
            rates = {
                "CURRENT": [
                    D(".96") - worsening,
                    D(".03"),
                    D(".008"),
                    D(".002") + worsening,
                ],
                "DPD_1_29": [
                    D(".25"),
                    D(".55") - worsening,
                    D(".15"),
                    D(".05") + worsening,
                ],
                "DPD_30_89": [
                    D(".08"),
                    D(".12"),
                    D(".55") - worsening,
                    D(".25") + worsening,
                ],
                "DEFAULT": [D(0), D(0), D(0), D(1)],
            }[b]
            moved = [money(amount * r) for r in rates[:3]]
            moved.append(amount - sum(moved))
            for dest, exposure in zip(buckets, moved):
                transitions.append(
                    dict(
                        month=date(2026, m, 1),
                        variant_id=variant,
                        segment=segment,
                        currency="INR",
                        from_bucket=b,
                        to_bucket=dest,
                        cohort_opening_balance_inr=amount,
                        transitioned_balance_inr=exposure,
                    )
                )
    for name, rows in [
        ("products", products),
        ("card_profiles", profiles),
        ("monthly_financials", facts),
        ("delinquency_monthly", balances),
        ("roll_rates_monthly", transitions),
    ]:
        write(name, rows)


if __name__ == "__main__":
    generate()
