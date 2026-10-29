# PAT and revolving-margin scenarios

The credit-card workspace (tenant 1 / workspace 1) includes the following additional fields in `cc_monthly_financials`:

| Field | Meaning |
| --- | --- |
| `ending_net_receivables_inr` | Total ending net receivables (ENR) |
| `average_net_receivables_inr` | Total average net receivables (ANR) |
| `revolving_enr_inr`, `revolving_anr_inr` | Revolving portions of ENR and ANR |
| `transactor_anr_inr` | Non-revolving portion of ANR |
| `transactor_share`, `revolve_rate` | Balance shares as decimal fractions; in this stationary seed they sum to one |
| `annual_revolve_yield` | Annual yield on revolving ANR |
| `pbt_inr`, `tax_expense_inr`, `pat_inr` | Pre-tax contribution profit, allocated synthetic tax and profit after tax |
| `tax_rate`, `tax_loss_benefit_recognized` | Explicit synthetic tax policy |

Existing `profit_inr` remains pre-tax contribution profit and equals `pbt_inr`. Existing P&L amounts and raw transaction/card/customer files are preserved. The new seed sets ENR=ANR within each month; they remain different concepts. Revolving allocations retain Decimal precision, including sub-paise fractions, until currency amounts are rounded. The revolve yield is calibrated from the existing blended interest rate and revolving share, rounded to eight decimal places; this preserves the original recorded income rather than changing historical P&L. Scenario interest changes are anchored to recorded income to avoid zero-shock drift.

Regenerate with `uv run python scripts/generate_credit_card_financials.py`.

## Spend growth, flat revolve rate and higher credit cost

```sh
uv run agent.py --tenant-id 1 --workspace-id 1 --trace --max-tokens 8192 \
  "If approved INR spends grow by 10%, the revolve rate stays flat so revolving ENR scales with spend, and the annual credit-cost rate rises by 50 bps, what is adjusted PAT? Apply credit cost to total ENR. Hold rewards, other fee income, all opex, network cost and funding cost flat. Use the dataset's tax policy and show base and adjusted PBT, tax and PAT."
```

For symbolic inputs X and Y, define `g = X/100`, `dc = Y/10000` and `t = days_in_period/365`:

- Revolving ENR and total ENR scale by `1+g` at constant revolve rate. Assume the ENR-to-ANR conversion is unchanged, so revolving ANR scales by the same factor. End balances alone would not establish interest accrual without this assumption.
- Revolve yield stays fixed. Adjusted interest income is recorded interest income times `1+g`.
- The default assumes interchange's take-rate stays fixed, so interchange income also scales by `1+g`. An explicit user request to hold interchange flat overrides that assumption.
- Credit-cost rate applies to **adjusted total ENR** by default: `CC' = ENR*(1+g)*(base annual credit-cost rate + dc)*t`. Round the resulting expense to paise per card. Both the exposure growth and the rate shock affect credit cost. A question specifying revolving ENR instead needs that alternative basis disclosed and its baseline reconciled.
- `PBT' = base PBT + change in interest + change in interchange - change in credit cost`. All the other named P&L lines remain at their recorded values, including funding cost even though balances grow.
- Recalculate tax, then `PAT' = PBT' - tax'`.

The fixture uses a flat **25% synthetic tax rate with full recognition of tax benefits on losses**. Tax is rounded half-away-from-zero to paise per card, then aggregated. This is an explicit demonstration policy, not a statutory tax rule. Per-card rounding can differ by a few paise from applying tax once to aggregate profit. If loss benefits are disabled, tax on a negative PBT is zero. User-supplied tax assumptions take precedence.

[Reference SQL](../queries/credit_cards/scenarios/10_flat_cost_pat.sql) contains example inputs X=10 and Y=50. It yields June baseline PAT **INR 345.28** and adjusted PAT **INR 2,946.94** with these assumptions. The original, more general spend-growth question in `QUES.md` uses a different scenario that scales spend-linked expenses and holds interest/balances fixed; that legacy rule must not be applied to this explicit flat-cost PAT question. Missing numerical X/Y values call for a formula or clarification, not arbitrary values from the example.

For correct currency rounding, multiply exposure by annual rate and days before dividing by the day-count basis; pre-rounding `days/365` can move a half-paise tie and change PAT. The reference SQL and independent Decimal tests check this case.

## COF, revolve yield and transactor-share sensitivities

```sh
uv run agent.py --tenant-id 1 --workspace-id 1 --trace --max-tokens 8192 \
  "What is the combined impact on annualized portfolio net interest margin if COF rises 20 bps, revolve yield rises 25 bps, and transactor share rises 5 percentage points? Hold total ANR and funded balance fixed. Show the change in revolving ANR, individual income and funding effects, the yield/mix interaction, and base versus adjusted NIM."
```

The default portfolio margin for this question is **annualized net interest margin (NIM) on total ANR**:

`NIM = SUM(interest_income - funding_cost) / SUM(total_ANR * t)`

This differs from contribution profit divided by exposure-years, PAT margin, or profit divided by revenue. An explicit request for another margin definition must be handled using that definition.

Let `dc = cof_bps_change/10000`, `dy = revolve_yield_bps_change/10000`, and `dt = transactor_share_pct_pt_change/100`. For each selected card-month:

- `T' = T + dt`, where T is the balance-based transactor share.
- `revolving_ANR' = total_ANR * (1 - T')`.
- A **+5** percentage-point change adds **0.05** to T: 40% becomes 45%, not 42%. Revolving ANR falls by 5% of total ANR; it does not rise or fall by 5% of its original revolving amount.
- Total ANR and funded balance remain fixed. COF changes funding expense; revolve yield changes interest income. Other P&L lines are unchanged in this NIM sensitivity.
- Combined change in net interest income is:

```text
ΔNII = SUM((base_revolving_ANR * dy
            - total_ANR * dt * base_revolve_yield
            - total_ANR * dt * dy
            - funded_balance * dc) * t)
```

The third term is the yield/mix interaction. Adding isolated COF, yield and mix effects without it produces the wrong combined scenario. Use each row's yield/share/balance, rather than multiplying portfolio totals by unweighted average rates.

`NIM' = (base NII + ΔNII) / SUM(total_ANR * t)`, and `ΔNIM_bps = (NIM' - NIM) * 10000`. Anchor deltas to recorded income/funding costs so zero shocks reproduce the baseline despite rounded historical values and yields.

[Reference SQL](../queries/credit_cards/scenarios/11_portfolio_nim.sql) includes example inputs +20bps, +25bps and +5pp. For June it produces a combined NIM change of approximately **-178.57bps**. The example CTE is editable; its values are not agent defaults. Income sensitivities retain bounded Decimal precision before display rounding.

Reject a share outside [0,1] for any selected card, negative rates under this synthetic model, spend growth below -100%, or a zero NIM denominator. Do not clip shares, discard invalid cards or report a partially valid portfolio. Reference queries expose `invalid_input_rows` and return null for primary adjusted results if any row is invalid; the agent must report the validation failure instead of those diagnostic calculations.

Both questions are direct arithmetic and can use SQL alone. Tests cover zero-shock identity, PBT/tax/PAT reconciliation, held-flat costs, independent Decimal scenario results, percentage-point direction, combined interactions and invalid inputs. These are fixture/calculation checks, not proof that every LLM-generated query follows the definitions.

## Published configured query for the documented NIM request

The documented combined-NIM question now uses a workspace-scoped SQL recipe configured under `profiles.credit_cards.sql_recipes.portfolio_nim` in `domain_instructions.json`. The recipe recognizes the documented full question (including fixed total ANR/funded balance and the requested outputs), with variable numeric changes and rising/falling directions. It does not discard additional period filters, grouping requests or other objectives: unmatched requests continue through normal planning.

After intent classification, the planner publishes a schema step and one SQL step, plus `sql_recipe.recipe_id`, the three signed parameters and the template's SHA-256. The executor validates the workspace and template hash, substitutes validated Decimal parameters into [the approved template](../queries/credit_cards/scenarios/11_portfolio_nim.template.sql), and runs it through the existing read-only, scoped DataFusion access. It does not ask an LLM to rewrite the formula or start Monty. A changed template requires a newly published plan.

The checked query filters the latest financial month, uses actual ANR columns rather than ENR aliases, applies row-level yield and share effects before aggregation, and anchors adjusted NII to recorded base NII exactly once. It returns the signed inputs, decimal conversions, ANR change, component effects, combined NII, NIM rates/percentages and rounded display/magnitude fields for summarization. Invalid share/rate inputs stop the calculation rather than producing a partial adjusted portfolio.

For the documented June +20bps COF / +25bps yield / +5pp share scenario, expected results are:

| Measure | Result |
| --- | ---: |
| Base annualized NIM | 16.3053% |
| Adjusted annualized NIM | 14.5195% |
| NIM change | -178.57bps |
| Revolving ANR change | INR -160,312.50 |
| COF NII impact | INR -527.05 |
| Yield NII impact | INR +445.11 |
| Mix NII impact | INR -4,591.00 |
| Yield/mix interaction | INR -32.94 |
| Combined NII impact | INR -4,705.89 |

The configuration and template contain domain-specific rules; `query_recipes.py`, planning, publication and execution provide generic recipe handling. Other question types retain their existing planning path. Automated checks cover signed parameter extraction, scope isolation, numeric bounds, hash mismatches, invalid share scenarios, no Monty/model rewrite, single-query execution and independence from ENR columns.
