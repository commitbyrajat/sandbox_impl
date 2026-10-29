# Credit-card financial question coverage

`QUES.md` contains nine business questions. All use **tenant 1 / workspace 1**. Its `cli.py` command examples describe question text; this repository's executable is `agent.py`:

```sh
uv run agent.py --tenant-id 1 --workspace-id 1 --trace --max-tokens 8192 \
  "what factors in Diners Black are pulling down profitability"
```

The credit-card profile in `domain_instructions.json` now defines the terminology, data grains, period defaults, financial formulas and forecast assumptions. `datasource.json` registers the extension tables. General planning and execution remain domain agnostic.

| Question in QUES.md | Analysis | Reference SQL |
| --- | --- | --- |
| Which variant is dragging profitability? | Rank latest-month contribution profit, disclose negative versus lower positive profit and profit/card. | `queries/credit_cards/01_variant_profitability.sql` |
| What factors in Diners Black pull profitability down? | Show its income/expense components; compare costs and per-card economics using the same period. Avoid causal claims. | `queries/credit_cards/02_diners_black_drivers.sql` |
| Which variant is good for sourcing with high profitability and less opex? | Compare profit/card and opex/card with credit cost and scale; show Pareto-efficient alternatives instead of inventing weights. | `queries/credit_cards/03_sourcing_comparison.sql` |
| Which segment causes high losses? | Rank commercial segments by credit cost and normalized rate/per-card burden; report association, not causation. | `queries/credit_cards/04_segment_losses.sql` |
| Segment profitability/card and spend/card comparison | Aggregate INR card-month facts by profile segment; divide by distinct cards including zero-spend cards. | `queries/credit_cards/05_segment_per_card.sql` |
| Spend +10% and credit cost +50bps | Adjust spend-linked contribution and annual credit cost on receivables; show baseline, impacts and adjusted profit. | `queries/credit_cards/06_spend_credit_scenario.sql` |
| Funding cost changes 20bps | Show both increase and decrease cases if direction is unspecified; calculate profit and annualized receivables-margin impacts. | `queries/credit_cards/07_funding_sensitivity.sql` |
| Which P&L parameters drove month-on-month change? | Signed income/expense bridge from the preceding to latest month; component changes reconcile to total profit change. | `queries/credit_cards/08_monthly_profit_bridge.sql` |
| Forecast credit cost from delinquency and roll-rate movements | Inspect latest/prior bucket balances and conditional default rates; forecast one month on a fixed book using latest exposures, transition rates and LGD. | `queries/credit_cards/09_credit_cost_forecast.sql` and `support/risk_movements.sql` |

These analyses are primarily aggregations, ratios, linear scenarios or a one-step matrix calculation, so SQL can answer them. Python is appropriate for a separately requested iterative or resampling algorithm with published assumptions; a multi-horizon calculation alone does not necessarily require it. It is not required simply because a question mentions a scenario or forecast.

When no period is specified, financial/risk questions use the latest financial month (June 2026 in the seed); month-on-month comparisons use May versus June. Explicit user periods take precedence. Default profit is pre-tax contribution profit. Distinct-card and card-month denominators are different and must be labeled. The later PAT extension supplies an explicit synthetic tax policy and recorded PBT/tax/PAT; see [PAT and margin scenarios](credit-card-pat-and-margin.md). No actual acquisition-conversion model, capital allocation or statistically validated future credit-loss model is present.

The original spending data is unchanged. The extension is fully synthetic; in particular the Diners Black economics do not describe a real product. Regeneration details and all assumptions are in [the dataset README](../data/parquet/README.md).

Reference SQL is executed by local DataFusion tests. Those tests verify data keys, spending/P&L reconciliation, transition-cohort conservation, scenario direction and a balanced P&L bridge; they do not establish that every live LLM-generated query is correct. Published plans and executed SQL remain reviewable with `--trace`.

See the [README execution-mode matrix](../README.md#execution-mode-analysis-of-recent-financial-questions) for the SQL-only classification of all eleven recent financial questions and examples of additional requirements that justify SQL + Python.
