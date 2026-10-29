# Sample credit card data

Entirely synthetic data, generated with random seed 42.

- `customers.parquet`: 20 customers. Primary key: `customer_id`.
- `credit_cards.parquet`: 40 cards, with 1–3 cards per customer. Primary key: `card_id`; joins to customers on `customer_id`. Only fictional last-four digits are stored.
- `transactions.parquet`: 1,000 transactions from January–June 2026. Primary key: `transaction_id`; joins to cards on `card_id` and customers on `customer_id`. Every card has multiple transactions.

Transactions include purchases, negative-amount refunds linked through `original_transaction_id`, and approved, declined, or pending statuses. Timestamps are UTC. Amounts and credit limits are decimal(12, 2). Amounts are in the transaction currency; credit limits are in the card billing currency. Filter to approved transactions and group by currency when calculating spending; no exchange rates are supplied. Files use Snappy compression.

## Synthetic profitability and credit-risk extension

Run `uv run python scripts/generate_credit_card_financials.py` from the project root to rebuild the following files. The generator is deterministic and reads, but does not rewrite, the original three files.

| Registered table / file | Grain | Rows |
| --- | --- | ---: |
| `cc_products` / `products.parquet` | Analytical variant | 4 |
| `cc_card_profiles` / `card_profiles.parquet` | Card | 40 |
| `cc_monthly_financials` / `monthly_financials.parquet` | Card and month, January–June 2026 | 240 |
| `cc_delinquency_monthly` / `delinquency_monthly.parquet` | Month, variant, segment, delinquency bucket | 288 |
| `cc_roll_rates_monthly` / `roll_rates_monthly.parquet` | Month, variant, segment, origin and destination bucket | 1,152 |

The analytical variants are **Diners Black**, **Platinum Plus**, **Everyday Rewards** and **Classic**. These are fictional product-economics labels assigned by card ID, independent of the original network/tier metadata. They do not describe actual issuer products. Profiles assign each customer consistently to SALARIED, SELF_EMPLOYED or AFFLUENT; these are synthetic commercial segments, not credit scores.

All extension amounts are INR. Monthly `approved_net_spend_inr` reconciles exactly to approved INR transactions by card/month, including negative refunds. Other balances, rates, fees, expenses and risk exposures are generated assumptions. They are not inferred from transactions or converted from the original billing-currency credit limits. Zero-spend card-months remain in the financial table.

Profit equals interest, fee and interchange income minus funding cost, credit cost, fixed/variable opex, rewards and network cost. Annual funding/credit rates use actual days/365. Opex is fixed plus variable operating expense. Diners Black is deliberately modeled with high fixed opex, rewards and credit costs to provide a loss-making variant for analysis. SELF_EMPLOYED has a higher synthetic credit-cost multiplier; this is an imposed data-generation assumption, not an empirical causal finding.

For spend +10%, only stored spend-linked income and expenses scale by 10%; balances, interest, fees, funding costs and fixed opex stay fixed. A +50bps annual credit-cost change applies `0.005 * average_receivables * days/365`. A +/-20bps funding change applies `+/-0.002 * funded_balance * days/365`. Scenario amounts should be distinguished from rounded accounting entries. Annualized contribution margin on receivables divides profit by receivables exposure-years; revenue profit margin divides profit by revenue.

Risk buckets are CURRENT, DPD_1_29, DPD_30_89 and DEFAULT. Each origin cohort has four destination rows whose `transitioned_balance_inr` sums exactly to the repeated `cohort_opening_balance_inr`. Use the opening balance once per cohort. Risk observations are synthetic aggregate cohorts, not individual borrower trajectories or a complete stock-flow accounting of lending, payments and recoveries. Their opening-cohort balances are set to the matching synthetic book exposures for illustration; consecutive monthly snapshots need not reconcile without additional business flows.

The next-month fixed-book expected-loss illustration applies the latest conditional default transition probabilities to matching non-default exposures and multiplies by LGD (0.60). DEFAULT is absorbing and is excluded from new defaults. No new lending, repayments, recoveries, changing rates, model fitting or validated forecast accuracy is assumed. P&L credit-cost accrual uses its separate annual-rate assumption and is not asserted to equal this forecast.

The agent must aggregate/filter before retrieving risk or financial rows: SQL results are capped at 200 rows, and Monty rejects truncated inputs. See `queries/credit_cards/` for working DataFusion queries covering `QUES.md`.

### PAT and revolving balances

Monthly financials now also store total ENR/ANR, revolving ENR/ANR, transactor ANR/share, revolve rate/yield, PBT, tax and PAT. The original pre-tax P&L values remain unchanged. Product rows add transactor-share and synthetic tax-rate assumptions. See [PAT and margin scenario definitions](../../docs/credit-card-pat-and-margin.md) for balance allocation, the synthetic 25% tax/loss-benefit policy, explicit held-flat costs, basis-point versus percentage-point units and reproducible SQL examples.
