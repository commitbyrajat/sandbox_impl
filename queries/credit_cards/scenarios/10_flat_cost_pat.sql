-- EXAMPLE inputs, not defaults: X=10 percent, Y=50 annual basis points.
-- Credit-cost basis: total ENR. Flat revolve share and ENR:ANR conversion.
WITH inputs AS (
 SELECT CAST('10' AS DECIMAL(8,4)) AS spend_growth_pct,
        CAST('50' AS DECIMAL(8,4)) AS credit_cost_increase_bps
), base AS (
 SELECT f.*,i.*,
 CAST(i.spend_growth_pct/100 AS DECIMAL(12,8)) AS g,
 CAST(i.credit_cost_increase_bps/10000 AS DECIMAL(12,8)) AS dc,
 CAST(f.days_in_period AS DECIMAL(8,4))/f.day_count_basis AS period_fraction
 FROM cc_monthly_financials f CROSS JOIN inputs i
 WHERE month=(SELECT MAX(month) FROM cc_monthly_financials)
), adjusted AS (
 SELECT *,
 revolving_enr_inr*(1+g) AS adjusted_revolving_enr_inr,
 revolving_anr_inr*(1+g) AS adjusted_revolving_anr_inr,
 ending_net_receivables_inr*(1+g) AS adjusted_total_enr_inr,
 CAST(ROUND(interest_income_inr*(1+g),2) AS DECIMAL(18,2)) AS adjusted_interest_income_inr,
 CAST(ROUND(interchange_income_inr*(1+g),2) AS DECIMAL(18,2)) AS adjusted_interchange_income_inr,
 CAST(ROUND(ending_net_receivables_inr*(1+g)*(annual_credit_cost_rate+dc)*days_in_period/day_count_basis,2) AS DECIMAL(18,2)) AS adjusted_credit_cost_inr,
 CASE WHEN g < -1 OR annual_credit_cost_rate+dc < 0 THEN 1 ELSE 0 END AS invalid_input
 FROM base
), profit AS (
 SELECT *,pbt_inr+adjusted_interest_income_inr-interest_income_inr
 +adjusted_interchange_income_inr-interchange_income_inr
 -(adjusted_credit_cost_inr-credit_cost_inr) AS adjusted_pbt_inr
 FROM adjusted
), taxed AS (
 SELECT *,CAST(ROUND(CASE WHEN adjusted_pbt_inr<0 AND NOT tax_loss_benefit_recognized
 THEN 0 ELSE adjusted_pbt_inr*tax_rate END,2) AS DECIMAL(18,2)) AS adjusted_tax_expense_inr
 FROM profit
)
SELECT month,currency,MAX(spend_growth_pct) AS spend_growth_pct,MAX(credit_cost_increase_bps) AS credit_cost_increase_bps,
SUM(fee_income_inr) AS held_flat_fee_income_inr,SUM(opex_inr) AS held_flat_opex_inr,
SUM(rewards_cost_inr) AS held_flat_rewards_cost_inr,SUM(funding_cost_inr) AS held_flat_funding_cost_inr,
SUM(network_cost_inr) AS held_flat_network_cost_inr,
SUM(invalid_input) AS invalid_input_rows,SUM(pbt_inr) AS base_pbt_inr,SUM(tax_expense_inr) AS base_tax_inr,SUM(pat_inr) AS base_pat_inr,
SUM(revolving_enr_inr) AS base_revolving_enr_inr,SUM(adjusted_revolving_enr_inr) AS adjusted_revolving_enr_inr,
SUM(adjusted_interest_income_inr-interest_income_inr) AS interest_impact_inr,
SUM(adjusted_interchange_income_inr-interchange_income_inr) AS interchange_impact_inr,
SUM(adjusted_credit_cost_inr-credit_cost_inr) AS credit_cost_impact_inr,
CASE WHEN SUM(invalid_input)=0 THEN SUM(adjusted_pbt_inr) END AS adjusted_pbt_inr,
CASE WHEN SUM(invalid_input)=0 THEN SUM(adjusted_tax_expense_inr) END AS adjusted_tax_inr,
CASE WHEN SUM(invalid_input)=0 THEN SUM(adjusted_pbt_inr-adjusted_tax_expense_inr) END AS adjusted_pat_inr
FROM taxed GROUP BY month,currency;
