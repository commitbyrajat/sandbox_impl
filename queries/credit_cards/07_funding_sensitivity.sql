WITH base AS (SELECT f.*,p.variant_id,p.segment,v.variant_name FROM cc_monthly_financials f
JOIN cc_card_profiles p ON p.card_id=f.card_id
JOIN cc_products v ON v.variant_id=p.variant_id
WHERE f.month=(SELECT MAX(month) FROM cc_monthly_financials)), totals AS (SELECT month,SUM(profit_inr) AS base_profit_inr,
SUM(funded_balance_inr*days_in_period/day_count_basis)*CAST('0.002' AS DECIMAL(8,6)) AS additional_funding_cost_inr,
SUM(average_receivables_inr*days_in_period/day_count_basis) AS receivable_exposure_years
FROM base GROUP BY month) SELECT *,20 AS funding_rate_change_bps,base_profit_inr-additional_funding_cost_inr AS profit_if_rate_up_20bps_inr,
base_profit_inr+additional_funding_cost_inr AS profit_if_rate_down_20bps_inr,
-additional_funding_cost_inr/receivable_exposure_years AS annualized_margin_change_if_rate_up FROM totals;
