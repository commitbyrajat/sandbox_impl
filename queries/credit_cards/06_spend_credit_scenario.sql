WITH base AS (SELECT f.*,p.variant_id,p.segment,v.variant_name FROM cc_monthly_financials f
JOIN cc_card_profiles p ON p.card_id=f.card_id
JOIN cc_products v ON v.variant_id=p.variant_id
WHERE f.month=(SELECT MAX(month) FROM cc_monthly_financials)), totals AS (SELECT month,SUM(profit_inr) AS base_profit_inr,
SUM(spend_linked_revenue_inr-spend_linked_cost_inr)*CAST('0.10' AS DECIMAL(8,6)) AS spend_growth_impact_inr,
SUM(average_receivables_inr*days_in_period/day_count_basis)*CAST('0.005' AS DECIMAL(8,6)) AS additional_credit_cost_inr
FROM base GROUP BY month) SELECT *,10 AS spend_growth_pct,50 AS credit_cost_increase_bps,base_profit_inr+spend_growth_impact_inr-additional_credit_cost_inr AS adjusted_profit_inr FROM totals;
