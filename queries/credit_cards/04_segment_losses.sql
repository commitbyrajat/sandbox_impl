WITH base AS (SELECT f.*,p.variant_id,p.segment,v.variant_name FROM cc_monthly_financials f
JOIN cc_card_profiles p ON p.card_id=f.card_id
JOIN cc_products v ON v.variant_id=p.variant_id
WHERE f.month=(SELECT MAX(month) FROM cc_monthly_financials)) SELECT month,segment,COUNT(DISTINCT card_id) AS cards,
SUM(profit_inr) AS profit_inr,SUM(credit_cost_inr) AS credit_cost_inr,
SUM(profit_inr)/COUNT(DISTINCT card_id) AS profit_per_card_inr,
SUM(approved_net_spend_inr)/COUNT(DISTINCT card_id) AS spend_per_card_inr,
SUM(credit_cost_inr)/SUM(average_receivables_inr*days_in_period/day_count_basis) AS annualized_credit_cost_rate
FROM base GROUP BY month,segment ORDER BY credit_cost_inr DESC;
