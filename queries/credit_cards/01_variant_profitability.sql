WITH base AS (SELECT f.*,p.variant_id,p.segment,v.variant_name FROM cc_monthly_financials f
JOIN cc_card_profiles p ON p.card_id=f.card_id
JOIN cc_products v ON v.variant_id=p.variant_id
WHERE f.month=(SELECT MAX(month) FROM cc_monthly_financials)) SELECT month,variant_name,COUNT(DISTINCT card_id) AS cards,SUM(profit_inr) AS profit_inr,
SUM(profit_inr)/COUNT(DISTINCT card_id) AS profit_per_card_inr,
SUM(opex_inr)/COUNT(DISTINCT card_id) AS opex_per_card_inr,
SUM(credit_cost_inr) AS credit_cost_inr FROM base GROUP BY month,variant_name ORDER BY profit_inr;
