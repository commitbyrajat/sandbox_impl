WITH base AS (SELECT f.*,p.variant_id,p.segment,v.variant_name FROM cc_monthly_financials f
JOIN cc_card_profiles p ON p.card_id=f.card_id
JOIN cc_products v ON v.variant_id=p.variant_id
WHERE f.month=(SELECT MAX(month) FROM cc_monthly_financials)), variants AS (SELECT month,variant_name,COUNT(DISTINCT card_id) AS cards,SUM(profit_inr) AS profit_inr,
SUM(profit_inr)/COUNT(DISTINCT card_id) AS profit_per_card_inr,
SUM(opex_inr)/COUNT(DISTINCT card_id) AS opex_per_card_inr,
SUM(credit_cost_inr) AS credit_cost_inr FROM base GROUP BY month,variant_name) SELECT a.month,a.variant_name,a.cards,a.profit_inr,a.profit_per_card_inr,a.opex_per_card_inr,a.credit_cost_inr,
COUNT(b.variant_name)=0 AS pareto_efficient FROM variants a LEFT JOIN variants b ON
b.profit_per_card_inr >= a.profit_per_card_inr AND b.opex_per_card_inr <= a.opex_per_card_inr AND
(b.profit_per_card_inr > a.profit_per_card_inr OR b.opex_per_card_inr < a.opex_per_card_inr)
GROUP BY a.month,a.variant_name,a.cards,a.profit_inr,a.profit_per_card_inr,a.opex_per_card_inr,a.credit_cost_inr
ORDER BY a.profit_per_card_inr DESC;
