WITH base AS (SELECT f.*,p.variant_id,p.segment,v.variant_name FROM cc_monthly_financials f
JOIN cc_card_profiles p ON p.card_id=f.card_id
JOIN cc_products v ON v.variant_id=p.variant_id
WHERE f.month=(SELECT MAX(month) FROM cc_monthly_financials)) SELECT month,variant_name,SUM(interest_income_inr) AS interest_income_inr,SUM(fee_income_inr) AS fee_income_inr,SUM(interchange_income_inr) AS interchange_income_inr,SUM(funding_cost_inr) AS funding_cost_inr,SUM(credit_cost_inr) AS credit_cost_inr,SUM(fixed_opex_inr) AS fixed_opex_inr,SUM(variable_opex_inr) AS variable_opex_inr,SUM(rewards_cost_inr) AS rewards_cost_inr,SUM(network_cost_inr) AS network_cost_inr,SUM(profit_inr) AS profit_inr FROM base WHERE variant_name='Diners Black' GROUP BY month,variant_name;
