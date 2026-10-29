WITH probabilities AS (
 SELECT month,variant_id,segment,currency,from_bucket,
 SUM(transitioned_balance_inr)/MAX(cohort_opening_balance_inr) AS default_probability
 FROM cc_roll_rates_monthly WHERE month=(SELECT MAX(month) FROM cc_roll_rates_monthly)
 AND to_bucket='DEFAULT' AND from_bucket <> 'DEFAULT'
 GROUP BY month,variant_id,segment,currency,from_bucket)
SELECT d.month AS as_of_month,CAST(d.month+INTERVAL '1 month' AS DATE) AS forecast_month,d.currency,
MAX(d.loss_given_default) AS loss_given_default,MAX(d.loss_given_default)*100 AS loss_given_default_pct,
SUM(d.balance_inr*p.default_probability*d.loss_given_default) AS expected_credit_loss_inr
FROM cc_delinquency_monthly d JOIN probabilities p ON d.month=p.month AND d.variant_id=p.variant_id
AND d.segment=p.segment AND d.currency=p.currency AND d.dpd_bucket=p.from_bucket
GROUP BY d.month,d.currency;
