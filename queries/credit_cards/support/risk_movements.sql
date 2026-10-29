WITH probabilities AS (
 SELECT month,variant_id,segment,currency,from_bucket,
 SUM(CASE WHEN to_bucket='DEFAULT' THEN transitioned_balance_inr ELSE 0 END)
   /MAX(cohort_opening_balance_inr) AS default_probability
 FROM cc_roll_rates_monthly
 WHERE month >= (SELECT MAX(month)-INTERVAL '1 month' FROM cc_roll_rates_monthly)
 GROUP BY month,variant_id,segment,currency,from_bucket
), monthly AS (
 SELECT d.month,d.currency,
 SUM(CASE WHEN d.dpd_bucket IN ('DPD_1_29','DPD_30_89') THEN d.balance_inr ELSE 0 END) AS delinquent_balance_inr,
 SUM(CASE WHEN d.dpd_bucket='DEFAULT' THEN d.balance_inr ELSE 0 END) AS default_balance_inr,
 SUM(CASE WHEN d.dpd_bucket<>'DEFAULT' THEN d.balance_inr*p.default_probability ELSE 0 END)
   /SUM(CASE WHEN d.dpd_bucket<>'DEFAULT' THEN d.balance_inr ELSE 0 END) AS weighted_default_roll_rate
 FROM cc_delinquency_monthly d JOIN probabilities p ON d.month=p.month AND d.variant_id=p.variant_id
 AND d.segment=p.segment AND d.currency=p.currency AND d.dpd_bucket=p.from_bucket
 GROUP BY d.month,d.currency
)
SELECT *,delinquent_balance_inr-LAG(delinquent_balance_inr) OVER(PARTITION BY currency ORDER BY month) AS delinquent_balance_change_inr,
weighted_default_roll_rate-LAG(weighted_default_roll_rate) OVER(PARTITION BY currency ORDER BY month) AS default_roll_rate_change
FROM monthly ORDER BY month,currency;
