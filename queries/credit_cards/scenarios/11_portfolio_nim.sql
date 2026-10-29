-- EXAMPLE inputs: COF +20bps, revolve yield +25bps, transactor share +5 percentage points.
WITH inputs AS (
 SELECT CAST('20' AS DECIMAL(8,4)) AS cof_bps_change,
 CAST('25' AS DECIMAL(8,4)) AS revolve_yield_bps_change,
 CAST('5' AS DECIMAL(8,4)) AS transactor_share_pct_pt_change
), base AS (
 SELECT f.*,i.*,
 CAST(i.cof_bps_change/10000 AS DECIMAL(12,8)) AS dcof,
 CAST(i.revolve_yield_bps_change/10000 AS DECIMAL(12,8)) AS dyield,
 CAST(i.transactor_share_pct_pt_change/100 AS DECIMAL(12,8)) AS dshare,
 CAST(CAST(f.days_in_period AS DECIMAL(20,16))/f.day_count_basis AS DECIMAL(20,16)) AS period_fraction
 FROM cc_monthly_financials f CROSS JOIN inputs i
 WHERE month=(SELECT MAX(month) FROM cc_monthly_financials)
), impacts AS (
 SELECT *,average_net_receivables_inr*(1-transactor_share-dshare) AS adjusted_revolving_anr_inr,
 CAST(-funded_balance_inr*dcof*period_fraction AS DECIMAL(24,6)) AS cof_nii_impact_inr,
 CAST(revolving_anr_inr*dyield*period_fraction AS DECIMAL(24,6)) AS yield_nii_impact_inr,
 CAST(-average_net_receivables_inr*dshare*annual_revolve_yield*period_fraction AS DECIMAL(24,6)) AS mix_nii_impact_inr,
 CAST(-average_net_receivables_inr*dshare*dyield*period_fraction AS DECIMAL(24,6)) AS yield_mix_interaction_inr,
 CASE WHEN transactor_share+dshare<0 OR transactor_share+dshare>1
 OR annual_funding_rate+dcof<0 OR annual_revolve_yield+dyield<0 THEN 1 ELSE 0 END AS invalid_input
 FROM base
), totals AS (
 SELECT month,currency,MAX(cof_bps_change) AS cof_bps_change,MAX(revolve_yield_bps_change) AS revolve_yield_bps_change,
 MAX(transactor_share_pct_pt_change) AS transactor_share_pct_pt_change,MAX(dcof) AS cof_rate_change,MAX(dyield) AS yield_rate_change,MAX(dshare) AS transactor_share_fraction_change,SUM(invalid_input) AS invalid_input_rows,
 SUM(average_net_receivables_inr*period_fraction) AS anr_exposure_years,
 SUM(revolving_anr_inr) AS base_revolving_anr_inr,SUM(adjusted_revolving_anr_inr) AS adjusted_revolving_anr_inr,
 SUM(interest_income_inr-funding_cost_inr) AS base_nii_inr,
 SUM(cof_nii_impact_inr) AS cof_nii_impact_inr,SUM(yield_nii_impact_inr) AS yield_nii_impact_inr,
 SUM(mix_nii_impact_inr) AS mix_nii_impact_inr,SUM(yield_mix_interaction_inr) AS yield_mix_interaction_inr
 FROM impacts GROUP BY month,currency
), combined AS (
 SELECT *,CAST(cof_nii_impact_inr+yield_nii_impact_inr+mix_nii_impact_inr+yield_mix_interaction_inr AS DECIMAL(24,6)) AS combined_nii_impact_inr FROM totals
), metrics AS (
SELECT *,CAST(base_nii_inr AS DECIMAL(24,6))/NULLIF(CAST(anr_exposure_years AS DECIMAL(24,6)),0) AS base_nim_rate,
CASE WHEN invalid_input_rows=0 THEN base_nii_inr+combined_nii_impact_inr END AS adjusted_nii_inr,
CASE WHEN invalid_input_rows=0 THEN CAST(base_nii_inr+combined_nii_impact_inr AS DECIMAL(24,6))/NULLIF(CAST(anr_exposure_years AS DECIMAL(24,6)),0) END AS adjusted_nim_rate,
CASE WHEN invalid_input_rows=0 THEN CAST(combined_nii_impact_inr/NULLIF(CAST(anr_exposure_years AS DECIMAL(24,6)),0) AS DECIMAL(16,10))*10000 END AS nim_change_bps
FROM combined)
SELECT *,ROUND(base_nim_rate*100,4) AS base_nim_pct,ROUND(adjusted_nim_rate*100,4) AS adjusted_nim_pct,
adjusted_revolving_anr_inr-base_revolving_anr_inr AS change_revolving_anr_inr,
ROUND(ABS(nim_change_bps),2) AS nim_change_magnitude_bps,
ROUND(ABS(adjusted_revolving_anr_inr-base_revolving_anr_inr),2) AS revolving_anr_change_magnitude_inr,
ROUND(base_nii_inr,2) AS base_nii_inr_display, ROUND(ABS(base_nii_inr),2) AS base_nii_inr_magnitude,
ROUND(adjusted_nii_inr,2) AS adjusted_nii_inr_display, ROUND(ABS(adjusted_nii_inr),2) AS adjusted_nii_inr_magnitude,
ROUND(cof_nii_impact_inr,2) AS cof_nii_impact_inr_display, ROUND(ABS(cof_nii_impact_inr),2) AS cof_nii_impact_inr_magnitude,
ROUND(yield_nii_impact_inr,2) AS yield_nii_impact_inr_display, ROUND(ABS(yield_nii_impact_inr),2) AS yield_nii_impact_inr_magnitude,
ROUND(mix_nii_impact_inr,2) AS mix_nii_impact_inr_display, ROUND(ABS(mix_nii_impact_inr),2) AS mix_nii_impact_inr_magnitude,
ROUND(yield_mix_interaction_inr,2) AS yield_mix_interaction_inr_display, ROUND(ABS(yield_mix_interaction_inr),2) AS yield_mix_interaction_inr_magnitude,
ROUND(combined_nii_impact_inr,2) AS combined_nii_impact_inr_display, ROUND(ABS(combined_nii_impact_inr),2) AS combined_nii_impact_inr_magnitude
FROM metrics;
