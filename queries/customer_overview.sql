-- These ten overlapping fictional names link the sample datasets for a demo.
-- Production data needs an explicit, stable customer identity mapping.
-- Aggregate each source before joining so transactions do not multiply holdings.
WITH spending AS (
    SELECT c.full_name, count(*) AS approved_card_transactions,
           sum(t.amount) AS net_card_spending_inr
    FROM cc_customers c
    JOIN cc_transactions t ON c.customer_id = t.customer_id
    WHERE t.status = 'APPROVED' AND t.currency = 'INR'
    GROUP BY c.full_name
), portfolios AS (
    SELECT c.full_name, count(DISTINCT h.folio_id) AS folios,
           count(*) AS scheme_holdings, sum(h.market_value) AS portfolio_value_inr
    FROM sw_customers c
    JOIN sw_holdings h ON c.customer_id = h.customer_id
    GROUP BY c.full_name
)
SELECT p.full_name, s.approved_card_transactions, s.net_card_spending_inr,
       p.folios, p.scheme_holdings, p.portfolio_value_inr
FROM portfolios p
JOIN spending s ON p.full_name = s.full_name
ORDER BY p.full_name;
