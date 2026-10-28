# Sample credit card data

Entirely synthetic data, generated with random seed 42.

- `customers.parquet`: 20 customers. Primary key: `customer_id`.
- `credit_cards.parquet`: 40 cards, with 1–3 cards per customer. Primary key: `card_id`; joins to customers on `customer_id`. Only fictional last-four digits are stored.
- `transactions.parquet`: 1,000 transactions from January–June 2026. Primary key: `transaction_id`; joins to cards on `card_id` and customers on `customer_id`. Every card has multiple transactions.

Transactions include purchases, negative-amount refunds linked through `original_transaction_id`, and approved, declined, or pending statuses. Timestamps are UTC. Amounts and credit limits are decimal(12, 2). Amounts are in the transaction currency; credit limits are in the card billing currency. Filter to approved transactions and group by currency when calculating spending; no exchange rates are supplied. Files use Snappy compression.
