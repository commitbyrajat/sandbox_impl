# SmartWealth sample PostgreSQL database

All names, AMCs, schemes, NAVs, and transactions are fictional. All monetary amounts are INR. This fixture covers January–June 2026.

Start from this directory:

```sh
docker compose up -d
docker compose exec postgres psql -U smartwealth -d smartwealth
```

Connect from your host using `localhost:5433`, database/user `smartwealth`, password `smartwealth_local`. Override the defaults with `SMARTWEALTH_DB_PORT` and `SMARTWEALTH_DB_PASSWORD` environment variables before starting. These credentials are for local sample use.

The database includes:

| Relation | Rows | Description |
| --- | ---: | --- |
| customers | 12 | Different cities and risk profiles |
| amcs | 6 | Fictional asset management companies |
| schemes | 18 | Equity, debt, and hybrid schemes per AMC |
| folios | 24 | Two folios with different AMCs per customer |
| transactions | 432 | Purchases, SIPs, redemptions; completed, pending, failed |
| holdings (view) | 72 | Three scheme holdings per folio, calculated from completed transactions |

Foreign keys ensure that each folio and its schemes belong to the same AMC. Holdings use the latest seeded NAV dated June 30, 2026. `net_invested_amount` subtracts redemption proceeds from purchases; it is not a tax cost basis or realized profit calculation.

Example portfolio query:

```sql
SELECT c.full_name, f.folio_number, a.amc_name, s.scheme_name,
       h.units, h.net_invested_amount, h.market_value, h.nav_date
FROM holdings h
JOIN customers c USING (customer_id)
JOIN folios f USING (folio_id)
JOIN amcs a ON a.amc_id = h.amc_id
JOIN schemes s ON s.scheme_id = h.scheme_id
ORDER BY c.customer_id, f.folio_id, s.scheme_id;
```

`docker compose down` stops the database and retains data. Initialization scripts run only when the data volume is empty. To deliberately discard all local database changes and reload the sample data:

```sh
docker compose down -v
docker compose up -d
```
