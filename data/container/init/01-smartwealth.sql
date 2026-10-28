BEGIN;

CREATE TABLE customers (
    customer_id integer PRIMARY KEY,
    full_name text NOT NULL,
    email text UNIQUE NOT NULL,
    city text NOT NULL,
    risk_profile text NOT NULL CHECK (risk_profile IN ('CONSERVATIVE', 'MODERATE', 'AGGRESSIVE')),
    joined_on date NOT NULL
);

CREATE TABLE amcs (
    amc_id integer PRIMARY KEY,
    amc_name text UNIQUE NOT NULL
);

CREATE TABLE schemes (
    scheme_id integer PRIMARY KEY,
    amc_id integer NOT NULL REFERENCES amcs,
    scheme_name text UNIQUE NOT NULL,
    category text NOT NULL,
    plan text NOT NULL DEFAULT 'DIRECT',
    option text NOT NULL DEFAULT 'GROWTH',
    currency char(3) NOT NULL DEFAULT 'INR',
    latest_nav numeric(14,4) NOT NULL CHECK (latest_nav > 0),
    nav_date date NOT NULL,
    UNIQUE (scheme_id, amc_id)
);

CREATE TABLE folios (
    folio_id integer PRIMARY KEY,
    customer_id integer NOT NULL REFERENCES customers,
    amc_id integer NOT NULL REFERENCES amcs,
    folio_number text UNIQUE NOT NULL,
    opened_on date NOT NULL,
    holding_mode text NOT NULL DEFAULT 'SINGLE',
    UNIQUE (folio_id, amc_id)
);

CREATE TABLE transactions (
    transaction_id integer PRIMARY KEY,
    folio_id integer NOT NULL,
    scheme_id integer NOT NULL,
    amc_id integer NOT NULL,
    transaction_date date NOT NULL,
    transaction_type text NOT NULL CHECK (transaction_type IN ('PURCHASE', 'SIP', 'REDEMPTION')),
    units numeric(18,6) NOT NULL CHECK (units > 0),
    nav numeric(14,4) NOT NULL CHECK (nav > 0),
    amount numeric(18,2) NOT NULL CHECK (amount > 0),
    status text NOT NULL CHECK (status IN ('COMPLETED', 'PENDING', 'FAILED')),
    FOREIGN KEY (folio_id, amc_id) REFERENCES folios (folio_id, amc_id),
    FOREIGN KEY (scheme_id, amc_id) REFERENCES schemes (scheme_id, amc_id),
    CHECK (amount = round(units * nav, 2))
);

CREATE INDEX transactions_folio_scheme_idx ON transactions (folio_id, scheme_id, transaction_date);
CREATE INDEX folios_customer_idx ON folios (customer_id);

INSERT INTO customers
SELECT i, names[i], 'customer' || i || '@example.com',
       (ARRAY['Mumbai', 'Bengaluru', 'Delhi', 'Pune', 'Chennai', 'Hyderabad'])[(i-1)%6+1],
       (ARRAY['CONSERVATIVE', 'MODERATE', 'AGGRESSIVE'])[(i-1)%3+1],
       DATE '2024-01-01' + i * 7
FROM generate_series(1,12) AS g(i)
CROSS JOIN (SELECT ARRAY['Aarav Sharma', 'Priya Patel', 'Rohan Mehta', 'Ananya Singh',
    'Vikram Rao', 'Neha Gupta', 'Arjun Nair', 'Isha Shah', 'Kabir Das',
    'Meera Iyer', 'Dev Kapoor', 'Sara Menon'] AS names) AS n;

INSERT INTO amcs VALUES
    (1, 'Sample Horizon AMC'), (2, 'Sample Riverstone AMC'),
    (3, 'Sample Summit AMC'), (4, 'Sample Evergreen AMC'),
    (5, 'Sample Lighthouse AMC'), (6, 'Sample Bluebird AMC');

INSERT INTO schemes
SELECT (a.amc_id-1)*3+s, a.amc_id,
       a.amc_name || ' ' || (ARRAY['Equity Fund', 'Bond Fund', 'Hybrid Fund'])[s],
       (ARRAY['EQUITY', 'DEBT', 'HYBRID'])[s], 'DIRECT', 'GROWTH', 'INR',
       30 + a.amc_id * 7 + s * 5, DATE '2026-06-30'
FROM amcs a CROSS JOIN generate_series(1,3) AS g(s);

INSERT INTO folios
SELECT (c.customer_id-1)*2+n, c.customer_id,
       ((c.customer_id+n-2)%6)+1,
       'SW-' || lpad(((c.customer_id-1)*2+n)::text,6,'0'),
       DATE '2025-01-01' + c.customer_id, 'SINGLE'
FROM customers c CROSS JOIN generate_series(1,2) AS g(n);

-- Each folio holds all three schemes from its AMC. Amounts are in INR.
-- All units are positive; transaction_type determines the cash/units direction.
INSERT INTO transactions
SELECT ((f.folio_id-1)*3 + (s.scheme_id-1)%3)*6 + n,
       f.folio_id, s.scheme_id, f.amc_id,
       (DATE '2026-01-05' + (n-1) * INTERVAL '1 month')::date,
       CASE WHEN n=1 THEN 'PURCHASE' WHEN n=6 THEN 'REDEMPTION' ELSE 'SIP' END,
       CASE WHEN n=6 THEN 5 ELSE 20 + f.customer_id + n END,
       s.latest_nav - (6-n)*0.75,
       round((CASE WHEN n=6 THEN 5 ELSE 20 + f.customer_id + n END)
             * (s.latest_nav - (6-n)*0.75), 2),
       CASE WHEN n=5 AND f.folio_id%4=0 THEN 'FAILED'
            WHEN n=6 AND f.folio_id%5=0 THEN 'PENDING'
            ELSE 'COMPLETED' END
FROM folios f JOIN schemes s ON s.amc_id=f.amc_id
CROSS JOIN generate_series(1,6) AS g(n);

-- Live holdings stay consistent with completed transactions after inserts/updates.
CREATE VIEW holdings AS
SELECT f.customer_id, t.folio_id, t.scheme_id, t.amc_id,
       sum(CASE WHEN t.transaction_type='REDEMPTION' THEN -t.units ELSE t.units END) AS units,
       sum(CASE WHEN t.transaction_type='REDEMPTION' THEN -t.amount ELSE t.amount END) AS net_invested_amount,
       s.latest_nav, s.nav_date,
       round(sum(CASE WHEN t.transaction_type='REDEMPTION' THEN -t.units ELSE t.units END)
             * s.latest_nav, 2) AS market_value
FROM transactions t
JOIN folios f ON f.folio_id=t.folio_id
JOIN schemes s ON s.scheme_id=t.scheme_id
WHERE t.status='COMPLETED'
GROUP BY f.customer_id, t.folio_id, t.scheme_id, t.amc_id, s.latest_nav, s.nav_date
HAVING sum(CASE WHEN t.transaction_type='REDEMPTION' THEN -t.units ELSE t.units END)>0;

COMMENT ON VIEW holdings IS 'Synthetic holdings as of scheme nav_date. Net invested amount is purchases minus redemption proceeds, not tax cost basis.';

COMMIT;
