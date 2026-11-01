# Staged SQL generation and execution

Large semantic queries are automatically compiled into a dependency-ordered set
of smaller SQL statements. This applies to both MetricFlow and the catalog
compiler and contains no domain-specific query templates.

The LLM selects business metrics, dimensions, filters and observation grain in
QueryIR. SQLGlot then generates the executable stages from complete CTE/derived
relations. These internal stages do not require extra LLM calls. Small queries
retain the existing single-statement path.

For the default credit-loss forecast, the catalog query becomes:

1. Aggregate transitions at the origin grain for the latest common period.
2. Select the exposure snapshot for that same period.
3. Join those typed results and compute the requested metrics at the forecast
   observation grain. Only this final dataset is supplied to Monty.

Each intermediate is fully materialized as Arrow record batches in the
request's DataFusion context. Registering a lazy view would let DataFusion inline
the entire query again; materialization creates actual execution boundaries.
Decimal, date, timestamp and null types are preserved. Intermediates do not use
the final result's 200-row preview limit.

## Validation and observability

Before reading intermediate data, the host checks the original query's owned
sources and business join lineage using SQLGlot, then compiles **every** stage
with DataFusion using empty typed intermediate providers. DataFusion never needs
to compile the entire expanded statement as one logical plan. Source-key,
constant and reconciliation guards still run. Staging does not fix a missing
metric definition or a mismatched output contract.
Before SQL compilation, configured observation policies can complete exact
catalog-metric projections and explicitly declared composite observation keys;
these bindings are visible in the plan and trace. See
[projection normalization](semantic-queries.md#projection-aliases-and-correction-feedback).

Stages execute in dependency order. The final result must pass its original
output contract before it becomes a SQL producer result. Temporary providers are
removed on success and failure. Partial stages are not published as input to
Monty, reused between requests or left registered after an error.

`--trace` reports `SQL_STAGES_PLAN`, `SQL_STAGING` and `SQL_STAGE n/N` with
statement sizes, intermediate row counts, bytes and timing. The private local
artifact under `artifacts/sql_stages/tenant-*/workspace-*` contains every executable
statement and dependency. Execution validation receipts retain stage schemas and
row counts. The next LLM turn receives compact stage metadata and the final SQL,
rather than another full copy of every intermediate statement.

## Settings

No new CLI flags are needed. Optional environment variables:

| Variable | Default | Meaning |
| --- | --- | --- |
| `SQL_STAGING_ENABLED` | `true` | Enable automatic staging. |
| `SQL_STAGING_MAX_CHARS` | `6000` | Stage when original SQL exceeds this many characters. |
| `SQL_STAGING_MAX_SELECTS` | `6` | Stage at this many SELECT scopes. |
| `SQL_STAGING_MAX_JOINS` | `3` | Stage at this many joins. |
| `SQL_STAGING_MAX_STAGES` | `64` | Maximum number of statements, including the final query. |
| `SQL_STAGING_MAX_ROWS` | `100000` | Maximum rows per intermediate; exceeding it fails, without truncation. |
| `SQL_STAGING_MAX_BYTES` | `64000000` | Maximum total retained Arrow data bytes across intermediates. |

Any complexity threshold can activate staging. These trigger thresholds do not
guarantee a particular statement size: SQL can only be separated where a complete
relation is independent of an outer row. Each executable statement still has a
hard 20,000-character limit; the original compilation input is capped at 500,000.
An unsplittable oversized stage returns a diagnostic asking for a simpler logical
query. Recursive CTEs and volatile SQL functions are rejected on the staging path.
Correlated subqueries remain with their enclosing query. Materialization adds
memory use and can reduce cross-stage optimizer opportunities, so small queries
stay on the existing path.

The implementation uses DataFusion's documented
[Arrow record-batch registration](https://datafusion.apache.org/python/autoapi/datafusion/context/index.html)
and [streaming DataFrame execution](https://datafusion.apache.org/python/autoapi/datafusion/dataframe/index.html).
