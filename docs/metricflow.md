# MetricFlow integration

The application uses the pinned **MetricFlow 0.213.0 Python core** to compile
metric requests. It constructs a real semantic manifest, validates it with
MetricFlow's manifest validator, builds a native dataflow plan and obtains SQL
through `MetricFlowEngine.explain()`. It does not invoke a warehouse client, run
the dbt CLI or cache query results. The tenant-selected LLM still produces the
typed QueryIR; MetricFlow adds no model calls.

[MetricFlow's official repository](https://github.com/dbt-labs/metricflow)
describes its semantic planning and SQL generation. This integration supplies
the semantic manifest programmatically from the existing JSON catalogs rather
than requiring a separate dbt project.

## Responsibilities

1. PydanticAI selects workspace-owned metric IDs, dimensions, filters and time
   in the published QueryIR.
2. The catalog compiler resolves allowed relationships and prepares source
   relations at their declared grain. It compiles checks for unique keys,
   repeated constants, reconciliations and metric bounds.
3. The adapter converts those sources into scoped MetricFlow semantic models,
   entities, dimensions and metrics. MetricFlow resolves native entity paths
   and compiles aggregate, ratio, derived and cumulative requests.
4. SQLGlot bridges the DuckDB renderer's syntax, restores business aliases and
   retains ordered arithmetic calculations. It verifies that every native base
   relation belongs to the prepared workspace sources or the internal time spine.
5. Existing SQLGlot qualification, join checks and the **DataFusion compile gate**
   run before data execution. Source and result-contract checks still apply.
6. SQL rows retain their actual Arrow/Python types for Monty, scientific
   callbacks and optional evidence-validated summarization.

MetricFlow has no native DataFusion renderer. SQLGlot transpilation does not
establish compatibility: DataFusion remains the final compilation authority.
Unsupported functions, nonempty native bind parameters, scope violations and
unexpected native planning errors stop execution and return correction feedback.
The adapter cannot execute warehouse queries. Its compile-only client's query,
execute and dry-run methods raise immediately.

## Workspace configuration

Each workspace in `datasource.json` contains:

```json
"semantic_planner": {
  "backend": "metricflow",
  "unsupported": "catalog",
  "time_spine_start": "2026-01-01",
  "time_spine_end": "2027-01-01"
}
```

Both sample workspaces enable this configuration. New workspaces without the
property retain `backend: catalog`; saved/manual legacy SQL plans keep their
existing behavior. `unsupported: error` makes MetricFlow support mandatory.

`unsupported: catalog` permits a documented compatibility path for explicitly
unsupported shapes, including whole-population completion and dimension-only
lookups, global predicates on non-anchor sources, and snapshot selection that
does not scope every temporal metric source. Native metric dataflows aggregate
sources independently; those relational shapes use the catalog compiler to avoid
changing LEFT-join membership or creating extra NULL groups. Whole-population completion must preserve registered members with no
activity, so it uses the existing population-anchored compiler. Trace output
reports `METRICFLOW_FALLBACK` and its reason; validation receipts record the
requested and actual backend. Invalid business definitions, source defects,
unrecognized source references, unexpected MetricFlow exceptions and DataFusion
errors never trigger fallback.

No engine or results are shared across requests. MetricFlow compilation is
serialized because the library resets generated node IDs. Column names/types,
entities, relationships and metrics come only from the resolved tenant/workspace.

## Ratio and cohort semantics

A catalog formula dividing two named metrics becomes a native MetricFlow ratio.
Other supported catalog formulas become native derived metrics. The adapter
does not contain credit-card or SmartWealth table names or formulas.

The probability-of-default definition in `domain_instructions.json` counts
transitions into DEFAULT from non-default origins, divided by those origins'
opening balances. Its horizon is one month. Destination rows are first reduced
to their origin-cohort grain using the relationship's configured preaggregation.
Opening balances must agree before MAX is applied and are counted once per
origin. Already-defaulted origins are excluded from both inputs, even in a
variant/segment breakdown. Missing cohorts or zero denominators remain NULL;
they do not become zero-risk observations. Existing guards reject inconsistent
opening balances and unreconciled transitions.

Source sums retain Decimal money. Derived division uses the pipeline's explicit
float64/NULL-on-zero boundary to avoid Decimal128 precision growth. Computed
DataFusion output types, rather than JSON display types, govern Monty inputs.

## Cumulative metrics

Define the business metric in `profiles.<profile>.semantic_metrics`:

```json
"cumulative_approved_net_spending": {
  "description": "Cumulative approved INR net spending",
  "cumulative": {
    "input_metric": "approved_net_spending",
    "period_aggregation": "last"
  }
}
```

The referenced base metric owns its filters and aggregation. `last` returns the
month-end cumulative value when grouping by month. `first` and `average` must be
chosen explicitly when those meanings are wanted. Monthly date keys label the
**start** of each month; they do not become month-end dates. Tool results and
planner receipts expose `time_groupings` and `cumulative_period_aggregation`
so the response can distinguish the label from the cumulative measurement.
Optional fields are
`window_count` plus `window_grain` (day/week/month/quarter/year), or
`grain_to_date` (month/quarter/year); these window choices are mutually exclusive.

QueryIR must select a bounded half-open time range and an explicit day, month or
year transform on its time dimension. Use the time selection alone instead of
duplicate predicates on the snapshot date. MetricFlow can then widen source
history for rolling windows before trimming the displayed result period. The
internal daily time spine is a read-only VALUES relation with an inclusive start
and exclusive end, limited to 730 days. Configure it to cover the requested period.

Currently cumulative inputs must be base/simple metrics with a real time column.
Derived or ratio inputs, static models and out-of-spine requests are rejected.
A cumulative request, including one referenced by a derived metric, can never
fall back to an instantaneous catalog calculation. Unsupported semantics must be
modeled explicitly rather than guessed.

The sample catalogs include credit-card `cumulative_approved_net_spending` and
SmartWealth `cumulative_purchases`. The latter includes completed PURCHASE
transactions; SIPs remain separate.

## Audit evidence and verification

`METRICFLOW_PLAN` points to a private JSON file under
`artifacts/metricflow/tenant-<id>/workspace-<id>/`. Before executing the data query,
the application writes the resolved semantic manifest, prepared source SQL,
native dataflow text, original MetricFlow SQL, bridged DataFusion SQL, QueryIR,
catalog/manifest/dataflow/statement hashes and the MetricFlow version. Files are
created with mode 0600 and ignored by Git. Publication failure stops execution.
These compilation artifacts describe a proposal; successful execution is recorded
separately in result validation receipts and `SQL_KPI`.

`QUERY_COMPILED` includes the planner receipt. `SQL_KPI` records `planner_backend`,
so compilation acceptance and explicit fallback can be measured separately from
semantic accuracy. Compilation and result-contract success do not prove the LLM
interpreted every business question correctly.

Run the numerical and integration checks without an LLM API call:

```sh
uv run python -m unittest discover -s tests -p test_metricflow_adapter.py
uv run python -m unittest discover -s tests
```

These tests use real MetricFlow and DataFusion for AMC joins/distinct counts,
cross-model ratios, cohort default probabilities, empty counts and cumulative
and rolling windows. They compare results with independent SQL, check missing
destinations/zero denominators/source consistency and enforce tenant scope.
A FunctionModel exercises the execution tools and passes actual typed SQL rows
to the real Monty sandbox. This validates the integration without claiming that
every live model proposal will be correct on its first attempt.
