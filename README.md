# Query the sample datasets with DataFusion Python

```sh
docker compose -f data/container/docker-compose.yml up -d --wait
uv sync
uv run main.py
```

`main.py` registers both datasets in a single `datafusion.SessionContext`:

| Source | DataFusion table names |
| --- | --- |
| Parquet files in `data/parquet` | `cc_customers`, `cc_credit_cards`, `cc_transactions` |
| PostgreSQL in `data/container` | `sw_customers`, `sw_amcs`, `sw_schemes`, `sw_folios`, `sw_transactions`, `sw_holdings` |

Parquet is scanned directly by DataFusion. Psycopg reads the PostgreSQL relations into Arrow batches in one consistent, read-only transaction; DataFusion queries those in-memory snapshots. Create a new context to refresh PostgreSQL data. This example loads the entire sample database into memory, with no PostgreSQL query pushdown or `datafusion-federation` dependency.

The default query in `queries/customer_overview.sql` joins card spending and mutual fund portfolios. It matches ten overlapping fictional full names for demonstration only: the datasets have separate IDs and differently formatted emails. A real integration needs a stable customer mapping. The query includes only INR card transactions, subtracts approved refunds, and aggregates each source before joining to avoid duplicated totals. Customers without approved INR transactions are excluded.

Custom SQL:

```sh
uv run main.py --sql "SELECT merchant_category, currency, sum(amount) AS spending FROM cc_transactions WHERE status = 'APPROVED' GROUP BY merchant_category, currency"
uv run main.py --sql "SELECT a.amc_name, count(*) AS holdings, sum(h.market_value) AS market_value_inr FROM sw_holdings h JOIN sw_amcs a ON h.amc_id = a.amc_id GROUP BY a.amc_name"
uv run main.py --sql-file queries/customer_overview.sql --limit 10
```

From Python:

```python
from main import create_context

ctx = create_context()
df = ctx.sql("SELECT * FROM sw_holdings LIMIT 5")
df.show()
batches = df.collect()  # PyArrow RecordBatches
```

PostgreSQL defaults to `127.0.0.1:5433`, database/user `smartwealth`, password `smartwealth_local`. Override with `SMARTWEALTH_DB_HOST`, `SMARTWEALTH_DB_PORT`, `SMARTWEALTH_DB_NAME`, `SMARTWEALTH_DB_USER`, and `SMARTWEALTH_DB_PASSWORD`.

References: [DataFusion Python data sources](https://datafusion.apache.org/python/user-guide/data-sources.html), [Psycopg cursor API](https://www.psycopg.org/psycopg3/docs/api/cursors.html).

## Ask questions with Pydantic AI and Monty

```sh
uv sync
docker compose -f data/container/docker-compose.yml up -d --wait
# If .env does not exist, copy .env.example to .env and fill in your key.
uv run agent.py --tenant-id 1 --workspace-id 1 "How many customers and transactions are there?" --trace
uv run agent.py --tenant-id 1 --workspace-id 1 "Show total approved INR card spending by customer."
uv run agent.py --tenant-id 2 --workspace-id 2 "Which AMC has the highest total holding value?"
```

Configure `LITELLM_MASTER_KEY` and `LITELLM_BASE_URL` (default `http://localhost:4000/v1`) in `.env` or the environment. Environment variables override `.env`. The local `.env` is ignored by Git; `.env.example` contains placeholders. Models are selected by tenant from `datasource.json`; a global `LITELLM_MODEL` setting does not override the tenant mapping.

Every agent request requires **both `tenant_id` and `workspace_id`**. Ownership is validated before classification, model construction or data loading. The selected tenant model serves classification, planning, execution and final summarization.

| Tenant ID | Model ID | Owned workspace ID | Workspace | Data source |
| --- | --- | --- | --- | --- |
| `1` | `qwen-coder` | `1` | Credit Cards | Parquet (`cc_*` tables) |
| `2` | `ornith-reasoner` | `2` | SmartWealth | PostgreSQL snapshot (`sw_*` tables) |

A tenant's `workspaces` mapping can contain multiple workspaces. Workspace IDs are resolved within a tenant, so the same workspace ID can exist under different tenants with different data. Model definitions are reusable across tenants. Each model can specify `base_url_env` and `api_key_env` for its connection settings; defaults are `LITELLM_BASE_URL` and `LITELLM_MASTER_KEY`. Credentials stay in environment variables.

Missing/unknown IDs and mismatched pairs such as tenant `1` with workspace `2` fail before any model or source is accessed. The question and model cannot change the request's tenant or workspace. A bare "How many customers?" refers only to the authorized workspace. The standalone `main.py` remains a raw SQL demo; scoped requests use `agent.py` and its Python APIs.

`agent.py` delegates orchestration to **`pipeline.py`**, which implements a chain of responsibility. Each `PipelineStage` handles one responsibility, updates a request-local `PipelineContext`, then forwards it through `next_stage.handle()`. The default chain is:

1. **Resolve:** validate tenant/workspace ownership and domain configuration before creating a model client.
2. **Classify:** identify the business objective, explicit constraints and independent sub-objectives, using workspace terminology without choosing tools.
3. **Plan:** choose SQL or SQL plus Python automatically, with ordered steps, expected outputs, validation checks and reasons. Defer unsupported or ambiguous objectives without blocking ready ones.
4. **Publish:** save the scoped plan and display it before data access. Publication failure stops the chain.
5. **Load sources:** open only the chosen workspace sources. PostgreSQL is a consistent read-only snapshot; Parquet is registered for scanning.
6. **Execute:** run the planned steps in order, retaining typed SQL results, Python outputs and completed-step evidence. Failed tool calls remain pending for correction.
7. **Summarize:** use the workspace vocabulary and tone to turn execution evidence into a business answer. This final agent has no query or code tools. It receives actual SQL/Python outputs and the executor draft, preserves values, units and dates, and omits implementation chatter. The application appends deferred clarification questions once.

`--classify-only` stops after classification. The `plan_question()` Python API stops after planning. CLI `--plan-only` stops after publication. Clarification-only plans bypass source loading and execution; the final stage passes through the domain-aware clarification without another model call. Exceptions stop successor handlers.

Responsibilities are separated into modules:

| Module | Responsibility |
| --- | --- |
| `agent.py` | CLI and public wrappers that run the pipeline |
| `pipeline.py` | Stage adapters, request context, preview stops and injected services |
| `pipeline_graph.py` | Typed `pydantic-graph` lifecycle and preview branches |
| `planning.py` | Intent/plan schemas and classifier/planner agents |
| `model_config.py` | Tenant-selected LiteLLM client configuration |
| `data_access.py` | Workspace table allowlist, read-only SQL, result storage and source loading |
| `execution.py` | Ordered execution tools, Monty lifecycle and execution evidence |
| `scientific_capabilities.json` | Domain-independent scientific signatures, numeric policy and RNG semantics supplied before code generation |
| `scientific_models.py` | Pydantic scientific requests, resource settings and Monty import type stubs |
| `scientific_bridge.py` | Allowlisted async Monty module and supervised CPython worker lifecycle |
| `scientific_worker.py` | Trusted SciPy/NumPy numerical operations and structured results |
| `sql_diagnostics.py` | SQLGlot preflight, engine result types, query fingerprints and repair diagnostics |
| `sql_joins.py` | Relationship keys bound to configured business meaning and grain rules |
| `result_contracts.py` | SQL output aliases/types, unique observation grain and population checks before step completion |
| `python_validation.py` | Preflight for sandbox imports, SQL-backed bindings, scientific API keys and directly traceable SQL aliases |
| `python_input_context.py` | Stringified `data` object and actual Python/Arrow types supplied before Python generation |
| `calculation_contracts.py` | Method families, Python output contracts, invariant checks and final evidence validation |
| `repair_feedback.py` | Typed repair destinations, rejected artifacts, authoritative contracts and retry budgets |
| `summarization.py` | Final domain-aware presentation, requested-topic guards and deferred questions |
| `summary_evidence.py` | Decimal totals and numeric/date checks against actual execution evidence |
| `plan_publication.py` | Scoped pre-execution JSON publications |
| `domain_instructions.py` | Scoped domain profile lookup for all four model roles |

The default lifecycle is **resolve → classify → plan → publish → load_sources → execute → validate_results → summarize**, implemented with typed nodes in `pydantic-graph`. Classification, plan and publication previews terminate at their corresponding nodes. The graph is in process; it does not provide durable checkpoints or resumability across process restarts.

`PipelineServices` supplies replaceable adapters for scope resolution, model creation, classification, planning, publication, loading, execution, validation and summarization. Stages use those adapters rather than importing the CLI. `Pipeline(..., stages=[...])` preserves the custom handler-chain API, and `set_next()` explicitly links those custom handlers. Each run creates a fresh context. The executor's tool permissions select the pending schema/SQL/Python role; these are logical generation responsibilities using the tenant's mapped model, not separate built-in agent classes.

See [validation and repair architecture](docs/validation-and-repair.md) for the executable graph, contracts, retry destinations and verification limits.

For example, replace presentation without changing classification, loading or execution:

```python
from pipeline import Pipeline, PipelineRequest, PipelineServices

async def custom_summary(question, plan, execution, model):
    # Return a business-facing summary; execution holds actual query/code outputs.
    return execution.draft if execution is not None else plan.response

services = PipelineServices(summarize=custom_summary)
context = await Pipeline(services).run(
    PipelineRequest("How many customers?", tenant_id="1", workspace_id="1")
)
print(context.answer)
```

The historical source-backed [interactive architecture diagram](.archify/architecture-analysis-pipeline-20261028-221642/analysis-pipeline.html) documents the earlier pipeline, tenant/model/domain mapping, source loading and Monty boundary. It predates the graph refactor and separate result-validation stage. [Architecture validation receipt](.archify/architecture-analysis-pipeline-20261028-221642/analysis-pipeline.finalize-summary.json).

The historical [agent stage responsibilities diagram](.archify/workflow-agent-stages-20261028-221642/agent-stages.html) shows the earlier handler chain and also predates the graph refactor. For the current lifecycle use [the graph generated from the executable nodes](docs/pipeline-graph.mmd). [Historical stage diagram receipt](.archify/workflow-agent-stages-20261028-221642/review-3/agent-stages.finalize-summary.json).

Both diagrams cite source at the pinned project commit recorded in their source-provenance files ([architecture](.archify/architecture-analysis-pipeline-20261028-221642/source-provenance.json), [stages](.archify/workflow-agent-stages-20261028-221642/source-provenance.json)). Source bytes were checked against that commit; the SSH repository origin uses local-only links. Automated specification, artifact and browser checks pass for both diagrams. No perceptual screenshot review was performed.

Plans are saved under `artifacts/plans/tenant-<id>/workspace-<id>/<publication-id>.json`. Each record includes its timestamp, question, intent, selected model, workspace and full plan. The record also includes `execution_summary.python_required`, the chosen approach and its explanation. Every request gets a new file with owner-only permissions; the default artifact directory is Git-ignored. These are local publications, with no external service involved. Records have status `planned`: they preserve the pre-execution plan, not the eventual run outcome. Use `--trace` to inspect actual SQL and Python.

Override the publication directory when needed:

```sh
uv run agent.py --tenant-id 2 --workspace-id 2 \
  --plan-dir /tmp/smartwealth-plans "Show holdings by AMC"
```

`--plan-only` also publishes the plan and then exits without opening data. `--classify-only` returns intent without creating a plan publication. Python callers can pass `plan_dir=Path(...)` to `ask()`.

`run_sql` performs retrieval, filtering, joins, aggregates, rankings, ratios and SQL-supported calculations. Only workspace tables and read-only statements are allowed, including inside CTEs. Results are retained by ID with exact typed values. Ordinary SQL plans do not start Monty. Hybrid plans expose `run_python` only for the identified non-SQL calculation, using complete prior SQL results as input. The executor applies the plan's validation checks when interpreting results and must state meaningful limitations.

`--trace` prints the tenant ID, workspace ID, mapped model, classified intent, actionable plan, completed step IDs, actual SQL and any executed Python. These records describe operations and evidence.

Every pipeline stage logs a start and end message to stderr, including status and
elapsed seconds, even without `--trace`. Failed, timed out and cancelled stages
also emit their end message. `--trace` adds SQL/Python tool progress while the
request is running. A model timeout identifies
its stage and mapped model. The former generic tenant/database message did not
identify slow inference accurately.

Local inference can exceed 90 seconds, particularly when Ollama loads a different
tenant model. All model stages share `LITELLM_TIMEOUT_SECONDS` (default 300 seconds
per SDK attempt), `LITELLM_REQUEST_DEADLINE_SECONDS` (default at least 330 seconds
for a model request), `LITELLM_MAX_RETRIES` (default 0), and
`LITELLM_MAX_OUTPUT_TOKENS` (default 4096). Configure these in `.env` as shown in
`.env.example`. The total deadline must be at least the attempt timeout. These
settings do not override a shorter LiteLLM/backend server timeout.

`AGENT_SUMMARIZATION_ENABLED=true` enables the final summarization stage (default).
Set it to `false` in `.env` to skip that LLM call, or override it for one command:

```sh
AGENT_SUMMARIZATION_ENABLED=false uv run agent.py --tenant-id 1 --workspace-id 1 --trace --max-tokens 8192 \
  "How uncertain is average PAT per customer? Show the average and a 95% range."
```

When disabled, the CLI returns the executor's report and preserves deferred
clarification questions. SQL/Python execution and tool validation still run;
the separate domain summary and its numeric/topic guards are bypassed. Requests
that need no data execution return the plan's clarification/response. Logs show
`STAGE summarize: end (skipped, ...)`. Accepted values are `true`/`false`, `1`/`0`,
`yes`/`no`, and `on`/`off` (case insensitive). Invalid values fail before any stage.

Set `AGENT_TOOL_RETRIES=5` in `.env` to allow more SQL/Python corrections, or
override it for one command:

```sh
AGENT_TOOL_RETRIES=5 uv run agent.py --tenant-id 1 --workspace-id 1 --trace --max-tokens 8192 \
  "Split customers into three review groups with roughly equal January–June 2026 approved INR net spending. Show members and totals."
```

The default is **3 retries per executor tool**, excluding the initial attempt;
`0` disables retries. Shell environment values take precedence over `.env`.
This controls `describe_sources`, `run_sql` and `run_python`, independently of
`LITELLM_MAX_RETRIES` (HTTP retries) and output-validation retries. The executor's
overall limits of 15 model requests and 25 tool calls still apply.

A missing Python key must be corrected using the actual SQL output aliases;
increasing retries does not repair code by itself. Python failures return the
error and stored result columns to the LLM. A `run_sql` retry-limit error after
a Python failure can mean the model tried to rerun SQL while the Python step was
pending; `--trace` now reports these rejected out-of-order calls explicitly.

Every LLM-generated `run_sql` proposal follows an enforced preflight loop:

1. SQLGlot parses a copy of the statement and qualifies its columns against the
   owned workspace's actual Arrow schemas. CTE/subquery projection failures are
   reported with the visible aliases and missing fields.
2. Configured join keys are checked through resolvable CTE projections, including
   every component of compound keys. Wrong keys cannot be justified by matching
   field names or types. Coarser joins between grouped CTEs are accepted when both
   inputs expose unique keys at that grain. Computed or opaque lineage remains
   explicitly unverified.
3. DataFusion compiles the original SQL without fetching result rows. Its result
   schema is authoritative about computed types; SQLGlot's inferred types are
   advisory and can differ in Decimal precision. SQLGlot has no dedicated
   DataFusion dialect in the installed version, so it does not replace this check.
4. A failed proposal returns JSON diagnostics through `ModelRetry`, with scope
   issues, relevant source types and configured join meaning. The pending step
   and latest correction requirements are refreshed in the model instructions.
   The LLM generates the revised SQL; the application does not rewrite it.
5. Retrieved rows must satisfy the SQL step's `output_contract` before they are
   stored or the step completes. Missing fields, incorrect types, duplicate/null
   grain keys, row-count violations and configured population mismatches return
   correction feedback while keeping that SQL step pending.

`datasource.json` also contains `sources.<source_id>.columns.<table>`, with every
column's Arrow `type` and `nullable` flag for both workspaces. The scoped catalog
is supplied to the planner and executor before query generation. `describe_sources`
verifies it against the loaded source schemas; schema drift stops execution with
an explicit registry mismatch. After changing the datasets, refresh the registry:

```bash
uv run python scripts/refresh_datasource_schema.py
```

The refresh inspects all configured sources, so PostgreSQL must be available. Start
a new agent process after refreshing. Source types describe base columns;
DataFusion supplies the actual types of aggregated and computed result columns.

New LLM-generated plans require an `output_contract` for every SQL step, including
`required_columns` (aliases, coarse types, nullability and meanings), `grain`,
optional row-count bounds and optional `population` (`table`, `source_key`,
`result_key`). Each Python step declares its SQL producer IDs in `input_steps`.
A complete customer-feature contract therefore rejects a date-only lookup even
when that lookup is valid SQL. Preliminary lookups belong inside the feature query
or in separately planned steps. Existing manually constructed plans can omit
contracts for compatibility; new generated plans cannot.

The LLM planner uses action-specific step schemas: SQL requires an output contract,
Python requires upstream `input_steps`, and schema/Python steps use
`output_contract: null`. This reports independent shape errors together instead
of exhausting retries on one error at a time. With `--trace`, `PLAN_REJECTED`
shows each rejection's field paths and validation messages, including the final
rejection. Planner output retries remain separate from `AGENT_TOOL_RETRIES`.

Profiles may prioritize business-owned instructions under `question_guidance`:
each named rule supplies a `request_pattern` and planner/executor instruction
lists. Matching guidance is placed beside the current question/execution state.
The credit-card PAT-uncertainty guidance preserves the latest-month default,
registered-customer population and customer-level bootstrap. It supplies no
prewritten plan, SQL, Python or answer; explicit user periods take precedence.

For contract-based plans, only the tool for the pending step is advertised. During
Python, the host supplies current result IDs, exact aliases, Arrow/Python types,
row counts and an explicit `HOST_DATA_OBJECT_JSON` snapshot before code generation.
The snapshot contains the current Python step's SQL inputs and is complete up to
24,000 UTF-8 bytes. Larger snapshots are explicitly marked as previews, with
included/stored row counts per result. Decimal values display as exact JSON
strings and dates as ISO strings; the accompanying type map explains their actual
`decimal.Decimal`/`datetime.date` types. Do not parse this JSON or copy it into code:
read the already supplied `data[result_id]`. It is refreshed on retries and after
SQL input repair. Python still executes in
Monty on the complete stored rows. Code/alias errors return feedback for correction.
The scientific capability catalog lists the bootstrap result keys explicitly
(`estimate`, `lower`, `upper`, `standard_error`, `bootstrap_distribution`, `method`).
It is an allowlisted bridge, so SciPy-style names such as `scientific.mean` or
`confidence_interval` are unavailable. Python retry feedback includes this API
contract alongside the actual error and stored SQL columns.
Hybrid summaries lead with the calculated Python metrics. Numeric validation
rejects invented rounded examples and asks the model to remove the unsupported
claim or copy an actual result, while preserving the validation checks.
If the SQL inputs need correction, `request_sql_repair` can reopen a declared SQL
producer at most **twice per request**. It preserves the published contract,
invalidates downstream results and Python caches, and issues fresh result IDs.
Stale result IDs cannot be reused after repair.

`datasource.json` supplies machine-readable `relationship_keys`. Each entry has a
relationship ID, `left_table`, `right_table` and `column_pairs`. The same ID under
`profiles.<profile_id>.join_semantics` in `domain_instructions.json` supplies its
`meaning`, `left_grain`, `right_grain`, configured `cardinality` and
`aggregation_rules`. Both are bound in the schema tool's `_sql_guidance` before
SQL generation and returned with relevant correction feedback. Existing narrative
instructions remain in place, and legacy relationship equality strings remain
supported. Business rules belong in configuration, not domain-specific Python
branches. Configured cardinality is not an observed uniqueness check or proof
that a query preserves measures; fan-out/grain warnings must still be reviewed.

Failed deterministic preflight/compilation proposals are fingerprinted per request
and step. Comments, formatting and unquoted identifier case do not bypass the
duplicate guard; string literals and quoted identifier case remain significant.
An unchanged rejected query receives feedback again without repeating preflight
or DataFusion compilation. Each duplicate still consumes the configured tool
retry budget. Execution/storage failures are not cached, so transient failures
can retry unchanged SQL. Separate requests have separate failure histories.

`--trace` records attempt numbers, fingerprints and complete `SQL_REJECTED`
diagnostics, making repeated errors distinguishable from identical submissions.
The raw `DataSources.query()` API and configured recipe path retain their existing
checks; this preflight applies to generated executor tool proposals. The preflight
does not prove business arithmetic or all join semantics. Population completeness
is checked separately when declared in the output contract.
Configured recipes are selected by the host before LLM planning; generated plans
must leave `sql_recipe` null. Invented recipe IDs/hashes return planner correction
feedback instead of bypassing the generated-SQL preflight path.

Override the output budget for one run with `--max-tokens` (a positive integer),
or set `LITELLM_MAX_OUTPUT_TOKENS` in `.env` for the default. Precedence is
`--max-tokens` / `ask(..., max_tokens=...)`, then the environment/`.env`, then 4096.
The override applies to classification, planning, execution and summarization;
it is isolated to that request. Reasoning models can consume their output budget
before producing visible text, so a larger budget may be needed even for a short
answer. Backend model/context limits and request timeouts still apply.

```sh
uv run agent.py --tenant-id 2 --workspace-id 2 --trace --max-tokens 8192 \
  "How variable were completed monthly net inflows from January through June 2026? Show all six monthly totals, the average, minimum and maximum. Explain the separate contributions from purchases, SIPs and redemptions."
```

For this question, net inflow includes completed purchases **plus SIPs**, minus
completed redemptions. All types use `amount`; there is no `proceeds` column.
The average, minimum and maximum must be calculated from monthly net totals in
SQL and supplied as evidence to the summary stage.

Automatic SDK retries are disabled to avoid repeating slow inference behind the proxy's own
retry policy; SQL/code correction retries remain enabled. A full pipeline includes
several model requests, so its total duration can exceed one request's deadline.

Hybrid execution sends at most 12 preview rows from each large SQL result to the
model. The host retains every stored row for Monty; previews do not change the
calculation's input or relax the 200-row SQL result limit. The final summary uses
complete Python outputs and verified aggregates. Requested confidence levels and
executed method parameters (such as seed and resample count) are validated
separately from monetary results. The credit-card domain guidance specifies
nearest-rank indices explicitly to avoid percentage/proportion mistakes. Run the examples sequentially
when the local backend shares memory between tenant models.


The separate **`datasource.json`** mapping defines reusable `models`, then `tenants.<tenant_id>.model_id` and `tenants.<tenant_id>.workspaces.<workspace_id>.sources`. The model ID is resolved from the tenant; the exact source definitions come from the owned workspace. Each workspace has a name, a `domain_profile` reference, and source definitions with storage locations, table names and relationships. `datasource_mapping.py` validates it and shares the registry between classification, loading and query enforcement. No credentials are stored in the mapping. Connection settings remain in environment variables.

Each model can set `structured_output_mode` to `tool` (the default), `native`, or
`prompted`. `qwen-coder` uses `native`: classification and planning send a JSON
schema in `response_format` and still validate the returned Pydantic models.
This avoids Ollama's Qwen XML tool-call parser for structured results. Other
models retain their configured mode; `ornith-reasoner` defaults to `tool`.
The executor's `describe_sources`, `run_sql` and `run_python` remain function
tools. Native structured output changes the result format, not how those tools
execute. See [Pydantic output modes](https://pydantic.dev/docs/ai/core-concepts/output/)
and [Ollama structured outputs](https://docs.ollama.com/capabilities/structured-outputs).

Provider failures now report the model, HTTP status when available and sanitized
provider message. An Ollama XML parser error is a model-server failure before
Pydantic can validate its output; increasing `AGENT_TOOL_RETRIES` does not fix a
planner HTTP failure. The generic tenant/database hint previously hid that cause.

Domain-specific instructions are configured separately in **`domain_instructions.json`**, under `profiles.<profile_id>`. Each profile contains:

- `description` and `glossary` for interpreting workspace terminology.
- `common_instructions` shared by all four model roles.
- `classifier_instructions` for interpreting objectives and domain-specific ambiguities.
- `planner_instructions` for valid calculations, aggregation grain, joins and checks.
- `executor_instructions` for executing and interpreting results correctly.
- `summarizer_instructions` and `tone` for the final business-facing answer.
- `action_policies`: named lists of actual business decision rules; empty until configured. Each entry needs nonempty condition-to-action instructions. Supplied user rules are also respected.
- `policy_absence_pattern`: blocks claims that policies are absent when named rules were supplied.
- `topic_scopes`: request/answer patterns that prevent unrelated topics or disclaimers in summaries.
- `summary_additive_fields` and `summary_partition_fields`: explicitly permitted total fields and grouping boundaries for deterministic summary checks.

A workspace selects its profile in `datasource.json`:

```json
{
  "name": "SmartWealth",
  "domain_profile": "smartwealth",
  "sources": { "smartwealth": { "...": "source configuration" } }
}
```

`domain_instructions.py` resolves the tenant/workspace ownership first, then looks up the referenced profile. The classifier receives common plus classifier guidance, the planner common plus planner guidance, the executor common plus executor guidance, and the final summarizer common plus summarizer guidance and the configured tone. Stage-specific rules do not leak into other stages, and another workspace's profile is never injected. Unknown profile references fail rather than falling back to a different domain. Profiles load at process startup; restart a long-running process after editing them.

`planning.py`, the executor and summarizer base prompts contain agent responsibilities and execution constraints only. SmartWealth terminology, next-action examples, valuation rules and credit card spending rules live in the separate domain mapping. Adding or changing domain guidance requires configuration edits rather than changing those prompts.

Preview the task classification alone, or classify and plan without opening data sources:

```sh
uv run agent.py --tenant-id 2 --workspace-id 2 "Which AMC has the highest total holding value?" --classify-only
uv run agent.py --tenant-id 2 --workspace-id 2 "Which AMC has the highest total holding value?" --plan-only
```

Example intent:

```json
{
  "intent": "aggregation",
  "objective": "Find the AMC with the highest total holding value",
  "constraints": [],
  "clarification": null,
  "objectives": [
    {"objective": "Find the AMC with the highest total holding value", "status": "ready", "clarification": null}
  ]
}
```

Example execution plan:

```json
{
  "tenant_id": "2",
  "workspace_id": "2",
  "sources": ["smartwealth"],
  "execution_mode": "sql",
  "execution_reason": "The requested totals and ranking can be obtained directly by querying the holding records.",
  "steps": [
    {
      "step_id": "s1",
      "action": "schema",
      "instruction": "Inspect holding and AMC columns",
      "expected_output": "Join keys, value column and valuation date"
    },
    {
      "step_id": "s2",
      "action": "sql",
      "instruction": "Sum holding market value per AMC, rank descending and return the highest",
      "expected_output": "AMC name, total INR value and valuation date"
    }
  ],
  "validation_checks": ["Keep INR units and report the holdings valuation date"],
  "python_reason": null,
  "response": null
}
```

## Examples: SQL only

SQL handles filtering, joins, grouping, sorting, ratios and ordinary aggregates. These questions should produce an `execution_mode: "sql"` plan with schema inspection and SQL steps, without starting Monty.

```sh
# Credit cards: aggregate approved purchases and negative refunds by month/currency.
uv run agent.py --tenant-id 1 --workspace-id 1 --trace \
  "Show monthly approved net card spending by currency. Keep currencies separate."

# SmartWealth: join holdings to AMCs and aggregate without multiplying folios.
uv run agent.py --tenant-id 2 --workspace-id 2 --trace \
  "Show total market value in INR, distinct schemes and distinct folios by AMC, with the valuation date."
```

Representative SQL for the second question:

```sql
SELECT a.amc_name,
       COUNT(DISTINCT h.scheme_id) AS schemes,
       COUNT(DISTINCT h.folio_id) AS folios,
       SUM(h.market_value) AS market_value_inr,
       MIN(h.nav_date) AS first_nav_date,
       MAX(h.nav_date) AS last_nav_date
FROM sw_holdings h
JOIN sw_amcs a ON a.amc_id = h.amc_id
GROUP BY a.amc_name
ORDER BY market_value_inr DESC;
```

The sample has six AMCs, each with three schemes and four distinct folios. Combined market value is INR 635,082, valued on June 30, 2026. No Python is needed for these calculations.

## Examples: SQL followed by Python

The user asks a business question; the planner identifies whether an iterative calculation or resampling is needed and selects `execution_mode: "sql_python"` automatically. The user does not need to name SQL, Python or an algorithm. The published plan must explain the choice in `execution_reason` and the additional calculation in `python_reason`, and include schema inspection, SQL preparation and a Python step. Monty receives the complete stored SQL results; it does not connect to the database or read Parquet itself.

```sh
# Business question: estimate uncertainty in average daily spending.
uv run agent.py --tenant-id 1 --workspace-id 1 --trace \
  "How uncertain is our average daily approved INR net card spending? Give a 95% confidence range based on the observed days."

# Business question: balance an administrative review workload.
uv run agent.py --tenant-id 2 --workspace-id 2 --trace \
  "Split AMCs into three review groups with roughly equal total holding value. Show the group members and totals."
```

The workspace's planner guidance supplies documented analysis defaults. For the first question it chooses a nonparametric bootstrap (1,000 resamples, seed 42, nearest-rank percentile bounds) after querying observed daily totals. For the second it chooses a largest-first greedy assignment after querying AMC values, with deterministic tie breaks. That is approximate administrative grouping, not an optimal allocation or investment recommendation. These decisions are made and published by the planner, without asking the business user to select the implementation.

A published plan for the review-group question will show an explanation like:

```text
Approach: Query the data, then run an additional calculation (SQL + Python)
Python required: yes
Why: Balancing review groups needs repeated assignments after totaling AMC holdings.
Additional calculation: Assign each AMC to the group with the smallest running total.
s1 [schema] Inspect the available holding and AMC fields.
s2 [sql] Total holding value for each AMC.
s3 [python] Balance the AMCs across three review groups using the published method.
```

For a simple totals question, it instead publishes `Python required: no` and explains why a data query is sufficient. If SQL alone answers the question, no Python step is added and Monty is not started.

The following SQL and Python snippets illustrate what the agent generates internally; they are not instructions the business user must include. For the confidence-range example, representative SQL is:

```sql
SELECT CAST(transaction_timestamp AS DATE) AS spend_date,
       SUM(amount) AS net_spend
FROM cc_transactions
WHERE status = 'APPROVED' AND currency = 'INR'
GROUP BY CAST(transaction_timestamp AS DATE)
ORDER BY spend_date;
```

A subsequent `run_python` call can process the first SQL result (`query_1`) like this:

```python
import random
from decimal import Decimal

random.seed(42)
amounts = [row['net_spend'] for row in data['query_1']]
n = len(amounts)
if not n:
    raise ValueError('No observed spending days')

means = sorted(
    sum((random.choice(amounts) for _ in range(n)), Decimal('0')) / n
    for _ in range(1000)
)
{
    'observed_days': n,
    'mean_daily_inr': sum(amounts, Decimal('0')) / n,
    'bootstrap_95pct_interval_inr': [means[24], means[974]],
    'confidence_level_pct': 95,
    'resamples': 1000,
    'seed': 42,
}
```

Verified against the sample data in DataFusion and Monty: 163 observed days, mean daily net spending about INR 16,939.84, and the seeded bootstrap interval about INR 15,172.07–18,715.49. These endpoints are for the representative `random.choice` implementation above; another valid resampling implementation can produce slightly different endpoints. This uses days with approved INR transactions; absent days are not inserted as zeros. The interval describes resampling of the synthetic observations, not a spending forecast.

Use `--plan-only` with any example to publish and inspect the proposed mode and steps before running it. Plans are generated by the selected tenant model; `--trace` shows the actual queries and code it executes.

## More SQL + Python business questions

These examples target iterative assignment, resampling or clustering after SQL preparation. The prompts describe the business outcome without requiring the user to request a programming language. The planner should publish the method and assumptions it chooses. These are suggested hybrid analyses supported by the available fields; they have not all been exercised end to end against the configured LLMs.

### Workspace 1: uncertainty in average customer spending

```sh
uv run agent.py --tenant-id 1 --workspace-id 1 --trace \
  "How sensitive is average approved INR net spending per customer to changes in our customer mix during January–June 2026? Give a 95% uncertainty range using the observed customers as the basis. Include every registered customer, including those with no qualifying spending, and show the actual observed mean separately."
```

SQL should aggregate approved INR transactions per customer and left-join the customer register, assigning zero to customers with no qualifying transactions. A suitable Python calculation resamples whole customers and calculates the average for each resample. Preserve negative approved refunds in net spending. This addresses variation in a hypothetical customer mix, rather than uncertainty in the recorded total or a prediction of future spending. The sample contains 20 registered customers.

### Workspace 1: compare spending between two periods

```sh
uv run agent.py --tenant-id 1 --workspace-id 1 --trace \
  "Compare average daily approved INR net spending in January–March versus April–June 2026. Using only days with qualifying transactions, assess how unusual the observed difference would be if those daily totals were randomly reassigned between the two periods. Show the difference, day counts and a two-sided randomization result."
```

SQL should return one net-spending total per observed day and its original period. Python can repeatedly shuffle period assignments while retaining the two original group sizes, comparing the absolute simulated mean difference with the absolute observed difference. Publish the number of shuffles, seed, sign convention and p-value calculation. This illustrative randomization assumes days are exchangeable under the null hypothesis; the synthetic data cannot establish that real spending lacks seasonality or serial dependence. The result does not establish that the period caused a change.

### Workspace 1: balance customer-review workloads

```sh
uv run agent.py --tenant-id 1 --workspace-id 1 --trace \
  "Split customers into three operations review groups with roughly equal January–June 2026 approved INR net spending. Keep each customer in one group. Assign customers from highest to lowest spending, each time choosing the group with the lowest running total. Show members, group totals and the largest-to-smallest group gap."
```

SQL should produce one row per registered customer, including zero for missing qualifying spending. Python should maintain the three running totals and assign each customer once; break customer ties by customer ID and group ties by group number. This greedy procedure is an approximate administrative workload split. It does not establish optimal groups or create customer risk scores. Report negative customer net totals if present instead of silently clipping them.

### Workspace 2: uncertainty in a difference between customer groups

```sh
uv run agent.py --tenant-id 2 --workspace-id 2 --trace \
  "Compare average current holding value per customer for CONSERVATIVE and AGGRESSIVE customers. Give the observed CONSERVATIVE-minus-AGGRESSIVE difference and a 95% uncertainty range based on changes in the observed customer mix within each profile. Keep each customer's whole portfolio together, and show group sizes and the valuation date."
```

SQL should total holdings per customer and join `sw_customers` for the stored risk-profile labels. A suitable Python calculation independently resamples whole customers within each profile and returns the difference between resampled means. The seed contains four customers in each profile, so disclose the very small groups. Use the actual observed group means for the point estimate. The interval describes a customer-composition scenario; it does not measure investment returns, prove that a risk profile causes higher holdings or provide investment advice.

### Workspace 2: group customers by their holding mix

```sh
uv run agent.py --tenant-id 2 --workspace-id 2 --trace \
  "Group customers into three descriptive segments with similar proportions of holding value in EQUITY, DEBT and HYBRID schemes. Show each segment's customers, average category proportions, customer count and total holding value at the supplied valuation date. Explain how the groups were formed."
```

SQL should aggregate market value by customer and scheme category, joining holdings to schemes on `scheme_id`. Python can run a small iterative clustering calculation on each customer's three category proportions. Publish the initialization, distance measure, seed when applicable, iteration limit, convergence rule and empty-cluster handling. Compute proportions using each customer's total market value; define handling of zero-value portfolios before dividing. These are descriptive portfolio-mix segments, separate from the stored risk-profile labels. Similar synthetic portfolios may yield weak separation; do not invent meaningful distinctions or recommendations. Monty has no scikit-learn dependency: the generated calculation must use supported Python primitives.

### Workspace 2: balance folio-review workloads

```sh
uv run agent.py --tenant-id 2 --workspace-id 2 --trace \
  "Allocate folios into four operations review queues with roughly equal current holding value. Keep each folio intact. Take the highest-value remaining folio and assign it to the queue with the lowest running value. Show each queue's folio members, count and total, the overall total, and the largest-to-smallest queue gap. Include the valuation date."
```

SQL should sum holdings per folio before attaching folio/customer/AMC labels, preserving one row per folio. Python should carry out the repeated lowest-total assignment, with folio-ID and queue-number tie breaks. Each of the 24 folios should appear exactly once, and the queue totals should reconcile to INR 635,082 in the seed at NAV date 2026-06-30. This groups administrative work by recorded value; it does not move assets, split folios or make an investment allocation.

For resampling examples, publish the sample unit, seed, resample count and interval definition. For iterative examples, publish deterministic tie breaks or initialization and state approximation/convergence limits. Retain all requested outputs in one Python calculation when they share inputs. Every selected SQL result must fit the 200-row limit without truncation; aggregate at the customer, day, category or folio grain before passing it to Monty. The planner can select SQL alone if it identifies a suitable direct query for a requested calculation; do not add a Python step solely to match an example heading.

## Workspace 2: uncertainty and sensitivity examples

These examples use **tenant 2 / workspace 2** (SmartWealth). Inspection of the seeded PostgreSQL data through DataFusion found:

| Available evidence | Sample coverage |
| --- | --- |
| Customer portfolios | 12 customers, 24 folios, 18 schemes and 72 holding rows |
| Current holding valuation | INR 635,082 in total; mean per customer INR 52,923.50; NAV date 2026-06-30 |
| Transaction history | 432 transactions on six dates: the fifth of each month, January–June 2026 |
| Transaction statuses | 402 completed, 18 failed SIP transactions and 12 pending redemptions |
| Pending redemption proceeds | INR 4,080 at the recorded transaction amounts |
| Coverage per AMC | Six AMCs, each with four distinct customers and six transaction dates |

All amounts are INR. `sw_transactions` has no `customer_id`: join it to `sw_folios` on `folio_id` to identify customers. `sw_holdings` already contains `customer_id`. Purchases, SIPs and redemptions all have positive transaction amounts; subtract completed redemptions when calculating net inflows. Failed and pending transactions do not affect current holdings.

### Customer-mix uncertainty in average holding value

```sh
uv run agent.py --tenant-id 2 --workspace-id 2 --trace \
  "If our customer mix changed, how stable would average holding value per customer be? Give a 95% uncertainty range based on these observed customers, keep each customer's whole portfolio together, and show the observed average, customer count and valuation date."
```

A suitable approach is to total holdings **per customer** in SQL, then resample whole customer portfolios in Python and calculate the mean for each resample. Report the actual observed mean separately from the mean of the simulated means. The observed mean is INR 52,923.50 across 12 customers. Treating the 72 holding rows as independent customers would overstate the sample size.

The range describes hypothetical changes in the mix of the observed customers. The stored total and observed mean are known exactly at the supplied NAV; this range does not measure uncertainty in that recorded valuation or predict future NAVs. The dataset is synthetic and is not a representative sample of investors.

### Customer-mix uncertainty in AMC concentration

```sh
uv run agent.py --tenant-id 2 --workspace-id 2 --trace \
  "Which AMC has the largest share of current holding value? How sensitive is that same AMC's share to changes in our customer mix? Give a 95% uncertainty range based on the observed customer portfolios, keeping each customer's holdings together across all AMCs. Include the valuation date."
```

SQL can prepare a customer-by-AMC valuation table. A suitable Python calculation resamples customers, carrying each selected customer's entire portfolio across all AMCs, and recalculates the selected AMC's share of total value. Select the largest AMC from the original portfolio and keep that AMC fixed across resamples; selecting a new winner in each resample answers a different question. Preserve customers with zero exposure to the selected AMC and count repeated customer selections with their full multiplicity.

This is sensitivity to customer composition. It does not estimate investment performance or the probability that an AMC will outperform. Only four distinct customers contribute holdings to each AMC in this sample, so disclose the limited coverage.

For both customer-mix questions, the planner should publish its chosen method, sampling unit, resample count, seed and interval definition before execution. The SmartWealth profile currently has no dedicated uncertainty-analysis defaults; these are suitable analysis approaches, not guaranteed preconfigured algorithms. Review the published plan, and expect a focused clarification if the model cannot establish the intended assumptions. If a bootstrap is selected, keep related statistics in one Python step and return the observed statistic, interval bounds and method parameters together.

### Observed variability in monthly net inflows

```sh
uv run agent.py --tenant-id 2 --workspace-id 2 --trace \
  "How variable were completed monthly net inflows from January through June 2026? Show all six monthly totals, the average, minimum and maximum. Explain the separate contributions from purchases, SIPs and redemptions."
```

This descriptive question can be answered with **SQL only**. The six completed monthly net-inflow totals are INR 121,125; 127,038; 133,059; 139,188; 110,889; and -19,140, respectively. June contains redemptions, while the earlier months contain purchases or SIPs. The minimum–maximum range describes observed variation; it is not a 95% confidence interval. Six deliberately patterned monthly observations do not establish a stable process for forecasting future inflows. Do not treat the 432 transaction rows as 432 independent months.

### Bounds on redemption payouts with pending requests

```sh
uv run agent.py --tenant-id 2 --workspace-id 2 --trace \
  "For January through June 2026, show completed redemption proceeds and pending requested proceeds by AMC. What are the lower and upper totals for completed plus pending requests if none or all of the pending requests complete at their recorded amounts? Keep current holdings unchanged in this report."
```

This scenario can also be answered with **SQL only**. Across all AMCs, completed redemption proceeds are INR 19,140 and pending requested proceeds are INR 4,080. The conditional bounds are therefore INR 19,140–23,220. These are scenario bounds at recorded transaction amounts, not a statistical confidence interval or a prediction that pending requests will complete. Do not add pending redemptions to completed holdings or treat pending status as failure.

Use `--plan-only` in place of `--trace` to inspect any proposed plan without loading data or executing calculations. The current dataset has a single latest scheme NAV snapshot and transaction-time NAVs, rather than a regular historical NAV series. Examples asking for reliable future portfolio-value ranges, volatility or value-at-risk require additional price history and explicit modeling assumptions.

Compound requests are answered partially when appropriate:

```sh
uv run agent.py --tenant-id 2 --workspace-id 2 "Show holdings by AMC and what is their next best action?" --trace
```

The SmartWealth glossary defines AMC as Asset Management Company. The classifier marks the holdings aggregation ready and uses configured/user-provided action rules when complete, or records a separate question about the meaning of "next best action" when criteria are missing (portfolio review, customer outreach, rebalancing, or another policy). SQL retrieves the holdings by AMC; the final response includes those results and the pending action-policy question. The application preserves and appends deferred questions even if the executor omits them. No action policy or investment recommendation is invented from the holdings. A holdings-only answer does not mention next-action availability at all. The default SmartWealth profile currently has an empty `action_policies` mapping: clarification examples are not actual decision rules. Add the business-owned conditions and actions under named entries when available; do not assume rules are absent when the user supplies them in the question. Clarifications asking which tenant/workspace/dataset to use, or asking for a glossary-defined acronym, are rejected and retried.

## Summary integrity

The final agent receives actual SQL and Python results, executed query text and host-verified Decimal totals. `summary_evidence.py` totals only profile-approved additive columns, keeps configured currency/date partitions separate, and never totals truncated result sets. Unsupported numeric values or ISO dates cause a correction retry; the executor draft is not accepted as numeric evidence. Drafts containing unsupported numeric/date claims are omitted from the summarizer input. Negative currency forms such as `-₹27,094.29` and explicitly labeled loss magnitudes are checked against the corresponding signed values. These checks catch an invented grand total such as INR 615,082 when the returned AMC rows total INR 635,082. They do not replace semantic validation of joins or prove every narrative interpretation.

The summary must attribute fields using executed SQL, rather than guessed columns in a pre-execution plan. It must not label planned validation checks as completed checks. Unrequested next-action and policy disclaimers are rejected using workspace topic-scope rules.

## DataFusion and the Monty sandbox

**Monty does not run DataFusion.** DataFusion's `SessionContext` executes read-only SQL in the host Python process (`data_access.py`). PostgreSQL rows are first loaded by Psycopg into Arrow; Parquet is scanned directly by DataFusion. These operations occur outside Monty.

For a hybrid plan, the host retrieves complete authorized query results and passes their typed rows into Monty as `data[result_id]`. The Monty worker performs only the generated Python calculation. It has no database connection, filesystem mount, query callback, DataFusion/PyArrow object or native DataFusion import. Python outputs return to the host and become evidence for the final summarizer. Tests verify that `import datafusion` fails inside Monty, while SQL succeeds in the host.

SQL returns at most 200 rows with an explicit truncation flag. Python accepts selected `result_ids` only after SQL has run; it rejects unknown IDs and truncated results. Its `data` input maps result IDs to their typed rows, so dates and Decimal amounts survive the boundary. For example, `sum(row['value'] for row in data['query_1'])` processes a prior SQL result. Each invocation uses a fresh Monty checkout with a 32 MB heap and a five-second interpreter execution budget. The approved scientific module is the only host-function bridge. Generated Python receives no filesystem mounts, network APIs or database credentials. SciPy runs in a separate supervised worker with independent limits. DataFusion also runs outside Monty's resource limits; use the supplied small datasets.

PostgreSQL snapshots refresh after classification and planning at the start of each applicable question. Schema and query results are sent to the configured LiteLLM model as context. LiteLLM must expose the model IDs declared in the tenant mapping. The CLI returns a clear error if a model or selected database is unavailable.

### SciPy calculations through Monty

Generated code imports the allowlisted **`scientific` module** and awaits its
functions. Scientific algorithms execute in a separate CPython worker; Monty
retains the generated code, complete authorized input rows and custom business
calculations. Direct `import scipy` or `import numpy` inside Monty remains
unsupported. The application process does not import the scientific libraries.

```text
SQL contracts → typed data[result_id] → Monty
                                        ↕ approved async numerical calls
                                  SciPy/NumPy worker
```

`scientific_capabilities.json` supplies signatures and policies to planning and
code generation. Both workspaces use the same domain-independent operations:

| Operation | Implementation and output |
| --- | --- |
| `await scientific.kmeans(matrix, clusters=3)` | SciPy `kmeans2`; population scaling, farthest-first or k-means++ initialization, labels in input order, centroids, counts and convergence |
| `await scientific.bootstrap(values)` | SciPy bootstrap for mean/median/sum/population std; confidence bounds and bootstrap distribution |
| `await scientific.sample("normal", {"loc": 0, "scale": 1}, size=1000)` | Batched SciPy draws for general Monte Carlo calculations; supports normal, uniform, binomial and Bernoulli |
| `await scientific.binomial(10, 0.5, size=1000)` | Binomial draws plus theoretical mean/variance |
| `await scientific.bernoulli(0.5, size=1000)` | Coin flips (0/1) with the requested success probability |
| `await scientific.resample_indices(n, n_resamples=1000)` | NumPy index samples with replacement, for custom statistics on original Decimal rows |
| `await scientific.monte_carlo_test(values)` | SciPy Monte Carlo hypothesis test; statistic, p-value and null distribution |

General Monte Carlo outcome simulations use `sample`; a hypothesis-test p-value
does not answer a portfolio-loss distribution question. For example, probabilities
for several cohorts can be passed as `{"p": probabilities}` with
`size=[1000, len(probabilities)]` to obtain one batched Bernoulli simulation matrix.
Monty can apply the resulting 0/1 defaults to original Decimal exposures/LGD.

Example generated clustering code:

```python
import scientific
from decimal import Decimal

rows = sorted(data["query_1"], key=lambda row: row["customer_id"])
columns = ["spend_per_card", "profit_per_card", "opex_per_card"]
out = await scientific.kmeans(
    [[row[name] for name in columns] for row in rows],
    clusters=3, standardize=True, initialization="farthest_first", seed=42,
)
groups = []
for label in sorted(set(out["labels"])):
    members = [row for row, assigned in zip(rows, out["labels"]) if assigned == label]
    groups.append({
        "segment": label + 1,
        "members": [row["customer_id"] for row in members],
        "count": len(members),
        "typical_values": {
            name: sum((row[name] for row in members), Decimal("0")) / len(members)
            for name in columns
        },
    })
{"groups": groups, "method": out["method"], "warnings": out["warnings"]}
```

The column aliases must match the actual SQL output contract. Scientific arrays
and interval calculations use `float64`; original SQL amounts stay Decimal for
accounting totals, observed monetary means and cluster monetary averages.
Custom exact-decimal resampling can use `resample_indices` and calculate the
statistic in Monty. Do not pass sandbox lambdas, functions or code strings into
the worker; the approved statistic names select trusted host functions.

Each call uses `Generator(PCG64)` with `SeedSequence([seed, stream])`. The defaults
are seed 42 and stream 0. Identical seed/stream inputs reproduce draws; use
different streams for independent batches. These draws differ from Python
`random`, even with seed 42. The pinned worker libraries are NumPy **2.5.4** and
SciPy **1.18.1**. Results and `SCIENTIFIC_RESULT` trace events include actual
versions, RNG and method parameters. New seeded results need not match older
handwritten algorithms bit for bit.

Bootstrap defaults are 1000 resamples, a 95% percentile interval and exact
nearest-rank indices (24/974 for 1000 resamples). SciPy basic/BCa intervals require
`quantile_method="linear"`; these methods are explicit choices, not silent
replacements. Empty/degenerate input, fewer distinct clustering profiles and
non-convergence must be disclosed using actual returned results/warnings.

Defaults in `.env.example` bound callbacks to 64 suspensions per Monty checkout,
30 seconds per scientific call, 120 seconds accumulated scientific execution per
request, 512 MiB monitored worker RSS, 250000 array cells, 20000000 work units,
10000 observations/resamples, 500 clustering iterations and 4 MB wire messages.
Scientific imports and calls consume suspensions, so batch sampling instead of
calling once per draw. The RSS supervisor checks every 25 ms and terminates an
over-budget worker; transient overshoot is possible. Linux additionally enforces
a virtual-address-space ceiling of four times the RSS budget (at least 1 GiB) to
allow native-library mappings. Monty's heap/time limits do not cover SciPy.

The numerical worker receives validated numeric requests over JSON, a minimal
environment with no LLM/database credentials, and a fixed operation allowlist.
It never evaluates generated Python. Timeout/cancellation kills the worker;
a later call can start a fresh process. Failures become normal Python-tool
correction feedback. Unawaited scientific results are rejected. SQL-only plans
do not create Monty or scientific workers; hybrid plans start the numerical
worker lazily on their first scientific call.

Run the local integration tests with PostgreSQL running (no LLM/API key required):

```sh
uv run python -m unittest discover -s tests -v
```

The tests cover chain order, preview stops, service replacement, publication-failure propagation, domain-aware tool-free summarization, DataFusion exclusion from Monty, business-language questions with automatic execution selection, published Python-required decisions and reasons, rejected requests for the user to choose a language, plan publication before data access, scoped plan records, owner-only file permissions, publication failure stopping execution, domain-agnostic base prompts, stage-specific profile injection, workspace profile isolation, invalid profile references, workspace glossary context, clarification retries, partial compound answers with real SQL evidence, preservation of pending questions, tenant/workspace ownership, mapped model selection across all stages, multiple workspaces with distinct storage locations, repeated workspace IDs under different tenants, intent-only classification, separate planning before data loading, enforced step order, corrected-query retries, rejected premature answers, required/unknown workspace IDs, scoped classifier metadata, rejected workspace switching, blocked cross-workspace SQL in both directions, validated intent plans, classification before loading, selected-source isolation, SQL-only execution without Monty, combined SQL/Python execution, clarification without source access, CTE and external-source restrictions, decimal precision, truncation handling, and sandbox filesystem isolation. Agent tests use Pydantic AI's deterministic `FunctionModel` and make no model API calls.

Implementation follows the [Monty Python quickstart](https://pydantic.dev/docs/monty/quickstart/python/) and [Pydantic AI compatible model configuration](https://pydantic.dev/docs/ai/models/openai/).

Python API callers must also supply both IDs:

```python
import asyncio
from agent import ask, classify_question, plan_question

answer = asyncio.run(ask("How many customers?", tenant_id="1", workspace_id="1"))
intent = asyncio.run(classify_question("Which AMC has the largest holdings?", tenant_id="2", workspace_id="2"))
intent, plan = asyncio.run(plan_question("Which AMC has the largest holdings?", tenant_id="2", workspace_id="2"))
```

Before sandbox execution, Python preparation checks unresolved global names, including names used inside lambdas, comprehensions and helper functions. A missing `Decimal` import is supplied from the sandbox-native `decimal` module. Other undefined names cause a correction request naming the missing identifiers. Replacing the host-provided `data` with copied literal rows is rejected; deriving a local view from the supplied input remains supported. Each snippet must define its helpers and import supported modules; only the allowlisted scientific module can issue external callbacks. Its imports and calls consume a bounded suspension allowance; all scientific functions must be awaited.

Python calculation results must be structured, non-null values read from stored SQL inputs. A final `print(result)` is normalized to return `result`. Copied input rows are rejected when code does not read `data`; a null result leaves the calculation step pending for correction. Summarization requires the completed Python results, preventing printed-only output from triggering unsupported-number retries.

Adjacent Python plan steps are consolidated before publication, preserving their requested outputs in one calculation. Related statistics should reuse intermediate values. Within one request, identical normalized Python code on identical stored SQL inputs and the same scientific library/settings fingerprint reuses its successful result; `PYTHON REUSED` records identify the original step without repeating the code or invoking Monty. Separate requests have separate caches. Failed calculations remain retryable.

## Execution-mode analysis of recent financial questions

**All nine questions in `QUES.md` and both recent PAT/margin questions can use SQL only under the documented definitions and default assumptions. None requires Monty merely because it asks for a scenario, forecast or attribution.** Here, SQL only means `execution_mode: "sql"`: DataFusion performs the calculations, the Python host orchestrates the pipeline, and the LLM still classifies, plans and summarizes. Monty is not started.

All eleven questions below belong to **tenant 1 / workspace 1**. Run the commands from the project root. The first nine preserve the exact question text from `QUES.md`, using this repository's `agent.py` entry point. The PAT command supplies example X=10%, Y=50bps; the NIM command supplies COF +20bps, yield +25bps and transactor share +5pp. Replace those example values with the intended scenario. The classifications describe the implemented reference calculations; they are not a claim that every live model-generated plan has been verified.

| Question | Execution mode | Why SQL is sufficient | Reference | Executable command |
| --- | --- | --- | --- | --- |
| Which variant is dragging profitability? | **SQL only** | Join product/card metadata, sum contribution profit, calculate profit/card and sort the variants. | [Variant ranking](queries/credit_cards/01_variant_profitability.sql) | `uv run agent.py --tenant-id 1 --workspace-id 1 --trace --max-tokens 8192 "which variant is dragging profitability"` |
| What factors in Diners Black are pulling down profitability? | **SQL only** | Filter the variant and aggregate its income and expense components; comparisons use the same month and denominator. This is accounting contribution analysis, not a causal model. | [P&L components](queries/credit_cards/02_diners_black_drivers.sql) | `uv run agent.py --tenant-id 1 --workspace-id 1 --trace --max-tokens 8192 "what factors in Diners Black are pulling down profitability"` |
| Which variant is good for sourcing customers with high profitability and less opex? | **SQL only** | Calculate profit/card and opex/card, then use a self-join to identify Pareto-efficient alternatives. No iterative optimizer is needed for this comparison. | [Sourcing comparison](queries/credit_cards/03_sourcing_comparison.sql) | `uv run agent.py --tenant-id 1 --workspace-id 1 --trace --max-tokens 8192 "which variant is good to source more customers with high profitability, less opex"` |
| Which segment causes high losses? | **SQL only** | Group by commercial segment, rank credit cost, and normalize by card count or receivables exposure. Report association and the synthetic assumptions, not proven causation. | [Segment losses](queries/credit_cards/04_segment_losses.sql) | `uv run agent.py --tenant-id 1 --workspace-id 1 --trace --max-tokens 8192 "which segment causes high losses"` |
| Segment profitability per card and spend per card comparison | **SQL only** | Aggregate profit and approved INR spend, dividing by distinct cards with zero-spend cards retained. | [Per-card comparison](queries/credit_cards/05_segment_per_card.sql) | `uv run agent.py --tenant-id 1 --workspace-id 1 --trace --max-tokens 8192 "segment profitability per card, spend per card comparison"` |
| If spends grow 10% and credit cost rises 50bps, what is adjusted profitability? | **SQL only** | Apply the legacy pre-tax scenario's fixed decimal multipliers to spend-linked contribution and receivables-based credit cost. | [Legacy spend scenario](queries/credit_cards/06_spend_credit_scenario.sql) | `uv run agent.py --tenant-id 1 --workspace-id 1 --trace --max-tokens 8192 "if spends grow by 10% and credit cost rises by 50 bps what is adjusted profitability"` |
| Impact of a 20bps cost-of-funds change on portfolio margin | **SQL only** | Calculate the annual-rate shock on funded exposure using days/365, then divide the income/profit impact by the specified margin denominator. Show both directions when unspecified. | [Funding sensitivity](queries/credit_cards/07_funding_sensitivity.sql) | `uv run agent.py --tenant-id 1 --workspace-id 1 --trace --max-tokens 8192 "impact of cost of funds change of 20 bps on portfolio margin"` |
| Which P&L parameters contributed most to month-on-month profitability change? | **SQL only** | Calculate signed component differences between two months, rank absolute contributions, and reconcile the bridge to total profit change. | [Monthly bridge](queries/credit_cards/08_monthly_profit_bridge.sql) | `uv run agent.py --tenant-id 1 --workspace-id 1 --trace --max-tokens 8192 "which p&l parameters contributed most to month-on-month profitability change"` |
| Forecast credit cost from delinquency and roll-rate movements | **SQL only, for the default one-month fixed-book forecast** | Aggregate transition probabilities, join matching non-default exposures, multiply by default probabilities and LGD, and sum. Previous/current movements are ordinary grouped/window calculations. This is a conditional projection, not a fitted predictive model. | [Credit-cost forecast](queries/credit_cards/09_credit_cost_forecast.sql), [observed movements](queries/credit_cards/support/risk_movements.sql) | `uv run agent.py --tenant-id 1 --workspace-id 1 --trace --max-tokens 8192 "forecast credit cost from delinquency and roll-rate movements"` |
| Spends grow X%, revolve rate stays flat so revolving ENR scales, credit cost rises Y bps; what is adjusted PAT with other P&L lines flat? | **SQL only, once X/Y and assumptions are supplied** | Scale ENR/ANR and interest/interchange income, recalculate credit cost on the declared base, hold specified costs flat, then calculate PBT, tax and PAT at the documented rounding grain. | [Flat-cost PAT](queries/credit_cards/scenarios/10_flat_cost_pat.sql) | `uv run agent.py --tenant-id 1 --workspace-id 1 --trace --max-tokens 8192 "If approved INR spends grow by 10%, the revolve rate stays flat so revolving ENR scales with spend, and the annual credit-cost rate rises by 50 bps, what is adjusted PAT? Apply credit cost to total ENR. Hold rewards, other fee income, all opex, network cost and funding cost flat. Keep interchange take-rate unchanged and use the dataset tax policy. Show base and adjusted PBT, tax and PAT."` |
| Combined COF, revolve-yield and transactor-share shifts: impact on portfolio margin | **SQL only, once signed changes are supplied** | Calculate the changed revolving ANR, yield and funding cost; include the yield-by-mix interaction and aggregate annualized NIM. `transactor_share_pct_pt_change=+5` adds 0.05 to the share and reduces revolving ANR at fixed total ANR. | [Combined NIM sensitivity](queries/credit_cards/scenarios/11_portfolio_nim.sql) | `uv run agent.py --tenant-id 1 --workspace-id 1 --trace --max-tokens 8192 "What is the combined impact on annualized portfolio net interest margin if COF rises 20 bps, revolve yield rises 25 bps, and transactor share rises 5 percentage points? Hold total ANR and funded balance fixed. Show the change in revolving ANR, component effects, the yield-mix interaction, and base versus adjusted NIM."` |

The two spend scenarios are different: the legacy question scales spend-linked expenses and holds interest/balances fixed; the explicit flat-cost PAT question scales revolving balances/income while holding the named costs flat. Do not reuse one scenario's assumptions for the other. PAT also requires the declared tax policy, whereas the original profitability examples use pre-tax contribution profit.

A question containing literal placeholders X/Y or unspecified rate changes may need clarification or a symbolic formula before numerical execution. Likewise, a forecast with an unspecified horizon/model needs its assumptions resolved. Missing inputs do not, by themselves, make a question SQL + Python. The reference templates contain example input values, not permission to substitute those values for missing user parameters.

### Changes to the question that would justify SQL + Python

| Added requirement | SQL preparation | Python calculation | Executable command |
| --- | --- | --- | --- |
| A bootstrap uncertainty interval for average profit or PAT per customer | Aggregate the selected month's P&L at customer grain. | Resample whole customers, recompute the requested statistic and return percentile bounds. | `uv run agent.py --tenant-id 1 --workspace-id 1 --trace --max-tokens 8192 "How uncertain is average PAT per customer? Show the average and a 95% range."` |
| A simulation-based distribution of future credit losses under explicit stochastic assumptions | Retrieve aggregated exposures and transition probabilities, with defined horizon and scenarios. | Simulate paths, track defaults without double counting, and calculate loss quantiles. A single conditional expected-loss sum still needs only SQL. | `uv run agent.py --tenant-id 1 --workspace-id 1 --trace --max-tokens 8192 "What could next month's credit losses be? Show expected loss and a loss level exceeded in only 5% of scenarios."` |
| Balance variants, customers or folios into review queues using repeated lowest-total assignment | Aggregate the workload measure at the entity grain. | Perform the greedy assignment while maintaining running queue totals. This is administrative grouping. | `uv run agent.py --tenant-id 1 --workspace-id 1 --trace --max-tokens 8192 "Split customers into three review groups with roughly equal January–June 2026 approved INR net spending. Show members and totals."` |
| Iteratively cluster customer profitability/spending patterns | Prepare one row of agreed features per customer. | Run the published iterative clustering method and return membership and centroids. Do not invent a risk score. | `uv run agent.py --tenant-id 1 --workspace-id 1 --trace --max-tokens 8192 "Group customers into three segments with similar spending, profitability and operating cost per card. Show members and typical values."` |

The questions use business language. The credit-card domain profile supplies the calculation defaults; the user does not need to choose a language, sampling method or clustering algorithm. These four question families require a `sql_python` plan: SQL prepares complete inputs, and the planned Python step executes in Monty. A SQL-only plan for a matching request is rejected before publication.

The configured defaults are:

- **PAT uncertainty:** latest financial month, whole-customer PAT totals including every registered customer, 1,000 resamples, seed 42 and a 95% percentile interval. Report the actual mean separately.
- **Credit-loss range:** next month on a fixed book, 1,000 simulations, seed 42, latest matched transition probabilities and stored LGD. Each non-default variant/segment/bucket exposure is one independent default-or-no-default block. Return mean loss and the 95th-percentile loss. These coarse block assumptions are published; the result is not calibrated borrower-level risk.
- **Review groups:** total approved INR net spending per customer, including zero-spend customers; largest-first assignment to the group with the smallest running total. Break ties by customer ID and group number; report the split as approximate.
- **Customer segments:** latest-month customer features for profit, approved INR spending and opex per card. Standardize features, ignore zero-variance features and use deterministic farthest-first initialization followed by up to 50 k-means iterations. Publish convergence/empty-cluster rules; return membership and typical values in original units. These are descriptive groups, not risk scores.

Explicit user assumptions override defaults. These additional calculations do not change the SQL-only classification of the eleven financial questions above. Ordinary ratios, Pareto comparisons, linear sensitivities, fixed scenario grids and a one-step conditional expected-loss sum remain SQL work. Six synthetic monthly snapshots do not establish forecast validity. Python uses supported Monty primitives and its existing time/memory limits; it cannot process truncated SQL inputs.

To inspect the actual planner decision for a question without executing it:

```sh
uv run agent.py --tenant-id 1 --workspace-id 1 --plan-only --max-tokens 8192 \
  "If approved INR spends grow by 10%, the revolve rate stays flat, and the annual credit-cost rate rises by 50 bps, what is adjusted PAT? Apply credit cost to total ENR. Hold rewards, other fees, opex, network cost and funding cost flat. Use the dataset tax policy."
```

The published plan should say `Python required: no` for this deterministic scenario. Use `--trace` for an executed run to review actual SQL and any Python calls. Reference-query and calculation tests support the capability classifications above; this README table does not replace reviewing a generated plan.

## Credit-card profitability questions

The nine questions in `QUES.md` are supported by synthetic product, card-profile, monthly P&L and delinquency/roll-rate Parquet extensions in tenant 1 / workspace 1. See [question coverage and reference SQL](docs/credit-card-question-coverage.md) and [data assumptions/regeneration](data/parquet/README.md). Financial questions default to the latest available month, use explicit INR/rate/day-count definitions, and disclose synthetic scenario/forecast assumptions.

## Credit-card PAT and portfolio-margin scenarios

The extended dataset supports flat-cost adjusted PAT and combined COF/revolve-yield/transactor-share sensitivities. See [field definitions, formulas and runnable questions](docs/credit-card-pat-and-margin.md). In particular, `transactor_share_pct_pt_change=+5` means adding 0.05 to the transactor share at fixed total ANR, which reduces revolving ANR. PAT is separate from pre-tax profit, and explicit held-flat P&L lines override the older spend-linked-cost scenario.

The documented combined-NIM command uses a workspace-configured, parameterized SQL recipe. Its published plan includes signed inputs and a template hash; execution runs the checked query once without model-generated formula rewrites. The query uses ANR, row-level yield/mix effects, the interaction and recorded base NII. See [recipe behavior and verified NIM results](docs/credit-card-pat-and-margin.md#published-configured-query-for-the-documented-nim-request). Unmatched wording or extra filters/objectives retain normal planning rather than being silently ignored.
