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
| `pipeline.py` | Handler chain, request context, preview stops and injected service adapters |
| `planning.py` | Intent/plan schemas and classifier/planner agents |
| `model_config.py` | Tenant-selected LiteLLM client configuration |
| `data_access.py` | Workspace table allowlist, read-only SQL, result storage and source loading |
| `execution.py` | Ordered execution tools, Monty lifecycle and execution evidence |
| `summarization.py` | Final domain-aware presentation, requested-topic guards and deferred questions |
| `summary_evidence.py` | Decimal totals and numeric/date checks against actual execution evidence |
| `plan_publication.py` | Scoped pre-execution JSON publications |
| `domain_instructions.py` | Scoped domain profile lookup for all four model roles |

`PipelineServices` supplies replaceable adapters for scope resolution, model creation, classification, planning, publication, loading, execution and summarization. Stages use those adapters rather than importing the CLI. `Pipeline(..., stages=[...])` supports a different handler chain, and `set_next()` explicitly links handlers. Each run creates a fresh context, so state and data are not shared between requests.

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

The source-backed [interactive architecture diagram](.archify/architecture-analysis-pipeline-20261028-221642/analysis-pipeline.html) documents the pipeline, tenant/model/domain mapping, source loading and Monty boundary. It includes request-local token overrides, stage logs, Python deduplication, native Decimal preparation, input/result checks and final summary validation. [Architecture validation receipt](.archify/architecture-analysis-pipeline-20261028-221642/analysis-pipeline.finalize-summary.json).

The separate [agent stage responsibilities diagram](.archify/workflow-agent-stages-20261028-221642/agent-stages.html) details **resolve → classify → plan → publish → load → execute → summarize**, showing each stage's inputs, responsibilities, outputs and conditions. It also shows preview exits, failure propagation, tool retries, result reuse and clarification passthrough. These are pipeline stages rather than a persisted state machine. [Stage diagram validation receipt](.archify/workflow-agent-stages-20261028-221642/review-3/agent-stages.finalize-summary.json).

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

The final agent receives actual SQL and Python results, executed query text and host-verified Decimal totals. `summary_evidence.py` totals only profile-approved additive columns, keeps configured currency/date partitions separate, and never totals truncated result sets. Unsupported numeric values or ISO dates cause a correction retry; the executor draft is not accepted as numeric evidence. These checks catch an invented grand total such as INR 615,082 when the returned AMC rows total INR 635,082. They do not replace semantic validation of joins or prove every narrative interpretation.

The summary must attribute fields using executed SQL, rather than guessed columns in a pre-execution plan. It must not label planned validation checks as completed checks. Unrequested next-action and policy disclaimers are rejected using workspace topic-scope rules.

## DataFusion and the Monty sandbox

**Monty does not run DataFusion.** DataFusion's `SessionContext` executes read-only SQL in the host Python process (`data_access.py`). PostgreSQL rows are first loaded by Psycopg into Arrow; Parquet is scanned directly by DataFusion. These operations occur outside Monty.

For a hybrid plan, the host retrieves complete authorized query results and passes their typed rows into Monty as `data[result_id]`. The Monty worker performs only the generated Python calculation. It has no database connection, filesystem mount, query callback, DataFusion/PyArrow object or native DataFusion import. Python outputs return to the host and become evidence for the final summarizer. Tests verify that `import datafusion` fails inside Monty, while SQL succeeds in the host.

SQL returns at most 200 rows with an explicit truncation flag. Python accepts selected `result_ids` only after SQL has run; it rejects unknown IDs and truncated results. Its `data` input maps result IDs to their typed rows, so dates and Decimal amounts survive the boundary. For example, `sum(row['value'] for row in data['query_1'])` processes a prior SQL result. Each invocation uses a fresh Monty worker checkout with a 32 MB heap and a five-second execution budget. No host functions, filesystem mounts, network access, or database credentials are exposed to generated Python. DataFusion runs outside Monty's resource limits; use the supplied small datasets.

PostgreSQL snapshots refresh after classification and planning at the start of each applicable question. Schema and query results are sent to the configured LiteLLM model as context. LiteLLM must expose the model IDs declared in the tenant mapping. The CLI returns a clear error if a model or selected database is unavailable.

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

Before sandbox execution, Python preparation checks unresolved global names, including names used inside lambdas, comprehensions and helper functions. A missing `Decimal` import is supplied from the sandbox-native `decimal` module. Other undefined names cause a correction request naming the missing identifiers. Replacing the host-provided `data` with copied literal rows is rejected; deriving a local view from the supplied input remains supported. Each snippet must define its helpers and import supported modules; sandbox suspensions and external host callbacks remain disabled.

Python calculation results must be structured, non-null values read from stored SQL inputs. A final `print(result)` is normalized to return `result`. Copied input rows are rejected when code does not read `data`; a null result leaves the calculation step pending for correction. Summarization requires the completed Python results, preventing printed-only output from triggering unsupported-number retries.

Adjacent Python plan steps are consolidated before publication, preserving their requested outputs in one calculation. Related statistics should reuse intermediate values. Within one request, identical normalized Python code on identical stored SQL inputs reuses its successful result; `PYTHON REUSED` records identify the original step without repeating the code or invoking Monty. Separate requests have separate caches. Failed calculations remain retryable.
