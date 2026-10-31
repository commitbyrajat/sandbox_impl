# Validation and repair architecture

`agent.py` calls `Pipeline.run()`. The default pipeline uses the typed nodes in
`pipeline_graph.py`, built with `pydantic_graph.GraphBuilder`. Stage adapters own
their work, logging and injected services; the graph owns lifecycle transitions.
The custom `Pipeline(..., stages=[...])` chain remains supported.

The [generated lifecycle graph](pipeline-graph.mmd) comes from
`LIFECYCLE_GRAPH.render()`. It includes preview exits. State is request-local and
in memory; the graph does not implement process-restart persistence.

Before every Python-generation request, `python_input_context.py` stringifies
the actual selected `data` binding under `HOST_DATA_OBJECT_JSON`, followed by
runtime Python types, computed Arrow types, nullability and row counts. It uses
current producer result IDs after SQL repair. JSON strings representing Decimal
and date values are display-only; Monty receives the original native values.
Snapshots over 24,000 UTF-8 bytes are explicitly marked as previews, and the
complete stored rows remain available to the sandbox. SQL-truncated inputs are
marked unusable rather than presented as complete data.

## Validation boundaries

| Boundary | Responsibility | Authoritative inputs |
| --- | --- | --- |
| Intent and plan | Describe the business outcome; select a suitable algorithm and declare contracts | Caller tenant/workspace, classified intent, datasource registry, domain instructions, scientific registry |
| SQL generation and validation | Generate scoped SQL, validate CTE projections, relationship keys, compilation, types, observation grain and population before retaining results | Loaded Arrow schemas verified against `datasource.json`; configured join meanings; SQL output contract |
| Python generation and preflight | Generate code using current result IDs, exact aliases/types and supported functions; reject unsupported imports, copied replacement inputs and directly traceable invalid keys | Actual DataFusion output metadata and preview; scientific signatures, input examples and exact result fields |
| Monty execution | Run generated code, with numerical operations delegated through the scientific bridge | Complete stored SQL rows, original Decimal values, allowlisted worker callbacks |
| Calculation and final evidence validation | Check finite outputs and declared fields, counts, member coverage and total reconciliation before success | Python result contract and original SQL producer rows |
| Optional summarization | Present validated results using domain vocabulary; reject unsupported numeric/date claims or unrelated topics | SQL/Python evidence, audit receipts and scoped domain guidance |

The classifier's optional `analysis_kind` describes the requested outcome:
`clustering`, `balancing`, `uncertainty`, `simulation`, `hypothesis_test` or
`custom`. New generated Python steps require `calculation.kind`,
`calculation.method` and `result_contract`. The planner validates the declared
method against the scientific registry. For example, `scientific.kmeans`
implements clustering and cannot be declared as a total-balancing algorithm.
Legacy manually constructed plans can omit these new fields.

These are logical generation and validation roles. The tenant-selected PydanticAI
agent generates the next proposal through the currently permitted tool. Validators
are deterministic Python checks; no extra LLM critic is required for structural
validation. PydanticAI owns its bounded proposal/retry loop inside the graph's
execution node.

## Explicit correction destinations

`RepairFeedback` in `repair_feedback.py` contains `target`, `phase`, `step_id`,
the exact `diagnostic`, rejected `artifact`, authoritative schemas/contracts,
`retries_remaining`, and corrective instructions. It is sent to the LLM through
PydanticAI retry messages. Trace output includes `REPAIR_ROUTE`.

The executor also uses PydanticAI's `ProcessHistory` capability to put the active
repair in a focused generation turn immediately before the next model call. It
includes the plain diagnostic, rejected artifact and required next tool, while
retaining the original tool-call/return pairs. It replaces only its own previous
handoff and removes stale handoffs after the step succeeds. Trace output includes
`GENERATION_HANDOFF`. Rejected artifacts precede the error and authoritative
values, so the final instruction points to the correction. This uses the existing retry/model-call allowance; it does
not add another agent call or reset a budget. The latest dynamic correction state
also carries a structured object rather than a doubly encoded JSON string.

| Failure | Destination | Evidence retained |
| --- | --- | --- |
| Invalid plan shape or unsuitable declared method | `planner` | Question, classified intent and scoped registry |
| Invalid SQL structure, compilation or observation contract | `sql_generator` | Earlier completed steps and the failed SQL diagnostic |
| SQL observations are genuinely inadequate for the pending calculation | `sql_input_repair` | Unaffected SQL results; the published contract remains unchanged |
| Unsupported import/API key, wrong row alias, sandbox runtime error or failed calculation invariant | `python_generator` | Successful SQL datasets and their exact types |
| Unsupported summary claim | `summarizer` | All execution evidence; only presentation is regenerated |

Planner output correction permits three retries, execution tools use
`AGENT_TOOL_RETRIES`, SQL input repair permits at most two reopens, and final
summarization permits two retries. These budgets are separate and are not reset
by changing a repair label. HTTP retries remain controlled by `LITELLM_MAX_RETRIES`.

A successful SQL step is retained once. Rejected SQL fingerprints are not executed
again. SQL input repair invalidates the requested producer and affected Python
outputs, while unaffected SQL results can be replayed without querying DataFusion.
Python failures do not implicitly restart SQL. Cached Python results are checked
against the current result contract before reuse.

## Calculation contracts

Python's `result_contract` is separate from SQL's `output_contract`. Required
fields use dot paths and coarse types. Simple `$.groups[*].total` paths normalize
to `groups.*.total`; expressions and filters are rejected. `$` selects the whole
result. Invariants independently require their selected fields to exist, including
fields not duplicated in `required_fields`. `*` selects list entries or dictionary
values. Invariants are a discriminated union: a count check requires an integer
`expected_count`; input comparisons require `input_step` and `input_column`.
Count checks count wildcard matches, or the entries in a directly selected
collection (`groups` and `groups.*` both count the groups).
An input step must be one of that Python step's declared SQL producers.

```json
{
  "required_fields": {
    "groups": "list",
    "groups.*.total": "numeric"
  },
  "invariants": [
    {"kind": "count_equals", "output_path": "groups.*", "expected_count": 3},
    {
      "kind": "members_equal_input",
      "output_path": "groups.*.members.*",
      "input_step": "observations",
      "input_column": "entity_id"
    },
    {
      "kind": "sum_equals_input",
      "output_path": "groups.*.total",
      "input_step": "observations",
      "input_column": "value",
      "tolerance": "0.01"
    }
  ]
}
```

These checks do not evaluate model-provided Python expressions. Numeric
reconciliation uses Decimal values and the contract's declared tolerance.
Static Python checks intentionally cover directly traceable bindings and keys;
Binding `data = data[result_id]` remains supported and is tracked as a row list;
using a separate `rows` variable keeps access to the input mapping clearer.
Dynamic aliases, computed keys and arbitrary algorithm semantics still depend on
runtime behavior and the declared invariants. A valid schema or successful query
does not prove the business interpretation is correct.

Setting `AGENT_SUMMARIZATION_ENABLED=false` skips the presentation model and its
numeric/topic guards. The deterministic `validate_results` stage still runs, and
the executor report and pending clarification questions remain available.

The summary context retains computed estimates, bounds and full member lists.
Long raw working arrays named `bootstrap_distribution`, `null_distribution`,
`samples` or `indices` are replaced with an explicit omission marker and count.
The complete original calculation output remains stored and is used for evidence
validation. Markdown emphasis on method parameters does not turn those values
into permissible monetary claims.

## Diagram status

The executable graph above is generated from the current code. The historical
Archify diagrams linked in the README remain pinned to their earlier commits.
An additional Archify workflow design is retained under
`.archify/workflow-validation-repair-20261031-115627/candidate.json`; its layout
validation is incomplete (two shared route corridors and one label overlap).
It is a working-tree design draft, not a validated source-backed artifact.
