# NumPy, pandas and SciPy in Monty

The sandbox registers `scientific`, `numpy`, `pandas`, `scipy` and `statistics`.
Every exposed library function is **async** and runs through the existing supervised
CPython worker. The pinned libraries are NumPy 2.5.4, pandas 3.1.0 and SciPy 1.18.1.
Generated code continues to execute in Monty; installing a package alone does not
make it importable there. This follows Monty's documented
[external-module mechanism](https://pydantic.dev/docs/monty/concepts/host-functions/#importing-host-modules).

These are bounded facades, not complete replacements for the native libraries.
`numpy.array`/`asarray` return lists; `pandas.DataFrame` returns records. Use the
function APIs below instead of ndarray/Series/DataFrame methods. Host objects,
arbitrary callbacks, files, network requests, SQL and dynamic imports are unavailable.
Unknown functions and missing awaits return actionable preflight errors.

## Available functions

| Module | Exposed API |
| --- | --- |
| `numpy` | `array`, `asarray`, `reshape`, `sum`, `mean`, `median`, `std`, `var`, `min`, `max`, `percentile`, `quantile`, `sort`, `sqrt`, `abs`, `add`, `subtract`, `multiply`, `divide`, `dot` |
| `numpy.random` namespace | Seeded `binomial`, `normal`, `uniform`, `random` |
| `pandas` | `DataFrame`, `groupby`, `merge`, `sort_values`, `dropna`, `to_numeric` |
| `scipy.stats` | `bernoulli.rvs`, `binom.rvs`, `norm.rvs`, `uniform.rvs`, `bootstrap` |
| `scipy.cluster.vq` | `kmeans2`, returning `(centroids, labels)` with seeded kmeans++ initialization |
| `statistics` | `mean`, `fmean`, `median`, `pstdev`, `stdev`, `pvariance`, `variance` |
| `scientific` | Existing high-level clustering, bootstrap, sampling, resampling and Monte Carlo test APIs |

`scientific.bernoulli_sum(probabilities, weights, n_simulations=1000, quantile=0.95)`
provides a trusted weighted-event distribution. SciPy produces a
`[n_simulations, observations]` draw matrix; NumPy sums across observations to give
one portfolio outcome per scenario. It returns `expected_value` (analytical),
`simulated_mean`, `quantile`, `quantile_level`, `exceedance_fraction`, `samples`
and `method`. The analytical expectation is distinct from the finite-simulation
mean. All numerical outputs use float64.

Use `import numpy as np` and `np.random.binomial`, or `from numpy import random`.
`import numpy.random` is unsupported: Monty reserves its bundled `random` module,
so this facade uses a restricted attribute namespace. `default_rng` generator
objects are not exposed; supply `seed` and `stream` directly to sampling functions.

`scipy.stats.bootstrap` accepts a named statistic (`mean`, `median`, `sum`, `std`)
and returns the documented `estimate`, `lower`, `upper`, `standard_error`,
`bootstrap_distribution`, `method` dictionary, rather than SciPy's native
BootstrapResult. For standardization and convergence metadata, use
`scientific.kmeans`; the lower-level `kmeans2` facade does not standardize features.

The full signatures and result shapes live in `module_catalog()` in
`external_module_models.py`. They are supplied to the executor before Python
generation together with the stringified input object and its actual types.
The planner receives high-level algorithms; low-level library signatures remain
in the code-generation stage to avoid expanding the planning context unnecessarily.
External modules also provide `.pyi` declarations for Monty type checking.
The pending `run_python` tool's code argument includes the exact output contract.
Required return keys remain unchanged even when the final business answer uses
different labels. A result-validation failure names the missing key and the
actual returned keys before the model's next correction attempt.

## Examples inside generated Monty code

Input fields depend on the actual SQL output; these snippets illustrate the API.

The default credit-loss scenario uses the weighted-event library operation:

```python
import scientific

rows = data['query_1']
probabilities = [float(row['probability_of_default']) for row in rows]
weights = [row['exposure_amount'] * row['loss_given_default'] for row in rows]
result = await scientific.bernoulli_sum(
    probabilities, weights, n_simulations=1000, quantile=0.95, seed=42
)
{
    'expected_loss': result['expected_value'],
    'simulated_mean': result['simulated_mean'],
    'loss_95th_percentile': result['quantile'],
    'method': result['method'],
}
```

`question_guidance.credit_loss_distribution.observation_policy` in the credit-card
domain profile binds the configured origin table's complete observation grain and
the library method before plan validation. The generic `observation_planning.py`
adds missing key dimensions, anchors the input at its observation source, removes
an inapplicable single-key population constraint, sets a positive input row bound,
and scopes latest-common inputs. It appends identity sort keys for repeatable seeded
draw assignment. These bindings appear in `OBSERVATION_GRAIN_BOUND` and the published
plan. They contain no SQL or generated Python template. Ordinary grouped reports
and unrelated analysis methods retain their own contracts.

The configured default therefore retains all monthly origin cohorts instead of
silently pooling delinquency buckets into larger default events. The executor
requires the published scientific method to run and checks the observation count
for a single-input, single-batch weighted simulation.

```python
import numpy as np

rows = data['query_1']
values = await np.asarray([row['amount'] for row in rows], dtype=np.float64)
mean = await np.mean(values)
bounds = await np.percentile(values, [2.5, 97.5], method='inverted_cdf')
{'mean': mean, 'lower': bounds[0], 'upper': bounds[1]}
```

Those percentiles describe an input distribution; confidence bounds for its
estimated mean require a bootstrap such as `await scientific.bootstrap(values)`.
`inverted_cdf` implements nearest-rank quantiles; `linear` is the NumPy default.

```python
import pandas as pd

records = await pd.DataFrame(data['query_1'])
grouped = await pd.groupby(records, by='category', agg={'amount': 'sum'})
await pd.sort_values(grouped, by='amount', ascending=False)
```

```python
import numpy as np
from scipy import stats
from decimal import Decimal

cohorts = data['query_1']
probabilities = [float(row['probability_of_default']) for row in cohorts]
draws = await stats.bernoulli.rvs(
    probabilities, size=[1000, len(cohorts)], seed=42, stream=0
)
losses = [
    sum((row['exposure_amount'] * row['loss_given_default'] * Decimal(default)
         for row, default in zip(cohorts, scenario)), Decimal('0'))
    for scenario in draws
]
{
    'mean_loss': await np.mean(losses),
    'loss_95th_percentile': await np.percentile(losses, 95, method='inverted_cdf'),
    'seed': 42,
    'scenarios': 1000,
}
```

This is a conditional whole-cohort default simulation, not a validated borrower
model. The prior opening cohort balance is a PD denominator; current exposure
at risk is a separate measure. A discrete 95th percentile may have fewer than
5% of outcomes strictly above the threshold because of ties.

## Types, limits and SQL repair

NumPy reductions and SciPy statistics use float64. Keep original SQL Decimal
values for exact accounting; the example multiplies/sums losses as Decimal
before requesting statistical summaries. Pandas record transformations preserve
Decimal/date/datetime cells and Decimal sums via an explicit wire codec.
Pandas statistical reductions and `to_numeric` may return approximate numeric
types. Missing tabular values become `None`; NumPy requires finite numeric inputs
and rejects empty mean/percentile inputs instead of returning fabricated zeros.

Sampling uses Generator(PCG64) initialized by SeedSequence([seed, stream]). All
operations share the existing worker time, RSS memory, cell/work, wire-size and
Monty suspension limits. `SCIENTIFIC_RESULT` records actual library APIs, versions
and random parameters. The worker receives no model code or application credentials.

`latest`/`latest_common` queries reject additional hardcoded snapshot predicates.
The supplied June 30 equality conflicted with the latest monthly key, June 1.
Use the dynamic snapshot, or choose an explicit range for an intentional period.
After `request_sql_repair`, `run_sql` exposes and requires revised `query=QueryIR`.
Unchanged logical or compiled queries are rejected before another data execution;
dependent old result IDs are invalidated.

Run the integration checks without an LLM API:

```sh
uv run python -m unittest discover -s tests -p test_external_modules.py
```

Tests exercise real Monty imports/type stubs and native NumPy/pandas/SciPy parity,
Decimal/date/null transport, seeded vector draws, limits, blocked IO, contradictory
snapshots, repair-tool schemas and a SQL-to-Monty credit-loss simulation.
