# HDFC Insight

React + TypeScript, initialized and managed with Yarn, with Vite for development.
The interface connects to the existing Python analysis pipeline through the API
in `../api.py`. The supplied logo is preserved; see [brand analysis](BRAND.md).

## Run locally

From the repository root, start the Python API:

```sh
uv sync
uv run uvicorn api:app --host 127.0.0.1 --port 8010
```

In another terminal:

```sh
cd ui
yarn install --frozen-lockfile
yarn dev
```

Open **http://127.0.0.1:5173**. Vite proxies `/api` to `127.0.0.1:8010`, so no
cross-origin configuration or browser API keys are needed. For a different API
address, start Vite with `INSIGHT_API_URL=http://127.0.0.1:PORT yarn dev`.

The API uses the same `.env`, tenant mappings, domain instructions, model budgets
and data sources as `agent.py`. Tenant/workspace metadata comes from the Python
registry. The selector enforces workspace ownership before model execution.

## Interaction

- Select a tenant, then one of its workspaces. Suggestions change with the domain.
- Enter a question and select **Run analysis**, or use **⌘/Ctrl + Enter**.
- Watch all eight real lifecycle stages. Select a stage to inspect its outcome,
  source/result metadata, queries or calculation results. **Activity** shows tool
  calls, SQL sub-stages and correction attempts; **Follow live** follows the current stage.
- **Run analysis → Outcome** separates **SQL**, **Python** (when used) and **Results**.
  Numbered SQL cards list every recorded statement in execution order. Source panels
  include syntax highlighting, line numbers, visual line wrapping, expand/collapse,
  and exact-source copy/download controls. Python panels show the retained code
  actually executed in Monty and its SQL input IDs. Reused results are labelled
  separately; unavailable source is stated explicitly. Existing in-session runs
  captured before this update do not gain missing Python source retroactively.
- The final Markdown answer appears only after the result event. Tables render
  normally, raw HTML is not enabled, and answers can be copied or downloaded.
- **Stop** closes the stream and cancels the request task. A synchronous DataFusion
  operation already running in a thread may finish before its resources are released.
- History is session-local React state, limited to 20 runs. It is not persisted to
  browser storage or presented as a shared audit log.
- Stage failures retain completed evidence and do not display a fabricated answer.
  A broken connection is reported rather than treated as a successful completion.

## Build and serve with Python

```sh
cd ui
yarn build
cd ..
uv run uvicorn api:app --host 127.0.0.1 --port 8010
```

Open **http://127.0.0.1:8010**. Start/restart the API after building so it mounts
`ui/dist`. The service is configured for local use; no authentication system is
introduced by the tenant picker.

## Endpoints

| Endpoint                           | Purpose                                                                    |
| ---------------------------------- | -------------------------------------------------------------------------- |
| `GET /api/health`                  | API reachability; does not claim the model/database are online.            |
| `GET /api/tenants`                 | Public tenant names/IDs and lifecycle-stage definitions.                   |
| `GET /api/tenants/{id}/workspaces` | Workspaces owned by the chosen tenant, domain and registered-table counts. |
| `POST /api/questions/stream`       | Validated request with SSE lifecycle, activity, result or error events.    |

The POST accepts:

```json
{
  "tenant_id": "1",
  "workspace_id": "1",
  "question": "How many registered customers are in this workspace?",
  "max_tokens": 8192
}
```

It streams `ready`, `stage`, `activity`, `result` and `error` events and heartbeat
comments. Events use request-local context variables, including callbacks from
worker threads. Provider keys, connection locations and absolute publication
paths are not included in discovery or stage-outcome payloads. The browser never
calls a model provider directly. Interactive API documentation: `/docs`.

## Verify

```sh
uv run python -m unittest discover -s tests
cd ui
yarn build
yarn playwright install chromium
yarn test
```

Browser tests start a separate API on port 8012 and Vite on 5174. Their delayed,
deterministic pipeline is confined to `tests/ui_fixture_server.py`; `api:app` always
uses the real pipeline. Tests cover progressive event delivery, stage evidence,
tenant selection, errors, stopping, history, mobile layout, chunked UTF-8 SSE,
exact source copy/download and reused-result labels. The Python evidence fixture
runs its sample calculation in Monty.
