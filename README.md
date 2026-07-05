# LLM Orchestration API

A small FastAPI service used to work through real LLM orchestration patterns end to end — not slideware, an actual LangGraph graph you can `curl` and inspect. Each pattern gets its own endpoint and its own write-up.

## Patterns

| # | Pattern | Endpoint | Write-up |
|---|---------|----------|----------|
| 1 | **Adaptive Routing** — one query, one of three model tiers (`small` / `frontier` / `specialist`), picked by an embedding classifier with an LLM fallback | `POST /query` | [`html/routing_adaptive_architecture.html`](html/routing_adaptive_architecture.html) |
| 2 | **Orchestrator-Workers** — a lead LLM decomposes the query and dispatches it to whichever of three specialist workers (research / code / analysis) are relevant, in parallel, then synthesizes one answer | `POST /orchestrate` | [`html/orchestrator_workers_architecture.html`](html/orchestrator_workers_architecture.html) |
| 3 | **Autonomous (ReAct)** — reason → act → observe, looping until the agent decides it's done | — | coming next |

Original planning docs: [`html/plan_adaptive_routing.html`](html/plan_adaptive_routing.html) (Part 1). See also [`html/centroid_explainer.html`](html/centroid_explainer.html) for a deeper dive into the embedding-centroid router behind Part 1.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # or: uv venv
pip install -r requirements.txt                      # or: uv pip install -r requirements.txt

cp .env.example .env                                 # then fill in OPENAI_API_KEY
```

Requires Python 3.12 (see `.python-version`).

## Run

```bash
python server.py
# or
uvicorn server:app --reload
```

Defaults to `http://localhost:8000`. Override with the `HOST` / `PORT` / `DEBUG` env vars.

## API

### `GET /health`
```bash
curl localhost:8000/health
```

### `POST /query` — Adaptive Routing
```bash
curl -X POST localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"question": "hello there"}'
```
Returns `{"answer", "route", "route_reason"}` — `route` is one of `small` / `frontier` / `specialist`.

### `POST /orchestrate` — Orchestrator-Workers
```bash
curl -X POST localhost:8000/orchestrate \
  -H 'Content-Type: application/json' \
  -d '{"question": "Should we migrate our REST API to GraphQL, and what would the implementation look like?"}'
```
Returns `{"answer", "workers_used", "plan", "worker_results"}` — `workers_used` shows which of `research` / `code` / `analysis` the orchestrator actually dispatched to for this question.

## Project structure

```
graph/      GraphBuilder, OrchestratorGraphBuilder — LangGraph StateGraph wiring per pattern
nodes/      node functions (router, responders, orchestrator/worker/synthesis nodes)
prompts/    ChatPromptTemplate definitions per pattern
state/      TypedDict state shapes per pattern
llm/        OpenAILLM wrapper + the embedding-centroid router
models/     Pydantic request/response models
html/       per-pattern write-ups (this README links to them above)
assets/     diagrams referenced by the write-ups, including the real compiled graph PNGs
server.py   FastAPI app and endpoints
```
