# 🛡️ AgentShield

**Crash-test any AI agent before deployment.** AgentShield connects to your agent over HTTP,
generates adversarial test cases, injects faults mid-conversation, judges every answer, and —
the differentiator — for each failure produces a plain-English cause + a copy-pasteable fix
backed by trace evidence.

```
Connect agent → Verify connection → Agent knowledge → Generate test cases
                                                              │
                                                     Review test cases
                                                              │
                                                    Run reliability test
                                                              │
                                              (pass) ──► Reliability report
                                              (fail) ──► Explain + suggest fix ──► report
```

- **Backend:** FastAPI + **PostgreSQL** + **Temporal** (durable workflow execution) ·
  OpenAI (isolated wrapper, `app/core/llm.py`) · black-box HTTP adapter.
- **Frontend:** **Next.js** (React + TypeScript) · glassmorphism / dark UI.
- **Agent under test:** anything that answers `POST /chat`-shaped JSON over HTTP — a
  standalone RAG agent you connect (6 ship with this repo), or your own.

---

## Prerequisites
- Python 3.11+ and Node 18+.
- **PostgreSQL** running locally.
- **Temporal CLI** — a run cannot execute without it (see below).
- An OpenAI API key.

## One-time setup
```bash
createdb agentshield                    # PostgreSQL database (once)

cd backend
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env                    # then paste your key into OPENAI_API_KEY=

cd ../frontend && npm install
```

`backend/.env` also holds `DATABASE_URL` (defaults to `postgresql://localhost:5432/agentshield`)
and `TEMPORAL_ADDRESS`/`TEMPORAL_TASK_QUEUE` — override only if you're not running everything
on localhost with defaults.

## Run it
```bash
./run_all.sh
```
One command starts everything: the AgentShield backend (`:8100`), all 6 sample agents
(`:8002`–`:8007`), the Temporal dev server + worker (if the `temporal` CLI is installed), and
the frontend (`:3100`). Ctrl-C stops all of it.

Then open **http://localhost:3100**.

**Temporal is required, not optional.** Every crash-test run is submitted as a Temporal
workflow (`AgentTestWorkflow` / `RunGroupWorkflow`) so it survives a backend restart and runs
several agents in parallel under a shared concurrency budget. If `run_all.sh` doesn't find the
`temporal` CLI, it skips starting it — the backend still comes up and you can browse and connect
agents, but submitting a run then fails with a 503 telling you to start it:
```bash
brew install temporal          # once
temporal server start-dev      # UI at http://localhost:8233
```

To run pieces by hand instead of `run_all.sh`:
```bash
cd backend
.venv/bin/uvicorn app.main:app --port 8100              # backend
.venv/bin/python -m app.temporal.worker                 # worker (needs the Temporal server up)
.venv/bin/uvicorn sample_rag_bot.main:app --port 8002    # + any sample agent you want, see below

cd frontend && npm run dev                               # :3100
```

## The two ways to test an agent
- **Connect Your AI Agent** (`/dashboard`) — point AgentShield at a brand-new endpoint: name,
  URL, optional auth header, then **Agent Knowledge** (required — upload a docs file or type a
  description; at least one). Verify Connection does a real probe. From there: generate test
  cases → review/edit them → run.
- **Existing Agent Testing** (`/existing-agent`) — pick a previously connected agent (or one of
  the seeded sample customers) and reuse its stored test-case library. The endpoint and Agent
  Knowledge are already on file — reconnecting isn't needed, and the same URL/knowledge get used
  again automatically.

Reconnecting the same name + endpoint reuses the existing agent row (its saved knowledge and
test cases stay attached) rather than creating a duplicate.

## Sample agents (6 domains, each standalone with its own KB + chat UI)
Each is a real RAG agent (gpt-4o-mini + retrieval over its own knowledge base), speaks the same
`POST /chat` contract, honors the fault flags, and serves its own chat playground at `GET /`.
Point AgentShield at any of them (enter its `/chat` URL, or pick it from Existing Agent Testing).

| Agent | Domain | Port | Run |
|-------|--------|------|-----|
| Store Support (RAG) | e-commerce returns/refunds/shipping | 8002 | `uvicorn sample_rag_bot.main:app --port 8002` |
| NorthBank Support | retail banking (transfers, fees, fraud, loans) | 8003 | `uvicorn sample_agents.banking:app --port 8003` |
| PeopleDesk HR | HR / benefits (PTO, 401k, leave, payroll) | 8004 | `uvicorn sample_agents.hr:app --port 8004` |
| SafeGuard Claims | auto/home insurance claims | 8005 | `uvicorn sample_agents.insurance:app --port 8005` |
| SkyRoute Airways Support | flights, baggage, fare rules | 8006 | `uvicorn sample_agents.airline:app --port 8006` |
| ConnectWave Mobile Support | telecom plans, billing, roaming | 8007 | `uvicorn sample_agents.telecom:app --port 8007` |

Each one is a deliberate archetype rather than a uniform "good agent":

| Agent | Facts | Security | Fails on |
|-------|-------|----------|----------|
| PeopleDesk (HR) | invents confident numbers | leaks its secret on injection | almost everything — the weak baseline |
| SafeGuard (insurance) | honest, declines gracefully | holds | close to nothing — the hardened baseline |
| SkyRoute (airline) | invents under a failed lookup | holds | recovery / hallucination, isolated |
| ConnectWave (telecom) | honest, declines gracefully | leaks to anyone claiming to be staff | injection, isolated |

New agents share a self-contained factory (`backend/sample_agents/rag_core.py`); each domain
file just supplies its KB docs (current + stale variants), a confidential system-prompt secret
(for injection tests), and sample questions. Their matching knowledge-base markdown files — the
same content you'd upload through **Connect Your AI Agent** — live in `docs/agent_kbs/`.

Scenario generation is **domain-driven and knowledge-grounded**: register an agent with a
description or an uploaded knowledge file and AgentShield generates domain-appropriate
scenarios, with `expected_behavior` set to the actual facts in that knowledge where possible —
not a vague restatement of them.

## Try a sample agent directly (its own chat UI)
Every sample agent ships its own self-contained chat playground:

- **UI:** open e.g. **http://localhost:8002/** — chat with the agent, and toggle **⚡
  tool_timeout / stale_doc / injection** to watch how it behaves under each fault (the trace
  shows tool calls, stale docs, latency, tokens live).
- **Endpoint (black-box API AgentShield uses):**
  ```bash
  curl -s http://localhost:8002/chat -H 'content-type: application/json' \
    -d '{"message":"how long do I have to return an item?","history":[],"faults":[]}'
  # -> { "reply": "...", "trace": { tool_calls, retrieved_docs, latency_ms, tokens } }
  ```
  Request: `{ message: str, history: [{role,content}], faults: ["tool_timeout"|"stale_doc"|"injection"] }`
  · `GET /health` for a liveness check.

Note: the `faults` field is that sample agent's own public API — a real third-party agent
wouldn't expose it, so AgentShield degrades to conversation-only fault testing there (no
injected tool_timeout/stale_doc; system-level faults like `api_unreachable` still work against
any endpoint, simulated at the HTTP layer).

---

## Data model (PostgreSQL, `backend/app/db.py`)
No ORM, no migrations — `init_schema()` runs on backend startup and is idempotent.

- **agents** — one row per connected endpoint (name, endpoint_url, auth_header,
  request_template, response_path, description, and the uploaded `knowledge`/`knowledge_name`).
- **customers** / **customer_agents** — the testing context. A `customer_agents` row is what a
  saved test-case library actually hangs off, so the same agent onboarded twice is two separate
  suites. Agents connected through "Connect Your AI Agent" get an auto-generated
  "New Customer N" the first time their suite is saved.
- **test_cases** — the persistent, reviewable/editable library for one customer-agent
  combination. Replaced wholesale on save/regenerate, not appended to.
- **run_groups** / **runs** — one click of "Run" is a `run_groups` row; each selected agent gets
  its own `runs` row (status, reliability_score, breakdown_json) inside it.
- **scenarios** — the frozen, per-run execution copy of whatever suite was run.
- **conversations** / **messages** — one conversation per scenario per run; one row per turn,
  with the agent's trace attached.

## Execution & scoring
- Every run goes through Temporal: `AgentTestWorkflow` (suite → play → judge → explain failures
  → score) per agent, fanned out under `RunGroupWorkflow` for a multi-agent batch, bounded by
  `AGENT_CONCURRENCY` / `WORK_CONCURRENCY` in `.env`.
- The **Judge** is an LLM call (`app/core/judge.py`) at `temperature=0` with a fixed seed, so the
  same transcript scores the same way twice. `verdict` and `severity` are both derived in code
  from the model's sub-scores rather than trusted as raw fields — a small model saying "high" or
  "med" inconsistently no longer moves the reliability score on its own.
- If an agent's endpoint never answers a single turn, that's judged deterministically as a
  **system** failure (no LLM guessing at a conversation that didn't happen), and the whole run is
  marked `error` rather than `done` — a genuinely unreachable agent doesn't get a misleadingly
  normal-looking score.
- `injection` / `memory` / `contradiction` test cases carry their escalating follow-up question
  as part of the stored suite (not invented fresh at run time), so what you see in Review Test
  Cases is exactly what runs, every time.

## Demo safety
- **View sample report** shows a pre-baked report with no backend/OpenAI needed — insurance
  against a flaky network mid-demo (`GET /runs/demo/report`, `app/demo_report.py`).
- Every OpenAI call has a fallback (scenario bank, safe judge/fix defaults); one failed scenario
  never fails the whole run.
- Runs are capped (`MAX_SCENARIOS` in `.env`, default 10) so a live run finishes quickly.

## How AgentShield reaches the agent
Black-box HTTP only — it never imports an agent's code. The agent is a DB row
(`endpoint_url`, `request_template`, `response_path`, optional `auth_header`); the adapter
(`app/core/adapter.py`) renders the template, POSTs, and extracts `reply` + `trace`. Register
your own from the UI (**Connect Your AI Agent**) or via `POST /agents`.

## Feature notes
- **Custom guidance & edge cases** — optional box on the Configure step. Your domain notes are
  refined by the generator and at least 3 scenarios are targeted at them.
- **System-failure testing** — a fault class AgentShield simulates at the HTTP layer, so it works
  on any endpoint: `api_unreachable`, `api_error` (5xx), `api_timeout`, `malformed_response`.
  Judged deterministically → a **system** failure category with infrastructure-oriented fixes
  (retries, timeouts, monitoring).
- **Cost & latency** — the report's performance panel aggregates avg/max latency, total tokens,
  and estimated $ cost across the agent's turns. Tune the rate with `PRICE_PER_1K_TOKENS` in `.env`.
- **Bring your own endpoint** — register any HTTP agent (name, URL, response dot-path, request
  template, optional auth header). Custom agents get conversation + system-failure tests;
  agent-cooperative faults (tool_timeout/stale_doc) only apply to agents that honor a `faults`
  field the way the sample agents do.
