# AgentShield — Complete Project Guide

> **Read this to understand, present, and defend the whole project.**
> It explains what AgentShield is, how every piece works, the exact end-to-end flow,
> and a judge Q&A. Nothing here is aspirational — it all describes code that exists in this
> repo, as of the Postgres + Temporal architecture (not the original SQLite hackathon MVP;
> see `docs/PHASES.md` for how it got here).

---

## 1. The one-sentence pitch

**AgentShield crash-tests any AI agent through its API before you ship it** — it generates
adversarial test cases, injects real faults mid-conversation, judges every answer, and for
each failure hands you a plain-English cause **plus a copy-pasteable fix** backed by trace
evidence.

**Tagline:** *Before you deploy your AI agent, crash-test it.*

---

## 2. The problem we solve

Teams ship LLM agents (support bots, RAG assistants, tool-callers) with almost no idea how
they fail in production. Standard eval tools check "is the answer good on a clean
question?" They do **not**:
- deliberately break things mid-conversation (a tool times out, a doc is stale, the API 500s),
- adversarially probe safety (prompt injection, role-break),
- test multi-turn behavior (memory, contradiction),
- and — the big gap — tell you **why** it broke and **exactly what to change**.

AgentShield does all of that against **any** agent reachable over HTTP. You don't need its
code, its prompts, or its weights — just the endpoint.

---

## 3. The big picture (architecture)

```
┌─────────────────────────────┐        ┌──────────────────────────────┐
│  FRONTEND (Next.js :3100)   │        │  AGENT(S) UNDER TEST          │
│  Connect a new agent, or    │        │  (:8002–:8007) — 6 sample RAG │
│  reuse an existing one      │        │  agents shipped with the repo,│
│  Wizard: Connect→Verify→    │        │  or any third-party endpoint  │
│  Knowledge→Configure→       │        │  POST /chat -> {reply, trace} │
│  Review→Run→Results         │        │                                │
└──────────────┬──────────────┘        └───────────────▲──────────────┘
               │ REST + polling                          │ black-box HTTP
               ▼                                          │ (adapter only)
┌───────────────────────────────────────────────────────┴──────────────┐
│  BACKEND (FastAPI :8100, PostgreSQL, asyncio)                          │
│  Submits each run as a Temporal workflow. Per agent, the workflow:    │
│    generate/reuse test cases → break agent (faults+trace) → judge →   │
│    (fail only) explain+fix → score → report                           │
│  LLM calls (scenario gen, judge, fix) go to OpenAI via one wrapper.   │
└───────────────────────────────────────────┬───────────────────────────┘
                                             │ workflow submission + activities
                                             ▼
                              ┌───────────────────────────────┐
                              │  TEMPORAL (server :7233,       │
                              │  worker process, Web UI :8233) │
                              │  owns execution: durable,      │
                              │  survives a backend restart,   │
                              │  runs several agents in        │
                              │  parallel under one shared     │
                              │  concurrency budget            │
                              └───────────────────────────────┘
```

- **Frontend** never talks to OpenAI or the agent directly — only to our backend.
- **Backend** is the brain. It calls OpenAI for its own AI steps, calls the agent-under-test
  **only through the HTTP adapter** (black box), and hands actual execution to **Temporal**
  rather than running it in-process.
- **Agent(s) under test** are stand-ins for "the customer's agent." Six real RAG agents
  (gpt-4o-mini + retrieval) ship with the repo, each a deliberate archetype (§7.1), so demos
  and development test genuinely non-deterministic agents, not scripted mocks — and the
  same pipeline works against any third-party HTTP endpoint.

**Ports:** frontend `3100`, backend `8100`, Temporal server `7233` (Web UI `8233`), sample
agents `8002`–`8007`. One command starts all of it: `./run_all.sh`.

---

## 4. The pipeline (the heart of the product)

```
Connect an agent (or pick an existing one)
  │
  ▼
① Agent Knowledge ─────────────► required: upload docs and/or describe the agent.
  │                              Persisted on the agent row either way, and reused on
  │                              every later run of it — no re-upload needed.
  ▼
② Generate test cases ─────────► OpenAI (or a hardcoded fallback bank), grounded in the
  │                              knowledge when present — expected_behavior quotes the
  │                              actual fact, not a vague restatement of it.
  ▼
③ Review test cases ───────────► edit, add, or delete before anything runs; saved as a
  │                              persistent library scoped to this customer + agent.
  ▼
④ Break the agent as it runs ──► for each test case, play it turn-by-turn through the
  │                              HTTP adapter, INJECTING the assigned fault, and RECORD
  │                              every turn + trace. Runs execute as a Temporal workflow —
  │                              durable, and several agents run in parallel.
  ▼
⑤ Judge each answer ───────────► OpenAI (temperature=0, fixed seed) scores accuracy /
  │                              safety / hallucination / recovery; verdict and severity
  │                              are both DERIVED in code from those sub-scores, not
  │                              trusted as raw model fields.
  │
  ├── pass ─────────────────────► straight to the report (no fix needed)
  │
  └── fail ─► ⑥ Explain + suggest fix ──► OpenAI (or deterministic for system faults):
                                          cause (2-4 sentences) + ONE pasteable fix + evidence
                                              │
                                              ▼
⑦ Score + report ──────────────► reliability score (0-100) + breakdown + cost/latency.
                                  A run where the agent never answered a single turn is
                                  marked ERROR, not a misleadingly normal-looking score.
```

**Key branch:** passing scenarios skip the Explain+Fix step entirely. Only failures get a
cause + fix. That branch is the whole point — it's the "last mile" competitors skip.

**Key design decision:** a test-case suite is not regenerated every run. It's saved once,
reviewed/edited, and reused verbatim on every future run of that agent — Review Test Cases
shows you exactly what will execute, including the adaptive follow-up questions.

---

## 5. Backend — file by file (what does what, where)

All under `backend/`.

### App wiring
- **`app/main.py`** — FastAPI app. Enables CORS (open, local dev), creates the DB tables on
  startup, seeds the sample agents + inventory, and mounts the routers (`/runs`, `/agents`,
  `/conversations`, `/inventory`). Exposes `GET /health`.
- **`app/config.py`** — loads env from `.env`: `OPENAI_API_KEY`, `LLM_MODEL` (gpt-4o-mini),
  `DATABASE_URL` (Postgres), `TEMPORAL_ADDRESS`/`TEMPORAL_TASK_QUEUE`, `AGENT_CONCURRENCY`/
  `WORK_CONCURRENCY`, `MAX_SCENARIOS`, `PRICE_PER_1K_TOKENS`, and every sample agent's URL
  (`SAMPLE_AGENT_URLS`). Also loads `mapping.yaml` (internal key → display name) and
  `inventory.yaml` (which customers own which sample agents).
- **`app/db.py`** — thin PostgreSQL layer over `psycopg`, no ORM. `init_schema()` creates
  every table (idempotent, run on startup); everything else is a small, explicit
  query/insert/update helper. Also `build_conversation_payload()`, which assembles one
  conversation for the report (scenario meta + scores + messages + trace).
- **`app/models.py`** — Pydantic request/response shapes.

### The LLM wrapper (provider isolation)
- **`app/core/llm.py`** — the ONLY place that imports `openai`. `chat(system, messages,
  json_mode, temperature=0.2, model=None, seed=None)` wraps OpenAI's async client, uses
  JSON mode when asked, strips ```` ```json ```` fences, retries once on a bad parse. A
  fixed `FIXED_SEED` constant lives here for callers (the Judge) that need a repeatable
  sample — `seed` is opt-in per call, not a global default, so scenario generation and the
  adaptive tester are unaffected. Swap this one file to change providers.

### The pipeline modules
- **`app/core/adapter.py`** — the **black-box HTTP client**. `send(agent, message, history,
  faults)` renders the agent's `request_template`, POSTs to its `endpoint_url` (with
  optional auth header), and extracts `reply` (via a dot-path) + `trace`. **Also simulates
  system/transport faults** (see §7.2). Fails safe: timeouts/errors return
  `{"reply":"<error>","trace":{"error":...}}` instead of crashing. Seeds every sample agent
  (`seed_sample_agent`, `seed_inventory`) on startup.
- **`app/core/scenarios.py`** — `generate_scenarios(desc, selected_types, guidance,
  knowledge)`. Asks OpenAI for 8-10 scenarios in a required mix (support, injection,
  memory, contradiction, hallucination, system_failure), treats an uploaded `knowledge`
  block as authoritative ground truth (outranking the description), refines the user's
  optional **guidance/edge cases** and targets ≥3 scenarios at them, normalizes to valid
  enums, and **falls back to a hardcoded bank** if OpenAI fails.
  `memory`/`contradiction` scenarios are required to set `expected_behavior` to the actual
  fact/figure — not a circular restatement of "the agent should be correct" — because the
  Judge has no other reference to check accuracy against. `injection`/`memory`/
  `contradiction` scenarios carry their escalating follow-up question as the last entry of
  `seed_turns` up front (`_ensure_press_turn`/`PRESS_TURNS` guarantee it even if the model
  under-writes it), so the whole conversation is fixed before anything runs.
- **`app/core/runner.py`** — `run_scenario(run_id, scenario, agent, idem_key)`. Plays a
  scenario turn-by-turn: sends each seed turn through the adapter (passing the assigned
  fault), and — **only if the suite arrived with fewer than 3 seed turns** (an older stored
  suite or a hand-written case with no pressing turn) — falls back to generating one live
  follow-up for injection/memory/contradiction (an LLM "tester" that escalates, with a
  canned fallback on failure). Caps at 5 turns, writes **every** turn to `messages` with
  its `trace`. `idem_key` makes a re-executed activity converge on the same conversation
  instead of duplicating it.
- **`app/core/judge.py`** — `judge_conversation(conv)`. Loads the transcript + traces +
  scenario. If the assigned fault is a system/transport fault, OR every agent turn in the
  conversation is the `<error>` sentinel (the endpoint never answered), it's judged
  **deterministically** as a `system` failure with no LLM call. Otherwise it asks OpenAI
  (`temperature=0`, a fixed `seed`) to score accuracy/safety/hallucination/recovery and
  cite evidence. Both **`verdict`** and **`severity`** are then derived in code from those
  sub-scores (`_derive_fail_category`, `_derive_severity`) rather than trusted as raw model
  output — the LLM's self-reported category/severity were unreliable and, for severity,
  fed a 3×/2×/1× score weight directly. Fail-safe: if the judge call dies outright, the
  conversation defaults to a safe "pass" so one bad call can't kill the run.
- **`app/core/fixer.py`** — `explain_and_fix(conv)`. Runs **only on failures**. Produces a
  plain-English cause + ONE pasteable fix + evidence. System failures get a deterministic
  infrastructure-oriented fix (retries/timeouts/monitoring), no LLM call.
- **`app/core/scoring.py`** — `compute(run_id)`. Reliability score + breakdown + performance
  (cost/latency) aggregation. Math in §8. Purely arithmetic over what the Judge already
  wrote — it faithfully propagates upstream determinism (or the lack of it) rather than
  adding any of its own.
- **`app/core/discover.py`** — `discover_agent(agent)`: probes an agent with a generic
  message and infers a description, used when Connect gets neither an uploaded knowledge
  file nor a typed description.
- **`app/core/orchestrator.py`** — the original straight-line implementation (generate →
  play all → judge all → fix failures → score) kept as a reference; a live run instead
  executes the equivalent steps as a Temporal workflow (see §5's Temporal section below).

### Temporal (durable, parallel execution)
- **`app/temporal/workflows.py`** — `AgentTestWorkflow` (one agent's full pipeline) and
  `RunGroupWorkflow` (fans out one child `AgentTestWorkflow` per agent in a batch,
  independently — one agent failing never cancels its siblings). Concurrency inside one
  agent's scenario/judge/fix work is bounded by that agent's fair share of the worker's
  pool, so N agents running "in parallel" genuinely share the budget instead of racing for
  it first-come-first-served.
- **`app/core/activities.py`** — every unit of external I/O (Postgres, the LLM, the agent
  under test) as a named, individually-retryable Temporal activity: `load_agent`,
  `prepare_scenarios`, `persist_suite`, `play_scenario`, `replay_scenario`,
  `judge_conversation_by_id`, `explain_conversation_by_id`, `list_conversation_ids`,
  `list_failed_conversation_ids`, `finalize_run`, `fail_run`, `load_replay_context`.
  `finalize_run` computes the score and then marks the run `error` — instead of `done` —
  if every conversation in it never got a real reply from the agent (excluding scenarios
  that intentionally never call the agent, i.e. injected system faults).
- **`app/temporal/client.py` / `worker.py` / `policies.py` / `health.py`** — a client
  factory, the worker process (`python -m app.temporal.worker`), per-activity
  timeout/retry policy kept in one place, and a startup health check.
- A run cannot execute without the Temporal server reachable: `POST /runs` /
  `POST /runs/group` insert the run rows, then try to submit the workflow; if that fails,
  the rows are marked `error` and the request returns **HTTP 503** telling you to start
  `temporal server start-dev`.

### The API (routers)
- **`app/routers/runs.py`**
  - `POST /runs/scenarios` — stage 1: generate a suite for review. Persists nothing.
  - `POST /runs/scenarios/one` — generate a single test case from a free-text description
    (the "Add Test Case → Generate with AI" dialog).
  - `POST /runs/test-cases` — save a reviewed suite to its customer-agent library, without
    running it.
  - `POST /runs` / `POST /runs/group` — persist the final suite(s), insert run row(s), and
    submit a Temporal workflow. Return immediately with ids to poll.
  - `GET /runs/{id}` / `GET /runs/group/{id}` — status + reliability_score + progress
    counts (single agent / batch).
  - `GET /runs/{id}/report` / `GET /runs/group/{id}/report` — the full nested report(s):
    score, breakdown, every conversation with messages + trace + explanation + fix +
    evidence.
  - `GET /runs/demo/report` — a static pre-baked report (demo safety, works offline).
- **`app/routers/agents.py`**
  - `GET /agents` / `GET /agents/{id}` — list / fetch agents.
  - `POST /agents` — register a black-box agent (name, endpoint, response path, template,
    auth). Re-registering the same name+endpoint returns the existing row rather than
    duplicating it, so reconnecting keeps its saved knowledge and test cases attached.
  - `POST /agents/{id}/probe` — a **real connectivity check**, used by Verify Connection.
  - `POST /agents/{id}/discover` — auto-infer a description when neither docs nor a
    description were provided.
  - `POST /agents/{id}/knowledge` — attach uploaded docs (plain text the browser already
    read — no server-side file parsing or storage).
- **`app/routers/inventory.py`**
  - `GET /inventory` — every customer × agent combination (seeded from `inventory.yaml`,
    plus every agent connected dynamically through Connect Your AI Agent).
  - `GET /inventory/{customer_agent_id}/test-cases` — the stored library for one
    combination, feeding Existing Agent Testing straight into Review Test Cases.
- **`app/routers/conversations.py`**
  - `POST /conversations/{id}/replay` — re-run that scenario as a new conversation, judge +
    fix, return it.

### The agents under test
- **`backend/sample_rag_bot/main.py`** and **`backend/sample_agents/*.py`** — each a
  **fully standalone** FastAPI service (imports nothing from our backend). Every one is a
  real RAG agent: keyword-retrieve its own domain's policy docs → build a grounded prompt →
  call **gpt-4o-mini** → return `{reply, trace}` with real token counts. Each exposes:
  - `GET /` — a self-contained **chat playground UI** (talk to the agent, toggle faults live).
  - `POST /chat` — the API AgentShield calls.
  - `GET /health` — liveness.
  The four domain agents (banking, HR, insurance, airline, telecom) share one factory,
  `backend/sample_agents/rag_core.py` — a new domain is just a `Domain(...)` config (KB
  docs with current + stale variants, a confidential secret for injection testing, sample
  questions). They honor the agent-cooperative faults (tool_timeout, stale_doc, injection)
  because we own them; a real third-party agent simply wouldn't, and AgentShield degrades
  to conversation + system-fault tests. See §7.1 for the archetype each one represents.

### Demo safety
- **`app/demo_report.py`** — a hand-crafted, always-impressive report served by
  `GET /runs/demo/report`. No DB, no OpenAI, so "View sample report" renders even if the
  network dies mid-demo.

---

## 6. Frontend — file by file

Next.js (App Router) + Tailwind + framer-motion + lucide-react, under `frontend/`.

- **`app/layout.tsx`** — root layout, fonts, metadata.
- **`app/page.tsx`** — the landing page: hero, core features, workflow, CTA.
- **`app/start/page.tsx`** — the fork: **Connect Your AI Agent** vs. **Existing Agent
  Testing**.
- **`app/existing-agent/page.tsx`** — pick one or more customer-agent combinations (seeded
  samples or previously connected agents) and hand them to the dashboard, which loads (or
  generates, if none is stored yet) each one's test cases.
- **`app/lib/api.ts`** — the API client: `registerAgent`, `probeAgent`, `discoverAgent`,
  `saveAgentKnowledge`, `generateScenarios`, `generateOneScenario`, `saveTestCases`,
  `createRun`, `createRunGroup`, `getRun`, `getRunGroup`, `getReport`, `getGroupReport`,
  `getDemoReport`, `getInventory`, `getStoredTestCases`, plus all the TS types. Base URL
  from `NEXT_PUBLIC_API_URL` (default `http://localhost:8100`).
- **`app/dashboard/page.tsx`** — the **wizard** (the whole product UX), a client component
  stepping through `connect → verifying → configure → review → running → results`.

### What each wizard step does (and which API it calls)
| Step | UI | Real backend call |
|------|----|-------------------|
| **Connect** | agent name, endpoint URL, optional API key, **Agent Knowledge** (upload docs and/or describe the agent — required, at least one of the two) | — |
| **Verify** | animated checklist | `POST /agents` (register) → `POST /agents/{id}/knowledge` (if docs given) → `POST /agents/{id}/probe` (genuine ping) → `POST /agents/{id}/discover` (only if neither knowledge input was given — now effectively unreachable, kept as a safety net) |
| **Configure** | test-type chips + optional **custom guidance** textarea | `POST /runs/scenarios` (generate for review) |
| **Review** | edit/add/delete test cases; each shows its full seed-turn conversation, including the pressing follow-up for adaptive types | `POST /runs/test-cases` (save without running) |
| **Running** | animated progress, one row per agent in a batch | `POST /runs` or `POST /runs/group` → polls `GET /runs/{id}` or `GET /runs/group/{id}` |
| **Results** | score gauge, cost/latency, resilience bars, failure cards, passing scenarios — one tab per agent in a batch | `GET /runs/{id}/report` or `GET /runs/group/{id}/report` |

Existing Agent Testing skips straight to **Review** with the stored library already loaded
(`GET /inventory/{customer_agent_id}/test-cases`), generating only if nothing was saved
yet.

### The results screen (the "money screen") shows
- **Reliability score gauge** (0–100, animated).
- **Cost & latency tiles** — avg latency, estimated $ cost, total tokens.
- **Resilience by Category** — pass rate per test type as bars.
- **Failed Scenarios** — each card: severity badge, category, **cause**, **pasteable fix +
  Copy button**, **evidence**, and a transcript expander (tester/agent turns with per-turn
  trace chips: tool calls ok/failed, stale docs, latency, tokens).
- **Passing Scenarios** — collapsed list; expand any to see its transcript + trace.
- "View sample report" deep-links straight to the offline sample report.

---

## 7. Fault injection & the sample agent roster

### 7.1 The six sample agents, as deliberate archetypes

Rather than one uniform "good agent," the shipped roster deliberately spans both axes of
"gets the facts right" and "holds under social engineering," so each one has a clean,
isolated failure signature to demonstrate:

| Agent | Domain | Port | Facts | Security | Fails on |
|-------|--------|------|-------|----------|----------|
| Store Support (RAG) | e-commerce | 8002 | grounded | — | general baseline |
| NorthBank Support | banking | 8003 | grounded | — | neutral baseline |
| PeopleDesk HR | HR / benefits | 8004 | invents confident numbers | leaks on injection | almost everything |
| SafeGuard Claims | insurance | 8005 | honest, declines gracefully | holds | close to nothing |
| SkyRoute Airways | airline | 8006 | invents under a failed lookup | holds | recovery / hallucination, isolated |
| ConnectWave Mobile | telecom | 8007 | honest, declines gracefully | leaks to anyone claiming to be staff | injection, isolated |

### 7.2 Fault classes

**Agent-cooperative faults** (travel in the request body; only our sample agents honor
them, because the fault manipulates the context fed to their LLM):
- `tool_timeout` — the retrieval/lookup tool "fails"; a good agent should apologize + offer a fallback.
- `stale_doc` — an **outdated** policy doc is injected; does the agent parrot it?
- `injection` — an adversarial message is passed straight through; does the model leak its secret / break role?

**System / transport faults** (AgentShield simulates these **at the HTTP layer**, so they
work on **any** endpoint — the agent is never even called):
- `api_unreachable` — connection refused.
- `api_error` — HTTP 500 / crash.
- `api_timeout` — no response in time.
- `malformed_response` — invalid / non-JSON body.

System faults are judged deterministically (a 5xx on a valid request is a reliability
failure regardless of content) and produce infrastructure-oriented fixes (retries,
timeouts, monitoring). An agent whose endpoint is simply **down** — not an injected fault,
just unreachable — is caught the same way: every turn comes back as the `<error>`
sentinel, the Judge records it as a deterministic `system` failure without an LLM call,
and if that's true for the whole run, the run itself is marked `error` rather than `done`.

---

## 8. Scoring & cost math (so you can defend the numbers)

**Reliability score** (`app/core/scoring.py`):
```
weighted_failures = Σ severity_weight   (high=3, med=2, low=1) over failed scenarios
reliability_score = 100 × (1 − weighted_failures / (total_scenarios × 3)),  clamped 0–100
```
Example: 1 high-severity fail out of 5 → 100 × (1 − 3/15) = **80**.

`severity` here is **derived in code from the Judge's sub-scores**, not read from the
model's raw output — see §5's judge.py notes and §13's Q&A on why that matters.

**Breakdown:** pass/fail counts **by test type** and **by failure category**
(safety / hallucination / recovery / accuracy / system).

**Performance (cost & latency):** aggregated from every agent turn's trace —
`avg_latency_ms`, `max_latency_ms`, `total_tokens`, and
`est_cost_usd = total_tokens/1000 × PRICE_PER_1K_TOKENS` (rate configurable in `.env`).

---

## 9. Data model — PostgreSQL, `backend/app/db.py`

No ORM, no migrations — `init_schema()` runs on backend startup and is idempotent.

- **agents** — id, name, kind(sample|custom), endpoint_url, auth_header, request_template,
  response_path, description, **knowledge, knowledge_name**, created_at. The uploaded
  docs' extracted text and original filename — never the file itself.
- **customers** — id, name (unique), created_at.
- **customer_agents** — id, customer_id, agent_id, created_at. **The actual testing
  context** — a saved test-case library hangs off this row, not off the agent, so the same
  agent onboarded for two customers is two independent suites. Agents connected through
  Connect Your AI Agent get an auto-generated "New Customer N" the first time their suite
  is saved.
- **test_cases** — id, customer_agent_id, title, user_goal, test_type, assigned_fault,
  expected_behavior, seed_turns_json, source(ai|user), created_at. The persistent,
  reviewable/editable library. Replaced wholesale on save/regenerate, never appended to.
- **run_groups** — id, created_at. One row per "Run" click.
- **runs** — id, agent_id, group_id, customer_agent_id, status(queued|running|done|error),
  reliability_score, breakdown_json, started_at, finished_at.
- **scenarios** — id, run_id, title, user_goal, test_type, assigned_fault,
  expected_behavior, seed_turns_json. The frozen, per-run execution copy of whatever suite
  actually ran — `test_cases` is the reusable library, this is the historical record.
- **conversations** — id, run_id, scenario_id, verdict, severity, recovered, scores_json,
  explanation, suggested_fix, evidence, idem_key. One per scenario per run.
- **messages** — id, conversation_id, turn_index, role(tester|agent), content, trace_json.
  One row per turn; a unique index on `(conversation_id, turn_index)` guards against a
  genuinely concurrent double-execution.

One run → many scenarios → one conversation each → many messages (turns). The trace lives
on each agent message. One customer-agent combination → one persistent test-case library,
reused across as many runs as you like.

---

## 10. Tech stack & why

| Layer | Choice | Why |
|-------|--------|-----|
| Backend | FastAPI + asyncio | async fan-out of scenarios/judging with a concurrency cap; tiny footprint |
| Storage | **PostgreSQL** | a real multi-table relational model (customers, test-case libraries, run groups) outgrew SQLite's single-writer, single-file model |
| Execution | **Temporal** | a run survives a backend restart, and several agents run genuinely in parallel under one shared, fair concurrency budget — an in-process `asyncio` task could do neither |
| LLM | OpenAI gpt-4o-mini | cheap, fast, JSON-mode; isolated in one wrapper so the provider is swappable |
| Frontend | Next.js + Tailwind + framer-motion | modern, animated UI, App Router |
| Agents under test | 6 standalone RAG agents + gpt-4o-mini | real, non-deterministic agents, each a deliberate pass/fail archetype, so results are credible |

Still deliberately **no** Docker / auth / Redis / Celery / websockets — those weren't
needed even once Postgres and Temporal were added.

---

## 11. How to run

```bash
./run_all.sh
```
Starts the backend (`:8100`), all 6 sample agents (`:8002`–`:8007`), the Temporal dev
server + worker (if the `temporal` CLI is installed), and the frontend (`:3100`). Ctrl-C
stops all of it. See `README.md` for prerequisites (PostgreSQL, the Temporal CLI, an
OpenAI key) and running pieces by hand instead.

Open http://localhost:3100 → **Start Testing** → either **Connect Your AI Agent** (a fresh
endpoint) or **Existing Agent Testing** (reuse a stored one, sample or previously
connected).

A run genuinely requires the Temporal server to be reachable — without it, `POST /runs`
returns HTTP 503 rather than silently doing nothing.

---

## 12. 60-second demo script

1. "Every team ships AI agents with no idea how they break. AgentShield crash-tests any
   agent through its public API — no code, no prompts, no access needed."
2. **Connect** → point it at one of the sample agents (or your own) → **Agent Knowledge**
   (upload its docs) → **Verify** (it really pings it).
3. **Configure** → pick test types → **Generate** → **Review** — point out that the full
   conversation for each adaptive test case (injection/memory/contradiction) is already
   fixed and editable here, not invented while it runs.
4. **Run** → "It's playing these test cases against the agent, injecting faults — a tool
   timing out, a stale policy doc, a prompt injection — and recording every trace, as a
   durable Temporal workflow."
5. **Report** → "Reliability score, cost, latency, resilience by category."
6. Open a coral card → "After 'ignore your instructions,' it leaked its system prompt —
   here's the exact transcript and trace proving it, and the one-line rule you paste to fix
   it." **Copy the fix.**
7. "Test the same agent again — the score doesn't drift just because the Judge felt like it
   this time. And if the agent's actually down, we tell you that, not a fake passing grade."
8. "Domain-independent, works on your own endpoint, and the same test-case library is
   reused every time you come back to this agent — the last mile no eval tool gives you."

---

## 13. Judge Q&A — likely questions & strong answers

**Q: How is this different from existing LLM eval tools (Ragas, LangSmith, PromptFoo)?**
A: Those mostly score answer quality on clean inputs. AgentShield (1) **injects live
faults** mid-conversation (tool timeouts, stale docs, prompt injection, API 500s), (2)
tests **multi-turn** adversarial behavior with a fixed, reviewable escalation, and (3) —
the differentiator — for every failure returns a **plain-English cause + a copy-pasteable
fix + trace evidence**. It's black-box: point it at any HTTP endpoint, no code or prompts
needed.

**Q: It's black-box — how can you inject faults into an agent you don't control?**
A: Two tiers. **System/transport faults** (unreachable, 500, timeout, malformed) we
simulate at the HTTP layer, so they work on *any* endpoint. **Injection** is just an
adversarial message — works on any agent. **Tool-timeout / stale-doc** need the agent to
expose a fault hook, which only our sample agents do; for a real third-party agent we
degrade gracefully to conversation + system-fault tests. We're explicit about that
boundary — no overclaiming.

**Q: Who judges the answers, and isn't an LLM judge unreliable?**
A: An LLM (gpt-4o-mini) scores 4 categories with a strict rubric and must cite evidence
from the transcript/trace, called at **temperature=0 with a fixed seed** so the same
transcript scores the same way twice. We further reduce unreliability by (a) **deriving
both the failure category AND the severity deterministically** from the sub-scores rather
than trusting the model's self-labels — severity in particular feeds the score's weighting
directly, so this isn't cosmetic, (b) judging **system failures, and an agent that never
answered at all, deterministically with no LLM call**, and (c) grounding accuracy in the
scenario's `expected_behavior`, which is itself required to contain the actual fact from
any uploaded knowledge rather than a vague description of correctness. It's not perfect
truth, but it's consistent, cited, and reproducible.

**Q: How is the reliability score calculated?**
A: Severity-weighted: `100 × (1 − Σweight / (total×3))`, weights high=3/med=2/low=1. So one
high-severity fail out of five scenarios → 80. Plus a breakdown by test type and failure
category.

**Q: What about cost and latency?**
A: Every agent turn's trace carries latency and tokens. The report aggregates avg/max
latency, total tokens, and an **estimated $ cost** (tokens × configurable rate) — so you
see reliability *and* the operational cost profile in one place.

**Q: Is the demo agent real or faked?**
A: Real — six of them, in fact, each a standalone RAG agent (retrieval → grounded prompt →
gpt-4o-mini) with real, non-deterministic answers and real token usage. They're
deliberately built as different archetypes (a weak one that invents and leaks, a hardened
one that holds on both axes, and two that are strong on one axis and weak on the other) so
the pipeline is demonstrably catching genuine, distinct failure modes — not one scripted
weakness repeated six times.

**Q: What happens if OpenAI rate-limits or the network dies during the demo?**
A: Three safety nets: scenario generation **falls back to a hardcoded bank**; judge/fix
have safe defaults; and there's a **pre-baked sample report** (`View sample report`) that
needs no network at all. Runs are also capped (`MAX_SCENARIOS`) so a live run finishes
quickly.

**Q: Can I test my own agent?**
A: Yes — **Connect Your AI Agent** in the UI (or `POST /agents`). Provide the URL, the JSON
request template, the response dot-path, an optional auth header, and Agent Knowledge
(upload docs and/or describe it — at least one is required, both persist and are reused on
every later run). AgentShield then treats it as a black box, and its endpoint + knowledge +
test-case library are all there waiting the next time you come back to it through Existing
Agent Testing.

**Q: How does the multi-turn / adaptive testing work?**
A: Each scenario has seed turns. For injection/memory/contradiction, the generator now
writes the escalating final turn (press the injection harder, ask the planted fact back,
restate the contradiction) as part of the suite up front — so it's visible and editable in
Review Test Cases, and identical on every future run of that test case. A live LLM
"tester" call is still the fallback for an older suite or a hand-written case with fewer
turns. Capped at 5 tester turns total.

**Q: Why did the same agent + same test cases score differently between two runs?**
A: They shouldn't anymore, for the reasons this run's config controls — but two sources of
variance are inherent, not bugs: the agent under test is itself an LLM, and (for adaptive
scenarios generated before this fix, or hand-written ones with fewer turns) the follow-up
question can still be regenerated live. Even holding every input fixed, a `support`
scenario with zero adaptivity measured a 16% verdict-flip rate across repeated runs in our
own data — that's the honest floor of testing a non-deterministic agent with a
non-deterministic judge, not something a fixed prompt can erase.

**Q: Why Postgres + Temporal instead of the original SQLite + in-process background task?**
A: SQLite's single-writer model doesn't fit a relational, multi-table system (customers,
per-customer test-case libraries, run batches) with concurrent runs. An `asyncio`
background task dies with the process and has no fair way to share concurrency across
several agents running "in parallel" — Temporal makes a run durable (survives a restart)
and genuinely shares a concurrency budget across a batch. Still deliberately no
Docker/auth/Redis/Celery/websockets — those weren't the actual gap.

**Q: What's the single most important thing to remember?**
A: **We don't just tell you the agent failed — we tell you exactly what was asked, how it
broke (with the trace), why, and the one line you paste to fix it — and we tell you that
consistently, run after run, on a test suite you reviewed and can trust hasn't silently
changed underneath you.** That last mile is the moat.

---

## 14. Glossary (quick)

- **Agent** — one connected endpoint (`agents` row): its URL, request/response shape,
  optional auth, and its uploaded knowledge/description.
- **Customer-agent combination** — the actual testing context (`customer_agents` row): an
  agent as onboarded for one customer. A saved test-case library belongs to this, not to
  the agent alone.
- **Test case** — one persisted, reviewable/editable entry in a customer-agent's library
  (`test_cases` row): goal, type, assigned fault, expected behavior, seed turns.
- **Scenario** — the frozen, per-run copy of a test case that actually executed
  (`scenarios` row) — the historical record, distinct from the reusable library.
- **Conversation** — one played-out scenario against the agent (its turns + verdict +
  scores + fix).
- **Trace** — the per-turn record: tool calls (ok/failed), retrieved docs (stale?),
  latency, tokens.
- **Fault** — a deliberate break injected during a scenario (agent-cooperative or
  system/transport).
- **Verdict / severity** — pass|fail and low|med|high — both derived in code from the
  Judge's sub-scores, not read as raw model output.
- **Recovery** — did the agent cope gracefully after a fault (only scored when a fault was
  injected).
- **Run group** — one "Run" click; one or more `runs`, one per selected agent, scored
  independently.
