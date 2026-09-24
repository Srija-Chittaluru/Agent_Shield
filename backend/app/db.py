"""PostgreSQL helpers + schema. Deliberately tiny — no ORM, no migrations (hackathon MVP).

`init_schema()` is the single, idempotent schema-creation entry point; it runs on FastAPI
startup. Rows come back as plain dicts (psycopg `dict_row`), so callers use row["col"].
"""
from datetime import datetime, timezone

import psycopg
from psycopg.rows import dict_row

from app.config import DATABASE_URL


# Customer names for agents connected through "Connect Your AI Agent". There is no login
# yet, so each newly connected agent gets its own generated "New Customer N" placeholder,
# to be replaced by the real customer identity once auth exists.
NEW_CUSTOMER_PREFIX = "New Customer"


def get_conn() -> psycopg.Connection:
    """Return a connection whose rows are dicts (access by column name)."""
    return psycopg.connect(DATABASE_URL, row_factory=dict_row)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_schema() -> None:
    """Create the tables if they don't exist. Idempotent."""
    conn = get_conn()
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS agents (
            id                SERIAL PRIMARY KEY,
            name              TEXT NOT NULL,
            kind              TEXT NOT NULL,          -- sample | custom
            endpoint_url      TEXT,
            auth_header       TEXT,
            request_template  TEXT,                   -- JSON string w/ {message},{history},{faults}
            response_path     TEXT,                   -- dot-path to reply in the response JSON
            description       TEXT,
            created_at        TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS runs (
            id               SERIAL PRIMARY KEY,
            agent_id         INTEGER NOT NULL REFERENCES agents(id),
            status           TEXT NOT NULL,           -- queued | running | done | error
            reliability_score DOUBLE PRECISION,
            breakdown_json   TEXT,
            started_at       TEXT,
            finished_at      TEXT
        );

        CREATE TABLE IF NOT EXISTS scenarios (
            id                SERIAL PRIMARY KEY,
            run_id            INTEGER NOT NULL REFERENCES runs(id),
            title             TEXT,
            user_goal         TEXT,
            test_type         TEXT,   -- support | memory | injection | contradiction | hallucination
            assigned_fault    TEXT,   -- none | tool_timeout | stale_doc | injection
            expected_behavior TEXT,
            seed_turns_json   TEXT
        );

        CREATE TABLE IF NOT EXISTS conversations (
            id             SERIAL PRIMARY KEY,
            run_id         INTEGER NOT NULL REFERENCES runs(id),
            scenario_id    INTEGER NOT NULL REFERENCES scenarios(id),
            verdict        TEXT,           -- pass | fail | null(unjudged)
            severity       TEXT,           -- low | med | high
            recovered      INTEGER,        -- 0 | 1 | null
            scores_json    TEXT,
            explanation    TEXT,
            suggested_fix  TEXT,
            evidence       TEXT
        );

        CREATE TABLE IF NOT EXISTS messages (
            id               SERIAL PRIMARY KEY,
            conversation_id  INTEGER NOT NULL REFERENCES conversations(id),
            turn_index       INTEGER NOT NULL,
            role             TEXT NOT NULL,   -- tester | agent
            content          TEXT,
            trace_json       TEXT
        );

        -- Customer -> agent -> test cases. A customer_agents row IS the testing context:
        -- the same agent onboarded for two customers is two rows, tested separately.
        CREATE TABLE IF NOT EXISTS customers (
            id          SERIAL PRIMARY KEY,
            name        TEXT NOT NULL UNIQUE,
            created_at  TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS customer_agents (
            id           SERIAL PRIMARY KEY,
            customer_id  INTEGER NOT NULL REFERENCES customers(id) ON DELETE CASCADE,
            agent_id     INTEGER NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
            created_at   TEXT NOT NULL,
            UNIQUE (customer_id, agent_id)
        );

        -- The persistent test-case library for one customer-agent combination.
        -- Same columns as `scenarios` (which stays the per-run execution copy).
        CREATE TABLE IF NOT EXISTS test_cases (
            id                SERIAL PRIMARY KEY,
            customer_agent_id INTEGER NOT NULL REFERENCES customer_agents(id) ON DELETE CASCADE,
            title             TEXT,
            user_goal         TEXT,
            test_type         TEXT,   -- support | memory | injection | contradiction | hallucination
            assigned_fault    TEXT,   -- none | tool_timeout | stale_doc | injection | api_*
            expected_behavior TEXT,
            seed_turns_json   TEXT,
            source            TEXT,   -- ai | user
            created_at        TEXT NOT NULL
        );

        -- One "Run Selected Agents" click = one run_groups row holding N runs, one per
        -- selected agent. Each run keeps its own scenarios, conversations and score, so
        -- agents stay independently reportable and comparable.
        CREATE TABLE IF NOT EXISTS run_groups (
            id          SERIAL PRIMARY KEY,
            created_at  TEXT NOT NULL
        );

        ALTER TABLE runs ADD COLUMN IF NOT EXISTS group_id
            INTEGER REFERENCES run_groups(id);
        ALTER TABLE runs ADD COLUMN IF NOT EXISTS customer_agent_id
            INTEGER REFERENCES customer_agents(id);

        -- The agent's own docs, uploaded on the Connect step. Stored so every later
        -- run for this agent stays grounded without re-uploading the file.
        ALTER TABLE agents ADD COLUMN IF NOT EXISTS knowledge TEXT;
        ALTER TABLE agents ADD COLUMN IF NOT EXISTS knowledge_name TEXT;

        -- Which testing modality this agent is: chat (default, text/HTTP) or voice.
        -- The Temporal workflow reads this to pick play_scenario vs play_voice_scenario.
        ALTER TABLE agents ADD COLUMN IF NOT EXISTS modality TEXT NOT NULL DEFAULT 'chat';

        -- Which wire protocol a voice-modality agent speaks. Only 'http_json' (the
        -- existing TTS -> adapter.send() -> STT contract) is implemented today;
        -- 'websocket' and 'twilio' are recognized names app.core.voice_caller rejects
        -- with a controlled error until a later phase implements them. Meaningless for
        -- modality='chat'. Defaulted so every existing agent keeps working unchanged.
        ALTER TABLE agents ADD COLUMN IF NOT EXISTS voice_protocol TEXT NOT NULL DEFAULT 'http_json';

        -- Which HTTP verb the adapter uses to call endpoint_url. 'POST' (default) sends
        -- request_template as the JSON body, unchanged from before this column existed.
        -- 'GET' sends message/history/faults as query parameters instead (see
        -- query_param_map) for black-box agents whose API only accepts GET.
        ALTER TABLE agents ADD COLUMN IF NOT EXISTS http_method TEXT NOT NULL DEFAULT 'POST';

        -- Only meaningful when http_method='GET'. JSON object mapping the semantic
        -- fields the adapter knows about ("message", "history", "faults") to the actual
        -- query parameter names the target agent expects, e.g. '{"message":"q"}'. A
        -- field omitted from the map is simply not sent. NULL/empty falls back to
        -- app.core.adapter.DEFAULT_QUERY_PARAM_MAP.
        ALTER TABLE agents ADD COLUMN IF NOT EXISTS query_param_map TEXT;

        -- Idempotency, so a re-executed unit of work converges on the same rows instead
        -- of appending new ones. This is the precondition for turning on retries.
        --
        -- conversations.idem_key: a caller-supplied natural key for "this scenario, in
        -- this run". PostgreSQL treats NULLs as distinct in a unique index, so every
        -- pre-existing row — and every replay, which deliberately wants a NEW
        -- conversation — keeps a NULL key and is unaffected.
        ALTER TABLE conversations ADD COLUMN IF NOT EXISTS idem_key TEXT;
        CREATE UNIQUE INDEX IF NOT EXISTS conversations_idem_key_uq
            ON conversations (idem_key);

        -- One row per turn. Guards against a genuinely concurrent double-execution,
        -- which clear_messages() alone cannot prevent.
        CREATE UNIQUE INDEX IF NOT EXISTS messages_conv_turn_uq
            ON messages (conversation_id, turn_index);

        -- The AI Caller: when set (non-empty), app.core.runner.run_scenario() plays this
        -- scenario DYNAMICALLY — app.core.ai_caller generates one natural next utterance
        -- per turn from this CUSTOMER context/behavior text + the live transcript,
        -- instead of replaying seed_turns_json verbatim. This describes the SIMULATED
        -- USER calling the agent under test — never the voice agent's own behavior — see
        -- app.core.ai_caller's module docstring for the role separation this enforces.
        -- NULL/empty (every scenario that predates this column) keeps running the
        -- original scripted seed_turns path, unchanged.
        --
        -- Named customer_context (not "persona") deliberately: an earlier version of
        -- this column WAS called persona, which reads ambiguously as "persona of the
        -- voice agent" to anyone skimming the schema — renamed before any real run data
        -- depended on the old name, so this is a straight add+drop, not a migration.
        ALTER TABLE scenarios ADD COLUMN IF NOT EXISTS customer_context TEXT;
        ALTER TABLE scenarios DROP COLUMN IF EXISTS persona;
        ALTER TABLE test_cases ADD COLUMN IF NOT EXISTS customer_context TEXT;
        ALTER TABLE test_cases DROP COLUMN IF EXISTS persona;

        -- Dynamic-mode-only: caps how many caller<->agent exchanges the AI Caller gets
        -- before the scenario ends regardless of outcome. NULL falls back to
        -- app.core.runner.DEFAULT_MAX_TURNS. Meaningless (ignored) when customer_context
        -- is empty.
        ALTER TABLE scenarios ADD COLUMN IF NOT EXISTS max_turns INTEGER;
        ALTER TABLE test_cases ADD COLUMN IF NOT EXISTS max_turns INTEGER;

        -- Conversation recording feature removed — these two columns (path to a WAV
        -- file, and whether it held agent-only audio) existed solely to support it.
        -- Dropped rather than left behind, same pattern already used for `persona`.
        ALTER TABLE conversations DROP COLUMN IF EXISTS recording_path;
        ALTER TABLE conversations DROP COLUMN IF EXISTS recording_agent_only;

        -- Flow-aware / node-based voice testing — Phase 1 only (upload, parse, store,
        -- display; nothing here is wired to execution yet). One row per uploaded flow
        -- definition version for an agent: re-uploading a corrected flow adds a new row
        -- rather than overwriting, so earlier versions stay available. Nodes/edges are
        -- JSON-in-TEXT, matching this schema's existing style (seed_turns_json,
        -- breakdown_json, ...) rather than a normalized graph schema — a handful of
        -- nodes per flow doesn't warrant one.
        CREATE TABLE IF NOT EXISTS agent_flows (
            id             SERIAL PRIMARY KEY,
            agent_id       INTEGER NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
            name           TEXT NOT NULL,
            source_format  TEXT NOT NULL,   -- json | yaml
            raw_source     TEXT NOT NULL,
            nodes_json     TEXT NOT NULL,
            edges_json     TEXT NOT NULL,
            created_at     TEXT NOT NULL
        );

        -- How this row's nodes_json/edges_json were derived from raw_source: 'deterministic'
        -- (app.core.flow_parser's alias/wrapper scanner) or 'llm' (app.core.flow_llm_extractor,
        -- used only when the scanner found nothing confident). NULL for rows written before
        -- this column existed — treated as 'deterministic' by callers, since that was the
        -- only path back then. No separate "version" column: id + created_at (already
        -- ordered newest-first by list_agent_flows) are sufficient.
        ALTER TABLE agent_flows ADD COLUMN IF NOT EXISTS extraction_method TEXT;

        -- Phase 2: which flow/node a scenario or test case was authored from, and the
        -- rich per-turn script it was authored WITH (expected_agent_behavior +
        -- caller_line per turn — see app.core.node_script). NULL for every scenario
        -- that predates this feature or wasn't authored from a flow. A later phase
        -- flattens node_script_json into this row's existing seed_turns_json (the
        -- caller_line values, in order) and expected_behavior (the
        -- expected_agent_behavior values, numbered) so the EXISTING scripted runner
        -- and Judge play/grade it unchanged; node_script_json itself is kept only so
        -- the per-turn structure stays editable/redisplayable later.
        ALTER TABLE scenarios ADD COLUMN IF NOT EXISTS flow_id
            INTEGER REFERENCES agent_flows(id);
        ALTER TABLE scenarios ADD COLUMN IF NOT EXISTS node_id TEXT;
        ALTER TABLE scenarios ADD COLUMN IF NOT EXISTS node_script_json TEXT;
        ALTER TABLE test_cases ADD COLUMN IF NOT EXISTS flow_id
            INTEGER REFERENCES agent_flows(id);
        ALTER TABLE test_cases ADD COLUMN IF NOT EXISTS node_id TEXT;
        ALTER TABLE test_cases ADD COLUMN IF NOT EXISTS node_script_json TEXT;
        """
    )
    conn.commit()
    conn.close()
    print("[db] schema initialized")


# ---------------------------------------------------------------------------
# Small query helpers (no ORM). Rows come back as dicts.
# ---------------------------------------------------------------------------
def get_agent(agent_id: int) -> dict | None:
    conn = get_conn()
    row = conn.execute("SELECT * FROM agents WHERE id = %s", (agent_id,)).fetchone()
    conn.close()
    return row


def get_agent_by_kind(kind: str) -> dict | None:
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM agents WHERE kind = %s ORDER BY id LIMIT 1", (kind,)
    ).fetchone()
    conn.close()
    return row


def list_agents() -> list[dict]:
    conn = get_conn()
    rows = conn.execute("SELECT * FROM agents ORDER BY id").fetchall()
    conn.close()
    return rows


def insert_agent(
    name: str,
    kind: str,
    endpoint_url: str,
    response_path: str,
    request_template: str,
    auth_header: str | None = None,
    description: str | None = None,
    modality: str = "chat",
    voice_protocol: str = "http_json",
    http_method: str = "POST",
    query_param_map: str | None = None,
) -> int:
    conn = get_conn()
    cur = conn.execute(
        """INSERT INTO agents
           (name, kind, endpoint_url, auth_header, request_template,
            response_path, description, created_at, modality, voice_protocol,
            http_method, query_param_map)
           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id""",
        (name, kind, endpoint_url, auth_header, request_template,
         response_path, description, now_iso(), modality, voice_protocol,
         http_method, query_param_map),
    )
    agent_id = cur.fetchone()["id"]
    conn.commit()
    conn.close()
    return agent_id


def update_agent_connection(
    agent_id: int,
    auth_header: str | None,
    request_template: str,
    response_path: str,
    modality: str,
    voice_protocol: str,
    http_method: str,
    query_param_map: str | None,
) -> None:
    """Refresh a re-registered agent's connection details in place.

    Re-registering the same name+endpoint (POST /agents) reuses the existing row so
    its customer context and stored test cases stay attached — but that means a
    corrected auth_header/protocol/create-call body typed on a SECOND attempt was
    previously discarded silently (only description ever got updated), so a wrong
    API key or protocol typed on attempt 1 stuck around forever no matter how many
    times the form was resubmitted. This makes every reconnect attempt authoritative.
    """
    conn = get_conn()
    conn.execute(
        """UPDATE agents SET auth_header = %s, request_template = %s, response_path = %s,
           modality = %s, voice_protocol = %s, http_method = %s, query_param_map = %s
           WHERE id = %s""",
        (auth_header, request_template, response_path, modality, voice_protocol,
         http_method, query_param_map, agent_id),
    )
    conn.commit()
    conn.close()


def update_agent_description(agent_id: int, description: str) -> None:
    conn = get_conn()
    conn.execute("UPDATE agents SET description = %s WHERE id = %s", (description, agent_id))
    conn.commit()
    conn.close()


def set_agent_knowledge(agent_id: int, knowledge: str, knowledge_name: str | None = None) -> None:
    """Attach the uploaded docs to the agent so later runs can ground themselves on them."""
    conn = get_conn()
    conn.execute(
        "UPDATE agents SET knowledge = %s, knowledge_name = %s WHERE id = %s",
        (knowledge, knowledge_name, agent_id),
    )
    conn.commit()
    conn.close()


import json as _json


def insert_run_group() -> int:
    """Create the batch that a set of same-click runs belong to."""
    conn = get_conn()
    cur = conn.execute(
        "INSERT INTO run_groups (created_at) VALUES (%s) RETURNING id", (now_iso(),)
    )
    gid = cur.fetchone()["id"]
    conn.commit()
    conn.close()
    return gid


def insert_run(
    agent_id: int,
    group_id: int | None = None,
    customer_agent_id: int | None = None,
) -> int:
    """One run against one agent. `group_id` ties it to the batch it was launched with."""
    conn = get_conn()
    cur = conn.execute(
        """INSERT INTO runs (agent_id, status, started_at, group_id, customer_agent_id)
           VALUES (%s, 'running', %s, %s, %s) RETURNING id""",
        (agent_id, now_iso(), group_id, customer_agent_id),
    )
    run_id = cur.fetchone()["id"]
    conn.commit()
    conn.close()
    return run_id


def get_runs_for_group(group_id: int) -> list[dict]:
    """Every run in a batch, oldest first, with the agent and customer names joined in.

    Ordered by id so the response lines up with the order the client sent its targets.
    """
    conn = get_conn()
    rows = conn.execute(
        """SELECT r.*, a.name AS agent_name, c.name AS customer_name
           FROM runs r
           JOIN agents a               ON a.id = r.agent_id
           LEFT JOIN customer_agents ca ON ca.id = r.customer_agent_id
           LEFT JOIN customers c        ON c.id = ca.customer_id
           WHERE r.group_id = %s
           ORDER BY r.id""",
        (group_id,),
    ).fetchall()
    conn.close()
    return rows


def run_group_exists(group_id: int) -> bool:
    conn = get_conn()
    row = conn.execute("SELECT id FROM run_groups WHERE id = %s", (group_id,)).fetchone()
    conn.close()
    return row is not None


def update_run(
    run_id: int,
    status: str | None = None,
    reliability_score: float | None = None,
    breakdown_json: str | None = None,
    finished: bool = False,
) -> None:
    sets, vals = [], []
    if status is not None:
        sets.append("status = %s"); vals.append(status)
    if reliability_score is not None:
        sets.append("reliability_score = %s"); vals.append(reliability_score)
    if breakdown_json is not None:
        sets.append("breakdown_json = %s"); vals.append(breakdown_json)
    if finished:
        sets.append("finished_at = %s"); vals.append(now_iso())
    if not sets:
        return
    vals.append(run_id)
    conn = get_conn()
    conn.execute(f"UPDATE runs SET {', '.join(sets)} WHERE id = %s", vals)
    conn.commit()
    conn.close()


def get_run(run_id: int) -> dict | None:
    conn = get_conn()
    row = conn.execute("SELECT * FROM runs WHERE id = %s", (run_id,)).fetchone()
    conn.close()
    return row


def insert_scenario(run_id: int, s: dict) -> int:
    conn = get_conn()
    cur = conn.execute(
        """INSERT INTO scenarios
           (run_id, title, user_goal, test_type, assigned_fault,
            expected_behavior, seed_turns_json)
           VALUES (%s,%s,%s,%s,%s,%s,%s) RETURNING id""",
        (run_id, s.get("title"), s.get("user_goal"), s.get("test_type"),
         s.get("assigned_fault"), s.get("expected_behavior"),
         _json.dumps(s.get("seed_turns", []))),
    )
    sid = cur.fetchone()["id"]
    conn.commit()
    conn.close()
    return sid


def replace_scenarios(run_id: int, scenarios: list[dict]) -> list[int]:
    """Make `scenarios` the run's scenario set. Returns the new ids, in input order.

    Safe to call twice: the run's existing scenarios are removed first, so a re-executed
    persist step converges instead of duplicating the whole suite. Scenario titles are
    LLM-generated and do collide within a single run, so there is no usable natural key
    here — replacing the set is what makes this idempotent.

    Must run BEFORE any conversation exists for the run (it does: scenarios are persisted
    in stage 3, conversations are created in stage 4). If that order is ever broken the
    delete hits the conversations foreign key and fails loudly — far better than silently
    duplicating a suite.
    """
    conn = get_conn()
    conn.execute("DELETE FROM scenarios WHERE run_id = %s", (run_id,))
    ids: list[int] = []
    for s in scenarios:
        cur = conn.execute(
            """INSERT INTO scenarios
               (run_id, title, user_goal, test_type, assigned_fault,
                expected_behavior, seed_turns_json, customer_context, max_turns,
                flow_id, node_id, node_script_json)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id""",
            (run_id, s.get("title"), s.get("user_goal"), s.get("test_type"),
             s.get("assigned_fault"), s.get("expected_behavior"),
             _json.dumps(s.get("seed_turns", [])), s.get("customer_context"), s.get("max_turns"),
             # Phase 3 (flow-node tests only): which flow/node this scenario was authored
             # from, and its rich per-turn script. None/NULL for every ordinary scenario,
             # which simply doesn't carry these keys — no behavior change for them.
             s.get("flow_id"), s.get("node_id"), s.get("node_script_json")),
        )
        ids.append(cur.fetchone()["id"])
    conn.commit()
    conn.close()
    return ids


def insert_conversation(run_id: int, scenario_id: int) -> int:
    conn = get_conn()
    cur = conn.execute(
        "INSERT INTO conversations (run_id, scenario_id) VALUES (%s, %s) RETURNING id",
        (run_id, scenario_id),
    )
    cid = cur.fetchone()["id"]
    conn.commit()
    conn.close()
    return cid


def get_or_create_conversation(
    run_id: int, scenario_id: int, idem_key: str | None = None
) -> int:
    """Conversation id for this scenario, creating it only the first time.

    The same `idem_key` always resolves to the same row, so a re-executed scenario
    reuses its conversation instead of leaving an orphan behind. Pass `idem_key=None`
    (the default) when a NEW conversation is genuinely wanted — that is what replay
    does — in which case this is just `insert_conversation`.
    """
    if idem_key is None:
        return insert_conversation(run_id, scenario_id)

    conn = get_conn()
    cur = conn.execute(
        """INSERT INTO conversations (run_id, scenario_id, idem_key)
           VALUES (%s, %s, %s)
           ON CONFLICT (idem_key) DO UPDATE SET run_id = EXCLUDED.run_id
           RETURNING id""",
        (run_id, scenario_id, idem_key),
    )
    cid = cur.fetchone()["id"]
    conn.commit()
    conn.close()
    return cid


def clear_messages(conversation_id: int) -> int:
    """Drop a conversation's transcript so a re-executed scenario writes a clean one.

    A retry restarts turn numbering at 0, but the transcript can legitimately diverge
    mid-way (the adaptive follow-up is LLM-generated), so overwriting turn by turn could
    strand tail turns from a longer earlier attempt. Clearing first avoids that.
    """
    conn = get_conn()
    n = conn.execute(
        "DELETE FROM messages WHERE conversation_id = %s", (conversation_id,)
    ).rowcount
    conn.commit()
    conn.close()
    return n


def insert_message(
    conversation_id: int, turn_index: int, role: str, content: str, trace: dict | None
) -> int:
    conn = get_conn()
    cur = conn.execute(
        """INSERT INTO messages (conversation_id, turn_index, role, content, trace_json)
           VALUES (%s,%s,%s,%s,%s) RETURNING id""",
        (conversation_id, turn_index, role, content,
         _json.dumps(trace) if trace is not None else None),
    )
    mid = cur.fetchone()["id"]
    conn.commit()
    conn.close()
    return mid


def get_scenario(scenario_id: int) -> dict | None:
    conn = get_conn()
    row = conn.execute("SELECT * FROM scenarios WHERE id = %s", (scenario_id,)).fetchone()
    conn.close()
    return row


def get_conversation(conversation_id: int) -> dict | None:
    conn = get_conn()
    row = conn.execute("SELECT * FROM conversations WHERE id = %s", (conversation_id,)).fetchone()
    conn.close()
    return row


def get_conversations_for_run(run_id: int) -> list[dict]:
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM conversations WHERE run_id = %s ORDER BY id", (run_id,)
    ).fetchall()
    conn.close()
    return rows


def get_messages(conversation_id: int) -> list[dict]:
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM messages WHERE conversation_id = %s ORDER BY turn_index, id",
        (conversation_id,),
    ).fetchall()
    conn.close()
    return rows


def update_conversation_verdict(
    conversation_id: int,
    verdict: str,
    severity: str | None,
    recovered: bool | None,
    scores: dict | None,
    evidence: str | None,
) -> None:
    conn = get_conn()
    conn.execute(
        """UPDATE conversations
           SET verdict=%s, severity=%s, recovered=%s, scores_json=%s, evidence=%s
           WHERE id=%s""",
        (verdict, severity,
         (None if recovered is None else (1 if recovered else 0)),
         _json.dumps(scores) if scores is not None else None,
         evidence, conversation_id),
    )
    conn.commit()
    conn.close()


def update_conversation_fix(
    conversation_id: int, explanation: str, suggested_fix: str, evidence: str | None
) -> None:
    conn = get_conn()
    if evidence is not None:
        conn.execute(
            "UPDATE conversations SET explanation=%s, suggested_fix=%s, evidence=%s WHERE id=%s",
            (explanation, suggested_fix, evidence, conversation_id),
        )
    else:
        conn.execute(
            "UPDATE conversations SET explanation=%s, suggested_fix=%s WHERE id=%s",
            (explanation, suggested_fix, conversation_id),
        )
    conn.commit()
    conn.close()


def build_conversation_payload(conversation_id: int) -> dict | None:
    """Assemble one conversation for the report: scenario meta + scores + messages+trace."""
    conv = get_conversation(conversation_id)
    if conv is None:
        return None
    conv = dict(conv)
    scenario = get_scenario(conv["scenario_id"])
    scenario = dict(scenario) if scenario else {}

    messages = []
    for m in get_messages(conversation_id):
        m = dict(m)
        trace = None
        if m.get("trace_json"):
            try:
                trace = _json.loads(m["trace_json"])
            except (_json.JSONDecodeError, TypeError):
                trace = None
        messages.append({
            "turn_index": m["turn_index"], "role": m["role"],
            "content": m["content"], "trace": trace,
        })

    scores = None
    if conv.get("scores_json"):
        try:
            scores = _json.loads(conv["scores_json"])
        except (_json.JSONDecodeError, TypeError):
            scores = None

    return {
        "id": conv["id"],
        "scenario_title": scenario.get("title"),
        "user_goal": scenario.get("user_goal"),
        "test_type": scenario.get("test_type"),
        "assigned_fault": scenario.get("assigned_fault"),
        "expected_behavior": scenario.get("expected_behavior"),
        "verdict": conv.get("verdict"),
        "severity": conv.get("severity"),
        "recovered": (None if conv.get("recovered") is None else bool(conv["recovered"])),
        "scores": scores,
        "explanation": conv.get("explanation"),
        "suggested_fix": conv.get("suggested_fix"),
        "evidence": conv.get("evidence"),
        "messages": messages,
    }


def run_counts(run_id: int) -> dict:
    conn = get_conn()
    scen = conn.execute("SELECT COUNT(*) c FROM scenarios WHERE run_id=%s", (run_id,)).fetchone()["c"]
    convs = conn.execute("SELECT COUNT(*) c FROM conversations WHERE run_id=%s", (run_id,)).fetchone()["c"]
    msgs = conn.execute(
        "SELECT COUNT(*) c FROM messages m JOIN conversations c ON m.conversation_id=c.id WHERE c.run_id=%s",
        (run_id,),
    ).fetchone()["c"]
    judged = conn.execute(
        "SELECT COUNT(*) c FROM conversations WHERE run_id=%s AND verdict IS NOT NULL", (run_id,)
    ).fetchone()["c"]
    conn.close()
    return {"scenarios": scen, "conversations": convs, "messages": msgs, "judged": judged}


# ---------------------------------------------------------------------------
# Customer -> agent -> test cases.
# ---------------------------------------------------------------------------
def get_or_create_customer(name: str) -> int:
    """Customer id for `name`, inserting the row the first time. Idempotent."""
    conn = get_conn()
    cur = conn.execute(
        """INSERT INTO customers (name, created_at) VALUES (%s, %s)
           ON CONFLICT (name) DO UPDATE SET name = EXCLUDED.name
           RETURNING id""",
        (name, now_iso()),
    )
    cid = cur.fetchone()["id"]
    conn.commit()
    conn.close()
    return cid


def get_agent_by_name_and_endpoint(name: str, endpoint_url: str) -> dict | None:
    """The agent row for this exact name+endpoint, if it was already registered.

    Reconnecting the same agent must reuse its row so its customer context and stored
    test cases stay attached instead of forking a duplicate.
    """
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM agents WHERE name = %s AND endpoint_url = %s ORDER BY id LIMIT 1",
        (name, endpoint_url),
    ).fetchone()
    conn.close()
    return row


def get_agent_by_name(name: str) -> dict | None:
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM agents WHERE name = %s ORDER BY id LIMIT 1", (name,)
    ).fetchone()
    conn.close()
    return row


def get_or_create_customer_agent(customer_id: int, agent_id: int) -> int:
    """Id of the customer-agent testing context, inserting it the first time.

    This row — not the agent — is what test cases hang off, so the same agent
    onboarded for two customers stays two separate contexts.
    """
    conn = get_conn()
    cur = conn.execute(
        """INSERT INTO customer_agents (customer_id, agent_id, created_at)
           VALUES (%s, %s, %s)
           ON CONFLICT (customer_id, agent_id) DO UPDATE SET customer_id = EXCLUDED.customer_id
           RETURNING id""",
        (customer_id, agent_id, now_iso()),
    )
    caid = cur.fetchone()["id"]
    conn.commit()
    conn.close()
    return caid


def list_customer_agents() -> list[dict]:
    """Every customer-agent combination, with the customer, agent names, and modality joined in."""
    conn = get_conn()
    rows = conn.execute(
        """SELECT ca.id, ca.customer_id, ca.agent_id,
                  c.name AS customer_name, a.name AS agent_name, a.modality AS agent_modality
           FROM customer_agents ca
           JOIN customers c ON c.id = ca.customer_id
           JOIN agents a    ON a.id = ca.agent_id
           ORDER BY c.id, a.id"""
    ).fetchall()
    conn.close()
    return rows


def insert_test_case(
    customer_agent_id: int,
    tc: dict,
    source: str = "ai",
    flow_id: int | None = None,
    node_id: str | None = None,
    node_script_json: str | None = None,
) -> int:
    """Store one generated/edited test case against a customer-agent combination.

    A single INSERT — it never touches any other test case already saved for this
    combination (contrast replace_test_cases, which replaces the whole suite).
    `flow_id`/`node_id`/`node_script_json` are set only for a flow-node test (Phase 3);
    every other caller omits them and the row keeps them NULL, unchanged from before
    these columns existed.
    """
    conn = get_conn()
    cur = conn.execute(
        """INSERT INTO test_cases
           (customer_agent_id, title, user_goal, test_type, assigned_fault,
            expected_behavior, seed_turns_json, source, created_at,
            flow_id, node_id, node_script_json)
           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id""",
        (customer_agent_id, tc.get("title"), tc.get("user_goal"), tc.get("test_type"),
         tc.get("assigned_fault"), tc.get("expected_behavior"),
         _json.dumps(tc.get("seed_turns", [])), source, now_iso(),
         flow_id, node_id, node_script_json),
    )
    tid = cur.fetchone()["id"]
    conn.commit()
    conn.close()
    return tid


def update_test_case(test_case_id: int, tc: dict, node_script_json: str | None = None) -> None:
    """Update an existing test_cases row in place, keeping its id.

    Used for re-saving a flow-node test after an edit/regenerate: reuses the row a
    prior save already created instead of inserting a duplicate (contrast
    insert_test_case, which always creates a new row).
    """
    conn = get_conn()
    conn.execute(
        """UPDATE test_cases
           SET title=%s, user_goal=%s, test_type=%s, assigned_fault=%s,
               expected_behavior=%s, seed_turns_json=%s, node_script_json=%s
           WHERE id=%s""",
        (tc.get("title"), tc.get("user_goal"), tc.get("test_type"), tc.get("assigned_fault"),
         tc.get("expected_behavior"), _json.dumps(tc.get("seed_turns", [])), node_script_json,
         test_case_id),
    )
    conn.commit()
    conn.close()


def get_test_case(test_case_id: int) -> dict | None:
    conn = get_conn()
    row = conn.execute("SELECT * FROM test_cases WHERE id = %s", (test_case_id,)).fetchone()
    conn.close()
    return row


def get_customer_agent(customer_agent_id: int) -> dict | None:
    """One customer-agent combination with the customer, agent names, and modality joined in."""
    conn = get_conn()
    row = conn.execute(
        """SELECT ca.id, ca.customer_id, ca.agent_id,
                  c.name AS customer_name, a.name AS agent_name, a.modality AS agent_modality
           FROM customer_agents ca
           JOIN customers c ON c.id = ca.customer_id
           JOIN agents a    ON a.id = ca.agent_id
           WHERE ca.id = %s""",
        (customer_agent_id,),
    ).fetchone()
    conn.close()
    return row


def get_customer_agent_by_agent_id(agent_id: int) -> dict | None:
    """The first customer-agent context for this agent, if one already exists.

    Same joined shape as get_customer_agent, filtered by agent instead of by combo id.
    Used when saving a flow-node test (Phase 3) so it attaches to a genuine existing
    customer context when there is one, rather than unconditionally minting a new
    "New Customer N" the way default_customer_agent does for an agent with none yet.
    """
    conn = get_conn()
    row = conn.execute(
        """SELECT ca.id, ca.customer_id, ca.agent_id,
                  c.name AS customer_name, a.name AS agent_name, a.modality AS agent_modality
           FROM customer_agents ca
           JOIN customers c ON c.id = ca.customer_id
           JOIN agents a    ON a.id = ca.agent_id
           WHERE ca.agent_id = %s
           ORDER BY ca.id LIMIT 1""",
        (agent_id,),
    ).fetchone()
    conn.close()
    return row


def default_customer_agent(agent_id: int) -> int:
    """The customer-agent context for an agent connected through "Connect Your AI Agent".

    Test cases hang off a customer_agents row, and these agents have no customer yet, so
    each one gets its own generated "New Customer N". Creating this row is what makes the
    agent show up in Existing Agent Testing — which is why it happens on save, not on
    connect. Reused on every later save, so revisiting the flow adds no duplicates.
    """
    conn = get_conn()
    like = f"{NEW_CUSTOMER_PREFIX} %"
    existing = conn.execute(
        """SELECT ca.id FROM customer_agents ca
           JOIN customers c ON c.id = ca.customer_id
           WHERE ca.agent_id = %s AND c.name LIKE %s
           ORDER BY ca.id LIMIT 1""",
        (agent_id, like),
    ).fetchone()
    if existing:
        conn.close()
        return existing["id"]
    # First save for this agent — take the next free number.
    taken = conn.execute("SELECT name FROM customers WHERE name LIKE %s", (like,)).fetchall()
    conn.close()
    used = [
        int(tail) for r in taken
        if (tail := r["name"][len(NEW_CUSTOMER_PREFIX):].strip()).isdigit()
    ]
    name = f"{NEW_CUSTOMER_PREFIX} {max(used, default=0) + 1}"
    return get_or_create_customer_agent(get_or_create_customer(name), agent_id)


def replace_test_cases(customer_agent_id: int, cases: list[dict], sources: list[str] | None = None) -> int:
    """Make `cases` the stored suite for THIS combination only. Returns how many were saved.

    Scoped by customer_agent_id, so saving Customer 1 / NorthBank never touches
    Customer 2 / NorthBank. Replaces in one transaction.
    """
    conn = get_conn()
    conn.execute("DELETE FROM test_cases WHERE customer_agent_id = %s", (customer_agent_id,))
    ts = now_iso()
    for i, tc in enumerate(cases):
        source = (sources[i] if sources and i < len(sources) else None) or "ai"
        conn.execute(
            """INSERT INTO test_cases
               (customer_agent_id, title, user_goal, test_type, assigned_fault,
                expected_behavior, seed_turns_json, source, created_at, customer_context, max_turns)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            (customer_agent_id, tc.get("title"), tc.get("user_goal"), tc.get("test_type"),
             tc.get("assigned_fault"), tc.get("expected_behavior"),
             _json.dumps(tc.get("seed_turns", [])), source, ts,
             tc.get("customer_context"), tc.get("max_turns")),
        )
    conn.commit()
    conn.close()
    return len(cases)


def get_test_cases(customer_agent_id: int) -> list[dict]:
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM test_cases WHERE customer_agent_id = %s ORDER BY id",
        (customer_agent_id,),
    ).fetchall()
    conn.close()
    return rows


# ---------------------------------------------------------------------------
# Flow-aware / node-based voice testing — Phase 1 (upload/parse/store/display only).
# ---------------------------------------------------------------------------
def insert_agent_flow(
    agent_id: int,
    name: str,
    source_format: str,
    raw_source: str,
    nodes: list[dict],
    edges: list[dict],
    extraction_method: str = "deterministic",
) -> int:
    """Store one uploaded+parsed flow definition as a new version for this agent."""
    conn = get_conn()
    cur = conn.execute(
        """INSERT INTO agent_flows
           (agent_id, name, source_format, raw_source, nodes_json, edges_json,
            created_at, extraction_method)
           VALUES (%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id""",
        (agent_id, name, source_format, raw_source,
         _json.dumps(nodes), _json.dumps(edges), now_iso(), extraction_method),
    )
    flow_id = cur.fetchone()["id"]
    conn.commit()
    conn.close()
    return flow_id


def list_agent_flows(agent_id: int) -> list[dict]:
    """Every uploaded flow version for this agent, newest first. No nodes/edges payload
    — callers wanting those fetch the single flow via get_agent_flow.
    """
    conn = get_conn()
    rows = conn.execute(
        """SELECT id, agent_id, name, source_format, created_at, extraction_method
           FROM agent_flows WHERE agent_id = %s ORDER BY id DESC""",
        (agent_id,),
    ).fetchall()
    conn.close()
    return rows


def get_agent_flow(flow_id: int) -> dict | None:
    conn = get_conn()
    row = conn.execute("SELECT * FROM agent_flows WHERE id = %s", (flow_id,)).fetchone()
    conn.close()
    return row


if __name__ == "__main__":
    init_schema()
