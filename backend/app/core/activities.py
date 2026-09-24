"""Activity boundary — every unit of external I/O that a run performs, named in one place.

Split out from the orchestrator so that workflow-level logic (sequencing, fan-out,
failure policy) holds no I/O of its own. Each function below is exactly one unit of work
against the outside world: PostgreSQL, the LLM, or the agent under test.

Every activity here is:
  * addressed by plain serializable arguments — ints, dicts, lists — and never by a live
    database row, because a row cannot cross a process boundary;
  * idempotent, so re-executing it converges instead of duplicating (see the idem_key
    work in app.db);
  * a single unit of work, so it can later carry its own timeout and retry policy.

Each one is registered as a Temporal Activity via @activity.defn, and the sync/async
split below is a deliberate contract, not an accident:

  * SYNC ACTIVITY (plain ``def``) — the body blocks on psycopg. Temporal runs these on the
    worker's ThreadPoolExecutor, so a slow query never stalls the worker's event loop.
    Writing them as ``async def`` would be a lie that blocks the loop on every query.
  * ASYNC ACTIVITY (``async def``) — genuinely awaitable I/O: the LLM, or the agent under
    test over httpx. These run on the event loop, where they belong.

Timeouts and retry policies are NOT set here. They belong to the caller, and live in
app.temporal.policies so they can be reviewed in one place.

Nothing here decides *when* it runs — that is the orchestrator's job. Every function is
also callable directly, which is what the current asyncio path still does; @activity.defn
only attaches metadata.
"""
import json as _json

from temporalio import activity

from app.config import MAX_SCENARIOS
from app.core.adapter import SYSTEM_FAULTS
from app.core.fixer import explain_and_fix
from app.core.judge import _endpoint_never_responded, judge_conversation
from app.core.runner import run_scenario
from app.core.scenarios import generate_scenarios
from app.core.scoring import compute
from app.core.voice_caller import (
    call_voice_agent,
    close_voice_session,
    peek_opening_greeting,
    voice_session_ended,
)
from app.db import (
    get_agent,
    get_conversation,
    get_conversations_for_run,
    get_messages,
    get_run,
    get_scenario,
    replace_scenarios,
    update_run,
)


# ---------------------------------------------------------------------------
# Agent + suite
# ---------------------------------------------------------------------------
@activity.defn
def load_agent(agent_id: int) -> dict | None:
    """SYNC ACTIVITY. The agent row as a plain dict, or None if it no longer exists."""
    agent = get_agent(agent_id)
    return dict(agent) if agent is not None else None


@activity.defn
async def prepare_scenarios(
    agent: dict, selected_types: list[str], guidance: str = "", knowledge: str = ""
) -> list[dict]:
    """LLM (+ HTTP when auto-discovering). Build the scenario bank for one agent.

    Pure generation — nothing is persisted and nothing is executed. Used both by the
    review endpoint (POST /runs/scenarios) and by a run that was given no reviewed suite.
    """
    # Resolve the agent's domain profile. Priority: uploaded docs > description >
    # auto-discovery (probe the agent and infer what it does).
    # Docs uploaded on the Connect step are stored on the agent, so a caller that has no
    # docs in hand (e.g. the Existing Agent path) still generates grounded test cases.
    knowledge = knowledge or (agent.get("knowledge") or "")
    description = agent.get("description") or ""
    if not (knowledge and knowledge.strip()):
        # Imported here rather than at module scope: discovery is the uncommon path and
        # pulls in the HTTP adapter.
        from app.core.discover import discover_agent, looks_generic

        if looks_generic(description):
            print(f"[activities] no docs/description — auto-discovering agent {agent.get('id')}")
            description = await discover_agent(agent)

    # Generate scenarios (falls back to a hardcoded bank on LLM failure).
    scenarios = await generate_scenarios(
        description or "A customer-support agent.",
        selected_types,
        guidance,
        knowledge,
    )
    # Demo safety: cap the number of scenarios so a live run stays fast.
    return scenarios[:MAX_SCENARIOS]


@activity.defn
def persist_suite(run_id: int, scenarios: list[dict]) -> list[int]:
    """SYNC ACTIVITY. Make `scenarios` the run's scenario set; returns ids in input order.

    Replaces rather than appends, so re-executing cannot duplicate the suite.
    """
    return replace_scenarios(run_id, scenarios)


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------
@activity.defn
async def play_scenario(run_id: int, scenario: dict, agent: dict) -> int:
    """HTTP + LLM + DB. Play one scenario against the agent; returns its conversation id.

    Keyed on (run, scenario), so re-executing reuses the same conversation and rewrites
    its transcript instead of leaving a half-finished orphan behind — an orphan would be
    scored as a free pass.
    """
    return await run_scenario(
        run_id, scenario, agent,
        idem_key=f"run:{run_id}:scenario:{scenario['_id']}",
    )


@activity.defn
async def play_voice_scenario(run_id: int, scenario: dict, agent: dict) -> int:
    """HTTP + TTS/STT + LLM + DB. Play one scenario against a voice-modality agent;
    returns its conversation id.

    Phase 2B: the scenario's turns are bridged onto a real voice-contract endpoint via
    app.core.voice_caller.call_voice_agent — TTS the tester's text, POST the resulting
    audio through the SAME black-box HTTP adapter chat uses, STT the agent's spoken
    reply back to text. run_scenario() itself is unchanged and fully reused: only which
    function plays a turn differs (`send_fn`), so persistence, idempotency, and turn
    generation (scripted or dynamic) all behave identically to the chat path. The Judge,
    Scoring, and Fixer only ever see the resulting text — they remain completely
    unaware anything was ever audio.

    Phase 3B: `close_fn=close_voice_session` is the GUARANTEED cleanup hook a
    persistent Twilio/native_ws call/session needs (http_json/websocket ignore it — see
    that function's docstring). Passed unconditionally, for every voice run, because it
    is a no-op for every protocol that doesn't need it.

    `greeting_fn=peek_opening_greeting` gives a DYNAMIC scenario (see app.core.ai_caller)
    a chance to read native_ws's unprompted opening greeting before generating the AI
    Caller's first line — also unconditional and a no-op for every other protocol, and
    entirely unused by a scripted scenario regardless of protocol.
    """
    return await run_scenario(
        run_id, scenario, _with_call_variables(agent, scenario),
        idem_key=f"run:{run_id}:scenario:{scenario['_id']}:voice",
        send_fn=call_voice_agent,
        close_fn=close_voice_session,
        greeting_fn=peek_opening_greeting,
        # Lets a flow scenario stop once the call is over; ignored by every other kind.
        session_ended_fn=voice_session_ended,
    )


def _with_call_variables(agent: dict, scenario: dict) -> dict:
    """The agent as its transport should see it for this scenario: with the scenario's
    call-creation test data (a saved script's setup.call) attached as
    agent["call_variables"], which app.core.voice_native_ws fills into the call it
    creates. A scenario without any leaves the agent exactly as loaded."""
    values = scenario.get("call_variables")
    return {**agent, "call_variables": values} if values else agent


@activity.defn
async def replay_scenario(run_id: int, scenario: dict, agent: dict) -> int:
    """HTTP + LLM + DB. Play a scenario as a deliberately NEW conversation.

    Unkeyed on purpose: replay exists to produce a second, independent conversation for
    a scenario that already has one, so the caller can compare before and after a fix.

    Modality-aware for the same reason play_voice_scenario is: a voice agent's replay
    must go through app.core.voice_caller (TTS/STT), or it would send raw text into a
    `message` field that endpoint expects as base64 audio. `close_fn`/`greeting_fn`
    follow the same modality-aware pattern as `send_fn` — None for chat, so both are
    skipped entirely.
    """
    is_voice = agent.get("modality") == "voice"
    send_fn = call_voice_agent if is_voice else None
    close_fn = close_voice_session if is_voice else None
    greeting_fn = peek_opening_greeting if is_voice else None
    session_ended_fn = voice_session_ended if is_voice else None
    if is_voice:
        agent = _with_call_variables(agent, scenario)
    return await run_scenario(
        run_id, scenario, agent, idem_key=None,
        send_fn=send_fn, close_fn=close_fn, greeting_fn=greeting_fn,
        session_ended_fn=session_ended_fn,
    )


@activity.defn
async def judge_conversation_by_id(conversation_id: int) -> dict:
    """DB + LLM. Score one conversation and persist its verdict.

    Takes an id rather than a row: a psycopg row is not serializable across a process
    boundary, so the row is loaded inside the activity.
    """
    conv = get_conversation(conversation_id)
    if conv is None:
        return {}
    return await judge_conversation(conv)


@activity.defn
async def explain_conversation_by_id(conversation_id: int) -> dict:
    """DB + LLM. Explain one failure and suggest a fix. Id in, for the same reason."""
    conv = get_conversation(conversation_id)
    if conv is None:
        return {}
    return await explain_and_fix(conv)


# ---------------------------------------------------------------------------
# Run bookkeeping
# ---------------------------------------------------------------------------
@activity.defn
def list_conversation_ids(run_id: int) -> list[int]:
    """SYNC ACTIVITY. Every conversation in the run, oldest first."""
    return [c["id"] for c in get_conversations_for_run(run_id)]


@activity.defn
def list_failed_conversation_ids(run_id: int) -> list[int]:
    """SYNC ACTIVITY. Only the conversations the judge failed — the explain+fix branch."""
    return [c["id"] for c in get_conversations_for_run(run_id) if c["verdict"] == "fail"]


def _agent_never_reachable(run_id: int) -> bool:
    """True when every scenario that was SUPPOSED to reach the agent failed to connect.

    Reuses the Judge's per-conversation check, so "the run never reached the agent" means
    exactly "every conversation never reached the agent" — one definition, not two.

    Scenarios carrying an injected system fault are excluded: AgentShield deliberately
    never calls the endpoint for those, so their empty transcripts say nothing about
    whether the agent was up. A suite made up entirely of them is a normal run.
    """
    genuine = []
    for c in get_conversations_for_run(run_id):
        scenario = get_scenario(dict(c)["scenario_id"])
        fault = (dict(scenario).get("assigned_fault") if scenario else None) or "none"
        if fault not in SYSTEM_FAULTS:
            genuine.append(dict(c))
    if not genuine:
        return False
    return all(
        _endpoint_never_responded([dict(m) for m in get_messages(c["id"])])
        for c in genuine
    )


@activity.defn
def finalize_run(run_id: int) -> dict:
    """SYNC ACTIVITY. Score the run, persist the breakdown, and mark it done.

    A run whose agent never once answered is marked "error" instead of "done". The score
    is still computed and stored (0.0 — every conversation is a system failure), but the
    status then says the endpoint was unreachable rather than that the agent scored badly.
    Partially reachable runs are untouched: one answered turn makes it a real result.
    """
    result = compute(run_id)
    status = "error" if _agent_never_reachable(run_id) else "done"
    update_run(run_id, status=status, finished=True)
    return {**result, "status": status}


@activity.defn
def fail_run(run_id: int) -> None:
    """SYNC ACTIVITY. Mark a run as errored so it never hangs in 'running'."""
    update_run(run_id, status="error", finished=True)


# ---------------------------------------------------------------------------
# Replay context
# ---------------------------------------------------------------------------
@activity.defn
def load_replay_context(conversation_id: int) -> dict | None:
    """SYNC ACTIVITY. Everything a replay needs, in one round trip.

    Returns ``{"run_id", "scenario", "agent"}`` — the scenario shaped exactly as the
    runner expects it — or None if any part of the chain is missing.
    """
    conv = get_conversation(conversation_id)
    if conv is None:
        return None
    conv = dict(conv)

    scenario_row = get_scenario(conv["scenario_id"])
    if scenario_row is None:
        return None
    s = dict(scenario_row)

    run = get_run(conv["run_id"])
    agent_row = get_agent(run["agent_id"]) if run else None
    if agent_row is None:
        return None

    return {
        "run_id": conv["run_id"],
        "scenario": {
            "_id": s["id"],
            "title": s["title"],
            "user_goal": s["user_goal"],
            "test_type": s["test_type"],
            "assigned_fault": s["assigned_fault"],
            "expected_behavior": s["expected_behavior"],
            "seed_turns": _json.loads(s["seed_turns_json"] or "[]"),
            **_replay_call_variables(s.get("node_script_json")),
        },
        "agent": dict(agent_row),
    }


def _replay_call_variables(node_script_json: str | None) -> dict:
    """A replayed interrupt scenario creates its call with the same test data: its
    saved setup.call, kept in the run's own scenarios.node_script_json."""
    try:
        script = _json.loads(node_script_json) if node_script_json else None
    except ValueError:
        return {}
    call = ((script or {}).get("setup") or {}).get("call") if isinstance(script, dict) else None
    return {"call_variables": call} if call else {}
