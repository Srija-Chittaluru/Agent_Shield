"""Running a saved interrupt script (Part 6) through the EXISTING runner.

The real Maya fixture is planned with the real planner, a planner-consistent script is
built (tests.routers.test_flow_interrupt_scripts._script_for) and validated with the
real validator, then converted with interrupt_script_to_scenario and played by the real
run_scenario()/_run_scripted(). Only the transport (a scripted fake, or the real
native_ws transport over a fake socket) and app.db are faked. The AI Caller may never be
used at run time.

_run_scripted now passes every seed_turns line through _reactive_scripted_line, which
calls chat() to react to the agent's actual last reply (see runner.py). The autouse
fixture below stands chat() in with `fakes_llm.passthrough_chat`, which echoes the
planned line straight back, so every "exact saved line" assertion below still holds for
the same reason it always did — a real reactive rewrite is covered separately in
test_run_scenario_flow_node.py.
"""
import json

import pytest

from app.core import runner, voice_caller
from app.core import voice_native_ws as nws
from app.core.node_script import PATH_TEST_TYPE, interrupt_script_to_scenario
from app.routers import flows as R
from tests.core.fakes_llm import passthrough_chat
from tests.core.fakes_native_ws import FakeAsyncClient, FakeConnect, FakeWebSocket, turn_frame
from tests.core.test_run_scenario_session_end import _Agent, _greeting, fake_db  # noqa: F401  (fixture)
from tests.routers.test_flow_interrupt_scenarios import PARSED
from tests.routers.test_flow_interrupt_scripts import INTERRUPTS, KINDS, READBACK, RECORD, _script_for

from app.core.node_script import validate_interrupt_script

SENTINEL = runner._AGENT_ERROR_SENTINEL
GRAPH = {"nodes": PARSED["nodes"], "edges": PARSED["edges"]}
PLAN = {s["interrupt"]["id"]: s for s in R._interrupt_scenarios(GRAPH, INTERRUPTS)[1]}
NAMES = {n["id"]: n.get("name") or n["id"] for n in PARSED["nodes"]}


@pytest.fixture(autouse=True)
def no_generated_caller_text(monkeypatch):
    """The AI Caller must never fire for a scripted scenario (still enforced); chat() IS
    now used, by _reactive_scripted_line, but stood in with a deterministic passthrough
    so these tests stay exact and network-free."""
    def boom(*a, **kw):
        raise AssertionError("scripted flow scenario must not use the AI Caller")

    monkeypatch.setattr(runner, "next_utterance", boom)
    monkeypatch.setattr(runner, "chat", passthrough_chat)


@pytest.fixture(autouse=True)
def clean_sessions():
    nws._SESSIONS.clear()
    yield
    nws._SESSIONS.clear()


def _saved(key):
    """(planner item, validated script) for one interrupt of the Maya flow."""
    item = PLAN[key]
    script = validate_interrupt_script(
        _script_for(item), item, node_kinds=KINDS, readback_steps=READBACK,
        interrupts=INTERRUPTS, record=RECORD,
    )
    return item, script


def _scenario(key):
    item, script = _saved(key)
    s = interrupt_script_to_scenario(
        9, item["id"], item["name"], item, NAMES,
        script["test_goal"], script["turns"], script["setup"], script["expectations"],
    )
    s["_id"] = 1
    return s, script


def _spoken(script):
    return [t["caller_line"] for t in script["turns"] if t["type"] in ("caller", "interrupt")]


def _interrupt_line(script):
    return next(t["caller_line"] for t in script["turns"] if t["type"] == "interrupt")


async def _run(scenario, send_fn, session_ended_fn):
    return await runner.run_scenario(
        run_id=1, scenario=scenario, agent={"voice_protocol": "native_ws"},
        send_fn=send_fn, greeting_fn=_greeting, session_ended_fn=session_ended_fn,
    )


def _ends_on_end_status(agent, session_key, trace):
    return bool(trace and trace.get("end_status"))


# ---------------------------------------------------------------------------
# Conversion: the saved script becomes the existing flow_path scenario shape
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("key", ["human_request", "wrong_number", "hold"])
def test_conversion_uses_the_saved_lines_verbatim(key):
    s, script = _scenario(key)
    assert s["test_type"] == PATH_TEST_TYPE and s["customer_context"] == "" and s["max_turns"] is None
    # Only caller + interrupt lines are spoken; listen/readback send nothing.
    assert s["seed_turns"] == _spoken(script)
    stored = json.loads(s["node_script_json"])
    assert stored["kind"] == "interrupt" and stored["turns"] == script["turns"]
    assert stored["setup"] == script["setup"] and stored["expectations"] == script["expectations"]
    assert stored["interrupt"]["key"] == key
    assert stored["interrupt"]["injection_step"] == len(PLAN[key]["segments"][0]["steps"])


def test_interrupt_line_is_at_its_planned_position():
    s, script = _scenario("hold")
    before = [t for t in script["turns"][: script["turns"].index(
        next(t for t in script["turns"] if t["type"] == "interrupt"))] if t["type"] == "caller"]
    assert s["seed_turns"].index(_interrupt_line(script)) == len(before)


def test_judge_reference_carries_every_expectation():
    s, script = _scenario("wrong_number")
    exp = s["expected_behavior"]
    assert exp.startswith("Planned path: ")
    assert "Interrupt: the caller raises 'wrong_number'" in exp and "end the call" in exp
    assert f"Expected outcome: {script['expectations']['outcome']}" in exp
    assert f"Expected end status: {script['expectations']['end_status']}" in exp
    assert all(f"- {g}" in exp for g in script["expectations"]["bug_guards"])
    assert all(t["expected_agent_behavior"] in exp for t in script["turns"])
    assert "user.first_name = Daniel" in exp
    assert s["user_goal"] == script["test_goal"]


# ---------------------------------------------------------------------------
# Execution through the real runner
# ---------------------------------------------------------------------------
async def test_greeting_first_then_every_saved_line_in_order_goto(fake_db):
    s, script = _scenario("human_request")
    agent = _Agent([{"reply": f"agent reply {i}", "trace": {}} for i in range(20)])
    conv = await _run(s, agent, _ends_on_end_status)

    assert agent.sent == _spoken(script)  # the goto continuation, exactly as saved
    transcript = fake_db.messages[conv]
    assert transcript[0] == {"turn_index": 0, "role": "agent", "content": "Hi, I'm Maya.", "trace": None}
    assert [m["role"] for m in transcript] == ["agent"] + ["tester", "agent"] * len(agent.sent)
    # Real replies are recorded as returned; nothing is manufactured from expectations.
    assert [m["content"] for m in transcript if m["role"] == "agent"][1:] == [
        f"agent reply {i}" for i in range(len(agent.sent))
    ]


async def test_resume_plays_the_saved_continuation(fake_db):
    s, script = _scenario("hold")
    agent = _Agent([{"reply": "ok", "trace": {}} for _ in range(30)])
    await _run(s, agent, _ends_on_end_status)
    idx = agent.sent.index(_interrupt_line(script))
    assert agent.sent[idx + 1:] == _spoken(script)[idx + 1:]
    assert len(agent.sent[idx + 1:]) > 0


async def test_end_interrupt_that_ends_the_call(fake_db):
    s, script = _scenario("wrong_number")
    agent = _Agent([
        {"reply": "Sorry to bother you, goodbye.", "trace": {"end_status": "No Answer | Retry Scheduled"}},
    ])
    conv = await _run(s, agent, _ends_on_end_status)
    assert agent.sent == [_interrupt_line(script)]
    assert fake_db.messages[conv][-1]["trace"]["end_status"] == "No Answer | Retry Scheduled"


async def test_end_interrupt_the_agent_does_not_honour_is_not_faked(fake_db):
    """The agent keeps talking instead of ending: its real reply is recorded, nothing is
    sent after the interrupt (the script has nothing after it), and no end is invented."""
    s, script = _scenario("wrong_number")
    agent = _Agent([{"reply": "Oh! Anyway, is now a good time?", "trace": {}}])
    conv = await _run(s, agent, _ends_on_end_status)
    assert agent.sent == [_interrupt_line(script)]
    last = fake_db.messages[conv][-1]
    assert last["content"] == "Oh! Anyway, is now a good time?" and not (last["trace"] or {}).get("end_status")


async def test_early_hangup_stops_the_remaining_lines(fake_db):
    s, script = _scenario("human_request")
    agent = _Agent([
        {"reply": "Sure.", "trace": {}},
        {"reply": "Goodbye.", "trace": {"end_status": "Contacted | Needs Review"}},
    ])
    conv = await _run(s, agent, _ends_on_end_status)
    assert agent.sent == _spoken(script)[:2]
    assert len(fake_db.messages[conv]) == 5


async def test_call_ended_in_the_greeting_sends_nothing(fake_db):
    s, _ = _scenario("hold")
    agent = _Agent([])
    conv = await _run(s, agent, lambda a, k, t: True)
    assert agent.sent == []
    assert [m["role"] for m in fake_db.messages[conv]] == ["agent"]


async def test_script_longer_than_the_old_five_turn_cap_is_played_in_full(fake_db):
    s, script = _scenario("hold")
    assert len(s["seed_turns"]) > runner.MAX_TESTER_TURNS
    agent = _Agent([{"reply": "ok", "trace": {}} for _ in range(40)])
    await _run(s, agent, _ends_on_end_status)
    assert agent.sent == _spoken(script)


# ---------------------------------------------------------------------------
# Real native_ws transport over a fake socket
# ---------------------------------------------------------------------------
async def test_native_ws_end_interrupt(fake_db, monkeypatch):
    s, script = _scenario("wrong_number")
    line = _interrupt_line(script)
    ws = FakeWebSocket(incoming=[
        turn_frame("agent", "Hi, this is Maya. Is now a good time?"),
        turn_frame("caller", line),
        turn_frame("agent", "Sorry for the trouble. Goodbye.", end_status="No Answer | Retry Scheduled"),
    ])
    calls_log: list[dict] = []
    monkeypatch.setattr(nws.websockets, "connect", FakeConnect([ws]))
    monkeypatch.setattr(nws.httpx, "AsyncClient", FakeAsyncClient(calls_log).factory())

    conv = await runner.run_scenario(
        run_id=1, scenario=s,
        agent={"voice_protocol": "native_ws", "endpoint_url": "http://localhost:8000"},
        send_fn=voice_caller.call_voice_agent, close_fn=voice_caller.close_voice_session,
        greeting_fn=voice_caller.peek_opening_greeting, session_ended_fn=voice_caller.voice_session_ended,
    )
    assert [json.loads(f)["text"] for f in ws.sent] == [line]
    assert [m["content"] for m in fake_db.messages[conv]] == [
        "Hi, this is Maya. Is now a good time?", line, "Sorry for the trouble. Goodbye.",
    ]
    assert ws.closed is True
    assert len([c for c in calls_log if c["url"].endswith("/end")]) == 1
