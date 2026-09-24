"""A flow scenario ("flow_path") stops as soon as its voice session has ended (Phase F).

Runner-level tests use the REAL run_scenario()/_run_scripted() with app.db faked
in-memory; transport-level tests drive the REAL native_ws transport over a fake socket
(same fakes as test_run_scenario_native_ws.py), and unit-test each transport's
read-only session_ended() helper.
"""
import json
from types import SimpleNamespace

import pytest

from app.core import activities, runner, twilio_bridge, voice_caller
from app.core import voice_native_ws as nws
from app.core.node_script import path_script_to_scenario
from tests.core.fakes_native_ws import FakeAsyncClient, FakeConnect, FakeWebSocket, turn_frame

SENTINEL = runner._AGENT_ERROR_SENTINEL


class _FakeDB:
    def __init__(self):
        self.messages: dict[int, list[dict]] = {}
        self._next = 1

    def get_or_create_conversation(self, run_id, scenario_id, idem_key=None):
        cid, self._next = self._next, self._next + 1
        self.messages[cid] = []
        return cid

    def clear_messages(self, conversation_id):
        self.messages[conversation_id] = []

    def insert_message(self, conversation_id, turn_index, role, content, trace):
        self.messages[conversation_id].append(
            {"turn_index": turn_index, "role": role, "content": content, "trace": trace}
        )


@pytest.fixture()
def fake_db(monkeypatch):
    db = _FakeDB()
    monkeypatch.setattr(runner, "get_or_create_conversation", db.get_or_create_conversation)
    monkeypatch.setattr(runner, "clear_messages", db.clear_messages)
    monkeypatch.setattr(runner, "insert_message", db.insert_message)
    return db


@pytest.fixture(autouse=True)
def no_generated_caller_text(monkeypatch):
    def boom(*a, **kw):
        raise AssertionError("no caller text may be generated")

    monkeypatch.setattr(runner, "next_utterance", boom)
    monkeypatch.setattr(runner, "chat", boom)


@pytest.fixture(autouse=True)
def clean_sessions():
    nws._SESSIONS.clear()
    twilio_bridge._SESSIONS_BY_KEY.clear()
    yield
    nws._SESSIONS.clear()
    twilio_bridge._SESSIONS_BY_KEY.clear()


class _Agent:
    """Scripted transport: `results[i]` is the i-th turn's {"reply", "trace"}."""

    def __init__(self, results):
        self._results = list(results)
        self.sent: list[str] = []

    async def __call__(self, agent, message, history, faults, session_key=None, is_last_turn=False):
        self.sent.append(message)
        return self._results.pop(0) if self._results else {"reply": "ok", "trace": {}}


async def _greeting(agent, session_key):
    return "Hi, I'm Maya."


def _ends_on_end_status(agent, session_key, trace):
    return bool(trace and trace.get("end_status"))


def _flow_path(n):
    path = [f"n{i}" for i in range(n)]
    turns = [
        {"step": i + 1, "node_id": f"n{i}", "expected_agent_behavior": "x", "caller_line": f"Line {i + 1}."}
        for i in range(n)
    ]
    s = path_script_to_scenario(7, "sid", "Retry / Recovery", path, {}, "goal", turns)
    s["_id"] = 1
    return s


async def _run(scenario, send_fn, session_ended_fn, agent=None):
    return await runner.run_scenario(
        run_id=1, scenario=scenario, agent=agent or {"voice_protocol": "native_ws"},
        send_fn=send_fn, greeting_fn=_greeting, session_ended_fn=session_ended_fn,
    )


# ---------------------------------------------------------------------------
# A. Normal flow_path — unchanged
# ---------------------------------------------------------------------------
async def test_live_call_plays_every_line_agent_first(fake_db):
    agent = _Agent([{"reply": f"reply {i}", "trace": {}} for i in range(8)])
    conv = await _run(_flow_path(8), agent, _ends_on_end_status)
    assert agent.sent == [f"Line {i + 1}." for i in range(8)]
    transcript = fake_db.messages[conv]
    assert transcript[0]["role"] == "agent" and transcript[0]["content"] == "Hi, I'm Maya."
    assert len(transcript) == 17


# ---------------------------------------------------------------------------
# B. Early hang-up
# ---------------------------------------------------------------------------
async def test_agent_ending_the_call_stops_the_script(fake_db):
    """The agent's 2nd reply ends the call (end_status): lines 3-8 are never sent and
    the transcript is exactly the turns that happened."""
    agent = _Agent([
        {"reply": "Can you talk now?", "trace": {}},
        {"reply": "Thanks, take care!", "trace": {"end_status": "Retry Scheduled | In Progress"}},
    ])
    conv = await _run(_flow_path(8), agent, _ends_on_end_status)

    assert agent.sent == ["Line 1.", "Line 2."]
    transcript = fake_db.messages[conv]
    assert [(m["role"], m["content"]) for m in transcript] == [
        ("agent", "Hi, I'm Maya."),
        ("tester", "Line 1."), ("agent", "Can you talk now?"),
        ("tester", "Line 2."), ("agent", "Thanks, take care!"),
    ]
    assert SENTINEL not in [m["content"] for m in transcript]


async def test_call_dying_mid_script_keeps_one_failed_attempt_and_stops(fake_db):
    """No end_status — the session just dies while line 3 is being delivered. That one
    failed turn (with its real error) is kept; nothing further is sent or recorded."""
    dead = {"closed": False}

    async def send(agent, message, history, faults, session_key=None, is_last_turn=False):
        send.sent.append(message)
        if len(send.sent) == 3:
            dead["closed"] = True
            return {"reply": SENTINEL, "trace": {"error": "native_ws turn error: received 1000 (OK)"}}
        return {"reply": f"reply {len(send.sent)}", "trace": {}}

    send.sent = []
    conv = await _run(_flow_path(8), send, lambda a, k, t: dead["closed"])

    assert send.sent == ["Line 1.", "Line 2.", "Line 3."]
    transcript = fake_db.messages[conv]
    assert len(transcript) == 7
    assert [m["content"] for m in transcript if m["content"] == SENTINEL] == [SENTINEL]
    assert transcript[-1]["trace"]["error"].startswith("native_ws turn error")


async def test_session_ending_on_the_last_line_changes_nothing(fake_db):
    agent = _Agent([{"reply": "a", "trace": {}}, {"reply": "bye", "trace": {"end_status": "Done"}}])
    await _run(_flow_path(2), agent, _ends_on_end_status)
    assert agent.sent == ["Line 1.", "Line 2."]


async def test_without_a_session_check_a_flow_scenario_behaves_as_in_phase_e(fake_db):
    agent = _Agent([{"reply": "bye", "trace": {"end_status": "Done"}}] * 8)
    await _run(_flow_path(8), agent, None)
    assert len(agent.sent) == 8


# ---------------------------------------------------------------------------
# C. Every other scenario is unchanged
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("test_type", ["support", "happy_path", "flow_node", "memory"])
async def test_other_scripted_scenarios_ignore_session_end(fake_db, test_type):
    """Even when the session reports ended after every turn, a non-flow scenario keeps
    its existing behaviour: it plays up to the 5-turn cap, exactly as before."""
    agent = _Agent([{"reply": SENTINEL, "trace": {"end_status": "Done"}}] * 8)
    scenario = {
        "_id": 2, "test_type": test_type, "assigned_fault": "none",
        "seed_turns": [f"line {i}" for i in range(8)], "customer_context": "",
    }
    await _run(scenario, agent, lambda a, k, t: True)
    assert agent.sent == [f"line {i}" for i in range(runner.MAX_TESTER_TURNS)]


# ---------------------------------------------------------------------------
# D. Transports
# ---------------------------------------------------------------------------
def _wire_native_ws(monkeypatch, frames):
    ws = FakeWebSocket(incoming=frames)
    calls_log: list[dict] = []
    monkeypatch.setattr(nws.websockets, "connect", FakeConnect([ws]))
    monkeypatch.setattr(nws.httpx, "AsyncClient", FakeAsyncClient(calls_log).factory())
    return ws, calls_log


async def _run_native_ws(scenario):
    return await runner.run_scenario(
        run_id=1, scenario=scenario,
        agent={"voice_protocol": "native_ws", "endpoint_url": "http://localhost:8000"},
        send_fn=voice_caller.call_voice_agent, close_fn=voice_caller.close_voice_session,
        greeting_fn=voice_caller.peek_opening_greeting,
        session_ended_fn=voice_caller.voice_session_ended,
    )


def _utterances(ws):
    return [json.loads(f)["text"] for f in ws.sent]


async def test_native_ws_agent_hangup_with_end_status(fake_db, monkeypatch):
    ws, calls_log = _wire_native_ws(monkeypatch, [
        turn_frame("agent", "Hi, I'm Maya."),
        turn_frame("caller", "Line 1."), turn_frame("agent", "Do you have a minute?"),
        turn_frame("caller", "Line 2."),
        turn_frame("agent", "Thanks, take care!", end_status="Retry Scheduled | In Progress"),
    ])
    conv = await _run_native_ws(_flow_path(8))

    assert _utterances(ws) == ["Line 1.", "Line 2."]  # nothing sent into the ended call
    transcript = fake_db.messages[conv]
    assert [m["content"] for m in transcript] == [
        "Hi, I'm Maya.", "Line 1.", "Do you have a minute?", "Line 2.", "Thanks, take care!",
    ]
    assert transcript[-1]["trace"]["end_status"] == "Retry Scheduled | In Progress"
    # Finalized normally: socket closed, /end posted exactly once.
    assert ws.closed is True
    assert len([c for c in calls_log if c["url"].endswith("/end")]) == 1


async def test_native_ws_socket_dropping_without_end_status(fake_db, monkeypatch):
    ws, calls_log = _wire_native_ws(monkeypatch, [
        turn_frame("agent", "Hi, I'm Maya."),
        turn_frame("caller", "Line 1."), turn_frame("agent", "Go on."),
        turn_frame("caller", "Line 2."), turn_frame("agent", "Mm-hm."),
        # nothing more: the next recv raises, as a closed socket does
    ])
    conv = await _run_native_ws(_flow_path(8))

    assert _utterances(ws) == ["Line 1.", "Line 2.", "Line 3."]
    transcript = fake_db.messages[conv]
    assert len(transcript) == 7
    assert transcript[-1]["content"] == SENTINEL
    assert "native_ws turn error" in transcript[-1]["trace"]["error"]
    assert not any("already ended" in (m["trace"] or {}).get("error", "") for m in transcript if m["trace"])
    assert ws.closed is True


def test_native_ws_session_ended_is_read_only():
    assert nws.session_ended("k", {"end_status": "Done"}) is True
    assert nws.session_ended("missing", {}) is False
    nws._SESSIONS["live"] = nws._Session(call_id="c")
    assert nws.session_ended("live", {"call_id": "c"}) is False
    nws._SESSIONS["dead"] = nws._Session(call_id="c", closed=True)
    assert nws.session_ended("dead", None) is True
    assert nws._SESSIONS["live"].closed is False  # unchanged by asking


def test_twilio_session_ended_is_read_only():
    twilio_bridge._SESSIONS_BY_KEY["live"] = SimpleNamespace(closed=False)
    twilio_bridge._SESSIONS_BY_KEY["dead"] = SimpleNamespace(closed=True)
    assert twilio_bridge.session_ended("live") is False
    assert twilio_bridge.session_ended("dead") is True
    assert twilio_bridge.session_ended("missing") is False


@pytest.mark.parametrize("protocol", ["http_json", "websocket", None])
def test_stateless_transports_never_report_an_ended_session(protocol):
    agent = {"voice_protocol": protocol} if protocol else {}
    assert voice_caller.voice_session_ended(agent, "k", {"error": "boom", "end_status": "x"}) is False


def test_voice_session_ended_dispatches_by_protocol():
    twilio_bridge._SESSIONS_BY_KEY["k"] = SimpleNamespace(closed=True)
    assert voice_caller.voice_session_ended({"voice_protocol": "twilio"}, "k", None) is True
    assert voice_caller.voice_session_ended({"voice_protocol": "native_ws"}, "k", {"end_status": "Done"}) is True
    assert voice_caller.voice_session_ended({"voice_protocol": "native_ws"}, "k", {}) is False


async def test_a_stateless_transport_failure_does_not_stop_a_flow_scenario(fake_db):
    agent = _Agent([{"reply": SENTINEL, "trace": {"error": "http 500"}}] + [{"reply": "ok", "trace": {}}] * 3)
    await _run(_flow_path(4), agent, voice_caller.voice_session_ended, agent={"voice_protocol": "http_json"})
    assert len(agent.sent) == 4


# ---------------------------------------------------------------------------
# Wiring: the voice activities hand the runner the session check
# ---------------------------------------------------------------------------
async def test_voice_activities_pass_the_session_check(monkeypatch):
    seen = []

    async def fake_run_scenario(*a, **kw):
        seen.append(kw)
        return 1

    monkeypatch.setattr(activities, "run_scenario", fake_run_scenario)
    await activities.play_voice_scenario(1, {"_id": 1}, {"modality": "voice"})
    await activities.replay_scenario(1, {"_id": 1}, {"modality": "voice"})
    await activities.replay_scenario(1, {"_id": 1}, {"modality": "chat"})
    assert seen[0]["session_ended_fn"] is voice_caller.voice_session_ended
    assert seen[1]["session_ended_fn"] is voice_caller.voice_session_ended
    assert seen[2]["session_ended_fn"] is None


# ---------------------------------------------------------------------------
# Phase G — the call ends during the opening greeting
# ---------------------------------------------------------------------------
_ENDED_GREETING = turn_frame(
    "agent", "Hi, I'm Maya. Sorry, we can't take this call right now. Goodbye.",
    end_status="Contacted | Registration Declined",
)


async def test_normal_greeting_still_leads_to_caller_line_one(fake_db, monkeypatch):
    ws, _ = _wire_native_ws(monkeypatch, [
        turn_frame("agent", "Hi, I'm Maya."),
        turn_frame("caller", "Line 1."), turn_frame("agent", "Go on."),
        turn_frame("caller", "Line 2."), turn_frame("agent", "Thanks."),
    ])
    conv = await _run_native_ws(_flow_path(2))
    assert _utterances(ws) == ["Line 1.", "Line 2."]
    assert [m["role"] for m in fake_db.messages[conv]] == ["agent", "tester", "agent", "tester", "agent"]


async def test_call_ended_in_the_greeting_sends_no_caller_line(fake_db, monkeypatch):
    ws, calls_log = _wire_native_ws(monkeypatch, [_ENDED_GREETING])
    conv = await _run_native_ws(_flow_path(8))

    assert _utterances(ws) == []  # caller line 1 was never sent
    transcript = fake_db.messages[conv]
    assert transcript == [{
        "turn_index": 0, "role": "agent",
        "content": "Hi, I'm Maya. Sorry, we can't take this call right now. Goodbye.", "trace": None,
    }]  # no artificial caller turn, no transport error
    # Finalized normally.
    assert ws.closed is True
    assert len([c for c in calls_log if c["url"].endswith("/end")]) == 1


async def test_greeting_that_ends_the_call_is_ignored_by_other_scenarios(fake_db, monkeypatch):
    """flow_node keeps its existing behaviour: it still sends its first line (which
    here the fake socket happens to answer)."""
    ws, _ = _wire_native_ws(monkeypatch, [
        _ENDED_GREETING, turn_frame("caller", "line 0"), turn_frame("agent", "..."),
    ])
    scenario = {
        "_id": 3, "test_type": "flow_node", "assigned_fault": "none",
        "seed_turns": ["line 0"], "customer_context": "",
    }
    await _run_native_ws(scenario)
    assert _utterances(ws) == ["line 0"]


async def test_setup_failure_without_a_greeting_is_unchanged(fake_db, monkeypatch):
    """No greeting at all because the call couldn't be set up: the flow scenario
    records its one failed attempt (with the error) as in Phase F — it is not silently
    dropped."""
    monkeypatch.setattr(nws.websockets, "connect", FakeConnect([], fail=True))
    monkeypatch.setattr(nws.httpx, "AsyncClient", FakeAsyncClient([]).factory())
    conv = await _run_native_ws(_flow_path(8))
    transcript = fake_db.messages[conv]
    assert [m["role"] for m in transcript] == ["tester", "agent"]
    assert transcript[1]["content"] == SENTINEL


def test_session_ended_reports_a_greeting_that_ended_the_call():
    nws._SESSIONS["k"] = nws._Session(call_id="c", opening_end_status="Contacted | Needs Review")
    assert nws.session_ended("k", None) is True
    nws._SESSIONS["live"] = nws._Session(call_id="c", opening_end_status=None)
    assert nws.session_ended("live", None) is False


async def test_greeting_end_status_is_captured_by_the_existing_greeting_path(monkeypatch):
    _wire_native_ws(monkeypatch, [_ENDED_GREETING])
    text = await voice_caller.peek_opening_greeting(
        {"voice_protocol": "native_ws", "endpoint_url": "http://localhost:8000"}, "g",
    )
    assert text.startswith("Hi, I'm Maya.")  # the greeting helper still returns text only
    assert nws._SESSIONS["g"].opening_end_status == "Contacted | Registration Declined"
    assert voice_caller.voice_session_ended({"voice_protocol": "native_ws"}, "g", None) is True


async def test_runner_level_ended_greeting_with_scripted_doubles(fake_db):
    agent = _Agent([])
    conv = await _run(_flow_path(8), agent, lambda a, k, t: True)
    assert agent.sent == []
    assert [m["role"] for m in fake_db.messages[conv]] == ["agent"]
