"""Part 7 — native_ws agent-turn synchronization, call variables and trace plumbing.

The target speaks each step as its own "turn" event and drops a caller line sent
while it is still producing consecutive turns (Maya: greet, then ask_good_time). These
tests drive the REAL transport (and, where noted, the REAL run_scenario) over a timed
fake socket: frames are released in batches — batch 0 when the socket opens, batch k
after the k-th caller line is sent — each frame after its own small delay, and recv()
blocks when nothing is due, exactly like a live socket. So "the caller waited" is a
real ordering fact, checked from the socket's own event log.
"""
import asyncio
import json

import pytest

from app.core import runner, voice_caller
from app.core import voice_native_ws as nws
from app.core.node_script import path_script_to_scenario
from tests.core.fakes_native_ws import FakeAsyncClient, FakeConnect, turn_frame
from tests.core.test_run_scenario_session_end import fake_db  # noqa: F401  (fixture)

AGENT = {"voice_protocol": "native_ws", "endpoint_url": "http://localhost:8000"}
MAYA_TEMPLATE = '{"client": "client_a", "flow": "maya_renewal", "patient_ref": "{{patient_ref}}"}'


@pytest.fixture(autouse=True)
def fast_settle(monkeypatch):
    monkeypatch.setattr(nws, "SETTLE_S", 0.15)
    monkeypatch.setattr(nws, "AUDIO_QUIET_S", 0.1)
    monkeypatch.setattr(nws, "SPEAKING_MAX_S", 2.0)
    monkeypatch.setattr(nws, "MAX_OUTPUT_S", 5.0)


@pytest.fixture(autouse=True)
def clean_sessions():
    nws._SESSIONS.clear()
    yield
    nws._SESSIONS.clear()


def speaking(on: bool) -> str:
    return json.dumps({"type": "speaking", "speaking": on})


class TimedSocket:
    """batches[k] = [(delay_s, frame), ...] released after the k-th send (batch 0 on
    open). `log` records ("recv", frame) / ("send", text) in real order."""

    def __init__(self, batches):
        self._batches = [list(b) for b in batches]
        self._queue: asyncio.Queue = asyncio.Queue()
        self.log: list[tuple] = []
        self.sent: list[str] = []
        self.closed = False
        self._release(0)

    def _release(self, k):
        if k >= len(self._batches):
            return
        frames = self._batches[k]

        async def feed():
            for delay, frame in frames:
                await asyncio.sleep(delay)
                await self._queue.put(frame)

        asyncio.get_event_loop().create_task(feed())

    async def send(self, data):
        text = json.loads(data)["text"]
        self.sent.append(text)
        self.log.append(("send", text))
        self._release(len(self.sent))

    async def recv(self):
        frame = await self._queue.get()
        if isinstance(frame, Exception):
            raise frame
        self.log.append(("recv", frame))
        return frame

    async def close(self, code=1000, reason=""):
        self.closed = True


def _wire(monkeypatch, ws):
    calls_log: list[dict] = []
    monkeypatch.setattr(nws.websockets, "connect", FakeConnect([ws]))
    monkeypatch.setattr(nws.httpx, "AsyncClient", FakeAsyncClient(calls_log).factory())
    return calls_log


def _agent_texts_before_first_send(ws):
    out = []
    for kind, item in ws.log:
        if kind == "send":
            return out
        if isinstance(item, str) and json.loads(item).get("type") == "turn":
            out.append(json.loads(item)["text"])
    return out


async def _run(scenario, agent=AGENT):
    return await runner.run_scenario(
        run_id=1, scenario=scenario, agent=agent,
        send_fn=voice_caller.call_voice_agent, close_fn=voice_caller.close_voice_session,
        greeting_fn=voice_caller.peek_opening_greeting, session_ended_fn=voice_caller.voice_session_ended,
    )


def _flow_path(lines):
    path = [f"n{i}" for i in range(len(lines))]
    turns = [{"step": i + 1, "node_id": n, "expected_agent_behavior": "x", "caller_line": line}
             for i, (n, line) in enumerate(zip(path, lines))]
    s = path_script_to_scenario(9, "sid", "Path", path, {}, "goal", turns)
    s["_id"] = 1
    return s


# ---------------------------------------------------------------------------
# Synchronization
# ---------------------------------------------------------------------------
async def test_one_agent_turn_then_the_caller_speaks(fake_db, monkeypatch):
    ws = TimedSocket([
        [(0, turn_frame("agent", "Hi, is now a good time?"))],
        [(0, turn_frame("caller", "Yes.")), (0.02, turn_frame("agent", "Great, thanks."))],
    ])
    _wire(monkeypatch, ws)
    conv = await _run(_flow_path(["Yes."]))
    assert ws.sent == ["Yes."]
    assert [m["content"] for m in fake_db.messages[conv]] == ["Hi, is now a good time?", "Yes.", "Great, thanks."]
    assert "agent_turns" not in fake_db.messages[conv][2]["trace"]  # single turn: trace as before


async def test_two_opening_turns_are_both_received_before_the_first_line(fake_db, monkeypatch):
    """Maya's shape: the greeting, then — a moment later, unprompted — its first question."""
    ws = TimedSocket([
        [(0, turn_frame("agent", "Hello, this is Maya.")),
         (0.08, turn_frame("agent", "Would you have a couple of minutes?"))],
        [(0, turn_frame("caller", "wrong number")),
         (0.02, turn_frame("agent", "Sorry to trouble you.", end_status="No Answer | Retry Scheduled",
                           interrupt_key="wrong_number"))],
    ])
    _wire(monkeypatch, ws)
    conv = await _run(_flow_path(["wrong number", "never sent"]))

    assert _agent_texts_before_first_send(ws) == ["Hello, this is Maya.", "Would you have a couple of minutes?"]
    assert ws.sent == ["wrong number"]  # the call ended: the second line is never sent
    transcript = fake_db.messages[conv]
    assert transcript[0]["content"] == "Hello, this is Maya. Would you have a couple of minutes?"
    assert transcript[-1]["trace"]["end_status"] == "No Answer | Retry Scheduled"
    assert transcript[-1]["trace"]["interrupt_key"] == "wrong_number"


async def test_many_consecutive_turns_with_speaking_and_audio_frames(fake_db, monkeypatch):
    """Four consecutive agent turns, interleaved with speaking events and binary audio.
    The agent is still speaking for longer than SETTLE_S between two of them, which
    must not end the wait."""
    ws = TimedSocket([
        [(0, turn_frame("agent", "Greeting."))],
        [(0, turn_frame("caller", "Yes.")), (0, speaking(True)), (0, b"\x00\x01"),
         (0.01, turn_frame("agent", "Thanks for confirming.")),
         (0.3, b"\x00\x02"),                         # still speaking, > SETTLE_S later
         (0.01, turn_frame("agent", "One more thing.")),
         (0.01, speaking(False)),
         (0.1, speaking(True)),                       # starts again inside SETTLE_S
         (0.01, turn_frame("agent", "Is your address the same?")),
         (0.01, speaking(False))],
        [(0, turn_frame("caller", "It is.")), (0.01, turn_frame("agent", "Perfect."))],
    ])
    _wire(monkeypatch, ws)
    conv = await _run(_flow_path(["Yes.", "It is."]))

    assert ws.sent == ["Yes.", "It is."]
    second_send = [i for i, (k, _) in enumerate(ws.log) if k == "send"][1]
    received = [json.loads(f)["text"] for k, f in ws.log[:second_send]
                if k == "recv" and isinstance(f, str) and json.loads(f).get("type") == "turn"]
    assert received[-1] == "Is your address the same?"
    reply = fake_db.messages[conv][2]
    assert reply["content"] == "Thanks for confirming. One more thing. Is your address the same?"
    assert reply["trace"]["agent_turns"] == ["Thanks for confirming.", "One more thing.", "Is your address the same?"]


async def test_turn_arriving_later_than_settle_while_audio_still_plays(fake_db, monkeypatch):
    """Maya's live timing: no speaking events; the greeting's audio streams in real time
    and the next turn arrives LATER than SETTLE_S after the first — but while audio is
    still arriving. The caller must still wait for it."""
    audio = [(0.05, b"\x00" * 320) for _ in range(8)]  # 0.4 s of streamed audio
    ws = TimedSocket([
        [(0, turn_frame("agent", "Hello, this is Maya."))] + audio
        + [(0.01, turn_frame("agent", "Would you have a couple of minutes?"))] + audio,
        [(0, turn_frame("caller", "hold on")), (0.01, turn_frame("agent", "Of course, I'll wait.", interrupt_key="hold"))],
    ])
    _wire(monkeypatch, ws)
    monkeypatch.setattr(nws, "SETTLE_S", 0.1)  # < the 0.4 s between the two turns
    conv = await _run(_flow_path(["hold on"]))
    assert _agent_texts_before_first_send(ws) == ["Hello, this is Maya.", "Would you have a couple of minutes?"]
    last = fake_db.messages[conv][-1]
    assert last["content"] == "Of course, I'll wait." and last["trace"]["interrupt_key"] == "hold"


async def test_output_is_capped(monkeypatch):
    monkeypatch.setattr(nws, "MAX_OUTPUT_S", 0.3)
    ws = TimedSocket([[(0, turn_frame("agent", "Hi."))] + [(0.05, b"\x00") for _ in range(40)]])
    _wire(monkeypatch, ws)
    loop = asyncio.get_running_loop()
    t0 = loop.time()
    assert await nws.peek_greeting(AGENT, "k") == "Hi."
    assert loop.time() - t0 < 1.0


async def test_agent_ends_the_call_during_consecutive_turns(fake_db, monkeypatch):
    ws = TimedSocket([
        [(0, turn_frame("agent", "Hi."))],
        [(0, turn_frame("caller", "I'm busy.")),
         (0.01, turn_frame("agent", "No problem.")),
         (0.05, turn_frame("agent", "Goodbye.", end_status="Contacted | Callback"))],
    ])
    calls_log = _wire(monkeypatch, ws)
    conv = await _run(_flow_path(["I'm busy.", "never sent", "never sent"]))
    assert ws.sent == ["I'm busy."]
    last = fake_db.messages[conv][-1]
    assert last["content"] == "No problem. Goodbye."
    assert last["trace"]["end_status"] == "Contacted | Callback"
    assert ws.closed and len([c for c in calls_log if c["url"].endswith("/end")]) == 1


async def test_call_ended_in_a_consecutive_opening_turn_sends_nothing(fake_db, monkeypatch):
    ws = TimedSocket([[(0, turn_frame("agent", "Hi.")),
                       (0.05, turn_frame("agent", "We can't take this call. Goodbye.", end_status="Declined"))]])
    _wire(monkeypatch, ws)
    conv = await _run(_flow_path(["never sent"]))
    assert ws.sent == []
    assert [m["content"] for m in fake_db.messages[conv]] == ["Hi. We can't take this call. Goodbye."]


async def test_agent_without_speaking_events_settles_after_its_last_turn(monkeypatch):
    ws = TimedSocket([[(0, turn_frame("agent", "One.")), (0.05, turn_frame("agent", "Two."))]])
    _wire(monkeypatch, ws)
    greeting = await nws.peek_greeting(AGENT, "k")
    assert greeting == "One. Two."


async def test_socket_closing_after_a_turn_returns_what_was_said(monkeypatch):
    ws = TimedSocket([[(0, turn_frame("agent", "Hello.")), (0.01, RuntimeError("closed"))]])
    _wire(monkeypatch, ws)
    assert await nws.peek_greeting(AGENT, "k") == "Hello."


@pytest.mark.parametrize("test_type", ["flow_node", "support"])
async def test_node_tests_and_other_voice_scenarios_use_the_same_reader(fake_db, monkeypatch, test_type):
    """A node test (agent-first) and a plain voice scenario both get the whole reply."""
    ws = TimedSocket([
        [(0, turn_frame("agent", "Hello.")), (0.05, turn_frame("agent", "How can I help?"))],
        [(0, turn_frame("caller", "Hi.")), (0.01, turn_frame("agent", "Sure.")), (0.05, turn_frame("agent", "Anything else?"))],
    ])
    _wire(monkeypatch, ws)
    scenario = {"_id": 2, "test_type": test_type, "assigned_fault": "none", "seed_turns": ["Hi."], "customer_context": ""}
    conv = await _run(scenario)
    agent_msgs = [m["content"] for m in fake_db.messages[conv] if m["role"] == "agent"]
    assert agent_msgs[-1] == "Sure. Anything else?"
    if test_type == "flow_node":
        assert agent_msgs[0] == "Hello. How can I help?" and _agent_texts_before_first_send(ws) == ["Hello.", "How can I help?"]


async def test_interrupt_scenario_resume_plays_its_continuation(fake_db, monkeypatch):
    from tests.core.test_run_scenario_interrupt import _scenario, _spoken

    s, script = _scenario("hold")
    lines = _spoken(script)
    batches = [[(0, turn_frame("agent", "Hello.")), (0.03, turn_frame("agent", "Is now a good time?"))]]
    for i, line in enumerate(lines):
        batches.append([(0, turn_frame("caller", line)), (0.01, turn_frame("agent", f"reply {i}"))])
    ws = TimedSocket(batches)
    _wire(monkeypatch, ws)
    await _run(s)
    assert ws.sent == lines
    assert _agent_texts_before_first_send(ws) == ["Hello.", "Is now a good time?"]


# ---------------------------------------------------------------------------
# Trace
# ---------------------------------------------------------------------------
async def test_trace_keeps_existing_fields_and_adds_interrupt_key(monkeypatch):
    ws = TimedSocket([
        [(0, turn_frame("agent", "Hi."))],
        [(0, turn_frame("caller", "hold on")),
         (0.01, turn_frame("agent", "Of course, take your time.", interrupt_key="hold", answers={"good_time": "Yes"}))],
    ])
    _wire(monkeypatch, ws)
    result = await nws._call_via_native_ws(AGENT, "hold on", session_key="k")
    assert result["reply"] == "Of course, take your time."
    assert result["trace"] == {
        "call_id": "call-1", "opening_line": "Hi.",
        "answers": {"good_time": "Yes"}, "interrupt_key": "hold",
    }


async def test_trace_without_interrupt_is_unchanged(monkeypatch):
    ws = TimedSocket([[(0, turn_frame("agent", "Hi."))],
                      [(0, turn_frame("caller", "x")), (0.01, turn_frame("agent", "ok", interrupt_key=None))]])
    _wire(monkeypatch, ws)
    result = await nws._call_via_native_ws(AGENT, "x", session_key="k")
    assert result["trace"] == {"call_id": "call-1", "opening_line": "Hi."}


# ---------------------------------------------------------------------------
# Call variables
# ---------------------------------------------------------------------------
def test_template_without_variables_is_unchanged():
    agent = {**AGENT, "request_template": '{"client": "client_a", "flow": "maya_renewal"}'}
    assert nws._create_call_body(agent) == {"client": "client_a", "flow": "maya_renewal"}
    assert nws._create_call_body({**AGENT, "request_template": "not json"}) == {}
    assert nws._create_call_body(AGENT) == {}


def test_declared_variable_not_supplied_is_left_out():
    agent = {**AGENT, "request_template": MAYA_TEMPLATE}
    assert nws._create_call_body(agent) == {"client": "client_a", "flow": "maya_renewal"}
    assert nws.call_template_variables(agent) == ["patient_ref"]


def test_supplied_variable_fills_its_mapped_field():
    agent = {**AGENT, "request_template": '{"client": "c", "patient_ref": "{{ patient }}", "language": "{{lang}}"}',
             "call_variables": {"patient": " adult-en ", "lang": "en-US"}}
    assert nws._create_call_body(agent) == {"client": "c", "patient_ref": "adult-en", "language": "en-US"}


@pytest.mark.parametrize("agent, message", [
    ({**AGENT, "request_template": '{"client": "c"}', "call_variables": {"patient_ref": "x"}}, "not mapped"),
    ({**AGENT, "request_template": "not json", "call_variables": {"patient_ref": "x"}}, "not a JSON object"),
    ({"voice_protocol": "http_json", "request_template": MAYA_TEMPLATE, "call_variables": {"patient_ref": "x"}}, "only supported for native_ws"),
    ({**AGENT, "request_template": MAYA_TEMPLATE, "call_variables": {"patient_ref": ""}}, "non-empty"),
])
def test_invalid_call_variables_fail_clearly(agent, message):
    with pytest.raises(nws.CallVariableError, match=message):
        nws._create_call_body(agent)


async def test_call_variables_reach_call_creation(fake_db, monkeypatch):
    from app.core import activities

    ws = TimedSocket([[(0, turn_frame("agent", "Hi."))], [(0, turn_frame("caller", "x")), (0, turn_frame("agent", "ok"))]])
    calls_log = _wire(monkeypatch, ws)
    agent = {**AGENT, "modality": "voice", "request_template": MAYA_TEMPLATE}
    scenario = {**_flow_path(["x"]), "call_variables": {"patient_ref": "adult-en"}}
    await _run(scenario, agent=activities._with_call_variables(agent, scenario))
    create = next(c for c in calls_log if c["url"].endswith("/api/calls"))
    assert create["json"] == {"client": "client_a", "flow": "maya_renewal", "patient_ref": "adult-en"}


async def test_unmapped_call_variable_places_no_call_and_says_why(fake_db, monkeypatch):
    ws = TimedSocket([[(0, turn_frame("agent", "Hi."))]])
    calls_log = _wire(monkeypatch, ws)
    agent = {**AGENT, "request_template": '{"client": "c"}', "call_variables": {"patient_ref": "adult-en"}}
    conv = await _run(_flow_path(["x"]), agent=agent)
    assert calls_log == []  # nothing was created
    last = fake_db.messages[conv][-1]
    assert last["content"] == runner._AGENT_ERROR_SENTINEL and "not mapped" in last["trace"]["error"]


def test_activities_attach_call_variables_only_when_present():
    from app.core import activities

    agent = {"id": 1, "modality": "voice"}
    assert activities._with_call_variables(agent, {"_id": 1}) is agent
    assert activities._with_call_variables(agent, {"call_variables": {"a": "b"}})["call_variables"] == {"a": "b"}
    assert activities._replay_call_variables(json.dumps({"setup": {"call": {"patient_ref": "adult-en"}}})) == {
        "call_variables": {"patient_ref": "adult-en"}}
    assert activities._replay_call_variables(None) == {} and activities._replay_call_variables("[]") == {}
