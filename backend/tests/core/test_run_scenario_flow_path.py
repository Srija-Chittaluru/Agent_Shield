"""Runner tests for executing a flow scenario (test_type "flow_path", Phase E) through
the REAL run_scenario() / _run_scripted() — plus regression guards proving every other
scripted scenario keeps the 5-turn cap.

Same faking approach as test_run_scenario_flow_node.py: app.db is faked in-memory and
send_fn/greeting_fn are scripted test doubles. One test drives the REAL native_ws
transport (fake socket only) to show the existing greeting handling is what's used.

_run_scripted now passes every seed_turns line through _reactive_scripted_line, which
calls chat() to react to the agent's actual last reply (see runner.py). The autouse
fixture below stands chat() in with `fakes_llm.passthrough_chat`, which echoes the
planned line straight back, so every "caller lines stay verbatim" assertion below still
holds for the same reason it always did — a real reactive rewrite is covered separately
in test_run_scenario_flow_node.py.
"""
import pytest

from app.core import runner, voice_caller
from app.core import voice_native_ws as nws
from app.core.node_script import PATH_TEST_TYPE, path_script_to_scenario
from tests.core.fakes_llm import passthrough_chat
from tests.core.fakes_native_ws import FakeAsyncClient, FakeConnect, FakeWebSocket, turn_frame


class _FakeDB:
    def __init__(self):
        self.messages: dict[int, list[dict]] = {}
        self._next_id = 1

    def get_or_create_conversation(self, run_id, scenario_id, idem_key=None):
        cid = self._next_id
        self._next_id += 1
        self.messages[cid] = []
        return cid

    def clear_messages(self, conversation_id):
        self.messages[conversation_id] = []
        return 0

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
def no_llm_caller(monkeypatch):
    """A scripted scenario must never reach the AI Caller or the adaptive follow-up LLM
    (both still forbidden); chat() IS now used, by _reactive_scripted_line, but stood in
    with a deterministic passthrough so these tests stay verbatim and network-free."""
    def boom(*a, **kw):
        raise AssertionError("scripted flow scenario must not use the AI Caller")

    monkeypatch.setattr(runner, "next_utterance", boom)
    monkeypatch.setattr(runner, "chat", passthrough_chat)


class _Agent:
    """Scripted send_fn: replies in order, records exactly what the caller sent."""

    def __init__(self, replies):
        self._replies = list(replies)
        self.sent: list[str] = []

    async def __call__(self, agent, message, history, faults, session_key=None, is_last_turn=False):
        self.sent.append(message)
        return {"reply": self._replies.pop(0) if self._replies else "ok", "trace": {}}


async def _greeting(agent, session_key):
    return "Hi, this is Maya from the clinic. Is now a good time?"


def _flow_path_scenario(n_turns, scenario_id=1):
    path = [f"n{i}" for i in range(n_turns)]
    turns = [
        {"step": i + 1, "node_id": f"n{i}", "expected_agent_behavior": f"Agent does step {i + 1}.",
         "caller_line": f"Caller line {i + 1}."}
        for i in range(n_turns)
    ]
    s = path_script_to_scenario(7, "abc", "Retry / Recovery", path, {}, "Verify the path.", turns)
    s["_id"] = scenario_id
    return s


async def _run(scenario, agent, greeting=_greeting):
    return await runner.run_scenario(
        run_id=1, scenario=scenario, agent={"voice_protocol": "native_ws"},
        send_fn=agent, greeting_fn=greeting,
    )


# ---------------------------------------------------------------------------
# B. Turn count
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n_turns", [8, 12])
async def test_flow_scenario_plays_every_script_turn(fake_db, n_turns):
    agent = _Agent([f"agent reply {i}" for i in range(n_turns)])
    conv = await _run(_flow_path_scenario(n_turns), agent)
    assert agent.sent == [f"Caller line {i + 1}." for i in range(n_turns)]
    transcript = fake_db.messages[conv]
    assert len(transcript) == 1 + 2 * n_turns  # greeting + one tester/agent pair per turn
    assert [m["role"] for m in transcript if m["role"] == "tester"] == ["tester"] * n_turns


@pytest.mark.parametrize("test_type", ["support", "happy_path", "flow_node"])
async def test_every_other_scripted_scenario_is_still_capped_at_five(fake_db, test_type):
    agent = _Agent([])
    scenario = {
        "_id": 2, "test_type": test_type, "assigned_fault": "none",
        "seed_turns": [f"line {i}" for i in range(8)], "customer_context": "",
    }
    await _run(scenario, agent)
    assert agent.sent == [f"line {i}" for i in range(runner.MAX_TESTER_TURNS)]
    assert runner.MAX_TESTER_TURNS == 5


async def test_flow_scenario_caller_lines_stay_verbatim_whatever_the_agent_says(fake_db):
    # The agent goes completely off-path; with chat() stood in by passthrough_chat (a
    # deterministic no-op), the caller still says the next saved line. A real chat()
    # would instead react to the agent's reply — see
    # test_run_scenario_flow_node.py::test_flow_node_caller_line_is_rephrased_to_answer_what_the_agent_actually_asked.
    agent = _Agent(["I can't help with that.", "Goodbye.", "What?", "Hello?"])
    await _run(_flow_path_scenario(4), agent)
    assert agent.sent == ["Caller line 1.", "Caller line 2.", "Caller line 3.", "Caller line 4."]


async def test_flow_scenario_agent_replies_are_recorded_unaltered(fake_db):
    replies = ["Sure — what's your name?", "Hmm, I don't see that name.", "Thanks!"]
    conv = await _run(_flow_path_scenario(3), _Agent(replies))
    agent_turns = [m["content"] for m in fake_db.messages[conv] if m["role"] == "agent"]
    assert agent_turns[1:] == replies  # [0] is the greeting


# ---------------------------------------------------------------------------
# C. Agent-first behavior
# ---------------------------------------------------------------------------
async def test_agent_greeting_is_received_before_caller_turn_one(fake_db):
    agent = _Agent(["reply 1", "reply 2"])
    conv = await _run(_flow_path_scenario(2), agent)
    transcript = fake_db.messages[conv]
    assert [(m["turn_index"], m["role"]) for m in transcript] == [
        (0, "agent"), (1, "tester"), (2, "agent"), (3, "tester"), (4, "agent"),
    ]
    assert transcript[0]["content"] == "Hi, this is Maya from the clinic. Is now a good time?"
    assert transcript[1]["content"] == "Caller line 1."


async def test_flow_scenario_fails_cleanly_if_the_opening_turn_cannot_be_fetched(fake_db):
    async def broken(agent, session_key):
        raise RuntimeError("connection refused")

    agent = _Agent([])
    conv = await _run(_flow_path_scenario(8), agent, greeting=broken)
    assert agent.sent == []
    assert fake_db.messages[conv][0]["content"] == runner._AGENT_ERROR_SENTINEL


@pytest.fixture
def native_ws(monkeypatch):
    nws._SESSIONS.clear()
    yield
    nws._SESSIONS.clear()


async def test_flow_scenario_over_the_real_native_ws_transport(fake_db, monkeypatch, native_ws):
    """The existing greeting handling (voice_caller.peek_opening_greeting ->
    voice_native_ws) and the existing turn transport play a 7-turn flow script on ONE
    session, agent first, every line verbatim."""
    frames = [turn_frame("agent", "Hi, I'm Maya.")]
    for i in range(7):
        frames += [turn_frame("caller", f"Caller line {i + 1}."), turn_frame("agent", f"Agent turn {i + 1}.")]
    ws = FakeWebSocket(incoming=frames)
    connect = FakeConnect([ws])
    calls_log: list[dict] = []
    monkeypatch.setattr(nws.websockets, "connect", connect)
    monkeypatch.setattr(nws.httpx, "AsyncClient", FakeAsyncClient(calls_log).factory())

    conv = await runner.run_scenario(
        run_id=1, scenario=_flow_path_scenario(7),
        agent={"voice_protocol": "native_ws", "endpoint_url": "http://localhost:8000"},
        send_fn=voice_caller.call_voice_agent, close_fn=voice_caller.close_voice_session,
        greeting_fn=voice_caller.peek_opening_greeting,
    )

    transcript = fake_db.messages[conv]
    assert transcript[0] == {"turn_index": 0, "role": "agent", "content": "Hi, I'm Maya.", "trace": None}
    assert [m["content"] for m in transcript if m["role"] == "tester"] == [f"Caller line {i + 1}." for i in range(7)]
    assert [m["content"] for m in transcript[2::2]] == [f"Agent turn {i + 1}." for i in range(7)]
    sent = [__import__("json").loads(f)["text"] for f in ws.sent]
    assert sent == [f"Caller line {i + 1}." for i in range(7)]
    assert connect.call_count == 1 and ws.closed is True


# ---------------------------------------------------------------------------
# path_script_to_scenario: the compatibility reshape
# ---------------------------------------------------------------------------
def test_saved_script_becomes_a_scripted_scenario_for_the_existing_pipeline():
    path = ["greeting", "ask_name", "validate_name", "retry_name"]
    turns = [
        {"step": 1, "node_id": "greeting", "expected_agent_behavior": "Greets.", "caller_line": "Hi."},
        {"step": 4, "node_id": "retry_name", "expected_agent_behavior": "Asks again.", "caller_line": "It's Srija."},
    ]
    names = {"greeting": "Greeting", "ask_name": "Ask Name", "validate_name": "Validate Name", "retry_name": "Ask Name Again"}
    s = path_script_to_scenario(7, "sid", "Branch Scenario", path, names, "Verify the retry.", turns)
    assert s["test_type"] == PATH_TEST_TYPE == "flow_path"
    assert s["seed_turns"] == ["Hi.", "It's Srija."]
    assert s["customer_context"] == ""  # scripted, never the AI Caller
    assert s["user_goal"] == "Verify the retry."
    assert s["assigned_fault"] == "none"
    assert s["expected_behavior"].splitlines() == [
        "Planned path: Greeting -> Ask Name -> Validate Name -> Ask Name Again",
        "1. At step 1 (Greeting): Greets.",
        "2. At step 4 (Ask Name Again): Asks again.",
    ]
    assert s["flow_id"] == 7 and s["node_id"] is None
    assert __import__("json").loads(s["node_script_json"]) == {"scenario_id": "sid", "path": path, "turns": turns}


def test_flow_path_survives_scenario_normalization():
    from app.core.scenarios import normalize_scenarios

    assert normalize_scenarios([{"test_type": "flow_path", "seed_turns": ["x"]}])[0]["test_type"] == "flow_path"
    assert "flow_path" not in runner.ADAPTIVE_TYPES
