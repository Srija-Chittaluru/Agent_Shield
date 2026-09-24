"""Regression tests: a flow-node SCRIPTED voice scenario (scenario["test_type"] ==
"flow_node" — see app.core.node_script) must wait for the Voice Agent's unprompted
opening greeting before sending its first scripted caller line, exactly like
app.core.runner._run_dynamic already does for the AI Caller — see runner.py's
_run_scripted docstring for why.

Same faking approach as test_run_scenario_dynamic.py / test_run_scenario_native_ws.py:
app.db is faked in-memory, greeting_fn/send_fn are scripted test doubles (never a real
LLM or network), and the genuine run_scenario()/_run_scripted() code is exercised
unmodified.
"""
import pytest

from app.core import runner


class _FakeDB:
    """Conversations keyed by an incrementing id — mirrors the other run_scenario
    integration tests' fake exactly."""

    def __init__(self):
        self.messages: dict[int, list[dict]] = {}
        self._next_id = 1
        self._by_key: dict[tuple, int] = {}

    def get_or_create_conversation(self, run_id, scenario_id, idem_key=None):
        key = idem_key or (run_id, scenario_id, self._next_id)
        if key in self._by_key:
            return self._by_key[key]
        cid = self._next_id
        self._next_id += 1
        self._by_key[key] = cid
        self.messages[cid] = []
        return cid

    def clear_messages(self, conversation_id):
        n = len(self.messages.get(conversation_id, []))
        self.messages[conversation_id] = []
        return n

    def insert_message(self, conversation_id, turn_index, role, content, trace):
        self.messages[conversation_id].append(
            {"turn_index": turn_index, "role": role, "content": content, "trace": trace}
        )
        return len(self.messages[conversation_id])


@pytest.fixture()
def fake_db(monkeypatch):
    db = _FakeDB()
    monkeypatch.setattr(runner, "get_or_create_conversation", db.get_or_create_conversation)
    monkeypatch.setattr(runner, "clear_messages", db.clear_messages)
    monkeypatch.setattr(runner, "insert_message", db.insert_message)
    return db


class _FakeSendAgent:
    """A scripted send_fn standing in for a voice transport: returns one reply per
    call, in order, and records every call it received — same shape
    test_run_scenario_dynamic.py's _FakeSendAgent uses."""

    def __init__(self, replies: list[str]):
        self._replies = list(replies)
        self.calls: list[dict] = []

    async def __call__(self, agent, message, history, faults, session_key=None, is_last_turn=False):
        self.calls.append({
            "message": message, "history": list(history), "faults": list(faults),
            "session_key": session_key, "is_last_turn": is_last_turn,
        })
        reply = self._replies.pop(0) if self._replies else runner._AGENT_ERROR_SENTINEL
        return {"reply": reply, "trace": {}}


def _flow_node_scenario(**overrides) -> dict:
    """Mirrors app.core.node_script.to_scenario_dict's output shape exactly enough
    for run_scenario() to see a real flow-node scenario."""
    base = {
        "_id": 1,
        "title": "Flow node: ask_for_email",
        "user_goal": "Verify the agent asks for the caller's email address.",
        "test_type": runner.NODE_TEST_TYPE,
        "assigned_fault": "none",
        "expected_behavior": "1. Expected behavior: The agent should ask for the caller's email address.",
        "seed_turns": ["I need help with my account."],
        # Empty, deliberately — see to_scenario_dict: forces the scripted path.
        "customer_context": "",
        "max_turns": None,
        "flow_id": 7,
        "node_id": "ask_for_email",
    }
    base.update(overrides)
    return base


async def test_flow_node_scenario_waits_for_greeting_before_first_caller_line(monkeypatch, fake_db):
    """1 & 6: Agent opening -> Caller -> Agent -> Caller -> Agent, correctly ordered
    and persisted exactly as the Judge would read it (turn_index strictly increasing,
    role-tagged)."""
    async def fake_greeting(agent, session_key):
        return "Hi, this is Maya. How can I help you today?"

    send_fn = _FakeSendAgent(["Could you provide your email address?", "Got it, thanks."])
    scenario = _flow_node_scenario(
        seed_turns=["I need help with my account.", "I don't have an email right now."]
    )

    conv_id = await runner.run_scenario(
        run_id=1, scenario=scenario, agent={"voice_protocol": "native_ws"},
        send_fn=send_fn, greeting_fn=fake_greeting,
    )

    transcript = fake_db.messages[conv_id]
    assert [(m["turn_index"], m["role"]) for m in transcript] == [
        (0, "agent"), (1, "tester"), (2, "agent"), (3, "tester"), (4, "agent"),
    ]
    assert transcript[0]["content"] == "Hi, this is Maya. How can I help you today?"
    assert transcript[1]["content"] == "I need help with my account."
    assert transcript[2]["content"] == "Could you provide your email address?"
    assert transcript[3]["content"] == "I don't have an email right now."
    assert transcript[4]["content"] == "Got it, thanks."


async def test_flow_node_caller_lines_match_the_saved_script_exactly(monkeypatch, fake_db):
    """2: the exact seed_turns text is sent, verbatim, in order — never anything
    generated by the AI Caller."""
    def boom(*a, **kw):
        raise AssertionError("ai_caller must not be called for a flow-node scripted scenario")

    monkeypatch.setattr(runner, "next_utterance", boom)

    async def fake_greeting(agent, session_key):
        return "Hi, this is Maya."

    send_fn = _FakeSendAgent(["reply one", "reply two"])
    scenario = _flow_node_scenario(seed_turns=["line one, exactly as scripted", "line two, exactly as scripted"])

    await runner.run_scenario(
        run_id=1, scenario=scenario, agent={"voice_protocol": "native_ws"},
        send_fn=send_fn, greeting_fn=fake_greeting,
    )

    assert [c["message"] for c in send_fn.calls] == [
        "line one, exactly as scripted", "line two, exactly as scripted",
    ]


async def test_flow_node_agent_replies_are_real_not_expected_behavior_text(monkeypatch, fake_db):
    """3: the recorded agent turns are whatever send_fn actually returned — never the
    scenario's expected_behavior text (that stays evaluation-only, for the Judge)."""
    async def fake_greeting(agent, session_key):
        return "Hi, this is Maya."

    send_fn = _FakeSendAgent(["Sure — could you give me your email address?"])
    scenario = _flow_node_scenario(
        seed_turns=["I need help."],
        expected_behavior="1. Expected behavior: The agent should ask for the caller's email address.",
    )

    conv_id = await runner.run_scenario(
        run_id=1, scenario=scenario, agent={"voice_protocol": "native_ws"},
        send_fn=send_fn, greeting_fn=fake_greeting,
    )

    agent_lines = [m["content"] for m in fake_db.messages[conv_id] if m["role"] == "agent"]
    assert "Sure — could you give me your email address?" in agent_lines
    assert not any("Expected behavior" in line for line in agent_lines)


async def test_non_flow_node_scripted_scenario_is_unaffected(monkeypatch, fake_db):
    """5: the fix is scoped to test_type == "flow_node" only — any other scripted
    test_type keeps opening with the caller's first line exactly as before, even when
    a greeting_fn is available (mirrors test_run_scenario_native_ws.py's existing,
    still-passing "support" scenario)."""
    async def fake_greeting(agent, session_key):
        return "Hi, this is Maya."

    send_fn = _FakeSendAgent(["ok"])
    scenario = {
        "_id": 9, "test_type": "support", "assigned_fault": "none",
        "seed_turns": ["hello"],
    }

    conv_id = await runner.run_scenario(
        run_id=1, scenario=scenario, agent={"voice_protocol": "native_ws"},
        send_fn=send_fn, greeting_fn=fake_greeting,
    )

    transcript = fake_db.messages[conv_id]
    assert transcript[0]["role"] == "tester"
    assert transcript[0]["content"] == "hello"


async def test_flow_node_scenario_opens_cold_when_no_greeting_is_available(monkeypatch, fake_db):
    """Backward-compatible fallback: a protocol with no opening-greeting concept
    (greeting_fn returns None/"" rather than raising) still plays the script normally,
    same as it always has."""
    async def fake_greeting(agent, session_key):
        return None

    send_fn = _FakeSendAgent(["ok"])
    scenario = _flow_node_scenario(seed_turns=["I need help."])

    conv_id = await runner.run_scenario(
        run_id=1, scenario=scenario, agent={"voice_protocol": "http_json"},
        send_fn=send_fn, greeting_fn=fake_greeting,
    )

    transcript = fake_db.messages[conv_id]
    assert transcript[0]["role"] == "tester"
    assert transcript[0]["content"] == "I need help."


async def test_flow_node_scenario_fails_cleanly_when_the_opening_turn_cannot_be_fetched(monkeypatch, fake_db):
    """7: if the transport fails while establishing/draining the opening turn, the
    scenario must record a clean failure and stop — never blindly send the first
    scripted caller line into a session that may not even exist."""
    async def fake_greeting(agent, session_key):
        raise RuntimeError("native_ws call setup error: connection refused")

    send_fn = _FakeSendAgent(["should never be reached"])
    scenario = _flow_node_scenario(seed_turns=["I need help with my account."])

    conv_id = await runner.run_scenario(
        run_id=1, scenario=scenario, agent={"voice_protocol": "native_ws"},
        send_fn=send_fn, greeting_fn=fake_greeting,
    )

    transcript = fake_db.messages[conv_id]
    assert len(transcript) == 1
    assert transcript[0]["role"] == "agent"
    assert transcript[0]["content"] == runner._AGENT_ERROR_SENTINEL
    assert send_fn.calls == []  # the caller line was never sent
