"""Integration tests for the DYNAMIC (AI-Caller-driven) run_scenario() loop.

app.core.ai_caller.next_utterance is mocked (scripted per-call), since real LLM
creativity isn't unit-testable — these tests verify the LOOP itself: greeting-first-turn
wiring, reactivity (the transcript passed to the next call reflects the latest agent
reply), termination (done flag / max_turns / transport failure), and that the complete
transcript lands in the DB exactly as app.core.judge would read it. DB calls are faked
in-memory (same pattern as test_run_scenario_native_ws.py) — no real Postgres needed.
"""
import pytest

from app.core import runner


class _FakeDB:
    """Conversations keyed by an incrementing id — enough to fake idempotency for what
    these tests exercise. Mirrors test_run_scenario_native_ws.py's fake exactly."""

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


def _dynamic_scenario(**overrides) -> dict:
    base = {
        "_id": 1,
        "title": "Change phone number",
        "user_goal": "Change the registered phone number",
        "test_type": "support",
        "assigned_fault": "none",
        "expected_behavior": "Agent verifies identity, then updates the number.",
        "customer_context": "Maya, a polite customer who answers questions but volunteers nothing extra.",
        "seed_turns": [],
    }
    base.update(overrides)
    return base


def _scripted_caller(turns: list[dict]):
    """A fake ai_caller.next_utterance that plays back a fixed script of
    {"utterance", "done"} dicts, one per call — lets a test assert exact loop behavior
    without a real LLM, while still exercising the real run_scenario()/_run_dynamic()
    code and its actual call signature."""
    calls: list[dict] = []
    it = iter(turns)

    async def fake(objective, customer_context, expected_behavior, transcript, turn_number, max_turns):
        calls.append({
            "objective": objective, "customer_context": customer_context,
            "transcript": list(transcript), "turn_number": turn_number, "max_turns": max_turns,
        })
        return next(it)

    fake.calls = calls
    return fake


class _FakeSendAgent:
    """A scripted send_fn standing in for a voice/chat transport: returns one reply per
    call, in order (or the error sentinel once exhausted), and records every call."""

    def __init__(self, replies: list[str]):
        self._replies = list(replies)
        self.calls: list[dict] = []

    async def __call__(self, agent, message, history, faults, session_key=None, is_last_turn=False):
        self.calls.append({
            "message": message, "history": list(history), "faults": list(faults),
            "session_key": session_key, "is_last_turn": is_last_turn,
        })
        reply = self._replies.pop(0) if self._replies else runner.AGENT_ERROR_SENTINEL
        return {"reply": reply, "trace": {}}


async def test_agent_speaks_first_and_caller_reacts_to_the_greeting(monkeypatch, fake_db):
    """1 & 2: the greeting is recorded as turn 0, and the AI Caller's first generation
    call receives it in its transcript."""
    caller = _scripted_caller([
        {"utterance": "Hi, yes, I have a minute.", "done": False},
        {"utterance": "Great, thanks!", "done": True},
    ])
    monkeypatch.setattr(runner, "next_utterance", caller)

    async def fake_greeting(agent, session_key):
        return "Hi, this is Maya calling about your registration. Got a minute?"

    send_fn = _FakeSendAgent(["Sure, what can I help with?", "You're welcome!"])

    conv_id = await runner.run_scenario(
        run_id=1, scenario=_dynamic_scenario(), agent={"id": 1},
        send_fn=send_fn, greeting_fn=fake_greeting,
    )

    transcript = fake_db.messages[conv_id]
    assert transcript[0]["role"] == "agent"
    assert transcript[0]["content"] == "Hi, this is Maya calling about your registration. Got a minute?"
    assert caller.calls[0]["turn_number"] == 0
    assert caller.calls[0]["transcript"][0]["content"].startswith("Hi, this is Maya")


async def test_caller_response_reflects_the_agents_latest_reply(monkeypatch, fake_db):
    """3: whatever the agent actually said last is what the NEXT ai_caller call sees —
    the reactivity the whole feature is about, not a fixed script."""
    caller = _scripted_caller([
        {"utterance": "I need to update my phone number.", "done": False},
        {"utterance": "It's 555-1234.", "done": True},
    ])
    monkeypatch.setattr(runner, "next_utterance", caller)
    send_fn = _FakeSendAgent(["Sure — what's the new number?", "Updated. Anything else?"])

    await runner.run_scenario(run_id=1, scenario=_dynamic_scenario(), agent={"id": 1}, send_fn=send_fn)

    second_call_transcript = caller.calls[1]["transcript"]
    assert {"role": "agent", "content": "Sure — what's the new number?"} in second_call_transcript


async def test_multi_turn_conversation_is_fully_recorded(monkeypatch, fake_db):
    """4: several turns, all persisted in order."""
    caller = _scripted_caller([
        {"utterance": "Hi, I need help with X.", "done": False},
        {"utterance": "Here's more detail.", "done": False},
        {"utterance": "Got it, thanks!", "done": True},
    ])
    monkeypatch.setattr(runner, "next_utterance", caller)
    send_fn = _FakeSendAgent(["Sure, tell me more.", "Okay, one sec.", "All done!"])

    conv_id = await runner.run_scenario(
        run_id=1, scenario=_dynamic_scenario(max_turns=8), agent={"id": 1}, send_fn=send_fn,
    )
    transcript = fake_db.messages[conv_id]
    assert [m["role"] for m in transcript] == ["tester", "agent", "tester", "agent", "tester", "agent"]
    assert len(caller.calls) == 3


async def test_different_objectives_and_customer_contexts_reach_the_caller_unmixed(monkeypatch, fake_db):
    """6: run_scenario() passes each scenario's OWN objective/customer_context through untouched —
    the actual "different behavior" is the LLM's job, verified at the prompt level in
    test_ai_caller.py; this confirms the wiring doesn't share/hardcode them."""
    caller = _scripted_caller([{"utterance": "line one", "done": True}])
    monkeypatch.setattr(runner, "next_utterance", caller)
    send_fn = _FakeSendAgent(["ok"])

    await runner.run_scenario(
        run_id=1,
        scenario=_dynamic_scenario(user_goal="Reset a forgotten password", customer_context="Anxious, apologetic."),
        agent={"id": 1}, send_fn=send_fn,
    )
    assert caller.calls[0]["objective"] == "Reset a forgotten password"
    assert caller.calls[0]["customer_context"] == "Anxious, apologetic."


async def test_max_turns_cap_stops_the_conversation(monkeypatch, fake_db):
    """7: the caller never signals done, so the scenario's own max_turns cuts it off."""
    caller = _scripted_caller([{"utterance": f"turn {i}", "done": False} for i in range(20)])
    monkeypatch.setattr(runner, "next_utterance", caller)
    send_fn = _FakeSendAgent([f"reply {i}" for i in range(20)])

    conv_id = await runner.run_scenario(
        run_id=1, scenario=_dynamic_scenario(max_turns=3), agent={"id": 1}, send_fn=send_fn,
    )
    assert len(caller.calls) == 3
    assert len(fake_db.messages[conv_id]) == 6  # 3 tester + 3 agent turns, no more


async def test_default_max_turns_used_when_scenario_sets_none(monkeypatch, fake_db):
    caller = _scripted_caller([{"utterance": f"turn {i}", "done": False} for i in range(20)])
    monkeypatch.setattr(runner, "next_utterance", caller)
    send_fn = _FakeSendAgent([f"reply {i}" for i in range(20)])

    await runner.run_scenario(
        run_id=1, scenario=_dynamic_scenario(max_turns=None), agent={"id": 1}, send_fn=send_fn,
    )
    assert len(caller.calls) == runner.DEFAULT_MAX_TURNS


async def test_conversation_terminates_when_caller_signals_done(monkeypatch, fake_db):
    """8: proper termination — done=True on turn 2 of a max_turns=8 scenario stops the
    loop right there, not at the cap."""
    caller = _scripted_caller([
        {"utterance": "question", "done": False},
        {"utterance": "thanks, that resolves it", "done": True},
    ])
    monkeypatch.setattr(runner, "next_utterance", caller)
    send_fn = _FakeSendAgent(["answer", "you're welcome"])

    conv_id = await runner.run_scenario(
        run_id=1, scenario=_dynamic_scenario(max_turns=8), agent={"id": 1}, send_fn=send_fn,
    )
    assert len(caller.calls) == 2  # stopped after done=True, not at 8
    assert len(fake_db.messages[conv_id]) == 4


async def test_transport_failure_stops_the_dynamic_loop(monkeypatch, fake_db):
    caller = _scripted_caller([{"utterance": f"turn {i}", "done": False} for i in range(20)])
    monkeypatch.setattr(runner, "next_utterance", caller)
    send_fn = _FakeSendAgent([])  # empty -> every call returns the error sentinel

    conv_id = await runner.run_scenario(
        run_id=1, scenario=_dynamic_scenario(max_turns=8), agent={"id": 1}, send_fn=send_fn,
    )
    transcript = fake_db.messages[conv_id]
    assert transcript[-1]["content"] == runner.AGENT_ERROR_SENTINEL
    assert len(caller.calls) == 1  # did not keep generating caller lines against a dead session


async def test_complete_transcript_is_exactly_what_the_judge_would_read(monkeypatch, fake_db):
    """9: run_scenario()'s only durable output IS the messages table — judge.py reads it
    with get_messages(conversation_id), completely unmodified by this feature. Confirm
    the dynamic loop leaves a clean, correctly-ordered, fully role-tagged transcript."""
    caller = _scripted_caller([
        {"utterance": "I need help with X", "done": False},
        {"utterance": "thanks", "done": True},
    ])
    monkeypatch.setattr(runner, "next_utterance", caller)
    send_fn = _FakeSendAgent(["sure, here you go", "anytime"])

    conv_id = await runner.run_scenario(run_id=1, scenario=_dynamic_scenario(), agent={"id": 1}, send_fn=send_fn)
    transcript = fake_db.messages[conv_id]
    assert [(m["turn_index"], m["role"]) for m in transcript] == [
        (0, "tester"), (1, "agent"), (2, "tester"), (3, "agent"),
    ]
    assert transcript[0]["content"] == "I need help with X"
    assert transcript[3]["content"] == "anytime"


async def test_dynamic_scenario_works_over_the_plain_chat_transport_too(monkeypatch, fake_db):
    """Dynamic mode isn't voice-only: a chat scenario (no send_fn override, no
    greeting_fn) opens cold and still conducts a real multi-turn conversation."""
    caller = _scripted_caller([
        {"utterance": "Hi, I need help with my order.", "done": False},
        {"utterance": "Great, thanks!", "done": True},
    ])
    monkeypatch.setattr(runner, "next_utterance", caller)

    async def fake_chat_send(agent, message, history, faults):
        return {"reply": f"How can I help with: {message}", "trace": {}}

    monkeypatch.setattr(runner, "send", fake_chat_send)

    conv_id = await runner.run_scenario(run_id=1, scenario=_dynamic_scenario(), agent={"id": 1})
    transcript = fake_db.messages[conv_id]
    assert transcript[0]["role"] == "tester"  # no greeting_fn -> opens cold, no turn 0 agent line
    assert transcript[0]["content"] == "Hi, I need help with my order."
    assert len(transcript) == 4


async def test_scripted_scenario_without_customer_context_never_touches_ai_caller(monkeypatch, fake_db):
    """10/11 (regression): a scenario with NO customer_context — exactly what every pre-existing
    scenario looks like — must keep running the original scripted path untouched."""
    def boom(*a, **kw):
        raise AssertionError("ai_caller must not be called for a scripted scenario")

    monkeypatch.setattr(runner, "next_utterance", boom)

    async def fake_chat_send(agent, message, history, faults):
        return {"reply": f"echo: {message}", "trace": {}}

    monkeypatch.setattr(runner, "send", fake_chat_send)

    scenario = {
        "_id": 2, "test_type": "support", "assigned_fault": "none",
        "seed_turns": ["Hello, I need help."],
        # no "customer_context" key at all
    }
    conv_id = await runner.run_scenario(run_id=1, scenario=scenario, agent={"id": 1})
    transcript = fake_db.messages[conv_id]
    assert transcript[0]["content"] == "Hello, I need help."
    assert transcript[1]["content"] == "echo: Hello, I need help."


async def test_black_box_agent_trace_never_reaches_the_ai_caller(monkeypatch, fake_db):
    """The AI Caller must simulate the CUSTOMER, never a persona of the agent under
    test — which structurally requires it to never see the agent's internal trace
    (node/field names, call ids, tool state). Rich, LangGraph-shaped trace data is
    attached to every agent reply here; the assertion is that none of it ever appears
    in the transcript _run_dynamic() hands to next_utterance()."""
    caller = _scripted_caller([
        {"utterance": "Hi, I need to register.", "done": False},
        {"utterance": "Sure, that's fine.", "done": True},
    ])
    monkeypatch.setattr(runner, "next_utterance", caller)

    class _RichTraceSend:
        async def __call__(self, agent, message, history, faults, session_key=None, is_last_turn=False):
            return {
                "reply": "Could you confirm your date of birth?",
                "trace": {
                    "call_id": "abc-123",
                    "field_key": "verify_caller",
                    "answers": {"consent": "Yes"},
                    "node": "ask_consent",
                },
            }

    conv_id = await runner.run_scenario(
        run_id=1, scenario=_dynamic_scenario(max_turns=2), agent={"id": 1}, send_fn=_RichTraceSend(),
    )

    # The DB row legitimately keeps the trace (report/debugging need it) ...
    agent_messages = [m for m in fake_db.messages[conv_id] if m["role"] == "agent"]
    assert agent_messages[0]["trace"]["field_key"] == "verify_caller"

    # ... but NONE of it ever reached the AI Caller's transcript argument.
    for call in caller.calls:
        for turn in call["transcript"]:
            assert set(turn.keys()) == {"role", "content"}
            serialized = str(turn).lower()
            for leaky_term in ("call_id", "field_key", "verify_caller", "ask_consent", "node"):
                assert leaky_term not in serialized
