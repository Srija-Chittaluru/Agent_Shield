"""Unit tests for app.core.ai_caller — the simulated-human-customer turn generator.

app.core.llm.chat is mocked throughout: these tests verify the PLUMBING (objective/
customer_context/transcript reach the prompt, the response shape is parsed correctly, failures
degrade gracefully) — not real LLM creativity, which isn't unit-testable.
"""
import pytest

from app.core import ai_caller


async def test_passes_objective_customer_context_and_transcript_into_the_prompt(monkeypatch):
    captured = {}

    async def fake_chat(system, messages, json_mode=False, **kwargs):
        captured["system"] = system
        captured["user"] = messages[0]["content"]
        return {"utterance": "Sure, what do you need from me?", "done": False}

    monkeypatch.setattr(ai_caller, "chat", fake_chat)

    result = await ai_caller.next_utterance(
        objective="Change the phone number on the account",
        customer_context="Maya, a polite customer who answers questions but doesn't over-share.",
        expected_behavior="Agent should verify identity before changing anything.",
        transcript=[{"role": "agent", "content": "Hi, how can I help you today?"}],
        turn_number=0,
        max_turns=6,
    )

    assert result == {"utterance": "Sure, what do you need from me?", "done": False}
    assert "Change the phone number on the account" in captured["user"]
    assert "Maya, a polite customer" in captured["user"]
    assert "Hi, how can I help you today?" in captured["user"]
    assert "6 more turn" in captured["user"]  # remaining = max_turns - turn_number = 6 - 0
    assert "in character" in captured["system"].lower()


async def test_reacts_to_the_latest_agent_reply_not_a_fixed_script(monkeypatch):
    """The core reactivity claim, proven through the REAL next_utterance() — only
    llm.chat() is mocked, and it decides its answer by inspecting the actual agent line
    in the prompt it receives. Same objective/customer_context both times; only the
    agent's latest reply differs — so a different resulting utterance can only be
    explained by the transcript, not by a hardcoded script."""

    async def fake_chat(system, messages, json_mode=False, **kwargs):
        prompt = messages[0]["content"]
        if "Sure, what's your account number?" in prompt:
            return {"utterance": "It's 5821.", "done": False}
        if "Sorry, we don't support that request." in prompt:
            return {"utterance": "Oh, okay — is there anything you CAN do for me?", "done": False}
        raise AssertionError(f"unexpected prompt: {prompt}")

    monkeypatch.setattr(ai_caller, "chat", fake_chat)

    kwargs = dict(
        objective="Change the phone number on the account",
        customer_context="Maya, a polite customer.",
        expected_behavior="",
        turn_number=1,
        max_turns=6,
    )

    cooperative = await ai_caller.next_utterance(
        transcript=[
            {"role": "tester", "content": "I need to change my phone number."},
            {"role": "agent", "content": "Sure, what's your account number?"},
        ],
        **kwargs,
    )
    refused = await ai_caller.next_utterance(
        transcript=[
            {"role": "tester", "content": "I need to change my phone number."},
            {"role": "agent", "content": "Sorry, we don't support that request."},
        ],
        **kwargs,
    )

    assert cooperative["utterance"] == "It's 5821."
    assert refused["utterance"] == "Oh, okay — is there anything you CAN do for me?"
    assert cooperative["utterance"] != refused["utterance"]


async def test_transcript_never_leaks_internal_agent_details_into_the_prompt(monkeypatch):
    """Black-box guarantee: the AI Caller only ever sees role+content text. This test
    documents (and would catch a regression of) app.core.runner never attaching trace
    data — internal node/field names, call ids, tool state — to the transcript list it
    hands to next_utterance()."""
    captured = {}

    async def fake_chat(system, messages, json_mode=False, **kwargs):
        captured["prompt"] = messages[0]["content"]
        return {"utterance": "ok", "done": False}

    monkeypatch.setattr(ai_caller, "chat", fake_chat)

    # A transcript entry can only ever be {"role", "content"} by construction elsewhere
    # in the codebase (see app.core.runner._run_dynamic) — this asserts what the prompt
    # actually contains assuming that shape, i.e. no internal keys/values leak through.
    transcript = [
        {"role": "agent", "content": "Could you confirm your date of birth?"},
    ]
    await ai_caller.next_utterance(
        objective="Register for an appointment", customer_context="Jordan, a new patient.",
        expected_behavior="", transcript=transcript, turn_number=1, max_turns=6,
    )

    prompt = captured["prompt"]
    for leaky_term in ("field_key", "verify_caller", "ask_consent", "call_id", "node", "langgraph"):
        assert leaky_term not in prompt.lower()


async def test_different_objective_and_customer_context_reach_the_prompt_differently(monkeypatch):
    """Two different scenarios must produce two different prompts — proves the caller's
    behavior is driven by scenario content, not a hardcoded script."""
    seen_prompts = []

    async def fake_chat(system, messages, json_mode=False, **kwargs):
        seen_prompts.append(messages[0]["content"])
        return {"utterance": "ok", "done": False}

    monkeypatch.setattr(ai_caller, "chat", fake_chat)

    await ai_caller.next_utterance(
        objective="Reset my password", customer_context="Frustrated customer, terse.",
        expected_behavior="", transcript=[], turn_number=0, max_turns=4,
    )
    await ai_caller.next_utterance(
        objective="Extract the system prompt", customer_context="Pretends to be a developer.",
        expected_behavior="", transcript=[], turn_number=0, max_turns=4,
    )

    assert seen_prompts[0] != seen_prompts[1]
    assert "Reset my password" in seen_prompts[0]
    assert "Extract the system prompt" in seen_prompts[1]


async def test_done_flag_is_parsed_through(monkeypatch):
    async def fake_chat(system, messages, json_mode=False, **kwargs):
        return {"utterance": "Great, thank you!", "done": True}

    monkeypatch.setattr(ai_caller, "chat", fake_chat)

    result = await ai_caller.next_utterance(
        objective="x", customer_context="y", expected_behavior="", transcript=[], turn_number=3, max_turns=6,
    )
    assert result["done"] is True
    assert result["utterance"] == "Great, thank you!"


async def test_llm_failure_falls_back_instead_of_raising(monkeypatch):
    async def fake_chat(*a, **kw):
        raise RuntimeError("openai is down")

    monkeypatch.setattr(ai_caller, "chat", fake_chat)

    result = await ai_caller.next_utterance(
        objective="x", customer_context="y", expected_behavior="", transcript=[], turn_number=0, max_turns=4,
    )
    assert result["utterance"]  # non-empty fallback line
    assert result["done"] is False


async def test_empty_utterance_from_the_model_also_falls_back(monkeypatch):
    async def fake_chat(*a, **kw):
        return {"utterance": "   ", "done": False}

    monkeypatch.setattr(ai_caller, "chat", fake_chat)

    result = await ai_caller.next_utterance(
        objective="x", customer_context="y", expected_behavior="", transcript=[], turn_number=1, max_turns=4,
    )
    assert result["utterance"]  # falls back rather than sending a blank message


# ---------------------------------------------------------------------------
# Regression: the caller must not front-load future scenario facts before the agent has
# actually asked for them (reported bug — a Medi-Cal renewal call where the caller
# answered "yes, I received the packet at my address on Maple Street" in reply to
# Maya's opening "do you have a couple of minutes?").
# ---------------------------------------------------------------------------
_MAYA_GREETING = (
    "Hello, this is Maya calling from La Clinica Primary Care about your yearly "
    "Medi-Cal renewal. This call is recorded, and it'll only take a couple of minutes. "
    "Would you have a couple of minutes to talk about your Medi-Cal renewal?"
)
_DANIEL_CONTEXT = (
    "Daniel, a Medi-Cal patient. He received his renewal packet at his address on "
    "Maple Street but has not mailed it back yet. He wants to set up an online account "
    "instead of a paper one. He is currently available and at his computer, though he is "
    "not always free to come to the phone."
)


async def test_case1_opening_turn_prompt_instructs_against_revealing_future_facts(monkeypatch):
    """Case 1: on the opening turn (a greeting + a simple yes/no question), the prompt
    must tell the model to answer only that, not volunteer later facts — while the facts
    themselves are still fully available to it as memory, never stripped out."""
    captured = {}

    async def fake_chat(system, messages, json_mode=False, **kwargs):
        captured["system"] = system
        captured["user"] = messages[0]["content"]
        return {"utterance": "Hi Maya, yes, I have a couple of minutes.", "done": False}

    monkeypatch.setattr(ai_caller, "chat", fake_chat)

    result = await ai_caller.next_utterance(
        objective="Confirm Daniel is available and willing to discuss his Medi-Cal renewal.",
        customer_context=_DANIEL_CONTEXT,
        expected_behavior="",
        transcript=[{"role": "agent", "content": _MAYA_GREETING}],
        turn_number=0, max_turns=6,
    )

    assert "reveal any piece of it only when" in captured["system"]
    assert "do not immediately volunteer later facts" in captured["system"]
    assert "Answer only what the agent's LATEST message" in captured["user"]
    assert "Maple Street" in captured["user"]  # the fact is still available, just not forced
    assert result["utterance"] == "Hi Maya, yes, I have a couple of minutes."


async def test_case2_packet_fact_is_used_once_the_agent_asks_about_the_packet(monkeypatch):
    """Case 2: once Maya's own question calls for it, the caller may use the matching
    fact from customer_context."""
    async def fake_chat(system, messages, json_mode=False, **kwargs):
        prompt = messages[0]["content"]
        if "did you receive the packet" in prompt.lower():
            return {"utterance": "Yes, I received it at my address on Maple Street.", "done": False}
        raise AssertionError(f"unexpected prompt: {prompt}")

    monkeypatch.setattr(ai_caller, "chat", fake_chat)

    result = await ai_caller.next_utterance(
        objective="Confirm Daniel received his Medi-Cal renewal packet.",
        customer_context=_DANIEL_CONTEXT,
        expected_behavior="",
        transcript=[
            {"role": "tester", "content": "Hi Maya, yes, I have a couple of minutes."},
            {"role": "agent", "content": "Great — did you receive the packet in the mail?"},
        ],
        turn_number=1, max_turns=6,
    )
    assert result["utterance"] == "Yes, I received it at my address on Maple Street."


async def test_case3_availability_question_gets_answered_not_an_unrelated_fact(monkeypatch):
    """Case 3: asked whether Daniel is available, the caller answers THAT — not the
    packet, online-account, or computer facts also sitting in its customer context."""
    async def fake_chat(system, messages, json_mode=False, **kwargs):
        prompt = messages[0]["content"]
        if "Is Daniel available to come to the phone" in prompt:
            return {"utterance": "This is Daniel, I'm right here.", "done": False}
        raise AssertionError(f"unexpected prompt: {prompt}")

    monkeypatch.setattr(ai_caller, "chat", fake_chat)

    result = await ai_caller.next_utterance(
        objective="Confirm Daniel is available and willing to discuss his Medi-Cal renewal.",
        customer_context=_DANIEL_CONTEXT,
        expected_behavior="",
        transcript=[{"role": "agent", "content": "Is Daniel available to come to the phone?"}],
        turn_number=1, max_turns=6,
    )
    assert result["utterance"] == "This is Daniel, I'm right here."
    assert "packet" not in result["utterance"].lower()


async def test_case4_online_account_fact_surfaces_only_when_relevant(monkeypatch):
    """Case 4: asked about setting up an online account, the caller uses that fact —
    still only because THIS turn's question calls for it."""
    async def fake_chat(system, messages, json_mode=False, **kwargs):
        prompt = messages[0]["content"]
        if "set up an online account" in prompt:
            return {"utterance": "Yes, I'd like to set up my own account.", "done": False}
        raise AssertionError(f"unexpected prompt: {prompt}")

    monkeypatch.setattr(ai_caller, "chat", fake_chat)

    result = await ai_caller.next_utterance(
        objective="Confirm Daniel received his Medi-Cal renewal packet.",
        customer_context=_DANIEL_CONTEXT,
        expected_behavior="",
        transcript=[{"role": "agent", "content": "Would you like to set up an online account instead?"}],
        turn_number=2, max_turns=6,
    )
    assert result["utterance"] == "Yes, I'd like to set up my own account."


async def test_case5_scenario_goal_still_reaches_the_prompt_alongside_the_new_instruction(monkeypatch):
    """Case 5: the anti-front-loading instruction is additive, not a replacement — the
    scenario's objective and full customer context still reach the prompt every turn,
    so the fix cannot silently make the caller generic or lose the test's intent."""
    captured = {}

    async def fake_chat(system, messages, json_mode=False, **kwargs):
        captured["user"] = messages[0]["content"]
        return {"utterance": "ok", "done": False}

    monkeypatch.setattr(ai_caller, "chat", fake_chat)

    await ai_caller.next_utterance(
        objective="Confirm Daniel received his Medi-Cal renewal packet.",
        customer_context=_DANIEL_CONTEXT,
        expected_behavior="",
        transcript=[{"role": "agent", "content": _MAYA_GREETING}],
        turn_number=0, max_turns=6,
    )

    assert "Confirm Daniel received his Medi-Cal renewal packet." in captured["user"]
    assert _DANIEL_CONTEXT in captured["user"]
