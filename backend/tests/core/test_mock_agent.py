"""Dry run: an agent with kind="mock" has no endpoint — app.core.adapter.send() hands each
turn to an LLM that plays it from its system prompt. The LLM is faked; no network.
"""
from app.core import adapter, judge, mock_agent, voice_caller
from app.core.adapter import AGENT_ERROR_SENTINEL

_MOCK = {"id": 7, "kind": "mock", "endpoint_url": "", "modality": "chat",
         "system_prompt": "You are Acme Bank's assistant.", "description": "Banking help.",
         "knowledge": "Overdraft fee: $25.", "knowledge_name": "fees.md"}


def _fake_llm(monkeypatch, reply="Sure, happy to help.") -> list:
    calls = []

    async def chat(system, messages, **kw):
        calls.append({"system": system, "messages": messages})
        return reply

    monkeypatch.setattr(mock_agent, "chat", chat)
    return calls


async def test_mock_agent_is_played_by_the_llm_not_called_over_http(monkeypatch):
    calls = _fake_llm(monkeypatch)
    history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]

    result = await adapter.send(_MOCK, "What is the overdraft fee?", history, [])

    assert result["reply"] == "Sure, happy to help."
    assert result["trace"]["simulated"] is True
    assert calls[0]["system"].startswith("You are Acme Bank's assistant.")
    assert "Overdraft fee: $25." in calls[0]["system"]
    assert calls[0]["messages"] == [*history, {"role": "user", "content": "What is the overdraft fee?"}]


async def test_without_a_system_prompt_the_description_is_used(monkeypatch):
    calls = _fake_llm(monkeypatch)
    await adapter.send({**_MOCK, "system_prompt": None}, "hi", [], [])
    assert "Banking help." in calls[0]["system"]


async def test_tool_timeout_withholds_the_knowledge(monkeypatch):
    calls = _fake_llm(monkeypatch)
    result = await adapter.send(_MOCK, "fee?", [], ["tool_timeout"])

    assert "Overdraft fee: $25." not in calls[0]["system"]
    assert result["trace"]["tool_calls"][0]["ok"] is False


async def test_stale_doc_marks_the_knowledge_outdated(monkeypatch):
    calls = _fake_llm(monkeypatch)
    result = await adapter.send(_MOCK, "fee?", [], ["stale_doc"])

    assert "OUTDATED" in calls[0]["system"]
    assert result["trace"]["retrieved_docs"][0]["stale"] is True


async def test_system_faults_still_fail_before_the_stand_in_is_asked(monkeypatch):
    calls = _fake_llm(monkeypatch)
    result = await adapter.send(_MOCK, "hi", [], ["api_timeout"])

    assert result["reply"] == AGENT_ERROR_SENTINEL
    assert calls == []


async def test_an_llm_failure_is_an_agent_error_not_a_crash(monkeypatch):
    async def broken(*a, **kw):
        raise RuntimeError("rate limited")

    monkeypatch.setattr(mock_agent, "chat", broken)
    result = await adapter.send(_MOCK, "hi", [], [])

    assert result["reply"] == AGENT_ERROR_SENTINEL
    assert "rate limited" in result["trace"]["error"]


async def test_a_voice_mock_answers_in_text_without_placing_a_call(monkeypatch):
    calls = _fake_llm(monkeypatch)
    result = await voice_caller.call_voice_agent(
        {**_MOCK, "modality": "voice"}, "hi", [], [], session_key=1, is_last_turn=True,
    )

    assert result["reply"] == "Sure, happy to help."
    assert "phone call" in calls[0]["system"]


def test_the_fixer_is_told_to_fix_the_system_prompt(monkeypatch):
    monkeypatch.setattr(judge, "get_run", lambda rid: {"id": rid, "agent_id": 7})
    monkeypatch.setattr(judge, "get_agent", lambda aid: _MOCK)

    block = judge._agent_block({"id": 1, "run_id": 9})

    assert "DRY RUN" in block
    assert "You are Acme Bank's assistant." in block
