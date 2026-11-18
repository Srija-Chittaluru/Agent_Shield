"""The Judge and the fixer are told what the agent under test does (its saved
description), so they judge and fix it against ITS purpose — never assuming it is a
customer-support agent. DB and LLM are faked.
"""
import json

from app.core import fixer, judge

_SCENARIO = {"id": 1, "title": "t", "user_goal": "Move $50 to savings", "test_type": "support",
             "assigned_fault": "none", "expected_behavior": "Confirms the transfer."}
_MSGS = [
    {"turn_index": 0, "role": "tester", "content": "Move $50 to savings", "trace_json": None},
    {"turn_index": 1, "role": "agent", "content": "Done.", "trace_json": json.dumps({})},
]
_VERDICT = {"scores": {"accuracy": 0.9, "safety": 1.0, "hallucination": 1.0, "recovery": None},
            "verdict": "pass", "severity": "low", "recovered": None, "fail_category": None,
            "evidence": "agent: Done."}


def _fake_agent(monkeypatch, description):
    monkeypatch.setattr(judge, "get_run", lambda rid: {"id": rid, "agent_id": 3})
    monkeypatch.setattr(judge, "get_agent", lambda aid: {"id": aid, "description": description})


def _fake_llm(monkeypatch, module, reply) -> list:
    calls = []

    async def chat(system, messages, **kw):
        calls.append({"system": system, "user": messages[0]["content"]})
        return reply

    monkeypatch.setattr(module, "chat", chat)
    return calls


async def _judge_prompt(monkeypatch, description) -> dict:
    _fake_agent(monkeypatch, description)
    monkeypatch.setattr(judge, "get_scenario", lambda sid: _SCENARIO)
    monkeypatch.setattr(judge, "get_messages", lambda cid: _MSGS)
    monkeypatch.setattr(judge, "update_conversation_verdict", lambda *a: None)
    calls = _fake_llm(monkeypatch, judge, _VERDICT)
    await judge.judge_conversation({"id": 5, "scenario_id": 1, "run_id": 9})
    return calls[0]


async def test_judge_is_told_what_the_agent_does(monkeypatch):
    prompt = await _judge_prompt(monkeypatch, "A retail-banking assistant for transfers.")

    assert prompt["user"].startswith("AGENT UNDER TEST\n  A retail-banking assistant for transfers.\n\nSCENARIO")
    assert "customer-support" not in prompt["system"]


async def test_judge_prompt_is_unchanged_for_an_agent_without_a_description(monkeypatch):
    prompt = await _judge_prompt(monkeypatch, None)

    assert prompt["user"].startswith("SCENARIO")


async def test_long_description_is_capped(monkeypatch):
    prompt = await _judge_prompt(monkeypatch, "x" * 5000)

    assert "x" * judge.AGENT_DESCRIPTION_MAX_CHARS + "\n" in prompt["user"]
    assert "x" * (judge.AGENT_DESCRIPTION_MAX_CHARS + 1) not in prompt["user"]


async def test_fixer_is_told_what_the_agent_does(monkeypatch):
    _fake_agent(monkeypatch, "An HR assistant that books leave.")
    monkeypatch.setattr(fixer, "get_scenario", lambda sid: _SCENARIO)
    monkeypatch.setattr(fixer, "get_messages", lambda cid: _MSGS)
    monkeypatch.setattr(fixer, "update_conversation_fix", lambda *a: None)
    calls = _fake_llm(monkeypatch, fixer, {"explanation": "e", "suggested_fix": "f", "evidence": "x"})

    await fixer.explain_and_fix({"id": 5, "scenario_id": 1, "run_id": 9, "severity": "high"})

    assert calls[0]["user"].startswith("AGENT UNDER TEST\n  An HR assistant that books leave.\n\nSCENARIO")
    assert "customer-support" not in calls[0]["system"]
