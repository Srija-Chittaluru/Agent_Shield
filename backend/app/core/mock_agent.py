"""Dry run — an LLM stands in for an agent that has no endpoint (agents.kind = "mock").

The user gives the agent's system prompt (and/or description + knowledge docs) instead of
an endpoint. Each turn, the LLM plays that agent from those instructions, so the whole
pipeline — scenarios, Judge, Explain + Fix, report — runs unchanged and predicts where
the real agent would break before it is ever connected.

Reached only through app.core.adapter.send(), the single place a turn reaches an agent,
so every caller (runs, replay, discovery, probe) gets the stand-in with no changes of its
own. Same return shape as send(): {"reply": str, "trace": dict}.

Agent-cooperative faults are simulated the way the sample RAG agent applies them, so a
dry run exercises the same recovery behaviour:
  tool_timeout -> the knowledge lookup failed; no docs are provided this turn
  stale_doc    -> the knowledge is marked outdated, as a stale retrieval would serve it
  injection    -> the message is passed through unchanged (it is itself the attack)
"""
import time
from typing import Any

from app.core.llm import chat

# Every agent registered for a dry run carries this `agents.kind`.
MOCK_KIND = "mock"

# Cap on the knowledge given to the stand-in each turn, so a large doc never crowds out
# the conversation.
KNOWLEDGE_MAX_CHARS = 6000

# A real agent is not deterministic either; a little temperature keeps the stand-in from
# being unrealistically consistent across runs.
TEMPERATURE = 0.4


def is_mock(agent: Any) -> bool:
    return (agent or {}).get("kind") == MOCK_KIND


def _system_prompt(agent: dict, knowledge: str) -> str:
    instructions = (agent.get("system_prompt") or "").strip()
    if not instructions:
        description = (agent.get("description") or "").strip() or "A conversational AI assistant."
        instructions = f"You are this assistant: {description}"

    parts = [instructions]
    if agent.get("modality") == "voice":
        parts.append(
            "You are speaking on a phone call: reply in short, natural spoken sentences, "
            "with no markdown, lists, links or emojis."
        )
    if knowledge:
        parts.append(f"KNOWLEDGE YOU CAN USE\n{knowledge}")
    return "\n\n".join(parts)


def _knowledge_for_turn(agent: dict, faults: list) -> tuple[str, list[dict], list[dict]]:
    """The knowledge the stand-in sees this turn, plus the trace entries describing it."""
    knowledge = (agent.get("knowledge") or "").strip()[:KNOWLEDGE_MAX_CHARS]
    if not knowledge:
        return "", [], []

    name = agent.get("knowledge_name") or "knowledge"
    if "tool_timeout" in faults:
        tool = {"name": "retrieve_docs", "result": "timeout", "ok": False}
        return "(The knowledge lookup timed out — no documents are available this turn.)", [tool], []

    stale = "stale_doc" in faults
    if stale:
        knowledge = (
            "NOTE: these documents were retrieved from an OUTDATED snapshot and may no "
            f"longer be accurate.\n{knowledge}"
        )
    tool = {"name": "retrieve_docs", "result": "ok", "ok": True}
    return knowledge, [tool], [{"id": "knowledge", "title": name, "stale": stale}]


async def simulated_reply(agent: Any, message: str, history: list, faults: list) -> dict:
    """One turn played by the stand-in. Never raises: an LLM failure becomes an error
    trace, exactly like an unreachable real agent."""
    from app.core.adapter import AGENT_ERROR_SENTINEL

    agent = dict(agent)
    start = time.time()
    knowledge, tool_calls, retrieved_docs = _knowledge_for_turn(agent, faults)
    # The runner already keeps history as {role: user|assistant, content}.
    messages = [*history, {"role": "user", "content": message}]

    try:
        reply = await chat(
            system=_system_prompt(agent, knowledge), messages=messages, temperature=TEMPERATURE,
        )
    except Exception as e:
        return {"reply": AGENT_ERROR_SENTINEL, "trace": {"error": f"simulated agent failed: {e}"}}

    return {
        "reply": str(reply).strip(),
        "trace": {
            "simulated": True,
            "tool_calls": tool_calls,
            "retrieved_docs": retrieved_docs,
            "latency_ms": int((time.time() - start) * 1000),
        },
    }
