"""Auto-discovery — when the user gives no docs and no description, AgentShield figures out
what the agent does by probing it with a few neutral questions and summarizing the replies.
"""
from typing import Any

from app.core.adapter import AGENT_ERROR_SENTINEL, send
from app.core.llm import chat

# Neutral, domain-agnostic probes. We only look at HOW/WHAT the agent answers.
PROBES = [
    "Hi! What can you help me with?",
    "What topics or questions do you handle?",
    "Who are you and what is your role?",
]


def looks_generic(description: str | None) -> bool:
    """True if the stored description tells us nothing useful about the domain."""
    if not description or not description.strip():
        return True
    d = description.strip().lower()
    return d.endswith("agent under test.") or len(d) < 25


async def discover_agent(agent: Any) -> str:
    """Probe the agent and infer a 1-2 sentence description of its domain + capabilities.

    Falls back to a safe generic description if probing or the LLM summary fails.
    """
    replies: list[str] = []
    history: list[dict] = []
    for probe in PROBES:
        try:
            r = await send(agent, probe, history, [])
            reply = r.get("reply", "")
            if reply and reply != AGENT_ERROR_SENTINEL:
                replies.append(f"Q: {probe}\nA: {reply}")
                history.append({"role": "user", "content": probe})
                history.append({"role": "assistant", "content": reply})
        except Exception:
            continue

    if not replies:
        return "A conversational AI assistant (domain could not be auto-detected)."

    transcript = "\n\n".join(replies)[:4000]
    try:
        result = await chat(
            system=(
                "You infer what an AI agent does from a short sample of its own answers. "
                "Respond in json as {\"description\": \"...\"}. The description must be 1-2 "
                "sentences naming the agent's DOMAIN and the main topics/capabilities it handles "
                "(e.g. 'A retail banking assistant handling transfers, fees, fraud, and loans'). "
                "Base it strictly on the sample; if unclear, say so."
            ),
            messages=[{"role": "user", "content": f"Agent's answers to neutral questions:\n{transcript}\n\nInfer its description as json."}],
            json_mode=True,
        )
        desc = result.get("description") if isinstance(result, dict) else None
        if desc and isinstance(desc, str) and desc.strip():
            return desc.strip()[:400]
    except Exception:
        pass
    return "A conversational AI assistant (domain auto-detected from its replies)."
