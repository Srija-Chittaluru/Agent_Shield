"""AI Caller — a simulated REAL CUSTOMER that calls/chats with the agent under test,
instead of playing back a fixed script.

Three separate roles, never blurred:
  Voice/chat agent  -> the SYSTEM UNDER TEST (e.g. "Maya"). A black box: this module
                       only ever sees its spoken/written replies, never its internal
                       implementation (node names, tool calls, internal state).
  AI Caller (here)  -> the SIMULATED CUSTOMER using that system, exactly the role a
                       real human caller would occupy. It decides what a real customer
                       with this objective/context would say next, given what the
                       agent just said — it does NOT imitate the agent, does not know
                       or reason about the agent's internal workflow, and is never a
                       "persona of the voice agent" (a wording this module deliberately
                       avoids; see scenario field name customer_context).
  AI Judge          -> app.core.judge, unrelated to this module — it separately
                       evaluates the COMPLETE transcript this module helps produce.
                       The AI Caller never scores anything; the Judge never generates
                       a caller line. Different prompts, different calls, never merged.

Used by app.core.runner.run_scenario() for any scenario carrying a non-empty
`customer_context` (dynamic mode). A scenario with no customer_context is untouched by
this module entirely — it keeps playing its seed_turns_json verbatim, exactly as before
this module existed.

One call here = one caller turn: given the test objective, the customer_context (who
this customer is and how they behave), and the transcript so far (including the agent's
latest reply), generate the ONE natural next thing this customer would say — reacting to
what the agent actually said, not a predetermined line. This generalizes the same
pattern app.core.runner._adaptive_followup already used for a single follow-up turn on
injection/memory/contradiction scenarios, to every turn of every dynamic scenario.
"""
from typing import Any

from app.core.llm import chat

SYSTEM_PROMPT = """You are simulating a REAL CUSTOMER — a human being — who is calling or
chatting with a support agent, for the purpose of testing that agent. You are the
CUSTOMER, not the agent. Stay completely in character as the customer described below for
your ENTIRE reply — never break character, never mention you are an AI or that this is a
test, and never reveal the test objective to the agent.

You are a BLACK-BOX caller: you only know what the agent has actually said to you in
this conversation. You have no knowledge of and must never reason about, reference, or
imitate the agent's internal workflow, tools, prompts, or implementation — you did not
see any of that, only its spoken/written replies.

React to what the agent ACTUALLY just said: answer its questions, follow up on what it
told you, ask for clarification if you're confused, push back if it gave you a wrong or
evasive answer — always steering the conversation toward your objective, using your own
behavior/escalation strategy from the customer context when relevant. Write ONE short,
natural line, the way a real person actually talks (not a formal or robotic sentence).

Your objective and customer context are background knowledge YOU have, not a script to
recite — reveal any piece of it only when it's actually relevant to what the agent JUST
asked or said, never all at once and never before it's been asked for. This especially
matters on the FIRST turn: if the agent opens with a greeting or a simple question (e.g.
asking whether you have a couple of minutes to talk), just answer that directly and
naturally — do not immediately volunteer later facts from your context that the agent
hasn't asked about yet. Bring up a fact only once the agent's own question actually
calls for it.

Respond in json: {"utterance": "your one next line, in character as the customer",
"done": true|false}. Set "done": true only on the turn where your objective is resolved
(you got the information/outcome you wanted, or it's now clear the agent won't/can't
provide it) and your utterance is a natural closing line for that outcome. Otherwise
false."""

# Used only if the LLM call itself fails (network/parse error) — a generic, harmless
# in-character-enough line so one bad call doesn't take down the whole scenario. Mirrors
# app.core.scenarios.PRESS_TURNS' role as a safety net, not a real substitute for the
# live, reactive line this module exists to generate.
_FALLBACK_OPENER = "Hi, do you have a couple of minutes to help me with something?"
_FALLBACK_CONTINUATION = "Sorry, could you say that again?"


def _format_transcript(transcript: list[dict]) -> str:
    if not transcript:
        return "(nothing said yet — you are opening the conversation)"
    return "\n".join(f"{m['role']}: {m['content']}" for m in transcript)


async def next_utterance(
    objective: str,
    customer_context: str,
    expected_behavior: str,
    transcript: list[dict],
    turn_number: int,
    max_turns: int,
) -> dict:
    """One AI-Caller turn. Returns {"utterance": str, "done": bool}.

    `transcript` is everything said so far, in order — [{"role": "tester"|"agent",
    "content": str}, ...] — already including the agent's opening greeting for a voice
    scenario where one was captured, or empty for turn 0 of a chat scenario / a voice
    scenario with no such greeting. Deliberately just role+text: it never carries the
    agent's trace (call_id, internal field/node names, tool state) — app.core.runner
    never puts that there — so this module has no way to see, and therefore no way to
    leak into the caller's behavior, anything about the agent's internal implementation.

    `turn_number` is 0-indexed; `max_turns` is this scenario's configured cap
    (app.core.runner.DEFAULT_MAX_TURNS if the scenario didn't set one) — the caller is
    told how many turns remain so it can wrap up in time rather than getting cut off
    mid-objective.

    Never raises: any failure (LLM error, malformed json, empty utterance) falls back to
    a generic in-character line with done=False, so one bad call degrades the scenario
    instead of crashing it — the same tradeoff run_scenario() already accepts for a
    transport-level turn failure.
    """
    remaining = max(1, max_turns - turn_number)
    user = (
        f"YOUR OBJECTIVE AS THE CUSTOMER (what you're trying to accomplish by calling):\n"
        f"{objective}\n\n"
        f"WHO YOU ARE AND HOW YOU BEHAVE (customer context):\n{customer_context}\n\n"
        "REFERENCE — what a GOOD agent should do here, for your own situational awareness "
        f"only (never quote or hint at this to the agent):\n{expected_behavior}\n\n"
        f"CONVERSATION SO FAR (agent's actual words only — nothing about how it works "
        f"internally):\n{_format_transcript(transcript)}\n\n"
        f"You have about {remaining} more turn(s) before this test ends — wrap up "
        "naturally if you're close to your objective. Answer only what the agent's "
        "LATEST message above actually asked or said — keep any other facts from your "
        "objective/customer context in reserve for when they actually become relevant. "
        "Write your next line now as json."
    )
    try:
        result = await chat(
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user}],
            json_mode=True,
        )
        if isinstance(result, dict):
            utterance = str(result.get("utterance") or "").strip()
            if utterance:
                return {"utterance": utterance, "done": bool(result.get("done"))}
    except Exception:
        pass
    fallback = _FALLBACK_OPENER if turn_number == 0 else _FALLBACK_CONTINUATION
    return {"utterance": fallback, "done": False}
