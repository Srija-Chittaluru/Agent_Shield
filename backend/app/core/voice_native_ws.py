"""Native persistent-WebSocket voice transport — voice_protocol == "native_ws".

Bridges a text scenario onto a target voice agent that speaks its own call-session
protocol: create a call over HTTP, then hold ONE WebSocket connection open for the
WHOLE scenario, driving each turn with a `simulated_utterance` JSON message and
reading back the agent's `"turn"` event as the reply. Modeled on
app.core.twilio_bridge's persistent-session pattern, but far simpler: this module is
a WebSocket *client*, not a server accepting an inbound stream, so there is no
webhook/router half to it at all.

Field reuse on the `agents` row (no schema changes — the same trick twilio_bridge
already plays with `endpoint_url` meaning something else for its own protocol):
    endpoint_url      -> HTTP base URL, e.g. "http://localhost:8000". Used for both
                         POST {base}/api/calls and, scheme-swapped (http -> ws,
                         https -> wss), the WebSocket URL — exactly what the target's
                         own frontend does (same-origin, protocol-swapped).
    request_template  -> the JSON body for POST /api/calls, e.g. {"client": "acme"}.
    auth_header       -> "Authorization: Bearer <token>", sent on the two HTTP calls
                         only — the WebSocket handshake itself carries no auth (the
                         call_id is the capability, per the target's own security
                         model: see its app/api/security.py).

Wire contract below is the TARGET's own (not ours) — see its
app/api/routes/simulator.py and app/transports/orchestrator_processor.py:
    POST {base}/api/calls          {request_template body} -> {"call_id": "<uuid>"}
    WS   {base}/api/ws/{call_id}   opened once, held for the whole scenario
        server -> client (JSON text frame), the only ones we act on:
            {"type": "turn", "role": "agent"|"caller", "text": "...", "end_status": ...}
        client -> server:
            {"type": "simulated_utterance", "text": "..."}
    POST {base}/api/calls/{call_id}/end   -> best-effort finalize

The instant the socket opens, the agent speaks first, unprompted — at least one "agent"
turn event arrives before we ever send anything.

One reply may be SEVERAL consecutive agent turns: a target that has more to say without
asking anything (a greeting, then its first question) speaks each as its own turn, and
a caller line sent between two of them is dropped by the target (its current turn owns
the moment). So every read — the opening and each reply — takes the agent's whole
output (_read_agent_output): consecutive agent turns until the agent has finished
speaking and nothing more follows, joined into one reply. run_scenario()'s turn loop is
tester-first (seed turn -> reply), so that greeting has no transcript slot of its
own; it is captured as trace["opening_line"] on the FIRST turn rather than dropped
or misread as the reply to that turn's tester message.

Session store: one process-local dict, keyed by run_scenario()'s `session_key`
(its stable per-conversation id), same as twilio_bridge.py's `_SESSIONS_BY_KEY`. A
session is stored BEFORE the HTTP call that creates it is placed — same reason
twilio_bridge does this — so a later turn on the SAME session_key, if the first
turn's setup itself failed, sees "this session existed but died" and fails fast
rather than silently placing a second call. Final removal happens only in
close_native_ws_session(), guaranteed exactly once by run_scenario()'s `finally`.

Failure handling mirrors every other transport: any failure (HTTP create-call
error, WebSocket connect/send/recv error, an unexpected close, a malformed
message) resolves to the same AGENT_ERROR_SENTINEL ("<error>") the rest of the
system already knows how to score as a system failure — no Judge change needed.
"""
import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx
import websockets

from app.core.adapter import DEFAULT_TIMEOUT_S

# Matches app.core.judge.AGENT_ERROR_SENTINEL / app.core.voice_caller.AGENT_ERROR_SENTINEL
# exactly — kept as a literal, same tradeoff already made in twilio_bridge.py.
AGENT_ERROR_SENTINEL = "<error>"

# When one exchange's agent output counts as finished, so the caller may speak (see
# _read_agent_output). The agent composes a consecutive turn while its previous line is
# still being heard, so the wait is measured both from its last turn and from its last
# audio:
#   SETTLE_S       — no new agent turn for this long (covers composing the next turn
#                    after a short line; the target's model latency varies, ~1-3 s)
#   AUDIO_QUIET_S  — and no audio frame for this long (the target streams its speech in
#                    real time, so audio still arriving means it is still talking)
#   SPEAKING_MAX_S — while the target itself reports speaking, at most this long
#   MAX_OUTPUT_S   — never wait longer than this after the first turn, whatever arrives
SETTLE_S = 4.0
AUDIO_QUIET_S = 1.0
SPEAKING_MAX_S = DEFAULT_TIMEOUT_S
MAX_OUTPUT_S = 120.0


@dataclass
class _Session:
    """One AgentShield test conversation's persistent native_ws call.

    Created (empty) and stored under its session_key BEFORE the HTTP call that
    would fill in call_id/base_url/ws is even placed — see the module docstring.
    """
    call_id: str = ""
    base_url: str = ""
    ws: Any = None
    closed: bool = False
    opening_line: Optional[str] = None
    # The greeting turn's end_status, if the agent ended the call in its very first
    # turn. Only read by session_ended(); nothing else changes because of it.
    opening_end_status: Optional[str] = None
    # Frames read while collecting one reply that belong to the next exchange (a
    # caller echo); consumed first by the next read.
    pending: list = field(default_factory=list)
    # Why setting the call up failed, when it did — reported by every later turn.
    setup_error: Optional[str] = None


_SESSIONS: dict[Any, _Session] = {}


def _ws_url(base_url: str, call_id: str) -> str:
    if base_url.startswith("https://"):
        ws_base = "wss://" + base_url[len("https://"):]
    elif base_url.startswith("http://"):
        ws_base = "ws://" + base_url[len("http://"):]
    else:
        ws_base = base_url
    return f"{ws_base.rstrip('/')}/api/ws/{call_id}"


def _auth_headers(agent: Any) -> dict[str, str]:
    headers: dict[str, str] = {}
    if agent.get("auth_header"):
        name, _, value = agent["auth_header"].partition(":")
        if name and value:
            headers[name.strip()] = value.strip()
    return headers


# A request_template value that is exactly "{{name}}" is a call variable: filled from
# the scenario's test data (agent["call_variables"]) when it supplies `name`, and left
# out of the body when it does not — so the body is then exactly the template without
# that field.
_CALL_VARIABLE = re.compile(r"^\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}$")


class CallVariableError(ValueError):
    """Test data names a call variable the agent's request_template does not map."""


def _parsed_template(agent: Any) -> Optional[dict]:
    raw = agent.get("request_template")
    if not raw:
        return {}
    try:
        body = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return None
    return body if isinstance(body, dict) else None


def call_template_variables(agent: Any) -> list[str]:
    """The call variables this agent's request_template declares, in template order."""
    body = _parsed_template(agent) or {}
    names = []
    for value in body.values():
        m = _CALL_VARIABLE.match(value) if isinstance(value, str) else None
        if m and m.group(1) not in names:
            names.append(m.group(1))
    return names


def check_call_variables(agent: Any, values: Optional[dict]) -> list[str]:
    """Problems with supplying `values` to this agent's call creation (empty if none)."""
    if not values:
        return []
    if (agent.get("voice_protocol") or "") != "native_ws":
        return [f"call variables are only supported for native_ws voice agents; this agent uses {agent.get('voice_protocol')!r}"]
    if _parsed_template(agent) is None:
        return ["the agent's request_template is not a JSON object, so it cannot take call variables"]
    declared = call_template_variables(agent)
    unknown = [k for k in values if k not in declared]
    if unknown:
        return [
            f"call variable(s) {', '.join(unknown)} are not mapped by the agent's request_template "
            f"(it declares: {', '.join(declared) or 'none'}); add e.g. \"{unknown[0]}\": \"{{{{{unknown[0]}}}}}\""
        ]
    return [f"call variable {k!r} must be a non-empty string" for k, v in values.items() if not isinstance(v, str) or not v.strip()]


def _create_call_body(agent: Any) -> dict:
    """`request_template`, parsed as the JSON body for POST /api/calls, with its call
    variables filled from agent["call_variables"] (see _CALL_VARIABLE).

    Without call variables supplied, falls back to `{}` on anything unparsable, so a
    misconfigured agent fails as a clean HTTP 4xx from the target (caught by
    _open_session) rather than crashing this module — unchanged. Supplied call variables
    the template cannot take raise CallVariableError instead of being dropped.
    """
    values = agent.get("call_variables") or {}
    problems = check_call_variables(agent, values)
    if problems:
        raise CallVariableError("; ".join(problems))
    body = _parsed_template(agent)
    if body is None:
        return {}
    filled = {}
    for key, value in body.items():
        m = _CALL_VARIABLE.match(value) if isinstance(value, str) else None
        if m is None:
            filled[key] = value
        elif m.group(1) in values:
            filled[key] = values[m.group(1)].strip()
    return filled


async def _next_agent_turn(ws: Any) -> Optional[dict]:
    """Read frames until an agent-role "turn" event arrives.

    Skips the "caller" echo of the tester's own utterance, "speaking"/
    "interruption" frames, and any binary frame (the target's own real TTS audio,
    PCM16 mono 16kHz, no header — see the module docstring) — none of those are
    the "turn" JSON event we're waiting for, so they're silently skipped rather
    than breaking turn detection.

    Raises (propagates) if the socket closes or errors before a "turn" event is
    ever seen — callers treat that exactly like any other transport failure.
    """
    while True:
        raw = await ws.recv()
        if not isinstance(raw, str):
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        if payload.get("type") == "turn" and payload.get("role") == "agent":
            return payload


async def _read_agent_output(session: _Session, first_timeout: float = DEFAULT_TIMEOUT_S) -> list[dict]:
    """The agent's whole output for one exchange: every consecutive agent "turn" event
    until the agent has finished.

    Waits up to `first_timeout` for the first agent turn (raising, as _next_agent_turn
    does, if none arrives or the socket fails first). After that the output is finished
    when any of these happens:
      - a turn carries `end_status` (the call is ending — returned at once);
      - the agent has gone quiet: no new agent turn for SETTLE_S AND no audio frame
        for AUDIO_QUIET_S (while the target reports `{"type": "speaking", "speaking":
        true}`, up to SPEAKING_MAX_S instead). A target that streams no audio and sends
        no speaking events settles SETTLE_S after its last turn;
      - a caller "turn" echo arrives (the next exchange has begun; kept for the next read);
      - the socket fails or closes (what was collected is returned; the next send then
        fails and tombstones the session exactly as before);
      - MAX_OUTPUT_S has passed since the first turn.
    """
    loop = asyncio.get_running_loop()
    turns: list[dict] = []
    speaking = False
    last_turn = last_audio = last_speaking = cap = 0.0

    def settle_deadline() -> float:
        if speaking:
            return min(cap, last_speaking + SPEAKING_MAX_S)
        return min(cap, max(last_turn + SETTLE_S, last_audio + AUDIO_QUIET_S))

    first_deadline = loop.time() + first_timeout
    while True:
        if session.pending:
            raw = session.pending.pop(0)
        else:
            remaining = (settle_deadline() if turns else first_deadline) - loop.time()
            if remaining <= 0:
                if turns:
                    return turns
                raise asyncio.TimeoutError("no agent turn")
            try:
                raw = await asyncio.wait_for(session.ws.recv(), timeout=remaining)
            except Exception:
                if turns:
                    return turns
                raise
        now = loop.time()
        if not isinstance(raw, str):
            last_audio = now  # the target's speech audio: it is still talking
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        kind = payload.get("type")
        if kind == "speaking":
            speaking = bool(payload.get("speaking"))
            last_speaking = now
            continue
        if kind != "turn":
            continue
        if payload.get("role") == "caller":
            if turns:
                session.pending.insert(0, raw)
                return turns
            continue  # the echo of the line just sent
        if payload.get("role") != "agent":
            continue
        if not turns:
            cap = now + MAX_OUTPUT_S
        turns.append(payload)
        last_turn = now
        if payload.get("end_status"):
            return turns


def _combine_turns(turns: list[dict]) -> tuple[str, dict]:
    """(reply text, trace fields) for consecutive agent turns. The reply is their texts
    joined in order. Trace: `end_status` and `answers` as the latest turn reporting them
    (answers is the target's running total), `interrupt_key` as the first turn reporting
    one, and — only when there was more than one turn — each turn's text as `agent_turns`."""
    reply = " ".join(str(t.get("text") or "").strip() for t in turns if str(t.get("text") or "").strip())
    info: dict = {}
    for t in turns:
        if t.get("end_status"):
            info["end_status"] = t["end_status"]
        if t.get("answers"):
            info["answers"] = t["answers"]
        if t.get("interrupt_key") and "interrupt_key" not in info:
            info["interrupt_key"] = t["interrupt_key"]
    if len(turns) > 1:
        info["agent_turns"] = [str(t.get("text") or "") for t in turns]
    return reply, info


async def _open_session(agent: Any, session: _Session) -> None:
    """Fill in `session` in place: create the call over HTTP, open the WebSocket,
    and drain the agent's unprompted opening greeting.

    Raises on a genuine setup failure (bad endpoint_url, HTTP error, no call_id in
    the response, WebSocket connect failure) — the caller tombstones `session` on
    any exception here. Failing to capture the greeting itself is NOT such a
    failure (a slow/silent agent shouldn't block the whole session), so that part
    has its own narrower try/except.
    """
    base_url = (agent.get("endpoint_url") or "").rstrip("/")
    if not base_url:
        raise ValueError(
            "voice_protocol 'native_ws' requires an endpoint_url (the agent's HTTP base URL)"
        )
    session.base_url = base_url

    body = _create_call_body(agent)
    headers = _auth_headers(agent)
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT_S) as client:
        resp = await client.post(f"{base_url}/api/calls", json=body, headers=headers)
        resp.raise_for_status()
        data = resp.json()
    call_id = data.get("call_id") if isinstance(data, dict) else None
    if not call_id:
        raise ValueError(f"POST /api/calls did not return a call_id: {data!r}")
    session.call_id = str(call_id)

    session.ws = await websockets.connect(_ws_url(base_url, session.call_id))

    try:
        # The whole opening — a greeting may be followed straight away by the agent's
        # first question, and the caller must not speak between them.
        greeting, info = _combine_turns(await _read_agent_output(session))
        session.opening_line = greeting or None
        session.opening_end_status = info.get("end_status") or None
    except Exception:
        session.opening_line = None


async def _call_via_native_ws(
    agent: Any,
    message: str,
    history: Optional[list] = None,
    faults: Optional[list] = None,
    session_key: Optional[Any] = None,
    is_last_turn: bool = False,
) -> dict:
    """One voice turn over the target's native persistent-WebSocket protocol.

    FIRST turn for a given `session_key`: creates the call and opens the socket
    (see `_open_session`). EVERY turn (first and subsequent): sends this turn's
    text as a `simulated_utterance` on the SAME open socket and waits for the
    agent's `"turn"` reply.

    `history`/`faults` are accepted for signature parity with every other voice
    transport (app.core.voice_caller.call_voice_agent calls all of them the same
    way) but unused here: the target holds its own conversation state across turns
    on the open socket, the same reason app.core.twilio_bridge ignores them too.
    """
    session = _SESSIONS.get(session_key) if session_key is not None else None

    if session is not None and session.closed:
        return {
            "reply": AGENT_ERROR_SENTINEL,
            "trace": {
                "error": f"native_ws call setup error: {session.setup_error}" if session.setup_error
                else "native_ws session for this conversation already ended (a prior turn failed)"
            },
        }

    if session is None:
        # Stored BEFORE _open_session's HTTP call is placed — see module docstring.
        session = _Session()
        if session_key is not None:
            _SESSIONS[session_key] = session
        try:
            await _open_session(agent, session)
        except Exception as e:
            session.closed = True
            return {"reply": AGENT_ERROR_SENTINEL, "trace": {"error": f"native_ws call setup error: {e}"}}

    trace: dict = {"call_id": session.call_id}
    if session.opening_line is not None:
        trace["opening_line"] = session.opening_line
        session.opening_line = None  # only the very first turn carries it

    try:
        await session.ws.send(json.dumps({"type": "simulated_utterance", "text": message}))
        turns = await _read_agent_output(session)
    except Exception as e:
        session.closed = True  # tombstone — a later turn must fail fast, not reconnect silently
        return {"reply": AGENT_ERROR_SENTINEL, "trace": {**trace, "error": f"native_ws turn error: {e}"}}

    if not turns:
        session.closed = True
        return {"reply": AGENT_ERROR_SENTINEL, "trace": {**trace, "error": "socket closed with no agent reply"}}

    reply, info = _combine_turns(turns)
    trace.update(info)
    return {"reply": reply, "trace": trace}


async def peek_greeting(agent: Any, session_key: Any) -> Optional[str]:
    """Ensure the session for `session_key` is open, and return the agent's unprompted
    opening greeting WITHOUT sending any simulated_utterance yet.

    Used only by a dynamic (AI-Caller-driven) scenario's first turn, so the caller's
    opening line can react to what the agent actually greeted with — see
    app.core.runner._run_dynamic. Not used at all by a scripted scenario, which still
    gets its opening_line the original way: attached to trace on its first REAL turn,
    inside _call_via_native_ws above.

    Same "store the session before the HTTP call" and tombstone-on-failure rules as
    _call_via_native_ws, so a scenario that calls this first and then plays real turns
    through _call_via_native_ws shares exactly ONE session/call/socket — never a second
    one. Consumes (clears) session.opening_line once read, so that first real turn's own
    trace does not redundantly repeat it.

    Returns None on any failure, or for a session already tombstoned by an earlier
    failure — the caller then simply opens the conversation cold, exactly like it always
    does for chat and for http_json/websocket voice, which have no equivalent concept.
    """
    session = _SESSIONS.get(session_key) if session_key is not None else None
    if session is not None and session.closed:
        return None

    if session is None:
        session = _Session()
        if session_key is not None:
            _SESSIONS[session_key] = session
        try:
            await _open_session(agent, session)
        except Exception as e:
            session.closed = True
            session.setup_error = str(e)
            return None

    greeting = session.opening_line
    session.opening_line = None
    return greeting


async def close_native_ws_session(session_key: Any) -> None:
    """Guaranteed final cleanup for one AgentShield conversation. Called exactly
    once from run_scenario()'s `finally`, via app.core.voice_caller.close_voice_session,
    regardless of how the scenario ended. Idempotent — a session already tombstoned
    by a failed turn, or no session ever created at all, is a no-op.
    """
    session = _SESSIONS.pop(session_key, None)
    if session is None:
        return
    session.closed = True
    if session.ws is not None:
        try:
            await session.ws.close(code=1000, reason="scenario finished")
        except Exception:
            pass
    if session.call_id and session.base_url:
        try:
            async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT_S) as client:
                await client.post(f"{session.base_url}/api/calls/{session.call_id}/end")
        except Exception:
            pass


def session_ended(session_key: Any, trace: Optional[dict] = None) -> bool:
    """Read-only: has this conversation's call ended? True when the agent's latest turn
    carried the target's `end_status` (its final call disposition — it is only ever
    sent on the call's last agent turn), when the opening greeting itself carried one
    (the agent ended the call in its first turn), or when a failed turn already
    tombstoned the session. Changes no session state; used by the runner to stop
    sending caller lines into a call that is over."""
    if trace and trace.get("end_status"):
        return True
    session = _SESSIONS.get(session_key)
    return session is not None and (session.closed or bool(session.opening_end_status))
