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

The instant the socket opens, the agent speaks first, unprompted — one "agent" turn
event arrives before we ever send anything. run_scenario()'s turn loop is
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
from dataclasses import dataclass
from typing import Any, Optional

import httpx
import websockets

from app.core.adapter import DEFAULT_TIMEOUT_S

# Matches app.core.judge.AGENT_ERROR_SENTINEL / app.core.voice_caller.AGENT_ERROR_SENTINEL
# exactly — kept as a literal, same tradeoff already made in twilio_bridge.py.
AGENT_ERROR_SENTINEL = "<error>"


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


def _create_call_body(agent: Any) -> dict:
    """`request_template`, parsed as the JSON body for POST /api/calls.

    Falls back to `{}` on anything unparsable, so a misconfigured agent fails as a
    clean HTTP 4xx from the target (caught by _open_session) rather than crashing
    this module.
    """
    raw = agent.get("request_template")
    if not raw:
        return {}
    try:
        body = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return {}
    return body if isinstance(body, dict) else {}


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
        greeting = await asyncio.wait_for(_next_agent_turn(session.ws), timeout=DEFAULT_TIMEOUT_S)
        session.opening_line = greeting.get("text") if greeting else None
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
                "error": "native_ws session for this conversation already ended (a prior turn failed)"
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
        turn = await asyncio.wait_for(_next_agent_turn(session.ws), timeout=DEFAULT_TIMEOUT_S)
    except Exception as e:
        session.closed = True  # tombstone — a later turn must fail fast, not reconnect silently
        return {"reply": AGENT_ERROR_SENTINEL, "trace": {**trace, "error": f"native_ws turn error: {e}"}}

    if not turn:
        session.closed = True
        return {"reply": AGENT_ERROR_SENTINEL, "trace": {**trace, "error": "socket closed with no agent reply"}}

    reply = str(turn.get("text") or "")
    if turn.get("end_status"):
        trace["end_status"] = turn["end_status"]
    if turn.get("answers"):
        trace["answers"] = turn["answers"]

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
        except Exception:
            session.closed = True
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
