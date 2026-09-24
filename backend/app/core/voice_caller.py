"""AI Caller — the voice-modality "turn" function for app.core.runner.run_scenario().

`call_voice_agent()` is a small dispatcher over per-protocol transports, keyed on the
agent's configured `voice_protocol` (default "http_json" — see app.db). It returns the
exact ``{"reply": str, "trace": dict}`` shape ``adapter.send()`` already returns, so
``run_scenario()`` can call this in place of ``adapter.send()`` with NO change to its
turn loop, persistence, idempotency, or adaptive-follow-up logic, REGARDLESS of which
transport actually plays the turn. Only WHICH function plays a turn differs; everything
downstream — messages, Judge, scoring, finalization — only ever sees text, exactly as
before. Chat-modality runs never import or execute this module.

Currently implemented:
  "http_json"  -> _call_via_http_json(): TTS the tester's text -> base64 WAV
      -> app.core.adapter.send() [REUSED, unchanged HTTP infra: request templating,
         auth header, timeout/error handling, system-fault simulation]
      -> base64 WAV reply -> STT -> text (agent turn)
  "websocket"  -> _call_via_websocket(): same TTS/STT, one ws:// or wss:// connection
      PER TURN (opened, used once, closed — no persistent multi-turn session yet).
  "twilio"     -> app.core.twilio_bridge._call_via_twilio(): same TTS/STT, but ONE
      PHONE CALL for the WHOLE scenario (Phase 3B) — every turn reuses the same Call
      SID and Media Stream, via `session_key`/`is_last_turn` (see close_voice_session
      below). See that module's docstring; it is a much larger subsystem than the
      other two, since Twilio's Media Streams protocol has no notion of "one turn" at
      all — segmenting the continuous stream into turns is that module's real job.
  "native_ws"  -> app.core.voice_native_ws._call_via_native_ws(): text only, NO
      TTS/STT — a target that speaks its own call-session WebSocket protocol
      directly (create a call over HTTP, then one PERSISTENT WebSocket for the
      WHOLE scenario, driven with a `simulated_utterance` message per turn). Same
      persistent-session shape as twilio, via the same `session_key`/`is_last_turn`
      convention, but far simpler — no audio, no inbound webhook/router half.

Failure handling deliberately mirrors app.core.adapter.send(): any failure (TTS, the
transport call itself, STT, or an unsupported/unknown protocol) resolves to the SAME
sentinel app.core.judge.AGENT_ERROR_SENTINEL ("<error>") that a transport failure
already produces for a chat agent, so the existing judge._endpoint_never_responded() /
_record_system_failure() machinery scores it exactly like an unreachable chat agent —
no new failure category for the Judge to learn.
"""
import asyncio
import base64
import json
from typing import Any, Optional

import websockets

from app.core.adapter import DEFAULT_TIMEOUT_S, send as http_send
from app.core.llm import speech_to_text, text_to_speech

# Matches app.core.judge.AGENT_ERROR_SENTINEL exactly — kept as a literal (not an
# import) to avoid a runner/activities -> judge import for a single string constant;
# judge.py's copy is the canonical one and asserts against this same value.
AGENT_ERROR_SENTINEL = "<error>"

# Protocols recognized by name but with no transport implementation yet. Kept apart
# from a genuinely unknown/typo'd value so the two fail with distinguishable messages.
_KNOWN_UNIMPLEMENTED_PROTOCOLS: set[str] = set()


async def call_voice_agent(
    agent: Any,
    message: str,
    history: Optional[list] = None,
    faults: Optional[list] = None,
    session_key: Optional[Any] = None,
    is_last_turn: bool = False,
) -> dict:
    """One voice turn — dispatches on agent["voice_protocol"] (default "http_json").

    Drop-in replacement for app.core.adapter.send() when passed as run_scenario's
    send_fn — same signature (plus the two trailing keyword-only additions below),
    same return shape, for every protocol.

    `session_key`/`is_last_turn` are run_scenario()'s stable per-conversation id and
    its best-effort "no more turns are coming" signal (see runner.py's docstring for
    exactly what "best-effort" means). http_json and websocket both ignore them —
    neither holds any state across turns. Only twilio's persistent call/session
    (Phase 3B) actually uses them.
    """
    protocol = agent.get("voice_protocol") or "http_json"

    if protocol == "http_json":
        return await _call_via_http_json(agent, message, history, faults)

    if protocol == "websocket":
        return await _call_via_websocket(agent, message, history, faults)

    if protocol == "twilio":
        # Imported lazily (not at module top) so an environment with no Twilio
        # configured, or a hiccup installing the twilio SDK, never breaks chat,
        # http_json, or websocket voice testing — only this one branch.
        from app.core.twilio_bridge import _call_via_twilio

        return await _call_via_twilio(
            agent, message, history, faults,
            session_key=session_key, is_last_turn=is_last_turn,
        )

    if protocol == "native_ws":
        # Imported lazily for the same reason the twilio branch is: an environment
        # with no `websockets`/`httpx` hiccup, or simply no native_ws agent ever
        # configured, never affects chat, http_json, websocket, or twilio testing.
        from app.core.voice_native_ws import _call_via_native_ws

        return await _call_via_native_ws(
            agent, message, history, faults,
            session_key=session_key, is_last_turn=is_last_turn,
        )

    if protocol in _KNOWN_UNIMPLEMENTED_PROTOCOLS:
        return {
            "reply": AGENT_ERROR_SENTINEL,
            "trace": {"error": f"voice_protocol '{protocol}' is not implemented yet"},
        }

    return {
        "reply": AGENT_ERROR_SENTINEL,
        "trace": {"error": f"unknown voice_protocol '{protocol}'"},
    }


async def peek_opening_greeting(agent: Any, session_key: Any) -> Optional[str]:
    """Best-effort: for a voice_protocol with an unprompted agent greeting (native_ws
    only, today), ensure the session is open and return that greeting's text WITHOUT
    sending any turn yet.

    Used only by a dynamic (AI-Caller-driven) scenario's first turn (see
    app.core.runner._run_dynamic), so the caller's opening line can react to what the
    agent actually said. Returns None for every other protocol (http_json, websocket,
    twilio) and for chat — none of those expose a pre-turn greeting the way native_ws's
    call-session protocol does — in which case the dynamic scenario simply opens cold,
    exactly like it already would without this function existing at all.
    """
    protocol = agent.get("voice_protocol") or "http_json"
    if protocol == "native_ws":
        from app.core.voice_native_ws import peek_greeting

        return await peek_greeting(agent, session_key)
    return None


async def close_voice_session(agent: Any, session_key: Any) -> None:
    """Called exactly once from run_scenario()'s finally, for every voice-modality
    scenario regardless of how it ended — see run_scenario()'s `close_fn` parameter.

    Session teardown is a no-op for http_json/websocket: neither holds any state
    across turns, so there is nothing to close. twilio's persistent call/session
    (Phase 3B) and native_ws's persistent call/socket both need it; it is the
    GUARANTEED cleanup path (a safety net for the case where no turn's
    `is_last_turn=True` ever actually fired — see runner.py's docstring).
    """
    protocol = agent.get("voice_protocol") or "http_json"
    if protocol == "twilio":
        from app.core.twilio_bridge import close_twilio_session

        await close_twilio_session(session_key)
    elif protocol == "native_ws":
        from app.core.voice_native_ws import close_native_ws_session

        await close_native_ws_session(session_key)


async def _call_via_http_json(
    agent: Any,
    message: str,
    history: Optional[list] = None,
    faults: Optional[list] = None,
) -> dict:
    """The original (and, today, only implemented) voice transport: TTS the tester's
    text, POST it through the existing black-box HTTP adapter, STT the agent's spoken
    reply. Behavior is unchanged from before this module supported dispatch.
    """
    # --- TTS: tester's text -> base64 WAV ------------------------------------
    try:
        audio_in = await text_to_speech(message)
        audio_in_b64 = base64.b64encode(audio_in).decode("ascii")
    except Exception as e:
        return {"reply": AGENT_ERROR_SENTINEL, "trace": {"error": f"tts error: {e}"}}

    # --- HTTP: reuse the existing black-box adapter, completely unchanged ----
    # This is what makes faults, auth headers, request templating, and system-fault
    # simulation (api_unreachable/api_error/api_timeout/malformed_response) all keep
    # working for a voice agent with zero duplicated logic.
    result = await http_send(agent, audio_in_b64, history, faults)
    reply_b64 = result.get("reply", "")
    trace: dict = dict(result.get("trace") or {})

    # A system/transport fault (or any other adapter-level failure) already comes back
    # as the same sentinel judge.py recognizes — pass it straight through unchanged
    # rather than trying to STT it as audio.
    if reply_b64 == AGENT_ERROR_SENTINEL or trace.get("error"):
        return {"reply": AGENT_ERROR_SENTINEL, "trace": trace}

    # --- STT: agent's spoken reply -> text ------------------------------------
    try:
        audio_out = base64.b64decode(reply_b64) if reply_b64 else b""
        if not audio_out:
            raise ValueError("voice agent returned no audio")
        transcript_out = await speech_to_text(audio_out)
        if not transcript_out:
            raise ValueError("STT produced an empty transcript")
    except Exception as e:
        trace["error"] = f"stt error: {e}"
        return {"reply": AGENT_ERROR_SENTINEL, "trace": trace}

    return {"reply": transcript_out, "trace": trace}


# ---------------------------------------------------------------------------
# WebSocket transport (Phase 2)
# ---------------------------------------------------------------------------
# Wire contract for a "websocket" voice_protocol agent — one connection PER TURN,
# opened, used once, and closed (no persistent multi-turn session in this phase):
#
#   Client -> Server (one JSON text frame):
#     {"type": "audio", "audio": "<base64 WAV>",
#      "history": [{"role": "tester"|"agent", "content": "..."}], "faults": [...]}
#
#   Server -> Client (one JSON text frame):
#     {"type": "audio", "audio": "<base64 WAV>", "trace": {...}}
#     or, on a server-side failure:
#     {"type": "error", "error": "..."}
#
# `history`/`faults` carry the exact same shapes as the http_json contract's request
# body — only the transport differs, not what is sent. backend/sample_voice_bot's
# /voice-chat/ws is the reference implementation of the server side.
def _validate_ws_url(url: str) -> Optional[str]:
    """None if `url` looks like a usable ws(s):// endpoint, else an error message."""
    if not url:
        return "voice_protocol 'websocket' requires an endpoint_url, got none"
    if not (url.startswith("ws://") or url.startswith("wss://")):
        return (
            "voice_protocol 'websocket' requires a ws:// or wss:// endpoint_url, "
            f"got: {url}"
        )
    return None


async def _call_via_websocket(
    agent: Any,
    message: str,
    history: Optional[list] = None,
    faults: Optional[list] = None,
) -> dict:
    """One voice turn over a fresh WebSocket connection (opened, used once, closed).

    Same TTS/STT as _call_via_http_json — only how the audio travels differs. The
    agent's `auth_header` ("Header: value", the same convention adapter.send() already
    uses) is sent as a handshake header, since the WebSocket upgrade IS an HTTP
    request and carries arbitrary headers exactly like a normal one.
    """
    url = (agent.get("endpoint_url") or "").strip()
    bad_url = _validate_ws_url(url)
    if bad_url:
        return {"reply": AGENT_ERROR_SENTINEL, "trace": {"error": bad_url}}

    # --- TTS: tester's text -> base64 WAV ------------------------------------
    try:
        audio_in = await text_to_speech(message)
        audio_in_b64 = base64.b64encode(audio_in).decode("ascii")
    except Exception as e:
        return {"reply": AGENT_ERROR_SENTINEL, "trace": {"error": f"tts error: {e}"}}

    headers: dict[str, str] = {}
    if agent.get("auth_header"):
        name, _, value = agent["auth_header"].partition(":")
        if name and value:
            headers[name.strip()] = value.strip()

    request = {
        "type": "audio",
        "audio": audio_in_b64,
        "history": history or [],
        "faults": faults or [],
    }

    async def _one_turn() -> Any:
        async with websockets.connect(url, additional_headers=headers or None) as ws:
            await ws.send(json.dumps(request))
            raw = await ws.recv()
        return json.loads(raw)

    # One timeout bounding connect + send + recv together, same value adapter.send()
    # uses for its own HTTP call — so a hung/unreachable voice agent fails in a
    # comparable amount of time regardless of transport.
    try:
        response = await asyncio.wait_for(_one_turn(), timeout=DEFAULT_TIMEOUT_S)
    except asyncio.TimeoutError:
        return {
            "reply": AGENT_ERROR_SENTINEL,
            "trace": {"error": f"websocket timeout after {DEFAULT_TIMEOUT_S}s"},
        }
    except Exception as e:
        # Connection refused, handshake rejected (e.g. bad auth), malformed JSON,
        # or any other transport failure — degrade gracefully like adapter.send() does.
        return {"reply": AGENT_ERROR_SENTINEL, "trace": {"error": f"websocket error: {e}"}}

    if not isinstance(response, dict):
        return {
            "reply": AGENT_ERROR_SENTINEL,
            "trace": {"error": "websocket response was not a JSON object"},
        }

    trace: dict = dict(response.get("trace") or {})

    if response.get("type") == "error":
        trace.setdefault("error", str(response.get("error") or "voice agent reported an error"))
        return {"reply": AGENT_ERROR_SENTINEL, "trace": trace}

    reply_b64 = response.get("audio")
    if not reply_b64:
        trace.setdefault("error", "websocket response had no 'audio' field")
        return {"reply": AGENT_ERROR_SENTINEL, "trace": trace}

    # --- STT: agent's spoken reply -> text ------------------------------------
    try:
        audio_out = base64.b64decode(reply_b64)
        if not audio_out:
            raise ValueError("voice agent returned no audio")
        transcript_out = await speech_to_text(audio_out)
        if not transcript_out:
            raise ValueError("STT produced an empty transcript")
    except Exception as e:
        trace["error"] = f"stt error: {e}"
        return {"reply": AGENT_ERROR_SENTINEL, "trace": trace}

    return {"reply": transcript_out, "trace": trace}
