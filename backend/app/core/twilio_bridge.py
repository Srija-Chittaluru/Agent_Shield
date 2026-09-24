"""Twilio voice transport — Phase 3B: ONE CALL FOR THE WHOLE SCENARIO.

Bridges a text scenario's turns onto a REAL PHONE CALL via Twilio Programmable Voice +
Media Streams. Unlike http_json/websocket, Twilio's Media Streams protocol is a
continuous, unsegmented mulaw/8kHz audio pipe with no "end of turn" signal — the real
engineering content of this module is translating that into the same discrete
"one turn's audio in, one turn's audio out" shape every other transport already uses,
AND (Phase 3B) doing so repeatedly over ONE persistent call rather than one call each.

    _call_via_twilio(agent, message, history, faults, session_key, is_last_turn)
        FIRST turn for a given `session_key` (run_scenario()'s stable per-conversation
        id — see runner.py):
            1. TTS the tester's text (REUSED, unchanged: app.core.llm.text_to_speech)
            2. convert WAV -> mulaw/8kHz (this module, pure Python, no audioop)
            3. originate ONE outbound call via the Twilio REST API, pointing its TwiML
               webhook at THIS process with a fresh, unguessable correlation key
            4. wait for that call's Media Stream to connect back to /twilio/media
        EVERY turn (first and subsequent — steps 5-8 repeat per turn on the SAME call):
            5. hand this turn's audio to the (already-connected, already-running)
               Media Stream handler and wait for it to come back with the agent's
               buffered reply (a simple silence/timeout heuristic decides when the
               agent has stopped speaking — NOT production-grade VAD)
            6. convert the buffered mulaw back to WAV, STT it (REUSED, unchanged)
            7. return {"reply": text, "trace": dict} to run_scenario()
        Only once run_scenario()'s turn loop is fully done (`close_voice_session()`,
        called from its `finally` — see runner.py) does the call actually hang up.

Session store (app.routers.twilio <-> this module): two in-memory dicts pointing at
the same _Session objects — `_SESSIONS_BY_KEY` (keyed by run_scenario()'s
`session_key`, so `_call_via_twilio` can find/reuse the SAME session across every
turn of one scenario) and `_SESSIONS_BY_CONV_KEY` (keyed by a random, unguessable
`conv_key`, threaded through Twilio's own request plumbing — the webhook's query
string, then the <Stream> custom parameter, then the Media Stream's
`start.customParameters` — so the webhook/stream can find the right session without
ever seeing `session_key` itself). No new database table; nothing here outlives one
test conversation.

A session is created BEFORE the outbound REST call is placed (so a webhook/stream
that arrives immediately can never race ahead of us). On any failure it is marked
`closed` (and the call hung up) but deliberately NOT removed from the dicts yet — a
subsequent turn on the SAME session_key must still be able to see "this session
existed but died" and fail fast, rather than silently starting a second call. Final
removal happens only in `close_twilio_session()`, guaranteed exactly once by
run_scenario()'s `finally`. Turns are serialized through one `asyncio.Queue` per
session — the Media Stream handler processes exactly one turn at a time, by
construction, so audio from one turn can never leak into another's buffer.

Failure handling mirrors every other transport: any failure (bad phone number, Twilio
API error, the call never connects, a dead session reused after a prior failure, a
malformed Media Stream message, empty/invalid audio, STT failure) resolves to the
same AGENT_ERROR_SENTINEL ("<error>") the rest of the system already knows how to
score as a system failure — no Judge change needed.

NO audioop: it is deprecated and removed in Python 3.13. mu-law encode/decode here is
a small, self-contained implementation of the standard ITU-T G.711 algorithm.
"""
import asyncio
import base64
import io
import re
import struct
import time
import uuid
import wave
from dataclasses import dataclass, field
from typing import Any, Optional

from twilio.rest import Client as TwilioClient
from twilio.twiml.voice_response import Connect, Stream, VoiceResponse

from app.config import PUBLIC_BASE_URL, TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM_NUMBER
from app.core.llm import speech_to_text, text_to_speech

# Matches app.core.judge.AGENT_ERROR_SENTINEL / app.core.voice_caller.AGENT_ERROR_SENTINEL
# exactly. Duplicated as a literal rather than imported, same tradeoff already made
# between judge.py and voice_caller.py — importing from voice_caller.py here would be
# circular (voice_caller dispatches INTO this module).
AGENT_ERROR_SENTINEL = "<error>"

# --- Telephony / timing constants (POC-tuned, not production-calibrated) -----
SAMPLE_RATE_HZ = 8000                       # Twilio Media Streams' fixed rate
FRAME_MS = 20                                # Twilio's own frame duration
FRAME_BYTES = SAMPLE_RATE_HZ * FRAME_MS // 1000  # 160 bytes/frame (8-bit mu-law, mono)
SEND_PACING_S = FRAME_MS / 1000.0            # real-time pacing between outbound frames

# Silence/timeout heuristic for "the agent stopped talking" — NOT production-grade VAD.
# Twilio streams audio continuously regardless of loudness, so this is a simple energy
# threshold over decoded PCM16, not silence-vs-no-frames. Both constants would need
# real tuning against a real phone line before this is anything but a POC.
SILENCE_RMS_THRESHOLD = 400                  # rough |PCM16| magnitude
SILENCE_DURATION_S = 1.5                     # how long "quiet" must persist to end a turn
MAX_LISTEN_S = 20.0                          # hard safety cap regardless of silence

# Bounds the WHOLE call: ring + answer + webhook + stream-connect + speak + listen.
# Generous relative to http_json/websocket's 30s, because a real phone call's
# ring/answer alone can take several seconds before any of our own logic even starts.
CALL_TIMEOUT_S = 60.0

_PHONE_RE = re.compile(r"^\+[1-9]\d{7,14}$")  # loose E.164 check


# ---------------------------------------------------------------------------
# mu-law <-> linear PCM16 — standard ITU-T G.711 algorithm, pure Python.
# Deliberately NOT using audioop: it is deprecated and removed in Python 3.13.
# ---------------------------------------------------------------------------
_MULAW_BIAS = 0x84
_MULAW_MAX = 0x1FFF


def _linear_to_mulaw_sample(sample: int) -> int:
    sign = 0x00
    if sample < 0:
        sample = -sample
        sign = 0x80
    sample = min(sample, _MULAW_MAX) + _MULAW_BIAS
    exponent = 7
    mask = 0x4000
    while exponent > 0 and not (sample & mask):
        exponent -= 1
        mask >>= 1
    mantissa = (sample >> (exponent + 3)) & 0x0F
    return ~(sign | (exponent << 4) | mantissa) & 0xFF


def _mulaw_to_linear_sample(byte: int) -> int:
    byte = ~byte & 0xFF
    sign = byte & 0x80
    exponent = (byte >> 4) & 0x07
    mantissa = byte & 0x0F
    sample = ((mantissa << 3) + _MULAW_BIAS) << exponent
    sample -= _MULAW_BIAS
    return -sample if sign else sample


def pcm16_to_mulaw(pcm: bytes) -> bytes:
    """Little-endian 16-bit PCM mono -> 8-bit mu-law."""
    n = len(pcm) // 2
    samples = struct.unpack(f"<{n}h", pcm[: n * 2])
    return bytes(_linear_to_mulaw_sample(s) for s in samples)


def mulaw_to_pcm16(mulaw: bytes) -> bytes:
    """8-bit mu-law -> little-endian 16-bit PCM mono."""
    samples = [_mulaw_to_linear_sample(b) for b in mulaw]
    return struct.pack(f"<{len(samples)}h", *samples) if samples else b""


def _avg_abs_amplitude(pcm: bytes) -> float:
    n = len(pcm) // 2
    if n == 0:
        return 0.0
    samples = struct.unpack(f"<{n}h", pcm[: n * 2])
    return sum(abs(s) for s in samples) / n


# ---------------------------------------------------------------------------
# WAV <-> raw PCM16 helpers (stdlib `wave` only — no new dependency).
# ---------------------------------------------------------------------------
def _wav_bytes_to_pcm16(wav_bytes: bytes) -> tuple[bytes, int]:
    """Returns (raw little-endian PCM16 mono samples, sample_rate)."""
    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        if w.getsampwidth() != 2:
            raise ValueError(f"expected 16-bit PCM WAV, got sampwidth={w.getsampwidth()}")
        raw = w.readframes(w.getnframes())
        rate = w.getframerate()
        channels = w.getnchannels()
    if channels > 1:
        raw = _downmix_to_mono(raw, channels)
    return raw, rate


def _downmix_to_mono(raw: bytes, channels: int) -> bytes:
    n = len(raw) // 2
    samples = struct.unpack(f"<{n}h", raw[: n * 2])
    mono = [
        sum(samples[i : i + channels]) // channels
        for i in range(0, len(samples) - len(samples) % channels, channels)
    ]
    return struct.pack(f"<{len(mono)}h", *mono)


def _pcm16_to_wav_bytes(pcm: bytes, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()


def _resample_pcm16(pcm: bytes, in_rate: int, out_rate: int) -> bytes:
    """Linear-interpolation resample. Simple and self-contained, not hi-fi — fine for
    a voice-testing POC, not for audio-quality-sensitive use."""
    if in_rate == out_rate or not pcm:
        return pcm
    n = len(pcm) // 2
    samples = struct.unpack(f"<{n}h", pcm[: n * 2])
    n_out = max(1, int(n * out_rate / in_rate))
    out = []
    for i in range(n_out):
        pos = i * (n - 1) / (n_out - 1) if n_out > 1 else 0.0
        lo = int(pos)
        hi = min(lo + 1, n - 1)
        frac = pos - lo
        val = samples[lo] * (1 - frac) + samples[hi] * frac
        out.append(max(-32768, min(32767, int(val))))
    return struct.pack(f"<{len(out)}h", *out)


def wav_to_mulaw_frames(wav_bytes: bytes) -> list[bytes]:
    """TTS output (whatever rate) -> a list of 160-byte (20ms) 8kHz mu-law frames."""
    pcm, rate = _wav_bytes_to_pcm16(wav_bytes)
    pcm = _resample_pcm16(pcm, rate, SAMPLE_RATE_HZ)
    mulaw = pcm16_to_mulaw(pcm)
    return [mulaw[i : i + FRAME_BYTES] for i in range(0, len(mulaw), FRAME_BYTES)]


def mulaw_buffer_to_wav(mulaw: bytes) -> bytes:
    """Accumulated 8kHz mu-law audio -> a WAV file speech_to_text() can consume
    directly. No upsampling needed — Whisper handles 8kHz telephony-quality audio
    natively."""
    pcm = mulaw_to_pcm16(mulaw)
    return _pcm16_to_wav_bytes(pcm, SAMPLE_RATE_HZ)


# ---------------------------------------------------------------------------
# Persistent session store — in-memory only, one entry per AgentShield test
# conversation (not per turn), no DB table. See the module docstring.
# ---------------------------------------------------------------------------
# Sentinel pushed onto a session's turn_queue to wake a Media Stream handler that is
# idly waiting for the next turn, so it can exit its loop and tear the call down.
# A plain object() rather than None: None is a plausible (if unlikely) real queue
# item in other contexts, this can never be mistaken for one.
_CLOSE_SENTINEL = object()


@dataclass
class _TurnRequest:
    """One turn's outbound audio, and the future its result resolves on."""
    frames: list[bytes]
    result: "asyncio.Future[dict]"
    is_last_turn: bool


@dataclass
class _Session:
    """One AgentShield test conversation's persistent Twilio call — created on its
    first turn, reused by every subsequent turn, torn down once.

    Looked up by TWO keys pointing at the same object: `session_key` (run_scenario's
    stable id, for _call_via_twilio to find/reuse across turns) and `conv_key` (an
    unguessable per-call token Twilio's webhook/stream correlate on).
    """
    session_key: Any
    conv_key: str
    turn_queue: "asyncio.Queue" = field(default_factory=asyncio.Queue)
    call_sid: Optional[str] = None
    stream_sid: Optional[str] = None
    closed: bool = False
    current_turn: Optional[_TurnRequest] = None  # the one turn in flight, if any


_SESSIONS_BY_KEY: dict[Any, _Session] = {}
_SESSIONS_BY_CONV_KEY: dict[str, _Session] = {}


def _twiml_url(conv_key: str) -> str:
    base = (PUBLIC_BASE_URL or "").rstrip("/")
    return f"{base}/twilio/twiml?conv_key={conv_key}"


def _stream_url() -> str:
    base = (PUBLIC_BASE_URL or "").rstrip("/")
    # wss:// regardless of whether PUBLIC_BASE_URL is https:// or http://, since a
    # tunnel (ngrok) terminates TLS in front of this plain-http process either way.
    host = base.split("://", 1)[-1]
    return f"wss://{host}/twilio/media"


def session_exists(conv_key: str) -> bool:
    """Whether a session is currently waiting on this correlation key. Used by the
    /twilio/twiml webhook to reject a request that doesn't correspond to a live
    session (see app.routers.twilio's security note)."""
    return conv_key in _SESSIONS_BY_CONV_KEY


def build_twiml(conv_key: str) -> str:
    """The TwiML returned from POST /twilio/twiml — connects the call's entire audio
    to our Media Stream, passing `conv_key` through as a custom Stream parameter.
    Fetched exactly once per call (Twilio never re-fetches it for later turns)."""
    response = VoiceResponse()
    connect = Connect()
    stream = Stream(url=_stream_url())
    stream.parameter(name="conv_key", value=conv_key)
    connect.append(stream)
    response.append(connect)
    return str(response)


async def _hangup(session: _Session) -> None:
    """Best-effort, safe to call more than once or on an already-ended call."""
    if not session.call_sid:
        return
    try:
        client = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        await asyncio.to_thread(client.calls(session.call_sid).update, status="completed")
    except Exception:
        pass  # best-effort; the call will time out on Twilio's side otherwise


async def _teardown(session: _Session, error: Optional[str] = None) -> None:
    """Mark a session closed and hang up its call. Deliberately does NOT remove it
    from the lookup dicts — a later turn on the SAME session_key must still be able
    to see "this session existed but is dead" and fail fast rather than starting a
    second call. Final removal is close_twilio_session()'s job. Idempotent."""
    if session.closed:
        return
    session.closed = True
    if session.current_turn is not None and not session.current_turn.result.done():
        session.current_turn.result.set_result({"audio": b"", "error": error or "session closed"})
    session.turn_queue.put_nowait(_CLOSE_SENTINEL)  # wake a handler waiting on the queue
    await _hangup(session)


async def close_twilio_session(session_key: Any) -> None:
    """Guaranteed final cleanup for one AgentShield conversation. Called exactly once
    from run_scenario()'s `finally`, via app.core.voice_caller.close_voice_session,
    regardless of how the scenario ended. Idempotent — safe even if a turn failure
    already tore the session down, or no Twilio session was ever created at all."""
    session = _SESSIONS_BY_KEY.pop(session_key, None)
    if session is None:
        return
    _SESSIONS_BY_CONV_KEY.pop(session.conv_key, None)
    await _teardown(session)


# ---------------------------------------------------------------------------
# The AI Caller entry point for voice_protocol == "twilio"
# ---------------------------------------------------------------------------
async def _call_via_twilio(
    agent: Any,
    message: str,
    history: Optional[list] = None,
    faults: Optional[list] = None,
    session_key: Optional[Any] = None,
    is_last_turn: bool = False,
) -> dict:
    """One voice turn. The FIRST call for a given `session_key` places ONE outbound
    phone call and waits for its Media Stream to connect; every call (first and
    subsequent) hands its turn's audio to that same, already-running stream and
    waits for the agent's reply. The call itself is only hung up once
    close_twilio_session() runs, after run_scenario()'s whole turn loop is done.

    `agent["endpoint_url"]` is reused to hold the target's E.164 phone number for this
    protocol (there is no URL involved at all) — validated explicitly, never assumed,
    and only on the first turn (a later turn reuses the already-validated session).
    """
    session = _SESSIONS_BY_KEY.get(session_key) if session_key is not None else None

    if session is not None and session.closed:
        return {
            "reply": AGENT_ERROR_SENTINEL,
            "trace": {"error": "twilio session for this conversation already ended (a prior turn failed)"},
        }

    if session is None:
        # --- FIRST turn for this session_key: validate + originate ONE call -----
        to_number = (agent.get("endpoint_url") or "").strip()
        if not _PHONE_RE.match(to_number):
            looks_like_url = "://" in to_number
            hint = " (this looks like a URL, not a phone number)" if looks_like_url else ""
            return {
                "reply": AGENT_ERROR_SENTINEL,
                "trace": {
                    "error": (
                        f"voice_protocol 'twilio' requires agent.endpoint_url to be an "
                        f"E.164 phone number (e.g. +15551234567){hint}, got: {to_number or '(empty)'}"
                    )
                },
            }
        if not (TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN and TWILIO_FROM_NUMBER and PUBLIC_BASE_URL):
            return {
                "reply": AGENT_ERROR_SENTINEL,
                "trace": {"error": "Twilio is not configured (missing TWILIO_* / PUBLIC_BASE_URL env vars)"},
            }

        conv_key = uuid.uuid4().hex
        session = _Session(session_key=session_key, conv_key=conv_key)
        # Stored BEFORE the REST call is placed — see _Session's docstring.
        _SESSIONS_BY_KEY[session_key] = session
        _SESSIONS_BY_CONV_KEY[conv_key] = session

        try:
            client = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
            call = await asyncio.to_thread(
                client.calls.create, to=to_number, from_=TWILIO_FROM_NUMBER, url=_twiml_url(conv_key),
            )
            session.call_sid = call.sid
        except Exception as e:
            await _teardown(session)  # tombstone it — a later turn must fail fast, not retry
            return {"reply": AGENT_ERROR_SENTINEL, "trace": {"error": f"twilio call origination error: {e}"}}

    # --- TTS: this turn's text -> WAV -> mu-law frames (every turn) ------------
    try:
        audio_wav = await text_to_speech(message)
        frames = wav_to_mulaw_frames(audio_wav)
    except Exception as e:
        return {"reply": AGENT_ERROR_SENTINEL, "trace": {"error": f"tts error: {e}"}}

    request = _TurnRequest(
        frames=frames, result=asyncio.get_event_loop().create_future(), is_last_turn=is_last_turn,
    )
    session.current_turn = request
    session.turn_queue.put_nowait(request)  # picked up by handle_media_stream()

    trace: dict = {"call_sid": session.call_sid}
    try:
        outcome = await asyncio.wait_for(request.result, timeout=CALL_TIMEOUT_S)
    except asyncio.TimeoutError:
        await _teardown(session, error=f"turn timed out after {CALL_TIMEOUT_S}s")
        return {
            "reply": AGENT_ERROR_SENTINEL,
            "trace": {**trace, "error": f"twilio turn timed out after {CALL_TIMEOUT_S}s"},
        }
    finally:
        session.current_turn = None

    if outcome.get("error"):
        return {"reply": AGENT_ERROR_SENTINEL, "trace": {**trace, "error": outcome["error"]}}

    mulaw_reply = outcome.get("audio") or b""
    if not mulaw_reply:
        return {"reply": AGENT_ERROR_SENTINEL, "trace": {**trace, "error": "no audio received from voice agent"}}

    # --- STT: this turn's buffered mu-law reply -> WAV -> text -----------------
    try:
        wav_reply = mulaw_buffer_to_wav(mulaw_reply)
        transcript_out = await speech_to_text(wav_reply)
        if not transcript_out:
            raise ValueError("STT produced an empty transcript")
    except Exception as e:
        return {"reply": AGENT_ERROR_SENTINEL, "trace": {**trace, "error": f"stt error: {e}"}}

    return {"reply": transcript_out, "trace": trace}


# ---------------------------------------------------------------------------
# Media Stream event handling — called from app.routers.twilio's WebSocket route.
# All Twilio-protocol logic lives here; the router is a thin FastAPI adapter. This
# handler stays alive for the WHOLE call, processing one turn per loop iteration —
# turns are strictly serialized through `turn_queue`, so audio from one turn can
# never leak into another's buffer.
# ---------------------------------------------------------------------------
async def handle_media_stream(websocket: Any) -> None:
    """Drive one Twilio Media Stream connection for the WHOLE call it belongs to:
    wait for `start`, then repeatedly take the next queued turn, send its audio,
    buffer the reply until silence/timeout, resolve that turn's future, and loop —
    until told to stop (the last turn resolving, an explicit close, or the stream
    itself ending). `websocket` is a FastAPI WebSocket, already accepted by the
    caller.
    """
    session: Optional[_Session] = None

    try:
        # --- wait for "start" (which carries our correlation key) -------------
        while session is None:
            msg = await websocket.receive_json()
            event = msg.get("event")
            if event == "start":
                start = msg.get("start") or {}
                stream_sid = start.get("streamSid") or msg.get("streamSid")
                conv_key = (start.get("customParameters") or {}).get("conv_key")
                session = _SESSIONS_BY_CONV_KEY.get(conv_key) if conv_key else None
                if session is None:
                    # Unknown/stale/duplicate stream — nothing waiting on it.
                    await websocket.close()
                    return
                session.stream_sid = stream_sid
            elif event == "stop":
                # Call ended before we ever got a usable start — nothing to do.
                await websocket.close()
                return
            # "connected" and anything else: ignore and keep waiting for "start".

        # --- main loop: one turn per queue item, for the life of the call ------
        while True:
            request = await session.turn_queue.get()
            if request is _CLOSE_SENTINEL:
                break

            # send this turn's audio, paced at ~real time
            for frame in request.frames:
                await websocket.send_json({
                    "event": "media",
                    "streamSid": session.stream_sid,
                    "media": {"payload": _b64(frame)},
                })
                await asyncio.sleep(SEND_PACING_S)

            # buffer the agent's reply until silence/timeout — a FRESH buffer every
            # turn, so Turn 1's audio can never become part of Turn 2's response
            buffer = bytearray()
            silence_since: Optional[float] = None
            deadline = time.monotonic() + MAX_LISTEN_S
            stream_ended = False
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    msg = await asyncio.wait_for(websocket.receive_json(), timeout=remaining)
                except asyncio.TimeoutError:
                    break
                event = msg.get("event")
                if event == "media":
                    frame = _b64decode(msg["media"]["payload"])
                    buffer.extend(frame)
                    energy = _avg_abs_amplitude(mulaw_to_pcm16(frame))
                    if energy < SILENCE_RMS_THRESHOLD:
                        if silence_since is None:
                            silence_since = time.monotonic()
                        elif time.monotonic() - silence_since >= SILENCE_DURATION_S:
                            break
                    else:
                        silence_since = None
                elif event == "stop":
                    stream_ended = True
                    break
                # ignore "mark" and anything else

            if not request.result.done():
                request.result.set_result({"audio": bytes(buffer), "error": None})

            if stream_ended:
                await _teardown(session, error="media stream ended")
                break
            if request.is_last_turn:
                # Fast path: tear down immediately rather than waiting for
                # run_scenario()'s guaranteed close_fn to get around to it — that
                # call still happens (and is a no-op here, since _teardown is
                # idempotent), it's just no longer the thing doing the work.
                await _teardown(session)
                break
    except Exception as e:
        if session is not None:
            await _teardown(session, error=f"media stream error: {e}")
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _b64decode(data: str) -> bytes:
    return base64.b64decode(data)
