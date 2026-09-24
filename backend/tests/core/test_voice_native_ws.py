"""Unit tests for app.core.voice_native_ws — the native_ws voice_protocol transport.

No real network, no real DB: httpx and websockets are replaced with the scripted
fakes in fakes_native_ws.py. Each test clears the module-level session registry
first, since it's process-global state keyed by session_key.
"""
import pytest

from app.core import voice_native_ws as nws
from tests.core.fakes_native_ws import FakeAsyncClient, FakeConnect, FakeWebSocket, turn_frame


@pytest.fixture(autouse=True)
def _clear_sessions():
    nws._SESSIONS.clear()
    yield
    nws._SESSIONS.clear()


def _agent(**overrides) -> dict:
    base = {
        "endpoint_url": "http://localhost:8000",
        "auth_header": "Authorization: Bearer qa-token",
        "request_template": '{"client": "acme_health"}',
    }
    base.update(overrides)
    return base


async def test_single_turn_captures_greeting_and_reply(monkeypatch):
    ws = FakeWebSocket(incoming=[
        turn_frame("agent", "Hi, this is the scheduling line."),  # unprompted greeting
        turn_frame("caller", "I need an appointment"),            # echo of the tester's line
        turn_frame("agent", "Sure, what day works?"),              # the actual reply
    ])
    connect = FakeConnect([ws])
    calls_log: list[dict] = []
    http_client = FakeAsyncClient(calls_log, call_id="call-abc")

    monkeypatch.setattr(nws.websockets, "connect", connect)
    monkeypatch.setattr(nws.httpx, "AsyncClient", http_client.factory())

    result = await nws._call_via_native_ws(
        _agent(), "I need an appointment", session_key="conv-1",
    )

    assert result["reply"] == "Sure, what day works?"
    assert result["trace"]["opening_line"] == "Hi, this is the scheduling line."
    assert result["trace"]["call_id"] == "call-abc"

    # Create-call body came from request_template, with the auth header attached.
    create_call = calls_log[0]
    assert create_call["url"] == "http://localhost:8000/api/calls"
    assert create_call["json"] == {"client": "acme_health"}
    assert create_call["headers"] == {"Authorization": "Bearer qa-token"}

    # The tester's line went out as a simulated_utterance, not raw text.
    assert ws.sent == ['{"type": "simulated_utterance", "text": "I need an appointment"}']

    # WS URL is the HTTP base, scheme-swapped, with the returned call_id.
    assert connect.urls == ["ws://localhost:8000/api/ws/call-abc"]


async def test_second_turn_omits_opening_line_and_reuses_connection(monkeypatch):
    ws = FakeWebSocket(incoming=[
        turn_frame("agent", "Hi there."),
        turn_frame("caller", "hello"),
        turn_frame("agent", "How can I help?"),
        turn_frame("caller", "book me for Tuesday"),
        turn_frame("agent", "Tuesday it is."),
    ])
    connect = FakeConnect([ws])
    calls_log: list[dict] = []
    http_client = FakeAsyncClient(calls_log, call_id="call-xyz")
    monkeypatch.setattr(nws.websockets, "connect", connect)
    monkeypatch.setattr(nws.httpx, "AsyncClient", http_client.factory())

    first = await nws._call_via_native_ws(_agent(), "hello", session_key="conv-2")
    second = await nws._call_via_native_ws(_agent(), "book me for Tuesday", session_key="conv-2")

    assert first["reply"] == "How can I help?"
    assert first["trace"]["opening_line"] == "Hi there."
    assert second["reply"] == "Tuesday it is."
    assert "opening_line" not in second["trace"]  # only the first turn carries it

    # Exactly one call created and one socket opened for the whole scenario.
    create_calls = [c for c in calls_log if c["url"].endswith("/api/calls")]
    assert len(create_calls) == 1
    assert connect.call_count == 1
    assert ws.sent == [
        '{"type": "simulated_utterance", "text": "hello"}',
        '{"type": "simulated_utterance", "text": "book me for Tuesday"}',
    ]


async def test_new_scenario_gets_its_own_call_id_and_connection(monkeypatch):
    ws_a = FakeWebSocket(incoming=[turn_frame("agent", "hi A"), turn_frame("caller", "x"), turn_frame("agent", "reply A")])
    ws_b = FakeWebSocket(incoming=[turn_frame("agent", "hi B"), turn_frame("caller", "y"), turn_frame("agent", "reply B")])
    connect = FakeConnect([ws_a, ws_b])
    calls_log: list[dict] = []
    ids = iter(["call-A", "call-B"])

    class _SeqClient:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, json=None, headers=None):
            calls_log.append({"url": url, "json": json, "headers": headers})
            return _Resp({"call_id": next(ids)})

    class _Resp:
        def __init__(self, data):
            self._data = data

        def raise_for_status(self):
            pass

        def json(self):
            return self._data

    monkeypatch.setattr(nws.websockets, "connect", connect)
    monkeypatch.setattr(nws.httpx, "AsyncClient", _SeqClient)

    result_a = await nws._call_via_native_ws(_agent(), "x", session_key="scenario-A")
    result_b = await nws._call_via_native_ws(_agent(), "y", session_key="scenario-B")

    assert result_a["trace"]["call_id"] == "call-A"
    assert result_b["trace"]["call_id"] == "call-B"
    assert connect.call_count == 2
    assert connect.urls == [
        "ws://localhost:8000/api/ws/call-A",
        "ws://localhost:8000/api/ws/call-B",
    ]


async def test_close_sends_code_1000_and_calls_end_endpoint(monkeypatch):
    ws = FakeWebSocket(incoming=[turn_frame("agent", "hi"), turn_frame("caller", "x"), turn_frame("agent", "reply")])
    connect = FakeConnect([ws])
    calls_log: list[dict] = []
    http_client = FakeAsyncClient(calls_log, call_id="call-close")
    monkeypatch.setattr(nws.websockets, "connect", connect)
    monkeypatch.setattr(nws.httpx, "AsyncClient", http_client.factory())

    await nws._call_via_native_ws(_agent(), "x", session_key="conv-close")
    await nws.close_native_ws_session("conv-close")

    assert ws.closed is True
    assert ws.close_args[0] == 1000
    end_calls = [c for c in calls_log if c["url"].endswith("/end")]
    assert len(end_calls) == 1
    assert end_calls[0]["url"] == "http://localhost:8000/api/calls/call-close/end"
    assert "conv-close" not in nws._SESSIONS


async def test_close_is_idempotent(monkeypatch):
    ws = FakeWebSocket(incoming=[turn_frame("agent", "hi"), turn_frame("caller", "x"), turn_frame("agent", "reply")])
    connect = FakeConnect([ws])
    calls_log: list[dict] = []
    http_client = FakeAsyncClient(calls_log, call_id="call-idem")
    monkeypatch.setattr(nws.websockets, "connect", connect)
    monkeypatch.setattr(nws.httpx, "AsyncClient", http_client.factory())

    await nws._call_via_native_ws(_agent(), "x", session_key="conv-idem")
    await nws.close_native_ws_session("conv-idem")
    await nws.close_native_ws_session("conv-idem")  # must not raise, must not double-call /end

    end_calls = [c for c in calls_log if c["url"].endswith("/end")]
    assert len(end_calls) == 1

    # Closing a session_key that never had a session at all is also a no-op.
    await nws.close_native_ws_session("never-existed")


async def test_setup_failure_tombstones_session_no_retry_on_next_turn(monkeypatch):
    connect = FakeConnect([])  # never reached — create-call fails first
    calls_log: list[dict] = []
    http_client = FakeAsyncClient(calls_log, fail_create=True)
    monkeypatch.setattr(nws.websockets, "connect", connect)
    monkeypatch.setattr(nws.httpx, "AsyncClient", http_client.factory())

    first = await nws._call_via_native_ws(_agent(), "hello", session_key="conv-fail")
    second = await nws._call_via_native_ws(_agent(), "again", session_key="conv-fail")

    assert first["reply"] == nws.AGENT_ERROR_SENTINEL
    assert "native_ws call setup error" in first["trace"]["error"]
    assert second["reply"] == nws.AGENT_ERROR_SENTINEL
    assert "already ended" in second["trace"]["error"]

    # Only ONE create-call attempt ever happened, despite two turns.
    create_calls = [c for c in calls_log if c["url"].endswith("/api/calls")]
    assert len(create_calls) == 1
    assert connect.call_count == 0


async def test_turn_failure_tombstones_session_no_reconnect_on_next_turn(monkeypatch):
    ws = FakeWebSocket(incoming=[turn_frame("agent", "hi")])  # nothing left after the greeting
    connect = FakeConnect([ws])
    calls_log: list[dict] = []
    http_client = FakeAsyncClient(calls_log, call_id="call-turnfail")
    monkeypatch.setattr(nws.websockets, "connect", connect)
    monkeypatch.setattr(nws.httpx, "AsyncClient", http_client.factory())

    # First turn: the socket has nothing queued after the greeting, so recv() in
    # _next_agent_turn raises (FakeWebSocket's "no more scripted frames").
    first = await nws._call_via_native_ws(_agent(), "hello", session_key="conv-turnfail")
    second = await nws._call_via_native_ws(_agent(), "again", session_key="conv-turnfail")

    assert first["reply"] == nws.AGENT_ERROR_SENTINEL
    assert "native_ws turn error" in first["trace"]["error"]
    assert second["reply"] == nws.AGENT_ERROR_SENTINEL
    assert "already ended" in second["trace"]["error"]

    # No second connection was ever attempted after the first turn's failure.
    assert connect.call_count == 1


async def test_peek_greeting_returns_and_clears_the_opening_line(monkeypatch):
    ws = FakeWebSocket(incoming=[turn_frame("agent", "Hi, this is Maya calling about your registration.")])
    connect = FakeConnect([ws])
    calls_log: list[dict] = []
    http_client = FakeAsyncClient(calls_log, call_id="call-peek")
    monkeypatch.setattr(nws.websockets, "connect", connect)
    monkeypatch.setattr(nws.httpx, "AsyncClient", http_client.factory())

    greeting = await nws.peek_greeting(_agent(), "conv-peek")

    assert greeting == "Hi, this is Maya calling about your registration."
    # Reading it via peek must not leave it to also show up as a nested "opening_line".
    session = nws._SESSIONS["conv-peek"]
    assert session.opening_line is None
    assert connect.call_count == 1


async def test_peek_greeting_then_real_turn_share_one_connection_and_dont_duplicate_it(monkeypatch):
    ws = FakeWebSocket(incoming=[
        turn_frame("agent", "Hi, this is Maya."),
        turn_frame("caller", "hello"),
        turn_frame("agent", "How can I help?"),
    ])
    connect = FakeConnect([ws])
    calls_log: list[dict] = []
    http_client = FakeAsyncClient(calls_log, call_id="call-shared")
    monkeypatch.setattr(nws.websockets, "connect", connect)
    monkeypatch.setattr(nws.httpx, "AsyncClient", http_client.factory())

    greeting = await nws.peek_greeting(_agent(), "conv-shared")
    turn = await nws._call_via_native_ws(_agent(), "hello", session_key="conv-shared")

    assert greeting == "Hi, this is Maya."
    assert turn["reply"] == "How can I help?"
    assert "opening_line" not in turn["trace"]  # peek already consumed it — no duplicate
    # ONE connection for the whole scenario, not one for the peek and another for the turn.
    assert connect.call_count == 1
    create_calls = [c for c in calls_log if c["url"].endswith("/api/calls")]
    assert len(create_calls) == 1


async def test_peek_greeting_returns_none_on_setup_failure(monkeypatch):
    connect = FakeConnect([])
    calls_log: list[dict] = []
    http_client = FakeAsyncClient(calls_log, fail_create=True)
    monkeypatch.setattr(nws.websockets, "connect", connect)
    monkeypatch.setattr(nws.httpx, "AsyncClient", http_client.factory())

    greeting = await nws.peek_greeting(_agent(), "conv-peek-fail")

    assert greeting is None
    # Tombstoned exactly like a failed real turn would be — a later call must fail fast.
    assert nws._SESSIONS["conv-peek-fail"].closed is True


async def test_peek_greeting_returns_none_once_session_already_tombstoned(monkeypatch):
    nws._SESSIONS["conv-dead"] = nws._Session(closed=True)
    greeting = await nws.peek_greeting(_agent(), "conv-dead")
    assert greeting is None


# ---------------------------------------------------------------------------
# Binary frames must not break turn detection (the conversation-recording feature
# that used to capture them has been removed — see _next_agent_turn's docstring).
# ---------------------------------------------------------------------------

async def test_next_agent_turn_skips_binary_and_non_turn_text_frames():
    """Caller echo, speaking/interruption text frames, and any binary frame the
    target sends must all be silently skipped — only the "turn"/agent text event
    stops the loop."""
    import json

    ws = FakeWebSocket(incoming=[
        turn_frame("caller", "echo of what the tester said"),
        json.dumps({"type": "speaking", "speaking": True}),
        b"some-binary-frame-the-target-sends",
        turn_frame("agent", "the real reply"),
    ])
    result = await nws._next_agent_turn(ws)
    assert result["text"] == "the real reply"
