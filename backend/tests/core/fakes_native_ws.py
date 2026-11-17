"""Test doubles for app.core.voice_native_ws — no real network, no real DB.

Shared by test_voice_native_ws.py (unit-level) and test_run_scenario_native_ws.py
(full run_scenario() integration).
"""
import json
from typing import Any, Optional


def turn_frame(role: str, text: str, **extra: Any) -> str:
    """One server -> client "turn" JSON text frame, as the target agent sends it."""
    payload = {"type": "turn", "role": role, "text": text, **extra}
    return json.dumps(payload)


class FakeWebSocket:
    """A scripted stand-in for a websockets client connection.

    `incoming` is consumed strictly in order by `recv()`, regardless of when
    `send()` is called — good enough for driving a fixed turn-by-turn script
    without needing real concurrency.
    """

    def __init__(self, incoming: Optional[list[str]] = None):
        self._incoming = list(incoming or [])
        self.sent: list[str] = []
        self.closed = False
        self.close_args: Optional[tuple] = None

    async def send(self, data: str) -> None:
        self.sent.append(data)

    async def recv(self) -> str:
        if not self._incoming:
            raise RuntimeError("FakeWebSocket: no more scripted frames (socket closed)")
        return self._incoming.pop(0)

    async def close(self, code: int = 1000, reason: str = "") -> None:
        self.closed = True
        self.close_args = (code, reason)

    def push(self, msg: str) -> None:
        self._incoming.append(msg)


class FakeConnect:
    """Stands in for `websockets.connect`. Returns one FakeWebSocket per call, in
    the order given by `sockets`, so a test can assert exactly how many distinct
    connections were opened (one per scenario, reused across that scenario's turns).
    """

    def __init__(self, sockets: list[FakeWebSocket], fail: bool = False):
        self._sockets = list(sockets)
        self.fail = fail
        self.urls: list[str] = []
        self.call_count = 0

    async def __call__(self, url: str, *args: Any, **kwargs: Any) -> FakeWebSocket:
        self.call_count += 1
        self.urls.append(url)
        if self.fail:
            raise RuntimeError("connection refused")
        return self._sockets.pop(0)


class _FakeResponse:
    def __init__(self, data: dict, status_code: int = 200):
        self._data = data
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"http {self.status_code}")

    def json(self) -> dict:
        return self._data


class FakeAsyncClient:
    """Stands in for httpx.AsyncClient — logs every POST, and answers
    POST ...`create_path` with `call_id` (or raises, if `fail_create`). `create_response`
    replaces that answer's body, for an agent whose response is shaped differently.
    """

    calls_log: list[dict] = []  # overwritten per-instance in __init__; class attr is just a type hint

    def __init__(
        self,
        calls_log: list[dict],
        call_id: str = "call-1",
        fail_create: bool = False,
        create_path: str = "/api/calls",
        create_response: Optional[dict] = None,
    ):
        self.calls_log = calls_log
        self.call_id = call_id
        self.fail_create = fail_create
        self.create_path = create_path
        self.create_response = create_response

    def factory(self):
        """Return a callable usable as `httpx.AsyncClient` (i.e. `Client(timeout=...)`)."""
        outer = self

        class _Client:
            def __init__(self, *a: Any, **kw: Any) -> None:
                pass

            async def __aenter__(self) -> "_Client":
                return self

            async def __aexit__(self, *a: Any) -> bool:
                return False

            async def post(self, url: str, json: Optional[dict] = None, headers: Optional[dict] = None):
                outer.calls_log.append({"url": url, "json": json, "headers": headers})
                if url.endswith(outer.create_path):
                    if outer.fail_create:
                        raise RuntimeError("connection refused")
                    return _FakeResponse(outer.create_response or {"call_id": outer.call_id})
                return _FakeResponse({"ok": True})

        return _Client
