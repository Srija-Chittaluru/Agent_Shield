"""/agents API — list the seeded sample agent + register your own (black-box) agent."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core.adapter import AGENT_ERROR_SENTINEL, DEFAULT_REQUEST_TEMPLATE
from app.core.mock_agent import MOCK_KIND
from app.db import (
    get_agent,
    get_agent_by_name_and_endpoint,
    insert_agent,
    list_agents,
    set_agent_knowledge,
    update_agent_connection,
    update_agent_description,
)

router = APIRouter(prefix="/agents", tags=["agents"])


class RegisterAgent(BaseModel):
    name: str
    # Required for a real agent; left empty for a dry run, which has no endpoint.
    endpoint_url: str = ""
    response_path: str = "reply"
    request_template: str = DEFAULT_REQUEST_TEMPLATE
    auth_header: str | None = None
    description: str | None = None
    # "chat" (default, text/HTTP) or "voice" — which execution path the Temporal
    # workflow uses for runs against this agent.
    modality: str = "chat"
    # Which wire protocol a voice-modality agent speaks. Only "http_json" (the
    # existing TTS -> HTTP -> STT contract) is implemented; irrelevant for chat.
    voice_protocol: str = "http_json"
    # "POST" (default, unchanged behavior) or "GET" for black-box agents whose API only
    # accepts GET. GET agents ignore request_template and use query_param_map instead.
    http_method: str = "POST"
    # Only used when http_method="GET". JSON string mapping "message"/"history"/"faults"
    # to the query parameter names the target agent expects, e.g. '{"message":"q"}'.
    query_param_map: str | None = None
    # Whether the agent reports flow signals (answers / interrupt_key / end_status) in its
    # trace. None = not declared: the Judge decides per conversation from the trace.
    reports_flow_signals: bool | None = None
    # Only for voice_protocol="native_ws": JSON object overriding any part of its wire
    # format (see app.core.voice_native_ws.NATIVE_WS_DEFAULTS). None = the defaults.
    protocol_config: str | None = None
    # Dry run: store the agent as kind="mock" — an LLM plays it from `system_prompt`
    # (or, without one, its description + knowledge) instead of calling an endpoint.
    simulated: bool = False
    system_prompt: str | None = None


class AgentKnowledge(BaseModel):
    """The agent's docs, read as text in the browser on the Connect step."""
    text: str
    name: str | None = None


def _row_to_dict(row) -> dict:
    r = dict(row)
    return {
        "id": r["id"], "name": r["name"], "kind": r["kind"],
        "modality": r.get("modality") or "chat",
        "voice_protocol": r.get("voice_protocol") or "http_json",
        "endpoint_url": r["endpoint_url"], "response_path": r["response_path"],
        "http_method": r.get("http_method") or "POST",
        "query_param_map": r.get("query_param_map"),
        "reports_flow_signals": r.get("reports_flow_signals"),
        "protocol_config": r.get("protocol_config"),
        "description": r["description"], "created_at": r["created_at"],
        "simulated": r["kind"] == MOCK_KIND,
        "system_prompt": r.get("system_prompt"),
        "knowledge_name": r.get("knowledge_name"),
        "knowledge_chars": len(r.get("knowledge") or ""),
    }


@router.get("")
def get_agents() -> dict:
    return {"agents": [_row_to_dict(a) for a in list_agents()]}


@router.get("/{agent_id}")
def get_one(agent_id: int) -> dict:
    a = get_agent(agent_id)
    if a is None:
        raise HTTPException(status_code=404, detail=f"agent {agent_id} not found")
    return _row_to_dict(a)


@router.post("")
def register(body: RegisterAgent) -> dict:
    """Register a custom (black-box) agent. Faults won't be injected; trace = whatever it returns.

    Re-registering the same name+endpoint returns the existing agent instead of inserting a
    duplicate, so reconnecting keeps the agent's customer context and saved test cases — but
    every OTHER connection detail (auth header, protocol, request template/create-call body,
    http method) is refreshed from this submission too, so correcting a wrong API key or
    protocol on a second attempt actually takes effect instead of being silently discarded.
    """
    if body.simulated:
        # A dry-run agent has no endpoint to call. Blank it so re-registering the same
        # name finds this row again instead of forking a duplicate.
        body.endpoint_url = ""
    elif not body.endpoint_url.strip():
        raise HTTPException(status_code=422, detail="endpoint_url is required (or choose a dry run)")

    if body.protocol_config:
        from app.core.voice_native_ws import protocol_config

        try:
            protocol_config({"protocol_config": body.protocol_config})
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))

    existing = get_agent_by_name_and_endpoint(body.name, body.endpoint_url)
    if existing:
        update_agent_connection(
            existing["id"],
            auth_header=body.auth_header,
            request_template=body.request_template,
            response_path=body.response_path,
            modality=body.modality,
            voice_protocol=body.voice_protocol,
            http_method=body.http_method,
            query_param_map=body.query_param_map,
            reports_flow_signals=body.reports_flow_signals,
            protocol_config=body.protocol_config,
            system_prompt=body.system_prompt,
        )
        if body.description:
            update_agent_description(existing["id"], body.description)
        return {"agent_id": existing["id"]}

    agent_id = insert_agent(
        name=body.name,
        kind=MOCK_KIND if body.simulated else "custom",
        endpoint_url=body.endpoint_url,
        response_path=body.response_path,
        request_template=body.request_template,
        auth_header=body.auth_header,
        description=body.description,
        modality=body.modality,
        voice_protocol=body.voice_protocol,
        http_method=body.http_method,
        query_param_map=body.query_param_map,
        reports_flow_signals=body.reports_flow_signals,
        protocol_config=body.protocol_config,
        system_prompt=body.system_prompt,
    )
    return {"agent_id": agent_id}


@router.post("/{agent_id}/probe")
async def probe(agent_id: int) -> dict:
    """Real connectivity check: send one trivial message through the agent's actual
    transport and report back.

    Powers the UI's 'Verify Connection' step — it genuinely reaches the endpoint.
    Chat agents go through app.core.adapter.send() directly, unchanged. Voice agents
    go through app.core.voice_caller.call_voice_agent() instead — the SAME dispatcher
    a real run uses — so this probes whatever voice_protocol the agent is actually
    configured with (http_json's TTS/STT round-trip, a one-shot websocket turn, or a
    real native_ws call-session), rather than a raw HTTP POST no voice contract
    actually accepts. A one-off session_key is used and always closed, so a
    persistent-session protocol (native_ws) never leaks a live call past this probe.
    """
    a = get_agent(agent_id)
    if a is None:
        raise HTTPException(status_code=404, detail=f"agent {agent_id} not found")

    message = "Connectivity test from AgentShield. Please reply 'ok'."
    if (a.get("modality") or "chat") == "voice":
        import uuid

        from app.core.voice_caller import call_voice_agent, close_voice_session

        session_key = f"probe:{agent_id}:{uuid.uuid4().hex}"
        try:
            result = await call_voice_agent(
                a, message, [], [], session_key=session_key, is_last_turn=True,
            )
        finally:
            await close_voice_session(a, session_key)
    else:
        from app.core.adapter import send

        result = await send(a, message, [], [])

    reply = result.get("reply", "")
    trace = result.get("trace", {}) or {}
    ok = reply != AGENT_ERROR_SENTINEL and not trace.get("error")
    return {"ok": ok, "reply": reply[:300], "error": trace.get("error")}


@router.post("/{agent_id}/discover")
async def discover(agent_id: int) -> dict:
    """Auto-discover what the agent does by probing it; save + return the inferred description.

    Used when the user uploads no docs and gives no description — AgentShield figures it out.
    """
    from app.core.discover import discover_agent

    a = get_agent(agent_id)
    if a is None:
        raise HTTPException(status_code=404, detail=f"agent {agent_id} not found")
    description = await discover_agent(a)
    update_agent_description(agent_id, description)
    return {"description": description}


@router.post("/{agent_id}/knowledge")
def save_knowledge(agent_id: int, body: AgentKnowledge) -> dict:
    """Attach the agent's docs, so every later run for it generates grounded test cases.

    Takes the text the browser already read from the file — no server-side file parsing.
    """
    if get_agent(agent_id) is None:
        raise HTTPException(status_code=404, detail=f"agent {agent_id} not found")
    set_agent_knowledge(agent_id, body.text, body.name)
    return {"agent_id": agent_id, "knowledge_name": body.name, "chars": len(body.text)}
