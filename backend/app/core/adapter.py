"""Black-box HTTP adapter — the ONLY way AgentShield touches an agent-under-test.

Given an `agents` row (endpoint_url + request_template + response_path + optional
auth_header), render the request, POST it, and pull `reply` + `trace` back out. It never
imports the agent's code — everything goes over HTTP. Errors degrade gracefully so one
bad turn can't crash a run.
"""
import json
from typing import Any, Optional

import httpx

from app.config import (
    AGENT_DESCRIPTIONS,
    INVENTORY,
    SAMPLE_AGENT_URLS,
    SAMPLE_RAG_BOT_URL,
    agent_description,
    agent_display_name,
)
from app.db import (
    get_agent_by_kind,
    get_agent_by_name,
    get_or_create_customer,
    get_or_create_customer_agent,
    insert_agent,
    update_agent_description,
)

# The reply a turn gets when the agent produced no usable reply — connection refused, bad
# status, timeout, unparseable body, or a failed voice transport. It is not agent output:
# the Judge scores it as a system failure. Defined here once; every transport and check
# imports it.
AGENT_ERROR_SENTINEL = "<error>"

# The standard request template for our sample RAG agent. Placeholders:
#   {message} -> JSON-escaped string (sits inside quotes)
#   {history} -> JSON array          (no quotes)
#   {faults}  -> JSON array          (no quotes)
DEFAULT_REQUEST_TEMPLATE = '{"message":"{message}","history":{history},"faults":{faults}}'

# For http_method="GET" agents: maps the semantic fields the adapter knows about to the
# query parameter names the target agent actually expects. A field left out of an agent's
# own query_param_map is simply not sent (e.g. a GET agent that only accepts ?message=...
# and has no concept of history/faults).
DEFAULT_QUERY_PARAM_MAP = {"message": "message", "history": "history", "faults": "faults"}

# Internal identifier of the sample RAG agent — matches its /health "service" value and
# its key in mapping.yaml.
SAMPLE_RAG_BOT_KEY = "sample_rag_bot"
DEFAULT_TIMEOUT_S = 30.0

# Agent-cooperative faults (only our sample RAG agent honors these — they travel in the
# request body and the agent decides how to behave).
AGENT_FAULTS = {"tool_timeout", "stale_doc", "injection"}

# System / transport faults. AgentShield simulates these itself at the HTTP layer, so they
# work against ANY agent (black box) — the endpoint never even gets a well-formed call.
SYSTEM_FAULTS = {"api_unreachable", "api_error", "api_timeout", "malformed_response"}

_SYSTEM_FAULT_TRACE = {
    "api_unreachable": "connection refused — agent endpoint unreachable",
    "api_error": "HTTP 500 — agent returned an internal server error",
    "api_timeout": "request timed out — agent did not respond in time",
    "malformed_response": "malformed response — agent returned invalid / non-JSON body",
}


def _render_body(template: str, message: str, history: list, faults: list) -> dict:
    """Substitute placeholders into the JSON template and parse to a dict.

    Falls back to a canonical body if the template can't be rendered/parsed, so a
    malformed custom template never kills the run.
    """
    try:
        body_str = (
            template
            .replace("{message}", json.dumps(message)[1:-1])   # escaped, no surrounding quotes
            .replace("{history}", json.dumps(history))
            .replace("{faults}", json.dumps(faults))
        )
        return json.loads(body_str)
    except (json.JSONDecodeError, ValueError):
        return {"message": message, "history": history, "faults": faults}


def _render_query_params(param_map: dict, message: str, history: list, faults: list) -> dict:
    """Build query params for a GET agent from its query_param_map.

    history/faults are JSON-encoded since query strings only carry text; message is sent
    as-is. Semantic fields the map doesn't mention are left out entirely.
    """
    values = {"message": message, "history": json.dumps(history), "faults": json.dumps(faults)}
    return {
        param_name: values[field]
        for field, param_name in param_map.items()
        if field in values and param_name
    }


def auth_headers(agent: Any) -> dict[str, str]:
    """The agent's `auth_header` — stored as one "Header: value" string — as a headers
    dict. Empty when unset or without both a name and a value. Shared by every transport."""
    raw = agent.get("auth_header")
    if not raw:
        return {}
    name, _, value = raw.partition(":")
    return {name.strip(): value.strip()} if name and value else {}


def _extract_by_path(data: Any, dot_path: str) -> Any:
    """Walk a dot-path like 'reply' or 'data.choices.0.text' through dicts/lists."""
    cur = data
    for part in dot_path.split("."):
        if isinstance(cur, list):
            try:
                cur = cur[int(part)]
            except (ValueError, IndexError):
                return None
        elif isinstance(cur, dict):
            cur = cur.get(part)
        else:
            return None
    return cur


async def send(
    agent: Any,  # an `agents` row (dict) with the agent columns
    message: str,
    history: Optional[list] = None,
    faults: Optional[list] = None,
) -> dict:
    """Send one turn to the agent. Returns {"reply": str, "trace": dict}.

    On timeout / HTTP / parse error, returns a sentinel reply + an error trace instead
    of raising — the runner records it like any other turn.
    """
    history = history or []
    faults = faults or []
    agent = dict(agent)  # accept a row or a plain dict uniformly

    # System/transport faults are simulated here — the agent is never called, so this works
    # for any black-box endpoint. Record the failure like a normal (bad) turn.
    sys_fault = next((f for f in faults if f in SYSTEM_FAULTS), None)
    if sys_fault:
        return {
            "reply": AGENT_ERROR_SENTINEL,
            "trace": {"error": _SYSTEM_FAULT_TRACE[sys_fault], "system_fault": sys_fault, "latency_ms": 0, "tokens": 0},
        }

    # Only agent-cooperative faults are forwarded to the agent's own API.
    faults = [f for f in faults if f in AGENT_FAULTS]

    response_path = agent.get("response_path") or "reply"
    url = agent.get("endpoint_url")
    http_method = (agent.get("http_method") or "POST").upper()
    headers = {"content-type": "application/json", **auth_headers(agent)}

    try:
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT_S) as client:
            if http_method == "GET":
                raw_map = agent.get("query_param_map")
                try:
                    param_map = json.loads(raw_map) if raw_map else DEFAULT_QUERY_PARAM_MAP
                except (json.JSONDecodeError, ValueError, TypeError):
                    param_map = DEFAULT_QUERY_PARAM_MAP
                params = _render_query_params(param_map, message, history, faults)
                resp = await client.get(url, params=params, headers=headers)
            else:
                template = agent.get("request_template") or DEFAULT_REQUEST_TEMPLATE
                body = _render_body(template, message, history, faults)
                resp = await client.post(url, json=body, headers=headers)
            resp.raise_for_status()
            data = resp.json()
    except httpx.TimeoutException:
        return {"reply": AGENT_ERROR_SENTINEL, "trace": {"error": "timeout"}}
    except httpx.HTTPStatusError as e:
        return {"reply": AGENT_ERROR_SENTINEL, "trace": {"error": f"http {e.response.status_code}"}}
    except Exception as e:  # connection refused, bad JSON, etc.
        return {"reply": AGENT_ERROR_SENTINEL, "trace": {"error": str(e)}}

    reply = _extract_by_path(data, response_path)
    if reply is None:
        reply = json.dumps(data)  # last resort: show what came back
    trace = data.get("trace", {}) if isinstance(data, dict) else {}
    return {"reply": str(reply), "trace": trace}


def seed_sample_agent() -> int:
    """Ensure the RAG customer-support agent exists as the sample `agents` row.

    Idempotent — returns the existing sample agent id if already seeded.
    """
    existing = get_agent_by_kind("sample")
    if existing:
        return existing["id"]
    return insert_agent(
        # Internal identifier -> actual name, resolved via backend/mapping.yaml.
        name=agent_display_name(SAMPLE_RAG_BOT_KEY),
        kind="sample",
        endpoint_url=SAMPLE_RAG_BOT_URL,
        response_path="reply",
        request_template=DEFAULT_REQUEST_TEMPLATE,
        # What it does, from backend/mapping.yaml's `descriptions`.
        description=agent_description(SAMPLE_RAG_BOT_KEY),
    )


def seed_inventory() -> int:
    """Seed customers + their agents from backend/inventory.yaml. Idempotent.

    One `agents` row per distinct agent (matched by name, so NorthBank Support is stored
    once) and one `customer_agents` row per customer-agent pair — which is what makes
    "NorthBank Support for Customer 1" and "…for Customer 2" separate testing contexts.

    Call AFTER seed_sample_agent(): that one claims the first kind='sample' row.
    """
    pairs = 0
    for customer in INVENTORY:
        customer_id = get_or_create_customer(customer["name"])
        for key in customer["agents"]:
            name = agent_display_name(key)
            agent = get_agent_by_name(name)
            if agent and key in AGENT_DESCRIPTIONS:
                # Seeding matches on name, so an already-seeded row keeps whatever
                # description it was created with — refresh it from mapping.yaml.
                update_agent_description(agent["id"], AGENT_DESCRIPTIONS[key])
            agent_id = agent["id"] if agent else insert_agent(
                name=name,
                kind="sample",
                endpoint_url=SAMPLE_AGENT_URLS.get(key, ""),
                response_path="reply",
                request_template=DEFAULT_REQUEST_TEMPLATE,
                description=agent_description(key),
            )
            get_or_create_customer_agent(customer_id, agent_id)
            pairs += 1
    print(f"[db] inventory seeded: {pairs} customer-agent combinations")
    return pairs
