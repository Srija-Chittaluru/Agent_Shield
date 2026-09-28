"""/flows API — upload/parse a Voice Agent's flow (Phase 1), generate a node's test
goal + deterministic script (Phase 2), and save/run it (Phase 3).

Phase 3 turns a reviewed node script into a REAL AgentShield test case and sends it
through the existing Temporal execution pipeline (app.temporal.workflows) — the exact
same AgentTestWorkflow/RunGroupWorkflow, activities, scripted runner
(app.core.runner._run_scripted, unmodified), voice protocols, recording, and Judge
every other test uses. This module adds no second voice-testing engine; see
app.core.node_script.to_scenario_dict for how a script becomes that existing shape.
"""
import hashlib
import json as _json
import re

import yaml
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.config import TEMPORAL_TASK_QUEUE, WORK_CONCURRENCY
from app.core.flow_graph import (
    BRANCH,
    DEFAULT_MAX_SCENARIOS,
    HAPPY_PATH,
    HARD_MAX_SCENARIOS,
    PRIMARY_PATH,
    RECOVERY,
    RETRY,
    TERMINAL,
    analyze_graph,
    generate_candidate_paths,
)
from app.core.flow_llm_extractor import FlowExtractionError, extract_flow_with_llm
from app.core.flow_parser import FlowAmbiguousError, FlowParseError, is_modular_step_flow, parse_flow
from app.core.flow_scenario_planner import plan_flow_scenarios
from app.core.voice_native_ws import call_template_variables, check_call_variables
from app.core.node_script import (
    NodeScriptError,
    generate_interrupt_script,
    generate_node_script,
    generate_path_script,
    interrupt_script_to_scenario,
    path_script_to_scenario,
    prerequisite_path,
    to_scenario_dict,
    validate_interrupt_script,
    validate_path_script,
    validate_script,
)
from app.core.scenarios import normalize_scenarios
from app.temporal.client import get_client
from app.temporal.workflows import AgentTestInput, RunGroupInput, RunGroupWorkflow
from app.db import (
    default_customer_agent,
    delete_flow_scenario,
    get_agent,
    get_agent_flow,
    get_flow_scenario,
    get_customer_agent_by_agent_id,
    get_test_case,
    insert_agent_flow,
    insert_flow_scenarios,
    insert_run,
    insert_run_group,
    insert_test_case,
    list_agent_flows,
    list_flow_scenarios,
    update_run,
    update_test_case,
    upsert_flow_scenario_script,
)

router = APIRouter(tags=["flows"])


class UploadFlow(BaseModel):
    """The file's contents, read as text in the browser — same convention as
    POST /agents/{id}/knowledge. Unlike knowledge, this content IS actually parsed
    and validated server-side; see app.core.flow_parser.
    """
    content: str
    filename: str | None = None
    # Explicit override; if omitted, inferred from filename extension, falling back to
    # trying JSON then YAML.
    source_format: str | None = None


def _format_hint(filename: str | None, explicit: str | None) -> str | None:
    if explicit:
        return explicit
    if filename:
        lower = filename.lower()
        if lower.endswith((".yaml", ".yml")):
            return "yaml"
        if lower.endswith(".json"):
            return "json"
    return None


@router.post("/agents/{agent_id}/flows")
async def upload_flow(agent_id: int, body: UploadFlow) -> dict:
    """Parse+validate an uploaded flow definition and store it as a new version.

    Deterministic extraction (app.core.flow_parser) is tried first, free and instant.
    Only when it can't confidently find a node collection at all (FlowAmbiguousError)
    does this fall back to LLM-assisted extraction (app.core.flow_llm_extractor) — and
    that output is validated with the exact same rules before being trusted. If neither
    path can identify a flow, nothing is persisted and the diagnostics collected along
    the way are returned so the user can see WHY, instead of a bare schema complaint.
    """
    if get_agent(agent_id) is None:
        raise HTTPException(status_code=404, detail=f"agent {agent_id} not found")

    fmt = _format_hint(body.filename, body.source_format)
    try:
        parsed = parse_flow(body.content, fmt)
    except FlowAmbiguousError as e:
        try:
            parsed = await extract_flow_with_llm(body.content, fmt or "json")
        except FlowExtractionError as llm_error:
            raise HTTPException(
                status_code=422,
                detail={
                    "message": (
                        "AgentShield could not confidently identify conversation "
                        "nodes in this file."
                    ),
                    "errors": llm_error.errors,
                    "diagnostics": e.diagnostics,
                },
            ) from llm_error
    except FlowParseError as e:
        raise HTTPException(status_code=422, detail={"errors": e.errors}) from e

    name = body.filename or parsed["agent_name"]
    flow_id = insert_agent_flow(
        agent_id=agent_id, name=name, source_format=parsed["source_format"],
        raw_source=body.content, nodes=parsed["nodes"], edges=parsed["edges"],
        extraction_method=parsed["extraction_method"],
    )
    return {
        "flow_id": flow_id, "agent_id": agent_id, "name": name,
        "agent_name": parsed["agent_name"], "source_format": parsed["source_format"],
        "extraction_method": parsed["extraction_method"],
        "nodes": parsed["nodes"], "edges": parsed["edges"],
    }


@router.get("/agents/{agent_id}/flows")
def list_flows(agent_id: int) -> dict:
    if get_agent(agent_id) is None:
        raise HTTPException(status_code=404, detail=f"agent {agent_id} not found")
    return {
        "flows": [
            {
                "id": f["id"], "agent_id": f["agent_id"], "name": f["name"],
                "source_format": f["source_format"], "created_at": f["created_at"],
                "extraction_method": f.get("extraction_method") or "deterministic",
            }
            for f in list_agent_flows(agent_id)
        ]
    }


@router.get("/flows/{flow_id}")
def get_flow(flow_id: int) -> dict:
    row = get_agent_flow(flow_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"flow {flow_id} not found")
    return {
        "id": row["id"], "agent_id": row["agent_id"], "name": row["name"],
        "source_format": row["source_format"], "created_at": row["created_at"],
        "extraction_method": row.get("extraction_method") or "deterministic",
        "nodes": _json.loads(row["nodes_json"]), "edges": _json.loads(row["edges_json"]),
    }


class GenerateNodeScript(BaseModel):
    """`test_goal` is an optional user-provided seed intent for the test — e.g. "test an
    incorrect name before the correct one". Omit it to have the node's own purpose/
    expected_inputs drive what gets generated.
    """
    test_goal: str | None = None


def _find_node(nodes: list[dict], node_id: str) -> dict | None:
    return next((n for n in nodes if n["id"] == node_id), None)


@router.post("/flows/{flow_id}/nodes/{node_id}/script")
async def generate_script(flow_id: int, node_id: str, body: GenerateNodeScript) -> dict:
    """Generate a test goal + deterministic caller script for one node (Phase 2).

    Draft only — nothing here is persisted to scenarios/test_cases (that's the next
    phase) and nothing here executes anything against the Voice Agent. Prerequisite
    nodes (per the flow's edges) are woven into the script as plain pass-through turns
    ahead of this node's own turns; see app.core.node_script.
    """
    flow = get_agent_flow(flow_id)
    if flow is None:
        raise HTTPException(status_code=404, detail=f"flow {flow_id} not found")

    nodes = _json.loads(flow["nodes_json"])
    edges = _json.loads(flow["edges_json"])
    node = _find_node(nodes, node_id)
    if node is None:
        raise HTTPException(
            status_code=404, detail=f"node '{node_id}' not found in flow {flow_id}"
        )

    prereqs = prerequisite_path(nodes, edges, node_id)

    try:
        generated = await generate_node_script(node, prereqs, body.test_goal)
    except NodeScriptError as e:
        raise HTTPException(status_code=422, detail={"errors": e.errors}) from e

    return {
        "flow_id": flow_id, "node_id": node_id, "node_name": node["name"],
        "test_goal": generated["test_goal"], "script": generated["script"],
    }


def _resolve_customer_agent_id(agent_id: int) -> int:
    """Which customer-agent context a flow-node test for this agent belongs under.

    Prefers a genuine existing one (e.g. from inventory.yaml) over minting a new
    "New Customer N" — default_customer_agent only ever looks for/creates the latter,
    which would otherwise fork an agent that already has a real customer context into
    two separate places its test cases live.
    """
    existing = get_customer_agent_by_agent_id(agent_id)
    if existing:
        return existing["id"]
    return default_customer_agent(agent_id)


class SaveNodeTest(BaseModel):
    """The reviewed test goal + script for one node, ready to persist (Phase 3)."""
    test_goal: str
    script: list[dict]
    # Set when re-saving a script this node's workspace already saved once — updates
    # that row in place instead of inserting a duplicate. Omitted/None on first save.
    test_id: int | None = None


@router.post("/flows/{flow_id}/nodes/{node_id}/test")
def save_node_test(flow_id: int, node_id: str, body: SaveNodeTest) -> dict:
    """Persist a reviewed node script as a real test_cases row.

    Same table, same columns (title/user_goal/test_type/assigned_fault/
    expected_behavior/seed_turns_json/source) every other test case uses, plus the
    flow_id/node_id/node_script_json columns added for this feature — see
    app.core.node_script.to_scenario_dict for the conversion. Without `test_id` this is
    a single INSERT that never touches any other test case already saved for this
    agent (contrast the dashboard's "save reviewed suite", which replaces the whole
    set). With `test_id`, it UPDATEs that same row in place instead — re-saving an
    edited/regenerated script for a node the workspace already saved once must not
    silently accumulate duplicate test cases.

    Rejects a malformed script outright (missing/empty caller_line or
    expected_agent_behavior, non-string fields, empty or oversized script) — never
    silently repaired.
    """
    flow = get_agent_flow(flow_id)
    if flow is None:
        raise HTTPException(status_code=404, detail=f"flow {flow_id} not found")

    nodes = _json.loads(flow["nodes_json"])
    node = _find_node(nodes, node_id)
    if node is None:
        raise HTTPException(
            status_code=404, detail=f"node '{node_id}' not found in flow {flow_id}"
        )

    if not body.test_goal or not body.test_goal.strip():
        raise HTTPException(
            status_code=422, detail={"errors": ["test_goal must be a non-empty string."]}
        )

    try:
        script = validate_script(body.script)
    except NodeScriptError as e:
        raise HTTPException(status_code=422, detail={"errors": e.errors}) from e

    raw = to_scenario_dict(flow_id, node_id, node["name"], body.test_goal.strip(), script)
    normalized_list = normalize_scenarios([raw])
    if not normalized_list:
        raise HTTPException(status_code=500, detail="failed to normalize the reviewed test case")
    normalized = normalized_list[0]
    # normalize_scenarios() only knows the fields every OTHER scenario type has, so it
    # drops these three — reattach them from the pre-normalization dict.
    normalized["flow_id"] = flow_id
    normalized["node_id"] = node_id
    normalized["node_script_json"] = raw["node_script_json"]

    if body.test_id is not None:
        existing = get_test_case(body.test_id)
        if (
            existing is None
            or existing.get("flow_id") != flow_id
            or existing.get("node_id") != node_id
        ):
            raise HTTPException(
                status_code=404,
                detail=f"test {body.test_id} not found for flow {flow_id}, node '{node_id}'",
            )
        update_test_case(body.test_id, normalized, node_script_json=normalized["node_script_json"])
        test_id = body.test_id
        customer_agent_id = existing["customer_agent_id"]
    else:
        customer_agent_id = _resolve_customer_agent_id(flow["agent_id"])
        test_id = insert_test_case(
            customer_agent_id, normalized, source="user",
            flow_id=flow_id, node_id=node_id, node_script_json=normalized["node_script_json"],
        )

    return {
        "test_id": test_id, "flow_id": flow_id, "node_id": node_id, "node_name": node["name"],
        "customer_agent_id": customer_agent_id,
        "test_goal": normalized["user_goal"], "script": script,
    }


@router.post("/flows/{flow_id}/nodes/{node_id}/tests/{test_id}/run")
async def run_node_test(flow_id: int, node_id: str, test_id: int) -> dict:
    """Start a saved node test through the EXISTING Temporal run pipeline.

    This is a batch of one — the same pattern POST /runs already uses for a single
    agent (app.routers.runs.create_run) — submitting the identical RunGroupWorkflow /
    AgentTestWorkflow every other run goes through. It deliberately does NOT call
    app.routers.runs._launch_group: that helper also re-persists its target's reviewed
    suite via replace_test_cases, which REPLACES a customer-agent's entire test_cases
    library. That is correct when reviewing/saving a whole suite on the dashboard, but
    would silently delete every other saved test case for this agent the first time
    someone ran a single node test. The test_cases row already exists (from
    POST .../test above), so nothing needs to be (re)persisted here — only a new run
    and the workflow submission.

    Returns the existing run_id/group_id: the frontend polls/loads the result through
    the existing GET /runs/{run_id} and GET /runs/{run_id}/report, unchanged.
    """
    flow = get_agent_flow(flow_id)
    if flow is None:
        raise HTTPException(status_code=404, detail=f"flow {flow_id} not found")

    test_case = get_test_case(test_id)
    if (
        test_case is None
        or test_case.get("flow_id") != flow_id
        or test_case.get("node_id") != node_id
    ):
        raise HTTPException(
            status_code=404,
            detail=f"test {test_id} not found for flow {flow_id}, node '{node_id}'",
        )

    agent = get_agent(flow["agent_id"])
    if agent is None:
        raise HTTPException(status_code=404, detail=f"agent {flow['agent_id']} not found")

    # Rebuild the scenario shape from the SAVED row — already normalized/validated at
    # save time, so this is a straight reshape, not a re-derivation.
    scenario = {
        "title": test_case["title"],
        "user_goal": test_case["user_goal"],
        "test_type": test_case["test_type"],
        "assigned_fault": test_case["assigned_fault"],
        "expected_behavior": test_case["expected_behavior"],
        "seed_turns": _json.loads(test_case["seed_turns_json"] or "[]"),
        # Empty, deliberately — forces the scripted path, never the dynamic AI Caller.
        "customer_context": "",
        "max_turns": None,
        "flow_id": test_case["flow_id"],
        "node_id": test_case["node_id"],
        "node_script_json": test_case["node_script_json"],
    }

    run_id, group_id = await _start_single_scenario_run(
        agent["id"], test_case["customer_agent_id"], scenario
    )
    return {
        "run_id": run_id, "group_id": group_id,
        "flow_id": flow_id, "node_id": node_id, "test_id": test_id,
    }


async def _start_single_scenario_run(
    agent_id: int, customer_agent_id: int | None, scenario: dict
) -> tuple[int, int]:
    """Submit ONE scenario through the existing RunGroupWorkflow / AgentTestWorkflow —
    a batch of one, exactly as a node test has always been run. Shared by node tests
    and flow scenarios so both use the identical execution path. Returns
    (run_id, group_id); raises 503 (and marks the run errored) if Temporal is down.
    """
    group_id = insert_run_group()
    run_id = insert_run(agent_id, group_id, customer_agent_id)

    try:
        client = await get_client()
        await client.start_workflow(
            RunGroupWorkflow.run,
            RunGroupInput(
                group_id=group_id,
                targets=[AgentTestInput(
                    run_id=run_id, agent_id=agent_id, scenarios=[scenario],
                    # The sole run in this batch — it may use the worker's whole share.
                    work_share=WORK_CONCURRENCY,
                )],
                agent_concurrency=1,
            ),
            # Same naming convention as app.routers.runs._group_workflow_id: derived
            # from group_id, so re-submitting the same group id cannot start a second
            # copy of it.
            id=f"agentshield-run-group-{group_id}",
            task_queue=TEMPORAL_TASK_QUEUE,
        )
    except Exception as e:
        # The rows exist but nothing will ever execute them — mark errored rather than
        # leaving the client polling a run stuck in "running" forever.
        update_run(run_id, status="error", finished=True)
        raise HTTPException(
            status_code=503,
            detail=(
                "could not reach the Temporal server, so the test was not started "
                f"({type(e).__name__}). Start it with: temporal server start-dev"
            ),
        ) from e
    return run_id, group_id


# ---------------------------------------------------------------------------
# Flow-level scenario planning (Phase B): deterministic preview + explicit save.
#
# app.core.flow_graph is the ONLY path planner — these endpoints just load a stored
# flow, hand its nodes/edges to it, and attach an id and a display name to each
# candidate. No LLM, no test_cases, no runs. Preview never writes; saving is a
# separate explicit POST that is additive and idempotent.
# ---------------------------------------------------------------------------
# Structural labels only — they describe the kind of path, never what it means for
# the conversation (e.g. a branch is never called "Incorrect Name"). Semantic naming
# is left to the later LLM script-generation phase.
SCENARIO_NAMES = {
    HAPPY_PATH: "Happy Path",
    PRIMARY_PATH: "Primary Path",
    BRANCH: "Branch Scenario",
    TERMINAL: "Terminal Scenario",
    RETRY: "Retry Loop",
    RECOVERY: "Retry / Recovery",
}


class ScenarioPlanRequest(BaseModel):
    max_scenarios: int = Field(DEFAULT_MAX_SCENARIOS, ge=1, le=HARD_MAX_SCENARIOS)


class SaveScenariosRequest(BaseModel):
    """Save the reviewed plan. With `scenario_ids` (ids from a preview), exactly those
    scenarios are saved; without it, the whole plan for `max_scenarios` is."""
    scenario_ids: list[str] | None = None
    max_scenarios: int = Field(DEFAULT_MAX_SCENARIOS, ge=1, le=HARD_MAX_SCENARIOS)


def _load_flow_graph(flow_id: int) -> tuple[dict, dict]:
    """(flow row, {"nodes", "edges"}) for a stored flow — 404 if missing, 422 if its
    stored graph is malformed.

    Uses the stored nodes_json/edges_json, which ARE the flow parser's normalized output
    (or LLM-extracted output validated by the parser's own rules at upload time).
    raw_source is deliberately not re-parsed: for a flow that needed LLM extraction,
    that would call the LLM again.
    """
    flow = get_agent_flow(flow_id)
    if flow is None:
        raise HTTPException(status_code=404, detail=f"flow {flow_id} not found")
    try:
        nodes = _json.loads(flow["nodes_json"])
        edges = _json.loads(flow["edges_json"])
    except (TypeError, ValueError):
        raise HTTPException(
            status_code=422, detail={"errors": [f"flow {flow_id} has an unreadable stored graph."]}
        )
    errors: list[str] = []
    if not isinstance(nodes, list) or not all(
        isinstance(n, dict) and n.get("id") not in (None, "") for n in nodes
    ):
        errors.append("stored nodes must be a list of objects that each have an 'id'.")
    if not isinstance(edges, list) or not all(
        isinstance(e, dict) and e.get("from") not in (None, "") and e.get("to") not in (None, "")
        for e in edges
    ):
        errors.append("stored edges must be a list of objects that each have 'from' and 'to'.")
    if errors:
        raise HTTPException(status_code=422, detail={"errors": errors})
    return flow, {"nodes": nodes, "edges": edges}


def _scenario_key(path: list[str]) -> str:
    """Deterministic id from the ordered path. The planner never returns two identical
    paths for one flow, so this is unique per flow — and stable across calls."""
    return hashlib.sha1(_json.dumps(path).encode()).hexdigest()[:16]


def _plan_scenarios(graph: dict, max_scenarios: int) -> list[dict]:
    """flow_graph's candidates, unchanged, plus an `id` and a display `name`.

    A repeated category is numbered by order of appearance ("Branch Scenario",
    "Branch Scenario 2", ...). Because the planner's output for a smaller cap is a
    prefix of its output for a larger one, ids AND names are the same in every preview
    regardless of max_scenarios.
    """
    seen: dict[str, int] = {}
    scenarios = []
    for candidate in generate_candidate_paths(graph, max_scenarios):
        category = candidate["category"]
        seen[category] = seen.get(category, 0) + 1
        base = SCENARIO_NAMES.get(category, category.replace("_", " ").title())
        scenarios.append({
            "id": _scenario_key(candidate["path"]),
            "name": base if seen[category] == 1 else f"{base} {seen[category]}",
            **candidate,
        })
    return scenarios


def _stored_scenario(row: dict) -> dict:
    path = _json.loads(row["path_json"])
    return {
        "id": row["scenario_key"],
        "scenario_row_id": row["id"],
        "category": row["category"],
        "name": row["name"],
        "test_goal": row["test_goal"],
        "path": path,
        "covered_edges": _json.loads(row["covered_edges_json"]),
        "covered_terminals": _json.loads(row["covered_terminals_json"]),
        "contains_retry": row["contains_retry"],
        "path_length": len(path),
        "created_at": row["created_at"],
        **_stored_script(row.get("script_json")),
    }


def _stored_script(script_json: str | None) -> dict:
    """The saved script. A path script is stored as its turns list (Phase D); an
    interrupt script as {"turns", "setup", "expectations"} in the same column."""
    script = _json.loads(script_json) if script_json else None
    if isinstance(script, dict):
        return {"turns": script.get("turns"), "setup": script.get("setup"), "expectations": script.get("expectations")}
    return {"turns": script}


# ---------------------------------------------------------------------------
# Interrupt scenarios (whole-flow planning): a modular flow's stored raw source is
# re-parsed deterministically to recover its interrupts, then planned by
# app.core.flow_scenario_planner. Scripts: _generate_interrupt_draft / _save_interrupt_script;
# runs: _run_interrupt_scenario.
# ---------------------------------------------------------------------------
INTERRUPT_CATEGORY = "interrupt"
INTERRUPT_NO_SCRIPT = "Interrupt scenarios do not support script generation yet."
INTERRUPT_NO_RUN = "This interrupt scenario has no saved script — generate and save one before running."


def _modular_source(flow: dict) -> bool:
    """True iff this stored flow's raw source is a modular step flow that may be
    re-parsed deterministically. Never for an LLM-extracted flow: re-parsing it would
    call the LLM again."""
    if (flow.get("extraction_method") or "deterministic") == "llm" or not flow.get("raw_source"):
        return False
    try:
        return is_modular_step_flow(yaml.safe_load(flow["raw_source"]))
    except yaml.YAMLError:
        return False


def _load_planning_graph(flow_id: int) -> tuple[dict, dict, list | None]:
    """(flow row, graph, interrupts) for whole-flow planning.

    Not a modular flow: the stored graph, interrupts None — exactly what every endpoint
    used before. Modular flow: a fresh deterministic parse of the stored raw source,
    which must describe the SAME graph as the stored one (same node ids in order, same
    edges in order); otherwise 409 rather than planning against a different graph.
    Nothing is written.
    """
    flow, stored = _load_flow_graph(flow_id)
    if not _modular_source(flow):
        return flow, stored, None
    try:
        fresh = parse_flow(flow["raw_source"], flow.get("source_format"))
    except (FlowParseError, FlowAmbiguousError) as e:
        errors = getattr(e, "errors", None) or [str(e)]
        raise HTTPException(status_code=422, detail={"errors": errors}) from e

    stored_ids = [str(n["id"]) for n in stored["nodes"]]
    fresh_ids = [str(n["id"]) for n in fresh["nodes"]]
    stored_edges = [(str(e["from"]), str(e["to"])) for e in stored["edges"]]
    fresh_edges = [(str(e["from"]), str(e["to"])) for e in fresh["edges"]]
    if stored_ids != fresh_ids or stored_edges != fresh_edges:
        raise HTTPException(
            status_code=409,
            detail={"errors": [
                f"flow {flow_id}'s stored graph ({len(stored_ids)} nodes, {len(stored_edges)} edges) "
                f"no longer matches its source ({len(fresh_ids)} nodes, {len(fresh_edges)} edges). "
                "Re-upload the flow to refresh it."
            ]},
        )
    return flow, {"nodes": fresh["nodes"], "edges": fresh["edges"]}, fresh.get("interrupts", [])


def _interrupt_scenario_key(interrupt_key: str, step: str | None, base_path: list[str]) -> str:
    """Deterministic id from the interrupt's identity, where it is injected, and the
    base path — never the path alone, so interrupts sharing a path (every `end`
    interrupt fired at the same caller turn) cannot collide. Same convention as
    _scenario_key."""
    identity = {"interrupt": interrupt_key, "at": step, "base": base_path}
    return hashlib.sha1(_json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]


def _interrupt_scenarios(graph: dict, interrupts: list) -> tuple[dict | None, list[dict]]:
    """(base path info, every interrupt scenario) for a modular flow's graph."""
    plan = plan_flow_scenarios({**graph, "interrupts": interrupts}, HARD_MAX_SCENARIOS)
    base = plan["base"]
    base_path = base["path"] if base else []
    base_info = None
    if base is not None:
        base_scenario = _plan_scenarios(graph, HARD_MAX_SCENARIOS)[base["candidate_index"]]
        base_info = {"scenario_id": base_scenario["id"], "name": base_scenario["name"], "path": base_path}

    terminals = set(analyze_graph(graph).terminals)
    items = []
    for s in plan["interrupt_scenarios"]:
        key = s["interrupt"]["id"]
        item = {
            **s,
            "id": _interrupt_scenario_key(key, s["placement"].get("step"), base_path),
            "name": key.replace("_", " ").title(),
            "category": INTERRUPT_CATEGORY,
        }
        if s["plannable"]:
            step_segments = [g["steps"] for g in s["segments"] if "steps" in g]
            item["covered_terminals"] = [n for n in s["path"] if n in terminals]
            # A resume revisits the interrupted step by design; only a repeat WITHIN a
            # segment is a graph retry.
            item["contains_retry"] = any(len(steps) != len(set(steps)) for steps in step_segments)
        items.append(item)
    return base_info, items


def _is_planned_interrupt(flow_id: int, scenario_id: str) -> bool:
    """Whether `scenario_id` names one of this flow's (unsaved) interrupt scenarios."""
    try:
        _, graph, interrupts = _load_planning_graph(flow_id)
    except HTTPException:
        return False
    if interrupts is None:
        return False
    return any(s["id"] == scenario_id for s in _interrupt_scenarios(graph, interrupts)[1])


# Any `user.<field>` the flow references — in prompt placeholders ({user.first_name})
# or in conditions (answer.dob == user.dob).
_RECORD_FIELD = re.compile(r"\b(user\.[A-Za-z_][A-Za-z0-9_]*)")
READBACK_FIELD_TYPES = ("date", "address")


def _modular_details(flow: dict) -> dict:
    """Facts about each step of a modular flow's stored source, for prompting and
    validation: kind, what the agent asks/says, the field it collects, and the call
    status if it ends there. Plus which steps allow a readback — an ask step whose
    declared field type is date/address, or that declares a `readback` prompt — and the
    `user.*` record fields the flow's prompts reference."""
    raw = flow["raw_source"]
    doc = yaml.safe_load(raw)
    lang = doc.get("default_language") or next(iter(doc.get("languages") or []), None)

    def text(block):
        return (block.get("text") or block.get("intent") or "").strip() if isinstance(block, dict) else ""

    details, readback_steps = {}, set()
    for step in doc["steps"]:
        prompt = (step.get("prompt") or {}).get(lang) or {}
        d = {"kind": step.get("kind")}
        if text(prompt.get("ask")):
            d["ask"] = text(prompt["ask"])
        if text(prompt.get("say")):
            d["say"] = text(prompt["say"])
        field = step.get("field")
        if isinstance(field, dict) and field.get("key"):
            d["field"] = {k: field[k] for k in ("key", "type", "label") if k in field}
            if step.get("kind") == "ask" and (field.get("type") in READBACK_FIELD_TYPES or "readback" in prompt):
                readback_steps.add(step["id"])
        if step.get("status"):
            d["status"] = step["status"]
        details[step["id"]] = d
    return {
        "details": details,
        "readback_steps": readback_steps,
        "record_fields": sorted(set(_RECORD_FIELD.findall(raw))),
        "flow_name": str(doc.get("flow_key") or flow.get("name") or "flow"),
    }


def _interrupt_context(flow_id: int, scenario_id: str) -> tuple[dict, dict, list, dict]:
    """(flow, graph, interrupts, planned scenario) for an interrupt scenario id,
    re-planned from the stored source. A saved row must still match the plan."""
    flow, graph, interrupts = _load_planning_graph(flow_id)
    if interrupts is None:
        raise HTTPException(status_code=404, detail=f"scenario '{scenario_id}' not found for flow {flow_id}")
    item = next((s for s in _interrupt_scenarios(graph, interrupts)[1] if s["id"] == scenario_id), None)
    if item is None:
        raise HTTPException(status_code=409, detail=f"interrupt scenario '{scenario_id}' no longer matches flow {flow_id}'s plan")
    if not item["plannable"]:
        raise HTTPException(status_code=409, detail=f"interrupt scenario '{item['name']}' cannot be scripted: {item['reason']}")
    row = get_flow_scenario(flow_id, scenario_id)
    if row is not None and _json.loads(row["path_json"]) != item["path"]:
        raise HTTPException(status_code=409, detail=f"interrupt scenario '{scenario_id}' no longer matches flow {flow_id}'s plan")
    return flow, graph, interrupts, item


def _record_values(values: dict | None, allowed: list[str]) -> dict:
    """Test data the user supplied for the flow's `user.*` record fields, trimmed; blank
    values dropped. An unknown field is rejected rather than silently kept."""
    record = {k.strip(): v.strip() for k, v in (values or {}).items() if isinstance(v, str) and v.strip()}
    unknown = [k for k in record if k not in allowed]
    if unknown:
        raise HTTPException(status_code=422, detail={"errors": [
            f"unknown record field(s): {', '.join(unknown)}; this flow uses: {', '.join(allowed) or 'none'}"
        ]})
    return record


CALL_PREFIX = "call."


def _split_test_data(values: dict | None) -> tuple[dict, dict]:
    """(record values, call variables): test data named `call.<name>` is for creating
    the voice agent's call (see app.core.voice_native_ws call variables); everything
    else is the flow's `user.*` record."""
    record, call = {}, {}
    for k, v in (values or {}).items():
        key = k.strip() if isinstance(k, str) else k
        if isinstance(key, str) and key.startswith(CALL_PREFIX):
            call[key[len(CALL_PREFIX):]] = v
        else:
            record[k] = v
    return record, call


def _agent_for_flow(flow: dict) -> dict:
    agent = get_agent(flow["agent_id"])
    if agent is None:
        raise HTTPException(status_code=404, detail=f"agent {flow['agent_id']} not found")
    return dict(agent)


def _call_fields(agent: dict) -> list[str]:
    """The call variables the flow's agent can take from test data (native_ws only)."""
    return call_template_variables(agent) if (agent.get("voice_protocol") or "") == "native_ws" else []


def _call_values(values: dict | None, agent: dict, status_code: int = 422) -> dict:
    """Call variables for this agent, trimmed; blank values dropped. Any the agent's
    request_template does not map is rejected — never silently dropped."""
    call = {k.strip(): v.strip() for k, v in (values or {}).items() if isinstance(v, str) and v.strip()}
    problems = check_call_variables(agent, call)
    if problems:
        raise HTTPException(status_code=status_code, detail={"errors": problems})
    return call


def _interrupt_validation_args(graph: dict, interrupts: list, modular: dict) -> dict:
    return {
        "node_kinds": {str(n["id"]): str(n.get("type") or "") for n in graph["nodes"]},
        "readback_steps": modular["readback_steps"],
        "interrupts": interrupts,
    }


@router.post("/flows/{flow_id}/scenarios/preview")
def preview_flow_scenarios(flow_id: int, body: ScenarioPlanRequest | None = None) -> dict:
    """Plan candidate test scenarios for a stored flow — CALCULATE ONLY.

    Deterministic (same flow -> same response), and side-effect free: no LLM call, no
    database write, no test case, no run.
    """
    body = body or ScenarioPlanRequest()
    flow, graph, interrupts = _load_planning_graph(flow_id)
    facts = analyze_graph(graph)
    scenarios = _plan_scenarios(graph, body.max_scenarios)
    response = {
        "flow_id": flow_id,
        "max_scenarios": body.max_scenarios,
        "scenario_count": len(scenarios),
        "graph": {
            "roots": facts.roots,
            "roots_inferred": facts.roots_inferred,
            "entry_points": facts.entry_points,
            "terminals": facts.terminals,
            "branch_nodes": facts.branch_nodes,
            "back_edges": [[u, v] for u, v in facts.back_edges],
            "unreachable": facts.unreachable,
        },
        # id -> name only; the full node objects stay available via GET /flows/{id}.
        "node_names": {str(n["id"]): str(n.get("name") or n["id"]) for n in graph["nodes"]},
        "scenarios": scenarios,
    }
    # Modular flows only: the interrupt scenarios and the base path they are built on.
    # Every other flow's response is exactly as before.
    if interrupts is not None:
        base_info, interrupt_items = _interrupt_scenarios(graph, interrupts)
        shown = interrupt_items[:body.max_scenarios]
        response.update({
            "base_path": base_info,
            "interrupt_scenarios": shown,
            "interrupt_scenario_count": len(shown),
            "interrupt_scenario_total": len(interrupt_items),
            # The `user.*` record fields the flow references — test data for scripts.
            "record_fields": _modular_details(flow)["record_fields"],
            # Call variables the flow's agent takes from test data, as `call.<name>`.
            "call_fields": [CALL_PREFIX + n for n in _call_fields(dict(get_agent(flow["agent_id"]) or {}))],
        })
    return response


@router.post("/flows/{flow_id}/scenarios")
def save_flow_scenarios(flow_id: int, body: SaveScenariosRequest | None = None) -> dict:
    """Explicitly save planned scenarios for a flow.

    Scenarios are re-planned server-side from the stored flow rather than accepted from
    the client, so a saved path can never contain an invented node or edge. Additive
    and idempotent: already-saved scenarios are left as they are, nothing is deleted or
    replaced. Not connected to test_cases or execution.
    """
    body = body or SaveScenariosRequest()
    _, graph, interrupts = _load_planning_graph(flow_id)

    if body.scenario_ids is None:
        chosen = _plan_scenarios(graph, body.max_scenarios)
    else:
        # Resolve against the largest plan: every id any preview can show is in it.
        # Interrupt scenarios are re-planned server-side too — the client only names ids.
        available = _plan_scenarios(graph, HARD_MAX_SCENARIOS)
        interrupt_items = _interrupt_scenarios(graph, interrupts)[1] if interrupts is not None else []
        known = {s["id"] for s in available} | {s["id"] for s in interrupt_items}
        unknown = [i for i in body.scenario_ids if i not in known]
        if unknown:
            raise HTTPException(
                status_code=422,
                detail={"errors": [f"unknown scenario id(s) for flow {flow_id}: {', '.join(unknown)}"]},
            )
        wanted = set(body.scenario_ids)
        unplannable = [s for s in interrupt_items if s["id"] in wanted and not s["plannable"]]
        if unplannable:
            raise HTTPException(
                status_code=422,
                detail={"errors": [f"interrupt scenario '{s['name']}' cannot be saved: {s['reason']}" for s in unplannable]},
            )
        chosen = [s for s in available if s["id"] in wanted] + [s for s in interrupt_items if s["id"] in wanted]

    inserted = insert_flow_scenarios(flow_id, chosen)
    return {
        "flow_id": flow_id,
        "inserted": inserted,
        "scenarios": [_stored_scenario(r) for r in list_flow_scenarios(flow_id)],
    }


@router.get("/flows/{flow_id}/scenarios")
def get_flow_scenarios(flow_id: int) -> dict:
    """Every scenario saved for this flow."""
    if get_agent_flow(flow_id) is None:
        raise HTTPException(status_code=404, detail=f"flow {flow_id} not found")
    return {"flow_id": flow_id, "scenarios": [_stored_scenario(r) for r in list_flow_scenarios(flow_id)]}


# ---------------------------------------------------------------------------
# Flow scenario scripts (Phase D): generate / save / delete the conversation script
# for ONE planned scenario. Drafting and storage only — nothing here creates a test
# case, starts a run, or places a call.
# ---------------------------------------------------------------------------
def _resolve_scenario(flow_id: int, graph: dict, scenario_id: str) -> dict:
    """The scenario `scenario_id` names, resolved entirely server-side.

    A saved scenario uses its stored path; an unsaved one is looked up in the planner's
    (deterministic) plan. The client never supplies a path. Either way the path must
    hash back to `scenario_id` and use only real edges of this flow, so no other path
    can be substituted for the one the id names.
    """
    row = get_flow_scenario(flow_id, scenario_id)
    if row is not None:
        if row["category"] == INTERRUPT_CATEGORY:
            raise HTTPException(status_code=409, detail=INTERRUPT_NO_SCRIPT)
        scenario = _stored_scenario(row)
    else:
        scenario = next(
            (s for s in _plan_scenarios(graph, HARD_MAX_SCENARIOS) if s["id"] == scenario_id), None
        )
    if scenario is None:
        if _is_planned_interrupt(flow_id, scenario_id):
            raise HTTPException(status_code=409, detail=INTERRUPT_NO_SCRIPT)
        raise HTTPException(
            status_code=404, detail=f"scenario '{scenario_id}' not found for flow {flow_id}"
        )

    _verify_path(flow_id, graph, scenario_id, scenario["path"])
    return scenario


def _graph_node_kinds(graph: dict) -> dict[str, str]:
    """node id -> type, for validate_path_script's no-reply steps."""
    return {str(n["id"]): str(n.get("type") or "") for n in graph["nodes"]}


def _verify_path(flow_id: int, graph: dict, scenario_id: str, path: list[str]) -> None:
    """409 unless `path` is exactly the path `scenario_id` names and is walkable in this
    flow: non-empty, every node exists, every consecutive pair is a real edge. Never
    repairs a path."""
    edges = {(str(e["from"]), str(e["to"])) for e in graph["edges"]}
    node_ids = {str(n["id"]) for n in graph["nodes"]}
    if (
        _scenario_key(path) != scenario_id
        or not path
        or any(n not in node_ids for n in path)
        or any((a, b) not in edges for a, b in zip(path, path[1:]))
    ):
        raise HTTPException(
            status_code=409,
            detail=f"scenario '{scenario_id}' no longer matches flow {flow_id}'s graph",
        )


class GenerateFlowScript(BaseModel):
    """Interrupt scenarios only: test data for the flow's `user.*` record fields
    (e.g. {"user.first_name": "Daniel"}), used verbatim in the script."""
    test_data: dict[str, str] | None = None


def _is_interrupt_request(flow_id: int, scenario_id: str) -> bool:
    row = get_flow_scenario(flow_id, scenario_id)
    if row is not None:
        return row["category"] == INTERRUPT_CATEGORY
    return _is_planned_interrupt(flow_id, scenario_id)


@router.post("/flows/{flow_id}/scenarios/{scenario_id}/script")
async def generate_flow_scenario_script(
    flow_id: int, scenario_id: str, body: GenerateFlowScript | None = None
) -> dict:
    """Draft a deterministic conversation script for one planned scenario. Calling it
    again regenerates the script for the SAME path (and, for an interrupt scenario, the
    same interrupt at the same point). Nothing is saved."""
    if _is_interrupt_request(flow_id, scenario_id):
        return await _generate_interrupt_draft(flow_id, scenario_id, (body or GenerateFlowScript()).test_data)
    _, graph = _load_flow_graph(flow_id)
    scenario = _resolve_scenario(flow_id, graph, scenario_id)
    nodes_by_id = {str(n["id"]): n for n in graph["nodes"]}
    try:
        generated = await generate_path_script(
            scenario["path"], nodes_by_id, analyze_graph(graph).out_adj, scenario["category"]
        )
    except NodeScriptError as e:
        raise HTTPException(status_code=422, detail={"errors": e.errors}) from e
    return {
        "flow_id": flow_id, "scenario_id": scenario_id,
        "name": scenario["name"], "category": scenario["category"], "path": scenario["path"],
        "test_goal": generated["test_goal"], "turns": generated["turns"],
    }


class SaveFlowScenarioScript(BaseModel):
    test_goal: str
    turns: list[dict]
    # Interrupt scripts only; ignored for a path script.
    setup: dict | None = None
    expectations: dict | None = None


@router.put("/flows/{flow_id}/scenarios/{scenario_id}")
def save_flow_scenario_script(flow_id: int, scenario_id: str, body: SaveFlowScenarioScript) -> dict:
    """Save a reviewed/edited script for one scenario (idempotent upsert into
    flow_scenarios). The script is re-validated against the server-resolved path, so
    an edit cannot break path correspondence. Creates no test case."""
    if _is_interrupt_request(flow_id, scenario_id):
        return _save_interrupt_script(flow_id, scenario_id, body)
    _, graph = _load_flow_graph(flow_id)
    scenario = _resolve_scenario(flow_id, graph, scenario_id)
    try:
        script = validate_path_script(body.model_dump(), scenario["path"], _graph_node_kinds(graph))
    except NodeScriptError as e:
        raise HTTPException(status_code=422, detail={"errors": e.errors}) from e
    row = upsert_flow_scenario_script(flow_id, scenario, script["test_goal"], script["turns"])
    return _stored_scenario(row)


@router.delete("/flows/{flow_id}/scenarios/{scenario_id}")
def delete_saved_flow_scenario(flow_id: int, scenario_id: str) -> dict:
    """Delete one saved scenario and its script — nothing else."""
    if get_agent_flow(flow_id) is None:
        raise HTTPException(status_code=404, detail=f"flow {flow_id} not found")
    if not delete_flow_scenario(flow_id, scenario_id):
        raise HTTPException(
            status_code=404, detail=f"no saved scenario '{scenario_id}' for flow {flow_id}"
        )
    return {"flow_id": flow_id, "scenario_id": scenario_id, "deleted": True}


# ---------------------------------------------------------------------------
# Run a saved flow scenario (Phase E) through the EXISTING pipeline: the same
# RunGroupWorkflow / AgentTestWorkflow, scripted runner, voice transport and Judge a
# node test uses. The only flow-specific piece is path_script_to_scenario(), which
# reshapes the saved script into the scenario dict that pipeline already consumes.
# Nothing is written to test_cases; the run's own scenarios row is the normal per-run
# execution copy every run has.
# ---------------------------------------------------------------------------
@router.post("/flows/{flow_id}/scenarios/{scenario_id}/run")
async def run_flow_scenario(flow_id: int, scenario_id: str) -> dict:
    """Execute one SAVED flow scenario with its SAVED script. Takes no body: the path
    and script come only from the database, and are re-verified before anything
    starts — an invalid one is refused without creating a run or placing a call."""
    flow, graph = _load_flow_graph(flow_id)

    row = get_flow_scenario(flow_id, scenario_id)
    if row is None:
        if _is_planned_interrupt(flow_id, scenario_id):
            raise HTTPException(status_code=409, detail=INTERRUPT_NO_RUN)
        raise HTTPException(
            status_code=404,
            detail=f"no saved scenario '{scenario_id}' for flow {flow_id} — save it before running",
        )
    if row["category"] == INTERRUPT_CATEGORY:
        return await _run_interrupt_scenario(flow_id, scenario_id, row)
    scenario = _stored_scenario(row)
    if not scenario["turns"] or not scenario["test_goal"]:
        raise HTTPException(
            status_code=409,
            detail=f"scenario '{scenario_id}' has no saved script — generate and save one before running",
        )

    _verify_path(flow_id, graph, scenario_id, scenario["path"])
    try:
        script = validate_path_script(
            {"test_goal": scenario["test_goal"], "turns": scenario["turns"]}, scenario["path"],
            _graph_node_kinds(graph),
        )
    except NodeScriptError as e:
        raise HTTPException(status_code=422, detail={"errors": e.errors}) from e

    agent = get_agent(flow["agent_id"])
    if agent is None:
        raise HTTPException(status_code=404, detail=f"agent {flow['agent_id']} not found")

    node_names = {str(n["id"]): str(n.get("name") or n["id"]) for n in graph["nodes"]}
    run_input = path_script_to_scenario(
        flow_id, scenario_id, scenario["name"], scenario["path"], node_names,
        script["test_goal"], script["turns"],
    )
    # Group the run under the agent's existing customer context if it has one; never
    # create one just to run a flow scenario (runs.customer_agent_id is nullable).
    customer_agent = get_customer_agent_by_agent_id(agent["id"])
    run_id, group_id = await _start_single_scenario_run(
        agent["id"], customer_agent["id"] if customer_agent else None, run_input
    )
    return {"run_id": run_id, "group_id": group_id, "flow_id": flow_id, "scenario_id": scenario_id}


# ---------------------------------------------------------------------------
# Interrupt scenario scripts: generate / save. Everything planner-controlled comes from
# the server-side re-plan; the model and the client only supply dialogue and
# expectations, which validate_interrupt_script checks against that plan. A saved
# interrupt script runs through the same pipeline as a path script
# (_run_interrupt_scenario).
# ---------------------------------------------------------------------------
async def _generate_interrupt_draft(flow_id: int, scenario_id: str, test_data: dict | None) -> dict:
    flow, graph, interrupts, item = _interrupt_context(flow_id, scenario_id)
    modular = _modular_details(flow)
    record_data, call_data = _split_test_data(test_data)
    record = _record_values(record_data, modular["record_fields"])
    call = _call_values(call_data, _agent_for_flow(flow)) if call_data else {}
    try:
        script = await generate_interrupt_script(
            item, details=modular["details"], record=record, flow_name=modular["flow_name"],
            edges=graph["edges"], **_interrupt_validation_args(graph, interrupts, modular),
        )
    except NodeScriptError as e:
        raise HTTPException(status_code=422, detail={"errors": e.errors}) from e
    if call:
        script["setup"]["call"] = call
    return {
        "flow_id": flow_id, "scenario_id": scenario_id, "name": item["name"],
        "category": INTERRUPT_CATEGORY, "path": item["path"], **script,
    }


def _save_interrupt_script(flow_id: int, scenario_id: str, body: SaveFlowScenarioScript) -> dict:
    flow, graph, interrupts, item = _interrupt_context(flow_id, scenario_id)
    modular = _modular_details(flow)
    setup = body.setup or {}
    record = _record_values(setup.get("record"), modular["record_fields"])
    call = _call_values(setup.get("call"), _agent_for_flow(flow)) if setup.get("call") else {}
    submitted = {
        "test_goal": body.test_goal, "turns": body.turns,
        "assumed_values": setup.get("assumed_values") or {}, "expectations": body.expectations,
    }
    try:
        script = validate_interrupt_script(
            submitted, item, record=record, **_interrupt_validation_args(graph, interrupts, modular),
        )
    except NodeScriptError as e:
        raise HTTPException(status_code=422, detail={"errors": e.errors}) from e
    if call:
        script["setup"]["call"] = call
    stored = {"turns": script["turns"], "setup": script["setup"], "expectations": script["expectations"]}
    row = upsert_flow_scenario_script(flow_id, item, script["test_goal"], stored)
    return _stored_scenario(row)


async def _run_interrupt_scenario(flow_id: int, scenario_id: str, row: dict) -> dict:
    """Execute one SAVED interrupt scenario with its SAVED script, through the same
    _start_single_scenario_run a path script uses. Before anything starts, the flow is
    re-planned and the saved script re-validated against that plan; if the flow has
    changed since it was saved, 409 — an outdated script is never run, regenerated or
    repaired."""
    try:
        flow, graph, interrupts, item = _interrupt_context(flow_id, scenario_id)
    except HTTPException as e:
        if e.status_code != 404:
            raise
        # The row is saved but the flow no longer plans interrupts at all: stale.
        raise HTTPException(
            status_code=409, detail=f"interrupt scenario '{scenario_id}' no longer matches flow {flow_id}'s plan"
        ) from e
    saved = _stored_scenario(row)
    if not saved["turns"] or not saved["test_goal"]:
        raise HTTPException(status_code=409, detail=INTERRUPT_NO_RUN)

    modular = _modular_details(flow)
    setup = saved.get("setup") or {}
    record = setup.get("record") or {}
    stale = [k for k in record if k not in modular["record_fields"]]
    submitted = {
        "test_goal": saved["test_goal"], "turns": saved["turns"],
        "assumed_values": setup.get("assumed_values") or {}, "expectations": saved.get("expectations"),
    }
    try:
        if stale:
            raise NodeScriptError([f"record field(s) no longer used by the flow: {', '.join(stale)}"])
        script = validate_interrupt_script(
            submitted, item, record=record, **_interrupt_validation_args(graph, interrupts, modular),
        )
    except NodeScriptError as e:
        raise HTTPException(status_code=409, detail={"errors": [
            f"the saved script for '{item['name']}' no longer matches flow {flow_id}; regenerate and save it before running",
            *e.errors,
        ]}) from e

    agent = _agent_for_flow(flow)
    if setup.get("call"):
        # The agent's call configuration may have changed since the script was saved.
        script["setup"]["call"] = _call_values(setup["call"], agent, status_code=409)

    node_names = {str(n["id"]): str(n.get("name") or n["id"]) for n in graph["nodes"]}
    run_input = interrupt_script_to_scenario(
        flow_id, scenario_id, item["name"], item, node_names,
        script["test_goal"], script["turns"], script["setup"], script["expectations"],
    )
    customer_agent = get_customer_agent_by_agent_id(agent["id"])
    run_id, group_id = await _start_single_scenario_run(
        agent["id"], customer_agent["id"] if customer_agent else None, run_input
    )
    return {"run_id": run_id, "group_id": group_id, "flow_id": flow_id, "scenario_id": scenario_id}
