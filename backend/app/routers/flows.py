"""/flows API — upload/parse a Voice Agent's flow (Phase 1), generate a node's test
goal + deterministic script (Phase 2), and save/run it (Phase 3).

Phase 3 turns a reviewed node script into a REAL AgentShield test case and sends it
through the existing Temporal execution pipeline (app.temporal.workflows) — the exact
same AgentTestWorkflow/RunGroupWorkflow, activities, scripted runner
(app.core.runner._run_scripted, unmodified), voice protocols, recording, and Judge
every other test uses. This module adds no second voice-testing engine; see
app.core.node_script.to_scenario_dict for how a script becomes that existing shape.
"""
import json as _json

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.config import TEMPORAL_TASK_QUEUE, WORK_CONCURRENCY
from app.core.flow_llm_extractor import FlowExtractionError, extract_flow_with_llm
from app.core.flow_parser import FlowAmbiguousError, FlowParseError, parse_flow
from app.core.node_script import (
    NodeScriptError,
    generate_node_script,
    prerequisite_path,
    to_scenario_dict,
    validate_script,
)
from app.core.scenarios import normalize_scenarios
from app.temporal.client import get_client
from app.temporal.workflows import AgentTestInput, RunGroupInput, RunGroupWorkflow
from app.db import (
    default_customer_agent,
    get_agent,
    get_agent_flow,
    get_customer_agent_by_agent_id,
    get_test_case,
    insert_agent_flow,
    insert_run,
    insert_run_group,
    insert_test_case,
    list_agent_flows,
    update_run,
    update_test_case,
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

    group_id = insert_run_group()
    run_id = insert_run(agent["id"], group_id, test_case["customer_agent_id"])

    try:
        client = await get_client()
        await client.start_workflow(
            RunGroupWorkflow.run,
            RunGroupInput(
                group_id=group_id,
                targets=[AgentTestInput(
                    run_id=run_id, agent_id=agent["id"], scenarios=[scenario],
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

    return {
        "run_id": run_id, "group_id": group_id,
        "flow_id": flow_id, "node_id": node_id, "test_id": test_id,
    }
