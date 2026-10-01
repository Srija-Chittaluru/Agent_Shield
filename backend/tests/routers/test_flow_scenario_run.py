"""Router-level tests for POST /flows/{flow_id}/scenarios/{scenario_id}/run (Phase E).

Isolated FastAPI app, app.db calls monkeypatched, Temporal client faked so we can see
exactly what would be submitted. Uses the REAL _load_flow_graph / _verify_path /
validate_path_script / path_script_to_scenario code.
"""
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routers import flows as flows_router

NODES = [
    ("greeting", "Greeting"), ("ask_name", "Ask Name"), ("validate_name", "Validate Name"),
    ("retry_name", "Ask Name Again"), ("ask_dob", "Ask DOB"), ("ask_email", "Ask Email"),
    ("complete", "Complete"),
]
EDGES = [
    ("greeting", "ask_name"), ("ask_name", "validate_name"), ("validate_name", "ask_dob"),
    ("validate_name", "retry_name"), ("retry_name", "validate_name"),
    ("ask_dob", "ask_email"), ("ask_email", "complete"),
]
FLOWS = {
    fid: {
        "id": fid, "agent_id": 83, "name": "flow.json", "source_format": "json", "created_at": "t",
        "nodes_json": json.dumps([{"id": n, "name": name} for n, name in NODES]),
        "edges_json": json.dumps([{"from": a, "to": b} for a, b in EDGES]),
    }
    for fid in (1, 2)
}
RECOVERY = [
    "greeting", "ask_name", "validate_name", "retry_name", "validate_name", "ask_dob", "ask_email", "complete",
]
RECOVERY_ID = flows_router._scenario_key(RECOVERY)


def _turns(path):
    return [
        {"step": i + 1, "node_id": n, "expected_agent_behavior": f"Agent handles {n}.",
         "caller_line": f"Caller line {i + 1}."}
        for i, n in enumerate(path)
    ]


def _row(flow_id=1, key=RECOVERY_ID, path=RECOVERY, turns="default", test_goal="Verify recovery."):
    return {
        "id": 10, "flow_id": flow_id, "scenario_key": key, "category": "recovery",
        "name": "Retry / Recovery", "test_goal": test_goal, "path_json": json.dumps(path),
        "covered_edges_json": json.dumps([list(p) for p in zip(path, path[1:])]),
        "covered_terminals_json": json.dumps(["complete"]), "contains_retry": True,
        "created_at": "t",
        "script_json": json.dumps(_turns(path) if turns == "default" else turns) if turns is not None else None,
    }


class _Recorder:
    def __init__(self):
        self.calls: list[tuple] = []
        self.started: list[dict] = []


@pytest.fixture
def rec(monkeypatch):
    r = _Recorder()
    saved: dict[tuple, dict] = {(1, RECOVERY_ID): _row()}
    r.saved = saved

    class _Client:
        async def start_workflow(self, fn, inp, id, task_queue):
            r.started.append({"input": inp, "id": id, "task_queue": task_queue})

    async def get_client():
        r.calls.append(("get_client",))
        return _Client()

    def insert_run_group():
        r.calls.append(("insert_run_group",))
        return 501

    def insert_run(agent_id, group_id, customer_agent_id):
        r.calls.append(("insert_run", agent_id, group_id, customer_agent_id))
        return 901

    monkeypatch.setattr(flows_router, "get_agent_flow", lambda fid: FLOWS.get(fid))
    monkeypatch.setattr(flows_router, "get_flow_scenario", lambda fid, key: saved.get((fid, key)))
    monkeypatch.setattr(flows_router, "get_agent", lambda aid: {"id": aid, "name": "Maya", "modality": "voice"})
    monkeypatch.setattr(flows_router, "get_customer_agent_by_agent_id", lambda aid: {"id": 1027})
    monkeypatch.setattr(flows_router, "get_client", get_client)
    monkeypatch.setattr(flows_router, "insert_run_group", insert_run_group)
    monkeypatch.setattr(flows_router, "insert_run", insert_run)
    monkeypatch.setattr(flows_router, "update_run", lambda *a, **kw: r.calls.append(("update_run", kw)))

    def no_test_cases(*a, **kw):
        raise AssertionError("running a flow scenario must not write test_cases")

    monkeypatch.setattr(flows_router, "insert_test_case", no_test_cases)
    monkeypatch.setattr(flows_router, "update_test_case", no_test_cases)
    return r


@pytest.fixture
def client(rec):
    app = FastAPI()
    app.include_router(flows_router.router)
    return TestClient(app)


def _run(client, flow_id=1, key=RECOVERY_ID, **kw):
    return client.post(f"/flows/{flow_id}/scenarios/{key}/run", **kw)


def _assert_nothing_started(rec):
    assert rec.calls == [] and rec.started == []


# ---------------------------------------------------------------------------
# A. Execution
# ---------------------------------------------------------------------------
def test_saved_scenario_runs_through_the_existing_workflow(client, rec):
    res = _run(client)
    assert res.status_code == 200, res.text
    assert res.json() == {"run_id": 901, "group_id": 501, "flow_id": 1, "scenario_id": RECOVERY_ID}

    assert ("insert_run", 83, 501, 1027) in rec.calls
    [started] = rec.started
    assert started["id"] == "agentshield-run-group-501"
    [target] = started["input"].targets
    assert target.run_id == 901 and target.agent_id == 83
    [scenario] = target.scenarios
    assert scenario["test_type"] == "flow_path"
    assert scenario["customer_context"] == ""
    assert scenario["seed_turns"] == [f"Caller line {i + 1}." for i in range(8)]  # all 8, in order
    assert scenario["user_goal"] == "Verify recovery."
    assert scenario["expected_behavior"].startswith(
        "Planned path: Greeting -> Ask Name -> Validate Name -> Ask Name Again -> Validate Name"
    )
    assert "8. At step 8 (Complete): Agent handles complete." in scenario["expected_behavior"]
    assert scenario["flow_id"] == 1
    assert json.loads(scenario["node_script_json"])["path"] == RECOVERY


def test_run_ignores_any_client_supplied_script_or_path(client, rec):
    res = _run(client, json={"path": ["complete"], "turns": [{"caller_line": "hijack"}], "test_goal": "x"})
    assert res.status_code == 200
    scenario = rec.started[0]["input"].targets[0].scenarios[0]
    assert "hijack" not in scenario["seed_turns"]
    assert json.loads(scenario["node_script_json"])["path"] == RECOVERY


def test_agent_without_a_customer_context_gets_one_created_on_run(client, rec, monkeypatch):
    """Actually running a flow scenario is a meaningful action (same threshold
    save_node_test uses): an agent with no customer context yet gets one created, so it
    starts showing up in Existing Agent Testing / Existing Test Cases."""
    monkeypatch.setattr(flows_router, "get_customer_agent_by_agent_id", lambda aid: None)
    monkeypatch.setattr(flows_router, "default_customer_agent", lambda aid: 4242)
    assert _run(client).status_code == 200
    assert ("insert_run", 83, 501, 4242) in rec.calls


def test_temporal_down_is_503_and_the_run_is_marked_errored(client, rec, monkeypatch):
    async def down():
        raise ConnectionError("nope")

    monkeypatch.setattr(flows_router, "get_client", down)
    res = _run(client)
    assert res.status_code == 503
    assert ("update_run", {"status": "error", "finished": True}) in rec.calls


def test_missing_flow_is_404(client, rec):
    assert _run(client, flow_id=999).status_code == 404
    _assert_nothing_started(rec)


def test_unsaved_or_unknown_scenario_is_404(client, rec):
    happy = flows_router._scenario_key(["greeting", "ask_name", "validate_name", "ask_dob", "ask_email", "complete"])
    assert _run(client, key=happy).status_code == 404  # a real planned path, but never saved
    assert _run(client, key="0000000000000000").status_code == 404
    _assert_nothing_started(rec)


def test_scenario_saved_for_another_flow_is_404(client, rec):
    # Flow 2 has the same graph, but the scenario was saved only for flow 1.
    assert _run(client, flow_id=2).status_code == 404
    _assert_nothing_started(rec)


@pytest.mark.parametrize("overrides", [{"turns": None}, {"turns": []}, {"test_goal": None}])
def test_scenario_without_a_saved_script_cannot_run(client, rec, overrides):
    rec.saved[(1, RECOVERY_ID)] = _row(**overrides)
    res = _run(client)
    assert res.status_code == 409
    assert "no saved script" in res.json()["detail"]
    _assert_nothing_started(rec)


# ---------------------------------------------------------------------------
# D. Path integrity / invalid scripts — refused before anything starts (E)
# ---------------------------------------------------------------------------
def _bad_turns(mutate):
    turns = _turns(RECOVERY)
    mutate(turns)
    return turns


@pytest.mark.parametrize("turns,needle", [
    (_bad_turns(lambda t: t.reverse()), "turns follow the path in order"),
    (_bad_turns(lambda t: t[2].update(node_id="ask_dob")), "must be 'validate_name'"),
    (_bad_turns(lambda t: t[0].update(caller_line="")), "caller_line"),
    (_bad_turns(lambda t: t[1].update(caller_line="My name is [name].")), "placeholder"),
    (_bad_turns(lambda t: t[0].update(step=99)), "step must be"),
])
def test_invalid_saved_script_is_refused_before_execution(client, rec, turns, needle):
    rec.saved[(1, RECOVERY_ID)] = _row(turns=turns)
    res = _run(client)
    assert res.status_code == 422
    assert needle in " ".join(res.json()["detail"]["errors"])
    _assert_nothing_started(rec)


@pytest.mark.parametrize("path", [
    ["greeting", "ask_name", "ghost"],        # invalid node
    ["greeting", "complete"],                 # invalid edge
    ["ask_name", "greeting"],                 # reversed edge
])
def test_invalid_saved_path_is_refused_before_execution(client, rec, path):
    key = flows_router._scenario_key(path)
    rec.saved[(1, key)] = _row(key=key, path=path)
    res = _run(client, key=key)
    assert res.status_code == 409
    _assert_nothing_started(rec)


def test_path_that_does_not_match_its_id_is_refused(client, rec):
    # Stored under the recovery id, but holding a different (valid) path.
    other = ["greeting", "ask_name", "validate_name", "retry_name"]
    rec.saved[(1, RECOVERY_ID)] = _row(path=other)
    assert _run(client).status_code == 409
    _assert_nothing_started(rec)
