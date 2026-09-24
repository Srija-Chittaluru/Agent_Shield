"""Router tests for interrupt scenarios (whole-flow planning, read-only phase):

    POST /flows/{id}/scenarios/preview   modular flows add interrupt_scenarios + base_path
    POST /flows/{id}/scenarios           interrupt scenario ids can be saved
    run                                  explicitly 409 for interrupt scenarios
                                         (scripts: see test_flow_interrupt_scripts.py)

Isolated FastAPI app, in-memory fake of flow_scenarios, the REAL parser / flow_graph /
flow_scenario_planner / node_script, and the real Maya fixture as a stored flow row.
"""
import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import app.core.llm as llm
from app.core import node_script
from app.core.flow_graph import HARD_MAX_SCENARIOS
from app.core.flow_parser import parse_flow
from app.routers import flows as R

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "flows" / "maya_renewal_modular.yaml"
RAW = FIXTURE.read_text()
# Flow 9's agent as configured: native_ws, with one call variable in its template.
MAYA_AGENT = {
    "id": 87, "name": "Maya", "modality": "voice", "voice_protocol": "native_ws",
    "endpoint_url": "http://localhost:8000",
    "request_template": '{"client": "client_a", "flow": "maya_renewal", "patient_ref": "{{patient_ref}}"}',
}
PARSED = parse_flow(RAW, "yaml")
END_INTERRUPTS = ["do_not_call", "voicemail", "busy", "wrong_number", "self_harm", "unsafe_at_home"]


def _row(flow_id, nodes, edges, **extra):
    return {
        "id": flow_id, "agent_id": 87, "name": "maya_renewal_modular.yaml", "source_format": "yaml",
        "created_at": "t", "extraction_method": "deterministic",
        "nodes_json": json.dumps(nodes), "edges_json": json.dumps(edges), **extra,
    }


# Stored rows as they really are: nodes/edges from the sync, no interrupts, no `field`.
STORED_NODES = [{k: v for k, v in n.items() if k != "field"} for n in PARSED["nodes"]]
FLOWS = {
    9: _row(9, STORED_NODES, PARSED["edges"], raw_source=RAW),
    1: {  # generic JSON flow, no raw modular source
        "id": 1, "agent_id": 1, "name": "flow.json", "source_format": "json", "created_at": "t",
        "nodes_json": json.dumps([{"id": "a", "name": "A"}, {"id": "b", "name": "B"}]),
        "edges_json": json.dumps([{"from": "a", "to": "b"}]),
    },
    10: _row(10, STORED_NODES, PARSED["edges"], raw_source=RAW, extraction_method="llm"),
    11: _row(11, STORED_NODES, PARSED["edges"][:-1], raw_source=RAW),   # stale stored graph
}


class _Table:
    """flow_scenarios in memory with the real UNIQUE (flow_id, scenario_key) semantics."""

    def __init__(self):
        self.rows: list[dict] = []

    def _find(self, flow_id, key):
        return next((r for r in self.rows if r["flow_id"] == flow_id and r["scenario_key"] == key), None)

    def _new(self, flow_id, s):
        return {
            "id": len(self.rows) + 1, "flow_id": flow_id, "scenario_key": s["id"], "category": s["category"],
            "name": s["name"], "test_goal": None, "path_json": json.dumps(s["path"]),
            "covered_edges_json": json.dumps(s["covered_edges"]),
            "covered_terminals_json": json.dumps(s["covered_terminals"]),
            "contains_retry": bool(s["contains_retry"]), "created_at": "t", "script_json": None,
        }

    def insert(self, flow_id, scenarios):
        n = 0
        for s in scenarios:
            if self._find(flow_id, s["id"]) is None:
                self.rows.append(self._new(flow_id, s))
                n += 1
        return n

    def upsert(self, flow_id, scenario, test_goal, turns):
        row = self._find(flow_id, scenario["id"])
        if row is None:
            row = self._new(flow_id, scenario)
            self.rows.append(row)
        row["test_goal"], row["script_json"] = test_goal, json.dumps(turns)
        return row

    def get(self, flow_id, key):
        return self._find(flow_id, key)

    def list(self, flow_id):
        return [r for r in self.rows if r["flow_id"] == flow_id]

    def delete(self, flow_id, key):
        row = self._find(flow_id, key)
        if row:
            self.rows.remove(row)
        return row is not None


@pytest.fixture
def table(monkeypatch):
    t = _Table()
    for name, fn in [("insert_flow_scenarios", t.insert), ("upsert_flow_scenario_script", t.upsert),
                     ("get_flow_scenario", t.get), ("list_flow_scenarios", t.list),
                     ("delete_flow_scenario", t.delete)]:
        monkeypatch.setattr(R, name, fn)
    return t


@pytest.fixture
def forbidden(monkeypatch):
    """Anything that would create test cases, runs, calls, flows or LLM traffic fails."""
    touched = []

    def trap(name):
        def boom(*a, **kw):
            touched.append(name)
            raise AssertionError(f"must not call {name}")
        return boom

    for name in ("insert_test_case", "update_test_case", "insert_run", "insert_run_group", "update_run",
                 "get_client", "insert_agent_flow", "extract_flow_with_llm", "generate_node_script"):
        monkeypatch.setattr(R, name, trap(name))
    monkeypatch.setattr(llm, "chat", trap("llm.chat"))
    return touched


@pytest.fixture
def client(monkeypatch, table, forbidden):
    app = FastAPI()
    app.include_router(R.router)
    monkeypatch.setattr(R, "get_agent_flow", lambda fid: FLOWS.get(fid))
    monkeypatch.setattr(R, "get_agent", lambda aid: dict(MAYA_AGENT, id=aid))
    return TestClient(app)


def _preview(client, flow_id=9, **body):
    res = client.post(f"/flows/{flow_id}/scenarios/preview", json=body or None)
    assert res.status_code == 200, res.text
    return res.json()


def _interrupts(body):
    return {s["interrupt"]["id"]: s for s in body["interrupt_scenarios"]}


# ---------------------------------------------------------------------------
# Preview
# ---------------------------------------------------------------------------
def test_modular_preview_adds_all_12_interrupt_scenarios_and_the_base(client):
    body = _preview(client)
    assert body["interrupt_scenario_total"] == body["interrupt_scenario_count"] == 12
    assert [s["interrupt"]["id"] for s in body["interrupt_scenarios"]] == [i["id"] for i in PARSED["interrupts"]]
    assert all(s["category"] == "interrupt" and s["plannable"] for s in body["interrupt_scenarios"])
    base = body["base_path"]
    assert base["path"][0] == "greet" and base["path"][-1] == "closing"
    assert base["scenario_id"] in {s["id"] for s in R._plan_scenarios(
        {"nodes": STORED_NODES, "edges": PARSED["edges"]}, HARD_MAX_SCENARIOS)}


def test_normal_scenarios_and_ids_are_unchanged(client):
    body = _preview(client)
    stored_graph = {"nodes": STORED_NODES, "edges": PARSED["edges"]}
    assert body["scenarios"] == R._plan_scenarios(stored_graph, 20)
    assert body["scenario_count"] == 20


def test_six_end_interrupts_share_a_path_but_not_an_id(client):
    by_id = _interrupts(_preview(client))
    ends = [by_id[k] for k in END_INTERRUPTS]
    assert {tuple(s["path"]) for s in ends} == {("greet", "ask_good_time")}
    assert len({s["id"] for s in ends}) == 6
    all_ids = [s["id"] for s in by_id.values()]
    assert len(set(all_ids)) == 12
    assert not set(all_ids) & {s["id"] for s in _preview(client)["scenarios"]}


def test_interrupt_details_are_preserved(client):
    by_id = _interrupts(_preview(client))
    human = by_id["human_request"]
    assert human["name"] == "Human Request"
    assert human["placement"] == {"policy": "after_greeting", "convention": True, "step": "ask_good_time"}
    assert human["segments"][1] == {"kind": "interrupt", "interrupt_id": "human_request",
                                     "outcome": "goto", "target": "route_counsellor_hours"}
    assert "target" not in by_id["hold"]["interrupt"]  # resume: no fake target
    assert by_id["wrong_number"]["segments"][1]["end_status"] == "No Answer | Retry Scheduled"
    # The interrupt jump is never an edge.
    assert ["ask_good_time", "route_counsellor_hours"] not in human["covered_edges"]


def test_preview_is_deterministic(client):
    assert _preview(client) == _preview(client)


def test_scenario_cap_applies_to_both_groups(client):
    body = _preview(client, max_scenarios=5)
    assert body["scenario_count"] == 5
    assert body["interrupt_scenario_count"] == 5 and body["interrupt_scenario_total"] == 12
    # A smaller cap shows a prefix of the same deterministic list.
    assert body["interrupt_scenarios"] == _preview(client)["interrupt_scenarios"][:5]


def test_generic_flow_preview_is_unchanged(client):
    body = _preview(client, flow_id=1)
    assert set(body) == {"flow_id", "max_scenarios", "scenario_count", "graph", "node_names", "scenarios"}


def test_llm_extracted_flow_is_never_reparsed(client, monkeypatch):
    monkeypatch.setattr(R, "parse_flow", lambda *a, **kw: pytest.fail("must not re-parse an LLM-extracted flow"))
    body = _preview(client, flow_id=10)
    assert "interrupt_scenarios" not in body


def test_stale_stored_graph_is_refused_with_a_clear_error(client, table):
    res = client.post("/flows/11/scenarios/preview")
    assert res.status_code == 409
    message = res.json()["detail"]["errors"][0]
    assert "no longer matches its source" in message and "73 edges" in message and "74 edges" in message
    assert client.post("/flows/11/scenarios").status_code == 409
    assert table.rows == []


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
def _save(client, ids, flow_id=9, **extra):
    return client.post(f"/flows/{flow_id}/scenarios", json={"scenario_ids": ids, **extra})


def test_all_12_interrupt_scenarios_save_and_coexist(client, table):
    ids = [s["id"] for s in _preview(client)["interrupt_scenarios"]]
    res = _save(client, ids)
    assert res.status_code == 200 and res.json()["inserted"] == 12
    rows = [r for r in table.rows if r["category"] == "interrupt"]
    assert len(rows) == 12 and len({r["scenario_key"] for r in rows}) == 12
    ends = {r["name"] for r in rows if json.loads(r["path_json"]) == ["greet", "ask_good_time"]}
    assert ends == {"Do Not Call", "Voicemail", "Busy", "Wrong Number", "Self Harm", "Unsafe At Home"}


def test_repeated_save_is_idempotent(client, table):
    ids = [s["id"] for s in _preview(client)["interrupt_scenarios"]]
    _save(client, ids)
    again = _save(client, ids).json()
    assert again["inserted"] == 0 and len(table.rows) == 12


def test_saved_interrupt_row_holds_only_existing_columns_and_real_edges(client, table):
    human = _interrupts(_preview(client))["human_request"]
    _save(client, [human["id"]])
    [row] = table.rows
    assert set(row) == {"id", "flow_id", "scenario_key", "category", "name", "test_goal", "path_json",
                        "covered_edges_json", "covered_terminals_json", "contains_retry", "created_at", "script_json"}
    assert row["script_json"] is None and row["test_goal"] is None
    edges = {(e["from"], e["to"]) for e in PARSED["edges"]}
    assert all(tuple(e) in edges for e in json.loads(row["covered_edges_json"]))
    assert "detection" not in json.dumps(row) and "keywords" not in json.dumps(row)


def test_normal_scenario_still_saves_alongside(client, table):
    body = _preview(client)
    normal, interrupt = body["scenarios"][0]["id"], body["interrupt_scenarios"][0]["id"]
    assert _save(client, [normal, interrupt]).json()["inserted"] == 2
    assert sorted(r["category"] for r in table.rows) == ["interrupt", "primary_path"]


def test_client_cannot_invent_or_alter_an_interrupt_scenario(client, table):
    real = _interrupts(_preview(client))["human_request"]
    invented = R._interrupt_scenario_key("made_up_interrupt", "ask_good_time", real["path"])
    moved = R._interrupt_scenario_key("human_request", "ask_dob", _preview(client)["base_path"]["path"])
    for bad in (invented, moved):
        res = _save(client, [bad])
        assert res.status_code == 422 and "unknown scenario id" in res.json()["detail"]["errors"][0]
    # Extra fields are ignored: the saved row is what the server planned.
    _save(client, [real["id"]], path=["closing"], target="done", interrupt_key="busy", injection_point="closing")
    [row] = table.rows
    assert json.loads(row["path_json"]) == real["path"] and row["name"] == "Human Request"


def test_existing_normal_script_is_preserved(client, table, monkeypatch):
    normal = _preview(client)["scenarios"][0]
    turns = [{"step": 2, "node_id": normal["path"][1], "expected_agent_behavior": "Asks if now is a good time.",
              "caller_line": "Sorry, now is really not a good time for me."}]
    assert client.put(f"/flows/9/scenarios/{normal['id']}", json={"test_goal": "g", "turns": turns}).status_code == 200
    _save(client, [s["id"] for s in _preview(client)["interrupt_scenarios"]] + [normal["id"]])
    row = next(r for r in table.rows if r["scenario_key"] == normal["id"])
    assert json.loads(row["script_json"]) == turns and row["test_goal"] == "g"


# ---------------------------------------------------------------------------
# Running an interrupt scenario needs a saved script
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("saved", [False, True])
def test_interrupt_run_without_a_script_returns_409(client, table, saved):
    # Running a scripted interrupt scenario is covered in test_flow_interrupt_run.py;
    # without a saved script it is refused.
    iid = _interrupts(_preview(client))["hold"]["id"]
    if saved:
        _save(client, [iid])
    run = client.post(f"/flows/9/scenarios/{iid}/run")
    assert run.status_code == 409 and "no saved script" in run.json()["detail"]


def test_interrupt_scenario_can_still_be_deleted(client, table):
    iid = _interrupts(_preview(client))["busy"]["id"]
    _save(client, [iid])
    assert client.delete(f"/flows/9/scenarios/{iid}").status_code == 200
    assert table.rows == []


def test_normal_scenario_scripts_still_work(client, monkeypatch):
    normal = _preview(client)["scenarios"][0]

    async def fake_chat(system, messages, json_mode=False, **kw):
        return {"test_goal": "g", "turns": [{"step": 2, "node_id": normal["path"][1],
                                              "expected_agent_behavior": "x", "caller_line": "Not right now, sorry."}]}

    monkeypatch.setattr(node_script, "chat", fake_chat)
    assert client.post(f"/flows/9/scenarios/{normal['id']}/script").status_code == 200
    # A normal id that isn't saved still runs into the existing "save it first" 404.
    assert client.post(f"/flows/9/scenarios/{normal['id']}/run").status_code == 404
    assert client.post("/flows/9/scenarios/0000000000000000/run").status_code == 404


def test_nothing_outside_flow_scenarios_is_touched(client, forbidden):
    ids = [s["id"] for s in _preview(client)["interrupt_scenarios"]]
    _save(client, ids)
    client.post(f"/flows/9/scenarios/{ids[0]}/script")
    client.post(f"/flows/9/scenarios/{ids[0]}/run")
    assert forbidden == []
