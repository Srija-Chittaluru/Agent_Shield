"""POST /flows/{id}/scenarios/{sid}/run for a SAVED interrupt scenario (Part 6).

Real parser / planner / validator and the real Maya fixture as stored flow 9; the
flow_scenarios table, the run rows and the Temporal client are faked, so we can see
exactly what would be submitted to the existing RunGroupWorkflow without running it.
"""
import copy
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core import node_script
from app.core.node_script import PATH_TEST_TYPE
from app.routers import flows as R
from tests.routers.test_flow_interrupt_scenarios import FLOWS, _Table
from tests.routers.test_flow_interrupt_scripts import RECORD, _script_for


class _Rec:
    def __init__(self):
        self.started: list = []
        self.runs: list = []


@pytest.fixture
def table(monkeypatch):
    t = _Table()
    for name, fn in [("insert_flow_scenarios", t.insert), ("upsert_flow_scenario_script", t.upsert),
                     ("get_flow_scenario", t.get), ("list_flow_scenarios", t.list), ("delete_flow_scenario", t.delete)]:
        monkeypatch.setattr(R, name, fn)
    return t


@pytest.fixture
def rec(monkeypatch):
    r = _Rec()

    class _Client:
        async def start_workflow(self, fn, inp, id, task_queue):
            r.started.append({"input": inp, "id": id})

    async def get_client():
        return _Client()

    def insert_run(agent_id, group_id, customer_agent_id):
        r.runs.append((agent_id, group_id, customer_agent_id))
        return 901

    def no(*a, **kw):
        raise AssertionError("must not write test_cases or call the LLM")

    monkeypatch.setattr(R, "get_client", get_client)
    monkeypatch.setattr(R, "insert_run_group", lambda: 501)
    monkeypatch.setattr(R, "insert_run", insert_run)
    monkeypatch.setattr(R, "update_run", lambda *a, **kw: None)
    monkeypatch.setattr(R, "get_agent", lambda aid: {"id": aid, "name": "Maya", "modality": "voice"})
    monkeypatch.setattr(R, "get_customer_agent_by_agent_id", lambda aid: {"id": 1027})
    monkeypatch.setattr(R, "insert_test_case", no)
    monkeypatch.setattr(R, "update_test_case", no)
    monkeypatch.setattr(node_script, "chat", no)
    return r


@pytest.fixture
def flows(monkeypatch):
    f = copy.deepcopy(FLOWS)
    monkeypatch.setattr(R, "get_agent_flow", lambda fid: f.get(fid))
    return f


@pytest.fixture
def client(table, rec, flows):
    app = FastAPI()
    app.include_router(R.router)
    return TestClient(app)


@pytest.fixture
def plan(client):
    body = client.post("/flows/9/scenarios/preview").json()
    return {s["interrupt"]["id"]: s for s in body["interrupt_scenarios"]}


def _save_script(client, item):
    script = _script_for(item)
    res = client.put(f"/flows/9/scenarios/{item['id']}", json={
        "test_goal": script["test_goal"], "turns": script["turns"],
        "setup": {"record": RECORD}, "expectations": script["expectations"],
    })
    assert res.status_code == 200, res.json()
    return res.json()


def _submitted(rec):
    return rec.started[-1]["input"].targets[0].scenarios[0]


@pytest.mark.parametrize("key", ["human_request", "wrong_number", "hold"])
def test_saved_interrupt_script_runs_through_the_existing_pipeline(client, plan, rec, key):
    item = plan[key]
    saved = _save_script(client, item)
    res = client.post(f"/flows/9/scenarios/{item['id']}/run")
    assert res.status_code == 200, res.json()
    assert res.json() == {"run_id": 901, "group_id": 501, "flow_id": 9, "scenario_id": item["id"]}
    assert rec.runs == [(FLOWS[9]["agent_id"], 501, 1027)]
    assert rec.started[0]["id"] == "agentshield-run-group-501"

    s = _submitted(rec)
    assert s["test_type"] == PATH_TEST_TYPE and s["customer_context"] == ""
    spoken = [t["caller_line"] for t in saved["turns"] if t["type"] in ("caller", "interrupt")]
    assert s["seed_turns"] == spoken
    stored = json.loads(s["node_script_json"])
    assert stored["turns"] == saved["turns"] and stored["expectations"] == saved["expectations"]
    assert stored["interrupt"]["outcome"] == item["interrupt"]["outcome"]


def test_end_interrupt_is_the_last_spoken_line(client, plan, rec):
    item = plan["wrong_number"]
    saved = _save_script(client, item)
    client.post(f"/flows/9/scenarios/{item['id']}/run")
    interrupt = next(t for t in saved["turns"] if t["type"] == "interrupt")
    assert _submitted(rec)["seed_turns"][-1] == interrupt["caller_line"]


def test_unsaved_or_unscripted_interrupt_is_409(client, plan, rec, table):
    item = plan["hold"]
    assert client.post(f"/flows/9/scenarios/{item['id']}/run").status_code == 409
    assert client.post("/flows/9/scenarios", json={"scenario_ids": [item["id"]]}).status_code == 200
    res = client.post(f"/flows/9/scenarios/{item['id']}/run")
    assert res.status_code == 409 and "no saved script" in res.json()["detail"]
    assert rec.started == [] and rec.runs == []


def test_stale_saved_path_is_409(client, plan, rec, table):
    item = plan["hold"]
    _save_script(client, item)
    table.rows[0]["path_json"] = json.dumps(item["path"][:-1])
    res = client.post(f"/flows/9/scenarios/{item['id']}/run")
    assert res.status_code == 409 and rec.started == []


def test_saved_script_that_no_longer_validates_is_409(client, plan, rec, table):
    item = plan["human_request"]
    _save_script(client, item)
    script = json.loads(table.rows[0]["script_json"])
    next(t for t in script["turns"] if t["type"] == "interrupt")["interrupt_key"] = "busy"
    table.rows[0]["script_json"] = json.dumps(script)
    res = client.post(f"/flows/9/scenarios/{item['id']}/run")
    assert res.status_code == 409
    errors = res.json()["detail"]["errors"]
    assert "no longer matches" in errors[0] and any("interrupt_key" in e for e in errors[1:])
    assert rec.started == []


def test_flow_changed_since_save_is_409(client, plan, rec, flows):
    item = plan["wrong_number"]
    _save_script(client, item)
    # The interrupt disappears from the source: the saved scenario is no longer planned.
    flows[9]["raw_source"] = flows[9]["raw_source"].replace("wrong_number", "wrong_num_renamed")
    res = client.post(f"/flows/9/scenarios/{item['id']}/run")
    assert res.status_code == 409 and rec.started == []


def test_record_field_no_longer_in_the_flow_is_409(client, plan, rec, table):
    item = plan["hold"]
    _save_script(client, item)
    script = json.loads(table.rows[0]["script_json"])
    script["setup"]["record"]["user.nickname"] = "Danny"
    table.rows[0]["script_json"] = json.dumps(script)
    res = client.post(f"/flows/9/scenarios/{item['id']}/run")
    assert res.status_code == 409 and rec.started == []


# ---------------------------------------------------------------------------
# Call-creation test data (setup.call) — Part 7
# ---------------------------------------------------------------------------
from tests.routers.test_flow_interrupt_scenarios import MAYA_AGENT  # noqa: E402


@pytest.fixture
def maya(monkeypatch):
    agent = dict(MAYA_AGENT)
    monkeypatch.setattr(R, "get_agent", lambda aid: dict(agent, id=aid))
    return agent


def _save_with_call(client, item, call):
    script = _script_for(item)
    return client.put(f"/flows/9/scenarios/{item['id']}", json={
        "test_goal": script["test_goal"], "turns": script["turns"],
        "setup": {"record": RECORD, "call": call}, "expectations": script["expectations"],
    })


def test_preview_lists_the_agents_call_fields(client, maya):
    body = client.post("/flows/9/scenarios/preview").json()
    assert body["call_fields"] == ["call.patient_ref"]


def test_saved_call_data_is_passed_to_the_run(client, plan, rec, maya, table):
    item = plan["wrong_number"]
    res = _save_with_call(client, item, {"patient_ref": " adult-en "})
    assert res.status_code == 200 and res.json()["setup"]["call"] == {"patient_ref": "adult-en"}
    assert client.post(f"/flows/9/scenarios/{item['id']}/run").status_code == 200
    s = _submitted(rec)
    assert s["call_variables"] == {"patient_ref": "adult-en"}
    assert "Call created with: patient_ref = adult-en" in s["expected_behavior"]
    assert json.loads(s["node_script_json"])["setup"]["call"] == {"patient_ref": "adult-en"}


def test_scripts_without_call_data_run_as_before(client, plan, rec, maya):
    item = plan["hold"]
    _save_script(client, item)
    client.post(f"/flows/9/scenarios/{item['id']}/run")
    assert "call_variables" not in _submitted(rec)


def test_unmapped_call_data_is_rejected_on_save(client, plan, maya, table):
    res = _save_with_call(client, plan["hold"], {"patient_id": "x"})
    assert res.status_code == 422 and "not mapped" in res.json()["detail"]["errors"][0]
    assert table.rows == []


def test_call_data_for_an_agent_without_a_mapping_is_rejected(client, plan, table):
    # rec's default agent: voice, no native_ws template.
    res = _save_with_call(client, plan["hold"], {"patient_ref": "adult-en"})
    assert res.status_code == 422 and table.rows == []


def test_agent_mapping_removed_since_save_is_409(client, plan, rec, maya):
    item = plan["hold"]
    assert _save_with_call(client, item, {"patient_ref": "adult-en"}).status_code == 200
    maya["request_template"] = '{"client": "client_a", "flow": "maya_renewal"}'
    res = client.post(f"/flows/9/scenarios/{item['id']}/run")
    assert res.status_code == 409 and "not mapped" in res.json()["detail"]["errors"][0]
    assert rec.started == []
