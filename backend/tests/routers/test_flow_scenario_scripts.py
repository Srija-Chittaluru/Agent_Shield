"""Router-level tests for flow scenario scripts (Phase D):

    POST   /flows/{flow_id}/scenarios/{scenario_id}/script   generate / regenerate (draft)
    PUT    /flows/{flow_id}/scenarios/{scenario_id}          save reviewed script (upsert)
    DELETE /flows/{flow_id}/scenarios/{scenario_id}          delete saved scenario

Isolated FastAPI app + in-memory fake of the flow_scenarios table. The REAL planner
(app.core.flow_graph) and the REAL generate/validate code (app.core.node_script) run;
only the LLM call itself (node_script.chat) is faked.
"""
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core import node_script
from app.routers import flows as flows_router


def _flow_row(flow_id, nodes, edges):
    return {
        "id": flow_id, "agent_id": 42, "name": "flow.json", "source_format": "json",
        "created_at": "t", "extraction_method": "deterministic",
        "nodes_json": json.dumps([{"id": n, "name": name, "type": "", "purpose": f"{name} step"} for n, name in nodes]),
        "edges_json": json.dumps([{"from": a, "to": b} for a, b in edges]),
    }


SPEC_NODES = [
    ("greeting", "Greeting"), ("ask_name", "Ask Name"), ("validate_name", "Validate Name"),
    ("retry_name", "Ask Name Again"), ("ask_dob", "Ask DOB"), ("ask_email", "Ask Email"),
    ("complete", "Complete"),
]
SPEC_EDGES = [
    ("greeting", "ask_name"), ("ask_name", "validate_name"), ("validate_name", "ask_dob"),
    ("validate_name", "retry_name"), ("retry_name", "validate_name"),
    ("ask_dob", "ask_email"), ("ask_email", "complete"),
]
FLOWS = {
    1: _flow_row(1, SPEC_NODES, SPEC_EDGES),
    2: _flow_row(2, [("a", "A"), ("b", "B")], [("a", "b")]),
}
HAPPY = ["greeting", "ask_name", "validate_name", "ask_dob", "ask_email", "complete"]
BRANCH = ["greeting", "ask_name", "validate_name", "retry_name"]
HAPPY_ID = flows_router._scenario_key(HAPPY)
BRANCH_ID = flows_router._scenario_key(BRANCH)


def _script_for(path, goal="Verify the path.", **overrides):
    turns = [
        {"step": i + 1, "node_id": n, "expected_agent_behavior": f"Agent handles {n}.",
         "caller_line": f"Caller reply number {i + 1}."}
        for i, n in enumerate(path[:5])
    ]
    return {"test_goal": goal, "turns": turns, **overrides}


class _FakeScenarioTable:
    """flow_scenarios in memory, with the real UNIQUE (flow_id, scenario_key) semantics."""

    def __init__(self):
        self.rows: list[dict] = []

    def _find(self, flow_id, key):
        return next((r for r in self.rows if r["flow_id"] == flow_id and r["scenario_key"] == key), None)

    def get(self, flow_id, key):
        return self._find(flow_id, key)

    def upsert(self, flow_id, scenario, test_goal, turns):
        row = self._find(flow_id, scenario["id"])
        if row is None:
            row = {
                "id": len(self.rows) + 1, "flow_id": flow_id, "scenario_key": scenario["id"],
                "category": scenario["category"], "name": scenario["name"],
                "path_json": json.dumps(scenario["path"]),
                "covered_edges_json": json.dumps(scenario["covered_edges"]),
                "covered_terminals_json": json.dumps(scenario["covered_terminals"]),
                "contains_retry": scenario["contains_retry"], "created_at": "t",
            }
            self.rows.append(row)
        row["test_goal"] = test_goal
        row["script_json"] = json.dumps(turns)
        return row

    def insert_plan(self, flow_id, scenarios):
        n = 0
        for s in scenarios:
            if self._find(flow_id, s["id"]) is None:
                self.upsert(flow_id, s, None, [])
                self._find(flow_id, s["id"])["script_json"] = None
                n += 1
        return n

    def delete(self, flow_id, key):
        row = self._find(flow_id, key)
        if row is None:
            return False
        self.rows.remove(row)
        return True

    def list(self, flow_id):
        return [r for r in self.rows if r["flow_id"] == flow_id]


@pytest.fixture
def table(monkeypatch):
    t = _FakeScenarioTable()
    monkeypatch.setattr(flows_router, "get_flow_scenario", t.get)
    monkeypatch.setattr(flows_router, "upsert_flow_scenario_script", t.upsert)
    monkeypatch.setattr(flows_router, "delete_flow_scenario", t.delete)
    monkeypatch.setattr(flows_router, "insert_flow_scenarios", t.insert_plan)
    monkeypatch.setattr(flows_router, "list_flow_scenarios", t.list)
    return t


@pytest.fixture
def llm(monkeypatch):
    """Fake node_script.chat: records every prompt; replies with `llm.reply` (a dict, or
    a callable taking the prompt and returning one)."""

    class _LLM:
        prompts: list[str] = []
        reply = None

    fake = _LLM()
    fake.prompts = []

    async def chat(system, messages, json_mode=False, **kw):
        prompt = messages[0]["content"]
        fake.prompts.append(prompt)
        return fake.reply(prompt) if callable(fake.reply) else fake.reply

    monkeypatch.setattr(node_script, "chat", chat)
    return fake


@pytest.fixture
def forbid_execution(monkeypatch):
    def boom(*a, **kw):
        raise AssertionError("scripts must not create test cases or runs")

    for name in ("insert_test_case", "update_test_case", "insert_run", "insert_run_group", "update_run", "get_client"):
        monkeypatch.setattr(flows_router, name, boom)


@pytest.fixture
def client(monkeypatch, table, forbid_execution):
    app = FastAPI()
    app.include_router(flows_router.router)
    monkeypatch.setattr(flows_router, "get_agent_flow", lambda flow_id: FLOWS.get(flow_id))
    return TestClient(app)


# ---------------------------------------------------------------------------
# Generate
# ---------------------------------------------------------------------------
def test_generate_script_for_a_planned_scenario(client, llm):
    llm.reply = _script_for(HAPPY)
    res = client.post(f"/flows/1/scenarios/{HAPPY_ID}/script")
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["flow_id"] == 1 and body["scenario_id"] == HAPPY_ID
    assert body["name"] == "Happy Path" and body["category"] == "happy_path"
    assert body["path"] == HAPPY
    assert body["test_goal"] == "Verify the path."
    assert [t["node_id"] for t in body["turns"]] == HAPPY[:5]
    assert all(set(t) == {"step", "node_id", "expected_agent_behavior", "caller_line"} for t in body["turns"])


def test_generation_prompt_carries_the_exact_server_side_path(client, llm):
    llm.reply = _script_for(BRANCH)
    client.post(f"/flows/1/scenarios/{BRANCH_ID}/script")
    prompt = llm.prompts[0]
    for i, node_id in enumerate(BRANCH):
        assert f"Step {i + 1}:\nname: {dict(SPEC_NODES)[node_id]}\nid: {node_id}" in prompt
    assert "Step 5" not in prompt  # nothing beyond the path
    assert "other branches not taken from here: Ask DOB" in prompt


def test_generate_looks_up_the_requested_flow(client, llm):
    ab = flows_router._scenario_key(["a", "b"])
    llm.reply = _script_for(["a", "b"])
    assert client.post(f"/flows/2/scenarios/{ab}/script").json()["path"] == ["a", "b"]
    # The same id means nothing on another flow.
    assert client.post(f"/flows/1/scenarios/{ab}/script").status_code == 404


def test_generate_uses_the_saved_path_for_a_saved_scenario(client, llm, table):
    llm.reply = _script_for(HAPPY)
    client.put(f"/flows/1/scenarios/{HAPPY_ID}", json=_script_for(HAPPY))
    res = client.post(f"/flows/1/scenarios/{HAPPY_ID}/script")
    assert res.status_code == 200 and res.json()["path"] == HAPPY


def test_regenerate_uses_the_same_path_and_saves_nothing(client, llm, table):
    replies = iter([_script_for(HAPPY, goal="First draft."), _script_for(HAPPY, goal="Second draft.")])
    llm.reply = lambda prompt: next(replies)
    first = client.post(f"/flows/1/scenarios/{HAPPY_ID}/script").json()
    second = client.post(f"/flows/1/scenarios/{HAPPY_ID}/script").json()
    assert first["path"] == second["path"] == HAPPY
    assert llm.prompts[0] == llm.prompts[1]
    assert (first["test_goal"], second["test_goal"]) == ("First draft.", "Second draft.")
    assert table.rows == []


def test_client_supplied_path_is_ignored(client, llm):
    llm.reply = _script_for(HAPPY)
    res = client.post(f"/flows/1/scenarios/{HAPPY_ID}/script", json={"path": ["complete", "greeting"]})
    assert res.json()["path"] == HAPPY
    assert "Step 1:\nname: Greeting" in llm.prompts[0]


@pytest.mark.parametrize("reply,needle", [
    ("not json at all", "JSON object"),
    ({"test_goal": "g", "turns": "nope"}, "turns must be a non-empty array"),
    ({"turns": _script_for(HAPPY)["turns"]}, "test_goal"),
    ({"test_goal": "g", "turns": [{"step": 1, "node_id": "greeting", "expected_agent_behavior": "x"}]}, "caller_line"),
    ({"test_goal": "g", "turns": [{"step": 1, "node_id": "ask_email", "expected_agent_behavior": "x", "caller_line": "y"}]}, "must be 'greeting'"),
    ({"test_goal": "g", "turns": [{"step": 9, "node_id": "complete", "expected_agent_behavior": "x", "caller_line": "y"}]}, "step must be"),
])
def test_malformed_model_output_is_rejected_not_repaired(client, llm, reply, needle):
    llm.reply = reply
    res = client.post(f"/flows/1/scenarios/{HAPPY_ID}/script")
    assert res.status_code == 422
    assert needle in " ".join(res.json()["detail"]["errors"])


def test_unknown_scenario_is_404(client, llm):
    assert client.post("/flows/1/scenarios/0000000000000000/script").status_code == 404
    assert llm.prompts == []


def test_missing_flow_is_404(client, llm):
    assert client.post(f"/flows/999/scenarios/{HAPPY_ID}/script").status_code == 404
    assert client.put(f"/flows/999/scenarios/{HAPPY_ID}", json=_script_for(HAPPY)).status_code == 404
    assert client.delete(f"/flows/999/scenarios/{HAPPY_ID}").status_code == 404


@pytest.mark.parametrize("stored_path", [
    ["greeting", "complete"],                 # not a real edge in the flow
    ["greeting", "ask_name", "ghost"],         # not a real node
])
def test_tampered_stored_path_is_refused(client, llm, table, stored_path):
    table.upsert(1, {
        "id": HAPPY_ID, "category": "happy_path", "name": "Happy Path", "path": stored_path,
        "covered_edges": [], "covered_terminals": [], "contains_retry": False,
    }, "g", [])
    assert client.post(f"/flows/1/scenarios/{HAPPY_ID}/script").status_code == 409
    assert llm.prompts == []


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
def test_save_persists_scenario_and_script(client, table):
    res = client.put(f"/flows/1/scenarios/{HAPPY_ID}", json=_script_for(HAPPY, goal="  Verify it.  "))
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["id"] == HAPPY_ID and body["path"] == HAPPY and body["name"] == "Happy Path"
    assert body["test_goal"] == "Verify it."
    assert [t["node_id"] for t in body["turns"]] == HAPPY[:5]
    listed = client.get("/flows/1/scenarios").json()["scenarios"]
    assert [s["id"] for s in listed] == [HAPPY_ID] and listed[0]["turns"] == body["turns"]


def test_repeated_save_is_idempotent_and_updates_in_place(client, table):
    client.put(f"/flows/1/scenarios/{HAPPY_ID}", json=_script_for(HAPPY, goal="v1"))
    client.put(f"/flows/1/scenarios/{HAPPY_ID}", json=_script_for(HAPPY, goal="v1"))
    edited = _script_for(HAPPY, goal="v2")
    edited["turns"][1]["caller_line"] = "My name is Srija Chittaluru."
    body = client.put(f"/flows/1/scenarios/{HAPPY_ID}", json=edited).json()
    assert len(table.rows) == 1
    assert body["test_goal"] == "v2"
    assert body["turns"][1]["caller_line"] == "My name is Srija Chittaluru."
    assert body["path"] == HAPPY


def test_save_adds_a_script_to_a_plan_only_scenario(client, table):
    client.post("/flows/1/scenarios")  # Phase B bulk plan save — no scripts
    assert all(r["script_json"] is None for r in table.rows)
    client.put(f"/flows/1/scenarios/{HAPPY_ID}", json=_script_for(HAPPY))
    assert len(table.rows) == 3
    assert client.get("/flows/1/scenarios").json()["scenarios"][0]["turns"]


@pytest.mark.parametrize("mutate,needle", [
    (lambda s: s["turns"][0].update(node_id="complete"), "must be 'greeting'"),
    (lambda s: s["turns"].reverse(), "turns follow the path in order"),
    (lambda s: s["turns"][0].update(caller_line=""), "caller_line"),
    (lambda s: s.update(test_goal=""), "test_goal"),
    (lambda s: s["turns"][0].update(caller_line="I'm [name]."), "placeholder"),
])
def test_edited_script_is_revalidated_against_the_path_on_save(client, table, mutate, needle):
    script = _script_for(HAPPY)
    mutate(script)
    res = client.put(f"/flows/1/scenarios/{HAPPY_ID}", json=script)
    assert res.status_code == 422
    assert needle in " ".join(res.json()["detail"]["errors"])
    assert table.rows == []


def test_save_ignores_a_client_supplied_path(client, table):
    body = client.put(
        f"/flows/1/scenarios/{HAPPY_ID}", json={**_script_for(HAPPY), "path": ["complete"]}
    ).json()
    assert body["path"] == HAPPY


def test_save_of_unknown_scenario_is_404(client, table):
    assert client.put("/flows/1/scenarios/0000000000000000", json=_script_for(HAPPY)).status_code == 404
    assert table.rows == []


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------
def test_delete_removes_only_that_saved_scenario(client, table):
    client.put(f"/flows/1/scenarios/{HAPPY_ID}", json=_script_for(HAPPY))
    client.put(f"/flows/1/scenarios/{BRANCH_ID}", json=_script_for(BRANCH))
    ab = flows_router._scenario_key(["a", "b"])
    client.put(f"/flows/2/scenarios/{ab}", json=_script_for(["a", "b"]))

    res = client.delete(f"/flows/1/scenarios/{HAPPY_ID}")
    assert res.status_code == 200
    assert res.json() == {"flow_id": 1, "scenario_id": HAPPY_ID, "deleted": True}
    assert [s["id"] for s in client.get("/flows/1/scenarios").json()["scenarios"]] == [BRANCH_ID]
    assert len(client.get("/flows/2/scenarios").json()["scenarios"]) == 1
    # The flow itself is untouched and the path can still be previewed and re-saved.
    assert HAPPY_ID in [s["id"] for s in client.post("/flows/1/scenarios/preview").json()["scenarios"]]


def test_delete_of_an_unsaved_scenario_is_404(client, table):
    assert client.delete(f"/flows/1/scenarios/{HAPPY_ID}").status_code == 404


def test_delete_twice(client, table):
    client.put(f"/flows/1/scenarios/{HAPPY_ID}", json=_script_for(HAPPY))
    assert client.delete(f"/flows/1/scenarios/{HAPPY_ID}").status_code == 200
    assert client.delete(f"/flows/1/scenarios/{HAPPY_ID}").status_code == 404
