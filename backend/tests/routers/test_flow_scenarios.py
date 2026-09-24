"""Router-level tests for flow-level scenario planning (Phase B):

    POST /flows/{flow_id}/scenarios/preview   calculate only
    POST /flows/{flow_id}/scenarios           explicit, additive, idempotent save
    GET  /flows/{flow_id}/scenarios           list saved

Isolated FastAPI app + monkeypatched app.db calls, same convention as the other
routers/test_*.py files. app.core.flow_graph is NOT mocked — these exercise the real
planner end to end.
"""
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import app.core.llm as llm
from app.core import flow_graph
from app.routers import flows as flows_router


def _flow_row(flow_id, node_ids, edges, names=None, **overrides):
    names = names or {}
    row = {
        "id": flow_id, "agent_id": 42, "name": "flow.json", "source_format": "json",
        "created_at": "t", "extraction_method": "deterministic",
        "nodes_json": json.dumps([
            {"id": n, "name": names.get(n, n), "type": "", "purpose": "", "expected_inputs": []}
            for n in node_ids
        ]),
        "edges_json": json.dumps([{"from": a, "to": b} for a, b in edges]),
    }
    row.update(overrides)
    return row


SPEC_NODES = ["greeting", "ask_name", "validate_name", "retry_name", "ask_dob", "ask_email", "complete"]
SPEC_EDGES = [
    ("greeting", "ask_name"), ("ask_name", "validate_name"),
    ("validate_name", "ask_dob"), ("validate_name", "retry_name"),
    ("retry_name", "validate_name"), ("ask_dob", "ask_email"), ("ask_email", "complete"),
]
SPEC_NAMES = {
    "greeting": "Greeting", "ask_name": "Ask Name", "validate_name": "Validate Name",
    "retry_name": "Ask Name Again", "ask_dob": "Ask DOB", "ask_email": "Ask Email",
    "complete": "Complete",
}

FLOWS = {
    1: _flow_row(1, SPEC_NODES, SPEC_EDGES, SPEC_NAMES),
    2: _flow_row(2, ["a", "b", "c"], [("a", "b"), ("b", "c")]),
    3: _flow_row(3, ["a", "b", "c", "d"], [("a", "b"), ("b", "c"), ("b", "d")]),
    4: _flow_row(4, ["a", "b", "c"], [("a", "b"), ("b", "c"), ("c", "b")]),
    5: _flow_row(
        5, ["start", "decide", "complete", "transfer", "hangup"],
        [("start", "decide"), ("decide", "complete"), ("decide", "transfer"), ("decide", "hangup")],
    ),
    6: _flow_row(6, ["a", "b", "c"], [("a", "c"), ("b", "c")]),
    7: _flow_row(7, ["a", "b", "orphan"], [("a", "b")]),
    90: _flow_row(90, ["a"], [], nodes_json="{not json"),
    91: _flow_row(91, ["a"], [], nodes_json=json.dumps({"id": "a"})),
    92: _flow_row(92, ["a"], [], edges_json=json.dumps([{"from": "a"}])),
    93: _flow_row(93, ["a"], [], nodes_json=json.dumps([{"name": "no id"}])),
}


def _comb_row(flow_id, levels):
    ids, edges = ["root"], [("root", "b0")]
    for i in range(levels):
        ids += [f"b{i}", f"x{i}", f"y{i}"]
        nxt = f"b{i + 1}" if i + 1 < levels else "end"
        edges += [(f"b{i}", f"x{i}"), (f"b{i}", f"y{i}"), (f"x{i}", nxt), (f"y{i}", nxt)]
    return _flow_row(flow_id, ids + ["end"], edges)


FLOWS[8] = _comb_row(8, 30)  # 31 candidates available


class _FakeScenarioStore:
    """In-memory stand-in for flow_scenarios with the real table's UNIQUE
    (flow_id, scenario_key) + ON CONFLICT DO NOTHING semantics."""

    def __init__(self):
        self.rows: list[dict] = []

    def insert(self, flow_id, scenarios):
        inserted = 0
        for s in scenarios:
            if any(r["flow_id"] == flow_id and r["scenario_key"] == s["id"] for r in self.rows):
                continue
            self.rows.append({
                "id": len(self.rows) + 1, "flow_id": flow_id, "scenario_key": s["id"],
                "category": s["category"], "name": s["name"], "test_goal": s.get("test_goal"),
                "path_json": json.dumps(s["path"]),
                "covered_edges_json": json.dumps(s["covered_edges"]),
                "covered_terminals_json": json.dumps(s["covered_terminals"]),
                "contains_retry": bool(s["contains_retry"]), "created_at": "t",
            })
            inserted += 1
        return inserted

    def list(self, flow_id):
        return [r for r in self.rows if r["flow_id"] == flow_id]


@pytest.fixture
def store(monkeypatch):
    s = _FakeScenarioStore()
    monkeypatch.setattr(flows_router, "insert_flow_scenarios", s.insert)
    monkeypatch.setattr(flows_router, "list_flow_scenarios", s.list)
    return s


@pytest.fixture
def client(monkeypatch, store):
    app = FastAPI()
    app.include_router(flows_router.router)
    monkeypatch.setattr(flows_router, "get_agent_flow", lambda flow_id: FLOWS.get(flow_id))
    return TestClient(app)


@pytest.fixture
def no_side_effects(monkeypatch):
    """Make every LLM / test-case / run / workflow entry point the router can reach
    blow up if touched."""
    calls: list[str] = []

    def forbid(name):
        def boom(*a, **kw):
            calls.append(name)
            raise AssertionError(f"preview must not call {name}")
        return boom

    for name in (
        "generate_node_script", "extract_flow_with_llm", "insert_test_case", "update_test_case",
        "insert_run", "insert_run_group", "update_run", "get_client", "insert_agent_flow",
        "insert_flow_scenarios",
    ):
        monkeypatch.setattr(flows_router, name, forbid(name))
    monkeypatch.setattr(llm, "chat", forbid("llm.chat"))
    return calls


def _preview(client, flow_id, **body):
    res = client.post(f"/flows/{flow_id}/scenarios/preview", json=body or None)
    assert res.status_code == 200, res.text
    return res.json()


def _paths(body):
    return [s["path"] for s in body["scenarios"]]


# ---------------------------------------------------------------------------
# Preview: planning results
# ---------------------------------------------------------------------------
def test_spec_flow_preview_returns_planner_output_with_ids_and_names(client):
    body = _preview(client, 1)
    assert body["flow_id"] == 1
    assert body["scenario_count"] == 3
    assert [(s["category"], s["name"]) for s in body["scenarios"]] == [
        ("happy_path", "Happy Path"),
        ("branch", "Branch Scenario"),
        ("recovery", "Retry / Recovery"),
    ]
    assert _paths(body)[1] == ["greeting", "ask_name", "validate_name", "retry_name"]
    assert body["node_names"]["retry_name"] == "Ask Name Again"
    assert body["graph"]["back_edges"] == [["retry_name", "validate_name"]]


def test_preview_is_exactly_flow_graph_output_plus_id_and_name(client):
    body = _preview(client, 1)
    graph = {
        "nodes": json.loads(FLOWS[1]["nodes_json"]), "edges": json.loads(FLOWS[1]["edges_json"]),
    }
    expected = flow_graph.generate_candidate_paths(graph)
    for scenario, candidate in zip(body["scenarios"], expected, strict=True):
        extra = {k: v for k, v in scenario.items() if k not in candidate}
        assert set(extra) == {"id", "name"}
        assert {k: scenario[k] for k in candidate} == candidate


# 1. Linear
def test_linear_flow_has_one_primary_path(client):
    body = _preview(client, 2)
    assert _paths(body) == [["a", "b", "c"]]
    assert body["scenarios"][0]["category"] == "primary_path"
    assert body["scenarios"][0]["name"] == "Primary Path"
    assert body["graph"]["roots"] == ["a"] and body["graph"]["terminals"] == ["c"]


# 2. Branch
def test_branch_flow_covers_both_outgoing_edges(client):
    body = _preview(client, 3)
    covered = {tuple(e) for s in body["scenarios"] for e in s["covered_edges"]}
    assert {("b", "c"), ("b", "d")} <= covered
    assert body["graph"]["branch_nodes"] == ["b"]


# 3. Retry loop
def test_retry_loop_returns_bounded_retry_path(client):
    body = _preview(client, 4)
    retry = [s for s in body["scenarios"] if s["category"] == "retry"]
    assert [s["path"] for s in retry] == [["a", "b", "c", "b"]]
    assert retry[0]["name"] == "Retry Loop"
    assert retry[0]["contains_retry"] is True
    for s in body["scenarios"]:
        assert all(s["path"].count(n) <= 2 for n in s["path"])


# 4. Multiple terminals
def test_multiple_terminals_are_all_represented(client):
    body = _preview(client, 5)
    assert {s["path"][-1] for s in body["scenarios"]} == {"complete", "transfer", "hangup"}
    # Repeated categories are numbered, not renamed with guessed meanings.
    assert [s["name"] for s in body["scenarios"]] == [
        "Happy Path", "Branch Scenario", "Branch Scenario 2",
    ]


# 5. Multiple roots
def test_multiple_roots_each_get_a_primary_path(client):
    body = _preview(client, 6)
    assert body["graph"]["roots"] == ["a", "b"]
    assert _paths(body) == [["a", "c"], ["b", "c"]]
    assert [s["name"] for s in body["scenarios"]] == ["Primary Path", "Primary Path 2"]


# 6. Unreachable nodes
def test_unreachable_nodes_are_reported_and_excluded_from_paths(client):
    body = _preview(client, 7)
    assert body["graph"]["unreachable"] == ["orphan"]
    assert all("orphan" not in s["path"] for s in body["scenarios"])
    assert body["node_names"]["orphan"] == "orphan"


# 7. Scenario limit
def test_scenario_limit_defaults_to_20_and_is_respected(client):
    assert _preview(client, 8)["scenario_count"] == 20
    small = _preview(client, 8, max_scenarios=5)
    assert small["scenario_count"] == 5 and small["max_scenarios"] == 5


def test_smaller_limit_is_a_prefix_so_ids_and_names_are_stable(client):
    big = _preview(client, 8, max_scenarios=30)["scenarios"]
    small = _preview(client, 8, max_scenarios=7)["scenarios"]
    assert small == big[:7]


@pytest.mark.parametrize("bad", [0, -1, 51, 1000])
def test_out_of_range_limit_is_rejected(client, bad):
    res = client.post("/flows/8/scenarios/preview", json={"max_scenarios": bad})
    assert res.status_code == 422


def test_preview_is_deterministic(client):
    assert _preview(client, 1) == _preview(client, 1)


# 8. Not found
def test_unknown_flow_is_404(client):
    assert client.post("/flows/999/scenarios/preview").status_code == 404
    assert client.post("/flows/999/scenarios").status_code == 404
    assert client.get("/flows/999/scenarios").status_code == 404


# 9. Malformed flow
@pytest.mark.parametrize("flow_id", [90, 91, 92, 93])
def test_malformed_stored_graph_is_422(client, flow_id):
    res = client.post(f"/flows/{flow_id}/scenarios/preview")
    assert res.status_code == 422
    assert res.json()["detail"]["errors"]


# 10-12. No LLM, no test_cases, no runs, no writes at all
def test_preview_has_no_side_effects(client, no_side_effects):
    for flow_id in (1, 2, 3, 4, 5, 6, 7, 8):
        _preview(client, flow_id)
    assert no_side_effects == []


def test_preview_does_not_save_scenarios(client, store):
    _preview(client, 1)
    assert store.rows == []


# ---------------------------------------------------------------------------
# Explicit save
# ---------------------------------------------------------------------------
def test_save_persists_the_previewed_plan(client, store):
    preview = _preview(client, 1)
    res = client.post("/flows/1/scenarios")
    assert res.status_code == 200
    body = res.json()
    assert body["inserted"] == 3
    saved = body["scenarios"]
    assert [s["id"] for s in saved] == [s["id"] for s in preview["scenarios"]]
    for stored, planned in zip(saved, preview["scenarios"]):
        for key in ("category", "name", "path", "covered_edges", "covered_terminals", "contains_retry", "path_length"):
            assert stored[key] == planned[key]
        assert stored["test_goal"] is None


def test_save_is_idempotent_and_never_replaces(client, store):
    client.post("/flows/1/scenarios")
    again = client.post("/flows/1/scenarios").json()
    assert again["inserted"] == 0
    assert len(again["scenarios"]) == 3
    assert len(store.rows) == 3


def test_save_selected_ids_only(client, store):
    preview = _preview(client, 1)
    wanted = [preview["scenarios"][2]["id"], preview["scenarios"][0]["id"]]
    body = client.post("/flows/1/scenarios", json={"scenario_ids": wanted}).json()
    assert body["inserted"] == 2
    # Stored in plan order, not request order.
    assert [s["category"] for s in body["scenarios"]] == ["happy_path", "recovery"]


def test_save_resolves_ids_beyond_the_default_limit(client, store):
    beyond = _preview(client, 8, max_scenarios=30)["scenarios"][25]
    body = client.post("/flows/8/scenarios", json={"scenario_ids": [beyond["id"]]}).json()
    assert [s["id"] for s in body["scenarios"]] == [beyond["id"]]
    assert body["scenarios"][0]["name"] == beyond["name"]


def test_save_rejects_unknown_ids_without_writing(client, store):
    res = client.post("/flows/1/scenarios", json={"scenario_ids": ["made-up"]})
    assert res.status_code == 422
    assert store.rows == []


def test_saving_one_flow_does_not_touch_another(client, store):
    client.post("/flows/1/scenarios")
    client.post("/flows/2/scenarios")
    assert len(client.get("/flows/1/scenarios").json()["scenarios"]) == 3
    assert len(client.get("/flows/2/scenarios").json()["scenarios"]) == 1


def test_save_creates_no_test_cases_or_runs(client, monkeypatch, store):
    def boom(*a, **kw):
        raise AssertionError("save must not create test cases or runs")

    for name in ("insert_test_case", "update_test_case", "insert_run", "insert_run_group", "get_client"):
        monkeypatch.setattr(flows_router, name, boom)
    monkeypatch.setattr(llm, "chat", boom)
    assert client.post("/flows/1/scenarios").status_code == 200


def test_list_is_empty_before_any_save(client):
    assert client.get("/flows/1/scenarios").json() == {"flow_id": 1, "scenarios": []}
