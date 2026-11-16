"""Unit tests for app.core.activities.load_replay_context — the scenario a replay is
handed must be shaped so run_scenario() takes the SAME path (dynamic vs scripted) the
original run took. DB lookups are mocked; this verifies the reshape only.
"""
import json

from app.core import activities
from app.core import runner


def _stub_db(monkeypatch, scenario_row: dict) -> None:
    monkeypatch.setattr(activities, "get_conversation", lambda cid: {"id": cid, "run_id": 7, "scenario_id": 3})
    monkeypatch.setattr(activities, "get_scenario", lambda sid: scenario_row)
    monkeypatch.setattr(activities, "get_run", lambda rid: {"id": rid, "agent_id": 11})
    monkeypatch.setattr(activities, "get_agent", lambda aid: {"id": aid, "modality": "chat"})


def _row(**overrides) -> dict:
    row = {
        "id": 3, "title": "t", "user_goal": "g", "test_type": "support",
        "assigned_fault": "none", "expected_behavior": "e",
        "seed_turns_json": json.dumps(["Hi"]), "node_script_json": None,
        "customer_context": None, "max_turns": None,
    }
    row.update(overrides)
    return row


def test_dynamic_scenario_keeps_customer_context_and_max_turns(monkeypatch):
    _stub_db(monkeypatch, _row(seed_turns_json="[]", customer_context="Frustrated caller, order #123", max_turns=4))

    ctx = activities.load_replay_context(99)

    assert ctx["scenario"]["customer_context"] == "Frustrated caller, order #123"
    assert ctx["scenario"]["max_turns"] == 4


def test_scripted_scenario_replays_as_scripted(monkeypatch):
    _stub_db(monkeypatch, _row())

    ctx = activities.load_replay_context(99)

    assert ctx["scenario"]["seed_turns"] == ["Hi"]
    assert ctx["scenario"]["customer_context"] == ""
    assert ctx["scenario"]["max_turns"] is None
    assert ctx["run_id"] == 7 and ctx["agent"]["id"] == 11


async def test_replayed_dynamic_scenario_takes_dynamic_path(monkeypatch):
    """End to end through run_scenario's branch: a replayed dynamic scenario must reach
    _run_dynamic with its max_turns, not fall back to _run_scripted."""
    _stub_db(monkeypatch, _row(seed_turns_json="[]", customer_context="Caller wants a refund", max_turns=2))
    scenario = activities.load_replay_context(99)["scenario"]

    calls = []

    async def fake_dynamic(conv_id, scen, *a, **k):
        calls.append(("dynamic", scen["max_turns"]))

    async def fake_scripted(*a, **k):
        calls.append(("scripted", None))

    monkeypatch.setattr(runner, "get_or_create_conversation", lambda *a: 1)
    monkeypatch.setattr(runner, "clear_messages", lambda *a: None)
    monkeypatch.setattr(runner, "_run_dynamic", fake_dynamic)
    monkeypatch.setattr(runner, "_run_scripted", fake_scripted)

    await runner.run_scenario(7, scenario, {"id": 11})

    assert calls == [("dynamic", 2)]


def test_replay_still_carries_call_variables(monkeypatch):
    script = {"kind": "interrupt", "setup": {"call": {"patient_ref": "adult-en"}}}
    _stub_db(monkeypatch, _row(node_script_json=json.dumps(script)))

    ctx = activities.load_replay_context(99)

    assert ctx["scenario"]["call_variables"] == {"patient_ref": "adult-en"}
    assert ctx["scenario"]["customer_context"] == ""
