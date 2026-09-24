"""Tests for app.core.flow_scenario_planner: base-path selection from flow_graph's
candidates, and one interrupt scenario per declared interrupt. Pure unit tests; the
real Maya fixture plus small synthetic flows.
"""
from pathlib import Path

import pytest
import yaml

from app.core import flow_scenario_planner
from app.core.flow_graph import HARD_MAX_SCENARIOS, generate_candidate_paths
from app.core.flow_parser import parse_flow
from app.core.flow_scenario_planner import plan_flow_scenarios

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "flows" / "maya_renewal_modular.yaml"

UNCONDITIONAL = [
    "do_not_call", "voicemail", "busy", "wrong_number", "hold", "human_request",
    "bereavement", "incarceration", "self_harm", "unsafe_at_home",
]


@pytest.fixture(scope="module")
def maya():
    flow = parse_flow(FIXTURE.read_text(), "yaml")
    return flow, plan_flow_scenarios(flow)


def _by_id(plan):
    return {s["interrupt"]["id"]: s for s in plan["interrupt_scenarios"]}


def _steps(scenario, kind):
    return next(s["steps"] for s in scenario["segments"] if s["kind"] == kind)


def _modular(steps, interrupts=None):
    doc = {"entry": steps[0]["id"], "steps": steps}
    if interrupts is not None:
        doc["interrupts"] = interrupts
    return parse_flow(yaml.safe_dump(doc, sort_keys=False), "yaml")


# ---------------------------------------------------------------------------
# flow_graph is reused, not replaced
# ---------------------------------------------------------------------------
def test_path_scenarios_are_exactly_flow_graphs_output(maya):
    flow, plan = maya
    assert plan["path_scenarios"] == generate_candidate_paths(flow)


def test_imports_nothing_but_flow_graph():
    """No LLM, database, network, or parser: the planner only consumes parsed data."""
    source = Path(flow_scenario_planner.__file__).read_text()
    imports = [line.strip() for line in source.splitlines() if line.startswith(("import ", "from "))]
    assert imports == ["import re", "from typing import Optional", "from app.core.flow_graph import ("]


# ---------------------------------------------------------------------------
# Base path (Maya)
# ---------------------------------------------------------------------------
def test_base_is_an_existing_candidate_not_the_early_exit(maya):
    flow, plan = maya
    base = plan["base"]
    all_candidates = generate_candidate_paths(flow, HARD_MAX_SCENARIOS)
    assert base["path"] == all_candidates[base["candidate_index"]]["path"]
    assert base["path"] != plan["path_scenarios"][0]["path"]  # the early-exit primary path
    assert plan["path_scenarios"][0]["path"][-1] == "end_another_time"
    assert base["path"][0] == "greet" and base["path"][-1] == "closing"


def test_base_is_the_renewal_completion_conversation(maya):
    _, plan = maya
    path = plan["base"]["path"]
    for step in ("ask_identity", "ask_dob", "thank_for_verifying", "ask_packet_received",
                 "ask_already_sent", "ask_online_account", "ask_more_questions", "ask_send_date"):
        assert step in path, step
    # It takes route_dob's "date matches" case, NOT its declared default (say_dob_failed):
    # the default is not assumed to be the happy path.
    assert path[path.index("route_dob") + 1] == "thank_for_verifying"
    assert "say_dob_failed" not in path


def test_base_has_the_most_caller_turns_among_eligible_candidates(maya):
    flow, plan = maya
    kinds = {n["id"]: n["type"] for n in flow["nodes"]}
    asks = lambda p: sum(kinds[n] == "ask" for n in p)  # noqa: E731
    eligible = [c for c in generate_candidate_paths(flow, HARD_MAX_SCENARIOS)
                if c["path"][0] == "greet" and not c["contains_retry"]]
    assert asks(plan["base"]["path"]) == max(asks(c["path"]) for c in eligible)


def test_base_does_not_depend_on_the_display_cap(maya):
    flow, plan = maya
    assert plan_flow_scenarios(flow, max_scenarios=1)["base"] == plan["base"]
    assert len(plan_flow_scenarios(flow, max_scenarios=3)["path_scenarios"]) == 3


# ---------------------------------------------------------------------------
# Interrupt scenarios (Maya)
# ---------------------------------------------------------------------------
def test_one_plannable_scenario_per_interrupt_in_file_order(maya):
    flow, plan = maya
    assert [s["interrupt"]["id"] for s in plan["interrupt_scenarios"]] == [i["id"] for i in flow["interrupts"]]
    assert all(s["plannable"] for s in plan["interrupt_scenarios"])
    assert [s["key"] for s in plan["interrupt_scenarios"]][:2] == ["interrupt:do_not_call", "interrupt:voicemail"]


def test_unconditional_interrupts_fire_at_the_first_caller_turn_after_the_greeting(maya):
    _, plan = maya
    by_id = _by_id(plan)
    for key in UNCONDITIONAL:
        s = by_id[key]
        assert s["placement"] == {"policy": "after_greeting", "convention": True, "step": "ask_good_time"}, key
        assert _steps(s, "prefix") == ["greet", "ask_good_time"]


def test_prove_identity_fires_after_the_dob_step(maya):
    _, plan = maya
    s = _by_id(plan)["prove_identity"]
    assert s["placement"] == {
        "policy": "after_field_confirmed", "field": "dob", "field_step": "ask_dob",
        "convention": True, "step": "ask_packet_received",
    }
    assert _steps(s, "prefix")[-3:] == ["route_dob", "thank_for_verifying", "ask_packet_received"]
    assert _steps(s, "target") == ["ask_agent_verified"]
    # ask_agent_verified's only way on is its declared `resume`: back to the interrupted step.
    resumed = next(g for g in s["segments"] if g["kind"] == "resumed")
    assert resumed["via"] == "ask_agent_verified"
    assert resumed["steps"][0] == "ask_packet_received" and resumed["steps"][-1] == "closing"


def test_prove_identity_preverify_fires_at_the_dob_step_and_resumes(maya):
    _, plan = maya
    s = _by_id(plan)["prove_identity_preverify"]
    assert s["placement"]["policy"] == "before_field_confirmed"
    assert s["placement"]["step"] == "ask_dob" == s["placement"]["field_step"]
    assert _steps(s, "prefix")[-1] == "ask_dob"
    assert _steps(s, "resumed")[0] == "ask_dob"


def test_when_conditions_are_carried_verbatim_not_evaluated(maya):
    _, plan = maya
    by_id = _by_id(plan)
    assert by_id["prove_identity"]["interrupt"]["when"] == "status.dob == 'confirmed'"
    assert by_id["prove_identity_preverify"]["interrupt"]["when"] == "status.dob != 'confirmed'"


@pytest.mark.parametrize("key", ["do_not_call", "voicemail", "busy", "wrong_number", "self_harm", "unsafe_at_home"])
def test_end_interrupts_stop_the_call_with_their_end_status(maya, key):
    flow, plan = maya
    s = _by_id(plan)[key]
    declared = next(i for i in flow["interrupts"] if i["id"] == key)
    assert [g["kind"] for g in s["segments"]] == ["prefix", "interrupt"]
    assert s["segments"][1]["end_status"] == declared["end_status"]
    assert s["path"] == ["greet", "ask_good_time"]


def test_hold_resumes_the_interrupted_step_and_follows_the_base(maya):
    _, plan = maya
    s = _by_id(plan)["hold"]
    assert _steps(s, "resumed") == plan["base"]["path"][1:]


@pytest.mark.parametrize("key,target_steps", [
    ("bereavement", ["say_bereavement_connect", "end_bereavement_transfer"]),
    ("incarceration", ["say_incarceration_connect", "end_incarceration_transfer"]),
])
def test_goto_interrupts_continue_from_their_target(maya, key, target_steps):
    _, plan = maya
    s = _by_id(plan)[key]
    assert s["segments"][1] == {"kind": "interrupt", "interrupt_id": key, "outcome": "goto", "target": target_steps[0]}
    assert _steps(s, "target") == target_steps


def test_human_request_continues_through_the_counsellor_steps(maya):
    _, plan = maya
    s = _by_id(plan)["human_request"]
    target = _steps(s, "target")
    assert target[0] == "route_counsellor_hours"
    assert s["continuation_options"] > 1  # other continuations exist; one is chosen by the same ranking


def test_every_move_inside_a_segment_is_a_real_edge_and_the_jump_is_not(maya):
    flow, plan = maya
    edges = {(e["from"], e["to"]) for e in flow["edges"]}
    for s in plan["interrupt_scenarios"]:
        assert s["all_moves_are_edges"], s["key"]
        assert all(tuple(e) in edges for e in s["covered_edges"])
        prefix_end = _steps(s, "prefix")[-1]
        nxt = next((g["steps"][0] for g in s["segments"] if g["kind"] in ("target", "resumed")), None)
        if nxt is not None and s["interrupt"]["outcome"] == "goto":
            assert (prefix_end, nxt) not in edges  # the interrupt is the jump
            assert [prefix_end, nxt] not in s["covered_edges"]


def test_planning_is_deterministic(maya):
    flow, plan = maya
    assert plan_flow_scenarios(flow) == plan


# ---------------------------------------------------------------------------
# Synthetic
# ---------------------------------------------------------------------------
def test_generic_flow_uses_flow_graphs_primary_path_and_has_no_interrupts():
    flow = parse_flow(yaml.safe_dump({"nodes": [
        {"id": "a", "name": "A"}, {"id": "b", "name": "B"}, {"id": "c", "name": "C"},
    ], "edges": [{"from": "a", "to": "b"}, {"from": "a", "to": "c"}]}), "yaml")
    plan = plan_flow_scenarios(flow)
    assert plan["base"]["path"] == plan["path_scenarios"][0]["path"] == ["a", "b"]
    assert plan["interrupt_scenarios"] == []


def test_declared_default_breaks_a_tie_between_equally_full_paths():
    flow = _modular([
        {"id": "start", "kind": "ask", "next": "route"},
        {"id": "route", "kind": "branch", "cases": [{"when": "x", "goto": "other"}], "default": "normal"},
        {"id": "other", "kind": "ask", "next": "end_a"}, {"id": "normal", "kind": "ask", "next": "end_b"},
        {"id": "end_a", "kind": "end"}, {"id": "end_b", "kind": "end"},
    ])
    plan = plan_flow_scenarios(flow)
    assert plan["path_scenarios"][0]["path"] == ["start", "route", "other", "end_a"]  # first-declared
    assert plan["base"]["path"] == ["start", "route", "normal", "end_b"]              # default wins the tie


def test_default_is_not_assumed_to_be_the_fuller_path():
    # The default exits immediately; the case continues the conversation (like route_dob).
    flow = _modular([
        {"id": "ask1", "kind": "ask", "next": "route"},
        {"id": "route", "kind": "branch", "cases": [{"when": "ok", "goto": "ask2"}], "default": "fail"},
        {"id": "ask2", "kind": "ask", "next": "ask3"}, {"id": "ask3", "kind": "ask", "next": "done"},
        {"id": "fail", "kind": "end"}, {"id": "done", "kind": "end"},
    ])
    assert plan_flow_scenarios(flow)["base"]["path"] == ["ask1", "route", "ask2", "ask3", "done"]


def test_unplaceable_interrupts_are_reported_not_forced():
    flow = _modular(
        [{"id": "a", "kind": "say", "next": "b"}, {"id": "b", "kind": "say", "next": "c"}, {"id": "c", "kind": "end"}],
        [
            {"key": "no_turn", "resume": "end"},                                   # no caller turn exists
            {"key": "odd_when", "resume": "end", "when": "answer.x == 'y'"},       # not a status.<field> condition
            {"key": "no_field", "resume": "end", "when": "status.zzz == 'confirmed'"},  # nobody collects zzz
        ],
    )
    by_id = _by_id(plan_flow_scenarios(flow))
    assert by_id["no_turn"]["plannable"] is False and "no caller turn" in by_id["no_turn"]["reason"]
    assert by_id["odd_when"]["plannable"] is False and "not tied to a collected field" in by_id["odd_when"]["reason"]
    assert by_id["no_field"]["plannable"] is False and "collects field 'zzz'" in by_id["no_field"]["reason"]
    assert all(s["path"] == [] for s in by_id.values())


def test_goto_target_that_no_candidate_reaches_is_just_the_target():
    flow = _modular(
        [{"id": "a", "kind": "say", "next": "q"}, {"id": "q", "kind": "ask", "next": "z"},
         {"id": "z", "kind": "end"}, {"id": "island", "kind": "end"}],
        [{"key": "jump", "resume": "goto:island"}],
    )
    s = plan_flow_scenarios(flow)["interrupt_scenarios"][0]
    assert _steps(s, "target") == ["island"]
    assert s["path"] == ["a", "q", "island"]
    assert s["covered_edges"] == [["a", "q"]]


def test_field_metadata_is_parsed_only_for_modular_steps(maya):
    flow, _ = maya
    nodes = {n["id"]: n for n in flow["nodes"]}
    assert nodes["ask_dob"]["field"] == {"key": "dob", "save_to": "answer.dob"}
    assert "field" not in nodes["greet"]
    generic = parse_flow(yaml.safe_dump({"steps": [{"id": "a", "name": "A", "field": {"key": "k"}}]}), "yaml")
    assert "field" not in generic["nodes"][0]
