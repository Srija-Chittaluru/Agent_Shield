"""Tests for the modular step flow adapter in app.core.flow_parser (the
maya_renewal_modular.yaml format): every declared transition form becomes an edge,
nothing undeclared does.

Pure unit tests, plus a regression test against the real flow file
(tests/fixtures/flows/maya_renewal_modular.yaml — identical to the stored flows 3/4/6/9)
that checks structural facts rather than edge counts.
"""
from pathlib import Path

import pytest
import yaml

from app.core.flow_graph import _reachable_from, analyze_graph
from app.core.flow_parser import FlowParseError, parse_flow

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "flows" / "maya_renewal_modular.yaml"


def _parse(doc: dict) -> dict:
    return parse_flow(yaml.safe_dump(doc, sort_keys=False), "yaml")


def _flow(steps, **top) -> dict:
    return {"flow_key": "t", "entry": steps[0]["id"], **top, "steps": steps}


def _edges(result, source=None):
    return [e for e in result["edges"] if source is None or e["from"] == source]


def _pairs(result):
    return {(e["from"], e["to"]) for e in result["edges"]}


def _errors(doc) -> str:
    with pytest.raises(FlowParseError) as exc:
        _parse(doc)
    return " | ".join(exc.value.errors)


# ---------------------------------------------------------------------------
# 1–6. Each declared form
# ---------------------------------------------------------------------------
def test_next():
    r = _parse(_flow([{"id": "a", "kind": "say", "next": "b"}, {"id": "b", "kind": "end"}]))
    assert r["edges"] == [{"from": "a", "to": "b", "type": "next"}]


def test_cases_goto_with_label_and_when_preserved():
    r = _parse(_flow([
        {"id": "route", "kind": "branch", "cases": [
            {"when": 'answer.x == "No"', "label": "said no", "goto": "no_path"},
            {"when": 'answer.x == "Maybe"', "label": "unsure", "goto": "maybe_path"},
        ], "default": "yes_path"},
        {"id": "no_path", "kind": "end"}, {"id": "maybe_path", "kind": "end"}, {"id": "yes_path", "kind": "end"},
    ]))
    assert _edges(r, "route") == [
        {"from": "route", "to": "no_path", "type": "case", "label": "said no", "when": 'answer.x == "No"'},
        {"from": "route", "to": "maybe_path", "type": "case", "label": "unsure", "when": 'answer.x == "Maybe"'},
        {"from": "route", "to": "yes_path", "type": "default", "label": "default"},
    ]


def test_default():
    r = _parse(_flow([{"id": "route", "kind": "branch", "default": "b"}, {"id": "b", "kind": "end"}]))
    assert r["edges"] == [{"from": "route", "to": "b", "type": "default", "label": "default"}]


@pytest.mark.parametrize("event", ["on_decline", "on_exhaust"])
def test_answer_goto(event):
    r = _parse(_flow([
        {"id": "ask", "kind": "ask", "next": "ok", "answer": {event: "goto:fallback"}},
        {"id": "ok", "kind": "end"}, {"id": "fallback", "kind": "end"},
    ]))
    assert {"from": "ask", "to": "fallback", "type": "answer", "label": event} in r["edges"]


def test_answer_goto_as_mapping():
    r = _parse(_flow([
        {"id": "ask", "kind": "ask", "answer": {"on_decline": {"goto": "fallback"}}},
        {"id": "fallback", "kind": "end"},
    ]))
    assert r["edges"] == [{"from": "ask", "to": "fallback", "type": "answer", "label": "on_decline"}]


def test_non_transition_answer_settings_are_not_edges():
    r = _parse(_flow([
        {"id": "ask", "kind": "ask", "answer": {"on_decline": "record", "requirement": "required"}},
        {"id": "b", "kind": "end"},
    ]))
    assert _edges(r, "ask") == []


# ---------------------------------------------------------------------------
# 7–8. advance
# ---------------------------------------------------------------------------
def test_advance_uses_declared_list_order():
    # Ids deliberately NOT in alphabetical order: "zeta" is declared right after "ask".
    r = _parse(_flow([
        {"id": "ask", "kind": "ask", "answer": {"on_exhaust": "advance"}},
        {"id": "zeta", "kind": "end"},
        {"id": "alpha", "kind": "end"},
    ]))
    assert r["edges"] == [
        {"from": "ask", "to": "zeta", "type": "answer", "label": "on_exhaust", "advance": True},
    ]


def test_flow_defaults_apply_to_ask_steps_that_do_not_override_them():
    r = _parse(_flow([
        {"id": "say", "kind": "say", "next": "ask"},
        {"id": "ask", "kind": "ask"},                                    # inherits both
        {"id": "route", "kind": "branch", "default": "ask2"},
        {"id": "ask2", "kind": "ask", "answer": {"on_decline": "goto:bye"}},  # overrides one
        {"id": "after", "kind": "say", "next": "bye"},
        {"id": "bye", "kind": "end"},
    ], defaults={"answer": {"on_decline": "advance", "on_exhaust": "advance", "requirement": "required"}}))
    assert _edges(r, "ask") == [
        {"from": "ask", "to": "route", "type": "answer", "label": "on_decline", "advance": True, "from_defaults": True},
        {"from": "ask", "to": "route", "type": "answer", "label": "on_exhaust", "advance": True, "from_defaults": True},
    ]
    assert _edges(r, "ask2") == [
        {"from": "ask2", "to": "bye", "type": "answer", "label": "on_decline"},
        {"from": "ask2", "to": "after", "type": "answer", "label": "on_exhaust", "advance": True, "from_defaults": True},
    ]
    # Defaults describe answers, so only `ask` steps get them.
    assert _edges(r, "say") == [{"from": "say", "to": "ask", "type": "next"}]
    assert _edges(r, "route") == [{"from": "route", "to": "ask2", "type": "default", "label": "default"}]


def test_advance_declared_on_the_last_step_is_an_error():
    assert "no next step to advance to" in _errors(_flow([
        {"id": "a", "kind": "say", "next": "b"},
        {"id": "b", "kind": "ask", "answer": {"on_exhaust": "advance"}},
    ]))


def test_inherited_advance_on_the_last_step_adds_nothing_and_is_not_an_error():
    r = _parse(_flow([
        {"id": "a", "kind": "say", "next": "b"}, {"id": "b", "kind": "ask"},
    ], defaults={"answer": {"on_decline": "advance"}}))
    assert _edges(r, "b") == []


# ---------------------------------------------------------------------------
# 9. resume
# ---------------------------------------------------------------------------
def test_resume_is_kept_as_control_metadata_never_as_an_edge():
    r = _parse(_flow([
        {"id": "ask", "kind": "ask", "next": "route", "answer": {"on_decline": "goto:resume"}},
        {"id": "route", "kind": "branch",
         "cases": [{"when": "answer.ok == 'Yes'", "label": "satisfied", "goto": "resume"}],
         "default": "again"},
        {"id": "again", "kind": "end"},
    ]))
    assert all(e["to"] != "resume" for e in r["edges"])
    nodes = {n["id"]: n for n in r["nodes"]}
    assert nodes["ask"]["control_transitions"] == [{"type": "answer", "label": "on_decline", "target": "resume"}]
    assert nodes["route"]["control_transitions"] == [
        {"type": "case", "label": "satisfied", "when": "answer.ok == 'Yes'", "target": "resume"},
    ]
    assert "control_transitions" not in nodes["again"]
    # route's only real exit is its default.
    assert _edges(r, "route") == [{"from": "route", "to": "again", "type": "default", "label": "default"}]


# ---------------------------------------------------------------------------
# 10. multiple outgoing branches
# ---------------------------------------------------------------------------
def test_cases_plus_default_make_a_branch_node():
    r = _parse(_flow([
        {"id": "route", "kind": "branch", "cases": [{"when": "x", "goto": "b"}], "default": "c"},
        {"id": "b", "kind": "end"}, {"id": "c", "kind": "end"},
    ]))
    assert analyze_graph(r).branch_nodes == ["route"]


# ---------------------------------------------------------------------------
# 11–12. backward compatibility, no inference
# ---------------------------------------------------------------------------
def test_generic_steps_without_the_modular_shape_are_unchanged():
    # No top-level `entry` / no `kind`: the existing generic path (next-only).
    r = parse_flow(yaml.safe_dump({"steps": [
        {"id": "a", "name": "A", "next": "b", "default": "c"},
        {"id": "b", "name": "B"}, {"id": "c", "name": "C"},
    ]}), "yaml")
    assert r["edges"] == [{"from": "a", "to": "b"}]
    assert "control_transitions" not in r["nodes"][0]


def test_entry_but_steps_without_kind_stay_generic():
    r = parse_flow(yaml.safe_dump({"entry": "a", "steps": [
        {"id": "a", "next": "b", "default": "c"}, {"id": "b"}, {"id": "c"},
    ]}), "yaml")
    assert r["edges"] == [{"from": "a", "to": "b"}]


def test_literal_nodes_schema_is_unchanged():
    r = parse_flow(yaml.safe_dump({
        "entry": "a",
        "nodes": [{"id": "a", "name": "A", "kind": "ask"}, {"id": "b", "name": "B", "kind": "end"}],
        "edges": [{"from": "a", "to": "b"}],
    }), "yaml")
    assert r["edges"] == [{"from": "a", "to": "b"}]


def test_list_order_alone_never_creates_an_edge():
    r = _parse(_flow([
        {"id": "a", "kind": "say", "next": "b"},
        {"id": "b", "kind": "say"},            # declares no exit
        {"id": "c", "kind": "end"},            # comes next in the list — must NOT be linked
    ], defaults={"answer": {"on_decline": "advance"}}))  # a `say` step never inherits
    assert _pairs(r) == {("a", "b")}


def test_interrupts_are_not_edges():
    r = _parse(_flow(
        [{"id": "a", "kind": "ask", "next": "b"}, {"id": "b", "kind": "end"}, {"id": "human", "kind": "end"}],
        interrupts=[{"key": "human_request", "resume": "goto:human", "detection": {"en-US": {"keywords": ["a person"]}}}],
    ))
    assert all(e["to"] != "human" for e in r["edges"])


def test_parsing_is_deterministic():
    doc = _flow([{"id": "r", "kind": "branch", "cases": [{"goto": "b"}], "default": "c"},
                 {"id": "b", "kind": "end"}, {"id": "c", "kind": "end"}])
    assert _parse(doc) == _parse(doc)


# ---------------------------------------------------------------------------
# 13. malformed transitions — same strictness as a declared edges list
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("step,needle", [
    ({"id": "a", "kind": "branch", "cases": [{"goto": "nope"}]}, "cases[0].goto points to unknown step 'nope'"),
    ({"id": "a", "kind": "branch", "default": "nope"}, "default points to unknown step 'nope'"),
    ({"id": "a", "kind": "ask", "answer": {"on_decline": "goto:nope"}}, "answer.on_decline points to unknown step 'nope'"),
    ({"id": "a", "kind": "branch", "cases": {"goto": "b"}}, "cases must be a list"),
    ({"id": "a", "kind": "branch", "cases": [{"label": "no goto"}]}, "cases[0] must be an object with a 'goto'"),
    ({"id": "a", "kind": "branch", "cases": [{"goto": ""}]}, "cases[0].goto must name a step"),
    ({"id": "a", "kind": "ask", "answer": "goto:b"}, "answer must be an object"),
])
def test_malformed_declared_transition_is_rejected(step, needle):
    assert needle in _errors(_flow([step, {"id": "b", "kind": "end"}]))


def test_unresolvable_next_keeps_the_existing_lenient_behaviour():
    r = _parse(_flow([{"id": "a", "kind": "say", "next": "nope"}, {"id": "b", "kind": "end"}]))
    assert r["edges"] == []


# ---------------------------------------------------------------------------
# Regression: the real maya_renewal_modular.yaml
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def maya():
    source = FIXTURE.read_text()
    return yaml.safe_load(source), parse_flow(source, "yaml")


def test_real_flow_keeps_every_step_in_declared_order(maya):
    doc, flow = maya
    assert [n["id"] for n in flow["nodes"]] == [s["id"] for s in doc["steps"]]
    assert len(flow["nodes"]) == len(doc["steps"]) == 58
    assert flow["extraction_method"] == "deterministic"


def test_real_flow_every_edge_is_a_declared_transition(maya):
    """Independent cross-check against the YAML: each edge must be exactly one declared
    transition, and every declared step-to-step transition must be an edge."""
    doc, flow = maya
    steps = doc["steps"]
    order = [s["id"] for s in steps]
    defaults = doc["defaults"]["answer"]

    def resolve(step_id, value):
        target = value[5:] if isinstance(value, str) and value.startswith("goto:") else value
        return order[order.index(step_id) + 1] if target == "advance" else target

    declared = set()
    for s in steps:
        if "next" in s:
            declared.add((s["id"], s["next"], "next", None))
        for case in s.get("cases") or []:
            declared.add((s["id"], case["goto"], "case", case.get("label")))
        if "default" in s:
            declared.add((s["id"], s["default"], "default", "default"))
        answer = dict(s.get("answer") or {})
        if s["kind"] == "ask":
            answer = {**{k: v for k, v in defaults.items() if k.startswith("on_")}, **answer}
        for event, value in answer.items():
            if event.startswith("on_") and (value == "advance" or str(value).startswith("goto:")):
                declared.add((s["id"], resolve(s["id"], value), "answer", event))
    declared = {d for d in declared if d[1] != "resume"}

    got = {(e["from"], e["to"], e["type"], e.get("label")) for e in flow["edges"]}
    assert got == declared


def test_real_flow_renewal_steps_are_connected_from_the_entry(maya):
    _, flow = maya
    facts = analyze_graph(flow)
    reach = _reachable_from(["greet"], facts.out_adj)
    for step in (
        "ask_good_time", "ask_identity", "ask_dob", "route_dob", "thank_for_verifying",
        "ask_packet_received", "route_packet", "ask_address_ok", "route_address",
        "ask_already_sent", "route_sent", "ask_online_account", "route_online_account",
        "ask_has_email", "route_prereqs", "ask_more_questions", "ask_send_date", "closing",
    ):
        assert step in reach, step
    assert {("ask_identity", "route_identity"), ("route_identity", "ask_dob")} <= _pairs(flow)


def test_real_flow_branches_retries_and_answer_paths(maya):
    _, flow = maya
    edges = flow["edges"]

    def has(frm, to, type_, label=None, **extra):
        return any(
            e["from"] == frm and e["to"] == to and e["type"] == type_
            and (label is None or e.get("label") == label)
            and all(e.get(k) == v for k, v in extra.items())
            for e in edges
        )

    # Packet received / not received.
    assert has("route_packet", "ask_address_ok", "case", "packet never arrived")
    assert has("route_packet", "ask_already_sent", "default")
    # DOB: match, retry, fail.
    assert has("route_dob", "thank_for_verifying", "case", "date matches")
    assert has("route_dob", "ask_dob", "case", "ask once more")
    assert has("route_dob", "say_dob_failed", "default")
    assert ("route_dob", "ask_dob") in analyze_graph(flow).back_edges
    # Explicit answer decline / exhaust.
    assert has("ask_good_time", "say_another_time", "answer", "on_decline")
    assert has("ask_identity", "say_dob_declined", "answer", "on_exhaust")
    assert has("ask_dob", "route_dob", "answer", "on_exhaust", advance=True)
    # `advance` inherited from the flow's defaults block.
    assert has("ask_packet_received", "route_packet", "answer", "on_decline", advance=True, from_defaults=True)
    for node in ("route_good_time", "route_identity", "route_dob", "route_packet", "route_sent"):
        assert node in analyze_graph(flow).branch_nodes, node


def test_real_flow_resume_stays_metadata(maya):
    _, flow = maya
    assert all(e["to"] != "resume" for e in flow["edges"])
    with_resume = {n["id"] for n in flow["nodes"] if "control_transitions" in n}
    assert with_resume == {
        "ask_agent_verified", "route_agent_verified", "ask_agent_verified_again", "route_agent_verified_again",
    }


def test_real_flow_has_no_interrupt_or_adjacency_edges(maya):
    doc, flow = maya
    # Steps only an interrupt leads to stay unlinked until interrupts are modelled.
    interrupt_targets = {
        i["resume"][5:] for i in doc["interrupts"] if str(i.get("resume", "")).startswith("goto:")
    }
    assert interrupt_targets == {
        "route_counsellor_hours", "ask_agent_verified", "say_bereavement_connect", "say_incarceration_connect",
    }
    assert not any(e["to"] in interrupt_targets for e in flow["edges"])
    # A closing line that declares no exit is not linked to the step after it.
    for say_step in ("close_no_packet", "close_already_sent", "closing"):
        assert not any(e["from"] == say_step for e in flow["edges"]), say_step
