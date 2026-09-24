"""Tests for interrupt normalization in the modular step flow adapter
(app.core.flow_parser.modular_interrupts): the top-level `interrupts:` list becomes a
separate `interrupts` list in the parse result — never edges.
"""
import json
from pathlib import Path

import pytest
import yaml

from app.core.flow_parser import FlowParseError, parse_flow

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "flows" / "maya_renewal_modular.yaml"


def _parse(doc: dict) -> dict:
    return parse_flow(yaml.safe_dump(doc, sort_keys=False), "yaml")


def _flow(interrupts=None, **extra) -> dict:
    doc = {"entry": "a", "steps": [
        {"id": "a", "kind": "ask", "next": "b"},
        {"id": "b", "kind": "branch", "cases": [{"when": "x", "label": "l", "goto": "c"}], "default": "d"},
        {"id": "c", "kind": "end"}, {"id": "d", "kind": "end"}, {"id": "desk", "kind": "end"},
    ], **extra}
    if interrupts is not None:
        doc["interrupts"] = interrupts
    return doc


def _errors(doc) -> str:
    with pytest.raises(FlowParseError) as exc:
        _parse(doc)
    return " | ".join(exc.value.errors)


# ---------------------------------------------------------------------------
# The real Maya fixture
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def maya():
    source = FIXTURE.read_text()
    return yaml.safe_load(source), parse_flow(source, "yaml")


def test_real_flow_every_interrupt_is_extracted_in_order(maya):
    doc, flow = maya
    assert [i["id"] for i in flow["interrupts"]] == [e["key"] for e in doc["interrupts"]]
    assert len(flow["interrupts"]) == 12


def test_real_flow_outcomes_and_targets(maya):
    _, flow = maya
    by_id = {i["id"]: i for i in flow["interrupts"]}
    assert {k: (v["outcome"], v.get("target")) for k, v in by_id.items()} == {
        "do_not_call": ("end", None),
        "voicemail": ("end", None),
        "busy": ("end", None),
        "wrong_number": ("end", None),
        "hold": ("resume", None),
        "human_request": ("goto", "route_counsellor_hours"),
        "prove_identity": ("goto", "ask_agent_verified"),
        "prove_identity_preverify": ("resume", None),
        "bereavement": ("goto", "say_bereavement_connect"),
        "incarceration": ("goto", "say_incarceration_connect"),
        "self_harm": ("end", None),
        "unsafe_at_home": ("end", None),
    }
    step_ids = {n["id"] for n in flow["nodes"]}
    assert all(i["target"] in step_ids for i in flow["interrupts"] if i["outcome"] == "goto")


def test_real_flow_trigger_phrases_are_preserved_verbatim(maya):
    doc, flow = maya
    for raw, item in zip(doc["interrupts"], flow["interrupts"], strict=True):
        assert item["detection"] == {
            lang: {f: spec[f] for f in ("keywords", "negative") if f in spec}
            for lang, spec in raw["detection"].items()
        }
    human = next(i for i in flow["interrupts"] if i["id"] == "human_request")
    assert human["detection"]["en-US"]["keywords"]  # a real, non-empty phrase list


def test_real_flow_metadata_is_preserved(maya):
    doc, flow = maya
    for raw, item in zip(doc["interrupts"], flow["interrupts"], strict=True):
        assert item["priority"] == raw["priority"]
        assert item.get("when") == raw.get("when")
        assert item.get("end_status") == raw.get("end_status")
        assert item.get("critical") == raw.get("critical")
        assert item.get("max_turns") == raw.get("max_turns")
        assert "prompt" not in item  # agent wording stays in the raw source
    by_id = {i["id"]: i for i in flow["interrupts"]}
    assert by_id["prove_identity"]["when"] == "status.dob == 'confirmed'"
    assert by_id["prove_identity_preverify"]["when"] == "status.dob != 'confirmed'"
    assert by_id["bereavement"]["critical"] is True


def test_real_flow_interrupts_add_no_edges_and_change_no_nodes(maya):
    doc, flow = maya
    without = dict(doc)
    del without["interrupts"]
    plain = _parse(without)
    assert flow["edges"] == plain["edges"]
    assert flow["nodes"] == plain["nodes"]
    assert "interrupts" not in plain
    targets = {i["target"] for i in flow["interrupts"] if "target" in i}
    assert not any(e["to"] in targets for e in flow["edges"])


# ---------------------------------------------------------------------------
# Synthetic
# ---------------------------------------------------------------------------
def test_multiple_interrupts_of_each_outcome():
    r = _parse(_flow([
        {"key": "human", "priority": 8, "detection": {"en-US": {"keywords": ["a person"], "negative": ["no person"]}},
         "resume": "goto:desk", "max_turns": 1},
        {"key": "hang_up", "priority": 1, "detection": {"en-US": {"keywords": ["stop"]}},
         "resume": "end", "end_status": "Declined", "critical": True},
        {"key": "hold", "priority": 20, "detection": {"en-US": {"keywords": ["one sec"]}},
         "resume": "resume", "when": "status.x == 'y'"},
    ]))
    assert r["interrupts"] == [
        {"id": "human", "priority": 8, "detection": {"en-US": {"keywords": ["a person"], "negative": ["no person"]}},
         "outcome": "goto", "target": "desk", "max_turns": 1},
        {"id": "hang_up", "priority": 1, "detection": {"en-US": {"keywords": ["stop"]}},
         "outcome": "end", "end_status": "Declined", "critical": True},
        {"id": "hold", "priority": 20, "detection": {"en-US": {"keywords": ["one sec"]}},
         "outcome": "resume", "when": "status.x == 'y'"},
    ]


def test_ordinary_transitions_are_unchanged_by_interrupts():
    base = _parse(_flow())
    with_interrupts = _parse(_flow([{"key": "h", "resume": "goto:desk"}]))
    assert with_interrupts["edges"] == base["edges"]
    assert with_interrupts["nodes"] == base["nodes"]
    assert base["edges"] == [
        {"from": "a", "to": "b", "type": "next"},
        {"from": "b", "to": "c", "type": "case", "label": "l", "when": "x"},
        {"from": "b", "to": "d", "type": "default", "label": "default"},
    ]


def test_modular_flow_without_interrupts_has_no_interrupts_key():
    assert "interrupts" not in _parse(_flow())


def test_empty_interrupts_list():
    assert _parse(_flow([]))["interrupts"] == []


def test_existing_resume_control_transitions_are_unaffected():
    doc = _flow([{"key": "h", "resume": "resume"}])
    doc["steps"][0]["answer"] = {"on_decline": "goto:resume"}
    r = _parse(doc)
    assert r["nodes"][0]["control_transitions"] == [{"type": "answer", "label": "on_decline", "target": "resume"}]
    assert r["interrupts"] == [{"id": "h", "outcome": "resume"}]


@pytest.mark.parametrize("interrupts,needle", [
    ([{"key": "h", "resume": "goto:nowhere"}], "interrupt 'h': resume points to unknown step 'nowhere'"),
    ([{"key": "h"}], "interrupt 'h': resume must be 'end', 'resume' or 'goto:<step>'"),
    ([{"key": "h", "resume": "desk"}], "resume must be 'end', 'resume' or 'goto:<step>', got 'desk'"),
    ([{"key": "h", "resume": "goto:"}], "resume must be 'end', 'resume' or 'goto:<step>'"),
    ([{"resume": "end"}], "interrupts[0] must have a 'key'"),
    ([{"key": "h", "resume": "end"}, {"key": "h", "resume": "end"}], "Duplicate interrupt key: 'h'"),
    (["not an object"], "interrupts[0] must be an object"),
    ({"key": "h"}, "interrupts must be a list"),
    ([{"key": "h", "resume": "end", "priority": "high"}], "priority must be an integer"),
    ([{"key": "h", "resume": "end", "detection": ["stop"]}], "detection must be an object keyed by language"),
    ([{"key": "h", "resume": "end", "detection": {"en-US": {"keywords": "stop"}}}], "detection.en-US.keywords must be a list of strings"),
])
def test_malformed_interrupts_are_rejected(interrupts, needle):
    assert needle in _errors(_flow(interrupts))


# ---------------------------------------------------------------------------
# Other formats are untouched
# ---------------------------------------------------------------------------
def test_generic_flow_with_an_interrupts_key_is_unchanged():
    # Not the modular shape (no entry/kind): `interrupts` is ignored, as before.
    r = parse_flow(json.dumps({
        "nodes": [{"id": "a", "name": "A"}, {"id": "b", "name": "B"}],
        "edges": [{"from": "a", "to": "b"}],
        "interrupts": [{"key": "h", "resume": "goto:b"}],
    }), "json")
    assert set(r) == {"agent_name", "nodes", "edges", "source_format", "extraction_method"}
    assert r["edges"] == [{"from": "a", "to": "b"}]


def test_generic_steps_flow_is_unchanged():
    r = parse_flow(yaml.safe_dump({"steps": [{"id": "a", "name": "A", "next": "b"}, {"id": "b", "name": "B"}],
                                   "interrupts": [{"key": "h", "resume": "goto:b"}]}), "yaml")
    assert "interrupts" not in r
    assert r["edges"] == [{"from": "a", "to": "b"}]
