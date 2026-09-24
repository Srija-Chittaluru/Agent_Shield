"""Interrupt scenario scripts: generate -> validate -> edit -> save (running: test_flow_interrupt_run.py).

Real parser / planner / validator and the real Maya fixture as stored flow 9; only the
LLM call (node_script.chat) and the flow_scenarios table are faked. `_script_for` builds
a planner-consistent script for any interrupt scenario; each negative test breaks
exactly one protected rule of it.
"""
import copy
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core import node_script
from app.core.node_script import NodeScriptError, _triggers, validate_interrupt_script
from app.routers import flows as R
from tests.routers.test_flow_interrupt_scenarios import FLOWS, MAYA_AGENT, PARSED, _Table

KINDS = {n["id"]: n["type"] for n in PARSED["nodes"]}
READBACK = {"ask_dob", "ask_new_address"}
INTERRUPTS = PARSED["interrupts"]
RECORD = {"user.first_name": "Daniel", "user.last_name": "Alvarez"}


def _safe_keyword(interrupt):
    for spec in interrupt["detection"].values():
        for kw in spec.get("keywords", []):
            line = f"Actually, {kw}."
            if _triggers(line, interrupt) and not any(
                _triggers(line, o) for o in INTERRUPTS
                if o["id"] != interrupt["id"] and not o.get("when") and not interrupt.get("when")
            ):
                return kw
    raise AssertionError(interrupt["id"])


def _script_for(item):
    """A valid script: a caller turn at every ask step except the injection step, a
    listen at the greeting, a readback + confirmation at the DOB step, the interrupt."""
    interrupt = item["interrupt"]
    injection = len(item["segments"][0]["steps"])
    turns, position = [], 0
    for seg in item["segments"]:
        if seg["kind"] == "interrupt":
            turns.append({"type": "interrupt", "interrupt_key": interrupt["id"],
                          "caller_line": f"Actually, {_safe_keyword(interrupt)}.",
                          "expected_agent_behavior": "The agent handles the interrupt as configured."})
            continue
        for node in seg["steps"]:
            position += 1
            if node == "greet":
                turns.append({"type": "listen", "step": position, "node_id": node,
                              "expected_agent_behavior": "The agent introduces itself and the call."})
            if KINDS[node] != "ask" or position == injection:
                continue
            if node == "ask_dob":
                turns += [
                    {"type": "caller", "step": position, "node_id": node,
                     "expected_agent_behavior": "The agent asks for the date of birth.",
                     "caller_line": "March 14th, 1985."},
                    {"type": "readback", "step": position, "node_id": node,
                     "expected_agent_behavior": "The agent reads back March 14 1985 and asks if that is right."},
                    {"type": "caller", "step": position, "node_id": node,
                     "expected_agent_behavior": "The agent accepts the confirmation.",
                     "caller_line": "Yes, that's right."},
                ]
                continue
            turns.append({"type": "caller", "step": position, "node_id": node,
                          "expected_agent_behavior": f"The agent handles {node}.",
                          "caller_line": f"My reply for step {position}."})
    expectations = {"outcome": "The agent follows the planned handling.", "bug_guards": ["No unexpected close."]}
    if interrupt.get("end_status"):
        expectations["end_status"] = interrupt["end_status"]
    return {"test_goal": f"Verify the {interrupt['id']} interrupt.", "turns": turns, "assumed_values": {},
            "expectations": expectations}


@pytest.fixture
def table(monkeypatch):
    t = _Table()
    for name, fn in [("insert_flow_scenarios", t.insert), ("upsert_flow_scenario_script", t.upsert),
                     ("get_flow_scenario", t.get), ("list_flow_scenarios", t.list), ("delete_flow_scenario", t.delete)]:
        monkeypatch.setattr(R, name, fn)
    return t


@pytest.fixture
def llm(monkeypatch):
    class _LLM:
        reply = None
        calls: list = []

    fake = _LLM()
    fake.calls = []

    async def chat(system, messages, json_mode=False, **kw):
        fake.calls.append({"system": system, "prompt": messages[0]["content"]})
        return fake.reply(messages[0]["content"]) if callable(fake.reply) else copy.deepcopy(fake.reply)

    monkeypatch.setattr(node_script, "chat", chat)
    return fake


@pytest.fixture
def client(monkeypatch, table, llm):
    def boom(*a, **kw):
        raise AssertionError("no test cases, runs or calls may be created")

    for name in ("insert_test_case", "update_test_case", "insert_run", "insert_run_group", "update_run", "get_client"):
        monkeypatch.setattr(R, name, boom)
    monkeypatch.setattr(R, "get_agent_flow", lambda fid: FLOWS.get(fid))
    monkeypatch.setattr(R, "get_agent", lambda aid: dict(MAYA_AGENT, id=aid))
    app = FastAPI()
    app.include_router(R.router)
    return TestClient(app)


@pytest.fixture
def plan(client):
    body = client.post("/flows/9/scenarios/preview").json()
    return {s["interrupt"]["id"]: s for s in body["interrupt_scenarios"]}, body


def _generate(client, sid, test_data=RECORD):
    return client.post(f"/flows/9/scenarios/{sid}/script", json={"test_data": test_data} if test_data is not None else None)


def _validate(item, script, record=RECORD):
    return validate_interrupt_script(script, item, node_kinds=KINDS, readback_steps=READBACK,
                                     interrupts=INTERRUPTS, record=record)


def _errors(item, script, record=RECORD):
    with pytest.raises(NodeScriptError) as exc:
        _validate(item, script, record)
    return " | ".join(exc.value.errors)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def test_every_maya_interrupt_generates_a_valid_draft(client, llm, plan, table):
    by_id, _ = plan
    for key, item in by_id.items():
        llm.reply = _script_for(item)
        res = _generate(client, item["id"])
        assert res.status_code == 200, (key, res.text)
        body = res.json()
        assert body["category"] == "interrupt" and body["path"] == item["path"]
        assert [t for t in body["turns"] if t["type"] == "interrupt"][0]["interrupt_key"] == key
        assert body["setup"] == {"record": RECORD}
        assert body["expectations"]["outcome"]
        assert body["expectations"].get("end_status") == item["interrupt"].get("end_status")
    assert table.rows == []  # drafts are never saved


def test_prompt_carries_the_planned_scenario(client, llm, plan):
    item = plan[0]["prove_identity"]
    llm.reply = _script_for(item)
    _generate(client, item["id"])
    call = llm.calls[-1]
    assert call["system"] is node_script.INTERRUPT_SYSTEM_PROMPT
    prompt = call["prompt"]
    assert "STEP 6 [prefix] ask_dob (ask)" in prompt and "collects field 'dob' (date)" in prompt
    assert "caller answer -> readback -> caller confirmation" in prompt  # ask_dob allows it
    assert "turns allowed here: NONE — the interrupt below replaces" in prompt
    assert "turns allowed here: none — silent routing" in prompt
    assert "turns allowed here: an optional listen turn only" in prompt
    assert ">>> INTERRUPT 'prove_identity' here" in prompt and "step 9" in prompt
    assert "applies when: status.dob == 'confirmed'" in prompt
    assert "the agent then goes to: ask_agent_verified" in prompt
    assert "how do i know you're real" in prompt and "must NOT contain" in prompt
    assert "OTHER INTERRUPTS" in prompt and "user.first_name: Daniel" in prompt
    assert "May I please speak with {user.first_name}?" in prompt  # the step's real ask text


def test_normal_scenario_generation_is_unchanged(client, llm, plan):
    normal = plan[1]["scenarios"][0]
    llm.reply = {"test_goal": "g", "turns": [{"step": 2, "node_id": normal["path"][1],
                                             "expected_agent_behavior": "x", "caller_line": "Not right now."}]}
    res = client.post(f"/flows/9/scenarios/{normal['id']}/script")
    assert res.status_code == 200
    assert llm.calls[-1]["system"] is node_script.PATH_SYSTEM_PROMPT
    assert "setup" not in res.json()


def test_unknown_record_field_is_rejected_before_any_llm_call(client, llm, plan):
    res = _generate(client, plan[0]["hold"]["id"], {"user.ssn": "123"})
    assert res.status_code == 422 and "unknown record field" in res.json()["detail"]["errors"][0]
    assert llm.calls == []


def test_generation_alone_never_touches_a_saved_script(client, llm, plan, table):
    item = plan[0]["hold"]
    script = _script_for(item)
    assert client.put(f"/flows/9/scenarios/{item['id']}", json={
        "test_goal": script["test_goal"], "turns": script["turns"],
        "setup": {"record": RECORD}, "expectations": script["expectations"]}).status_code == 200
    saved = table.rows[0]["script_json"]
    llm.reply = {**script, "test_goal": "A different draft."}
    assert _generate(client, item["id"]).status_code == 200
    llm.reply = {"turns": "garbage"}
    assert _generate(client, item["id"]).status_code == 422  # failed regenerate
    assert table.rows[0]["script_json"] == saved


# ---------------------------------------------------------------------------
# Validation — one broken rule each
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("key", ["human_request", "wrong_number", "hold", "prove_identity",
                                 "prove_identity_preverify", "bereavement"])
def test_planner_consistent_scripts_validate(plan, key):
    item = plan[0][key]
    out = _validate(item, _script_for(item))
    assert sum(t["type"] == "interrupt" for t in out["turns"]) == 1


def _mutate(item, fn):
    script = _script_for(item)
    fn(script)
    return script


def _turn(script, predicate):
    return next(t for t in script["turns"] if predicate(t))


def test_wrong_interrupt_is_rejected(plan):
    item = plan[0]["human_request"]
    s = _mutate(item, lambda s: _turn(s, lambda t: t["type"] == "interrupt").update(interrupt_key="busy"))
    assert "interrupt_key must be 'human_request'" in _errors(item, s)


def test_missing_and_extra_interrupts_are_rejected(plan):
    item = plan[0]["human_request"]
    missing = _mutate(item, lambda s: s["turns"].remove(_turn(s, lambda t: t["type"] == "interrupt")))
    assert "must contain the 'human_request' interrupt" in _errors(item, missing)
    extra = _mutate(item, lambda s: s["turns"].append(dict(_turn(s, lambda t: t["type"] == "interrupt"))))
    assert "only one interrupt is allowed" in _errors(item, extra)


def test_interrupt_at_the_wrong_point_is_rejected(plan):
    item = plan[0]["prove_identity"]  # injected at step 9
    def move_later(s):
        i = next(i for i, t in enumerate(s["turns"]) if t["type"] == "interrupt")
        s["turns"].insert(i + 2, s["turns"].pop(i))
    assert "must come at step 9" in _errors(item, _mutate(item, move_later))
    def answer_injection_step(s):
        i = next(i for i, t in enumerate(s["turns"]) if t["type"] == "interrupt")
        s["turns"].insert(i, {"type": "caller", "step": 9, "node_id": "ask_packet_received",
                              "expected_agent_behavior": "x", "caller_line": "Yes, it arrived."})
    assert "raises the interrupt instead of answering" in _errors(item, _mutate(item, answer_injection_step))


def test_wrong_target_or_outcome_on_the_interrupt_is_rejected(plan):
    item = plan[0]["human_request"]
    s = _mutate(item, lambda s: _turn(s, lambda t: t["type"] == "interrupt").update(target="closing"))
    assert "target must be 'route_counsellor_hours'" in _errors(item, s)
    s = _mutate(item, lambda s: _turn(s, lambda t: t["type"] == "interrupt").update(outcome="end"))
    assert "outcome must be 'goto'" in _errors(item, s)
    s = _mutate(item, lambda s: _turn(s, lambda t: t["type"] == "interrupt").update(step=2))
    assert "an interrupt is not a step" in _errors(item, s)


def test_invented_node_or_wrong_order_is_rejected(plan):
    item = plan[0]["hold"]
    identity_step = item["path"].index("ask_identity") + 1
    s = _mutate(item, lambda s: _turn(s, lambda t: t.get("node_id") == "ask_identity").update(node_id="closing"))
    assert f"node_id must be 'ask_identity' (the node at step {identity_step})" in _errors(item, s)
    s = _mutate(item, lambda s: _turn(s, lambda t: t.get("node_id") == "ask_identity").update(step=99))
    assert "step must be an integer from 1 to" in _errors(item, s)
    def swap(s):
        a = next(i for i, t in enumerate(s["turns"]) if t.get("node_id") == "ask_identity")
        b = next(i for i, t in enumerate(s["turns"]) if t.get("node_id") == "ask_packet_received")
        s["turns"][a], s["turns"][b] = s["turns"][b], s["turns"][a]
    assert "turns follow the path in order" in _errors(item, _mutate(item, swap))


def test_turns_outside_the_planned_segments_are_rejected(plan):
    item = plan[0]["wrong_number"]  # end: nothing may follow the interrupt
    s = _mutate(item, lambda s: s["turns"].append({"type": "caller", "step": 2, "node_id": "ask_good_time",
                                                   "expected_agent_behavior": "x", "caller_line": "More."}))
    errors = _errors(item, s)
    assert "call ends at the 'wrong_number' interrupt" in errors or "before the interrupt point" in errors


def test_turn_kinds_must_match_the_step(plan):
    item = plan[0]["hold"]
    s = _mutate(item, lambda s: _turn(s, lambda t: t.get("node_id") == "ask_identity").update(type="listen", caller_line=""))
    assert "listen turns belong at say/end steps" in _errors(item, s)
    s = _mutate(item, lambda s: _turn(s, lambda t: t.get("type") == "listen").update(type="caller", caller_line="Hello."))
    assert "caller turns belong at ask steps" in _errors(item, s)
    s = _mutate(item, lambda s: _turn(s, lambda t: t.get("type") == "listen").update(caller_line="Hi!"))
    assert "agent-only; remove its caller_line" in _errors(item, s)


def _dob_turns(script):
    return [i for i, t in enumerate(script["turns"]) if t.get("node_id") == "ask_dob"]


def test_readback_follows_the_answer_and_is_confirmed(plan):
    item = plan[0]["prove_identity"]
    out = _validate(item, _script_for(item))
    assert [t["type"] for t in out["turns"] if t.get("node_id") == "ask_dob"] == ["caller", "readback", "caller"]


def test_readback_rules(plan):
    item = plan[0]["prove_identity"]
    def drop_confirmation(s):
        s["turns"].pop(_dob_turns(s)[2])
    assert "must be followed by the caller confirming" in _errors(item, _mutate(item, drop_confirmation))
    def readback_before_answer(s):
        a, rb, _ = _dob_turns(s)
        s["turns"][a], s["turns"][rb] = s["turns"][rb], s["turns"][a]
    assert "must directly follow the caller's answer" in _errors(item, _mutate(item, readback_before_answer))
    def second_answer_without_readback(s):
        s["turns"].pop(_dob_turns(s)[1])
    assert "turns follow the path in order" in _errors(item, _mutate(item, second_answer_without_readback))
    def readback_where_not_allowed(s):
        i = next(i for i, t in enumerate(s["turns"]) if t.get("node_id") == "ask_identity")
        step = s["turns"][i]["step"]
        s["turns"][i + 1:i + 1] = [
            {"type": "readback", "step": step, "node_id": "ask_identity", "expected_agent_behavior": "x"},
            {"type": "caller", "step": step, "node_id": "ask_identity", "expected_agent_behavior": "x", "caller_line": "Yes."},
        ]
    assert "does not allow a readback" in _errors(item, _mutate(item, readback_where_not_allowed))


def test_interrupt_line_must_trigger_its_interrupt_and_nothing_else(plan):
    item = plan[0]["human_request"]
    no_trigger = _mutate(item, lambda s: _turn(s, lambda t: t["type"] == "interrupt").update(caller_line="I have a question."))
    assert "must contain one of the 'human_request' trigger phrases" in _errors(item, no_trigger)
    negative = next(n for spec in item["interrupt"]["detection"].values() for n in spec.get("negative", []))
    with_negative = _mutate(item, lambda s: _turn(s, lambda t: t["type"] == "interrupt").update(
        caller_line=f"I'd like to talk to a person, {negative}."))
    assert "none of its excluded phrases" in _errors(item, with_negative)
    both = _mutate(item, lambda s: _turn(s, lambda t: t["type"] == "interrupt").update(
        caller_line="I want to talk to a person, wrong number anyway."))
    assert "would also trigger interrupt 'wrong_number'" in _errors(item, both)


def test_ordinary_caller_lines_may_not_trigger_an_interrupt(plan):
    item = plan[0]["hold"]
    s = _mutate(item, lambda s: _turn(s, lambda t: t.get("node_id") == "ask_identity").update(
        caller_line="Sorry, wrong number."))
    assert "would trigger interrupt 'wrong_number'" in _errors(item, s)


def test_placeholders_and_bad_shapes_are_rejected(plan):
    item = plan[0]["hold"]
    s = _mutate(item, lambda s: _turn(s, lambda t: t.get("node_id") == "ask_identity").update(caller_line="I'm [name]."))
    assert "placeholder" in _errors(item, s)
    assert "Script must be a JSON object" in _errors(item, "not json")
    assert "expectations must be an object" in _errors(item, _mutate(item, lambda s: s.pop("expectations")))
    assert "test_goal must be" in _errors(item, _mutate(item, lambda s: s.update(test_goal="")))


def test_end_status_is_planner_controlled(plan):
    item = plan[0]["wrong_number"]
    s = _mutate(item, lambda s: s["expectations"].update(end_status="Contacted | Completed"))
    assert "end_status must be 'No Answer | Retry Scheduled'" in _errors(item, s)
    out = _validate(item, _mutate(item, lambda s: s["expectations"].pop("end_status")))
    assert out["expectations"]["end_status"] == "No Answer | Retry Scheduled"


def test_assumed_values_only_without_supplied_record(plan):
    item = plan[0]["hold"]
    s = _mutate(item, lambda s: s.update(assumed_values={"user.dob": "March 14th, 1985"}))
    assert "record values were supplied" in _errors(item, s)
    out = _validate(item, s, record={})
    assert out["setup"] == {"record": {}, "assumed_values": {"user.dob": "March 14th, 1985"}}


# ---------------------------------------------------------------------------
# Outcome shapes
# ---------------------------------------------------------------------------
def test_goto_end_and_resume_scripts(plan):
    goto = _validate(plan[0]["human_request"], _script_for(plan[0]["human_request"]))
    after = goto["turns"][[t["type"] for t in goto["turns"]].index("interrupt") + 1:]
    assert after and after[0]["node_id"] in plan[0]["human_request"]["segments"][2]["steps"]
    end = _validate(plan[0]["wrong_number"], _script_for(plan[0]["wrong_number"]))
    assert end["turns"][-1]["type"] == "interrupt"
    resume = _validate(plan[0]["hold"], _script_for(plan[0]["hold"]))
    i = [t["type"] for t in resume["turns"]].index("interrupt")
    assert resume["turns"][i + 1]["node_id"] == "ask_good_time"  # the interrupted step again


# ---------------------------------------------------------------------------
# Save / reload
# ---------------------------------------------------------------------------
def _put(client, item, script, record=RECORD):
    return client.put(f"/flows/9/scenarios/{item['id']}", json={
        "test_goal": script["test_goal"], "turns": script["turns"],
        "setup": {"record": record, "assumed_values": script.get("assumed_values") or {}},
        "expectations": script["expectations"]})


def test_valid_interrupt_script_saves_and_reloads(client, plan, table):
    item = plan[0]["bereavement"]
    res = _put(client, item, _script_for(item))
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["category"] == "interrupt" and body["setup"]["record"] == RECORD
    assert body["expectations"]["outcome"] and any(t["type"] == "interrupt" for t in body["turns"])
    [row] = table.rows
    stored = json.loads(row["script_json"])
    assert set(stored) == {"turns", "setup", "expectations"}
    listed = client.get("/flows/9/scenarios").json()["scenarios"][0]
    assert listed["turns"] == body["turns"] and listed["expectations"] == body["expectations"]


def test_invalid_or_tampered_script_is_not_saved(client, plan, table):
    item = plan[0]["human_request"]
    for mutate in (lambda s: _turn(s, lambda t: t["type"] == "interrupt").update(interrupt_key="busy"),
                   lambda s: _turn(s, lambda t: t["type"] == "interrupt").update(target="closing"),
                   lambda s: s["turns"].remove(_turn(s, lambda t: t["type"] == "interrupt"))):
        res = _put(client, item, _mutate(item, mutate))
        assert res.status_code == 422
    assert table.rows == []


def test_repeated_save_updates_one_row(client, plan, table):
    item = plan[0]["hold"]
    _put(client, item, _script_for(item))
    _put(client, item, _mutate(item, lambda s: s.update(test_goal="Edited goal.")))
    assert len(table.rows) == 1 and table.rows[0]["test_goal"] == "Edited goal."


def test_normal_script_save_keeps_the_list_format(client, plan, table):
    normal = plan[1]["scenarios"][0]
    turns = [{"step": 2, "node_id": normal["path"][1], "expected_agent_behavior": "x", "caller_line": "Not now."}]
    assert client.put(f"/flows/9/scenarios/{normal['id']}", json={"test_goal": "g", "turns": turns}).status_code == 200
    assert isinstance(json.loads(table.rows[0]["script_json"]), list)


def test_generate_splits_call_data_from_the_record(client, plan, llm):
    item = plan[0]["hold"]
    llm.reply = _script_for(item)
    res = _generate(client, item["id"], test_data={**RECORD, "call.patient_ref": "adult-en"})
    assert res.status_code == 200, res.json()
    setup = res.json()["setup"]
    assert setup["record"] == RECORD and setup["call"] == {"patient_ref": "adult-en"}
    assert "adult-en" not in llm.calls[0]["prompt"]  # call data is not dialogue material


def test_generate_rejects_unmapped_call_data(client, plan, llm):
    res = _generate(client, plan[0]["hold"]["id"], test_data={**RECORD, "call.patient_id": "x"})
    assert res.status_code == 422 and llm.calls == []
