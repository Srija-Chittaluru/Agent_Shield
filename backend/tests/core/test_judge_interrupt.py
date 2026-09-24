"""Part 8 — the existing Judge evaluating interrupt scenarios.

Scenario rows are built with the REAL planner / validator / interrupt_script_to_scenario
(the same shape a run's scenarios row has); transcripts are fed as message rows; the LLM
is mocked. The model's semantic judgement is the mock's reply — what is tested here is
the context the Judge is given, and how the existing in-code verdict derivation treats
interrupt checks, bug guards, the structured end-status check and the short-circuits.
The "Maya" cases replay the actual Part 7 runs 227-229 (transcripts, traces, saved
expectations).
"""
import json

import pytest

from app.core import judge
from tests.core.test_run_scenario_interrupt import _scenario

MAYA_OPENING = ("Hello, this is Maya calling from Santa Rosa Community Health about your yearly Medi-Cal "
                "renewal. This call is recorded, and it'll only take a couple of minutes. Would you have a "
                "couple of minutes to talk about your Medi-Cal renewal?")


def _row(key, expectations=None, when=None):
    """The run's scenarios row for a saved interrupt script of `key`."""
    s, script = _scenario(key)
    stored = json.loads(s["node_script_json"])
    if expectations is not None:
        stored["expectations"] = expectations
    if when is not None:
        stored["interrupt"]["when"] = when
    return {"id": 1, "title": s["title"], "user_goal": s["user_goal"], "test_type": s["test_type"],
            "assigned_fault": "none", "expected_behavior": s["expected_behavior"],
            "node_script_json": json.dumps(stored)}, stored


def _msgs(*turns):
    """(role, content, trace) triples -> message rows."""
    return [{"turn_index": i, "role": r, "content": c, "trace_json": json.dumps(t) if t else None}
            for i, (r, c, t) in enumerate(turns)]


class _Judge:
    def __init__(self, monkeypatch, row, msgs, reply, protocol="native_ws"):
        self.calls, self.saved = [], None
        monkeypatch.setattr(judge, "get_scenario", lambda sid: row)
        monkeypatch.setattr(judge, "get_messages", lambda cid: msgs)
        monkeypatch.setattr(judge, "get_run", lambda rid: {"id": rid, "agent_id": 87})
        monkeypatch.setattr(judge, "get_agent", lambda aid: {"id": aid, "voice_protocol": protocol})
        monkeypatch.setattr(judge, "update_conversation_verdict", lambda *a: setattr(self, "saved", a))

        async def chat(system, messages, **kw):
            self.calls.append({"system": system, "user": messages[0]["content"]})
            if reply is None:
                raise AssertionError("the LLM must not be called")
            return json.loads(json.dumps(reply))

        monkeypatch.setattr(judge, "chat", chat)

    async def __call__(self):
        return await judge.judge_conversation({"id": 5, "scenario_id": 1, "run_id": 229})

    @property
    def user(self):
        return self.calls[0]["user"]


def _reply(accuracy=0.9, recognized=True, handled=True, outcome=True, expected=True, violated=(),
           observed="The agent handled the interrupt.", evidence="agent: ..."):
    return {"scores": {"accuracy": accuracy, "safety": 1.0, "hallucination": 1.0, "recovery": None},
            "verdict": "pass", "severity": "low", "recovered": None, "fail_category": None,
            "evidence": evidence,
            "interrupt": {"recognized": recognized, "handled": handled, "outcome_met": outcome,
                          "expected_outcome_met": expected, "violated_bug_guards": list(violated),
                          "observed": observed}}


# ---------------------------------------------------------------------------
# Context given to the Judge
# ---------------------------------------------------------------------------
async def test_interrupt_context_is_added_to_the_existing_prompt(monkeypatch):
    row, stored = _row("human_request", when="answer.good_time == 'Yes'")
    j = _Judge(monkeypatch, row, _msgs(("agent", "Hi.", None), ("tester", "I want to talk to a person.", None),
                                         ("agent", "Sure, one moment.", {"interrupt_key": "human_request"})), _reply())
    await j()
    system, user = j.calls[0]["system"], j.user
    assert system.startswith(judge.SYSTEM_PROMPT) and system.endswith(judge.INTERRUPT_RULES)
    assert "PLANNED INTERRUPT (fixed by the test plan)" in user
    assert "interrupt: human_request" in user and "goto — continue toward the planned target step 'route_counsellor_hours'" in user
    assert f"instead of answering step {stored['interrupt']['injection_step']} (ask_good_time)" in user
    assert "applies when: answer.good_time == 'Yes' (context only" in user
    assert f"expected outcome: {stored['expectations']['outcome']}" in user
    assert "1. No unexpected close." in user
    assert "user.first_name = Daniel" in user
    assert "interrupt keys the agent reported: human_request (turn 2)" in user
    assert "[turn 1] tester: I want to talk to a person." in user
    # The existing SCENARIO block is still there, unchanged in form.
    assert f"expected_behavior: {row['expected_behavior']}" in user and "test_type: flow_path" in user


async def test_normal_flow_path_prompt_is_unchanged(monkeypatch):
    row = {"id": 1, "user_goal": "g", "test_type": "flow_path", "assigned_fault": "none",
           "expected_behavior": "Planned path: A -> B", "node_script_json": json.dumps({"scenario_id": "x", "path": ["A"], "turns": []})}
    msgs = _msgs(("agent", "Hi.", None), ("tester", "Hello.", None), ("agent", "Bye.", {"call_id": "c"}))
    j = _Judge(monkeypatch, row, msgs, {"scores": {"accuracy": 0.9, "safety": 1.0, "hallucination": 1.0}, "evidence": "e"})
    result = await j()
    assert j.calls[0]["system"] == judge.SYSTEM_PROMPT
    assert j.user == (
        "SCENARIO\n  goal: g\n  expected_behavior: Planned path: A -> B\n  test_type: flow_path\n"
        "  assigned_fault: none\n\nTRANSCRIPT\nagent: Hi.\ntester: Hello.\nagent: Bye.\n\n"
        f"AGENT TRACE (per agent turn)\n{json.dumps([{'call_id': 'c'}], indent=2)}\n\n"
        "Judge this conversation now as json."
    )
    assert result["verdict"] == "pass" and result["evidence"] == "e"


# ---------------------------------------------------------------------------
# goto
# ---------------------------------------------------------------------------
GOTO_MSGS = _msgs(("agent", "Hi. Is now a good time?", None), ("tester", "I want to talk to a person.", None),
                  ("agent", "Our counsellors are available now. Would you like one?", {"interrupt_key": "human_request"}))


async def test_goto_recognized_and_correct_continuation_passes(monkeypatch):
    result = await _Judge(monkeypatch, _row("human_request")[0], GOTO_MSGS, _reply())()
    assert result["verdict"] == "pass"
    assert result["evidence"].startswith("Planned interrupt: human_request (goto → route_counsellor_hours). "
                                         "Observed interrupt key: human_request; end status: none.")


async def test_goto_recognized_but_wrong_continuation_fails(monkeypatch):
    result = await _Judge(monkeypatch, _row("human_request")[0], GOTO_MSGS,
                          _reply(accuracy=0.8, outcome=False, observed="It went back to verifying identity."))()
    assert result["verdict"] == "fail" and result["fail_category"] == "accuracy"
    assert result["scores"]["accuracy"] == 0.5  # consistent with the in-code fail
    assert "planned outcome not reached" in result["evidence"]
    assert "Observed: It went back to verifying identity." in result["evidence"]


async def test_goto_not_recognized_fails(monkeypatch):
    msgs = _msgs(("agent", "Is now a good time?", None), ("tester", "I want to talk to a person.", None),
                 ("agent", "Great, may I speak with Daniel?", {"answers": {"good_time": "Yes"}}))
    j = _Judge(monkeypatch, _row("human_request")[0], msgs, _reply(accuracy=0.3, recognized=False, handled=False, outcome=False))
    result = await j()
    assert "interrupt keys the agent reported: none — this transport reports interrupt keys, so none was recognized" in j.user
    assert result["verdict"] == "fail" and "planned interrupt not recognized" in result["evidence"]


async def test_missing_key_is_not_called_unrecognized_when_the_transport_cannot_report_keys(monkeypatch):
    j = _Judge(monkeypatch, _row("human_request")[0], GOTO_MSGS[:2] + [dict(GOTO_MSGS[2], trace_json=None)], _reply(),
               protocol="http_json")
    await j()
    assert "none (this transport may not report interrupt keys; judge from the transcript)" in j.user


# ---------------------------------------------------------------------------
# end
# ---------------------------------------------------------------------------
def _end_msgs(reply="Sorry to trouble you. Goodbye.", **trace):
    return _msgs(("agent", "Is now a good time?", None), ("tester", "Actually, wrong number.", None),
                 ("agent", reply, trace or None))


async def test_end_recognized_with_correct_end_passes(monkeypatch):
    row, stored = _row("wrong_number")
    msgs = _end_msgs(interrupt_key="wrong_number", end_status=stored["interrupt"]["end_status"])
    j = _Judge(monkeypatch, row, msgs, _reply())
    result = await j()
    assert "end status reported: 'No Answer | Retry Scheduled' (turn 2) — the agent ended the call" in j.user
    assert result["verdict"] == "pass"


async def test_end_with_the_wrong_end_status_fails_even_if_the_model_passes_it(monkeypatch):
    msgs = _end_msgs(interrupt_key="wrong_number", end_status="Retry Scheduled | In Progress")
    result = await _Judge(monkeypatch, _row("wrong_number")[0], msgs, _reply(accuracy=1.0))()
    assert result["verdict"] == "fail"
    assert "planned end status 'No Answer | Retry Scheduled', observed 'Retry Scheduled | In Progress'" in result["evidence"]


async def test_end_recognized_but_the_call_continues_fails(monkeypatch):
    msgs = _end_msgs(reply="Oh, sorry! Anyway, may I speak with Daniel?", interrupt_key="wrong_number")
    j = _Judge(monkeypatch, _row("wrong_number")[0], msgs, _reply(accuracy=0.4, outcome=False, expected=False))
    result = await j()
    assert "end status reported: none — the agent did not end the call with a status" in j.user
    assert result["verdict"] == "fail" and "planned outcome not reached" in result["evidence"]


async def test_end_not_recognized_fails(monkeypatch):
    msgs = _end_msgs(reply="Great, let's continue.")
    result = await _Judge(monkeypatch, _row("wrong_number")[0], msgs,
                          _reply(accuracy=0.2, recognized=False, handled=False, outcome=False, expected=False))()
    assert result["verdict"] == "fail" and result["severity"] == "high"


async def test_matching_end_status_alone_never_makes_a_scenario_pass(monkeypatch):
    row, stored = _row("wrong_number")
    msgs = _end_msgs(interrupt_key="wrong_number", end_status=stored["interrupt"]["end_status"])
    result = await _Judge(monkeypatch, row, msgs, _reply(accuracy=0.2, handled=False))()
    assert result["verdict"] == "fail"


# ---------------------------------------------------------------------------
# resume
# ---------------------------------------------------------------------------
def _resume_msgs(after_resume):
    return _msgs(("agent", "Is now a good time?", None), ("tester", "Actually, hold on.", None),
                 ("agent", "Of course, I'll wait.", {"interrupt_key": "hold"}),
                 ("tester", "Yes, I have time.", None),
                 ("agent", after_resume, {"interrupt_key": "hold"}))


async def test_resume_with_the_question_asked_again_passes(monkeypatch):
    msgs = _resume_msgs("Welcome back! Would you have a couple of minutes to talk?")
    assert (await _Judge(monkeypatch, _row("hold")[0], msgs, _reply())())["verdict"] == "pass"


async def test_resume_without_the_question_asked_again_fails(monkeypatch):
    j = _Judge(monkeypatch, _row("hold")[0], _resume_msgs(""), _reply(accuracy=0.6, outcome=False))
    result = await j()
    assert "empty agent replies (said nothing) at turn(s): 4" in j.user
    assert "resume event in the trace" in j.calls[0]["system"]
    assert result["verdict"] == "fail"


async def test_resume_not_recognized_fails(monkeypatch):
    msgs = _msgs(("agent", "Is now a good time?", None), ("tester", "Actually, hold on.", None),
                 ("agent", "No problem, I'll call back later. Goodbye.", {"end_status": "Retry Scheduled | In Progress"}))
    result = await _Judge(monkeypatch, _row("hold")[0], msgs, _reply(accuracy=0.2, recognized=False, outcome=False))()
    assert result["verdict"] == "fail"


# ---------------------------------------------------------------------------
# Bug guards / expected outcome
# ---------------------------------------------------------------------------
GUARDED = {"outcome": "The agent routes the caller to a counsellor.",
           "bug_guards": ["The agent does not ask for availability again.", "The agent does not end the call."]}


async def test_bug_guards_satisfied(monkeypatch):
    j = _Judge(monkeypatch, _row("human_request", GUARDED)[0], GOTO_MSGS, _reply(violated=[]))
    result = await j()
    assert "1. The agent does not ask for availability again." in j.user and "2. The agent does not end the call." in j.user
    assert result["verdict"] == "pass"


async def test_bug_guard_violated_fails_and_names_the_guard(monkeypatch):
    result = await _Judge(monkeypatch, _row("human_request", GUARDED)[0], GOTO_MSGS, _reply(accuracy=0.9, violated=[2, 7, True]))()
    assert result["verdict"] == "fail"
    assert "bug guard 2 violated: The agent does not end the call." in result["evidence"]
    assert "bug guard 7" not in result["evidence"]  # out-of-range numbers are ignored


async def test_expected_outcome_satisfied(monkeypatch):
    assert (await _Judge(monkeypatch, _row("human_request", GUARDED)[0], GOTO_MSGS, _reply(expected=True))())["verdict"] == "pass"


async def test_expected_outcome_mismatched(monkeypatch):
    result = await _Judge(monkeypatch, _row("human_request", GUARDED)[0], GOTO_MSGS, _reply(accuracy=0.9, expected=False))()
    assert result["verdict"] == "fail" and "expected outcome not met" in result["evidence"]


async def test_low_accuracy_alone_still_fails_without_checks(monkeypatch):
    reply = _reply(accuracy=0.3)
    del reply["interrupt"]
    assert (await _Judge(monkeypatch, _row("hold")[0], _resume_msgs(""), reply)())["verdict"] == "fail"


# ---------------------------------------------------------------------------
# Deterministic short-circuits are unchanged
# ---------------------------------------------------------------------------
async def test_interrupt_scenario_that_never_reached_the_agent_is_a_system_failure(monkeypatch):
    msgs = _msgs(("tester", "Actually, hold on.", None), ("agent", "<error>", {"error": "native_ws call setup error: boom"}))
    result = await _Judge(monkeypatch, _row("hold")[0], msgs, None)()
    assert result["verdict"] == "fail" and result["fail_category"] == "system"
    assert result["evidence"] == "trace error: native_ws call setup error: boom"


async def test_injected_system_fault_short_circuit_is_unchanged(monkeypatch):
    row = dict(_row("hold")[0], assigned_fault="api_timeout")
    result = await _Judge(monkeypatch, row, _resume_msgs("x"), None)()
    assert result["fail_category"] == "system" and result["severity"] == "med"


# ---------------------------------------------------------------------------
# Realistic Maya cases — the actual Part 7 runs
# ---------------------------------------------------------------------------
async def test_maya_wrong_number_run_227(monkeypatch):
    row, _ = _row("wrong_number", {"outcome": "The call ends with the status 'No Answer | Retry Scheduled'.",
                                   "bug_guards": [], "end_status": "No Answer | Retry Scheduled"})
    msgs = _msgs(("agent", MAYA_OPENING, None), ("tester", "wrong number", None),
                 ("agent", "I'm sorry to have troubled you — thank you, take care!",
                  {"call_id": "43faf6ec", "end_status": "No Answer | Retry Scheduled", "interrupt_key": "wrong_number"}))
    j = _Judge(monkeypatch, row, msgs, _reply(accuracy=1.0))
    result = await j()
    assert "interrupt keys the agent reported: wrong_number (turn 2)" in j.user
    assert "'No Answer | Retry Scheduled' (turn 2) — the agent ended the call" in j.user
    assert result["verdict"] == "pass"
    assert result["evidence"].startswith("Planned interrupt: wrong_number (end). Observed interrupt key: wrong_number; "
                                         "end status: 'No Answer | Retry Scheduled'.")


async def test_maya_hold_run_228_empty_resume_does_not_auto_pass(monkeypatch):
    row, _ = _row("hold", {"outcome": "The agent successfully resumes the conversation after the hold interrupt.",
                           "bug_guards": ["The agent must return to the interrupted step and ask the same question again.",
                                          "The call status should remain 'In Progress' after the hold."]})
    msgs = _msgs(("agent", MAYA_OPENING, None), ("tester", "hold on", None),
                 ("agent", "Of course, take your time — I'll wait.", {"call_id": "61d613b8", "interrupt_key": "hold"}),
                 ("tester", "Yes, I have time.", None),
                 ("agent", "", {"call_id": "61d613b8", "interrupt_key": "hold"}))
    # The model recognizes the hold but finds the resume missing.
    j = _Judge(monkeypatch, row, msgs, _reply(accuracy=0.4, outcome=False, expected=False, violated=[1],
                                              observed="It acknowledged the hold but never re-asked the question."))
    result = await j()
    assert "interrupt keys the agent reported: hold (turn 2), hold (turn 4)" in j.user
    assert "empty agent replies (said nothing) at turn(s): 4" in j.user
    assert result["verdict"] == "fail"
    assert "bug guard 1 violated: The agent must return to the interrupted step" in result["evidence"]


async def test_maya_hold_run_228_structured_facts_alone_do_not_pass_it(monkeypatch):
    """Even if the model's checks come back empty, the recognized key and resume event
    are not treated as success: the verdict rests on the model's accuracy score."""
    row, _ = _row("hold")
    msgs = _msgs(("agent", MAYA_OPENING, None), ("tester", "hold on", None),
                 ("agent", "Of course, take your time — I'll wait.", {"interrupt_key": "hold"}),
                 ("tester", "Yes, I have time.", None), ("agent", "", {"interrupt_key": "hold"}))
    reply = _reply(accuracy=0.3)
    reply["interrupt"] = {}
    assert (await _Judge(monkeypatch, row, msgs, reply)())["verdict"] == "fail"


async def test_maya_human_request_run_229_treated_as_decline(monkeypatch):
    row, _ = _row("human_request", {
        "outcome": "The agent successfully routes the caller to a counsellor after handling the interrupt.",
        "bug_guards": ["The agent does not ask for availability again after the interrupt.",
                       "The agent correctly identifies the interrupt and transitions to the counselling route."]})
    msgs = _msgs(("agent", MAYA_OPENING, None), ("tester", "I want to talk to a person.", None),
                 ("agent", "Would you have a couple of minutes to talk about your Medi-Cal renewal?",
                  {"call_id": "7de9cda9", "end_status": "Retry Scheduled | In Progress"}))
    j = _Judge(monkeypatch, row, msgs, _reply(
        accuracy=0.1, recognized=False, handled=False, outcome=False, expected=False, violated=[1, 2],
        observed="It re-asked the availability question and ended the call as a decline."))
    result = await j()
    assert "interrupt keys the agent reported: none — this transport reports interrupt keys, so none was recognized" in j.user
    assert "'Retry Scheduled | In Progress' (turn 2) — the agent ended the call" in j.user
    assert result["verdict"] == "fail" and result["severity"] == "high"
    ev = result["evidence"]
    assert ev.startswith("Planned interrupt: human_request (goto → route_counsellor_hours). Observed interrupt key: none; "
                         "end status: 'Retry Scheduled | In Progress'.")
    assert "planned interrupt not recognized" in ev and "bug guard 2 violated" in ev


# ---------------------------------------------------------------------------
# Part 9 — missing interrupt checks, structured failures, Judge errors
# ---------------------------------------------------------------------------
class _Sequence:
    """Judge whose model returns `replies` in order (an Exception is raised)."""

    def __init__(self, monkeypatch, row, msgs, replies, protocol="native_ws"):
        self.base = _Judge(monkeypatch, row, msgs, {}, protocol)
        self.replies = list(replies)

        async def chat(system, messages, **kw):
            self.base.calls.append({"system": system, "messages": messages})
            r = self.replies.pop(0)
            if isinstance(r, Exception):
                raise r
            return json.loads(json.dumps(r))

        monkeypatch.setattr(judge, "chat", chat)

    async def __call__(self):
        return await self.base()


async def test_missing_interrupt_checks_are_asked_for_once_and_used(monkeypatch):
    first = _reply(accuracy=0.9)
    del first["interrupt"]
    j = _Sequence(monkeypatch, _row("human_request")[0], GOTO_MSGS,
                  [first, _reply(accuracy=0.9, outcome=False, observed="It never offered a counsellor.")])
    result = await j()
    assert len(j.base.calls) == 2
    retry_msgs = j.base.calls[1]["messages"]
    assert [m["role"] for m in retry_msgs] == ["user", "assistant", "user"] and '"interrupt" object' in retry_msgs[2]["content"]
    assert result["verdict"] == "fail" and "planned outcome not reached" in result["evidence"]


async def test_checks_still_missing_after_retry_fall_back_to_accuracy_and_trace(monkeypatch):
    bad = _reply(accuracy=0.9)
    bad["interrupt"] = {"recognized": "yes"}  # malformed
    result = await _Sequence(monkeypatch, _row("human_request")[0], GOTO_MSGS, [bad, bad])()
    assert result["verdict"] == "pass"  # accuracy 0.9 and the trace shows the planned key
    assert "did not return its interrupt checks" in result["evidence"]


async def test_planned_key_never_reported_fails_without_the_models_checks(monkeypatch):
    """Maya run 229's shape, but the model (wrongly) passes it and omits its checks."""
    msgs = _msgs(("agent", MAYA_OPENING, None), ("tester", "I want to talk to a person.", None),
                 ("agent", "Would you have a couple of minutes to talk about your Medi-Cal renewal?",
                  {"end_status": "Retry Scheduled | In Progress"}))
    lenient = _reply(accuracy=1.0)
    del lenient["interrupt"]
    result = await _Sequence(monkeypatch, _row("human_request")[0], msgs, [lenient, lenient])()
    assert result["verdict"] == "fail"
    assert "the agent never reported the planned interrupt 'human_request'" in result["evidence"]


async def test_end_interrupt_without_an_end_status_fails_on_a_reporting_transport(monkeypatch):
    msgs = _end_msgs(reply="Sorry! Goodbye.", interrupt_key="wrong_number")
    result = await _Judge(monkeypatch, _row("wrong_number")[0], msgs, _reply(accuracy=1.0))()
    assert result["verdict"] == "fail" and "the call did not end (no end status reported)" in result["evidence"]


async def test_structured_checks_do_not_apply_where_the_transport_reports_nothing(monkeypatch):
    msgs = _end_msgs(reply="Sorry! Goodbye.")
    result = await _Judge(monkeypatch, _row("wrong_number")[0], msgs, _reply(accuracy=1.0), protocol="http_json")()
    assert result["verdict"] == "pass"


async def test_judge_error_never_passes_an_interrupt_scenario(monkeypatch):
    j = _Sequence(monkeypatch, _row("hold")[0], _resume_msgs("Welcome back! Is now a good time?"),
                  [RuntimeError("model down")])
    result = await j()
    assert result["verdict"] == "fail" and result["fail_category"] == "system"
    assert "Judge unavailable (RuntimeError)" in result["evidence"] and "not counted as a pass" in result["evidence"]
    assert j.base.saved[1:3] == ("fail", "med")


async def test_judge_error_on_a_normal_scenario_keeps_the_existing_fallback(monkeypatch):
    row = {"id": 1, "user_goal": "g", "test_type": "flow_path", "assigned_fault": "none", "expected_behavior": "x"}
    msgs = _msgs(("tester", "Hi.", None), ("agent", "Hello.", None))
    result = await _Sequence(monkeypatch, row, msgs, [RuntimeError("model down")])()
    assert result["verdict"] == "pass" and result["evidence"] == "judge unavailable"
