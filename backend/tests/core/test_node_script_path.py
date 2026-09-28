"""Tests for the flow-scenario (path) script helpers in app.core.node_script:
validate_path_script, the path prompt, and generate_path_script with a faked LLM.
The pre-existing node-script helpers keep their own tests in test_node_script.py.
"""
import pytest

from app.core import node_script
from app.core.node_script import NodeScriptError, generate_path_script, validate_path_script

PATH = ["greeting", "ask_name", "validate_name", "retry_name", "validate_name", "ask_dob", "complete"]


def _turn(step, node_id, caller="Hi, this is Rahul Sharma.", behavior="Agent does its job."):
    return {"step": step, "node_id": node_id, "expected_agent_behavior": behavior, "caller_line": caller}


GOOD = {
    "test_goal": "Verify the agent recovers from a wrong name.",
    "turns": [
        _turn(1, "greeting", "Hi, I'm calling about my account."),
        _turn(2, "ask_name", "My name is Rahul."),
        _turn(4, "retry_name", "Sorry — it's Srija Chittaluru."),
        _turn(6, "ask_dob", "March 3rd, 1991."),
    ],
}


def _errors(result, path=PATH):
    with pytest.raises(NodeScriptError) as exc:
        validate_path_script(result, path)
    return " | ".join(exc.value.errors)


def test_accepts_a_well_formed_script_and_normalizes_it():
    messy = {
        "test_goal": "  Verify recovery.  ",
        "turns": [dict(_turn(2, "ask_name"), caller_line="  My name is Rahul.  ", extra="ignored")],
    }
    assert validate_path_script(messy, PATH) == {
        "test_goal": "Verify recovery.",
        "turns": [{
            "step": 2, "node_id": "ask_name",
            "expected_agent_behavior": "Agent does its job.", "caller_line": "My name is Rahul.",
        }],
    }


def test_revisited_node_is_anchored_by_position():
    # validate_name is at steps 3 AND 5; both are legal anchors, in order.
    script = {"test_goal": "g", "turns": [_turn(3, "validate_name"), _turn(5, "validate_name")]}
    assert [t["step"] for t in validate_path_script(script, PATH)["turns"]] == [3, 5]


def test_full_good_script_passes():
    assert len(validate_path_script(GOOD, PATH)["turns"]) == 4


@pytest.mark.parametrize("result,needle", [
    ("not a dict", "JSON object"),
    ({"turns": GOOD["turns"]}, "test_goal"),
    ({"test_goal": "  ", "turns": GOOD["turns"]}, "test_goal"),
    ({"test_goal": "g"}, "turns must be a non-empty array"),
    ({"test_goal": "g", "turns": []}, "turns must be a non-empty array"),
    ({"test_goal": "g", "turns": "nope"}, "turns must be a non-empty array"),
])
def test_rejects_malformed_top_level(result, needle):
    assert needle in _errors(result)


def test_rejects_missing_or_empty_caller_line():
    no_line = {"test_goal": "g", "turns": [{"step": 1, "node_id": "greeting", "expected_agent_behavior": "x"}]}
    assert "caller_line" in _errors(no_line)
    blank = {"test_goal": "g", "turns": [_turn(1, "greeting", caller="   ")]}
    assert "caller_line" in _errors(blank)


def test_rejects_missing_expected_behavior():
    turn = {"step": 1, "node_id": "greeting", "caller_line": "Hi."}
    assert "expected_agent_behavior" in _errors({"test_goal": "g", "turns": [turn]})


@pytest.mark.parametrize("step", [0, 8, -1, "2", 2.0, True, None])
def test_rejects_a_step_outside_the_path(step):
    assert "step must be an integer from 1 to 7" in _errors({"test_goal": "g", "turns": [_turn(step, "greeting")]})


def test_rejects_a_node_that_is_not_at_that_step():
    # "ask_email" exists nowhere in the path; "ask_dob" exists but not at step 2.
    assert "must be 'ask_name'" in _errors({"test_goal": "g", "turns": [_turn(2, "ask_email")]})
    assert "must be 'ask_name'" in _errors({"test_goal": "g", "turns": [_turn(2, "ask_dob")]})


@pytest.mark.parametrize("steps", [[3, 2], [2, 2]])
def test_rejects_turns_that_reorder_or_repeat_steps(steps):
    turns = [_turn(s, PATH[s - 1]) for s in steps]
    assert "turns follow the path in order" in _errors({"test_goal": "g", "turns": turns})


@pytest.mark.parametrize("line", ["My name is [name].", "Born on {dob}.", "Email me at <email>."])
def test_rejects_placeholders_in_caller_lines(line):
    assert "placeholder" in _errors({"test_goal": "g", "turns": [_turn(1, "greeting", caller=line)]})


def test_a_long_path_may_have_more_turns_than_a_node_script():
    # A path script is bounded by its path (one turn per step), not by MAX_SCRIPT_TURNS.
    path = [f"n{i}" for i in range(10)]
    turns = [_turn(i + 1, f"n{i}") for i in range(node_script.MAX_SCRIPT_TURNS + 3)]
    assert len(validate_path_script({"test_goal": "g", "turns": turns}, path)["turns"]) == 8


def test_turns_can_never_outnumber_path_steps():
    path = ["a", "b"]
    turns = [_turn(1, "a"), _turn(2, "b"), _turn(2, "b")]
    errors = _errors({"test_goal": "g", "turns": turns}, path)
    assert "turns follow the path in order" in errors and "maximum is 2" in errors


def test_node_scripts_keep_the_five_turn_cap():
    six = [{"expected_agent_behavior": "x", "caller_line": "y"}] * 6
    with pytest.raises(NodeScriptError) as exc:
        node_script.validate_script(six)
    assert "the maximum is 5" in exc.value.errors[0]


def test_reports_every_problem_not_just_the_first():
    bad = {"test_goal": "", "turns": [_turn(9, "x"), _turn(1, "wrong", caller="[name]")]}
    errors = _errors(bad)
    assert "test_goal" in errors and "step must be" in errors and "must be 'greeting'" in errors
    assert "placeholder" in errors


def test_prompt_lists_the_exact_path_in_order_with_loops_and_untaken_branches():
    nodes = {n: {"id": n, "name": n.replace("_", " ").title()} for n in set(PATH) | {"give_up"}}
    out_adj = {"validate_name": ["ask_dob", "retry_name", "give_up"], "retry_name": ["validate_name"]}
    prompt = node_script._build_path_prompt(PATH, nodes, out_adj, "recovery")
    positions = [prompt.index(f"Step {i + 1}") for i in range(len(PATH))]
    assert positions == sorted(positions)
    for node_id in PATH:
        assert f"id: {node_id}" in prompt
    assert "Step 5 (returns to the node from step 3)" in prompt
    # At step 3 the path goes to retry_name; the other real options are named.
    step3 = prompt[prompt.index("Step 3"):prompt.index("Step 4")]
    assert "other branches not taken from here: Ask Dob, Give Up" in step3
    assert "SCENARIO KIND (structural only): recovery" in prompt


async def test_generate_path_script_validates_model_output(monkeypatch):
    seen = {}

    async def fake_chat(system, messages, json_mode=False, **kw):
        seen.update(system=system, prompt=messages[0]["content"], json_mode=json_mode)
        return GOOD

    monkeypatch.setattr(node_script, "chat", fake_chat)
    out = await generate_path_script(PATH, {}, {}, "recovery")
    assert out["test_goal"] == GOOD["test_goal"]
    assert [t["step"] for t in out["turns"]] == [1, 2, 4, 6]
    assert seen["json_mode"] is True
    assert seen["system"] is node_script.PATH_SYSTEM_PROMPT


async def test_generate_path_script_rejects_bad_model_output(monkeypatch):
    async def fake_chat(*a, **kw):
        return {"test_goal": "g", "turns": [_turn(1, "not_on_path")]}

    monkeypatch.setattr(node_script, "chat", fake_chat)
    with pytest.raises(NodeScriptError):
        await generate_path_script(PATH, {}, {}, "recovery")


def test_path_prompt_is_separate_from_the_node_prompt():
    assert node_script.PATH_SYSTEM_PROMPT is not node_script.SYSTEM_PROMPT
    assert "test script for ONE node" in node_script.SYSTEM_PROMPT
    assert "ONE EXACT PATH" in node_script.PATH_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Steps where the caller never replies (say / branch / end in a modular flow)
# ---------------------------------------------------------------------------
from tests.routers.test_flow_interrupt_scenarios import PARSED  # noqa: E402

MAYA_NODES = {str(n["id"]): n for n in PARSED["nodes"]}
MAYA_KINDS = {k: str(n.get("type") or "") for k, n in MAYA_NODES.items()}
# "Branch Scenario 13" of the Maya modular flow.
BRANCH_13 = ["greet", "ask_good_time", "route_good_time", "ask_identity", "route_identity", "ask_dob",
             "route_dob", "thank_for_verifying", "ask_packet_received", "route_packet", "ask_already_sent",
             "route_sent", "ask_online_account", "route_online_account", "ask_has_email", "route_prereqs",
             "close_counsellor_setup", "route_counsellor_direct", "end_counsellor_transfer"]
ASK_LINES = {2: "Yes, I have a couple of minutes.", 4: "Yes, that's me.", 6: "March 14th, 1985.",
             9: "Yes, I got the packet.", 11: "Yes, I sent it back.", 13: "Yes, I'd like my own account.",
             15: "I have an email and I'm at my computer."}


def _branch_13(extra=()):
    turns = [_turn(step, BRANCH_13[step - 1], line) for step, line in ASK_LINES.items()]
    turns += [_turn(step, BRANCH_13[step - 1], line) for step, line in extra]
    return {"test_goal": "g", "turns": sorted(turns, key=lambda t: t["step"])}


def test_asks_only_script_is_valid_on_a_modular_flow():
    out = validate_path_script(_branch_13(), BRANCH_13, MAYA_KINDS)
    assert [t["step"] for t in out["turns"]] == [2, 4, 6, 9, 11, 13, 15]


def test_the_reported_model_output_is_rejected_with_a_clear_reason():
    """The actual failing draft: the agent's greeting as the caller's first line, and
    empty lines at the two say steps."""
    script = _branch_13([(1, "Hello, this is John Smith calling from ABC Health Services about my renewal."),
                         (8, ""), (17, "")])
    with pytest.raises(NodeScriptError) as exc:
        validate_path_script(script, BRANCH_13, MAYA_KINDS)
    errors = exc.value.errors
    assert errors == [
        "turns[0]: step 1 (greet) is a 'say' step — the agent only speaks there and the caller does not reply; "
        "remove this turn (the caller answers at the next 'ask' step).",
        "turns[4]: step 8 (thank_for_verifying) is a 'say' step — the agent only speaks there and the caller "
        "does not reply; remove this turn (the caller answers at the next 'ask' step).",
        "turns[9]: step 17 (close_counsellor_setup) is a 'say' step — the agent only speaks there and the "
        "caller does not reply; remove this turn (the caller answers at the next 'ask' step).",
    ]
    assert not any("caller_line must be a non-empty string" in e for e in errors)


@pytest.mark.parametrize("step, kind", [(3, "branch"), (19, "end")])
def test_turns_at_branch_and_end_steps_are_rejected(step, kind):
    with pytest.raises(NodeScriptError) as exc:
        validate_path_script(_branch_13([(step, "Okay.")]), BRANCH_13, MAYA_KINDS)
    assert f"is a '{kind}' step" in exc.value.errors[0]


def test_flows_without_step_kinds_are_unaffected():
    # Untyped / differently typed flows (e.g. a JSON flow) keep the existing rules.
    assert len(validate_path_script(GOOD, PATH, {n: "" for n in PATH})["turns"]) == 4
    assert len(validate_path_script(GOOD, PATH, {"greeting": "greeting"})["turns"]) == 4
    assert len(validate_path_script(_branch_13([(1, "Hello.")]), BRANCH_13)["turns"]) == 8  # no kinds given


def test_path_prompt_marks_which_steps_get_a_turn():
    prompt = node_script._build_path_prompt(BRANCH_13, MAYA_NODES, {}, "branch")
    step1 = prompt[prompt.index("Step 1:"):prompt.index("Step 2:")]
    step2 = prompt[prompt.index("Step 2:"):prompt.index("Step 3:")]
    step3 = prompt[prompt.index("Step 3:"):prompt.index("Step 4:")]
    assert "caller: NO turn at this step (the agent only speaks there)" in step1
    assert "caller: replies here" in step2
    assert "caller: NO turn at this step (the agent routes internally there)" in step3
    assert 'NEVER gets a turn — not even one with an empty caller_line' in node_script.PATH_SYSTEM_PROMPT


async def test_generate_path_script_applies_the_rule(monkeypatch):
    async def fake_chat(*a, **kw):
        return _branch_13([(8, "")])

    monkeypatch.setattr(node_script, "chat", fake_chat)
    with pytest.raises(NodeScriptError) as exc:
        await generate_path_script(BRANCH_13, MAYA_NODES, {}, "branch")
    assert "step 8 (thank_for_verifying) is a 'say' step" in exc.value.errors[0]
