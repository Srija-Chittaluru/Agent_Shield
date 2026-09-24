"""Node script generation for flow-aware / node-based voice testing (Phase 2).

Turns a selected flow node (+ its prerequisite path) into:
  - a concise, test-oriented "test_goal" focused on THAT node
  - a deterministic, turn-by-turn caller script:
    [{"expected_agent_behavior": str, "caller_line": str}, ...]

This module is authoring-time only, called once per "Generate Test Script" click. It
never runs during a scenario's execution: app.core.runner, app.core.ai_caller and
app.core.voice_caller are untouched and know nothing about this module. A later phase
flattens a reviewed script into the existing scenario shape — seed_turns =
[t["caller_line"] for t in script], expected_behavior = the numbered
expected_agent_behavior values — so the EXISTING scripted runner (_run_scripted) plays
it verbatim. There is no runtime AI-Caller improvisation here: the LLM writes the
script once, a human reviews/edits it, and only the reviewed caller lines are ever
spoken.

CRITICAL role separation (mirrors app.core.ai_caller's docstring): "caller_line" is
what the SIMULATED CUSTOMER says. It must never mention nodes, flows, graphs,
functions, APIs, internal state, workflow names, traces, or any other implementation
detail of the Voice Agent under test — a real caller has no idea any of that exists.
"expected_agent_behavior" is an evaluation reference for the Judge only — it is NEVER
given to the Voice Agent as a line to speak, and the real Voice Agent's response is
never scripted or assumed.
"""
import json
import re
from typing import Optional

from app.core.llm import chat

NODE_TEST_TYPE = "flow_node"

# Prerequisite turns + the node-under-test's own turns, combined. MUST match
# app.core.runner.MAX_TESTER_TURNS: _run_scripted() (reused unmodified to execute a
# saved node script — see Phase 3) hard-stops after that many tester turns regardless
# of how many seed_turns are stored, so a script longer than this would have its tail
# turns silently never played. runner.py is deliberately not touched by this feature,
# so this cap follows it rather than the other way around.
MAX_SCRIPT_TURNS = 5


class NodeScriptError(ValueError):
    """Raised with every validation problem found, not just the first. Malformed model
    output is rejected outright here — never silently repaired.
    """

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("; ".join(errors))


def validate_script(script: object, max_turns: int = MAX_SCRIPT_TURNS) -> list[dict]:
    """Validate `script` is a well-formed node script; return it normalized (order
    preserved, both fields trimmed). Raises NodeScriptError otherwise. `max_turns`
    defaults to MAX_SCRIPT_TURNS; only a flow path script passes a different bound.
    """
    if not isinstance(script, list) or not script:
        raise NodeScriptError(["script must be a non-empty array of turns."])

    errors: list[str] = []
    if len(script) > max_turns:
        errors.append(f"script has {len(script)} turns; the maximum is {max_turns}.")

    turns: list[dict] = []
    for i, t in enumerate(script):
        if not isinstance(t, dict):
            errors.append(f"script[{i}] must be an object.")
            continue
        caller_line = t.get("caller_line")
        behavior = t.get("expected_agent_behavior")
        if not isinstance(caller_line, str) or not caller_line.strip():
            errors.append(f"script[{i}].caller_line must be a non-empty string.")
            continue
        if not isinstance(behavior, str) or not behavior.strip():
            errors.append(f"script[{i}].expected_agent_behavior must be a non-empty string.")
            continue
        turns.append({
            "expected_agent_behavior": behavior.strip(),
            "caller_line": caller_line.strip(),
        })

    if errors:
        raise NodeScriptError(errors)
    return turns


def prerequisite_path(nodes: list[dict], edges: list[dict], target_id: str) -> list[dict]:
    """The chain of nodes that must be passed through before `target_id`, root-first,
    NOT including the target itself.

    Deliberately NOT a graph engine: this just walks one predecessor per step (a node
    with several incoming edges uses the first one found in `edges`), which is enough
    to flatten a linear onboarding path into one ordered script without building
    branch-choice UI or any execution-time graph state. A cycle or missing node simply
    stops the walk early rather than looping forever.
    """
    by_id = {n["id"]: n for n in nodes}
    parents: dict[str, str] = {}
    for e in edges:
        parents.setdefault(e["to"], e["from"])  # first-found predecessor wins

    chain: list[str] = []
    seen: set[str] = {target_id}
    current = target_id
    while current in parents:
        prev = parents[current]
        if prev not in by_id or prev in seen:
            break
        chain.append(prev)
        seen.add(prev)
        current = prev

    chain.reverse()
    return [by_id[nid] for nid in chain]


SYSTEM_PROMPT = """You are a test-script writer for a SCRIPTED voice-agent tester. You write \
a deterministic, turn-by-turn test script for ONE node in a voice agent's conversation flow. \
Respond ONLY with a json object.

ROLE SEPARATION — read carefully:
- You are writing lines for the CALLER (a simulated customer), never for the Voice Agent under
  test. The Voice Agent is a black box; its real responses will be captured live later, never
  scripted. "expected_agent_behavior" is an EVALUATION reference for a judge, not a line anyone
  will say out loud — never write a literal sentence for the agent to speak.
- The caller must sound like a real human customer on a phone call: natural, concise, spoken
  language. NEVER mention nodes, flows, graphs, functions, APIs, internal state, workflow names,
  traces, or any other implementation detail of the system under test — a real caller has no
  idea any of that exists and would never reference it.
- Keep the caller's lines SHORT and to the point. Avoid unnecessary chit-chat.

WHAT YOU RECEIVE:
- The node under test (its name, type, purpose, and expected_inputs if any).
- Any PREREQUISITE nodes that must be passed through first, in order, to reach the node under
  test in a real conversation (their name/purpose only).
- A test intent (either given by the user, or left for you to infer from the node's purpose).

WHAT TO PRODUCE:
{
  "test_goal": "one or two sentences describing WHAT this test verifies, concise and
                test-oriented, focused ONLY on the node under test — never a generic summary
                of the whole flow (e.g. never 'verify greeting, auth and account help')",
  "script": [
    {"expected_agent_behavior": "...", "caller_line": "..."},
    ...
  ]
}

SCRIPT RULES:
- If prerequisite nodes are given, the script MUST begin with one short, natural, FULLY
  COOPERATIVE caller turn per prerequisite node (in the given order) that plausibly gets a real
  agent through that step — e.g. a greeting node gets a simple opening line; an authentication
  node gets a turn where the caller gives correct-looking information. These prerequisite turns
  exist ONLY to put a real agent into the right state — they are not what this test is about,
  and the goal must not describe them as the point of the test.
- After the prerequisite turns, write the turns that actually TEST the node under test. This is
  where the test intent lives — including deliberately incorrect/edge-case caller lines if the
  test intent calls for it (e.g. giving a wrong name before the right one).
- Every turn needs both fields. "caller_line" is EXACTLY what the simulated caller will say,
  verbatim, with no further improvisation — write it as a complete, natural spoken sentence.
- "expected_agent_behavior" describes what a good agent should do in response to that caller
  line — never a literal sentence for the agent to say, always a description of correct
  behavior a judge can check the real transcript against.
- Use at most 5 turns total (prerequisites + node-under-test turns combined). Prefer fewer,
  focused turns over padding."""


def _describe_node(node: dict) -> str:
    lines = [f"name: {node.get('name')}", f"id: {node.get('id')}"]
    if node.get("type"):
        lines.append(f"type: {node['type']}")
    if node.get("purpose"):
        lines.append(f"purpose: {node['purpose']}")
    if node.get("expected_inputs"):
        lines.append(f"expected_inputs: {', '.join(node['expected_inputs'])}")
    return "\n".join(lines)


def _build_prompt(
    node: dict, prerequisite_nodes: list[dict], user_goal_hint: Optional[str]
) -> str:
    parts = ["NODE UNDER TEST:", _describe_node(node)]
    if prerequisite_nodes:
        parts.append("\nPREREQUISITE NODES (in order, must be passed through first):")
        for i, p in enumerate(prerequisite_nodes, 1):
            parts.append(f"{i}. {_describe_node(p)}")
    else:
        parts.append("\nThis node has no prerequisites — the script opens directly on it.")
    hint = (user_goal_hint or "").strip()
    parts.append(
        "\nTEST INTENT: "
        + (hint if hint else
           "(none given — infer an appropriate, concrete test from the node's own "
           "purpose and expected_inputs.)")
    )
    parts.append("\nGenerate the test_goal and script now as json.")
    return "\n".join(parts)


async def generate_node_script(
    node: dict,
    prerequisite_nodes: list[dict],
    user_goal_hint: Optional[str] = None,
) -> dict:
    """Generate {"test_goal": str, "script": [...]} for `node` via one LLM call.

    `prerequisite_nodes` (root-first, NOT including `node` itself — see
    prerequisite_path()) are woven in as plain pass-through turns ahead of the
    node-under-test's own turns. `user_goal_hint`, if given, steers what the final
    test_goal verifies; otherwise one is written from the node's own purpose/
    expected_inputs alone.

    Raises NodeScriptError if the model's output fails validation. Unlike
    app.core.scenarios/app.core.ai_caller, there is deliberately NO canned fallback
    bank here: a malformed script must surface as a clear error for the user to
    retry/edit, never a silently substituted generic script.
    """
    result = await chat(
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": _build_prompt(node, prerequisite_nodes, user_goal_hint)}],
        json_mode=True,
    )

    if not isinstance(result, dict):
        raise NodeScriptError(["Model did not return a JSON object."])

    test_goal = result.get("test_goal")
    if not isinstance(test_goal, str) or not test_goal.strip():
        raise NodeScriptError(["Model did not return a non-empty 'test_goal' string."])

    script = validate_script(result.get("script"))
    return {"test_goal": test_goal.strip(), "script": script}


def to_scenario_dict(flow_id: int, node_id: str, node_name: str, test_goal: str, script: list[dict]) -> dict:
    """Flatten a reviewed node script into the EXISTING scenario/test-case shape
    (app.core.scenarios._normalize's canonical fields), so it can be saved and run
    through the unmodified existing pipeline (Phase 3):

      seed_turns        = [turn["caller_line"] for turn in script]   (played verbatim
                           by the existing app.core.runner._run_scripted — never
                           app.core.ai_caller, since customer_context stays empty)
      expected_behavior  = the expected_agent_behavior values, numbered, for the
                           existing Judge (app.core.judge) to grade the REAL Voice
                           Agent responses against — never shown to the caller/agent
      user_goal          = test_goal, so the existing Judge prompt's "goal:" line
                           carries the node's test intent as evaluation context
      test_type          = NODE_TEST_TYPE ("flow_node"), assigned_fault = "none"

    `script` is assumed already validated (see validate_script) — this function does
    not re-validate it.
    """
    seed_turns = [turn["caller_line"] for turn in script]
    expected_behavior = "\n".join(
        f"{i}. Expected behavior: {turn['expected_agent_behavior']}"
        for i, turn in enumerate(script, start=1)
    )
    return {
        "title": f"Flow node: {node_name}",
        "user_goal": test_goal,
        "test_type": NODE_TEST_TYPE,
        "assigned_fault": "none",
        "expected_behavior": expected_behavior,
        "seed_turns": seed_turns,
        # Empty, deliberately: this forces app.core.runner.run_scenario() onto the
        # SCRIPTED path (_run_scripted), never the dynamic AI Caller — the whole point
        # of a flow-node test is that the caller improvises nothing.
        "customer_context": "",
        "max_turns": None,
        "flow_id": flow_id,
        "node_id": node_id,
        "node_script_json": json.dumps(script),
    }


# ---------------------------------------------------------------------------
# Flow scenario (path) scripts — same shape and rules as a node script, but for one
# whole candidate path planned by app.core.flow_graph. Everything above is unchanged;
# this section only reuses validate_script() and the same chat() call.
#
# The path is authoritative. Each turn is anchored to one position in it: "step" is
# the 1-based index into the path and "node_id" must be the node at that index. The
# validator enforces that steps only move forward through the path, so a script can
# never add a node, reorder the path, or branch somewhere the path doesn't go. (Not
# every step gets a turn — the caller only speaks where the conversation needs a
# reply — and which steps need one isn't derivable from the graph, so coverage is
# asked of the model rather than enforced.) Because steps strictly increase, a script
# has at most one turn per step: its length is bounded by the path, not by
# MAX_SCRIPT_TURNS, which would make a path with more than 5 caller replies
# unscriptable.
#
# Turn semantics match execution (app.core.runner drains the agent's opening turn
# first): at each step the agent speaks/acts, then the caller replies.
# ---------------------------------------------------------------------------
PATH_SYSTEM_PROMPT = """You are a test-script writer for a SCRIPTED voice-agent tester. You write \
a deterministic, turn-by-turn caller script that drives a voice agent along ONE EXACT PATH \
through its conversation flow. Respond ONLY with a json object.

ROLE SEPARATION — read carefully:
- You are writing lines for the CALLER (a simulated customer), never for the Voice Agent under
  test. The Voice Agent is a black box; its real responses will be captured live later, never
  scripted. "expected_agent_behavior" is an EVALUATION reference for a judge, not a line anyone
  will say out loud — never write a literal sentence for the agent to speak.
- The caller must sound like a real human customer on a phone call: natural, concise, spoken
  language. NEVER mention nodes, steps, flows, graphs, paths, branches, functions, APIs, internal
  state, or any other implementation detail — a real caller has no idea any of that exists.

THE PATH IS FIXED. You receive the path as numbered steps. You must NOT add, remove, reorder or
skip past steps, invent other steps, or steer the conversation down a different branch than the
one this path takes. Where a step lists "other branches not taken", the caller's reply at the
step before it must plausibly lead the agent onto THIS path's next step instead. Where the path
returns to an earlier node, the caller's first attempt must plausibly cause that loop, and the
later attempt must let the conversation move on.

EXECUTION ORDER: the agent speaks first at every step, then the caller replies. So each turn is:
- "expected_agent_behavior": what a good agent should do at that step (describe it, don't quote it)
- "caller_line": EXACTLY what the caller says in reply, verbatim — a complete, natural spoken
  sentence with real concrete details (an actual name, an actual date). NEVER a placeholder such
  as [name], {date} or <email>: the line is spoken word for word.

WHAT TO PRODUCE:
{
  "test_goal": "one or two sentences describing what this path verifies",
  "turns": [
    {"step": <step number>, "node_id": "<that step's node id>",
     "expected_agent_behavior": "...", "caller_line": "..."},
    ...
  ]
}

TURN RULES:
- One turn per step where the caller has to reply, in path order; "step" values strictly increase.
- Steps where the caller needs to say nothing (e.g. the agent is only routing or deciding
  internally) get no turn.
- Cover the path from its first step to its end."""

_PLACEHOLDER = re.compile(r"\{[^{}]*\}|\[[^\[\]]*\]|<[^<>]*>")


def _build_path_prompt(
    path: list[str], nodes_by_id: dict[str, dict], out_adj: dict[str, list[str]], category: str
) -> str:
    parts = [f"SCENARIO KIND (structural only): {category}", "", "PATH:"]
    for i, node_id in enumerate(path):
        node = nodes_by_id.get(node_id, {"id": node_id, "name": node_id})
        header = f"Step {i + 1}"
        if node_id in path[:i]:
            header += f" (returns to the node from step {path.index(node_id) + 1})"
        parts.append(f"{header}:\n{_describe_node(node)}")
        if i + 1 < len(path):
            taken = path[i + 1]
            others = [t for t in out_adj.get(node_id, []) if t != taken]
            if others:
                names = ", ".join(str(nodes_by_id.get(t, {}).get("name") or t) for t in others)
                parts.append(f"  other branches not taken from here: {names}")
    parts.append("\nGenerate the test_goal and turns now as json.")
    return "\n".join(parts)


def validate_path_script(result: object, path: list[str]) -> dict:
    """Validate a flow scenario script against its path; return
    {"test_goal": str, "turns": [{"step", "node_id", "expected_agent_behavior",
    "caller_line"}, ...]} normalized. Raises NodeScriptError with every problem found.

    Used both on model output and on a user's edited script before it is saved, so an
    edit can't break path correspondence either.
    """
    if not isinstance(result, dict):
        raise NodeScriptError(["Script must be a JSON object."])

    errors: list[str] = []
    test_goal = result.get("test_goal")
    if not isinstance(test_goal, str) or not test_goal.strip():
        errors.append("test_goal must be a non-empty string.")

    raw_turns = result.get("turns")
    steps: list[tuple[int, str]] = []
    if not isinstance(raw_turns, list) or not raw_turns:
        errors.append("turns must be a non-empty array.")
        raw_turns = []

    last_step = 0
    for i, t in enumerate(raw_turns):
        if not isinstance(t, dict):
            continue  # reported by validate_script below
        step = t.get("step")
        if not isinstance(step, int) or isinstance(step, bool) or not 1 <= step <= len(path):
            errors.append(f"turns[{i}].step must be an integer from 1 to {len(path)} (a step of the path).")
            continue
        if step <= last_step:
            errors.append(
                f"turns[{i}].step {step} must come after step {last_step}: turns follow the path in order."
            )
        last_step = max(last_step, step)
        expected_node = path[step - 1]
        if t.get("node_id") != expected_node:
            errors.append(
                f"turns[{i}].node_id must be '{expected_node}' (the node at step {step}), "
                f"got {t.get('node_id')!r}."
            )
        caller_line = t.get("caller_line")
        if isinstance(caller_line, str) and _PLACEHOLDER.search(caller_line):
            errors.append(
                f"turns[{i}].caller_line contains a placeholder "
                f"('{_PLACEHOLDER.search(caller_line).group(0)}'); write the exact words to speak."
            )
        steps.append((step, expected_node))

    fields: list[dict] = []
    if raw_turns:
        try:
            fields = validate_script(raw_turns, max_turns=len(path))
        except NodeScriptError as e:
            errors.extend(e.errors)

    if errors:
        raise NodeScriptError(errors)

    return {
        "test_goal": test_goal.strip(),
        "turns": [
            {"step": step, "node_id": node_id, **f}
            for (step, node_id), f in zip(steps, fields, strict=True)
        ],
    }


async def generate_path_script(
    path: list[str], nodes_by_id: dict[str, dict], out_adj: dict[str, list[str]], category: str
) -> dict:
    """Generate {"test_goal", "turns"} for one planned path via one LLM call, then
    validate it against that path. `out_adj` (successors per node) is only used to tell
    the model which branches the path does NOT take. Raises NodeScriptError on invalid
    output — no fallback script, same policy as generate_node_script."""
    result = await chat(
        system=PATH_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": _build_path_prompt(path, nodes_by_id, out_adj, category)}],
        json_mode=True,
    )
    return validate_path_script(result, path)


# Test type for an executed flow scenario (Phase E). Its own type rather than
# NODE_TEST_TYPE so the runner can give it the agent-first opening AND play every
# script turn (a flow script may exceed MAX_SCRIPT_TURNS) without changing anything
# for node tests. Like flow_node it is not an adaptive type and has no customer
# context, so the runner plays it scripted and verbatim.
PATH_TEST_TYPE = "flow_path"


def path_script_to_scenario(
    flow_id: int, scenario_id: str, name: str, path: list[str], node_names: dict[str, str],
    test_goal: str, turns: list[dict],
) -> dict:
    """Flatten a saved, validated flow scenario script into the EXISTING scenario shape
    the Temporal workflow / runner / Judge already consume — the path counterpart of
    to_scenario_dict:

      seed_turns        = [turn["caller_line"] ...]   played verbatim, in order
      user_goal         = test_goal
      expected_behavior = the planned path, then each turn's expected_agent_behavior
                          labelled with its step — the Judge's reference; never spoken
      test_type         = PATH_TEST_TYPE, customer_context = "" (scripted, never the
                          dynamic AI Caller)

    `turns` is assumed already validated against `path` (validate_path_script).
    """
    label = lambda n: node_names.get(n) or n  # noqa: E731
    expected = "\n".join(
        [f"Planned path: {' -> '.join(label(n) for n in path)}"]
        + [
            f"{i}. At step {t['step']} ({label(t['node_id'])}): {t['expected_agent_behavior']}"
            for i, t in enumerate(turns, start=1)
        ]
    )
    return {
        "title": f"Flow scenario: {name}",
        "user_goal": test_goal,
        "test_type": PATH_TEST_TYPE,
        "assigned_fault": "none",
        "expected_behavior": expected,
        "seed_turns": [t["caller_line"] for t in turns],
        "customer_context": "",
        "max_turns": None,
        "flow_id": flow_id,
        # A path covers many nodes; node_id stays NULL and the path lives in the script.
        "node_id": None,
        "node_script_json": json.dumps({"scenario_id": scenario_id, "path": path, "turns": turns}),
    }


# ---------------------------------------------------------------------------
# Interrupt scenario scripts — a planned whole-flow path (app.core.flow_scenario_planner)
# with ONE interrupt raised at a planned point. Same generate -> validate -> save cycle
# and the same chat() call as path scripts; everything above is unchanged.
#
# Turns, in conversation order:
#   caller    {"type": "caller", "step", "node_id", "expected_agent_behavior", "caller_line"}
#             the agent acts at an `ask` step, the caller replies verbatim
#   listen    {"type": "listen", "step", "node_id", "expected_agent_behavior"}
#             the agent speaks at a `say`/`end` step and the caller does NOT reply
#   readback  {"type": "readback", "step", "node_id", "expected_agent_behavior"}
#             the agent reads a collected value back; the very next turn must be the
#             caller confirming it at the same step
#   interrupt {"type": "interrupt", "interrupt_key", "caller_line", "expected_agent_behavior"}
#             the caller raises the planned interrupt INSTEAD of answering at the
#             injection step. It is not a node and not an edge.
# `step` is the 1-based position in the planner's flat path (its segments joined), so a
# step before the interrupt is in the prefix and a step after it is in the target /
# resumed segment. Everything planner-controlled (path, interrupt key, injection point,
# target, outcome, end status) is checked, never taken from the model.
# ---------------------------------------------------------------------------
CALLER, LISTEN, READBACK, INTERRUPT = "caller", "listen", "readback", "interrupt"
LISTEN_KINDS = ("say", "end")
CALLER_KIND = "ask"
MAX_BUG_GUARDS = 10

INTERRUPT_SYSTEM_PROMPT = """You are generating a test script for a PRE-PLANNED voice-agent \
scenario. Respond ONLY with a json object.

The supplied path is authoritative. Do not change the path. Do not invent nodes or transitions.
The caller lines are the simulated caller's exact utterances, spoken verbatim later. The real
voice agent's responses are NOT generated by you — "expected_agent_behavior" only describes what
a good agent should do, for a judge; never write a sentence for the agent to say.

The interrupt is authoritative. Do not move it. Do not replace it. Do not add another interrupt.
The interrupt jump is not an ordinary graph edge. The caller utterance must trigger the declared
interrupt: it must contain one of its trigger phrases and none of its "must not" phrases.

TURNS, in conversation order (each turn references a numbered STEP of the path). "type" is exactly
one of caller | listen | readback | interrupt — never a step kind such as ask or say:
- {"type": "caller", "step": N, "node_id": "...", "expected_agent_behavior": "...", "caller_line": "..."}
  only at an ASK step: the agent asks, then the caller replies. One caller turn per ask step you
  cover, in path order.
- {"type": "listen", "step": N, "node_id": "...", "expected_agent_behavior": "..."}
  only at a SAY or END step, when it is worth checking what the agent says there; the caller does
  not reply. Do not add one for every agent utterance. NEVER write a listen turn at an ASK step:
  the agent's question there is described in that step's caller turn "expected_agent_behavior".
- {"type": "readback", "step": N, "node_id": "...", "expected_agent_behavior": "..."}
  only at a step marked "readback allowed", for a value the caller JUST gave: the agent repeats
  that value back. Order at that step: caller answer -> readback -> caller confirming it (a second
  caller turn at the same step). A readback is never the agent asking or re-asking a question.
- {"type": "interrupt", "interrupt_key": "...", "caller_line": "...", "expected_agent_behavior": "..."}
  exactly once, where the path says INTERRUPT. It REPLACES the caller's answer at that step: do
  not also write a caller turn for that step before the interrupt.
BRANCH steps are silent routing: never write a turn for them. Steps never go backwards. After a
resume, the interrupted question is asked again as the next step: answer it with a normal caller
turn.

CALLER LINES: natural, concise, concrete; no placeholders like [name], {dob} or <email>. Each
line must clearly produce the transition the path takes next. Apart from the interrupt line, no
caller line may contain any interrupt trigger phrase listed below. Use supplied record values
exactly; never invent personal details that were not supplied — if the caller must state one and
none was supplied, choose a clearly fictional value and list it in "assumed_values".

WHAT TO PRODUCE:
{
  "test_goal": "one or two sentences: what this scenario verifies",
  "turns": [ ... ],
  "assumed_values": {"field": "value", ...},
  "expectations": {
    "outcome": "what should have happened by the end of the call",
    "bug_guards": ["a specific check a judge can apply, only if the flow facts below support it", ...]
  }
}
Only include bug guards backed by the flow facts given (declared end statuses, fields, attempt
limits, transfers, the interrupt's settings). Omit any you cannot back up."""


def _norm_phrase(text: str) -> str:
    return " ".join(str(text).lower().replace("’", "'").split())


def _phrases(interrupt: dict, field: str) -> list[str]:
    return [p for spec in (interrupt.get("detection") or {}).values() for p in spec.get(field, [])]


def _triggers(line: str, interrupt: dict) -> bool:
    text = _norm_phrase(line)
    return any(_norm_phrase(k) in text for k in _phrases(interrupt, "keywords")) and not any(
        _norm_phrase(n) in text for n in _phrases(interrupt, "negative")
    )


def _allowed_turns(kind: Optional[str], at_injection: bool, readback: bool) -> str:
    if at_injection:
        return "NONE — the interrupt below replaces the caller's answer to this step"
    if kind == CALLER_KIND:
        text = "exactly one caller turn"
        if readback:
            text += ('; optionally the readback sequence instead: caller answer -> readback -> '
                     'caller confirmation (all three at this step)')
        return text
    if kind in LISTEN_KINDS:
        return "an optional listen turn only — never a caller turn"
    return "none — silent routing, write no turn"


def _describe_step(position: int, node_id: str, details: dict, segment: str, readback_steps: set,
                   at_injection: bool = False) -> str:
    d = details.get(node_id, {})
    lines = [f"STEP {position} [{segment}] {node_id} ({d.get('kind') or 'step'})"]
    for key in ("ask", "say"):
        if d.get(key):
            lines.append(f"  agent {key}s: {d[key][:220]}")
    if d.get("field"):
        f = d["field"]
        lines.append(f"  collects field '{f.get('key')}' ({f.get('type')}): {f.get('label')}")
    if d.get("status"):
        lines.append(f"  call status if the call ends here: {d['status']}")
    lines.append(f"  turns allowed here: {_allowed_turns(d.get('kind'), at_injection, node_id in readback_steps)}")
    return "\n".join(lines)


def _build_interrupt_prompt(
    scenario: dict, details: dict, interrupts: list[dict], record: dict,
    readback_steps: set, flow_name: str, edges: list[dict],
) -> str:
    interrupt = scenario["interrupt"]
    edge_info = {}
    for e in edges:
        edge_info.setdefault((str(e["from"]), str(e["to"])), []).append(
            e.get("label") or e.get("type") or "transition"
        )
    parts = [
        f"FLOW: {flow_name}",
        f"SCENARIO: {scenario['name']} (interrupt scenario)",
        f"PLANNED GOAL: verify how the agent handles the '{interrupt['id']}' interrupt "
        f"(outcome: {interrupt['outcome']}) raised at {scenario['placement']['step']}.",
        "",
        "PATH:",
    ]
    position = 0
    injection = len(scenario["segments"][0]["steps"])
    for seg in scenario["segments"]:
        if seg["kind"] == "interrupt":
            desc = [
                f">>> INTERRUPT '{interrupt['id']}' here — the caller says it INSTEAD of answering "
                f"step {position}. Write NO caller turn for step {position}.",
                f"    outcome: {interrupt['outcome']}",
            ]
            if interrupt.get("target"):
                desc.append(f"    the agent then goes to: {interrupt['target']}")
            if interrupt["outcome"] == "resume":
                desc.append("    the agent then returns to the interrupted step and carries on")
            if interrupt["outcome"] == "end":
                desc.append(f"    the call then ends (status: {interrupt.get('end_status') or 'not declared'})")
            if interrupt.get("when"):
                desc.append(f"    applies when: {interrupt['when']} (already satisfied by the steps before it)")
            for field in ("priority", "critical", "max_turns"):
                if field in interrupt:
                    desc.append(f"    {field}: {interrupt[field]}")
            desc.append(f"    trigger phrases (use one): {', '.join(_phrases(interrupt, 'keywords'))}")
            negatives = _phrases(interrupt, "negative")
            if negatives:
                desc.append(f"    must NOT contain: {', '.join(negatives)}")
            parts.extend(desc)
            continue
        for node_id in seg["steps"]:
            position += 1
            parts.append(_describe_step(position, node_id, details, seg["kind"], readback_steps,
                                        at_injection=seg["kind"] == "prefix" and position == injection))
            nxt = scenario["path"][position] if position < len(scenario["path"]) else None
            if nxt and (node_id, nxt) in edge_info and seg["steps"].index(node_id) < len(seg["steps"]) - 1:
                parts.append(f"  -> next: {nxt} (via {', '.join(edge_info[(node_id, nxt)])})")
    others = [i for i in interrupts if i["id"] != interrupt["id"]]
    if others:
        parts.append("\nOTHER INTERRUPTS — no caller line may contain these trigger phrases:")
        for i in others:
            parts.append(f"  {i['id']}: {', '.join(_phrases(i, 'keywords'))}")
    parts.append("\nRECORD VALUES (use exactly):" if record else "\nRECORD VALUES: none supplied.")
    for k, v in record.items():
        parts.append(f"  {k}: {v}")
    parts.append("\nGenerate the test_goal, turns, assumed_values and expectations now as json.")
    return "\n".join(parts)


def validate_interrupt_script(
    result: object, scenario: dict, *, node_kinds: dict[str, str], readback_steps: set,
    interrupts: list[dict], record: dict,
) -> dict:
    """Validate an interrupt scenario script against its planner scenario; return it
    normalized as {"test_goal", "turns", "setup", "expectations"}. Raises
    NodeScriptError listing every problem. Used on model output AND on a user's edit
    before saving; nothing is ever repaired."""
    if not isinstance(result, dict):
        raise NodeScriptError(["Script must be a JSON object."])
    interrupt = scenario["interrupt"]
    path = scenario["path"]
    injection = len(scenario["segments"][0]["steps"])
    errors: list[str] = []

    test_goal = result.get("test_goal")
    if not isinstance(test_goal, str) or not test_goal.strip():
        errors.append("test_goal must be a non-empty string.")

    raw_turns = result.get("turns")
    if not isinstance(raw_turns, list) or not raw_turns:
        errors.append("turns must be a non-empty array.")
        raw_turns = []

    turns: list[dict] = []
    seen_interrupt = False
    last_step = 0
    caller_steps: set[int] = set()
    pending_readback: Optional[int] = None
    previous: Optional[tuple[str, Optional[int]]] = None  # (type, step) of the turn before
    for i, t in enumerate(raw_turns):
        if not isinstance(t, dict):
            errors.append(f"turns[{i}] must be an object.")
            continue
        kind = t.get("type", CALLER)
        behavior = t.get("expected_agent_behavior")
        if not isinstance(behavior, str) or not behavior.strip():
            errors.append(f"turns[{i}].expected_agent_behavior must be a non-empty string.")
        line = t.get("caller_line")

        if pending_readback is not None and not (kind == CALLER and t.get("step") == pending_readback):
            errors.append(f"turns[{i}]: a readback at step {pending_readback} must be followed by the caller confirming it at that step.")
        confirming = pending_readback is not None and kind == CALLER and t.get("step") == pending_readback
        pending_readback = None
        current = previous
        previous = (kind, t.get("step"))

        if kind == INTERRUPT:
            if seen_interrupt:
                errors.append(f"turns[{i}]: only one interrupt is allowed.")
            seen_interrupt = True
            if t.get("interrupt_key") != interrupt["id"]:
                errors.append(f"turns[{i}].interrupt_key must be '{interrupt['id']}', got {t.get('interrupt_key')!r}.")
            for protected in ("step", "node_id"):
                if protected in t:
                    errors.append(f"turns[{i}]: an interrupt is not a step — remove '{protected}'.")
            if "outcome" in t and t["outcome"] != interrupt["outcome"]:
                errors.append(f"turns[{i}].outcome must be '{interrupt['outcome']}'.")
            if "target" in t and t["target"] != interrupt.get("target"):
                errors.append(f"turns[{i}].target must be {interrupt.get('target')!r}.")
            if last_step > injection:
                errors.append(f"turns[{i}]: the interrupt must come at step {injection}, not after step {last_step}.")
            if injection in caller_steps:
                errors.append(f"turns[{i}]: the interrupt replaces the caller's answer at step {injection}; remove that caller turn.")
            if not isinstance(line, str) or not line.strip():
                errors.append(f"turns[{i}].caller_line must be a non-empty string.")
            else:
                if not _triggers(line, interrupt):
                    errors.append(
                        f"turns[{i}].caller_line must contain one of the '{interrupt['id']}' trigger phrases "
                        "and none of its excluded phrases."
                    )
                for other in interrupts:
                    if other["id"] != interrupt["id"] and not other.get("when") and not interrupt.get("when") and _triggers(line, other):
                        errors.append(f"turns[{i}].caller_line would also trigger interrupt '{other['id']}'.")
            turns.append({"type": INTERRUPT, "interrupt_key": interrupt["id"],
                          "caller_line": (line or "").strip(), "expected_agent_behavior": (behavior or "").strip()})
            continue

        if kind not in (CALLER, LISTEN, READBACK):
            errors.append(f"turns[{i}].type must be caller, listen, readback or interrupt, got {kind!r}.")
            continue
        step = t.get("step")
        if not isinstance(step, int) or isinstance(step, bool) or not 1 <= step <= len(path):
            errors.append(f"turns[{i}].step must be an integer from 1 to {len(path)} (a step of the path).")
            continue
        node_id = path[step - 1]
        if t.get("node_id") != node_id:
            errors.append(f"turns[{i}].node_id must be '{node_id}' (the node at step {step}), got {t.get('node_id')!r}.")
        # At an answered step only its readback, then the caller's confirmation, may follow.
        same_step_ok = confirming or (kind == READBACK and current == (CALLER, step))
        if step < last_step or (step == last_step and last_step in caller_steps and not same_step_ok):
            errors.append(f"turns[{i}].step {step} must come after step {last_step}: turns follow the path in order.")
        if not seen_interrupt and step > injection:
            errors.append(f"turns[{i}]: step {step} is after the interrupt point (step {injection}) but comes before the interrupt.")
        if seen_interrupt and step <= injection:
            errors.append(f"turns[{i}]: step {step} is before the interrupt point but comes after the interrupt.")
        if seen_interrupt and interrupt["outcome"] == "end":
            errors.append(f"turns[{i}]: the call ends at the '{interrupt['id']}' interrupt; nothing can follow it.")
        last_step = max(last_step, step)
        node_kind = node_kinds.get(node_id)

        if kind == CALLER:
            if node_kind != CALLER_KIND:
                errors.append(f"turns[{i}]: caller turns belong at ask steps; step {step} ({node_id}) is '{node_kind}'.")
            if not seen_interrupt and step == injection:
                errors.append(f"turns[{i}]: at step {step} the caller raises the interrupt instead of answering.")
            if not isinstance(line, str) or not line.strip():
                errors.append(f"turns[{i}].caller_line must be a non-empty string.")
            else:
                for other in interrupts:
                    if _triggers(line, other):
                        errors.append(f"turns[{i}].caller_line would trigger interrupt '{other['id']}'.")
            caller_steps.add(step)
        else:
            if line not in (None, ""):
                errors.append(f"turns[{i}]: a {kind} turn is agent-only; remove its caller_line.")
            if kind == LISTEN and node_kind not in LISTEN_KINDS:
                errors.append(f"turns[{i}]: listen turns belong at say/end steps; step {step} ({node_id}) is '{node_kind}'.")
            if kind == READBACK:
                if node_id not in readback_steps:
                    errors.append(f"turns[{i}]: step {step} ({node_id}) does not allow a readback.")
                elif current != (CALLER, step):
                    errors.append(f"turns[{i}]: a readback repeats the value the caller just gave, so it must directly follow the caller's answer at step {step}.")
                pending_readback = step
        if isinstance(line, str) and _PLACEHOLDER.search(line):
            errors.append(f"turns[{i}].caller_line contains a placeholder ('{_PLACEHOLDER.search(line).group(0)}').")
        turn = {"type": kind, "step": step, "node_id": node_id, "expected_agent_behavior": (behavior or "").strip()}
        if kind == CALLER:
            turn["caller_line"] = (line or "").strip()
        turns.append(turn)

    if pending_readback is not None:
        errors.append(f"the readback at step {pending_readback} must be followed by the caller confirming it.")
    if not seen_interrupt:
        errors.append(f"the script must contain the '{interrupt['id']}' interrupt.")
    interrupt_line = next((t["caller_line"] for t in turns if t["type"] == INTERRUPT), "")
    if _PLACEHOLDER.search(interrupt_line):
        errors.append("the interrupt caller_line contains a placeholder.")

    assumed = result.get("assumed_values") or {}
    if not isinstance(assumed, dict) or not all(isinstance(k, str) and isinstance(v, str) and v.strip() for k, v in assumed.items()):
        errors.append("assumed_values must map field names to non-empty strings.")
        assumed = {}
    elif record and assumed:
        errors.append("record values were supplied, so assumed_values must be empty.")

    expectations = result.get("expectations")
    outcome, guards = None, []
    if not isinstance(expectations, dict):
        errors.append("expectations must be an object with 'outcome' and 'bug_guards'.")
    else:
        outcome = expectations.get("outcome")
        if not isinstance(outcome, str) or not outcome.strip():
            errors.append("expectations.outcome must be a non-empty string.")
        guards = expectations.get("bug_guards") or []
        if not isinstance(guards, list) or not all(isinstance(g, str) and g.strip() for g in guards):
            errors.append("expectations.bug_guards must be a list of non-empty strings.")
            guards = []
        elif len(guards) > MAX_BUG_GUARDS:
            errors.append(f"expectations.bug_guards has {len(guards)} entries; the maximum is {MAX_BUG_GUARDS}.")
        declared = interrupt.get("end_status")
        if expectations.get("end_status") not in (None, "", declared):
            errors.append(f"expectations.end_status must be {declared!r} (the interrupt's declared status).")

    if errors:
        raise NodeScriptError(errors)

    result_expectations = {"outcome": outcome.strip(), "bug_guards": [g.strip() for g in guards]}
    if interrupt.get("end_status"):
        result_expectations["end_status"] = interrupt["end_status"]
    setup = {"record": dict(record)}
    if assumed:
        setup["assumed_values"] = {k: v.strip() for k, v in assumed.items()}
    return {"test_goal": test_goal.strip(), "turns": turns, "setup": setup, "expectations": result_expectations}


async def generate_interrupt_script(
    scenario: dict, *, details: dict, node_kinds: dict[str, str], readback_steps: set,
    interrupts: list[dict], record: dict, flow_name: str, edges: list[dict],
) -> dict:
    """One LLM call, then validate_interrupt_script. No fallback, no repair."""
    result = await chat(
        system=INTERRUPT_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": _build_interrupt_prompt(
            scenario, details, interrupts, record, readback_steps, flow_name, edges,
        )}],
        json_mode=True,
    )
    return validate_interrupt_script(
        result, scenario, node_kinds=node_kinds, readback_steps=readback_steps,
        interrupts=interrupts, record=record,
    )


# ---------------------------------------------------------------------------
# Running a saved interrupt script — through the SAME scenario shape a path script
# runs as (PATH_TEST_TYPE), so the existing runner plays it unchanged: the agent's
# opening greeting first, every scripted line (no MAX_TESTER_TURNS cap), and a stop
# as soon as the voice session has ended.
# ---------------------------------------------------------------------------
def interrupt_script_to_scenario(
    flow_id: int, scenario_id: str, name: str, scenario: dict, node_names: dict[str, str],
    test_goal: str, turns: list[dict], setup: dict, expectations: dict,
) -> dict:
    """Flatten a saved, validated interrupt script into the existing scenario shape.

      seed_turns        = the caller_line of every caller turn and of the interrupt
                          turn, in script order, spoken verbatim. listen/readback turns
                          send nothing: they describe what the REAL agent should say in
                          its reply, so they are Judge reference only.
      expected_behavior = the planned path, the interrupt, the test record, every turn's
                          expected_agent_behavior, the expected outcome, end status and
                          bug guards — the Judge's reference; never spoken
      node_script_json  = the whole saved script plus the planned interrupt, kept on the
                          run's own scenarios row

    `turns`/`setup`/`expectations` are assumed already validated against `scenario`
    (validate_interrupt_script).
    """
    label = lambda n: node_names.get(n) or n  # noqa: E731
    path = scenario["path"]
    interrupt = scenario["interrupt"]
    injection = len(scenario["segments"][0]["steps"])
    outcome = interrupt["outcome"]
    if outcome == "goto":
        effect = f"the agent should go to {label(interrupt.get('target'))}"
    elif outcome == "end":
        effect = f"the agent should end the call (status: {interrupt.get('end_status') or 'not declared'})"
    else:
        effect = "the agent should handle it, then return to the interrupted step"

    lines = [
        f"Planned path: {' -> '.join(label(n) for n in path)}",
        f"Interrupt: the caller raises '{interrupt['id']}' instead of answering step {injection} "
        f"({label(path[injection - 1])}); {effect}.",
    ]
    record = setup.get("record") or {}
    if record:
        lines.append("Test record: " + "; ".join(f"{k} = {v}" for k, v in record.items()))
    if setup.get("assumed_values"):
        lines.append("Assumed values: " + "; ".join(f"{k} = {v}" for k, v in setup["assumed_values"].items()))
    if setup.get("call"):
        lines.append("Call created with: " + "; ".join(f"{k} = {v}" for k, v in setup["call"].items()))
    lines.append("Script (caller lines are spoken verbatim; listen/readback turns are what the agent should say, with no caller line):")
    for i, t in enumerate(turns, start=1):
        kind = t.get("type", CALLER)
        if kind == INTERRUPT:
            lines.append(f"{i}. INTERRUPT '{t['interrupt_key']}' — caller: \"{t['caller_line']}\" — expected: {t['expected_agent_behavior']}")
        elif kind == CALLER:
            lines.append(f"{i}. At step {t['step']} ({label(t['node_id'])}) — caller: \"{t['caller_line']}\" — expected: {t['expected_agent_behavior']}")
        else:
            lines.append(f"{i}. At step {t['step']} ({label(t['node_id'])}) — agent {kind}, no caller line — expected: {t['expected_agent_behavior']}")
    lines.append(f"Expected outcome: {expectations['outcome']}")
    if expectations.get("end_status"):
        lines.append(f"Expected end status: {expectations['end_status']}")
    if expectations.get("bug_guards"):
        lines.append("Bug guards (must NOT happen):")
        lines.extend(f"- {g}" for g in expectations["bug_guards"])

    return {
        "title": f"Flow scenario: {name}",
        "user_goal": test_goal,
        "test_type": PATH_TEST_TYPE,
        "assigned_fault": "none",
        "expected_behavior": "\n".join(lines),
        "seed_turns": [t["caller_line"] for t in turns if t.get("type", CALLER) in (CALLER, INTERRUPT)],
        "customer_context": "",
        "max_turns": None,
        "flow_id": flow_id,
        "node_id": None,
        # The voice agent's call-creation test data (setup.call), if any; attached to
        # the agent by app.core.activities for its transport. Absent otherwise.
        **({"call_variables": dict(setup["call"])} if setup.get("call") else {}),
        "node_script_json": json.dumps({
            "scenario_id": scenario_id,
            "kind": "interrupt",
            "path": path,
            "interrupt": {
                "key": interrupt["id"], "outcome": outcome, "target": interrupt.get("target"),
                "end_status": interrupt.get("end_status"), "injection_step": injection,
                # Context for the Judge only; the plan already placed the interrupt.
                **({"when": interrupt["when"]} if interrupt.get("when") else {}),
            },
            "test_goal": test_goal,
            "turns": turns,
            "setup": setup,
            "expectations": expectations,
        }),
    }
