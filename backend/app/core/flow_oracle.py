"""Deterministic branch oracle for flow-path scenarios.

A flow_path scenario's saved script (app.core.node_script.path_script_to_scenario)
plans ONE representative path through the agent's flow graph. But because the caller's
actual spoken line can be reactively rephrased (app.core.runner._reactive_scripted_line)
to fit whatever the real agent just said, a live conversation can genuinely end up on a
DIFFERENT — but still perfectly valid — branch of the SAME flow than the one originally
planned. Judging the agent against the planned path's later steps in that case is a
false failure: those steps were never applicable once a different branch was taken.

CRITICAL: the ACTUAL node the agent is at can only be trusted from what the agent
ITSELF reported, never from the scenario's planned position. A native_ws turn's own
trace carries the field values it has actually collected so far (see
app.core.voice_native_ws._combine_turns's "answers", e.g. {"is_patient": "No"}) whenever
the target agent's own API reports them — the exact same "answer.FIELD" values a
modular-format flow's own branch edges declare in their `when` condition (see
app.core.flow_parser.modular_step_transitions). Nothing here is invented or guessed:
this module only replays the flow's OWN declared conditions against the agent's OWN
reported answers, and stops exactly where that stops being possible.

This module answers, from the agent's OWN parsed implementation (its flow's nodes_json/
edges_json — the exact representation app.core.flow_parser produced and
app.core.node_script/flow_graph already consume elsewhere) — never invented, never
agent-specific:

  - which node the flow's own transitions say is active, given the answers the agent
    has actually reported so far;
  - what labeled branch options (case/default/answer edges, with their `when` condition
    text when the source flow declared one) the flow itself offers from there; and
  - which later planned steps are now structurally UNREACHABLE from there — pure
    forward reachability over the parsed edges' from/to, so it works for any flow
    regardless of source format, even one with no condition metadata at all.

Requires the target to report field answers at all (native_ws only, and only when the
target's own implementation does so). A scenario/transport that never reports answers,
or a flow with no stored edges, simply gets no oracle (branch_oracle returns None) —
callers degrade to whatever context they already had (see app.core.judge).
"""
import json
import re
from typing import Optional

from app.core.flow_graph import analyze_graph
from app.db import get_agent_flow

_WHEN_RE = re.compile(r'^answer\.(\w+)\s*(==|!=)\s*"([^"]*)"$')


def eval_when(when: Optional[str], answers: dict) -> Optional[bool]:
    """Evaluate a simple `answer.FIELD == "VALUE"` / `!=` condition (the only form seen
    in a modular-format flow's case edges) against the agent's own reported answers.

    Returns None — unresolvable, never guessed — for anything that doesn't match this
    exact pattern, or when the referenced field hasn't actually been answered yet.
    """
    if not when:
        return None
    m = _WHEN_RE.match(when.strip())
    if not m:
        return None
    field, op, value = m.groups()
    if field not in answers:
        return None
    actual = str(answers[field])
    return (actual == value) if op == "==" else (actual != value)


def outgoing_options(edges: list[dict], node_id: str) -> list[dict]:
    """Every edge FROM node_id, exactly as the flow's own parser recorded it — `type`/
    `label`/`when` only when the source flow actually declared them (modular step-flow
    format); a generic/LLM-extracted flow's edges carry only `to`. Never invented."""
    return [
        {k: e[k] for k in ("to", "type", "label", "when") if k in e}
        for e in edges if e.get("from") == node_id
    ]


def reachable_from(edges: list[dict], start: str) -> set[str]:
    """Pure structural forward reachability (from/to only — ignores type/label/when),
    so this works for ANY parsed flow, modular or generic."""
    adj: dict[str, list[str]] = {}
    for e in edges:
        frm = e.get("from")
        if frm is not None:
            adj.setdefault(frm, []).append(e.get("to"))
    seen = {start}
    stack = [start]
    while stack:
        cur = stack.pop()
        for nxt in adj.get(cur, []):
            if nxt is not None and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return seen


def _field_keys(nodes: list[dict]) -> dict[str, Optional[str]]:
    """node_id -> the field it collects (node["field"]["key"]), or None — the same
    "save_to: answer.X" convention a modular flow's `when` conditions read from."""
    keys: dict[str, Optional[str]] = {}
    for n in nodes:
        nid = n.get("id")
        if nid is None:
            continue
        field = n.get("field") or {}
        keys[str(nid)] = field.get("key") if isinstance(field, dict) else None
    return keys


def _walk(nodes: list[dict], edges: list[dict], root: str, answers: dict) -> tuple[str, set[str]]:
    """The actual walk (see resolve_active_node) — also returns every node it passed
    through, so a planned step already walked THROUGH isn't later mistaken for one that
    is now unreachable (it's earlier in the chain than the active node, not off on some
    other branch)."""
    field_keys = _field_keys(nodes)
    current = root
    visited = {root}
    while True:
        fk = field_keys.get(current)
        if fk and fk not in answers:
            return current, visited  # an ask node whose own collected field is still unanswered
        outgoing = outgoing_options(edges, current)
        if len(outgoing) == 1:
            nxt = outgoing[0].get("to")
        else:
            cases = [o for o in outgoing if o.get("type") == "case"]
            if not cases:
                return current, visited  # several edges, none condition-based -- can't resolve
            taken = next((o for o in cases if eval_when(o.get("when"), answers) is True), None)
            if taken is None:
                if any(eval_when(o.get("when"), answers) is None for o in cases):
                    return current, visited  # at least one case is still unresolvable -- stop here
                taken = next((o for o in outgoing if o.get("type") == "default"), None)
            if taken is None:
                return current, visited
            nxt = taken.get("to")
        if nxt is None or nxt in visited:
            return current, visited
        visited.add(nxt)
        current = nxt


def resolve_active_node(nodes: list[dict], edges: list[dict], root: str, answers: dict) -> str:
    """Walk the flow from `root`, advancing only where the next step is certain:

    - an `ask` node that collects a field (node["field"]["key"]) never auto-advances
      past itself until that exact field actually appears in `answers` — its single
      outgoing edge ("on_exhaust" handing off to routing) is not proof an answer was
      actually given, only that the ask step exists;
    - otherwise, a single outgoing edge is a purely linear step and is always taken
      (a `say`/`branch` node, or an `ask` with no declared field);
    - a `case`/`default` branch is taken only when the agent's own reported answers
      actually resolve its condition(s).

    Stops at the first node where none of that applies — e.g. an unanswered ask node,
    several outcome-specific edges (like a decline vs. a normal answer) that answers
    alone can't distinguish between, or a case referencing a field not yet answered.
    That stopping point IS the agent's actual current node, as far as its own reported
    state can show.
    """
    return _walk(nodes, edges, root, answers)[0]


def _latest_answers(traces: list[dict]) -> dict:
    """Every "answers" dict the agent's own turns reported, merged in order — a later
    turn's value for a field overrides an earlier one, same as a real conversation
    progressively confirming/updating what it has collected."""
    merged: dict = {}
    for t in traces:
        a = t.get("answers") if isinstance(t, dict) else None
        if isinstance(a, dict):
            merged.update(a)
    return merged


def branch_oracle(flow_id: Optional[int], traces: list[dict], planned_turns: list[dict]) -> Optional[dict]:
    """{"active_node", "options", "unreachable"} resolved from the agent's OWN reported
    answers against its OWN parsed flow — or None when there isn't enough deterministic
    data to say anything (no flow_id, no reported answers at all, the flow has no
    stored edges, or no entry point could be determined).
    """
    answers = _latest_answers(traces)
    if not flow_id or not answers:
        return None
    flow = get_agent_flow(flow_id)
    if not flow:
        return None
    try:
        nodes = json.loads(flow.get("nodes_json") or "[]")
        edges = json.loads(flow.get("edges_json") or "[]")
    except (TypeError, ValueError):
        return None
    if not isinstance(edges, list) or not edges:
        return None
    nodes = nodes if isinstance(nodes, list) else []
    facts = analyze_graph({"nodes": nodes, "edges": edges})
    if not facts.entry_points:
        return None
    active_node, walked = _walk(nodes, edges, facts.entry_points[0], answers)
    # A planned step already WALKED THROUGH is earlier in the chain than the active
    # node, not off on some other branch — never "no longer reachable" just because a
    # purely-forward reachability check from the active node alone can't see backward.
    covered = walked | reachable_from(edges, active_node)
    unreachable = [t for t in planned_turns if t.get("node_id") not in covered]
    return {
        "active_node": active_node,
        "options": outgoing_options(edges, active_node),
        "unreachable": unreachable,
    }
