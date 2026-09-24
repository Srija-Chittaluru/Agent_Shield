"""Interrupt-aware whole-flow scenario planning — deterministic, no LLM, no I/O.

Consumes a parsed flow ({"nodes", "edges", and optionally "interrupts"} — see
app.core.flow_parser) and produces:

  - path_scenarios       app.core.flow_graph's candidates, unchanged
  - base                 ONE of those candidates, selected as the base conversation
                         that interrupt scenarios are built on
  - interrupt_scenarios  one scenario per declared interrupt

Division of labour:
  flow_graph             graph/path coverage (primary path, branches, terminals,
                         retries, cap). Untouched and not re-implemented here.
  flow_scenario_planner  selects an existing planned path as the base, then places each
                         interrupt on it and builds that interrupt's scenario from
                         flow_graph's own candidates.

BASE SELECTION. flow_graph's primary path follows each step's first-declared exit,
which in a flow that lists branch `cases` before its `default` is often an early exit.
The base is instead chosen from flow_graph's candidates that start at the flow's first
entry point and contain no retry, ranked by:
  1. most `ask` steps (caller turns) — the fullest conversation;
  2. fewest departures from a declared normal exit (a `next` or `default` edge). This
     preference applies only where the graph's edges carry that type; `default` is NOT
     assumed to be the happy path (in the Maya flow, route_dob's default is failure);
  3. flow_graph's own order.
The same ranking picks the continuation after a `goto` interrupt.

INTERRUPT PLACEMENT. A flow never says at which step an interrupt fires, so placement is
a PLANNING CONVENTION, recorded as such on every scenario — never a fact about the
agent:
  after_greeting          no `when`: the first caller turn (`ask` step) after the
                          opening step of the base path
  after_field_confirmed   `when: status.<field> == '<value>'`: the first caller turn
                          after the step that collects <field> (its `field.key`)
  before_field_confirmed  `when: status.<field> != '<value>'`: at that collecting step,
                          before the value can have been confirmed
`when` is never evaluated: only which field it names, and whether it is `==` or `!=`.
Any other condition leaves the interrupt unplaced (plannable False, with a reason).

OUTCOMES (from the interrupt's declared `resume`):
  goto    prefix -> interrupt -> target -> continuation along the graph from the
          target. If that continuation ends at a step whose only way on is a declared
          `resume`, the call resumes at the interrupted step and follows the base on.
  end     prefix -> interrupt -> call ends with the interrupt's end_status
  resume  prefix -> interrupt -> the interrupted step again -> the rest of the base

Every step-to-step move inside a segment is a real edge; the jump INTO the interrupt's
target (or back to the interrupted step) is the interrupt itself, never an edge.
"""
import re
from typing import Optional

from app.core.flow_graph import (
    DEFAULT_MAX_SCENARIOS,
    HARD_MAX_SCENARIOS,
    analyze_graph,
    generate_candidate_paths,
)

NORMAL_EXIT_TYPES = ("next", "default")
CALLER_TURN_TYPE = "ask"

AFTER_GREETING = "after_greeting"
AFTER_FIELD_CONFIRMED = "after_field_confirmed"
BEFORE_FIELD_CONFIRMED = "before_field_confirmed"

_STATUS_CONDITION = re.compile(r"^\s*status\.([A-Za-z_][A-Za-z0-9_]*)\s*(==|!=)\s*'[^']*'\s*$")


def plan_flow_scenarios(flow: dict, max_scenarios: int = DEFAULT_MAX_SCENARIOS) -> dict:
    """{"path_scenarios", "base", "interrupt_scenarios"} for a parsed flow.

    `max_scenarios` caps path_scenarios exactly as flow_graph does. Interrupt scenarios
    are one per declared interrupt (so bounded by the flow itself), also capped at
    HARD_MAX_SCENARIOS. The base is chosen from the full candidate set so it does not
    depend on the display cap.
    """
    path_scenarios = generate_candidate_paths(flow, max_scenarios)
    all_candidates = generate_candidate_paths(flow, HARD_MAX_SCENARIOS)
    ctx = _Context(flow, all_candidates)

    base = ctx.select_base()
    interrupt_scenarios = []
    if base is not None:
        for interrupt in (flow.get("interrupts") or [])[:HARD_MAX_SCENARIOS]:
            interrupt_scenarios.append(ctx.interrupt_scenario(interrupt, base))

    return {
        "path_scenarios": path_scenarios,
        "base": (
            {"candidate_index": base[0], "path": list(base[1]["path"]), "category": base[1]["category"]}
            if base is not None else None
        ),
        "interrupt_scenarios": interrupt_scenarios,
    }


class _Context:
    def __init__(self, flow: dict, candidates: list[dict]):
        self.candidates = candidates
        self.facts = analyze_graph(flow)
        self.nodes = {str(n["id"]): n for n in flow.get("nodes") or []}
        self.edge_pairs = {(str(e["from"]), str(e["to"])) for e in flow.get("edges") or []}
        self.normal_exits: dict[str, set[str]] = {}
        for e in flow.get("edges") or []:
            if e.get("type") in NORMAL_EXIT_TYPES:
                self.normal_exits.setdefault(str(e["from"]), set()).add(str(e["to"]))

    # --- ranking -----------------------------------------------------------------
    def _is_caller_turn(self, node_id: str) -> bool:
        return self.nodes.get(node_id, {}).get("type") == CALLER_TURN_TYPE

    def _rank(self, index: int, path: list[str]) -> tuple:
        asks = sum(1 for n in path if self._is_caller_turn(n))
        departures = sum(
            1 for a, b in zip(path, path[1:]) if a in self.normal_exits and b not in self.normal_exits[a]
        )
        return (-asks, departures, index)

    def _best(self, options: list[tuple[int, list[str]]]) -> Optional[tuple[int, list[str]]]:
        return min(options, key=lambda o: self._rank(*o)) if options else None

    def select_base(self) -> Optional[tuple[int, dict]]:
        if not self.facts.entry_points or not self.candidates:
            return None
        entry = self.facts.entry_points[0]
        eligible = [
            (i, c["path"]) for i, c in enumerate(self.candidates)
            if c["path"][0] == entry and not c["contains_retry"]
        ]
        best = self._best(eligible)
        if best is None:
            return 0, self.candidates[0]
        return best[0], self.candidates[best[0]]

    # --- placement ---------------------------------------------------------------
    def _first_caller_turn(self, path: list[str], start: int) -> Optional[int]:
        return next((i for i in range(start, len(path)) if self._is_caller_turn(path[i])), None)

    def _placement(self, interrupt: dict, base_path: list[str]) -> dict:
        when = interrupt.get("when")
        if not when:
            index = self._first_caller_turn(base_path, 1)
            if index is None:
                return {"policy": AFTER_GREETING, "unplaced": "the base path has no caller turn after its opening step"}
            return {"policy": AFTER_GREETING, "index": index}

        match = _STATUS_CONDITION.match(when)
        if not match:
            return {"policy": None, "unplaced": f"condition {when!r} is not tied to a collected field"}
        field, op = match.group(1), match.group(2)
        policy = AFTER_FIELD_CONFIRMED if op == "==" else BEFORE_FIELD_CONFIRMED
        collector = next(
            (i for i, n in enumerate(base_path) if self.nodes.get(n, {}).get("field", {}).get("key") == field), None
        )
        if collector is None:
            return {"policy": policy, "field": field,
                    "unplaced": f"no step on the base path collects field '{field}'"}
        if policy == BEFORE_FIELD_CONFIRMED:
            return {"policy": policy, "field": field, "field_step": base_path[collector], "index": collector}
        index = self._first_caller_turn(base_path, collector + 1)
        if index is None:
            return {"policy": policy, "field": field, "field_step": base_path[collector],
                    "unplaced": f"no caller turn after '{base_path[collector]}' on the base path"}
        return {"policy": policy, "field": field, "field_step": base_path[collector], "index": index}

    # --- continuation after a goto target ----------------------------------------
    def _continuations_from(self, target: str) -> list[list[str]]:
        """flow_graph's own candidates that pass through `target`, from `target` on —
        ranked best first. Just [target] if no candidate reaches it."""
        options = []
        seen = set()
        for i, c in enumerate(self.candidates):
            if c["contains_retry"] or target not in c["path"]:
                continue
            tail = c["path"][c["path"].index(target):]
            if tuple(tail) not in seen:
                seen.add(tuple(tail))
                options.append((i, tail))
        if not options:
            return [[target]]
        return [p for _, p in sorted(options, key=lambda o: self._rank(*o))]

    def _resumes(self, node_id: str) -> bool:
        """The step's only way on is a declared `resume` (no outgoing edge)."""
        node = self.nodes.get(node_id, {})
        declares_resume = any(t.get("target") == "resume" for t in node.get("control_transitions") or [])
        return declares_resume and not self.facts.out_adj.get(node_id)

    # --- scenario ----------------------------------------------------------------
    def interrupt_scenario(self, interrupt: dict, base: tuple[int, dict]) -> dict:
        base_path = base[1]["path"]
        placement = self._placement(interrupt, base_path)
        scenario = {
            "key": f"interrupt:{interrupt['id']}",
            "category": "interrupt",
            "interrupt": dict(interrupt),
            "placement": {
                **{k: v for k, v in placement.items() if k not in ("index", "unplaced")},
                "convention": True,  # a planning choice, not something the flow declares
            },
            "base_candidate_index": base[0],
        }
        if "unplaced" in placement:
            scenario.update({"plannable": False, "reason": placement["unplaced"], "segments": [], "path": []})
            return scenario

        k = placement["index"]
        scenario["placement"]["step"] = base_path[k]
        prefix = base_path[:k + 1]
        outcome = interrupt["outcome"]
        segments: list[dict] = [
            {"kind": "prefix", "steps": prefix},
            {"kind": "interrupt", "interrupt_id": interrupt["id"], "outcome": outcome,
             **({"target": interrupt["target"]} if outcome == "goto" else {}),
             **({"end_status": interrupt["end_status"]} if interrupt.get("end_status") else {})},
        ]
        if outcome == "goto":
            options = self._continuations_from(interrupt["target"])
            chosen = options[0]
            segments.append({"kind": "target", "steps": chosen})
            scenario["continuation_options"] = len(options)
            if self._resumes(chosen[-1]):
                segments.append({"kind": "resumed", "steps": base_path[k:], "via": chosen[-1]})
        elif outcome == "resume":
            segments.append({"kind": "resumed", "steps": base_path[k:]})
        # outcome "end": the call ends at the interrupt.

        step_segments = [s["steps"] for s in segments if "steps" in s]
        path = [n for steps in step_segments for n in steps]
        covered = [[a, b] for steps in step_segments for a, b in zip(steps, steps[1:])]
        scenario.update({
            "plannable": True,
            "segments": segments,
            "path": path,
            "path_length": len(path),
            "covered_edges": covered,
            "all_moves_are_edges": all((a, b) in self.edge_pairs for a, b in covered),
        })
        return scenario
