"""Flow-definition parser for flow-aware / node-based voice testing.

Accepts an uploaded JSON or YAML flow description — in WHATEVER shape the external
voice-agent framework that produced it uses — and normalizes it into AgentShield's
internal representation:

    {"agent_name": str, "nodes": [...], "edges": [...], "source_format": "json"|"yaml"}

The uploaded file is external input; it does not have to match this shape. Two
extraction paths produce it:

  1. DETERMINISTIC (this module, always tried first, free, instant): if the file uses
     AgentShield's own literal `nodes` key, that exact original behavior is preserved
     unchanged. Otherwise `_find_node_candidate()` looks for a plausible node/state/step
     collection under a small set of common aliases (states, steps, screens, intents,
     actions, handlers, routes, branches, subflows, ...), including one level of
     nesting through a small whitelist of wrapper keys (flow, config, data, workflow,
     agent) — bounded, never an open-ended recursive search.
  2. LLM-ASSISTED (app.core.flow_llm_extractor, tried only when step 1 finds nothing
     confident): reuses the existing app.core.llm.chat() infra. Its output is run
     through the exact same node/edge validation as the deterministic path
     (normalize_nodes_tolerant / normalize_explicit_edges below) — never trusted
     blindly.

Structural rules enforced either way: at least one node exists, node ids are unique,
every EXPLICIT edge points at a real node, and every node has an id and a name
something downstream can display/target. Edges are never invented from ordering —
only from explicit transition-like fields (see normalize_explicit_edges /
_extract_inline_edges). No graph engine anywhere in this module.
"""
import json
from typing import Optional

import yaml

REQUIRED_NODE_FIELDS = ("id", "name")

# --- alias tables for the tolerant/generalized extraction path, and for validating
# LLM-extracted output (same rules, either source). ---
ID_ALIASES = ("id", "node_id", "key", "slug", "name")
NAME_ALIASES = ("name", "title", "label", "id")
PURPOSE_ALIASES = ("purpose", "description", "prompt", "instructions", "goal", "objective")
TYPE_ALIASES = ("type", "node_type", "kind")
FROM_ALIASES = ("from", "source")
TO_ALIASES = ("to", "target")
# Single-value "points at the next node" fields, used both inline on a node object and
# as a fallback target field inside an explicit edge/transition object.
INLINE_NEXT_ALIASES = ("next", "goto", "next_state", "next_node")

# Keys that MAY hold a list of node-like objects, checked in this priority order (a
# literal "nodes" match always wins when present — see parse_flow).
NODE_LIST_KEYS = (
    "nodes", "states", "steps", "screens", "flows", "intents",
    "actions", "handlers", "routes", "branches", "subflows",
)
# The ONLY keys _find_node_candidate() will descend into looking for a nested node
# collection — never an arbitrary key, so this cannot run away on an unrelated file.
WRAPPER_KEYS = ("flow", "config", "data", "workflow", "agent")
# How many wrapper levels deep the search goes (e.g. config -> workflow -> states is
# depth 2). Bounded on purpose — "controlled search", not open recursion.
MAX_WRAPPER_DEPTH = 3


class FlowParseError(ValueError):
    """Raised with every validation problem found for a STRUCTURALLY IDENTIFIED but
    malformed flow (duplicate ids, an edge pointing nowhere, etc) — not just the first.
    """

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("; ".join(errors))


class FlowAmbiguousError(ValueError):
    """Raised when no deterministic scan could confidently find a node collection at
    all. Carries `.diagnostics` (detected top-level keys/shapes) so the caller can
    either try LLM-assisted extraction (app.core.flow_llm_extractor) or show the user
    something more useful than a bare "define nodes" message.
    """

    def __init__(self, diagnostics: dict):
        self.diagnostics = diagnostics
        super().__init__("Could not confidently identify conversation nodes in this file.")


def _load_raw(source_text: str, source_format: str | None) -> tuple[object, str]:
    """Return (parsed_object, format_actually_used). Raises FlowParseError on bad syntax."""
    fmt = (source_format or "").lower()
    if fmt == "yml":
        fmt = "yaml"

    if fmt == "json":
        try:
            return json.loads(source_text), "json"
        except json.JSONDecodeError as e:
            raise FlowParseError([f"Invalid JSON: {e}"]) from e

    if fmt == "yaml":
        try:
            return yaml.safe_load(source_text), "yaml"
        except yaml.YAMLError as e:
            raise FlowParseError([f"Invalid YAML: {e}"]) from e

    # No (or unrecognized) format hint: try JSON first, then YAML. Every valid JSON
    # document is also valid YAML, so trying JSON first is what correctly labels plain
    # JSON content as "json" rather than "yaml".
    try:
        return json.loads(source_text), "json"
    except json.JSONDecodeError:
        pass
    try:
        loaded = yaml.safe_load(source_text)
    except yaml.YAMLError as e:
        raise FlowParseError([f"File is neither valid JSON nor valid YAML: {e}"]) from e
    return loaded, "yaml"


def _first_present(d: dict, aliases: tuple[str, ...]):
    """The first non-empty value found under any of `aliases`, or None."""
    for k in aliases:
        v = d.get(k)
        if v not in (None, ""):
            return v
    return None


def _looks_like_node(item: object) -> bool:
    """A dict "looks like" a flow node/state/step if it carries at least one
    id/name/purpose-ish field — enough to rule out an array of unrelated objects
    (e.g. a list of plain strings, or a list of {"email": ...} user records).
    """
    if not isinstance(item, dict):
        return False
    aliases = set(ID_ALIASES) | set(NAME_ALIASES) | set(PURPOSE_ALIASES)
    return any(item.get(k) not in (None, "") for k in aliases)


def _looks_like_node_list(val: object) -> bool:
    return isinstance(val, list) and len(val) > 0 and all(_looks_like_node(x) for x in val)


def _find_node_candidate(obj: object, path: str = "", depth: int = 0):
    """Bounded, whitelist-only search for a plausible node collection.

    Checks every key in NODE_LIST_KEYS at the current level (in priority order) before
    descending into any wrapper; only descends into the fixed WRAPPER_KEYS whitelist,
    never an arbitrary key, and only up to MAX_WRAPPER_DEPTH levels — this cannot run
    away on a large or unrelated file. Returns (source_path, raw_items) or None.
    """
    if not isinstance(obj, dict):
        return None
    for key in NODE_LIST_KEYS:
        val = obj.get(key)
        if _looks_like_node_list(val):
            return (f"{path}{key}", val)
    if depth >= MAX_WRAPPER_DEPTH:
        return None
    for key in WRAPPER_KEYS:
        val = obj.get(key)
        if isinstance(val, dict):
            found = _find_node_candidate(val, f"{path}{key}.", depth + 1)
            if found:
                return found
    return None


def _normalize_nodes_strict(raw_nodes: list) -> tuple[list[dict], list[str]]:
    """EXACT original behavior for the literal top-level 'nodes' key — unchanged, so
    every flow already written in AgentShield's own schema keeps working byte-for-byte.
    """
    errors: list[str] = []
    nodes: list[dict] = []
    seen_ids: set[str] = set()
    for i, n in enumerate(raw_nodes):
        if not isinstance(n, dict):
            errors.append(f"nodes[{i}] must be an object.")
            continue
        missing = [f for f in REQUIRED_NODE_FIELDS if not n.get(f)]
        if missing:
            errors.append(f"nodes[{i}] is missing required field(s): {', '.join(missing)}.")
            continue
        node_id = str(n["id"])
        if node_id in seen_ids:
            errors.append(f"Duplicate node id: '{node_id}'.")
            continue
        seen_ids.add(node_id)
        expected_inputs = n.get("expected_inputs") or []
        if not isinstance(expected_inputs, list):
            errors.append(f"nodes[{i}] ('{node_id}'): expected_inputs must be a list.")
            expected_inputs = []
        nodes.append({
            "id": node_id,
            "name": str(n["name"]),
            "type": str(n.get("type") or ""),
            "purpose": str(n.get("purpose") or ""),
            "expected_inputs": [str(x) for x in expected_inputs],
            "source_path": f"nodes[{i}]",
        })
    return nodes, errors


def normalize_nodes_tolerant(raw_items: list, source_path: str) -> tuple[list[dict], list[str]]:
    """Alias-based node extraction for anything found by _find_node_candidate, and
    reused (public — imported by app.core.flow_llm_extractor) to validate LLM-extracted
    nodes with the exact same rules, so LLM output is never trusted more loosely than a
    deterministic hit.

    Unlike the strict path, a missing id/name is NOT a hard error — it falls back to a
    synthetic positional id / the id itself as the name. That is bookkeeping (so every
    node stays addressable), never invented CONTENT (purpose/type default to "").
    """
    errors: list[str] = []
    nodes: list[dict] = []
    seen_ids: set[str] = set()
    for i, n in enumerate(raw_items):
        if not isinstance(n, dict):
            errors.append(f"{source_path}[{i}] must be an object.")
            continue
        node_id = _first_present(n, ID_ALIASES)
        node_id = str(node_id) if node_id not in (None, "") else f"node_{i}"
        if node_id in seen_ids:
            errors.append(f"Duplicate node id: '{node_id}'.")
            continue
        seen_ids.add(node_id)
        name = _first_present(n, NAME_ALIASES)
        expected_inputs = n.get("expected_inputs") or []
        if not isinstance(expected_inputs, list):
            expected_inputs = []
        nodes.append({
            "id": node_id,
            "name": str(name) if name not in (None, "") else node_id,
            "type": str(_first_present(n, TYPE_ALIASES) or ""),
            "purpose": str(_first_present(n, PURPOSE_ALIASES) or ""),
            "expected_inputs": [str(x) for x in expected_inputs],
            "source_path": f"{source_path}[{i}]",
        })
    return nodes, errors


def _extract_inline_edges(raw_items: list, nodes: list[dict]) -> list[dict]:
    """Per-node inline transition hints (next/goto/next_state/next_node pointing at
    another node's resolved id). Lenient BY DESIGN: a reference that doesn't resolve to
    a known node id is silently dropped rather than failing the whole upload — this is
    an opportunistic signal picked out of an unknown-shaped file, not a user-declared
    edges list. Never invents an edge from ordering; only from an explicit field value.
    """
    if len(raw_items) != len(nodes):
        return []
    ids = {n["id"] for n in nodes}
    edges: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for raw_item, node in zip(raw_items, nodes):
        if not isinstance(raw_item, dict):
            continue
        target = _first_present(raw_item, INLINE_NEXT_ALIASES)
        if target in (None, ""):
            continue
        target = str(target)
        pair = (node["id"], target)
        if target in ids and pair not in seen:
            edges.append({"from": node["id"], "to": target})
            seen.add(pair)
    return edges


def normalize_explicit_edges(raw_edges: list, seen_ids: set[str]) -> tuple[list[dict], list[str]]:
    """Top-level explicit edges/transitions array. STRICT (public — imported by
    app.core.flow_llm_extractor for the same reason as normalize_nodes_tolerant): a
    malformed or dangling reference here is a real error, since this is a user- (or
    model-) declared transitions list, not an opportunistic inline hint.
    """
    errors: list[str] = []
    edges: list[dict] = []
    for i, e in enumerate(raw_edges):
        if not isinstance(e, dict):
            errors.append(f"edges[{i}] must be an object.")
            continue
        frm = _first_present(e, FROM_ALIASES)
        to = _first_present(e, TO_ALIASES) or _first_present(e, INLINE_NEXT_ALIASES)
        if frm in (None, "") or to in (None, ""):
            errors.append(f"edges[{i}] must have both 'from' and 'to'.")
            continue
        frm, to = str(frm), str(to)
        if frm not in seen_ids:
            errors.append(f"edges[{i}] references unknown node id in 'from': '{frm}'.")
            continue
        if to not in seen_ids:
            errors.append(f"edges[{i}] references unknown node id in 'to': '{to}'.")
            continue
        edges.append({"from": frm, "to": to})
    return edges, errors


# ---------------------------------------------------------------------------
# Format adapter: modular step flows (e.g. maya_renewal_modular.yaml).
#
#   entry: greet
#   defaults: {answer: {on_decline: advance, on_exhaust: advance, ...}}
#   steps:
#     - {id: ask_x, kind: ask, next: route_x, answer: {on_decline: "goto:say_y"}}
#     - {id: route_x, kind: branch, cases: [{when: ..., label: ..., goto: z}], default: w}
#
# In this format transitions are declared in several places, not just `next`; the
# generic scanner above only reads `next`. This adapter reads every DECLARED form and
# nothing else:
#
#   next: <step>                     -> type "next"
#   cases[].goto: <step>             -> type "case" (label + when kept as metadata)
#   default: <step>                  -> type "default"
#   answer.on_*: "goto:<step>"       -> type "answer" (label = the on_* event); for an
#                                       `ask` step, events it doesn't set come from the
#                                       explicit flow-level `defaults.answer` block
#
# A target may be `advance`, which this format defines as "the next step in the
# declared list order" — resolved against the parsed step list. A target of `resume`
# depends on call history (return to wherever the call was interrupted), so it never
# becomes an edge: it is kept on the node as `control_transitions`.
#
# Deliberately NOT read: step adjacency where no transition is declared (no implicit
# fall-through), `transfer` blocks, and anything inferred from `when` text, names or
# descriptions. The flow-level `interrupts:` list is normalized separately — never
# as edges — by modular_interrupts() below.
# ---------------------------------------------------------------------------
ADVANCE = "advance"
RESUME = "resume"


def is_modular_step_flow(raw: object) -> bool:
    """A top-level `entry` plus a non-empty `steps` list whose every item has an `id`
    and a `kind` — the shape of this format. Anything else keeps the generic path."""
    if not isinstance(raw, dict) or not isinstance(raw.get("entry"), str):
        return False
    steps = raw.get("steps")
    return (
        isinstance(steps, list) and bool(steps)
        and all(isinstance(s, dict) and s.get("id") and s.get("kind") for s in steps)
    )


def _goto_value(value: object) -> Optional[str]:
    """The target named by a transition value: "goto:X" or {"goto": X} -> X; a bare
    string (as `next`/`default`/`cases[].goto` use) -> itself. None if not a goto."""
    if isinstance(value, dict):
        value = value.get("goto")
    if not isinstance(value, str) or not value.strip():
        return None
    value = value.strip()
    return value[len("goto:"):].strip() if value.startswith("goto:") else value


def modular_step_transitions(
    raw: dict, nodes: list[dict]
) -> tuple[list[dict], dict[str, list[dict]], list[str]]:
    """(edges, control_transitions by node id, errors) for a modular step flow.

    `nodes` are the already-normalized steps, in file order. A declared target that
    names no step is an error (these are declared transitions, same strictness as a
    top-level edges list) — except under `next`, which keeps the generic parser's
    lenient inline-hint behaviour of dropping an unresolvable reference.
    """
    steps = raw["steps"]
    if len(steps) != len(nodes):
        return [], {}, []
    position = {n["id"]: i for i, n in enumerate(nodes)}
    defaults_answer = (raw.get("defaults") or {}).get("answer") or {}
    if not isinstance(defaults_answer, dict):
        defaults_answer = {}

    edges: list[dict] = []
    seen: set[tuple] = set()
    controls: dict[str, list[dict]] = {}
    errors: list[str] = []

    def emit(step_id: str, value: object, meta: dict, where: str, *, lenient: bool = False,
             declared_on_step: bool = True) -> None:
        target = _goto_value(value)
        if target is None:
            if not lenient:
                errors.append(f"step '{step_id}': {where} must name a step, got {value!r}.")
            return
        edge_meta = dict(meta)
        if target == RESUME:
            controls.setdefault(step_id, []).append({**edge_meta, "target": RESUME})
            return
        if target == ADVANCE:
            index = position[step_id] + 1
            if index >= len(nodes):
                # Only a transition declared on the step itself is malformed here; an
                # inherited default simply has nowhere to advance to.
                if declared_on_step:
                    errors.append(
                        f"step '{step_id}': {where} is 'advance', but it is the last step — "
                        "there is no next step to advance to."
                    )
                return
            target = nodes[index]["id"]
            edge_meta["advance"] = True
        elif target not in position:
            if not lenient:
                errors.append(f"step '{step_id}': {where} points to unknown step '{target}'.")
            return
        edge = {"from": step_id, "to": target, **edge_meta}
        key = tuple(sorted(edge.items()))
        if key not in seen:
            seen.add(key)
            edges.append(edge)

    for step, node in zip(steps, nodes):
        step_id = node["id"]

        if "next" in step:
            emit(step_id, step["next"], {"type": "next"}, "next", lenient=True)

        cases = step.get("cases")
        if cases is not None:
            if not isinstance(cases, list):
                errors.append(f"step '{step_id}': cases must be a list.")
                cases = []
            for i, case in enumerate(cases):
                if not isinstance(case, dict) or "goto" not in case:
                    errors.append(f"step '{step_id}': cases[{i}] must be an object with a 'goto'.")
                    continue
                meta = {"type": "case"}
                if case.get("label") not in (None, ""):
                    meta["label"] = str(case["label"])
                if case.get("when") not in (None, ""):
                    meta["when"] = str(case["when"])
                emit(step_id, case["goto"], meta, f"cases[{i}].goto")

        if "default" in step:
            emit(step_id, step["default"], {"type": "default", "label": "default"}, "default")

        answer = step.get("answer")
        if answer is not None and not isinstance(answer, dict):
            errors.append(f"step '{step_id}': answer must be an object.")
            answer = {}
        own = {k: v for k, v in (answer or {}).items() if str(k).startswith("on_")}
        inherited = (
            {k: v for k, v in defaults_answer.items() if str(k).startswith("on_") and k not in own}
            if step.get("kind") == "ask" else {}
        )
        for events, from_defaults in ((own, False), (inherited, True)):
            for event, value in events.items():
                # Only transition-shaped values ("advance", "goto:X", {"goto": X},
                # "resume") are transitions; any other answer setting is not.
                target = _goto_value(value)
                if target is None or (
                    target not in (ADVANCE, RESUME)
                    and not (isinstance(value, dict) or str(value).strip().startswith("goto:"))
                ):
                    continue
                meta = {"type": "answer", "label": str(event)}
                if from_defaults:
                    meta["from_defaults"] = True
                where = f"defaults.answer.{event}" if from_defaults else f"answer.{event}"
                emit(step_id, value, meta, where, declared_on_step=not from_defaults)

    return edges, controls, errors


END = "end"


def modular_interrupts(raw: dict, step_ids: set[str]) -> tuple[list[dict], list[str]]:
    """(interrupts, errors) for a modular step flow's top-level `interrupts:` list.

    Interrupts are events the caller can raise at any point in the call — the file
    never ties them to particular steps — so they are NOT edges. Each is normalized to:

        {"id": <key>, "priority": int, "detection": {lang: {"keywords": [...],
         "negative": [...]}}, "outcome": "goto" | "end" | "resume",
         "target": <step id>   (outcome "goto" only),
         "when": str, "end_status": str, "critical": bool, "max_turns": int  (when declared)}

    `outcome`/`target` come from the entry's `resume` field: "goto:<step>" hands the
    call to that step, "end" ends it (with `end_status`), "resume" returns to wherever
    the call was interrupted — which depends on call history, so no target is guessed.
    `when` is kept verbatim, never evaluated. The agent's `prompt` wording is left in
    the raw source. Nothing about where an interrupt can fire is inferred.
    """
    entries = raw.get("interrupts")
    if not isinstance(entries, list):
        return [], ["interrupts must be a list."]

    interrupts: list[dict] = []
    errors: list[str] = []
    seen: set[str] = set()
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            errors.append(f"interrupts[{i}] must be an object.")
            continue
        key = entry.get("key")
        if not isinstance(key, str) or not key.strip():
            errors.append(f"interrupts[{i}] must have a 'key'.")
            continue
        key = key.strip()
        if key in seen:
            errors.append(f"Duplicate interrupt key: '{key}'.")
            continue
        seen.add(key)

        item: dict = {"id": key}
        if "priority" in entry:
            if isinstance(entry["priority"], bool) or not isinstance(entry["priority"], int):
                errors.append(f"interrupt '{key}': priority must be an integer.")
                continue
            item["priority"] = entry["priority"]

        detection = entry.get("detection")
        if detection is not None:
            normalized_detection = {}
            if not isinstance(detection, dict):
                errors.append(f"interrupt '{key}': detection must be an object keyed by language.")
                continue
            bad = False
            for lang, spec in detection.items():
                if not isinstance(spec, dict):
                    errors.append(f"interrupt '{key}': detection.{lang} must be an object.")
                    bad = True
                    continue
                phrases = {}
                for field in ("keywords", "negative"):
                    if field not in spec:
                        continue
                    values = spec[field]
                    if not isinstance(values, list) or not all(isinstance(v, str) for v in values):
                        errors.append(f"interrupt '{key}': detection.{lang}.{field} must be a list of strings.")
                        bad = True
                        continue
                    phrases[field] = list(values)
                normalized_detection[str(lang)] = phrases
            if bad:
                continue
            item["detection"] = normalized_detection

        resume = entry.get("resume")
        target = _goto_value(resume) if isinstance(resume, str) and resume.strip().startswith("goto:") else None
        if resume == END:
            item["outcome"] = END
        elif resume == RESUME:
            item["outcome"] = RESUME
        elif target:
            if target not in step_ids:
                errors.append(f"interrupt '{key}': resume points to unknown step '{target}'.")
                continue
            item["outcome"] = "goto"
            item["target"] = target
        else:
            errors.append(
                f"interrupt '{key}': resume must be 'end', 'resume' or 'goto:<step>', got {resume!r}."
            )
            continue

        if entry.get("when") not in (None, ""):
            item["when"] = str(entry["when"])
        if entry.get("end_status") not in (None, ""):
            item["end_status"] = str(entry["end_status"])
        if "critical" in entry:
            item["critical"] = bool(entry["critical"])
        if "max_turns" in entry:
            item["max_turns"] = entry["max_turns"]
        interrupts.append(item)

    return interrupts, errors


def _diagnostics(raw: dict) -> dict:
    """What Part-7-style failures show the user instead of a bare "define nodes"
    message: a summary of what IS in the file, so they can see why nothing matched.
    """
    top_level_summary: dict[str, str] = {}
    for k, v in raw.items():
        if isinstance(v, list):
            top_level_summary[k] = f"list[{len(v)}]"
        elif isinstance(v, dict):
            top_level_summary[k] = f"object({', '.join(list(v.keys())[:6])})"
        else:
            top_level_summary[k] = type(v).__name__
    possible_sections = [k for k in raw if k in NODE_LIST_KEYS or k in WRAPPER_KEYS]
    return {
        "top_level_keys": list(raw.keys()),
        "top_level_summary": top_level_summary,
        "possible_sections": possible_sections,
    }


def parse_flow(source_text: str, source_format: Optional[str] = None) -> dict:
    """Parse+validate `source_text` into the normalized flow shape.

    `source_format` ("json" | "yaml" | "yml"), if given, is treated as authoritative
    (e.g. inferred from the uploaded filename's extension). Omit it to auto-detect.

    Raises FlowParseError (with `.errors`) on invalid syntax or a structurally
    identified-but-malformed flow. Raises FlowAmbiguousError (with `.diagnostics`) when
    no node collection could be confidently identified at all — the caller (see
    app.routers.flows) is expected to try app.core.flow_llm_extractor next.
    """
    if not source_text or not source_text.strip():
        raise FlowParseError(["Flow file is empty."])

    raw, detected_format = _load_raw(source_text, source_format)
    if not isinstance(raw, dict):
        raise FlowParseError(["Flow file must contain a JSON/YAML object at the top level."])

    errors: list[str] = []

    agent_block = raw.get("agent")
    agent_name = (
        agent_block.get("name") if isinstance(agent_block, dict) else None
    ) or raw.get("agent_name") or "Unnamed Agent"

    # A literal top-level "nodes" key keeps the EXACT original behavior (including its
    # exact error messages) — every flow already written in AgentShield's own schema is
    # untouched by everything below. Only when it's absent (or not a list at all) do we
    # fall through to the generalized alias/wrapper search.
    explicit_nodes = raw.get("nodes")
    if isinstance(explicit_nodes, list):
        if not explicit_nodes:
            errors.append("Flow must define at least one node under 'nodes'.")
            nodes, raw_node_items = [], []
        else:
            nodes, node_errors = _normalize_nodes_strict(explicit_nodes)
            errors.extend(node_errors)
            raw_node_items = explicit_nodes
    else:
        candidate = _find_node_candidate(raw)
        if candidate is None:
            raise FlowAmbiguousError(_diagnostics(raw))
        source_path, raw_node_items = candidate
        nodes, node_errors = normalize_nodes_tolerant(raw_node_items, source_path)
        errors.extend(node_errors)

    seen_ids = {n["id"] for n in nodes}

    # Explicit top-level transitions (strict) ...
    raw_edges = raw.get("edges")
    if raw_edges is None:
        raw_edges = raw.get("transitions") or []
    if not isinstance(raw_edges, list):
        errors.append("'edges' must be a list if present.")
        raw_edges = []
    explicit_edges, edge_errors = normalize_explicit_edges(raw_edges, seen_ids)
    errors.extend(edge_errors)

    # ... plus per-node transitions: for a modular step flow, every form that format
    # declares (see modular_step_transitions); otherwise the lenient inline hints
    # (next/goto/...), never inferred from ordering — see _extract_inline_edges.
    interrupts = None
    if explicit_nodes is None and is_modular_step_flow(raw) and raw_node_items is raw["steps"]:
        inline_edges, controls, transition_errors = modular_step_transitions(raw, nodes)
        errors.extend(transition_errors)
        for node in nodes:
            if node["id"] in controls:
                node["control_transitions"] = controls[node["id"]]
        # The answer field a step collects (`field.key`, `field.save_to`), kept so a
        # condition such as `status.dob == 'confirmed'` can be tied to the step that
        # collects `dob` without guessing from names. Data only; nothing reads it here.
        for step, node in zip(raw["steps"], nodes):
            field = step.get("field")
            if isinstance(field, dict) and isinstance(field.get("key"), str) and field["key"].strip():
                node["field"] = {"key": field["key"].strip()}
                if isinstance(field.get("save_to"), str):
                    node["field"]["save_to"] = field["save_to"]
        if "interrupts" in raw:
            interrupts, interrupt_errors = modular_interrupts(raw, seen_ids)
            errors.extend(interrupt_errors)
    else:
        inline_edges = _extract_inline_edges(raw_node_items, nodes)
    seen_pairs = {(e["from"], e["to"]) for e in explicit_edges}
    edges = explicit_edges + [e for e in inline_edges if (e["from"], e["to"]) not in seen_pairs]

    if errors:
        raise FlowParseError(errors)

    result = {
        "agent_name": str(agent_name),
        "nodes": nodes,
        "edges": edges,
        "source_format": detected_format,
        "extraction_method": "deterministic",
    }
    # Only a modular step flow that declares `interrupts:` gets this key; every other
    # flow's result is exactly as before.
    if interrupts is not None:
        result["interrupts"] = interrupts
    return result
