"""Tests for app.core.flow_graph — deterministic graph analysis + path planning.

Pure unit tests: no DB, no FastAPI, no LLM. Flows are written in the exact normalized
shape app.core.flow_parser produces ({"nodes": [...], "edges": [...]}).
"""
import pytest

from app.core import flow_graph
from app.core.flow_graph import (
    BRANCH,
    DEFAULT_MAX_SCENARIOS,
    HAPPY_PATH,
    HARD_MAX_SCENARIOS,
    MAX_REVISITS_PER_NODE,
    PRIMARY_PATH,
    RECOVERY,
    RETRY,
    analyze_graph,
    generate_candidate_paths,
)


def _flow(node_ids, edges, **node_overrides):
    """Build a normalized flow. `edges` is a list of (from, to) pairs; `node_overrides`
    maps a node id to extra fields (name/type/purpose) for that node."""
    nodes = []
    for nid in node_ids:
        node = {"id": nid, "name": nid, "type": "", "purpose": "", "expected_inputs": []}
        node.update(node_overrides.get(nid, {}))
        nodes.append(node)
    return {"nodes": nodes, "edges": [{"from": a, "to": b} for a, b in edges]}


def _paths(candidates):
    return [c["path"] for c in candidates]


def _assert_invariants(flow, candidates):
    """Properties every candidate must satisfy for ANY flow."""
    real_edges = {(e["from"], e["to"]) for e in flow["edges"]}
    node_ids = {n["id"] for n in flow["nodes"]}
    for c in candidates:
        path = c["path"]
        assert path, "empty path"
        assert set(path) <= node_ids, "path contains an invented node"
        for a, b in zip(path, path[1:]):
            assert (a, b) in real_edges, f"path uses an invented edge {a}->{b}"
        for n in set(path):
            assert path.count(n) <= 1 + MAX_REVISITS_PER_NODE, f"{n} revisited too often"
        assert c["path_length"] == len(path)
        assert c["covered_edges"] == [[a, b] for a, b in zip(path, path[1:])]
        assert c["contains_retry"] == (len(path) != len(set(path)))
    assert len({tuple(c["path"]) for c in candidates}) == len(candidates), "duplicate path"


# The worked example from the feature spec.
SPEC_FLOW = _flow(
    ["greeting", "ask_name", "validate_name", "retry_name", "ask_dob", "ask_email", "complete"],
    [
        ("greeting", "ask_name"),
        ("ask_name", "validate_name"),
        ("validate_name", "ask_dob"),
        ("validate_name", "retry_name"),
        ("retry_name", "validate_name"),
        ("ask_dob", "ask_email"),
        ("ask_email", "complete"),
    ],
)


# ---------------------------------------------------------------------------
# 1. Simple linear flow
# ---------------------------------------------------------------------------
def test_linear_flow_has_one_root_one_terminal_and_one_primary_path():
    flow = _flow(["a", "b", "c"], [("a", "b"), ("b", "c")])

    facts = analyze_graph(flow)
    assert facts.roots == ["a"]
    assert facts.roots_inferred is False
    assert facts.terminals == ["c"]
    assert facts.branch_nodes == []
    assert facts.back_edges == []
    assert facts.unreachable == []

    candidates = generate_candidate_paths(flow)
    assert _paths(candidates) == [["a", "b", "c"]]
    # "c" says nothing about success, so the path is not claimed as a happy path.
    assert candidates[0]["category"] == PRIMARY_PATH
    _assert_invariants(flow, candidates)


# ---------------------------------------------------------------------------
# 2. Simple branch
# ---------------------------------------------------------------------------
def test_branch_node_is_detected_and_both_outgoing_edges_are_covered():
    flow = _flow(["a", "b", "c", "d"], [("a", "b"), ("b", "c"), ("b", "d")])

    assert analyze_graph(flow).branch_nodes == ["b"]

    candidates = generate_candidate_paths(flow)
    covered = {tuple(e) for c in candidates for e in c["covered_edges"]}
    assert ("b", "c") in covered
    assert ("b", "d") in covered
    assert _paths(candidates) == [["a", "b", "c"], ["a", "b", "d"]]
    assert [c["category"] for c in candidates] == [PRIMARY_PATH, BRANCH]
    assert all(c["branch_node_ids"] == ["b"] for c in candidates)
    _assert_invariants(flow, candidates)


# ---------------------------------------------------------------------------
# 3. Multiple terminals
# ---------------------------------------------------------------------------
def test_every_reachable_terminal_is_represented():
    flow = _flow(
        ["start", "decide", "complete", "transfer", "hangup"],
        [("start", "decide"), ("decide", "complete"), ("decide", "transfer"), ("decide", "hangup")],
    )

    facts = analyze_graph(flow)
    assert facts.terminals == ["complete", "transfer", "hangup"]

    candidates = generate_candidate_paths(flow)
    ended_on = {c["path"][-1] for c in candidates}
    assert ended_on == {"complete", "transfer", "hangup"}
    _assert_invariants(flow, candidates)


def test_terminals_at_different_depths_are_all_represented():
    flow = _flow(
        ["a", "b", "mid", "done", "voicemail"],
        [("a", "mid"), ("a", "b"), ("b", "mid"), ("b", "voicemail"), ("mid", "done")],
    )
    candidates = generate_candidate_paths(flow)
    assert {c["path"][-1] for c in candidates} >= {"done", "voicemail"}
    _assert_invariants(flow, candidates)


# ---------------------------------------------------------------------------
# 4. Retry loop
# ---------------------------------------------------------------------------
def test_retry_loop_is_detected_and_traversal_is_bounded():
    flow = _flow(["a", "b", "c"], [("a", "b"), ("b", "c"), ("c", "b")])

    facts = analyze_graph(flow)
    assert facts.back_edges == [("c", "b")]
    assert facts.back_edge_cycles == [["b", "c", "b"]]
    assert facts.terminals == []  # the loop has no exit

    candidates = generate_candidate_paths(flow)
    # The primary walk goes round the loop until the revisit allowance runs out — it
    # terminates even though no terminal exists.
    assert candidates[0]["category"] == PRIMARY_PATH
    assert candidates[0]["path"] == ["a", "b", "c", "b", "c"]
    # b has no other way out of the loop, so the cycle becomes a bounded RETRY path,
    # not a recovery.
    retry = [c for c in candidates if c["category"] == RETRY]
    assert _paths(retry) == [["a", "b", "c", "b"]]
    assert retry[0]["contains_retry"] is True
    _assert_invariants(flow, candidates)


def test_retry_with_an_exit_becomes_a_recovery_path():
    candidates = generate_candidate_paths(SPEC_FLOW)
    recovery = [c for c in candidates if c["category"] == RECOVERY]
    assert _paths(recovery) == [[
        "greeting", "ask_name", "validate_name", "retry_name",
        "validate_name", "ask_dob", "ask_email", "complete",
    ]]
    assert recovery[0]["covered_terminals"] == ["complete"]
    _assert_invariants(SPEC_FLOW, candidates)


def test_self_loop_is_a_back_edge_and_stays_bounded():
    flow = _flow(["a", "b", "done"], [("a", "b"), ("b", "b"), ("b", "done")])
    facts = analyze_graph(flow)
    assert facts.back_edges == [("b", "b")]
    candidates = generate_candidate_paths(flow)
    assert ["a", "b", "b", "done"] in _paths(candidates)
    _assert_invariants(flow, candidates)


# ---------------------------------------------------------------------------
# 5. Multiple roots
# ---------------------------------------------------------------------------
def test_multiple_roots_are_all_recognized_and_each_gets_a_path():
    flow = _flow(["a", "b", "c"], [("a", "c"), ("b", "c")])

    facts = analyze_graph(flow)
    assert facts.roots == ["a", "b"]
    assert facts.entry_points == ["a", "b"]
    assert facts.roots_inferred is False

    candidates = generate_candidate_paths(flow)
    assert _paths(candidates) == [["a", "c"], ["b", "c"]]
    _assert_invariants(flow, candidates)


# ---------------------------------------------------------------------------
# 6. Unreachable node
# ---------------------------------------------------------------------------
def test_isolated_node_is_unreachable_and_never_appears_in_a_path():
    flow = _flow(["a", "b", "c"], [("a", "b")])

    facts = analyze_graph(flow)
    assert facts.unreachable == ["c"]
    assert facts.reachable == {"a", "b"}
    # c has no incoming edge so it IS structurally a root, but it leads nowhere, so it
    # is not an entry point.
    assert facts.roots == ["a", "c"]
    assert facts.entry_points == ["a"]

    candidates = generate_candidate_paths(flow)
    assert all("c" not in c["path"] for c in candidates)
    _assert_invariants(flow, candidates)


def test_node_reachable_only_from_an_unrooted_cycle_is_unreachable():
    flow = _flow(["a", "b", "x", "y"], [("a", "b"), ("x", "y"), ("y", "x")])
    facts = analyze_graph(flow)
    assert facts.roots == ["a"]
    assert set(facts.unreachable) == {"x", "y"}
    # The cycle is still detected even though no root reaches it.
    assert facts.back_edges == [("y", "x")]
    candidates = generate_candidate_paths(flow)
    assert all(not ({"x", "y"} & set(c["path"])) for c in candidates)


# ---------------------------------------------------------------------------
# 7. Duplicate paths
# ---------------------------------------------------------------------------
def test_candidate_paths_are_never_duplicated():
    # Two branches that immediately reconverge: several coverage goals are satisfiable
    # by the same few paths.
    flow = _flow(
        ["a", "b", "x", "y", "end"],
        [("a", "b"), ("b", "x"), ("b", "y"), ("x", "end"), ("y", "end")],
    )
    candidates = generate_candidate_paths(flow)
    assert _paths(candidates) == [["a", "b", "x", "end"], ["a", "b", "y", "end"]]
    _assert_invariants(flow, candidates)


def test_planner_rejects_an_identical_path_added_twice():
    planner = flow_graph._Planner(analyze_graph(SPEC_FLOW), max_scenarios=10)
    path = ["greeting", "ask_name", "validate_name"]
    assert planner.add(path, BRANCH) is True
    assert planner.add(list(path), RETRY) is False
    assert len(planner.results) == 1


def test_duplicate_edges_do_not_make_a_node_branch():
    flow = {
        "nodes": [{"id": n, "name": n} for n in ("a", "b", "c")],
        "edges": [{"from": "a", "to": "b"}, {"from": "b", "to": "c"}, {"from": "b", "to": "c"}],
    }
    facts = analyze_graph(flow)
    assert facts.branch_nodes == []
    assert facts.out_adj["b"] == ["c"]
    assert _paths(generate_candidate_paths(flow)) == [["a", "b", "c"]]


# ---------------------------------------------------------------------------
# 8. Scenario cap
# ---------------------------------------------------------------------------
def _comb_flow(levels: int) -> dict:
    """root -> b0; each b_i -> x_i | y_i; both -> b_(i+1); b_levels -> end.
    `levels` branch nodes, each with 2 distinct options."""
    ids = ["root"]
    edges = [("root", "b0")]
    for i in range(levels):
        ids += [f"b{i}", f"x{i}", f"y{i}"]
        nxt = f"b{i + 1}" if i + 1 < levels else "end"
        edges += [(f"b{i}", f"x{i}"), (f"b{i}", f"y{i}"), (f"x{i}", nxt), (f"y{i}", nxt)]
    ids.append("end")
    return _flow(ids, edges)


def test_max_scenarios_is_respected_and_keeps_the_highest_priority_candidates():
    flow = _comb_flow(10)  # 1 primary + 10 uncovered branch options = 11 candidates
    uncapped = generate_candidate_paths(flow)
    assert len(uncapped) == 11

    capped = generate_candidate_paths(flow, max_scenarios=4)
    assert len(capped) == 4
    assert capped == uncapped[:4]
    assert capped[0]["category"] == PRIMARY_PATH
    _assert_invariants(flow, capped)


def test_max_scenarios_defaults_to_20_and_is_clamped_to_the_hard_maximum():
    flow = _comb_flow(60)  # 61 candidates available
    assert len(generate_candidate_paths(flow)) == DEFAULT_MAX_SCENARIOS
    assert len(generate_candidate_paths(flow, max_scenarios=1000)) == HARD_MAX_SCENARIOS
    assert len(generate_candidate_paths(flow, max_scenarios=0)) == 1


def test_branch_coverage_does_not_grow_exponentially():
    # 2^12 = 4096 distinct root-to-end paths exist; coverage needs only 13.
    candidates = generate_candidate_paths(_comb_flow(12), max_scenarios=HARD_MAX_SCENARIOS)
    assert len(candidates) == 13
    covered = {tuple(e) for c in candidates for e in c["covered_edges"]}
    for i in range(12):
        assert (f"b{i}", f"x{i}") in covered
        assert (f"b{i}", f"y{i}") in covered


# ---------------------------------------------------------------------------
# 9. No edges
# ---------------------------------------------------------------------------
def test_single_node_without_edges_yields_a_single_node_path():
    flow = _flow(["only"], [])

    facts = analyze_graph(flow)
    assert facts.roots == ["only"]
    assert facts.entry_points == ["only"]
    assert facts.terminals == ["only"]
    assert facts.unreachable == []

    candidates = generate_candidate_paths(flow)
    assert _paths(candidates) == [["only"]]
    assert candidates[0]["path_length"] == 1
    assert candidates[0]["covered_edges"] == []
    assert candidates[0]["covered_terminals"] == ["only"]


def test_empty_flow_yields_nothing():
    facts = analyze_graph({"nodes": [], "edges": []})
    assert facts.roots == [] and facts.terminals == [] and facts.unreachable == []
    assert generate_candidate_paths({"nodes": [], "edges": []}) == []


# ---------------------------------------------------------------------------
# 10. Cyclic graph with no indegree-zero node
# ---------------------------------------------------------------------------
def test_fully_cyclic_graph_falls_back_to_first_node_as_inferred_root():
    flow = _flow(["a", "b"], [("a", "b"), ("b", "a")])

    facts = analyze_graph(flow)
    assert facts.roots == ["a"]
    assert facts.roots_inferred is True
    assert facts.terminals == []
    assert facts.back_edges == [("b", "a")]
    assert facts.unreachable == []

    candidates = generate_candidate_paths(flow)
    assert _paths(candidates) == [["a", "b", "a", "b"], ["a", "b", "a"]]
    assert [c["category"] for c in candidates] == [PRIMARY_PATH, RETRY]
    _assert_invariants(flow, candidates)


def test_inferred_root_follows_file_order_not_node_name():
    flow = _flow(["z", "a"], [("z", "a"), ("a", "z")])
    facts = analyze_graph(flow)
    assert facts.roots == ["z"]
    assert facts.roots_inferred is True


# ---------------------------------------------------------------------------
# Happy path labelling
# ---------------------------------------------------------------------------
def test_spec_example_produces_happy_path_branch_and_recovery():
    candidates = generate_candidate_paths(SPEC_FLOW)
    assert [(c["category"], c["path"]) for c in candidates] == [
        (HAPPY_PATH, ["greeting", "ask_name", "validate_name", "ask_dob", "ask_email", "complete"]),
        (BRANCH, ["greeting", "ask_name", "validate_name", "retry_name"]),
        (RECOVERY, [
            "greeting", "ask_name", "validate_name", "retry_name",
            "validate_name", "ask_dob", "ask_email", "complete",
        ]),
    ]
    _assert_invariants(SPEC_FLOW, candidates)


@pytest.mark.parametrize("field,value", [
    ("name", "Call Complete"),
    ("type", "success"),
    ("purpose", "The caller's request has been resolved."),
])
def test_primary_path_is_happy_only_when_its_terminal_denotes_success(field, value):
    flow = _flow(["a", "end"], [("a", "end")], end={field: value})
    assert generate_candidate_paths(flow)[0]["category"] == HAPPY_PATH


@pytest.mark.parametrize("terminal_meta", [
    {"name": "End"},
    {"name": "Goodbye", "purpose": "Say goodbye and hang up."},
    {"name": "Transfer", "purpose": "Transfer to a human agent."},
])
def test_primary_path_stays_neutral_when_success_is_not_evident(terminal_meta):
    flow = _flow(["a", "end"], [("a", "end")], end=terminal_meta)
    assert generate_candidate_paths(flow)[0]["category"] == PRIMARY_PATH


def test_a_path_that_never_reaches_a_terminal_is_never_happy():
    flow = _flow(
        ["a", "complete_loop"], [("a", "complete_loop"), ("complete_loop", "a")],
    )
    # "complete_loop" contains a success keyword but is not a terminal.
    assert all(c["category"] != HAPPY_PATH for c in generate_candidate_paths(flow))


def test_edge_order_not_node_text_decides_the_primary_path():
    # The first-declared option is followed even though the OTHER option looks like
    # success — node text only labels, it never steers.
    flow = _flow(
        ["a", "b", "failed", "completed"],
        [("a", "b"), ("b", "failed"), ("b", "completed")],
    )
    candidates = generate_candidate_paths(flow)
    assert candidates[0]["path"] == ["a", "b", "failed"]
    assert candidates[0]["category"] == PRIMARY_PATH
    assert candidates[1]["path"] == ["a", "b", "completed"]
    assert candidates[1]["category"] == BRANCH


# ---------------------------------------------------------------------------
# Determinism / robustness
# ---------------------------------------------------------------------------
def test_output_is_deterministic():
    assert generate_candidate_paths(SPEC_FLOW) == generate_candidate_paths(SPEC_FLOW)
    assert generate_candidate_paths(_comb_flow(8)) == generate_candidate_paths(_comb_flow(8))


def test_dangling_edges_are_ignored_rather_than_invented_into_the_graph():
    flow = {
        "nodes": [{"id": "a", "name": "a"}, {"id": "b", "name": "b"}],
        "edges": [{"from": "a", "to": "b"}, {"from": "b", "to": "ghost"}],
    }
    facts = analyze_graph(flow)
    assert facts.out_adj == {"a": ["b"], "b": []}
    assert _paths(generate_candidate_paths(flow)) == [["a", "b"]]


def test_dense_cyclic_graph_terminates_and_respects_all_invariants():
    ids = [f"n{i}" for i in range(8)]
    edges = [(a, b) for a in ids for b in ids if a != b]  # complete digraph
    ids.append("exit")
    edges.append(("n7", "exit"))
    flow = _flow(ids, edges)
    candidates = generate_candidate_paths(flow, max_scenarios=HARD_MAX_SCENARIOS)
    assert 0 < len(candidates) <= HARD_MAX_SCENARIOS
    _assert_invariants(flow, candidates)


def test_coverage_guarantees_hold_on_random_graphs():
    """Property check over many seeded random graphs (cyclic, dense, sparse, with
    isolated nodes): every invariant holds, and whenever the cap isn't binding, every
    reachable branch edge and every reachable terminal is covered."""
    import random

    rng = random.Random(1234)
    for _ in range(500):
        ids = [f"n{i}" for i in range(rng.randint(1, 10))]
        density = rng.choice([0.1, 0.2, 0.35])
        edges = [(a, b) for a in ids for b in ids if rng.random() < density]
        flow = _flow(ids, edges)
        facts = analyze_graph(flow)
        candidates = generate_candidate_paths(flow, max_scenarios=HARD_MAX_SCENARIOS)
        _assert_invariants(flow, candidates)
        if len(candidates) == HARD_MAX_SCENARIOS:
            continue

        covered_edges = {tuple(e) for c in candidates for e in c["covered_edges"]}
        for b in facts.branch_nodes:
            if b in facts.reachable:
                for t in facts.out_adj[b]:
                    assert (b, t) in covered_edges, (edges, b, t)
        covered_terminals = {t for c in candidates for t in c["covered_terminals"]}
        for t in facts.terminals:
            if t in facts.reachable:
                assert t in covered_terminals, (edges, t)


def test_no_llm_or_network_is_used():
    import app.core.flow_graph as mod
    source = open(mod.__file__).read()
    for forbidden in ("app.core.llm", "openai", "httpx", "requests", "import app.db", "from app.db"):
        assert forbidden not in source
