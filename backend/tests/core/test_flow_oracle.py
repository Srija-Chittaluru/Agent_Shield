"""Unit tests for app.core.flow_oracle — the deterministic branch oracle derived from
an agent's own parsed flow (nodes/edges) and its own reported field answers.

EDGES mirrors the real shape found in an actual Maya flow's stored edges_json (case/
default/answer edges with `type`/`label`/`when`, an `ask` node handing off to its own
`route_*` branch node via a single unconditional edge) — condensed to a small, fully
controlled graph so reachability is unambiguous:

    start --(answer: on_exhaust, the only edge)--> route_start
    route_start --case(x == "No")--> dead_end                  (terminal)
    route_start --default--------------------------> branch_b
    branch_b --(next, the only edge)-----------------> ask_extra
    ask_extra --answer(on_decline)--> decline_target            (terminal)
    ask_extra --answer(on_exhaust)--> ok_target                 (terminal)
"""
from app.core import flow_oracle as fo

EDGES = [
    {"from": "start", "to": "route_start", "type": "answer", "label": "on_exhaust"},
    {"from": "route_start", "to": "dead_end", "type": "case", "label": "declined", "when": 'answer.x == "No"'},
    {"from": "route_start", "to": "branch_b", "type": "default", "label": "default"},
    {"from": "branch_b", "to": "ask_extra", "type": "next"},
    {"from": "ask_extra", "to": "decline_target", "type": "answer", "label": "on_decline"},
    {"from": "ask_extra", "to": "ok_target", "type": "answer", "label": "on_exhaust"},
]


# ---------------------------------------------------------------------------
# eval_when
# ---------------------------------------------------------------------------
def test_eval_when_matches_equality():
    assert fo.eval_when('answer.x == "No"', {"x": "No"}) is True
    assert fo.eval_when('answer.x == "No"', {"x": "Yes"}) is False


def test_eval_when_matches_inequality():
    assert fo.eval_when('answer.x != "No"', {"x": "Yes"}) is True
    assert fo.eval_when('answer.x != "No"', {"x": "No"}) is False


def test_eval_when_unresolvable_when_field_not_answered():
    assert fo.eval_when('answer.x == "No"', {}) is None


def test_eval_when_unresolvable_for_anything_else():
    assert fo.eval_when(None, {"x": "No"}) is None
    assert fo.eval_when('answer.x == "No" and answer.y == "Yes"', {"x": "No", "y": "Yes"}) is None


# ---------------------------------------------------------------------------
# outgoing_options / reachable_from
# ---------------------------------------------------------------------------
def test_outgoing_options_reads_type_label_when_verbatim():
    opts = fo.outgoing_options(EDGES, "route_start")
    assert opts == [
        {"to": "dead_end", "type": "case", "label": "declined", "when": 'answer.x == "No"'},
        {"to": "branch_b", "type": "default", "label": "default"},
    ]


def test_outgoing_options_on_a_terminal_node_is_empty():
    assert fo.outgoing_options(EDGES, "dead_end") == []


def test_reachable_from_dead_end_is_only_itself():
    assert fo.reachable_from(EDGES, "dead_end") == {"dead_end"}


def test_reachable_from_branch_b_includes_downstream_nodes():
    assert fo.reachable_from(EDGES, "branch_b") == {"branch_b", "ask_extra", "decline_target", "ok_target"}


# ---------------------------------------------------------------------------
# resolve_active_node
# ---------------------------------------------------------------------------
def test_resolves_through_a_single_unconditional_edge():
    # "start" has exactly one outgoing edge regardless of its type label — always taken.
    assert fo.resolve_active_node([], EDGES[:1], "start", {}) == "route_start"


def test_resolves_the_matching_case_branch():
    assert fo.resolve_active_node([], EDGES, "start", {"x": "No"}) == "dead_end"


def test_falls_back_to_default_when_no_case_matches():
    assert fo.resolve_active_node([], EDGES, "start", {"x": "Yes"}) == "ask_extra"  # default -> branch_b -> ask_extra


def test_stops_at_a_case_node_whose_field_is_not_yet_answered():
    assert fo.resolve_active_node([], EDGES, "start", {}) == "route_start"


def test_stops_at_a_node_with_multiple_non_case_outcomes():
    # ask_extra's two "answer" edges (decline vs exhaust) aren't condition-based --
    # answers alone can never resolve which outcome actually fired.
    assert fo.resolve_active_node([], EDGES, "start", {"x": "Yes"}) == "ask_extra"


def test_a_cycle_does_not_loop_forever():
    cyclic = EDGES + [{"from": "dead_end", "to": "route_start", "type": "next"}]
    assert fo.resolve_active_node([], cyclic, "start", {"x": "No"}) == "dead_end"


# ---------------------------------------------------------------------------
# resolve_active_node: an ask node's own collected field gates advancing past it,
# even through an otherwise-single outgoing edge — a single edge only means "this is
# the only structural option", never "an answer was actually given".
# ---------------------------------------------------------------------------
ASK_CHAIN_EDGES = [
    {"from": "ask_dob", "to": "ask_packet", "type": "next"},
    {"from": "ask_packet", "to": "ask_address", "type": "next"},
]
ASK_CHAIN_NODES = [
    {"id": "ask_dob", "type": "ask", "field": {"key": "dob"}},
    {"id": "ask_packet", "type": "ask", "field": {"key": "packet_received"}},
    {"id": "ask_address", "type": "ask", "field": {"key": "address_ok"}},
]


def test_does_not_leapfrog_an_unanswered_ask_node_even_through_a_single_edge():
    # Nothing answered yet -- must stop at the very first ask node, not run to the end
    # of the chain just because every node in it happens to have only one outgoing edge.
    assert fo.resolve_active_node(ASK_CHAIN_NODES, ASK_CHAIN_EDGES, "ask_dob", {}) == "ask_dob"


def test_advances_exactly_as_far_as_the_reported_answers_go():
    answers = {"dob": "1990-01-01"}
    assert fo.resolve_active_node(ASK_CHAIN_NODES, ASK_CHAIN_EDGES, "ask_dob", answers) == "ask_packet"
    answers["packet_received"] = "Yes"
    assert fo.resolve_active_node(ASK_CHAIN_NODES, ASK_CHAIN_EDGES, "ask_dob", answers) == "ask_address"


def test_a_branch_node_with_no_declared_field_is_never_gated():
    # route_start/branch_b/dead_end/ask_extra carry no "field" key in EDGES' fixture
    # nodes -- only a genuine ask node with one is ever gated.
    assert fo.resolve_active_node([], EDGES[:1], "start", {}) == "route_start"


# ---------------------------------------------------------------------------
# branch_oracle (end to end, with a fake get_agent_flow)
# ---------------------------------------------------------------------------
def _turns(*node_ids):
    return [{"step": i + 1, "node_id": n, "expected_agent_behavior": f"does {n}"} for i, n in enumerate(node_ids)]


def _wire_flow(monkeypatch, edges=EDGES, nodes=None):
    nodes = nodes or [{"id": n, "type": "ask"} for n in {e["from"] for e in edges} | {e["to"] for e in edges}]
    import json as _json
    monkeypatch.setattr(
        fo, "get_agent_flow",
        lambda fid: {"id": fid, "nodes_json": _json.dumps(nodes), "edges_json": _json.dumps(edges)},
    )


def test_no_oracle_without_a_flow_id(monkeypatch):
    _wire_flow(monkeypatch)
    assert fo.branch_oracle(None, [{"answers": {"x": "No"}}], _turns("branch_b")) is None


def test_no_oracle_when_nothing_reported_any_answers(monkeypatch):
    _wire_flow(monkeypatch)
    assert fo.branch_oracle(1, [{"interrupt_key": "hold"}], _turns("branch_b")) is None


def test_no_oracle_when_the_flow_has_no_stored_edges(monkeypatch):
    monkeypatch.setattr(fo, "get_agent_flow", lambda fid: {"id": fid, "nodes_json": "[]", "edges_json": "[]"})
    assert fo.branch_oracle(1, [{"answers": {"x": "No"}}], _turns("branch_b")) is None


def test_declined_branch_marks_the_other_branchs_steps_unreachable(monkeypatch):
    _wire_flow(monkeypatch)
    oracle = fo.branch_oracle(1, [{"answers": {"x": "No"}}], _turns("branch_b", "ask_extra"))
    assert oracle["active_node"] == "dead_end"
    assert oracle["options"] == []  # a valid terminal point
    assert [t["node_id"] for t in oracle["unreachable"]] == ["branch_b", "ask_extra"]


def test_continuing_branch_keeps_downstream_steps_as_still_reachable(monkeypatch):
    _wire_flow(monkeypatch)
    oracle = fo.branch_oracle(1, [{"answers": {"x": "Yes"}}], _turns("ok_target"))
    assert oracle["active_node"] == "ask_extra"
    assert {o["to"] for o in oracle["options"]} == {"decline_target", "ok_target"}
    assert oracle["unreachable"] == []  # ok_target is still structurally reachable from ask_extra


def test_later_agent_turns_answers_merge_with_earlier_ones(monkeypatch):
    _wire_flow(monkeypatch)
    # x answered on an earlier turn, y (irrelevant here) answered later -- both merge.
    oracle = fo.branch_oracle(1, [{"answers": {"x": "No"}}, {"answers": {"y": "whatever"}}], _turns())
    assert oracle["active_node"] == "dead_end"
