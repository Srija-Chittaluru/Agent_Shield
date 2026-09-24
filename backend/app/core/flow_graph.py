"""Deterministic flow-graph analysis + coverage-oriented path planning.

Phase A of flow/path-based voice testing. Takes the ALREADY-NORMALIZED flow shape
app.core.flow_parser produces ({"nodes": [...], "edges": [...]}) and answers, with no
LLM, no I/O and no randomness:

  analyze_graph()            roots, terminals, branch nodes, back-edges (cycles),
                             reachable / unreachable nodes.
  generate_candidate_paths() a bounded, deduplicated, priority-ordered list of
                             meaningful root-to-end paths to turn into test scenarios.

Output depends ONLY on nodes + edges (and their order in the file), so the same flow
always yields the same candidates — the planner is predictable and unit-testable. A
later phase hands each candidate path to an LLM to write the conversation script;
nothing here writes any caller or agent text.

NO EDGE INVENTION. Edges carry no condition/label today (flow_parser only keeps
from/to), and this module never adds one: an outgoing edge is just "an alternative
branch". The only place node TEXT is read at all is `_looks_successful()`, used solely
to decide whether a primary path's terminal may be LABELLED "happy_path" rather than
the neutral "primary_path" — it never influences which edges a path takes.

NOT FULL ENUMERATION. Listing every root-to-terminal path is exponential in the number
of branch nodes. Instead paths are built to satisfy coverage goals, in priority order:

  1. primary path per entry point  — first-declared edge at every branch
  2. branch coverage               — every distinct outgoing edge of every branch node,
                                     continued to its nearest terminal
  3. terminal coverage             — every reachable terminal not yet ended on. A
                                     guarantee rather than a usual source of paths:
                                     step 2's continuations already reach every
                                     terminal in practice, so this rarely adds one.
  4. retry / recovery              — one bounded path per back-edge (cycle)

Each goal skips what earlier paths already cover, so the candidate count is bounded by
roughly (entry points + branch edges + terminals + back-edges), never by path
combinations. The list is then capped at `max_scenarios`; because it is built in
priority order, capping always drops the lowest-priority candidates first.

LOOPS. A node may appear at most 1 + MAX_REVISITS_PER_NODE times in one path (twice,
by default) — enough for "fail once, retry, succeed" and never unbounded. Every path is
additionally capped at `_max_path_length()` nodes, so no traversal can run forever.
"""
from collections import deque
from dataclasses import dataclass
from typing import Optional

MAX_REVISITS_PER_NODE = 1
DEFAULT_MAX_SCENARIOS = 20
HARD_MAX_SCENARIOS = 50

# Categories a candidate can carry. Informational only — nothing downstream branches
# on them yet.
HAPPY_PATH = "happy_path"
PRIMARY_PATH = "primary_path"
BRANCH = "branch"
TERMINAL = "terminal"
RETRY = "retry"
RECOVERY = "recovery"

# Read ONLY to label a primary path whose terminal clearly denotes success. Deliberately
# short and unambiguous: a terminal named e.g. "end" or "goodbye" stays "primary_path",
# because those don't say whether the call SUCCEEDED.
SUCCESS_KEYWORDS = (
    "complete", "completed", "success", "successful", "succeeded",
    "confirmed", "resolved", "finished",
)


@dataclass
class GraphFacts:
    """Structural facts about one flow. All lists are in the flow file's own order."""

    node_ids: list[str]
    # Distinct successors per node, in first-declared edge order (a duplicated edge is
    # counted once — it doesn't make a node branch).
    out_adj: dict[str, list[str]]
    in_degree: dict[str, int]
    # Every indegree-0 node. If none exists (every node sits on a cycle), the first
    # node in file order is used instead and `roots_inferred` is True.
    roots: list[str]
    roots_inferred: bool
    # The roots paths actually start from: those with at least one outgoing edge, so an
    # isolated node doesn't count as a flow entry. Falls back to `roots` when no root
    # has an outgoing edge (e.g. a single-node flow).
    entry_points: list[str]
    terminals: list[str]
    branch_nodes: list[str]
    # (from, to) edges whose target is an ancestor on the DFS stack — each closes a cycle.
    back_edges: list[tuple[str, str]]
    # For back_edges[i] = (u, v): the cycle walked as [v, ..., u, v].
    back_edge_cycles: list[list[str]]
    reachable: set[str]
    unreachable: list[str]


def analyze_graph(flow: dict) -> GraphFacts:
    """Compute GraphFacts for a normalized flow ({"nodes": [...], "edges": [...]})."""
    nodes = flow.get("nodes") or []
    edges = flow.get("edges") or []
    node_ids = [str(n["id"]) for n in nodes]
    id_set = set(node_ids)

    out_adj: dict[str, list[str]] = {nid: [] for nid in node_ids}
    in_degree: dict[str, int] = {nid: 0 for nid in node_ids}
    for e in edges:
        frm, to = str(e["from"]), str(e["to"])
        # flow_parser already rejects dangling edges; skip rather than crash if a caller
        # hands us something it didn't validate.
        if frm not in id_set or to not in id_set or to in out_adj[frm]:
            continue
        out_adj[frm].append(to)
        in_degree[to] += 1

    roots = [nid for nid in node_ids if in_degree[nid] == 0]
    roots_inferred = False
    if not roots and node_ids:
        roots = [node_ids[0]]
        roots_inferred = True

    entry_points = [r for r in roots if out_adj[r]] or list(roots)
    terminals = [nid for nid in node_ids if not out_adj[nid]]
    branch_nodes = [nid for nid in node_ids if len(out_adj[nid]) > 1]
    back_edges, back_edge_cycles = _find_back_edges(node_ids, out_adj)
    reachable = _reachable_from(entry_points, out_adj)
    unreachable = [nid for nid in node_ids if nid not in reachable]

    return GraphFacts(
        node_ids=node_ids, out_adj=out_adj, in_degree=in_degree,
        roots=roots, roots_inferred=roots_inferred, entry_points=entry_points,
        terminals=terminals, branch_nodes=branch_nodes,
        back_edges=back_edges, back_edge_cycles=back_edge_cycles,
        reachable=reachable, unreachable=unreachable,
    )


def generate_candidate_paths(flow: dict, max_scenarios: int = DEFAULT_MAX_SCENARIOS) -> list[dict]:
    """Bounded, deduplicated, coverage-ordered candidate paths for `flow`.

    Each candidate is a plain dict:
        {"category": str, "path": [node_id, ...],
         "covered_edges": [[from, to], ...], "covered_terminals": [node_id, ...],
         "contains_retry": bool, "branch_node_ids": [node_id, ...], "path_length": int}

    `max_scenarios` is clamped to [1, HARD_MAX_SCENARIOS].
    """
    max_scenarios = max(1, min(HARD_MAX_SCENARIOS, int(max_scenarios)))
    facts = analyze_graph(flow)
    node_by_id = {str(n["id"]): n for n in (flow.get("nodes") or [])}
    planner = _Planner(facts, max_scenarios)

    # 1) One primary path per entry point: always follow the first-declared edge.
    for root in facts.entry_points:
        if planner.full:
            break
        walk = _greedy_walk(root, facts.out_adj, planner.max_len)
        end = walk[-1]
        category = (
            HAPPY_PATH
            if end in planner.terminal_set and _looks_successful(node_by_id.get(end, {}))
            else PRIMARY_PATH
        )
        planner.add(walk, category)

    # 2) Branch coverage: every distinct outgoing edge of every reachable branch node.
    for bnode in facts.branch_nodes:
        if bnode not in facts.reachable:
            continue
        for target in facts.out_adj[bnode]:
            if planner.full:
                break
            if (bnode, target) in planner.covered_edges:
                continue
            path = _path_through_edge(planner, bnode, target)
            if path:
                planner.add(path, BRANCH)

    # 3) Terminal coverage: every reachable terminal not yet ended on by some path.
    for term in facts.terminals:
        if planner.full:
            break
        if term in planner.covered_terminals or term not in facts.reachable:
            continue
        root = planner.root_reaching(term)
        path = _shortest_path(root, term, facts.out_adj) if root is not None else None
        if path:
            planner.add(path, TERMINAL)

    # 4) Retry / recovery: walk each cycle once, then — if the loop's entry node has
    # another way out — take it and continue to the nearest terminal.
    for (_, loop_entry), cycle in zip(facts.back_edges, facts.back_edge_cycles):
        if planner.full:
            break
        if loop_entry not in facts.reachable:
            continue
        root = planner.root_reaching(loop_entry)
        lead_in = _shortest_path(root, loop_entry, facts.out_adj) if root is not None else None
        if not lead_in:
            continue
        retry_path = lead_in + cycle[1:]

        # The continuation after the retry must move FORWARD: it may not re-enter any
        # node the path already visited, or it would just be looping again.
        escape = next((t for t in facts.out_adj[loop_entry] if t != cycle[1]), None)
        tail = (
            _bfs(escape, lambda n: n in planner.terminal_set, facts.out_adj, blocked=set(retry_path))
            if escape is not None and escape not in retry_path
            else None
        )
        if tail and planner.add(retry_path + tail, RECOVERY):
            continue
        planner.add(retry_path, RETRY)

    return planner.results


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------
class _Planner:
    """Accumulates candidates for one generate_candidate_paths() call: enforces the
    revisit/length caps, drops duplicate paths, tracks what's already covered, and
    stops accepting once `max_scenarios` is reached.
    """

    def __init__(self, facts: GraphFacts, max_scenarios: int):
        self.facts = facts
        self.max_scenarios = max_scenarios
        self.max_len = _max_path_length(len(facts.node_ids))
        self.terminal_set = set(facts.terminals)
        self.branch_set = set(facts.branch_nodes)
        self.results: list[dict] = []
        self.seen_paths: set[tuple[str, ...]] = set()
        self.covered_edges: set[tuple[str, str]] = set()
        self.covered_terminals: set[str] = set()
        self._reach_by_root = {r: _reachable_from([r], facts.out_adj) for r in facts.entry_points}

    @property
    def full(self) -> bool:
        return len(self.results) >= self.max_scenarios

    def root_reaching(self, node_id: str) -> Optional[str]:
        """The first entry point (in file order) from which `node_id` is reachable."""
        return next((r for r in self.facts.entry_points if node_id in self._reach_by_root[r]), None)

    def add(self, raw_path: list[str], category: str) -> bool:
        """Cap, dedupe and record one candidate. Returns True iff it was added."""
        if self.full:
            return False
        path = _cap_path(raw_path, self.max_len)
        if not path:
            return False
        key = tuple(path)
        if key in self.seen_paths:
            return False
        self.seen_paths.add(key)
        edges = list(zip(path, path[1:]))
        self.covered_edges.update(edges)
        terminals = [n for n in path if n in self.terminal_set]
        self.covered_terminals.update(terminals)
        self.results.append({
            "category": category,
            "path": path,
            "covered_edges": [[a, b] for a, b in edges],
            "covered_terminals": terminals,
            "contains_retry": len(path) != len(set(path)),
            "branch_node_ids": list(dict.fromkeys(n for n in path if n in self.branch_set)),
            "path_length": len(path),
        })
        return True


def _max_path_length(n_nodes: int) -> int:
    """Hard ceiling on nodes in any one path: every node visited the maximum number of
    times allowed, plus slack. Bounds even a pathological graph."""
    return (1 + MAX_REVISITS_PER_NODE) * max(n_nodes, 1) + 2


def _cap_path(path: list[str], max_len: int) -> list[str]:
    """Truncate `path` just before any node would exceed its revisit allowance, or at
    `max_len` — whichever comes first. The universal safety net every candidate passes
    through, however it was built."""
    visits: dict[str, int] = {}
    capped: list[str] = []
    for n in path:
        visits[n] = visits.get(n, 0) + 1
        if visits[n] > 1 + MAX_REVISITS_PER_NODE or len(capped) >= max_len:
            break
        capped.append(n)
    return capped


def _greedy_walk(start: str, out_adj: dict[str, list[str]], max_len: int) -> list[str]:
    """From `start`, repeatedly take the first-declared successor that hasn't used up
    its revisit allowance. Stops at a terminal, when every successor is exhausted, or
    at `max_len` — so it always terminates, even on a pure cycle."""
    path = [start]
    visits = {start: 1}
    current = start
    while len(path) < max_len:
        nxt = next((v for v in out_adj[current] if visits.get(v, 0) < 1 + MAX_REVISITS_PER_NODE), None)
        if nxt is None:
            break
        path.append(nxt)
        visits[nxt] = visits.get(nxt, 0) + 1
        current = nxt
    return path


def _path_through_edge(planner: _Planner, u: str, v: str) -> Optional[list[str]]:
    """Shortest entry-point -> u, then the edge u -> v, then the shortest continuation
    from v to a terminal that does NOT re-enter the lead-in. If none exists (v leads
    back into the path, e.g. a retry branch), the path stops at v — covering the branch
    itself; the loop is exercised separately as a retry/recovery candidate."""
    root = planner.root_reaching(u)
    if root is None:
        return None
    lead_in = _shortest_path(root, u, planner.facts.out_adj)
    if lead_in is None:
        return None
    tail = None
    if v not in lead_in:
        tail = _bfs(v, lambda n: n in planner.terminal_set, planner.facts.out_adj, blocked=set(lead_in))
    return lead_in + (tail or [v])


def _bfs(
    start: str, is_goal, out_adj: dict[str, list[str]], blocked: frozenset | set = frozenset()
) -> Optional[list[str]]:
    """Breadth-first search from `start` to the first node satisfying `is_goal`,
    exploring successors in declared order (so ties resolve deterministically) and
    never entering a `blocked` node. Returns the node path, or None if no goal is
    reachable. BFS paths never repeat a node."""
    if is_goal(start):
        return [start]
    prev: dict[str, Optional[str]] = {start: None}
    queue = deque([start])
    while queue:
        u = queue.popleft()
        for v in out_adj[u]:
            if v in prev or v in blocked:
                continue
            prev[v] = u
            if is_goal(v):
                path = [v]
                while prev[path[-1]] is not None:
                    path.append(prev[path[-1]])
                return path[::-1]
            queue.append(v)
    return None


def _shortest_path(start: str, goal: str, out_adj: dict[str, list[str]]) -> Optional[list[str]]:
    return _bfs(start, lambda n: n == goal, out_adj)


def _reachable_from(starts: list[str], out_adj: dict[str, list[str]]) -> set[str]:
    seen = set(starts)
    queue = deque(starts)
    while queue:
        u = queue.popleft()
        for v in out_adj[u]:
            if v not in seen:
                seen.add(v)
                queue.append(v)
    return seen


def _find_back_edges(
    node_ids: list[str], out_adj: dict[str, list[str]]
) -> tuple[list[tuple[str, str]], list[list[str]]]:
    """Classic white/gray/black DFS edge classification, iterative (no recursion limit
    on a large flow). An edge u -> v is a back-edge iff v is gray — still on the DFS
    stack, i.e. an ancestor of u — so it closes a cycle. Visits every node, so cycles in
    components no root reaches are found too. Also returns, per back-edge, the cycle as
    it sat on the stack: [v, ..., u, v]."""
    white, gray, black = 0, 1, 2
    color = {n: white for n in node_ids}
    back_edges: list[tuple[str, str]] = []
    cycles: list[list[str]] = []

    for start in node_ids:
        if color[start] != white:
            continue
        color[start] = gray
        stack = [(start, iter(out_adj[start]))]
        while stack:
            u, successors = stack[-1]
            descended = False
            for v in successors:
                if color[v] == white:
                    color[v] = gray
                    stack.append((v, iter(out_adj[v])))
                    descended = True
                    break
                if color[v] == gray:
                    back_edges.append((u, v))
                    on_stack = [node for node, _ in stack]
                    cycles.append(on_stack[on_stack.index(v):] + [v])
            if not descended:
                color[u] = black
                stack.pop()

    return back_edges, cycles


def _looks_successful(node: dict) -> bool:
    text = " ".join(str(node.get(f) or "") for f in ("id", "name", "type", "purpose")).lower()
    return any(kw in text for kw in SUCCESS_KEYWORDS)
