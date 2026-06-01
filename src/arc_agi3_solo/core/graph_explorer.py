"""Frame-graph explorer used by the GraphAgent.

Ported from `vendor/just-explore/graph_explorer.py` (dolphin-in-a-coma, MIT,
Copyright (c) 2025 Evgenii Rudakov). Matplotlib visualization and demo code
removed; public API preserved verbatim so it can be unit-tested against the
upstream tests if needed.

The explorer treats each unique frame-hash as a node and each action index as
an outgoing edge candidate. It tracks tested/untested edges, priority groups
(salient → less-salient action candidates), and shortest-path distances from
any frontier node, so the agent can both (a) try new edges greedily at the
frontier and (b) plan a path back to the frontier from any explored node.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, Hashable, List, Optional, Set, Tuple

import numpy as np
import random

INFINITY = np.iinfo(np.int32).max

# Per-edge metadata. Group: priority bucket. Result: 1 success, -1 fail, 0 untested.
# Target: name of destination node (empty until tested). Distance: BFS distance
# from the frontier (maintained by GraphExplorer). Errors: probe-error count.
edge_dtype = np.dtype([
    ("group", "i4"),
    ("result", "i4"),
    ("target", "U32"),
    ("distance", "i4"),
    ("errors", "i4"),
])


@dataclass
class NodeInfo:
    name: Hashable
    total_candidates: int
    num_groups: int = 1
    active_group: int = 0
    group2remaining_candidate_ids: List[Set[int]] = field(default_factory=list)
    edge_data: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=edge_dtype))
    error_threshold: int = 3
    closed: bool = False
    distance: Optional[float] = 0

    def __post_init__(self) -> None:
        assert self.name is not None, "Node name must be provided"
        if self.num_groups > 1 and self.group2remaining_candidate_ids is None:
            raise ValueError("group2remaining_candidate_ids required when num_groups > 1")
        if self.num_groups == 1 and not self.group2remaining_candidate_ids:
            self.group2remaining_candidate_ids = [set(range(self.total_candidates))]
        self.group2remaining_candidate_ids = [set(s) for s in self.group2remaining_candidate_ids]
        self.edge_data = np.zeros(self.total_candidates, dtype=edge_dtype)
        for gid, remaining in enumerate(self.group2remaining_candidate_ids):
            self.edge_data["group"][list(remaining)] = gid

    @property
    def has_open(self) -> bool:
        return bool((self.edge_data["result"] == 0).any())

    def has_open_group(self, group_id: int) -> bool:
        for gid in range(group_id + 1):
            if self.group2remaining_candidate_ids[gid]:
                return True
        return False

    def record_test(self, edge_idx: int, success: int, target_node: Optional[Hashable] = None) -> bool:
        edge_group_id = self.edge_data[edge_idx]["group"]
        assert (
            self.edge_data["result"][edge_idx] == 0
            and self.edge_data["target"][edge_idx] == ""
            and self.edge_data["distance"][edge_idx] == 0
        ), "edge must be untested before recording"

        if success == -1:
            self.edge_data["errors"][edge_idx] += 1
            if self.edge_data["errors"][edge_idx] >= self.error_threshold:
                self.edge_data["errors"][edge_idx] = 0
                new_group_id = edge_group_id + 1
                if new_group_id > self.num_groups - 1:
                    self.group2remaining_candidate_ids[edge_group_id].discard(edge_idx)
                    self.edge_data["result"][edge_idx] = -1
                    self.edge_data["distance"][edge_idx] = INFINITY
                    return True
                self.edge_data["group"][edge_idx] = new_group_id
                self.group2remaining_candidate_ids[new_group_id].add(edge_idx)
                self.group2remaining_candidate_ids[edge_group_id].discard(edge_idx)
            return False

        self.group2remaining_candidate_ids[edge_group_id].discard(edge_idx)
        if success == 1:
            self.edge_data["target"][edge_idx] = str(target_node)
            self.edge_data["distance"][edge_idx] = -1
            self.edge_data["result"][edge_idx] = 1
        elif success == 0:
            self.edge_data["distance"][edge_idx] = INFINITY
            self.edge_data["result"][edge_idx] = -1
        return True


class GraphExplorer:
    def __init__(
        self,
        start_node: Optional[Hashable] = None,
        num_candidates: Optional[int] = None,
        group2remaining_candidate_ids: Optional[List[Set[int]]] = None,
        n_groups: int = 1,
        verbose_level: int = 0,
    ) -> None:
        self._verbose_level = verbose_level
        self._n_groups = max(1, n_groups)
        self.reset()

    def reset(self) -> None:
        self._nodes: Dict[Hashable, NodeInfo] = {}
        self._G: Dict[Hashable, Set[Tuple[int, Hashable]]] = defaultdict(set)
        self._G_rev: Dict[Hashable, Set[Tuple[int, Hashable]]] = defaultdict(set)
        self._frontier: Set[Hashable] = set()
        self._dist: Dict[Hashable, int] = {}
        self._next: Dict[Hashable, Tuple[int, Hashable]] = {}
        self._active_group: int = 0
        self.suspicious_transitions: Dict[Tuple[Hashable, int, Hashable], int] = {}
        self.suspicious_transitions_threshold: int = 3
        self._empty = True

    def initialize(
        self,
        start_node: Optional[Hashable] = None,
        num_candidates: Optional[int] = None,
        group2remaining_candidate_ids: Optional[List[Set[int]]] = None,
    ) -> None:
        if start_node is not None:
            self._add_new_node(start_node, num_candidates, group2remaining_candidate_ids=group2remaining_candidate_ids)

    def record_test(
        self,
        node: Hashable,
        edge_idx: Hashable,
        success: bool,
        target_node: Optional[Hashable] = None,
        target_num_candidates: Optional[int] = None,
        group2remaining_candidate_ids: Optional[List[Set[int]]] = None,
        suspicious_transition: bool = False,
    ) -> None:
        if node not in self._nodes:
            raise KeyError(f"unknown node {node!r}")
        node_info = self._nodes[node]

        if node_info.closed:
            if target_node == self._nodes[node].edge_data["target"][edge_idx]:
                return
            dist_to_frontier = self._dist.get(target_node, 0)
            prev_target_node = self._nodes[node].edge_data["target"][edge_idx]
            prev_dist_to_frontier = self._dist.get(prev_target_node, INFINITY)
            if dist_to_frontier >= prev_dist_to_frontier:
                return

        if suspicious_transition:
            key = (node, edge_idx, target_node)
            self.suspicious_transitions[key] = self.suspicious_transitions.get(key, 0) + 1
            if self.suspicious_transitions[key] < self.suspicious_transitions_threshold:
                return

        node_info.record_test(edge_idx, success, target_node)

        if success == 1:
            if target_node is None:
                raise ValueError("target_node required when success=True")
            if target_node not in self._nodes:
                if target_num_candidates is None:
                    raise ValueError("target_num_candidates required for a new node")
                self._add_new_node(target_node, target_num_candidates, group2remaining_candidate_ids=group2remaining_candidate_ids)
            self._G[node].add((edge_idx, target_node))
            self._G_rev[target_node].add((edge_idx, node))

            if not self._nodes[node].has_open_group(self.active_group):
                self._close_node(node)
            if self._nodes[target_node].has_open_group(self.active_group):
                self._rebuild_distances()
            else:
                self._close_node(target_node)
                self._maybe_advance_group(target_node)
        else:
            if not self._nodes[node].has_open_group(self.active_group):
                self._close_node(node)
                self._maybe_advance_group(node)

    def get_distance(self, node: Hashable) -> Optional[int]:
        d = self._dist.get(node)
        return None if d is None or d == float("inf") else d

    def get_next_hop(self, node: Hashable) -> Optional[Hashable]:
        if node in self._frontier:
            return node
        nxt = self._next.get(node)
        if nxt is None:
            return None
        if isinstance(nxt, tuple) and len(nxt) == 2:
            return nxt[1]
        return nxt

    def edge_info(self, node: Hashable, edge_idx: Hashable) -> np.ndarray:
        return self._nodes[node].edge_data[edge_idx]

    def is_finished(self) -> bool:
        return not self._frontier

    @property
    def active_group(self) -> int:
        return self._active_group

    @property
    def empty(self) -> bool:
        return self._empty

    def _add_new_node(
        self,
        node: Hashable,
        n_candidates: int,
        group2remaining_candidate_ids: Optional[List[Set[int]]] = None,
    ) -> None:
        if n_candidates < 1:
            raise ValueError("num_candidates must be positive")
        self._nodes[node] = NodeInfo(node, n_candidates, self._n_groups, group2remaining_candidate_ids=group2remaining_candidate_ids)
        self._G[node] = set()
        self._G_rev[node] = set()
        if self._empty:
            self._empty = False
        if self._nodes[node].has_open_group(self.active_group):
            self._frontier.add(node)
        else:
            self._close_node(node)
            self._maybe_advance_group(node)

    def _close_node(self, node: Hashable) -> None:
        info = self._nodes[node]
        if info.closed:
            return
        info.closed = True
        self._frontier.discard(node)
        self._rebuild_distances()

    def _rebuild_distances(self) -> None:
        self._dist.clear()
        self._next.clear()
        dq = deque(self._frontier)
        for _, info in self._nodes.items():
            info.distance = INFINITY
            self._dist[info.name] = INFINITY
        for src in self._frontier:
            self._nodes[src].distance = 0
            self._dist[src] = 0
        while dq:
            v = dq.popleft()
            v_dist = self._dist.get(v, INFINITY)
            for edge_idx, u in self._G_rev.get(v, ()):
                u_info = self._nodes[u]
                u_dist = self._dist.get(u, INFINITY)
                u_info.edge_data["distance"][edge_idx] = v_dist + 1
                if u_dist > u_info.edge_data["distance"][edge_idx]:
                    u_info.distance = u_info.edge_data["distance"][edge_idx]
                    self._dist[u] = u_info.edge_data["distance"][edge_idx]
                    self._next[u] = (edge_idx, v)
                    dq.append(u)

    def _maybe_advance_group(self, current_node: Hashable) -> None:
        distance = self._nodes[current_node].distance
        while distance == INFINITY and self.active_group < self._n_groups - 1:
            self._active_group += 1
            self._dist.clear()
            self._next.clear()
            self._frontier.clear()
            for _, info in self._nodes.items():
                info.active_group = self.active_group
                if info.has_open_group(self.active_group):
                    self._frontier.add(info.name)
                    info.closed = False
            self._rebuild_distances()
            distance = self._dist.get(current_node)

    def choose_edge(self, node: Hashable, return_reasoning: bool = False):
        info = self._nodes[node]
        if info.has_open_group(self.active_group):
            untested: List[int] = []
            for gid in range(self.active_group + 1):
                untested.extend(info.group2remaining_candidate_ids[gid])
            if not untested:
                raise ValueError("no untested edges in active group while group reports open")
            edge_idx = random.choice(untested)
            reasoning = f"random untested edge {edge_idx} from group<={self.active_group}"
        else:
            lowest_dist = info.distance
            candidates = [
                i for i, d in enumerate(info.edge_data)
                if d["distance"] <= lowest_dist and d["result"] == 1 and d["group"] <= self.active_group
            ]
            edge_idx = random.choice(candidates)
            reasoning = f"BFS-next edge {edge_idx} at dist {lowest_dist}"
        if return_reasoning:
            return edge_idx, reasoning
        return edge_idx
