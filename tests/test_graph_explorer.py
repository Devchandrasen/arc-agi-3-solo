"""Unit tests for GraphExplorer + NodeInfo.

Locks in the bug fixes from commit 6276f99 (closed-node assertion loop) and
ce21323 (level-up KeyError). Both are easy to re-introduce when refactoring
for later phases, so we pin them here.
"""

from __future__ import annotations

import pytest

from arc_agi3_solo.core.graph_explorer import GraphExplorer, NodeInfo


def test_node_info_initial_state_all_untested() -> None:
    info = NodeInfo(name="A", total_candidates=4)
    assert info.has_open
    assert (info.edge_data["result"] == 0).all()
    assert (info.edge_data["target"] == "").all()


def test_explorer_records_simple_transition_and_creates_target() -> None:
    gx = GraphExplorer(n_groups=1)
    gx.initialize(start_node="A", num_candidates=2)
    gx.record_test("A", 0, 1, target_node="B", target_num_candidates=3)
    assert "B" in gx._nodes
    assert (0, "B") in gx._G["A"]


def test_explorer_fail_marks_edge_failed() -> None:
    gx = GraphExplorer(n_groups=1)
    gx.initialize(start_node="A", num_candidates=2)
    gx.record_test("A", 0, 0)  # no transition observed
    assert gx._nodes["A"].edge_data["result"][0] == -1


def test_closed_node_replay_does_not_assert() -> None:
    """Regression test for commit 6276f99.

    Once a source node is closed (all edges tested), re-recording any
    transition from it should be a no-op, not an AssertionError.
    """
    gx = GraphExplorer(n_groups=1)
    gx.initialize(start_node="A", num_candidates=1)
    gx.record_test("A", 0, 1, target_node="B", target_num_candidates=1)  # closes A
    assert gx._nodes["A"].closed
    # Now replay the same edge with a different target -- must not raise.
    gx.record_test("A", 0, 1, target_node="C", target_num_candidates=1)
    # The original target should still be B (no rewiring, by design).
    assert gx._nodes["A"].edge_data["target"][0] == "B"


def test_closed_node_replay_with_closer_target_does_not_assert() -> None:
    """The original bug fired specifically on the 'closer target' branch.

    Build a graph where the original target gets de-prioritised so a replay
    with a *closer* target would have triggered the assertion. After the
    fix it must just return without mutating state.
    """
    gx = GraphExplorer(n_groups=1)
    gx.initialize(start_node="A", num_candidates=1)
    gx.record_test("A", 0, 1, target_node="B", target_num_candidates=1)
    assert gx._nodes["A"].closed
    # Forge a 'closer' frontier-distance for hypothetical target C.
    gx._dist["C"] = 0
    gx._dist["B"] = 10
    # Must not raise even though the closed-node closer-target branch would
    # historically have fallen through to NodeInfo.record_test's assertion.
    gx.record_test("A", 0, 1, target_node="C", target_num_candidates=1)


def test_choose_edge_returns_untested_when_open() -> None:
    gx = GraphExplorer(n_groups=1)
    gx.initialize(start_node="A", num_candidates=3)
    edge = gx.choose_edge("A")
    assert edge in {0, 1, 2}
    assert gx._nodes["A"].edge_data["result"][edge] == 0


def test_choose_edge_after_all_tested_uses_bfs() -> None:
    gx = GraphExplorer(n_groups=1)
    gx.initialize(start_node="A", num_candidates=2)
    # Test both A edges: 0 -> B (open), 1 -> fail. Closes A.
    gx.record_test("A", 0, 1, target_node="B", target_num_candidates=1)
    gx.record_test("A", 1, 0)
    assert gx._nodes["A"].closed
    # At A there are no open edges; choose_edge must return the BFS-next edge.
    edge = gx.choose_edge("A")
    assert gx._nodes["A"].edge_data["result"][edge] == 1


def test_error_then_success_keeps_assertions_happy() -> None:
    """Edges that flip from probe-error to success should not trip the assert."""
    gx = GraphExplorer(n_groups=2)
    gx.initialize(
        start_node="A",
        num_candidates=2,
        group2remaining_candidate_ids=[{0}, {1}],
    )
    # Three probe errors on edge 0 promote it to group 1 (error_threshold=3).
    gx.record_test("A", 0, -1)
    gx.record_test("A", 0, -1)
    gx.record_test("A", 0, -1)
    # Now record success on the still-untested edge 0 (group 1).
    gx.record_test(
        "A", 0, 1, target_node="B",
        target_num_candidates=1,
        group2remaining_candidate_ids=[{0}, set()],
    )
    assert "B" in gx._nodes


@pytest.mark.parametrize("n_candidates", [1, 4, 16])
def test_frontier_drains_on_full_exploration(n_candidates: int) -> None:
    gx = GraphExplorer(n_groups=1)
    gx.initialize(start_node="A", num_candidates=n_candidates)
    for i in range(n_candidates):
        gx.record_test("A", i, 0)  # every edge fails
    assert gx.is_finished()
