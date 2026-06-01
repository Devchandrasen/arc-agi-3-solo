"""Graph-exploration agent (P1 baseline).

Ports the dolphin-in-a-coma `just-explore` strategy onto our minimal Agent base.
Idea: hash every frame, treat actions as edges, BFS to nearest open frontier.
Action candidates are bucketed into 5 priority groups (most-promising first);
the explorer only escalates to lower-priority groups when higher ones are
exhausted.

Reference: vendor/just-explore/agents/heuristic_agent.py (MIT, Rudakov 2025).
"""

from __future__ import annotations

import logging
import random
import time
from typing import Any, Optional

import numpy as np
from arcengine import FrameData, GameAction, GameState

from arc_agi3_solo.agents.base import Agent
from arc_agi3_solo.core.frame_processor import FrameProcessor
from arc_agi3_solo.core.graph_explorer import GraphExplorer

logger = logging.getLogger(__name__)


SIMPLE_ACTION_ID2GAME_ACTION = {
    1: GameAction.ACTION1,
    2: GameAction.ACTION2,
    3: GameAction.ACTION3,
    4: GameAction.ACTION4,
    5: GameAction.ACTION5,
}


class GraphAgent(Agent):
    """Frame-graph exploration with priority-group ordering on action candidates."""

    MAX_ACTIONS: int = 1_000_000
    N_GROUPS: int = 5
    TOTAL_TIME_BUDGET_S: float = 7.9 * 60 * 60

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        random.seed(int(time.time() * 1e6) ^ (hash(self.game_id) & 0xFFFFFFFF))

        self.frame_processor = FrameProcessor()
        self.graph_explorer = GraphExplorer(n_groups=self.N_GROUPS)

        self.status_bar_mask: Optional[np.ndarray] = None
        self.hashed_frame2action_results: dict = {}
        self.hashed_frame2transitions: dict = {}

        self.level_first_frame: Optional[str] = None
        self.last_hashed_frame: Optional[str] = None
        self.last_action: Optional[int] = None
        self.last_action_object: GameAction = GameAction.RESET
        self.last_levels_completed: int = 0
        self.level_up: bool = True
        self.failed: bool = False
        self.last_transition_suspicious: bool = False

        self.time_start = time.time()

    @property
    def name(self) -> str:
        return f"{super().name}.{self.MAX_ACTIONS}"

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

    def _get_frame_buffers(self, hashed_frame: str, num_actions: int):
        results = self.hashed_frame2action_results.setdefault(hashed_frame, np.zeros(num_actions))
        transitions = self.hashed_frame2transitions.setdefault(hashed_frame, [0] * num_actions)
        return results, transitions

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        if latest_frame.state is GameState.NOT_PLAYED:
            self.last_hashed_frame = None
            self.last_action = None
            if self.failed:
                self.level_up = True
                self.failed = False
            return GameAction.RESET

        if latest_frame.state is GameState.GAME_OVER:
            self.last_transition_suspicious = True
            return GameAction.RESET

        cur_levels = int(latest_frame.levels_completed)
        if cur_levels > self.last_levels_completed:
            self.level_up = True
            self.status_bar_mask = None
        self.last_levels_completed = cur_levels

        latest_np = np.array(latest_frame.frame, dtype=np.uint8)
        if latest_np.size == 0:
            return self._random_fallback(latest_frame)
        num_frames = latest_np.shape[0]
        latest_np = latest_np[-1]

        if self.level_up:
            seg_for_sb, segs_for_sb = self.frame_processor.segment_frame(latest_np)
            _, mask = self.frame_processor.identify_status_bars(seg_for_sb, segs_for_sb)
            self.status_bar_mask = mask
            self.hashed_frame2action_results = {}
            self.hashed_frame2transitions = {}

        latest_np[self.status_bar_mask] = 16
        segmented_frame, frame_segments = self.frame_processor.segment_frame(latest_np)
        available = list(latest_frame.available_actions or [])

        num_click_actions = 0
        num_actions = 0
        arrow_actions: list[GameAction] = []

        if 6 in available:
            num_click_actions = len(frame_segments)
            num_actions += num_click_actions
            action_groups = self.frame_processor.frame_segments_to_action_groups(
                frame_segments, n_groups=self.N_GROUPS
            )
        else:
            action_groups = [set() for _ in range(self.N_GROUPS)]

        for aid in available:
            if aid in SIMPLE_ACTION_ID2GAME_ACTION:
                arrow_actions.append(SIMPLE_ACTION_ID2GAME_ACTION[aid])
                action_groups[0].add(num_actions)
                num_actions += 1

        latest_np[latest_np == 16] = 0
        hashed_frame = self.frame_processor.hash_frame(latest_np)

        if self.level_up:
            self.level_first_frame = hashed_frame
            self.graph_explorer.reset()
            self.graph_explorer.initialize(
                start_node=hashed_frame,
                num_candidates=num_actions,
                group2remaining_candidate_ids=action_groups,
            )
            self.level_up = False

        if self.last_hashed_frame is not None:
            transition = hashed_frame != self.last_hashed_frame
            suspicious = (hashed_frame == self.level_first_frame and num_frames > 1) or self.last_transition_suspicious
            self.last_transition_suspicious = False

            prev_results, prev_transitions = self._get_frame_buffers(
                self.last_hashed_frame,
                len(self.hashed_frame2action_results[self.last_hashed_frame]),
            )
            if transition:
                prev_results[self.last_action] = 1
                prev_transitions[self.last_action] = hashed_frame
            else:
                prev_results[self.last_action] = -1
                prev_transitions[self.last_action] = None

            try:
                self.graph_explorer.record_test(
                    self.last_hashed_frame,
                    self.last_action,
                    int(transition),
                    hashed_frame,
                    target_num_candidates=num_actions,
                    group2remaining_candidate_ids=action_groups,
                    suspicious_transition=suspicious,
                )
            except KeyError:
                logger.warning("graph_explorer lost prior node; resetting from current frame")
                self.graph_explorer.reset()
                self.graph_explorer.initialize(
                    start_node=hashed_frame,
                    num_candidates=num_actions,
                    group2remaining_candidate_ids=action_groups,
                )

        cur_results, _ = self._get_frame_buffers(hashed_frame, num_actions)
        available_mask = np.where(cur_results != -1)[0]
        if len(available_mask) == 0:
            return self._random_fallback(latest_frame)

        if hashed_frame in self.graph_explorer._nodes:
            action_id = self.graph_explorer.choose_edge(hashed_frame)
        else:
            action_id = int(random.choice(available_mask))

        arrow_control = action_id >= num_click_actions

        if arrow_control:
            action = arrow_actions[action_id - num_click_actions]
            action.reasoning = {"desired_action": str(action.value), "my_reason": "arrow"}
        else:
            mask = segmented_frame == action_id
            pts = np.argwhere(mask)
            y, x = pts[random.randint(0, len(pts) - 1)]
            action = GameAction.ACTION6
            action.set_data({"x": int(x), "y": int(y)})
            action.reasoning = {"desired_action": str(action.value), "my_reason": f"click seg {action_id}"}

        self.last_hashed_frame = hashed_frame
        self.last_action = action_id
        self.last_action_object = action
        return action

    def _random_fallback(self, latest_frame: FrameData) -> GameAction:
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            return GameAction.RESET
        action = random.choice([a for a in GameAction if a is not GameAction.RESET])
        if action.is_complex():
            action.set_data({"x": random.randint(0, 63), "y": random.randint(0, 63)})
            action.reasoning = {"desired_action": str(action.value), "my_reason": "fallback"}
        else:
            action.reasoning = "fallback"
        return action

    def main(self) -> None:
        self.timer = time.time()
        while (
            not self.is_done(self.frames, self.frames[-1])
            and self.action_counter <= self.MAX_ACTIONS
        ):
            try:
                action = self.choose_action(self.frames, self.frames[-1])
            except Exception:
                logger.exception("choose_action crashed; recovering with last action")
                self.failed = True
                self.level_up = True
                action = self.last_action_object
            frame = self._step(action)
            if frame is not None:
                self._append(frame)
            self.action_counter += 1
            if time.time() - self.time_start > self.TOTAL_TIME_BUDGET_S:
                logger.info("time budget exhausted for %s", self.game_id)
                break
