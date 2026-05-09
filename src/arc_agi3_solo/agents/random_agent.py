"""Random-action baseline.

Picks a uniformly random non-RESET action each step (random (x,y) for ACTION6),
respecting `available_actions`. Resets when the game is NOT_PLAYED or GAME_OVER.

This exists to (a) sanity-check the framework end-to-end and (b) give us a
floor score we can A/B against later phases.
"""

from __future__ import annotations

import logging
import random
import time
from typing import Any

from arcengine import FrameData, GameAction, GameState

from arc_agi3_solo.agents.base import Agent

logger = logging.getLogger(__name__)


class RandomAgent(Agent):
    """Uniformly random over `available_actions`, with random (x,y) for ACTION6."""

    MAX_ACTIONS: int = 5000

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Per-game seed so different games don't share an RNG stream.
        random.seed(int(time.time() * 1e6) ^ (hash(self.game_id) & 0xFFFFFFFF))

    @property
    def name(self) -> str:
        return f"{super().name}.{self.MAX_ACTIONS}"

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            return GameAction.RESET

        avail = latest_frame.available_actions or [
            a.value for a in GameAction if a is not GameAction.RESET
        ]
        avail = [a for a in avail if a != 0]  # drop RESET
        if not avail:
            return GameAction.RESET

        action = GameAction.from_id(random.choice(avail))
        if action.is_complex():
            action.set_data({"x": random.randint(0, 63), "y": random.randint(0, 63)})
            action.reasoning = {"desired_action": action.value, "my_reason": "random"}
        else:
            action.reasoning = "random"
        return action
