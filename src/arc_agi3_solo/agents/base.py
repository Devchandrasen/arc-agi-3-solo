"""Minimal Agent base class.

Functionally identical to vendor `agents.agent.Agent` (so subclasses port over
verbatim) but with no dependency on the vendor's `agents/__init__.py`, which
eagerly imports langgraph, smolagents, openai, etc. That dep tree is huge and
unnecessary for our exploration agents.

The interface to implement is just:

    is_done(frames, latest_frame)  -> bool
    choose_action(frames, latest_frame) -> GameAction

`main()` runs the loop. Pass `arc_env: EnvironmentWrapper` (from `arc_agi`).
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Optional

from arc_agi import EnvironmentWrapper
from arcengine import FrameData, FrameDataRaw, GameAction, GameState

logger = logging.getLogger(__name__)


class Agent(ABC):
    """Drop-in replacement for vendor's Agent base class, dependency-light."""

    MAX_ACTIONS: int = 80

    def __init__(
        self,
        card_id: str,
        game_id: str,
        agent_name: str,
        arc_env: EnvironmentWrapper,
        ROOT_URL: str = "",  # unused offline; kept for vendor-compat signature
        record: bool = False,  # unused; vendor's Recorder isn't wired in here
        tags: Optional[list[str]] = None,
    ) -> None:
        self.card_id = card_id
        self.game_id = game_id
        self.agent_name = agent_name
        self.arc_env = arc_env
        self.ROOT_URL = ROOT_URL
        self.tags = tags or []
        self.frames: list[FrameData] = [FrameData(levels_completed=0)]
        self.action_counter: int = 0
        self.timer: float = 0.0
        self.guid: str = ""

    @property
    def name(self) -> str:
        return f"{self.game_id}.{self.__class__.__name__.lower()}"

    @property
    def state(self) -> GameState:
        return self.frames[-1].state

    @property
    def levels_completed(self) -> int:
        return int(self.frames[-1].levels_completed)

    @property
    def seconds(self) -> float:
        return round(time.time() - self.timer, 2)

    @property
    def fps(self) -> float:
        if self.action_counter == 0:
            return 0.0
        return round(self.action_counter / max(self.seconds, 0.1), 2)

    def main(self) -> None:
        """Play the game until WIN, MAX_ACTIONS, or unrecoverable error."""
        self.timer = time.time()
        # Prime the loop with a synthetic NOT_PLAYED frame so the first
        # choose_action sees a valid state and returns RESET.
        while (
            not self.is_done(self.frames, self.frames[-1])
            and self.action_counter <= self.MAX_ACTIONS
        ):
            action = self.choose_action(self.frames, self.frames[-1])
            frame = self._step(action)
            if frame is not None:
                self._append(frame)
                logger.info(
                    "%s | %s | step %d | levels %d | fps %.1f",
                    self.game_id,
                    action.name,
                    self.action_counter,
                    frame.levels_completed,
                    self.fps,
                )
            self.action_counter += 1

    def _step(self, action: GameAction) -> Optional[FrameData]:
        """Send `action` to the env and convert the raw response."""
        try:
            data = action.action_data.model_dump()
            raw: Optional[FrameDataRaw] = self.arc_env.step(
                action,
                data=data,
                reasoning=data.get("reasoning", {}),
            )
        except Exception:
            logger.exception("step failed for action %s", action.name)
            return None
        if raw is None:
            return None
        return FrameData(
            game_id=raw.game_id,
            frame=[arr.tolist() for arr in raw.frame],
            state=raw.state,
            levels_completed=raw.levels_completed,
            win_levels=raw.win_levels,
            guid=raw.guid,
            full_reset=raw.full_reset,
            available_actions=raw.available_actions,
        )

    def _append(self, frame: FrameData) -> None:
        self.frames.append(frame)
        if frame.guid:
            self.guid = frame.guid

    @abstractmethod
    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool: ...

    @abstractmethod
    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction: ...
