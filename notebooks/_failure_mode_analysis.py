"""Per-game failure-mode analysis for the 7 games where GraphAgent scored 0 in P1.

For each game we run a 5-min instrumented session and record:
- unique_frames           : how many distinct frame hashes the agent saw
- transitions             : count of state-changing actions
- non_transitions         : count of no-op actions
- game_overs              : count of GAME_OVER states (agent died)
- level_first_revisits    : how often we returned to the starting frame of the level
- click_actions           : clicks (ACTION6)
- arrow_actions           : arrow keys (ACTION1..5)
- available_action_dist   : histogram of len(available_actions)
- segment_counts          : histogram of number of segments per frame
- final_levels_completed  : sanity check (should be 0 for the 7 target games)
- last_5_frame_hashes     : the last 5 distinct frame hashes (does it cycle?)

Output: runs/failure_modes/<game_id>.json + first-frame.npy

Run with:
    $env:OPERATION_MODE = "offline"
    $env:ENVIRONMENTS_DIR = "C:\\...\\environment_files"
    python notebooks/_failure_mode_analysis.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from collections import Counter, deque
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

os.environ["OPERATION_MODE"] = "offline"
os.environ.setdefault("ENVIRONMENTS_DIR", str(ROOT / "data" / "environment_files"))

import numpy as np  # noqa: E402
from arc_agi import Arcade  # noqa: E402
from arcengine import GameAction, GameState  # noqa: E402

from arc_agi3_solo.agents.graph_agent import GraphAgent  # noqa: E402

OUT_DIR = ROOT / "runs" / "failure_modes"
OUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("failure_mode")
log.setLevel(logging.INFO)


class InstrumentedGraphAgent(GraphAgent):
    """GraphAgent with per-step stat collection."""

    def __init__(self, *a, **kw) -> None:
        super().__init__(*a, **kw)
        self.stats = {
            "unique_frames": set(),
            "transitions": 0,
            "non_transitions": 0,
            "game_overs": 0,
            "level_first_revisits": 0,
            "click_actions": 0,
            "arrow_actions": 0,
            "reset_actions": 0,
            "available_action_lens": Counter(),
            "segment_counts": Counter(),
        }
        self._first_frame_saved = False
        self._recent_distinct = deque(maxlen=5)

    def choose_action(self, frames, latest_frame):  # type: ignore[override]
        # Record GAME_OVER + available-action histogram BEFORE delegating.
        if latest_frame.state is GameState.GAME_OVER:
            self.stats["game_overs"] += 1
        if latest_frame.available_actions:
            self.stats["available_action_lens"][len(latest_frame.available_actions)] += 1

        prev_hash = self.last_hashed_frame
        action = super().choose_action(frames, latest_frame)
        cur_hash = self.last_hashed_frame

        if cur_hash is not None:
            if not self._first_frame_saved and latest_frame.frame:
                arr = np.asarray(latest_frame.frame, dtype=np.uint8)
                if arr.size:
                    np.save(OUT_DIR / f"{self.game_id}_first_frame.npy", arr[-1])
                    self._first_frame_saved = True
            self.stats["unique_frames"].add(cur_hash)
            if prev_hash is not None and cur_hash != prev_hash:
                self.stats["transitions"] += 1
                if cur_hash != self._recent_distinct[-1] if self._recent_distinct else True:
                    self._recent_distinct.append(cur_hash)
            elif prev_hash is not None and cur_hash == prev_hash:
                self.stats["non_transitions"] += 1
            if self.level_first_frame is not None and cur_hash == self.level_first_frame:
                self.stats["level_first_revisits"] += 1
        if action is GameAction.RESET:
            self.stats["reset_actions"] += 1
        elif action.is_complex():  # ACTION6 with x/y
            self.stats["click_actions"] += 1
        else:
            self.stats["arrow_actions"] += 1

        # Light segment-count sampling so we don't blow up the counter.
        if self.action_counter % 50 == 0 and latest_frame.frame:
            try:
                arr = np.asarray(latest_frame.frame, dtype=np.uint8)
                if arr.size:
                    _, comps = self.frame_processor.segment_frame(arr[-1])
                    self.stats["segment_counts"][len(comps) // 5 * 5] += 1
            except Exception:
                pass

        return action


def analyze(game_id: str, budget_s: float = 300.0) -> dict:
    GraphAgent.MAX_ACTIONS = 1_000_000
    GraphAgent.TOTAL_TIME_BUDGET_S = budget_s
    InstrumentedGraphAgent.MAX_ACTIONS = 1_000_000
    InstrumentedGraphAgent.TOTAL_TIME_BUDGET_S = budget_s

    arc = Arcade()
    card_id = arc.open_scorecard(tags=["failure_mode", game_id])
    env = arc.make(game_id, scorecard_id=card_id)
    if env is None:
        raise RuntimeError(f"env is None for {game_id}")

    t0 = time.time()
    agent = InstrumentedGraphAgent(
        card_id=card_id, game_id=game_id,
        agent_name="instrumented", arc_env=env, tags=["failure_mode"],
    )
    agent.main()
    elapsed = time.time() - t0

    arc.close_scorecard(card_id)

    result = {
        "game_id": game_id,
        "wall_s": round(elapsed, 1),
        "actions": agent.action_counter,
        "levels_completed": int(agent.frames[-1].levels_completed),
        "unique_frames": len(agent.stats["unique_frames"]),
        "transitions": agent.stats["transitions"],
        "non_transitions": agent.stats["non_transitions"],
        "game_overs": agent.stats["game_overs"],
        "level_first_revisits": agent.stats["level_first_revisits"],
        "click_actions": agent.stats["click_actions"],
        "arrow_actions": agent.stats["arrow_actions"],
        "reset_actions": agent.stats["reset_actions"],
        "available_action_lens": dict(agent.stats["available_action_lens"]),
        "segment_count_bins": dict(agent.stats["segment_counts"]),
        "transition_rate": (
            agent.stats["transitions"]
            / max(agent.stats["transitions"] + agent.stats["non_transitions"], 1)
        ),
    }

    out = OUT_DIR / f"{game_id}.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    log.info(
        "%s: %d uniq frames, %d transitions, %d game_overs, %d L0-revisits, "
        "%d clicks vs %d arrows in %ds",
        game_id, result["unique_frames"], result["transitions"], result["game_overs"],
        result["level_first_revisits"], result["click_actions"], result["arrow_actions"],
        result["wall_s"],
    )
    return result


def main() -> None:
    games = ["cn04", "g50t", "re86", "sc25", "su15", "tr87", "wa30"]
    log.info("failure-mode analysis on %d games (5 min each)", len(games))
    summary = []
    for gid in games:
        try:
            r = analyze(gid, budget_s=300.0)
            summary.append(r)
        except Exception:
            log.exception("analyzer crashed on %s; continuing", gid)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("wrote %s", OUT_DIR / "summary.json")


if __name__ == "__main__":
    main()
