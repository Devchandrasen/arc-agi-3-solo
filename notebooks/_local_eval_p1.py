"""Local 25-game eval for the GraphAgent (Phase 1 baseline).

Mirrors the submission notebook's run loop but writes results to
RESULTS_p1_local.json next to RESULTS.md. Runs in foreground; intended to
be launched in the background (e.g. PowerShell `run_in_background`).

Per-game wall-clock budget defaults to 5 minutes (25*5 = 125 min total).
Override with PER_GAME_BUDGET_S env var.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

os.environ["OPERATION_MODE"] = "offline"
os.environ.setdefault("ENVIRONMENTS_DIR", str(ROOT / "data" / "environment_files"))

PER_GAME_S = float(os.environ.get("PER_GAME_BUDGET_S", 300.0))

LOG_FILE = ROOT / "runs" / "p1_local_eval.log"
RESULTS_JSON = ROOT / "runs" / "p1_local_eval.json"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, mode="w"), logging.StreamHandler()],
)
log = logging.getLogger("p1_eval")

from arc_agi import Arcade  # noqa: E402

from arc_agi3_solo.agents.graph_agent import GraphAgent  # noqa: E402
from arc_agi3_solo.eval.runner import discover_games  # noqa: E402

GraphAgent.MAX_ACTIONS = 1_000_000
GraphAgent.TOTAL_TIME_BUDGET_S = PER_GAME_S

games = discover_games(os.environ["ENVIRONMENTS_DIR"])
log.info("p1 eval: %d games, %.1f min/game budget, log -> %s", len(games), PER_GAME_S / 60, LOG_FILE)

arc = Arcade()
card_id = arc.open_scorecard(tags=["graph", "p1-local"])
log.info("scorecard %s", card_id)

t0 = time.time()
per_game: list[dict] = []
for gid in games:
    g_start = time.time()
    try:
        env = arc.make(gid, scorecard_id=card_id)
        if env is None:
            log.warning("env is None for %s; skipping", gid)
            continue
        agent = GraphAgent(card_id=card_id, game_id=gid, agent_name="graph", arc_env=env, tags=["graph", "p1-local"])
        agent.main()
        elapsed = time.time() - g_start
        per_game.append({
            "game_id": gid,
            "levels_completed": int(agent.frames[-1].levels_completed),
            "actions": agent.action_counter,
            "wall_s": round(elapsed, 1),
        })
        log.info(
            "%s done: levels=%d actions=%d wall=%.1fs total=%.0fs",
            gid, agent.frames[-1].levels_completed, agent.action_counter, elapsed, time.time() - t0,
        )
    except Exception:
        log.exception("%s crashed; continuing", gid)
        per_game.append({"game_id": gid, "crashed": True})

scorecard = arc.close_scorecard(card_id)
dumped = scorecard.model_dump()

result = {
    "scorecard_id": card_id,
    "per_game_budget_s": PER_GAME_S,
    "total_wall_s": round(time.time() - t0, 1),
    "total_actions": dumped.get("total_actions"),
    "total_levels_completed": dumped.get("total_levels_completed"),
    "total_levels": dumped.get("total_levels"),
    "total_environments_completed": dumped.get("total_environments_completed"),
    "per_game": per_game,
    "scorecard": dumped,
}
RESULTS_JSON.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
log.info(
    "DONE: %d/%d levels, %d/%d envs, %d actions, wall=%.0fs -> %s",
    dumped.get("total_levels_completed", 0),
    dumped.get("total_levels", 0),
    dumped.get("total_environments_completed", 0),
    dumped.get("total_environments", 0),
    dumped.get("total_actions", 0),
    time.time() - t0,
    RESULTS_JSON,
)
