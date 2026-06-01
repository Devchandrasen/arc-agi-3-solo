"""Phase 2 A/B vs Phase 1 on the three Cluster-2 games (re86, tr87, wa30).

Same 5-min wall budget per game (matches P1 main eval). Outputs to
runs/p2_cluster2.json with per-game levels + cluster-collapse stats.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

os.environ["OPERATION_MODE"] = "offline"
os.environ.setdefault("ENVIRONMENTS_DIR", str(ROOT / "data" / "environment_files"))

LOG_FILE = ROOT / "runs" / "p2_cluster2.log"
RESULT_JSON = ROOT / "runs" / "p2_cluster2.json"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, mode="w"), logging.StreamHandler()],
)
log = logging.getLogger("p2_eval")

from arc_agi import Arcade  # noqa: E402

from arc_agi3_solo.agents.cnn_graph_agent import CNNGraphAgent  # noqa: E402

CNNGraphAgent.MAX_ACTIONS = 1_000_000
CNNGraphAgent.TOTAL_TIME_BUDGET_S = 300.0

GAMES = ["re86", "tr87", "wa30"]

arc = Arcade()
card_id = arc.open_scorecard(tags=["cnn-graph", "p2", "cluster2"])
log.info("scorecard %s, %d games, 5 min each", card_id, len(GAMES))

t0 = time.time()
per_game = []
for gid in GAMES:
    start = time.time()
    env = arc.make(gid, scorecard_id=card_id)
    if env is None:
        log.warning("env None for %s; skipping", gid)
        continue
    agent = CNNGraphAgent(
        card_id=card_id, game_id=gid, agent_name="cnn-graph", arc_env=env, tags=["cnn-graph"],
    )
    try:
        agent.main()
    except Exception:
        log.exception("agent crashed on %s; continuing", gid)
    elapsed = time.time() - start
    per_game.append({
        "game_id": gid,
        "actions": agent.action_counter,
        "levels_completed": int(agent.frames[-1].levels_completed),
        "n_hashes": agent.cluster_map.n_hashes,
        "n_clusters": agent.cluster_map.n_clusters,
        "collapse_ratio": round(agent.cluster_map.collapse_ratio(), 2),
        "wall_s": round(elapsed, 1),
    })
    log.info(
        "%s: levels=%d actions=%d hashes=%d clusters=%d (%.1fx) wall=%.0fs",
        gid, agent.frames[-1].levels_completed, agent.action_counter,
        agent.cluster_map.n_hashes, agent.cluster_map.n_clusters,
        agent.cluster_map.collapse_ratio(), elapsed,
    )

arc.close_scorecard(card_id)
total = {
    "per_game": per_game,
    "total_wall_s": round(time.time() - t0, 1),
    "total_actions": sum(g["actions"] for g in per_game),
    "total_levels": sum(g["levels_completed"] for g in per_game),
    "p1_baseline": {"re86": 0, "tr87": 0, "wa30": 0},
}
RESULT_JSON.write_text(json.dumps(total, indent=2), encoding="utf-8")
log.info("DONE: P2 cluster2 levels=%d -> %s", total["total_levels"], RESULT_JSON)
