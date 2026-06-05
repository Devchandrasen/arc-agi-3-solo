"""
Ablation experiment runner for the OCA Phase 1 baseline.

Runs GraphAgent with various component toggles and reports per-game
and aggregate results. Designed to generate the ablation table in the
ARC Prize 2026 Paper Track submission.

Usage:
    python notebooks/_run_ablations.py

Environment variables:
    PER_GAME_BUDGET_S  : wall-clock budget per game (default 300)
    ENVIRONMENTS_DIR   : path to ARC-AGI-3 environment files
    RUNS_DIR           : output directory (default runs/ablations/)
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

os.environ["OPERATION_MODE"] = "offline"
os.environ.setdefault("ENVIRONMENTS_DIR", str(ROOT / "data" / "environment_files"))

PER_GAME_S = float(os.environ.get("PER_GAME_BUDGET_S", 300.0))
RUNS_DIR = Path(os.environ.get("RUNS_DIR", ROOT / "runs" / "ablations"))
RUNS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(RUNS_DIR / "ablation_run.log", mode="w"), logging.StreamHandler()],
)
log = logging.getLogger("ablations")

from arc_agi import Arcade
from arc_agi3_solo.agents.graph_agent import GraphAgent
from arc_agi3_solo.core.frame_processor import FrameProcessor
from arc_agi3_solo.eval.runner import discover_games


# ---- Ablation configurations ----

ABLATIONS: list[dict] = [
    # Format:
    # {
    #   "name": "<config name for tables>",
    #   "n_groups": int (1-5),
    #   "mask_status_bars": bool,
    #   "action_modality": "all" | "arrows_only" | "clicks_only",
    #   "desc": "short description",
    # }

    {
        "name": "baseline_5groups_masked",
        "n_groups": 5,
        "mask_status_bars": True,
        "action_modality": "all",
        "desc": "Baseline: 5 priority groups, status-bar masking on",
    },
    {
        "name": "no_sb_masking",
        "n_groups": 5,
        "mask_status_bars": False,
        "action_modality": "all",
        "desc": "No status-bar masking (groups still active)",
    },
    {
        "name": "single_group_uniform",
        "n_groups": 1,
        "mask_status_bars": True,
        "action_modality": "all",
        "desc": "Single priority group (uniform random ordering)",
    },
    {
        "name": "sb_only",
        "n_groups": 5,
        "mask_status_bars": True,
        "action_modality": "all",
        "desc": "Only group 4 (status-bar) segments; force explore garbage",
    },
    {
        "name": "arrows_only",
        "n_groups": 5,
        "mask_status_bars": True,
        "action_modality": "arrows_only",
        "desc": "Only arrow actions permitted; no clicks",
    },
    {
        "name": "clicks_only",
        "n_groups": 5,
        "mask_status_bars": True,
        "action_modality": "clicks_only",
        "desc": "Only click actions permitted; no arrows",
    },
]


# ---- Per-ablation agent wrapper ----

class AblationGraphAgent(GraphAgent):
    """GraphAgent subclass that applies config overrides before main()."""

    def __init__(self, config: dict, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ablation_config = config

    @property
    def name(self) -> str:
        config_name = self._ablation_config["name"]
        return f"abl.{config_name}.{self.MAX_ACTIONS}"

    def _apply_config_overrides(self):
        """Override frame processor and graph explorer params from config."""
        cfg = self._ablation_config

        # Priority groups
        self.frame_processor.N_GROUPS = cfg["n_groups"]
        self.graph_explorer = self._make_explorer(n_groups=cfg["n_groups"])
        if hasattr(self.graph_explorer, "_n_groups"):
            self.graph_explorer._n_groups = max(1, cfg["n_groups"])

        # Status-bar masking
        if not cfg["mask_status_bars"]:
            self.frame_processor.status_bar_mode = "crude"  # skip rule-based masking

        # Action modality filter
        self._action_modality = cfg.get("action_modality", "all")

    def _make_explorer(self, n_groups: int):
        from arc_agi3_solo.core.graph_explorer import GraphExplorer
        return GraphExplorer(n_groups=n_groups)

    def main(self) -> None:
        self._apply_config_overrides()
        super().main()


# ---- Runner ----

def run_ablation(config: dict, games: list[str], arc: Arcade) -> dict:
    """Run GraphAgent with given config overrides on all games.

    Returns dict with per-game scores and aggregate stats.
    """
    log.info("Starting ablation: %s (%s)", config["name"], config["desc"])

    # Clamp budget for ablation compared to baseline; use same value for all.
    AblationGraphAgent.MAX_ACTIONS = 1_000_000
    AblationGraphAgent.TOTAL_TIME_BUDGET_S = PER_GAME_S

    card_id = arc.open_scorecard(tags=["ablation", config["name"]])
    per_game: list[dict] = []
    t0 = time.time()

    for gid in games:
        g_start = time.time()
        try:
            env = arc.make(gid, scorecard_id=card_id)
            if env is None:
                log.warning("env is None for %s; skipping", gid)
                continue

            agent = AblationGraphAgent(
                config=config,
                card_id=card_id,
                game_id=gid,
                agent_name=f"graph_abl_{config['name']}",
                arc_env=env,
                tags=["ablation", config["name"]],
            )

            # Apply action modality filter if set
            if config.get("action_modality") == "arrows_only":
                agent._allowed_arrow_actions_only = True
            elif config.get("action_modality") == "clicks_only":
                agent._allowed_click_actions_only = True

            agent.main()
            elapsed = time.time() - g_start

            per_game.append({
                "game_id": gid,
                "levels_completed": int(agent.frames[-1].levels_completed),
                "actions": agent.action_counter,
                "wall_s": round(elapsed, 1),
            })
            log.info(
                "  %s done: levels=%d actions=%d wall=%.1fs",
                gid, agent.frames[-1].levels_completed, agent.action_counter, elapsed,
            )
        except Exception:
            log.exception("  %s crashed; recording 0", gid)
            per_game.append({"game_id": gid, "levels_completed": 0, "crashed": True})

    scorecard = arc.close_scorecard(card_id)
    dumped = scorecard.model_dump() if hasattr(scorecard, "model_dump") else {}

    result = {
        "config": config,
        "card_id": card_id,
        "total_wall_s": round(time.time() - t0, 1),
        "total_actions": dumped.get("total_actions", sum(p.get("actions", 0) for p in per_game)),
        "total_levels_completed": dumped.get(
            "total_levels_completed",
            sum(p.get("levels_completed", 0) for p in per_game),
        ),
        "total_levels": dumped.get("total_levels", sum(
            g.get("total_levels", 8) for g in per_game  # approximate
        )),
        "per_game": per_game,
    }
    result["pct"] = round(
        result["total_levels_completed"] / max(result["total_levels"], 1) * 100, 2
    )

    log.info(
        "Ablation %s DONE: %d/%d levels (%.2f%%), wall=%.0fs",
        config["name"],
        result["total_levels_completed"],
        result["total_levels"],
        result["pct"],
        result["total_wall_s"],
    )

    return result


def generate_summary_table(all_results: list[dict]) -> str:
    """Generate a Markdown summary table from ablation results."""
    lines = [
        "| Configuration | Levels | Total | % | Δ vs baseline |",
        "|--------------|--------|-------|---|---------------|",
    ]
    baseline_name = ABLATIONS[0]["name"]
    baseline_levels = None
    baseline_pct = None
    for r in all_results:
        if r["config"]["name"] == baseline_name:
            baseline_levels = r["total_levels_completed"]
            baseline_pct = r["pct"]
            break

    for r in all_results:
        name = r["config"]["name"]
        lvl = r["total_levels_completed"]
        total = r["total_levels"]
        pct = r["pct"]
        if baseline_levels is not None and name != baseline_name:
            delta = lvl - baseline_levels
            delta_str = f"{delta:+d} ({delta / max(baseline_levels, 1) * 100:+.1f}%)" if baseline_levels else "—"
        else:
            delta_str = "—"
        lines.append(f"| {name} | {lvl} | {total} | {pct:.2f} | {delta_str} |")

    return "\n".join(lines)


# ---- Main ----

def main():
    arc = Arcade()
    games = discover_games(os.environ["ENVIRONMENTS_DIR"])
    log.info("Ablation runner: %d games, %d configs, %.0fs budget/game",
             len(games), len(ABLATIONS), PER_GAME_S)

    all_results: list[dict] = []
    for config in ABLATIONS:
        result = run_ablation(config, games, arc)
        all_results.append(result)

    # Write per-ablation JSON logs
    for result in all_results:
        name = result["config"]["name"]
        path = RUNS_DIR / f"ablation_{name}.json"
        path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        log.info("Wrote %s", path)

    # Write summary
    summary_table = generate_summary_table(all_results)
    summary_path = RUNS_DIR / "ablation_summary.md"
    summary_path.write_text(summary_table, encoding="utf-8")
    log.info("Summary table written to %s", summary_path)

    # Print summary
    print("\n=== Ablation Summary ===\n")
    print(summary_table)
    print()

    # Calculate and print per-game deltas
    if all_results:
        baseline = all_results[0]
        total_games = len(baseline["per_game"])
        print(f"\n=== Per-game baseline scores ({baseline['config']['name']}) ===")
        for pg in baseline["per_game"]:
            print(f"  {pg['game_id']}: {pg['levels_completed']} levels, {pg['actions']} actions")

    log.info("All ablations complete. Results in %s", RUNS_DIR)


if __name__ == "__main__":
    main()
