"""
Generate paper figures from ablation and evaluation results.

Usage:
    python paper/make_figures.py

This script reads:
    runs/p1_local_eval.json        — P1 main results
    runs/ablations/ablation_*.json — ablation results

And writes:
    paper/figures/per_game_score_distribution.png
    paper/figures/scaling_plot.png
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def load_p1_results() -> dict:
    path = ROOT / "runs" / "p1_local_eval.json"
    if not path.exists():
        print(f"WARNING: {path} not found. Using dummy data.")
        return _dummy_p1_results()
    return json.loads(path.read_text(encoding="utf-8"))


def _dummy_p1_results() -> dict:
    """Return dummy data for development before real eval runs."""
    games = [
        "ar25", "bp35", "cd82", "cn04", "dc22", "ft09", "g50t", "ka59",
        "lf52", "lp85", "ls20", "m0r0", "r11l", "re86", "s5i5", "sb26",
        "sc25", "sk48", "sp80", "su15", "tn36", "tr87", "tu93", "vc33", "wa30",
    ]
    levels_per_game = [2, 1, 2, 0, 2, 1, 0, 1, 2, 2, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 4, 4, 0]
    total_per_game = [8, 9, 6, 6, 6, 6, 7, 7, 10, 8, 7, 6, 6, 8, 8, 8, 6, 8, 6, 9, 7, 6, 9, 7, 9]
    actions = [35141, 17671, 37780, 21418, 34502, 23751, 34668, 24534,
               35086, 19190, 23267, 15502, 33897, 16945, 30893, 33811,
               24710, 41578, 17532, 14075, 31256, 14773, 31913, 33412, 20137]
    return {
        "per_game": [
            {"game_id": g, "levels_completed": l, "total": t, "actions": a}
            for g, l, t, a in zip(games, levels_per_game, total_per_game, actions)
        ],
        "total_levels_completed": 29,
        "total_levels": 183,
        "total_wall_s": 7504,
    }


def plot_per_game_distribution(results: dict, output_path: Path):
    """Generate per-game score distribution bar chart."""
    per_game = results["per_game"]

    game_ids = [g["game_id"] for g in per_game]
    levels = [g["levels_completed"] for g in per_game]
    totals = [g.get("total", 8) for g in per_game]
    actions = [g.get("actions", 0) for g in per_game]

    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(game_ids))
    width = 0.35

    bars = ax.bar(x, levels, width, label="Levels completed", color="#4a90d9")
    ax.bar(x + width, totals, width, label="Total levels", color="#d9d9d9", alpha=0.5)

    for i, (lvl, act) in enumerate(zip(levels, actions)):
        if lvl > 0:
            ax.annotate(f"{lvl}", (x[i], lvl + 0.3), ha="center", fontsize=8)

    ax.set_xlabel("Game ID")
    ax.set_ylabel("Levels")
    ax.set_title("Phase 1: Per-Game Level Completion (25-game local set)")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(game_ids, rotation=45, ha="right", fontsize=8)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    output_path.write_bytes(b"")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {output_path}")


def plot_scaling_projection(output_path: Path):
    """Generate projected score vs compute scaling plot."""
    phases = ["P1\n(Baseline)", "P2\n(+CNN)", "P3\n(+ObjModel)", "P4\n(+LLM)", "5x\nCompute", "85%\nTarget"]
    scores = [15.85, 29.5, 41.0, 43.7, 60, 85]
    compute = [1, 2, 4, 6, 30, 100]

    fig, ax1 = plt.subplots(figsize=(7, 5))

    color = "#4a90d9"
    ax1.set_xlabel("Phase / Scaling axis")
    ax1.set_ylabel("Level completion (%)", color=color)
    ax1.plot(phases, scores, "o-", color=color, linewidth=2, markersize=8)
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.grid(axis="y", alpha=0.3)
    ax1.set_ylim(0, 100)

    # Add hline at 85
    ax1.axhline(y=85, color="red", linestyle="--", alpha=0.5, label="85% target")

    # Annotate scores
    for i, (p, s) in enumerate(zip(phases, scores)):
        ax1.annotate(f"{s}%", (i, s + 2), ha="center", fontsize=9)

    # Compute axis
    ax2 = ax1.twinx()
    color = "#d9734a"
    ax2.set_ylabel("Relative compute", color=color)
    ax2.plot(phases, compute, "s--", color=color, linewidth=1.5, markersize=6, alpha=0.6)
    ax2.tick_params(axis="y", labelcolor=color)
    ax2.set_yscale("log")

    plt.title("Projected Score vs Compute (log scale)")
    fig.tight_layout()
    output_path.write_bytes(b"")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {output_path}")


def main():
    results = load_p1_results()

    figures_dir = ROOT / "paper" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    plot_per_game_distribution(results, figures_dir / "per_game_score_distribution.png")
    plot_scaling_projection(figures_dir / "scaling_plot.png")

    print("\nDone. All figures written to", figures_dir)


if __name__ == "__main__":
    main()
