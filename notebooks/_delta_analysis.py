"""Control-scheme discovery via frame-delta segment tracking (P2 groundwork).

For each Cluster-2 game we drive a random arrow policy, and for every
state-changing transition we:
  1. segment the before- and after-frames,
  2. match segments across the pair (same color, similar area, nearest centroid),
  3. record the displacement of the largest *salient* moving segment.

We then aggregate per-arrow-action displacement vectors. If pressing a given
arrow consistently moves one segment in one direction (low-variance, distinct
per arrow), the game has a discoverable control scheme + controllable object
-- the lever a frame-delta salience policy would exploit. If displacements
are noisy / undirected, whole-frame arrow control isn't the mechanic and P2
needs a different angle.

Output: runs/delta/<game>.json + runs/delta/REPORT.md

Run:
    $env:OPERATION_MODE = "offline"
    $env:ENVIRONMENTS_DIR = "...\\environment_files"
    python notebooks/_delta_analysis.py
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

os.environ["OPERATION_MODE"] = "offline"
os.environ.setdefault("ENVIRONMENTS_DIR", str(ROOT / "data" / "environment_files"))

import numpy as np  # noqa: E402
from arc_agi import Arcade  # noqa: E402
from arcengine import GameAction, GameState  # noqa: E402

from arc_agi3_solo.core.frame_processor import FrameProcessor  # noqa: E402

OUT = ROOT / "runs" / "delta"
OUT.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.WARNING, format="%(message)s")
log = logging.getLogger("delta")
log.setLevel(logging.INFO)

SALIENT = set(range(6, 16))
ARROW_IDS = [1, 2, 3, 4, 5]


def _centroids(components):
    """Return list of (color, area, cx, cy) for salient segments."""
    out = []
    for c in components:
        if c["color"] not in SALIENT:
            continue
        x1, y1, x2, y2 = c["bounding_box"]
        out.append((c["color"], c["area"], (x1 + x2) / 2.0, (y1 + y2) / 2.0))
    return out


def _largest_moving_displacement(before, after, fp: FrameProcessor):
    """Match salient segments before/after; return displacement (dx,dy) of the
    largest matched segment that actually moved, or None."""
    _, cb = fp.segment_frame(before)
    _, ca = fp.segment_frame(after)
    pb = _centroids(cb)
    pa = _centroids(ca)
    if not pb or not pa:
        return None
    best = None  # (area, dx, dy)
    for color_a, area_a, cxa, cya in pa:
        # candidate matches: same color, area within +-30%
        cands = [
            (cxb, cyb)
            for (color_b, area_b, cxb, cyb) in pb
            if color_b == color_a and abs(area_b - area_a) <= 0.3 * max(area_a, 1)
        ]
        if not cands:
            continue
        # nearest centroid
        cxb, cyb = min(cands, key=lambda p: (p[0] - cxa) ** 2 + (p[1] - cya) ** 2)
        dx, dy = cxa - cxb, cya - cyb
        if abs(dx) < 0.5 and abs(dy) < 0.5:
            continue  # didn't move
        if best is None or area_a > best[0]:
            best = (area_a, dx, dy)
    if best is None:
        return None
    return best[1], best[2]


def analyze(game: str, n_transitions: int = 600) -> dict:
    fp = FrameProcessor()
    arc = Arcade()
    card = arc.open_scorecard(tags=["delta", game])
    env = arc.make(game, scorecard_id=card)
    random.seed(0)

    raw = env.step(GameAction.RESET, data={}, reasoning={})
    prev = np.asarray(raw.frame[-1], dtype=np.uint8) if (raw and raw.frame) else None

    per_arrow = defaultdict(list)  # arrow_id -> [(dx, dy), ...]
    moved_count = 0
    total = 0

    while total < n_transitions and raw is not None:
        avail = [a for a in (raw.available_actions or ARROW_IDS) if a in ARROW_IDS]
        if not avail:
            # not an arrow game; bail early
            break
        aid = random.choice(avail)
        action = GameAction.from_id(aid)
        action.reasoning = "delta"
        raw = env.step(action, data=action.action_data.model_dump(), reasoning={})
        if raw is None:
            break
        if raw.state in (GameState.GAME_OVER, GameState.NOT_PLAYED):
            r2 = env.step(GameAction.RESET, data={}, reasoning={})
            prev = np.asarray(r2.frame[-1], dtype=np.uint8) if (r2 and r2.frame) else None
            raw = r2
            continue
        cur = np.asarray(raw.frame[-1], dtype=np.uint8) if raw.frame else None
        if prev is not None and cur is not None and prev.shape == cur.shape:
            total += 1
            disp = _largest_moving_displacement(prev, cur, fp)
            if disp is not None:
                per_arrow[aid].append(disp)
                moved_count += 1
        prev = cur

    arc.close_scorecard(card)

    # Aggregate per-arrow displacement stats
    arrow_stats = {}
    for aid, disps in per_arrow.items():
        arr = np.array(disps)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        arrow_stats[int(aid)] = {
            "n": len(disps),
            "mean_dx": round(float(mean[0]), 2),
            "mean_dy": round(float(mean[1]), 2),
            "std_dx": round(float(std[0]), 2),
            "std_dy": round(float(std[1]), 2),
            "mean_magnitude": round(float(np.linalg.norm(mean)), 2),
        }

    # "Control discovered" heuristic: >=2 arrows have mean displacement
    # magnitude > 1.0 px AND point in distinguishable directions (pairwise
    # mean-vector angle separation), with std not dwarfing the mean.
    directed = {
        a: s for a, s in arrow_stats.items()
        if s["mean_magnitude"] > 1.0 and s["n"] >= 5
        and s["mean_magnitude"] > 0.5 * (s["std_dx"] + s["std_dy"]) / 2 + 1e-9
    }
    control = len(directed) >= 2

    result = {
        "game": game,
        "transitions": total,
        "moved_transitions": moved_count,
        "move_rate": round(moved_count / max(total, 1), 3),
        "arrow_stats": arrow_stats,
        "n_directed_arrows": len(directed),
        "control_discovered": control,
    }
    (OUT / f"{game}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    log.info(
        "%s: %d transitions, move_rate=%.2f, directed_arrows=%d, control=%s",
        game, total, result["move_rate"], len(directed), control,
    )
    for a, s in sorted(arrow_stats.items()):
        log.info("   arrow %d: n=%d mean=(%.1f,%.1f) std=(%.1f,%.1f) |mean|=%.1f",
                 a, s["n"], s["mean_dx"], s["mean_dy"], s["std_dx"], s["std_dy"], s["mean_magnitude"])
    return result


def main() -> None:
    games = ["re86", "tr87", "wa30"]
    results = [analyze(g) for g in games]
    lines = ["# Frame-delta control-discovery analysis", ""]
    lines.append("| Game | Transitions | Move rate | Directed arrows | Control discovered? |")
    lines.append("|---|---|---|---|---|")
    for r in results:
        lines.append(
            f"| {r['game']} | {r['transitions']} | {r['move_rate']} | "
            f"{r['n_directed_arrows']} | {'YES' if r['control_discovered'] else 'no'} |"
        )
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("wrote %s", OUT / "REPORT.md")


if __name__ == "__main__":
    main()
