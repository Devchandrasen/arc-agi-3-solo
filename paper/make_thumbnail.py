"""Generate the 560x280 thumbnail for the Paper Track writeup.

Side-by-side: a real ARC-AGI-3 frame on the left, the same frame after
FrameProcessor's segmentation + status-bar mask + priority-group coloring
on the right. That visual *is* the Phase 1 method.

Output: paper/thumbnail.png (560 x 280, 100 DPI).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

os.environ["OPERATION_MODE"] = "offline"
os.environ.setdefault("ENVIRONMENTS_DIR", str(ROOT / "data" / "environment_files"))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from arc_agi import Arcade  # noqa: E402
from arc_agi3_solo.core.frame_processor import FrameProcessor  # noqa: E402

# Same ARC-AGI palette the engine uses internally.
ARC_COLORS = np.array([
    [255, 255, 255], [204, 204, 204], [153, 153, 153], [102, 102, 102],
    [51, 51, 51],    [0, 0, 0],       [255, 0, 0],     [0, 255, 0],
    [250, 61, 50],   [31, 147, 255],  [137, 216, 241], [255, 221, 0],
    [255, 133, 26],  [229, 58, 163],  [79, 205, 48],   [163, 86, 214],
], dtype=np.uint8)


def grab_frame() -> np.ndarray:
    arc = Arcade()
    card = arc.open_scorecard(tags=["thumb"])
    env = arc.make("vc33", scorecard_id=card)  # vc33 is the top-scoring game (57.1%)
    from arcengine import GameAction
    raw = env.step(GameAction.RESET, data={}, reasoning={})
    frames = raw.frame if raw is not None else None
    if frames is None or len(frames) == 0:
        raise RuntimeError("no frame returned by env")
    arr = np.asarray(frames[-1], dtype=np.uint8)
    arc.close_scorecard(card)
    return arr


def colorize(grid: np.ndarray) -> np.ndarray:
    return ARC_COLORS[np.clip(grid, 0, len(ARC_COLORS) - 1)]


def make_priority_view(frame: np.ndarray, fp: FrameProcessor):
    seg, comps = fp.segment_frame(frame)
    _, sb_mask = fp.identify_status_bars(seg, comps)
    groups = fp.frame_segments_to_action_groups(comps, n_groups=5)

    # Map each pixel to its priority group (0..4) for coloring; status bars get 5.
    pri = np.full(frame.shape, 5, dtype=np.uint8)  # 5 = unused/background
    for gid, sids in enumerate(groups):
        for sid in sids:
            pri[seg == sid] = gid
    pri[sb_mask] = 6  # status bars

    palette = np.array([
        [220, 38, 38],   # group 0: salient + medium  (red)
        [251, 146, 60],  # group 1: medium            (orange)
        [250, 204, 21],  # group 2: salient           (yellow)
        [59, 130, 246],  # group 3: other             (blue)
        [148, 163, 184], # group 4: status-bar-color  (slate)
        [240, 240, 240], # 5: background              (near-white)
        [30, 41, 59],    # 6: detected status bar     (dark slate)
    ], dtype=np.uint8)
    return palette[pri]


def main() -> None:
    frame = grab_frame()
    print(f"grabbed frame, shape={frame.shape}, unique={np.unique(frame)}")

    fp = FrameProcessor()
    raw_rgb = colorize(frame)
    pri_rgb = make_priority_view(frame.copy(), fp)

    fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.8), dpi=100)
    fig.patch.set_facecolor("#0b1220")
    for ax, img, title in [
        (axes[0], raw_rgb, "Raw frame"),
        (axes[1], pri_rgb, "Segmented + priority groups"),
    ]:
        ax.imshow(img, interpolation="nearest")
        ax.set_title(title, color="white", fontsize=10, pad=4)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    plt.subplots_adjust(left=0.02, right=0.98, top=0.85, bottom=0.05, wspace=0.05)
    out = ROOT / "paper" / "thumbnail.png"
    plt.savefig(out, dpi=100, facecolor=fig.get_facecolor())
    print(f"wrote {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
