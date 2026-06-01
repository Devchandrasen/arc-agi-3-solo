"""Generate notebooks/submission_v1.ipynb from src/arc_agi3_solo/.

Inlines the GraphExplorer, FrameProcessor, and GraphAgent source into a
single, self-contained submission notebook. Re-run this script whenever
those source files change.

    python notebooks/_build_submission_v1.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "arc_agi3_solo"
OUT = ROOT / "notebooks" / "submission_v1.ipynb"


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def code(lines: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines.splitlines(keepends=True),
    }


def md(lines: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": lines.splitlines(keepends=True),
    }


def strip_module_header(src: str) -> str:
    """Drop top-of-file docstring + `from __future__` lines when inlining."""
    out = []
    started = False
    skip_docstring = True
    in_docstring = False
    for line in src.splitlines(keepends=True):
        if not started and line.startswith('"""') and skip_docstring:
            in_docstring = not in_docstring
            if in_docstring is False:
                skip_docstring = False
            elif line.count('"""') >= 2:
                skip_docstring = False
                in_docstring = False
            continue
        if in_docstring:
            continue
        if line.startswith("from __future__"):
            continue
        if not started and not line.strip():
            continue
        started = True
        out.append(line)
    return "".join(out)


GE = strip_module_header(_read(SRC / "core" / "graph_explorer.py"))
FP = strip_module_header(_read(SRC / "core" / "frame_processor.py"))
GA_RAW = _read(SRC / "agents" / "graph_agent.py")
# In the notebook, `Agent` is defined inline (not imported from our package),
# and FrameProcessor / GraphExplorer are also inline. Rewrite imports.
GA = strip_module_header(GA_RAW)
GA = GA.replace(
    "from arc_agi3_solo.agents.base import Agent\n", "",
).replace(
    "from arc_agi3_solo.core.frame_processor import FrameProcessor\n", "",
).replace(
    "from arc_agi3_solo.core.graph_explorer import GraphExplorer\n", "",
)

AGENT_BASE = '''import logging, time
from abc import ABC, abstractmethod
from typing import Optional
from arc_agi import EnvironmentWrapper

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger()

class Agent(ABC):
    MAX_ACTIONS = 5000
    def __init__(self, card_id, game_id, agent_name, arc_env, tags=None, ROOT_URL="", record=False):
        self.card_id = card_id
        self.game_id = game_id
        self.agent_name = agent_name
        self.arc_env = arc_env
        self.tags = tags or []
        self.frames = [FrameData(levels_completed=0)]
        self.action_counter = 0
        self.timer = 0.0
        self.guid = ""
    @property
    def name(self): return f"{self.game_id}.{self.__class__.__name__.lower()}"
    @property
    def state(self): return self.frames[-1].state
    @property
    def levels_completed(self): return int(self.frames[-1].levels_completed)
    @property
    def seconds(self): return round(time.time() - self.timer, 2)
    @property
    def fps(self):
        if self.action_counter == 0: return 0.0
        return round(self.action_counter / max(self.seconds, 0.1), 2)
    def main(self):
        self.timer = time.time()
        while not self.is_done(self.frames, self.frames[-1]) and self.action_counter <= self.MAX_ACTIONS:
            action = self.choose_action(self.frames, self.frames[-1])
            frame = self._step(action)
            if frame is not None:
                self._append(frame)
            self.action_counter += 1
    def _step(self, action):
        try:
            data = action.action_data.model_dump()
            raw = self.arc_env.step(action, data=data, reasoning=data.get("reasoning", {}))
        except Exception:
            log.exception("step failed for %s", action.name)
            return None
        if raw is None: return None
        return FrameData(
            game_id=raw.game_id, frame=[a.tolist() for a in raw.frame], state=raw.state,
            levels_completed=raw.levels_completed, win_levels=raw.win_levels,
            guid=raw.guid, full_reset=raw.full_reset, available_actions=raw.available_actions,
        )
    def _append(self, frame):
        self.frames.append(frame)
        if frame.guid:
            self.guid = frame.guid
    @abstractmethod
    def is_done(self, frames, latest_frame): ...
    @abstractmethod
    def choose_action(self, frames, latest_frame): ...
'''

RUN_CELL = '''import json, time
from pathlib import Path

ENV_DIR = Path(os.environ["ENVIRONMENTS_DIR"])
games = sorted({p.name for p in ENV_DIR.iterdir()
                if p.is_dir() and any(level.is_dir() for level in p.iterdir())})
print(f"Discovered {len(games)} games")

# Per-game wall-clock budget. Kaggle ceiling = 9h; reserve 1h for setup/teardown.
TOTAL_BUDGET_S = 8.0 * 60 * 60
PER_GAME_S = TOTAL_BUDGET_S / max(len(games), 1)
GraphAgent.MAX_ACTIONS = 1_000_000
GraphAgent.TOTAL_TIME_BUDGET_S = PER_GAME_S

arc = Arcade()
card_id = arc.open_scorecard(tags=["graph", "v1"])
log.info(f"scorecard {card_id}, mode {arc.operation_mode}, per-game budget {PER_GAME_S/60:.1f} min")

t0 = time.time()
for gid in games:
    try:
        env = arc.make(gid, scorecard_id=card_id)
        if env is None:
            log.warning("env is None for %s; skipping", gid); continue
        agent = GraphAgent(card_id=card_id, game_id=gid, agent_name="graph", arc_env=env, tags=["graph", "v1"])
        agent.main()
        log.info(f"{gid}: levels {agent.frames[-1].levels_completed}, actions {agent.action_counter}, total {time.time()-t0:.0f}s")
    except Exception:
        log.exception(f"{gid} crashed; continuing")

scorecard = arc.close_scorecard(card_id)
dumped = scorecard.model_dump()
print(f"=== Final: {dumped['total_levels_completed']}/{dumped['total_levels']} levels, "
      f"{dumped['total_environments_completed']}/{dumped['total_environments']} envs, "
      f"{dumped['total_actions']} actions ===")
'''

DUMP_CELL = '''with open("/kaggle/working/scorecard.json", "w") as f:
    json.dump(dumped, f, indent=2, default=str)
print("wrote /kaggle/working/scorecard.json")
for env in dumped["environments"]:
    run = env["runs"][0] if env["runs"] else {}
    print(f"{env['id']:>22} | levels {env['levels_completed']:>2}/{env['level_count']:>2} | actions {env['actions']:>5} | resets {env['resets']}")
'''

INSTALL_CELL = '''import subprocess, sys, glob, os, fnmatch

# The exact mount path of the competition data under /kaggle/input can vary,
# so discover the wheel dir at runtime instead of hardcoding it.
def _find_file_dir(root, pattern):
    if not os.path.isdir(root):
        return None
    for dirpath, _dirs, files in os.walk(root):
        if any(fnmatch.fnmatch(f, pattern) for f in files):
            return dirpath
    return None

WHEEL_DIR = _find_file_dir("/kaggle/input", "arc_agi-*.whl")
print("WHEEL_DIR =", WHEEL_DIR)

try:
    import arc_agi  # already importable?
    print("arc_agi already importable; skipping install")
except ImportError:
    if WHEEL_DIR is None:
        raise RuntimeError(
            "Could not locate arc_agi wheels under /kaggle/input. "
            "Is the arc-prize-2026-arc-agi-3 competition data attached?"
        )
    wheels = sorted(glob.glob(os.path.join(WHEEL_DIR, "*.whl")))
    print(f"Found {len(wheels)} wheels in {WHEEL_DIR}")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--quiet", "--no-index",
        "--find-links", WHEEL_DIR, "arc-agi", "arcengine",
    ])
    print("install ok")
'''

OFFLINE_CELL = '''import os

# Discover the environment_files dir (per-game subdirs with metadata.json)
# wherever the competition data mounted under /kaggle/input.
def _find_env_dir(root):
    if not os.path.isdir(root):
        return None
    for dirpath, dirs, _files in os.walk(root):
        if os.path.basename(dirpath) == "environment_files":
            return dirpath
    return None

ENV_DIR = (
    _find_env_dir("/kaggle/input")
    or "/kaggle/input/arc-prize-2026-arc-agi-3/environment_files"
)
os.environ["OPERATION_MODE"] = "competition"
os.environ["ENVIRONMENTS_DIR"] = ENV_DIR
print("ENVIRONMENTS_DIR =", ENV_DIR, "exists:", os.path.isdir(ENV_DIR))

from arc_agi import Arcade
from arcengine import FrameData, FrameDataRaw, GameAction, GameState
print("arc_agi imported, mode =", os.environ["OPERATION_MODE"])
'''


nb = {
    "cells": [
        md(
            "# ARC-AGI-3 Solo — Submission v1 (graph exploration)\n\n"
            "Phase 1 baseline. Frame-graph exploration agent ported from the dolphin-in-a-coma "
            "`just-explore` 3rd-place strategy: hash every frame, treat actions as edges, BFS to "
            "the nearest open frontier with a 5-group priority over action candidates.\n\n"
            "Source modules live under `src/arc_agi3_solo/` in the repo and are mirrored inline "
            "here so the notebook is self-contained on Kaggle.\n\n"
            "License of ported code: MIT (Rudakov 2025, see vendor/just-explore/LICENSE)."
        ),
        md("## 1. Install vendored wheels"),
        code(INSTALL_CELL),
        md("## 2. Force COMPETITION (offline) mode"),
        code(OFFLINE_CELL),
        md("## 3. Inlined source: Agent base, GraphExplorer, FrameProcessor, GraphAgent"),
        code(AGENT_BASE),
        code(GE),
        code(FP),
        code(GA),
        md("## 4. Run graph agent across all games"),
        code(RUN_CELL),
        md("## 5. Dump scorecard"),
        code(DUMP_CELL),
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"wrote {OUT}")
