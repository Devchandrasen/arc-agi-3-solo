"""Smoke-validate submission_v1.ipynb against local env files.

Extracts code cells, patches the Kaggle wheel/env paths to local equivalents,
skips the pip install (venv already has the wheels), runs through.
Used only by `_build_submission_v1.py` author + CI; not shipped in the notebook.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB = ROOT / "notebooks" / "submission_v1.ipynb"
LOCAL_ENV_DIR = ROOT / "data" / "environment_files"

# We won't actually exec the full 25-game loop here; we'll cap to 1 game and 50 actions.
os.environ["OPERATION_MODE"] = "competition"
os.environ["ENVIRONMENTS_DIR"] = str(LOCAL_ENV_DIR)

with NB.open(encoding="utf-8") as f:
    nb = json.load(f)

cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
print(f"loaded {len(cells)} code cells")

g: dict = {"__name__": "__main__", "os": os, "sys": sys}
for i, cell in enumerate(cells):
    src = "".join(cell["source"])
    # Skip install (wheels already installed in venv)
    if "pip" in src and "wheels" in src.lower():
        print(f"  skipping cell {i}: pip install")
        continue
    # Patch the Kaggle env paths
    src = src.replace(
        '/kaggle/input/arc-prize-2026-arc-agi-3/environment_files',
        str(LOCAL_ENV_DIR).replace("\\", "/"),
    )
    # Cap the run loop so this smoke is fast
    src = src.replace(
        "TOTAL_BUDGET_S = 8.0 * 60 * 60",
        "TOTAL_BUDGET_S = 60.0",  # 1 minute total
    )
    src = src.replace(
        "for gid in games:",
        "for gid in games[:1]:",  # one game
    )
    src = src.replace(
        "GraphAgent.MAX_ACTIONS = 1_000_000",
        "GraphAgent.MAX_ACTIONS = 100",
    )
    # Redirect scorecard write away from /kaggle/working
    src = src.replace('"/kaggle/working/scorecard.json"', '"_validation_scorecard.json"')
    print(f"  executing cell {i} ({len(src.splitlines())} lines)...")
    exec(compile(src, f"<cell {i}>", "exec"), g)

print("\n[OK] submission_v1.ipynb validation completed without errors")
