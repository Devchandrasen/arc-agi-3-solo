# ARC-AGI-3 Solo

Kaggle ARC Prize 2026 — ARC-AGI-3 entry. Build agents that play interactive grid-world games with no instructions.

- Comp page: https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3
- Deadline: 2026-11-02 (final), Milestone 1: 2026-06-30, Milestone 2: 2026-09-30
- Prize pool: $850K (ARC-AGI-3 alone)
- Frontier AI baseline: 0.51%, human: 100%

**Also entered:** [ARC Prize 2026 Paper Track](https://www.kaggle.com/competitions/arc-prize-2026-paper-track) (juried, $450K pool). Plan in [docs/PAPER_TRACK_PLAN.md](docs/PAPER_TRACK_PLAN.md); draft in [paper/](paper/).

## Phases

| Phase | Approach | Target | Status |
|-------|----------|--------|--------|
| 0 | Scaffold + random baseline | submit | done (f17df1b) |
| 1 | Graph exploration (port of dolphin-in-a-coma 3rd-place) | ≥0.30 local | code shipped (ce21323); eval running |
| 2 | Online frame-change CNN | ≥0.50 local | pending |
| 3 | Object-centric world model + planning | ≥0.58 local | pending |
| 4 | Small offline LLM as triggered reasoner | ≥0.62 local | pending |
| 5 | Cross-benchmark universality + ablations | n/a | pending |
| 6 | Paper write + open-source cleanup | submission | pending |

Score history: [RESULTS.md](RESULTS.md).

## Layout

```
src/arc_agi3_solo/    # our agent code
  agents/             # agent implementations
  eval/               # offline runner / harness
  core/               # shared utilities (frame hashing, segmentation, ...)
notebooks/            # Kaggle submission notebooks
tests/                # pytest
docs/                 # API docs, design notes
vendor/               # cloned reference repos (gitignored)
data/                 # Kaggle dataset dump (gitignored)
```

## Local setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pytest
```

## Kaggle submission

- `notebooks/submission_v0.ipynb` — random baseline (Phase 0).
- `notebooks/submission_v1.ipynb` — graph-exploration baseline (Phase 1). Self-contained: `Agent` base, `GraphExplorer`, `FrameProcessor`, and `GraphAgent` are inlined. Regenerate from `src/` via `python notebooks/_build_submission_v1.py`.

Attach the `arc-prize-2026-arc-agi-3` competition dataset, run all cells, commit, submit.

## Local offline eval

```powershell
$env:OPERATION_MODE = "offline"
$env:ENVIRONMENTS_DIR = ".\data\environment_files"
$env:PER_GAME_BUDGET_S = "300"
python notebooks/_local_eval_p1.py
```

Writes `runs/p1_local_eval.{log,json}` with per-game scores.

## Reference repos (in `vendor/`)

- `arc-agi-3-agents/` — official agent framework (`arcprize/ARC-AGI-3-Agents`)
- `just-explore/` — 3rd-place preview-comp graph exploration agent (`dolphin-in-a-coma/arc-agi-3-just-explore`)

See [docs/agent_api.md](docs/agent_api.md) for the agent interface.
