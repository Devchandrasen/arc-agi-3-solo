# ARC-AGI-3 Solo

Kaggle ARC Prize 2026 — ARC-AGI-3 entry. Build agents that play interactive grid-world games with no instructions.

- Comp page: https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3
- Deadline: 2026-11-02 (final), Milestone 1: 2026-06-30, Milestone 2: 2026-09-30
- Prize pool: $850K (ARC-AGI-3 alone)
- Frontier AI baseline: 0.51%, human: 100%

## Phases

| Phase | Approach | Target | Status |
|-------|----------|--------|--------|
| 0 | Scaffold + random baseline | submit | in progress |
| 1 | Graph exploration (port of dolphin-in-a-coma 3rd-place) | ≥0.30 local | pending |
| 2 | Online frame-change CNN | ≥0.50 local | pending |
| 3 | Object-centric world model + planning | ≥0.58 local | pending |
| 4 | Small offline LLM as triggered reasoner | ≥0.62 local | pending |
| 5 | Optimization & error analysis | submission | pending |

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

Open `notebooks/submission_v0.ipynb`, attach the `arc-prize-2026-arc-agi-3` competition dataset, run all cells, commit, submit.

## Reference repos (in `vendor/`)

- `arc-agi-3-agents/` — official agent framework (`arcprize/ARC-AGI-3-Agents`)
- `just-explore/` — 3rd-place preview-comp graph exploration agent (`dolphin-in-a-coma/arc-agi-3-just-explore`)

See [docs/agent_api.md](docs/agent_api.md) for the agent interface.
