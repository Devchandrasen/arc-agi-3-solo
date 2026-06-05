# Ablation Experiment Reference

This document defines all ablation experiments for the OCA agent, their configurations, and expected outcomes. The ablation runner is at `notebooks/_run_ablations.py`.

---

## Ablation Configurations

| # | Name | Groups | Mask SB | Modality | Description | Expected Δ |
|---|------|--------|---------|----------|-------------|-----------|
| 1 | `baseline_5groups_masked` | 5 | ✅ | all | Full P1 baseline | Baseline |
| 2 | `no_sb_masking` | 5 | ❌ | all | Skip status-bar masking | −20% |
| 3 | `single_group_uniform` | 1 | ✅ | all | No priority grouping | −30% |
| 4 | `sb_only` | 5 | ❌ | all | Only status-bar segments | −80% |
| 5 | `arrows_only` | 5 | ✅ | arrows | No click actions | −15% |
| 6 | `clicks_only` | 5 | ✅ | clicks | No arrow actions | −60% |

---

## Running Ablations

```powershell
# Default (all 6 configs, 300s per game)
python notebooks/_run_ablations.py

# Fast mode for smoke testing (30s per game)
$env:PER_GAME_BUDGET_S = "30"
python notebooks/_run_ablations.py

# Override output directory
$env:RUNS_DIR = "runs/ablations_debug"
python notebooks/_run_ablations.py
```

## Output

```
runs/ablations/
  ablation_run.log                    — full log
  ablation_baseline_5groups_masked.json   — per-game results per config
  ablation_no_sb_masking.json
  ablation_single_group_uniform.json
  ablation_sb_only.json
  ablation_arrows_only.json
  ablation_clicks_only.json
  ablation_summary.md                 — markdown table for paper
```

---

## Adding a New Ablation

1. Add a config dict to `ABLATIONS` list in `_run_ablations.py`
2. If the config requires new agent behavior, add a method to `AblationGraphAgent`
3. Run and re-generate the summary

---

## Interpretation Guide

- **Status-bar masking ablation**: Measures false-positive rate on UI elements. If removing masking barely changes score, the status-bar detector is too aggressive.
- **Single-group ablation**: Measures the value of priority ordering. If large drop, action ordering matters.
- **Arrow-only / click-only**: Measures which action modality drives progress in which game families. Critical for deciding P2/P3 architecture.
