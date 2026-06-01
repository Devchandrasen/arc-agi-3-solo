# ARC-AGI-3 results ledger

Score-history table for every submission and local eval. Paper Track main results
table will be assembled from this file. Entry format:

| Date | Phase | Agent | Config | Local score | Local levels | LB score | Notes | Commit |

Local = run via `src/arc_agi3_solo/eval/runner.py` against `data/environment_files/`
(25 games, 183 levels). LB = Kaggle private/public score for a submitted notebook.

## Runs

| Date | Phase | Agent | Config | Local score | Local levels | LB score | Notes | Commit |
|---|---|---|---|---|---|---|---|---|
| 2026-06-01 | P0 | RandomAgent | MAX_ACTIONS=5000 | tbd | tbd | n/a | original baseline (commit f17df1b) | f17df1b |
| 2026-06-01 | P1 | GraphAgent | MAX_ACTIONS=50, ar25 only | 0 | 0/7 | n/a | smoke test, 51 actions, clean exit | unstaged |

## Notes

- All runs use `OPERATION_MODE=competition` (no internet, matches Kaggle eval env).
- Long-form runs of GraphAgent on all 25 games should be done locally and logged here
  before any code submission.
- Once GraphAgent runs a full 7.9h budget on all 25 games, that becomes the P1 row.
