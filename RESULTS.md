# ARC-AGI-3 results ledger

Score-history table for every submission and local eval. Paper Track main results
table will be assembled from this file. Local = run via the offline runner
against `data/environment_files/` (25 games, 183 levels). LB = Kaggle private/
public score for a submitted notebook.

## Runs

| Date | Phase | Agent | Config | Local levels | LB score | Commit |
|---|---|---|---|---|---|---|
| 2026-06-01 | P0 | RandomAgent | 1500 actions/game | 0/183 (0%) | n/a | f17df1b |
| 2026-06-01 | P1 | GraphAgent | 5 min/game, OFFLINE mode | **29/183 (15.85%)** | n/a | 6276f99 |

## Phase 1 — full eval breakdown

Run: `runs/p1_local_eval.json`, scorecard `48886404-5240-408b-9250-2d8b58249fc5`.
Total wall: 7504s (~2h05m). Total actions: 667417 (~89 actions/sec average).

| Game | Levels | Total | % | Actions | Wall s |
|---|---|---|---|---|---|
| ar25 | 2 | 8 | 25.0% | 35141 | 300.0 |
| bp35 | 1 | 9 | 11.1% | 17671 | 300.1 |
| cd82 | 2 | 6 | 33.3% | 37780 | 300.0 |
| cn04 | 0 | 6 | 0.0% | 21418 | 300.0 |
| dc22 | 2 | 6 | 33.3% | 34502 | 300.1 |
| ft09 | 1 | 6 | 16.7% | 23751 | 300.0 |
| g50t | 0 | 7 | 0.0% | 34668 | 300.1 |
| ka59 | 1 | 7 | 14.3% | 24534 | 300.1 |
| lf52 | 2 | 10 | 20.0% | 35086 | 300.2 |
| lp85 | 2 | 8 | 25.0% | 19190 | 300.1 |
| ls20 | 1 | 7 | 14.3% | 23267 | 300.5 |
| m0r0 | 1 | 6 | 16.7% | 15502 | 300.0 |
| r11l | 1 | 6 | 16.7% | 33897 | 300.1 |
| re86 | 0 | 8 | 0.0% | 16945 | 300.1 |
| s5i5 | 1 | 8 | 12.5% | 30893 | 300.0 |
| sb26 | 1 | 8 | 12.5% | 33811 | 300.1 |
| sc25 | 0 | 6 | 0.0% | 24710 | 300.1 |
| sk48 | 1 | 8 | 12.5% | 41578 | 300.1 |
| sp80 | 1 | 6 | 16.7% | 17532 | 300.1 |
| su15 | 0 | 9 | 0.0% | 14075 | 300.1 |
| tn36 | 1 | 7 | 14.3% | 31256 | 300.1 |
| tr87 | 0 | 6 | 0.0% | 14773 | 300.8 |
| tu93 | **4** | 9 | **44.4%** | 31913 | 300.6 |
| vc33 | **4** | 7 | **57.1%** | 33412 | 300.1 |
| wa30 | 0 | 9 | 0.0% | 20137 | 300.1 |
| **Total** | **29** | **183** | **15.85%** | **667417** | **7504** |

### Observations
- 19/25 games scored ≥1 level (~76% game coverage). 6/25 scored 0.
- Top scorers: vc33 (57.1%), tu93 (44.4%), cd82 (33.3%), dc22 (33.3%).
- Zero-scorers (cn04, g50t, re86, sc25, su15, tr87, wa30) are candidates for
  per-game failure-mode analysis in §Analysis of the paper. Hypotheses to
  test: more aggressive (a) status-bar masking, (b) salient-color heuristics,
  or (c) longer per-game budget.
- 0/25 environments fully completed — every game still has unsolved levels.
  P2-P4 work targets the harder levels, not just more levels of the same kind.

### Comparison to public leaderboard
Live ARC-AGI-3 LB as of 2026-06-01 (top 5):
1. Tufa Labs — 1.20
2. Redfield Rentals — 0.68
3. Barada Sahu — 0.66
4. Kevin E R MILLE — 0.66
5. SVG — 0.65

The Kaggle LB scoring metric weights per-environment completion (envs fully
solved), which is why top scores hover near 1% while our level-completion
rate is much higher. We currently have 0/25 envs fully completed (same as
the top LB entries, presumably). To translate our 15.85% level rate into an
LB number, we need a Kaggle submission run — see notebook v1. The LB ranking
will depend on which envs we partially solve vs which the leaders do.

## Notes
- All runs use `OPERATION_MODE=offline` for local testing (no network).
- The Kaggle submission notebook uses `OPERATION_MODE=competition` (the
  Kaggle eval env has the live API server on its own subnet).
- Future P2-P4 phases should add their rows above; do not delete past rows.
