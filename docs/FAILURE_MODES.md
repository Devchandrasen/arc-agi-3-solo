# Failure-mode analysis — Phase 1 zero-scoring games

Instrumented 5-min re-runs of `GraphAgent` on the 7 games that scored 0 in
the P1 main eval. Goal: identify *why* each game is hard, so the failure
modes inform which P2-P4 phase should fix which cluster.

## Per-game instrumented stats

| Game | Uniq frames | Transitions | Trans. rate | Game-overs | L0 revisits | Clicks | Arrows | Avail. acts | ~Segments |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cn04 | 8 798 | 15 726 | 82 % | 253 | 266 | 10 277 | 8 708 | 6 | ~10 |
| g50t | **139** | 18 143 | 70 % | 197 | **3 750** | 0 | 25 685 | 5 | ~15 |
| re86 | 10 672 | 18 534 | **98 %** | 186 | 192 | 0 | 18 636 | 5 | ~25 |
| sc25 | 5 492 | 12 187 | 54 % | 250 | 536 | 18 580 | 3 783 | 5 | ~25 |
| su15 | 5 392 | 14 490 | 33 % | **908** | 1 792 | 42 848 | 248 | **2** | ~25 |
| tr87 | 10 148 | 16 711 | **99 %** | 130 | 135 | 0 | 16 712 | 4 | ~60 |
| wa30 | 10 021 | 17 627 | 82 % | 106 | 112 | 0 | 21 265 | 5 | ~10 |

(`su15` solved 1 level in this re-run; main eval got 0. The per-game results
are noisy on the borderline games — the bucket assignment below is the
signal that's robust.)

## Three failure clusters

### Cluster 1 — Cycle trap (g50t)
- **Signature:** tiny state space (139 unique frames) + huge L0-revisit count
  (3 750, ~21 % of every transition lands back at the level-start frame) +
  arrows-only action set (0 clicks).
- **Diagnosis:** The 5-priority bucketing collapses for arrow-only games —
  every arrow goes into group 0, the graph BFS exhausts a small reachable
  set quickly, and there's no mechanism to escape. The game likely requires
  *sequencing* or *holding* arrow inputs (multi-step commitments) which our
  single-step graph doesn't model.
- **Which phase fixes it:** **P4 (triggered offline LLM reasoner).** When
  the agent detects BFS exhaustion + repeated L0-revisits, hand the frame
  history to the LLM to spot the structural pattern that requires a
  multi-step sequence.

### Cluster 2 — Wide-but-arrows (re86, tr87, wa30)
- **Signature:** huge state space (10 k+ unique frames) + very high
  transition rate (98–99 % on re86/tr87) + few game-overs (≤200) +
  arrows-only.
- **Diagnosis:** The agent discovers thousands of distinct states but
  *cannot find a level-completing one* in 5 min. Random-edge-BFS is the
  bottleneck — these games have rich state spaces with sparse rewards, and
  the agent needs *salience over arrow consequences* (which arrow moves the
  important object vs which moves a distractor).
- **Which phase fixes it:** **P2 (online frame-change CNN).** A CNN trained
  on frame-deltas labels which arrow caused which kind of change; the agent
  can then prefer arrows that change more.

### Cluster 3 — Death-trap click-heavy (cn04, sc25, su15)
- **Signature:** moderate state space (5–9 k frames) + lots of game-overs
  (250–908) + lots of clicks (10–43 k). High reset rate (~1 reset per
  50 actions on su15).
- **Diagnosis:** The agent *sees* actionable UI (high click count) but
  clicking the wrong segment kills it. Status-bar masking + salient-color
  priority isn't enough — some segments are "safe" and some are "deadly"
  and the agent has no way to learn the difference.
- **Which phase fixes it:** **P3 (object-centric world model).** A learned
  forward model predicts whether a clicked object leads to GAME_OVER, and
  the planner avoids those segments.

## Implications for the plan

| Phase | Plan-original target | Refined by this analysis |
|---|---|---|
| P2 (frame-change CNN) | "≥0.50 local" | Should specifically unblock Cluster 2 (re86, tr87, wa30 = 3 games, 25 levels = 13.7 % of total levels) |
| P3 (object-centric world model + planning) | "≥0.58 local" | Should unblock Cluster 3 (cn04, sc25, su15 = 3 games, 21 levels = 11.5 %) |
| P4 (triggered offline LLM reasoner) | "≥0.62 local" | Should handle Cluster 1 + the tail of cluster 2/3 |

If P2 unblocks Cluster 2 fully, the level rate jumps to (29 + 25) / 183 =
**29.5 %** without any other change. P3 on top would add Cluster 3:
(29 + 25 + 21) / 183 = **41.0 %**. These are upper bounds (assume full
clearance of the cluster), but they show the per-phase return on
engineering investment.

## Open data

- `runs/failure_modes/<game_id>.json` — per-game instrumented stats
- `runs/failure_modes/<game_id>_first_frame.npy` — level-0 frame snapshot
  used as the visual anchor for each game's failure mode
- `runs/failure_modes/summary.json` — all 7 games combined
