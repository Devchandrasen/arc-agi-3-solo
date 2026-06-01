# Phase 2 findings — state-space clustering does NOT help Cluster-2 games

**Verdict:** Negative result. Whole-frame clustering (random-CNN embedding +
online nearest-cluster assignment) does not improve scores on the three
Cluster-2 games (re86, tr87, wa30). All three stayed at **0 levels**, same
as Phase 1. The premise behind the approach is refuted by the data.

## What we tried

### v0 — global mean-pool embedding (commit 1c…)
- Frozen random-init CNN, 32-D embedding via **global mean pool**, fixed
  `eps = 0.025` L2 threshold on the cluster map.
- Result on the 3 games: 0 / 0 / 0 levels.
- Collapse ratios: re86 122×, tr87 **2682×**, wa30 **2517×**.
- **Diagnosis:** global mean pooling is position-invariant. Frames that
  differ only in *where* an object is map to nearly identical embeddings,
  so thousands of distinct states collapse into 1-2 clusters. The graph
  becomes a single node — useless.

### v0.1 — coarse 4×4 spatial-pool embedding (commit this one)
- Replaced global mean pool with a 4×4 grid pool → 512-D embedding that
  preserves approximate object position.
- Re-profiled pairwise embedding distances on 80 sampled frames per game:

  | Game | dim | p10 | p25 | median | p75 | max |
  |---|---|---|---|---|---|---|
  | re86 | 512 | 0.117 | 0.149 | 0.171 | 0.195 | 0.236 |
  | tr87 | 512 | 0.025 | 0.036 | 0.045 | 0.052 | 0.070 |
  | wa30 | 512 | 0.035 | 0.048 | 0.059 | 0.066 | 0.109 |

## Why clustering is the wrong lever

Two findings kill the approach:

1. **Per-game distance scales vary 3-4×** (re86 median 0.171 vs tr87 0.045).
   No single global `eps` can work; it over-merges tight games and
   under-merges spread ones. Adaptive per-game `eps` could fix *this*, but…

2. **There is no within-cluster / between-cluster gap.** The pairwise
   distance distributions are smooth and unimodal (p10-to-max spans a
   factor of ~2 with no bimodality). That means the frames don't lie on a
   low-dimensional state manifold with tight clusters — they are *genuinely
   distinct states*. The Cluster-2 hypothesis ("10k frame hashes but ~100
   real game states") is **false for these games**: the state space really
   is large.

If the state space is genuinely large, collapsing it either (a) loses real
distinctions (over-merge → 0 levels, as in v0) or (b) doesn't collapse at
all (under-merge → same as P1). There is no eps that helps.

## What the data says the real P2 should be

The failure-mode analysis (`docs/FAILURE_MODES.md`) was more specific than
"frame-change CNN": Cluster-2 games need **salience over arrow
*consequences*** — the agent must learn *which action changes the
important part of the frame*, not merely enumerate states faster.

That points at a **frame-delta** model, not a whole-frame state model:
- Compute the per-pixel delta between consecutive frames after each action.
- Characterize *what kind* of change each action causes (which region,
  how large, what color transitions).
- Bias action selection toward actions whose deltas resemble
  "progress-like" changes (e.g. movement of a salient object, not flicker
  of a status bar).

This is genuinely "online frame-change CNN" in the spirit of the original
plan — but operating on *deltas between frames* (the change signal), which
is the part the whole-frame embedder threw away.

## Decision

- Keep the v0.1 embedder + cluster map in the tree (`core/novelty.py`,
  `agents/cnn_graph_agent.py`) — they're tested, documented, and the
  negative result is paper material (Novelty + Theory + Completeness rubric
  axes all reward a well-characterized failed hypothesis).
- **Pivot P2** from state-clustering to frame-delta action-salience. Tracked
  in `docs/PAPER_TRACK_PLAN.md`.
- The paper's §Experiments ablation table gains a row: "state clustering
  (v0/v0.1): no gain — state space is genuinely large, not pixel-inflated."
