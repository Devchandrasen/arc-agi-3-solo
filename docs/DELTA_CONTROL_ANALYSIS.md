# Frame-delta control-discovery analysis (P2 groundwork)

**Method:** drive a random arrow policy; for each state-changing transition,
segment before/after frames, match salient segments (color + area + nearest
centroid), and record the displacement of the largest matched moving segment.
Aggregate per-arrow displacement vectors. A clean control scheme shows up as
≥2 arrows with consistent, low-variance, distinct displacement directions.

| Game | Transitions | Move rate | Directed arrows | Control discovered? |
|---|---|---|---|---|
| re86 | 600 | 0.907 | 4 | **YES** |
| tr87 | 600 | 0.515 | 0 | no |
| wa30 | 600 | 0.848 | 4 | **YES** |

## Discovered control schemes

**wa30** — clean 4-way avatar (≈2.5px/step):
- arrow 1 → UP   (dy −2.7, std 1.5)
- arrow 2 → DOWN (dy +2.8, std 1.5)
- arrow 3 → LEFT (dx −2.5, std 1.4)
- arrow 4 → RIGHT(dx +2.1, std 1.9)
- arrow 5 → ~no-op (mag 0.5)

**re86** — clean 4-way avatar (≈1.5px/step):
- arrow 1 → UP (dy −1.26), 2 → DOWN (dy +1.44), 3 → LEFT (dx −1.5),
  4 → RIGHT (dx +1.08), 5 → ~no-op (mag 0.5)

**tr87** — no stable avatar. Only arrows 1/2 move anything, with huge variance
(std 8.8/5.5): multiple objects move per step, so "largest moving segment"
isn't a single controllable avatar. tr87 needs a different mechanic (likely
multi-object or non-spatial) — candidate for the P4 LLM reasoner, not P2.

## Implication for P2

2/3 Cluster-2 games expose a **controllable avatar with a discoverable
control scheme** — the exploitable structure the random-walk graph agent
ignores. The P2 frame-delta policy should:
1. **Calibrate** the control scheme online (correlate each arrow with avatar
   displacement over the first ~100 actions).
2. **Track** the avatar segment frame-to-frame.
3. **Navigate purposefully** — move the avatar toward salient targets /
   unexplored regions instead of stepping randomly.

This is the concrete, data-backed P2 build (supersedes the refuted
whole-frame clustering in `P2_FINDINGS.md`).
