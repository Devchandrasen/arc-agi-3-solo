# ARC Prize 2026 — Paper Track Plan

**Target prize pool:** Outstanding Papers (>4.5/5 avg → share of $375K, multiple winners).
**Anchor code track:** ARC-AGI-3 (workspace already scaffolded; field wide open, #1 = 1.20%).
**Owner:** solo entry, local RTX 5060 8GB + Windows/PowerShell.

---

## 1. Hard dates (work backward from these)

| Date | Gate |
|---|---|
| 2026-06-30 | ARC-AGI-3 Milestone #1 — must have a non-random submission on the board |
| 2026-09-30 | ARC-AGI-3 Milestone #2 — paper-quality results frozen |
| 2026-11-02 | Code submissions close |
| 2026-11-08 | Paper due |
| 2026-12-04 | Winners announced |

Today = 2026-06-01. **5 months to lock in the result for the paper**, 5.5 to write it.

## 2. The rubric → how each axis is won

Six categories scored 0–5; >4.5 average means at minimum 27/30 with no category below ~3.5.

| Category | What scores 5 | Our angle |
|---|---|---|
| **Accuracy** | Top-tier leaderboard score | Aim for **top-5 on ARC-AGI-3** (≥0.65 currently). Stretch: beat 1.20. |
| **Universality** | Method generalizes beyond ARC | Show same architecture solving (a) ARC-AGI-3 unseen games (b) a non-ARC test (e.g. MiniGrid or Atari-100k subset). One extra benchmark is enough. |
| **Progress** | Plausible path to 85% | Explicit scaling story: parameter count, compute, data axes — what would push score 10×. Don't hand-wave. |
| **Theory** | Says *why* it works, not what | Frame as inductive bias matching task statistics (object permanence + bounded planning horizon). Cite cognitive priors. |
| **Completeness** | Ablations, error analysis, failure modes | Per-game score table, per-action-type accuracy, ablation of each component, named failure modes with examples. |
| **Novelty** | Not in any public repo | The *combination* — frame-change CNN + object world model + triggered offline LLM reasoner under no-internet — appears to be unclaimed. Verify by lit search. |

## 3. Candidate thesis (refine before writing code)

> **Working title:** *Offline Compositional Agents for ARC-AGI-3: Object-Centric World Models with Triggered Symbolic Reasoning*
>
> Claim: ARC-AGI-3's interactive grids reward agents that (1) build an object-permanence representation from frame deltas, (2) plan over a learned forward model with a bounded horizon, and (3) escalate to slower symbolic reasoning only when a confidence trigger fires. We achieve **X%** on the private set using a fully offline pipeline — no API calls, single 8GB GPU — and present scaling laws suggesting Y% is reachable with Z× compute.

Why this is defensible:
- The *combination* of all three components in a single agent isn't published as of mid-2026 (verify in §6).
- It rides existing literature (object-centric repr, MuZero-style planning, mixture-of-experts triggering) without copying any single paper.
- "Offline" + "triggered escalation" is the **inductive bias to ARC's no-internet rule itself** — that's the novelty hook judges remember.

## 4. Code roadmap (paper-aware version of your 5-phase plan)

Each phase below now has a **paper-deliverable** column: what graph/table/ablation it produces.

| Phase | Code goal | Local target | Paper deliverable | Deadline |
|---|---|---|---|---|
| **P1** Graph exploration baseline (port `just-explore`) | Reproduce 3rd-place preview score | ≥0.30 | Baseline row in main results table + analysis of which game families it solves | 2026-06-25 (before M1) | **DONE: 29/183 = 15.85% local.** |
| **P2a** ~~Whole-frame state clustering~~ | ~~Collapse aliased frames~~ | — | Negative-result ablation row | done | **REFUTED — see `docs/P2_FINDINGS.md`. State spaces are genuinely large, not pixel-inflated; clustering can't help. Code kept (`core/novelty.py`).** |
| **P2** Online frame-**delta** salience model | Characterize *what each action changes* (region/size/color of the frame delta); bias action selection toward "progress-like" deltas | ≥0.50 | Ablation: delta-salience on/off; per-game delta-type histogram | 2026-07-31 | Pivoted from clustering per P2a finding. |
| **P3** Object-centric world model + bounded MCTS | Learned forward model, plan k=4 steps | ≥0.58 | Scaling plot: score vs planning horizon, score vs model size | 2026-08-31 |
| **P4** Triggered offline LLM reasoner | Small (≤3B) quantized LLM called only when policy entropy > τ | ≥0.65 | Trigger-rate vs accuracy tradeoff curve; case studies of triggered solves | 2026-09-25 (before M2) |
| **P5** Cross-benchmark universality | Run same agent on MiniGrid-BabyAI or similar | n/a | Universality table (rubric axis #2) | 2026-10-10 |
| **P6** Error analysis + ablations | Per-game breakdown, named failure modes | n/a | Completeness section material | 2026-10-25 |
| **P7** Paper write + open-source cleanup | LaTeX + license headers | n/a | Submitted paper + repo | 2026-11-08 |

**Code submission to anchor the paper** = whichever Phase produces the best private score by 2026-11-02. The paper's linked submission doesn't need to be #1; it needs to *exist and match the paper*.

## 5. Risks and pre-mitigations

| Risk | Mitigation |
|---|---|
| 8GB VRAM caps the triggered LLM | Pre-commit to ≤3B params quantized (Qwen2.5-3B-Q4, Phi-3-mini-Q4); benchmark inference latency on RTX 5060 in Phase 4 week 1. Fall back to a 1.5B if needed. |
| `vendor/` is gitignored but paper rubric expects open-source artifacts | Before P7, port any vendored code we actually depend on into `src/` with proper attribution + Apache-2.0/GPLv3 headers. |
| Novelty collapses (someone publishes the same combo) | Monthly lit-check: arxiv-sanity + Twitter (`#ARCprize`, `#ARCAGI3`). If overlap appears, pivot the "triggering" mechanism to something more specific (e.g. uncertainty-gated program synthesis instead of LLM). |
| Single-author paper hits Completeness ceiling | Compensate with exhaustive ablations + a public eval-script repo that judges can run. Reproducibility is the leverage. |
| Score on linked submission is 0 → Accuracy = 0 | Keep the Phase 1 graph-explore submission live on the board as a fallback before each milestone, even after newer phases are running. |
| Internet-disabled Kaggle eval breaks LLM cell | Smoke-test the submission notebook in OFFLINE mode (set `OPERATION_MODE=competition`, unplug, run end-to-end) at end of each phase. |

## 6. Required lit-search before P3 starts (by 2026-07-15)

Must cite + differentiate from:
- Greenblatt 2024 (GPT-4o program synthesis on ARC-AGI-1, 42%)
- The ARChitects 2024 (TTT, 53.5% private on ARC-AGI-1)
- Akyürek et al. 2024 (TTT 47.5%)
- Hodel DSL line of work
- MuZero / EfficientZero (planning over learned models)
- Slot Attention / object-centric repr literature (Locatello et al.)
- Any 2026 ARC-AGI-3 preprints — set a weekly arxiv alert for `ARC-AGI-3`, `interactive grid reasoning`

## 7. Paper structure (target: 8 pages + appendix)

Hard rule from the rubric brief: *"shorter and clearer is always better. No filler, no unnecessary equations."*

```
1. Abstract              (150 words)
2. Introduction          (1 page)    — why ARC, why interactive, what we claim
3. Prior work            (1 page)    — TTT line, program synthesis line, offline RL line; differentiate
4. Method                (2 pages)   — architecture diagram + 3 sub-modules
5. Experiments           (2 pages)   — main table, ablations, scaling plots, universality
6. Analysis              (1 page)    — failure modes, trigger behaviour, qualitative cases
7. Discussion            (0.5 page)  — path to 85%, what's missing
8. Conclusion            (0.25 page)
   References, appendix (reproducibility, hyperparams, per-game scores)
```

## 8. Immediate next actions (this week)

1. Port `just-explore` baseline (P1 start) — primary engineering task.
2. Open `paper/` directory with a LaTeX skeleton + `references.bib` so writing accretes from day 1, not at the end.
3. Set up a `RESULTS.md` ledger: every submission's date, score, config hash, notebook commit — this becomes the main results table.
4. Subscribe to arxiv `cs.AI` + `cs.LG` filtered for "ARC-AGI" weekly.

## 9. Decision log

- 2026-06-01: Chose ARC-AGI-3 anchor over ARC-AGI-2 (wider field, existing workspace, more theory surface).
- 2026-06-01: Targeting Outstanding Pool (>4.5), not Top 3 — better expected value for a solo entry.
- 2026-06-01: Thesis locked tentatively as "Offline Compositional Agents" — revisit after P3 results.
