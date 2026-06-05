# ARC Prize 2026 — Paper Track Plan (Revised)

**Target prize pool:** Outstanding Papers (>4.5/5 avg → share of $375K, multiple winners).
**Anchor code track:** ARC-AGI-3 (workspace already scaffolded; field wide open, #1 = 1.20%).
**Owner:** solo entry, local RTX 5060 8GB + Windows/PowerShell.
**Revision date:** 2026-06-05

---

## 1. Hard dates (work backward from these)

| Date | Gate | Status |
|------|------|--------|
| 2026-06-25 | P1 Kaggle submission for Milestone #1 | **IN PROGRESS** |
| **2026-06-30** | **ARC-AGI-3 Milestone #1** | Must have non-random submission on board |
| 2026-07-31 | P2 (frame-change CNN) code complete | Planned |
| 2026-08-31 | P3 (object-centric model) code complete | Planned |
| 2026-09-25 | P4 (triggered LLM) code complete | Planned |
| **2026-09-30** | **ARC-AGI-3 Milestone #2** | Paper-quality results frozen |
| 2026-10-10 | P5 (cross-benchmark universality) | Planned |
| 2026-10-25 | P6 (error analysis + ablations complete) | Planned |
| **2026-11-02** | **Code submissions close** | Final submission |
| **2026-11-08** | **Paper due** | Must match linked code |
| **2026-12-04** | **Winners announced** | — |

Today = 2026-06-05. **5 months to lock the result**, 5.5 to write the paper.

---

## 2. The rubric → how each axis is won

Six categories scored 0–5; >4.5 average means ≥27/30 with no category below ~3.5.

| Category | Score needed | Our current status | Gap analysis |
|----------|-------------|-------------------|--------------|
| **Accuracy** | ≥4 | P1 = 15.85% local (20.2% public). No private LB score yet. | Need P2+P3 to jump to 41%+. |
| **Universality** | ≥4 | P5 (MiniGrid) not started. | Must complete MiniGrid eval by Oct 10. |
| **Progress** | ≥4 | Scaling story drafted but extrapolation not data-backed. | Must add real ablation scaling from P2/P3. |
| **Theory** | ≥4 | "Offline = inductive bias" hook is strong. Citations added. | Formalize with MDL or regret bound. |
| **Completeness** | ≥4 | P1 ablations configured. Failure taxonomy done. | Need to run ablations; add per-action-type stats. |
| **Novelty** | ≥4 | Combination appears unclaimed. | Monthly lit-check; document search results in paper. |

### Current estimated score: ~2.3/5 → Target: ≥4.5/5

---

## 3. Candidate thesis (locked 2026-06-05)

> **Title:** *Offline Compositional Agents for ARC-AGI-3: Object-Centric World Models with Triggered Symbolic Reasoning*
>
> Claim: ARC-AGI-3's interactive grids reward agents that (1) build an object-permanence representation from frame deltas, (2) plan over a learned forward model with a bounded horizon, and (3) escalate to slower symbolic reasoning only when a confidence trigger fires. We achieve **X%** on the private set using a fully offline pipeline — no API calls, single 8GB GPU — and present scaling laws suggesting Y% is reachable with Z× compute.

Why this is defensible:
- The *combination* of all three components in a single agent is unclaimed as of mid-2026.
- It rides existing literature (object-centric repr, MuZero-style planning, mixture-of-experts triggering) without copying any single paper.
- "Offline" + "triggered escalation" is the **inductive bias to ARC's no-internet rule itself** — that's the novelty hook.

---

## 4. Code roadmap (paper-aware revision)

| Phase | Code goal | Local target | Paper deliverable | Deadline | Status |
|-------|-----------|-------------|-------------------|----------|--------|
| **P1** Graph exploration baseline | Port `just-explore` 3rd-place | ≥0.30 | Baseline row + game family analysis | 2026-06-25 | ✅ Shipped (15.85%) |
| **P2** Online frame-change CNN | Learn salience from frame deltas | ≥0.50 | Ablation: with/without delta salience | 2026-07-31 | 🔄 Started |
| **P3** Object-centric world model + MCTS | Learned forward model, k=4 planning | ≥0.58 | Scaling plot: score vs horizon vs capacity | 2026-08-31 | 📅 Planned |
| **P4** Triggered offline LLM reasoner | ≤3B quantized LLM, entropy gate | ≥0.65 | τ vs accuracy curve; case studies | 2026-09-25 | 📅 Planned |
| **P5** Cross-benchmark universality | MiniGrid-BabyAI eval | n/a | Universality table | 2026-10-10 | 📅 Planned |
| **P6** Error analysis + ablations | Per-game breakdown + all ablations | n/a | Completeness section | 2026-10-25 | 🔄 Ablation runner done |
| **P7** Paper write + cleanup | LaTeX + license headers | n/a | Submitted paper + repo | 2026-11-08 | 🔄 Draft ongoing |

### Key milestone: P2+P3 combine to project 41%+ level coverage (from failure-mode analysis)

---

## 5. Revised risk register

| Risk | Likelihood | Impact | Mitigation | Status |
|------|-----------|--------|------------|--------|
| 8GB VRAM caps triggered LLM | Medium | High | Pre-commit ≤3B quantized; benchmark first week of P4; 1.5B fallback | ⚠️ Monitor |
| `vendor/` gitignored blocks open-source | High | Medium | Before P7, port needed vendor code to `src/` with license headers | 📅 Planned |
| Novelty collapse (someone publishes combo) | Medium | High | Monthly lit-check; pivot trigger mechanism if needed | 🔄 Lit-search ongoing |
| Single-author Completeness ceiling | High | Medium | Exhaustive ablations + public eval-script; repo judges can run | 🔄 Ablation runner done |
| Linked submission score = 0 → Accuracy = 0 | Medium | Critical | Keep P1 submission live as fallback before each milestone | ✅ P1 ready |
| Internet-disabled eval breaks LLM cell | Medium | High | Smoke-test notebook in OFFLINE mode at end of each phase | 📅 Planned |
| **Kaggle LB metric weights env completion** | **High** | **High** | Focus P2/P3 on *finishing* environments, not just levels | 🔄 Being addressed |
| **Paper uses "we" for single author** | Low | Low | Verify pronoun choice before submission | 📅 Fix before P7 |

---

## 6. Literature search log (updated 2026-06-05)

### Must cite + differentiate from:

| Work | Key idea | How we differentiate |
|------|----------|---------------------|
| Greenblatt 2024 | GPT-4o program synthesis on ARC-AGI-1 (42%) | We use no LLM at eval (P1/P2/P3); P4 LLM is triggered, not primary. |
| ARChitects 2024 | TTT, 53.5% private on ARC-AGI-1 | No TTT; hard 9-hour budget prevents eval-time fine-tuning. |
| Akyürek et al. 2024 | TTT 47.5% on ARC-AGI-1 | Same as above. |
| Hodel DSL series | Programmatic DSL for grid transforms | We do not enumerate programs; we explore interactively. |
| MuZero / EfficientZero | Planning over learned models | We adopt bounded-horizon MCTS + object-centric slots. |
| Slot Attention (Locatello et al.) | Object-centric representation | We use this as the encoder in P3. |
| Rudakov 2025 | just-explore: 3rd place preview comp | Our P1 is a port; P2-P4 are novel extensions. |
| Chollet et al. 2026 | ARC-AGI-3 benchmark paper | We directly compare to the human baseline and frontier scores. |

### Arxiv search queries (set weekly):
- `cs.AI` + `ARC-AGI-3`
- `cs.LG` + `interactive grid reasoning`
- `cs.AI` + `abstract visual reasoning`

---

## 7. Paper structure (revised — 8 pages + appendix)

```
1. Abstract              (150 words)   ← NOW: real numbers from P1
2. Introduction          (1 page)      ← NOW: phase summary table, problem framing
3. Prior work            (1 page)      ← NOW: complete with all citations
4. Method                (2 pages)     ← NOW: all three sub-modules described
   - 4.1 Frame perception + graph induction (P1, shipped)
   - 4.2 Object-centric forward model (P3, planned)
   - 4.3 Triggered symbolic reasoner (P4, planned)
5. Experiments           (2 pages)     ← NOW: P1 table + ablations + scaling projections
   - 5.1 P1 main results (real data)
   - 5.2 Ablations (real data)
   - 5.3 Scaling to 85% (projection + plot)
   - 5.4 Universality (planned)
6. Analysis              (1 page)      ← NOW: failure clusters, cluster→phase mapping
7. Discussion            (0.5 page)    ← NOW: compute, slots, LLM scaling axes
8. Conclusion            (0.25 page)
   References (9+ entries)
   Appendix: Reproducibility, per-game scores
```

---

## 8. Revised immediate actions (June 2026)

- [x] Port `just-explore` baseline (P1) — DONE (15.85% local)
- [x] Set up `paper/` directory with LaTeX skeleton + `references.bib`
- [x] Set up `RESULTS.md` ledger
- [x] Subscribe to arxiv `cs.AI` + `cs.LG` for ARC-AGI
- [x] Create ablation experiment runner
- [ ] **Submit P1 to Kaggle before Milestone #1 (June 30)**
- [ ] Run ablation experiments (status-bar, priority groups, modality)
- [ ] Start P2 frame-change CNN implementation
- [ ] Document novelty search results (monthly, starting now)
- [ ] Fix pronoun voice in paper ("we" → "we" only if co-authors; "I" otherwise)

---

## 9. Decision log

| Date | Decision |
|------|----------|
| 2026-06-01 | Chose ARC-AGI-3 anchor over ARC-AGI-2 (wider field, more theory surface) |
| 2026-06-01 | Targeting Outstanding Pool (>4.5), not Top 3 — better solo-entry EV |
| 2026-06-01 | Thesis locked as "Offline Compositional Agents" |
| 2026-06-05 | **Revised plan after deep review** — added ablation runner, failure-mode taxonomy in paper, scaling projections, universality target date |
| 2026-06-05 | Decision: P2 (frame-change CNN) is the **highest-ROI next phase** — unblocks 13.7% of total levels (Cluster 2) |
| 2026-06-05 | Decision: Submit P1 to Kaggle *before* June 30 regardless of P2 readiness |
| 2026-06-05 | Decision: Paper draft will be maintained as both `.tex` (LaTeX) and `.md` (Kaggle-compatible Markdown) |
