# Gap Analysis & Phased Roadmap for 2027 Season

**Date:** 2026-05-03
**Baseline:** `meta_region_poolaware` at 11.2% P(1st) on corrected data (15-year LOYO, N=30 pool, team-identity scoring)

---

## Identified Gaps

### Gap 1: Pool-Specific Opponent Modeling (HIGH IMPACT)

**What successful projects do:** Tools like TeamRankings and Fantasy Cruncher explicitly model the opponent field's tendencies and optimize "probability of beating everyone" rather than "expected points." They learn which games opponents cluster on, where the field is chalk-heavy, and where contrarian picks have the highest EV.

**What this repo does:** The poolaware selector simulates against opponents, but the opponent model uses generic ESPN national pick rates + seed-based correlation. It does not learn THIS pool's specific biases from the 105 real brackets (2023-2026, 18/25/32/30 entries).

**Why it matters:** The council unanimously agreed this is the #1 untapped axis. The opponent model is the second term in P(my score > max(opponent scores)) and is currently the least calibrated component. Cross-year behavioral model infrastructure exists (`src/simulation/pool_history_opponent_model.py`) but is a fallback, not the primary opponent source.

**Evidence:** The First Principles advisor's framing: "Stop asking 'which picks are most likely correct?' Start asking 'which picks, conditional on the tournament outcome distribution, maximize the probability of beating THIS field?'"

---

### Gap 2: Year-Regime Adaptation (MEDIUM IMPACT)

**What successful projects do:** No public tool does this well, but the concept is established: chalk years and chaos years require different strategies. A bracket optimized for a chalk year (all 1-seeds to F4) performs terribly in a chaos year (2023: FDU over Purdue, Princeton over Arizona) and vice versa.

**What this repo does:** The poolaware selector implicitly adapts by choosing among ~25 candidates with different risk levels (0.1-0.9). But the risk dimension is a construction parameter, not informed by any pre-tournament regime signal. The `compute_field_volatility_signal()` function exists and correctly distinguishes chalk (2025: 0.744) from volatile (2026: 0.528) fields, but it is unused.

**Why it matters:** In chalk years, the pool is competitive (many people pick correctly), so differentiation is hard. In chaos years, most brackets collapse, so a bracket that gets the right upsets has huge separation. The candidate pool should be skewed toward the predicted regime.

**Evidence:** Tournament oracle analysis shows ranker gap correlates with regime (chaos years: +820 torvik gap; chalk years: near zero). The field volatility signal has r=-0.668 with F4 seed-mean (p=0.006, n=15).

---

### Gap 3: Champion Pick Specialization (MEDIUM IMPACT, HIGH DIFFICULTY)

**What successful projects do:** Most tools pick the highest-rated 1-seed as champion and move on. No public project has solved champion prediction OOS.

**What this repo does:** The system picks a 1-seed as champion 13/14 times. Champion accuracy is 6/14 (43%) via poolaware's multi-candidate approach. The champion diagnostic proved that champion pick is the binding P(1st) constraint.

**Why it matters:** The session summary notes "going from 2/14 to 4/14 champion accuracy would likely push P(1st) above 12%." Each correct champion pick is worth 320 points of direct scoring plus indirect value from path-consistent later rounds.

**Evidence:** Highest-barthag 1-seed is champion only 2/14 times. The signal that determines WHICH 1-seed wins is not captured by any rating system (schedule strength, injury luck, matchup-specific advantages all matter).

**Caveat:** This may be fundamentally unpredictable. The project has already confirmed "champion pick is ~random among 1-seeds." Further investment here is speculative.

---

### Gap 4: Market-Implied Probabilities as Primary Input (LOW IMPACT)

**What successful projects do:** FiveThirtyEight and KenPom-derived tools used betting market lines (futures, game spreads) as the primary probability backbone rather than pure rating systems. Markets aggregate information from many models + insider knowledge.

**What this repo does:** Unified odds data exists (87K games, 19 seasons) and Vegas R1 was tested as a GBM feature, but torvik barthag remains the primary probability source. The odds data has never been used as the primary round_probs input to construction.

**Why it matters:** Markets are theoretically better probability estimators than any single rating system. However, the BSS=0 ceiling suggests that no public probability source (including markets) meaningfully outperforms seed-implied probabilities in tournament prediction. This gap may be theoretical rather than practical.

**Evidence:** Vegas R1 as a GBM feature produced identical brackets to the base model. Market efficiency likely means this gap cannot be exploited.

---

### Gap 5: Upset Detector as Poolaware Candidate (CLOSED - NO IMPACT)

**Status:** Tested and killed 2026-05-03. Calibrated upset probability adjustments (merged 8-feature detector+specialist model, R32 boost for 12v4 and 11v3, 2 boost modes x 3 bases x 3 risks = ~18 candidates) produced 10.93% P(1st) WITH upset candidates vs 11.20% WITHOUT. Upset candidate selected in only 1/15 years (2011) and hurt P(1st) by 2pp. Extra candidates add selection noise. The "killed" label from `meta_region_upset` (7.9%) stands.

---

## Phased Roadmap

### Phase 0: Foundation (Pre-season, ~1 day)

**Goal:** Ensure infrastructure is clean and validated before building on it.

| Task | Effort | Dependencies | Acceptance | Status |
|------|--------|-------------|------------|--------|
| O16: Add 2027 to TOURNAMENT_START_DATES + test gate | 2 hrs | None | Test fails loudly for missing year | **DONE** — 2027-03-16 added, test gate updated |
| O2: Validate local four-factors vs Torvik | 4 hrs | None | Per-season r >= 0.99, no systematic bias | **CLOSED** — Torvik overlay in production, r≈0.97 at data-source ceiling, 4 tests in test_validate_four_factors.py |
| O5: MC sim count stability | 2 hrs | None | 3x identical inputs -> identical top-20 order | **CLOSED** — n_tournaments=5000 locked, TestRankStability proves 3-run identity |
| Fix tournament data errors (2023, 2024) | 1 hr | Manual result lookup | Zero `build_actual_outcome` warnings | **DONE** — 6 games fixed in 2023 (4 S16 + 1 E8 + 1 score), 5 games fixed in 2024 (1 R64 + 2 R32 + 1 S16 + 1 E8) |

---

### Phase 1: Pool-Specific Opponent Model (Gap 1) — REJECTED 2026-05-04

**Goal:** Replace generic ESPN opponent model with one trained on real pool behavior.

**Result:** Tested blend weights 0.0–0.7. All degrade P(1st) monotonically.

| Blend Weight | P(1st) | vs Baseline (11.2%) |
|--------------|--------|---------------------|
| Baseline (`pool`) | **11.20%** | — |
| 0.0 (pure ESPN, n=30) | 9.60% | -1.6pp |
| 0.1 | 8.67% | -2.5pp |
| 0.2 | 8.13% | -3.1pp |
| 0.3 | 7.20% | -4.0pp |
| 0.7 | 4.53% | -6.7pp |

**Root cause:** The baseline's opponent model for pool years (2023-2026) uses **actual year-specific pool brackets** — the synthetic model can't beat literal ground truth. For non-pool years, ESPN national picks are already closer to actual pool behavior than the cross-year behavioral model.

**Bug fixed along the way:** `build_pool_behavioral_model()` had a compounding bug — it computed per-game conditional win rates instead of round advancement probabilities, producing near-uniform late-round distributions. Fixed by direct counting of per-seed advancement rates from pool data. The model now correctly captures pool chalk bias (0.976 vs actual 0.978 for CHAMP), but this isn't enough to overcome the accuracy gap vs year-specific data.

**Key finding:** This pool is only ~8pp more chalky than ESPN national average in late rounds. The signal is real but too small to exploit with a seed-level generalization model. Year-specific ESPN picks already capture most team-level preferences.

**Infrastructure preserved:** `build_blended_pool_opponent_model()`, `--opponent pool_calibrated` CLI, `--pool-blend-weight` arg, 8 unit tests. Available for future use if more pool years accumulate.

---

### Phase 2: Regime-Adaptive Candidate Generation (Gap 2) — REJECTED 2026-05-04

**Goal:** Use the pre-tournament field volatility signal to skew the candidate pool toward the predicted regime.

**Result:** Implemented regime-conditional risk levels (chalk: 0.1-0.5, chaos: 0.5-0.9, mixed: 0.1-0.9) + forced 2-seed champions in chaos years. Signal validated (r=-0.668, p=0.006) and correctly classifies years (3 chalk, 7 chaos, 5 mixed). But P(1st) = 10.80% vs baseline 11.20%.

| Metric | Value |
|--------|-------|
| Years improved | 2/15 |
| Years regressed | 3/15 |
| Years tied | 10/15 |
| Acceptance gate (≥8/15) | **FAIL** |

**Root cause:** The MC selector (200 trials) already implicitly adapts to the regime by selecting whichever candidate happens to score highest against simulated tournament outcomes. Skewing the candidate pool doesn't provide better options because the uniform sweep already covers the optimal risk level for each year. The 2024 regression (-6pp) suggests that removing low-risk candidates in "chaos" years can backfire when the actual tournament has mixed outcomes.

**Lesson:** The poolaware selector's existing architecture (diverse candidates + MC selection) is already regime-adaptive by construction. Explicit regime conditioning adds complexity without benefit. The candidate diversity at uniform risk levels is sufficient.

---

### Phase 3: Champion Pick Improvement (Gap 3) — REJECTED 2026-05-04

**Goal:** Improve champion accuracy from 6/14 (43%) toward 8/14 (57%).

**Step 3.1 result (diagnostic):** Barthag rank among 1-seeds is nearly uniform for actual champions: #1=3/11, #2=4/11, #3=3/11, #4=1/11. **No predictable signal.** Champion selection among 1-seeds is confirmed random w.r.t. pre-tournament ratings.

**Step 3.2 (diversity approach):** Since no classifier signal exists, tested adding champion-variant diversity instead: 4×3=12 forced-champion candidates (4 one-seeds × 3 risk levels: 0.3/0.5/0.7) + 4 massey-based champion candidates. Total 16 champion variants (vs current 4).

**Result:** P(1st) = 11.33% vs 11.20% baseline (+0.13pp). Only 2/15 years improved, 2/15 regressed. **Neutral — no meaningful improvement.**

**Root cause:** The MC selector already evaluates exhaustive_champion candidates (which try all 64 teams). Adding more forced-champion variants at different risk levels doesn't produce meaningfully different brackets because the non-champion picks in the bracket (R64-E8) are what matter most for scoring, and those are already well-covered by the risk-sweep candidates.

**Lesson:** Champion pick among 1-seeds is genuinely unpredictable from available pre-tournament data. The 43% accuracy (6/14) is near the ceiling for this signal environment. Improvement requires new data sources (injury reports, late-season momentum, betting market futures) not currently in the pipeline.

---

### Phase 4: Candidate Diversity Expansion (Gap 2 supplement, ~1 day)

**Goal:** Add more probability base variants to the poolaware candidate pool.

| Candidate type | Effort | Rationale |
|----------------|--------|-----------|
| 70/30 torvik/massey blend | 5 min | `blend` selected 7/14 years — more ratios may help |
| 60/40 torvik/massey blend | 5 min | Same |
| Walk-forward massey_best system | 2 hrs | Use historically best-performing Massey system per year |
| Market-implied round probs (unified odds) | 4 hrs | Gap 4 — use odds as a probability base, not just a feature |

**Implementation:** Add new entries to `_pa_prob_bases` list in the poolaware dispatch. Each new base automatically crosses with all risk levels and construction modes. Dedup handles identical brackets.

**Backtest:** Run 15-year LOOY. Accept if P(1st) improves in >= 8/15 years.

---

## Timeline

| Phase | Target | Dependencies | Expected P(1st) | Status |
|-------|--------|-------------|-----------------|--------|
| Phase 0 | Pre-season | None | 11.2% (foundation) | **DONE** 2026-05-04 |
| Phase 1 | Early off-season | Phase 0 | 12-13% (if pool bias exploitable) | **REJECTED** 2026-05-04 — opponent model already near-optimal |
| Phase 2 | Mid off-season | Phase 0 | 11.5-12% (if volatility signal predictive) | **REJECTED** 2026-05-04 — MC selector already implicitly adapts |
| Phase 3 | Late off-season | Phase 2 done | Speculative (may be null) | **REJECTED** 2026-05-04 — champion among 1-seeds is random |
| Phase 4 | Anytime | None | 11.2-11.5% (low-effort diversity) | **REJECTED** 2026-05-04 — more candidates = more selection noise |

**Production target for 2027:** 12%+ P(1st) via Phase 2 (regime adaptation) + Phase 4 (candidate diversity). Phase 1 (opponent model) proved the existing model is already near-optimal.

---

## Decision Framework

After each phase, apply the acceptance gate:
- **P(1st) improves in >= 8/15 years** -> Accept, integrate into production
- **P(1st) improves in 5-7/15 years** -> Investigate year-by-year; accept if gains are in regime-appropriate years
- **P(1st) improves in < 5/15 years** -> Reject, document, move on

Never optimize MeanRank, P(top25%), or MeanScore — they don't pay out in winner-take-all.
