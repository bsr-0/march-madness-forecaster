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

| Task | Effort | Dependencies | Acceptance |
|------|--------|-------------|------------|
| O16: Add 2027 to TOURNAMENT_START_DATES + test gate | 2 hrs | None | Test fails loudly for missing year |
| O2: Validate local four-factors vs Torvik | 4 hrs | None | Per-season r >= 0.99, no systematic bias |
| O5: MC sim count stability | 2 hrs | None | 3x identical inputs -> identical top-20 order |
| Fix 2 remaining E8 data errors (2023, 2024) | 1 hr | Manual result lookup | Zero `build_actual_outcome` warnings |

---

### Phase 1: Pool-Specific Opponent Model (Gap 1, ~3 days)

**Goal:** Replace generic ESPN opponent model with one trained on real pool behavior.

**Step 1.1: Analyze pool bracket data (1 day)**
- Load 105 real brackets from `data/pool_history/pool_hist_results.json`
- Measure: per-game pick distributions, chalk bias by round, correlation structure, favorite teams/upsets
- Compare pool distributions vs ESPN national distributions
- Identify: where does THIS pool diverge most from the national average?
- Output: diagnostic report with per-round pick heatmaps

**Step 1.2: Build pool-calibrated opponent generator (1 day)**
- Module: enhance `src/simulation/pool_history_opponent_model.py`
- For LOYO year Y, train on pool brackets from years != Y
- Generate synthetic opponent brackets that match the pool's empirical pick distribution, chalk bias, and correlation structure
- Replace `generate_opponent_brackets()` in poolaware dispatch with pool-calibrated generator

**Step 1.3: Backtest and compare (1 day)**
- Run full 15-year LOYO with pool-calibrated opponents vs current ESPN-based opponents
- Acceptance gate: P(1st) improvement in >= 8/15 years
- If accepted: update production config. If rejected: document and move on.

**Key risk:** Only 4 years of pool data (2023-2026). The pool-specific model may overfit to a small sample. Mitigation: blend pool empirical rates with ESPN national rates (e.g., 70/30) rather than replacing entirely.

---

### Phase 2: Regime-Adaptive Candidate Generation (Gap 2, ~2 days)

**Goal:** Use the pre-tournament field volatility signal to skew the candidate pool toward the predicted regime.

**Step 2.1: Validate the volatility signal (0.5 day)**
- `compute_field_volatility_signal()` already exists in `src/data/features/custom_ratings.py`
- Validate: does the signal computed at Selection Sunday predict whether the F4 has low-seeds (chaos) or all 1-seeds (chalk)?
- Measure: correlation between signal and actual F4 seed-mean across 15 years
- If r < 0.4: skip this phase (signal too weak to act on)

**Step 2.2: Regime-conditional candidate generation (1 day)**
- When volatility signal predicts chalk (signal > 0.65): generate more low-risk candidates (risk 0.1-0.3), fewer high-risk
- When volatility signal predicts chaos (signal < 0.45): generate more high-risk candidates (risk 0.7-0.9), add exhaustive_champion variants with non-1-seed champions
- Mixed regime: keep current balanced sweep
- Implementation: adjust the `_pa_risk_levels` tuple in poolaware dispatch based on volatility signal

**Step 2.3: Backtest (0.5 day)**
- Compare regime-adaptive poolaware vs current uniform-risk poolaware
- Acceptance gate: P(1st) improvement in >= 8/15 years

**Key risk:** Overfitting a regime classifier to 15 data points. Mitigation: use a simple threshold (not ML), keep the current balanced candidates as fallback, only SKEW the distribution rather than eliminating candidates.

---

### Phase 3: Champion Pick Improvement (Gap 3, ~2 days, SPECULATIVE)

**Goal:** Improve champion accuracy from 6/14 (43%) toward 8/14 (57%).

**Step 3.1: Champion feature analysis (0.5 day)**
- For each of 15 years, catalog the actual champion's pre-tournament profile: barthag rank, conference strength, tournament experience, injury status, momentum, draw difficulty
- Identify: what features distinguish the actual champion from the other 1-seeds?
- If no consistent signal: stop here (confirms it's unpredictable)

**Step 3.2: Champion classifier (1 day)**
- Only if Step 3.1 finds signal
- Simple model (logistic regression) trained on 1-seed features to predict which 1-seed wins
- Walk-forward: for year Y, train on years < Y
- Output: P(champion) for each 1-seed, used as weight for forced-champion candidates in poolaware

**Step 3.3: Integration and backtest (0.5 day)**
- Weight forced-champion candidates by classifier confidence rather than equal weight
- Backtest against uniform-weight baseline

**Key risk:** The project already proved "champion pick is ~random among 1-seeds." This phase has the highest probability of producing a null result. Only pursue if Phase 1 and 2 are complete.

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

| Phase | Target | Dependencies | Expected P(1st) |
|-------|--------|-------------|-----------------|
| Phase 0 | Pre-season (anytime) | None | 11.2% (no change, foundation) |
| Phase 1 | Early off-season | Phase 0 | 12-13% (if pool bias is exploitable) |
| Phase 2 | Mid off-season | Phase 0 | 11.5-12% (if volatility signal is predictive) |
| Phase 3 | Late off-season | Phase 1+2 done | Speculative (may be null) |
| Phase 4 | Anytime | None | 11.2-11.5% (low-effort diversity) |

**Production target for 2027:** 12%+ P(1st) via Phase 1 (opponent model) + Phase 2 (regime adaptation). This would represent a 4x improvement over the seed baseline (3.1%) and the best-known result for a single-entry bracket in a 30-person pool.

---

## Decision Framework

After each phase, apply the acceptance gate:
- **P(1st) improves in >= 8/15 years** -> Accept, integrate into production
- **P(1st) improves in 5-7/15 years** -> Investigate year-by-year; accept if gains are in regime-appropriate years
- **P(1st) improves in < 5/15 years** -> Reject, document, move on

Never optimize MeanRank, P(top25%), or MeanScore — they don't pay out in winner-take-all.
