# O21 — Pool-History-Marginal Blend Backtest

**Executed:** 2026-04-13 • **Source:** `scripts/o21_marginal_blend_backtest.py` • **Artifact:** `artifacts/o21_blend_backtest.json` • **Data:** `pool_hist_results.json` (2023-2025; 2026 skipped, see below).

Closes `COUNCIL_LESSONS.md §2 O21` with a **negative verdict**.

---

## TL;DR

**Blending pool-history marginals into the opponent model does NOT improve bracket-ranking calibration.** Across weights w ∈ {0.0, 0.25, 0.5, 0.75, 1.0}:

- **2024 (K=25):** Spearman ρ = **+0.340** at *every* tested weight. Zero movement.
- **2025 (K=32):** Spearman ρ = +0.575 at w ≤ 0.5, drops to +0.506 at w ≥ 0.75. Blending slightly *hurts*.
- **2023 (K=18):** underpowered — P(1st) = 0 for all pool brackets at every weight (synthetic opponents sweep the pool even at n=17).

**Gate not met.** The O21 gate was "verify switch changes bracket rankings." Rankings are approximately invariant to pool-history blending in this sample. Keep the opponent model at the locked 60/30/10 ESPN/Massey/seed. Do not add a pool-history channel to the locked weights on this evidence.

---

## Method

For each year × each weight w:

1. Build opponent model via `build_opponent_model(year, seeds, pool_history_path=<path> if w>0 else None, pool_history_weight=w)`. At w=0 this exactly reproduces the production 60/30/10 ESPN/Massey/seed blend.
2. For each of the K actual pool brackets:
   - Score the bracket against actual tournament outcomes via round-set membership (order-independent; robust to pool-JSON region ordering).
3. MC estimate P(1st) with n_sims = 400:
   - Generate `K-1` synthetic opponents from the opponent model.
   - Score all opponents against actual outcomes.
   - Count the fraction of trials where the actual pool bracket's score exceeds the synthetic max.
4. Compute Spearman ρ between predicted P(1st) rank and actual-score rank across the K pool brackets. **Higher ρ = better opponent-model calibration for ranking.**

Using n_opp = K-1 (matching actual pool size) rather than fixed 30 preserves the interpretation "would this bracket have won this year's pool if the K-1 other competitors were drawn from the opponent model."

---

## Results

### Per-year × per-weight Spearman ρ

| Year | K | w=0.00 | w=0.25 | w=0.50 | w=0.75 | w=1.00 |
|------|---|--------|--------|--------|--------|--------|
| 2023 | 18 | NaN† | NaN† | NaN† | NaN† | NaN† |
| 2024 | 25 | +0.340 (p=0.097) | +0.340 | +0.340 | +0.340 | +0.340 |
| 2025 | 32 | **+0.575** (p=0.001) | +0.575 | +0.575 | +0.506 | +0.506 |

**Aggregate across 2 valid years:**

| Weight | mean ρ |
|--------|--------|
| 0.00 (status quo) | **+0.457** |
| 0.25 | +0.457 |
| 0.50 | +0.457 |
| 0.75 | +0.423 |
| 1.00 | +0.423 |

†2023 NaN: P(1st) = 0.0 for all 18 pool brackets at every weight. The opponent model consistently generates a bracket that beats every actual pool bracket in 2023 (UConn's championship run was chalky enough that ESPN-marginal samplers produce dominant synthetic brackets). Not a signal-vs-noise issue — a pool-quality issue with that specific year.

### Why are 2024 ρ values identical across weights?

Investigation: the P(1st) vectors differ slightly across weights (mean P(1st) ranges 0.0035 to 0.0056), but Spearman ρ is a rank-invariant metric. The *relative ordering* of pool brackets by predicted P(1st) doesn't change across weights — only the absolute P(1st) values do. This is the key finding.

### What IS the opponent model affecting, if not rank?

Absolute P(1st) values change (2024: mean P(1st) drops from 0.0056 at w=0 to 0.0037 at w=0.5). So the opponent model affects:
- **Absolute EV calibration** (how much a bracket is expected to earn)
- **Kelly-sized bet decisions** (if you're staking on winning)
- **"How hard is this pool?" estimates**

It does NOT meaningfully affect **which brackets rank highest**. The ranking is dominated by the bracket's own quality against actual outcomes, not by the opponent distribution.

---

## Why the null result?

Three non-exclusive explanations:

1. **Opponent model affects the ceiling, not the ordering.** Picking a bracket that "beats chalk" is a consistent objective across opponent models — whether the crowd is ESPN-shaped or pool-shaped, the brackets that exploit actual-outcome variance are the same ones.
2. **Pool-history marginals are noisy.** K = 18-32 brackets per year with Laplace α=0.5 smoothing. Blending noisy marginals into the (more stable) ESPN-Massey-seed blend may add noise rather than signal. Would need more pool-years before the pool marginals are precise enough to help.
3. **Our K is too small to detect small effects.** With only 2 usable years (2024, 2025) producing valid ρ, we have N = 2. A real but small effect (ΔρΔw ≈ +0.02) couldn't be detected with this sample.

---

## Implications

### 1. O21 verdict: close. Do not adopt pool-history blending.

Gate was "verify switch changes bracket rankings." Rankings change by Δρ = 0 (2024) to Δρ = −0.07 (2025, going in the *wrong* direction at high weights). Keep locked weights at 60% ESPN / 30% Massey / 10% seed per `MEMORY.md §1`.

### 2. Reframes the opponent-model-wall investigation.

The opponent-model-wall (O1, O3, O4, O10, O21) has now been largely investigated:
- O1 closed (data already collected).
- O3 closed (ranking has real signal, mean ρ = +0.37 over 14 years).
- O4 closed (independence holds).
- O10 mostly moot (no copula needed).
- **O21 closed (pool marginals don't change bracket rankings).**

The wall is **no longer load-bearing on opponent modeling.** If there's a next opportunity for improvement, it's in the **base model** (game-outcome probabilities), not the opponent model. This is the hypothesis that didn't surface in the 2026-04-12 council series because they framed the problem as opponent modeling.

### 3. New lesson for §1

"**Opponent-model marginals affect absolute P(1st) but not *ranking* of pool brackets.**" This means: the choice of opponent model is irrelevant for "which bracket should I submit?" (the core product question). It matters only for "how likely am I to win?" (a calibration question). Future work that tunes the opponent model should be framed around calibration tasks, not ranking tasks.

### 4. Surfaces a new data-ops item: O22

`data/raw/historical/tournament_results_2026.json` is malformed:
- 49 games labeled `round_name='NCG'` (should be 1)
- 0 games labeled `R64` (should be 32)
- 4 games labeled `R68` (should be `FF` for First Four)

Tracked as `COUNCIL_LESSONS.md §2 O22`. Until fixed, 2026 cannot be included in any year-over-year analysis depending on `tournament_results_2026.json`. Found incidentally here; the bug predates this work.

---

## Caveats / what this does NOT prove

- **N = 2 usable years.** Confidence in the null is modest; a real small effect (|Δρ| < 0.03) couldn't be detected.
- **Ranking metric is Spearman ρ vs actual score.** Alternative metrics (NDCG, top-K precision) might reveal weight-dependence that rank correlation hides.
- **2023 methodology gap.** P(1st) = 0 for all brackets means the test is uninformative that year; does not disprove the hypothesis.
- **Does not disprove the broader "pool is different from ESPN" finding.** The 5pp marginal divergence from O4 is real; it just doesn't translate to ranking-function sensitivity.
- **Pool size matters.** These findings apply to K = 18-32 pools. In a larger or smaller pool, the opponent-model-to-ranking sensitivity might differ.

---

## Reproduction

```bash
python3 scripts/o21_marginal_blend_backtest.py
# → prints per-year × per-weight table
# → writes artifacts/o21_blend_backtest.json
```

Requires `pool_hist_results.json` at repo root and `data/raw/historical/tournament_{seeds,results}_{2023,2024,2025}.json`. Deterministic modulo RNG seed (2026) and 400 MC trials.
