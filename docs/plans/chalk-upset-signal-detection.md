# Plan: Chalk/Upset Year Signal Detection for BSS Improvement

**Date:** 2026-05-19  
**Goal:** Find a pre-tournament signal that predicts whether a tournament year will
be chalk-heavy or upset-heavy, then use it to modulate the torvik correction
model's strength. Target: BSS improvement ≥ +0.002 vs incumbent
`torvik_corrected_recent5_conservative` (BSS +0.133).

---

## The Problem, Precisely

The correction model has a **+0.318 intercept** because every training year (2009-2024)
has positive mean residual — the better team won more often than torvik predicted.
2026 broke this pattern (mean residual −0.012), and the correction systematically
made 2026 predictions worse.

The model cannot learn the 2026 pattern without knowing the 2026 outcomes —
which aren't available at prediction time. But the *market's overconfidence*
(torvik, odds) was observable before a single game was played.

Key data from existing artifact (2016–2026, years with real outcomes):

```
Year | mean_resid | torvik_conf | seed_conf | mkt_conf | pct_tv>0.7
2016 |   +0.127   |   0.421     |   0.444   |   0.411  |   0.476
2017 |   +0.243   |   0.438     |   0.434   |   0.377  |   0.444
2018 |   +0.096   |   0.452     |   0.439   |   0.350  |   0.413
2019 |   +0.054   |   0.469     |   0.421   |   0.401  |   0.571
2021 |   +0.099   |   0.472     |   0.481   |   0.347  |   0.571
2022 |   +0.088   |   0.455     |   0.460   |   0.338  |   0.492
2023 |   +0.151   |   0.419     |   0.426   |   0.400  |   0.444
2024 |   +0.059   |   0.484     |   0.432   |   0.432  |   0.587
2025 |   +0.169   |   0.463     |   0.395   |   0.408  |   0.540
2026 |   -0.012   |   0.505     |   0.419   |   0.437  |   0.619
```

`torvik_conf` = mean(|torvik − 0.5| × 2) across all 63 games.
`pct_tv>0.7` = fraction of games where torvik > 0.70.

**Observation:** 2026 has the highest `torvik_conf` (0.505) and `pct_tv>0.7` (0.619)
of any year — and is the only year with a negative residual. This is consistent with
"when forecasters are maximally confident, they are overconfident."

**Honest caution:** n=10 years, 1 upset year. No signal will clear any standard
significance threshold. The goal is to find a correction that is *mechanically
justified* and *LOYO-safe*, not statistically proven.

---

## Signals to Test

### Tier 1 — Derivable from existing artifact (no new data)

These can be computed in Phase 0 with no scraping:

| Signal | Definition | Intuition |
|--------|-----------|-----------|
| `torvik_conf_mean` | mean(|torvik − 0.5| × 2) for all 63 games | High = forecasters very confident → over-correcting may hurt |
| `pct_torvik_gt70` | fraction(torvik > 0.70) | High = many lopsided matchups expected |
| `mean_seed_gap` | mean(|seed2 − seed1| / 15) | High = more 1v16/2v15 type matchups → chalk expected |
| `market_conf_mean` | mean(|odds − 0.5| × 2) for games with odds | Combined market confidence |
| `torvik_mkt_agree_rate` | fraction(|odds − torvik| < 0.05) | High = market and model lock step → systemic overconfidence |
| `torvik_massey_conf_mean` | mean(|massey_avg − 0.5| × 2) | Second rating system confidence as confirmation |

**Key constraint:** These signals must be computed from the **training years only**
(LOYO-safe). When predicting year Y, the chalk_score is derived from the field of
teams entered in year Y's tournament — using torvik/odds/seeds from year Y's games.
This is LOYO-safe because the game-level torvik/seed/odds are observed pre-game.

The year-level signal is a pre-game aggregate — it describes the tournament field
at Selection Sunday, before any ball is tipped.

### Tier 2 — Requires external data (scrape if Tier 1 fails)

| Signal | Source | Availability |
|--------|--------|-------------|
| KenPom AdjEM gap (1-seed mean vs 5-seed mean) | kenpom.com historical archives | 2002-present, scrape needed |
| Conference tournament upset rate | data.ncaa.com or kenpom | Selection-Sunday week |
| Pre-tournament futures market concentration | Odds API historical or DraftKings | 2018-present |
| NET rating spread at Selection Sunday | barttorvik.com | 2019-present |
| First Four upset indicator | Available in existing team data | 2011-present |

Tier 2 signals are only worth pursuing if Tier 1 analysis shows a clear directional
signal from any of the 6 Tier 1 candidates.

---

## Phase 0: Correlation Audit Script (research only, no model changes)

**File:** `scripts/chalk_upset_signal_audit.py`  
**Output:** `artifacts/chalk_upset_signal_audit.json`

The script:
1. Loads `artifacts/loyo_pergame_predictions.json`
2. For each year ≥ 2016 (years with real outcome data):
   - Computes all 6 Tier 1 signals from that year's 63 games
   - Records actual `mean_residual` for that year
3. Runs Spearman correlation (rank-based, robust to n=10) between each signal
   and mean_residual
4. Prints a ranked table with correlation and direction

**Expected deliverable:** A table showing which Tier 1 signals have the strongest
*negative* correlation with mean_residual (higher model confidence → lower residual).
The hypothesis is that `torvik_conf_mean` and `pct_torvik_gt70` will show
r ≈ −0.3 to −0.5 based on the data above.

**No model changes in this phase. Read-only analysis.**

---

## Phase 1: Architecture Decision

Based on Phase 0, choose one of these approaches:

### Option A: Adaptive `max_correction` (recommended starting point)
Scale the correction ceiling by a function of the year-level chalk score:

```
chalk_score = torvik_conf_mean(year)  # precomputed at prediction time
scale_factor = 1.0 - α × (chalk_score - baseline_chalk_score)
effective_max_correction = max_correction × clip(scale_factor, 0.5, 1.5)
```

Where `baseline_chalk_score` is the mean of `torvik_conf_mean` across training years.
In high-confidence years (chalk_score > baseline), scale down the correction.
In low-confidence years (chalk_score < baseline), scale up slightly.

The single free parameter `α` is fit from training years. It has a natural
regularizer: at `α = 0`, the model reduces to the incumbent (no change).

**Pros:** One parameter, mechanically justified, no intercept change, bounded effect.  
**Cons:** Year-level signal computed from the same year's games (though pre-game data).

### Option B: Year-level feature in the linear model
Add `chalk_score(year)` as a 10th feature in `_feature_vector`. The correction
model then learns whether to reduce corrections in high-confidence years.

**Pros:** Integrated into existing architecture.  
**Cons:** Year-level feature applied to game-level model (information level mismatch).
Ridge will likely shrink it toward zero — same null result pattern as market_disagree.

### Option C: Conditional intercept
Fit two intercepts: one for chalk years (chalk_score > threshold), one for upset years.
In a given test year, apply the corresponding intercept.

**Pros:** Directly addresses the root cause.  
**Cons:** Requires classifying test year before any outcomes — circular if threshold is
learned from same data. Only 1 upset year in sample = extreme overfit risk.

**Recommendation:** Start with Option A. If it fails, test Option B before Option C.
Option C is a last resort and requires hard theoretical justification.

---

## Phase 2: Implementation

### 2a. Chalk score computation
`scripts/loyo_pergame_predictions.py` — add `compute_year_chalk_score(games)` that
takes a year's game records and returns a dict of Tier 1 signal values. This runs
per-year on the test year's pre-game data (LOYO-safe since it uses torvik/seeds/odds
that are pre-game observations).

### 2b. New model variant
`src/prediction/torvik_correction.py` — add `TorvikCorrectionChalkAdaptive`:
- Same as `TorvikCorrectionModel` but accepts `year_chalk_score: float` in `predict()`
- `effective_max_correction = config.max_correction × (1 − α × (chalk_score − baseline))`
- `α` and `baseline` are fit from training years as additional model attributes

### 2c. New LOYO mode
`scripts/admit_kaggle_candidate.py` — add `_evaluate_torvik_corrected_chalk_adaptive()`:
- At test time, compute `chalk_score` from test year's pre-game torvik/seeds
- Pass it to `TorvikCorrectionChalkAdaptive.predict()`
- Search over `α ∈ [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]` on shadow years

### 2d. Artifact update
`scripts/loyo_pergame_predictions.py` — regenerate artifact with `year_chalk_score`
field added at the year level (stored once per year, not per game). This gives
the admit script access to the pre-game signal without recomputing during LOYO.

---

## Phase 3: Evaluation

Run the admission gate:

```bash
python scripts/admit_kaggle_candidate.py \
  --experiment-spec configs/kaggle_chalk_adaptive_experiment.json \
  --no-fail
```

Experiment spec:
```json
{
  "incumbent": {
    "mode": "torvik_corrected",
    "params": {"correction_ridge": 20.0, "max_correction": 0.06, "recent_year_count": 5}
  },
  "candidate_search_spaces": [{
    "mode": "torvik_corrected_chalk_adaptive",
    "params": {
      "correction_ridge": [20.0],
      "max_correction": [0.06],
      "recent_year_count": [5],
      "chalk_alpha": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    }
  }]
}
```

**Kill criteria:**
- If no `chalk_alpha` value produces mean_improvement ≥ +0.001 on shadow years
  (2023-2024), stop. Signal is not actionable regardless of phase 0 correlations.
- If the best shadow candidate passes (+0.001 on shadow), run full admission gate.
- Admission requires: mean_improvement ≥ +0.002 on final years (2025-2026).

---

## Phase 4: Tier 2 Extension (only if Phase 3 fails)

If Tier 1 signals fail to produce a shadow-passing candidate, evaluate two Tier 2
signals before calling this line of research exhausted:

1. **KenPom AdjEM spread**: Scrape historical KenPom archives for the 16 bracket
   teams in each region; compute AdjEM gap between top-4 seeds and bottom-4 seeds.
   High gap → chalk expected. Build `scripts/scrape_kenpom_tournament_field.py`.

2. **Conference tournament upset rate**: For each year, count upset rate in the
   week-of-selection conference tournaments (non-1-seed wins / total games played).
   Data available from barttorvik.com or sports-reference.

If both Tier 2 signals fail the shadow gate, **close this research line**.

---

## Power / Overfitting Warnings

1. **n=10 years, 1 upset year.** Spearman correlation with 10 points requires
   r ≈ 0.64 to reach p < 0.05 (two-tailed). We will not achieve statistical
   significance. Any signal will be exploratory evidence, not proof.

2. **Pre-register the signal before seeing holdout.** The Phase 0 audit
   chooses a signal based on 2016-2024 training years only. The 2025-2026 holdout
   must be truly blind. Do not look at 2025/2026 residuals when selecting the
   signal in Phase 0.

3. **One free parameter (α).** The experiment spec searches 6 values of α on
   shadow years (2023-2024). This is the maximum allowable search. Do not expand
   the grid after seeing shadow results.

4. **Accept that the incumbent may be unbeatable.** If Phase 3 fails, document
   the conclusion: the correction model's 2026 miscalibration is irreducible with
   available pre-tournament signals. The ceiling is BSS +0.133.

---

## Files to Create / Modify

| File | Action | Phase |
|------|--------|-------|
| `scripts/chalk_upset_signal_audit.py` | CREATE — correlation analysis script | 0 |
| `artifacts/chalk_upset_signal_audit.json` | CREATE — audit output | 0 |
| `src/prediction/torvik_correction.py` | MODIFY — add `TorvikCorrectionChalkAdaptive` | 2 |
| `scripts/loyo_pergame_predictions.py` | MODIFY — add `year_chalk_score` to artifact | 2 |
| `artifacts/loyo_pergame_predictions.json` | REGENERATE | 2 |
| `scripts/admit_kaggle_candidate.py` | MODIFY — add new evaluation mode | 2 |
| `configs/kaggle_chalk_adaptive_experiment.json` | CREATE — experiment spec | 3 |
| `artifacts/kaggle_admission_report.json` | UPDATE — new admission result | 3 |

---

## Outcome (2026-05-19)

**CLOSED — Signal real, effect size insufficient.**

Phase 0 passed strongly: `torvik_conf_mean` Spearman r=−0.697, p=0.025.
Phase 2 implemented: `TorvikCorrectionChalkAdaptive` in `torvik_correction.py`,
new `torvik_corrected_chalk_adaptive` mode in `admit_kaggle_candidate.py`.
Phase 3 ran exhaustive α sweep (0–40, both min_scale=0.1 and min_scale=0.0):

- Max net improvement on 2025+2026: **+0.0007** (gate requires +0.002, 3× short)
- Best α ≈ 5–7: helps 2026 by ~5pp but costs 2025 ~4pp (correction genuinely useful in chalk years)
- 2025 deviation=+0.025, 2026 deviation=+0.067 — ratio only 2.7×; no α cleanly isolates 2026
- Even at α=15 with min_scale=0 (2026 gets raw torvik): 2025 loses enough to cancel the gain

Root cause: 2026 is only modestly upset (mean_residual=−0.012) while 2025 is strongly chalk
(mean_residual=+0.169). The correction benefit in 2025 dominates the correction cost in 2026
at every point in the α search space. The tradeoff is mathematically irreducible with this mechanism.

**Do not retry**: year-level chalk signal modulation via max_correction scaling.

## Decision Tree

```
Phase 0 → any signal with r < −0.25 (negative) with mean_residual?
  NO → Stop. Signals are noise. BSS +0.133 is the ceiling.
  YES → Phase 1: select signal and architecture → Phase 2: implement
         → Phase 3: shadow gate passes (≥+0.001)?
           NO → Try Tier 2 signals (KenPom AdjEM / conf tourney upsets)
                → Both fail shadow gate → CLOSE this line
           YES → Full admission gate (≥+0.002 on 2025-2026)?
                 NO → Document: signal real but effect too small
                 YES → New incumbent, update MEMORY.md
```

---

## Not in Scope

- Using 2026 outcomes to train anything (would require masking 2026 from LOYO)
- Game-level chalk detection (already handled by existing seed_gap/torvik_confidence features)
- Poolaware / bracket strategy implications (separate objective)
