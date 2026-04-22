# Rank Stability Comparison: 5K Shape vs 20K Team-Identity

## Context

Council report 2026-04-19 (selection-problem) Action #1: test whether the MC
ranker's top-5 bracket selection is stable across independent RNG seeds. If
unstable, the ranker selects from noise and the fix is variance reduction.

## Baseline: 5K tournaments, shape-encoded scoring

Config: `n_tournaments=5000`, `n_opponents=200`, shape-encoded `score_brackets_against_outcome`,
50 `f4_first_tv` brackets, 10 ranking runs per year.

Artifact: `artifacts/rank_stability_check_2026-04-19.json`

| Year | Jaccard | Spearman | Always top-5 | Verdict |
|------|---------|----------|-------------|---------|
| 2023 | 0.296 | 0.530 | 1/5 | UNSTABLE |
| 2024 | 0.175 | 0.560 | 0/5 | UNSTABLE |
| 2025 | 0.140 | 0.438 | 0/5 | UNSTABLE |
| 2026 | 0.150 | 0.525 | 0/5 | UNSTABLE |
| **Agg** | **0.190** | **0.513** | | **UNSTABLE** |

## After: 20K tournaments, team-identity scoring

Config: `n_tournaments=20000`, `n_opponents=200`, team-identity `score_brackets_team_identity`
with pre-decoded bracket picks, 50 `f4_first_tv` brackets, 10 ranking runs per year.

Artifact: `artifacts/rank_stability_check_20k_ti_2026-04-19.json`

| Year | Jaccard | Spearman | Always top-5 | Verdict |
|------|---------|----------|-------------|---------|
| 2023 | 0.606 | 0.828 | 2/5 | STABLE |
| 2024 | 0.480 | 0.752 | 0/5 | MARGINAL |
| 2025 | 0.504 | 0.781 | 2/5 | MARGINAL |
| 2026 | 0.489 | 0.820 | 0/5 | MARGINAL |
| **Agg** | **0.520** | **0.795** | | **MARGINAL** |

## Deltas

| Metric | 5K shape | 20K TI | Delta |
|--------|----------|--------|-------|
| Jaccard | 0.190 | 0.520 | +174% |
| Spearman | 0.513 | 0.795 | +55% |
| Verdict | UNSTABLE | MARGINAL | improved |

## Interpretation

Two changes were applied simultaneously: 4x more tournament simulations (variance
reduction) and team-identity scoring (correct ESPN payout rules per O26/O27). Both
contribute:

- **Variance reduction** (5K to 20K): SE drops from ~0.35% to ~0.17%, making the
  ~1% P(rank=1) spread across 50 brackets resolvable.
- **Team-identity scoring**: recovers ~0.25 Spearman (per O26-G1), which means the
  ranking surface has more real structure to resolve.

The aggregate verdict improved from UNSTABLE to MARGINAL. 2023 reached STABLE;
the remaining years are on the MARGINAL/STABLE boundary (Jaccard 0.48-0.50,
gate is 0.60). The remaining variance is primarily from opponent resampling
(synthetic opponents are regenerated per ranking run).

## Selection reframe: observed opponents (council Action #2)

Replaces synthetic opponents with actual pool brackets from
`pool_hist_results.json`. The opponent field is fixed — the only remaining
variance source is tournament outcome simulation.

| Year | Jaccard | Spearman | Always top-5 | Verdict |
|------|---------|----------|-------------|---------|
| 2023 | 0.881 | 0.980 | 4/5 | STABLE |
| 2024 | 0.830 | 0.975 | 4/5 | STABLE |
| 2025 | 0.844 | 0.979 | 4/5 | STABLE |
| 2026 | 0.933 | 0.983 | 4/5 | STABLE |
| **Agg** | **0.872** | **0.979** | | **STABLE** |

## Full comparison

| Config | Jaccard | Spearman | Verdict |
|--------|---------|----------|---------|
| 5K shape, synthetic opponents | 0.190 | 0.513 | UNSTABLE |
| 20K TI, synthetic opponents | 0.520 | 0.795 | MARGINAL |
| 20K TI, observed opponents | **0.872** | **0.979** | **STABLE** |

## Root cause decomposition

The instability had two sources:
1. **Tournament outcome variance** — fixed by raising n_tournaments 5K→20K
2. **Opponent-sampling variance** — fixed by using observed pool brackets

The opponent-sampling variance was the dominant source. With observed opponents,
the ranker consistently identifies the same top-5 brackets across 10 independent
RNG seeds. 4 of 5 top brackets are identical in every single run for all 4 years.

## Operational implication for 2027

**ESPN pools do NOT show opponent brackets before lock.** Picks are private
until the tournament starts. The observed-opponent mode cannot be used for
pre-submission selection in the ESPN Tournament Challenge format.

The observed-opponent results are still valuable for two purposes:
1. **Retroactive validation** — scoring system brackets against the actual pool
   post-tournament to measure how well the ranker would have done.
2. **Diagnosis** — proving that opponent-sampling variance (not tournament
   simulation variance) was the dominant instability source. This means the
   production ranker's quality depends heavily on how well the synthetic
   opponent model (ESPN national 60/30/10) approximates the actual pool.

For pre-submission selection, the system must use synthetic opponents derived
from ESPN national public pick percentages. The MARGINAL verdict (Jaccard 0.52,
Spearman 0.80) is the realistic operating point for 2027.
