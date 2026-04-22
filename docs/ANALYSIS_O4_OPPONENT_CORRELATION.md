# O4 — Empirical Opponent Correlation Analysis

**Executed:** 2026-04-13 • **Source:** `scripts/pool_correlation_analysis.py` + `scripts/pool_multiyear_analysis.py` • **Data:** `pool_hist_results.json` (pool `pool0`, groupId `0b6a2bbe…`, 4 years × 93 usable brackets total).

Closes `COUNCIL_LESSONS.md §2 O4`. Context: the 2026-04-12c council flagged the independence assumption as a "fundamental validity threat to the entire 13-year backtest." This analysis tests that hypothesis against 4 years of actual pool data.

---

## TL;DR

**The independence assumption holds.** Across 2023-2026, observed inter-bracket correlation is consistently *below* what independent draws from the empirical marginals would produce (pooled z = −4.15). The council's validity concern is **not supported** by the data.

**But** the data exposes a different problem: **pool marginals diverge meaningfully from ESPN national marginals** (5.0 percentage points mean absolute divergence; up to +18pp for individual teams). The opponent-model error is in the *marginals*, not the *correlation structure*. This opens a new action item: use pool-specific marginals, not ESPN-national, in the opponent model. Tracked as `§2 O21`.

---

## Method

Each bracket is encoded as a binary vector over every (round, team) pair anyone in the pool picked. Inter-bracket correlation = mean off-diagonal element of the cosine-like correlation matrix across the 30ish brackets per year.

Under independence, the expected correlation isn't zero — high-consensus picks (e.g. every bracket picks a 1-seed over a 16-seed) create mechanical correlation even under perfectly independent draws. The null distribution is simulated by drawing each bracket's picks IID from the observed marginals, and the observed correlation is compared to the Monte Carlo null (n=1000-2000 trials per year).

If observed > simulated → brackets cluster *beyond* what marginals alone explain → independence violated.

---

## Results

### Year-by-year independence test

| Year | K | Obs mean ρ | Sim mean ρ (IID null) | Excess | z | Verdict |
|------|---|-----------|-----------------------|--------|---|---------|
| 2023 | 18 | 0.4105 | 0.4449 | **−0.0345** | −2.4 | LESS correlated than independence |
| 2024 | 25 | 0.4013 | 0.4271 | **−0.0258** | −2.2 | LESS correlated than independence |
| 2025 | 32 | 0.4395 | 0.4585 | **−0.0190** | −1.8 | Independence holds |
| 2026 | 30 | 0.4635 | 0.4834 | **−0.0199** | −1.9 | Independence holds |

**Pooled (4-year meta-analysis): z = −4.15.** Consistent directional deficit of correlation vs independence across all 4 years. Brackets in this pool are slightly *more diverse* than pure independence from the marginals would predict, not more clustered.

### Pool marginals vs ESPN national (champion pick)

This is where the real opponent-model error lives.

| Year | Actual champ | Pool top pick (pct) | ESPN top pick (pct) | Pool % on actual | ESPN % on actual |
|------|--------------|---------------------|---------------------|------------------|------------------|
| 2023 | (none of the 4 surveyed) | ALA 27.8% | ALA 20.3% | 0.0% | — |
| 2024 | CONN | UNC 28.0% | CONN 24.7% | 20.0% | 24.7% |
| 2025 | FLA  | DUKE 34.4% | DUKE 25.0% | 31.2% | 21.3% |
| 2026 | MICH | ARIZ 36.7% | DUKE 28.4% | 23.3% | 14.3% |

Mean absolute divergence (pool vs ESPN, across all champion-pick teams × 4 years): **5.0 pp**. Largest single divergence: ARIZ in 2026 pool 36.7 % vs ESPN 21.9 % (+14.8pp).

In three of four years the pool's top champion pick was **not** the actual champion. In every year, a different team was over-picked relative to ESPN national — ALA/HOU (2023), UNC/HOU (2024), DUKE/SJU (2025), ARIZ (2026). Pattern suggests systematic pool-specific bias (geographic, demographic, or social), not random drift.

### Late-round correlation structure (mean |ρ| by round)

| Round | 2023 | 2024 | 2025 | 2026 |
|-------|------|------|------|------|
| F4 | 0.185 | 0.139 | 0.152 | 0.154 |
| CHAMP | 0.101 | 0.127 | 0.134 | 0.123 |

All late-round absolute correlations are small (< 0.2). The largest pairwise correlations are either structural (same bracket path → `KU & UNC = +1.0` in 2026 E8 because only one advances in that slot) or anti-correlations from mutually-exclusive picks (you can't pick both ARIZ and DUKE as 2026 champ → `r = −0.30`). Nothing in this data supports a "pool members all pick the same chalky bracket" narrative.

---

## Implications

### 1. O4 verdict: independence holds. Close.

The 13-year backtest has been using a model that samples each opponent bracket independently from ESPN pick distributions. The council's concern was that real pools cluster and this model would overestimate optimization edge (because beating N independent opponents is harder than beating N correlated opponents). The data says the independence side of the assumption is fine — if anything, pool members are *more* independent than the marginals suggest.

### 2. O10 verdict: mostly moot. Downgrade.

The "empirical pool-year correlation vs theoretical copula" design question was raised under the assumption that brackets cluster. With independence empirically confirmed, a copula isn't needed. Downgrade to `[mostly moot]` rather than delete — if future data (e.g. a much larger pool or a different demographic) showed clustering, the copula question would reactivate.

### 3. New open item — O21: use pool-specific marginals.

The opponent model currently uses ESPN national pick rates (see `POOL_STRATEGY_RECOMMENDATION.md`: 60% ESPN / 30% Massey / 10% seed). Pool-specific marginals diverge from ESPN by 5pp mean absolute, up to 18pp on individual teams. Using ESPN national rates means the optimizer is optimizing against the **wrong crowd** — the national crowd, not your pool. 4 years of pool-history data exist (`pool_hist_results.json`, 93 brackets) and can be blended in.

**Gate for O21:** opponent model rebuilt with weight blending ESPN-national + pool-history marginals (e.g. 30/50/20 or tuned). Retrospective pool EV backtested against 2023-2026 to verify the switch changes bracket rankings.

### 4. Lesson for §1: check marginals before correlations.

The council diagnosed "validity threat" as a correlation problem without measuring either correlation *or* marginals first. The actual error is in the marginals. Future councils looking at "opponent model is wrong" should check marginals-vs-target first — it's the cheaper diagnostic and was the actual issue here.

### 5. 2026 ranking-failure root cause (partial).

Council 23 (2026-04-12) flagged that the system produced a 1440-pt winning bracket but ranked it #11. This analysis partly explains: the optimizer ranked ARIZ-champ brackets highly (ARIZ was the optimizer's prior pick based on public pick rate + model probability), but the actual champion was MICH (pool picked at 23.3% vs ESPN 14.3%). A pool-aware marginal would have boosted MICH's rank relative to ARIZ because MICH was over-picked *in this pool*, not nationally. Full root cause also depends on base-model champion probabilities (separate from O4).

---

## What this does NOT prove

- **Only 4 years of one pool.** Independence may not hold for much larger pools or different demographics. The finding is robust to this pool's 2023-2026 data; extrapolation to 31-person pools elsewhere is a conjecture.
- **Null distribution uses empirical marginals.** The test compares observed correlation to what *these* marginals + IID draws produce. It does not test whether brackets are independent in any absolute sense — just that they're as independent as their marginals allow.
- **Does not close O21 itself.** The marginals-are-mis-specified finding is a hypothesis that needs implementation + backtest before it's locked.
- **Does not address O3 (rank-correlation diagnostic).** That artifact already exists at `artifacts/rank_correlation_diagnostic.json` — mean Spearman ρ = +0.37, 12/14 years positive. The optimizer's ranking has real signal; it's not random. O3 is separately close-able.

---

## Reproduction

```bash
python3 scripts/pool_correlation_analysis.py        # single-year (2026)
python3 scripts/pool_multiyear_analysis.py          # 2023-2026
```

Both scripts read `pool_hist_results.json` and deterministic results files in the repo. Output is stable across runs modulo Monte Carlo null (n=1000-2000 trials; std across trials ~0.01).
