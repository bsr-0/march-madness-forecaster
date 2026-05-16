# Kaggle Objective Policy

This document defines the repo's secondary objective for Kaggle-style tournament prediction work.

For the next-step external-data acquisition plan, see [docs/kaggle-external-data-roadmap.md](/Users/benrosen/march-madness-forecaster/docs/kaggle-external-data-roadmap.md).

## Objective Hierarchy

The project has two different optimization targets:

- Primary objective: winner-take-all pool performance, measured by `P(1st)` or related pool EV metrics.
- Secondary objective: Kaggle prediction performance, measured by held-out tournament `Brier` score.

For Kaggle work, `Brier` is the optimization target because Kaggle scoring rewards lower squared error directly.

`BSS` remains important, but as a diagnostic:

- Use `BSS` to confirm the model is still adding skill over the seed baseline.
- Do not treat `BSS` as the primary search objective when choosing among Kaggle candidates.

On a fixed test set, lower `Brier` and higher `BSS` usually rank models similarly anyway. The difference is mostly about governance and framing: optimize the actual contest metric, then use `BSS` as a sanity check.

## Recency Policy

For Kaggle model selection, the repo should use strong recency weighting.

Current policy choice:

- Treat `2023-2025` as the default priority-1 evaluation window for near-term `2027` prediction work.
- Use that 3-year block as the first window to inspect when judging whether a new idea is relevant enough to pursue.
- Treat the most recent 5 observed tournaments as collectively just as important as all older observed tournaments combined.
- Treat this as a modeling prior, not a theorem.
- Older years still matter as regularization and as protection against chasing noise.

This means candidate search should favor recent held-out performance much more than a simple equal-weight mean across all historical years.

Interpretation:

- `2023-2025` is the default "does this look relevant for 2027?" window.
- The broader recent-5 weighting rule is still the repo's default objective weighting rule.
- This does not, by itself, replace the stricter admission split used by `scripts/admit_kaggle_candidate.py`.

## Recommended Weighting Rule

When using year-level weighting:

- Give each older year weight `1.0`.
- Define the recent block as the latest 5 observed tournament years in the fit window.
- Set the per-year recent weight so the recent block gets about half of the total objective mass.

Formula:

`recent_year_weight = n_older_years / n_recent_years`

Example:

- If the fit window contains 12 older years and 5 recent years, use `recent_year_weight = 12 / 5 = 2.4`.
- That makes the recent 5 seasons contribute the same total weight as the earlier 12 seasons combined.

This policy should be recomputed whenever the fit window changes.

## Admission Rule For Kaggle Candidates

When choosing a Kaggle candidate:

- Keep the current admission script selector on held-out `Brier` unless the gate is explicitly revised later.
- Keep the strict final-holdout mean `Brier` as the admission gate basis.
- Check `2023-2025` first as the default priority evaluation slice for 2027-facing work.
- Prefer recent shadow/final years over pooled all-history summaries.
- Keep `BSS > 0` as a guardrail on each final holdout year.
- Keep a single-year regression cap so a candidate cannot win on average while blowing up one recent year.
- Report both weighted and unweighted summaries when possible.

Recommended reporting order:

- Weighted recent-aware held-out `Brier`
- Final-holdout mean `Brier`
- `2023-2025` mean `Brier`
- `BSS` / stability guardrails

## Why This Policy Exists

The repo's Kaggle path is trying to improve future tournament predictions, not maximize retrospective smoothness over very old seasons.

Reasons to overweight recent years:

- The current prediction stack is Torvik-centered and tuned for the modern data regime.
- Contest conditions, data quality, and model architecture are more relevant in recent seasons than in older ones.
- Recent years are a better proxy for the environment the next Kaggle submission will face.

Reasons not to ignore older years completely:

- Tournament samples are tiny.
- A model that only wins on the last few seasons can still be overfit.
- Older years are useful for shrinkage, stability, and sign checks.

## Repo Guidance

Use this policy for:

- Kaggle admission decisions in `scripts/admit_kaggle_candidate.py`
- Kaggle-oriented experiment memos and backtests
- Any future candidate-selection logic for `ensemble`, `torvik_corrected`, or successor models

Do not use this policy to override the repo's primary pool objective. It is strictly for the Kaggle secondary path.
