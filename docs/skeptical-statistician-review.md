# Skeptical Statistician Review

Blunt external-review-style summary of this repo from the perspective of an experienced sports statistician at a place like DraftKings.

Scope:
- Primary objective: ESPN pool EV / `P(1st)`
- Secondary objective: Kaggle-style game prediction performance (`Brier`, with `BSS` as guardrail)

This document is intentionally direct. It is meant to compress the repo's current strengths, weaknesses, and likely next steps into one memo that future agents can find quickly.

## Bottom line

If a skeptical, experienced sports statistician reviewed this repo, they would probably say:

1. The repo is stronger on experimental discipline than on raw predictive edge.
2. The main bottleneck is not "more model sophistication." It is missing or weak external signal.
3. The Kaggle path is reasonably well-governed and has probably extracted most of the available value from the current internal data.
4. The pool-EV path is directionally thoughtful, but the target is much noisier and more weakly identified than the repo sometimes wants it to be.
5. The highest-upside future work is data acquisition, not another round of internal blending or architecture changes.

## What they would respect

- The repo has a real dead-end ledger. It records failed ideas instead of quietly forgetting them.
- There is real temporal-discipline language throughout the codebase: LOYO, holdout enforcement, leakage checks, point-in-time guardrails.
- The codebase has already falsified many common bad ideas:
  - bigger generic ML stacks
  - more source blending
  - longer windows
  - contrarian pool heuristics without evidence
- The Kaggle path now has a clear objective policy:
  - optimize held-out `Brier`
  - use `BSS` as a guardrail
  - apply explicit recency weighting
- The current Kaggle baseline is at least admitted through an explicit gate rather than by narrative.

## What they would criticize

### 1. Too much of the remaining problem is a data problem, not a modeling problem

They would likely say the repo has already spent a lot of time optimizing combinations of:
- seeds
- torvik
- internal pipeline outputs
- public pick-rate proxies

But the most obvious missing institutional-grade signals are still absent:
- historical market odds
- injury / availability data
- pool-specific opponent data at useful scale
- player-level lineup / continuity signals

Their blunt read would be:
"You are mostly recombining second-tier public signals while missing the first-tier ones."

### 2. The pool-EV objective is much noisier than the repo's process language suggests

A serious statistician would probably be more skeptical of the primary pool objective than of the Kaggle objective.

Why:
- the target is winner-take-all or near-winner-take-all
- actual pool fields are small and idiosyncratic
- the actual historical sample of true opponent pools is tiny
- the mapping from estimated game probabilities to realized pool payout is very high variance

Their likely conclusion:
"The repo may be directionally right on pool strategy, but the empirical confidence around those recommendations is much weaker than for the Kaggle path."

### 3. The repo likely overstates what can be learned from ESPN-national pick shares for a specific pool

Even though the repo already knows this is imperfect, an external reviewer would probably push harder on it.

They would likely say:
- national pick rates are a useful prior
- they are not a substitute for pool-specific behavior
- the misspecification in opponent marginals is probably more important than many of the internal ranking refinements

In blunt terms:
"Your opponent model is probably good enough for toy backtests and weak enough to cap real EV progress."

### 4. Some recent improvements are real, but still live on small recent samples

A skeptical reviewer would not dismiss the recent-5 Kaggle correction result. But they would likely frame it as:
- promising
- plausible
- still fragile

They would want:
- uncertainty intervals
- year-by-year stability framing
- less rhetorical emphasis on point estimates alone

Especially for claims built on `2021+` slices, they would say:
"This may be the right policy choice, but do not confuse recency preference with certainty."

### 5. The repo has strong governance for what not to do, but weaker clarity on acquisition strategy

The negative evidence is good. The missing thing is a more explicit statement that:
- internal modeling is near diminishing returns
- future progress depends on better data
- data acquisition priority should be treated as a first-class roadmap

## Where they would probably agree with the repo

They would likely agree with these current repo positions:

- Do not reopen broad blend-complexity work by default.
- Do not chase bigger generic ML architectures without new signal.
- Keep the Kaggle objective Brier-first.
- Use recency weighting for Kaggle model selection.
- Preserve torvik as the base signal rather than replacing it with a noisier learned system.

They would probably view the current `torvik_corrected_recent5_conservative` Kaggle baseline as reasonable.

## Where they would push a different emphasis

They would likely shift emphasis in these ways:

### Primary EV / pool focus

Priority order they would likely prefer:

1. Better opponent-pool data
2. Injury / availability information
3. Market odds
4. Women's-path opportunity if contest scope includes it
5. Only then more bracket-selection refinement

Reason:
For pool EV, the largest misses usually come from:
- misreading the field
- missing live availability changes
- misunderstanding which tails are worth owning

not from another small recalibration of baseline win probabilities.

### Secondary Kaggle / Brier focus

Priority order they would likely prefer:

1. Historical market odds
2. Torvik + market correction
3. Nonlinear monotone torvik calibrator
4. Injury / availability data
5. Women's-path improvement

Reason:
For game-level probability scoring, calibration-grade external signal matters more than bracket-portfolio cleverness.

## Highest-probability future moves

If this reviewer had to recommend the most practical future roadmap across both objectives, it would probably look like this:

| Priority | Idea | Main objective helped | Expected impact | Confidence |
|---|---|---|---|---|
| 1 | Historical market odds | Secondary first, primary second | High | High |
| 2 | Injury / availability data | Primary first, secondary second | Medium-High | Medium |
| 3 | Torvik + market correction | Secondary first | High | High |
| 4 | Better pool-specific opponent data | Primary | Medium-High | Medium |
| 5 | Women's-path improvement | Both | Medium-High | Medium-High |
| 6 | Nonlinear monotone torvik calibrator | Secondary | Medium | Medium-High |
| 7 | Roster continuity / transfer stability | Both | Medium | Medium-Low |
| 8 | More blend complexity | Neither materially | Low | High |

## What they would likely say is exhausted

They would likely say the following are close to exhausted unless new data arrives:

- generic model complexity
- more internal-source blending
- more alpha tuning
- more handcrafted bucket logic
- longer historical windows as a default answer

Their summary would be:
"You do not have a model architecture problem anymore. You have a signal acquisition problem."

## Sharpest single criticism

If forced to give one sharp criticism of the repo, it would probably be:

"The codebase is more mature than the data stack."

Meaning:
- methodology is fairly disciplined
- experimentation history is fairly honest
- but the data sources still look more like a strong public-data research project than like a professional betting or DFS operation

## Sharpest single positive note

If forced to give one sharp positive note, it would probably be:

"This repo is unusually willing to keep negative results and stop re-learning the same lesson."

That is a real strength. Many sports-model codebases are weaker here.

## Final verdict

The likely external verdict is:

- The repo is credible.
- The repo is not naive.
- The repo has already ruled out many easy but wrong ideas.
- The next real gains are likely to come from external data quality, not from cleverer internal modeling.
- The Kaggle path is closer to production-grade statistical thinking than the pool-EV path, because the target is cleaner and the evaluation loop is more stable.
- The pool-EV path is still worth pursuing, but should be described more cautiously and treated as a thinner-signal, higher-variance optimization problem.

## Practical implication for future agents

If you are proposing future work:

- For Kaggle / `Brier`: default to market-data or monotone-calibration ideas.
- For pool EV: default to opponent-data or injury-data ideas.
- Do not reopen blend-complexity work unless new external signal arrives.
- Do not claim strong pool-EV improvements without acknowledging the small-sample and field-modeling limitations.
