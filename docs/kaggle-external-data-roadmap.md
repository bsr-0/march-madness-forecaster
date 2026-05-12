# Kaggle External Data Roadmap

This document defines the next serious path for improving the repo's Kaggle secondary objective:

- optimize held-out tournament `Brier`
- keep `BSS` as a guardrail

For the concrete vendor-feasibility execution checklist, see [docs/kaggle-market-odds-phase1-checklist.md](/Users/benrosen/march-madness-forecaster/docs/kaggle-market-odds-phase1-checklist.md).

The internal-only path is now effectively exhausted. Future improvement should be treated as a data-acquisition problem first.

## Current status

The current admitted Kaggle baseline is `torvik_corrected_recent5_conservative`.

The last serious internal-only nonlinear candidate (`torvik_isotonic`) failed against that baseline on final holdout:

- incumbent final mean Brier: `0.1308`
- isotonic candidate final mean Brier: `0.1351`
- incumbent final mean BSS: `+0.1313`
- isotonic candidate final mean BSS: `+0.1025`

Conclusion:

- do not reopen internal-only calibration work by default
- the highest-value future work is new external signal

## Priority order

1. Historical market odds
2. Injury / availability data
3. Better pool-specific opponent data
4. Roster continuity / transfer stability

For the Kaggle track specifically, historical market odds are the first priority by a large margin.

## Why historical market odds are first

This is the clearest missing institutional-grade signal in the repo.

Reasons:

- market probabilities are stronger calibration anchors than most public model blends
- they directly target the competition metric (`Brier`)
- they add genuinely new information instead of another recombination of torvik, seeds, and internal pipeline outputs
- they are point-in-time compatible if captured with correct timestamps

## Exact data to acquire

Minimum required fields per game:

- `game_id`
- `season`
- `date`
- `team_1`
- `team_2`
- `book`
- `market_type`
- `opening_moneyline_team_1`
- `opening_moneyline_team_2`
- `closing_moneyline_team_1`
- `closing_moneyline_team_2`
- `opening_spread`
- `closing_spread`
- `timestamp_open`
- `timestamp_close`

Preferred derived fields:

- de-vigged opening implied probability
- de-vigged closing implied probability
- consensus opening probability across books
- consensus closing probability across books
- opener-to-close movement
- spread-to-moneyline agreement checks

Required historical coverage:

- at least `2021-2026`
- ideally `2020+` if acquisition is cheap

## Vendor priority

### 1. The Odds API

Best first acquisition path if the goal is practical historical NCAA odds coverage with limited integration work.

Why:

- has historical odds coverage in the modern window the repo actually cares about
- likely enough for the `2021+` Kaggle evaluation regime
- lower integration burden than a full enterprise vendor

Use it if:

- the historical endpoint includes tournament games with enough book coverage
- rate limits and pricing are acceptable

### 2. SportsDataIO

Best fallback if broader structured sports data is desired in one vendor relationship.

Why:

- can also support injuries and roster-related expansion later
- more likely to support richer historical metadata

Use it if:

- The Odds API historical coverage is too thin
- one-vendor consolidation matters more than lightweight integration

### 3. Sportradar

Consider only if the project is intentionally moving toward an enterprise-grade data stack.

Why:

- potentially strong long-run vendor
- likely more expensive and heavier-weight than necessary for the immediate Kaggle question

## Data quality acceptance checks

Do not trust the feed until these pass:

1. Tournament coverage check
- every NCAA tournament game in the target years is present or missingness is explicitly measured and acceptable

2. Point-in-time check
- the line timestamp used for a game is from before game start
- no post-tip updates leak into the modeling artifact

3. Book-consensus check
- consensus probability is not dominated by one noisy book
- outlier books can be filtered or downweighted

4. De-vig sanity check
- implied probabilities sum correctly after margin removal
- edge cases with missing sides are handled deterministically

5. Team-identity resolution check
- team names match the repo's canonical IDs cleanly
- no silent fallback to weak fuzzy matching

## Recommended modeling path once odds exist

Do not start with a large model.

Start with a constrained candidate:

- base signal: `torvik`
- external signal: market closing implied probability
- model family: bounded monotone correction or small linear/logit correction

Preferred first candidate:

- input 1: `logit(torvik_prob)`
- input 2: `logit(market_prob)`
- input 3: signed `seed_gap`
- optional input 4: opener-to-close movement

Governance:

- keep torvik as the base
- keep probability clipping
- admit only through the same recent-5 Brier-first gate

## What not to do

- do not add odds and then immediately build a large generic ensemble
- do not use market data without a timestamp audit
- do not replace the current admission gate with a looser one
- do not reopen broad blend-complexity work just because a new source appears

## Execution plan

Phase 1:

- choose vendor
- confirm historical tournament coverage for `2021-2026`
- define schema and canonical team mapping

Use [docs/kaggle-market-odds-phase1-checklist.md](/Users/benrosen/march-madness-forecaster/docs/kaggle-market-odds-phase1-checklist.md) as the execution document for this phase.

Phase 2:

- ingest raw historical odds
- build de-vigged consensus probability artifact
- add point-in-time validation checks

Phase 3:

- add one new candidate family: `torvik + market correction`
- run the existing admission gate against `torvik_corrected_recent5_conservative`

Phase 4:

- if admitted, promote to new Kaggle baseline
- if not admitted, keep current baseline and treat market integration as incomplete or low-value

## Closure rule

For the Kaggle track, the next serious reopen condition is:

- historical market odds land in the repo with acceptable `2021-2026` tournament coverage

Until that happens, the Brier/BSS internal-modeling thread should be treated as closed.
