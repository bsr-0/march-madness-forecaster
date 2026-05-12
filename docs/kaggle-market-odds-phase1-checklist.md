# Kaggle Market Odds Phase 1 Checklist

Purpose:

- close the vendor-feasibility question before writing ingestion or modeling code
- determine whether historical market odds are good enough to reopen the Kaggle Brier/BSS track

This checklist is the concrete execution plan for Phase 1 of [docs/kaggle-external-data-roadmap.md](/Users/benrosen/march-madness-forecaster/docs/kaggle-external-data-roadmap.md).

## Phase 1 goal

Answer one question cleanly:

Can the repo get usable point-in-time NCAA tournament odds for `2021-2026` with enough coverage and metadata to support a `torvik + market` correction model?

If the answer is no, stop. Do not start ingestion or modeling work.

## Primary vendor order

1. `The Odds API`
2. `SportsDataIO`
3. `Sportradar`

The decision rule is practical, not theoretical:

- use the cheapest / simplest vendor that clears the coverage and timestamp bar
- escalate to the next vendor only if the previous one fails the acceptance checks

## Required outputs

Phase 1 is complete only when these exist:

- one short vendor comparison memo
- one coverage table for `2021-2026`
- one schema draft for the chosen source
- one explicit go / no-go decision

Recommended artifact paths:

- `docs/kaggle-market-odds-vendor-review.md`
- `artifacts/kaggle_market_odds_coverage.json`
- `artifacts/kaggle_market_odds_sample_rows.json`

## Checklist

### 1. Vendor access and pricing

- confirm the vendor exposes historical NCAA men's tournament odds
- confirm whether the relevant endpoint is available on the expected plan tier
- record rate limits
- record pricing assumptions
- record whether opening and closing lines are both available

Acceptance:

- vendor can supply at least closing lines for `2021-2026`
- access cost is acceptable relative to the secondary Kaggle objective

Failure:

- no historical tournament access
- pricing is too high for the likely value of the Kaggle track

### 2. Tournament coverage audit

For each year `2021-2026`:

- count expected NCAA tournament games
- count games returned by the vendor
- compute coverage rate
- identify whether First Four / play-in games are included
- identify whether championship and Final Four games are included

Acceptance:

- overall tournament game coverage is at least `95%`
- no year is below `90%`
- missingness is explainable and not concentrated in key rounds

Failure:

- missingness is broad, inconsistent, or unexplained
- coverage is materially worse in recent years

### 3. Timestamp / PIT audit

For a sample across years and rounds:

- capture event start time
- capture odds timestamp
- verify the intended "closing" line is still pre-tip
- verify no post-game or in-game rows are mislabeled as pregame

Acceptance:

- every audited row is clearly pregame
- "close" can be defined deterministically from available timestamps

Failure:

- no reliable timestamp
- line snapshots are ambiguous relative to tip time
- pregame and live/in-game rows are mixed

### 4. Book coverage and consensus quality

For each sample game:

- count books available
- note whether books are stable across years
- inspect outlier books
- confirm consensus can be built without one-book domination

Acceptance:

- typical tournament game has multiple books
- consensus is stable enough to compute a defensible market probability

Failure:

- too many single-book rows
- large vendor-year instability in book coverage

### 5. Moneyline / spread usability

Check whether both markets are present and usable:

- moneyline available?
- spread available?
- enough rows with both to support movement and consistency checks?

Acceptance:

- moneyline is broadly available
- spread is at least frequent enough to support QA and optional features

Failure:

- only partial market coverage with no reliable probability anchor

### 6. Team identity mapping

Take a cross-year sample and verify:

- team names map cleanly to repo canonical IDs
- no silent fuzzy matching is needed for many rows
- play-in teams and naming variants are manageable

Acceptance:

- mapping can be implemented deterministically with a small alias layer

Failure:

- vendor naming is too unstable to trust without broad fuzzy heuristics

### 7. Schema draft

Produce a first-pass schema for the raw artifact:

- event metadata
- team identifiers
- book name
- market type
- opening / closing values
- timestamps

Acceptance:

- schema is sufficient for:
  - de-vigged closing implied probability
  - opener-to-close movement
  - point-in-time validation

### 8. Go / no-go decision

At the end of Phase 1, make one explicit call:

- `GO`: chosen vendor clears the bar; start ingestion
- `NO-GO`: odds source is too weak / expensive; keep Brier/BSS track closed

Do not leave the result ambiguous.

## Decision rubric

Use this simple rubric:

| Area | Weight | Pass condition |
|---|---:|---|
| Tournament coverage | High | `>=95%` overall, no year `<90%` |
| PIT safety | High | reliable pregame timestamps |
| Book depth | Medium | enough books for consensus |
| Market richness | Medium | closing moneyline at minimum |
| Name mapping | Medium | deterministic mapping feasible |
| Cost / effort | Medium | justified by likely Kaggle value |

If either coverage or PIT safety fails, the source should be rejected.

## Non-goals for Phase 1

Do not do these in Phase 1:

- no new model training
- no admission-gate reruns
- no production ingestion pipeline
- no broad abstraction for multiple vendors

This phase is only about proving the source is worth integrating.

## Exit conditions

Phase 1 is done when one of these is true:

1. `GO`
- chosen vendor documented
- coverage and timestamp audits pass
- schema drafted
- proceed to ingestion

2. `NO-GO`
- vendor(s) documented
- failure reasons written down
- Kaggle Brier/BSS track remains closed pending a different source

## Recommended next action after Phase 1

If `GO`:

- build a raw odds snapshot artifact for `2021-2026`

If `NO-GO`:

- stop Kaggle model work
- do not reopen internal-only tuning
- revisit only when a better odds source appears
