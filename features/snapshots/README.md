# Tier 3 Feature Snapshots

This directory stores dated CSV snapshots of Tier 3 (external rating) features
for Point-in-Time (PIT) compliance in LOYO cross-validation.

## Protocol (v2, Section 2.3)

Tier 3 features (KenPom AdjEM, NET ranking, Sagarin, BPI, etc.) are sourced
from third-party systems that update continuously. For each LOYO fold year,
these features must use the Selection Sunday morning snapshot — not
end-of-season values.

## Expected File Format

```
kenpom_YYYY-MM-DD.csv
net_YYYY-MM-DD.csv
sagarin_YYYY-MM-DD.csv
```

Where the date is Selection Sunday (or the latest available pre-tournament
date) for that season. Each CSV should contain columns:
- `team_id` or `team_name`
- The rating value(s) for that system

## Current State

The pipeline's Tier 3 features (`external_rating_composite`,
`external_rating_spread`) are currently derived from live-scraped data via
`src/data/features/public_advanced_metrics.py`, which fetches ratings with
point-in-time awareness. Historical LOYO folds use imputed composites rather
than true archived snapshots.

To strengthen PIT compliance for backtesting, populate this directory with
historical Selection Sunday snapshots for each LOYO fold year (2016-2025,
excluding 2020).
