# Historical Feature Research (2022-2025)

## Objective
Define leakage-safe variables for March Madness modeling using public, reproducible sources and documented methodology.

## Academic and Public References Reviewed

1. Academic / preprint references
- [A Bayesian Approach to NCAA Tournament Predictions (arXiv)](https://arxiv.org/abs/2503.21790)
- [Machine Learning Approaches to Predicting March Madness Outcomes and Evaluating Bracket Success (arXiv)](https://arxiv.org/abs/1701.07316)
- [Who is likely to win NCAA championship: a sports analytics approach (International Journal of Sports Science)](https://www.researchgate.net/publication/319418470_Who_is_likely_to_win_NCAA_championship_a_sports_analytics_approach)

2. Public methodology and production-style data docs
- [FiveThirtyEight March Madness methodology](https://fivethirtyeight.com/methodology/how-our-march-madness-predictions-work-2/)
- [KenPom Ratings Glossary](https://kenpom.com/blog/ratings-glossary/)
- [sportsipy NCAAB docs](https://sportsipy.readthedocs.io/en/stable/ncaab.html)
- [cbbpy (PyPI)](https://pypi.org/project/cbbpy/)
- [sportsdataverse (PyPI)](https://pypi.org/project/sportsdataverse/)
- [gamezoneR docs](https://gamezoner.sportsdataverse.org/)
- [scikit-learn: data leakage pitfalls](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage)
- [scikit-learn: probability calibration](https://scikit-learn.org/stable/modules/calibration.html)

## High-Value Variables and Evidence

1. Possession-based efficiency (`off/def/net` per 100 possessions)
- Source: [KenPom Ratings Glossary](https://kenpom.com/blog/ratings-glossary/)
- Why: Opponent-adjusted efficiency margin is a stronger predictive signal than raw win-loss.
- Pipeline: `off_eff_game`, `def_eff_game`, `net_eff_game`, rolling and expanding pregame versions.

2. Four Factors (`eFG%`, `TO%`, `ORB%`, `FT rate`)
- Source: [NBA Four Factors explainer](https://www.nba.com/stats/help/glossary#four_factors)
- Why: Four Factors are the canonical decomposition of team quality in basketball outcomes.
- Pipeline: `efg_game`, `to_rate_game`, `ft_rate_game`, `orb_rate_game`, `drb_rate_game`, `three_rate_game`, plus rolling priors.

3. Strength of schedule context (`SRS`, `SOS`)
- Source: [Sports Reference advanced season team table](https://www.sports-reference.com/cbb/seasons/men/2025-advanced-school-stats.html)
- Why: Team results are not comparable without schedule difficulty adjustment.
- Pipeline: prior-season `srs`/`sos` merged as leakage-safe priors.

4. Calibration and probabilistic reliability
- Source: [scikit-learn probability calibration docs](https://scikit-learn.org/stable/modules/calibration.html)
- Why: Tournament strategy requires calibrated probabilities, not only ranking quality.
- Pipeline implication: materialized labels and features are generated for downstream calibration-ready training.

5. Temporal leakage control
- Source: [scikit-learn common pitfalls (data leakage)](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage)
- Why: Using full-season aggregates for in-season games leaks future information.
- Pipeline: all game-derived predictors use `shift(1)` before expanding/rolling; season aggregates are shifted to prior season only.

6. Shot-quality priors (`xP`, shot mix)
- Source: [FiveThirtyEight methodology](https://fivethirtyeight.com/methodology/how-our-march-madness-predictions-work-2/) and [cbbpy](https://pypi.org/project/cbbpy/) + [sportsdataverse](https://pypi.org/project/sportsdataverse/) play-by-play ecosystems
- Why: Shot profile and expected points style are useful for separating process from variance.
- Pipeline: optional prior-season ShotQuality team features merged if available.

7. Public project practice (external-signal blending)
- Source: [FiveThirtyEight methodology](https://fivethirtyeight.com/methodology/how-our-march-madness-predictions-work-2/), [sportsipy docs](https://sportsipy.readthedocs.io/en/stable/ncaab.html), [gamezoneR docs](https://gamezoner.sportsdataverse.org/)
- Why: Strong public systems blend possession metrics, schedule context, and richer event data.
- Pipeline: optional prior-season joins for KenPom/Torvik/ShotQuality/roster/transfer signals.

8. Conference and roster-continuity context
- Source: [sportsipy NCAAB docs](https://sportsipy.readthedocs.io/en/stable/ncaab.html), [FiveThirtyEight methodology](https://fivethirtyeight.com/methodology/how-our-march-madness-predictions-work-2/)
- Why: conference strength and roster continuity influence early- and mid-season team quality estimates.
- Pipeline: prior conference-level means (`SRS`, `SOS`, `off/def rating`) and roster continuity (`minutes_returning_share`, continuity learning-rate proxy, upperclass share).

## Data Source Coverage Implemented

1. Base historical game data
- Primary: `cbbpy` game IDs + box scores.
- Artifacts: `historical_games_<season>.json` with `games` and `team_games`.

2. Team season metrics
- Primary: `sportsipy` (when available), fallback to Sports Reference scraper.
- Artifacts: `team_metrics_<season>.json`.

3. Optional prior-season feature sources
- `kenpom_<season>.json`
- `torvik_<season>.json`
- `shotquality_teams_<season>.json`
- `rosters_<season>.json`
- `transfer_portal_<season>.json`
- `odds_<season>.json` (optional market priors, if available)

All optional sources are merged as **prior-season-only** features to prevent leakage.

## Materialized Tables

1. `team_game_features_<start>_<end>.parquet|csv`
- One row per team-game.
- Contains:
  - game-level outcomes (targets)
  - leakage-safe rolling and expanding priors
  - rest/schedule density variables
  - opponent pregame context joins
  - prior-season external priors

2. `matchup_features_<start>_<end>.parquet|csv`
- One row per game.
- Contains team-vs-team differential and average features for model training.

3. `materialization_manifest_<start>_<end>.json`
- Leakage check results
- Missingness and season coverage report
- Variable coverage audit (critical vs optional feature families)
- Study-alignment scorecard against documented methodologies
- Output artifact paths and formats

## Current Limitations and Gaps

1. Market odds history is optional and not ingested automatically from a live API.
- Impact: reduced quality of market-implied priors.
- Mitigation: provide `odds_<season>.json` with `implied_win_probability`/`title_odds`.

2. Team identity harmonization across sources is heuristic (name/alias matching).
- Impact: some external priors may not map for edge-case naming differences.
- Mitigation: materialization emits match scores and coverage audit; add explicit alias maps for unresolved schools.

3. True injury feed history is not available from free/public standardized APIs at scale.
- Impact: injury uncertainty is approximated through roster/transfer proxies.
- Mitigation: integrate proprietary injury feeds where available.

4. Full location/travel distance features require stable venue geocoding and team location history.
- Impact: only rest/fatigue proxies are included by default.
- Mitigation: add geocoded venue/team coordinates and derive travel burden features.
