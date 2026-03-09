"""Sample loading for historical LOYO training.

Extracts ``_load_year_samples_incremental`` from ``SOTAPipeline`` as a
module-level function with explicit ``config`` parameter.

Implements Agent Directive V7 S2 Phase 1 (sample loading decomposition).
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date
from typing import Dict, List, Optional, Tuple

import numpy as np

from ...data.features.proprietary_metrics import (
    IncrementalMetricsEngine,
    _team_id,
    team_games_to_game_records,
)
from ..config import (
    MIN_SEASON_FEATURE_COMPLETENESS,
    SOTAPipelineConfig,
    TOURNAMENT_START_DATES,
    _infer_tournament_round_weight,
)

logger = logging.getLogger(__name__)


def load_year_samples(
    games_path: str,
    metrics_path: str,
    feature_dim: int,
    year: int,
    include_tournament: bool = False,
    prior_elo: Optional[Dict[str, float]] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Deprecated entry point — raises ``NotImplementedError``.

    Use :func:`load_year_samples_incremental` instead.
    """
    raise NotImplementedError(
        "_load_year_samples() is deprecated and must not be called. "
        "It uses season-end team_metrics aggregates as training features, "
        "violating point-in-time constraints and causing temporal leakage. "
        "Use _load_year_samples_incremental() instead."
    )


def load_year_samples_incremental(
    config: SOTAPipelineConfig,
    games_path: str,
    metrics_path: str,
    feature_dim: int,
    year: int,
    include_tournament: bool = False,
    prior_elo: Optional[Dict[str, float]] = None,
) -> tuple:
    """Load historical year samples with TRUE point-in-time features.

    For each training game at date D, features are computed using ONLY
    games played before D via ``IncrementalMetricsEngine``.

    Args:
        config: Pipeline configuration (for kaggle_dir, etc.).
        games_path: Path to ``historical_games_{year}.json``.
        metrics_path: Path to team metrics JSON for conference map.
        feature_dim: Expected feature vector dimension.
        year: Season year.
        include_tournament: If True, include tournament games.
        prior_elo: Prior-season end-of-season Elo for carryover.

    Returns:
        Tuple of ``(X, y, margins, end_elo, round_weights)`` or shorter
        tuples on early exit.
    """
    # ── 1. Load game data ─────────────────────────────────────────────
    with open(games_path, "r") as f:
        payload = json.load(f)

    if isinstance(payload, dict):
        team_games_raw = payload.get("team_games", [])
    else:
        team_games_raw = payload  # legacy list format
    if not team_games_raw:
        logger.warning(
            "Year %d: no team_games (box-score) data — skipping to avoid "
            "season-end data leakage.",
            year,
        )
        return np.empty((0, feature_dim)), np.array([]), {}

    game_records = team_games_to_game_records(team_games_raw, year)
    if len(game_records) < 100:
        logger.warning("Year %d: only %d GameRecords — skipping.", year, len(game_records))
        return np.empty((0, feature_dim)), np.array([]), {}

    # ── 2. Load auxiliary data (conference map, seeds, rosters) ────────
    conference_map = None
    try:
        with open(metrics_path, "r") as f:
            metrics_payload = json.load(f)
        if isinstance(metrics_payload, dict):
            for tid, info in metrics_payload.items():
                if isinstance(info, dict) and "conference" in info:
                    if conference_map is None:
                        conference_map = {}
                    conference_map[tid] = info["conference"]
    except Exception as _conf_exc:
        logger.debug("Conference map extraction from metrics failed: %s", _conf_exc)

    # FIX-CONF: Fallback to Kaggle MTeamConferences.csv
    if not conference_map and config.kaggle_dir:
        try:
            from ...data.kaggle_loader import KaggleDataLoader

            _kloader = KaggleDataLoader(config.kaggle_dir)
            _kconf = _kloader.load_team_conferences(year)
            if _kconf:
                conference_map = _kconf
                logger.info(
                    "FIX-CONF: Loaded %d conference assignments from Kaggle "
                    "for year %d — conference_adj_em will use true conf peers.",
                    len(_kconf),
                    year,
                )
        except Exception as _kconf_exc:
            logger.debug("Kaggle conference fallback failed for year %d: %s", year, _kconf_exc)

    if conference_map:
        logger.debug(
            "FIX-CONF: Year %d has conference map with %d teams across %d conferences.",
            year,
            len(conference_map),
            len(set(conference_map.values())),
        )
    else:
        logger.debug(
            "FIX-CONF: Year %d has NO conference map — conference_adj_em "
            "will use schedule-frequency-based opponent clustering.",
            year,
        )

    # Tournament seeds
    team_seeds: Dict[str, int] = {}
    seeds_path = os.path.join(os.path.dirname(games_path), f"tournament_seeds_{year}.json")
    if not os.path.isfile(seeds_path):
        seeds_path = os.path.join(
            os.path.dirname(games_path), "historical", f"tournament_seeds_{year}.json"
        )
    if os.path.isfile(seeds_path):
        try:
            with open(seeds_path, "r") as f:
                seeds_data = json.load(f)
            if isinstance(seeds_data, list):
                for entry in seeds_data:
                    tid = entry.get("team_id", "")
                    seed = int(entry.get("seed", 0))
                    if tid and seed:
                        team_seeds[_team_id(tid)] = seed
        except Exception as _seed_exc:
            logger.debug("Tournament seeds loading failed for year %d: %s", year, _seed_exc)

    # Massey Ordinals composite
    team_massey_composite: Dict[str, float] = {}
    team_massey_spread: Dict[str, float] = {}
    _massey_loaded = False
    massey_cache_path = os.path.join(
        os.path.dirname(games_path),
        f"external_massey_composite_{year}.json",
    )
    if not os.path.isfile(massey_cache_path):
        massey_cache_path = os.path.join(
            os.path.dirname(games_path),
            "historical",
            f"external_massey_composite_{year}.json",
        )
    if os.path.isfile(massey_cache_path):
        try:
            with open(massey_cache_path, "r") as f:
                massey_data = json.load(f)
            for entry in massey_data:
                tid = entry.get("team_id", "")
                if tid:
                    team_massey_composite[_team_id(tid)] = entry.get("normalized", 0.0)
            _massey_loaded = True
            logger.info(
                "Gap #1: Loaded Massey composite cache for year %d (%d teams)",
                year,
                len(team_massey_composite),
            )
        except Exception as _massey_cache_exc:
            logger.debug(
                "Massey composite cache load failed for year %d: %s", year, _massey_cache_exc
            )
    if not _massey_loaded and config.kaggle_dir:
        try:
            from ...data.scrapers.external_ratings import ExternalRatingsLoader

            _loader = ExternalRatingsLoader(cache_dir=os.path.dirname(games_path))
            n_cached = _loader.populate_from_massey_ordinals(config.kaggle_dir, year)
            if n_cached > 0:
                all_ratings = _loader.load_all(year)
                if all_ratings:
                    composites = _loader.compute_composite(all_ratings)
                    for tid, comp in composites.items():
                        team_massey_composite[tid] = comp.composite_rating
                        team_massey_spread[tid] = comp.rating_spread
                    _massey_loaded = True
                    logger.info(
                        "Gap #1: Computed Massey composite from Kaggle for year %d (%d teams)",
                        year,
                        len(team_massey_composite),
                    )
        except Exception as e:
            logger.debug("Gap #1: Massey ordinals not available for year %d: %s", year, e)

    if team_massey_composite:
        logger.info(
            "FIX-MASSEY: Year %d Massey composite coverage: %d teams (source=%s).",
            year,
            len(team_massey_composite),
            "cache" if _massey_loaded and not config.kaggle_dir else "kaggle",
        )
    else:
        logger.info(
            "FIX-MASSEY: Year %d has NO Massey composite data — "
            "diff_external_rating_composite will be 0.0 for all training "
            "samples this year. Provide external_massey_composite_%d.json "
            "or set kaggle_dir to enable.",
            year,
            year,
        )

    # Roster features
    team_roster_features: Dict[str, Dict] = {}
    roster_path = os.path.join(
        os.path.dirname(games_path), "historical", f"cbbpy_rosters_{year}.json"
    )
    if not os.path.isfile(roster_path):
        roster_path = os.path.join(os.path.dirname(games_path), f"cbbpy_rosters_{year}.json")
    if os.path.isfile(roster_path):
        try:
            with open(roster_path, "r") as f:
                roster_data = json.load(f)
            if isinstance(roster_data, dict):
                for tid, info in roster_data.items():
                    if isinstance(info, dict):
                        team_roster_features[tid] = info
        except Exception as _roster_exc:
            logger.debug("Roster features loading failed for year %d: %s", year, _roster_exc)

    # ── 3. Create incremental engine ──────────────────────────────────
    inc_engine = IncrementalMetricsEngine(
        game_records,
        conference_map=conference_map,
        prior_elo=prior_elo,
    )

    # ── 4. Identify training games ────────────────────────────────────
    seen_gids: set = set()
    all_games: list = []
    for g in sorted(game_records, key=lambda r: r.game_date):
        if g.game_id in seen_gids:
            continue
        seen_gids.add(g.game_id)
        all_games.append(g)

    _t_start = TOURNAMENT_START_DATES.get(year, date(year, 3, 14))
    tournament_cutoff = _t_start.isoformat()
    if include_tournament:
        training_games = all_games
    else:
        training_games = [g for g in all_games if g.game_date < tournament_cutoff]

    # Gap #6: Data quality filtering
    training_games = [
        g
        for g in training_games
        if g.points > 0 and g.opp_points > 0 and abs(g.points - g.opp_points) <= 80
    ]

    if year <= 2009:
        pre_filter_count = len(training_games)
        training_games = [
            g
            for g in training_games
            if (
                getattr(g, "fgm", 0)
                + getattr(g, "fga", 0)
                + getattr(g, "opp_fgm", 0)
                + getattr(g, "opp_fga", 0)
            )
            > 0
        ]
        filtered_count = pre_filter_count - len(training_games)
        if filtered_count > 0:
            logger.info(
                "Gap #6: Filtered %d/%d zero-stat games from year %d",
                filtered_count,
                pre_filter_count,
                year,
            )

    if not training_games:
        return (
            np.empty((0, feature_dim)),
            np.array([]),
            np.array([]),
            inc_engine.get_end_of_season_elo(),
        )

    # ── 5. Build feature vectors ──────────────────────────────────────
    X_list: list = []
    y_list: list = []
    margins_list: list = []
    round_weight_list: list = []
    skipped = 0

    min_games = getattr(config, "game_level_min_games_per_team", 5)

    for g in training_games:
        if min_games > 0:
            n1 = inc_engine.games_played_before(g.team_id, g.game_date)
            n2 = inc_engine.games_played_before(g.opponent_id, g.game_date)
            if n1 < min_games or n2 < min_games:
                skipped += 1
                continue

        metrics = inc_engine.compute_as_of(g.game_date)
        if not metrics:
            skipped += 1
            continue

        m1 = metrics.get(g.team_id)
        m2 = metrics.get(g.opponent_id)
        if m1 is None or m2 is None:
            skipped += 1
            continue

        # Seeds: only attach after selection cutoff to prevent leakage.
        if g.game_date > tournament_cutoff:
            seed1 = team_seeds.get(g.team_id, 0)
            seed2 = team_seeds.get(g.opponent_id, 0)
        else:
            seed1, seed2 = 0, 0

        # Massey: only attach after selection cutoff (end-of-season aggregate).
        if g.game_date > tournament_cutoff:
            _mc1 = team_massey_composite.get(g.team_id, 0.0)
            _mc2 = team_massey_composite.get(g.opponent_id, 0.0)
            _ms1 = team_massey_spread.get(g.team_id, 0.0)
            _ms2 = team_massey_spread.get(g.opponent_id, 0.0)
        else:
            _mc1 = _mc2 = _ms1 = _ms2 = 0.0

        v1 = IncrementalMetricsEngine.metrics_to_team_vector(
            m1,
            seed=seed1,
            external_rating_composite=_mc1,
            external_rating_spread=_ms1,
        )
        v2 = IncrementalMetricsEngine.metrics_to_team_vector(
            m2,
            seed=seed2,
            external_rating_composite=_mc2,
            external_rating_spread=_ms2,
        )

        # Overlay roster features
        rf1 = team_roster_features.get(g.team_id, {})
        rf2 = team_roster_features.get(g.opponent_id, {})
        if rf1 or rf2:
            v1[15] = rf1.get("roster_continuity", 0.0)
            v1[17] = rf1.get("avg_experience", 0.0)
            v2[15] = rf2.get("roster_continuity", 0.0)
            v2[17] = rf2.get("avg_experience", 0.0)

        matchup = IncrementalMetricsEngine.build_matchup_vector(v1, v2, seed1, seed2)

        if len(matchup) < feature_dim:
            padded = np.zeros(feature_dim, dtype=np.float64)
            padded[: len(matchup)] = matchup
            matchup = padded
        elif len(matchup) > feature_dim:
            matchup = matchup[:feature_dim]

        X_list.append(matchup)
        y_list.append(1 if g.points > g.opp_points else 0)
        margins_list.append(float(g.points - g.opp_points))

        rw = 1.0
        if include_tournament and g.game_date > tournament_cutoff:
            rw = _infer_tournament_round_weight(g.game_date, year)
        round_weight_list.append(rw)

    if not X_list:
        return (
            np.empty((0, feature_dim)),
            np.array([]),
            np.array([]),
            inc_engine.get_end_of_season_elo(),
            np.array([]),
        )

    X_arr = np.stack(X_list)
    y_arr = np.array(y_list, dtype=int)
    margins_arr = np.array(margins_list, dtype=np.float64)
    round_weights_arr = np.array(round_weight_list, dtype=np.float64)

    # FIX-DQ: Feature completeness validation
    completeness = float(np.mean(np.abs(X_arr) > 1e-8))
    col_activity = np.mean(np.abs(X_arr) > 1e-8, axis=0)
    n_dead_cols = int(np.sum(col_activity < 0.01))

    if completeness < MIN_SEASON_FEATURE_COMPLETENESS:
        logger.warning(
            "FIX-DQ: Year %d feature completeness %.2f < %.2f threshold; "
            "skipping season. %d/%d features dead (< 1%% non-zero).",
            year,
            completeness,
            MIN_SEASON_FEATURE_COMPLETENESS,
            n_dead_cols,
            feature_dim,
        )
        return (
            np.empty((0, feature_dim)),
            np.array([]),
            np.array([]),
            inc_engine.get_end_of_season_elo(),
            np.array([]),
        )

    if n_dead_cols > 0:
        logger.debug(
            "FIX-DQ: Year %d has %d dead feature columns (< 1%% non-zero).",
            year,
            n_dead_cols,
        )

    logger.info(
        "Year %d (incremental): %d training samples from %d games "
        "(%d skipped, %d unique dates). feature_dim=%d. "
        "completeness=%.2f. dead_features=%d. tournament_round_weighted=%d.",
        year,
        len(X_list),
        len(training_games),
        skipped,
        len(inc_engine._unique_dates),
        feature_dim,
        completeness,
        n_dead_cols,
        int(np.sum(round_weights_arr > 1.0)),
    )

    # Symmetric augmentation
    if getattr(config, "enable_symmetric_augmentation", True):
        from ...ml.training.symmetric import symmetric_augment

        X_arr, y_arr, margins_arr, round_weights_arr, _ = symmetric_augment(
            X_arr,
            y_arr,
            margins_arr,
            round_weights_arr,
        )

    return X_arr, y_arr, margins_arr, inc_engine.get_end_of_season_elo(), round_weights_arr
