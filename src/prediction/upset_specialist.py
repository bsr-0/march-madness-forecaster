"""Upset specialist classifier for R64 coin-flip matchups.

Targets the four seed pairings where upsets are near 50/50:
  8v9, 7v10, 6v11, 5v12

Trains a logistic regression on per-game features derived from
pre-tournament data (torvik barthag, custom ratings). The specialist
can then override the default bracket picks on games where it has
high confidence in the upset.

Walk-forward safe: training only uses years < test_year.
Point-in-time safe: all features are pre-Selection-Sunday.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Seed matchups considered "coin flips" — historical upset rates ~35-55%.
UPSET_ELIGIBLE_MATCHUPS: Set[Tuple[int, int]] = {(5, 12), (6, 11), (7, 10), (8, 9)}

HIST_DIR = Path("data/raw/historical")

# Feature names for the specialist model.
FEATURE_NAMES = (
    "seed_diff",
    "barthag_diff",
    "srs_diff",
    "elo_slope_diff",
    "close_game_pct_diff",
    "ncsos_diff",
    "conf_tourney_depth_diff",
    "detector_upset_score",
)


def _load_torvik_barthag_map(year: int, data_root: Path = Path("data")) -> Dict[str, float]:
    """Load {canonical_team_id: barthag} from torvik_{year}.json."""
    for prefix in [data_root / "raw" / "historical", data_root / "raw"]:
        path = prefix / f"torvik_{year}.json"
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            teams = data.get("teams", [])
            return {t["team_id"]: float(t["barthag"]) for t in teams if t.get("barthag") is not None}
    return {}


def _load_custom_ratings_canonical(
    year: int,
    data_root: Path = Path("data"),
) -> Dict[str, Dict[str, float]]:
    """Load all custom ratings as {rating_name: {canonical_team_id: value}}.

    Returns ratings for: srs, elo_slope, close_game_pct, ncsos, conf_tourney_depth.
    """
    from src.data.features.custom_ratings import (
        compute_close_game_pct,
        compute_conf_tourney_depth,
        compute_elo_slope,
        compute_ncsos,
        compute_srs_ratings,
        ratings_to_canonical,
    )

    result: Dict[str, Dict[str, float]] = {}
    loaders = {
        "srs": compute_srs_ratings,
        "elo_slope": compute_elo_slope,
        "close_game_pct": compute_close_game_pct,
        "ncsos": compute_ncsos,
    }

    for name, func in loaders.items():
        try:
            raw = func(year, data_root)
            result[name] = ratings_to_canonical(raw, data_root)
        except Exception:
            logger.debug("Failed to load %s for %d", name, year)
            result[name] = {}

    # conf_tourney_depth has a different signature (no cutoff_day)
    try:
        raw = compute_conf_tourney_depth(year, data_root)
        result["conf_tourney_depth"] = ratings_to_canonical(raw, data_root)
    except Exception:
        logger.debug("Failed to load conf_tourney_depth for %d", year)
        result["conf_tourney_depth"] = {}

    return result


def _load_seeds_and_results(
    year: int,
    data_root: Path = Path("data"),
) -> Tuple[Dict[str, int], Dict[str, str], List[Dict[str, Any]]]:
    """Load seeds, regions, and tournament results for a year.

    Returns (seeds, regions, r64_games) where:
        seeds: {team_id: seed_number}
        regions: {team_id: region_name}
        r64_games: list of game dicts from tournament_results with round_name == 'R64'
    """
    hist = data_root / "raw" / "historical"

    # Seeds
    with open(hist / f"tournament_seeds_{year}.json") as f:
        seed_data = json.load(f)
    seeds = {}
    regions = {}
    for t in seed_data["teams"]:
        seeds[t["team_id"]] = int(t["seed"])
        regions[t["team_id"]] = t["region"]

    # Results
    with open(hist / f"tournament_results_{year}.json") as f:
        results_data = json.load(f)
    r64_games = [g for g in results_data["games"] if g["round_name"] == "R64"]

    return seeds, regions, r64_games


def _game_to_features(
    higher_seed_id: str,
    lower_seed_id: str,
    higher_seed: int,
    lower_seed: int,
    barthag: Dict[str, float],
    custom: Dict[str, Dict[str, float]],
    detector_scores: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """Build feature vector for a single upset-eligible game.

    Convention: higher_seed is the favored team (lower seed number).
    Features measure higher_seed - lower_seed (positive = favored team is stronger).
    The 8th feature is the underdog's rule-based detector upset score (0-1).
    """
    b_high = barthag.get(higher_seed_id, 0.5)
    b_low = barthag.get(lower_seed_id, 0.5)

    feats = [
        higher_seed - lower_seed,  # seed_diff (always negative, e.g. 5-12 = -7)
        b_high - b_low,  # barthag_diff
        custom.get("srs", {}).get(higher_seed_id, 0.0) - custom.get("srs", {}).get(lower_seed_id, 0.0),
        custom.get("elo_slope", {}).get(higher_seed_id, 0.0) - custom.get("elo_slope", {}).get(lower_seed_id, 0.0),
        custom.get("close_game_pct", {}).get(higher_seed_id, 0.5)
        - custom.get("close_game_pct", {}).get(lower_seed_id, 0.5),
        custom.get("ncsos", {}).get(higher_seed_id, 0.0) - custom.get("ncsos", {}).get(lower_seed_id, 0.0),
        custom.get("conf_tourney_depth", {}).get(higher_seed_id, 0.0)
        - custom.get("conf_tourney_depth", {}).get(lower_seed_id, 0.0),
        (detector_scores or {}).get(lower_seed_id, 0.0),  # underdog's detector score
    ]
    return np.array(feats, dtype=np.float64)


def _get_detector_scores(
    year: int,
    seeds: Dict[str, int],
    data_root: Path = Path("data"),
) -> Dict[str, float]:
    """Compute rule-based detector upset scores for all tournament teams.

    Returns {team_id: upset_score} or empty dict if detector unavailable.
    """
    try:
        from src.prediction.upset_detector import compute_upset_scores

        canonical_ids = list(seeds.keys())
        return compute_upset_scores(year, canonical_ids, seeds, data_root)
    except Exception:
        return {}


def build_upset_training_data(
    train_years: Sequence[int],
    data_root: Path = Path("data"),
) -> Tuple[np.ndarray, np.ndarray]:
    """Build (X, y) for upset-eligible R64 games across train_years.

    Only includes 5v12, 6v11, 7v10, 8v9 matchups.
    y = 1 if the lower-seeded team (upset) won, 0 if the higher seed won.

    Returns (X, y) where X has shape (n_games, 8) and y has shape (n_games,).
    """
    all_X = []
    all_y = []

    for year in train_years:
        try:
            seeds, regions, r64_games = _load_seeds_and_results(year, data_root)
        except (FileNotFoundError, KeyError) as e:
            logger.debug("Skipping year %d: %s", year, e)
            continue

        barthag = _load_torvik_barthag_map(year, data_root)
        custom = _load_custom_ratings_canonical(year, data_root)
        detector_scores = _get_detector_scores(year, seeds, data_root)

        for game in r64_games:
            s1 = game["team1_seed"]
            s2 = game["team2_seed"]
            seed_pair = tuple(sorted([s1, s2]))

            if seed_pair not in UPSET_ELIGIBLE_MATCHUPS:
                continue

            # Determine higher seed (favored) and lower seed (underdog)
            if s1 <= s2:
                higher_seed_id = game["team1_id"]
                lower_seed_id = game["team2_id"]
                higher_seed = s1
                lower_seed = s2
                # upset = lower seed won = team1 did NOT win
                upset = not game["team1_won"]
            else:
                higher_seed_id = game["team2_id"]
                lower_seed_id = game["team1_id"]
                higher_seed = s2
                lower_seed = s1
                # upset = lower seed won = team1 DID win
                upset = game["team1_won"]

            feats = _game_to_features(
                higher_seed_id,
                lower_seed_id,
                higher_seed,
                lower_seed,
                barthag,
                custom,
                detector_scores,
            )
            all_X.append(feats)
            all_y.append(1 if upset else 0)

    if not all_X:
        return np.empty((0, len(FEATURE_NAMES))), np.empty(0)

    return np.array(all_X), np.array(all_y)


def train_upset_specialist(
    X: np.ndarray,
    y: np.ndarray,
) -> Any:
    """Train logistic regression on upset-eligible games.

    Returns a fitted model with .predict_proba(X) -> P(upset).
    Uses LogisticRegression with L2 regularization — the dataset is small
    (~16 games/year, ~224 over 14 years) so a simple model is appropriate.
    """
    from sklearn.linear_model import LogisticRegression

    if len(X) < 10:
        logger.warning("Too few training samples (%d), returning None", len(X))
        return None

    # Replace any NaN with 0 (missing custom ratings)
    X_clean = np.nan_to_num(X, nan=0.0)

    model = LogisticRegression(
        C=1.0,
        penalty="l2",
        solver="lbfgs",
        max_iter=500,
        random_state=42,
    )
    model.fit(X_clean, y)
    return model


def _predict_upset_eligible_games(
    year: int,
    first_round: Sequence[str],
    seeds: Dict[str, int],
    model: Any,
    data_root: Path = Path("data"),
) -> List[Tuple[int, str, str, int, int, float]]:
    """Shared logic: predict P(upset) for all upset-eligible R64 games.

    Returns list of (game_idx, higher_seed_id, lower_seed_id,
    higher_seed, lower_seed, p_upset).
    """
    if model is None:
        return []

    barthag = _load_torvik_barthag_map(year, data_root)
    custom = _load_custom_ratings_canonical(year, data_root)
    detector_scores = _get_detector_scores(year, seeds, data_root)

    results = []
    for game_idx in range(32):
        t1 = first_round[game_idx * 2]
        t2 = first_round[game_idx * 2 + 1]

        s1 = seeds.get(t1, 0)
        s2 = seeds.get(t2, 0)

        seed_pair = tuple(sorted([s1, s2]))
        if seed_pair not in UPSET_ELIGIBLE_MATCHUPS:
            continue

        if s1 <= s2:
            higher_seed_id, lower_seed_id = t1, t2
            higher_seed, lower_seed = s1, s2
        else:
            higher_seed_id, lower_seed_id = t2, t1
            higher_seed, lower_seed = s2, s1

        feats = _game_to_features(
            higher_seed_id,
            lower_seed_id,
            higher_seed,
            lower_seed,
            barthag,
            custom,
            detector_scores,
        )
        X = np.nan_to_num(feats.reshape(1, -1), nan=0.0)
        proba = model.predict_proba(X)[0]
        p_upset = proba[1]

        results.append((game_idx, higher_seed_id, lower_seed_id, higher_seed, lower_seed, p_upset))

    return results


def get_upset_overrides(
    year: int,
    first_round: Sequence[str],
    seeds: Dict[str, int],
    model: Any,
    data_root: Path = Path("data"),
    threshold: float = 0.55,
) -> Dict[int, str]:
    """Predict upset probabilities for upset-eligible R64 games.

    For each upset-eligible game, if the specialist's P(upset) > threshold,
    return an override indicating the lower seed should win.

    Returns:
        Dict mapping R64 game index (0-31) to the predicted winner team_id.
    """
    predictions = _predict_upset_eligible_games(
        year,
        first_round,
        seeds,
        model,
        data_root,
    )

    overrides: Dict[int, str] = {}
    for game_idx, higher_id, lower_id, hs, ls, p_upset in predictions:
        if p_upset > threshold:
            overrides[game_idx] = lower_id
            logger.info(
                "Year %d game %d: %s(%d) vs %s(%d) -> upset P=%.3f > %.2f, picking %s",
                year,
                game_idx,
                higher_id,
                hs,
                lower_id,
                ls,
                p_upset,
                threshold,
                lower_id,
            )
        elif p_upset < (1.0 - threshold):
            overrides[game_idx] = higher_id

    return overrides


def get_upset_probabilities(
    year: int,
    first_round: Sequence[str],
    seeds: Dict[str, int],
    model: Any,
    data_root: Path = Path("data"),
) -> Dict[int, Tuple[str, str, float]]:
    """Return calibrated P(upset) for all upset-eligible R64 games.

    Returns:
        {game_idx: (higher_seed_id, lower_seed_id, p_upset)} for every
        upset-eligible game (5v12, 6v11, 7v10, 8v9).
    """
    predictions = _predict_upset_eligible_games(
        year,
        first_round,
        seeds,
        model,
        data_root,
    )
    return {game_idx: (higher_id, lower_id, p_upset) for game_idx, higher_id, lower_id, _, _, p_upset in predictions}


def apply_upset_overrides(
    bracket: np.ndarray,
    overrides: Dict[int, str],
    first_round: Sequence[str],
    seeds: Dict[str, int],
) -> np.ndarray:
    """Apply upset specialist overrides to an existing bracket.

    Flips R64 picks where the specialist disagrees with the base bracket.
    Does NOT propagate to later rounds — the region beam search already
    builds path-consistent brackets, so only R64 picks are changed.
    Later rounds keep whatever the beam search chose.

    Args:
        bracket: (63,) boolean array from the base bracket builder.
        overrides: {game_index: winner_team_id} from get_upset_overrides.
        first_round: 64 team IDs in bracket order.
        seeds: {team_id: seed_number}.

    Returns:
        Modified (63,) boolean array with overrides applied.
    """
    if not overrides:
        return bracket

    result = bracket.copy()

    for game_idx, winner in overrides.items():
        if game_idx >= 32:
            continue  # Only R64

        t1 = first_round[game_idx * 2]
        # True = team1 (higher bracket position) wins
        result[game_idx] = winner == t1

    return result
