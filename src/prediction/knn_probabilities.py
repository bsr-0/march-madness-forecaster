"""KNN matchup probability base (catalog: KNN).

Predicts tournament team strength by finding the K most similar historical
tournament teams and aggregating their actual tournament performance.

Algorithm:
1. For each season, compute an 8-dim feature vector per team from
   regular-season games (DayNum < 133 = Selection Sunday, walk-forward safe).
2. For each tournament team in the test year, find the K nearest neighbors
   in the historical database (years strictly < test year).
3. Aggregate neighbors' actual tournament wins into a barthag-equivalent
   strength estimate, weighted by inverse distance.
4. Blend with a seed-based prior for regularization.

Features (all from MRegularSeasonCompactResults.csv):
  0. seed (tournament seed, 1-16)
  1. win_pct (regular-season win percentage)
  2. avg_margin (average point differential)
  3. sos (strength of schedule, iterative)
  4. away_win_pct (road + neutral win percentage)
  5. close_game_pct (win% in games decided by <= 7 points)
  6. ppg (points per game scored)
  7. opp_ppg (points per game allowed)

Output: Dict[canonical_id, barthag in (0, 1)] — same shape as
``load_elo_barthag`` or ``load_market_ratings``, consumed by
``build_torvik_round_probabilities`` for MC simulation.
"""

from __future__ import annotations

import csv
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_CUTOFF_DAY = 133  # Selection Sunday
DEFAULT_K = 15
MIN_HISTORY_YEARS = 5  # need at least this many prior tournament years
SOS_ITERATIONS = 10
CLOSE_GAME_MARGIN = 7
SEED_PRIOR_WEIGHT = 0.3  # blend: (1-w)*knn + w*seed_prior


def _load_season_games(
    season: int,
    data_root: Path = Path("data"),
    cutoff_day: int = DEFAULT_CUTOFF_DAY,
) -> List[Tuple[int, int, int, int, str]]:
    """Load regular-season games as (winner_id, loser_id, w_score, l_score, location).

    Only includes games with DayNum < cutoff_day to prevent tournament leakage.
    """
    for filename in ("MRegularSeasonCompactResults.csv", "MRegularSeasonDetailedResults.csv"):
        path = data_root / "kaggle" / filename
        if not path.exists():
            continue
        games = []
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if int(row["Season"]) != season:
                    continue
                if int(row["DayNum"]) >= cutoff_day:
                    continue
                games.append(
                    (
                        int(row["WTeamID"]),
                        int(row["LTeamID"]),
                        int(row["WScore"]),
                        int(row["LScore"]),
                        row.get("WLoc", "N"),
                    )
                )
        if games:
            return games
    return []


def _load_tournament_seeds(
    season: int,
    data_root: Path = Path("data"),
) -> Dict[int, int]:
    """Load tournament seeds as {kaggle_team_id: seed_number}.

    Parses the Kaggle MNCAATourneySeeds.csv format where seed is like 'W01'.
    """
    path = data_root / "kaggle" / "MNCAATourneySeeds.csv"
    if not path.exists():
        return {}
    seeds: Dict[int, int] = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["Season"]) != season:
                continue
            seed_str = row["Seed"]
            seed_num = int(seed_str[1:3])
            seeds[int(row["TeamID"])] = seed_num
    return seeds


def _load_tournament_wins(
    season: int,
    data_root: Path = Path("data"),
) -> Dict[int, int]:
    """Count tournament wins per team for a season.

    Returns {kaggle_team_id: n_tournament_wins}. Range 0-6 (champion has 6).
    Teams that made the tournament but lost R64 have 0 wins.
    """
    path = data_root / "kaggle" / "MNCAATourneyCompactResults.csv"
    if not path.exists():
        return {}
    wins: Dict[int, int] = defaultdict(int)
    for filename in ("MNCAATourneyCompactResults.csv", "MNCAATourneyDetailedResults.csv"):
        fpath = data_root / "kaggle" / filename
        if not fpath.exists():
            continue
        with open(fpath) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if int(row["Season"]) != season:
                    continue
                wins[int(row["WTeamID"])] += 1
        break  # use first available file
    return dict(wins)


def compute_season_features(
    season: int,
    data_root: Path = Path("data"),
    cutoff_day: int = DEFAULT_CUTOFF_DAY,
) -> Dict[int, np.ndarray]:
    """Compute 8-dim feature vector per team for a season.

    Returns {kaggle_team_id: np.array of shape (8,)}.
    Only teams with >= 10 games are included.
    """
    games = _load_season_games(season, data_root, cutoff_day)
    if not games:
        return {}

    # Accumulate per-team stats
    team_wins: Dict[int, int] = defaultdict(int)
    team_losses: Dict[int, int] = defaultdict(int)
    team_points_for: Dict[int, int] = defaultdict(int)
    team_points_against: Dict[int, int] = defaultdict(int)
    team_margins: Dict[int, List[int]] = defaultdict(list)
    team_opponents: Dict[int, List[int]] = defaultdict(list)
    # Away/neutral games (team was not home)
    team_away_wins: Dict[int, int] = defaultdict(int)
    team_away_games: Dict[int, int] = defaultdict(int)
    # Close games (margin <= 7)
    team_close_wins: Dict[int, int] = defaultdict(int)
    team_close_games: Dict[int, int] = defaultdict(int)

    for w_id, l_id, w_score, l_score, loc in games:
        margin = w_score - l_score

        # Winner stats
        team_wins[w_id] += 1
        team_points_for[w_id] += w_score
        team_points_against[w_id] += l_score
        team_margins[w_id].append(margin)
        team_opponents[w_id].append(l_id)

        # Loser stats
        team_losses[l_id] += 1
        team_points_for[l_id] += l_score
        team_points_against[l_id] += w_score
        team_margins[l_id].append(-margin)
        team_opponents[l_id].append(w_id)

        # Away/neutral tracking
        if loc == "H":
            # Winner was home → loser was away
            team_away_games[l_id] += 1
        elif loc == "A":
            # Winner was away
            team_away_games[w_id] += 1
            team_away_wins[w_id] += 1
        else:
            # Neutral
            team_away_games[w_id] += 1
            team_away_wins[w_id] += 1
            team_away_games[l_id] += 1

        # Close game tracking
        if margin <= CLOSE_GAME_MARGIN:
            team_close_games[w_id] += 1
            team_close_wins[w_id] += 1
            team_close_games[l_id] += 1

    # Compute SOS iteratively
    all_teams = set(team_wins.keys()) | set(team_losses.keys())
    team_avg_margin: Dict[int, float] = {}
    for t in all_teams:
        n = team_wins[t] + team_losses[t]
        if n > 0:
            team_avg_margin[t] = sum(team_margins[t]) / n
        else:
            team_avg_margin[t] = 0.0

    sos: Dict[int, float] = {t: 0.0 for t in all_teams}
    for _ in range(SOS_ITERATIONS):
        new_sos: Dict[int, float] = {}
        for t in all_teams:
            opps = team_opponents[t]
            if opps:
                new_sos[t] = np.mean([team_avg_margin[o] + sos[o] for o in opps])
            else:
                new_sos[t] = 0.0
        sos = new_sos

    # Load seeds for this season (0 if not a tournament team)
    tourney_seeds = _load_tournament_seeds(season, data_root)

    # Build feature vectors
    result: Dict[int, np.ndarray] = {}
    for t in all_teams:
        n_games = team_wins[t] + team_losses[t]
        if n_games < 10:
            continue

        win_pct = team_wins[t] / n_games
        avg_margin = team_avg_margin[t]
        ppg = team_points_for[t] / n_games
        opp_ppg = team_points_against[t] / n_games

        away_n = team_away_games[t]
        away_win_pct = team_away_wins[t] / away_n if away_n > 0 else win_pct

        close_n = team_close_games[t]
        close_pct = team_close_wins[t] / close_n if close_n > 0 else 0.5

        seed = tourney_seeds.get(t, 0)

        result[t] = np.array(
            [
                float(seed),
                win_pct,
                avg_margin,
                sos[t],
                away_win_pct,
                close_pct,
                ppg,
                opp_ppg,
            ],
            dtype=np.float64,
        )

    return result


def build_historical_database(
    test_year: int,
    data_root: Path = Path("data"),
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Build historical feature matrix from tournament teams in years < test_year.

    Returns:
        X: Feature matrix (N, 8) — z-score normalized
        y: Tournament wins per team (N,) — range 0-6
        n_years: Number of years included

    Only tournament teams (those with seeds) are included.
    """
    all_features: List[np.ndarray] = []
    all_wins: List[int] = []
    n_years = 0

    # Kaggle data starts at 1985, tournament compact results too
    start_year = 1985
    for year in range(start_year, test_year):
        season_feats = compute_season_features(year, data_root)
        tourney_seeds = _load_tournament_seeds(year, data_root)
        tourney_wins = _load_tournament_wins(year, data_root)

        if not tourney_seeds or not season_feats:
            continue

        year_count = 0
        for tid, seed in tourney_seeds.items():
            if tid not in season_feats:
                continue
            feat = season_feats[tid].copy()
            # Overwrite seed feature with actual tournament seed
            feat[0] = float(seed)
            wins = tourney_wins.get(tid, 0)
            all_features.append(feat)
            all_wins.append(wins)
            year_count += 1

        if year_count > 0:
            n_years += 1

    if not all_features:
        return np.array([]), np.array([]), 0

    X = np.stack(all_features)
    y = np.array(all_wins, dtype=np.float64)
    return X, y, n_years


def _seed_fallback_barthag(seed: int) -> float:
    """Seed-based barthag fallback (same formula as elo_probabilities)."""
    return max(0.10, 1.0 - seed * 0.04)


def _knn_predict(
    query: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    k: int,
    mean: np.ndarray,
    std: np.ndarray,
) -> float:
    """Find K nearest neighbors and return inverse-distance-weighted mean of labels.

    Args:
        query: (8,) raw feature vector for the test team
        X_train: (N, 8) z-scored training features
        y_train: (N,) training labels (tournament wins, 0-6)
        k: number of neighbors
        mean: (8,) feature means for z-scoring
        std: (8,) feature stds for z-scoring

    Returns:
        Weighted mean of neighbor labels (0-6 scale).
    """
    # Z-score the query using training stats
    q_normed = (query - mean) / std

    # Euclidean distances to all training points
    dists = np.sqrt(np.sum((X_train - q_normed) ** 2, axis=1))

    # Get K nearest
    k_actual = min(k, len(dists))
    if k_actual == len(dists):
        idx = np.arange(k_actual)
    else:
        idx = np.argpartition(dists, k_actual)[:k_actual]
    k_dists = dists[idx]
    k_labels = y_train[idx]

    # Inverse-distance weighting (add epsilon to avoid division by zero)
    eps = 1e-6
    weights = 1.0 / (k_dists + eps)
    return float(np.average(k_labels, weights=weights))


def load_knn_barthag(
    year: int,
    seeds: Dict[str, int],
    data_root: Optional[Path] = None,
    k: int = DEFAULT_K,
) -> Optional[Dict[str, float]]:
    """KNN-derived barthag per tournament team for the given year.

    Args:
        year: Tournament year.
        seeds: Canonical tournament team ID -> seed (1-16).
        data_root: Repo data root. Defaults to <project>/data.
        k: Number of nearest neighbors.

    Returns:
        Dict[canonical_id, barthag in (0, 1)] for every team in seeds.
        Returns None when insufficient historical data.
    """
    if data_root is None:
        data_root = Path(__file__).resolve().parent.parent.parent / "data"
    data_root = Path(data_root)

    # Build historical database (walk-forward: years < test_year only)
    X_hist, y_hist, n_years = build_historical_database(year, data_root)

    if n_years < MIN_HISTORY_YEARS or len(X_hist) == 0:
        logger.warning("KNN: only %d history years for %d, need %d", n_years, year, MIN_HISTORY_YEARS)
        return None

    # Z-score normalization fitted on historical data
    mean = X_hist.mean(axis=0)
    std = X_hist.std(axis=0)
    std[std < 1e-8] = 1.0  # avoid division by zero for constant features
    X_normed = (X_hist - mean) / std

    # Compute features for test-year teams
    season_feats = compute_season_features(year, data_root)

    # Build Kaggle ID -> canonical ID mapping
    from src.data.features.custom_ratings import ratings_to_canonical

    # We need the reverse: canonical_id -> kaggle_team_id for the test year
    kaggle_seeds = _load_tournament_seeds(year, data_root)
    # Build kaggle_id -> canonical_id via MTeams.csv
    teams_path = data_root / "kaggle" / "MTeams.csv"
    kaggle_to_name: Dict[int, str] = {}
    if teams_path.exists():
        from src.data.normalize import normalize_team_id

        with open(teams_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                kaggle_to_name[int(row["TeamID"])] = normalize_team_id(row["TeamName"])

    # Map canonical seeds to kaggle IDs
    canonical_to_kaggle: Dict[str, int] = {}
    for kid, seed in kaggle_seeds.items():
        canon = kaggle_to_name.get(kid)
        if canon and canon in seeds:
            canonical_to_kaggle[canon] = kid

    # Predict barthag for each tournament team
    result: Dict[str, float] = {}
    for canonical_id, seed in seeds.items():
        kid = canonical_to_kaggle.get(canonical_id)

        if kid is not None and kid in season_feats:
            feat = season_feats[kid].copy()
            feat[0] = float(seed)  # use actual tournament seed
            knn_wins = _knn_predict(feat, X_normed, y_hist, k, mean, std)
            # Convert wins (0-6) to barthag (0-1)
            knn_barthag = knn_wins / 6.0
            # Blend with seed prior
            seed_barthag = _seed_fallback_barthag(seed)
            barthag = (1.0 - SEED_PRIOR_WEIGHT) * knn_barthag + SEED_PRIOR_WEIGHT * seed_barthag
        else:
            # No season data — fall back to seed
            barthag = _seed_fallback_barthag(seed)
            logger.debug("KNN: no season data for %s (seed %d), using seed fallback", canonical_id, seed)

        # Clamp to valid range
        result[canonical_id] = float(np.clip(barthag, 0.05, 0.95))

    return result
