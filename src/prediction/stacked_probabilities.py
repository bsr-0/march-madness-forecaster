"""Stacked source (catalog B5).

Ridge regression meta-learner that learns optimal blend weights across
all Category-A probability bases from historical tournament game outcomes.

Every other blend in the catalog is a human-chosen convex combination
(0.5*x+0.5*y). Stacked is the only source where the data picks the
weights: fit Ridge on game-level barthag differentials from prior
tournaments, then apply the learned coefficients to blend the test
year's per-team barthag values.

Walk-forward contract: at test_year Y, the Ridge is fitted only on
tournament games from years strictly < Y. Each test year's weight
vector is computed in isolation. 2020 excluded (COVID — no tournament).

Algorithm:
1. For each prior year y in [start, test_year):
   a. Load barthag from every Category-A source for that year's field.
   b. Load tournament games from ``tournament_results_{y}.json``.
   c. For each game: build feature vector = per-source barthag diff
      (team1 − team2). Target = 1.0 if team1 won, 0.0 otherwise.
   d. Skip the year if any Category-A source is unavailable.
2. Collect all (feature_vector, outcome) pairs across usable years.
3. If fewer than ``min_prior_years`` usable years: return None.
4. Fit ``Ridge(alpha=1.0, fit_intercept=False)`` on the feature matrix.
5. For the test year: blended_barthag[team] = sum(coef_i × barthag_i[team]).
6. Normalize to [0, 1] via min-max across the field (preserves ranking).
7. Clip to [0.10, 0.99] per catalog convention.

Why Ridge, not OLS: with 8 highly correlated features (all measure
team strength), OLS can produce wildly unstable coefficients. Ridge
regularization (L2 penalty, alpha=1.0) shrinks coefficients toward
equal weight — exactly the right prior for nearly-interchangeable
barthag sources. The learned weights capture the marginal accuracy
of each source conditional on the others.

Why ``fit_intercept=False``: the barthag diffs are symmetric around
zero (equally likely that team1 is stronger or weaker than team2).
An intercept would bias the prediction toward one side, which is
conceptually wrong for a pairwise comparison model.

Data dependencies:
- ``data/raw/historical/tournament_seeds_{year}.json`` (all Category-A
  sources need seeds for fallback).
- ``data/raw/historical/tournament_results_{year}.json`` (target variable).
- All 8 Category-A source loaders (A1–A8).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Category-A source loaders (all must return Optional[Dict[str, float]])
from src.prediction.ap_probabilities import load_ap_strength_barthag
from src.prediction.elo_probabilities import load_elo_barthag
from src.prediction.market_probabilities import load_market_ratings, load_spread_power_ratings
from src.prediction.massey_best_probabilities import load_massey_best_barthag
from src.prediction.massey_probabilities import load_massey_avg_barthag

# Category-A base names in stable order (determines feature columns)
_BASE_NAMES = (
    "seed",  # A1
    "torvik",  # A2
    "odds",  # A3
    "elo",  # A4
    "massey_avg",  # A5
    "massey_best",  # A6
    "spread_power",  # A7
    "ap_strength",  # A8
)

_EXCLUDED_YEARS = {2020}
_VALID_ROUND_NAMES = {"R64", "R32", "S16", "E8", "F4", "CHAMP"}
_DEFAULT_MIN_PRIOR_YEARS = 3
_DEFAULT_RIDGE_ALPHA = 1.0


def _load_seeds(year: int, data_root: Path) -> Dict[str, int]:
    """Load {team_id → seed} from tournament_seeds_{year}.json."""
    path = data_root / "raw" / "historical" / f"tournament_seeds_{year}.json"
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    seeds: Dict[str, int] = {}
    if isinstance(data, dict) and "teams" in data:
        for t in data["teams"]:
            seeds[t["team_id"]] = t["seed"]
    return seeds


def _load_seed_barthag(seeds: Dict[str, int]) -> Dict[str, float]:
    """A1 seed: deterministic barthag from seed number. Always available."""
    return {tid: max(0.10, 1.0 - seed * 0.04) for tid, seed in seeds.items()}


def _load_torvik_barthag(year: int, seeds: Dict[str, int], data_root: Path) -> Optional[Dict[str, float]]:
    """A2 torvik: load from torvik_{year}.json."""
    path = data_root / "raw" / "historical" / f"torvik_{year}.json"
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    barthag: Dict[str, float] = {}
    for t in data.get("teams", []):
        tid = t.get("team_id", "")
        b = t.get("barthag")
        if tid in seeds and b is not None:
            barthag[tid] = float(b)
    # Fill missing teams with seed fallback
    for tid, seed in seeds.items():
        if tid not in barthag:
            barthag[tid] = max(0.10, 1.0 - seed * 0.04)
    return barthag


def _load_all_base_barthag(
    year: int,
    seeds: Dict[str, int],
    data_root: Path,
) -> Optional[Dict[str, Dict[str, float]]]:
    """Load barthag from all 8 Category-A bases for a given year.

    Returns {base_name: {team_id: barthag}} or None if any base is
    unavailable for the year.
    """
    result: Dict[str, Dict[str, float]] = {}

    # A1: seed (always available)
    result["seed"] = _load_seed_barthag(seeds)

    # A2: torvik
    torvik = _load_torvik_barthag(year, seeds, data_root)
    if torvik is None:
        return None
    result["torvik"] = torvik

    # A3: odds (Bradley-Terry from unified_odds)
    odds = load_market_ratings(year, seeds)
    if odds is None:
        return None
    result["odds"] = odds

    # A4: elo
    elo = load_elo_barthag(year, seeds, data_root)
    if elo is None:
        return None
    result["elo"] = elo

    # A5: massey_avg
    massey_avg = load_massey_avg_barthag(year, seeds, data_root)
    if massey_avg is None:
        return None
    result["massey_avg"] = massey_avg

    # A6: massey_best
    massey_best = load_massey_best_barthag(year, data_root)
    if massey_best is None:
        return None
    result["massey_best"] = massey_best

    # A7: spread_power
    spread_power = load_spread_power_ratings(year, seeds)
    if spread_power is None:
        return None
    result["spread_power"] = spread_power

    # A8: ap_strength
    ap = load_ap_strength_barthag(year, seeds, seeds.keys(), data_root)
    if ap is None:
        return None
    result["ap_strength"] = ap

    return result


def _load_tournament_games(year: int, data_root: Path) -> Optional[List[dict]]:
    """Load tournament games list for one year, or None if missing."""
    path = data_root / "raw" / "historical" / f"tournament_results_{year}.json"
    if not path.exists():
        return None
    with open(path) as f:
        raw = json.load(f)
    if isinstance(raw, dict) and "games" in raw:
        return raw["games"]
    if isinstance(raw, list):
        return raw
    return None


def _build_training_data(
    test_year: int,
    data_root: Path,
    min_prior_years: int = _DEFAULT_MIN_PRIOR_YEARS,
) -> Optional[Tuple[np.ndarray, np.ndarray, int]]:
    """Build (X, y, n_usable_years) for Ridge training.

    X: (n_games, 8) — barthag diff per Category-A base.
    y: (n_games,) — 1.0 if team1 won, 0.0 otherwise.

    Returns None if fewer than ``min_prior_years`` have all 8 bases.
    """
    all_features: List[List[float]] = []
    all_targets: List[float] = []
    n_usable_years = 0

    for year in range(2005, test_year):
        if year in _EXCLUDED_YEARS:
            continue

        seeds = _load_seeds(year, data_root)
        if not seeds:
            continue

        base_barthag = _load_all_base_barthag(year, seeds, data_root)
        if base_barthag is None:
            continue

        games = _load_tournament_games(year, data_root)
        if games is None:
            continue

        year_game_count = 0
        for g in games:
            rnd = g.get("round_name")
            if rnd not in _VALID_ROUND_NAMES:
                continue
            t1 = g.get("team1_id")
            t2 = g.get("team2_id")
            if not t1 or not t2:
                continue

            # Both teams must have barthag from every base
            row: List[float] = []
            skip = False
            for base in _BASE_NAMES:
                b1 = base_barthag[base].get(t1)
                b2 = base_barthag[base].get(t2)
                if b1 is None or b2 is None:
                    skip = True
                    break
                row.append(b1 - b2)
            if skip:
                continue

            actual = 1.0 if g.get("team1_won") else 0.0
            all_features.append(row)
            all_targets.append(actual)
            year_game_count += 1

        if year_game_count > 0:
            n_usable_years += 1

    if n_usable_years < min_prior_years:
        return None

    X = np.array(all_features, dtype=np.float64)
    y = np.array(all_targets, dtype=np.float64)
    return X, y, n_usable_years


def fit_stacked_weights(
    test_year: int,
    data_root: Path,
    min_prior_years: int = _DEFAULT_MIN_PRIOR_YEARS,
    ridge_alpha: float = _DEFAULT_RIDGE_ALPHA,
) -> Optional[np.ndarray]:
    """Walk-forward Ridge fit: learn blend weights from years < test_year.

    Returns coefficient array of shape (8,) — one weight per Category-A
    base in ``_BASE_NAMES`` order — or None if insufficient training data.
    """
    training = _build_training_data(test_year, data_root, min_prior_years)
    if training is None:
        return None
    X, y, _ = training

    from sklearn.linear_model import Ridge

    ridge = Ridge(alpha=ridge_alpha, fit_intercept=False)
    ridge.fit(X, y)
    return ridge.coef_


def load_stacked_barthag(
    test_year: int,
    seeds: Dict[str, int],
    data_root: Optional[Path] = None,
    min_prior_years: int = _DEFAULT_MIN_PRIOR_YEARS,
    ridge_alpha: float = _DEFAULT_RIDGE_ALPHA,
) -> Optional[Dict[str, float]]:
    """Ridge-blended barthag per tournament team for the given year.

    Args:
        test_year: Tournament year.
        seeds: Canonical tournament team ID → seed (1-16).
        data_root: Repo data root. Defaults to ``<project>/data``.
        min_prior_years: Minimum usable prior years for Ridge training.
        ridge_alpha: Ridge regularization strength.

    Returns:
        ``Dict[canonical_id, barthag ∈ [0.10, 0.99]]`` for every team in
        ``seeds``. Returns ``None`` if insufficient training data or if
        any Category-A source is unavailable for the test year.
    """
    if data_root is None:
        data_root = Path(__file__).resolve().parent.parent.parent / "data"

    # Step 1: Fit Ridge weights on years < test_year
    weights = fit_stacked_weights(test_year, data_root, min_prior_years, ridge_alpha)
    if weights is None:
        return None

    # Step 2: Load all Category-A barthag for the test year
    base_barthag = _load_all_base_barthag(test_year, seeds, data_root)
    if base_barthag is None:
        return None

    # Step 3: Blend barthag using learned weights
    raw_blended: Dict[str, float] = {}
    for tid in seeds:
        value = 0.0
        for i, base in enumerate(_BASE_NAMES):
            b = base_barthag[base].get(tid, 0.5)
            value += weights[i] * b
        raw_blended[tid] = value

    # Step 4: Normalize to [0, 1] via min-max, then clip to [0.10, 0.99]
    vals = list(raw_blended.values())
    v_min = min(vals)
    v_max = max(vals)
    spread = v_max - v_min
    if spread < 1e-12:
        # All teams blended to the same value — degenerate; fall back to
        # equal-weight average
        return None

    result: Dict[str, float] = {}
    for tid, v in raw_blended.items():
        normed = (v - v_min) / spread
        result[tid] = max(0.10, min(0.99, normed))
    return result


def build_stacked_round_probabilities(
    seeds: Dict[str, int],
    regions: Dict[str, str],
    test_year: int,
    data_root: Path,
    n_sims: int = 10000,
    min_prior_years: int = _DEFAULT_MIN_PRIOR_YEARS,
    ridge_alpha: float = _DEFAULT_RIDGE_ALPHA,
) -> Optional[Dict[str, Dict[str, float]]]:
    """End-to-end: fit Ridge weights → blend barthag → MC round_probs.

    Uses the same MC construction as torvik (``build_torvik_round_probabilities``).
    Returns None if insufficient training data or source unavailability.
    """
    barthag = load_stacked_barthag(
        test_year,
        seeds,
        data_root,
        min_prior_years=min_prior_years,
        ridge_alpha=ridge_alpha,
    )
    if barthag is None:
        return None

    from scripts.mc_pool_backtest import build_torvik_round_probabilities

    return build_torvik_round_probabilities(seeds, regions, barthag, n_sims=n_sims)
