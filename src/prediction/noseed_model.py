"""No-seed ML model for pool EV optimization.

Trains a logistic + GBM ensemble on non-seed Torvik four-factor features.
Backtest shows this produces better pool EV than seed-based probabilities
because it generates structural disagreement with the seed-thinking public,
creating leverage opportunities the optimizer can exploit.

Key finding: seed-based optimizer scores +0 vs chalk (can't disagree with
the crowd when your model IS seeds). No-seed model scores +9 mean edge.
"""

from __future__ import annotations

import json
import logging
from itertools import combinations
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.data.seed_pick_model import _compute_advancement_rates, _win_rate
from src.exceptions import LeakageError

logger = logging.getLogger(__name__)

HIST_DIR = Path("data/raw/historical")
DATA_DIR = Path("data/raw")

_VALID_PRETOURNAMENT_TYPES = {"pre_tournament"}


def _validate_pretournament(data: dict, filepath: Path) -> None:
    """Raise LeakageError if data file lacks pre-tournament provenance.

    Checks the data_type field written by the scraper/compute scripts.
    Raises on missing or wrong data_type — forces clean data regeneration
    rather than silently ingesting potentially contaminated files.
    """
    dt = data.get("data_type")
    if dt not in _VALID_PRETOURNAMENT_TYPES:
        raise LeakageError(
            f"{filepath}: data_type={dt!r}, expected one of {_VALID_PRETOURNAMENT_TYPES}. "
            f"File may contain post-tournament data (look-ahead bias). "
            f"Re-run rescrape_pretournament_torvik.py to regenerate from trank.php."
        )


TRAIN_YEARS = [
    2008,
    2009,
    2010,
    2011,
    2012,
    2013,
    2014,
    2015,
    2016,
    2017,
    2018,
    2019,
    2021,
    2022,
    2023,
    2024,
    2025,
]


def _sf(val, default):
    try:
        v = float(val)
        return v if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def _get_stat(stats, key, default):
    enriched = stats.get("enriched_stats", {})
    return _sf(stats.get(key) or enriched.get(key), default)


def _build_feature_vector(t1_stats: dict, t2_stats: dict) -> np.ndarray:
    """12-dim non-seed matchup feature vector."""
    return np.array(
        [
            _get_stat(t1_stats, "adj_offensive_efficiency", 100.0)
            - _get_stat(t2_stats, "adj_offensive_efficiency", 100.0),
            _get_stat(t1_stats, "adj_defensive_efficiency", 100.0)
            - _get_stat(t2_stats, "adj_defensive_efficiency", 100.0),
            _get_stat(t1_stats, "adj_tempo", 68.0) - _get_stat(t2_stats, "adj_tempo", 68.0),
            _get_stat(t1_stats, "effective_fg_pct", 0.50) - _get_stat(t2_stats, "effective_fg_pct", 0.50),
            _get_stat(t1_stats, "turnover_rate", 0.18) - _get_stat(t2_stats, "turnover_rate", 0.18),
            _get_stat(t1_stats, "offensive_reb_rate", 0.28) - _get_stat(t2_stats, "offensive_reb_rate", 0.28),
            _get_stat(t1_stats, "free_throw_rate", 0.35) - _get_stat(t2_stats, "free_throw_rate", 0.35),
            _get_stat(t1_stats, "opp_effective_fg_pct", 0.48) - _get_stat(t2_stats, "opp_effective_fg_pct", 0.48),
            _get_stat(t1_stats, "opp_turnover_rate", 0.18) - _get_stat(t2_stats, "opp_turnover_rate", 0.18),
            _get_stat(t1_stats, "defensive_reb_rate", 0.72) - _get_stat(t2_stats, "defensive_reb_rate", 0.72),
            _get_stat(t1_stats, "opp_free_throw_rate", 0.30) - _get_stat(t2_stats, "opp_free_throw_rate", 0.30),
            _get_stat(t1_stats, "barthag", 0.50) - _get_stat(t2_stats, "barthag", 0.50),
        ],
        dtype=np.float64,
    )


def _load_team_stats(year: int) -> dict:
    """Load Torvik team stats for a given year.

    Validates pre-tournament provenance on every file before reading. Raises
    LeakageError if a file's data_type indicates post-tournament data, preventing
    silent look-ahead bias from contaminating training or prediction.
    """
    stats = {}
    for prefix in [HIST_DIR, DATA_DIR]:
        torvik_path = prefix / f"torvik_{year}.json"
        if torvik_path.exists():
            with open(torvik_path) as f:
                data = json.load(f)
            _validate_pretournament(data, torvik_path)
            for t in data.get("teams", []):
                stats[t["team_id"]] = t
            break

    for prefix in [HIST_DIR, DATA_DIR]:
        ff_path = prefix / f"torvik_four_factors_{year}.json"
        if ff_path.exists():
            with open(ff_path) as f:
                ff_data = json.load(f)
            _validate_pretournament(ff_data, ff_path)
            if isinstance(ff_data, dict) and "teams" not in ff_data:
                for tid, ff in ff_data.items():
                    if tid in stats:
                        for k, v in ff.items():
                            if not k.startswith("_"):
                                stats[tid].setdefault(k, v)
                    else:
                        stats[tid] = ff
            break
    return stats


def _load_tournament_results(year: int) -> list:
    path = HIST_DIR / f"tournament_results_{year}.json"
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f).get("games", [])


class NoseedModel:
    """Ensemble of logistic regression + GBM trained without seed features."""

    def __init__(self, lr, scaler, gbm, train_years: Tuple[int, ...] = ()):
        self.lr = lr
        self.scaler = scaler
        self.gbm = gbm
        # Walk-forward provenance: the exact set of years used to fit this
        # model. Callers can assert `all(y < test_year for y in train_years)`
        # to prove no future-year data leaked into a backtest fold.
        self.train_years: Tuple[int, ...] = tuple(train_years)

    def predict_win_prob(self, t1_stats: dict, t2_stats: dict, sigma: float = 11.0) -> float:
        """Predict P(team1 wins) using no-seed ensemble."""
        X = _build_feature_vector(t1_stats, t2_stats).reshape(1, -1)
        p_lr = self.lr.predict_proba(self.scaler.transform(X))[0, 1]
        spread = self.gbm.predict(X)[0]
        p_gbm = 1.0 / (1.0 + np.exp(-spread / sigma))
        return 0.5 * p_lr + 0.5 * p_gbm


def train_noseed_model(max_year: Optional[int] = None) -> NoseedModel:
    """Train no-seed ensemble on historical tournament data.

    Args:
        max_year: If provided, only use training years < max_year
                  (for temporal honesty in backtesting). If None,
                  uses all available years.

    Returns:
        Trained NoseedModel.
    """
    train_years = [y for y in TRAIN_YEARS if max_year is None or y < max_year]
    if len(train_years) < 3:
        raise ValueError(f"Need >= 3 training years, got {len(train_years)}")

    X_list, y_list, margin_list = [], [], []

    # Symmetric augmentation: add both orientations of every game.
    # Since the feature vector is pure differentials (all dims are
    # t1_stat - t2_stat), swap(x) = -x. Appending both (x, y=1, +margin)
    # and (-x, y=0, -margin) enforces anti-symmetry as a hard data
    # constraint, forces the LR intercept to zero, and doubles the
    # effective training size (≈1000 games → ≈2000 rows). This replaces
    # the prior random-flip approach, which only used one orientation
    # per game and left the model's intercept as a free RNG-dependent
    # degree of freedom. Both orientations share the same year, so
    # LOYO integrity is preserved.
    for year in train_years:
        games = _load_tournament_results(year)
        stats = _load_team_stats(year)
        for g in games:
            if g.get("round_name") == "FF":
                continue
            t1, t2 = g["team1_id"], g["team2_id"]
            t1_stats = stats.get(t1, {})
            t2_stats = stats.get(t2, {})
            margin = g.get("team1_score", 0) - g.get("team2_score", 0)

            # Original orientation: raw data always has team1 as winner.
            X_list.append(_build_feature_vector(t1_stats, t2_stats))
            y_list.append(1)
            margin_list.append(margin)

            # Swapped orientation: same game from team2's perspective.
            X_list.append(_build_feature_vector(t2_stats, t1_stats))
            y_list.append(0)
            margin_list.append(-margin)

    X = np.array(X_list)
    y = np.array(y_list)
    margins = np.array(margin_list, dtype=float)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    lr = LogisticRegression(C=0.1, max_iter=1000, solver="lbfgs")
    lr.fit(X_scaled, y)

    gbm = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        min_samples_leaf=30,
        subsample=0.7,
        max_features=0.7,
        random_state=42,
    )
    gbm.fit(X, margins)

    logger.info("Trained no-seed model on %d games from %d years", len(X), len(train_years))
    return NoseedModel(lr, scaler, gbm, train_years=tuple(train_years))


def build_noseed_probabilities(
    model: NoseedModel,
    seeds: Dict[str, int],
    stats: Dict[str, dict],
) -> Dict[Tuple[str, str], float]:
    """Build pairwise win probabilities using the no-seed model."""
    probs: Dict[Tuple[str, str], float] = {}
    team_ids = list(seeds.keys())
    for t1, t2 in combinations(team_ids, 2):
        p = model.predict_win_prob(stats.get(t1, {}), stats.get(t2, {}))
        probs[(t1, t2)] = p
        probs[(t2, t1)] = 1.0 - p
    return probs


def build_noseed_round_probabilities(
    model: NoseedModel,
    seeds: Dict[str, int],
    stats: Dict[str, dict],
) -> Dict[str, Dict[str, float]]:
    """Build round advancement probs adjusted by no-seed model signal.

    Starts from seed-based advancement rates and adjusts based on
    each team's mean advantage/disadvantage vs the field according
    to the no-seed model.
    """
    seed_rates = _compute_advancement_rates()
    round_names = ["R64", "R32", "S16", "E8", "F4", "CHAMP"]
    result = {}

    for team_id, seed in seeds.items():
        base_rates = dict(seed_rates[seed])
        team_stats = stats.get(team_id, {})

        advantages = []
        for opp_id, opp_seed in seeds.items():
            if opp_id == team_id:
                continue
            opp_stats = stats.get(opp_id, {})
            noseed_p = model.predict_win_prob(team_stats, opp_stats)
            seed_p = _win_rate(seed, opp_seed)
            advantages.append(noseed_p - seed_p)

        mean_adv = np.mean(advantages) if advantages else 0.0

        adjusted = {}
        for i, rnd in enumerate(round_names):
            compound = (1.0 + mean_adv) ** (i + 1)
            adjusted[rnd] = max(0.001, min(0.99, base_rates[rnd] * compound))
        result[team_id] = adjusted

    return result


def build_blend_probabilities(
    seed_probs: Dict[Tuple[str, str], float],
    noseed_probs: Dict[Tuple[str, str], float],
    alpha: float = 0.5,
) -> Dict[Tuple[str, str], float]:
    """Blend seed-based and no-seed pairwise probabilities."""
    blended = {}
    for key in seed_probs:
        blended[key] = alpha * seed_probs[key] + (1.0 - alpha) * noseed_probs.get(key, 0.5)
    return blended


def build_blend_round_probabilities(
    seed_round: Dict[str, Dict[str, float]],
    noseed_round: Dict[str, Dict[str, float]],
    alpha: float = 0.5,
) -> Dict[str, Dict[str, float]]:
    """Blend seed-based and no-seed round probabilities."""
    blended = {}
    for team_id in seed_round:
        blended[team_id] = {}
        for rnd in seed_round[team_id]:
            sp = seed_round[team_id][rnd]
            np_ = noseed_round.get(team_id, {}).get(rnd, sp)
            blended[team_id][rnd] = alpha * sp + (1.0 - alpha) * np_
    return blended
