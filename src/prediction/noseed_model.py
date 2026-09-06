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
    # 2026 added for prospective spec 2027.v2 (2026-08-20). The 2026 season
    # concluded in April 2026, so it is ordinary historical data for any 2027
    # prediction. The walk-forward filter in train_noseed_model
    # (`y < max_year`) means this changes nothing for any test year <= 2026 --
    # only the 2027 prediction path sees it. 2026 remains permanently barred as
    # an EVALUATION season (PROSPECTIVE_2027.md); this concerns training only.
    2026,
]


def _sf(val, default):
    try:
        v = float(val)
        return v if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def _get_stat(stats, key, default):
    """Read one stat, falling back to ``enriched_stats`` then to ``default``.

    The ``is None`` checks are deliberate. This used to read
    ``stats.get(key) or enriched.get(key)``, which treats a legitimate ``0.0``
    as missing and silently substitutes the default — the classic falsy-value
    coalescing bug. Four-factor rates are never exactly zero in practice so it
    changed nothing measurable here, but it is wrong and would bite any future
    feature that can legitimately be zero.

    Note the per-key default is a *tolerance for one team missing one stat*, not
    a licence to run without a feature at all. When a key is absent for the
    whole payload, every team gets the same default, every differential is
    exactly 0.0 — a perfectly plausible feature value — and the model degrades
    to a coin flip with nothing in the output to indicate it. That is what
    :func:`validate_stats_payload` exists to catch.
    """
    val = stats.get(key)
    if val is None:
        val = (stats.get("enriched_stats") or {}).get(key)
    return _sf(val, default)


# Every key _build_feature_vector reads, paired with the dimension it feeds.
# Kept adjacent to that function so the two cannot drift apart.
REQUIRED_FEATURE_KEYS: Tuple[str, ...] = (
    "adj_offensive_efficiency",
    "adj_defensive_efficiency",
    "adj_tempo",
    "effective_fg_pct",
    "turnover_rate",
    "offensive_reb_rate",
    "free_throw_rate",
    "opp_effective_fg_pct",
    "opp_turnover_rate",
    "defensive_reb_rate",
    "opp_free_throw_rate",
    "barthag",
)


class FeatureSkewError(RuntimeError):
    """Raised when a stats payload cannot populate the model's feature vector.

    This is the train/serve skew guard. Its absence cost the project a
    measurable amount: ``mc_pool_backtest`` passed a four-factors-only dict to a
    model trained on 12 features, so ``adj_offensive_efficiency``,
    ``adj_defensive_efficiency``, ``adj_tempo`` and ``barthag`` were identically
    zero for every matchup. The model returned ~0.5 on everything — 17/32
    agreement with the seed favourite, i.e. chance, and 1-seeds over 16-seeds at
    0.474-0.540 — and nothing in the pipeline said so. See FINDINGS.md 6c.
    """


def validate_stats_payload(
    stats: Dict[str, dict],
    *,
    min_coverage: float = 0.5,
    context: str = "",
) -> None:
    """Assert a stats payload can actually populate the feature vector.

    Checks *coverage per feature key* across the payload rather than presence
    for a single team, because that is the shape of the failure worth catching:
    one team missing one stat is normal and the per-key default handles it,
    while a key missing for everyone is train/serve skew.

    Args:
        min_coverage: minimum fraction of teams that must carry each key.
        context: free-text label for the error message (e.g. "inference").

    Raises:
        FeatureSkewError: listing every under-covered key and its coverage.
    """
    if not stats:
        raise FeatureSkewError(f"empty stats payload{f' ({context})' if context else ''}")

    teams = [t for t in stats.values() if isinstance(t, dict)]
    if not teams:
        raise FeatureSkewError(f"stats payload has no team dicts{f' ({context})' if context else ''}")

    n = len(teams)
    bad = []
    for key in REQUIRED_FEATURE_KEYS:
        present = sum(
            1 for t in teams if t.get(key) is not None or (t.get("enriched_stats") or {}).get(key) is not None
        )
        coverage = present / n
        if coverage < min_coverage:
            bad.append((key, coverage))

    if bad:
        detail = ", ".join(f"{k} ({c:.0%})" for k, c in bad)
        raise FeatureSkewError(
            f"stats payload{f' ({context})' if context else ''} is missing required "
            f"feature keys across {n} teams: {detail}. Every team would fall back to "
            f"the same default, making those feature differentials identically zero "
            f"and the model's output uninformative. This is train/serve skew — supply "
            f"the full team-stats payload (noseed_model._load_team_stats), not a "
            f"four-factors-only dict."
        )


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

    Also refuses to return an unusable payload, which is the FINDINGS.md 6c
    failure reached by its other route. This function used to return ``{}`` when
    neither directory held ``torvik_{year}.json`` -- the ordinary state of a new
    season before its pre-tournament snapshot is scraped. Every team then fell
    through to per-key defaults, every differential became 0.0, and the model
    returned ~0.5 for every matchup while the pipeline reported success.

    The guard lives here rather than at a call site because there are six
    callers -- ``mc_pool_backtest``, ``build_candidate_artifact``, the ``mmf
    pool`` noseed and blend modes in ``cli/pool_cmds.py``, the recency fitter
    and ``train_noseed_model`` -- and a guard placed on one of them is exactly
    the mistake 6c-ii records: the first fix was attached to training, where the
    payload is assembled, rather than to inference, where the defect showed up.
    No caller wants an empty payload, so none needs an opt-out.

    Raises:
        FileNotFoundError: if no snapshot exists for ``year``.
        FeatureSkewError: if a snapshot exists but cannot populate the feature
            vector.
    """
    stats = {}
    ff_data = None
    for prefix in [HIST_DIR, DATA_DIR]:
        torvik_path = prefix / f"torvik_{year}.json"
        if torvik_path.exists():
            with open(torvik_path) as f:
                data = json.load(f)
            _validate_pretournament(data, torvik_path)
            for t in data.get("teams", []):
                stats[t["team_id"]] = t
            # Merged torvik_{year}.json already validated above — the nested
            # "four_factors" sub-dict was written atomically alongside it by
            # the same migration/rescrape run, so it needs no separate check.
            if "four_factors" in data:
                ff_data = data["four_factors"]
            break

    if ff_data is None:
        for prefix in [HIST_DIR, DATA_DIR]:
            ff_path = prefix / f"torvik_four_factors_{year}.json"
            if ff_path.exists():
                with open(ff_path) as f:
                    ff_data = json.load(f)
                _validate_pretournament(ff_data, ff_path)
                break

    if isinstance(ff_data, dict) and "teams" not in ff_data:
        for tid, ff in ff_data.items():
            if tid in stats:
                for k, v in ff.items():
                    if not k.startswith("_"):
                        stats[tid].setdefault(k, v)
            else:
                stats[tid] = ff

    if not stats:
        raise FileNotFoundError(
            f"No Torvik team stats for {year}: expected "
            f"{HIST_DIR / f'torvik_{year}.json'} or {DATA_DIR / f'torvik_{year}.json'}. "
            f"For a new season, generate a pre-tournament snapshot with "
            f"scripts/rescrape_pretournament_torvik.py. Refusing to continue: "
            f"an empty payload builds every matchup from per-key defaults and "
            f"returns ~0.5 for every team without failing."
        )
    validate_stats_payload(stats, context=f"team stats for {year}")
    return stats


def _load_tournament_results(year: int) -> list:
    ctx_path = HIST_DIR / f"tournament_context_{year}.json"
    data = None
    if ctx_path.exists():
        with open(ctx_path) as f:
            ctx = json.load(f)
        data = ctx.get("results")
    if data is None:
        path = HIST_DIR / f"tournament_results_{year}.json"
        if not path.exists():
            return []
        with open(path) as f:
            data = json.load(f)
    return data.get("games", [])


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
        validate_stats_payload(stats, context=f"training year {year}")
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
    validate_stats_payload(stats, context="build_noseed_probabilities")
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
    validate_stats_payload(stats, context="build_noseed_round_probabilities")
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
