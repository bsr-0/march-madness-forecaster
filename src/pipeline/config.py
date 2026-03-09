"""Pipeline configuration, data classes, and constants extracted from sota.py.

This module contains:
- SOTAPipelineConfig: All pipeline configuration knobs
- EVModeReport: EV mode optimization report structure
- _TrainedBaselineModel: Ensemble model wrapper
- Constants: TOURNAMENT_START_DATES, FIXED_FEATURE_SET, etc.
- Utility functions: compute_year_data_quality, _infer_tournament_round_weight

Extracted as part of Agent Directive V7 S12 (codebase decomposition).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..ml.ensemble.cfa import LightGBMRanker, XGBoostRanker

logger = logging.getLogger(__name__)

# Lazy import for optional deps — avoid circular import at module level
try:
    from ..ml.ensemble.spread_model import SpreadRegressor
except ImportError:
    SpreadRegressor = None  # type: ignore[misc,assignment]

try:
    from sklearn.linear_model import LogisticRegression
except ImportError:
    LogisticRegression = None  # type: ignore[misc,assignment]


# ---------------------------------------------------------------------------
# Hard tournament start dates by year.  Games on or after these dates are
# NCAA tournament games and MUST be excluded from regular-season training
# to prevent result leakage.
# ---------------------------------------------------------------------------
TOURNAMENT_START_DATES: Dict[int, date] = {
    2017: date(2017, 3, 14),
    2018: date(2018, 3, 13),
    2019: date(2019, 3, 19),
    2021: date(2021, 3, 18),
    2022: date(2022, 3, 15),
    2023: date(2023, 3, 14),
    2024: date(2024, 3, 19),
    2025: date(2025, 3, 18),
    2026: date(2026, 3, 17),
}


# C2: Fixed domain-knowledge feature set with published citations.
FIXED_FEATURE_SET = [
    "diff_adj_off_eff",
    "diff_adj_def_eff",
    "diff_adj_tempo",
    "diff_efg_pct",
    "diff_to_rate",
    "diff_orb_rate",
    "diff_ft_rate",
    "diff_opp_efg_pct",
    "diff_opp_to_rate",
    "diff_sos_adj_em",
    "diff_elo_rating",
    "diff_free_throw_pct",
    "diff_win_pct",
    "diff_three_pt_pct",
    "diff_three_pt_variance",
    "diff_avg_experience",
    "diff_roster_continuity",
    "abs_adj_off_eff",
    "abs_adj_def_eff",
    "abs_sos_adj_em",
    "seed_interaction",
    "seed_diff",
    "travel_advantage",
    "diff_external_rating_composite",
    "diff_external_rating_spread",
    "diff_momentum",
    "diff_tournament_resume",
    "diff_home_court_dependence",
]


SIMPLE_FEATURE_SET = [
    "diff_adj_off_eff",
    "diff_adj_def_eff",
    "diff_sos_adj_em",
    "diff_external_rating_composite",
    "diff_elo_rating",
    "diff_win_pct",
    "diff_free_throw_pct",
    "seed_interaction",
    "seed_diff",
    "diff_momentum",
]


KAGGLE_ROUND_WEIGHTS = {
    "R64": 1.0,
    "R32": 2.0,
    "S16": 4.0,
    "E8": 8.0,
    "F4": 16.0,
    "NCG": 32.0,
}


DATA_QUALITY_ERA_WEIGHTS = {
    2005: 0.0, 2006: 0.0, 2007: 0.10, 2008: 0.20, 2009: 0.30,
    2010: 0.55, 2011: 0.65, 2012: 0.75, 2013: 0.80, 2014: 0.85,
}


MIN_SEASON_FEATURE_COMPLETENESS = 0.20


def compute_year_data_quality(
    X: np.ndarray, year: int, feature_names: Optional[List[str]] = None,
) -> Dict:
    """Compute per-year data quality metrics for adaptive weighting."""
    n_samples, n_features = X.shape
    completeness = float(np.mean(np.abs(X) > 1e-8))
    col_vars = np.var(X, axis=0)
    n_active_features = int(np.sum(col_vars > 1e-8))
    feature_activity = n_active_features / max(n_features, 1)
    zero_cols = int(np.sum(np.all(np.abs(X) < 1e-8, axis=0)))
    zero_col_names: List[str] = []
    if feature_names and zero_cols > 0:
        for i, name in enumerate(feature_names):
            if i < n_features and np.all(np.abs(X[:, i]) < 1e-8):
                zero_col_names.append(name)
    n_nan = int(np.isnan(X).sum())
    n_inf = int(np.isinf(X).sum())
    bad_rate = (n_nan + n_inf) / max(X.size, 1)
    era_weight = DATA_QUALITY_ERA_WEIGHTS.get(year, 1.0)
    adaptive_weight = (
        0.3 * completeness
        + 0.3 * feature_activity
        + 0.2 * min(n_samples / 500.0, 1.0)
        + 0.2 * (1.0 - bad_rate)
    )
    combined_weight = min(era_weight, adaptive_weight) if era_weight < 0.5 else adaptive_weight
    return {
        "year": year,
        "n_samples": n_samples,
        "n_features": n_features,
        "completeness": round(completeness, 3),
        "feature_activity": round(feature_activity, 3),
        "n_active_features": n_active_features,
        "zero_columns": zero_cols,
        "zero_column_names": zero_col_names[:10],
        "nan_count": n_nan,
        "inf_count": n_inf,
        "bad_rate": round(bad_rate, 5),
        "era_weight": era_weight,
        "adaptive_weight": round(adaptive_weight, 3),
        "combined_weight": round(combined_weight, 3),
    }


def _infer_tournament_round_weight(game_date: str, year: int) -> float:
    """Infer tournament round weight from game date."""
    try:
        gd = datetime.strptime(game_date[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return 1.0
    day_of_march = (gd - date(year, 3, 1)).days
    if day_of_march >= 31:
        return 32.0 if day_of_march >= 33 else 16.0
    elif day_of_march >= 24:
        return 8.0
    elif day_of_march >= 22:
        return 4.0
    elif day_of_march >= 17:
        return 2.0
    else:
        return 1.0


class DataRequirementError(ValueError):
    """Raised when required real-world data is unavailable."""


@dataclass
class SOTAPipelineConfig:
    """Pipeline configuration knobs."""

    year: int = 2026
    num_simulations: int = 50000
    pool_size: int = 100
    random_seed: int = 2026
    dev_years: Optional[List[int]] = field(default_factory=lambda: list(range(2016, 2025)))
    holdout_years: Optional[List[int]] = field(default_factory=lambda: [2025])
    require_freeze_file: bool = False
    freeze_file: Optional[str] = None

    teams_json: Optional[str] = None
    torvik_json: Optional[str] = None
    historical_games_json: Optional[str] = None
    sports_reference_json: Optional[str] = None
    public_picks_json: Optional[str] = None
    scoring_rules_json: Optional[str] = None
    roster_json: Optional[str] = None
    transfer_portal_json: Optional[str] = None

    preseason_ap_json: Optional[str] = None
    coach_tournament_json: Optional[str] = None
    conf_champions_json: Optional[str] = None
    coach_data_cutoff_year: Optional[int] = None

    calibration_method: str = "temperature"
    scrape_live: bool = False
    data_cache_dir: str = "data/raw"
    injury_noise_samples: int = 10000
    enforce_feed_freshness: bool = True
    max_feed_age_hours: int = 168
    min_public_sources: int = 2
    min_rapm_players_per_team: int = 5
    min_calibration_samples_hard: int = 50

    enable_hyperparameter_tuning: bool = True
    optuna_n_trials: int = 15
    optuna_timeout: int = 300
    temporal_cv_splits: int = 5
    optimize_ensemble_weights: bool = True

    scoring_metric: str = "brier"
    enable_feature_scaling: bool = True
    enable_stacking: bool = False
    margin_first_training: bool = False

    enable_loyo_cv: bool = True
    loyo_years: Optional[List[int]] = None
    multi_year_games_dir: Optional[str] = "auto"

    enable_multi_year_training: bool = True
    training_years: Optional[List[int]] = None
    training_year_decay: float = 0.85
    training_year_min_weight: float = 0.15

    game_level_min_games_per_team: int = 5

    enable_spread_model: bool = True
    spread_sigma: float = 11.0
    spread_weight: float = 0.50

    enable_bayesian_bt: bool = True
    bayesian_bt_weight: float = 0.10
    bayesian_bt_prior_sigma: float = 200.0
    bayesian_bt_home_advantage: float = 0.0

    enable_gnn: bool = False
    enable_transformer: bool = False
    enable_player_embeddings: bool = False

    injury_report_json: Optional[str] = None
    enable_injury_severity_model: bool = True
    enable_positional_depth: bool = True

    enable_tournament_sigma: bool = True
    tournament_sigma_weight: float = 0.30
    tournament_expert_weight: float = 0.30

    enable_massey_standalone: bool = True
    massey_standalone_weight: float = 0.15
    massey_ordinals_csv: Optional[str] = None
    massey_auto_kaggle_download: bool = True

    enable_margin_first_ensemble: bool = True

    enable_brier_sharpening: bool = True
    brier_sharpening_alpha_bounds: Tuple[float, float] = (0.5, 2.0)
    enable_seed_overrides: bool = True
    seed_override_threshold: float = 0.08

    enable_goto_conversion: bool = True
    goto_conversion_margin_init: float = 0.05
    goto_conversion_margin_bounds: Tuple[float, float] = (0.0, 0.20)

    enable_womens_pipeline: bool = True
    womens_cache_dir: Optional[str] = None
    womens_seed_only_mode: bool = False
    womens_teams_csv: Optional[str] = None
    womens_model_complexity: str = "simple"
    womens_seed_prior_weight: float = 0.40
    womens_massey_blend_weight: float = 0.25

    enable_bracket_portfolio: bool = True
    portfolio_n_brackets: int = 1000
    portfolio_n_simulations: int = 50000

    enable_dual_submission: bool = True
    dual_strategy: str = "champion_boost"
    dual_max_deviations: int = 5
    dual_deviation_strength: float = 0.15
    dual_n_champion_candidates: int = 5

    mode: str = "calibration"
    ev_pool_size: int = 100
    ev_scoring_system: str = "standard"
    ev_target_percentile: float = 0.05
    ev_contrarian_strength: float = 1.0
    ev_enable_search: bool = False
    ev_enable_archetypes: bool = False
    ev_pool_type: str = "espn_national"
    ev_payout_structure: str = "tiered"
    ev_archetype_overrides: Optional[Dict[str, float]] = None
    ev_auto_refresh: bool = False
    kaggle_effective_pool_size: int = 3000

    betting_odds_json: Optional[str] = None
    enable_market_blend: bool = True
    market_blend_weight: float = 0.20

    # Compute budget management (S20)
    compute_budget_seconds: float = 3600.0
    enable_budget_degradation: bool = True
    use_agent_orchestration: bool = False

    def __post_init__(self):
        if self.mode not in ("calibration", "ev"):
            raise ValueError(f"Invalid mode '{self.mode}': must be 'calibration' or 'ev'")
        if self.mode == "ev":
            if self.ev_pool_size < 1:
                raise ValueError(f"ev_pool_size must be >= 1, got {self.ev_pool_size}")
            if not (0.0 < self.ev_target_percentile <= 0.5):
                raise ValueError(
                    f"ev_target_percentile must be in (0, 0.5], got {self.ev_target_percentile}"
                )
            if self.ev_scoring_system not in ("standard", "flat", "upset_bonus"):
                raise ValueError(
                    f"Invalid ev_scoring_system '{self.ev_scoring_system}': "
                    "must be 'standard', 'flat', or 'upset_bonus'"
                )
            valid_payouts = {
                "winner_take_all", "top_3", "top_10pct", "top_25pct", "tiered",
            }
            if self.ev_payout_structure not in valid_payouts:
                raise ValueError(
                    f"Invalid ev_payout_structure '{self.ev_payout_structure}': "
                    f"must be one of {sorted(valid_payouts)}"
                )


@dataclass
class EVModeReport:
    """Report structure for EV Mode optimization results."""

    pool_size: int = 0
    scoring_system: str = "standard"
    target_percentile: float = 0.05
    recommended_strategy: str = ""
    leverage_picks: List[Dict] = field(default_factory=list)
    fade_picks: List[Dict] = field(default_factory=list)
    model_vs_public_divergence: Dict[str, float] = field(default_factory=dict)
    bracket_portfolio_summary: Dict = field(default_factory=dict)
    win_probabilities: Dict[str, float] = field(default_factory=dict)
    competition_simulation: Dict = field(default_factory=dict)
    pareto_brackets: List[Dict] = field(default_factory=list)
    pool_ev_analysis: Dict[str, float] = field(default_factory=dict)
    picks_staleness_warning: Optional[str] = None

    def to_dict(self) -> Dict:
        d = {
            "mode": "ev",
            "pool_size": self.pool_size,
            "scoring_system": self.scoring_system,
            "target_percentile": self.target_percentile,
            "recommended_strategy": self.recommended_strategy,
            "leverage_picks": self.leverage_picks,
            "fade_picks": self.fade_picks,
            "model_vs_public_divergence": self.model_vs_public_divergence,
            "bracket_portfolio_summary": self.bracket_portfolio_summary,
            "win_probabilities": self.win_probabilities,
            "competition_simulation": self.competition_simulation,
            "pareto_brackets": self.pareto_brackets,
            "pool_ev_analysis": self.pool_ev_analysis,
        }
        if self.picks_staleness_warning:
            d["picks_staleness_warning"] = self.picks_staleness_warning
        return d


class _TrainedBaselineModel:
    """Wrapper for LightGBM, XGBoost, stacking meta-learner, or logistic fallback."""

    def __init__(self):
        self.lgb_model: Optional[LightGBMRanker] = None
        self.xgb_model: Optional[XGBoostRanker] = None
        self.logit_model = None  # LogisticRegression
        self.scaler: Optional[object] = None
        self.feature_dim: int = 80
        self.fixed_feature_indices: Optional[List[int]] = None
        self.fixed_weight_models: List = []
        self.fixed_weights: Dict[str, float] = {}
        self.stacking_meta: Optional[object] = None
        self.stacking_meta_type: str = "logistic"
        self.stacking_models: List = []
        self.spread_model: Optional[object] = None

    def predict_proba(self, x: np.ndarray) -> float:
        x_scaled = self._scale(x)
        if self.fixed_weight_models:
            return self._fixed_weight_predict(x_scaled)
        if self.stacking_meta is not None:
            return self._stacking_predict(x_scaled)
        if self.lgb_model is not None:
            return float(self.lgb_model.predict(x_scaled.reshape(1, -1))[0])
        if self.xgb_model is not None:
            return float(self.xgb_model.predict(x_scaled.reshape(1, -1))[0])
        if self.logit_model is None:
            return 0.5
        return float(self.logit_model.predict_proba(x_scaled.reshape(1, -1))[0][1])

    def predict_proba_batch(self, X: np.ndarray) -> np.ndarray:
        """Batch prediction for efficiency."""
        X_scaled = self._scale_batch(X)
        if self.fixed_weight_models:
            return self._fixed_weight_predict_batch(X_scaled)
        if self.stacking_meta is not None:
            return self._stacking_predict_batch(X_scaled)
        if self.lgb_model is not None:
            return self.lgb_model.predict(X_scaled)
        if self.xgb_model is not None:
            return self.xgb_model.predict(X_scaled)
        if self.logit_model is not None:
            return self.logit_model.predict_proba(X_scaled)[:, 1]
        return np.full(len(X_scaled), 0.5)

    def _fixed_weight_predict(self, x: np.ndarray) -> float:
        x_2d = x.reshape(1, -1)
        total = 0.0
        for name, model in self.fixed_weight_models:
            w = self.fixed_weights.get(name, 0.33)
            if isinstance(model, LightGBMRanker):
                total += w * float(model.predict(x_2d)[0])
            elif isinstance(model, XGBoostRanker):
                total += w * float(model.predict(x_2d)[0])
            elif SpreadRegressor is not None and isinstance(model, SpreadRegressor):
                total += w * float(model.predict_probability(x_2d)[0])
            else:
                total += w * float(model.predict_proba(x_2d)[0][1])
        return total

    def _fixed_weight_predict_batch(self, X: np.ndarray) -> np.ndarray:
        result = np.zeros(len(X))
        for name, model in self.fixed_weight_models:
            w = self.fixed_weights.get(name, 0.33)
            if isinstance(model, LightGBMRanker):
                result += w * model.predict(X)
            elif isinstance(model, XGBoostRanker):
                result += w * model.predict(X)
            elif SpreadRegressor is not None and isinstance(model, SpreadRegressor):
                result += w * model.predict_probability(X)
            else:
                result += w * model.predict_proba(X)[:, 1]
        return result

    def _select_features(self, x: np.ndarray) -> np.ndarray:
        if self.fixed_feature_indices is not None:
            n_cols = x.shape[-1] if x.ndim >= 1 else 0
            expected = len(self.fixed_feature_indices)
            if n_cols == expected:
                return x
            if x.ndim == 1:
                return x[self.fixed_feature_indices]
            return x[:, self.fixed_feature_indices]
        return x

    def _scale(self, x: np.ndarray) -> np.ndarray:
        x = self._select_features(x)
        if self.scaler is not None:
            return self.scaler.transform(x.reshape(1, -1))[0]
        return x

    def _scale_batch(self, X: np.ndarray) -> np.ndarray:
        X = self._select_features(X)
        if self.scaler is not None:
            return self.scaler.transform(X)
        return X

    def _stacking_predict(self, x: np.ndarray) -> float:
        meta_features = self._get_meta_features(x.reshape(1, -1))
        if self.stacking_meta_type == "lightgbm":
            raw = float(self.stacking_meta.predict(meta_features)[0])
            return float(np.clip(raw, 0.01, 0.99))
        return float(self.stacking_meta.predict_proba(meta_features)[0][1])

    def _stacking_predict_batch(self, X: np.ndarray) -> np.ndarray:
        meta_features = self._get_meta_features(X)
        if self.stacking_meta_type == "lightgbm":
            raw = self.stacking_meta.predict(meta_features)
            return np.clip(raw, 0.01, 0.99)
        return self.stacking_meta.predict_proba(meta_features)[:, 1]

    def _get_meta_features(self, X: np.ndarray) -> np.ndarray:
        base_cols = []
        for name, model in self.stacking_models:
            if name == "lgb" and isinstance(model, LightGBMRanker):
                base_cols.append(model.predict(X))
            elif name == "xgb" and isinstance(model, XGBoostRanker):
                base_cols.append(model.predict(X))
            elif name == "logit" and hasattr(model, "predict_proba"):
                base_cols.append(model.predict_proba(X)[:, 1])
        if not base_cols:
            return X
        base_arr = np.column_stack(base_cols)
        enriched = [base_arr]
        k = base_arr.shape[1]
        for i in range(k):
            for j in range(i + 1, k):
                enriched.append((base_arr[:, i] * base_arr[:, j]).reshape(-1, 1))
        enriched.append(np.max(base_arr, axis=1).reshape(-1, 1))
        enriched.append(np.min(base_arr, axis=1).reshape(-1, 1))
        enriched.append(np.std(base_arr, axis=1).reshape(-1, 1))
        return np.hstack(enriched)
