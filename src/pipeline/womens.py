"""Women's tournament prediction pipeline.

This is a lightweight parallel path for the women's side of the combined
Kaggle sample submission. It is intentionally simpler than the men's
pipeline, but it uses the same basic structure:

1. Load cached team data when available
2. Build matchup features
3. Train a simple logistic model from historical tournament results
4. Calibrate and shrink probabilities for tournament play
5. Fall back to seed-based estimates whenever live data is unavailable
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from ..data.features.womens_feature_engineering import (
    WomensFeatureEngineer,
    compute_seed_win_probability,
)
from ..data.scrapers.womens.herhoopstats import HerHoopStatsScraper, WomensTeamStats
from ..data.scrapers.womens.historical_results import WomensHistoricalResults
from ..data.scrapers.womens.ncaa_net import WomensNETScraper
from ..ml.calibration.post_processing import PostProcessingPipeline
from .probability_pipeline import apply_calibration, apply_final_clip, apply_tournament_shrinkage

logger = logging.getLogger(__name__)

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency guard
    LogisticRegression = None
    StandardScaler = None
    SKLEARN_AVAILABLE = False


class _TemperatureCalibrator:
    """Small self-contained temperature scaler for women's probabilities."""

    def __init__(self) -> None:
        self.temperature = 1.0
        self.fitted = False

    def fit(self, probabilities: np.ndarray, outcomes: np.ndarray) -> None:
        probs = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1.0 - 1e-6)
        y = np.asarray(outcomes, dtype=float)
        logits = np.log(probs / (1.0 - probs))

        best_t = 1.0
        best_brier = float("inf")
        for t in np.linspace(0.5, 2.5, 81):
            calibrated = 1.0 / (1.0 + np.exp(-logits / t))
            brier = float(np.mean((calibrated - y) ** 2))
            if brier < best_brier:
                best_brier = brier
                best_t = float(t)

        self.temperature = best_t
        self.fitted = True

    def calibrate(self, probabilities: np.ndarray) -> np.ndarray:
        probs = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1.0 - 1e-6)
        logits = np.log(probs / (1.0 - probs))
        return 1.0 / (1.0 + np.exp(-logits / max(self.temperature, 0.05)))


@dataclass
class WomensPipelineConfig:
    """Configuration for the women's tournament pipeline."""

    year: int = 2026
    cache_dir: str = "data/raw"
    calibration_method: str = "temperature"
    probability_profile: str = "production"
    clip_lo: float = 0.005
    clip_hi: float = 0.995
    seed_only_mode: bool = False
    enable_tournament_adaptation: bool = True
    tournament_shrinkage: float = 0.02
    enable_feature_scaling: bool = True
    model_weight: float = 0.40
    seed_weight: float = 0.60
    flb_threshold: float = 0.85
    flb_regression: float = 0.20

    def __post_init__(self) -> None:
        if self.probability_profile not in ("production", "experimental"):
            raise ValueError(
                f"Invalid probability_profile '{self.probability_profile}': must be 'production' or 'experimental'"
            )


class WomensPipeline:
    """Women-specific tournament prediction pipeline with robust fallbacks."""

    def __init__(self, config: Optional[WomensPipelineConfig] = None):
        self.config = config or WomensPipelineConfig()
        self.feature_engineer = WomensFeatureEngineer()
        self.team_stats: Dict[str, WomensTeamStats] = {}
        self.model = None
        self.scaler = None
        self.calibrator = None
        self.post_processor = PostProcessingPipeline(
            flb_threshold=self.config.flb_threshold,
            flb_regression=self.config.flb_regression,
            clip_lo=self.config.clip_lo,
            clip_hi=self.config.clip_hi,
        )
        self._trained = False

    def run(self) -> Dict[str, object]:
        """Load data, build features, and prepare the prediction path."""
        report: Dict[str, object] = {"status": "initialized"}

        scraper = HerHoopStatsScraper(cache_dir=self.config.cache_dir)
        self.team_stats = scraper.load_cached(self.config.year)
        report["teams_loaded"] = len(self.team_stats)

        net_scraper = WomensNETScraper(cache_dir=self.config.cache_dir)
        net_rankings = net_scraper.load_cached(self.config.year)
        report["net_rankings"] = len(net_rankings)

        self.feature_engineer.build_features(self.team_stats, net_rankings)
        report["features_built"] = len(self.feature_engineer.team_features)

        if not self.config.seed_only_mode and len(self.team_stats) >= 30:
            self._train_model()
        report["model_trained"] = self.model is not None

        self._fit_calibration()
        report["calibration_fitted"] = self.calibrator is not None

        self._trained = True
        report["status"] = "ready"
        return report

    def predict_probability(self, team1_id: str, team2_id: str) -> float:
        """Predict probability that team1 beats team2."""
        prob = self._raw_probability(team1_id, team2_id)
        prob = apply_calibration(prob, self.calibrator, self.config.clip_lo, self.config.clip_hi)
        if self.config.enable_tournament_adaptation:
            prob = apply_tournament_shrinkage(prob, self.config.tournament_shrinkage)

        if self.config.probability_profile == "experimental":
            seed1, seed2 = self._lookup_seeds(team1_id, team2_id)
            prob = self.post_processor.process(prob, seed1, seed2)

        return apply_final_clip(prob, self.config.clip_lo, self.config.clip_hi)

    def set_team_seeds(self, seed_map: Dict[str, int]) -> None:
        """Backfill teams from seed information when no cached stats exist."""
        if not seed_map:
            return

        scraper = HerHoopStatsScraper(cache_dir=self.config.cache_dir)
        generated = scraper.generate_seed_based_estimates(seed_map)
        for team_id, stats in generated.items():
            if team_id not in self.team_stats:
                self.team_stats[team_id] = stats

        net_scraper = WomensNETScraper(cache_dir=self.config.cache_dir)
        net_rankings = net_scraper.estimate_from_seeds(seed_map)
        self.feature_engineer.build_features(self.team_stats, net_rankings)

    def _lookup_seeds(self, team1_id: str, team2_id: str) -> tuple[int, int]:
        f1 = self.feature_engineer.team_features.get(team1_id)
        f2 = self.feature_engineer.team_features.get(team2_id)
        seed1 = f1.seed if f1 is not None else self.team_stats.get(team1_id, WomensTeamStats(team_name=team1_id)).seed
        seed2 = f2.seed if f2 is not None else self.team_stats.get(team2_id, WomensTeamStats(team_name=team2_id)).seed
        return seed1, seed2

    def _raw_probability(self, team1_id: str, team2_id: str) -> float:
        """Blend matchup-model and seed prior into a raw pre-calibration probability."""
        seed1, seed2 = self._lookup_seeds(team1_id, team2_id)
        seed_prob = float(compute_seed_win_probability(seed1, seed2)) if seed1 > 0 and seed2 > 0 else 0.5

        if self.model is None:
            return seed_prob

        features_fwd = self.feature_engineer.get_matchup_features(team1_id, team2_id)
        features_rev = self.feature_engineer.get_matchup_features(team2_id, team1_id)
        if features_fwd is None or features_rev is None:
            return seed_prob

        model_fwd = self._predict_with_model(features_fwd)
        model_rev = self._predict_with_model(features_rev)
        model_prob = (model_fwd + (1.0 - model_rev)) / 2.0

        total = self.config.model_weight + self.config.seed_weight
        return (self.config.model_weight * model_prob + self.config.seed_weight * seed_prob) / total

    def _predict_with_model(self, features: np.ndarray) -> float:
        if self.model is None:
            return 0.5
        x = features.reshape(1, -1)
        if self.scaler is not None:
            x = self.scaler.transform(x)
        pred = float(self.model.predict_proba(x)[0][1])
        return float(np.clip(pred, 0.01, 0.99))

    def _train_model(self) -> None:
        """Train a small logistic model from historical women's tournament results."""
        if not SKLEARN_AVAILABLE:
            logger.info("Women's pipeline: sklearn unavailable, staying seed-only")
            return

        history = WomensHistoricalResults(cache_dir=self.config.cache_dir)
        games = history.load_cached()
        if len(games) < 50:
            logger.info("Women's pipeline: insufficient historical games (%d), staying seed-only", len(games))
            return

        scraper = HerHoopStatsScraper(cache_dir=self.config.cache_dir)
        X_rows = []
        y_rows = []

        for game in games:
            if game.team1_seed <= 0 or game.team2_seed <= 0:
                continue

            seed_map = {
                f"hist_t1_{game.team1_seed}": game.team1_seed,
                f"hist_t2_{game.team2_seed}": game.team2_seed,
            }
            estimated_stats = scraper.generate_seed_based_estimates(seed_map)
            temp_engineer = WomensFeatureEngineer()
            temp_engineer.build_features(estimated_stats)

            fwd = temp_engineer.get_matchup_features(f"hist_t1_{game.team1_seed}", f"hist_t2_{game.team2_seed}")
            rev = temp_engineer.get_matchup_features(f"hist_t2_{game.team2_seed}", f"hist_t1_{game.team1_seed}")
            if fwd is None or rev is None:
                continue

            y = 1.0 if game.team1_won else 0.0
            X_rows.append(fwd)
            y_rows.append(y)
            X_rows.append(rev)
            y_rows.append(1.0 - y)

        if len(X_rows) < 50:
            logger.info("Women's pipeline: insufficient feature rows (%d), staying seed-only", len(X_rows))
            return

        X = np.asarray(X_rows, dtype=float)
        y = np.asarray(y_rows, dtype=float)

        if self.config.enable_feature_scaling:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)

        model = LogisticRegression(C=1.0, max_iter=500, solver="lbfgs")
        model.fit(X, y)
        self.model = model

    def _fit_calibration(self) -> None:
        """Fit a small temperature calibrator from historical tournament seed data."""
        if self.config.calibration_method == "none":
            return

        history = WomensHistoricalResults(cache_dir=self.config.cache_dir)
        games = history.load_cached()
        if len(games) < 30:
            return

        probabilities, outcomes = history.get_calibration_data()
        calibrator = _TemperatureCalibrator()
        calibrator.fit(np.asarray(probabilities, dtype=float), np.asarray(outcomes, dtype=float))
        self.calibrator = calibrator
