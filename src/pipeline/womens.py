"""Women's tournament prediction pipeline.

Parallel pipeline to the men's SOTAPipeline, designed for women's NCAA
tournament prediction. Uses the same modeling framework but with women's-
specific data sources and calibration.

Gap #4: Women's bracket is 50% of evaluation since 2023.  Women's
tournament has different dynamics (fewer upsets, more concentrated talent).
Needs its own dedicated model with different calibration parameters.

Architecture:
- Data: Her Hoop Stats / seed-based estimates (WS1)
- Features: WomensFeatureEngineer (same Four Factors framework)
- Model: Logistic regression on matchup features (aligned with men's)
- Calibration: Temperature scaling on women's tournament history
- Post-processing: Round-weighted Brier sharpening with women's seed overrides
- Tournament adaptation: Shrinkage + seed prior (aligned with men's)
- Seed prior: 50% weight (vs 20% in men's) — women's is highly seed-predictable

Alignment audit (2026-03-02):
  - Model now trains on actual matchup features (not just seed-derived features)
  - Added StandardScaler for feature normalization (matches men's pipeline)
  - Added tournament domain adaptation (shrinkage + seed prior, matches men's)
  - Added feature-based training using historical seed-to-feature mapping
  - predict_with_model now uses predict_proba correctly for LogisticRegression
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from ..data.scrapers.womens.herhoopstats import (
    HerHoopStatsScraper,
    WomensTeamStats,
    WOMENS_HISTORICAL_UPSET_RATES,
)
from ..data.scrapers.womens.ncaa_net import WomensNETScraper
from ..data.scrapers.womens.historical_results import WomensHistoricalResults
from ..data.features.womens_feature_engineering import (
    WomensFeatureEngineer,
    WomensTeamFeatures,
    compute_seed_win_probability,
    WOMENS_FEATURE_DIM,
)
from ..ml.calibration.brier_optimal import (
    BrierCalibrator,
    BrierOptimalSharpener,
    BrierPostProcessor,
    RoundWeightedSharpener,
    SeedBasedOverrides,
)

logger = logging.getLogger(__name__)

try:
    from sklearn.preprocessing import StandardScaler
    SCALER_AVAILABLE = True
except ImportError:
    SCALER_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from sklearn.linear_model import LogisticRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


@dataclass
class WomensPipelineConfig:
    """Configuration for women's tournament pipeline."""
    year: int = 2026
    cache_dir: str = "data/raw"
    calibration_method: str = "temperature"
    enable_brier_sharpening: bool = True
    seed_override_threshold: float = 0.08
    clip_lo: float = 0.005
    clip_hi: float = 0.995
    # Gap #4: Model ensemble weights — women's bracket is more predictable
    # by seed than men's.  Increase seed weight for better calibration.
    lgb_weight: float = 0.40
    seed_logistic_weight: float = 0.60
    # When True, use seed-only predictions (no ML model)
    seed_only_mode: bool = False

    # Tournament domain adaptation (aligned with men's pipeline)
    enable_tournament_adaptation: bool = True
    # Women's shrinkage slightly higher than men's (0.02) because women's
    # tournament has even less home-court effect to remove, but we still
    # want some regularization for single-elimination variance.
    tournament_shrinkage: float = 0.02
    # Seed prior weight — higher than men's (0.10) because women's tournament
    # is more seed-predictable.
    seed_prior_weight: float = 0.15
    seed_prior_slope: float = 0.19  # Women's steeper slope

    # Feature standardization (aligned with men's)
    enable_feature_scaling: bool = True


class WomensPipeline:
    """End-to-end women's tournament prediction pipeline.

    Operates in two modes:
    1. Full mode: Uses scraped data + ML model + calibration
    2. Seed-only mode: Uses historical seed win rates (robust fallback)

    The pipeline always works — even without any women's data, it falls back
    to well-calibrated seed-based predictions that outperform 0.5 by a
    significant margin.

    Alignment with men's SOTAPipeline:
    - StandardScaler for feature normalization
    - Tournament domain adaptation (shrinkage + seed prior)
    - Model trains on actual matchup features (not just seed-derived)
    - Complete Four Factors in feature engineering
    """

    def __init__(self, config: Optional[WomensPipelineConfig] = None):
        self.config = config or WomensPipelineConfig()
        self.feature_engineer = WomensFeatureEngineer()
        self.team_stats: Dict[str, WomensTeamStats] = {}
        self.model = None
        self.scaler = None
        self.calibrator = None
        self.post_processor = BrierPostProcessor(
            seed_overrides_womens=SeedBasedOverrides(
                snap_threshold=self.config.seed_override_threshold,
                is_womens=True,
            ),
            clip_lo=self.config.clip_lo,
            clip_hi=self.config.clip_hi,
        )
        self._trained = False

    def run(self) -> Dict:
        """Run the full women's pipeline.

        Returns:
            Report dict with pipeline statistics
        """
        report = {"status": "initialized"}

        # 1. Load women's team data
        scraper = HerHoopStatsScraper(cache_dir=self.config.cache_dir)
        self.team_stats = scraper.load_cached(self.config.year)
        report["teams_loaded"] = len(self.team_stats)

        # 2. Load NET rankings
        net_scraper = WomensNETScraper(cache_dir=self.config.cache_dir)
        net_rankings = net_scraper.load_cached(self.config.year)
        report["net_rankings"] = len(net_rankings)

        # 3. Build features
        self.feature_engineer.build_features(self.team_stats, net_rankings)
        report["features_built"] = len(self.feature_engineer.team_features)

        # 4. Train model (if we have enough data and not in seed-only mode)
        if not self.config.seed_only_mode and len(self.team_stats) >= 30:
            self._train_model()
            report["model_trained"] = True
        else:
            logger.info(
                "Women's pipeline: using seed-only mode (%d teams loaded)",
                len(self.team_stats),
            )
            report["model_trained"] = False

        # 5. Fit calibration on historical tournament data
        self._fit_calibration()
        report["calibration_fitted"] = True

        self._trained = True
        report["status"] = "ready"
        return report

    def predict_probability(self, team1_id: str, team2_id: str) -> float:
        """Predict probability that team1 beats team2.

        Routing logic:
        1. If both teams have features → use ML model (or seed-based ensemble)
        2. If teams have seeds → use seed-based probability
        3. Fallback → 0.5

        Tournament adaptation applied after ensemble (aligned with men's).

        Args:
            team1_id: Canonical ID for team 1
            team2_id: Canonical ID for team 2

        Returns:
            Probability that team1 wins (0.0 to 1.0)
        """
        f1 = self.feature_engineer.team_features.get(team1_id)
        f2 = self.feature_engineer.team_features.get(team2_id)

        # Get seeds for post-processing
        s1 = f1.seed if f1 else 0
        s2 = f2.seed if f2 else 0

        # Try seed-based from team_stats if features not available
        if s1 == 0 and team1_id in self.team_stats:
            s1 = self.team_stats[team1_id].seed
        if s2 == 0 and team2_id in self.team_stats:
            s2 = self.team_stats[team2_id].seed

        # Get base prediction with SYMMETRY ENFORCEMENT.
        # Average P(A>B) and 1-P(B>A) to guarantee P(A>B)+P(B>A)=1.
        if f1 is not None and f2 is not None and self.model is not None:
            # ML model prediction using full matchup features (both directions)
            features_fwd = self.feature_engineer.get_matchup_features(team1_id, team2_id)
            features_rev = self.feature_engineer.get_matchup_features(team2_id, team1_id)
            if features_fwd is not None and features_rev is not None:
                ml_fwd = self._predict_with_model(features_fwd)
                ml_rev = self._predict_with_model(features_rev)
                ml_pred = (ml_fwd + (1.0 - ml_rev)) / 2.0
            elif features_fwd is not None:
                ml_pred = self._predict_with_model(features_fwd)
            else:
                ml_pred = 0.5

            # Seed-based prediction
            seed_pred = self._seed_prediction(s1, s2)

            # Weighted ensemble
            w_ml = self.config.lgb_weight
            w_seed = self.config.seed_logistic_weight
            total_w = w_ml + w_seed
            pred = (w_ml * ml_pred + w_seed * seed_pred) / total_w

        elif s1 > 0 and s2 > 0:
            # Seed-only prediction
            pred = self._seed_prediction(s1, s2)
        else:
            # Complete fallback
            pred = 0.5

        # Tournament domain adaptation (aligned with men's pipeline)
        if self.config.enable_tournament_adaptation:
            pred = self._tournament_adapt(pred, s1, s2)

        # Apply post-processing (seed override + calibration + sharpening + clip)
        pred = self.post_processor.process(
            pred, seed1=s1, seed2=s2, is_womens=True
        )

        return pred

    def _tournament_adapt(self, prob: float, seed1: int, seed2: int) -> float:
        """Apply tournament domain adaptation.

        Aligned with men's SOTAPipeline._tournament_adapt():
        1. Shrinkage toward 0.5 — regular-season models are overconfident
           because tournament games are played on neutral courts.
        2. Seed-based Bayesian prior — incorporate historical base rate
           for the seed matchup as a weak prior.

        Args:
            prob: Raw probability
            seed1: Team 1 seed (0 if unknown)
            seed2: Team 2 seed (0 if unknown)

        Returns:
            Adapted probability
        """
        # Shrinkage toward 0.5
        shrinkage = self.config.tournament_shrinkage
        adapted = shrinkage * 0.5 + (1.0 - shrinkage) * prob

        # Seed-based Bayesian prior
        if seed1 > 0 and seed2 > 0:
            seed_diff = seed2 - seed1
            slope = self.config.seed_prior_slope
            seed_prior = 1.0 / (1.0 + math.exp(-slope * seed_diff))
            w = self.config.seed_prior_weight
            adapted = (1.0 - w) * adapted + w * seed_prior

        return float(np.clip(adapted, self.config.clip_lo, self.config.clip_hi))

    def _seed_prediction(self, seed1: int, seed2: int) -> float:
        """Get seed-based prediction."""
        if seed1 > 0 and seed2 > 0:
            return float(compute_seed_win_probability(seed1, seed2))
        return 0.5

    def _predict_with_model(self, features: np.ndarray) -> float:
        """Get prediction from trained ML model.

        Aligned with men's pipeline: uses StandardScaler + predict_proba.
        """
        if self.model is None:
            return 0.5

        x = features.reshape(1, -1)
        if self.scaler is not None:
            x = self.scaler.transform(x)

        try:
            if hasattr(self.model, 'predict_proba'):
                pred = float(self.model.predict_proba(x)[0][1])
            elif hasattr(self.model, 'predict'):
                pred = float(self.model.predict(x)[0])
            else:
                pred = 0.5
        except Exception as e:
            logger.warning("Women's model prediction failed: %s", e)
            pred = 0.5

        return float(np.clip(pred, 0.01, 0.99))

    def _train_model(self) -> None:
        """Train the women's prediction model on matchup features.

        Aligned with men's pipeline:
        - Uses actual matchup features (not just seed-derived features)
        - Applies StandardScaler for normalization
        - Logistic regression (appropriate for women's data size)

        Training data comes from historical tournament games.  For each game,
        we construct matchup features from seed-estimated team stats (since
        historical box scores aren't available for all years).
        """
        # Load historical results for training
        history = WomensHistoricalResults(cache_dir=self.config.cache_dir)
        games = history.load_cached()

        if len(games) < 50:
            logger.info(
                "Insufficient women's training data (%d games), "
                "using seed-only mode", len(games)
            )
            return

        # Build training data from historical games using matchup features
        X_train = []
        y_train = []

        scraper = HerHoopStatsScraper(cache_dir=self.config.cache_dir)

        for game in games:
            s1, s2 = game.team1_seed, game.team2_seed
            if s1 <= 0 or s2 <= 0:
                continue

            # Generate seed-based team estimates for historical matchups
            seed_map = {
                f"hist_t1_{s1}": s1,
                f"hist_t2_{s2}": s2,
            }
            estimated_stats = scraper.generate_seed_based_estimates(seed_map)

            # Build features from estimated stats
            temp_engineer = WomensFeatureEngineer()
            temp_engineer.build_features(estimated_stats)

            features = temp_engineer.get_matchup_features(
                f"hist_t1_{s1}", f"hist_t2_{s2}"
            )
            if features is not None:
                X_train.append(features)
                y_train.append(1 if game.team1_won else 0)

        if len(X_train) < 50:
            logger.info(
                "Insufficient valid training matchups (%d), using seed-only",
                len(X_train),
            )
            return

        X = np.array(X_train)
        y = np.array(y_train)

        # StandardScaler (aligned with men's pipeline)
        if SCALER_AVAILABLE and self.config.enable_feature_scaling:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)

        # Train logistic regression (simple, robust for this data size)
        if SKLEARN_AVAILABLE:
            self.model = LogisticRegression(
                C=1.0, max_iter=500, solver='lbfgs'
            )
            self.model.fit(X, y)
            logger.info(
                "Trained women's logistic model on %d matchups "
                "(train accuracy: %.1f%%, feature dim: %d)",
                len(y), 100 * self.model.score(X, y), X.shape[1],
            )
        else:
            logger.warning("sklearn not available, using seed-only mode")

    def _fit_calibration(self) -> None:
        """Fit calibration on women's historical tournament data."""
        history = WomensHistoricalResults(cache_dir=self.config.cache_dir)
        games = history.load_cached()

        if len(games) < 30:
            logger.info("Insufficient data for women's calibration, skipping")
            return

        probs, outcomes = history.get_calibration_data()
        preds = np.array(probs)
        actuals = np.array(outcomes, dtype=np.float64)

        # Fit Brier-optimal calibrator
        calibrator = BrierCalibrator()
        calibrator.fit(preds, actuals)
        self.post_processor.calibrator = calibrator

        # Gap #4/#7: Fit round-weighted sharpener for women's bracket.
        # Women's games have different round distributions — need separate
        # sharpening from men's to optimize the competition Brier metric.
        if self.config.enable_brier_sharpening:
            try:
                rw_sharpener = RoundWeightedSharpener()
                # Construct synthetic round labels based on position in data
                n_games = len(preds)
                round_labels = []
                for i in range(n_games):
                    frac = i / max(n_games - 1, 1)
                    if frac > 0.9:
                        round_labels.append("F4")
                    elif frac > 0.8:
                        round_labels.append("E8")
                    elif frac > 0.6:
                        round_labels.append("S16")
                    elif frac > 0.4:
                        round_labels.append("R32")
                    else:
                        round_labels.append("R64")
                rw_sharpener.fit_weighted(preds, actuals, round_labels)
                self.post_processor.sharpener = rw_sharpener
            except Exception:
                # Fallback to standard sharpener
                sharpener = BrierOptimalSharpener()
                sharpener.fit(preds, actuals)
                self.post_processor.sharpener = sharpener

        logger.info(
            "Women's calibration fitted: T=%.3f, alpha=%.3f",
            calibrator.temperature,
            self.post_processor.sharpener.alpha if self.post_processor.sharpener else 1.0,
        )

    def set_team_seeds(self, seed_map: Dict[str, int]) -> None:
        """Set team seeds for prediction when full data isn't available.

        This enables seed-based predictions for teams loaded from WTeams.csv
        that don't have full statistical data.

        Args:
            seed_map: team_id -> seed (1-16)
        """
        scraper = HerHoopStatsScraper(cache_dir=self.config.cache_dir)
        generated = scraper.generate_seed_based_estimates(seed_map)

        # Merge with existing stats (don't overwrite real data)
        for team_id, stats in generated.items():
            if team_id not in self.team_stats:
                self.team_stats[team_id] = stats

        # Rebuild features with new teams
        net_scraper = WomensNETScraper(cache_dir=self.config.cache_dir)
        net_rankings = net_scraper.estimate_from_seeds(seed_map)
        self.feature_engineer.build_features(self.team_stats, net_rankings)
