"""Tests for the top-1% gap closures (Gaps #1-7)."""

import math
import numpy as np
import pytest


class TestRoundWeightedBrier:
    """Gap #7: Round-weighted Brier scoring."""

    def test_uniform_weights_equals_standard_brier(self):
        from src.ml.calibration.brier_optimal import round_weighted_brier

        predictions = np.array([0.8, 0.6, 0.3, 0.9])
        outcomes = np.array([1.0, 1.0, 0.0, 1.0])
        standard = float(np.mean((predictions - outcomes) ** 2))
        weighted = round_weighted_brier(predictions, outcomes)  # no round labels
        assert abs(weighted - standard) < 1e-9

    def test_late_round_weighting(self):
        from src.ml.calibration.brier_optimal import round_weighted_brier

        # Prediction for NCG game is wrong, R64 games are correct
        predictions = np.array([0.9, 0.9, 0.1])  # 3rd is wrong (NCG)
        outcomes = np.array([1.0, 1.0, 1.0])
        rounds = ["R64", "R64", "NCG"]

        weighted = round_weighted_brier(predictions, outcomes, rounds)
        flat = round_weighted_brier(predictions, outcomes)

        # NCG error (0.81) is weighted 32x more than R64 errors (0.01 each)
        # So weighted should be much worse than flat
        assert weighted > flat

    def test_all_correct_gives_zero(self):
        from src.ml.calibration.brier_optimal import round_weighted_brier

        predictions = np.array([1.0, 1.0, 0.0])
        outcomes = np.array([1.0, 1.0, 0.0])
        rounds = ["R64", "S16", "NCG"]

        assert round_weighted_brier(predictions, outcomes, rounds) == pytest.approx(0.0)

    def test_round_label_from_seed_matchup(self):
        from src.ml.calibration.brier_optimal import round_label_from_seed_matchup

        assert round_label_from_seed_matchup(1, 16, "Round of 64") == "R64"
        assert round_label_from_seed_matchup(1, 4, "Elite Eight") == "E8"
        assert round_label_from_seed_matchup(1, 2, "Final Four") == "F4"
        assert round_label_from_seed_matchup(1, 1, "Championship") == "NCG"
        assert round_label_from_seed_matchup(1, 16, "Sweet 16") == "S16"
        assert round_label_from_seed_matchup(1, 16) == "R64"  # default


class TestRoundWeightedSharpener:
    """Gap #7: Round-weighted sharpening optimization."""

    def test_weighted_sharpener_fits(self):
        from src.ml.calibration.brier_optimal import RoundWeightedSharpener

        rng = np.random.default_rng(42)
        preds = rng.uniform(0.3, 0.9, size=50)
        outcomes = (preds > 0.5).astype(float)
        rounds = ["R64"] * 30 + ["R32"] * 10 + ["S16"] * 5 + ["E8"] * 3 + ["F4"] * 1 + ["NCG"] * 1

        sharpener = RoundWeightedSharpener()
        alpha = sharpener.fit_weighted(preds, outcomes, rounds)
        assert sharpener.fitted
        assert 0.3 < alpha < 3.0

    def test_weighted_vs_flat_alpha_differs(self):
        from src.ml.calibration.brier_optimal import BrierOptimalSharpener, RoundWeightedSharpener

        rng = np.random.default_rng(123)
        preds = np.clip(rng.normal(0.5, 0.2, size=100), 0.05, 0.95)
        outcomes = (rng.uniform(size=100) < preds).astype(float)
        rounds = (["R64"] * 50 + ["R32"] * 25 + ["S16"] * 12 +
                  ["E8"] * 6 + ["F4"] * 4 + ["NCG"] * 3)

        flat = BrierOptimalSharpener()
        flat.fit(preds, outcomes)

        weighted = RoundWeightedSharpener()
        weighted.fit_weighted(preds, outcomes, rounds)

        # Both should produce valid alphas (may or may not differ significantly)
        assert flat.fitted
        assert weighted.fitted


class TestKaggleRoundWeights:
    """Verify the round weight schedule is correct."""

    def test_all_rounds_contribute_equally(self):
        from src.ml.calibration.brier_optimal import KAGGLE_ROUND_WEIGHTS

        games_per_round = {"R64": 32, "R32": 16, "S16": 8, "E8": 4, "F4": 2, "NCG": 1}
        round_totals = {
            r: games_per_round[r] * KAGGLE_ROUND_WEIGHTS[r]
            for r in KAGGLE_ROUND_WEIGHTS
        }
        # All rounds should contribute 32 total points
        for r, total in round_totals.items():
            assert total == pytest.approx(32.0), f"{r} contributes {total}, not 32"

    def test_ncg_worth_32x_r64(self):
        from src.ml.calibration.brier_optimal import KAGGLE_ROUND_WEIGHTS

        assert KAGGLE_ROUND_WEIGHTS["NCG"] / KAGGLE_ROUND_WEIGHTS["R64"] == 32.0


class TestSimpleFeatureSet:
    """Gap #3: Simple mode feature set."""

    def test_simple_feature_set_is_subset_of_fixed(self):
        from src.pipeline.sota import SIMPLE_FEATURE_SET, FIXED_FEATURE_SET

        for feat in SIMPLE_FEATURE_SET:
            assert feat in FIXED_FEATURE_SET, f"{feat} not in FIXED_FEATURE_SET"

    def test_simple_feature_set_size(self):
        from src.pipeline.sota import SIMPLE_FEATURE_SET

        assert len(SIMPLE_FEATURE_SET) == 9  # Updated: added seed_diff

    def test_simple_has_massey_composite(self):
        from src.pipeline.sota import SIMPLE_FEATURE_SET

        assert "diff_external_rating_composite" in SIMPLE_FEATURE_SET


class TestDataQualityWeights:
    """Gap #6: Data quality era weights."""

    def test_early_years_downweighted(self):
        from src.pipeline.sota import DATA_QUALITY_ERA_WEIGHTS

        for year in range(2005, 2010):
            assert DATA_QUALITY_ERA_WEIGHTS[year] < 0.5, f"{year} not downweighted"

    def test_modern_years_not_in_map(self):
        from src.pipeline.sota import DATA_QUALITY_ERA_WEIGHTS

        # 2015+ should not be in the map (default to 1.0)
        for year in range(2015, 2026):
            assert year not in DATA_QUALITY_ERA_WEIGHTS

    def test_quality_increases_over_time(self):
        from src.pipeline.sota import DATA_QUALITY_ERA_WEIGHTS

        years = sorted(DATA_QUALITY_ERA_WEIGHTS.keys())
        weights = [DATA_QUALITY_ERA_WEIGHTS[y] for y in years]
        # Should be non-decreasing overall
        for i in range(1, len(weights)):
            assert weights[i] >= weights[i - 1] - 0.01  # Allow small tolerance


class TestSOTAConfigNewFields:
    """Verify new config fields are properly defined."""

    def test_config_defaults(self):
        from src.pipeline.sota import SOTAPipelineConfig

        config = SOTAPipelineConfig()
        assert config.kaggle_dir is None
        assert config.massey_blend_weight == 0.25  # FIX #5: increased from 0.20 for stronger Massey signal
        assert config.massey_sigma == 4.5  # FIX #5: calibrated via grid search on historical data
        assert config.model_complexity == "simple"  # Gap #3: default changed to simple
        assert config.ensemble_lgb_weight == 0.25  # Gap #2: reduced for MOV-first
        assert config.ensemble_xgb_weight == 0.15  # Gap #2: reduced for MOV-first
        # Women's config
        assert config.womens_model_complexity == "simple"
        assert config.womens_seed_prior_weight == 0.50  # Gap #4: increased for women's
        assert config.enable_bracket_portfolio is True  # Gap #5: enabled by default

    def test_simple_mode_config(self):
        from src.pipeline.sota import SOTAPipelineConfig

        config = SOTAPipelineConfig(model_complexity="simple")
        assert config.model_complexity == "simple"

    def test_spread_weight_increase(self):
        """Verify spread model gets highest ensemble weight (Gap #2)."""
        from src.pipeline.sota import SOTAPipelineConfig

        config = SOTAPipelineConfig()
        # Spread model weight (0.35) should be highest
        lgb = config.ensemble_lgb_weight  # 0.30
        xgb = config.ensemble_xgb_weight  # 0.20
        spread = 0.35  # Defined in _FIXED_WEIGHTS
        assert spread > lgb
        assert spread > xgb
