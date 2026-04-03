"""Tests for low-coverage pipeline and ML modules.

Covers:
- src/pipeline/stages/ev_mode.py
- src/pipeline/_optional_imports.py
- src/ml/ensemble/cfa.py
- src/ml/ensemble/margin_first_ensemble.py
- src/ml/ensemble/tournament_expert.py
- src/ml/calibration/post_processing.py
- src/ml/calibration/calibration.py
- src/ml/gnn/schedule_graph.py
- src/ml/transformer/game_sequence.py
- src/ml/research/research_loop.py
"""

import copy
import math
import sys
from dataclasses import asdict
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# 1. ev_mode.py
# ---------------------------------------------------------------------------
from src.pipeline.stages.ev_mode import (
    classify_pick_strategy,
    compute_leverage,
    expected_pool_rank,
    format_leverage_table,
)


class TestComputeLeverage:
    def test_positive_leverage(self):
        # Model sees higher prob than public -> positive leverage
        lev = compute_leverage(0.8, 0.4)
        assert lev > 0

    def test_zero_public_pick_rate(self):
        assert compute_leverage(0.7, 0.0) == 0.0

    def test_negative_public_pick_rate(self):
        assert compute_leverage(0.7, -0.1) == 0.0

    def test_scoring_weight_scales(self):
        lev1 = compute_leverage(0.8, 0.4, scoring_weight=1.0)
        lev2 = compute_leverage(0.8, 0.4, scoring_weight=2.0)
        assert pytest.approx(lev2, rel=1e-9) == 2.0 * lev1

    def test_very_small_public_rate_clamped(self):
        # public_pick_rate 0.005 -> denominator clamped to 0.01
        lev = compute_leverage(0.5, 0.005)
        expected = 1.0 * (0.5 - 0.005) / 0.01
        assert pytest.approx(lev, rel=1e-9) == expected


class TestClassifyPickStrategy:
    def test_neutral_small_leverage(self):
        assert classify_pick_strategy(0.05, 0.5, 0.5) == "neutral"

    def test_contrarian(self):
        # High positive leverage, public_pick_rate < 0.3
        assert classify_pick_strategy(0.5, 0.6, 0.2) == "contrarian"

    def test_chalk(self):
        # Positive leverage, public_pick_rate >= 0.3
        assert classify_pick_strategy(0.5, 0.6, 0.5) == "chalk"

    def test_fade(self):
        assert classify_pick_strategy(-0.5, 0.3, 0.7) == "fade"

    def test_neutral_fallthrough(self):
        # leverage > 0.1 but model_prob <= public_pick_rate and leverage > 0
        # This hits the final "return neutral"
        assert classify_pick_strategy(0.15, 0.3, 0.4) == "neutral"


class TestExpectedPoolRank:
    def test_single_entry_pool(self):
        assert expected_pool_rank(100.0, 1) == 1.0

    def test_zero_std(self):
        assert expected_pool_rank(100.0, 50, score_std=0.0) == 1.0

    def test_default_at_mean(self):
        # z=0, prob_beat=0.5, rank = pool_size * 0.5
        rank = expected_pool_rank(100.0, 100)
        assert rank == 50.0

    def test_large_pool(self):
        rank = expected_pool_rank(120.0, 1000)
        assert rank >= 1.0


class TestFormatLeverageTable:
    def test_empty_teams(self):
        result = format_leverage_table([])
        assert "Team" in result
        assert "-" * 65 in result

    def test_basic_formatting(self):
        teams = [
            {
                "team": "Duke",
                "model_prob": 0.75,
                "public_rate": 0.60,
                "leverage": 0.25,
                "strategy": "chalk",
            },
            {
                "team": "FGCU",
                "model_prob": 0.30,
                "public_rate": 0.10,
                "leverage": 2.0,
                "strategy": "contrarian",
            },
        ]
        result = format_leverage_table(teams, top_n=5)
        assert "Duke" in result
        assert "FGCU" in result

    def test_top_n_limits_output(self):
        teams = [{"team": f"T{i}", "leverage": i} for i in range(20)]
        result = format_leverage_table(teams, top_n=3)
        # Header + separator + 3 data lines
        lines = result.strip().split("\n")
        assert len(lines) == 5

    def test_missing_keys_use_defaults(self):
        teams = [{}]
        result = format_leverage_table(teams)
        assert "?" in result  # default team name


# ---------------------------------------------------------------------------
# 2. _optional_imports.py
# ---------------------------------------------------------------------------


class TestOptionalImports:
    def test_torch_flag_is_bool(self):
        from src.pipeline._optional_imports import TORCH_AVAILABLE

        assert isinstance(TORCH_AVAILABLE, bool)

    def test_sklearn_flag_is_bool(self):
        from src.pipeline._optional_imports import SKLEARN_AVAILABLE

        assert isinstance(SKLEARN_AVAILABLE, bool)

    def test_scaler_flag_is_bool(self):
        from src.pipeline._optional_imports import SCALER_AVAILABLE

        assert isinstance(SCALER_AVAILABLE, bool)

    def test_optuna_flag_is_bool(self):
        from src.pipeline._optional_imports import OPTUNA_AVAILABLE

        assert isinstance(OPTUNA_AVAILABLE, bool)

    def test_significance_flag_is_bool(self):
        from src.pipeline._optional_imports import SIGNIFICANCE_TESTING_AVAILABLE

        assert isinstance(SIGNIFICANCE_TESTING_AVAILABLE, bool)

    def test_ablation_flag_is_bool(self):
        from src.pipeline._optional_imports import ABLATION_AVAILABLE

        assert isinstance(ABLATION_AVAILABLE, bool)

    def test_spread_model_flag_is_bool(self):
        from src.pipeline._optional_imports import SPREAD_MODEL_AVAILABLE

        assert isinstance(SPREAD_MODEL_AVAILABLE, bool)

    def test_tournament_sigma_flag_is_bool(self):
        from src.pipeline._optional_imports import TOURNAMENT_SIGMA_AVAILABLE

        assert isinstance(TOURNAMENT_SIGMA_AVAILABLE, bool)

    def test_bayesian_bt_flag_is_bool(self):
        from src.pipeline._optional_imports import BAYESIAN_BT_AVAILABLE

        assert isinstance(BAYESIAN_BT_AVAILABLE, bool)


# ---------------------------------------------------------------------------
# 3. cfa.py
# ---------------------------------------------------------------------------
from src.ml.ensemble.cfa import (
    ModelPrediction,
    create_matchup_features,
)


class TestModelPrediction:
    def test_dataclass_creation(self):
        mp = ModelPrediction(
            model_name="baseline",
            win_probability=0.65,
            confidence=0.8,
        )
        assert mp.model_name == "baseline"
        assert mp.win_probability == 0.65
        assert mp.confidence == 0.8
        assert mp.features is None

    def test_with_features(self):
        mp = ModelPrediction(
            model_name="gnn",
            win_probability=0.55,
            confidence=0.7,
            features={"adj_em": 15.0},
        )
        assert mp.features == {"adj_em": 15.0}


class TestCreateMatchupFeatures:
    def test_returns_array_and_names(self):
        t1 = {"adj_efficiency_margin": 20.0, "adj_tempo": 70.0}
        t2 = {"adj_efficiency_margin": 10.0, "adj_tempo": 65.0}
        feats, names = create_matchup_features(t1, t2)
        assert isinstance(feats, np.ndarray)
        assert isinstance(names, list)
        assert len(feats) == len(names)

    def test_differential_features(self):
        t1 = {"adj_efficiency_margin": 20.0}
        t2 = {"adj_efficiency_margin": 10.0}
        feats, names = create_matchup_features(t1, t2)
        idx = names.index("diff_adj_efficiency_margin")
        assert feats[idx] == pytest.approx(10.0)

    def test_empty_stats_use_defaults(self):
        feats, names = create_matchup_features({}, {})
        assert len(feats) > 0
        # Tempo interaction with defaults: 68*68/4624 = 1.0
        idx = names.index("tempo_interaction")
        assert feats[idx] == pytest.approx(68 * 68 / 4624)

    def test_interaction_features_present(self):
        _, names = create_matchup_features({}, {})
        assert "tempo_interaction" in names
        assert "t1_off_vs_t2_def" in names
        assert "t2_off_vs_t1_def" in names


class TestLightGBMRankerImportGuard:
    def test_raises_without_lightgbm(self):
        from src.ml.ensemble.cfa import LightGBMRanker, LIGHTGBM_AVAILABLE

        if not LIGHTGBM_AVAILABLE:
            with pytest.raises(ImportError):
                LightGBMRanker()

    def test_predict_without_training_raises(self):
        from src.ml.ensemble.cfa import LightGBMRanker, LIGHTGBM_AVAILABLE

        if LIGHTGBM_AVAILABLE:
            ranker = LightGBMRanker()
            with pytest.raises(ValueError, match="not trained"):
                ranker.predict(np.zeros((1, 5)))

    def test_feature_importance_untrained(self):
        from src.ml.ensemble.cfa import LightGBMRanker, LIGHTGBM_AVAILABLE

        if LIGHTGBM_AVAILABLE:
            ranker = LightGBMRanker()
            assert ranker.get_feature_importance() == {}


class TestLightGBMMarginRegressor:
    def test_margin_to_probability(self):
        from src.ml.ensemble.cfa import LightGBMMarginRegressor, LIGHTGBM_AVAILABLE

        if not LIGHTGBM_AVAILABLE:
            pytest.skip("LightGBM not available")
        reg = LightGBMMarginRegressor()
        margins = np.array([0.0, 10.0, -10.0])
        probs = reg._margin_to_probability(margins)
        assert probs[0] == pytest.approx(0.5)
        assert probs[1] > 0.5
        assert probs[2] < 0.5

    def test_default_logistic_scale(self):
        from src.ml.ensemble.cfa import LightGBMMarginRegressor, LIGHTGBM_AVAILABLE

        if not LIGHTGBM_AVAILABLE:
            pytest.skip("LightGBM not available")
        reg = LightGBMMarginRegressor()
        assert reg.logistic_scale == 5.5

    def test_custom_logistic_scale(self):
        from src.ml.ensemble.cfa import LightGBMMarginRegressor, LIGHTGBM_AVAILABLE

        if not LIGHTGBM_AVAILABLE:
            pytest.skip("LightGBM not available")
        reg = LightGBMMarginRegressor(logistic_scale=8.0)
        assert reg.logistic_scale == 8.0

    def test_predict_margin_untrained_raises(self):
        from src.ml.ensemble.cfa import LightGBMMarginRegressor, LIGHTGBM_AVAILABLE

        if not LIGHTGBM_AVAILABLE:
            pytest.skip("LightGBM not available")
        reg = LightGBMMarginRegressor()
        with pytest.raises(ValueError, match="not trained"):
            reg.predict_margin(np.zeros((1, 5)))


# ---------------------------------------------------------------------------
# 4. margin_first_ensemble.py
# ---------------------------------------------------------------------------
from src.ml.ensemble.margin_first_ensemble import (
    DEFAULT_ROUND_SIGMAS,
    RoundSpecificCalibrator,
    _EXPERIMENTAL_WEIGHTS,
    _PRODUCTION_WEIGHTS,
)


class TestRoundSpecificCalibrator:
    def test_init_defaults(self):
        cal = RoundSpecificCalibrator()
        assert cal.calibrated is False
        assert cal.round_params == {}

    def test_convert_spread_to_prob_defaults(self):
        cal = RoundSpecificCalibrator()
        prob = cal.convert_spread_to_prob(0.0, "R64")
        assert prob == pytest.approx(0.5)

    def test_convert_spread_positive(self):
        cal = RoundSpecificCalibrator()
        prob = cal.convert_spread_to_prob(10.0, "R64")
        assert prob > 0.5

    def test_convert_spread_negative(self):
        cal = RoundSpecificCalibrator()
        prob = cal.convert_spread_to_prob(-10.0, "R64")
        assert prob < 0.5

    def test_convert_spreads_batch(self):
        cal = RoundSpecificCalibrator()
        spreads = np.array([-5.0, 0.0, 5.0])
        probs = cal.convert_spreads_to_probs(spreads, "R64")
        assert probs[0] < 0.5
        assert probs[1] == pytest.approx(0.5)
        assert probs[2] > 0.5

    def test_get_default_sigma(self):
        cal = RoundSpecificCalibrator()
        assert cal._get_default_sigma("R64") == DEFAULT_ROUND_SIGMAS["R64"]
        assert cal._get_default_sigma("NCG") == DEFAULT_ROUND_SIGMAS["NCG"]
        # Unknown round falls back to 10.5
        assert cal._get_default_sigma("UNKNOWN") == 10.5

    def test_calibrate_few_samples_uses_default(self):
        cal = RoundSpecificCalibrator()
        spreads = {"R64": np.array([1.0, 2.0])}
        outcomes = {"R64": np.array([1.0, 0.0])}
        cal.calibrate(spreads, outcomes, use_optuna=False)
        assert cal.calibrated is True
        assert cal.round_params["R64"]["k"] == 1.0

    def test_optimize_grid(self):
        cal = RoundSpecificCalibrator()
        np.random.seed(42)
        spreads = np.random.randn(50) * 10
        outcomes = (spreads > 0).astype(float)
        result = cal._optimize_grid(spreads, outcomes, "R64")
        assert "k" in result
        assert "sigma" in result
        assert "brier" in result


# ---------------------------------------------------------------------------
# 5. tournament_expert.py
# ---------------------------------------------------------------------------
from src.ml.ensemble.tournament_expert import TournamentExpert


class TestTournamentExpert:
    def test_init_defaults(self):
        te = TournamentExpert()
        assert te.sigma == 11.0
        assert te.blend_weight == 0.30
        assert te.trained is False

    def test_custom_init(self):
        te = TournamentExpert(sigma=9.0, blend_weight=0.5)
        assert te.sigma == 9.0
        assert te.blend_weight == 0.5

    def test_predict_untrained_returns_half(self):
        te = TournamentExpert()
        X = np.zeros((5, 10))
        probs = te.predict_probability(X)
        np.testing.assert_array_equal(probs, np.full(5, 0.5))

    def test_predict_untrained_1d(self):
        te = TournamentExpert()
        X = np.zeros(10)
        probs = te.predict_probability(X)
        np.testing.assert_array_equal(probs, np.full(1, 0.5))

    def test_blend_untrained_returns_regular(self):
        te = TournamentExpert()
        regular = np.array([0.6, 0.7, 0.8])
        X = np.zeros((3, 10))
        result = te.blend_with_regular(regular, X)
        np.testing.assert_array_equal(result, regular)

    def test_spread_to_prob(self):
        te = TournamentExpert(sigma=11.0)
        probs = te._spread_to_prob(np.array([0.0, 11.0, -11.0]))
        assert probs[0] == pytest.approx(0.5)
        assert probs[1] > 0.5
        assert probs[2] < 0.5

    def test_calibrate_sigma_insufficient_data(self):
        te = TournamentExpert()
        te.spread_model = MagicMock()
        original_sigma = te.sigma
        te._calibrate_sigma(np.zeros((5, 3)), np.zeros(5))
        # Should not change sigma with < 20 samples
        assert te.sigma == original_sigma

    def test_blend_weight_class_constant(self):
        assert TournamentExpert.BLEND_WEIGHT == 0.30

    def test_train_no_lightgbm(self):
        from src.ml.ensemble.tournament_expert import LIGHTGBM_AVAILABLE

        if LIGHTGBM_AVAILABLE:
            pytest.skip("LightGBM is available")
        te = TournamentExpert()
        result = te.train(
            np.zeros((100, 5)),
            np.zeros(100),
            np.zeros(100),
        )
        assert result["trained"] is False


# ---------------------------------------------------------------------------
# 6. post_processing.py
# ---------------------------------------------------------------------------
from src.ml.calibration.post_processing import (
    BrierOptimalClipper,
    FavoriteLongshotBiasCorrector,
    PostProcessingPipeline,
)


class TestFavoriteLongshotBiasCorrector:
    def test_no_correction_in_middle(self):
        flb = FavoriteLongshotBiasCorrector()
        assert flb.correct(0.6, 3, 6) == 0.6

    def test_correction_heavy_favorite(self):
        flb = FavoriteLongshotBiasCorrector(threshold=0.85, regression_strength=0.2)
        corrected = flb.correct(0.95, 1, 16)
        assert corrected < 0.95  # Should pull toward historical

    def test_correction_heavy_underdog(self):
        flb = FavoriteLongshotBiasCorrector(threshold=0.85, regression_strength=0.2)
        corrected = flb.correct(0.05, 16, 1)
        assert corrected > 0.05

    def test_seed_prior_table_used(self):
        table = {(1, 16): 0.99}
        flb = FavoriteLongshotBiasCorrector(seed_prior_table=table)
        hist = flb._get_seed_prior(1, 16)
        assert hist == 0.99

    def test_seed_prior_reverse_lookup(self):
        table = {(1, 16): 0.99}
        flb = FavoriteLongshotBiasCorrector(seed_prior_table=table)
        hist = flb._get_seed_prior(16, 1)
        assert hist == pytest.approx(0.01)

    def test_seed_prior_logistic_fallback(self):
        flb = FavoriteLongshotBiasCorrector()
        hist = flb._get_seed_prior(1, 16)
        expected = 1.0 / (1.0 + math.exp(0.175 * (1 - 16)))
        assert hist == pytest.approx(expected)

    def test_correct_batch(self):
        flb = FavoriteLongshotBiasCorrector()
        probs = np.array([0.5, 0.9, 0.1])
        seeds1 = np.array([5, 1, 16])
        seeds2 = np.array([5, 16, 1])
        result = flb.correct_batch(probs, seeds1, seeds2)
        assert result[0] == 0.5  # unchanged


class TestBrierOptimalClipper:
    def test_default_bounds(self):
        clipper = BrierOptimalClipper()
        assert clipper.clip(0.0001) == 0.001
        assert clipper.clip(0.9999) == 0.999

    def test_seed_specific_bounds(self):
        clipper = BrierOptimalClipper()
        # 1 vs 16 allows wider bounds
        clipped = clipper.clip(0.0001, seed1=1, seed2=16)
        assert clipped == 0.0005

    def test_no_seed_specific(self):
        clipper = BrierOptimalClipper(seed_specific=False)
        lo, hi = clipper._get_bounds(1, 16)
        assert lo == clipper.default_lo

    def test_clip_batch_no_seeds(self):
        clipper = BrierOptimalClipper()
        probs = np.array([0.0, 0.5, 1.0])
        result = clipper.clip_batch(probs)
        assert result[0] == 0.001
        assert result[2] == 0.999

    def test_clip_batch_with_seeds(self):
        clipper = BrierOptimalClipper()
        probs = np.array([0.5])
        seeds1 = np.array([5])
        seeds2 = np.array([12])
        result = clipper.clip_batch(probs, seeds1, seeds2)
        assert result[0] == 0.5


class TestPostProcessingPipeline:
    def test_pipeline_process(self):
        pp = PostProcessingPipeline()
        result = pp.process(0.5, 5, 12)
        assert 0.001 <= result <= 0.999

    def test_pipeline_clips_extreme(self):
        pp = PostProcessingPipeline()
        result = pp.process(0.0, 5, 12)
        assert result >= 0.001

    def test_pipeline_batch(self):
        pp = PostProcessingPipeline()
        probs = np.array([0.5, 0.95, 0.05])
        seeds1 = np.array([5, 1, 16])
        seeds2 = np.array([12, 16, 1])
        result = pp.process_batch(probs, seeds1, seeds2)
        assert len(result) == 3
        assert all(0.001 <= p <= 0.999 for p in result)


# ---------------------------------------------------------------------------
# 7. calibration.py
# ---------------------------------------------------------------------------
from src.ml.calibration.calibration import (
    BrierScoreOptimizer,
    CalibrationLeakageError,
    CalibrationMetrics,
    CalibrationPipeline,
    PlattScaling,
    TemperatureScaling,
    calculate_calibration_metrics,
)


class TestCalibrationMetrics:
    def test_str_representation(self):
        cm = CalibrationMetrics(
            brier_score=0.2,
            log_loss=0.5,
            expected_calibration_error=0.05,
            max_calibration_error=0.1,
            accuracy=0.75,
            prob_true=np.array([0.5]),
            prob_pred=np.array([0.5]),
        )
        s = str(cm)
        assert "Brier Score: 0.2000" in s
        assert "Log Loss: 0.5000" in s

    def test_str_with_auc(self):
        cm = CalibrationMetrics(
            brier_score=0.2,
            log_loss=0.5,
            expected_calibration_error=0.05,
            max_calibration_error=0.1,
            accuracy=0.75,
            prob_true=np.array([0.5]),
            prob_pred=np.array([0.5]),
            roc_auc=0.85,
        )
        assert "ROC-AUC: 0.8500" in str(cm)

    def test_str_with_ci(self):
        cm = CalibrationMetrics(
            brier_score=0.2,
            log_loss=0.5,
            expected_calibration_error=0.05,
            max_calibration_error=0.1,
            accuracy=0.75,
            prob_true=np.array([0.5]),
            prob_pred=np.array([0.5]),
            brier_ci_lower=0.18,
            brier_ci_upper=0.22,
        )
        assert "Brier 95% CI" in str(cm)


class TestBrierScoreOptimizer:
    def test_perfect_predictions(self):
        preds = np.array([1.0, 0.0, 1.0])
        outcomes = np.array([1.0, 0.0, 1.0])
        assert BrierScoreOptimizer.calculate(preds, outcomes) == 0.0

    def test_worst_predictions(self):
        preds = np.array([0.0, 1.0])
        outcomes = np.array([1.0, 0.0])
        assert BrierScoreOptimizer.calculate(preds, outcomes) == 1.0

    def test_coin_flip(self):
        preds = np.array([0.5, 0.5, 0.5, 0.5])
        outcomes = np.array([1.0, 0.0, 1.0, 0.0])
        assert BrierScoreOptimizer.calculate(preds, outcomes) == 0.25

    def test_decompose(self):
        np.random.seed(42)
        preds = np.random.rand(100)
        outcomes = (preds > 0.5).astype(float)
        rel, res, unc = BrierScoreOptimizer.decompose(preds, outcomes)
        assert rel >= 0
        assert res >= 0
        assert unc >= 0

    def test_skill_score_better_than_baseline(self):
        np.random.seed(42)
        outcomes = np.array([1.0, 0.0, 1.0, 0.0, 1.0])
        preds = np.array([0.9, 0.1, 0.8, 0.2, 0.7])
        bss = BrierScoreOptimizer.skill_score(preds, outcomes)
        assert bss > 0

    def test_skill_score_all_same_outcome(self):
        outcomes = np.array([1.0, 1.0, 1.0])
        preds = np.array([0.9, 0.8, 0.7])
        bss = BrierScoreOptimizer.skill_score(preds, outcomes)
        # bs_ref = 0 when all outcomes are 1
        assert bss == 0.0


class TestPlattScaling:
    def test_init(self):
        ps = PlattScaling()
        assert ps.a == 1.0
        assert ps.b == 0.0
        assert ps.fitted is False

    def test_calibrate_unfitted_raises(self):
        ps = PlattScaling()
        with pytest.raises(ValueError, match="Not fitted"):
            ps.calibrate(np.array([0.5]))

    def test_fit_and_calibrate(self):
        np.random.seed(42)
        ps = PlattScaling()
        preds = np.random.rand(100)
        outcomes = (preds > 0.5).astype(float)
        ps.fit(preds, outcomes)
        assert ps.fitted is True
        calibrated = ps.calibrate(preds)
        assert len(calibrated) == 100
        assert all(0 <= p <= 1 for p in calibrated)


class TestTemperatureScaling:
    def test_init(self):
        ts = TemperatureScaling()
        assert ts.temperature == 1.0
        assert ts.fitted is False

    def test_calibrate_unfitted_raises(self):
        ts = TemperatureScaling()
        with pytest.raises(ValueError, match="Not fitted"):
            ts.calibrate(np.array([0.5]))

    def test_fit_and_calibrate(self):
        np.random.seed(42)
        ts = TemperatureScaling()
        preds = np.clip(np.random.rand(200), 0.05, 0.95)
        outcomes = (preds > 0.5).astype(float)
        ts.fit(preds, outcomes)
        assert ts.fitted is True
        calibrated = ts.calibrate(preds)
        assert len(calibrated) == 200
        assert all(0 <= p <= 1 for p in calibrated)

    def test_temperature_preserves_ranking(self):
        np.random.seed(42)
        ts = TemperatureScaling()
        preds = np.array([0.3, 0.5, 0.7, 0.9])
        outcomes = np.array([0, 0, 1, 1]).astype(float)
        ts.fit(preds, outcomes)
        calibrated = ts.calibrate(preds)
        # Monotonic: ranking should be preserved
        for i in range(len(calibrated) - 1):
            assert calibrated[i] <= calibrated[i + 1]

    def test_bootstrap_ci(self):
        np.random.seed(42)
        ts = TemperatureScaling()
        preds = np.clip(np.random.rand(100), 0.05, 0.95)
        outcomes = (preds > 0.5).astype(float)
        ts.fit(preds, outcomes)
        lo, hi, T_values = ts.bootstrap_ci(preds, outcomes, n_bootstrap=50)
        assert lo <= hi
        assert len(T_values) == 50


class TestCalculateCalibrationMetrics:
    def test_basic_metrics(self):
        np.random.seed(42)
        preds = np.random.rand(50)
        outcomes = (preds > 0.5).astype(float)
        with patch.dict("sys.modules", {"sklearn.metrics": MagicMock()}):
            metrics = calculate_calibration_metrics(preds, outcomes)
        assert metrics.brier_score >= 0
        assert metrics.accuracy >= 0
        assert metrics.expected_calibration_error >= 0

    def test_per_bin_analysis(self):
        np.random.seed(42)
        preds = np.random.rand(50)
        outcomes = (preds > 0.5).astype(float)
        with patch.dict("sys.modules", {"sklearn.metrics": MagicMock()}):
            metrics = calculate_calibration_metrics(preds, outcomes)
        assert metrics.per_bin_analysis is not None
        assert len(metrics.per_bin_analysis) > 0
        assert "bin" in metrics.per_bin_analysis[0]

    def test_bootstrap_ci_computed(self):
        np.random.seed(42)
        preds = np.random.rand(50)
        outcomes = (preds > 0.5).astype(float)
        with patch.dict("sys.modules", {"sklearn.metrics": MagicMock()}):
            metrics = calculate_calibration_metrics(preds, outcomes)
        assert metrics.brier_ci_lower is not None
        assert metrics.brier_ci_upper is not None
        assert metrics.brier_ci_lower <= metrics.brier_ci_upper


def _mock_sklearn_metrics():
    """Context manager to mock sklearn.metrics for environments without sklearn."""
    return patch.dict("sys.modules", {"sklearn.metrics": MagicMock()})


class TestCalibrationPipeline:
    def test_none_method(self):
        pipeline = CalibrationPipeline(method="none")
        assert pipeline.calibrator is None
        preds = np.array([0.5, 0.6])
        np.testing.assert_array_equal(pipeline.calibrate(preds), preds)

    def test_temperature_method(self):
        pipeline = CalibrationPipeline(method="temperature")
        assert isinstance(pipeline.calibrator, TemperatureScaling)

    def test_fit_returns_metrics(self):
        np.random.seed(42)
        pipeline = CalibrationPipeline(method="temperature")
        preds = np.clip(np.random.rand(100), 0.05, 0.95)
        outcomes = (preds > 0.5).astype(float)
        with _mock_sklearn_metrics():
            metrics = pipeline.fit(preds, outcomes)
        assert isinstance(metrics, CalibrationMetrics)

    def test_downgrade_isotonic_small_sample(self):
        from src.ml.calibration.calibration import SKLEARN_AVAILABLE as CAL_SKLEARN

        if not CAL_SKLEARN:
            pytest.skip("sklearn not available for isotonic calibration test")
        np.random.seed(42)
        pipeline = CalibrationPipeline(method="isotonic", nested_cv=False)
        preds = np.clip(np.random.rand(50), 0.05, 0.95)
        outcomes = (preds > 0.5).astype(float)
        with _mock_sklearn_metrics():
            pipeline.fit(preds, outcomes)
        # Should have been downgraded to temperature
        assert pipeline.method == "temperature"
        assert pipeline._downgraded_from == "isotonic"

    def test_leakage_detection_strict(self):
        np.random.seed(42)
        pipeline = CalibrationPipeline(method="temperature", strict_mode=True)
        preds = np.clip(np.random.rand(100), 0.05, 0.95)
        outcomes = (preds > 0.5).astype(float)
        with _mock_sklearn_metrics():
            pipeline.fit(preds, outcomes)
            with pytest.raises(CalibrationLeakageError):
                pipeline.evaluate(preds, outcomes)

    def test_leakage_detection_non_strict(self):
        np.random.seed(42)
        pipeline = CalibrationPipeline(method="temperature", strict_mode=False)
        preds = np.clip(np.random.rand(100), 0.05, 0.95)
        outcomes = (preds > 0.5).astype(float)
        with _mock_sklearn_metrics():
            pipeline.fit(preds, outcomes)
            # Should warn but not raise
            pre, post = pipeline.evaluate(preds, outcomes)
        assert isinstance(pre, CalibrationMetrics)

    def test_evaluate_different_data(self):
        np.random.seed(42)
        pipeline = CalibrationPipeline(method="temperature")
        train_preds = np.clip(np.random.rand(100), 0.05, 0.95)
        train_outcomes = (train_preds > 0.5).astype(float)
        with _mock_sklearn_metrics():
            pipeline.fit(train_preds, train_outcomes)
            test_preds = np.clip(np.random.rand(50), 0.05, 0.95)
            test_outcomes = (test_preds > 0.5).astype(float)
            pre, post = pipeline.evaluate(test_preds, test_outcomes)
        assert isinstance(pre, CalibrationMetrics)
        assert isinstance(post, CalibrationMetrics)


# ---------------------------------------------------------------------------
# 8. schedule_graph.py
# ---------------------------------------------------------------------------
from src.ml.gnn.schedule_graph import (
    ScheduleEdge,
    ScheduleGraph,
    compute_multi_hop_sos,
)


class TestScheduleEdge:
    def test_adjusted_margin_home(self):
        edge = ScheduleEdge(
            game_id="g1",
            team1_id="A",
            team2_id="B",
            actual_margin=10.0,
            xp_margin=8.0,
            location_weight=1.0,
            game_date="2024-01-01",
        )
        # Home: adjustment = (1.0-0.5)*7.0 = 3.5
        assert edge.adjusted_margin == pytest.approx(6.5)

    def test_adjusted_margin_neutral(self):
        edge = ScheduleEdge(
            game_id="g1",
            team1_id="A",
            team2_id="B",
            actual_margin=10.0,
            xp_margin=8.0,
            location_weight=0.5,
            game_date="2024-01-01",
        )
        assert edge.adjusted_margin == pytest.approx(10.0)

    def test_adjusted_margin_away(self):
        edge = ScheduleEdge(
            game_id="g1",
            team1_id="A",
            team2_id="B",
            actual_margin=10.0,
            xp_margin=8.0,
            location_weight=0.0,
            game_date="2024-01-01",
        )
        # Away: adjustment = (0.0-0.5)*7.0 = -3.5
        assert edge.adjusted_margin == pytest.approx(13.5)

    def test_quality_adjusted_margin(self):
        edge = ScheduleEdge(
            game_id="g1",
            team1_id="A",
            team2_id="B",
            actual_margin=10.0,
            xp_margin=8.0,
            location_weight=0.5,
            game_date="2024-01-01",
        )
        expected = 0.35 * 10.0 + 0.65 * 8.0
        assert edge.quality_adjusted_margin == pytest.approx(expected)


class TestScheduleGraph:
    def _make_graph(self):
        teams = ["A", "B", "C"]
        g = ScheduleGraph(teams)
        g.add_game(
            ScheduleEdge(
                game_id="g1",
                team1_id="A",
                team2_id="B",
                actual_margin=10.0,
                xp_margin=8.0,
                location_weight=0.5,
                game_date="2024-01-01",
            )
        )
        g.add_game(
            ScheduleEdge(
                game_id="g2",
                team1_id="B",
                team2_id="C",
                actual_margin=5.0,
                xp_margin=4.0,
                location_weight=0.5,
                game_date="2024-01-15",
            )
        )
        g.add_game(
            ScheduleEdge(
                game_id="g3",
                team1_id="A",
                team2_id="C",
                actual_margin=-3.0,
                xp_margin=-2.0,
                location_weight=0.5,
                game_date="2024-02-01",
            )
        )
        return g

    def test_init(self):
        g = ScheduleGraph(["A", "B", "C"])
        assert g.n_teams == 3
        assert len(g.edges) == 0

    def test_add_games(self):
        g = self._make_graph()
        assert len(g.edges) == 3

    def test_adjacency_matrix_shape(self):
        g = self._make_graph()
        adj = g.get_adjacency_matrix()
        assert adj.shape == (3, 3)

    def test_adjacency_matrix_unweighted(self):
        g = self._make_graph()
        adj = g.get_adjacency_matrix(weighted=False)
        # Unweighted: weight=1.0, so adj[i,j] = (1-1)*r = 0, adj[j,i] = 1*r = 1
        # At least one direction should be nonzero for each edge
        for edge in g.edges:
            i = g.team_to_idx[edge.team1_id]
            j = g.team_to_idx[edge.team2_id]
            assert adj[i, j] + adj[j, i] == pytest.approx(1.0)

    def test_feature_matrix_default(self):
        g = ScheduleGraph(["A", "B"])
        feats = g.get_feature_matrix(feature_dim=4)
        assert feats.shape == (2, 4)
        np.testing.assert_array_equal(feats, np.zeros((2, 4)))

    def test_set_team_features(self):
        g = ScheduleGraph(["A", "B"])
        g.set_team_features("A", np.array([1.0, 2.0, 3.0]))
        feats = g.get_feature_matrix(feature_dim=4)
        assert feats[0, 0] == 1.0
        assert feats[0, 1] == 2.0

    def test_pagerank_sos(self):
        g = self._make_graph()
        scores = g.compute_pagerank_sos()
        assert len(scores) == 3
        assert all(v > 0 for v in scores.values())

    def test_temporal_decay(self):
        teams = ["A", "B"]
        g = ScheduleGraph(teams, temporal_decay=0.5)
        g.add_game(
            ScheduleEdge(
                game_id="g1",
                team1_id="A",
                team2_id="B",
                actual_margin=10.0,
                xp_margin=8.0,
                location_weight=0.5,
                game_date="2024-01-01",
            )
        )
        g.add_game(
            ScheduleEdge(
                game_id="g2",
                team1_id="A",
                team2_id="B",
                actual_margin=5.0,
                xp_margin=4.0,
                location_weight=0.5,
                game_date="2024-03-01",
            )
        )
        recency = g._compute_recency_multipliers()
        assert recency["g1"] < recency["g2"]  # Earlier game has less weight

    def test_temporal_decay_zero(self):
        g = ScheduleGraph(["A", "B"], temporal_decay=0.0)
        g.add_game(
            ScheduleEdge(
                game_id="g1",
                team1_id="A",
                team2_id="B",
                actual_margin=10.0,
                xp_margin=8.0,
                location_weight=0.5,
                game_date="2024-01-01",
            )
        )
        recency = g._compute_recency_multipliers()
        assert recency == {}

    def test_win_quality_metrics_no_edges(self):
        g = ScheduleGraph(["A", "B"])
        metrics = g.compute_win_quality_metrics()
        assert "A" in metrics
        assert metrics["A"]["best_win_percentile"] == 0.5

    def test_win_quality_metrics_with_edges(self):
        g = self._make_graph()
        metrics = g.compute_win_quality_metrics()
        assert "A" in metrics
        assert "best_win_percentile" in metrics["A"]
        assert "paper_tiger_score" in metrics["A"]
        assert "dominance_ratio" in metrics["A"]


class TestComputeMultiHopSos:
    def test_basic(self):
        teams = ["A", "B", "C"]
        g = ScheduleGraph(teams)
        g.add_game(
            ScheduleEdge(
                game_id="g1",
                team1_id="A",
                team2_id="B",
                actual_margin=10.0,
                xp_margin=8.0,
                location_weight=0.5,
                game_date="2024-01-01",
            )
        )
        g.add_game(
            ScheduleEdge(
                game_id="g2",
                team1_id="B",
                team2_id="C",
                actual_margin=5.0,
                xp_margin=4.0,
                location_weight=0.5,
                game_date="2024-01-15",
            )
        )
        scores = compute_multi_hop_sos(g, hops=2)
        assert len(scores) == 3

    def test_hops_parameter(self):
        teams = ["A", "B", "C"]
        g = ScheduleGraph(teams)
        g.add_game(
            ScheduleEdge(
                game_id="g1",
                team1_id="A",
                team2_id="B",
                actual_margin=10.0,
                xp_margin=8.0,
                location_weight=0.5,
                game_date="2024-01-01",
            )
        )
        scores1 = compute_multi_hop_sos(g, hops=1)
        scores3 = compute_multi_hop_sos(g, hops=3)
        # Both should return valid scores
        assert len(scores1) == 3
        assert len(scores3) == 3


# ---------------------------------------------------------------------------
# 9. game_sequence.py
# ---------------------------------------------------------------------------
from src.ml.transformer.game_sequence import (
    GameEmbedding,
    SeasonSequence,
    compute_momentum_features,
)


class TestGameEmbedding:
    def _make_game(self, **kwargs):
        defaults = dict(
            game_id="g1",
            team_id="A",
            opponent_id="B",
            game_date="2024-01-01",
            game_number=1,
            offensive_efficiency=110.0,
            defensive_efficiency=95.0,
            tempo=70.0,
            margin=15.0,
            win=True,
            is_conference_game=False,
            is_neutral_site=False,
        )
        defaults.update(kwargs)
        return GameEmbedding(**defaults)

    def test_to_vector_shape(self):
        g = self._make_game()
        v = g.to_vector()
        assert v.shape == (8,)

    def test_to_vector_values(self):
        g = self._make_game()
        v = g.to_vector()
        assert v[0] == pytest.approx(110.0 / 100.0)
        assert v[1] == pytest.approx(95.0 / 100.0)
        assert v[4] == 1.0  # win = True

    def test_opponent_rank_none(self):
        g = self._make_game(opponent_rank=None)
        v = g.to_vector()
        assert v[7] == pytest.approx(1.0 / 200)

    def test_opponent_rank_set(self):
        g = self._make_game(opponent_rank=10)
        v = g.to_vector()
        assert v[7] == pytest.approx(1.0 / 10)


class TestSeasonSequence:
    def _make_season(self, n_games=10):
        games = []
        for i in range(n_games):
            games.append(
                GameEmbedding(
                    game_id=f"g{i}",
                    team_id="A",
                    opponent_id="B",
                    game_date=f"2024-01-{i + 1:02d}",
                    game_number=i + 1,
                    offensive_efficiency=110.0 + i,
                    defensive_efficiency=95.0,
                    tempo=70.0,
                    margin=5.0 + i,
                    win=True,
                    is_conference_game=i > 5,
                    is_neutral_site=False,
                )
            )
        return SeasonSequence(team_id="A", games=games)

    def test_to_matrix(self):
        s = self._make_season(10)
        m = s.to_matrix()
        assert m.shape == (10, 8)

    def test_get_recent_window(self):
        s = self._make_season(20)
        recent = s.get_recent_window(5)
        assert recent.shape == (5, 8)

    def test_get_recent_window_larger_than_season(self):
        s = self._make_season(3)
        recent = s.get_recent_window(10)
        assert recent.shape == (3, 8)


class TestComputeMomentumFeatures:
    def test_empty_season(self):
        s = SeasonSequence(team_id="A", games=[])
        features = compute_momentum_features(s)
        assert features == {}

    def test_basic_features(self):
        games = []
        for i in range(15):
            games.append(
                GameEmbedding(
                    game_id=f"g{i}",
                    team_id="A",
                    opponent_id="B",
                    game_date=f"2024-01-{i + 1:02d}",
                    game_number=i + 1,
                    offensive_efficiency=100.0 + i,
                    defensive_efficiency=95.0,
                    tempo=70.0,
                    margin=float(i - 5),
                    win=i > 5,
                    is_conference_game=False,
                    is_neutral_site=False,
                )
            )
        s = SeasonSequence(team_id="A", games=games)
        features = compute_momentum_features(s)
        assert "momentum_margin" in features
        assert "momentum_offense" in features
        assert "momentum_defense" in features
        assert "current_streak" in features
        assert "recent_win_pct" in features
        assert "variance_margin" in features

    def test_win_streak(self):
        games = []
        for i in range(10):
            games.append(
                GameEmbedding(
                    game_id=f"g{i}",
                    team_id="A",
                    opponent_id="B",
                    game_date=f"2024-01-{i + 1:02d}",
                    game_number=i + 1,
                    offensive_efficiency=110.0,
                    defensive_efficiency=95.0,
                    tempo=70.0,
                    margin=10.0,
                    win=True,
                    is_conference_game=False,
                    is_neutral_site=False,
                )
            )
        s = SeasonSequence(team_id="A", games=games)
        features = compute_momentum_features(s)
        assert features["current_streak"] == 10

    def test_broken_streak(self):
        games = []
        for i in range(10):
            win = i != 8  # Loss at game 8 (0-indexed), only game 9 is streak
            games.append(
                GameEmbedding(
                    game_id=f"g{i}",
                    team_id="A",
                    opponent_id="B",
                    game_date=f"2024-01-{i + 1:02d}",
                    game_number=i + 1,
                    offensive_efficiency=110.0,
                    defensive_efficiency=95.0,
                    tempo=70.0,
                    margin=10.0 if win else -5.0,
                    win=win,
                    is_conference_game=False,
                    is_neutral_site=False,
                )
            )
        s = SeasonSequence(team_id="A", games=games)
        features = compute_momentum_features(s)
        assert features["current_streak"] == 1


# ---------------------------------------------------------------------------
# 10. research_loop.py
# ---------------------------------------------------------------------------
from src.ml.research.research_loop import (
    ConfigMutator,
    ImprovementCandidate,
    ImprovementGate,
    LoopEvaluator,
    ResearchCycleResult,
    ResearchLoop,
    ResearchTrajectory,
)


class TestImprovementCandidate:
    def test_dataclass(self):
        ic = ImprovementCandidate(
            name="test",
            config_key="spread_weight",
            old_value=0.5,
            new_value=0.6,
            brier_delta=-0.005,
            p_value=0.03,
            source="sensitivity",
        )
        assert ic.name == "test"
        assert ic.adopted is False

    def test_to_dict(self):
        ic = ImprovementCandidate(
            name="test",
            config_key="spread_weight",
            old_value=0.5,
            new_value=0.6,
            brier_delta=-0.005,
            p_value=0.03,
            source="sensitivity",
        )
        d = ic.to_dict()
        assert d["name"] == "test"
        assert d["brier_delta"] == -0.005


class TestResearchCycleResult:
    def test_defaults(self):
        r = ResearchCycleResult()
        assert r.cycle_id == ""
        assert r.n_experiments_run == 0

    def test_to_dict(self):
        r = ResearchCycleResult(cycle_id="c001", baseline_brier=0.2)
        d = r.to_dict()
        assert d["cycle_id"] == "c001"
        assert d["baseline_brier"] == 0.2


class TestResearchTrajectory:
    def test_empty(self):
        t = ResearchTrajectory()
        assert t.total_improvement == 0.0
        assert t.n_total_experiments == 0
        assert t.n_total_adopted == 0

    def test_with_cycles(self):
        t = ResearchTrajectory()
        t.cycles.append(
            ResearchCycleResult(
                cycle_id="c1",
                total_improvement=0.01,
                n_experiments_run=5,
                n_improvements_adopted=2,
                final_brier=0.19,
            )
        )
        t.cycles.append(
            ResearchCycleResult(
                cycle_id="c2",
                total_improvement=0.005,
                n_experiments_run=3,
                n_improvements_adopted=1,
                final_brier=0.185,
            )
        )
        assert t.total_improvement == pytest.approx(0.015)
        assert t.n_total_experiments == 8
        assert t.n_total_adopted == 3

    def test_brier_trajectory(self):
        t = ResearchTrajectory()
        t.cycles.append(ResearchCycleResult(cycle_id="c1", final_brier=0.2))
        t.cycles.append(ResearchCycleResult(cycle_id="c2", final_brier=0.19))
        traj = t.brier_trajectory()
        assert len(traj) == 2
        assert traj[0] == ("c1", 0.2)

    def test_brier_trajectory_skips_zero(self):
        t = ResearchTrajectory()
        t.cycles.append(ResearchCycleResult(cycle_id="c1", final_brier=0.0))
        assert t.brier_trajectory() == []


class TestConfigMutator:
    def test_list_mutable_params(self):
        params = ConfigMutator.list_mutable_params()
        assert "spread_weight" in params
        assert "spread_sigma" in params

    def test_generate_variants(self):
        config = MagicMock()
        config.spread_weight = 0.5
        variants = ConfigMutator.generate_variants(config, "spread_weight", n_points=5)
        assert len(variants) >= 5
        # Each variant is (config, value) tuple
        for cfg, val in variants:
            assert 0.0 <= val <= 1.0

    def test_generate_variants_unknown_param(self):
        config = MagicMock()
        variants = ConfigMutator.generate_variants(config, "nonexistent_param")
        assert variants == []

    def test_generate_variants_no_attr(self):
        config = MagicMock(spec=[])  # No attributes
        variants = ConfigMutator.generate_variants(config, "spread_weight")
        # getattr returns None, should return empty
        assert variants == []

    def test_apply_change(self):
        config = MagicMock()
        config.spread_weight = 0.5
        new_cfg = ConfigMutator.apply_change(config, "spread_weight", 0.7)
        assert new_cfg.spread_weight == 0.7


class TestLoopEvaluator:
    def test_default_evaluate(self):
        evaluator = LoopEvaluator()
        score = evaluator.evaluate(MagicMock())
        assert score == 0.25  # Placeholder

    def test_custom_eval_fn(self):
        evaluator = LoopEvaluator(eval_fn=lambda cfg: 0.18)
        score = evaluator.evaluate(MagicMock())
        assert score == 0.18

    def test_caching(self):
        call_count = 0

        def counting_fn(cfg):
            nonlocal call_count
            call_count += 1
            return 0.2

        evaluator = LoopEvaluator(eval_fn=counting_fn)
        evaluator.evaluate(MagicMock(), cache_key="k1")
        evaluator.evaluate(MagicMock(), cache_key="k1")
        assert call_count == 1

    def test_clear_cache(self):
        evaluator = LoopEvaluator(eval_fn=lambda cfg: 0.2)
        evaluator.evaluate(MagicMock(), cache_key="k1")
        evaluator.clear_cache()
        assert evaluator._cache == {}


class TestImprovementGate:
    def test_adopt_good_candidate(self):
        gate = ImprovementGate(min_brier_improvement=0.002, max_p_value=0.10)
        candidate = ImprovementCandidate(
            name="test",
            config_key="k",
            old_value=0.5,
            new_value=0.6,
            brier_delta=-0.005,
            p_value=0.03,
            source="sensitivity",
        )
        adopt, reason = gate.should_adopt(candidate)
        assert adopt is True
        assert "Passed" in reason

    def test_reject_small_improvement(self):
        gate = ImprovementGate(min_brier_improvement=0.002)
        candidate = ImprovementCandidate(
            name="test",
            config_key="k",
            old_value=0.5,
            new_value=0.6,
            brier_delta=-0.0001,
            p_value=0.01,
            source="sensitivity",
        )
        adopt, reason = gate.should_adopt(candidate)
        assert adopt is False
        assert "threshold" in reason

    def test_reject_high_p_value(self):
        gate = ImprovementGate(max_p_value=0.10)
        candidate = ImprovementCandidate(
            name="test",
            config_key="k",
            old_value=0.5,
            new_value=0.6,
            brier_delta=-0.01,
            p_value=0.20,
            source="sensitivity",
        )
        adopt, reason = gate.should_adopt(candidate)
        assert adopt is False
        assert "p-value" in reason

    def test_reject_worsening(self):
        gate = ImprovementGate()
        candidate = ImprovementCandidate(
            name="test",
            config_key="k",
            old_value=0.5,
            new_value=0.6,
            brier_delta=0.01,
            p_value=0.01,
            source="sensitivity",
        )
        adopt, reason = gate.should_adopt(candidate)
        assert adopt is False
        assert "worsens" in reason.lower()

    def test_reject_insufficient_folds(self):
        gate = ImprovementGate(
            min_brier_improvement=0.001,
            require_multiple_folds=True,
            min_folds_improved=0.6,
        )
        candidate = ImprovementCandidate(
            name="test",
            config_key="k",
            old_value=0.5,
            new_value=0.6,
            brier_delta=-0.01,
            p_value=0.01,
            source="sensitivity",
        )
        fold_deltas = [-0.01, 0.02, 0.03, 0.01, 0.02]  # Only 1/5 improved
        adopt, reason = gate.should_adopt(candidate, fold_deltas=fold_deltas)
        assert adopt is False
        assert "folds" in reason.lower()


class TestImprovementGateCohensD:
    """Tests for Cohen's d effect size check added to ImprovementGate."""

    def test_reject_low_cohens_d(self):
        """High variance across folds → low Cohen's d → reject."""
        gate = ImprovementGate(
            min_brier_improvement=0.001,
            max_p_value=0.10,
            min_cohens_d=0.2,
        )
        candidate = ImprovementCandidate(
            name="test",
            config_key="k",
            old_value=0.5,
            new_value=0.6,
            brier_delta=-0.005,
            p_value=0.04,
            source="param_sweep",
        )
        # Mean delta = -0.005, but huge variance → Cohen's d ≈ 0.05
        fold_deltas = [-0.005, -0.10, 0.09, -0.005, -0.08, 0.07, -0.01, 0.03]
        adopt, reason = gate.should_adopt(candidate, fold_deltas=fold_deltas)
        assert adopt is False
        assert "cohen" in reason.lower()

    def test_accept_high_cohens_d(self):
        """Consistent improvement across folds → high Cohen's d → accept."""
        gate = ImprovementGate(
            min_brier_improvement=0.001,
            max_p_value=0.10,
            min_cohens_d=0.2,
        )
        candidate = ImprovementCandidate(
            name="test",
            config_key="k",
            old_value=0.5,
            new_value=0.6,
            brier_delta=-0.01,
            p_value=0.03,
            source="param_sweep",
        )
        # Consistent negative deltas → high Cohen's d
        fold_deltas = [-0.012, -0.009, -0.011, -0.010, -0.008, -0.011, -0.010, -0.009]
        adopt, reason = gate.should_adopt(candidate, fold_deltas=fold_deltas)
        assert adopt is True

    def test_cohens_d_skipped_without_folds(self):
        """When no fold_deltas provided, Cohen's d check is skipped."""
        gate = ImprovementGate(
            min_brier_improvement=0.001,
            max_p_value=0.10,
            min_cohens_d=0.2,
        )
        candidate = ImprovementCandidate(
            name="test",
            config_key="k",
            old_value=0.5,
            new_value=0.6,
            brier_delta=-0.005,
            p_value=0.04,
            source="param_sweep",
        )
        adopt, reason = gate.should_adopt(candidate, fold_deltas=None)
        assert adopt is True


class TestParamSweepPairedTTest:
    """Tests for paired t-test support in ResearchLoop.run_param_sweep."""

    def test_param_sweep_uses_paired_ttest(self):
        """When eval_folds_fn is provided, p-values come from ttest_rel."""
        loop = ResearchLoop(log_path="/tmp/test_research_log.jsonl")
        rng = np.random.default_rng(42)

        # Simulate per-fold Brier scores: baseline vs a clearly better variant
        baseline_folds = [0.20, 0.22, 0.19, 0.21, 0.20, 0.23, 0.18, 0.21]

        def mock_eval_folds(cfg):
            sw = getattr(cfg, "spread_weight", 0.5)
            # Lower spread_weight → better scores
            offset = (sw - 0.5) * 0.04
            return [b + offset + rng.normal(0, 0.001) for b in baseline_folds]

        config = MagicMock()
        config.spread_weight = 0.5
        # ConfigMutator needs these attributes
        config.tournament_shrinkage = 0.06
        config.seed_prior_weight = 0.10

        candidates = loop.run_param_sweep(
            config,
            "spread_weight",
            eval_fn=lambda cfg: float(np.mean(mock_eval_folds(cfg))),
            eval_folds_fn=mock_eval_folds,
            n_points=5,
        )
        assert len(candidates) > 0
        for c in candidates:
            # p-values should not be the old heuristic values
            assert c.p_value not in (0.05, 0.5) or abs(c.brier_delta) < 1e-6
            assert len(c.fold_briers) > 0

    def test_param_sweep_falls_back_without_folds(self):
        """When only eval_fn is provided, fallback heuristic p-values are used
        (then Holm-corrected)."""
        loop = ResearchLoop(log_path="/tmp/test_research_log.jsonl")

        config = MagicMock()
        config.spread_weight = 0.5
        config.tournament_shrinkage = 0.06
        config.seed_prior_weight = 0.10

        candidates = loop.run_param_sweep(
            config,
            "spread_weight",
            eval_fn=lambda cfg: 0.20 + (getattr(cfg, "spread_weight", 0.5) - 0.5) * 0.1,
            n_points=5,
        )
        assert len(candidates) > 0
        # Scalar eval_fn → single-element fold_briers
        for c in candidates:
            assert len(c.fold_briers) == 1
        # After Holm-Bonferroni, adjusted p-values are >= the raw heuristic values
        for c in candidates:
            assert c.p_value >= 0.05 or c.p_value >= 0.5

    def test_param_sweep_applies_holm_bonferroni(self):
        """P-values from sweep are Holm-Bonferroni corrected (adjusted upward)."""
        loop = ResearchLoop(log_path="/tmp/test_research_log.jsonl")

        baseline_folds = [0.20, 0.22, 0.19, 0.21, 0.20, 0.23, 0.18, 0.21]

        def mock_eval_folds(cfg):
            sw = getattr(cfg, "spread_weight", 0.5)
            # Small consistent offset per variant
            offset = (sw - 0.5) * 0.02
            return [b + offset for b in baseline_folds]

        config = MagicMock()
        config.spread_weight = 0.5
        config.tournament_shrinkage = 0.06
        config.seed_prior_weight = 0.10

        candidates = loop.run_param_sweep(
            config,
            "spread_weight",
            eval_fn=lambda cfg: float(np.mean(mock_eval_folds(cfg))),
            eval_folds_fn=mock_eval_folds,
            n_points=5,
        )
        # With multiple candidates, Holm-Bonferroni adjusts p-values upward.
        # Adjusted p-values should be >= 0 and <= 1.
        assert len(candidates) > 1
        for c in candidates:
            assert 0.0 <= c.p_value <= 1.0


class TestResearchLoop:
    def test_init(self):
        loop = ResearchLoop(log_path="/tmp/test_research_log.jsonl")
        assert len(loop.trajectory.cycles) == 0
        assert loop._cycle_counter == 0

    def test_report_empty(self):
        loop = ResearchLoop(log_path="/tmp/test_research_log.jsonl")
        report = loop.report()
        assert "Research Loop Report" in report
        assert "Cycles completed: 0" in report

    def test_generate_research_report_empty(self):
        loop = ResearchLoop(log_path="/tmp/test_research_log.jsonl")
        report = loop.generate_research_report()
        assert report["trajectory"]["n_cycles"] == 0

    def test_extract_param_name(self):
        loop = ResearchLoop(log_path="/tmp/test_research_log.jsonl")
        assert loop._extract_param_name("Optimize spread_weight (current=0.5)") == "spread_weight"
        assert loop._extract_param_name("Something else") is None

    def test_run_cycle_no_diagnostics(self):
        loop = ResearchLoop(log_path="/tmp/test_research_log.jsonl")
        config = MagicMock(spec=[])  # Empty spec so no param sweeps generated
        result = loop.run_cycle(
            config,
            eval_fn=lambda cfg: 0.2,
            max_experiments=0,
        )
        assert result.cycle_id == "cycle_001"
        assert result.baseline_brier == 0.2
        assert len(loop.trajectory.cycles) == 1
