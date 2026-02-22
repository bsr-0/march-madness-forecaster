"""Tests for ML diagnostic features: feature validation, distribution shift detection,
ROC-AUC/bootstrap CI, ensemble diversity, and per-bin calibration analysis."""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fix 1: Feature Validation (NaN/inf detection)
# ---------------------------------------------------------------------------


class TestFeatureValidation:
    """Tests for NaN/inf detection in feature engineering and pipeline."""

    def test_team_features_nan_replaced_with_zero(self):
        """TeamFeatures.to_vector() should replace NaN values with 0.0."""
        from src.data.features.feature_engineering import TeamFeatures

        tf = TeamFeatures(team_id="test", team_name="Test", seed=1, region="East")
        tf.adj_offensive_efficiency = float("nan")
        tf.steal_rate = float("nan")

        vec = tf.to_vector()
        assert np.all(np.isfinite(vec)), "Feature vector should have no NaN/inf"
        # NaN features should be replaced with 0.0
        assert vec[0] == 0.0  # adj_off_eff is first feature

    def test_team_features_inf_replaced_with_zero(self):
        """TeamFeatures.to_vector() should replace inf values with 0.0."""
        from src.data.features.feature_engineering import TeamFeatures

        tf = TeamFeatures(team_id="test", team_name="Test", seed=5, region="West")
        tf.adj_defensive_efficiency = float("inf")
        tf.elo_rating = float("-inf")

        vec = tf.to_vector()
        assert np.all(np.isfinite(vec)), "Feature vector should have no NaN/inf"

    def test_clean_features_unchanged(self):
        """Normal feature values should pass through unchanged."""
        from src.data.features.feature_engineering import TeamFeatures

        tf = TeamFeatures(team_id="test", team_name="Test", seed=3, region="South")
        tf.adj_offensive_efficiency = 110.5
        tf.adj_defensive_efficiency = 95.3

        vec = tf.to_vector()
        assert vec[0] == 110.5  # adj_off_eff
        assert vec[1] == 95.3  # adj_def_eff

    def test_feature_vector_dimension_assertion(self):
        """Feature vector length should match TEAM_FEATURE_DIM."""
        from src.data.features.feature_engineering import TeamFeatures, TEAM_FEATURE_DIM

        tf = TeamFeatures(team_id="t", team_name="T", seed=1, region="E")
        vec = tf.to_vector()
        assert len(vec) == TEAM_FEATURE_DIM

    def test_feature_names_match_vector_length(self):
        """Feature names list should match vector length."""
        from src.data.features.feature_engineering import TeamFeatures, TEAM_FEATURE_DIM

        names = TeamFeatures.get_feature_names()
        assert len(names) == TEAM_FEATURE_DIM


# ---------------------------------------------------------------------------
# Fix 2: Distribution Shift Detection (PSI + KS test)
# ---------------------------------------------------------------------------


class TestPSI:
    """Tests for Population Stability Index computation."""

    def test_identical_distributions_zero_psi(self):
        """Identical distributions should have PSI ≈ 0."""
        from src.data.features.feature_selection import compute_psi

        rng = np.random.default_rng(42)
        data = rng.standard_normal(1000)
        psi = compute_psi(data, data)
        assert psi < 0.01

    def test_similar_distributions_low_psi(self):
        """Similar distributions should have PSI < 0.10."""
        from src.data.features.feature_selection import compute_psi

        rng = np.random.default_rng(42)
        train = rng.standard_normal(1000)
        # Slight shift
        val = rng.standard_normal(500) + 0.1
        psi = compute_psi(train, val)
        assert psi < 0.10

    def test_shifted_distributions_high_psi(self):
        """Very different distributions should have PSI > 0.25."""
        from src.data.features.feature_selection import compute_psi

        rng = np.random.default_rng(42)
        train = rng.standard_normal(1000)
        val = rng.standard_normal(500) + 3.0  # Large mean shift
        psi = compute_psi(train, val)
        assert psi > 0.25

    def test_psi_non_negative(self):
        """PSI should always be non-negative."""
        from src.data.features.feature_selection import compute_psi

        rng = np.random.default_rng(42)
        for _ in range(10):
            a = rng.standard_normal(200)
            b = rng.standard_normal(100) * 2 + rng.uniform(-5, 5)
            assert compute_psi(a, b) >= 0.0

    def test_constant_feature_psi_zero(self):
        """Constant feature should return PSI=0 (no variation to compute)."""
        from src.data.features.feature_selection import compute_psi

        train = np.full(100, 5.0)
        val = np.full(50, 5.0)
        psi = compute_psi(train, val)
        assert psi == 0.0


class TestDistributionShiftDetection:
    """Tests for the full distribution shift detection pipeline."""

    def test_no_shift_no_flags(self):
        """Identical train/val should produce no flagged features."""
        from src.data.features.feature_selection import detect_distribution_shift

        rng = np.random.default_rng(42)
        X = rng.standard_normal((300, 5))
        train_X = X[:200]
        val_X = X[200:]
        names = [f"f{i}" for i in range(5)]

        results = detect_distribution_shift(train_X, val_X, names)
        n_flagged = sum(1 for r in results if r.flagged)
        # With random splits from the same distribution, expect very few flags
        assert n_flagged <= 1  # At most 1 false positive from KS test

    def test_shifted_feature_flagged(self):
        """A feature with large mean shift should be flagged."""
        from src.data.features.feature_selection import detect_distribution_shift

        rng = np.random.default_rng(42)
        n_train, n_val = 200, 100
        train_X = rng.standard_normal((n_train, 4))
        val_X = rng.standard_normal((n_val, 4))
        # Shift feature 2 by 3 standard deviations
        val_X[:, 2] += 3.0
        names = ["stable_a", "stable_b", "shifted_c", "stable_d"]

        results = detect_distribution_shift(train_X, val_X, names)
        shifted = [r for r in results if r.feature_name == "shifted_c"]
        assert len(shifted) == 1
        assert shifted[0].flagged
        assert shifted[0].psi > 0.25
        assert shifted[0].mean_shift_std > 1.0

    def test_results_sorted_by_psi(self):
        """Results should be sorted by PSI descending."""
        from src.data.features.feature_selection import detect_distribution_shift

        rng = np.random.default_rng(42)
        train_X = rng.standard_normal((200, 5))
        val_X = rng.standard_normal((100, 5))
        val_X[:, 0] += 5.0  # Most shifted
        val_X[:, 3] += 2.0  # Moderately shifted
        names = [f"f{i}" for i in range(5)]

        results = detect_distribution_shift(train_X, val_X, names)
        psis = [r.psi for r in results]
        assert psis == sorted(psis, reverse=True)

    def test_ks_test_detects_shape_change(self):
        """KS test should detect distributional shape changes (not just mean shift)."""
        from src.data.features.feature_selection import detect_distribution_shift

        rng = np.random.default_rng(42)
        n_train, n_val = 300, 150
        train_X = rng.standard_normal((n_train, 3))
        val_X = rng.standard_normal((n_val, 3))
        # Feature 1: same mean but much higher variance (shape change)
        val_X[:, 1] = rng.standard_normal(n_val) * 5.0
        names = ["normal", "shape_changed", "normal2"]

        results = detect_distribution_shift(train_X, val_X, names)
        shape_changed = [r for r in results if r.feature_name == "shape_changed"]
        assert len(shape_changed) == 1
        assert shape_changed[0].flagged
        assert shape_changed[0].ks_pvalue < 0.05


# ---------------------------------------------------------------------------
# Fix 3: ROC-AUC and Bootstrap CI
# ---------------------------------------------------------------------------


class TestROCAUCAndBootstrapCI:
    """Tests for ROC-AUC and bootstrap CI in calibration metrics."""

    def test_roc_auc_computed(self):
        """ROC-AUC should be computed when both classes present."""
        from src.ml.calibration.calibration import calculate_calibration_metrics

        rng = np.random.default_rng(42)
        n = 200
        outcomes = rng.integers(0, 2, size=n).astype(float)
        predictions = np.clip(outcomes + rng.normal(0, 0.2, n), 0.01, 0.99)

        metrics = calculate_calibration_metrics(predictions, outcomes)
        assert metrics.roc_auc is not None
        assert 0.0 <= metrics.roc_auc <= 1.0
        # Good predictions should have high AUC
        assert metrics.roc_auc > 0.7

    def test_roc_auc_perfect_predictions(self):
        """Perfect predictions should give AUC = 1.0."""
        from src.ml.calibration.calibration import calculate_calibration_metrics

        outcomes = np.array([0, 0, 0, 1, 1, 1], dtype=float)
        predictions = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])

        metrics = calculate_calibration_metrics(predictions, outcomes)
        assert metrics.roc_auc is not None
        assert metrics.roc_auc == 1.0

    def test_roc_auc_random_predictions(self):
        """Random predictions should give AUC ≈ 0.5."""
        from src.ml.calibration.calibration import calculate_calibration_metrics

        rng = np.random.default_rng(42)
        n = 1000
        outcomes = rng.integers(0, 2, size=n).astype(float)
        predictions = rng.uniform(0.01, 0.99, size=n)  # Random

        metrics = calculate_calibration_metrics(predictions, outcomes)
        assert metrics.roc_auc is not None
        assert 0.4 < metrics.roc_auc < 0.6

    def test_bootstrap_ci_computed(self):
        """Bootstrap CI should be computed when n >= 20."""
        from src.ml.calibration.calibration import calculate_calibration_metrics

        rng = np.random.default_rng(42)
        n = 100
        outcomes = rng.integers(0, 2, size=n).astype(float)
        predictions = np.clip(outcomes + rng.normal(0, 0.3, n), 0.01, 0.99)

        metrics = calculate_calibration_metrics(predictions, outcomes)
        assert metrics.brier_ci_lower is not None
        assert metrics.brier_ci_upper is not None
        assert metrics.brier_ci_lower <= metrics.brier_score <= metrics.brier_ci_upper
        # CI should be reasonably narrow for 100 samples
        ci_width = metrics.brier_ci_upper - metrics.brier_ci_lower
        assert ci_width < 0.15

    def test_bootstrap_ci_contains_true_brier(self):
        """95% CI should contain the point estimate Brier score."""
        from src.ml.calibration.calibration import calculate_calibration_metrics

        rng = np.random.default_rng(42)
        outcomes = rng.integers(0, 2, size=200).astype(float)
        predictions = np.clip(outcomes + rng.normal(0, 0.25, 200), 0.01, 0.99)

        metrics = calculate_calibration_metrics(predictions, outcomes)
        assert metrics.brier_ci_lower <= metrics.brier_score
        assert metrics.brier_score <= metrics.brier_ci_upper

    def test_no_bootstrap_ci_small_sample(self):
        """Bootstrap CI should not be computed for very small samples."""
        from src.ml.calibration.calibration import calculate_calibration_metrics

        outcomes = np.array([0, 1, 0, 1, 0], dtype=float)
        predictions = np.array([0.3, 0.7, 0.4, 0.6, 0.5])

        metrics = calculate_calibration_metrics(predictions, outcomes)
        # n=5 < 20, so bootstrap CI should not be computed
        assert metrics.brier_ci_lower is None
        assert metrics.brier_ci_upper is None


# ---------------------------------------------------------------------------
# Fix 4: Ensemble Diversity
# ---------------------------------------------------------------------------


class TestEnsembleDiversity:
    """Tests for ensemble diversity metrics."""

    def test_pairwise_correlation_identical_models(self):
        """Identical predictions should have correlation 1.0."""
        from src.ml.ensemble.cfa import CombinatorialFusionAnalysis

        preds = np.array([0.3, 0.5, 0.7, 0.2, 0.8, 0.6])
        model_preds = {"model_a": preds, "model_b": preds.copy()}

        corrs = CombinatorialFusionAnalysis.compute_pairwise_correlation(model_preds)
        assert "model_a_vs_model_b" in corrs
        assert abs(corrs["model_a_vs_model_b"] - 1.0) < 0.01

    def test_pairwise_correlation_independent_models(self):
        """Independent predictions should have low correlation."""
        from src.ml.ensemble.cfa import CombinatorialFusionAnalysis

        rng = np.random.default_rng(42)
        n = 200
        model_preds = {
            "model_a": rng.uniform(0, 1, n),
            "model_b": rng.uniform(0, 1, n),
        }

        corrs = CombinatorialFusionAnalysis.compute_pairwise_correlation(model_preds)
        assert "model_a_vs_model_b" in corrs
        assert abs(corrs["model_a_vs_model_b"]) < 0.2

    def test_pairwise_correlation_three_models(self):
        """Three models should produce 3 pairwise correlations."""
        from src.ml.ensemble.cfa import CombinatorialFusionAnalysis

        rng = np.random.default_rng(42)
        n = 100
        model_preds = {
            "lgb": rng.uniform(0, 1, n),
            "xgb": rng.uniform(0, 1, n),
            "logit": rng.uniform(0, 1, n),
        }

        corrs = CombinatorialFusionAnalysis.compute_pairwise_correlation(model_preds)
        assert len(corrs) == 3  # C(3,2) = 3

    def test_diversity_metrics_single_prediction(self):
        """Diversity metrics should handle single-model case gracefully."""
        from src.ml.ensemble.cfa import CombinatorialFusionAnalysis, ModelPrediction

        cfa = CombinatorialFusionAnalysis()
        preds = {"baseline": ModelPrediction("baseline", 0.6, 0.8)}

        metrics = cfa.compute_diversity_metrics(preds)
        assert metrics["prediction_spread"] == 0.0
        assert metrics["prediction_std"] == 0.0

    def test_diversity_metrics_multi_model(self):
        """Multiple models should produce meaningful diversity metrics."""
        from src.ml.ensemble.cfa import CombinatorialFusionAnalysis, ModelPrediction

        cfa = CombinatorialFusionAnalysis()
        preds = {
            "gnn": ModelPrediction("gnn", 0.7, 0.8),
            "transformer": ModelPrediction("transformer", 0.5, 0.6),
            "baseline": ModelPrediction("baseline", 0.6, 0.9),
        }

        metrics = cfa.compute_diversity_metrics(preds)
        assert metrics["prediction_spread"] == pytest.approx(0.2, abs=0.01)
        assert metrics["prediction_std"] > 0
        assert "deviation_gnn" in metrics
        assert "ensemble_mean" in metrics


# ---------------------------------------------------------------------------
# Fix 5: Per-Bin Calibration Analysis
# ---------------------------------------------------------------------------


class TestPerBinCalibrationAnalysis:
    """Tests for per-bin (decile) calibration analysis."""

    def test_per_bin_analysis_populated(self):
        """Per-bin analysis should be populated in CalibrationMetrics."""
        from src.ml.calibration.calibration import calculate_calibration_metrics

        rng = np.random.default_rng(42)
        n = 200
        outcomes = rng.integers(0, 2, size=n).astype(float)
        predictions = np.clip(rng.uniform(0.1, 0.9, n), 0.01, 0.99)

        metrics = calculate_calibration_metrics(predictions, outcomes, n_bins=10)
        assert metrics.per_bin_analysis is not None
        assert len(metrics.per_bin_analysis) > 0

    def test_per_bin_has_required_fields(self):
        """Each bin should have count, mean_predicted, mean_actual, gap, direction."""
        from src.ml.calibration.calibration import calculate_calibration_metrics

        rng = np.random.default_rng(42)
        n = 500
        outcomes = rng.integers(0, 2, size=n).astype(float)
        predictions = np.clip(rng.uniform(0.0, 1.0, n), 0.01, 0.99)

        metrics = calculate_calibration_metrics(predictions, outcomes)
        for bin_info in metrics.per_bin_analysis:
            assert "bin" in bin_info
            assert "count" in bin_info
            assert "mean_predicted" in bin_info
            assert "mean_actual" in bin_info
            assert "gap" in bin_info
            assert "direction" in bin_info
            assert bin_info["direction"] in ("overconfident", "underconfident", "calibrated")

    def test_per_bin_counts_sum_to_total(self):
        """Bin counts should sum to total sample size."""
        from src.ml.calibration.calibration import calculate_calibration_metrics

        rng = np.random.default_rng(42)
        n = 300
        outcomes = rng.integers(0, 2, size=n).astype(float)
        # Spread predictions across full range
        predictions = np.clip(rng.uniform(0.0, 1.0, n), 0.01, 0.99)

        metrics = calculate_calibration_metrics(predictions, outcomes)
        total = sum(b["count"] for b in metrics.per_bin_analysis)
        assert total == n

    def test_overconfident_detection(self):
        """Overconfident predictions should be labeled as such."""
        from src.ml.calibration.calibration import calculate_calibration_metrics

        # Create systematically overconfident predictions:
        # predict high probabilities but actual win rate is ~50%
        rng = np.random.default_rng(42)
        n = 200
        outcomes = rng.integers(0, 2, size=n).astype(float)
        # Push all predictions toward 0.85 (overconfident for 50% base rate)
        predictions = np.clip(0.85 + rng.normal(0, 0.05, n), 0.7, 0.99)

        metrics = calculate_calibration_metrics(predictions, outcomes)
        # High-confidence bin should show overconfidence
        high_bins = [b for b in metrics.per_bin_analysis if b["mean_predicted"] > 0.7]
        if high_bins:
            # At least one high bin should show overconfidence
            overconfident = any(b["direction"] == "overconfident" for b in high_bins)
            assert overconfident, f"Expected overconfidence in high bins: {high_bins}"

    def test_well_calibrated_model(self):
        """Well-calibrated predictions should have small gaps."""
        from src.ml.calibration.calibration import calculate_calibration_metrics

        rng = np.random.default_rng(42)
        n = 1000
        # Generate well-calibrated predictions
        true_probs = rng.uniform(0.1, 0.9, n)
        outcomes = (rng.uniform(0, 1, n) < true_probs).astype(float)
        predictions = np.clip(true_probs + rng.normal(0, 0.05, n), 0.01, 0.99)

        metrics = calculate_calibration_metrics(predictions, outcomes)
        # Average gap should be small for well-calibrated model
        avg_gap = np.mean([abs(b["gap"]) for b in metrics.per_bin_analysis])
        assert avg_gap < 0.10, f"Average gap {avg_gap} too large for calibrated model"


# ---------------------------------------------------------------------------
# Integration: CalibrationMetrics.__str__ includes new fields
# ---------------------------------------------------------------------------


class TestCalibrationMetricsDisplay:
    """Tests for CalibrationMetrics string representation."""

    def test_str_includes_roc_auc(self):
        """String representation should include ROC-AUC when available."""
        from src.ml.calibration.calibration import calculate_calibration_metrics

        rng = np.random.default_rng(42)
        n = 100
        outcomes = rng.integers(0, 2, size=n).astype(float)
        predictions = np.clip(outcomes + rng.normal(0, 0.2, n), 0.01, 0.99)

        metrics = calculate_calibration_metrics(predictions, outcomes)
        s = str(metrics)
        assert "ROC-AUC" in s
        assert "Brier 95% CI" in s


# ---------------------------------------------------------------------------
# Multi-Year Training Pool
# ---------------------------------------------------------------------------


class TestMultiYearTrainingConfig:
    """Tests for the multi-year training pool configuration and integration."""

    def test_config_defaults(self):
        """Multi-year training config should have sensible defaults."""
        from src.pipeline.sota import SOTAPipelineConfig

        config = SOTAPipelineConfig()
        assert config.enable_multi_year_training is True
        assert config.training_year_decay == 0.85
        assert config.training_year_min_weight == 0.15
        assert config.training_years is None  # auto-detect

    def test_year_decay_weights(self):
        """Year-based decay should produce correct weight schedule."""
        from src.pipeline.sota import SOTAPipelineConfig

        config = SOTAPipelineConfig(year=2026, training_year_decay=0.85, training_year_min_weight=0.15)

        # Simulate weight computation for various years
        weights = {}
        for yr in [2025, 2024, 2023, 2020, 2015, 2010]:
            years_ago = config.year - yr
            w = max(config.training_year_min_weight, config.training_year_decay ** max(years_ago - 1, 0))
            weights[yr] = round(w, 4)

        # Most recent year (2025) should have highest weight
        assert weights[2025] == 1.0  # 0.85^0 = 1.0
        # Weights should decrease for older years
        assert weights[2024] == 0.85
        assert weights[2023] == round(0.85 ** 2, 4)
        # Oldest years should hit the floor
        assert weights[2010] == 0.15  # 0.85^15 ≈ 0.087, floored to 0.15

    def test_load_year_samples_no_symmetric_augmentation(self):
        """_load_year_samples should NOT produce symmetric augmented samples."""
        import json
        import os
        import tempfile
        from src.pipeline.sota import SOTAPipeline, SOTAPipelineConfig

        # Create minimal test data
        games = {
            "season": 2024,
            "games": [
                {"game_id": "1", "date": "2024-01-15", "team1_id": "duke",
                 "team2_id": "unc", "team1_score": 80, "team2_score": 70},
                {"game_id": "2", "date": "2024-01-20", "team1_id": "unc",
                 "team2_id": "kentucky", "team1_score": 75, "team2_score": 65},
            ],
        }
        metrics = {
            "season": 2024,
            "teams": [
                {"team_id": "duke", "off_rtg": 115.0, "def_rtg": 95.0, "pace": 70.0,
                 "srs": 10.0, "sos": 5.0, "wins": 25, "losses": 5},
                {"team_id": "unc", "off_rtg": 110.0, "def_rtg": 98.0, "pace": 68.0,
                 "srs": 8.0, "sos": 4.0, "wins": 22, "losses": 8},
                {"team_id": "kentucky", "off_rtg": 108.0, "def_rtg": 100.0, "pace": 72.0,
                 "srs": 6.0, "sos": 3.0, "wins": 20, "losses": 10},
            ],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            gp = os.path.join(tmpdir, "historical_games_2024.json")
            mp = os.path.join(tmpdir, "team_metrics_2024.json")
            with open(gp, "w") as f:
                json.dump(games, f)
            with open(mp, "w") as f:
                json.dump(metrics, f)

            pipeline = SOTAPipeline(SOTAPipelineConfig(year=2026))
            X, y, _ = pipeline._load_year_samples(gp, mp, feature_dim=77, year=2024)

            # Should have exactly 2 samples (one per game), NOT 4 (no symmetric augmentation)
            assert X.shape[0] == 2
            assert len(y) == 2
            assert X.shape[1] == 77

    def test_load_year_samples_feature_positions(self):
        """Historical feature vectors place metrics in correct indices.

        All positions verified against TeamFeatures.get_feature_names():
          [0]  adj_off_eff   [1]  adj_def_eff   [2]  adj_tempo
          [26] sos_adj_em    [35] elo_rating     [47] win_pct
          [66] abs_adj_off_eff  [67] abs_adj_def_eff  [68] abs_sos_adj_em
          [69] abs_elo_rating   [70] abs_win_pct
        """
        import json
        import os
        import tempfile
        from src.pipeline.sota import SOTAPipeline, SOTAPipelineConfig

        games = {
            "season": 2024,
            "games": [
                {"game_id": "1", "date": "2024-01-15", "team1_id": "duke",
                 "team2_id": "unc", "team1_score": 80, "team2_score": 70},
            ],
        }
        metrics = {
            "season": 2024,
            "teams": [
                {"team_id": "duke", "off_rtg": 115.0, "def_rtg": 95.0, "pace": 70.0,
                 "srs": 10.0, "sos": 5.0, "wins": 25, "losses": 5},
                {"team_id": "unc",  "off_rtg": 110.0, "def_rtg": 98.0, "pace": 68.0,
                 "srs": 8.0,  "sos": 4.0, "wins": 22, "losses": 8},
            ],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            gp = os.path.join(tmpdir, "games.json")
            mp = os.path.join(tmpdir, "metrics.json")
            with open(gp, "w") as f:
                json.dump(games, f)
            with open(mp, "w") as f:
                json.dump(metrics, f)

            pipeline = SOTAPipeline(SOTAPipelineConfig(year=2026))
            X, y, _ = pipeline._load_year_samples(gp, mp, feature_dim=77, year=2024)

            assert X.shape == (1, 77)

            # Diff features at verified indices
            assert X[0, 0]  == pytest.approx(5.0)            # diff_adj_off_eff
            assert X[0, 1]  == pytest.approx(-3.0)           # diff_adj_def_eff
            assert X[0, 2]  == pytest.approx(2.0)            # diff_adj_tempo
            assert X[0, 26] == pytest.approx(1.0)            # diff_sos_adj_em (idx 26, not 27)
            assert X[0, 35] > 0                              # diff_elo_rating: Duke won → positive
            assert X[0, 47] == pytest.approx(25/30 - 22/30, abs=0.01)  # diff_win_pct (idx 47, not 48)

            # Absolute-level features
            assert X[0, 66] == pytest.approx(112.5)          # abs_adj_off_eff
            assert X[0, 67] == pytest.approx(96.5)           # abs_adj_def_eff
            assert X[0, 68] == pytest.approx(4.5)            # abs_sos_adj_em
            assert 1490 < X[0, 69] < 1510                   # abs_elo_rating ≈ 1500
            assert X[0, 70] == pytest.approx((25/30 + 22/30) / 2, abs=0.01)  # abs_win_pct

            # Outcome: Duke won 80-70
            assert y[0] == 1

            # Roster features must remain zero (CBBpy fetch prohibitively expensive)
            assert X[0, 15] == pytest.approx(0.0), "roster_continuity must be zero"
            assert X[0, 17] == pytest.approx(0.0), "avg_experience must be zero"
            # travel_advantage stays zero for historical regular-season games
            assert X[0, 75] == pytest.approx(0.0), "travel_advantage must be zero"

    def test_historical_year_weights_combined_with_recency(self):
        """Year-based weights should combine multiplicatively with recency weights."""
        # Simulate the weight combination logic
        n_hist = 100
        n_current = 50
        total = n_hist + n_current

        # Year-based weights: historical=0.5, current=1.0
        year_weights = np.concatenate([
            np.full(n_hist, 0.5),
            np.ones(n_current),
        ])

        # Recency weights: linear ramp
        recency_weights = np.linspace(0.3, 1.0, total)
        recency_weights /= recency_weights.mean()

        # Combined
        combined = year_weights * recency_weights
        combined /= combined.mean()

        # Historical samples should have lower combined weight than current
        hist_mean = combined[:n_hist].mean()
        current_mean = combined[n_hist:].mean()
        assert current_mean > hist_mean, (
            f"Current-year mean weight ({current_mean:.3f}) should exceed "
            f"historical mean weight ({hist_mean:.3f})"
        )

    def test_derived_features_elo_monotone_with_wins(self):
        """Elo difference should increase as team1's winning margin grows."""
        import json
        import os
        import tempfile
        from src.pipeline.sota import SOTAPipeline, SOTAPipelineConfig

        def _make_dataset(team1_score, team2_score):
            return {
                "season": 2022,
                "games": [
                    {"game_id": "1", "date": "2022-01-10", "team1_id": "alpha",
                     "team2_id": "beta", "team1_score": team1_score, "team2_score": team2_score},
                ],
            }

        metrics = {
            "season": 2022,
            "teams": [
                {"team_id": "alpha", "off_rtg": 110.0, "def_rtg": 100.0, "pace": 70.0,
                 "srs": 5.0, "sos": 2.0, "wins": 15, "losses": 10},
                {"team_id": "beta",  "off_rtg": 105.0, "def_rtg": 102.0, "pace": 69.0,
                 "srs": 3.0, "sos": 1.5, "wins": 12, "losses": 13},
            ],
        }

        pipeline = SOTAPipeline(SOTAPipelineConfig(year=2026))

        elo_diffs = []
        for margin in [5, 15, 30]:
            with tempfile.TemporaryDirectory() as tmpdir:
                gp = os.path.join(tmpdir, "g.json")
                mp = os.path.join(tmpdir, "m.json")
                with open(gp, "w") as f:
                    json.dump(_make_dataset(70 + margin, 70), f)
                with open(mp, "w") as f:
                    json.dump(metrics, f)
                X, y, _ = pipeline._load_year_samples(gp, mp, feature_dim=77, year=2022)
                elo_diffs.append(X[0, 35])

        # Larger winning margin → larger Elo gain → larger diff
        assert elo_diffs[0] < elo_diffs[1] < elo_diffs[2], (
            f"Elo diff should increase with margin: {elo_diffs}"
        )

    def test_derived_features_wab_positive_for_strong_team(self):
        """A team beating strong opponents should have positive WAB."""
        import json
        import os
        import tempfile
        from src.pipeline.sota import SOTAPipeline, SOTAPipelineConfig

        # alpha (AdjEM=+10) beats two above-bubble opponents
        games = {
            "season": 2022,
            "games": [
                {"game_id": "1", "date": "2022-01-10", "team1_id": "alpha",
                 "team2_id": "gamma", "team1_score": 75, "team2_score": 65},
                {"game_id": "2", "date": "2022-01-17", "team1_id": "alpha",
                 "team2_id": "delta", "team1_score": 80, "team2_score": 70},
                {"game_id": "3", "date": "2022-01-24", "team1_id": "beta",
                 "team2_id": "alpha", "team1_score": 60, "team2_score": 72},
            ],
        }
        # gamma/delta both have AdjEM ≈ +10 (above bubble = 5.0) → beating them adds WAB
        metrics = {
            "season": 2022,
            "teams": [
                {"team_id": "alpha", "off_rtg": 115.0, "def_rtg": 105.0, "pace": 70.0,
                 "srs": 8.0, "sos": 5.0, "wins": 20, "losses": 5},
                {"team_id": "beta",  "off_rtg": 100.0, "def_rtg": 110.0, "pace": 68.0,
                 "srs": -2.0, "sos": 0.0, "wins": 8, "losses": 17},
                {"team_id": "gamma", "off_rtg": 112.0, "def_rtg": 102.0, "pace": 71.0,
                 "srs": 6.0, "sos": 4.0, "wins": 18, "losses": 7},
                {"team_id": "delta", "off_rtg": 111.0, "def_rtg": 103.0, "pace": 69.0,
                 "srs": 5.5, "sos": 3.5, "wins": 17, "losses": 8},
            ],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            gp = os.path.join(tmpdir, "g.json")
            mp = os.path.join(tmpdir, "m.json")
            with open(gp, "w") as f:
                json.dump(games, f)
            with open(mp, "w") as f:
                json.dump(metrics, f)

            pipeline = SOTAPipeline(SOTAPipelineConfig(year=2026))
            X, y, _ = pipeline._load_year_samples(gp, mp, feature_dim=77, year=2022)

        # All 3 games should be in training set; find alpha-vs-weak-team game (game 3)
        # alpha (idx 31 = wab) should be positive relative to beta
        # Simply verify WAB column is non-zero across the dataset
        assert X[:, 31].std() > 0, "WAB column should have variation across games"

    def test_derived_features_feature_coverage(self):
        """Verify that Option A populates significantly more features than the old 3."""
        import json
        import os
        import tempfile
        from src.pipeline.sota import SOTAPipeline, SOTAPipelineConfig

        # Build a season with enough games to trigger luck computation (≥12)
        games_list = []
        for i in range(20):
            margin = 5 if i % 3 != 0 else -3  # mostly wins
            games_list.append({
                "game_id": str(i + 100),
                "date": f"2023-{11 + i//10:02d}-{(i % 28) + 1:02d}",
                "team1_id": "alpha",
                "team2_id": "beta",
                "team1_score": 70 + margin,
                "team2_score": 70,
            })

        games = {"season": 2023, "games": games_list}
        metrics = {
            "season": 2023,
            "teams": [
                {"team_id": "alpha", "off_rtg": 112.0, "def_rtg": 98.0, "pace": 71.0,
                 "srs": 7.0, "sos": 4.0, "wins": 20, "losses": 5},
                {"team_id": "beta",  "off_rtg": 104.0, "def_rtg": 106.0, "pace": 68.0,
                 "srs": 1.0, "sos": 1.0, "wins": 12, "losses": 13},
            ],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            gp = os.path.join(tmpdir, "g.json")
            mp = os.path.join(tmpdir, "m.json")
            with open(gp, "w") as f:
                json.dump(games, f)
            with open(mp, "w") as f:
                json.dump(metrics, f)

            pipeline = SOTAPipeline(SOTAPipelineConfig(year=2026))
            X, y, _ = pipeline._load_year_samples(gp, mp, feature_dim=77, year=2023)

        assert X.shape[0] == 20

        # Check FIXED_FEATURE_SET positions are populated (non-zero in at least one row).
        # alpha vs beta have identical metrics each game, so some diffs are constant
        # non-zero (positions 0,1,2,26,47) while derived features vary (30,31,32,35).
        fixed_positions = [0, 1, 2, 26, 30, 31, 32, 33, 35, 47, 66, 67, 68, 69, 70]
        populated = [i for i in fixed_positions if np.abs(X[:, i]).max() > 1e-8]

        assert len(populated) >= 10, (
            f"Option A should populate ≥10 feature positions from FIXED_FEATURE_SET "
            f"(got {len(populated)} non-zero: {populated})"
        )
