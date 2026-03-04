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
# CombinatorialFusionAnalysis class was removed from cfa.py.
# The ensemble now uses fixed-weight averaging (LGB/XGB/Logistic)
# instead of CFA-style dynamic fusion.


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

    def test_load_year_samples_is_tombstoned(self):
        """_load_year_samples() must raise NotImplementedError.

        The method used season-end team_metrics aggregates as training features,
        causing temporal leakage.  It has been replaced by
        _load_year_samples_incremental() which computes features from box scores
        using IncrementalMetricsEngine.compute_as_of(game_date).
        """
        import pytest
        from src.pipeline.sota import SOTAPipeline, SOTAPipelineConfig

        pipeline = SOTAPipeline(SOTAPipelineConfig(year=2026))
        with pytest.raises(NotImplementedError, match="season-end"):
            pipeline._load_year_samples("g.json", "m.json", feature_dim=75, year=2024)

    def test_load_year_samples_feature_positions_tombstoned(self):
        """_load_year_samples() is tombstoned; confirm it raises immediately."""
        import pytest
        from src.pipeline.sota import SOTAPipeline, SOTAPipelineConfig

        pipeline = SOTAPipeline(SOTAPipelineConfig(year=2026))
        with pytest.raises(NotImplementedError, match="season-end"):
            pipeline._load_year_samples("g.json", "m.json", feature_dim=75, year=2024)

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

    def test_derived_features_elo_monotone_with_wins_tombstoned(self):
        """_load_year_samples() is tombstoned; this test confirms it raises."""
        import pytest
        from src.pipeline.sota import SOTAPipeline, SOTAPipelineConfig

        pipeline = SOTAPipeline(SOTAPipelineConfig(year=2026))
        with pytest.raises(NotImplementedError, match="season-end"):
            pipeline._load_year_samples("g.json", "m.json", feature_dim=75, year=2022)

    def test_derived_features_wab_tombstoned(self):
        """_load_year_samples() is tombstoned; this test confirms it raises."""
        import pytest
        from src.pipeline.sota import SOTAPipeline, SOTAPipelineConfig

        pipeline = SOTAPipeline(SOTAPipelineConfig(year=2026))
        with pytest.raises(NotImplementedError, match="season-end"):
            pipeline._load_year_samples("g.json", "m.json", feature_dim=75, year=2022)

    def test_derived_features_feature_coverage_tombstoned(self):
        """_load_year_samples() is tombstoned; this test confirms it raises."""
        import pytest
        from src.pipeline.sota import SOTAPipeline, SOTAPipelineConfig

        pipeline = SOTAPipeline(SOTAPipelineConfig(year=2026))
        with pytest.raises(NotImplementedError, match="season-end"):
            pipeline._load_year_samples("g.json", "m.json", feature_dim=75, year=2023)
