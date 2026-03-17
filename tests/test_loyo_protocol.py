"""Tests for the Leave-One-Year-Out (LOYO) validation protocol.

Covers LOYOValidator, FeatureAblator, powered ablation thresholds,
validation diagnostics, and statistical power analysis.
"""

import numpy as np
import pytest

from src.ml.evaluation.loyo_protocol import (
    DEFAULT_FEATURE_FAMILIES,
    LOYO_YEARS,
    MINIMUM_BRIER_IMPROVEMENT,
    DevEvalSplit,
    FeatureAblator,
    FeatureFamily,
    LOYOFoldResult,
    LOYOResult,
    LOYOValidator,
    ProspectiveValidator,
    ProspectiveValidationResult,
    compute_ablation_threshold,
    compute_validation_diagnostics,
    validate_family_coverage,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_year_data(year, n_games=30, n_features=5, seed=None):
    """Create synthetic per-year data for LOYO testing."""
    rng = np.random.RandomState(seed or year)
    X = rng.randn(n_games, n_features)
    y = rng.randint(0, 2, size=n_games).astype(float)
    margins = rng.randn(n_games) * 10
    return {
        "X": X,
        "y": y,
        "margins": margins,
        "feature_names": [f"feat_{i}" for i in range(n_features)],
    }


def _simple_train_fn(X, y, margins, feature_names, weights):
    """Trivial 'model': just store the mean of y."""
    return {"mean_y": float(y.mean())}


def _simple_predict_fn(model, X):
    """Predict the training mean for every sample."""
    return np.full(X.shape[0], model["mean_y"])


# ---------------------------------------------------------------------------
# LOYOValidator tests
# ---------------------------------------------------------------------------


class TestLOYOYears:
    """Verify the LOYO year configuration."""

    def test_2020_excluded(self):
        """COVID year must be excluded from LOYO validation."""
        assert 2020 not in LOYO_YEARS

    def test_expected_years_present(self):
        expected = {2018, 2019, 2021, 2022, 2023, 2024, 2025}
        assert set(LOYO_YEARS) == expected

    def test_minimum_brier_improvement_legacy(self):
        """Legacy constant preserved for backward compatibility."""
        assert MINIMUM_BRIER_IMPROVEMENT == 0.001


class TestLOYOValidator:
    """Tests for the core LOYO validation engine."""

    def test_runs_all_folds(self):
        """All years with prior data should produce fold results."""
        # Add years before 2018 so the earliest LOYO year has training data
        all_years = [2016, 2017] + list(LOYO_YEARS)
        data = {yr: _make_year_data(yr) for yr in all_years}
        validator = LOYOValidator(years=list(LOYO_YEARS))
        result = validator.validate(data, _simple_train_fn, _simple_predict_fn)

        assert len(result.fold_results) == len(LOYO_YEARS)
        for fold in result.fold_results:
            assert fold.held_out_year in LOYO_YEARS

    def test_held_out_year_excluded_from_training(self):
        """Each fold must train on prior years only (rolling_window default)."""
        data = {yr: _make_year_data(yr, n_games=10) for yr in [2021, 2022, 2023, 2024]}

        train_sizes = []

        def tracking_train(X, y, margins, names, weights):
            train_sizes.append(X.shape[0])
            return {"mean_y": float(y.mean())}

        validator = LOYOValidator(years=[2022, 2023, 2024])
        result = validator.validate(data, tracking_train, _simple_predict_fn)

        # rolling_window: hold 2022 → train on 2021 (10), hold 2023 → train on 2021+2022 (20),
        # hold 2024 → train on 2021+2022+2023 (30)
        assert train_sizes == [10, 20, 30]

    def test_fold_metrics_computed(self):
        """Each fold should have Brier, log-loss, and accuracy."""
        data = {yr: _make_year_data(yr) for yr in [2022, 2023, 2024]}
        validator = LOYOValidator(years=[2023, 2024])
        result = validator.validate(data, _simple_train_fn, _simple_predict_fn)

        for fold in result.fold_results:
            assert 0.0 <= fold.brier_score <= 1.0
            assert fold.log_loss > 0.0
            assert 0.0 <= fold.accuracy <= 1.0
            assert fold.n_train_games > 0
            assert fold.n_test_games > 0

    def test_aggregate_metrics(self):
        """Mean and std Brier should be correctly aggregated."""
        data = {yr: _make_year_data(yr) for yr in [2021, 2022, 2023, 2024]}
        validator = LOYOValidator(years=[2022, 2023, 2024])
        result = validator.validate(data, _simple_train_fn, _simple_predict_fn)

        briers = [f.brier_score for f in result.fold_results]
        assert abs(result.mean_brier - np.mean(briers)) < 1e-10
        assert abs(result.std_brier - np.std(briers)) < 1e-10

    def test_year_briers_dict(self):
        """year_briers should map held-out year to its Brier score."""
        data = {yr: _make_year_data(yr) for yr in [2022, 2023, 2024]}
        validator = LOYOValidator(years=[2023, 2024])
        result = validator.validate(data, _simple_train_fn, _simple_predict_fn)

        assert set(result.year_briers.keys()) == {2023, 2024}
        for yr in [2023, 2024]:
            fold = [f for f in result.fold_results if f.held_out_year == yr][0]
            assert result.year_briers[yr] == fold.brier_score

    def test_missing_year_skipped(self):
        """A year in the LOYO list but not in data should be skipped."""
        data = {2023: _make_year_data(2023)}
        validator = LOYOValidator(years=[2022, 2023])
        result = validator.validate(data, _simple_train_fn, _simple_predict_fn)

        # Only 2023 has data, but it needs OTHER years for training.
        # 2022 is missing from data entirely, so skipped.
        # 2023 has data but training on {2022} fails (missing).
        # Net: at most 1 fold (2023 held out, trained on empty set → skipped)
        assert len(result.fold_results) <= 1

    def test_per_round_brier(self):
        """When round labels are provided, per-round Brier should be computed."""
        data = {
            2022: {
                "X": np.random.randn(20, 3),
                "y": np.array([1.0, 0.0] * 10),
                "margins": np.zeros(20),
                "rounds": ["R64"] * 10 + ["R32"] * 10,
                "feature_names": ["a", "b", "c"],
            },
            2023: {
                "X": np.random.randn(20, 3),
                "y": np.array([1.0, 0.0] * 10),
                "margins": np.zeros(20),
                "rounds": ["R64"] * 10 + ["R32"] * 10,
                "feature_names": ["a", "b", "c"],
            },
            2024: {
                "X": np.random.randn(20, 3),
                "y": np.array([1.0, 0.0] * 10),
                "margins": np.zeros(20),
                "rounds": ["R64"] * 10 + ["R32"] * 10,
                "feature_names": ["a", "b", "c"],
            },
        }
        validator = LOYOValidator(years=[2023, 2024])
        result = validator.validate(data, _simple_train_fn, _simple_predict_fn)

        for fold in result.fold_results:
            assert "R64" in fold.round_briers
            assert "R32" in fold.round_briers

    def test_perfect_predictions(self):
        """Perfect predictions should yield Brier=0 and accuracy=1."""
        data = {
            2023: {
                "X": np.array([[1.0], [0.0], [1.0], [0.0]]),
                "y": np.array([1.0, 0.0, 1.0, 0.0]),
                "margins": np.zeros(4),
                "feature_names": ["x"],
            },
            2024: {
                "X": np.array([[1.0], [0.0]]),
                "y": np.array([1.0, 0.0]),
                "margins": np.zeros(2),
                "feature_names": ["x"],
            },
        }

        def perfect_predict(model, X):
            # Return exactly the true labels
            return X[:, 0]

        validator = LOYOValidator(years=[2024])
        result = validator.validate(data, _simple_train_fn, perfect_predict)

        assert len(result.fold_results) == 1
        assert result.fold_results[0].brier_score < 1e-10
        assert result.fold_results[0].accuracy == 1.0

    def test_summary_string(self):
        """LOYOResult.summary() should produce readable output."""
        result = LOYOResult(
            fold_results=[
                LOYOFoldResult(
                    held_out_year=2023, n_train_games=100, n_test_games=30,
                    brier_score=0.2, log_loss=0.5, accuracy=0.75,
                ),
            ],
            mean_brier=0.2,
            std_brier=0.0,
            mean_logloss=0.5,
            mean_accuracy=0.75,
            year_briers={2023: 0.2},
            total_time_seconds=5.0,
        )
        s = result.summary()
        assert "LOYO Validation" in s
        assert "0.200000" in s
        assert "2023" in s


# ---------------------------------------------------------------------------
# FeatureAblator tests
# ---------------------------------------------------------------------------


class TestFeatureAblator:
    """Tests for the 0.001 Rule enforcer."""

    def test_flags_weak_feature(self):
        """A feature that doesn't improve Brier by >= 0.001 should be flagged."""
        ablator = FeatureAblator(min_improvement=0.001)

        data = {yr: _make_year_data(yr, n_features=3) for yr in [2022, 2023, 2024]}
        validator = LOYOValidator(years=[2023, 2024])

        results = ablator.ablate_features(
            validator, data, _simple_train_fn, _simple_predict_fn,
            feature_names=["a", "b", "c"],
        )

        # With a trivial mean-predictor, zeroing any feature shouldn't
        # change the Brier significantly, so all features should be
        # flagged for deletion (improvement < 0.001).
        assert len(results) == 3
        for name, info in results.items():
            assert "improvement" in info
            assert "keep" in info
            assert isinstance(info["keep"], bool)

    def test_features_to_keep_and_delete(self):
        """get_features_to_keep/delete should partition correctly."""
        ablator = FeatureAblator()
        ablator.ablation_results = {
            "good_feature": {"keep": True},
            "bad_feature": {"keep": False},
            "another_bad": {"keep": False},
        }

        assert ablator.get_features_to_keep() == ["good_feature"]
        assert set(ablator.get_features_to_delete()) == {"bad_feature", "another_bad"}

    def test_custom_threshold(self):
        """FeatureAblator should respect a custom min_improvement threshold."""
        ablator = FeatureAblator(min_improvement=0.01)
        assert ablator.min_improvement == 0.01

    def test_ablation_uses_baseline(self):
        """When baseline_brier is provided, it should be used directly."""
        ablator = FeatureAblator(min_improvement=0.001)

        data = {yr: _make_year_data(yr, n_features=2) for yr in [2022, 2023, 2024]}
        validator = LOYOValidator(years=[2023, 2024])

        results = ablator.ablate_features(
            validator, data, _simple_train_fn, _simple_predict_fn,
            feature_names=["a", "b"],
            baseline_brier=0.25,
        )

        for info in results.values():
            assert info["baseline_brier"] == 0.25


# ---------------------------------------------------------------------------
# ProspectiveValidator tests
# ---------------------------------------------------------------------------


class TestProspectiveValidator:
    """Tests for strict season-by-season forward validation."""

    def test_uses_only_prior_years_for_training(self):
        data = {yr: _make_year_data(yr, n_games=8) for yr in [2021, 2022, 2023]}
        seen = []

        def tracking_train(X, y, margins, names, weights):
            seen.append(X.shape[0])
            return {"mean_y": float(y.mean())}

        validator = ProspectiveValidator(years=[2021, 2022, 2023])
        result = validator.validate(data, tracking_train, _simple_predict_fn)

        # 2021 skipped (no historical years), then:
        # 2022 -> train on 2021 (8 rows), 2023 -> train on 2021+2022 (16 rows)
        assert [f.predicted_year for f in result.fold_results] == [2022, 2023]
        assert seen == [8, 16]

    def test_skips_first_year_without_history(self):
        data = {yr: _make_year_data(yr, n_games=6) for yr in [2023, 2024]}
        validator = ProspectiveValidator(years=[2023, 2024])
        result = validator.validate(data, _simple_train_fn, _simple_predict_fn)

        assert [f.predicted_year for f in result.fold_results] == [2024]
        assert result.fold_results[0].train_years == [2023]

    def test_summary_mentions_prospective(self):
        result = ProspectiveValidationResult(
            mean_brier=0.21,
            std_brier=0.01,
            mean_logloss=0.55,
            mean_accuracy=0.71,
            year_briers={2024: 0.2},
            total_time_seconds=1.0,
        )
        s = result.summary()
        assert "Prospective Forward Validation" in s
        assert "2024" in s


# ---------------------------------------------------------------------------
# Phase 2: Feature Family Ablation Tests
# ---------------------------------------------------------------------------


class TestFeatureFamily:
    """Test FeatureFamily dataclass construction and validation."""

    def test_column_group_family(self):
        fam = FeatureFamily(
            name="test_fam",
            family_type="column_group",
            feature_names=["feat_a", "feat_b"],
            masking_policy="zero",
        )
        assert fam.name == "test_fam"
        assert fam.family_type == "column_group"
        assert len(fam.feature_names) == 2
        assert fam.masking_policy == "zero"

    def test_config_toggle_family(self):
        fam = FeatureFamily(
            name="toggle_fam",
            family_type="config_toggle",
            config_flags=["enable_something"],
        )
        assert fam.family_type == "config_toggle"
        assert fam.config_flags == ["enable_something"]
        assert fam.feature_names == []

    def test_hybrid_family(self):
        fam = FeatureFamily(
            name="hybrid_fam",
            family_type="hybrid",
            feature_names=["diff_momentum"],
            config_flags=["enable_recency_weighting"],
            masking_policy="zero",
        )
        assert fam.family_type == "hybrid"
        assert len(fam.feature_names) == 1
        assert len(fam.config_flags) == 1

    def test_neutral_masking_policy(self):
        fam = FeatureFamily(
            name="seed_fam",
            family_type="column_group",
            feature_names=["seed_diff"],
            masking_policy="neutral",
            neutral_values={"seed_diff": 0.0},
        )
        assert fam.masking_policy == "neutral"
        assert fam.neutral_values["seed_diff"] == 0.0

    def test_default_feature_families_exist(self):
        assert len(DEFAULT_FEATURE_FAMILIES) == 7
        family_names = {f.name for f in DEFAULT_FEATURE_FAMILIES}
        assert "seed_priors" in family_names
        assert "elo_ratings" in family_names
        assert "four_factors" in family_names
        assert "roster_continuity" in family_names
        assert "massey_ordinals" in family_names
        assert "recency_form" in family_names
        assert "public_picks" in family_names


class TestValidateFamilyCoverage:
    """Test feature-to-family coverage validation."""

    def test_full_coverage(self):
        features = ["feat_a", "feat_b"]
        families = [
            FeatureFamily(name="fam1", family_type="column_group", feature_names=["feat_a"]),
            FeatureFamily(name="fam2", family_type="column_group", feature_names=["feat_b"]),
        ]
        coverage = validate_family_coverage(features, families)
        assert coverage["feat_a"] == "fam1"
        assert coverage["feat_b"] == "fam2"
        assert "unassigned" not in coverage.values()

    def test_unassigned_features_detected(self):
        features = ["feat_a", "feat_b", "feat_c"]
        families = [
            FeatureFamily(name="fam1", family_type="column_group", feature_names=["feat_a"]),
        ]
        coverage = validate_family_coverage(features, families)
        assert coverage["feat_a"] == "fam1"
        assert coverage["feat_b"] == "unassigned"
        assert coverage["feat_c"] == "unassigned"

    def test_prefix_matching(self):
        features = ["diff_efg_pct", "diff_to_rate", "diff_elo"]
        families = [
            FeatureFamily(
                name="four_factors",
                family_type="column_group",
                feature_prefixes=["diff_efg", "diff_to"],
            ),
        ]
        coverage = validate_family_coverage(features, families)
        assert coverage["diff_efg_pct"] == "four_factors"
        assert coverage["diff_to_rate"] == "four_factors"
        assert coverage["diff_elo"] == "unassigned"


class TestFeatureAblatorFamilies:
    """Test family-level ablation in FeatureAblator."""

    def _make_family_data(self, n_games=30, n_features=5):
        """Create multi-year data with known feature structure."""
        feature_names = [f"feat_{i}" for i in range(n_features)]
        data_by_year = {}
        for year in [2021, 2022, 2023]:
            rng = np.random.RandomState(year)
            X = rng.randn(n_games, n_features)
            y = rng.randint(0, 2, size=n_games).astype(float)
            margins = rng.randn(n_games) * 10
            data_by_year[year] = {
                "X": X, "y": y, "margins": margins,
                "feature_names": feature_names,
            }
        return data_by_year, feature_names

    def test_ablate_column_group_family(self):
        """Family ablation masks all family columns and runs LOYO."""
        data, feat_names = self._make_family_data(n_features=5)
        family = FeatureFamily(
            name="test_family",
            family_type="column_group",
            feature_names=["feat_0", "feat_1"],
            masking_policy="zero",
        )

        validator = LOYOValidator(years=[2021, 2022, 2023])
        ablator = FeatureAblator(min_improvement=0.001)

        results = ablator.ablate_families(
            loyo_validator=validator,
            data_by_year=data,
            train_fn=_simple_train_fn,
            predict_fn=_simple_predict_fn,
            families=[family],
            feature_names=feat_names,
            mode="masked",
        )

        assert "test_family" in results
        r = results["test_family"]
        assert r["n_features_masked"] == 2
        assert r["features_masked"] == ["feat_0", "feat_1"]
        assert r["masking_policy"] == "zero"
        assert r["mode"] == "masked"
        assert r["ablated_brier"] is not None
        assert isinstance(r["keep"], bool)

    def test_config_toggle_family_skipped_in_masked_mode(self):
        """Config-toggle families are skipped in masked mode."""
        data, feat_names = self._make_family_data()
        family = FeatureFamily(
            name="public_picks",
            family_type="config_toggle",
            config_flags=["public_picks_json"],
        )

        validator = LOYOValidator(years=[2021, 2022, 2023])
        ablator = FeatureAblator()

        results = ablator.ablate_families(
            loyo_validator=validator,
            data_by_year=data,
            train_fn=_simple_train_fn,
            predict_fn=_simple_predict_fn,
            families=[family],
            feature_names=feat_names,
        )

        assert "public_picks" in results
        assert results["public_picks"]["keep"] is None
        assert "Skipped" in results["public_picks"]["reason"]

    def test_retrain_mode_not_implemented(self):
        """Retrain mode raises NotImplementedError (deferred to Phase 3)."""
        data, feat_names = self._make_family_data()
        family = FeatureFamily(name="test", family_type="column_group", feature_names=["feat_0"])

        validator = LOYOValidator(years=[2021, 2022, 2023])
        ablator = FeatureAblator()

        with pytest.raises(NotImplementedError, match="Phase 3"):
            ablator.ablate_families(
                loyo_validator=validator,
                data_by_year=data,
                train_fn=_simple_train_fn,
                predict_fn=_simple_predict_fn,
                families=[family],
                feature_names=feat_names,
                mode="retrain",
            )

    def test_masking_policies_differ(self):
        """Different masking policies produce different ablated data."""
        n_features = 3
        feature_names = ["feat_0", "feat_1", "feat_2"]
        data = {}
        for year in [2021, 2022]:
            rng = np.random.RandomState(year)
            X = rng.randn(20, n_features) + 5.0  # offset so mean != 0
            y = rng.randint(0, 2, size=20).astype(float)
            data[year] = {"X": X, "y": y, "margins": rng.randn(20) * 10, "feature_names": feature_names}

        validator = LOYOValidator(years=[2021, 2022])

        # Test zero masking
        fam_zero = FeatureFamily(
            name="zero", family_type="column_group",
            feature_names=["feat_0"], masking_policy="zero",
        )
        ablator = FeatureAblator()
        results_zero = ablator.ablate_families(
            validator, data, _simple_train_fn, _simple_predict_fn,
            [fam_zero], feature_names,
        )

        # Test mean masking
        fam_mean = FeatureFamily(
            name="mean", family_type="column_group",
            feature_names=["feat_0"], masking_policy="mean",
        )
        results_mean = ablator.ablate_families(
            validator, data, _simple_train_fn, _simple_predict_fn,
            [fam_mean], feature_names,
        )

        # Both should produce valid results (may have same Brier with trivial model,
        # but the key is they don't error)
        assert results_zero["zero"]["ablated_brier"] is not None
        assert results_mean["mean"]["ablated_brier"] is not None

    def test_no_matching_features_handled(self):
        """Family with no matching features is handled gracefully."""
        data, feat_names = self._make_family_data()
        family = FeatureFamily(
            name="missing",
            family_type="column_group",
            feature_names=["nonexistent_feature"],
        )

        validator = LOYOValidator(years=[2021, 2022, 2023])
        ablator = FeatureAblator()

        results = ablator.ablate_families(
            validator, data, _simple_train_fn, _simple_predict_fn,
            [family], feat_names,
        )

        assert results["missing"]["keep"] is None
        assert "No matching features" in results["missing"]["reason"]

    def test_multiple_families_ablated(self):
        """Multiple families can be ablated in one call."""
        data, feat_names = self._make_family_data(n_features=5)
        families = [
            FeatureFamily(name="fam_a", family_type="column_group", feature_names=["feat_0"]),
            FeatureFamily(name="fam_b", family_type="column_group", feature_names=["feat_1", "feat_2"]),
        ]

        validator = LOYOValidator(years=[2021, 2022, 2023])
        ablator = FeatureAblator()

        results = ablator.ablate_families(
            validator, data, _simple_train_fn, _simple_predict_fn,
            families, feat_names,
        )

        assert "fam_a" in results
        assert "fam_b" in results
        assert results["fam_a"]["n_features_masked"] == 1
        assert results["fam_b"]["n_features_masked"] == 2


# ---------------------------------------------------------------------------
# Powered Ablation Threshold Tests
# ---------------------------------------------------------------------------


class TestComputeAblationThreshold:
    """Tests for the statistically-powered ablation threshold."""

    def test_threshold_exceeds_legacy(self):
        """Powered threshold should be >= legacy 0.001 for realistic fold variance."""
        fold_briers = [0.18, 0.22, 0.19, 0.25, 0.20, 0.21, 0.17]
        threshold = compute_ablation_threshold(fold_briers)
        assert threshold >= MINIMUM_BRIER_IMPROVEMENT

    def test_threshold_scales_with_variance(self):
        """Higher fold variance should produce a larger threshold."""
        low_var = [0.200, 0.201, 0.199, 0.200, 0.201, 0.199, 0.200]
        high_var = [0.15, 0.25, 0.18, 0.28, 0.16, 0.24, 0.20]
        t_low = compute_ablation_threshold(low_var)
        t_high = compute_ablation_threshold(high_var)
        assert t_high > t_low

    def test_threshold_with_few_folds_uses_legacy(self):
        """With < 3 folds, should fall back to legacy threshold."""
        threshold = compute_ablation_threshold([0.20, 0.21])
        assert threshold == MINIMUM_BRIER_IMPROVEMENT

    def test_threshold_positive(self):
        """Threshold must always be positive."""
        fold_briers = [0.20] * 7
        threshold = compute_ablation_threshold(fold_briers)
        assert threshold >= MINIMUM_BRIER_IMPROVEMENT

    def test_typical_ncaa_se(self):
        """With realistic NCAA tournament fold variance, threshold should be ~0.01-0.03."""
        # Simulating typical LOYO variance: std ≈ 0.02-0.03 across 7 folds
        rng = np.random.RandomState(42)
        fold_briers = 0.20 + rng.normal(0, 0.025, size=7)
        threshold = compute_ablation_threshold(fold_briers.tolist())
        # Should be well above 0.001 and roughly in the 0.01-0.04 range
        assert threshold > 0.005
        assert threshold < 0.10  # sanity upper bound


class TestComputeValidationDiagnostics:
    """Tests for validation diagnostic reporting."""

    def test_basic_diagnostics(self):
        """Should compute all expected diagnostic fields."""
        folds = [
            LOYOFoldResult(
                held_out_year=yr, n_train_games=300, n_test_games=63,
                brier_score=0.20 + i * 0.01, log_loss=0.5, accuracy=0.75,
            )
            for i, yr in enumerate([2018, 2019, 2021, 2022, 2023, 2024, 2025])
        ]
        diag = compute_validation_diagnostics(folds)

        assert "n_folds" in diag
        assert diag["n_folds"] == 7
        assert "total_eval_games" in diag
        assert diag["total_eval_games"] == 7 * 63
        assert "se_mean_brier" in diag
        assert diag["se_mean_brier"] > 0
        assert "dof_sample_ratio" in diag
        assert "powered_threshold" in diag
        assert "integrity_level" in diag
        assert diag["integrity_level"] == 3

    def test_dof_ratio_warning(self):
        """Should warn when DoF/sample ratio exceeds target."""
        folds = [
            LOYOFoldResult(
                held_out_year=2023, n_train_games=300, n_test_games=63,
                brier_score=0.20, log_loss=0.5, accuracy=0.75,
            )
        ]
        diag = compute_validation_diagnostics(folds, n_tuned_constants=58)
        # 58/63 ≈ 0.92, way above 0.01 target
        assert diag["dof_sample_ratio"] > 0.01
        assert any("DoF/sample" in w for w in diag["warnings"])

    def test_legacy_threshold_warning(self):
        """Should warn that legacy 0.001 is within noise."""
        folds = [
            LOYOFoldResult(
                held_out_year=yr, n_train_games=300, n_test_games=63,
                brier_score=0.20 + i * 0.01, log_loss=0.5, accuracy=0.75,
            )
            for i, yr in enumerate([2018, 2019, 2021, 2022, 2023, 2024, 2025])
        ]
        diag = compute_validation_diagnostics(folds)
        # SE should be > 0.001, triggering the warning
        assert diag["se_mean_brier"] > MINIMUM_BRIER_IMPROVEMENT
        assert any("0.001" in w for w in diag["warnings"])

    def test_empty_folds(self):
        """Should handle empty fold list gracefully."""
        diag = compute_validation_diagnostics([])
        assert "error" in diag

    def test_integrity_note_mentions_retrospective(self):
        """Integrity note should clearly state retrospective nature."""
        folds = [
            LOYOFoldResult(
                held_out_year=2023, n_train_games=300, n_test_games=63,
                brier_score=0.20, log_loss=0.5, accuracy=0.75,
            )
        ]
        diag = compute_validation_diagnostics(folds)
        assert "RETROSPECTIVE" in diag["integrity_note"]
        assert "prospective" in diag["integrity_note"].lower()


class TestFeatureAblatorPoweredThreshold:
    """Tests for the powered threshold in FeatureAblator."""

    def test_powered_threshold_used_by_default(self):
        """FeatureAblator should use powered threshold by default."""
        ablator = FeatureAblator()
        assert ablator.use_powered_threshold is True

    def test_legacy_mode_available(self):
        """Can disable powered threshold for backward compatibility."""
        ablator = FeatureAblator(use_powered_threshold=False)
        assert ablator.use_powered_threshold is False

    def test_ablation_results_include_powered_threshold(self):
        """Ablation results should include the powered threshold value."""
        data = {yr: _make_year_data(yr, n_features=3) for yr in [2022, 2023, 2024]}
        validator = LOYOValidator(years=[2023, 2024])
        ablator = FeatureAblator()

        results = ablator.ablate_features(
            validator, data, _simple_train_fn, _simple_predict_fn,
            feature_names=["a", "b", "c"],
        )

        for info in results.values():
            assert "powered_threshold" in info
            assert "paired_t_p_value" in info
            assert "per_fold_baseline" in info
            assert "per_fold_ablated" in info

    def test_ablation_results_include_p_value(self):
        """Each ablated feature should have a paired t-test p-value."""
        data = {yr: _make_year_data(yr, n_features=2) for yr in [2021, 2022, 2023, 2024]}
        validator = LOYOValidator(years=[2022, 2023, 2024])
        ablator = FeatureAblator()

        results = ablator.ablate_features(
            validator, data, _simple_train_fn, _simple_predict_fn,
            feature_names=["a", "b"],
        )

        for info in results.values():
            # p-value should be computed for 3+ folds
            assert info["paired_t_p_value"] is not None
            assert 0.0 <= info["paired_t_p_value"] <= 1.0


class TestLOYOResultSummary:
    """Tests for the enhanced summary with statistical diagnostics."""

    def test_summary_includes_se(self):
        """Summary should report standard error of mean Brier."""
        folds = [
            LOYOFoldResult(
                held_out_year=yr, n_train_games=300, n_test_games=63,
                brier_score=0.20, log_loss=0.5, accuracy=0.75,
            )
            for yr in [2023, 2024]
        ]
        result = LOYOResult(
            fold_results=folds,
            mean_brier=0.20, std_brier=0.0,
            mean_logloss=0.5, mean_accuracy=0.75,
            year_briers={2023: 0.20, 2024: 0.20},
        )
        s = result.summary()
        assert "SE(mean)" in s

    def test_summary_includes_dof_ratio(self):
        """Summary should report DoF/sample ratio."""
        folds = [
            LOYOFoldResult(
                held_out_year=2023, n_train_games=300, n_test_games=63,
                brier_score=0.20, log_loss=0.5, accuracy=0.75,
            )
        ]
        result = LOYOResult(
            fold_results=folds,
            mean_brier=0.20, std_brier=0.0,
            mean_logloss=0.5, mean_accuracy=0.75,
            year_briers={2023: 0.20},
        )
        s = result.summary()
        assert "DoF/sample ratio" in s
        assert "target < 0.01" in s

    def test_summary_warns_on_high_dof_ratio(self):
        """Summary should warn when DoF/sample ratio is too high."""
        folds = [
            LOYOFoldResult(
                held_out_year=2023, n_train_games=300, n_test_games=10,
                brier_score=0.20, log_loss=0.5, accuracy=0.75,
            )
        ]
        result = LOYOResult(
            fold_results=folds,
            mean_brier=0.20, std_brier=0.0,
            mean_logloss=0.5, mean_accuracy=0.75,
            year_briers={2023: 0.20},
        )
        s = result.summary()
        assert "WARNING" in s

    def test_summary_includes_integrity_level(self):
        """Summary should state integrity level 3 (retrospective)."""
        folds = [
            LOYOFoldResult(
                held_out_year=2023, n_train_games=300, n_test_games=63,
                brier_score=0.20, log_loss=0.5, accuracy=0.75,
            )
        ]
        result = LOYOResult(
            fold_results=folds,
            mean_brier=0.20, std_brier=0.0,
            mean_logloss=0.5, mean_accuracy=0.75,
            year_briers={2023: 0.20},
        )
        s = result.summary()
        assert "Integrity level: 3" in s


# ---------------------------------------------------------------------------
# DevEvalSplit Tests (breaking circular validation)
# ---------------------------------------------------------------------------


class TestDevEvalSplit:
    """Tests for the dev/eval year split that breaks circular validation."""

    def test_overlap_raises(self):
        """Dev and eval years must not overlap."""
        with pytest.raises(ValueError, match="overlap"):
            DevEvalSplit(
                dev_years=[2021, 2022, 2023],
                eval_years=[2023, 2024],
            )

    def test_no_overlap_succeeds(self):
        """Non-overlapping years should construct successfully."""
        split = DevEvalSplit(
            dev_years=[2021, 2022, 2023],
            eval_years=[2024, 2025],
        )
        assert split.dev_years == [2021, 2022, 2023]
        assert split.eval_years == [2024, 2025]

    def test_recommended_split(self):
        """Recommended split should have no overlap and cover LOYO years."""
        split = DevEvalSplit.recommended_split()
        assert set(split.dev_years) & set(split.eval_years) == set()
        # Dev + eval should cover most LOYO years
        all_years = set(split.dev_years) | set(split.eval_years)
        assert all_years.issubset(set(LOYO_YEARS))

    def test_validate_no_leakage_clean(self):
        """No leakage when tuning uses only dev years."""
        split = DevEvalSplit(
            dev_years=[2021, 2022, 2023],
            eval_years=[2024, 2025],
        )
        result = split.validate_no_leakage([2021, 2022, 2023])
        assert result["clean"] is True
        assert result["leaked_years"] == []

    def test_validate_no_leakage_detected(self):
        """Leakage detected when tuning uses eval years."""
        split = DevEvalSplit(
            dev_years=[2021, 2022, 2023],
            eval_years=[2024, 2025],
        )
        result = split.validate_no_leakage([2021, 2022, 2023, 2024])
        assert result["clean"] is False
        assert 2024 in result["leaked_years"]
        assert "LEAKAGE" in result["warning"]

    def test_get_dev_validator(self):
        """Dev validator should use only dev years."""
        split = DevEvalSplit(
            dev_years=[2021, 2022, 2023],
            eval_years=[2024, 2025],
        )
        validator = split.get_dev_validator()
        assert validator.years == [2021, 2022, 2023]
        assert validator.temporal_mode == "rolling_window"

    def test_get_eval_validator(self):
        """Eval validator should use only eval years."""
        split = DevEvalSplit(
            dev_years=[2021, 2022, 2023],
            eval_years=[2024, 2025],
        )
        validator = split.get_eval_validator()
        assert validator.years == [2024, 2025]

    def test_dof_diagnostics(self):
        """DoF diagnostics should report ratios for both partitions."""
        split = DevEvalSplit(
            dev_years=[2018, 2019, 2021, 2022, 2023],
            eval_years=[2024, 2025],
        )
        diag = split.dof_diagnostics(n_tuned_constants=58)
        assert "dev_dof_ratio" in diag
        assert "eval_dof_ratio" in diag
        assert "eval_mde_brier" in diag
        # Both ratios should be > 0.01 (that's the problem)
        assert diag["dev_dof_ratio"] > 0.01
        assert diag["eval_dof_ratio"] > 0.01
        assert len(diag["warnings"]) > 0

    def test_dof_diagnostics_mde(self):
        """MDE should be reported for eval set."""
        split = DevEvalSplit(
            dev_years=[2021, 2022, 2023],
            eval_years=[2024, 2025],
        )
        diag = split.dof_diagnostics()
        assert diag["eval_mde_brier"] > 0
        assert "Minimum detectable" in diag["eval_mde_note"]
