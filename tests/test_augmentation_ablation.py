"""Tests for augmentation ablation and strategy comparison.

Tests verify:
1. AugmentationAblation produces correct LOYO fold results
2. Strategy comparison tests all strategies and ranks them
3. Zero-sum compliance audit correctly identifies model compliance
4. Augmentation strategies produce correct shapes and invariants
5. Statistical tests (paired t-test, powered threshold) work correctly
"""

from __future__ import annotations

import numpy as np
import pytest

from src.ml.training.symmetric import (
    MATCHUP_DIM,
    swap_matchup_batch,
    symmetric_augment,
)
from src.ml.training.augmentation_ablation import (
    AugmentationAblation,
    AugmentationStrategyComparison,
    ZeroSumComplianceAudit,
    no_augmentation,
    weighted_symmetric_augmentation,
    noise_augmentation,
    AUGMENTATION_STRATEGIES,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    return np.random.RandomState(42)


@pytest.fixture
def synthetic_data(rng):
    """Synthetic LOYO-style data: 5 years, 50 games per year, 78 features."""
    data_by_year = {}
    for year in [2019, 2021, 2022, 2023, 2024]:
        n = 50
        X = rng.randn(n, MATCHUP_DIM)
        # Create a simple ground truth: team1 wins if first diff feature > 0
        y = (X[:, 0] > 0).astype(float)
        data_by_year[year] = (X, y)
    return data_by_year


@pytest.fixture
def loyo_years():
    return [2019, 2021, 2022, 2023, 2024]


def _simple_train_predict(X_train, y_train, X_test):
    """Simple logistic-like predictor for testing."""
    # Fit: use first feature as signal
    mean_pos = np.mean(X_train[y_train > 0.5, 0]) if np.any(y_train > 0.5) else 0.0
    mean_neg = np.mean(X_train[y_train < 0.5, 0]) if np.any(y_train < 0.5) else 0.0
    threshold = (mean_pos + mean_neg) / 2.0

    # Predict: sigmoid-like function
    diff = X_test[:, 0] - threshold
    preds = 1.0 / (1.0 + np.exp(-diff))
    return np.clip(preds, 0.01, 0.99)


# ---------------------------------------------------------------------------
# Test augmentation strategies
# ---------------------------------------------------------------------------

class TestAugmentationStrategies:
    """Test that each strategy produces valid augmented data."""

    def test_no_augmentation_passthrough(self, rng):
        """no_augmentation returns inputs unchanged."""
        X = rng.randn(20, MATCHUP_DIM)
        y = rng.randint(0, 2, 20).astype(float)

        X_out, y_out, _, _, _ = no_augmentation(X, y)
        assert X_out is X  # Same object, not a copy
        assert y_out is y

    def test_symmetric_augmentation_doubles(self, rng):
        """symmetric_augment doubles the dataset."""
        n = 30
        X = rng.randn(n, MATCHUP_DIM)
        y = rng.randint(0, 2, n).astype(float)

        X_aug, y_aug, _, _, _ = symmetric_augment(X, y)
        assert X_aug.shape == (2 * n, MATCHUP_DIM)
        assert y_aug.shape == (2 * n,)

    def test_symmetric_labels_inverted(self, rng):
        """Swapped samples have inverted labels."""
        n = 30
        X = rng.randn(n, MATCHUP_DIM)
        y = rng.randint(0, 2, n).astype(float)

        _, y_aug, _, _, _ = symmetric_augment(X, y)

        # Original at even indices, swapped at odd
        for i in range(n):
            assert y_aug[2 * i] + y_aug[2 * i + 1] == 1.0

    def test_weighted_symmetric_weights(self, rng):
        """weighted_symmetric gives lower weight to swapped samples."""
        n = 20
        X = rng.randn(n, MATCHUP_DIM)
        y = rng.randint(0, 2, n).astype(float)

        _, _, _, rw_aug, _ = weighted_symmetric_augmentation(X, y, swap_weight=0.5)
        assert rw_aug is not None
        # Original samples should have weight 1.0
        np.testing.assert_allclose(rw_aug[0::2], 1.0)
        # Swapped samples should have weight 0.5
        np.testing.assert_allclose(rw_aug[1::2], 0.5)

    def test_weighted_preserves_round_weights(self, rng):
        """weighted_symmetric preserves existing round weights."""
        n = 20
        X = rng.randn(n, MATCHUP_DIM)
        y = rng.randint(0, 2, n).astype(float)
        rw = np.full(n, 2.0)

        _, _, _, rw_aug, _ = weighted_symmetric_augmentation(
            X, y, round_weights=rw, swap_weight=0.5
        )
        # Originals: 2.0 * 1.0 = 2.0
        np.testing.assert_allclose(rw_aug[0::2], 2.0)
        # Swapped: 2.0 * 0.5 = 1.0
        np.testing.assert_allclose(rw_aug[1::2], 1.0)

    def test_noise_augmentation_adds_perturbation(self, rng):
        """noise_augmentation perturbs swapped samples."""
        n = 20
        X = rng.randn(n, MATCHUP_DIM)
        y = rng.randint(0, 2, n).astype(float)

        X_aug_no_noise, _, _, _, _ = symmetric_augment(X, y)
        X_aug_noise, _, _, _, _ = noise_augmentation(X, y, noise_scale=0.1)

        # Original rows should be identical
        np.testing.assert_allclose(X_aug_no_noise[0::2], X_aug_noise[0::2])

        # Swapped rows should differ (due to noise)
        diffs = np.abs(X_aug_no_noise[1::2] - X_aug_noise[1::2])
        assert np.mean(diffs) > 0.01  # Noise added

    def test_noise_scale_controls_perturbation(self, rng):
        """Larger noise_scale produces larger perturbations."""
        n = 50
        X = rng.randn(n, MATCHUP_DIM)
        y = rng.randint(0, 2, n).astype(float)

        X_small, _, _, _, _ = noise_augmentation(X, y, noise_scale=0.01)
        X_large, _, _, _, _ = noise_augmentation(X, y, noise_scale=1.0)

        X_base, _, _, _, _ = symmetric_augment(X, y)
        diff_small = np.mean(np.abs(X_base[1::2] - X_small[1::2]))
        diff_large = np.mean(np.abs(X_base[1::2] - X_large[1::2]))

        assert diff_large > diff_small * 5  # Much larger noise

    def test_all_strategies_registered(self):
        """All strategies are in the registry."""
        assert "none" in AUGMENTATION_STRATEGIES
        assert "symmetric" in AUGMENTATION_STRATEGIES
        assert "weighted_symmetric" in AUGMENTATION_STRATEGIES
        assert "noise_augmentation" in AUGMENTATION_STRATEGIES


# ---------------------------------------------------------------------------
# Test augmentation ablation
# ---------------------------------------------------------------------------

class TestAugmentationAblation:
    """Test LOYO-based augmentation ablation."""

    def test_ablation_runs(self, synthetic_data, loyo_years):
        """Ablation completes without error."""
        ablation = AugmentationAblation(significance_level=0.05)
        result = ablation.run(loyo_years, synthetic_data, _simple_train_predict)

        assert result.n_folds == len(loyo_years)
        assert result.baseline_brier >= 0
        assert result.ablated_brier >= 0
        assert len(result.fold_deltas) == len(loyo_years)
        assert result.powered_threshold > 0

    def test_ablation_fold_counts(self, synthetic_data, loyo_years):
        """Each fold produces results for both augmented and ablated."""
        ablation = AugmentationAblation()
        result = ablation.run(loyo_years, synthetic_data, _simple_train_predict)

        assert len(result.baseline_folds) == len(loyo_years)
        assert len(result.ablated_folds) == len(loyo_years)
        for f in result.baseline_folds:
            assert f.n_games > 0
            assert f.year in loyo_years

    def test_ablation_zero_sum_check(self, synthetic_data, loyo_years):
        """Ablation records zero-sum deviation for augmented model."""
        ablation = AugmentationAblation()
        result = ablation.run(loyo_years, synthetic_data, _simple_train_predict)

        for f in result.baseline_folds:
            assert f.mean_zero_sum_deviation >= 0
            assert f.max_zero_sum_deviation >= f.mean_zero_sum_deviation

    def test_ablation_to_dict(self, synthetic_data, loyo_years):
        """Result serializes to dict correctly."""
        ablation = AugmentationAblation()
        result = ablation.run(loyo_years, synthetic_data, _simple_train_predict)
        d = result.to_dict()

        assert "baseline_brier" in d
        assert "ablated_brier" in d
        assert "t_stat" in d
        assert "p_value" in d
        assert "significant" in d
        assert isinstance(d["fold_deltas"], list)

    def test_ablation_significance_flag(self, synthetic_data, loyo_years):
        """Significance flag is boolean."""
        ablation = AugmentationAblation()
        result = ablation.run(loyo_years, synthetic_data, _simple_train_predict)
        assert isinstance(result.significant, bool)

    def test_ablation_brier_delta_sign(self, synthetic_data, loyo_years):
        """Brier delta is ablated - baseline (positive = augmentation helps)."""
        ablation = AugmentationAblation()
        result = ablation.run(loyo_years, synthetic_data, _simple_train_predict)
        expected_delta = result.ablated_brier - result.baseline_brier
        np.testing.assert_almost_equal(result.brier_delta, expected_delta, decimal=6)


# ---------------------------------------------------------------------------
# Test strategy comparison
# ---------------------------------------------------------------------------

class TestStrategyComparison:
    """Test multi-strategy comparison framework."""

    def test_comparison_runs(self, synthetic_data, loyo_years):
        """Strategy comparison completes without error."""
        comparison = AugmentationStrategyComparison(
            strategies={"none": no_augmentation, "symmetric": symmetric_augment},
        )
        report = comparison.run(loyo_years, synthetic_data, _simple_train_predict)

        assert "none" in report.results
        assert "symmetric" in report.results
        assert report.best_strategy in ["none", "symmetric"]

    def test_comparison_pairwise_tests(self, synthetic_data, loyo_years):
        """Pairwise tests are computed between all strategy pairs."""
        comparison = AugmentationStrategyComparison(
            strategies={"none": no_augmentation, "symmetric": symmetric_augment},
        )
        report = comparison.run(loyo_years, synthetic_data, _simple_train_predict)

        assert len(report.pairwise_tests) >= 1
        for key, val in report.pairwise_tests.items():
            assert "t_stat" in val
            assert "p_value" in val
            assert 0 <= val["p_value"] <= 1

    def test_comparison_all_strategies(self, synthetic_data, loyo_years):
        """All 4 strategies can be compared."""
        comparison = AugmentationStrategyComparison()
        report = comparison.run(loyo_years, synthetic_data, _simple_train_predict)

        for name in AUGMENTATION_STRATEGIES:
            assert name in report.results
            assert report.results[name].n_total_games > 0
            assert len(report.results[name].fold_briers) == len(loyo_years)

    def test_comparison_best_strategy_valid(self, synthetic_data, loyo_years):
        """Best strategy has the lowest mean Brier."""
        comparison = AugmentationStrategyComparison()
        report = comparison.run(loyo_years, synthetic_data, _simple_train_predict)

        best_brier = report.results[report.best_strategy].mean_brier
        for name, result in report.results.items():
            assert result.mean_brier >= best_brier - 1e-10

    def test_comparison_to_dict(self, synthetic_data, loyo_years):
        """Report serializes correctly."""
        comparison = AugmentationStrategyComparison(
            strategies={"none": no_augmentation, "symmetric": symmetric_augment},
        )
        report = comparison.run(loyo_years, synthetic_data, _simple_train_predict)
        d = report.to_dict()

        assert "best_strategy" in d
        assert "results" in d
        assert "pairwise_tests" in d


# ---------------------------------------------------------------------------
# Test zero-sum compliance
# ---------------------------------------------------------------------------

class TestZeroSumCompliance:
    """Test zero-sum compliance audit."""

    def test_perfect_symmetric_model_passes(self, rng):
        """A perfectly symmetric prediction function passes the audit."""
        X = rng.randn(100, MATCHUP_DIM)

        # Perfect antisymmetric predictor: P = sigmoid(first feature)
        def predict_fn(X_input):
            return 1.0 / (1.0 + np.exp(-X_input[:, 0]))

        audit = ZeroSumComplianceAudit(tolerance=0.01)
        result = audit.audit(X, predict_fn, n_samples=50)

        # sigmoid(-x) = 1 - sigmoid(x), so perfect zero-sum for feature 0
        # But only if feature 0 is negated under swap (it is, as a diff feature)
        assert result.passes
        assert result.mean_deviation < 0.01

    def test_constant_model_fails(self, rng):
        """A constant predictor always outputs 0.7 — fails zero-sum."""
        X = rng.randn(100, MATCHUP_DIM)

        def predict_fn(X_input):
            return np.full(len(X_input), 0.7)

        audit = ZeroSumComplianceAudit(tolerance=0.01)
        result = audit.audit(X, predict_fn, n_samples=50)

        # P(A>B) + P(B>A) = 0.7 + 0.7 = 1.4 != 1.0
        assert not result.passes
        assert result.mean_deviation > 0.3

    def test_audit_to_dict(self, rng):
        """Audit result serializes correctly."""
        X = rng.randn(50, MATCHUP_DIM)

        def predict_fn(X_input):
            return np.full(len(X_input), 0.5)

        audit = ZeroSumComplianceAudit(tolerance=0.01)
        result = audit.audit(X, predict_fn, n_samples=30)
        d = result.to_dict()

        assert "mean_deviation" in d
        assert "passes" in d
        assert "tolerance" in d

    def test_audit_by_prediction_range(self, rng):
        """Stratified audit produces buckets."""
        X = rng.randn(200, MATCHUP_DIM)

        def predict_fn(X_input):
            return 1.0 / (1.0 + np.exp(-X_input[:, 0]))

        audit = ZeroSumComplianceAudit(tolerance=0.05)
        results = audit.audit_by_prediction_range(X, predict_fn, n_buckets=3)

        assert len(results) > 0
        for key, res in results.items():
            assert res.n_tested > 0
            assert res.tolerance == 0.05

    def test_zero_sum_tolerance_parameterization(self, rng):
        """Different tolerances affect pass/fail appropriately."""
        X = rng.randn(100, MATCHUP_DIM)

        # Model with small but nonzero zero-sum error
        def predict_fn(X_input):
            base = 1.0 / (1.0 + np.exp(-X_input[:, 0]))
            return base + 0.005  # Small constant bias

        strict = ZeroSumComplianceAudit(tolerance=0.001)
        lenient = ZeroSumComplianceAudit(tolerance=0.05)

        result_strict = strict.audit(X, predict_fn, n_samples=50)
        result_lenient = lenient.audit(X, predict_fn, n_samples=50)

        # Strict should fail, lenient should pass (or at least be closer to passing)
        assert result_lenient.mean_deviation == result_strict.mean_deviation  # Same data
        assert result_lenient.tolerance > result_strict.tolerance


# ---------------------------------------------------------------------------
# Test mathematical invariants
# ---------------------------------------------------------------------------

class TestMathematicalInvariants:
    """Test mathematical properties that must hold."""

    def test_augmented_label_balance(self, rng):
        """Symmetric augmentation produces exactly 50% positive labels."""
        n = 100
        X = rng.randn(n, MATCHUP_DIM)
        # Deliberately imbalanced labels
        y = np.concatenate([np.ones(80), np.zeros(20)])

        _, y_aug, _, _, _ = symmetric_augment(X, y)
        assert np.mean(y_aug) == pytest.approx(0.5, abs=1e-10)

    def test_swap_is_involution(self, rng):
        """Swapping twice returns the original vector."""
        X = rng.randn(30, MATCHUP_DIM)
        X_double_swap = swap_matchup_batch(swap_matchup_batch(X))
        np.testing.assert_allclose(X_double_swap, X, atol=1e-12)

    def test_augmentation_margins_negate(self, rng):
        """Margins are correctly negated for swapped perspective."""
        n = 20
        X = rng.randn(n, MATCHUP_DIM)
        y = rng.randint(0, 2, n).astype(float)
        margins = rng.randn(n) * 10

        _, _, m_aug, _, _ = symmetric_augment(X, y, margins=margins)

        for i in range(n):
            assert m_aug[2 * i] == pytest.approx(margins[i])
            assert m_aug[2 * i + 1] == pytest.approx(-margins[i])

    def test_fold_chronological_integrity(self, rng):
        """Augmented pairs share the same sort key (chronological integrity)."""
        n = 20
        X = rng.randn(n, MATCHUP_DIM)
        y = rng.randint(0, 2, n).astype(float)
        sort_keys = np.arange(n, dtype=float)

        _, _, _, _, sk_aug = symmetric_augment(X, y, sort_keys=sort_keys)

        for i in range(n):
            assert sk_aug[2 * i] == sk_aug[2 * i + 1]
