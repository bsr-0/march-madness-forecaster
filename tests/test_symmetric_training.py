"""Comprehensive tests for symmetric training augmentation.

Tests verify:
1. Mathematical correctness of the swap operation
2. Zero-sum property enforcement
3. No information leakage across train/val splits
4. Sample doubling with correct label/margin inversion
5. Edge cases and integration with pipeline config
"""

from __future__ import annotations

import numpy as np
import pytest

from src.ml.training.symmetric import (
    DIFF_END,
    DIFF_START,
    ABS_END,
    ABS_START,
    INTERACT_START,
    INTERACT_END,
    MATCHUP_DIM,
    SEED_DIFF_IDX,
    swap_matchup_vector,
    swap_matchup_batch,
    symmetric_augment,
    verify_zero_sum_property,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    return np.random.RandomState(42)


@pytest.fixture
def random_matchup(rng):
    """A single random 78-dim matchup vector."""
    return rng.randn(MATCHUP_DIM)


@pytest.fixture
def random_matchup_batch(rng):
    """Batch of 50 random 78-dim matchup vectors."""
    return rng.randn(50, MATCHUP_DIM)


@pytest.fixture
def realistic_matchup():
    """A realistic matchup vector mimicking a 1-seed vs 16-seed game."""
    x = np.zeros(MATCHUP_DIM)
    # Differential features: 1-seed is much better
    x[0] = 0.4    # diff_adj_off (1-seed has better offense)
    x[1] = -0.3   # diff_adj_def (1-seed has better defense = lower)
    x[2] = 0.05   # diff_tempo
    x[3] = 0.15   # diff_efg
    x[63] = 0.8   # diff_seed_strength (big difference)
    x[64] = 0.5   # diff_external_rating_composite
    x[65] = 0.2   # diff_external_rating_spread

    # Absolute features
    x[66] = 0.8   # avg_adj_off
    x[67] = 0.3   # avg_adj_def
    x[68] = 0.5   # avg_sos_adj_em
    x[69] = 0.6   # avg_momentum
    x[70] = 0.4   # avg_roster_continuity

    # Interaction features
    x[71] = 0.3   # tempo_interaction
    x[72] = 0.1   # style_mismatch
    x[73] = 0.2   # h2h_record
    x[74] = 0.1   # common_opp_margin
    x[75] = -0.05 # travel_advantage
    x[76] = -0.88 # seed_interaction: (1*16)/128 - 1 = -0.875
    x[77] = -1.0  # seed_diff: (1-16)/15 = -1.0
    return x


# ===========================================================================
# Test class: swap_matchup_vector
# ===========================================================================

class TestSwapMatchupVector:
    """Tests for the single-vector swap operation."""

    def test_differential_features_negate(self, random_matchup):
        """All 66 differential features should negate."""
        swapped = swap_matchup_vector(random_matchup)
        np.testing.assert_allclose(
            swapped[DIFF_START:DIFF_END],
            -random_matchup[DIFF_START:DIFF_END],
            atol=1e-15,
        )

    def test_absolute_features_unchanged(self, random_matchup):
        """All 5 absolute features should be unchanged."""
        swapped = swap_matchup_vector(random_matchup)
        np.testing.assert_allclose(
            swapped[ABS_START:ABS_END],
            random_matchup[ABS_START:ABS_END],
            atol=1e-15,
        )

    def test_interaction_features_transform_correctly(self, random_matchup):
        """Interaction features should transform correctly under swap."""
        swapped = swap_matchup_vector(random_matchup)
        # Symmetric interactions: [71, 72, 76]
        for idx in [71, 72, 76]:
            assert swapped[idx] == pytest.approx(random_matchup[idx], abs=1e-15), f"idx {idx}"
        # h2h record flips perspective: 1 - x
        assert swapped[73] == pytest.approx(
            1.0 - random_matchup[73], abs=1e-15
        )
        # Common-opponent and travel are antisymmetric.
        assert swapped[74] == pytest.approx(-random_matchup[74], abs=1e-15)
        assert swapped[75] == pytest.approx(-random_matchup[75], abs=1e-15)
        # seed_diff remains antisymmetric.
        assert swapped[SEED_DIFF_IDX] == pytest.approx(
            -random_matchup[SEED_DIFF_IDX], abs=1e-15
        )

    def test_seed_diff_negates(self, random_matchup):
        """seed_diff at index 77 should negate."""
        swapped = swap_matchup_vector(random_matchup)
        assert swapped[SEED_DIFF_IDX] == pytest.approx(
            -random_matchup[SEED_DIFF_IDX], abs=1e-15
        )

    def test_involution_property(self, random_matchup):
        """Swap(swap(x)) should equal x exactly (involution)."""
        double_swapped = swap_matchup_vector(swap_matchup_vector(random_matchup))
        np.testing.assert_allclose(double_swapped, random_matchup, atol=1e-15)

    def test_does_not_modify_input(self, random_matchup):
        """The input array should not be modified."""
        original = random_matchup.copy()
        _ = swap_matchup_vector(random_matchup)
        np.testing.assert_array_equal(random_matchup, original)

    def test_realistic_matchup_swap(self, realistic_matchup):
        """Swap of realistic 1v16 matchup should produce 16v1 perspective."""
        swapped = swap_matchup_vector(realistic_matchup)

        # Differential: 16-seed now has "worse" offense from its perspective
        assert swapped[0] == pytest.approx(-0.4, abs=1e-15)  # diff_adj_off
        assert swapped[1] == pytest.approx(0.3, abs=1e-15)   # diff_adj_def
        assert swapped[63] == pytest.approx(-0.8, abs=1e-15)  # diff_seed_strength

        # Absolute unchanged
        assert swapped[66] == pytest.approx(0.8, abs=1e-15)  # avg_adj_off

        # Seed interaction unchanged (commutative)
        assert swapped[76] == pytest.approx(-0.88, abs=1e-15)

        # h2h_record flips perspective
        assert swapped[73] == pytest.approx(0.8, abs=1e-15)

        # Seed diff negated
        assert swapped[77] == pytest.approx(1.0, abs=1e-15)

    def test_zero_vector(self):
        """Swap of neutral vector should remain unchanged."""
        x = np.zeros(MATCHUP_DIM)
        x[73] = 0.5  # Neutral h2h prior is the identity under 1 - x transform.
        swapped = swap_matchup_vector(x)
        np.testing.assert_array_equal(swapped, x)

    def test_too_short_raises(self):
        """Vector shorter than 78 dims should raise ValueError."""
        with pytest.raises(ValueError, match="78 dimensions"):
            swap_matchup_vector(np.zeros(50))

    def test_padded_vector(self):
        """Vector longer than 78 dims should keep padding unchanged."""
        x = np.ones(100)
        swapped = swap_matchup_vector(x)

        # Dims beyond 78 unchanged
        np.testing.assert_allclose(swapped[78:], x[78:], atol=1e-15)

        # But diff features still negate
        np.testing.assert_allclose(
            swapped[DIFF_START:DIFF_END],
            -x[DIFF_START:DIFF_END],
            atol=1e-15,
        )


# ===========================================================================
# Test class: swap_matchup_batch
# ===========================================================================

class TestSwapMatchupBatch:
    """Tests for the vectorized batch swap."""

    def test_matches_scalar(self, random_matchup_batch):
        """Batch swap should match element-wise scalar swap."""
        batch_result = swap_matchup_batch(random_matchup_batch)
        for i in range(len(random_matchup_batch)):
            scalar_result = swap_matchup_vector(random_matchup_batch[i])
            np.testing.assert_allclose(batch_result[i], scalar_result, atol=1e-15)

    def test_output_shape(self, random_matchup_batch):
        """Output shape should match input shape."""
        result = swap_matchup_batch(random_matchup_batch)
        assert result.shape == random_matchup_batch.shape

    def test_involution(self, random_matchup_batch):
        """Double swap should return original."""
        double = swap_matchup_batch(swap_matchup_batch(random_matchup_batch))
        np.testing.assert_allclose(double, random_matchup_batch, atol=1e-15)

    def test_does_not_modify_input(self, random_matchup_batch):
        """Should not modify input array."""
        original = random_matchup_batch.copy()
        _ = swap_matchup_batch(random_matchup_batch)
        np.testing.assert_array_equal(random_matchup_batch, original)

    def test_1d_raises(self):
        """1D array should raise ValueError."""
        with pytest.raises(ValueError, match="2D array"):
            swap_matchup_batch(np.zeros(78))

    def test_too_few_columns_raises(self):
        """Array with fewer than 78 columns should raise."""
        with pytest.raises(ValueError, match="78 columns"):
            swap_matchup_batch(np.zeros((10, 50)))


# ===========================================================================
# Test class: symmetric_augment
# ===========================================================================

class TestSymmetricAugment:
    """Tests for the full augmentation pipeline."""

    def test_doubles_samples(self, rng):
        """Output should have exactly 2× the samples."""
        n = 30
        X = rng.randn(n, MATCHUP_DIM)
        y = rng.randint(0, 2, n)
        margins = rng.randn(n) * 10

        X_aug, y_aug, m_aug, _, _ = symmetric_augment(X, y, margins)

        assert X_aug.shape[0] == 2 * n
        assert len(y_aug) == 2 * n
        assert len(m_aug) == 2 * n

    def test_label_inversion(self, rng):
        """Swapped rows should have inverted labels."""
        n = 30
        X = rng.randn(n, MATCHUP_DIM)
        y = rng.randint(0, 2, n)

        X_aug, y_aug, _, _, _ = symmetric_augment(X, y)

        # Original rows at even indices, swapped at odd
        for i in range(n):
            assert y_aug[2 * i] == y[i]
            assert y_aug[2 * i + 1] == 1 - y[i]

    def test_margin_inversion(self, rng):
        """Swapped rows should have negated margins."""
        n = 30
        X = rng.randn(n, MATCHUP_DIM)
        y = rng.randint(0, 2, n)
        margins = rng.randn(n) * 10

        _, _, m_aug, _, _ = symmetric_augment(X, y, margins)

        for i in range(n):
            assert m_aug[2 * i] == pytest.approx(margins[i])
            assert m_aug[2 * i + 1] == pytest.approx(-margins[i])

    def test_feature_swap_in_augmented_rows(self, rng):
        """Odd-indexed rows should be the swap of even-indexed rows."""
        n = 20
        X = rng.randn(n, MATCHUP_DIM)
        y = rng.randint(0, 2, n)

        X_aug, _, _, _, _ = symmetric_augment(X, y)

        for i in range(n):
            original = X_aug[2 * i]
            swapped = X_aug[2 * i + 1]

            # Differential features negated
            np.testing.assert_allclose(
                swapped[DIFF_START:DIFF_END],
                -original[DIFF_START:DIFF_END],
                atol=1e-15,
            )

            # Absolute features unchanged
            np.testing.assert_allclose(
                swapped[ABS_START:ABS_END],
                original[ABS_START:ABS_END],
                atol=1e-15,
            )

            # seed_diff negated
            assert swapped[SEED_DIFF_IDX] == pytest.approx(
                -original[SEED_DIFF_IDX], abs=1e-15
            )

    def test_round_weights_preserved(self, rng):
        """Both perspectives should get the same round weight."""
        n = 20
        X = rng.randn(n, MATCHUP_DIM)
        y = rng.randint(0, 2, n)
        rw = np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0] * 3 + [1.0, 1.0])

        _, _, _, rw_aug, _ = symmetric_augment(X, y, round_weights=rw)

        for i in range(n):
            assert rw_aug[2 * i] == rw[i]
            assert rw_aug[2 * i + 1] == rw[i]

    def test_sort_keys_preserved(self, rng):
        """Both perspectives should share the same sort key."""
        n = 20
        X = rng.randn(n, MATCHUP_DIM)
        y = rng.randint(0, 2, n)
        sk = np.arange(n, dtype=float) * 1000

        _, _, _, _, sk_aug = symmetric_augment(X, y, sort_keys=sk)

        for i in range(n):
            assert sk_aug[2 * i] == sk[i]
            assert sk_aug[2 * i + 1] == sk[i]

    def test_none_optionals(self, rng):
        """Should handle None margins, round_weights, sort_keys."""
        n = 10
        X = rng.randn(n, MATCHUP_DIM)
        y = rng.randint(0, 2, n)

        X_aug, y_aug, m_aug, rw_aug, sk_aug = symmetric_augment(X, y)

        assert X_aug.shape[0] == 2 * n
        assert m_aug is None
        assert rw_aug is None
        assert sk_aug is None

    def test_empty_input(self):
        """Should handle empty arrays gracefully."""
        X = np.empty((0, MATCHUP_DIM))
        y = np.array([], dtype=int)

        X_aug, y_aug, _, _, _ = symmetric_augment(X, y)

        assert X_aug.shape == (0, MATCHUP_DIM)
        assert len(y_aug) == 0

    def test_single_sample(self, rng):
        """Should work for a single training sample."""
        X = rng.randn(1, MATCHUP_DIM)
        y = np.array([1])
        margins = np.array([12.0])

        X_aug, y_aug, m_aug, _, _ = symmetric_augment(X, y, margins)

        assert X_aug.shape == (2, MATCHUP_DIM)
        assert y_aug[0] == 1
        assert y_aug[1] == 0
        assert m_aug[0] == pytest.approx(12.0)
        assert m_aug[1] == pytest.approx(-12.0)

    def test_length_mismatch_raises(self, rng):
        """Should raise if X and y have different lengths."""
        X = rng.randn(10, MATCHUP_DIM)
        y = np.array([0, 1])

        with pytest.raises(ValueError, match="mismatch"):
            symmetric_augment(X, y)

    def test_interleaving_order(self, rng):
        """Augmented array should be interleaved [orig_0, swap_0, orig_1, ...]."""
        n = 5
        X = rng.randn(n, MATCHUP_DIM)
        y = rng.randint(0, 2, n)

        X_aug, _, _, _, _ = symmetric_augment(X, y)

        # Original rows at even indices
        for i in range(n):
            np.testing.assert_allclose(X_aug[2 * i], X[i], atol=1e-15)

        # Swapped rows at odd indices
        for i in range(n):
            expected_swap = swap_matchup_vector(X[i])
            np.testing.assert_allclose(X_aug[2 * i + 1], expected_swap, atol=1e-15)


# ===========================================================================
# Test class: Zero-sum property
# ===========================================================================

class TestZeroSumProperty:
    """Tests that verify the zero-sum learning principle."""

    def test_label_sums_to_one(self, rng):
        """For each game pair, y_orig + y_swap should equal 1."""
        n = 100
        X = rng.randn(n, MATCHUP_DIM)
        y = rng.randint(0, 2, n)

        _, y_aug, _, _, _ = symmetric_augment(X, y)

        for i in range(n):
            assert y_aug[2 * i] + y_aug[2 * i + 1] == 1

    def test_margin_sums_to_zero(self, rng):
        """For each game pair, margin_orig + margin_swap should equal 0."""
        n = 100
        X = rng.randn(n, MATCHUP_DIM)
        y = rng.randint(0, 2, n)
        margins = rng.randn(n) * 15

        _, _, m_aug, _, _ = symmetric_augment(X, y, margins)

        for i in range(n):
            assert m_aug[2 * i] + m_aug[2 * i + 1] == pytest.approx(0.0, abs=1e-15)

    def test_feature_antisymmetry(self, rng):
        """Differential features should sum to zero across perspectives."""
        n = 50
        X = rng.randn(n, MATCHUP_DIM)
        y = rng.randint(0, 2, n)

        X_aug, _, _, _, _ = symmetric_augment(X, y)

        for i in range(n):
            diff_sum = (
                X_aug[2 * i, DIFF_START:DIFF_END]
                + X_aug[2 * i + 1, DIFF_START:DIFF_END]
            )
            np.testing.assert_allclose(diff_sum, 0.0, atol=1e-15)

    def test_seed_diff_antisymmetry(self, rng):
        """seed_diff should sum to zero across perspectives."""
        n = 50
        X = rng.randn(n, MATCHUP_DIM)
        y = rng.randint(0, 2, n)

        X_aug, _, _, _, _ = symmetric_augment(X, y)

        for i in range(n):
            assert (
                X_aug[2 * i, SEED_DIFF_IDX]
                + X_aug[2 * i + 1, SEED_DIFF_IDX]
            ) == pytest.approx(0.0, abs=1e-15)

    def test_balanced_labels_after_augmentation(self, rng):
        """Augmented dataset should have exactly 50% positive labels."""
        n = 100
        X = rng.randn(n, MATCHUP_DIM)
        # Even with imbalanced original labels...
        y = np.ones(n, dtype=int)  # All team1 wins
        y[:10] = 0  # 90% positive rate

        _, y_aug, _, _, _ = symmetric_augment(X, y)

        # After augmentation, each game contributes one 1 and one 0
        pos_rate = np.mean(y_aug)
        assert pos_rate == pytest.approx(0.5, abs=1e-10)


# ===========================================================================
# Test class: No-leakage properties
# ===========================================================================

class TestNoLeakage:
    """Tests verifying no information leakage from symmetric augmentation."""

    def test_chronological_split_keeps_pairs_together(self, rng):
        """When split by sort_key, both perspectives of a game are on the same side."""
        n = 100
        X = rng.randn(n, MATCHUP_DIM)
        y = rng.randint(0, 2, n)
        # Sort keys represent dates — increasing
        sort_keys = np.sort(rng.rand(n) * 10000)

        _, _, _, _, sk_aug = symmetric_augment(X, y, sort_keys=sort_keys)

        # Split at the median sort key
        boundary = np.median(sk_aug)

        # For each pair, both should be on the same side
        for i in range(n):
            sk_orig = sk_aug[2 * i]
            sk_swap = sk_aug[2 * i + 1]

            # Both must be on the same side of the boundary
            assert sk_orig == sk_swap, (
                f"Game {i}: sort keys differ ({sk_orig} vs {sk_swap})"
            )
            # Therefore both are train or both are val
            if sk_orig < boundary:
                assert sk_swap < boundary
            else:
                assert sk_swap >= boundary

    def test_loyo_fold_integrity(self, rng):
        """Both perspectives should have the same year assignment."""
        # Simulate multi-year data
        years = [2018, 2019, 2021, 2022, 2023]
        all_X, all_y, all_years = [], [], []

        for yr in years:
            n_yr = 50
            X_yr = rng.randn(n_yr, MATCHUP_DIM)
            y_yr = rng.randint(0, 2, n_yr)
            all_X.append(X_yr)
            all_y.append(y_yr)
            all_years.extend([yr] * n_yr)

        X = np.vstack(all_X)
        y = np.concatenate(all_y)
        year_arr = np.array(all_years, dtype=float)

        # Augment with year as sort_key (same trick used in pipeline)
        X_aug, y_aug, _, _, yr_aug = symmetric_augment(
            X, y, sort_keys=year_arr
        )

        # Each pair should share the same year
        n_total = len(y)
        for i in range(n_total):
            assert yr_aug[2 * i] == yr_aug[2 * i + 1]

    def test_no_feature_information_leak(self, rng):
        """Swapped perspective should not reveal any new information."""
        n = 50
        X = rng.randn(n, MATCHUP_DIM)
        y = rng.randint(0, 2, n)

        X_aug, _, _, _, _ = symmetric_augment(X, y)

        # For each pair, the swapped version is fully determined by the original
        for i in range(n):
            orig = X_aug[2 * i]
            swap = X_aug[2 * i + 1]
            expected_swap = swap_matchup_vector(orig)
            np.testing.assert_allclose(swap, expected_swap, atol=1e-15)


# ===========================================================================
# Test class: verify_zero_sum_property diagnostic
# ===========================================================================

class TestVerifyZeroSum:
    """Tests for the zero-sum verification diagnostic."""

    def test_perfect_antisymmetric_model(self, rng):
        """A model that returns 1-p for swapped inputs has zero deviation."""
        X = rng.randn(100, MATCHUP_DIM)

        def perfect_model(X_in):
            # Simple logistic on first diff feature — antisymmetric by construction
            return 1.0 / (1.0 + np.exp(-X_in[:, 0] * 2))

        result = verify_zero_sum_property(X, perfect_model, n_samples=50)

        assert result["mean_deviation"] < 1e-10
        assert result["max_deviation"] < 1e-10
        assert result["fraction_within_tol"] == 1.0
        assert result["n_tested"] == 50

    def test_biased_model_has_deviation(self, rng):
        """A model that ignores features has constant output and deviates."""
        X = rng.randn(100, MATCHUP_DIM)

        def constant_model(X_in):
            return np.full(len(X_in), 0.7)

        result = verify_zero_sum_property(X, constant_model, n_samples=50)

        # 0.7 + 0.7 = 1.4, deviation = 0.4 per pair
        assert result["mean_deviation"] == pytest.approx(0.4, abs=1e-10)
        assert result["fraction_within_tol"] == 0.0

    def test_noisy_antisymmetric_model(self, rng):
        """A model with small noise should mostly pass."""
        X = rng.randn(200, MATCHUP_DIM)

        def noisy_model(X_in):
            p = 1.0 / (1.0 + np.exp(-X_in[:, 0] * 2))
            noise = rng.randn(len(p)) * 0.0001
            return np.clip(p + noise, 0, 1)

        result = verify_zero_sum_property(
            X, noisy_model, n_samples=100, tolerance=0.01
        )

        assert result["mean_deviation"] < 0.01
        assert result["fraction_within_tol"] > 0.9


# ===========================================================================
# Test class: Integration with build_matchup_vector
# ===========================================================================

class TestIntegrationWithMatchupVector:
    """Tests verifying symmetric swap is consistent with build_matchup_vector."""

    def test_swap_matches_reverse_construction(self):
        """swap(build_matchup(v1,v2)) should equal build_matchup(v2,v1)."""
        from src.data.features.proprietary_metrics import IncrementalMetricsEngine

        rng = np.random.RandomState(123)
        v1 = rng.randn(66).astype(np.float64)
        v2 = rng.randn(66).astype(np.float64)

        # Ensure seeds are reasonable
        seed1, seed2 = 3, 14

        # Build forward and reverse matchup vectors
        fwd = IncrementalMetricsEngine.build_matchup_vector(
            v1,
            v2,
            seed1,
            seed2,
            engine=None,
            team1_id="",
            team2_id="",
        )
        rev = IncrementalMetricsEngine.build_matchup_vector(
            v2,
            v1,
            seed2,
            seed1,
            engine=None,
            team1_id="",
            team2_id="",
        )

        # Swap the forward vector
        swapped = swap_matchup_vector(fwd)

        # Compare swapped forward with direct reverse construction
        # Differential: should match exactly
        np.testing.assert_allclose(
            swapped[DIFF_START:DIFF_END],
            rev[DIFF_START:DIFF_END],
            atol=1e-12,
            err_msg="Differential features don't match",
        )

        # Absolute: should match exactly (symmetric)
        np.testing.assert_allclose(
            swapped[ABS_START:ABS_END],
            rev[ABS_START:ABS_END],
            atol=1e-12,
            err_msg="Absolute features don't match",
        )

        # Interaction features
        # tempo_interaction: (v1[2]*v2[2]) = (v2[2]*v1[2]) → same
        assert swapped[71] == pytest.approx(rev[71], abs=1e-12)

        # style_mismatch: compare explicitly
        # Forward: ((v1[2]-v2[2]) * ((v1[0]-v1[1])-(v2[0]-v2[1]))) / 600
        # Reverse: ((v2[2]-v1[2]) * ((v2[0]-v2[1])-(v1[0]-v1[1]))) / 600
        # = (-(v1[2]-v2[2])) * (-(v1[0]-v1[1]-(v2[0]-v2[1]))) / 600
        # = same as forward
        assert swapped[72] == pytest.approx(rev[72], abs=1e-12)

        # seed_interaction: (s1*s2)/128 - 1 is commutative
        assert swapped[76] == pytest.approx(rev[76], abs=1e-12)

        # seed_diff: (s1-s2)/15 vs (s2-s1)/15 — negated
        assert swapped[77] == pytest.approx(rev[77], abs=1e-12)

    def test_swap_with_zero_seeds(self):
        """When seeds are zero, seed features should be zero and swap cleanly."""
        from src.data.features.proprietary_metrics import IncrementalMetricsEngine

        rng = np.random.RandomState(456)
        v1 = rng.randn(66).astype(np.float64)
        v2 = rng.randn(66).astype(np.float64)

        fwd = IncrementalMetricsEngine.build_matchup_vector(
            v1,
            v2,
            0,
            0,
            engine=None,
            team1_id="",
            team2_id="",
        )
        rev = IncrementalMetricsEngine.build_matchup_vector(
            v2,
            v1,
            0,
            0,
            engine=None,
            team1_id="",
            team2_id="",
        )

        swapped = swap_matchup_vector(fwd)
        np.testing.assert_allclose(swapped, rev, atol=1e-12)


# ===========================================================================
# Test class: Pipeline config integration
# ===========================================================================

class TestPipelineConfigIntegration:
    """Tests that the config option works correctly."""

    def test_config_has_enable_symmetric_augmentation(self):
        """SOTAPipelineConfig should have the new flag with default True."""
        from src.pipeline.sota import SOTAPipelineConfig
        config = SOTAPipelineConfig()
        assert hasattr(config, "enable_symmetric_augmentation")
        assert config.enable_symmetric_augmentation is True

    def test_config_disable(self):
        """Setting enable_symmetric_augmentation=False should be accepted."""
        from src.pipeline.sota import SOTAPipelineConfig
        config = SOTAPipelineConfig(enable_symmetric_augmentation=False)
        assert config.enable_symmetric_augmentation is False


# ===========================================================================
# Test class: Mathematical edge cases
# ===========================================================================

class TestMathematicalEdgeCases:
    """Edge cases for mathematical correctness."""

    def test_all_zeros_differential(self):
        """50-50 matchup: diff=0 → swap is identical."""
        x = np.zeros(MATCHUP_DIM)
        x[66:71] = [0.5, 0.3, 0.4, 0.6, 0.7]  # non-zero absolute
        x[71:77] = [0.3, 0.1, 0.5, 0.0, 0.0, -0.8]  # interactions

        swapped = swap_matchup_vector(x)

        # Diff is zero → swapped diff is also zero
        np.testing.assert_allclose(swapped[DIFF_START:DIFF_END], 0.0, atol=1e-15)
        # Absolute unchanged
        np.testing.assert_allclose(swapped[66:71], x[66:71], atol=1e-15)

    def test_large_values(self):
        """Extreme feature values should swap without overflow."""
        x = np.full(MATCHUP_DIM, 1e10)
        swapped = swap_matchup_vector(x)

        np.testing.assert_allclose(
            swapped[DIFF_START:DIFF_END], -1e10, atol=1e-5
        )
        assert swapped[SEED_DIFF_IDX] == pytest.approx(-1e10, rel=1e-10)

    def test_inf_values(self):
        """Inf values should negate correctly."""
        x = np.zeros(MATCHUP_DIM)
        x[0] = np.inf
        x[1] = -np.inf

        swapped = swap_matchup_vector(x)
        assert swapped[0] == -np.inf
        assert swapped[1] == np.inf

    def test_nan_passthrough(self):
        """NaN values should pass through (model handles NaN elsewhere)."""
        x = np.zeros(MATCHUP_DIM)
        x[5] = np.nan

        swapped = swap_matchup_vector(x)
        assert np.isnan(swapped[5])

    def test_augment_preserves_dtype(self, rng):
        """Augmented arrays should preserve float64 dtype."""
        X = rng.randn(10, MATCHUP_DIM).astype(np.float64)
        y = rng.randint(0, 2, 10).astype(np.int32)
        margins = rng.randn(10).astype(np.float64)

        X_aug, y_aug, m_aug, _, _ = symmetric_augment(X, y, margins)

        assert X_aug.dtype == np.float64
        assert m_aug.dtype == np.float64

    @pytest.mark.parametrize("n", [1, 2, 5, 10, 100, 1000])
    def test_various_batch_sizes(self, n):
        """Should work for various batch sizes."""
        rng = np.random.RandomState(n)
        X = rng.randn(n, MATCHUP_DIM)
        y = rng.randint(0, 2, n)

        X_aug, y_aug, _, _, _ = symmetric_augment(X, y)

        assert X_aug.shape == (2 * n, MATCHUP_DIM)
        assert len(y_aug) == 2 * n
        assert np.mean(y_aug) == pytest.approx(0.5, abs=1e-10)
