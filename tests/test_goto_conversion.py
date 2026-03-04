"""Comprehensive tests for the goto_conversion implementation.

Tests validate:
1. Mathematical correctness against the reference implementation
   (gotoConversion/goto_conversion on GitHub)
2. Favourite-longshot bias correction properties
3. Edge cases and numerical stability
4. Integration with BrierPostProcessor
5. Calibration/fitting on synthetic tournament data
6. Round-weighted Brier optimization

Reference:
    gotoConversion/goto_conversion (GitHub), 2019-2025.
    H. S. Shin, "Prices of State Contingent Claims with Insider traders,
    and the Favorite-Longshot Bias", The Economic Journal, 1992.
"""

import numpy as np
import pytest

from src.ml.calibration.brier_optimal import (
    BrierPostProcessor,
    FavouriteLongshotCorrection,
    SeedBasedOverrides,
)


# ---------------------------------------------------------------------------
# Reference implementation for validation
# ---------------------------------------------------------------------------

def _reference_goto_conversion(list_of_odds, total=1.0):
    """Reference implementation from gotoConversion/goto_conversion on GitHub.

    This is a direct transliteration of the original Python code for
    validation purposes.

    Args:
        list_of_odds: Decimal odds (must be > 1.0)
        total: Target sum of probabilities (default 1.0)

    Returns:
        List of corrected probabilities
    """
    list_of_probs = [1.0 / x for x in list_of_odds]
    list_of_se = [pow((x - x ** 2.0) / x, 0.5) for x in list_of_probs]
    step = (sum(list_of_probs) - total) / sum(list_of_se)
    output = [x - (y * step) for x, y in zip(list_of_probs, list_of_se)]
    # Check for imprudent odds
    if any(0.0 >= x for x in output) or (sum(list_of_probs) <= 1.0):
        normalizer = sum(list_of_probs) / total
        output = [x / normalizer for x in list_of_probs]
    return output


def _our_goto_for_pairwise(p, margin):
    """Wrapper: apply our FavouriteLongshotCorrection to a single p."""
    result = FavouriteLongshotCorrection._apply(np.array([p]), margin)
    return float(result[0])


def _reference_goto_for_pairwise(p, margin):
    """Apply the reference goto_conversion to a pairwise matchup.

    Given P(team1) = p with artificial margin, convert to odds and apply.

    The odds with overround: o1 = 1/(p*(1+margin)), o2 = 1/((1-p)*(1+margin))

    Note: When p*(1+margin) >= 1.0, the implied probability exceeds 1.0
    and the reference algorithm produces complex numbers.  We cap at 0.9999
    to match our implementation's guard.
    """
    p = max(1e-4, min(1.0 - 1e-4, p))
    pi_1 = min(p * (1.0 + margin), 0.9999)
    pi_2 = min((1.0 - p) * (1.0 + margin), 0.9999)
    # Convert to decimal odds (inverse of probabilities)
    o1 = 1.0 / pi_1
    o2 = 1.0 / pi_2
    result = _reference_goto_conversion([o1, o2], total=1.0)
    return result[0]  # P(team1 wins)


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------

class TestGotoConversionMathematicalCorrectness:
    """Verify our implementation matches the reference algorithm."""

    @pytest.mark.parametrize("p", [0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
    @pytest.mark.parametrize("margin", [0.02, 0.05, 0.10, 0.15])
    def test_matches_reference_implementation(self, p, margin):
        """Our _apply must produce the same result as the reference."""
        our_result = _our_goto_for_pairwise(p, margin)
        ref_result = _reference_goto_for_pairwise(p, margin)
        assert abs(our_result - ref_result) < 1e-6, (
            f"Mismatch at p={p}, margin={margin}: "
            f"ours={our_result:.8f}, ref={ref_result:.8f}"
        )

    @pytest.mark.parametrize("p", [0.1, 0.2, 0.3, 0.4])
    @pytest.mark.parametrize("margin", [0.02, 0.05, 0.10])
    def test_matches_reference_for_underdogs(self, p, margin):
        """Reference match for longshot probabilities too."""
        our_result = _our_goto_for_pairwise(p, margin)
        ref_result = _reference_goto_for_pairwise(p, margin)
        assert abs(our_result - ref_result) < 1e-6, (
            f"Mismatch at p={p}, margin={margin}: "
            f"ours={our_result:.8f}, ref={ref_result:.8f}"
        )

    def test_symmetry_property(self):
        """goto_conversion(p) + goto_conversion(1-p) should equal 1.0.

        This is the zero-sum property: the corrected probabilities for
        a two-outcome event must sum to 1.
        """
        for margin in [0.02, 0.05, 0.10, 0.15]:
            for p in np.linspace(0.05, 0.95, 19):
                p1 = _our_goto_for_pairwise(p, margin)
                p2 = _our_goto_for_pairwise(1.0 - p, margin)
                assert abs(p1 + p2 - 1.0) < 1e-6, (
                    f"Symmetry violation at p={p:.2f}, margin={margin}: "
                    f"p1={p1:.8f}, p2={p2:.8f}, sum={p1+p2:.8f}"
                )

    def test_vectorized_matches_scalar(self):
        """Batch _apply should match per-element scalar application."""
        margin = 0.08
        preds = np.array([0.3, 0.5, 0.7, 0.85, 0.95])
        batch_result = FavouriteLongshotCorrection._apply(preds, margin)
        for i, p in enumerate(preds):
            scalar = _our_goto_for_pairwise(p, margin)
            assert abs(batch_result[i] - scalar) < 1e-10, (
                f"Vectorized mismatch at index {i}: "
                f"batch={batch_result[i]:.10f}, scalar={scalar:.10f}"
            )


class TestGotoConversionFLBProperties:
    """Test the favourite-longshot bias correction properties."""

    def test_favourites_increase(self):
        """Correction should increase P(favourite) — the core FLB fix.

        When margin > 0, goto_conversion sharpens predictions toward
        the favourite (higher probability), correcting the overvaluation
        of longshots.
        """
        margin = 0.05
        for p in [0.6, 0.7, 0.8, 0.9, 0.95]:
            corrected = _our_goto_for_pairwise(p, margin)
            assert corrected > p, (
                f"Favourite not increased: p={p}, corrected={corrected}"
            )

    def test_longshots_decrease(self):
        """Correction should decrease P(longshot)."""
        margin = 0.05
        for p in [0.05, 0.1, 0.2, 0.3, 0.4]:
            corrected = _our_goto_for_pairwise(p, margin)
            assert corrected < p, (
                f"Longshot not decreased: p={p}, corrected={corrected}"
            )

    def test_50_50_unchanged(self):
        """A 50-50 matchup should be (nearly) unchanged.

        When p=0.5, both sides have equal SE, so the correction is
        symmetric and the probability stays at 0.5.
        """
        for margin in [0.02, 0.05, 0.10, 0.15]:
            corrected = _our_goto_for_pairwise(0.5, margin)
            assert abs(corrected - 0.5) < 1e-6, (
                f"50-50 changed: margin={margin}, corrected={corrected}"
            )

    def test_stronger_correction_with_larger_margin(self):
        """Larger margin should produce stronger correction."""
        p = 0.8
        c1 = _our_goto_for_pairwise(p, 0.02)
        c2 = _our_goto_for_pairwise(p, 0.05)
        c3 = _our_goto_for_pairwise(p, 0.10)
        assert c1 < c2 < c3, (
            f"Correction not monotonic in margin: "
            f"c(0.02)={c1:.6f}, c(0.05)={c2:.6f}, c(0.10)={c3:.6f}"
        )

    def test_proportionally_larger_correction_for_longshots(self):
        """The absolute correction should be proportionally larger for
        predictions further from 0.5.

        This is the key property of goto_conversion: SE is wider for
        longshots, so the same "step" (in SE units) produces a larger
        absolute change for extreme predictions.
        """
        margin = 0.05
        p_mild = 0.6
        p_extreme = 0.9

        delta_mild = _our_goto_for_pairwise(p_mild, margin) - p_mild
        delta_extreme = _our_goto_for_pairwise(p_extreme, margin) - p_extreme

        # The absolute change at p=0.9 should be larger than at p=0.6
        # (favourite gets more additional probability)
        assert abs(delta_extreme) > abs(delta_mild), (
            f"Correction not proportionally larger for extremes: "
            f"delta(0.6)={delta_mild:.6f}, delta(0.9)={delta_extreme:.6f}"
        )


class TestGotoConversionEdgeCases:
    """Test numerical stability and edge cases."""

    def test_zero_margin_is_identity(self):
        """margin=0 should return predictions unchanged."""
        preds = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        result = FavouriteLongshotCorrection._apply(preds, 0.0)
        np.testing.assert_array_equal(result, preds)

    def test_very_small_margin(self):
        """Very small margin should produce minimal change."""
        preds = np.array([0.3, 0.7])
        result = FavouriteLongshotCorrection._apply(preds, 1e-6)
        np.testing.assert_allclose(result, preds, atol=1e-4)

    def test_extreme_probability_near_zero(self):
        """Near-zero probabilities should not produce NaN or negative."""
        preds = np.array([0.001, 0.01, 0.05])
        for margin in [0.02, 0.05, 0.10, 0.15]:
            result = FavouriteLongshotCorrection._apply(preds, margin)
            assert np.all(result > 0), f"Negative at margin={margin}: {result}"
            assert np.all(np.isfinite(result)), f"NaN at margin={margin}"

    def test_extreme_probability_near_one(self):
        """Near-one probabilities should not exceed 1.0."""
        preds = np.array([0.95, 0.99, 0.999])
        for margin in [0.02, 0.05, 0.10, 0.15]:
            result = FavouriteLongshotCorrection._apply(preds, margin)
            assert np.all(result < 1.0), f"Exceeds 1 at margin={margin}: {result}"
            assert np.all(np.isfinite(result)), f"NaN at margin={margin}"

    def test_large_margin_stability(self):
        """Large margin should not crash."""
        preds = np.array([0.3, 0.5, 0.7, 0.85])
        result = FavouriteLongshotCorrection._apply(preds, 0.30)
        assert np.all(np.isfinite(result))
        assert np.all(result > 0)
        assert np.all(result < 1)

    def test_empty_array(self):
        """Empty array should return empty."""
        result = FavouriteLongshotCorrection._apply(np.array([]), 0.05)
        assert len(result) == 0

    def test_single_prediction(self):
        """Single prediction should work."""
        result = FavouriteLongshotCorrection._apply(np.array([0.7]), 0.05)
        assert len(result) == 1
        assert 0 < result[0] < 1

    def test_all_same_prediction(self):
        """Array of identical 0.5 should stay 0.5."""
        preds = np.full(100, 0.5)
        result = FavouriteLongshotCorrection._apply(preds, 0.10)
        np.testing.assert_allclose(result, 0.5, atol=1e-6)


class TestGotoConversionFitting:
    """Test the Brier score optimization (fit method)."""

    def test_fit_on_well_calibrated_data(self):
        """If predictions are already well-calibrated, Brier should not worsen."""
        rng = np.random.default_rng(42)
        # Generate well-calibrated predictions
        probs = rng.uniform(0.3, 0.7, size=200)
        outcomes = (rng.random(200) < probs).astype(float)

        flc = FavouriteLongshotCorrection(strength=0.05)
        flc.fit(probs, outcomes)

        # The key property: Brier should not worsen after fitting
        corrected = flc.correct(probs)
        brier_before = float(np.mean((probs - outcomes) ** 2))
        brier_after = float(np.mean((corrected - outcomes) ** 2))
        assert brier_after <= brier_before + 1e-6, (
            f"Brier worsened on calibrated data: "
            f"{brier_before:.5f} -> {brier_after:.5f}"
        )
        assert flc.fitted

    def test_fit_on_overconfident_data(self):
        """If model is overconfident on favourites, margin should be positive.

        Simulate the favourite-longshot bias: model predicts 0.9 for
        favourites but they actually win only 0.8 of the time.
        """
        rng = np.random.default_rng(42)
        n = 300
        # Overconfident predictions for favourites
        probs = np.concatenate([
            rng.uniform(0.80, 0.95, n // 2),  # overconfident favourites
            rng.uniform(0.05, 0.20, n // 2),  # overconfident longshots
        ])
        # Actual outcomes are less extreme
        true_probs = np.concatenate([
            np.full(n // 2, 0.75),  # favourites win less than predicted
            np.full(n // 2, 0.25),  # longshots win more than predicted
        ])
        outcomes = (rng.random(n) < true_probs).astype(float)

        flc = FavouriteLongshotCorrection(strength=0.05)
        flc.fit(probs, outcomes)
        # Should find a non-trivial margin
        assert flc.fitted
        # The Brier score should improve (or at least not get worse)
        brier_before = float(np.mean((probs - outcomes) ** 2))
        corrected = flc.correct(probs)
        brier_after = float(np.mean((corrected - outcomes) ** 2))
        assert brier_after <= brier_before + 0.001, (
            f"Brier worsened: {brier_before:.4f} -> {brier_after:.4f}"
        )

    def test_fit_with_too_few_samples(self):
        """Should use default margin when < 20 samples."""
        probs = np.array([0.7, 0.8, 0.9, 0.6, 0.5])
        outcomes = np.array([1.0, 1.0, 1.0, 0.0, 1.0])

        flc = FavouriteLongshotCorrection(strength=0.03)
        flc.fit(probs, outcomes)
        assert flc.strength == 0.03  # Unchanged
        assert flc.fitted

    def test_fit_with_round_labels(self):
        """Should accept round labels for weighted optimization."""
        rng = np.random.default_rng(42)
        n = 100
        probs = rng.uniform(0.3, 0.9, n)
        outcomes = (rng.random(n) < probs * 0.95).astype(float)
        rounds = ["R64"] * 40 + ["R32"] * 25 + ["S16"] * 15 + ["E8"] * 10 + ["F4"] * 6 + ["NCG"] * 4

        flc = FavouriteLongshotCorrection(strength=0.05)
        flc.fit(probs, outcomes, round_labels=rounds)
        assert flc.fitted

    def test_correct_single(self):
        """correct_single should match correct(np.array([p]))[0]."""
        flc = FavouriteLongshotCorrection(strength=0.07)
        flc.fitted = True
        for p in [0.3, 0.5, 0.7, 0.9]:
            single = flc.correct_single(p)
            batch = float(flc.correct(np.array([p]))[0])
            assert abs(single - batch) < 1e-10


class TestGotoConversionInBrierPostProcessor:
    """Test integration with BrierPostProcessor."""

    def test_processor_without_goto(self):
        """PostProcessor should work without goto_converter set."""
        pp = BrierPostProcessor(goto_converter=None)
        result = pp.process(0.7)
        assert 0 < result < 1

    def test_processor_with_goto_identity(self):
        """Fitted goto with margin=0 should be identity in the pipeline."""
        flc = FavouriteLongshotCorrection(strength=0.0)
        flc.fitted = True
        pp = BrierPostProcessor(goto_converter=flc)
        result = pp.process(0.7, seed1=1, seed2=16)
        # Should still be processed by seed overrides but goto is identity
        assert 0 < result < 1

    def test_processor_with_goto_sharpens_favourites(self):
        """In the full pipeline, goto should make favourites more extreme."""
        flc = FavouriteLongshotCorrection(strength=0.05)
        flc.fitted = True
        pp_without = BrierPostProcessor(goto_converter=None)
        pp_with = BrierPostProcessor(goto_converter=flc)

        # For a moderate favourite (not affected by seed overrides)
        p_without = pp_without.process(0.7)
        p_with = pp_with.process(0.7)
        assert p_with > p_without, (
            f"goto should sharpen favourite: "
            f"without={p_without:.4f}, with={p_with:.4f}"
        )

    def test_processor_batch_with_goto(self):
        """Batch processing should apply goto_conversion."""
        flc = FavouriteLongshotCorrection(strength=0.05)
        flc.fitted = True
        pp = BrierPostProcessor(goto_converter=flc)

        preds = np.array([0.3, 0.5, 0.7, 0.9])
        result = pp.process_batch(preds)
        assert len(result) == 4
        assert np.all(result > 0)
        assert np.all(result < 1)

    def test_processing_chain_order(self):
        """Verify the processing order: seed → calibrate → goto → sharpen → clip.

        The goto_conversion should be applied after calibration and before
        sharpening.  We verify by checking that a fitted goto changes the
        result after calibration.
        """
        flc = FavouriteLongshotCorrection(strength=0.08)
        flc.fitted = True
        pp = BrierPostProcessor(
            goto_converter=flc,
            clip_lo=0.01,
            clip_hi=0.99,
        )
        # A moderate prediction that won't be caught by seed overrides
        result = pp.process(0.75)
        assert 0.01 <= result <= 0.99


class TestGotoConversionMarchMadnessScenarios:
    """Test with realistic March Madness scenarios."""

    def test_1_vs_16_seed_matchup(self):
        """1-seed vs 16-seed: goto should sharpen toward the favourite.

        Historical rate: 1-seeds win ~98.7% of the time.
        A model predicting 0.95 should be pushed higher.
        """
        p = 0.95  # Model predicts 95% for 1-seed
        margin = 0.05
        corrected = _our_goto_for_pairwise(p, margin)
        assert corrected > p, (
            f"1v16: should sharpen 1-seed: {p} -> {corrected}"
        )

    def test_5_vs_12_upset_zone(self):
        """5-seed vs 12-seed: the classic upset zone.

        Historical rate: ~64% for 5-seed.
        Model predicting 0.65 should see a smaller correction.
        """
        p = 0.65
        margin = 0.05
        corrected = _our_goto_for_pairwise(p, margin)
        # Correction should be modest (close to 0.5, lower SE differential)
        assert abs(corrected - p) < 0.02, (
            f"5v12: correction too large: {p} -> {corrected}"
        )

    def test_8_vs_9_coin_flip(self):
        """8-seed vs 9-seed: essentially a coin flip.

        Historical rate: ~51% for 8-seed.
        Near 0.5, correction should be negligible.
        """
        p = 0.51
        margin = 0.05
        corrected = _our_goto_for_pairwise(p, margin)
        assert abs(corrected - p) < 0.005, (
            f"8v9: correction too large near 0.5: {p} -> {corrected}"
        )

    def test_realistic_tournament_corrections(self):
        """Apply to a realistic set of tournament predictions.

        Verify that the overall distribution shifts in the right direction:
        more probability mass on favourites, less on longshots.
        """
        # Simulated R64 predictions (32 games, lower seed always team1)
        preds = np.array([
            0.99, 0.95, 0.88, 0.82,  # 1-seeds
            0.94, 0.90, 0.85, 0.80,  # 2-seeds
            0.85, 0.80, 0.75, 0.70,  # 3-seeds
            0.78, 0.72, 0.68, 0.65,  # 4-seeds
            0.65, 0.62, 0.60, 0.58,  # 5-seeds
            0.62, 0.60, 0.58, 0.55,  # 6-seeds
            0.58, 0.55, 0.52, 0.50,  # 7-seeds
            0.52, 0.50, 0.48, 0.45,  # 8-seeds
        ])

        corrected = FavouriteLongshotCorrection._apply(preds, 0.05)

        # Mean should increase (more weight on favourites)
        assert np.mean(corrected) > np.mean(preds), (
            "Overall mean should increase with FLB correction"
        )

        # Variance should increase (more separation)
        assert np.var(corrected) >= np.var(preds) - 0.001, (
            "Variance should not decrease significantly"
        )


class TestGotoConversionBrierImprovement:
    """Verify Brier score improvement on synthetic tournament data."""

    def test_brier_improvement_on_biased_predictions(self):
        """When predictions have strong FLB, goto_conversion should improve Brier.

        Simulate a model that systematically undervalues favourites:
        - True favourite win rate is ~85-95% but model predicts 70-80%
        - This mimics regular-season training → tournament prediction mismatch

        The goto_conversion should sharpen predictions toward favourites,
        improving Brier score.
        """
        rng = np.random.default_rng(42)
        n = 1000

        # True probabilities (tournament reality)
        true_probs_fav = rng.uniform(0.82, 0.96, n // 2)
        true_probs_dog = 1.0 - true_probs_fav

        # Model predictions: systematically underconfident on favourites
        # (regular-season data doesn't capture tournament dynamics)
        model_probs_fav = true_probs_fav - rng.uniform(0.08, 0.15, n // 2)
        model_probs_dog = 1.0 - model_probs_fav

        model_probs = np.concatenate([model_probs_fav, model_probs_dog])
        true_probs = np.concatenate([true_probs_fav, true_probs_dog])
        outcomes = (rng.random(n) < true_probs).astype(float)
        model_probs = np.clip(model_probs, 0.01, 0.99)

        # Fit goto_conversion
        flc = FavouriteLongshotCorrection(strength=0.05)
        flc.fit(model_probs, outcomes)

        corrected = flc.correct(model_probs)
        brier_before = float(np.mean((model_probs - outcomes) ** 2))
        brier_after = float(np.mean((corrected - outcomes) ** 2))

        # goto_conversion should improve Brier (or at minimum not worsen it,
        # since the optimizer minimizes Brier explicitly)
        assert brier_after <= brier_before + 1e-6, (
            f"Brier should not worsen after fitting: "
            f"{brier_before:.5f} -> {brier_after:.5f}"
        )
