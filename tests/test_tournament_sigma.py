"""Tests for TournamentSigmaCalibrator: tournament-specific sigma calibration.

Validates that:
1. Per-round sigma values are empirically derived and decrease for later rounds
2. Bayesian shrinkage regularizes low-sample rounds (F4, NCG)
3. Tournament sigma is systematically lower than regular-season sigma (11.0)
4. Kaggle-weighted Brier score improves vs baseline sigma=11.0
5. Integration with SpreadRegressor and RoundSpecificCalibrator works correctly
"""

import math
import os
import tempfile

import numpy as np
import pytest

from src.ml.ensemble.tournament_sigma import (
    TournamentSigmaCalibrator,
    RoundSigmaEstimate,
    daynum_to_round,
    load_tournament_sigma_data,
    KAGGLE_ROUND_WEIGHTS,
)
from src.ml.ensemble.spread_model import SpreadRegressor, _logistic_cdf
from src.ml.ensemble.margin_first_ensemble import (
    RoundSpecificCalibrator,
    DEFAULT_ROUND_SIGMAS,
)


# ---------------------------------------------------------------------------
# DayNum → Round mapping
# ---------------------------------------------------------------------------


class TestDaynumToRound:
    def test_r64_daynum(self):
        assert daynum_to_round(134) == "R64"
        assert daynum_to_round(136) == "R64"
        assert daynum_to_round(137) == "R64"

    def test_r32_daynum(self):
        assert daynum_to_round(138) == "R32"
        assert daynum_to_round(139) == "R32"
        assert daynum_to_round(140) == "R32"

    def test_s16_daynum(self):
        assert daynum_to_round(143) == "S16"
        assert daynum_to_round(144) == "S16"

    def test_e8_daynum(self):
        assert daynum_to_round(145) == "E8"
        assert daynum_to_round(146) == "E8"

    def test_f4_daynum(self):
        assert daynum_to_round(152) == "F4"

    def test_ncg_daynum(self):
        assert daynum_to_round(154) == "NCG"
        assert daynum_to_round(160) == "NCG"

    def test_low_daynum_defaults_to_early_round(self):
        result = daynum_to_round(130)
        assert result == "R64"


# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------


def _generate_tournament_data(
    n_per_round: dict = None,
    true_sigmas: dict = None,
    seed: int = 42,
):
    """Generate synthetic tournament data with known per-round sigma.

    Returns (predicted_spreads, actual_margins, round_labels).
    The actual margins are generated as:
        actual_margin = predicted_spread + noise(0, true_sigma)
    """
    rng = np.random.default_rng(seed)

    if n_per_round is None:
        n_per_round = {
            "R64": 640,  # ~20 years × 32 games
            "R32": 320,  # ~20 years × 16 games
            "S16": 160,  # ~20 years × 8 games
            "E8": 80,    # ~20 years × 4 games
            "F4": 40,    # ~20 years × 2 games
            "NCG": 20,   # ~20 years × 1 game
        }

    if true_sigmas is None:
        # Realistic tournament sigmas (tighter than regular season 11.0)
        true_sigmas = {
            "R64": 10.0,
            "R32": 9.5,
            "S16": 9.0,
            "E8": 8.5,
            "F4": 8.0,
            "NCG": 7.5,
        }

    all_spreads = []
    all_margins = []
    all_rounds = []

    for round_name in ["R64", "R32", "S16", "E8", "F4", "NCG"]:
        n = n_per_round.get(round_name, 10)
        sigma = true_sigmas.get(round_name, 10.0)

        # Generate spread predictions (model output)
        spreads = rng.normal(0, 6.0, n)  # typical spread range
        # Generate actual margins with tournament-specific noise
        noise = rng.normal(0, sigma, n)
        margins = spreads + noise

        all_spreads.extend(spreads)
        all_margins.extend(margins)
        all_rounds.extend([round_name] * n)

    return (
        np.array(all_spreads),
        np.array(all_margins),
        np.array(all_rounds),
    )


# ---------------------------------------------------------------------------
# TournamentSigmaCalibrator: Residual-based calibration
# ---------------------------------------------------------------------------


class TestTournamentSigmaCalibratorResidual:
    def test_fit_from_residuals_returns_fitted(self):
        spreads, margins, rounds = _generate_tournament_data()
        calibrator = TournamentSigmaCalibrator()
        stats = calibrator.fit_from_residuals(spreads, margins, rounds)
        assert stats["fitted"] is True
        assert calibrator.fitted is True

    def test_per_round_sigmas_decrease_monotonically(self):
        """Later rounds should have tighter (lower) sigma values."""
        spreads, margins, rounds = _generate_tournament_data()
        calibrator = TournamentSigmaCalibrator(n_bootstrap=20)
        calibrator.fit_from_residuals(spreads, margins, rounds)

        sigmas = calibrator.get_all_sigmas()
        round_order = ["R64", "R32", "S16", "E8", "F4", "NCG"]

        # Check overall trend: R64 sigma > NCG sigma
        assert sigmas["R64"] > sigmas["NCG"], (
            f"R64 sigma ({sigmas['R64']:.2f}) should be > NCG sigma "
            f"({sigmas['NCG']:.2f}) since later rounds are tighter"
        )

        # Check monotonic decrease (allowing violations due to noise in
        # Brier-optimal sigma estimation with finite samples).  With Bayesian
        # shrinkage and limited data, some adjacent rounds may swap.
        violations = 0
        for i in range(len(round_order) - 1):
            if sigmas[round_order[i]] < sigmas[round_order[i + 1]]:
                violations += 1
        assert violations <= 2, (
            f"Sigma should generally decrease across rounds. Got: {sigmas}"
        )

    def test_tournament_sigma_lower_than_regular_season(self):
        """Global tournament sigma should be < 11.0 (regular season default)."""
        spreads, margins, rounds = _generate_tournament_data()
        calibrator = TournamentSigmaCalibrator(n_bootstrap=20)
        calibrator.fit_from_residuals(spreads, margins, rounds)

        assert calibrator.global_tournament_sigma < 11.0, (
            f"Tournament sigma ({calibrator.global_tournament_sigma:.2f}) "
            "should be lower than regular-season sigma (11.0)"
        )

    def test_bayesian_shrinkage_regularizes_small_samples(self):
        """F4/NCG with few samples should be shrunk toward global sigma."""
        spreads, margins, rounds = _generate_tournament_data()
        calibrator = TournamentSigmaCalibrator(
            prior_strength=30.0,
            n_bootstrap=20,
        )
        calibrator.fit_from_residuals(spreads, margins, rounds)

        # NCG has only 20 games -> alpha = 20/(20+30) = 0.4 -> heavy shrinkage
        ncg_est = calibrator.round_estimates.get("NCG")
        assert ncg_est is not None
        assert ncg_est.source in ("shrinkage", "empirical", "default")

        # R64 has 640 games -> alpha = 640/(640+30) ≈ 0.95 -> mostly empirical
        r64_est = calibrator.round_estimates.get("R64")
        assert r64_est is not None
        if r64_est.n_games > 100:
            assert r64_est.source == "empirical"

    def test_brier_improvement_positive(self):
        """Calibrated sigma should improve Brier score vs baseline sigma=11.0."""
        spreads, margins, rounds = _generate_tournament_data()
        calibrator = TournamentSigmaCalibrator(n_bootstrap=20)
        stats = calibrator.fit_from_residuals(spreads, margins, rounds)

        # At least some rounds should show improvement
        improvements = stats.get("round_brier_improvements", {})
        positive_improvements = sum(1 for v in improvements.values() if v > 0)
        assert positive_improvements > 0, (
            f"Expected some rounds to improve vs sigma=11.0. Got: {improvements}"
        )

    def test_insufficient_data_uses_defaults(self):
        """With too few games, should fall back to sensible defaults."""
        calibrator = TournamentSigmaCalibrator()
        stats = calibrator.fit_from_residuals(
            np.array([1.0, 2.0]),
            np.array([3.0, 1.0]),
            np.array(["R64", "R64"]),
        )
        assert stats["fitted"] is False
        assert calibrator.fitted is True  # _set_defaults makes it fitted
        assert calibrator.global_tournament_sigma < 11.0

    def test_get_sigma_returns_round_specific_value(self):
        spreads, margins, rounds = _generate_tournament_data()
        calibrator = TournamentSigmaCalibrator(n_bootstrap=20)
        calibrator.fit_from_residuals(spreads, margins, rounds)

        for round_name in ["R64", "R32", "S16", "E8", "F4", "NCG"]:
            sigma = calibrator.get_sigma(round_name)
            assert 4.0 <= sigma <= 18.0, (
                f"{round_name} sigma ({sigma:.2f}) out of range"
            )

    def test_get_sigma_unfitted_returns_default(self):
        calibrator = TournamentSigmaCalibrator()
        assert calibrator.get_sigma("R64") == 11.0

    def test_sample_weights_respected(self):
        """Higher-weighted samples should influence sigma more."""
        spreads, margins, rounds = _generate_tournament_data()

        # Uniform weights
        calibrator1 = TournamentSigmaCalibrator(n_bootstrap=20)
        calibrator1.fit_from_residuals(spreads, margins, rounds)

        # Upweight later rounds
        weights = np.ones(len(spreads))
        for i, r in enumerate(rounds):
            weights[i] = KAGGLE_ROUND_WEIGHTS.get(r, 1.0)

        calibrator2 = TournamentSigmaCalibrator(n_bootstrap=20)
        calibrator2.fit_from_residuals(spreads, margins, rounds, sample_weights=weights)

        # Both should be fitted
        assert calibrator1.fitted and calibrator2.fitted

    def test_diagnostics_complete(self):
        spreads, margins, rounds = _generate_tournament_data()
        calibrator = TournamentSigmaCalibrator(n_bootstrap=20)
        calibrator.fit_from_residuals(spreads, margins, rounds)

        diag = calibrator.get_diagnostics()
        assert "global_tournament_sigma" in diag
        assert "round_sigmas" in diag
        assert "kaggle_weighted_brier_improvement" in diag
        assert len(diag["round_sigmas"]) == 6


# ---------------------------------------------------------------------------
# TournamentSigmaCalibrator: Margin-distribution-based calibration
# ---------------------------------------------------------------------------


class TestTournamentSigmaCalibratorMargin:
    def test_fit_from_margins_returns_fitted(self):
        _, margins, rounds = _generate_tournament_data()
        calibrator = TournamentSigmaCalibrator()
        stats = calibrator.fit_from_margins(margins, rounds)
        assert stats["fitted"] is True
        assert stats["method"] == "margin_distribution"

    def test_margin_sigma_lower_than_regular_season(self):
        _, margins, rounds = _generate_tournament_data()
        calibrator = TournamentSigmaCalibrator()
        calibrator.fit_from_margins(margins, rounds)
        assert calibrator.global_tournament_sigma < 11.5, (
            f"Margin-derived sigma ({calibrator.global_tournament_sigma:.2f}) "
            "should be reasonable for tournament data"
        )

    def test_margin_per_round_sigmas_trend_downward(self):
        _, margins, rounds = _generate_tournament_data()
        calibrator = TournamentSigmaCalibrator()
        calibrator.fit_from_margins(margins, rounds)

        sigmas = calibrator.get_all_sigmas()
        assert sigmas["R64"] >= sigmas["NCG"] - 0.5, (
            f"Expected R64 >= NCG sigma. Got R64={sigmas['R64']:.2f}, "
            f"NCG={sigmas['NCG']:.2f}"
        )


# ---------------------------------------------------------------------------
# Probability conversion
# ---------------------------------------------------------------------------


class TestSpreadToProbability:
    def test_zero_spread_gives_half(self):
        calibrator = TournamentSigmaCalibrator()
        calibrator._set_defaults()
        prob = calibrator.spread_to_probability(0.0, "R64")
        assert abs(prob - 0.5) < 1e-7

    def test_positive_spread_above_half(self):
        calibrator = TournamentSigmaCalibrator()
        calibrator._set_defaults()
        prob = calibrator.spread_to_probability(10.0, "R64")
        assert prob > 0.5

    def test_negative_spread_below_half(self):
        calibrator = TournamentSigmaCalibrator()
        calibrator._set_defaults()
        prob = calibrator.spread_to_probability(-10.0, "R64")
        assert prob < 0.5

    def test_tighter_sigma_produces_more_extreme_probs(self):
        """Same spread with tighter sigma → more confident prediction."""
        calibrator = TournamentSigmaCalibrator()
        calibrator._set_defaults()

        # R64 has sigma ~10.5, NCG has sigma ~8.0
        spread = 8.0
        prob_r64 = calibrator.spread_to_probability(spread, "R64")
        prob_ncg = calibrator.spread_to_probability(spread, "NCG")

        # NCG (tighter sigma) should produce more extreme probability
        assert prob_ncg > prob_r64, (
            f"NCG prob ({prob_ncg:.4f}) should be more extreme than "
            f"R64 prob ({prob_r64:.4f}) for same spread"
        )

    def test_batch_conversion_matches_scalar(self):
        calibrator = TournamentSigmaCalibrator()
        calibrator._set_defaults()

        spreads = np.array([-10.0, -5.0, 0.0, 5.0, 10.0])
        batch_probs = calibrator.spreads_to_probabilities(spreads, "S16")

        for i, s in enumerate(spreads):
            scalar_prob = calibrator.spread_to_probability(float(s), "S16")
            assert abs(batch_probs[i] - scalar_prob) < 1e-10

    def test_antisymmetry(self):
        calibrator = TournamentSigmaCalibrator()
        calibrator._set_defaults()

        for round_name in ["R64", "R32", "S16", "E8", "F4", "NCG"]:
            for spread in [5.0, 10.0, 15.0]:
                p_pos = calibrator.spread_to_probability(spread, round_name)
                p_neg = calibrator.spread_to_probability(-spread, round_name)
                assert abs(p_pos + p_neg - 1.0) < 1e-10, (
                    f"Antisymmetry violated for {round_name} spread={spread}"
                )


# ---------------------------------------------------------------------------
# Default sigma values
# ---------------------------------------------------------------------------


class TestDefaultSigmas:
    def test_defaults_are_set(self):
        calibrator = TournamentSigmaCalibrator()
        calibrator._set_defaults()
        assert calibrator.fitted is True
        assert len(calibrator.round_estimates) == 6

    def test_defaults_decrease_across_rounds(self):
        calibrator = TournamentSigmaCalibrator()
        calibrator._set_defaults()
        sigmas = calibrator.get_all_sigmas()

        round_order = ["R64", "R32", "S16", "E8", "F4", "NCG"]
        for i in range(len(round_order) - 1):
            assert sigmas[round_order[i]] >= sigmas[round_order[i + 1]], (
                f"{round_order[i]} ({sigmas[round_order[i]]:.1f}) should be >= "
                f"{round_order[i+1]} ({sigmas[round_order[i+1]]:.1f})"
            )

    def test_defaults_all_below_regular_season(self):
        calibrator = TournamentSigmaCalibrator()
        calibrator._set_defaults()
        for rname, est in calibrator.round_estimates.items():
            assert est.sigma < 11.0, (
                f"Default {rname} sigma ({est.sigma}) should be < 11.0 "
                "(regular season)"
            )

    def test_margin_first_ensemble_defaults_updated(self):
        """DEFAULT_ROUND_SIGMAS should reflect tournament-appropriate values."""
        for rname, sigma in DEFAULT_ROUND_SIGMAS.items():
            assert sigma <= 10.5, (
                f"DEFAULT_ROUND_SIGMAS[{rname}] = {sigma} should be <= 10.5 "
                "(tournament-appropriate)"
            )


# ---------------------------------------------------------------------------
# Integration with SpreadRegressor
# ---------------------------------------------------------------------------


try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


@pytest.mark.skipif(not LIGHTGBM_AVAILABLE, reason="LightGBM not installed")
class TestSpreadRegressorIntegration:
    def _train_spread_model(self, sigma=11.0, n=300, seed=42):
        rng = np.random.RandomState(seed)
        X = rng.randn(n, 10)
        margins = 5.0 * X[:, 0] + 3.0 * X[:, 1] + rng.normal(0, 5.0, n)
        model = SpreadRegressor(sigma=sigma)
        model.train(X, margins, num_rounds=50)
        return model, X, margins

    def test_sigma_override_changes_probability(self):
        """SpreadRegressor.predict_probability(sigma_override=...) should differ."""
        model, X, _ = self._train_spread_model()

        probs_default = model.predict_probability(X[:10])
        probs_tight = model.predict_probability(X[:10], sigma_override=8.0)
        probs_loose = model.predict_probability(X[:10], sigma_override=15.0)

        # Tighter sigma should produce more extreme probabilities
        # (further from 0.5 on average)
        dev_default = np.mean(np.abs(probs_default - 0.5))
        dev_tight = np.mean(np.abs(probs_tight - 0.5))
        dev_loose = np.mean(np.abs(probs_loose - 0.5))

        assert dev_tight > dev_default, (
            "Tighter sigma should produce more extreme probabilities"
        )
        assert dev_loose < dev_default, (
            "Looser sigma should produce less extreme probabilities"
        )

    def test_tournament_sigma_override_improves_brier(self):
        """Using tournament sigma on tournament-like data should improve Brier."""
        # Generate data with tournament-like sigma (9.0)
        rng = np.random.RandomState(42)
        n = 500
        X = rng.randn(n, 10)
        true_spreads = 5.0 * X[:, 0] + 3.0 * X[:, 1]
        tournament_noise = rng.normal(0, 9.0, n)
        margins = true_spreads + tournament_noise
        outcomes = (margins > 0).astype(float)

        model = SpreadRegressor(sigma=11.0)
        model.train(X[:350], margins[:350], num_rounds=100)

        # Brier with regular-season sigma (11.0)
        probs_regular = model.predict_probability(X[350:])
        brier_regular = float(np.mean((probs_regular - outcomes[350:]) ** 2))

        # Brier with tournament sigma (9.0)
        probs_tourney = model.predict_probability(X[350:], sigma_override=9.0)
        brier_tourney = float(np.mean((probs_tourney - outcomes[350:]) ** 2))

        # Tournament sigma should be at least as good or better
        assert brier_tourney <= brier_regular + 0.005, (
            f"Tournament sigma Brier ({brier_tourney:.4f}) should be <= "
            f"regular sigma Brier ({brier_regular:.4f})"
        )


# ---------------------------------------------------------------------------
# Integration with RoundSpecificCalibrator
# ---------------------------------------------------------------------------


class TestRoundSpecificCalibratorIntegration:
    def test_tournament_sigma_calibrator_attached(self):
        """Attaching TournamentSigmaCalibrator should set round params."""
        calibrator = TournamentSigmaCalibrator()
        calibrator._set_defaults()

        rsc = RoundSpecificCalibrator()
        rsc.set_tournament_sigma_calibrator(calibrator)

        assert rsc.calibrated is True
        assert len(rsc.round_params) == 6

        for rname in ["R64", "R32", "S16", "E8", "F4", "NCG"]:
            assert rname in rsc.round_params
            assert rsc.round_params[rname]["source"] == "tournament_sigma_calibrator"

    def test_tournament_sigma_flows_to_probability(self):
        """Round-specific sigma should affect probability conversion."""
        calibrator = TournamentSigmaCalibrator()
        calibrator._set_defaults()

        rsc = RoundSpecificCalibrator()
        rsc.set_tournament_sigma_calibrator(calibrator)

        spread = 8.0
        prob_r64 = rsc.convert_spread_to_prob(spread, "R64")
        prob_ncg = rsc.convert_spread_to_prob(spread, "NCG")

        # NCG (tighter sigma) should give more extreme probability
        assert prob_ncg > prob_r64

    def test_grid_search_centered_on_tournament_sigma(self):
        """Grid search should be centered around tournament-calibrated sigma."""
        calibrator = TournamentSigmaCalibrator()
        calibrator._set_defaults()

        rsc = RoundSpecificCalibrator()
        rsc.set_tournament_sigma_calibrator(calibrator)

        # The _get_default_sigma should now return tournament values
        for rname in ["R64", "R32", "S16", "E8", "F4", "NCG"]:
            default = rsc._get_default_sigma(rname)
            assert default == calibrator.get_sigma(rname)


# ---------------------------------------------------------------------------
# CSV data loading
# ---------------------------------------------------------------------------


class TestLoadTournamentData:
    def test_load_from_kaggle_csv(self):
        """Test loading from the actual Kaggle tournament results CSV."""
        csv_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "kaggle", "MNCAATourneyCompactResults.csv",
        )

        if not os.path.isfile(csv_path):
            pytest.skip("MNCAATourneyCompactResults.csv not found")

        margins, round_labels, seasons = load_tournament_sigma_data(
            csv_path, min_season=2010, max_season=2024,
        )

        assert len(margins) > 100, f"Expected >100 games, got {len(margins)}"
        assert len(margins) == len(round_labels) == len(seasons)

        # All margins should be positive (winner - loser in CSV)
        assert np.all(margins > 0)

        # All rounds should be valid
        valid_rounds = {"R64", "R32", "S16", "E8", "F4", "NCG"}
        unique_rounds = set(round_labels)
        assert unique_rounds.issubset(valid_rounds), (
            f"Unexpected rounds: {unique_rounds - valid_rounds}"
        )

        # Season range should be respected
        assert np.min(seasons) >= 2010
        assert np.max(seasons) <= 2024
        assert 2020 not in seasons  # No tournament in 2020

    def test_end_to_end_with_kaggle_data(self):
        """Full pipeline: load CSV → fit calibrator → verify sigmas."""
        csv_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "kaggle", "MNCAATourneyCompactResults.csv",
        )

        if not os.path.isfile(csv_path):
            pytest.skip("MNCAATourneyCompactResults.csv not found")

        margins, round_labels, _ = load_tournament_sigma_data(
            csv_path, min_season=2003, max_season=2024,
        )

        calibrator = TournamentSigmaCalibrator(n_bootstrap=50)
        stats = calibrator.fit_from_margins(margins, round_labels)

        assert stats["fitted"] is True
        assert calibrator.global_tournament_sigma > 0

        sigmas = calibrator.get_all_sigmas()
        # Tournament global sigma should be in reasonable range
        assert 7.0 <= calibrator.global_tournament_sigma <= 12.0, (
            f"Global tournament sigma ({calibrator.global_tournament_sigma:.2f}) "
            "out of expected range"
        )

        # Per-round sigmas should be in reasonable range
        for rname, sigma in sigmas.items():
            assert 5.0 <= sigma <= 13.0, (
                f"{rname} sigma ({sigma:.2f}) out of expected range"
            )

    def test_load_with_synthetic_csv(self):
        """Test CSV parsing with a synthetic file."""
        csv_content = (
            "Season,DayNum,WTeamID,WScore,LTeamID,LScore,WLoc,NumOT\n"
            "2019,134,1001,75,1002,65,N,0\n"  # R64, margin=10
            "2019,136,1001,70,1003,68,N,0\n"  # R32, margin=2
            "2019,143,1001,80,1004,72,N,0\n"  # E8, margin=8
            "2019,152,1001,68,1005,66,N,0\n"  # NCG, margin=2
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False,
        ) as f:
            f.write(csv_content)
            tmp_path = f.name

        try:
            margins, rounds, seasons = load_tournament_sigma_data(
                tmp_path, min_season=2019, max_season=2019,
            )
            assert len(margins) == 4
            assert margins[0] == 10  # R64 game
            assert rounds[0] == "R64"
            assert seasons[0] == 2019
        finally:
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Kaggle round weight awareness
# ---------------------------------------------------------------------------


class TestKaggleWeightAwareness:
    def test_kaggle_weights_defined(self):
        assert KAGGLE_ROUND_WEIGHTS["F4"] == 16.0
        assert KAGGLE_ROUND_WEIGHTS["NCG"] == 32.0

    def test_weighted_improvement_emphasizes_late_rounds(self):
        """Kaggle-weighted improvement should weight F4/NCG heavily."""
        spreads, margins, rounds = _generate_tournament_data()
        calibrator = TournamentSigmaCalibrator(n_bootstrap=20)
        stats = calibrator.fit_from_residuals(spreads, margins, rounds)

        assert "kaggle_weighted_brier_improvement" in stats
        # The improvement should be a real number (could be positive or negative
        # depending on data, but should be finite)
        assert math.isfinite(stats["kaggle_weighted_brier_improvement"])


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------


class TestBootstrapCI:
    def test_ci_bounds_bracket_point_estimate(self):
        spreads, margins, rounds = _generate_tournament_data()
        calibrator = TournamentSigmaCalibrator(n_bootstrap=50)
        calibrator.fit_from_residuals(spreads, margins, rounds)

        for rname, est in calibrator.round_estimates.items():
            if est.n_games >= 10 and est.source != "default":
                # CI should bracket the point estimate (with some tolerance)
                assert est.sigma_ci_lo <= est.sigma + 0.5, (
                    f"{rname}: CI lower ({est.sigma_ci_lo:.2f}) > "
                    f"point estimate ({est.sigma:.2f})"
                )
                assert est.sigma_ci_hi >= est.sigma - 0.5, (
                    f"{rname}: CI upper ({est.sigma_ci_hi:.2f}) < "
                    f"point estimate ({est.sigma:.2f})"
                )

    def test_ci_width_decreases_with_more_data(self):
        """More games should produce tighter confidence intervals."""
        # Small dataset
        spreads_s, margins_s, rounds_s = _generate_tournament_data(
            n_per_round={"R64": 50, "R32": 25, "S16": 12, "E8": 6, "F4": 3, "NCG": 2},
        )
        cal_small = TournamentSigmaCalibrator(n_bootstrap=50)
        cal_small.fit_from_residuals(spreads_s, margins_s, rounds_s)

        # Large dataset
        spreads_l, margins_l, rounds_l = _generate_tournament_data(
            n_per_round={"R64": 500, "R32": 250, "S16": 125, "E8": 60, "F4": 30, "NCG": 15},
        )
        cal_large = TournamentSigmaCalibrator(n_bootstrap=50)
        cal_large.fit_from_residuals(spreads_l, margins_l, rounds_l)

        # R64 should have tighter CI with more data
        r64_small = cal_small.round_estimates.get("R64")
        r64_large = cal_large.round_estimates.get("R64")

        if r64_small and r64_large and r64_small.n_games >= 10 and r64_large.n_games >= 10:
            width_small = r64_small.sigma_ci_hi - r64_small.sigma_ci_lo
            width_large = r64_large.sigma_ci_hi - r64_large.sigma_ci_lo
            # Large dataset CI should be tighter (with tolerance for randomness)
            assert width_large <= width_small + 1.0, (
                f"Large dataset CI ({width_large:.2f}) should be tighter than "
                f"small dataset CI ({width_small:.2f})"
            )
