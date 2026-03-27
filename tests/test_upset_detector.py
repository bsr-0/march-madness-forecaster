"""Tests for the upset detection layer."""

import math
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pytest

from src.ml.ensemble.upset_detector import (
    HISTORICAL_UPSET_RATES,
    UpsetDetector,
    UpsetRiskTier,
    UpsetSignal,
    _classify_risk_tier,
    _get_historical_upset_rate,
    _logistic_upset_fallback,
)


# ---------------------------------------------------------------------------
# Mock TeamFeatures for testing (matches real TeamFeatures attributes)
# ---------------------------------------------------------------------------

@dataclass
class MockTeamFeatures:
    """Minimal mock matching the attributes UpsetDetector reads."""
    seed: int = 8
    three_pt_variance: float = 0.095
    pace_adjusted_variance: float = 0.0
    momentum: float = 0.0
    avg_experience: float = 2.0
    bench_depth_score: float = 0.0
    adj_efficiency_margin: float = 0.0


class TestHistoricalUpsetRates:
    """Test historical upset rate data integrity."""

    def test_all_r64_matchups_present(self):
        expected = [(1, 16), (2, 15), (3, 14), (4, 13),
                    (5, 12), (6, 11), (7, 10), (8, 9)]
        for matchup in expected:
            assert matchup in HISTORICAL_UPSET_RATES, f"Missing R64 matchup {matchup}"

    def test_rates_in_valid_range(self):
        for matchup, rate in HISTORICAL_UPSET_RATES.items():
            assert 0.0 < rate < 1.0, f"Rate {rate} out of range for {matchup}"

    def test_rates_increase_as_seeds_converge(self):
        """Higher-seed matchups should have higher upset rates."""
        assert HISTORICAL_UPSET_RATES[(1, 16)] < HISTORICAL_UPSET_RATES[(2, 15)]
        assert HISTORICAL_UPSET_RATES[(2, 15)] < HISTORICAL_UPSET_RATES[(3, 14)]
        assert HISTORICAL_UPSET_RATES[(5, 12)] < HISTORICAL_UPSET_RATES[(6, 11)]
        assert HISTORICAL_UPSET_RATES[(7, 10)] < HISTORICAL_UPSET_RATES[(8, 9)]

    def test_8v9_near_coin_flip(self):
        assert 0.40 < HISTORICAL_UPSET_RATES[(8, 9)] < 0.55


class TestRiskTierClassification:
    def test_very_high(self):
        assert _classify_risk_tier(0.45) == UpsetRiskTier.VERY_HIGH

    def test_high(self):
        assert _classify_risk_tier(0.30) == UpsetRiskTier.HIGH

    def test_moderate(self):
        assert _classify_risk_tier(0.20) == UpsetRiskTier.MODERATE

    def test_low(self):
        assert _classify_risk_tier(0.10) == UpsetRiskTier.LOW

    def test_minimal(self):
        assert _classify_risk_tier(0.02) == UpsetRiskTier.MINIMAL

    def test_boundary_40(self):
        assert _classify_risk_tier(0.40) == UpsetRiskTier.VERY_HIGH

    def test_boundary_25(self):
        assert _classify_risk_tier(0.25) == UpsetRiskTier.HIGH


class TestHistoricalLookup:
    def test_known_matchup(self):
        rate = _get_historical_upset_rate(1, 16)
        assert rate == pytest.approx(0.007, abs=0.001)

    def test_5v12_classic(self):
        rate = _get_historical_upset_rate(5, 12)
        assert rate == pytest.approx(0.356, abs=0.001)

    def test_unknown_matchup_returns_none(self):
        assert _get_historical_upset_rate(6, 16) is None

    def test_logistic_fallback_basic(self):
        """Logistic fallback should produce reasonable values."""
        rate = _logistic_upset_fallback(1, 16)
        assert 0.0 < rate < 0.10  # very low for 1v16

    def test_logistic_fallback_close_seeds(self):
        rate = _logistic_upset_fallback(7, 8)
        assert 0.35 < rate < 0.55  # close to coin flip


class TestUpsetDetector:
    """Core upset detection logic tests."""

    @pytest.fixture
    def detector(self):
        return UpsetDetector(prior_strength=0.20, adjustment_strength=0.15)

    def test_basic_detection_1v16(self, detector):
        """1v16: model says 95% favorite, historical says 99.3%."""
        signal = detector.detect(
            "duke", "fairleigh", seed1=1, seed2=16,
            model_prob_team1=0.95,
        )
        assert signal.favorite_id == "duke"
        assert signal.underdog_id == "fairleigh"
        assert signal.upset_prob < 0.10  # Very low upset chance
        assert signal.risk_tier == UpsetRiskTier.LOW or signal.risk_tier == UpsetRiskTier.MINIMAL
        assert signal.adjusted_prob > 0.90  # Still strongly favor favorite

    def test_5v12_upset_special(self, detector):
        """5v12: the classic upset matchup. Model says 65% favorite."""
        signal = detector.detect(
            "memphis", "drake", seed1=5, seed2=12,
            model_prob_team1=0.65,
        )
        assert signal.favorite_id == "memphis"
        assert signal.underdog_id == "drake"
        # Historical 5v12 upset rate is ~36% — should pull model toward it
        assert signal.upset_prob > 0.30
        assert signal.risk_tier in (UpsetRiskTier.HIGH, UpsetRiskTier.VERY_HIGH, UpsetRiskTier.MODERATE)

    def test_8v9_near_coinflip(self, detector):
        """8v9: essentially a coin flip."""
        signal = detector.detect(
            "team_a", "team_b", seed1=8, seed2=9,
            model_prob_team1=0.52,
        )
        assert 0.40 < signal.upset_prob < 0.55
        assert signal.risk_tier in (UpsetRiskTier.VERY_HIGH, UpsetRiskTier.HIGH)

    def test_same_seed_handling(self, detector):
        """Same seed matchup should return neutral signal."""
        signal = detector.detect(
            "team_a", "team_b", seed1=3, seed2=3,
            model_prob_team1=0.55,
        )
        assert signal.historical_prior == 0.5
        assert signal.adjusted_prob == 0.55  # No adjustment for same seed
        assert signal.upset_score == 0.0

    def test_underdog_is_team1(self, detector):
        """When team1 is the underdog (higher seed number)."""
        signal = detector.detect(
            "cinderella", "duke", seed1=12, seed2=5,
            model_prob_team1=0.35,
        )
        assert signal.underdog_id == "cinderella"
        assert signal.favorite_id == "duke"
        # model_upset_prob should be 0.35 (team1 IS the underdog)
        assert signal.model_upset_prob == pytest.approx(0.35, abs=0.01)

    def test_adjusted_prob_symmetry(self, detector):
        """Swapping teams should produce complementary adjusted probs."""
        signal_fwd = detector.detect("a", "b", 3, 14, 0.85)
        signal_rev = detector.detect("b", "a", 14, 3, 0.15)
        # Upset probs should be very close (same matchup, same model view)
        assert signal_fwd.upset_prob == pytest.approx(signal_rev.upset_prob, abs=0.01)

    def test_features_improve_detection(self, detector):
        """Team features should shift upset probability when signals are strong."""
        # Underdog (12-seed) with great efficiency, good momentum
        dog_feats = MockTeamFeatures(
            seed=12, adj_efficiency_margin=8.0, momentum=5.0,
            avg_experience=3.0, three_pt_variance=0.06,
        )
        # Favorite (5-seed) with mediocre efficiency, declining
        fav_feats = MockTeamFeatures(
            seed=5, adj_efficiency_margin=6.0, momentum=-2.0,
            avg_experience=1.5, three_pt_variance=0.14,
        )

        # Without features
        signal_no_feats = detector.detect(
            "fav", "dog", seed1=5, seed2=12,
            model_prob_team1=0.65,
        )
        # With features
        signal_with_feats = detector.detect(
            "fav", "dog", seed1=5, seed2=12,
            model_prob_team1=0.65,
            team1_features=fav_feats, team2_features=dog_feats,
        )
        # Features should increase upset probability (underdog is better than seed implies)
        assert signal_with_feats.upset_prob > signal_no_feats.upset_prob

    def test_output_in_valid_range(self, detector):
        """All probability outputs must be in [0, 1]."""
        for s1, s2 in [(1, 16), (5, 12), (8, 9), (12, 5)]:
            for model_p in [0.01, 0.3, 0.5, 0.7, 0.99]:
                signal = detector.detect("a", "b", s1, s2, model_p)
                assert 0.0 <= signal.upset_prob <= 1.0
                assert 0.0 <= signal.adjusted_prob <= 1.0
                assert 0.0 <= signal.upset_score <= 1.0
                assert 0.0 <= signal.confidence <= 1.0


class TestDetectAll:
    def test_batch_detection(self):
        detector = UpsetDetector()
        matchups = [
            {"team1_id": "duke", "team2_id": "fairleigh", "seed1": 1, "seed2": 16, "model_prob_team1": 0.95},
            {"team1_id": "baylor", "team2_id": "drake", "seed1": 5, "seed2": 12, "model_prob_team1": 0.65},
            {"team1_id": "team_a", "team2_id": "team_b", "seed1": 8, "seed2": 9, "model_prob_team1": 0.52},
        ]
        signals = detector.detect_all(matchups)
        assert len(signals) == 3
        # Should be sorted by upset_score descending
        assert signals[0].upset_score >= signals[1].upset_score
        assert signals[1].upset_score >= signals[2].upset_score

    def test_batch_with_features(self):
        detector = UpsetDetector()
        features = {
            "duke": MockTeamFeatures(seed=1, adj_efficiency_margin=28.0),
            "cinderella": MockTeamFeatures(seed=16, adj_efficiency_margin=-25.0),
        }
        matchups = [
            {"team1_id": "duke", "team2_id": "cinderella", "seed1": 1, "seed2": 16, "model_prob_team1": 0.97},
        ]
        signals = detector.detect_all(matchups, team_features=features)
        assert len(signals) == 1
        # Duke has EM 28 vs seed-expected 28 (residual 0)
        # Cinderella has EM -25 vs seed-expected -21 (residual -4)
        # Underdog underperforms seed → efficiency_signal < 0.5
        assert signals[0].efficiency_signal < 0.5


class TestUpsetDetectorFit:
    def test_fit_improves_or_maintains_brier(self):
        """Fitting should not make predictions worse on the training set."""
        rng = np.random.default_rng(42)
        n = 100
        seeds1 = rng.integers(1, 9, size=n)
        seeds2 = 17 - seeds1  # Canonical R64 matchups
        # Model probabilities (team1 = favorite)
        model_probs = np.clip(0.5 + 0.3 * (seeds2 - seeds1) / 15.0 + rng.normal(0, 0.1, n), 0.05, 0.95)
        # Outcomes: favorites win most, but some upsets
        outcomes = (rng.random(n) < model_probs).astype(float)

        detector = UpsetDetector()
        stats = detector.fit(model_probs, outcomes, seeds1, seeds2)

        assert stats["fitted"]
        assert stats["brier_after"] <= stats["brier_before"] + 0.001  # Allow tiny tolerance

    def test_fit_with_few_samples(self):
        """Should handle few samples gracefully."""
        detector = UpsetDetector()
        stats = detector.fit(
            np.array([0.7, 0.3]),
            np.array([1.0, 0.0]),
            np.array([1, 8]),
            np.array([16, 9]),
        )
        assert stats["fitted"]
        assert stats["method"] == "default"


class TestUpsetSummary:
    def test_summary_structure(self):
        detector = UpsetDetector()
        matchups = [
            {"team1_id": f"team_{i}", "team2_id": f"opp_{i}",
             "seed1": i, "seed2": 17 - i, "model_prob_team1": 0.65}
            for i in range(1, 9)
        ]
        signals = detector.detect_all(matchups)
        summary = detector.get_upset_summary(signals)

        assert "total_matchups" in summary
        assert summary["total_matchups"] == 8
        assert "by_tier" in summary
        assert "top_upset_picks" in summary
        assert "expected_upsets_r64" in summary

    def test_expected_upsets_reasonable(self):
        """Expected R64 upset count should be in historical range (7-10)."""
        detector = UpsetDetector()
        matchups = [
            {"team1_id": f"fav_{i}", "team2_id": f"dog_{i}",
             "seed1": i, "seed2": 17 - i,
             "model_prob_team1": 0.5 + 0.03 * (17 - 2 * i)}
            for i in range(1, 9)
        ]
        # 4 regions × 8 matchups each = 32 R64 games; we have 8 here (1 region)
        signals = detector.detect_all(matchups)
        summary = detector.get_upset_summary(signals)
        expected = summary["expected_upsets_r64"]
        # For one region of 8 matchups, historical average is ~2-3 upsets
        assert 1.0 < expected < 5.0


class TestUpsetSignalSerialization:
    def test_to_dict(self):
        signal = UpsetSignal(
            team1_id="duke", team2_id="drake", seed1=5, seed2=12,
            underdog_id="drake", favorite_id="duke",
            upset_prob=0.356, historical_prior=0.356, model_upset_prob=0.35,
            adjusted_prob=0.644, risk_tier=UpsetRiskTier.HIGH,
            upset_score=0.45, confidence=0.7,
        )
        d = signal.to_dict()
        assert d["upset_prob"] == 0.356
        assert d["risk_tier"] == "HIGH"
        assert d["underdog_id"] == "drake"
        assert isinstance(d, dict)


class TestAmplifierSignals:
    """Test that amplifier signals respond correctly to feature differences."""

    @pytest.fixture
    def detector(self):
        return UpsetDetector()

    def test_volatile_favorite_increases_upset_risk(self, detector):
        """When the favorite has high variance, upset risk should increase."""
        stable_fav = MockTeamFeatures(seed=3, three_pt_variance=0.06, pace_adjusted_variance=5.0)
        volatile_fav = MockTeamFeatures(seed=3, three_pt_variance=0.15, pace_adjusted_variance=15.0)
        dog = MockTeamFeatures(seed=14, three_pt_variance=0.08, pace_adjusted_variance=5.0)

        signal_stable = detector.detect("fav", "dog", 3, 14, 0.85,
                                        team1_features=stable_fav, team2_features=dog)
        signal_volatile = detector.detect("fav", "dog", 3, 14, 0.85,
                                          team1_features=volatile_fav, team2_features=dog)
        assert signal_volatile.volatility_signal > signal_stable.volatility_signal
        assert signal_volatile.upset_prob > signal_stable.upset_prob

    def test_hot_underdog_momentum(self, detector):
        """Underdog with strong momentum should have higher upset prob."""
        fav = MockTeamFeatures(seed=4, momentum=-1.0)
        cold_dog = MockTeamFeatures(seed=13, momentum=-2.0)
        hot_dog = MockTeamFeatures(seed=13, momentum=6.0)

        signal_cold = detector.detect("fav", "dog", 4, 13, 0.80,
                                      team1_features=fav, team2_features=cold_dog)
        signal_hot = detector.detect("fav", "dog", 4, 13, 0.80,
                                     team1_features=fav, team2_features=hot_dog)
        assert signal_hot.momentum_signal > signal_cold.momentum_signal

    def test_experienced_underdog(self, detector):
        """Experienced underdog roster should increase upset potential."""
        fav = MockTeamFeatures(seed=2, avg_experience=1.5)
        young_dog = MockTeamFeatures(seed=15, avg_experience=1.5)
        veteran_dog = MockTeamFeatures(seed=15, avg_experience=3.5)

        signal_young = detector.detect("fav", "dog", 2, 15, 0.92,
                                       team1_features=fav, team2_features=young_dog)
        signal_vet = detector.detect("fav", "dog", 2, 15, 0.92,
                                     team1_features=fav, team2_features=veteran_dog)
        assert signal_vet.experience_signal > signal_young.experience_signal
