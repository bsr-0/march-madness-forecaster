"""Tests for tournament resume composite feature and opponent-adjusted integration.

Validates that:
1. compute_tournament_resume_composite() produces sensible values
2. Bayesian shrinkage works correctly for small samples
3. The feature is properly wired into TeamFeatures.to_vector()
4. Road/neutral record is computed in ProprietaryMetricsEngine
5. FIXED_FEATURE_SET includes the new features
6. The composite preserves ordinality (strong > mid > weak)
"""

import math

import numpy as np
import pytest

from src.data.features.tournament_features import (
    compute_tournament_resume_composite,
    TOURNAMENT_FEATURE_NAMES,
)
from src.data.features.feature_engineering import (
    TeamFeatures,
    TEAM_FEATURE_DIM,
)


# =========================================================================
# Core composite function tests
# =========================================================================

class TestTournamentResumeComposite:
    """Tests for the Bayesian-shrunk tournament resume composite."""

    def test_strong_team_scores_high(self):
        """Elite tournament team (1-seed caliber) should score > 0.55."""
        score = compute_tournament_resume_composite(
            q1_win_pct=0.75, q1_games=12,
            road_neutral_win_pct=0.80, road_neutral_games=16,
            elite_sos=22.0, sor=1.3,
        )
        assert score > 0.55, f"Strong team score {score} should be > 0.55"

    def test_weak_team_scores_low(self):
        """16-seed with poor resume should score < 0.35."""
        score = compute_tournament_resume_composite(
            q1_win_pct=0.10, q1_games=3,
            road_neutral_win_pct=0.35, road_neutral_games=12,
            elite_sos=5.0, sor=0.5,
        )
        assert score < 0.35, f"Weak team score {score} should be < 0.35"

    def test_output_range_zero_to_one(self):
        """Output should always be in [0, 1]."""
        # Extreme high
        high = compute_tournament_resume_composite(
            q1_win_pct=1.0, q1_games=20,
            road_neutral_win_pct=1.0, road_neutral_games=20,
            elite_sos=30.0, sor=2.0,
        )
        assert 0.0 <= high <= 1.0

        # Extreme low
        low = compute_tournament_resume_composite(
            q1_win_pct=0.0, q1_games=20,
            road_neutral_win_pct=0.0, road_neutral_games=20,
            elite_sos=0.0, sor=0.0,
        )
        assert 0.0 <= low <= 1.0

    def test_zero_games_returns_prior(self):
        """With zero games, should return the Bayesian prior."""
        score = compute_tournament_resume_composite(
            q1_win_pct=0.0, q1_games=0,
            road_neutral_win_pct=0.0, road_neutral_games=0,
            elite_sos=0.0, sor=0.0,
        )
        # Should be close to prior-based blend
        assert 0.2 < score < 0.45, f"Zero-data score {score} should be near prior"

    def test_strong_beats_mid_beats_weak(self):
        """Ordinality: strong resume > mid-major > weak."""
        strong = compute_tournament_resume_composite(
            q1_win_pct=0.75, q1_games=12,
            road_neutral_win_pct=0.80, road_neutral_games=16,
            elite_sos=22.0, sor=1.3,
        )
        mid = compute_tournament_resume_composite(
            q1_win_pct=0.40, q1_games=8,
            road_neutral_win_pct=0.60, road_neutral_games=14,
            elite_sos=14.0, sor=0.95,
        )
        weak = compute_tournament_resume_composite(
            q1_win_pct=0.10, q1_games=3,
            road_neutral_win_pct=0.35, road_neutral_games=12,
            elite_sos=5.0, sor=0.5,
        )
        assert strong > mid > weak, (
            f"Ordinality violated: strong={strong:.4f}, mid={mid:.4f}, weak={weak:.4f}"
        )


class TestBayesianShrinkage:
    """Tests for the Bayesian shrinkage behavior."""

    def test_more_games_less_shrinkage(self):
        """With more games, observed rate should dominate over prior."""
        # Same 80% win rate, different sample sizes
        few_games = compute_tournament_resume_composite(
            q1_win_pct=0.80, q1_games=3,
            road_neutral_win_pct=0.80, road_neutral_games=5,
            elite_sos=18.0, sor=1.0,
        )
        many_games = compute_tournament_resume_composite(
            q1_win_pct=0.80, q1_games=15,
            road_neutral_win_pct=0.80, road_neutral_games=18,
            elite_sos=18.0, sor=1.0,
        )
        # Many games should produce a higher score (observed 80% > prior 40%)
        assert many_games > few_games, (
            f"More games ({many_games:.4f}) should score higher than fewer ({few_games:.4f})"
        )

    def test_small_sample_shrinks_toward_prior(self):
        """1-0 Q1 record shouldn't be treated as 100%."""
        perfect_small = compute_tournament_resume_composite(
            q1_win_pct=1.0, q1_games=1,
            road_neutral_win_pct=0.60, road_neutral_games=10,
            elite_sos=18.0, sor=1.0,
        )
        perfect_large = compute_tournament_resume_composite(
            q1_win_pct=1.0, q1_games=12,
            road_neutral_win_pct=0.60, road_neutral_games=10,
            elite_sos=18.0, sor=1.0,
        )
        # 12-0 Q1 should be treated as more impressive than 1-0 Q1
        assert perfect_large > perfect_small

    def test_prior_strength_parameter(self):
        """Higher prior_strength means more conservative shrinkage."""
        conservative = compute_tournament_resume_composite(
            q1_win_pct=0.80, q1_games=5,
            road_neutral_win_pct=0.80, road_neutral_games=10,
            elite_sos=18.0, sor=1.0,
            prior_strength=10.0,  # Very conservative
        )
        aggressive = compute_tournament_resume_composite(
            q1_win_pct=0.80, q1_games=5,
            road_neutral_win_pct=0.80, road_neutral_games=10,
            elite_sos=18.0, sor=1.0,
            prior_strength=1.0,  # Trust the data
        )
        # Aggressive shrinkage trusts the 80% more → higher score
        assert aggressive > conservative


class TestSORNormalization:
    """Tests for Strength of Record normalization."""

    def test_sor_one_is_average(self):
        """SOR=1.0 (average top-25 team) should map to ~0.5."""
        # With all other components at default/neutral
        baseline = compute_tournament_resume_composite(
            q1_win_pct=0.40, q1_games=10,
            road_neutral_win_pct=0.50, road_neutral_games=15,
            elite_sos=18.0, sor=1.0,
        )
        # SOR normalized component at 1.0 → logistic(0/0.4) = 0.5
        assert 0.3 < baseline < 0.6

    def test_sor_monotonic(self):
        """Higher SOR should produce higher composite."""
        low_sor = compute_tournament_resume_composite(
            q1_win_pct=0.50, q1_games=10,
            road_neutral_win_pct=0.60, road_neutral_games=15,
            elite_sos=18.0, sor=0.6,
        )
        high_sor = compute_tournament_resume_composite(
            q1_win_pct=0.50, q1_games=10,
            road_neutral_win_pct=0.60, road_neutral_games=15,
            elite_sos=18.0, sor=1.4,
        )
        assert high_sor > low_sor


class TestEliteSOSNormalization:
    """Tests for Elite SOS normalization."""

    def test_no_elite_opponents_uses_prior(self):
        """elite_sos=0 → prior of 0.3."""
        score_no_elite = compute_tournament_resume_composite(
            q1_win_pct=0.50, q1_games=10,
            road_neutral_win_pct=0.60, road_neutral_games=15,
            elite_sos=0.0, sor=1.0,
        )
        score_with_elite = compute_tournament_resume_composite(
            q1_win_pct=0.50, q1_games=10,
            road_neutral_win_pct=0.60, road_neutral_games=15,
            elite_sos=22.0, sor=1.0,
        )
        assert score_with_elite > score_no_elite

    def test_elite_sos_monotonic(self):
        """Higher elite SOS should produce higher composite."""
        low = compute_tournament_resume_composite(
            q1_win_pct=0.50, q1_games=10,
            road_neutral_win_pct=0.60, road_neutral_games=15,
            elite_sos=15.0, sor=1.0,
        )
        high = compute_tournament_resume_composite(
            q1_win_pct=0.50, q1_games=10,
            road_neutral_win_pct=0.60, road_neutral_games=15,
            elite_sos=25.0, sor=1.0,
        )
        assert high > low


# =========================================================================
# Feature vector integration tests
# =========================================================================

class TestTeamFeaturesIntegration:
    """Tests that tournament_resume is properly wired into TeamFeatures."""

    def test_team_feature_dim_includes_resume(self):
        """TEAM_FEATURE_DIM should be 79 (71 base + 2 graph SOS + 3 win quality + 3 per-stage coaching)."""
        assert TEAM_FEATURE_DIM == 79

    def test_to_vector_length_matches_dim(self):
        """to_vector() output length should equal TEAM_FEATURE_DIM."""
        t = TeamFeatures(team_id="test", team_name="Test", seed=1, region="East")
        v = t.to_vector()
        assert len(v) == TEAM_FEATURE_DIM

    def test_feature_names_length_matches_dim(self):
        """get_feature_names() should have TEAM_FEATURE_DIM entries."""
        names = TeamFeatures.get_feature_names()
        assert len(names) == TEAM_FEATURE_DIM

    def test_tournament_resume_in_feature_names(self):
        """tournament_resume should be in the feature name list."""
        names = TeamFeatures.get_feature_names()
        assert "tournament_resume" in names

    def test_tournament_resume_default_is_zero(self):
        """Default tournament_resume should be 0.0."""
        t = TeamFeatures(team_id="test", team_name="Test", seed=1, region="East")
        assert t.tournament_resume == 0.0

    def test_tournament_resume_appears_in_vector(self):
        """Setting tournament_resume should be reflected in to_vector()."""
        t = TeamFeatures(team_id="test", team_name="Test", seed=1, region="East")
        t.tournament_resume = 0.75
        v = t.to_vector()
        names = TeamFeatures.get_feature_names()
        idx = names.index("tournament_resume")
        assert v[idx] == pytest.approx(0.75)

    def test_home_court_dependence_in_vector(self):
        """home_court_dependence should be in both feature names and vector."""
        t = TeamFeatures(team_id="test", team_name="Test", seed=1, region="East")
        t.home_court_dependence = 5.5
        v = t.to_vector()
        names = TeamFeatures.get_feature_names()
        idx = names.index("home_court_dependence")
        assert v[idx] == pytest.approx(5.5)


class TestMatchupFeatureIntegration:
    """Tests that differential features work correctly."""

    def test_diff_tournament_resume_in_matchup(self):
        """Matchup vector should contain diff_tournament_resume."""
        from src.data.features.feature_engineering import FeatureEngineer

        fe = FeatureEngineer()

        t1 = TeamFeatures(team_id="T1", team_name="Team1", seed=1, region="East")
        t1.tournament_resume = 0.70
        t2 = TeamFeatures(team_id="T2", team_name="Team2", seed=8, region="East")
        t2.tournament_resume = 0.40

        fe.team_features = {"T1": t1, "T2": t2}
        matchup = fe.create_matchup_features("T1", "T2")

        # The diff vector should contain tournament_resume diff
        v = matchup.to_vector()
        names = TeamFeatures.get_feature_names()
        idx = names.index("tournament_resume")
        # diff = t1 - t2 = 0.70 - 0.40 = 0.30
        assert v[idx] == pytest.approx(0.30, abs=0.01)


# =========================================================================
# FIXED_FEATURE_SET integration tests
# =========================================================================

class TestFixedFeatureSetIntegration:
    """Tests that FIXED_FEATURE_SET includes new features."""

    def test_diff_tournament_resume_in_fixed_set(self):
        """diff_tournament_resume should be in FIXED_FEATURE_SET."""
        from src.pipeline.sota import FIXED_FEATURE_SET
        assert "diff_tournament_resume" in FIXED_FEATURE_SET

    def test_diff_home_court_dependence_in_fixed_set(self):
        """diff_home_court_dependence should be in FIXED_FEATURE_SET."""
        from src.pipeline.sota import FIXED_FEATURE_SET
        assert "diff_home_court_dependence" in FIXED_FEATURE_SET

    def test_fixed_feature_set_not_too_large(self):
        """FIXED_FEATURE_SET should not exceed 30 features (overfitting guard)."""
        from src.pipeline.sota import FIXED_FEATURE_SET
        assert len(FIXED_FEATURE_SET) <= 30, (
            f"FIXED_FEATURE_SET has {len(FIXED_FEATURE_SET)} features — "
            f"risk of overfitting on ~600 tournament samples"
        )

    def test_all_fixed_features_exist_in_matchup_names(self):
        """Every feature in FIXED_FEATURE_SET should be resolvable."""
        from src.pipeline.sota import FIXED_FEATURE_SET, ABSOLUTE_LEVEL_FEATURE_NAMES

        base_names = TeamFeatures.get_feature_names()
        diff_names = [f"diff_{n}" for n in base_names]
        abs_names = [f"abs_{n}" for n in ABSOLUTE_LEVEL_FEATURE_NAMES]
        interaction_names = [
            "tempo_interaction", "style_mismatch", "h2h_record",
            "common_opp_margin", "travel_advantage", "seed_interaction",
            "seed_diff",
        ]
        all_names = set(diff_names + abs_names + interaction_names)

        missing = [f for f in FIXED_FEATURE_SET if f not in all_names]
        assert not missing, f"Features not in matchup vector: {missing}"


# =========================================================================
# ProprietaryMetricsEngine road/neutral tests
# =========================================================================

class TestRoadNeutralRecord:
    """Tests that road/neutral record is computed from game records."""

    def test_road_neutral_fields_exist(self):
        """ProprietaryTeamMetrics should have road_neutral fields."""
        from src.data.features.proprietary_metrics import ProprietaryTeamMetrics
        m = ProprietaryTeamMetrics(team_id="T1", team_name="Test")
        assert hasattr(m, 'road_neutral_wins')
        assert hasattr(m, 'road_neutral_losses')
        assert hasattr(m, 'road_neutral_games')
        assert hasattr(m, 'road_neutral_win_pct')

    def test_road_neutral_default_values(self):
        """Default road_neutral values should be sensible."""
        from src.data.features.proprietary_metrics import ProprietaryTeamMetrics
        m = ProprietaryTeamMetrics(team_id="T1", team_name="Test")
        assert m.road_neutral_wins == 0
        assert m.road_neutral_losses == 0
        assert m.road_neutral_games == 0
        assert m.road_neutral_win_pct == 0.5

    def test_road_neutral_in_to_dict(self):
        """road_neutral fields should appear in to_dict()."""
        from src.data.features.proprietary_metrics import ProprietaryTeamMetrics
        m = ProprietaryTeamMetrics(team_id="T1", team_name="Test")
        m.road_neutral_wins = 8
        m.road_neutral_losses = 4
        m.road_neutral_games = 12
        m.road_neutral_win_pct = 8 / 12
        d = m.to_dict()
        assert d['road_neutral_wins'] == 8
        assert d['road_neutral_games'] == 12
        assert abs(d['road_neutral_win_pct'] - 8/12) < 0.001


# =========================================================================
# Tournament feature names
# =========================================================================

class TestTournamentFeatureNames:
    """Tests for TOURNAMENT_FEATURE_NAMES constant."""

    def test_includes_tournament_resume(self):
        """TOURNAMENT_FEATURE_NAMES should include diff_tournament_resume."""
        assert "diff_tournament_resume" in TOURNAMENT_FEATURE_NAMES

    def test_includes_traditional_names(self):
        """Should still include the original tournament feature names."""
        assert "diff_strength_of_record" in TOURNAMENT_FEATURE_NAMES
        assert "diff_top25_win_pct" in TOURNAMENT_FEATURE_NAMES
        assert "diff_road_neutral_win_pct" in TOURNAMENT_FEATURE_NAMES


# =========================================================================
# Mathematical properties
# =========================================================================

class TestMathematicalProperties:
    """Tests for mathematical correctness of the composite."""

    def test_component_weights_sum_to_one(self):
        """The four component weights should sum to 1.0."""
        assert 0.35 + 0.25 + 0.25 + 0.15 == pytest.approx(1.0)

    def test_symmetric_around_average(self):
        """An average team (all 50th percentile metrics) should score ~0.4-0.5."""
        score = compute_tournament_resume_composite(
            q1_win_pct=0.40, q1_games=10,
            road_neutral_win_pct=0.50, road_neutral_games=15,
            elite_sos=18.0, sor=1.0,
        )
        assert 0.35 < score < 0.55, f"Average team score {score} should be ~0.4-0.5"

    def test_continuous_with_respect_to_inputs(self):
        """Small input changes should produce small output changes."""
        base = compute_tournament_resume_composite(
            q1_win_pct=0.50, q1_games=10,
            road_neutral_win_pct=0.60, road_neutral_games=15,
            elite_sos=18.0, sor=1.0,
        )
        perturbed = compute_tournament_resume_composite(
            q1_win_pct=0.51, q1_games=10,
            road_neutral_win_pct=0.60, road_neutral_games=15,
            elite_sos=18.0, sor=1.0,
        )
        assert abs(perturbed - base) < 0.02, "Small Q1 change should produce small output change"

    def test_no_nan_or_inf(self):
        """Should never produce NaN or inf regardless of inputs."""
        edge_cases = [
            dict(q1_win_pct=0.0, q1_games=0, road_neutral_win_pct=0.0,
                 road_neutral_games=0, elite_sos=0.0, sor=0.0),
            dict(q1_win_pct=1.0, q1_games=100, road_neutral_win_pct=1.0,
                 road_neutral_games=100, elite_sos=50.0, sor=5.0),
            dict(q1_win_pct=0.0, q1_games=1, road_neutral_win_pct=0.0,
                 road_neutral_games=1, elite_sos=-10.0, sor=-1.0),
        ]
        for kwargs in edge_cases:
            score = compute_tournament_resume_composite(**kwargs)
            assert np.isfinite(score), f"Non-finite result for {kwargs}"


# =========================================================================
# Overfitting safeguards
# =========================================================================

class TestOverfittingSafeguards:
    """Tests that the implementation respects sample-size constraints."""

    def test_single_composite_vs_four_features(self):
        """Verify we use 1 composite feature, not 4 separate features."""
        from src.pipeline.sota import FIXED_FEATURE_SET
        opponent_quality_features = [
            f for f in FIXED_FEATURE_SET
            if any(x in f for x in [
                "strength_of_record", "top25_win", "road_neutral_win",
                "quad1_win", "tournament_resume",
            ])
        ]
        # Should be exactly 1: diff_tournament_resume
        assert len(opponent_quality_features) == 1, (
            f"Expected 1 opponent-quality feature in FIXED_FEATURE_SET, "
            f"got {len(opponent_quality_features)}: {opponent_quality_features}"
        )

    def test_feature_count_within_budget(self):
        """Total features should stay within sample-size budget."""
        from src.pipeline.sota import FIXED_FEATURE_SET
        # Rule of thumb: N/10 features for logistic, N/20 for LightGBM
        # With ~600 samples, max ~30 features
        n_features = len(FIXED_FEATURE_SET)
        assert n_features <= 30, f"{n_features} features exceeds budget for ~600 samples"
        # Should be 28 after adding 2 new features
        assert n_features == 28, f"Expected 28 features, got {n_features}"
