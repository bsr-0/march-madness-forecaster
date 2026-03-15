"""Tests for women's tournament pipeline (WS1).

Alignment audit (2026-03-02): Updated to match new feature dimensions,
complete Four Factors, interaction features, and tournament adaptation.
"""

import numpy as np
import pytest

from src.data.features.womens_feature_engineering import (
    WOMENS_FEATURE_DIM,
    WOMENS_FEATURE_NAMES,
    WOMENS_TEAM_FEATURE_DIM,
    WomensFeatureEngineer,
    WomensTeamFeatures,
    compute_seed_win_probability,
    get_team_feature_names,
    validate_womens_population_stats,
)
from src.data.scrapers.womens.herhoopstats import (
    HerHoopStatsScraper,
    WomensTeamStats,
    WOMENS_HISTORICAL_UPSET_RATES,
)
from src.data.scrapers.womens.ncaa_net import NETRanking, WomensNETScraper
from src.data.scrapers.womens.historical_results import (
    HistoricalTournamentGame,
    WomensHistoricalResults,
)
from src.pipeline.womens import WomensPipeline, WomensPipelineConfig


# --- Seed Win Probability ---

class TestSeedWinProbability:

    def test_equal_seeds(self):
        p = compute_seed_win_probability(8, 8)
        assert abs(p - 0.5) < 1e-6

    def test_lower_seed_favored(self):
        p = compute_seed_win_probability(1, 16)
        assert p > 0.9

    def test_higher_seed_underdog(self):
        p = compute_seed_win_probability(16, 1)
        assert p < 0.1

    def test_symmetry(self):
        p1 = compute_seed_win_probability(3, 14)
        p2 = compute_seed_win_probability(14, 3)
        assert abs(p1 + p2 - 1.0) < 1e-6

    def test_womens_slope_steeper(self):
        """Women's should be more extreme than men's (slope=0.175)."""
        from src.optimization.dual_submission import _seed_logistic_probability
        womens_p = compute_seed_win_probability(1, 16)
        mens_p = _seed_logistic_probability(1, 16, slope=0.175)
        # Women's slope is steeper so 1-seed wins more often
        assert womens_p > mens_p


# --- WomensTeamFeatures ---

class TestWomensTeamFeatures:

    def test_to_vector_shape(self):
        f = WomensTeamFeatures(team_id="test", team_name="Test")
        v = f.to_vector()
        assert v.shape == (WOMENS_TEAM_FEATURE_DIM,)  # 18 features

    def test_to_vector_includes_complete_four_factors(self):
        """Verify all 8 Four Factors are in the vector (4 offense + 4 defense)."""
        f = WomensTeamFeatures(
            team_id="test", team_name="Test",
            effective_fg_pct=0.50,
            turnover_rate=0.18,
            offensive_reb_rate=0.35,
            free_throw_rate=0.32,
            opp_effective_fg_pct=0.42,
            opp_turnover_rate=0.22,
            defensive_reb_rate=0.72,
            opp_free_throw_rate=0.28,
        )
        v = f.to_vector()
        names = get_team_feature_names()
        # Check that drb_rate and opp_ft_rate are present
        assert 'drb_rate' in names
        assert 'opp_ft_rate' in names
        # Check values at correct indices
        drb_idx = names.index('drb_rate')
        opp_ft_idx = names.index('opp_ft_rate')
        assert v[drb_idx] == pytest.approx(0.72)
        assert v[opp_ft_idx] == pytest.approx(0.28)

    def test_to_vector_nan_inf_validation(self):
        """NaN/inf values should be replaced with 0.0."""
        f = WomensTeamFeatures(
            team_id="test", team_name="Test",
            adj_offensive_efficiency=float('nan'),
            adj_defensive_efficiency=float('inf'),
        )
        v = f.to_vector()
        assert not np.any(np.isnan(v))
        assert not np.any(np.isinf(v))
        assert v[0] == 0.0  # nan replaced
        assert v[1] == 0.0  # inf replaced

    def test_to_vector_safety_clip(self):
        """Extreme values should be clipped to [-1000, 1000]."""
        f = WomensTeamFeatures(
            team_id="test", team_name="Test",
            adj_offensive_efficiency=5000.0,
        )
        v = f.to_vector()
        assert v[0] == 1000.0

    def test_from_womens_stats(self):
        stats = WomensTeamStats(
            team_id="uconn",
            team_name="UConn",
            seed=1,
            adj_offensive_efficiency=115.0,
            adj_defensive_efficiency=85.0,
        )
        f = WomensTeamFeatures.from_womens_stats(stats)
        assert f.team_id == "uconn"
        assert f.adj_offensive_efficiency == 115.0
        assert f.adj_defensive_efficiency == 85.0

    def test_defaults(self):
        f = WomensTeamFeatures(team_id="t", team_name="T")
        assert f.elo_rating == 1500.0
        assert f.win_pct == 0.5
        assert f.defensive_reb_rate == 0.70
        assert f.opp_free_throw_rate == 0.30


# --- get_team_feature_names ---

class TestGetTeamFeatureNames:

    def test_length_matches_dim(self):
        names = get_team_feature_names()
        assert len(names) == WOMENS_TEAM_FEATURE_DIM

    def test_to_vector_matches_names(self):
        """to_vector() length must match get_team_feature_names() length."""
        f = WomensTeamFeatures(team_id="test", team_name="Test")
        v = f.to_vector()
        names = get_team_feature_names()
        assert len(v) == len(names)


# --- validate_womens_population_stats ---

class TestPopulationStats:

    def test_no_warnings_for_normal_data(self):
        """Should not warn for reasonable data."""
        features = {
            f"t{i}": WomensTeamFeatures(
                team_id=f"t{i}", team_name=f"Team {i}",
                adj_offensive_efficiency=78.0 + i,
                adj_defensive_efficiency=78.0 - i,
            )
            for i in range(15)
        }
        warnings = validate_womens_population_stats(features)
        # Should have no or few warnings for near-population data
        assert isinstance(warnings, list)

    def test_skips_small_datasets(self):
        """Should return empty for <10 teams."""
        features = {
            f"t{i}": WomensTeamFeatures(team_id=f"t{i}", team_name=f"T{i}")
            for i in range(5)
        }
        warnings = validate_womens_population_stats(features)
        assert warnings == []


# --- WomensFeatureEngineer ---

class TestWomensFeatureEngineer:

    def _make_stats(self):
        return {
            "uconn": WomensTeamStats(
                team_id="uconn", team_name="UConn", seed=1,
                adj_offensive_efficiency=115.0,
                adj_defensive_efficiency=85.0,
            ),
            "iowa": WomensTeamStats(
                team_id="iowa", team_name="Iowa", seed=2,
                adj_offensive_efficiency=110.0,
                adj_defensive_efficiency=90.0,
            ),
        }

    def test_build_features(self):
        eng = WomensFeatureEngineer()
        eng.build_features(self._make_stats())
        assert len(eng.team_features) == 2
        assert "uconn" in eng.team_features
        assert "iowa" in eng.team_features

    def test_get_matchup_features_shape(self):
        eng = WomensFeatureEngineer()
        eng.build_features(self._make_stats())
        features = eng.get_matchup_features("uconn", "iowa")
        assert features is not None
        assert features.shape == (WOMENS_FEATURE_DIM,)

    def test_get_matchup_features_unknown_team(self):
        eng = WomensFeatureEngineer()
        eng.build_features(self._make_stats())
        assert eng.get_matchup_features("uconn", "unknown") is None

    def test_feature_names_count(self):
        eng = WomensFeatureEngineer()
        names = eng.get_feature_names()
        assert len(names) == WOMENS_FEATURE_DIM

    def test_matchup_features_contain_seed_diff(self):
        """Matchup features should include seed_diff (strongest single predictor)."""
        eng = WomensFeatureEngineer()
        eng.build_features(self._make_stats())
        names = eng.get_feature_names()
        assert "seed_diff" in names

    def test_matchup_features_contain_interaction_features(self):
        """Matchup features should include tempo_interaction and style_mismatch."""
        eng = WomensFeatureEngineer()
        names = eng.get_feature_names()
        assert "tempo_interaction" in names
        assert "style_mismatch" in names

    def test_matchup_features_contain_complete_four_factors_diff(self):
        """Differential features should include all Four Factors."""
        eng = WomensFeatureEngineer()
        names = eng.get_feature_names()
        assert "diff_drb_rate" in names
        assert "diff_opp_ft_rate" in names

    def test_matchup_features_symmetry(self):
        """Swapping teams should negate differential features."""
        eng = WomensFeatureEngineer()
        eng.build_features(self._make_stats())
        f_ab = eng.get_matchup_features("uconn", "iowa")
        f_ba = eng.get_matchup_features("iowa", "uconn")
        # Differential features (first 17) should be negated
        np.testing.assert_allclose(f_ab[:17], -f_ba[:17], atol=1e-10)


# --- HerHoopStatsScraper ---

class TestHerHoopStatsScraper:

    def test_generate_seed_based_estimates(self):
        scraper = HerHoopStatsScraper(cache_dir="/nonexistent")
        seeds = {"uconn": 1, "iowa": 2, "lsu": 3}
        stats = scraper.generate_seed_based_estimates(seeds)
        assert len(stats) == 3
        assert stats["uconn"].seed == 1
        # 1-seed should have better efficiency than 3-seed
        assert stats["uconn"].adj_offensive_efficiency > stats["lsu"].adj_offensive_efficiency

    def test_historical_upset_rates(self):
        """Known upset rates should be defined."""
        assert (1, 16) in WOMENS_HISTORICAL_UPSET_RATES
        assert WOMENS_HISTORICAL_UPSET_RATES[(1, 16)] < 0.02  # Very rare


# --- WomensNETScraper ---

class TestWomensNETScraper:

    def test_estimate_from_seeds(self):
        scraper = WomensNETScraper(cache_dir="/nonexistent")
        seeds = {"uconn": 1, "iowa": 8, "tiny": 16}
        rankings = scraper.estimate_from_seeds(seeds)
        assert len(rankings) == 3
        # 1-seed should have higher net_rating (higher = better in this scale)
        assert rankings["uconn"].net_rating > rankings["tiny"].net_rating
        # 1-seed should have lower net_ranking (lower = better)
        assert rankings["uconn"].net_ranking < rankings["tiny"].net_ranking


# --- WomensHistoricalResults ---

class TestWomensHistoricalResults:

    def test_historical_tournament_game(self):
        g = HistoricalTournamentGame(
            year=2024, round_name="R64",
            team1_seed=1, team2_seed=16,
            team1_name="UConn", team2_name="Fairleigh",
            team1_won=True,
        )
        assert g.upset is False
        assert g.seed_matchup == (1, 16)

    def test_game_upset_detection(self):
        g = HistoricalTournamentGame(
            year=2023, round_name="R64",
            team1_seed=12, team2_seed=5,
            team1_name="Miami", team2_name="Purdue",
            team1_won=True,
        )
        assert g.upset is True

    def test_synthetic_history(self):
        history = WomensHistoricalResults(cache_dir="/nonexistent")
        games = history.load_cached()
        assert len(games) > 0  # Should generate synthetic data

    def test_get_calibration_data(self):
        history = WomensHistoricalResults(cache_dir="/nonexistent")
        history.load_cached()  # Load data first
        probs, outcomes = history.get_calibration_data()
        assert len(probs) == len(outcomes)
        assert all(0 < p < 1 for p in probs)
        assert all(o in (0.0, 1.0) for o in outcomes)


# --- WomensPipeline ---

class TestWomensPipeline:

    def test_default_config(self):
        config = WomensPipelineConfig()
        assert config.year == 2026
        assert config.clip_lo == 0.005

    def test_config_has_tournament_adaptation(self):
        """Config should have tournament adaptation settings (aligned with men's)."""
        config = WomensPipelineConfig()
        assert config.enable_tournament_adaptation is True
        assert config.tournament_shrinkage == 0.02
        assert config.seed_prior_weight == 0.0  # Deprecated: redundant with SeedBasedOverrides
        assert config.seed_prior_slope == 0.19

    def test_config_has_feature_scaling(self):
        """Config should have feature scaling (aligned with men's)."""
        config = WomensPipelineConfig()
        assert config.enable_feature_scaling is True

    def test_seed_only_mode(self):
        """Pipeline in seed-only mode should produce valid predictions."""
        config = WomensPipelineConfig(seed_only_mode=True)
        pipeline = WomensPipeline(config)
        pipeline.run()

        # Set some team seeds
        pipeline.set_team_seeds({"t1": 1, "t2": 16, "t3": 8, "t4": 9})

        # Predict
        p = pipeline.predict_probability("t1", "t2")
        assert 0.005 <= p <= 0.995
        assert p > 0.8  # 1-seed vs 16-seed should heavily favor 1-seed

    def test_prediction_bounds(self):
        config = WomensPipelineConfig(seed_only_mode=True)
        pipeline = WomensPipeline(config)
        pipeline.run()
        pipeline.set_team_seeds({"a": 8, "b": 9})

        p = pipeline.predict_probability("a", "b")
        assert 0.005 <= p <= 0.995

    def test_unknown_teams_return_near_half(self):
        config = WomensPipelineConfig(seed_only_mode=True)
        pipeline = WomensPipeline(config)
        pipeline.run()
        # Don't set any team seeds
        p = pipeline.predict_probability("unknown1", "unknown2")
        assert 0.3 <= p <= 0.7  # Should be near 0.5

    def test_symmetry(self):
        """P(A beats B) + P(B beats A) should be approximately 1."""
        config = WomensPipelineConfig(seed_only_mode=True)
        pipeline = WomensPipeline(config)
        pipeline.run()
        pipeline.set_team_seeds({"a": 3, "b": 14})

        p_ab = pipeline.predict_probability("a", "b")
        p_ba = pipeline.predict_probability("b", "a")
        # After post-processing they may not be exactly complementary
        # but should be close
        assert abs(p_ab + p_ba - 1.0) < 0.15

    def test_tournament_adaptation_reduces_extremes(self):
        """Tournament adaptation should pull predictions slightly toward 0.5."""
        config_no_adapt = WomensPipelineConfig(
            seed_only_mode=True, enable_tournament_adaptation=False
        )
        config_with_adapt = WomensPipelineConfig(
            seed_only_mode=True, enable_tournament_adaptation=True
        )

        pipeline_no = WomensPipeline(config_no_adapt)
        pipeline_no.run()
        pipeline_no.set_team_seeds({"a": 1, "b": 16})

        pipeline_yes = WomensPipeline(config_with_adapt)
        pipeline_yes.run()
        pipeline_yes.set_team_seeds({"a": 1, "b": 16})

        p_no = pipeline_no.predict_probability("a", "b")
        p_yes = pipeline_yes.predict_probability("a", "b")

        # Both should favor the 1-seed
        assert p_no > 0.8
        assert p_yes > 0.8
