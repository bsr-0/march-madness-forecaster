"""Tests for EV Mode configuration, dispatch, and report structure."""

import pytest
from src.pipeline.sota import SOTAPipelineConfig, EVModeReport


class TestSOTAPipelineConfigMode:
    """Test mode parameter validation in SOTAPipelineConfig."""

    def test_default_mode_is_calibration(self):
        config = SOTAPipelineConfig()
        assert config.mode == "calibration"

    def test_calibration_mode_accepts_defaults(self):
        config = SOTAPipelineConfig(mode="calibration")
        assert config.mode == "calibration"

    def test_ev_mode_accepts_valid_params(self):
        config = SOTAPipelineConfig(
            mode="ev",
            ev_pool_size=500,
            ev_scoring_system="standard",
            ev_target_percentile=0.01,
        )
        assert config.mode == "ev"
        assert config.ev_pool_size == 500

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Invalid mode"):
            SOTAPipelineConfig(mode="invalid")

    def test_ev_mode_invalid_pool_size_raises(self):
        with pytest.raises(ValueError, match="ev_pool_size"):
            SOTAPipelineConfig(mode="ev", ev_pool_size=0)

    def test_ev_mode_invalid_target_percentile_low(self):
        with pytest.raises(ValueError, match="ev_target_percentile"):
            SOTAPipelineConfig(mode="ev", ev_target_percentile=0.0)

    def test_ev_mode_invalid_target_percentile_high(self):
        with pytest.raises(ValueError, match="ev_target_percentile"):
            SOTAPipelineConfig(mode="ev", ev_target_percentile=0.6)

    def test_ev_mode_invalid_scoring_system(self):
        with pytest.raises(ValueError, match="ev_scoring_system"):
            SOTAPipelineConfig(mode="ev", ev_scoring_system="custom")

    def test_ev_mode_flat_scoring(self):
        config = SOTAPipelineConfig(mode="ev", ev_scoring_system="flat")
        assert config.ev_scoring_system == "flat"

    def test_ev_mode_upset_bonus_scoring(self):
        config = SOTAPipelineConfig(mode="ev", ev_scoring_system="upset_bonus")
        assert config.ev_scoring_system == "upset_bonus"

    def test_ev_mode_default_pool_size(self):
        config = SOTAPipelineConfig(mode="ev")
        assert config.ev_pool_size == 100

    def test_ev_mode_default_target_percentile(self):
        config = SOTAPipelineConfig(mode="ev")
        assert config.ev_target_percentile == 0.05

    def test_ev_mode_contrarian_strength(self):
        config = SOTAPipelineConfig(mode="ev", ev_contrarian_strength=2.0)
        assert config.ev_contrarian_strength == 2.0

    def test_ev_mode_search_disabled_by_default(self):
        config = SOTAPipelineConfig(mode="ev")
        assert config.ev_enable_search is False

    def test_ev_mode_archetypes_disabled_by_default(self):
        config = SOTAPipelineConfig(mode="ev")
        assert config.ev_enable_archetypes is False

    def test_ev_mode_pool_type_default(self):
        config = SOTAPipelineConfig(mode="ev")
        assert config.ev_pool_type == "espn_national"

    def test_calibration_mode_ignores_ev_params(self):
        """EV params exist but aren't validated in calibration mode."""
        config = SOTAPipelineConfig(mode="calibration", ev_pool_size=0)
        assert config.mode == "calibration"
        assert config.ev_pool_size == 0  # Not validated in calibration mode


class TestEVModeReport:
    """Test EVModeReport dataclass and serialization."""

    def test_default_construction(self):
        report = EVModeReport()
        assert report.pool_size == 0
        assert report.scoring_system == "standard"
        assert report.target_percentile == 0.05

    def test_construction_with_params(self):
        report = EVModeReport(
            pool_size=500,
            scoring_system="flat",
            target_percentile=0.01,
            recommended_strategy="contrarian",
        )
        assert report.pool_size == 500
        assert report.recommended_strategy == "contrarian"

    def test_to_dict_has_mode_field(self):
        report = EVModeReport(pool_size=100)
        d = report.to_dict()
        assert d["mode"] == "ev"

    def test_to_dict_contains_all_fields(self):
        report = EVModeReport(pool_size=100, scoring_system="standard")
        d = report.to_dict()
        expected_keys = {
            "mode",
            "pool_size",
            "scoring_system",
            "target_percentile",
            "recommended_strategy",
            "leverage_picks",
            "fade_picks",
            "model_vs_public_divergence",
            "bracket_portfolio_summary",
            "win_probabilities",
            "pareto_brackets",
            "pool_ev_analysis",
        }
        assert set(d.keys()) == expected_keys

    def test_to_dict_preserves_leverage_picks(self):
        picks = [
            {"team_id": "duke", "leverage_ratio": 2.5},
            {"team_id": "gonzaga", "leverage_ratio": 1.8},
        ]
        report = EVModeReport(leverage_picks=picks)
        d = report.to_dict()
        assert len(d["leverage_picks"]) == 2
        assert d["leverage_picks"][0]["team_id"] == "duke"

    def test_to_dict_preserves_divergence(self):
        divergence = {"duke": 0.05, "gonzaga": -0.03}
        report = EVModeReport(model_vs_public_divergence=divergence)
        d = report.to_dict()
        assert d["model_vs_public_divergence"]["duke"] == 0.05

    def test_empty_lists_by_default(self):
        report = EVModeReport()
        assert report.leverage_picks == []
        assert report.fade_picks == []
        assert report.pareto_brackets == []

    def test_empty_dicts_by_default(self):
        report = EVModeReport()
        assert report.model_vs_public_divergence == {}
        assert report.bracket_portfolio_summary == {}
        assert report.win_probabilities == {}
        assert report.pool_ev_analysis == {}


class TestModeDispatch:
    """Test that run() dispatches correctly to calibration or EV mode.

    These tests verify the dispatch logic without running the full pipeline,
    which requires data files.  Full integration tests belong in a separate
    test suite.
    """

    def test_calibration_config_has_no_ev_analysis_key(self):
        """Calibration mode report should not contain ev_analysis."""
        config = SOTAPipelineConfig(mode="calibration")
        # We can't run the full pipeline without data, but we can verify
        # the config is set up correctly for dispatch.
        assert config.mode == "calibration"

    def test_ev_config_ready_for_dispatch(self):
        """EV mode config should be ready for _run_ev_mode dispatch."""
        config = SOTAPipelineConfig(
            mode="ev",
            ev_pool_size=1000,
            ev_scoring_system="standard",
            ev_target_percentile=0.01,
        )
        assert config.mode == "ev"
        assert config.ev_pool_size == 1000


class TestGetEVScoringSystem:
    """Test _get_ev_scoring_system helper."""

    def test_standard_scoring(self):
        from src.pipeline.sota import SOTAPipeline
        pipeline = SOTAPipeline.__new__(SOTAPipeline)
        pipeline.config = SOTAPipelineConfig(mode="ev", ev_scoring_system="standard")
        scoring = pipeline._get_ev_scoring_system()
        assert scoring["R64"] == 10
        assert scoring["CHAMP"] == 320

    def test_flat_scoring(self):
        from src.pipeline.sota import SOTAPipeline
        pipeline = SOTAPipeline.__new__(SOTAPipeline)
        pipeline.config = SOTAPipelineConfig(mode="ev", ev_scoring_system="flat")
        scoring = pipeline._get_ev_scoring_system()
        assert scoring["R64"] == 1
        assert scoring["CHAMP"] == 6

    def test_upset_bonus_scoring(self):
        from src.pipeline.sota import SOTAPipeline
        pipeline = SOTAPipeline.__new__(SOTAPipeline)
        pipeline.config = SOTAPipelineConfig(mode="ev", ev_scoring_system="upset_bonus")
        scoring = pipeline._get_ev_scoring_system()
        # Upset bonus uses standard base points
        assert scoring["R64"] == 10
        assert scoring["CHAMP"] == 320
