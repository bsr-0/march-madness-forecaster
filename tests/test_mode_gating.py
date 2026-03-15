"""Tests for mode-gating: calibration vs EV mode pipeline separation.

Validates that each mode only computes mode-relevant work:
- Calibration mode: pool analysis, leverage preview, bracket portfolio, Brier rubric
- EV mode: defers game theory to _build_ev_analysis, skips bracket portfolio

These tests use a mock pipeline to avoid needing real data files.
"""

import pytest
from unittest.mock import MagicMock, patch
from src.pipeline.sota import SOTAPipelineConfig, EVModeReport


class TestModeGatingConfig:
    """Verify config-level mode separation."""

    def test_calibration_mode_default(self):
        config = SOTAPipelineConfig()
        assert config.mode == "calibration"

    def test_ev_mode_has_pool_size(self):
        config = SOTAPipelineConfig(mode="ev", ev_pool_size=200, enforce_production_path=False)
        assert config.ev_pool_size == 200

    def test_calibration_ignores_ev_pool_size(self):
        """EV params exist but aren't validated in calibration mode."""
        config = SOTAPipelineConfig(mode="calibration", ev_pool_size=0)
        assert config.ev_pool_size == 0


class TestEVModeReportGating:
    """Verify EVModeReport includes competition_simulation field."""

    def test_ev_report_has_competition_simulation(self):
        report = EVModeReport()
        d = report.to_dict()
        assert "competition_simulation" in d
        assert d["competition_simulation"] == {}

    def test_ev_report_populated_competition_simulation(self):
        report = EVModeReport(
            competition_simulation={
                "best_bracket_id": "pareto_2",
                "effective_pool_size": 100,
                "n_tournaments": 1000,
            },
        )
        d = report.to_dict()
        assert d["competition_simulation"]["best_bracket_id"] == "pareto_2"
        assert d["competition_simulation"]["effective_pool_size"] == 100

    def test_ev_report_win_probabilities_separate_from_sim(self):
        """win_probabilities and competition_simulation are separate fields."""
        report = EVModeReport(
            win_probabilities={"top_5pct": 0.12},
            competition_simulation={"n_tournaments": 1000},
        )
        d = report.to_dict()
        assert d["win_probabilities"]["top_5pct"] == 0.12
        assert d["competition_simulation"]["n_tournaments"] == 1000


class TestModeGatingDispatch:
    """Test that run() dispatches to correct mode handler."""

    def test_calibration_mode_calls_calibration_method(self):
        """Calibration mode should call _run_calibration_mode."""
        from src.pipeline.sota import SOTAPipeline
        pipeline = SOTAPipeline.__new__(SOTAPipeline)
        pipeline.config = SOTAPipelineConfig(mode="calibration")

        pipeline._run_calibration_mode = MagicMock(return_value={"mode": "calibration"})
        pipeline._run_ev_mode = MagicMock(return_value={"mode": "ev"})

        result = pipeline.run()

        pipeline._run_calibration_mode.assert_called_once()
        pipeline._run_ev_mode.assert_not_called()
        assert result["mode"] == "calibration"

    def test_ev_mode_calls_ev_method(self):
        """EV mode should call _run_ev_mode."""
        from src.pipeline.sota import SOTAPipeline
        pipeline = SOTAPipeline.__new__(SOTAPipeline)
        pipeline.config = SOTAPipelineConfig(mode="ev", enforce_production_path=False)

        pipeline._run_calibration_mode = MagicMock(return_value={"mode": "calibration"})
        pipeline._run_ev_mode = MagicMock(return_value={"mode": "ev"})

        result = pipeline.run()

        pipeline._run_ev_mode.assert_called_once()
        pipeline._run_calibration_mode.assert_not_called()
        assert result["mode"] == "ev"


class TestReportModeField:
    """Test that the report contains a mode field."""

    def test_ev_report_to_dict_has_mode(self):
        report = EVModeReport(pool_size=100)
        d = report.to_dict()
        assert d["mode"] == "ev"


class TestRoundProbabilitiesFromSim:
    """Test _to_round_probabilities_from_sim helper."""

    def test_extracts_all_rounds(self):
        from src.pipeline.sota import SOTAPipeline
        pipeline = SOTAPipeline.__new__(SOTAPipeline)
        pipeline.team_struct = {
            "duke": MagicMock(name="Duke", seed=1, region="East"),
            "unc": MagicMock(name="UNC", seed=2, region="East"),
        }

        sim_data = {
            "championship_odds": {"duke": 0.15, "unc": 0.08},
            "final_four_odds": {"duke": 0.30, "unc": 0.20},
            "elite_eight_odds": {"duke": 0.55, "unc": 0.40},
            "sweet_sixteen_odds": {"duke": 0.75, "unc": 0.60},
            "round_of_32_odds": {"duke": 0.90, "unc": 0.85},
        }

        probs = pipeline._to_round_probabilities_from_sim(sim_data)

        assert probs["duke"]["R64"] == 1.0
        assert probs["duke"]["R32"] == 0.90
        assert probs["duke"]["S16"] == 0.75
        assert probs["duke"]["E8"] == 0.55
        assert probs["duke"]["F4"] == 0.30
        assert probs["duke"]["CHAMP"] == 0.15

        assert probs["unc"]["R64"] == 1.0
        assert probs["unc"]["CHAMP"] == 0.08

    def test_handles_missing_teams(self):
        """Teams in sim_data but not in team_struct should still appear."""
        from src.pipeline.sota import SOTAPipeline
        pipeline = SOTAPipeline.__new__(SOTAPipeline)
        pipeline.team_struct = {}

        sim_data = {
            "championship_odds": {"duke": 0.15},
            "final_four_odds": {},
            "elite_eight_odds": {},
            "sweet_sixteen_odds": {},
            "round_of_32_odds": {},
        }

        probs = pipeline._to_round_probabilities_from_sim(sim_data)
        assert "duke" in probs
        assert probs["duke"]["CHAMP"] == 0.15
        assert probs["duke"]["F4"] == 0.0  # Missing → 0.0

    def test_handles_empty_sim_data(self):
        from src.pipeline.sota import SOTAPipeline
        pipeline = SOTAPipeline.__new__(SOTAPipeline)
        pipeline.team_struct = {}

        probs = pipeline._to_round_probabilities_from_sim({})
        assert probs == {}


class TestChalkBracketFallback:
    """Test the static chalk bracket generator."""

    def test_chalk_picks_favorites(self):
        from src.pipeline.sota import SOTAPipeline

        first_round = ["A", "B", "C", "D"]
        matchup_probs = {
            ("A", "B"): 0.9,
            ("B", "A"): 0.1,
            ("C", "D"): 0.3,
            ("D", "C"): 0.7,
        }

        winners = SOTAPipeline._generate_chalk_winners(first_round, matchup_probs)

        # R64: A beats B (p=0.9), D beats C (p=0.7)
        assert winners[0] == "A"
        assert winners[1] == "D"
        # Final: A vs D
        assert len(winners) == 3  # 2 R64 + 1 final

    def test_chalk_handles_even_matchup(self):
        from src.pipeline.sota import SOTAPipeline

        first_round = ["A", "B"]
        matchup_probs = {("A", "B"): 0.5, ("B", "A"): 0.5}

        winners = SOTAPipeline._generate_chalk_winners(first_round, matchup_probs)
        assert len(winners) == 1
        # At exactly 0.5, picks t1 (A) per the >= 0.5 logic
        assert winners[0] == "A"


class TestCalibrationModeReportStructure:
    """Verify calibration report contains Kaggle-specific fields."""

    def test_calibration_report_expected_keys(self):
        """The rubric_evaluation phase_4 should have Kaggle-specific fields
        when in calibration mode (verified via config, not full pipeline run)."""
        config = SOTAPipelineConfig(mode="calibration")
        # In calibration mode, phase_4_game_theory should have:
        # public_consensus, leverage_ratio, pareto_front
        # We can't run the full pipeline, but verify config is correct
        assert config.mode == "calibration"
        # These fields are only set in calibration mode:
        assert hasattr(config, "pool_size")
        assert hasattr(config, "enable_bracket_portfolio")


class TestEVModeReportStructure:
    """Verify EV report contains all expected keys."""

    def test_ev_report_has_all_keys(self):
        report = EVModeReport(
            pool_size=100,
            scoring_system="standard",
            target_percentile=0.05,
        )
        d = report.to_dict()
        required_keys = {
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
            "competition_simulation",
            "pareto_brackets",
            "pool_ev_analysis",
        }
        assert set(d.keys()) == required_keys
