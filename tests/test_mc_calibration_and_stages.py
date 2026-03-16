"""Tests for mc_calibration helpers and pipeline stage metadata.

Covers:
- MCCalibrationResult dataclass and to_dict()
- _is_tournament_game date window logic
- _resolve_seed_id normalisation and matching
- _predict_fn_factory logistic prediction
- _build_bracket_teams region filtering
- FIRST_ROUND_MATCHUPS constant
- CalibrationStage / ModelTrainingStage / ReportingStage .name attributes
- _noop_ctx helpers
"""

from __future__ import annotations

import math
from contextlib import nullcontext

import pytest

from src.simulation.mc_calibration import (
    MCCalibrationResult,
    _is_tournament_game,
    _resolve_seed_id,
    _predict_fn_factory,
    _build_bracket_teams,
    FIRST_ROUND_MATCHUPS,
)
from src.pipeline.stages.calibrator import CalibrationStage, _noop_ctx as cal_noop
from src.pipeline.stages.model_trainer import ModelTrainingStage, _noop_ctx as model_noop
from src.pipeline.stages.reporter import ReportingStage


# ---------------------------------------------------------------------------
# FIRST_ROUND_MATCHUPS
# ---------------------------------------------------------------------------

class TestFirstRoundMatchups:
    def test_contains_all_eight_pairings(self):
        expected = {(1, 16), (2, 15), (3, 14), (4, 13),
                    (5, 12), (6, 11), (7, 10), (8, 9)}
        assert FIRST_ROUND_MATCHUPS == expected

    def test_is_a_set(self):
        assert isinstance(FIRST_ROUND_MATCHUPS, set)

    def test_each_pair_sums_to_17(self):
        for hi, lo in FIRST_ROUND_MATCHUPS:
            assert hi + lo == 17


# ---------------------------------------------------------------------------
# MCCalibrationResult
# ---------------------------------------------------------------------------

class TestMCCalibrationResult:
    def _make(self, **overrides):
        defaults = dict(
            noise_std=0.123456,
            regional_correlation=0.05678,
            dev_score=0.04567890,
            holdout_score=0.05432100,
            per_year_scores={2021: 0.04111, 2022: 0.05222},
        )
        defaults.update(overrides)
        return MCCalibrationResult(**defaults)

    def test_fields_stored(self):
        r = self._make()
        assert r.noise_std == pytest.approx(0.123456)
        assert r.regional_correlation == pytest.approx(0.05678)

    def test_to_dict_keys(self):
        d = self._make().to_dict()
        assert set(d.keys()) == {
            "noise_std", "regional_correlation",
            "dev_score", "holdout_score", "per_year_scores",
        }

    def test_to_dict_rounds_noise_std(self):
        d = self._make(noise_std=0.123456).to_dict()
        assert d["noise_std"] == round(0.123456, 4)

    def test_to_dict_rounds_regional_correlation(self):
        d = self._make(regional_correlation=0.056789).to_dict()
        assert d["regional_correlation"] == round(0.056789, 4)

    def test_to_dict_rounds_dev_score(self):
        d = self._make(dev_score=0.0456789).to_dict()
        assert d["dev_score"] == round(0.0456789, 5)

    def test_to_dict_rounds_holdout_score(self):
        d = self._make(holdout_score=0.0543219).to_dict()
        assert d["holdout_score"] == round(0.0543219, 5)

    def test_to_dict_holdout_none(self):
        d = self._make(holdout_score=None).to_dict()
        assert d["holdout_score"] is None

    def test_to_dict_per_year_keys_are_strings(self):
        d = self._make(per_year_scores={2021: 0.05}).to_dict()
        assert all(isinstance(k, str) for k in d["per_year_scores"])

    def test_to_dict_per_year_values_rounded(self):
        d = self._make(per_year_scores={2021: 0.0411111}).to_dict()
        assert d["per_year_scores"]["2021"] == round(0.0411111, 5)

    def test_to_dict_empty_per_year(self):
        d = self._make(per_year_scores={}).to_dict()
        assert d["per_year_scores"] == {}


# ---------------------------------------------------------------------------
# _is_tournament_game
# ---------------------------------------------------------------------------

class TestIsTournamentGame:
    def test_valid_tournament_date(self):
        # 2023 tournament starts 2023-03-14, ends window 2023-04-15
        assert _is_tournament_game("2023-03-17", 2023) is True

    def test_date_before_tournament(self):
        assert _is_tournament_game("2023-01-15", 2023) is False

    def test_date_after_tournament(self):
        assert _is_tournament_game("2023-04-20", 2023) is False

    def test_invalid_date_returns_false(self):
        assert _is_tournament_game("not-a-date", 2023) is False

    def test_empty_string_returns_false(self):
        assert _is_tournament_game("", 2023) is False

    def test_start_date_inclusive(self):
        # The start date itself should be in the window
        assert _is_tournament_game("2023-03-14", 2023) is True

    def test_end_date_inclusive(self):
        # April 15 is the end boundary
        assert _is_tournament_game("2023-04-15", 2023) is True

    def test_day_before_start_is_false(self):
        assert _is_tournament_game("2023-03-13", 2023) is False

    def test_partial_date_returns_false(self):
        assert _is_tournament_game("2023-03", 2023) is False


# ---------------------------------------------------------------------------
# _resolve_seed_id
# ---------------------------------------------------------------------------

class TestResolveSeedId:
    def test_exact_match(self):
        ids = ["alabama", "duke", "kansas"]
        assert _resolve_seed_id("alabama", ids) == "alabama"

    def test_prefix_match(self):
        # If normalized raw_id starts with a seed id, it should match
        ids_sorted = sorted(["north_carolina", "duke"], key=len, reverse=True)
        # "north_carolina_extra" after normalize -> "north_carolina_extra"
        # starts with "north_carolina"
        result = _resolve_seed_id("north_carolina_extra", ids_sorted)
        assert result == "north_carolina"

    def test_no_match(self):
        ids = ["alabama", "duke"]
        assert _resolve_seed_id("xyz_unknown_team", ids) is None

    def test_empty_string_returns_none(self):
        ids = ["alabama", "duke"]
        assert _resolve_seed_id("", ids) is None

    def test_empty_seed_list(self):
        assert _resolve_seed_id("alabama", []) is None

    def test_longer_prefix_preferred_when_sorted(self):
        # When seed_ids_sorted is sorted by length descending, the longer
        # matching prefix should be returned first.
        ids_sorted = sorted(["north", "north_carolina"], key=len, reverse=True)
        result = _resolve_seed_id("north_carolina_tar_heels", ids_sorted)
        assert result == "north_carolina"


# ---------------------------------------------------------------------------
# _predict_fn_factory
# ---------------------------------------------------------------------------

class TestPredictFnFactory:
    def _simple_setup(self):
        seed_info = {
            "team_a": {"seed": 1, "region": "East"},
            "team_b": {"seed": 16, "region": "East"},
        }
        strengths = {"team_a": 10.0, "team_b": -5.0}
        return seed_info, strengths

    def test_returns_callable(self):
        si, st = self._simple_setup()
        fn = _predict_fn_factory(si, st, em_slope=0.1, seed_slope=0.2)
        assert callable(fn)

    def test_output_between_0_and_1(self):
        si, st = self._simple_setup()
        fn = _predict_fn_factory(si, st, em_slope=0.1, seed_slope=0.2)
        p = fn("team_a", "team_b")
        assert 0.0 < p < 1.0

    def test_stronger_team_favored(self):
        si, st = self._simple_setup()
        fn = _predict_fn_factory(si, st, em_slope=0.1, seed_slope=0.2)
        p = fn("team_a", "team_b")
        assert p > 0.5

    def test_logistic_value_with_strengths(self):
        si, st = self._simple_setup()
        em_slope = 0.2
        fn = _predict_fn_factory(si, st, em_slope=em_slope, seed_slope=0.1)
        p = fn("team_a", "team_b")
        diff = 10.0 - (-5.0)  # 15.0
        expected = 1.0 / (1.0 + math.exp(-em_slope * diff))
        assert p == pytest.approx(expected)

    def test_unknown_teams_no_strengths_uses_seed(self):
        si = {
            "team_x": {"seed": 2, "region": "West"},
            "team_y": {"seed": 15, "region": "West"},
        }
        fn = _predict_fn_factory(si, {}, em_slope=0.1, seed_slope=0.3)
        p = fn("team_x", "team_y")
        seed_diff = 15 - 2  # 13
        expected = 1.0 / (1.0 + math.exp(-0.3 * seed_diff))
        assert p == pytest.approx(expected)

    def test_both_unknown_returns_half(self):
        fn = _predict_fn_factory({}, {}, em_slope=0.1, seed_slope=0.2)
        p = fn("no_team_1", "no_team_2")
        assert p == pytest.approx(0.5)

    def test_symmetric_around_equal_strengths(self):
        si = {
            "a": {"seed": 8, "region": "South"},
            "b": {"seed": 9, "region": "South"},
        }
        strengths = {"a": 5.0, "b": 5.0}
        fn = _predict_fn_factory(si, strengths, em_slope=0.1, seed_slope=0.2)
        p = fn("a", "b")
        assert p == pytest.approx(0.5)

    def test_reverse_order_gives_complement(self):
        si, st = self._simple_setup()
        fn = _predict_fn_factory(si, st, em_slope=0.15, seed_slope=0.2)
        p_ab = fn("team_a", "team_b")
        p_ba = fn("team_b", "team_a")
        assert p_ab + p_ba == pytest.approx(1.0)

    def test_one_strength_missing_falls_back_to_seed(self):
        si = {
            "team_a": {"seed": 3, "region": "East"},
            "team_b": {"seed": 14, "region": "East"},
        }
        # Only team_a has a strength; team_b does not -> fallback to seed
        strengths = {"team_a": 10.0}
        fn = _predict_fn_factory(si, strengths, em_slope=0.1, seed_slope=0.25)
        p = fn("team_a", "team_b")
        seed_diff = 14 - 3  # 11
        expected = 1.0 / (1.0 + math.exp(-0.25 * seed_diff))
        assert p == pytest.approx(expected)


# ---------------------------------------------------------------------------
# _build_bracket_teams
# ---------------------------------------------------------------------------

def _make_seed_info(regions, seeds_per_region=range(1, 17)):
    """Helper to build seed_info with specified regions and seeds."""
    info = {}
    for region in regions:
        for seed in seeds_per_region:
            tid = f"{region.lower()}_seed{seed}"
            info[tid] = {"seed": seed, "region": region}
    return info


class TestBuildBracketTeams:
    def test_four_complete_regions(self):
        regions = ["East", "West", "South", "Midwest"]
        si = _make_seed_info(regions)
        team_ids = sorted(si.keys())
        result = _build_bracket_teams(si, team_ids)
        assert len(result) == 4
        for region in regions:
            assert region in result
            assert len(result[region]) == 16

    def test_teams_sorted_by_seed(self):
        si = _make_seed_info(["East"])
        team_ids = sorted(si.keys())
        result = _build_bracket_teams(si, team_ids)
        seeds = [t.seed for t in result["East"]]
        assert seeds == list(range(1, 17))

    def test_incomplete_region_filtered(self):
        # Region with only seeds 1-15 should be excluded
        si = _make_seed_info(["East"], seeds_per_region=range(1, 16))
        team_ids = sorted(si.keys())
        result = _build_bracket_teams(si, team_ids)
        assert len(result) == 0

    def test_mix_complete_and_incomplete(self):
        si_complete = _make_seed_info(["East"])
        si_incomplete = _make_seed_info(["West"], seeds_per_region=range(1, 10))
        si = {**si_complete, **si_incomplete}
        team_ids = sorted(si.keys())
        result = _build_bracket_teams(si, team_ids)
        assert "East" in result
        assert "West" not in result

    def test_empty_team_ids(self):
        si = _make_seed_info(["East"])
        result = _build_bracket_teams(si, [])
        assert result == {}

    def test_team_not_in_seed_info_skipped(self):
        si = _make_seed_info(["East"])
        team_ids = sorted(si.keys()) + ["unknown_team"]
        result = _build_bracket_teams(si, team_ids)
        # Should still work; unknown_team is simply ignored
        assert len(result) == 1

    def test_returned_teams_have_correct_attributes(self):
        si = _make_seed_info(["South"])
        team_ids = sorted(si.keys())
        result = _build_bracket_teams(si, team_ids)
        team = result["South"][0]
        assert team.region == "South"
        assert team.seed == 1
        assert team.strength == 0.5  # default strength


# ---------------------------------------------------------------------------
# Pipeline stage names
# ---------------------------------------------------------------------------

class TestCalibrationStage:
    def test_name(self):
        assert CalibrationStage.name == "calibration"

    def test_noop_ctx_returns_context_manager(self):
        ctx = cal_noop()
        # Should be usable as a context manager
        with ctx:
            pass  # no error

    def test_noop_ctx_is_nullcontext(self):
        ctx = cal_noop()
        assert isinstance(ctx, type(nullcontext()))


class TestModelTrainingStage:
    def test_name(self):
        assert ModelTrainingStage.name == "model_training"

    def test_noop_ctx_returns_context_manager(self):
        ctx = model_noop()
        with ctx:
            pass

    def test_noop_ctx_is_nullcontext(self):
        ctx = model_noop()
        assert isinstance(ctx, type(nullcontext()))


class TestReportingStage:
    def test_name(self):
        assert ReportingStage.name == "reporting"

    def test_instance_name(self):
        stage = ReportingStage()
        assert stage.name == "reporting"
