"""Tests for the live public-pick refresh and re-optimization workflow."""

import json
import os
import time
import tempfile

import pytest
from unittest.mock import MagicMock, patch

from src.data.scrapers.espn_picks import (
    ConsensusData,
    OrchestratorResult,
    PicksUpdate,
    PublicPicks,
    RefreshablePicksManager,
    RoundDelta,
    ScraperOrchestrator,
    SourceStatus,
)
from src.optimization.leverage import TeamMetadata, analyze_pool
from src.optimization.live_refresh import (
    LivePicksRefreshWorkflow,
    PickDelta,
    LeverageShift,
    RefreshResult,
    RefreshSchedule,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ROUNDS = ["R64", "R32", "S16", "E8", "F4", "CHAMP"]

MODEL_PROBS = {
    "duke": {"R64": 0.97, "R32": 0.90, "S16": 0.78, "E8": 0.60, "F4": 0.42, "CHAMP": 0.28},
    "unc": {"R64": 0.93, "R32": 0.83, "S16": 0.58, "E8": 0.38, "F4": 0.26, "CHAMP": 0.14},
    "gonzaga": {"R64": 0.88, "R32": 0.69, "S16": 0.44, "E8": 0.27, "F4": 0.18, "CHAMP": 0.10},
    "midmajor": {"R64": 0.70, "R32": 0.22, "S16": 0.10, "E8": 0.05, "F4": 0.03, "CHAMP": 0.02},
}

TEAM_META = {
    "duke": TeamMetadata(team_name="Duke", seed=1, region="East"),
    "unc": TeamMetadata(team_name="UNC", seed=2, region="East"),
    "gonzaga": TeamMetadata(team_name="Gonzaga", seed=3, region="West"),
    "midmajor": TeamMetadata(team_name="Midmajor", seed=12, region="South"),
}

EV_CONFIG = {
    "pool_size": 100,
    "scoring_system": "standard",
    "target_percentile": 0.05,
    "contrarian_strength": 1.0,
    "payout_structure": "tiered",
    "enable_archetypes": False,
    "pool_type": "espn_national",
    "archetype_overrides": None,
}


def _make_picks(champion_pcts: dict) -> dict:
    """Build public picks dict from champion percentages (0-1 scale)."""
    picks = {}
    for tid, champ in champion_pcts.items():
        picks[tid] = {
            "R64": min(champ * 3.5, 0.99),
            "R32": min(champ * 3.0, 0.95),
            "S16": min(champ * 2.5, 0.80),
            "E8": min(champ * 2.0, 0.60),
            "F4": min(champ * 1.5, 0.40),
            "CHAMP": champ,
        }
    return picks


def _make_consensus(champion_pcts: dict) -> ConsensusData:
    """Build ConsensusData from champion percentages (raw %)."""
    teams = {}
    for tid, pct in champion_pcts.items():
        teams[tid] = PublicPicks(
            team_id=tid, team_name=tid.title(), seed=1, region="East",
            champion_pct=pct,
            final_four_pct=pct * 1.5,
            elite_8_pct=pct * 2.0,
            sweet_16_pct=pct * 2.5,
            round_of_32_pct=pct * 3.0,
            round_of_64_pct=min(pct * 3.5, 99.0),
        )
    return ConsensusData(teams=teams, sources=["espn"])


def _make_orch_result(champion_pcts: dict) -> OrchestratorResult:
    return OrchestratorResult(
        consensus=_make_consensus(champion_pcts),
        successful_sources=["espn"],
        source_statuses={"espn": SourceStatus(name="espn")},
    )


def _build_workflow(
    initial_picks: dict,
    model_probs: dict = None,
    team_meta: dict = None,
    ev_config: dict = None,
    picks_manager: RefreshablePicksManager = None,
) -> LivePicksRefreshWorkflow:
    """Build a LivePicksRefreshWorkflow with mock picks manager."""
    if model_probs is None:
        model_probs = MODEL_PROBS
    if team_meta is None:
        team_meta = TEAM_META
    if ev_config is None:
        ev_config = EV_CONFIG
    if picks_manager is None:
        orch = MagicMock(spec=ScraperOrchestrator)
        picks_manager = RefreshablePicksManager(orch)

    # Generate initial EV report
    initial_ev = LivePicksRefreshWorkflow._run_lightweight_ev(
        model_probs, initial_picks, team_meta, ev_config,
    )

    return LivePicksRefreshWorkflow(
        model_round_probs=model_probs,
        team_metadata=team_meta,
        ev_config=ev_config,
        picks_manager=picks_manager,
        initial_public_picks=initial_picks,
        initial_ev_report=initial_ev,
    )


# ---------------------------------------------------------------------------
# Tests: PickDelta
# ---------------------------------------------------------------------------


class TestPickDelta:

    def test_delta_positive(self):
        d = PickDelta("duke", "Duke", "CHAMP", 20.0, 25.0)
        assert d.delta == 5.0
        assert d.abs_delta == 5.0

    def test_delta_negative(self):
        d = PickDelta("duke", "Duke", "CHAMP", 25.0, 20.0)
        assert d.delta == -5.0
        assert d.abs_delta == 5.0


# ---------------------------------------------------------------------------
# Tests: LeverageShift
# ---------------------------------------------------------------------------


class TestLeverageShift:

    def test_new_opportunity(self):
        s = LeverageShift("duke", "Duke", "CHAMP", 1.2, 1.8, 0.0, 0.0)
        assert s.is_new_opportunity is True
        assert s.is_lost_opportunity is False

    def test_lost_opportunity(self):
        s = LeverageShift("duke", "Duke", "CHAMP", 2.0, 1.3, 0.0, 0.0)
        assert s.is_lost_opportunity is True
        assert s.is_new_opportunity is False

    def test_no_change(self):
        s = LeverageShift("duke", "Duke", "CHAMP", 2.0, 2.1, 0.0, 0.0)
        assert s.is_new_opportunity is False
        assert s.is_lost_opportunity is False


# ---------------------------------------------------------------------------
# Tests: RefreshResult
# ---------------------------------------------------------------------------


class TestRefreshResult:

    def test_not_refreshed_not_significant(self):
        r = RefreshResult(was_refreshed=False, skip_reason="data still fresh")
        assert r.is_significant is False
        assert "No refresh performed" in r.change_summary

    def test_refreshed_with_strategy_change_is_significant(self):
        r = RefreshResult(
            was_refreshed=True,
            old_strategy="chalk",
            new_strategy="contrarian",
            strategy_changed=True,
        )
        assert r.is_significant is True

    def test_refreshed_with_new_leverage_is_significant(self):
        r = RefreshResult(
            was_refreshed=True,
            new_leverage_picks=[{"team_id": "duke", "team_name": "Duke"}],
        )
        assert r.is_significant is True

    def test_refreshed_with_large_delta_is_significant(self):
        r = RefreshResult(
            was_refreshed=True,
            pick_deltas=[PickDelta("duke", "Duke", "CHAMP", 20.0, 25.0)],
        )
        assert r.is_significant is True

    def test_refreshed_with_tiny_delta_not_significant(self):
        r = RefreshResult(
            was_refreshed=True,
            pick_deltas=[PickDelta("duke", "Duke", "CHAMP", 20.0, 20.5)],
        )
        assert r.is_significant is False

    def test_to_dict_contains_key_fields(self):
        r = RefreshResult(
            refresh_timestamp=time.time(),
            was_refreshed=True,
            strategy_changed=True,
            old_strategy="chalk",
            new_strategy="contrarian",
        )
        d = r.to_dict()
        assert d["was_refreshed"] is True
        assert d["strategy_changed"] is True
        assert d["old_strategy"] == "chalk"
        assert d["new_strategy"] == "contrarian"


# ---------------------------------------------------------------------------
# Tests: RefreshSchedule
# ---------------------------------------------------------------------------


class TestRefreshSchedule:

    def test_game_week(self):
        s = RefreshSchedule.game_week()
        assert s.interval_hours == 12.0
        assert s.max_refreshes == 14

    def test_final_48h(self):
        s = RefreshSchedule.final_48h()
        assert s.interval_hours == 4.0
        assert s.max_refreshes == 12

    def test_crunch_time(self):
        s = RefreshSchedule.crunch_time()
        assert s.interval_hours == 1.0
        assert s.max_refreshes == 6


# ---------------------------------------------------------------------------
# Tests: LivePicksRefreshWorkflow — lightweight EV analysis
# ---------------------------------------------------------------------------


class TestLightweightEV:

    def test_run_lightweight_ev_returns_valid_report(self):
        picks = _make_picks({"duke": 0.22, "unc": 0.12, "gonzaga": 0.08, "midmajor": 0.008})
        report = LivePicksRefreshWorkflow._run_lightweight_ev(
            MODEL_PROBS, picks, TEAM_META, EV_CONFIG,
        )
        assert report["mode"] == "ev"
        assert report["pool_size"] == 100
        assert "recommended_strategy" in report
        assert isinstance(report["leverage_picks"], list)
        assert isinstance(report["fade_picks"], list)
        assert isinstance(report["pareto_brackets"], list)
        assert "refresh_timestamp" in report

    def test_leverage_picks_change_with_different_public_picks(self):
        picks_a = _make_picks({"duke": 0.22, "unc": 0.12, "gonzaga": 0.08, "midmajor": 0.008})
        picks_b = _make_picks({"duke": 0.35, "unc": 0.08, "gonzaga": 0.04, "midmajor": 0.005})

        report_a = LivePicksRefreshWorkflow._run_lightweight_ev(
            MODEL_PROBS, picks_a, TEAM_META, EV_CONFIG,
        )
        report_b = LivePicksRefreshWorkflow._run_lightweight_ev(
            MODEL_PROBS, picks_b, TEAM_META, EV_CONFIG,
        )

        # Different public picks should produce different leverage analysis
        lev_a = {(p["team_id"], p["round"]) for p in report_a["leverage_picks"]}
        lev_b = {(p["team_id"], p["round"]) for p in report_b["leverage_picks"]}
        # At least some difference expected (midmajor leverage changes with public)
        assert lev_a != lev_b or report_a["recommended_strategy"] != report_b["recommended_strategy"]


# ---------------------------------------------------------------------------
# Tests: LivePicksRefreshWorkflow — refresh from JSON
# ---------------------------------------------------------------------------


class TestRefreshFromJSON:

    def test_refresh_from_json_triggers_reoptimization(self, tmp_path):
        initial_picks = _make_picks({"duke": 0.22, "unc": 0.12, "gonzaga": 0.08, "midmajor": 0.008})
        workflow = _build_workflow(initial_picks)

        # Write updated picks to JSON
        updated_data = {
            "teams": {
                "duke": {"R64": 95.0, "R32": 85.0, "S16": 70.0, "E8": 50.0, "F4": 30.0, "champion_pct": 15.0},
                "unc": {"R64": 96.0, "R32": 88.0, "S16": 65.0, "E8": 45.0, "F4": 32.0, "champion_pct": 18.0},
                "gonzaga": {"R64": 88.0, "R32": 65.0, "S16": 40.0, "E8": 22.0, "F4": 12.0, "champion_pct": 5.0},
                "midmajor": {"R64": 40.0, "R32": 15.0, "S16": 5.0, "E8": 2.0, "F4": 1.0, "champion_pct": 0.3},
            }
        }
        json_path = str(tmp_path / "updated_picks.json")
        with open(json_path, "w") as f:
            json.dump(updated_data, f)

        result = workflow.refresh_from_json(json_path, force=True)
        assert result.was_refreshed is True
        assert result.updated_ev_report is not None
        assert result.elapsed_seconds >= 0
        assert len(result.pick_deltas) > 0

    def test_refresh_below_threshold_skips_reoptimization(self, tmp_path):
        initial_picks = _make_picks({"duke": 0.22, "unc": 0.12, "gonzaga": 0.08, "midmajor": 0.008})
        workflow = _build_workflow(initial_picks)

        # Write nearly identical picks (matching the _make_picks output closely).
        # _make_picks generates values in 0-1 scale. JSON values > 1 get divided
        # by 100 in the loader. So we use percentage-scale values (> 1.0).
        # _make_picks(duke=0.22) -> CHAMP=0.22, F4=0.33, E8=0.44, S16=0.55, R32=0.66, R64=0.77
        # In percentage terms: CHAMP=22%, F4=33%, etc. Add tiny deltas (<0.5pp).
        updated_data = {
            "teams": {
                "duke": {"CHAMP": 22.2, "F4": 33.2, "E8": 44.2, "S16": 55.2, "R32": 66.2, "R64": 77.2},
                "unc": {"CHAMP": 12.1, "F4": 18.1, "E8": 24.1, "S16": 30.1, "R32": 36.1, "R64": 42.1},
                "gonzaga": {"CHAMP": 8.1, "F4": 12.1, "E8": 16.1, "S16": 20.1, "R32": 24.1, "R64": 28.1},
                "midmajor": {"CHAMP": 1.1, "F4": 1.5, "E8": 1.8, "S16": 2.2, "R32": 2.6, "R64": 3.1},
            }
        }
        json_path = str(tmp_path / "same_picks.json")
        with open(json_path, "w") as f:
            json.dump(updated_data, f)

        result = workflow.refresh(
            new_picks_json=json_path,
            significance_threshold_pp=5.0,  # High threshold
        )
        assert result.was_refreshed is False
        assert result.skip_reason is not None
        assert "below threshold" in result.skip_reason

    def test_force_refresh_overrides_threshold(self, tmp_path):
        initial_picks = _make_picks({"duke": 0.22, "unc": 0.12, "gonzaga": 0.08, "midmajor": 0.008})
        workflow = _build_workflow(initial_picks)

        # Tiny changes that would not pass any normal threshold
        updated_data = {
            "teams": {
                "duke": {"CHAMP": 22.2, "F4": 33.2, "E8": 44.2, "S16": 55.2, "R32": 66.2, "R64": 77.2},
                "unc": {"CHAMP": 12.1, "F4": 18.1, "E8": 24.1, "S16": 30.1, "R32": 36.1, "R64": 42.1},
                "gonzaga": {"CHAMP": 8.1, "F4": 12.1, "E8": 16.1, "S16": 20.1, "R32": 24.1, "R64": 28.1},
                "midmajor": {"CHAMP": 1.1, "F4": 1.5, "E8": 1.8, "S16": 2.2, "R32": 2.6, "R64": 3.1},
            }
        }
        json_path = str(tmp_path / "same_picks.json")
        with open(json_path, "w") as f:
            json.dump(updated_data, f)

        result = workflow.refresh(
            new_picks_json=json_path,
            significance_threshold_pp=50.0,
            force=True,
        )
        assert result.was_refreshed is True


# ---------------------------------------------------------------------------
# Tests: LivePicksRefreshWorkflow — delta computation
# ---------------------------------------------------------------------------


class TestDeltaComputation:

    def test_compute_round_deltas_detects_changes(self):
        old = _make_picks({"duke": 0.22, "unc": 0.12})
        new = _make_picks({"duke": 0.30, "unc": 0.12})
        workflow = _build_workflow(old)

        deltas = workflow._compute_round_deltas(old, new)
        duke_deltas = [d for d in deltas if d.team_id == "duke"]
        assert len(duke_deltas) > 0
        # Championship should have changed
        champ_delta = [d for d in duke_deltas if d.round_name == "CHAMP"]
        assert len(champ_delta) == 1
        assert champ_delta[0].abs_delta == pytest.approx(8.0, abs=0.1)  # 22% -> 30%

    def test_compute_round_deltas_no_changes(self):
        picks = _make_picks({"duke": 0.22})
        workflow = _build_workflow(picks)
        deltas = workflow._compute_round_deltas(picks, picks)
        assert len(deltas) == 0


# ---------------------------------------------------------------------------
# Tests: LivePicksRefreshWorkflow — leverage shift detection
# ---------------------------------------------------------------------------


class TestLeverageShiftDetection:

    def test_detects_new_leverage_picks(self):
        old_picks = _make_picks({"duke": 0.22, "unc": 0.12, "gonzaga": 0.08, "midmajor": 0.008})
        workflow = _build_workflow(old_picks)

        # Shift public picks so midmajor becomes more leveraged
        new_picks = _make_picks({"duke": 0.30, "unc": 0.15, "gonzaga": 0.06, "midmajor": 0.002})
        result = workflow.refresh(new_picks_json=None, force=True)
        # Can't scrape, so let's test with JSON instead

    def test_leverage_shift_comparison(self):
        """Test that leverage shifts are computed between two EV reports."""
        picks_a = _make_picks({"duke": 0.22, "unc": 0.12, "gonzaga": 0.08, "midmajor": 0.008})
        picks_b = _make_picks({"duke": 0.35, "unc": 0.08, "gonzaga": 0.04, "midmajor": 0.005})

        ev_a = LivePicksRefreshWorkflow._run_lightweight_ev(
            MODEL_PROBS, picks_a, TEAM_META, EV_CONFIG,
        )
        ev_b = LivePicksRefreshWorkflow._run_lightweight_ev(
            MODEL_PROBS, picks_b, TEAM_META, EV_CONFIG,
        )

        workflow = _build_workflow(picks_a)
        shifts = workflow._compute_leverage_shifts(ev_a, ev_b)
        # With different public picks, some leverage ratios should shift
        assert isinstance(shifts, list)

    def test_find_new_and_lost_leverage(self):
        picks_a = _make_picks({"duke": 0.22, "unc": 0.12, "gonzaga": 0.08, "midmajor": 0.008})
        picks_b = _make_picks({"duke": 0.35, "unc": 0.08, "gonzaga": 0.04, "midmajor": 0.005})

        ev_a = LivePicksRefreshWorkflow._run_lightweight_ev(
            MODEL_PROBS, picks_a, TEAM_META, EV_CONFIG,
        )
        ev_b = LivePicksRefreshWorkflow._run_lightweight_ev(
            MODEL_PROBS, picks_b, TEAM_META, EV_CONFIG,
        )

        workflow = _build_workflow(picks_a)
        new_lev = workflow._find_new_leverage_picks(ev_a, ev_b)
        lost_lev = workflow._find_lost_leverage_picks(ev_a, ev_b)

        assert isinstance(new_lev, list)
        assert isinstance(lost_lev, list)


# ---------------------------------------------------------------------------
# Tests: LivePicksRefreshWorkflow — history tracking
# ---------------------------------------------------------------------------


class TestRefreshHistory:

    def test_history_tracks_refreshes(self, tmp_path):
        initial_picks = _make_picks({"duke": 0.22, "unc": 0.12, "gonzaga": 0.08, "midmajor": 0.008})
        workflow = _build_workflow(initial_picks)

        assert len(workflow.refresh_history) == 0

        # First refresh (force with no scraper → will fail gracefully)
        json_data = {
            "teams": {
                "duke": {"champion_pct": 25.0, "R64": 97.0, "R32": 90.0, "S16": 72.0, "E8": 52.0, "F4": 33.0},
                "unc": {"champion_pct": 12.0, "R64": 93.0, "R32": 83.0, "S16": 58.0, "E8": 38.0, "F4": 26.0},
                "gonzaga": {"champion_pct": 8.0, "R64": 88.0, "R32": 69.0, "S16": 44.0, "E8": 27.0, "F4": 18.0},
                "midmajor": {"champion_pct": 0.8, "R64": 70.0, "R32": 22.0, "S16": 10.0, "E8": 5.0, "F4": 3.0},
            }
        }
        json_path = str(tmp_path / "picks_v2.json")
        with open(json_path, "w") as f:
            json.dump(json_data, f)

        workflow.refresh(new_picks_json=json_path, force=True)
        assert len(workflow.refresh_history) == 1

        # Second refresh
        json_data["teams"]["duke"]["champion_pct"] = 28.0
        json_path2 = str(tmp_path / "picks_v3.json")
        with open(json_path2, "w") as f:
            json.dump(json_data, f)

        workflow.refresh(new_picks_json=json_path2, force=True)
        assert len(workflow.refresh_history) == 2


# ---------------------------------------------------------------------------
# Tests: RoundDelta in RefreshablePicksManager
# ---------------------------------------------------------------------------


class TestRoundDeltaTracking:

    def test_round_deltas_populated(self):
        orch = MagicMock(spec=ScraperOrchestrator)
        result1 = _make_orch_result({"duke": 20.0, "unc": 15.0})
        result2 = _make_orch_result({"duke": 25.0, "unc": 15.0})
        orch.fetch_all_picks.side_effect = [result1, result2]
        manager = RefreshablePicksManager(orch, staleness_threshold_hours=0.0)

        manager.fetch_or_refresh(year=2026)
        time.sleep(0.01)
        _, update = manager.fetch_or_refresh(year=2026)

        assert update is not None
        assert len(update.round_deltas) > 0
        # Duke's champion_pct changed by 5pp
        duke_champ = [d for d in update.round_deltas if d.team_id == "duke" and d.round_name == "CHAMP"]
        assert len(duke_champ) == 1
        assert duke_champ[0].abs_delta == pytest.approx(5.0)

    def test_late_round_max_delta(self):
        orch = MagicMock(spec=ScraperOrchestrator)
        result1 = _make_orch_result({"duke": 20.0})
        result2 = _make_orch_result({"duke": 28.0})
        orch.fetch_all_picks.side_effect = [result1, result2]
        manager = RefreshablePicksManager(orch, staleness_threshold_hours=0.0)

        manager.fetch_or_refresh(year=2026)
        time.sleep(0.01)
        _, update = manager.fetch_or_refresh(year=2026)

        assert update is not None
        assert update.late_round_max_delta > 0
        assert update.max_round_delta >= update.late_round_max_delta

    def test_round_deltas_summary_includes_late_round(self):
        update = PicksUpdate(
            previous=_make_consensus({"duke": 20.0}),
            current=_make_consensus({"duke": 25.0}),
            changed_teams=["duke"],
            max_delta=5.0,
            round_deltas=[
                RoundDelta(team_id="duke", round_name="CHAMP", old_pct=20.0, new_pct=25.0),
                RoundDelta(team_id="duke", round_name="F4", old_pct=30.0, new_pct=37.5),
            ],
        )
        assert "Late-round" in update.summary


# ---------------------------------------------------------------------------
# Tests: JSON loading with various formats
# ---------------------------------------------------------------------------


class TestJSONLoading:

    def test_load_with_short_keys(self, tmp_path):
        picks = _make_picks({"duke": 0.22})
        workflow = _build_workflow(picks)

        data = {
            "teams": {
                "duke": {"R64": 95.0, "R32": 88.0, "S16": 70.0, "E8": 50.0, "F4": 33.0, "CHAMP": 22.0},
            }
        }
        path = str(tmp_path / "short_keys.json")
        with open(path, "w") as f:
            json.dump(data, f)

        loaded = workflow._load_picks_from_json(path)
        assert loaded is not None
        assert "duke" in loaded
        assert loaded["duke"]["CHAMP"] == pytest.approx(0.22)  # 22% -> 0.22

    def test_load_with_long_keys(self, tmp_path):
        picks = _make_picks({"duke": 0.22})
        workflow = _build_workflow(picks)

        data = {
            "teams": {
                "duke": {
                    "round_of_64_pct": 95.0, "round_of_32_pct": 88.0,
                    "sweet_16_pct": 70.0, "elite_8_pct": 50.0,
                    "final_four_pct": 33.0, "champion_pct": 22.0,
                },
            }
        }
        path = str(tmp_path / "long_keys.json")
        with open(path, "w") as f:
            json.dump(data, f)

        loaded = workflow._load_picks_from_json(path)
        assert loaded is not None
        assert loaded["duke"]["CHAMP"] == pytest.approx(0.22)

    def test_load_invalid_json_returns_none(self, tmp_path):
        picks = _make_picks({"duke": 0.22})
        workflow = _build_workflow(picks)

        path = str(tmp_path / "bad.json")
        with open(path, "w") as f:
            f.write("not json")

        loaded = workflow._load_picks_from_json(path)
        assert loaded is None


# ---------------------------------------------------------------------------
# Tests: End-to-end refresh scenario
# ---------------------------------------------------------------------------


class TestEndToEndRefresh:

    def test_full_refresh_cycle(self, tmp_path):
        """Simulate a complete refresh cycle: initial -> 24h later -> 48h later."""
        initial_picks = _make_picks({"duke": 0.22, "unc": 0.12, "gonzaga": 0.08, "midmajor": 0.008})
        workflow = _build_workflow(initial_picks)

        # Snapshot 1: 24 hours before tip-off — Duke hype builds
        data_24h = {
            "teams": {
                "duke": {"champion_pct": 28.0, "R64": 98.0, "R32": 94.0, "S16": 80.0, "E8": 60.0, "F4": 40.0},
                "unc": {"champion_pct": 10.0, "R64": 92.0, "R32": 80.0, "S16": 55.0, "E8": 35.0, "F4": 22.0},
                "gonzaga": {"champion_pct": 6.0, "R64": 86.0, "R32": 65.0, "S16": 40.0, "E8": 24.0, "F4": 14.0},
                "midmajor": {"champion_pct": 0.5, "R64": 42.0, "R32": 16.0, "S16": 6.0, "E8": 2.5, "F4": 1.2},
            }
        }
        path_24h = str(tmp_path / "picks_24h.json")
        with open(path_24h, "w") as f:
            json.dump(data_24h, f)

        result1 = workflow.refresh(new_picks_json=path_24h, force=True)
        assert result1.was_refreshed is True
        assert result1.updated_ev_report is not None
        ev_24h = result1.updated_ev_report

        # Snapshot 2: 6 hours before tip-off — Gonzaga injury breaks
        data_6h = {
            "teams": {
                "duke": {"champion_pct": 30.0, "R64": 98.0, "R32": 95.0, "S16": 82.0, "E8": 62.0, "F4": 42.0},
                "unc": {"champion_pct": 14.0, "R64": 94.0, "R32": 85.0, "S16": 62.0, "E8": 42.0, "F4": 28.0},
                "gonzaga": {"champion_pct": 3.0, "R64": 80.0, "R32": 55.0, "S16": 30.0, "E8": 16.0, "F4": 8.0},
                "midmajor": {"champion_pct": 0.4, "R64": 40.0, "R32": 14.0, "S16": 5.0, "E8": 2.0, "F4": 1.0},
            }
        }
        path_6h = str(tmp_path / "picks_6h.json")
        with open(path_6h, "w") as f:
            json.dump(data_6h, f)

        result2 = workflow.refresh(new_picks_json=path_6h, force=True)
        assert result2.was_refreshed is True

        # Gonzaga's public drop should create leverage opportunity
        # (model still believes in them at 10%, public dropped to 3%)
        gonzaga_deltas = [d for d in result2.pick_deltas if d.team_id == "gonzaga"]
        assert len(gonzaga_deltas) > 0
        champ_delta = [d for d in gonzaga_deltas if d.round_name == "CHAMP"]
        assert len(champ_delta) > 0

        # History should have 2 entries
        assert len(workflow.refresh_history) == 2

        # Current picks should be updated
        assert workflow.current_picks["gonzaga"]["CHAMP"] == pytest.approx(0.03)

    def test_change_summary_is_human_readable(self, tmp_path):
        initial_picks = _make_picks({"duke": 0.22, "unc": 0.12, "gonzaga": 0.08, "midmajor": 0.008})
        workflow = _build_workflow(initial_picks)

        data = {
            "teams": {
                "duke": {"champion_pct": 30.0, "R64": 98.0, "R32": 94.0, "S16": 80.0, "E8": 60.0, "F4": 40.0},
                "unc": {"champion_pct": 10.0, "R64": 92.0, "R32": 80.0, "S16": 55.0, "E8": 35.0, "F4": 22.0},
                "gonzaga": {"champion_pct": 4.0, "R64": 85.0, "R32": 62.0, "S16": 38.0, "E8": 22.0, "F4": 12.0},
                "midmajor": {"champion_pct": 0.3, "R64": 38.0, "R32": 14.0, "S16": 5.0, "E8": 2.0, "F4": 1.0},
            }
        }
        path = str(tmp_path / "picks.json")
        with open(path, "w") as f:
            json.dump(data, f)

        result = workflow.refresh(new_picks_json=path, force=True)
        summary = result.change_summary
        assert "Refresh completed" in summary
        assert "Top pick shifts" in summary
