"""Tests for dynamic public pick refresh (Improvement 5)."""

import time
import pytest
from unittest.mock import MagicMock, patch

from src.data.scrapers.espn_picks import (
    ConsensusData,
    OrchestratorResult,
    PicksUpdate,
    PublicPicks,
    RefreshablePicksManager,
    ScraperOrchestrator,
    SourceHealth,
    SourceStatus,
)


def _make_consensus(champion_pcts: dict) -> ConsensusData:
    """Helper to build ConsensusData from champion percentages."""
    teams = {}
    for tid, pct in champion_pcts.items():
        teams[tid] = PublicPicks(
            team_id=tid, team_name=tid.title(), seed=1, region="East",
            champion_pct=pct,
        )
    return ConsensusData(teams=teams, sources=["espn"])


def _make_orchestrator_result(champion_pcts: dict) -> OrchestratorResult:
    return OrchestratorResult(
        consensus=_make_consensus(champion_pcts),
        successful_sources=["espn"],
        source_statuses={"espn": SourceStatus(name="espn")},
    )


class TestRefreshablePicksManagerStaleness:

    def test_initially_stale(self):
        orch = MagicMock(spec=ScraperOrchestrator)
        manager = RefreshablePicksManager(orch, staleness_threshold_hours=24.0)
        assert manager.is_stale is True
        assert manager.hours_since_refresh is None

    def test_not_stale_after_fetch(self):
        orch = MagicMock(spec=ScraperOrchestrator)
        orch.fetch_all_picks.return_value = _make_orchestrator_result({"duke": 20.0})
        manager = RefreshablePicksManager(orch, staleness_threshold_hours=24.0)

        manager.fetch_or_refresh(year=2026)
        assert manager.is_stale is False
        assert manager.hours_since_refresh is not None
        assert manager.hours_since_refresh < 0.01  # Just fetched

    def test_stale_after_threshold(self):
        orch = MagicMock(spec=ScraperOrchestrator)
        orch.fetch_all_picks.return_value = _make_orchestrator_result({"duke": 20.0})
        manager = RefreshablePicksManager(orch, staleness_threshold_hours=0.0001)

        manager.fetch_or_refresh(year=2026)
        time.sleep(0.5)  # Exceed the very short threshold
        assert manager.is_stale is True


class TestRefreshablePicksManagerFetch:

    def test_first_fetch_returns_result_no_update(self):
        orch = MagicMock(spec=ScraperOrchestrator)
        result = _make_orchestrator_result({"duke": 20.0})
        orch.fetch_all_picks.return_value = result
        manager = RefreshablePicksManager(orch)

        got_result, update = manager.fetch_or_refresh(year=2026)
        assert got_result is result
        assert update is None  # No previous data to compare against

    def test_cached_result_when_not_stale(self):
        orch = MagicMock(spec=ScraperOrchestrator)
        result = _make_orchestrator_result({"duke": 20.0})
        orch.fetch_all_picks.return_value = result
        manager = RefreshablePicksManager(orch, staleness_threshold_hours=24.0)

        manager.fetch_or_refresh(year=2026)
        got_result, update = manager.fetch_or_refresh(year=2026)
        # Should return cached, not re-fetch
        assert orch.fetch_all_picks.call_count == 1
        assert update is None

    def test_force_refresh_overrides_cache(self):
        orch = MagicMock(spec=ScraperOrchestrator)
        result1 = _make_orchestrator_result({"duke": 20.0})
        result2 = _make_orchestrator_result({"duke": 22.0})
        orch.fetch_all_picks.side_effect = [result1, result2]
        manager = RefreshablePicksManager(orch, staleness_threshold_hours=24.0)

        manager.fetch_or_refresh(year=2026)
        got_result, update = manager.fetch_or_refresh(year=2026, force_refresh=True)
        assert orch.fetch_all_picks.call_count == 2
        assert update is not None


class TestPicksUpdateDetection:

    def test_detects_changes(self):
        orch = MagicMock(spec=ScraperOrchestrator)
        result1 = _make_orchestrator_result({"duke": 20.0, "unc": 15.0})
        result2 = _make_orchestrator_result({"duke": 25.0, "unc": 15.0})
        orch.fetch_all_picks.side_effect = [result1, result2]
        manager = RefreshablePicksManager(orch, staleness_threshold_hours=0.0)

        manager.fetch_or_refresh(year=2026)
        time.sleep(0.01)
        _, update = manager.fetch_or_refresh(year=2026)

        assert update is not None
        assert "duke" in update.changed_teams
        assert update.max_delta == pytest.approx(5.0)
        assert "unc" not in update.changed_teams

    def test_no_changes(self):
        orch = MagicMock(spec=ScraperOrchestrator)
        result1 = _make_orchestrator_result({"duke": 20.0, "unc": 15.0})
        result2 = _make_orchestrator_result({"duke": 20.0, "unc": 15.0})
        orch.fetch_all_picks.side_effect = [result1, result2]
        manager = RefreshablePicksManager(orch, staleness_threshold_hours=0.0)

        manager.fetch_or_refresh(year=2026)
        time.sleep(0.01)
        _, update = manager.fetch_or_refresh(year=2026)

        assert update is not None
        assert len(update.changed_teams) == 0
        assert update.max_delta == 0.0

    def test_update_summary(self):
        update = PicksUpdate(
            previous=_make_consensus({"duke": 20.0}),
            current=_make_consensus({"duke": 25.0}),
            changed_teams=["duke"],
            max_delta=5.0,
            refresh_timestamp=time.time(),
            was_stale=True,
        )
        summary = update.summary
        assert "duke" in summary
        assert "5.0" in summary

    def test_update_summary_no_changes(self):
        update = PicksUpdate(
            previous=_make_consensus({"duke": 20.0}),
            current=_make_consensus({"duke": 20.0}),
            changed_teams=[],
            max_delta=0.0,
        )
        assert "No significant changes" in update.summary
