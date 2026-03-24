"""Phase 3 robustness tests: plausibility validation, min_sources enforcement, cache staleness."""

import json
import os
import tempfile
import time
from unittest.mock import patch

import pytest

from src.data.scrapers.espn_picks import (
    ConsensusData,
    ESPNPicksScraper,
    PublicPicks,
    ScraperOrchestrator,
    YahooPicksScraper,
    CBSPicksScraper,
    validate_consensus_plausibility,
)
from src.exceptions import DataRequirementError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_team(team_id, seed, region, champ):
    return PublicPicks(
        team_id=team_id,
        team_name=team_id.replace("_", " ").title(),
        seed=seed,
        region=region,
        round_of_64_pct=max(champ * 3, champ),
        round_of_32_pct=max(champ * 2.5, champ),
        sweet_16_pct=max(champ * 2, champ),
        elite_8_pct=max(champ * 1.5, champ),
        final_four_pct=max(champ * 1.2, champ),
        champion_pct=champ,
    )


def _realistic_consensus(n_teams=16):
    """Build a plausible consensus with 16 teams summing to ~100% champ."""
    base = {
        1: 22.0, 2: 15.0, 3: 8.0, 4: 5.0,
        5: 3.0, 6: 2.0, 7: 1.5, 8: 1.0,
        9: 0.8, 10: 0.5, 11: 0.4, 12: 0.3,
        13: 0.1, 14: 0.05, 15: 0.02, 16: 0.01,
    }
    teams = {}
    for seed, champ in list(base.items())[:n_teams]:
        tid = f"team_{seed}"
        teams[tid] = _make_team(tid, seed, "East", champ)
    return ConsensusData(teams=teams, sources=["espn"])


# ---------------------------------------------------------------------------
# Tests: validate_consensus_plausibility
# ---------------------------------------------------------------------------

class TestValidateConsensusPlausibility:

    def test_realistic_consensus_no_warnings(self):
        consensus = _realistic_consensus()
        warnings = validate_consensus_plausibility(consensus)
        assert warnings == []

    def test_empty_consensus(self):
        consensus = ConsensusData(teams={}, sources=["espn"])
        warnings = validate_consensus_plausibility(consensus)
        assert any("Empty consensus" in w for w in warnings)

    def test_too_few_teams(self):
        consensus = _realistic_consensus(n_teams=4)
        warnings = validate_consensus_plausibility(consensus, min_teams=16)
        assert any("Only 4 teams" in w for w in warnings)

    def test_single_team_concentration(self):
        consensus = _realistic_consensus()
        # Corrupt one team to 80% championship
        consensus.teams["team_1"] = _make_team("team_1", 1, "East", champ=80.0)
        warnings = validate_consensus_plausibility(consensus, max_single_champ_pct=60.0)
        assert any("80.0%" in w and "corruption" in w for w in warnings)

    def test_deflated_distribution(self):
        """All teams at ~1% — sum far below 50%."""
        teams = {}
        for i in range(1, 17):
            tid = f"team_{i}"
            teams[tid] = _make_team(tid, i, "East", champ=1.0)
        consensus = ConsensusData(teams=teams, sources=["espn"])
        warnings = validate_consensus_plausibility(consensus, min_champ_sum=50.0)
        assert any("deflated" in w for w in warnings)

    def test_inflated_distribution(self):
        """Sum way above 150%."""
        teams = {}
        for i in range(1, 17):
            tid = f"team_{i}"
            teams[tid] = _make_team(tid, i, "East", champ=20.0)
        consensus = ConsensusData(teams=teams, sources=["espn"])
        warnings = validate_consensus_plausibility(consensus, max_champ_sum=150.0)
        assert any("inflated" in w for w in warnings)

    def test_top_seed_zero_champ(self):
        """Seed 1 at 0% championship → warning."""
        consensus = _realistic_consensus()
        consensus.teams["team_1"] = _make_team("team_1", 1, "East", champ=0.0)
        warnings = validate_consensus_plausibility(consensus)
        assert any("seed 1" in w and "0%" in w for w in warnings)


# ---------------------------------------------------------------------------
# Tests: min_sources enforcement
# ---------------------------------------------------------------------------

class TestMinSourcesEnforcement:

    def _mock_fetch(self, source_data):
        def _side_effect(source_name, scraper, year):
            return source_data.get(source_name)
        return _side_effect

    def test_raises_when_below_min_sources(self):
        """Only ESPN returns data, min_sources=2 → DataRequirementError."""
        consensus = _realistic_consensus()
        data = {"espn": consensus}

        orch = ScraperOrchestrator()
        with patch.object(orch, "_fetch_with_retry", side_effect=self._mock_fetch(data)):
            with pytest.raises(DataRequirementError, match="Minimum required: 2"):
                orch.fetch_all_picks(year=2026, min_sources=2)

    def test_succeeds_at_min_sources(self):
        """Two sources return data, min_sources=2 → success."""
        consensus = _realistic_consensus()
        data = {"espn": consensus, "yahoo": consensus}

        orch = ScraperOrchestrator()
        with patch.object(orch, "_fetch_with_retry", side_effect=self._mock_fetch(data)):
            result = orch.fetch_all_picks(year=2026, min_sources=2)
        assert len(result.successful_sources) == 2

    def test_zero_sources_raises(self):
        """No sources return data → DataRequirementError."""
        orch = ScraperOrchestrator()
        with patch.object(orch, "_fetch_with_retry", side_effect=self._mock_fetch({})):
            with pytest.raises(DataRequirementError):
                orch.fetch_all_picks(year=2026, min_sources=1)

    def test_min_sources_one_allows_single(self):
        """min_sources=1 with one source → success."""
        consensus = _realistic_consensus()
        data = {"espn": consensus}

        orch = ScraperOrchestrator()
        with patch.object(orch, "_fetch_with_retry", side_effect=self._mock_fetch(data)):
            result = orch.fetch_all_picks(year=2026, min_sources=1)
        assert result.successful_sources == ["espn"]


# ---------------------------------------------------------------------------
# Tests: cache staleness
# ---------------------------------------------------------------------------

class TestCacheStaleness:

    def test_fresh_cache_loaded(self):
        """Cache file < 7 days old → loaded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            scraper = ESPNPicksScraper(cache_dir=tmpdir)
            cache_file = os.path.join(tmpdir, "test.json")
            with open(cache_file, "w") as f:
                json.dump({"teams": {}, "sources": ["espn"]}, f)
            # File just created → fresh
            result = scraper._load_from_cache("test.json")
            assert result is not None

    def test_stale_cache_rejected(self):
        """Cache file > max_age → rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            scraper = ESPNPicksScraper(cache_dir=tmpdir)
            cache_file = os.path.join(tmpdir, "test.json")
            with open(cache_file, "w") as f:
                json.dump({"teams": {}}, f)
            # Backdate the file by 10 days
            old_time = time.time() - 10 * 86400
            os.utime(cache_file, (old_time, old_time))

            result = scraper._load_from_cache("test.json")
            assert result is None

    def test_custom_max_age(self):
        """Cache file 2 hours old, max_age=1 hour → rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            scraper = ESPNPicksScraper(cache_dir=tmpdir)
            cache_file = os.path.join(tmpdir, "test.json")
            with open(cache_file, "w") as f:
                json.dump({"data": True}, f)
            old_time = time.time() - 2 * 3600
            os.utime(cache_file, (old_time, old_time))

            result = scraper._load_from_cache("test.json", max_age_seconds=3600)
            assert result is None

    def test_yahoo_cache_staleness(self):
        """Yahoo scraper also rejects stale cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            scraper = YahooPicksScraper(cache_dir=tmpdir)
            cache_file = os.path.join(tmpdir, "yahoo.json")
            with open(cache_file, "w") as f:
                json.dump({"teams": {}}, f)
            old_time = time.time() - 10 * 86400
            os.utime(cache_file, (old_time, old_time))

            result = scraper._load_cache("yahoo.json")
            assert result is None

    def test_cbs_cache_staleness(self):
        """CBS scraper also rejects stale cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            scraper = CBSPicksScraper(cache_dir=tmpdir)
            cache_file = os.path.join(tmpdir, "cbs.json")
            with open(cache_file, "w") as f:
                json.dump({"teams": {}}, f)
            old_time = time.time() - 10 * 86400
            os.utime(cache_file, (old_time, old_time))

            result = scraper._load_cache("cbs.json")
            assert result is None
