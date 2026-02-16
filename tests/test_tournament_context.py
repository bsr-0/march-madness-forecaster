"""Tests for TournamentContextScraper and SOTA pipeline integration."""

import json
import os
import tempfile

import pytest

from src.data.scrapers.tournament_context import TournamentContextScraper


# ---------------------------------------------------------------------------
# Unit tests for TournamentContextScraper helpers
# ---------------------------------------------------------------------------


def test_normalize_name():
    """_normalize_name should strip special chars and lowercase."""
    assert TournamentContextScraper._normalize_name("Duke") == "duke"
    assert TournamentContextScraper._normalize_name("North Carolina") == "north_carolina"
    assert TournamentContextScraper._normalize_name("St. Mary's") == "st__mary_s"
    assert TournamentContextScraper._normalize_name("") == ""
    assert TournamentContextScraper._normalize_name(None) == ""


def test_parse_rank():
    """_parse_rank should extract integers from rank text."""
    assert TournamentContextScraper._parse_rank("1") == 1
    assert TournamentContextScraper._parse_rank("14T") == 14
    assert TournamentContextScraper._parse_rank("25") == 25
    assert TournamentContextScraper._parse_rank("") is None
    assert TournamentContextScraper._parse_rank("NR") is None


def test_safe_int():
    """_safe_int should handle integers and malformed input."""
    assert TournamentContextScraper._safe_int("10") == 10
    assert TournamentContextScraper._safe_int("0") == 0
    assert TournamentContextScraper._safe_int("") == 0
    assert TournamentContextScraper._safe_int("abc") == 0


# ---------------------------------------------------------------------------
# Tests for JSON cache / load round-trip
# ---------------------------------------------------------------------------


def test_cache_roundtrip():
    """Caching should write and reload JSON correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        scraper = TournamentContextScraper(cache_dir=tmpdir)

        # Simulate caching
        scraper._save_cache("test_cache.json", {"rankings": {"duke": 5, "unc": 12}})
        loaded = scraper._load_cache("test_cache.json")
        assert loaded is not None
        assert loaded["rankings"]["duke"] == 5
        assert loaded["rankings"]["unc"] == 12


def test_load_cache_missing():
    """Loading from a non-existent cache should return None."""
    with tempfile.TemporaryDirectory() as tmpdir:
        scraper = TournamentContextScraper(cache_dir=tmpdir)
        assert scraper._load_cache("nonexistent.json") is None


def test_load_cache_no_cache_dir():
    """With no cache_dir, load_cache should always return None."""
    scraper = TournamentContextScraper(cache_dir=None)
    assert scraper._load_cache("anything.json") is None


def test_save_cache_no_cache_dir():
    """With no cache_dir, save_cache should be a no-op."""
    scraper = TournamentContextScraper(cache_dir=None)
    # Should not raise
    scraper._save_cache("anything.json", {"data": 1})


# ---------------------------------------------------------------------------
# Tests for static JSON loaders
# ---------------------------------------------------------------------------


def test_load_preseason_ap_from_json():
    """Static loader should extract rankings dict from JSON file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({"rankings": {"duke": 3, "gonzaga": 1}, "year": 2026}, f)
        f.flush()
        path = f.name

    try:
        result = TournamentContextScraper.load_preseason_ap_from_json(path)
        assert result == {"duke": 3, "gonzaga": 1}
    finally:
        os.unlink(path)


def test_load_coach_data_from_json():
    """Static loader should extract coaches dict from JSON file."""
    coach_data = {
        "coaches": {
            "jay_wright": {
                "name": "Jay Wright",
                "appearances": 12,
                "wins": 20,
                "losses": 10,
                "teams": ["Villanova"],
            }
        },
        "year": 2026,
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(coach_data, f)
        f.flush()
        path = f.name

    try:
        result = TournamentContextScraper.load_coach_data_from_json(path)
        assert "jay_wright" in result
        assert result["jay_wright"]["appearances"] == 12
    finally:
        os.unlink(path)


def test_load_conf_champions_from_json():
    """Static loader should extract champions dict from JSON file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({"champions": {"duke": "ACC", "gonzaga": "WCC"}, "year": 2026}, f)
        f.flush()
        path = f.name

    try:
        result = TournamentContextScraper.load_conf_champions_from_json(path)
        assert result == {"duke": "ACC", "gonzaga": "WCC"}
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# Tests for build_team_to_coach_appearances
# ---------------------------------------------------------------------------


def test_build_team_to_coach_appearances_exact_match():
    """Should map team to coach appearances via exact key match."""
    coach_data = {
        "john_calipari": {
            "name": "John Calipari",
            "appearances": 15,
            "wins": 28,
            "losses": 14,
            "teams": ["Kentucky", "Memphis"],
        },
        "mark_few": {
            "name": "Mark Few",
            "appearances": 22,
            "wins": 35,
            "losses": 12,
            "teams": ["Gonzaga"],
        },
    }
    team_to_coach = {
        "kentucky": "John Calipari",
        "gonzaga": "Mark Few",
        "duke": "Jon Scheyer",
    }
    scraper = TournamentContextScraper()
    result = scraper.build_team_to_coach_appearances(coach_data, team_to_coach)

    assert result["kentucky"] == 15
    assert result["gonzaga"] == 22
    # Duke's coach (Jon Scheyer) not in coach_data → 0
    assert result["duke"] == 0


def test_build_team_to_coach_appearances_fuzzy_last_name():
    """Should fuzzy-match by last name if exact key fails."""
    coach_data = {
        "bill_self": {
            "name": "Bill Self",
            "appearances": 20,
            "wins": 38,
            "losses": 17,
            "teams": ["Kansas"],
        },
    }
    # Use slightly different key that won't exact-match
    team_to_coach = {
        "kansas": "Coach Self",  # Won't exact match "bill_self"
    }
    scraper = TournamentContextScraper()
    result = scraper.build_team_to_coach_appearances(coach_data, team_to_coach)

    # Should find "self" via fuzzy last-name matching
    assert result["kansas"] == 20


# ---------------------------------------------------------------------------
# Test SOTA pipeline enrichment integration
# ---------------------------------------------------------------------------


def test_sota_enrich_tournament_context():
    """
    _enrich_tournament_context should inject AP rank, coach apps, and
    conf champion status into torvik_map and proprietary_map.
    """
    from src.pipeline.sota import SOTAPipeline, SOTAPipelineConfig

    # Write temp JSON artifacts
    with tempfile.TemporaryDirectory() as tmpdir:
        ap_path = os.path.join(tmpdir, "ap.json")
        with open(ap_path, "w") as f:
            json.dump({"rankings": {"duke": 5, "gonzaga": 1}}, f)

        coach_path = os.path.join(tmpdir, "coach.json")
        with open(coach_path, "w") as f:
            json.dump(
                {
                    "coaches": {
                        "jon_scheyer": {
                            "name": "Jon Scheyer",
                            "appearances": 3,
                            "wins": 5,
                            "losses": 2,
                            "teams": ["Duke"],
                        }
                    }
                },
                f,
            )

        champs_path = os.path.join(tmpdir, "champs.json")
        with open(champs_path, "w") as f:
            json.dump({"champions": {"gonzaga": "WCC"}}, f)

        config = SOTAPipelineConfig(
            preseason_ap_json=ap_path,
            coach_tournament_json=coach_path,
            conf_champions_json=champs_path,
        )
        pipeline = SOTAPipeline(config)

        # Build minimal team objects
        from src.models.team import Team

        teams = [
            Team(name="Duke", seed=2, region="East"),
            Team(name="Gonzaga", seed=1, region="West"),
        ]

        # Prepare torvik_map and proprietary_map with minimal data
        torvik_map = {
            "duke": {"effective_fg_pct": 0.55},
            "gonzaga": {"effective_fg_pct": 0.53},
        }
        proprietary_map = {
            "duke": {"adj_efficiency_margin": 20.0},
            "gonzaga": {"adj_efficiency_margin": 25.0},
        }

        pipeline._enrich_tournament_context(torvik_map, proprietary_map, teams)

        # Duke should have AP rank 5
        assert torvik_map["duke"]["preseason_ap_rank"] == 5
        assert proprietary_map["duke"]["preseason_ap_rank"] == 5

        # Gonzaga should have AP rank 1
        assert torvik_map["gonzaga"]["preseason_ap_rank"] == 1
        assert proprietary_map["gonzaga"]["preseason_ap_rank"] == 1

        # Gonzaga is conference tournament champion
        assert torvik_map["gonzaga"]["conf_tourney_champion"] == 1.0
        assert proprietary_map["gonzaga"]["conf_tourney_champion"] == 1.0

        # Duke is NOT conference tournament champion
        assert torvik_map["duke"]["conf_tourney_champion"] == 0.0
        assert proprietary_map["duke"]["conf_tourney_champion"] == 0.0


def test_sota_enrich_no_artifacts():
    """When no tournament context JSONs are provided, enrichment is a no-op."""
    from src.pipeline.sota import SOTAPipeline, SOTAPipelineConfig

    config = SOTAPipelineConfig()  # No context paths
    pipeline = SOTAPipeline(config)

    from src.models.team import Team

    teams = [Team(name="Duke", seed=2, region="East")]
    torvik_map = {"duke": {"effective_fg_pct": 0.55}}
    proprietary_map = {"duke": {"adj_efficiency_margin": 20.0}}

    pipeline._enrich_tournament_context(torvik_map, proprietary_map, teams)

    # Maps should be unchanged (no context keys added)
    assert "preseason_ap_rank" not in torvik_map["duke"]
    assert "coach_tournament_appearances" not in proprietary_map["duke"]


def test_feature_engineering_receives_context_from_torvik_data():
    """
    When torvik_data contains context keys (from enrichment),
    extract_team_features should populate the TeamFeatures fields.
    """
    from src.data.features.feature_engineering import FeatureEngineer

    eng = FeatureEngineer()
    features = eng.extract_team_features(
        team_id="duke",
        team_name="Duke",
        seed=3,
        region="East",
        proprietary_metrics={},
        torvik_data={
            "effective_fg_pct": 0.55,
            "turnover_rate": 0.16,
            "offensive_reb_rate": 0.32,
            "free_throw_rate": 0.35,
            "opp_effective_fg_pct": 0.48,
            "opp_turnover_rate": 0.20,
            "defensive_reb_rate": 0.72,
            "opp_free_throw_rate": 0.28,
            "preseason_ap_rank": 5,
            "coach_tournament_appearances": 12,
            "conf_tourney_champion": 1.0,
        },
    )

    assert features.preseason_ap_rank == 5
    assert features.coach_tournament_appearances == 12
    assert features.conf_tourney_champion == 1.0

    # Also verify it ends up in the vector
    vec = features.to_vector(include_embeddings=False)
    names = features.get_feature_names(include_embeddings=False)

    # Find the index of preseason_ap_rank in the vector
    ap_idx = names.index("preseason_ap_rank")
    # AP rank 5 → (26 - 5) / 25 = 0.84
    assert abs(vec[ap_idx] - (26.0 - 5) / 25.0) < 1e-6

    # Coach tournament exp (12 apps)
    coach_idx = names.index("coach_tournament_exp")
    import numpy as np
    assert abs(vec[coach_idx] - np.log1p(12) / np.log1p(30)) < 1e-6

    # Conf tourney champion = 1.0
    champ_idx = names.index("conf_tourney_champ")
    assert vec[champ_idx] == 1.0


def test_scraper_init_creates_cache_dir():
    """TournamentContextScraper should create cache directory if it doesn't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = os.path.join(tmpdir, "subdir", "cache")
        scraper = TournamentContextScraper(cache_dir=cache_path)
        assert os.path.isdir(cache_path)
        assert scraper.cache_dir == cache_path or str(scraper.cache_dir) == cache_path
