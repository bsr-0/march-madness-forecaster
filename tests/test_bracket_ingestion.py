"""Tests for Selection Sunday bracket ingestion pipeline."""

import json
import tempfile
from pathlib import Path

import pytest

from src.data.team_name_resolver import TeamNameResolver
from src.data.scrapers.bracket_ingestion import (
    BracketIngestionPipeline,
    BracketTeam,
    TournamentBracketData,
    BIGDANCE_AVAILABLE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_bracket_team(
    canonical_id: str,
    seed: int,
    region: str,
    display_name: str = "",
    rating: float = 25.0,
) -> BracketTeam:
    return BracketTeam(
        canonical_id=canonical_id,
        display_name=display_name or canonical_id.replace("_", " ").title(),
        seed=seed,
        region=region,
        rating=rating,
        source="test",
        name_confidence=1.0,
    )


def _make_full_bracket(season: int = 2026) -> TournamentBracketData:
    """Create a full 64-team bracket (16 seeds × 4 regions)."""
    teams = []
    regions = ["East", "West", "South", "Midwest"]
    # Use recognizable team names for the top seeds
    top_teams = {
        "East": ["duke", "alabama", "purdue", "connecticut",
                  "marquette", "baylor", "creighton", "clemson",
                  "drake", "nevada", "butler", "richmond",
                  "vermont", "oakland", "wagner", "howard"],
        "West": ["houston", "arizona", "kansas", "tennessee",
                  "gonzaga", "brigham_young", "florida", "wisconsin",
                  "northwestern", "colorado", "oregon", "dayton",
                  "yale", "samford", "princeton", "navy"],
        "South": ["north_carolina", "iowa_state", "texas", "auburn",
                   "san_diego_state", "illinois", "michigan_state", "florida_atlantic",
                   "memphis", "nebraska", "new_mexico", "grand_canyon",
                   "toledo", "kent_state", "morehead_state", "grambling"],
        "Midwest": ["kentucky", "marquette", "indiana", "virginia",
                     "texas_am", "kansas_state", "missouri", "villanova",
                     "notre_dame", "utah_state", "xavier", "james_madison",
                     "oral_roberts", "liberty", "furman", "stetson"],
    }
    for region in regions:
        for seed_idx, team_id in enumerate(top_teams[region], start=1):
            teams.append(_make_bracket_team(team_id, seed_idx, region))
    return TournamentBracketData(
        season=season,
        teams=teams,
        source="test",
        fetched_at="2026-03-15T00:00:00+00:00",
    )


def _make_manual_json(teams_data: list, season: int = 2026) -> str:
    """Write a manual bracket JSON file and return its path."""
    data = {"season": season, "teams": teams_data}
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, prefix="bracket_test_"
    )
    json.dump(data, tmp)
    tmp.close()
    return tmp.name


# ---------------------------------------------------------------------------
# BracketTeam tests
# ---------------------------------------------------------------------------


class TestBracketTeam:
    """Tests for the BracketTeam dataclass."""

    def test_creation(self):
        team = BracketTeam(
            canonical_id="duke",
            display_name="Duke",
            seed=1,
            region="East",
        )
        assert team.canonical_id == "duke"
        assert team.seed == 1
        assert team.region == "East"
        assert team.rating == 0.0  # default
        assert team.name_confidence == 1.0  # default

    def test_with_optional_fields(self):
        team = BracketTeam(
            canonical_id="duke",
            display_name="Duke",
            seed=1,
            region="East",
            rating=32.5,
            conference="ACC",
            source="bigdance",
            name_confidence=0.99,
        )
        assert team.rating == 32.5
        assert team.conference == "ACC"
        assert team.source == "bigdance"


# ---------------------------------------------------------------------------
# TournamentBracketData tests
# ---------------------------------------------------------------------------


class TestTournamentBracketData:
    """Tests for the TournamentBracketData container."""

    def test_n_teams(self):
        bracket = _make_full_bracket()
        assert bracket.n_teams == 64

    def test_first_round_matchups_count(self):
        bracket = _make_full_bracket()
        matchups = bracket.get_first_round_matchups()
        # 8 matchups per region × 4 regions = 32
        assert len(matchups) == 32

    def test_first_round_matchup_seeding(self):
        """1-seed plays 16-seed, 2 plays 15, etc."""
        bracket = _make_full_bracket()
        matchups = bracket.get_first_round_matchups()
        east_matchups = [m for m in matchups if m[0].region == "East"]
        seed_pairs = sorted([(m[0].seed, m[1].seed) for m in east_matchups])
        expected_pairs = [
            (1, 16), (2, 15), (3, 14), (4, 13),
            (5, 12), (6, 11), (7, 10), (8, 9),
        ]
        assert seed_pairs == expected_pairs

    def test_to_pipeline_json_structure(self):
        bracket = _make_full_bracket()
        result = bracket.to_pipeline_json()
        assert result["season"] == 2026
        assert result["source"] == "test"
        assert len(result["teams"]) == 64
        # Each team should have required fields
        team = result["teams"][0]
        assert "team_name" in team
        assert "team_id" in team
        assert "seed" in team
        assert "region" in team
        assert "school_slug" in team

    def test_to_pipeline_json_school_slug(self):
        bracket = TournamentBracketData(
            season=2026,
            teams=[_make_bracket_team("north_carolina", 1, "East")],
            source="test",
        )
        result = bracket.to_pipeline_json()
        assert result["teams"][0]["school_slug"] == "north-carolina"

    def test_to_espn_bracket_structure(self):
        bracket = _make_full_bracket()
        espn = bracket.to_espn_bracket()
        assert espn["season"] == 2026
        assert espn["format"] == "espn_tournament_challenge"
        assert "regions" in espn
        assert "first_round_matchups" in espn
        assert set(espn["regions"].keys()) == {"East", "West", "South", "Midwest"}

    def test_to_espn_bracket_matchups(self):
        bracket = _make_full_bracket()
        espn = bracket.to_espn_bracket()
        assert len(espn["first_round_matchups"]) == 32
        for matchup in espn["first_round_matchups"]:
            assert "higher_seed" in matchup
            assert "lower_seed" in matchup
            assert "region" in matchup
            assert matchup["higher_seed"]["seed"] < matchup["lower_seed"]["seed"]

    def test_to_espn_bracket_region_teams(self):
        bracket = _make_full_bracket()
        espn = bracket.to_espn_bracket()
        for region, data in espn["regions"].items():
            assert len(data["teams"]) == 16
            seeds = [t["seed"] for t in data["teams"]]
            assert sorted(seeds) == list(range(1, 17))

    def test_empty_bracket(self):
        bracket = TournamentBracketData(season=2026, teams=[], source="test")
        assert bracket.n_teams == 0
        assert bracket.get_first_round_matchups() == []

    def test_resolution_warnings(self):
        bracket = TournamentBracketData(
            season=2026,
            teams=[],
            source="test",
            resolution_warnings=["Low confidence for 'Unknown Team'"],
        )
        assert len(bracket.resolution_warnings) == 1


# ---------------------------------------------------------------------------
# BracketIngestionPipeline tests
# ---------------------------------------------------------------------------


class TestBracketIngestionPipelineManualJSON:
    """Tests for loading bracket from manual JSON files."""

    def test_load_manual_json(self):
        teams_data = [
            {"team_name": "Duke", "seed": 1, "region": "East"},
            {"team_name": "North Carolina", "seed": 2, "region": "East"},
            {"team_name": "Kentucky", "seed": 3, "region": "East"},
            {"team_name": "Kansas", "seed": 4, "region": "East"},
        ]
        path = _make_manual_json(teams_data)
        pipeline = BracketIngestionPipeline(season=2026, cache_dir=tempfile.mkdtemp())
        bracket = pipeline.fetch(source=path)

        assert bracket.n_teams == 4
        assert bracket.source == f"manual:{path}"

        ids = {t.canonical_id for t in bracket.teams}
        assert "duke" in ids
        assert "north_carolina" in ids
        assert "kentucky" in ids
        assert "kansas" in ids

    def test_manual_json_resolves_abbreviations(self):
        teams_data = [
            {"team_name": "UConn", "seed": 1, "region": "East"},
            {"team_name": "LSU", "seed": 2, "region": "West"},
            {"team_name": "BYU", "seed": 3, "region": "South"},
        ]
        path = _make_manual_json(teams_data)
        pipeline = BracketIngestionPipeline(season=2026, cache_dir=tempfile.mkdtemp())
        bracket = pipeline.fetch(source=path)

        ids = {t.canonical_id for t in bracket.teams}
        assert "connecticut" in ids
        assert "louisiana_state" in ids
        assert "brigham_young" in ids

    def test_manual_json_with_name_key(self):
        """Support both 'team_name' and 'name' keys."""
        teams_data = [
            {"name": "Duke", "seed": 1, "region": "East"},
        ]
        path = _make_manual_json(teams_data)
        pipeline = BracketIngestionPipeline(season=2026, cache_dir=tempfile.mkdtemp())
        bracket = pipeline.fetch(source=path)
        assert bracket.teams[0].canonical_id == "duke"

    def test_manual_json_low_confidence_warning(self):
        teams_data = [
            {"team_name": "xyzzy_not_a_real_team", "seed": 16, "region": "East"},
        ]
        path = _make_manual_json(teams_data)
        pipeline = BracketIngestionPipeline(season=2026, cache_dir=tempfile.mkdtemp())
        bracket = pipeline.fetch(source=path)
        assert len(bracket.resolution_warnings) > 0

    def test_manual_json_preserves_optional_fields(self):
        teams_data = [
            {
                "team_name": "Duke",
                "seed": 1,
                "region": "East",
                "rating": 30.5,
                "conference": "ACC",
            },
        ]
        path = _make_manual_json(teams_data)
        pipeline = BracketIngestionPipeline(season=2026, cache_dir=tempfile.mkdtemp())
        bracket = pipeline.fetch(source=path)
        team = bracket.teams[0]
        assert team.rating == 30.5
        assert team.conference == "ACC"


class TestBracketIngestionPipelineSave:
    """Tests for saving bracket to JSON."""

    def test_save_creates_file(self):
        bracket = _make_full_bracket()
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = BracketIngestionPipeline(season=2026, cache_dir=tmpdir)
            path = pipeline.save(bracket)
            assert Path(path).exists()

    def test_save_roundtrip(self):
        """Save then load should produce equivalent data."""
        bracket = _make_full_bracket()
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = BracketIngestionPipeline(season=2026, cache_dir=tmpdir)
            path = pipeline.save(bracket)

            # Load back
            with open(path, "r") as f:
                loaded = json.load(f)

            assert loaded["season"] == 2026
            assert len(loaded["teams"]) == 64
            assert "resolution_warnings" in loaded

    def test_save_custom_path(self):
        bracket = _make_full_bracket()
        with tempfile.TemporaryDirectory() as tmpdir:
            custom_path = str(Path(tmpdir) / "custom_bracket.json")
            pipeline = BracketIngestionPipeline(season=2026, cache_dir=tmpdir)
            result_path = pipeline.save(bracket, path=custom_path)
            assert result_path == custom_path
            assert Path(custom_path).exists()


class TestBracketIngestionPipelineAuto:
    """Tests for auto-source selection."""

    def test_auto_raises_when_no_source_and_bigdance_unavailable(self):
        """When bigdance is not installed and no fallback, should raise."""
        pipeline = BracketIngestionPipeline(
            season=2026, cache_dir=tempfile.mkdtemp()
        )
        if not BIGDANCE_AVAILABLE:
            with pytest.raises(RuntimeError, match="No bracket source available"):
                pipeline.fetch(source="auto")

    def test_bigdance_source_raises_when_unavailable(self):
        if not BIGDANCE_AVAILABLE:
            pipeline = BracketIngestionPipeline(
                season=2026, cache_dir=tempfile.mkdtemp()
            )
            with pytest.raises(ImportError, match="bigdance"):
                pipeline.fetch(source="bigdance")


class TestBracketIngestionPipelineResolver:
    """Tests for name resolution integration."""

    def test_custom_resolver(self):
        """Pipeline should accept a custom resolver."""
        extra = {"custom_team": ["My Custom School"]}
        resolver = TeamNameResolver(extra_aliases=extra)
        pipeline = BracketIngestionPipeline(
            season=2026,
            cache_dir=tempfile.mkdtemp(),
            resolver=resolver,
        )
        teams_data = [{"team_name": "My Custom School", "seed": 16, "region": "East"}]
        path = _make_manual_json(teams_data)
        bracket = pipeline.fetch(source=path)
        assert bracket.teams[0].canonical_id == "custom_team"

    def test_resolver_confidence_stored(self):
        teams_data = [
            {"team_name": "Duke", "seed": 1, "region": "East"},
        ]
        path = _make_manual_json(teams_data)
        pipeline = BracketIngestionPipeline(
            season=2026, cache_dir=tempfile.mkdtemp()
        )
        bracket = pipeline.fetch(source=path)
        assert bracket.teams[0].name_confidence >= 0.90


# ---------------------------------------------------------------------------
# Integration: end-to-end pipeline JSON → prediction pipeline format
# ---------------------------------------------------------------------------


class TestEndToEndPipelineJSON:
    """Full pipeline: manual JSON → bracket → pipeline JSON → ESPN."""

    def test_manual_to_pipeline_to_espn(self):
        """Full flow: create bracket JSON, ingest, convert for pipeline and ESPN."""
        teams_data = []
        regions = ["East", "West", "South", "Midwest"]
        team_pool = [
            "Duke", "North Carolina", "Kansas", "Kentucky",
            "Houston", "Arizona", "Purdue", "Alabama",
            "UConn", "Gonzaga", "Tennessee", "Marquette",
            "Baylor", "Creighton", "Auburn", "Iowa State",
        ]
        for i, region in enumerate(regions):
            for seed in range(1, 17):
                idx = (i * 16 + seed - 1) % len(team_pool)
                teams_data.append({
                    "team_name": team_pool[idx],
                    "seed": seed,
                    "region": region,
                })
        path = _make_manual_json(teams_data)

        pipeline = BracketIngestionPipeline(
            season=2026, cache_dir=tempfile.mkdtemp()
        )
        bracket = pipeline.fetch(source=path)
        assert bracket.n_teams == 64

        # Pipeline JSON
        pjson = bracket.to_pipeline_json()
        assert len(pjson["teams"]) == 64
        for t in pjson["teams"]:
            assert t["team_id"]  # canonical ID is populated
            assert t["school_slug"]  # slug is populated
            assert 1 <= t["seed"] <= 16

        # ESPN format
        espn = bracket.to_espn_bracket()
        assert len(espn["first_round_matchups"]) == 32

        # Save
        with tempfile.TemporaryDirectory() as tmpdir:
            saved = pipeline.save(bracket, str(Path(tmpdir) / "test.json"))
            with open(saved) as f:
                reloaded = json.load(f)
            assert reloaded["season"] == 2026
            assert len(reloaded["teams"]) == 64
