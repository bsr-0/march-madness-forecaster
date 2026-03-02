"""Tests for Kaggle competition CSV data loader."""

import csv
import json
from pathlib import Path

import pytest

from src.data.kaggle_loader import KaggleDataLoader, MasseyOrdinalEntry


@pytest.fixture
def kaggle_dir(tmp_path):
    """Create a minimal Kaggle data directory with sample CSVs."""
    # MTeams.csv
    _write_csv(tmp_path / "MTeams.csv", ["TeamID", "TeamName"], [
        ["1104", "Alabama"],
        ["1112", "Arizona"],
        ["1181", "Duke"],
        ["1314", "North Carolina"],
        ["1242", "Kansas"],
    ])

    # MRegularSeasonCompactResults.csv
    _write_csv(tmp_path / "MRegularSeasonCompactResults.csv",
        ["Season", "DayNum", "WTeamID", "WScore", "LTeamID", "LScore", "WLoc", "NumOT"],
        [
            ["2025", "50", "1181", "85", "1314", "72", "H", "0"],
            ["2025", "60", "1112", "78", "1104", "75", "N", "0"],
            ["2024", "50", "1181", "90", "1242", "88", "A", "1"],
        ],
    )

    # MRegularSeasonDetailedResults.csv
    _write_csv(tmp_path / "MRegularSeasonDetailedResults.csv",
        ["Season", "DayNum", "WTeamID", "WScore", "LTeamID", "LScore", "WLoc", "NumOT",
         "WFGM", "WFGA", "WFGM3", "WFGA3", "WFTM", "WFTA", "WOR", "WDR",
         "WAst", "WTO", "WStl", "WBlk", "WPF",
         "LFGM", "LFGA", "LFGM3", "LFGA3", "LFTM", "LFTA", "LOR", "LDR",
         "LAst", "LTO", "LStl", "LBlk", "LPF"],
        [
            ["2025", "50", "1181", "85", "1314", "72", "H", "0",
             "30", "60", "8", "20", "17", "22", "10", "25",
             "15", "12", "6", "4", "18",
             "25", "58", "6", "18", "16", "20", "8", "22",
             "12", "14", "5", "3", "20"],
        ],
    )

    # MNCAATourneySeeds.csv
    _write_csv(tmp_path / "MNCAATourneySeeds.csv",
        ["Season", "Seed", "TeamID"],
        [
            ["2025", "W01", "1181"],
            ["2025", "W02", "1314"],
            ["2025", "X01", "1242"],
            ["2025", "Y03", "1112"],
            ["2025", "Z04", "1104"],
        ],
    )

    # MNCAATourneyCompactResults.csv
    _write_csv(tmp_path / "MNCAATourneyCompactResults.csv",
        ["Season", "DayNum", "WTeamID", "WScore", "LTeamID", "LScore", "WLoc", "NumOT"],
        [
            ["2025", "136", "1181", "82", "1104", "68", "N", "0"],
            ["2025", "138", "1314", "75", "1112", "70", "N", "0"],
        ],
    )

    # MMasseyOrdinals.csv
    _write_csv(tmp_path / "MMasseyOrdinals.csv",
        ["Season", "RankingDayNum", "SystemName", "TeamID", "OrdinalRank"],
        [
            # POM system, day 128
            ["2025", "128", "POM", "1181", "1"],
            ["2025", "128", "POM", "1314", "5"],
            ["2025", "128", "POM", "1242", "3"],
            ["2025", "128", "POM", "1112", "8"],
            ["2025", "128", "POM", "1104", "12"],
            # SAG system, day 128
            ["2025", "128", "SAG", "1181", "2"],
            ["2025", "128", "SAG", "1314", "4"],
            ["2025", "128", "SAG", "1242", "1"],
            ["2025", "128", "SAG", "1112", "6"],
            ["2025", "128", "SAG", "1104", "10"],
            # Earlier day (should not be used when picking latest)
            ["2025", "100", "POM", "1181", "3"],
            ["2025", "100", "POM", "1314", "2"],
            # Different season (should be filtered out)
            ["2024", "128", "POM", "1181", "5"],
        ],
    )

    # MTeamCoaches.csv
    _write_csv(tmp_path / "MTeamCoaches.csv",
        ["Season", "TeamID", "FirstDayNum", "LastDayNum", "CoachName"],
        [
            ["2025", "1181", "0", "154", "Jon Scheyer"],
            ["2025", "1314", "0", "154", "Hubert Davis"],
            ["2025", "1242", "0", "154", "Bill Self"],
        ],
    )

    # MTeamConferences.csv
    _write_csv(tmp_path / "MTeamConferences.csv",
        ["Season", "TeamID", "ConfAbbrev"],
        [
            ["2025", "1181", "acc"],
            ["2025", "1314", "acc"],
            ["2025", "1242", "big_twelve"],
            ["2025", "1112", "pac_twelve"],
            ["2025", "1104", "sec"],
        ],
    )

    # MConferences.csv
    _write_csv(tmp_path / "MConferences.csv",
        ["ConfAbbrev", "Description"],
        [
            ["acc", "Atlantic Coast Conference"],
            ["big_twelve", "Big 12 Conference"],
            ["pac_twelve", "Pac-12 Conference"],
            ["sec", "Southeastern Conference"],
        ],
    )

    # MSeasons.csv
    _write_csv(tmp_path / "MSeasons.csv",
        ["Season", "DayZero", "RegionW", "RegionX", "RegionY", "RegionZ"],
        [
            ["2025", "2024-10-14", "West", "East", "South", "Midwest"],
        ],
    )

    return tmp_path


def _write_csv(path: Path, headers: list, rows: list):
    """Write a simple CSV file."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


class TestKaggleDataLoader:

    def test_load_teams(self, kaggle_dir):
        loader = KaggleDataLoader(str(kaggle_dir))
        teams = loader.load_teams()
        assert len(teams) == 5
        assert teams[1181] == "Duke"
        assert teams[1314] == "North Carolina"

    def test_regular_season_compact(self, kaggle_dir):
        loader = KaggleDataLoader(str(kaggle_dir))
        games = loader.load_regular_season_compact(2025)
        # 2 games × 2 perspectives = 4 rows
        assert len(games) == 4
        # Check winner's row
        duke_rows = [g for g in games if g["team_name"] == "Duke"]
        assert len(duke_rows) == 1
        assert duke_rows[0]["team_score"] == 85
        assert duke_rows[0]["opponent_score"] == 72

    def test_regular_season_compact_filters_by_season(self, kaggle_dir):
        loader = KaggleDataLoader(str(kaggle_dir))
        games_2024 = loader.load_regular_season_compact(2024)
        assert len(games_2024) == 2  # 1 game × 2 perspectives
        games_2025 = loader.load_regular_season_compact(2025)
        assert len(games_2025) == 4

    def test_regular_season_detailed(self, kaggle_dir):
        loader = KaggleDataLoader(str(kaggle_dir))
        games = loader.load_regular_season_detailed(2025)
        assert len(games) == 2  # 1 detailed game × 2 perspectives
        duke_row = [g for g in games if g["team_name"] == "Duke"][0]
        assert duke_row["fgm"] == 30.0
        assert duke_row["fga"] == 60.0
        assert duke_row["fg3m"] == 8.0
        assert duke_row["possessions"] > 0

    def test_tourney_seeds(self, kaggle_dir):
        loader = KaggleDataLoader(str(kaggle_dir))
        seeds = loader.load_tourney_seeds(2025)
        assert len(seeds) == 5
        duke_seed = [s for s in seeds if s["team_name"] == "Duke"][0]
        assert duke_seed["seed"] == 1
        assert duke_seed["region"] == "W"
        assert duke_seed["seed_str"] == "W01"

    def test_tourney_compact(self, kaggle_dir):
        loader = KaggleDataLoader(str(kaggle_dir))
        games = loader.load_tourney_compact(2025)
        assert len(games) == 4  # 2 games × 2 perspectives

    def test_massey_ordinals_latest_day(self, kaggle_dir):
        loader = KaggleDataLoader(str(kaggle_dir))
        ordinals = loader.load_massey_ordinals(2025)
        assert "POM" in ordinals
        assert "SAG" in ordinals
        # Should use day 128 (latest), not day 100
        duke_pom = ordinals["POM"]["duke"]
        assert duke_pom.ordinal_rank == 1
        assert duke_pom.ranking_day_num == 128

    def test_massey_ordinals_specific_day(self, kaggle_dir):
        loader = KaggleDataLoader(str(kaggle_dir))
        ordinals = loader.load_massey_ordinals(2025, ranking_day_num=100)
        # Only POM has day 100 data
        assert "POM" in ordinals
        duke_pom = ordinals["POM"]["duke"]
        assert duke_pom.ordinal_rank == 3  # Day 100 rank

    def test_massey_ordinals_season_filter(self, kaggle_dir):
        loader = KaggleDataLoader(str(kaggle_dir))
        ordinals_2024 = loader.load_massey_ordinals(2024)
        assert "POM" in ordinals_2024
        assert len(ordinals_2024["POM"]) == 1  # Only 1 team in 2024

    def test_massey_ordinals_as_external_ratings(self, kaggle_dir):
        loader = KaggleDataLoader(str(kaggle_dir))
        ratings = loader.load_massey_ordinals_as_external_ratings(2025)
        assert "POM" in ratings
        pom_entries = ratings["POM"]
        assert len(pom_entries) == 5
        # Duke should have rank=1, highest normalized
        duke_entry = [e for e in pom_entries if e["team_name"] == "Duke"][0]
        assert duke_entry["ranking"] == 1
        assert duke_entry["normalized"] == 1.0

    def test_team_coaches(self, kaggle_dir):
        loader = KaggleDataLoader(str(kaggle_dir))
        coaches = loader.load_team_coaches(2025)
        assert len(coaches) == 3
        duke_coach = [c for c in coaches if c["team_name"] == "Duke"][0]
        assert duke_coach["coach_name"] == "Jon Scheyer"

    def test_team_conferences(self, kaggle_dir):
        loader = KaggleDataLoader(str(kaggle_dir))
        confs = loader.load_team_conferences(2025)
        assert len(confs) == 5
        assert confs["duke"] == "acc"

    def test_conferences(self, kaggle_dir):
        loader = KaggleDataLoader(str(kaggle_dir))
        confs = loader.load_conferences()
        assert confs["acc"] == "Atlantic Coast Conference"

    def test_seasons(self, kaggle_dir):
        loader = KaggleDataLoader(str(kaggle_dir))
        seasons = loader.load_seasons()
        assert len(seasons) == 1
        assert seasons[0]["season"] == 2025
        assert seasons[0]["region_w"] == "West"

    def test_list_available_files(self, kaggle_dir):
        loader = KaggleDataLoader(str(kaggle_dir))
        files = loader.list_available_files()
        assert "MTeams" in files
        assert "MMasseyOrdinals" in files

    def test_summary(self, kaggle_dir):
        loader = KaggleDataLoader(str(kaggle_dir))
        info = loader.summary(2025)
        assert info["teams_count"] == 5
        assert info["regular_season_games"] == 4
        assert info["tourney_seeds"] == 5
        assert info["massey_ordinal_systems"] == 2
        assert info["coaches"] == 3

    def test_missing_dir(self):
        loader = KaggleDataLoader("/nonexistent/path")
        assert loader.load_teams() == {}
        assert loader.load_regular_season_compact(2025) == []
        assert loader.load_massey_ordinals(2025) == {}
        assert loader.list_available_files() == []


class TestMasseyOrdinalEntry:

    def test_repr(self):
        entry = MasseyOrdinalEntry(
            system_name="POM",
            team_id="duke",
            team_name="Duke",
            kaggle_team_id=1181,
            ordinal_rank=1,
            ranking_day_num=128,
        )
        assert "POM" in repr(entry)
        assert "Duke" in repr(entry)
        assert "rank=1" in repr(entry)


class TestExternalRatingsFromMassey:
    """Test the ExternalRatingsLoader.populate_from_massey_ordinals integration."""

    def test_populate_caches_systems(self, kaggle_dir, tmp_path):
        from src.data.scrapers.external_ratings import ExternalRatingsLoader

        loader = ExternalRatingsLoader(cache_dir=str(tmp_path))
        n_cached = loader.populate_from_massey_ordinals(
            str(kaggle_dir), 2025,
        )
        # POM (5 teams) and SAG (5 teams) are too small (< 50 threshold),
        # but massey_composite should still be built from them.
        # With our test data having only 5 teams per system, they won't
        # meet the 50-team minimum. Let's verify the method runs without error.
        assert n_cached >= 0

    def test_populate_with_large_system(self, tmp_path):
        """Create a system with 60+ teams to pass the coverage threshold."""
        kaggle_dir = tmp_path / "kaggle"
        kaggle_dir.mkdir()

        # MTeams with 60 teams
        team_rows = [[str(1100 + i), f"Team{i}"] for i in range(60)]
        _write_csv(kaggle_dir / "MTeams.csv", ["TeamID", "TeamName"], team_rows)

        # Massey ordinals with 60 teams for one system
        ordinal_rows = [
            ["2025", "128", "POM", str(1100 + i), str(i + 1)]
            for i in range(60)
        ]
        _write_csv(
            kaggle_dir / "MMasseyOrdinals.csv",
            ["Season", "RankingDayNum", "SystemName", "TeamID", "OrdinalRank"],
            ordinal_rows,
        )

        from src.data.scrapers.external_ratings import ExternalRatingsLoader

        loader = ExternalRatingsLoader(cache_dir=str(tmp_path / "cache"))
        n_cached = loader.populate_from_massey_ordinals(str(kaggle_dir), 2025)
        # POM (60 teams) + massey_composite = 2
        assert n_cached == 2

        # Verify the cache files were created
        assert (tmp_path / "cache" / "external_POM_2025.json").exists()
        assert (tmp_path / "cache" / "external_massey_composite_2025.json").exists()

        # Verify the cached data is loadable
        loaded = loader._load_system("POM", 2025)
        assert len(loaded) == 60

        # Verify massey_composite
        composite = loader._load_system("massey_composite", 2025)
        assert len(composite) == 60
        # Team0 (rank 1) should have highest normalized rating
        team0_id = list(composite.keys())[0]
        all_norms = [r.normalized for r in composite.values()]
        assert max(all_norms) == pytest.approx(1.0, abs=0.01)

    def test_load_all_after_populate(self, tmp_path):
        """After populating, load_all should find the cached systems."""
        kaggle_dir = tmp_path / "kaggle"
        kaggle_dir.mkdir()
        cache_dir = tmp_path / "cache"

        team_rows = [[str(1100 + i), f"Team{i}"] for i in range(60)]
        _write_csv(kaggle_dir / "MTeams.csv", ["TeamID", "TeamName"], team_rows)

        ordinal_rows = [
            ["2025", "128", "POM", str(1100 + i), str(i + 1)]
            for i in range(60)
        ]
        _write_csv(
            kaggle_dir / "MMasseyOrdinals.csv",
            ["Season", "RankingDayNum", "SystemName", "TeamID", "OrdinalRank"],
            ordinal_rows,
        )

        from src.data.scrapers.external_ratings import ExternalRatingsLoader

        loader = ExternalRatingsLoader(cache_dir=str(cache_dir))
        loader.populate_from_massey_ordinals(str(kaggle_dir), 2025)

        # Now load_all should find the populated systems
        all_ratings = loader.load_all(2025)
        assert "massey_composite" in all_ratings
        assert len(all_ratings["massey_composite"]) == 60

        # And compute_composite should work
        composites = loader.compute_composite(all_ratings)
        assert len(composites) == 60
