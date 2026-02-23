"""Tests for roster_enrichment cross-year eligibility/transfer inference."""

import json
import os
import tempfile

import pytest

from src.data.scrapers.roster_enrichment import RosterEnrichment


class TestNormalizePlayerName:
    def test_basic(self):
        assert RosterEnrichment.normalize_player_name("John Smith") == "john_smith"

    def test_suffix_jr(self):
        assert RosterEnrichment.normalize_player_name("Marcus Jones Jr.") == "marcus_jones"
        assert RosterEnrichment.normalize_player_name("Marcus Jones Jr") == "marcus_jones"

    def test_suffix_iii(self):
        assert RosterEnrichment.normalize_player_name("Robert Williams III") == "robert_williams"

    def test_suffix_sr(self):
        assert RosterEnrichment.normalize_player_name("Tim Duncan Sr.") == "tim_duncan"

    def test_dots_in_initials(self):
        assert RosterEnrichment.normalize_player_name("D.J. Burns") == "dj_burns"

    def test_hyphens(self):
        assert RosterEnrichment.normalize_player_name("Shai Gilgeous-Alexander") == "shai_gilgeous_alexander"

    def test_whitespace(self):
        assert RosterEnrichment.normalize_player_name("  John   Smith  ") == "john_smith"

    def test_empty(self):
        assert RosterEnrichment.normalize_player_name("") == ""
        assert RosterEnrichment.normalize_player_name(None) == ""


class TestPlayerKey:
    def test_espn_numeric_id(self):
        player = {"player_id": "4433225", "name": "John Smith"}
        assert RosterEnrichment.player_key(player) == "espn:4433225"

    def test_non_numeric_id_falls_back(self):
        player = {"player_id": "duke_john_smith", "name": "John Smith"}
        assert RosterEnrichment.player_key(player) == "john_smith"

    def test_missing_id_falls_back(self):
        player = {"name": "John Smith"}
        assert RosterEnrichment.player_key(player) == "john_smith"

    def test_empty_id_falls_back(self):
        player = {"player_id": "", "name": "John Smith"}
        assert RosterEnrichment.player_key(player) == "john_smith"


def _make_roster_json(year, teams):
    """Helper to build a roster payload dict.

    teams: list of (team_id, team_name, [(player_id, name, mpg, games_played)])
    """
    team_list = []
    for team_id, team_name, players in teams:
        player_list = []
        for pid, pname, mpg, gp in players:
            player_list.append({
                "player_id": pid,
                "name": pname,
                "position": "PG",
                "minutes_per_game": mpg,
                "games_played": gp,
                "games_started": gp,
                "rapm_offensive": None,
                "rapm_defensive": None,
                "warp": 0.1,
                "box_plus_minus": 2.0,
                "usage_rate": 20.0,
                "true_shooting_pct": 0.55,
                "effective_fg_pct": 0.50,
                "points_per_game": 10.0,
                "rebounds_per_game": 4.0,
                "assists_per_game": 3.0,
                "steals_per_game": 1.0,
                "blocks_per_game": 0.5,
                "turnovers_per_game": 2.0,
                "injury_status": "healthy",
                "is_transfer": False,
                "eligibility_year": 1,
            })
        team_list.append({
            "team_id": team_id,
            "team_name": team_name,
            "players": player_list,
            "stints": [],
        })
    return {"year": year, "teams": team_list}


@pytest.fixture
def enrichment_dir():
    """Create a temp dir with 3 years of synthetic roster data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Year 2022: freshman class
        y2022 = _make_roster_json(2022, [
            ("duke", "Duke", [
                ("1001", "Player A", 30.0, 30),
                ("1002", "Player B", 25.0, 28),
                ("1003", "Player C", 20.0, 30),
            ]),
            ("unc", "UNC", [
                ("2001", "Player D", 28.0, 30),
                ("2002", "Player E", 22.0, 25),
            ]),
        ])

        # Year 2023: A returns to Duke, B transfers to UNC, C is gone (graduated/left)
        y2023 = _make_roster_json(2023, [
            ("duke", "Duke", [
                ("1001", "Player A", 32.0, 32),  # returnee → SO
                ("3001", "Player F", 15.0, 20),  # new freshman
            ]),
            ("unc", "UNC", [
                ("2001", "Player D", 30.0, 32),  # returnee → SO
                ("2002", "Player E", 24.0, 30),  # returnee → SO
                ("1002", "Player B", 20.0, 28),  # transfer from Duke
            ]),
        ])

        # Year 2024: A still at Duke (JR), D still at UNC (JR), B still at UNC (JR)
        y2024 = _make_roster_json(2024, [
            ("duke", "Duke", [
                ("1001", "Player A", 33.0, 33),  # JR
                ("3001", "Player F", 20.0, 30),  # SO
                ("4001", "Player G", 10.0, 15),  # new freshman
            ]),
            ("unc", "UNC", [
                ("2001", "Player D", 31.0, 33),  # JR
                ("1002", "Player B", 22.0, 30),  # JR (was transfer last year, returnee now)
            ]),
        ])

        for year, data in [(2022, y2022), (2023, y2023), (2024, y2024)]:
            with open(os.path.join(tmpdir, f"cbbpy_rosters_{year}.json"), "w") as f:
                json.dump(data, f)

        yield tmpdir


class TestEligibilityYear:
    def test_freshman_no_prior(self):
        history = {"p1": [(2023, "duke", 20.0)]}
        assert RosterEnrichment.compute_eligibility_year("p1", 2023, history) == 1

    def test_sophomore_one_prior(self):
        history = {"p1": [(2022, "duke", 20.0), (2023, "duke", 25.0)]}
        assert RosterEnrichment.compute_eligibility_year("p1", 2023, history) == 2

    def test_junior_two_prior(self):
        history = {"p1": [(2021, "duke", 15.0), (2022, "duke", 20.0), (2023, "duke", 25.0)]}
        assert RosterEnrichment.compute_eligibility_year("p1", 2023, history) == 3

    def test_senior_three_prior(self):
        history = {"p1": [
            (2020, "duke", 10.0), (2021, "duke", 15.0),
            (2022, "duke", 20.0), (2023, "duke", 25.0),
        ]}
        assert RosterEnrichment.compute_eligibility_year("p1", 2023, history) == 4

    def test_graduate_cap_at_5(self):
        history = {"p1": [
            (2019, "duke", 5.0), (2020, "duke", 10.0), (2021, "duke", 15.0),
            (2022, "duke", 20.0), (2023, "duke", 25.0), (2024, "duke", 25.0),
        ]}
        # 5 prior years → would be 6, capped at 5
        assert RosterEnrichment.compute_eligibility_year("p1", 2024, history) == 5

    def test_unknown_player(self):
        history = {}
        assert RosterEnrichment.compute_eligibility_year("unknown", 2023, history) == 1

    def test_gap_year(self):
        """Player misses a year — still counts prior appearances."""
        history = {"p1": [(2020, "duke", 20.0), (2022, "duke", 25.0)]}
        # 2022: 1 prior year (2020) → SO
        assert RosterEnrichment.compute_eligibility_year("p1", 2022, history) == 2


class TestIsTransfer:
    def test_returnee(self):
        history = {"p1": [(2022, "duke", 20.0), (2023, "duke", 25.0)]}
        assert RosterEnrichment.compute_is_transfer("p1", 2023, "duke", history) is False

    def test_transfer(self):
        history = {"p1": [(2022, "duke", 20.0), (2023, "unc", 25.0)]}
        assert RosterEnrichment.compute_is_transfer("p1", 2023, "unc", history) is True

    def test_freshman_no_prior(self):
        history = {"p1": [(2023, "duke", 20.0)]}
        assert RosterEnrichment.compute_is_transfer("p1", 2023, "duke", history) is False

    def test_unknown_player(self):
        history = {}
        assert RosterEnrichment.compute_is_transfer("unknown", 2023, "duke", history) is False

    def test_gap_year_different_team(self):
        """Player on duke in 2020, absent 2021, appears at unc in 2022."""
        history = {"p1": [(2020, "duke", 20.0), (2022, "unc", 25.0)]}
        assert RosterEnrichment.compute_is_transfer("p1", 2022, "unc", history) is True

    def test_gap_year_same_team(self):
        """Player on duke in 2020, absent 2021, returns to duke in 2022."""
        history = {"p1": [(2020, "duke", 20.0), (2022, "duke", 25.0)]}
        assert RosterEnrichment.compute_is_transfer("p1", 2022, "duke", history) is False


class TestEnrichAll:
    def test_round_trip(self, enrichment_dir):
        enricher = RosterEnrichment(roster_dir=enrichment_dir)
        summary = enricher.enrich_all(start_year=2022, end_year=2024)

        assert summary["years_processed"] == 3
        assert summary["total_players_enriched"] > 0
        assert summary["total_transfers"] > 0

        # Verify 2023 enrichment
        with open(os.path.join(enrichment_dir, "cbbpy_rosters_2023.json")) as f:
            data = json.load(f)

        # Find Player A on Duke (returnee, SO)
        duke = next(t for t in data["teams"] if t["team_id"] == "duke")
        player_a = next(p for p in duke["players"] if p["player_id"] == "1001")
        assert player_a["eligibility_year"] == 2
        assert player_a["is_transfer"] is False

        # Find Player B on UNC (transfer from Duke, SO)
        unc = next(t for t in data["teams"] if t["team_id"] == "unc")
        player_b = next(p for p in unc["players"] if p["player_id"] == "1002")
        assert player_b["eligibility_year"] == 2
        assert player_b["is_transfer"] is True

        # Find Player F on Duke (new freshman)
        player_f = next(p for p in duke["players"] if p["player_id"] == "3001")
        assert player_f["eligibility_year"] == 1
        assert player_f["is_transfer"] is False

    def test_2024_continued_enrichment(self, enrichment_dir):
        enricher = RosterEnrichment(roster_dir=enrichment_dir)
        enricher.enrich_all(start_year=2022, end_year=2024)

        with open(os.path.join(enrichment_dir, "cbbpy_rosters_2024.json")) as f:
            data = json.load(f)

        # Player A at Duke: 3rd year → JR
        duke = next(t for t in data["teams"] if t["team_id"] == "duke")
        player_a = next(p for p in duke["players"] if p["player_id"] == "1001")
        assert player_a["eligibility_year"] == 3
        assert player_a["is_transfer"] is False

        # Player B at UNC: 3rd year, same team as last year → NOT transfer
        unc = next(t for t in data["teams"] if t["team_id"] == "unc")
        player_b = next(p for p in unc["players"] if p["player_id"] == "1002")
        assert player_b["eligibility_year"] == 3
        assert player_b["is_transfer"] is False  # was transfer in 2023, returnee in 2024

        # Player F at Duke: 2nd year → SO
        player_f = next(p for p in duke["players"] if p["player_id"] == "3001")
        assert player_f["eligibility_year"] == 2
        assert player_f["is_transfer"] is False

    def test_enrichment_metadata(self, enrichment_dir):
        enricher = RosterEnrichment(roster_dir=enrichment_dir)
        enricher.enrich_all(start_year=2022, end_year=2024)

        with open(os.path.join(enrichment_dir, "cbbpy_rosters_2023.json")) as f:
            data = json.load(f)

        meta = data["enrichment_metadata"]
        assert "enriched_at" in meta
        assert meta["total_players"] == 5  # A, F on Duke + D, E, B on UNC
        assert meta["transfers_detected"] == 1  # Player B
        assert isinstance(meta["eligibility_distribution"], dict)
        assert meta["years_cross_referenced"] == [2022, 2023, 2024]

    def test_first_year_all_freshmen(self, enrichment_dir):
        enricher = RosterEnrichment(roster_dir=enrichment_dir)
        enricher.enrich_all(start_year=2022, end_year=2024)

        with open(os.path.join(enrichment_dir, "cbbpy_rosters_2022.json")) as f:
            data = json.load(f)

        # First year in dataset — all players should be eligibility_year=1, is_transfer=False
        for team in data["teams"]:
            for player in team["players"]:
                assert player["eligibility_year"] == 1
                assert player["is_transfer"] is False

    def test_missing_years_graceful(self):
        """Enrichment works when only some years have roster files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Only create year 2023
            data = _make_roster_json(2023, [
                ("duke", "Duke", [("1001", "Player A", 30.0, 30)]),
            ])
            with open(os.path.join(tmpdir, "cbbpy_rosters_2023.json"), "w") as f:
                json.dump(data, f)

            enricher = RosterEnrichment(roster_dir=tmpdir)
            summary = enricher.enrich_all(start_year=2020, end_year=2025)

            assert summary["years_processed"] == 1
            assert summary["total_players_enriched"] == 1

    def test_separate_output_dir(self, enrichment_dir):
        """Output dir different from input dir."""
        with tempfile.TemporaryDirectory() as outdir:
            enricher = RosterEnrichment(roster_dir=enrichment_dir, output_dir=outdir)
            summary = enricher.enrich_all(start_year=2022, end_year=2024)

            assert summary["years_processed"] == 3
            # Output files should be in outdir
            assert os.path.exists(os.path.join(outdir, "cbbpy_rosters_2023.json"))
            # Original files should be unchanged
            with open(os.path.join(enrichment_dir, "cbbpy_rosters_2023.json")) as f:
                original = json.load(f)
            # Original should NOT have enrichment_metadata (it wasn't written back)
            assert "enrichment_metadata" not in original
