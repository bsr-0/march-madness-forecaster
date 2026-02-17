"""Tests for injury data integration: scraper, severity model, positional depth."""

import json
import numpy as np
import pytest

from src.data.models.player import InjuryStatus, Player, Position, Roster
from src.data.scrapers.injury_report import (
    InjuryReport,
    InjuryReportScraper,
    InjurySeverityModel,
    PositionalDepthChart,
    TeamInjuryReport,
    apply_injury_reports_to_roster,
    INJURY_SEVERITY_PRIORS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_player(
    name: str,
    position: Position = Position.POINT_GUARD,
    minutes: float = 25.0,
    rapm_o: float = 2.0,
    rapm_d: float = 1.0,
    injury: InjuryStatus = InjuryStatus.HEALTHY,
    games: int = 30,
    warp: float = 1.0,
    bpm: float = 3.0,
    usage: float = 20.0,
) -> Player:
    return Player(
        player_id=f"p_{name.lower().replace(' ', '_')}",
        name=name,
        team_id="test_team",
        position=position,
        minutes_per_game=minutes,
        games_played=games,
        rapm_offensive=rapm_o,
        rapm_defensive=rapm_d,
        injury_status=injury,
        warp=warp,
        box_plus_minus=bpm,
        usage_rate=usage,
    )


def _make_roster(n_players: int = 12) -> Roster:
    positions = [
        Position.POINT_GUARD, Position.SHOOTING_GUARD, Position.SMALL_FORWARD,
        Position.POWER_FORWARD, Position.CENTER,
        Position.POINT_GUARD, Position.SHOOTING_GUARD, Position.SMALL_FORWARD,
        Position.POWER_FORWARD, Position.CENTER,
        Position.SHOOTING_GUARD, Position.POWER_FORWARD,
    ]
    players = []
    for i in range(min(n_players, len(positions))):
        players.append(
            _make_player(
                f"Player {i+1}",
                position=positions[i],
                minutes=35.0 - i * 2,
                rapm_o=3.0 - i * 0.3,
                rapm_d=2.0 - i * 0.2,
            )
        )
    return Roster(team_id="test_team", players=players)


# ---------------------------------------------------------------------------
# InjuryReportScraper
# ---------------------------------------------------------------------------


class TestInjuryReportScraper:
    def test_load_from_json(self, tmp_path):
        data = {
            "timestamp": "2026-03-15T12:00:00Z",
            "source": "espn",
            "teams": {
                "duke": {
                    "players": [
                        {
                            "name": "John Smith",
                            "status": "questionable",
                            "injury_type": "ankle",
                            "expected_return": "day-to-day",
                        },
                        {
                            "name": "Bob Jones",
                            "status": "out",
                            "injury_type": "knee",
                        },
                    ]
                },
                "unc": {
                    "players": [
                        {
                            "name": "Mike Davis",
                            "status": "doubtful",
                            "injury_type": "back",
                        },
                    ]
                },
            },
        }
        json_path = tmp_path / "injuries.json"
        with open(json_path, "w") as f:
            json.dump(data, f)

        scraper = InjuryReportScraper()
        reports = scraper.load_from_json(str(json_path))

        assert "duke" in reports
        assert "unc" in reports
        assert len(reports["duke"].reports) == 2
        assert reports["duke"].reports[0].status == InjuryStatus.QUESTIONABLE
        assert reports["duke"].reports[1].status == InjuryStatus.OUT
        assert reports["unc"].reports[0].injury_type == "back"

    def test_status_normalization(self):
        scraper = InjuryReportScraper()
        assert scraper._normalize_status("questionable") == InjuryStatus.QUESTIONABLE
        assert scraper._normalize_status("day-to-day") == InjuryStatus.QUESTIONABLE
        assert scraper._normalize_status("gtd") == InjuryStatus.QUESTIONABLE
        assert scraper._normalize_status("game-time decision") == InjuryStatus.QUESTIONABLE
        assert scraper._normalize_status("doubtful") == InjuryStatus.DOUBTFUL
        assert scraper._normalize_status("out") == InjuryStatus.OUT
        assert scraper._normalize_status("out for season") == InjuryStatus.SEASON_ENDING
        assert scraper._normalize_status("healthy") == InjuryStatus.HEALTHY
        assert scraper._normalize_status("probable") == InjuryStatus.HEALTHY

    def test_load_list_format(self, tmp_path):
        data = {
            "timestamp": "2026-03-15T12:00:00Z",
            "teams": [
                {
                    "team_id": "duke",
                    "players": [
                        {"name": "Test Player", "status": "out", "injury_type": "knee"},
                    ],
                }
            ],
        }
        json_path = tmp_path / "injuries_list.json"
        with open(json_path, "w") as f:
            json.dump(data, f)

        scraper = InjuryReportScraper()
        reports = scraper.load_from_json(str(json_path))
        assert "duke" in reports
        assert len(reports["duke"].reports) == 1


# ---------------------------------------------------------------------------
# InjurySeverityModel
# ---------------------------------------------------------------------------


class TestInjurySeverityModel:
    def test_healthy_returns_full_availability(self):
        model = InjurySeverityModel(random_seed=42)
        report = InjuryReport(
            player_name="Healthy Player",
            team_id="test",
            status=InjuryStatus.HEALTHY,
        )
        mean, std = model.estimate_availability(report)
        assert mean == 1.0
        assert std == 0.0

    def test_season_ending_returns_zero(self):
        model = InjurySeverityModel(random_seed=42)
        report = InjuryReport(
            player_name="ACL Player",
            team_id="test",
            status=InjuryStatus.SEASON_ENDING,
        )
        mean, std = model.estimate_availability(report)
        assert mean == 0.0
        assert std == 0.0

    def test_ankle_questionable_reasonable(self):
        model = InjurySeverityModel(random_seed=42)
        report = InjuryReport(
            player_name="Ankle Sprain",
            team_id="test",
            status=InjuryStatus.QUESTIONABLE,
            injury_type="ankle",
            expected_return="day-to-day",
        )
        mean, std = model.estimate_availability(report)
        # Ankle + day-to-day + questionable should be moderate-high availability
        assert 0.3 < mean < 0.9
        assert std > 0

    def test_knee_doubtful_lower_than_ankle_questionable(self):
        model = InjurySeverityModel(random_seed=42)

        ankle_q = InjuryReport(
            player_name="A", team_id="t", status=InjuryStatus.QUESTIONABLE,
            injury_type="ankle", expected_return="day-to-day",
        )
        knee_d = InjuryReport(
            player_name="B", team_id="t", status=InjuryStatus.DOUBTFUL,
            injury_type="knee", expected_return="week-to-week",
        )

        ankle_mean, _ = model.estimate_availability(ankle_q)
        knee_mean, _ = model.estimate_availability(knee_d)
        assert knee_mean < ankle_mean

    def test_more_recovery_time_improves_availability(self):
        model = InjurySeverityModel(random_seed=42)
        report = InjuryReport(
            player_name="Test", team_id="t",
            status=InjuryStatus.QUESTIONABLE, injury_type="ankle",
        )
        mean_1day, _ = model.estimate_availability(report, days_until_game=1.0)
        mean_7day, _ = model.estimate_availability(report, days_until_game=7.0)
        assert mean_7day >= mean_1day

    def test_sample_availability_shape(self):
        model = InjurySeverityModel(random_seed=42)
        report = InjuryReport(
            player_name="Test", team_id="t",
            status=InjuryStatus.QUESTIONABLE, injury_type="ankle",
        )
        samples = model.sample_availability(report, n_samples=500)
        assert samples.shape == (500,)
        assert np.all(samples >= 0.0)
        assert np.all(samples <= 1.0)

    def test_severity_priors_cover_common_injuries(self):
        for injury_type in ["ankle", "knee", "concussion", "illness", "back", "hamstring"]:
            assert injury_type in INJURY_SEVERITY_PRIORS
            mean, std = INJURY_SEVERITY_PRIORS[injury_type]
            assert 0.0 <= mean <= 1.0
            assert 0.0 <= std <= 0.5


# ---------------------------------------------------------------------------
# PositionalDepthChart
# ---------------------------------------------------------------------------


class TestPositionalDepthChart:
    def test_analyze_returns_all_groups(self):
        roster = _make_roster()
        chart = PositionalDepthChart()
        depth = chart.analyze(roster)

        assert "guard" in depth
        assert "wing" in depth
        assert "forward" in depth
        assert "big" in depth

    def test_depth_score_range(self):
        roster = _make_roster()
        chart = PositionalDepthChart()
        depth = chart.analyze(roster)

        for group in depth.values():
            assert 0.0 <= group.depth_score <= 1.0

    def test_injury_impact_with_healthy_roster(self):
        roster = _make_roster()
        chart = PositionalDepthChart()
        impacts = chart.compute_injury_impact(roster)

        # Healthy roster should have near-zero impact
        assert impacts["total_impact"] < 0.1

    def test_injury_impact_increases_with_injuries(self):
        roster = _make_roster()
        chart = PositionalDepthChart()

        healthy_impacts = chart.compute_injury_impact(roster)

        # Injure the best player
        roster.players[0].injury_status = InjuryStatus.OUT
        injured_impacts = chart.compute_injury_impact(roster)

        assert injured_impacts["total_impact"] >= healthy_impacts["total_impact"]

    def test_positional_vulnerability_identifies_thin_positions(self):
        # Create roster with very thin guard depth
        players = [
            _make_player("Star PG", Position.POINT_GUARD, 35, 5.0, 3.0),
            _make_player("Bad PG backup", Position.POINT_GUARD, 5, 0.1, 0.1),
            _make_player("Good SG", Position.SHOOTING_GUARD, 30, 3.0, 2.0),
            _make_player("Good SF", Position.SMALL_FORWARD, 30, 3.0, 2.0),
            _make_player("Good PF", Position.POWER_FORWARD, 30, 3.0, 2.0),
            _make_player("Good C", Position.CENTER, 30, 3.0, 2.0),
            _make_player("Backup SF", Position.SMALL_FORWARD, 15, 1.5, 1.0),
            _make_player("Backup PF", Position.POWER_FORWARD, 15, 1.5, 1.0),
            _make_player("Backup C", Position.CENTER, 15, 1.5, 1.0),
        ]
        roster = Roster(team_id="thin_guards", players=players)

        chart = PositionalDepthChart()
        depth = chart.analyze(roster)

        # Guard depth should be lower than big depth
        assert depth["guard"].depth_score < depth["big"].depth_score

    def test_severity_model_integration(self):
        roster = _make_roster()
        roster.players[0].injury_status = InjuryStatus.QUESTIONABLE

        chart = PositionalDepthChart()
        model = InjurySeverityModel(random_seed=42)

        reports = {
            roster.players[0].name.lower(): InjuryReport(
                player_name=roster.players[0].name,
                team_id="test_team",
                status=InjuryStatus.QUESTIONABLE,
                injury_type="ankle",
                expected_return="day-to-day",
            )
        }

        impacts = chart.compute_injury_impact(
            roster,
            injury_reports=reports,
            severity_model=model,
        )
        # Should have non-trivial impact but less than OUT
        assert impacts["total_impact"] > 0


# ---------------------------------------------------------------------------
# apply_injury_reports_to_roster
# ---------------------------------------------------------------------------


class TestApplyInjuryReports:
    def test_updates_matching_players(self):
        roster = _make_roster()
        player_name = roster.players[0].name

        report = TeamInjuryReport(
            team_id="test_team",
            reports=[
                InjuryReport(
                    player_name=player_name,
                    team_id="test_team",
                    status=InjuryStatus.OUT,
                    injury_type="knee",
                )
            ],
        )

        updated = apply_injury_reports_to_roster(roster, report)
        assert updated == 1
        assert roster.players[0].injury_status == InjuryStatus.OUT

    def test_case_insensitive_matching(self):
        player = _make_player("John Smith")
        roster = Roster(team_id="test", players=[player])

        report = TeamInjuryReport(
            team_id="test",
            reports=[
                InjuryReport(
                    player_name="john smith",
                    team_id="test",
                    status=InjuryStatus.DOUBTFUL,
                )
            ],
        )

        updated = apply_injury_reports_to_roster(roster, report)
        assert updated == 1
        assert player.injury_status == InjuryStatus.DOUBTFUL

    def test_no_match_no_update(self):
        roster = _make_roster()
        report = TeamInjuryReport(
            team_id="test_team",
            reports=[
                InjuryReport(
                    player_name="Nonexistent Player",
                    team_id="test_team",
                    status=InjuryStatus.OUT,
                )
            ],
        )

        updated = apply_injury_reports_to_roster(roster, report)
        assert updated == 0

    def test_already_matching_status_not_counted(self):
        player = _make_player("Test Player", injury=InjuryStatus.OUT)
        roster = Roster(team_id="test", players=[player])

        report = TeamInjuryReport(
            team_id="test",
            reports=[
                InjuryReport(
                    player_name="Test Player",
                    team_id="test",
                    status=InjuryStatus.OUT,
                )
            ],
        )

        updated = apply_injury_reports_to_roster(roster, report)
        assert updated == 0  # Same status, no change
