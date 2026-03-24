"""Tests for scraper pydantic schema validation."""

import pytest
from src.data.scrapers.schemas import (
    BettingOddsSchema,
    CoachTournamentSchema,
    ConferenceSeedsSchema,
    ConsensusDataSchema,
    ExternalRatingSchema,
    InjuryEntrySchema,
    InjuryStatusCode,
    MarketConsensusSchema,
    PlayerStatsSchema,
    PositionCode,
    PreseasonAPSchema,
    PublicPicksSchema,
    RosterPayloadSchema,
    SchemaValidationError,
    TeamInjuryReportSchema,
    TeamRosterSchema,
    TournamentGameSchema,
    TransferEntrySchema,
    validate_betting_odds,
    validate_coach_tournament_data,
    validate_conference_seeds,
    validate_consensus_data,
    validate_injury_reports,
    validate_preseason_ap,
    validate_roster_payload,
    validate_tournament_games,
    validate_transfer_entries,
)
from pydantic import ValidationError


# -----------------------------------------------------------------------
# PublicPicksSchema
# -----------------------------------------------------------------------

class TestPublicPicksSchema:
    def test_valid_picks(self):
        p = PublicPicksSchema(
            team_id="duke",
            team_name="Duke",
            seed=1,
            region="East",
            round_of_64_pct=99.0,
            round_of_32_pct=92.0,
            sweet_16_pct=75.0,
            elite_8_pct=55.0,
            final_four_pct=35.0,
            champion_pct=22.0,
        )
        assert p.team_id == "duke"
        assert p.seed == 1

    def test_seed_out_of_range(self):
        with pytest.raises(ValidationError):
            PublicPicksSchema(team_id="duke", seed=17)

    def test_pct_out_of_range(self):
        with pytest.raises(ValidationError):
            PublicPicksSchema(team_id="duke", seed=1, champion_pct=101.0)

    def test_empty_team_id_rejected(self):
        with pytest.raises(ValidationError):
            PublicPicksSchema(team_id="", seed=1)

    def test_defaults(self):
        p = PublicPicksSchema(team_id="unc", seed=2)
        assert p.champion_pct == 0.0
        assert p.region == ""


# -----------------------------------------------------------------------
# BettingOddsSchema
# -----------------------------------------------------------------------

class TestBettingOddsSchema:
    def test_valid_odds(self):
        o = BettingOddsSchema(
            team_id="duke",
            team_name="Duke",
            season=2026,
            source="fanduel",
            championship_odds=500.0,
            implied_probability=0.167,
        )
        assert o.implied_probability == 0.167

    def test_probability_out_of_range(self):
        with pytest.raises(ValidationError):
            BettingOddsSchema(
                team_id="duke",
                season=2026,
                source="fanduel",
                implied_probability=1.5,
            )

    def test_confidence_out_of_range(self):
        with pytest.raises(ValidationError):
            BettingOddsSchema(
                team_id="duke",
                season=2026,
                source="fanduel",
                implied_probability=0.5,
                confidence=2.0,
            )


# -----------------------------------------------------------------------
# PlayerStatsSchema
# -----------------------------------------------------------------------

class TestPlayerStatsSchema:
    def test_valid_player(self):
        p = PlayerStatsSchema(
            player_id="duke_smith",
            name="John Smith",
            position=PositionCode.PG,
            minutes_per_game=30.5,
            games_played=25,
        )
        assert p.position == PositionCode.PG

    def test_minutes_too_high(self):
        with pytest.raises(ValidationError):
            PlayerStatsSchema(
                player_id="duke_smith",
                name="John Smith",
                minutes_per_game=55.0,
            )

    def test_usage_rate_range(self):
        with pytest.raises(ValidationError):
            PlayerStatsSchema(
                player_id="x",
                name="X",
                usage_rate=150.0,
            )


# -----------------------------------------------------------------------
# TournamentGameSchema
# -----------------------------------------------------------------------

class TestTournamentGameSchema:
    def test_valid_game(self):
        g = TournamentGameSchema(
            year=2024,
            round_name="R64",
            region="East",
            team1_id="duke",
            team1_seed=1,
            team1_score=75,
            team2_id="vermont",
            team2_seed=16,
            team2_score=60,
            team1_won=True,
        )
        assert g.team1_won is True

    def test_invalid_seed(self):
        with pytest.raises(ValidationError):
            TournamentGameSchema(
                year=2024,
                round_name="R64",
                team1_id="duke",
                team1_seed=17,
                team1_score=75,
                team2_id="vermont",
                team2_seed=16,
                team2_score=60,
            )


# -----------------------------------------------------------------------
# PreseasonAPSchema
# -----------------------------------------------------------------------

class TestPreseasonAPSchema:
    def test_valid_rankings(self):
        r = PreseasonAPSchema(rankings={"duke": 1, "unc": 5, "kansas": 25})
        assert r.rankings["duke"] == 1

    def test_rank_out_of_range(self):
        with pytest.raises(ValidationError):
            PreseasonAPSchema(rankings={"duke": 30})

    def test_rank_zero_rejected(self):
        with pytest.raises(ValidationError):
            PreseasonAPSchema(rankings={"duke": 0})


# -----------------------------------------------------------------------
# ConferenceSeedsSchema
# -----------------------------------------------------------------------

class TestConferenceSeedsSchema:
    def test_valid_seeds(self):
        s = ConferenceSeedsSchema(seeds={"ACC": {"duke": 1, "unc": 2}})
        assert s.seeds["ACC"]["duke"] == 1

    def test_zero_seed_rejected(self):
        with pytest.raises(ValidationError):
            ConferenceSeedsSchema(seeds={"ACC": {"duke": 0}})


# -----------------------------------------------------------------------
# CoachTournamentSchema
# -----------------------------------------------------------------------

class TestCoachTournamentSchema:
    def test_valid_coach(self):
        c = CoachTournamentSchema(
            name="Coach K",
            appearances=36,
            wins=101,
            losses=30,
            teams=["Duke"],
        )
        assert c.appearances == 36

    def test_negative_wins_rejected(self):
        with pytest.raises(ValidationError):
            CoachTournamentSchema(name="Bad", wins=-1)


# -----------------------------------------------------------------------
# Validate helper functions
# -----------------------------------------------------------------------

class TestValidateConsensusData:
    def test_valid_data(self):
        data = {
            "teams": {
                "duke": {
                    "team_name": "Duke",
                    "seed": 1,
                    "region": "East",
                    "round_of_64_pct": 99.0,
                    "champion_pct": 22.0,
                },
            },
            "sources": ["espn"],
            "timestamp": "2026-03-17T12:00:00Z",
        }
        result = validate_consensus_data(data)
        assert "duke" in result["teams"]

    def test_skips_invalid_team(self):
        data = {
            "teams": {
                "duke": {"team_name": "Duke", "seed": 1},
                "": {"team_name": "Empty", "seed": 1},  # invalid: empty ID
            },
        }
        result = validate_consensus_data(data)
        assert "duke" in result["teams"]

    def test_empty_teams(self):
        result = validate_consensus_data({"teams": {}})
        assert result["teams"] == {}


class TestValidateBettingOdds:
    def test_valid_odds_dict(self):
        odds = {
            "duke": {
                "team_name": "Duke",
                "championship_odds": 500.0,
                "implied_probability": 0.167,
                "timestamp": "2026-03-17",
                "confidence": 1.0,
            },
        }
        result = validate_betting_odds(odds, season=2026, source="fanduel")
        assert "duke" in result

    def test_skips_invalid_probability(self):
        odds = {
            "duke": {
                "team_name": "Duke",
                "championship_odds": 500.0,
                "implied_probability": 5.0,  # out of range
            },
        }
        result = validate_betting_odds(odds, season=2026, source="fanduel")
        assert "duke" not in result


class TestValidateRosterPayload:
    def test_valid_payload(self):
        payload = {
            "year": 2026,
            "teams": [
                {
                    "team_id": "duke",
                    "team_name": "Duke",
                    "players": [
                        {
                            "player_id": "duke_smith",
                            "name": "John Smith",
                            "position": "PG",
                            "minutes_per_game": 30.0,
                            "games_played": 25,
                        },
                    ],
                },
            ],
        }
        result = validate_roster_payload(payload)
        assert len(result["teams"]) == 1
        assert len(result["teams"][0]["players"]) == 1

    def test_skips_invalid_player(self):
        payload = {
            "year": 2026,
            "teams": [
                {
                    "team_id": "duke",
                    "team_name": "Duke",
                    "players": [
                        {
                            "player_id": "duke_smith",
                            "name": "John Smith",
                            "position": "PG",
                            "minutes_per_game": 30.0,
                        },
                        {
                            "player_id": "",  # invalid
                            "name": "Bad Player",
                            "minutes_per_game": 60.0,  # also invalid
                        },
                    ],
                },
            ],
        }
        result = validate_roster_payload(payload)
        assert len(result["teams"][0]["players"]) == 1

    def test_empty_payload_passthrough(self):
        assert validate_roster_payload({}) == {}
        assert validate_roster_payload(None) is None


class TestValidateTournamentGames:
    def test_valid_games(self):
        games = [
            {
                "year": 2024,
                "round_name": "R64",
                "region": "East",
                "team1_id": "duke",
                "team1_seed": 1,
                "team1_score": 75,
                "team2_id": "vermont",
                "team2_seed": 16,
                "team2_score": 60,
                "team1_won": True,
            },
        ]
        result = validate_tournament_games(games)
        assert len(result) == 1

    def test_skips_invalid_game(self):
        games = [
            {
                "year": 2024,
                "round_name": "R64",
                "team1_id": "duke",
                "team1_seed": 1,
                "team1_score": 75,
                "team2_id": "vermont",
                "team2_seed": 17,  # invalid
                "team2_score": 60,
            },
        ]
        result = validate_tournament_games(games)
        assert len(result) == 0


class TestValidateConferenceSeeds:
    def test_valid(self):
        result = validate_conference_seeds({"ACC": {"duke": 1, "unc": 2}})
        assert result["ACC"]["duke"] == 1

    def test_invalid_raises(self):
        with pytest.raises(SchemaValidationError):
            validate_conference_seeds({"ACC": {"duke": -1}})


class TestValidatePreseasonAP:
    def test_valid(self):
        result = validate_preseason_ap({"duke": 1, "kansas": 5})
        assert result["duke"] == 1

    def test_invalid_raises(self):
        with pytest.raises(SchemaValidationError):
            validate_preseason_ap({"duke": 30})


class TestValidateCoachData:
    def test_valid(self):
        result = validate_coach_tournament_data({
            "coach_k": {"name": "Coach K", "appearances": 36, "wins": 101, "losses": 30, "teams": ["Duke"]},
        })
        assert "coach_k" in result

    def test_skips_invalid(self):
        result = validate_coach_tournament_data({
            "good": {"name": "Good", "appearances": 5, "wins": 3, "losses": 2, "teams": []},
            "bad": {"name": "Bad", "wins": -1, "losses": 0, "appearances": 0, "teams": []},
        })
        assert "good" in result
        assert "bad" not in result


class TestValidateTransferEntries:
    def test_valid(self):
        entries = [{"player_name": "John Smith", "source_team_name": "Duke"}]
        result = validate_transfer_entries(entries)
        assert len(result) == 1

    def test_skips_empty_name(self):
        entries = [
            {"player_name": "John Smith"},
            {"player_name": ""},  # invalid
        ]
        result = validate_transfer_entries(entries)
        assert len(result) == 1


class TestValidateInjuryReports:
    def test_valid_dataclass_input(self):
        from src.data.scrapers.injury_report import (
            InjuryReport,
            TeamInjuryReport,
        )
        from src.data.models.player import InjuryStatus

        report = TeamInjuryReport(
            team_id="duke",
            reports=[
                InjuryReport(
                    player_name="John Smith",
                    team_id="duke",
                    status=InjuryStatus.QUESTIONABLE,
                    injury_type="ankle",
                ),
            ],
            report_date="2026-03-15",
        )
        result = validate_injury_reports({"duke": report})
        assert "duke" in result
        assert len(result["duke"]["reports"]) == 1

    def test_valid_dict_input(self):
        data = {
            "duke": {
                "team_id": "duke",
                "reports": [
                    {
                        "player_name": "John Smith",
                        "team_id": "duke",
                        "status": "questionable",
                        "injury_type": "ankle",
                    },
                ],
                "report_date": "2026-03-15",
            },
        }
        result = validate_injury_reports(data)
        assert "duke" in result


# -----------------------------------------------------------------------
# MarketConsensusSchema
# -----------------------------------------------------------------------

class TestMarketConsensusSchema:
    def test_valid(self):
        m = MarketConsensusSchema(
            team_probabilities={"duke": 0.15, "unc": 0.10},
            sources=["fanduel"],
            vig_adjusted=True,
        )
        assert m.team_probabilities["duke"] == 0.15

    def test_probability_out_of_range(self):
        with pytest.raises(ValidationError):
            MarketConsensusSchema(
                team_probabilities={"duke": 1.5},
                sources=["fanduel"],
            )


# -----------------------------------------------------------------------
# ExternalRatingSchema
# -----------------------------------------------------------------------

class TestExternalRatingSchema:
    def test_valid(self):
        r = ExternalRatingSchema(
            system_name="kenpom",
            team_name="Duke",
            team_id="duke",
            rating=25.5,
            ranking=3,
            normalized=0.95,
        )
        assert r.normalized == 0.95

    def test_normalized_out_of_range(self):
        with pytest.raises(ValidationError):
            ExternalRatingSchema(
                system_name="kenpom",
                normalized=1.5,
            )
