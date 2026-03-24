"""Pydantic output schemas for scraper validation.

Every scraper should validate its output through these schemas before
returning data to callers.  This catches upstream format changes (ESPN
API drift, HTML restructuring, etc.) early — at scrape time rather
than deep inside the ML pipeline where failures are harder to debug.

Usage::

    from .schemas import validate_roster_payload, validate_tournament_games

    payload = scraper.fetch_rosters(2026)
    payload = validate_roster_payload(payload)  # raises SchemaValidationError on bad data

    games = scraper.scrape_year(2024)
    games = validate_tournament_games(games)
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)


class SchemaValidationError(Exception):
    """Raised when scraper output fails schema validation."""


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class RoundName(str, Enum):
    R64 = "R64"
    R32 = "R32"
    S16 = "S16"
    E8 = "E8"
    F4 = "F4"
    NCG = "NCG"
    CHAMP = "CHAMP"


class PositionCode(str, Enum):
    PG = "PG"
    SG = "SG"
    SF = "SF"
    PF = "PF"
    C = "C"


class InjuryStatusCode(str, Enum):
    HEALTHY = "healthy"
    QUESTIONABLE = "questionable"
    DOUBTFUL = "doubtful"
    OUT = "out"
    SEASON_ENDING = "season_ending"


# ---------------------------------------------------------------------------
# ESPN Picks / Consensus
# ---------------------------------------------------------------------------

class PublicPicksSchema(BaseModel):
    """Validated public pick percentages for a single team."""

    team_id: str = Field(min_length=1)
    team_name: str = ""
    seed: int = Field(ge=0, le=16)
    region: str = ""
    round_of_64_pct: float = Field(ge=0.0, le=100.0, default=0.0)
    round_of_32_pct: float = Field(ge=0.0, le=100.0, default=0.0)
    sweet_16_pct: float = Field(ge=0.0, le=100.0, default=0.0)
    elite_8_pct: float = Field(ge=0.0, le=100.0, default=0.0)
    final_four_pct: float = Field(ge=0.0, le=100.0, default=0.0)
    champion_pct: float = Field(ge=0.0, le=100.0, default=0.0)

    @model_validator(mode="after")
    def round_percentages_are_monotonic(self) -> "PublicPicksSchema":
        """Each subsequent round should have equal or lower pick %."""
        pcts = [
            self.round_of_64_pct,
            self.round_of_32_pct,
            self.sweet_16_pct,
            self.elite_8_pct,
            self.final_four_pct,
            self.champion_pct,
        ]
        for i in range(len(pcts) - 1):
            if pcts[i + 1] > pcts[i] + 0.5:  # small tolerance
                logger.warning(
                    "Non-monotonic pick pcts for %s: R%d=%.1f > R%d=%.1f",
                    self.team_id, i + 2, pcts[i + 1], i + 1, pcts[i],
                )
        return self


class ConsensusDataSchema(BaseModel):
    """Validated consensus pick data from one or more sources."""

    teams: Dict[str, PublicPicksSchema] = Field(default_factory=dict)
    sources: List[str] = Field(default_factory=list)
    timestamp: Optional[str] = None


# ---------------------------------------------------------------------------
# Betting Markets
# ---------------------------------------------------------------------------

class BettingOddsSchema(BaseModel):
    """Validated odds for a single team from a sportsbook."""

    team_id: str = Field(min_length=1)
    team_name: str = ""
    season: int = Field(ge=2000, le=2100)
    source: str = Field(min_length=1)
    championship_odds: float = 0.0
    implied_probability: float = Field(ge=0.0, le=1.0)
    timestamp: str = ""
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)


class MarketConsensusSchema(BaseModel):
    """Validated aggregated market consensus."""

    team_probabilities: Dict[str, float] = Field(default_factory=dict)
    sources: List[str] = Field(default_factory=list)
    timestamp: str = ""
    vig_adjusted: bool = False

    @field_validator("team_probabilities")
    @classmethod
    def probabilities_in_range(cls, v: Dict[str, float]) -> Dict[str, float]:
        for team_id, prob in v.items():
            if not (0.0 <= prob <= 1.0):
                raise ValueError(
                    f"Probability for {team_id} out of range: {prob}"
                )
        return v


# ---------------------------------------------------------------------------
# Roster / Player Metrics (cbbpy_rosters, player_metrics)
# ---------------------------------------------------------------------------

class PlayerStatsSchema(BaseModel):
    """Validated player statistics from roster scraper."""

    player_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    position: PositionCode = PositionCode.PG
    minutes_per_game: float = Field(ge=0.0, le=50.0, default=0.0)
    games_played: int = Field(ge=0, default=0)
    games_started: int = Field(ge=0, default=0)
    rapm_offensive: float = Field(ge=-20.0, le=20.0, default=0.0)
    rapm_defensive: float = Field(ge=-20.0, le=20.0, default=0.0)
    warp: float = Field(ge=-5.0, le=10.0, default=0.0)
    box_plus_minus: float = Field(ge=-30.0, le=30.0, default=0.0)
    usage_rate: float = Field(ge=0.0, le=100.0, default=0.0)
    true_shooting_pct: float = Field(ge=0.0, le=1.5, default=0.0)
    effective_fg_pct: float = Field(ge=0.0, le=1.5, default=0.0)
    points_per_game: float = Field(ge=0.0, le=60.0, default=0.0)
    rebounds_per_game: float = Field(ge=0.0, le=30.0, default=0.0)
    assists_per_game: float = Field(ge=0.0, le=20.0, default=0.0)
    steals_per_game: float = Field(ge=0.0, le=10.0, default=0.0)
    blocks_per_game: float = Field(ge=0.0, le=10.0, default=0.0)
    turnovers_per_game: float = Field(ge=0.0, le=15.0, default=0.0)
    injury_status: InjuryStatusCode = InjuryStatusCode.HEALTHY
    is_transfer: bool = False
    eligibility_year: int = Field(ge=1, le=6, default=1)


class TeamRosterSchema(BaseModel):
    """Validated team roster."""

    team_id: str = Field(min_length=1)
    team_name: str = Field(min_length=1)
    players: List[PlayerStatsSchema] = Field(min_length=1)


class RosterPayloadSchema(BaseModel):
    """Validated full roster payload from cbbpy scraper."""

    year: int = Field(ge=2000, le=2100)
    teams: List[TeamRosterSchema] = Field(default_factory=list)
    timestamp: Optional[str] = None
    source: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Conference Seeds
# ---------------------------------------------------------------------------

class ConferenceSeedsSchema(BaseModel):
    """Validated conference tournament seeds."""

    seeds: Dict[str, Dict[str, int]] = Field(default_factory=dict)

    @field_validator("seeds")
    @classmethod
    def seeds_are_positive(cls, v: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, int]]:
        for conf, teams in v.items():
            for team_id, seed in teams.items():
                if seed < 1:
                    raise ValueError(
                        f"Invalid seed {seed} for {team_id} in {conf}"
                    )
        return v


# ---------------------------------------------------------------------------
# Tournament Results
# ---------------------------------------------------------------------------

class TournamentGameSchema(BaseModel):
    """Validated single tournament game result."""

    year: int = Field(ge=1985, le=2100)
    round_name: str = Field(min_length=1)
    region: str = ""
    team1_id: str = Field(min_length=1)
    team1_seed: int = Field(ge=0, le=16)
    team1_score: int = Field(ge=0)
    team2_id: str = Field(min_length=1)
    team2_seed: int = Field(ge=0, le=16)
    team2_score: int = Field(ge=0)
    team1_won: bool = True

    @model_validator(mode="after")
    def score_matches_winner(self) -> "TournamentGameSchema":
        if self.team1_score > 0 and self.team2_score > 0:
            if self.team1_score > self.team2_score and not self.team1_won:
                logger.warning(
                    "Score mismatch: %s (%d) > %s (%d) but team1_won=False",
                    self.team1_id, self.team1_score,
                    self.team2_id, self.team2_score,
                )
            elif self.team2_score > self.team1_score and self.team1_won:
                logger.warning(
                    "Score mismatch: %s (%d) < %s (%d) but team1_won=True",
                    self.team1_id, self.team1_score,
                    self.team2_id, self.team2_score,
                )
        return self


# ---------------------------------------------------------------------------
# Tournament Context
# ---------------------------------------------------------------------------

class PreseasonAPSchema(BaseModel):
    """Validated preseason AP rankings."""

    rankings: Dict[str, int] = Field(default_factory=dict)

    @field_validator("rankings")
    @classmethod
    def ranks_in_range(cls, v: Dict[str, int]) -> Dict[str, int]:
        for team_id, rank in v.items():
            if not (1 <= rank <= 25):
                raise ValueError(
                    f"AP rank for {team_id} out of range: {rank}"
                )
        return v


class CoachTournamentSchema(BaseModel):
    """Validated coach tournament experience record."""

    name: str = ""
    appearances: int = Field(ge=0, default=0)
    wins: int = Field(ge=0, default=0)
    losses: int = Field(ge=0, default=0)
    teams: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def wins_losses_consistent(self) -> "CoachTournamentSchema":
        if self.wins + self.losses > 0 and self.appearances == 0:
            logger.warning(
                "Coach %s has wins=%d losses=%d but appearances=0",
                self.name, self.wins, self.losses,
            )
        return self


# ---------------------------------------------------------------------------
# Injury Report
# ---------------------------------------------------------------------------

class InjuryEntrySchema(BaseModel):
    """Validated single injury report entry."""

    player_name: str = Field(min_length=1)
    team_id: str = Field(min_length=1)
    status: InjuryStatusCode = InjuryStatusCode.QUESTIONABLE
    injury_type: str = ""
    expected_return: str = ""
    report_date: str = ""
    source: str = ""


class TeamInjuryReportSchema(BaseModel):
    """Validated team injury report."""

    team_id: str = Field(min_length=1)
    reports: List[InjuryEntrySchema] = Field(default_factory=list)
    report_date: str = ""


# ---------------------------------------------------------------------------
# External Ratings
# ---------------------------------------------------------------------------

class ExternalRatingSchema(BaseModel):
    """Validated external rating entry."""

    system_name: str = Field(min_length=1)
    team_name: str = ""
    team_id: str = ""
    rating: float = 0.0
    ranking: int = Field(ge=0, default=0)
    normalized: float = Field(ge=0.0, le=1.0, default=0.0)


# ---------------------------------------------------------------------------
# Transfer Portal
# ---------------------------------------------------------------------------

class TransferEntrySchema(BaseModel):
    """Validated transfer portal entry."""

    player_id: str = ""
    player_name: str = Field(min_length=1)
    source_team_name: str = ""
    destination_team_name: str = ""
    entry_date: str = ""


# ---------------------------------------------------------------------------
# Validation helper functions
# ---------------------------------------------------------------------------

def validate_consensus_data(data: dict) -> dict:
    """Validate ESPN picks / consensus data dict.

    Returns the validated dict (with defaults filled in).
    Raises SchemaValidationError on structural issues.
    """
    try:
        # Validate each team entry
        teams = data.get("teams", {})
        validated_teams = {}
        skipped = 0
        for team_id, team_data in teams.items():
            try:
                team_data_with_id = {**team_data, "team_id": team_id}
                validated = PublicPicksSchema(**team_data_with_id)
                validated_teams[team_id] = validated.model_dump()
            except Exception as e:
                logger.warning("Skipping invalid team %s: %s", team_id, e)
                skipped += 1

        if skipped:
            logger.warning(
                "Consensus validation: skipped %d/%d teams",
                skipped, len(teams),
            )

        return {
            "teams": validated_teams,
            "sources": data.get("sources", []),
            "timestamp": data.get("timestamp"),
        }
    except Exception as e:
        raise SchemaValidationError(f"Consensus data validation failed: {e}") from e


def validate_betting_odds(
    odds_map: dict, season: int, source: str
) -> dict:
    """Validate betting odds dict (team_id -> odds info).

    Returns validated dict. Drops invalid entries with warnings.
    """
    validated = {}
    skipped = 0
    for team_id, odds_data in odds_map.items():
        try:
            if isinstance(odds_data, dict):
                entry = BettingOddsSchema(
                    team_id=team_id,
                    season=season,
                    source=source,
                    **{k: v for k, v in odds_data.items()
                       if k not in ("team_id", "season", "source")},
                )
            else:
                # Already a BettingMarketOdds dataclass
                entry = BettingOddsSchema(
                    team_id=odds_data.team_id,
                    team_name=odds_data.team_name,
                    season=odds_data.season,
                    source=odds_data.source,
                    championship_odds=odds_data.championship_odds,
                    implied_probability=odds_data.implied_probability,
                    timestamp=getattr(odds_data, "timestamp", ""),
                    confidence=getattr(odds_data, "confidence", 1.0),
                )
            validated[team_id] = entry.model_dump()
        except Exception as e:
            logger.warning("Skipping invalid odds for %s: %s", team_id, e)
            skipped += 1

    if skipped:
        logger.warning(
            "Betting odds validation: skipped %d/%d teams", skipped, len(odds_map)
        )
    return validated


def validate_roster_payload(payload: dict) -> dict:
    """Validate the full roster payload from cbbpy scraper.

    Returns validated dict. Drops invalid teams/players with warnings.
    """
    if not payload or not isinstance(payload, dict):
        return payload

    year = payload.get("year", 0)
    teams = payload.get("teams", [])
    if not teams:
        return payload

    validated_teams = []
    skipped_teams = 0
    skipped_players = 0

    for team_data in teams:
        try:
            players = team_data.get("players", [])
            valid_players = []
            for p in players:
                try:
                    validated_p = PlayerStatsSchema(**p)
                    valid_players.append(validated_p.model_dump())
                except Exception as e:
                    logger.warning(
                        "Skipping invalid player %s on %s: %s",
                        p.get("name", "?"), team_data.get("team_id", "?"), e,
                    )
                    skipped_players += 1

            if valid_players:
                validated_teams.append({
                    "team_id": team_data["team_id"],
                    "team_name": team_data["team_name"],
                    "players": valid_players,
                })
            else:
                skipped_teams += 1
        except Exception as e:
            logger.warning(
                "Skipping invalid team %s: %s",
                team_data.get("team_id", "?"), e,
            )
            skipped_teams += 1

    if skipped_teams or skipped_players:
        logger.warning(
            "Roster validation: skipped %d teams, %d players",
            skipped_teams, skipped_players,
        )

    result = {**payload, "teams": validated_teams}
    return result


def validate_conference_seeds(seeds: dict) -> dict:
    """Validate conference tournament seeds dict.

    Returns validated dict. Raises SchemaValidationError on structural issues.
    """
    try:
        validated = ConferenceSeedsSchema(seeds=seeds)
        return validated.model_dump()["seeds"]
    except Exception as e:
        raise SchemaValidationError(f"Conference seeds validation failed: {e}") from e


def validate_tournament_games(games: list) -> list:
    """Validate tournament game results list.

    Returns validated list. Drops invalid games with warnings.
    """
    validated = []
    skipped = 0
    for game in games:
        try:
            g = TournamentGameSchema(**game)
            validated.append(g.model_dump())
        except Exception as e:
            logger.warning("Skipping invalid tournament game: %s (%s)", e, game)
            skipped += 1

    if skipped:
        logger.warning(
            "Tournament games validation: skipped %d/%d games",
            skipped, len(games),
        )
    return validated


def validate_preseason_ap(rankings: dict) -> dict:
    """Validate preseason AP rankings. Raises on bad data."""
    try:
        validated = PreseasonAPSchema(rankings=rankings)
        return validated.model_dump()["rankings"]
    except Exception as e:
        raise SchemaValidationError(f"Preseason AP validation failed: {e}") from e


def validate_coach_tournament_data(coaches: dict) -> dict:
    """Validate coach tournament experience data.

    Returns validated dict. Drops invalid entries with warnings.
    """
    validated = {}
    skipped = 0
    for key, coach_data in coaches.items():
        try:
            c = CoachTournamentSchema(**coach_data)
            validated[key] = c.model_dump()
        except Exception as e:
            logger.warning("Skipping invalid coach %s: %s", key, e)
            skipped += 1

    if skipped:
        logger.warning(
            "Coach data validation: skipped %d/%d entries",
            skipped, len(coaches),
        )
    return validated


def validate_transfer_entries(entries: list) -> list:
    """Validate transfer portal entries list.

    Returns validated list. Drops invalid entries with warnings.
    """
    validated = []
    skipped = 0
    for entry in entries:
        try:
            t = TransferEntrySchema(**entry)
            validated.append(t.model_dump())
        except Exception as e:
            logger.warning("Skipping invalid transfer entry: %s", e)
            skipped += 1

    if skipped:
        logger.warning(
            "Transfer portal validation: skipped %d/%d entries",
            skipped, len(entries),
        )
    return validated


def validate_injury_reports(reports: dict) -> dict:
    """Validate team injury reports dict.

    Returns validated dict. Drops invalid entries with warnings.
    """
    validated = {}
    skipped = 0
    for team_id, report_data in reports.items():
        try:
            if isinstance(report_data, dict):
                entries = report_data.get("reports", report_data.get("players", []))
                report_date = report_data.get("report_date", "")
            else:
                # TeamInjuryReport dataclass
                entries = [
                    {
                        "player_name": r.player_name,
                        "team_id": r.team_id,
                        "status": r.status.value if hasattr(r.status, "value") else r.status,
                        "injury_type": r.injury_type,
                        "expected_return": r.expected_return,
                        "report_date": r.report_date,
                        "source": r.source,
                    }
                    for r in report_data.reports
                ]
                report_date = report_data.report_date

            valid_entries = []
            for entry in entries:
                try:
                    if isinstance(entry, dict):
                        e = InjuryEntrySchema(**entry)
                    else:
                        e = InjuryEntrySchema(
                            player_name=entry.player_name,
                            team_id=entry.team_id,
                            status=entry.status.value if hasattr(entry.status, "value") else entry.status,
                            injury_type=entry.injury_type,
                            expected_return=entry.expected_return,
                            report_date=entry.report_date,
                            source=entry.source,
                        )
                    valid_entries.append(e.model_dump())
                except Exception as e_err:
                    logger.warning(
                        "Skipping invalid injury entry for %s: %s",
                        team_id, e_err,
                    )
                    skipped += 1

            validated[team_id] = {
                "team_id": team_id,
                "reports": valid_entries,
                "report_date": report_date,
            }
        except Exception as e:
            logger.warning("Skipping invalid injury report for %s: %s", team_id, e)
            skipped += 1

    if skipped:
        logger.warning(
            "Injury report validation: skipped %d entries", skipped,
        )
    return validated
