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
        """Each subsequent round should have equal or lower pick %.

        Hard-fails when a later round exceeds the prior by >2pp (clearly
        invalid data, not a rounding artifact).  Warns for 0.5–2pp
        violations which can arise from legitimate rounding.
        """
        round_labels = ["R64", "R32", "S16", "E8", "F4", "CHAMP"]
        pcts = [
            self.round_of_64_pct,
            self.round_of_32_pct,
            self.sweet_16_pct,
            self.elite_8_pct,
            self.final_four_pct,
            self.champion_pct,
        ]
        for i in range(len(pcts) - 1):
            if pcts[i + 1] > pcts[i] + 2.0:
                raise ValueError(
                    f"Non-monotonic pick pcts for {self.team_id}: "
                    f"{round_labels[i + 1]}={pcts[i + 1]:.1f} > "
                    f"{round_labels[i]}={pcts[i]:.1f} (exceeds 2pp tolerance)"
                )
            elif pcts[i + 1] > pcts[i] + 0.5:
                logger.warning(
                    "Non-monotonic pick pcts for %s: %s=%.1f > %s=%.1f",
                    self.team_id, round_labels[i + 1], pcts[i + 1],
                    round_labels[i], pcts[i],
                )
        return self


class ConsensusDataSchema(BaseModel):
    """Validated consensus pick data from one or more sources."""

    teams: Dict[str, PublicPicksSchema] = Field(default_factory=dict)
    sources: List[str] = Field(default_factory=list)
    timestamp: Optional[str] = None


# ---------------------------------------------------------------------------
# Bracket Structure Validation (cross-team constraints)
# ---------------------------------------------------------------------------


class BracketStructureError(Exception):
    """Raised when public picks violate hard bracket-math constraints.

    Individual team validation (monotonicity, range) is handled by
    ``PublicPicksSchema``.  This error covers *cross-team* invariants
    that only make sense when looking at the full 64-team bracket:
    championship picks summing to ~100%, first-round matchup
    complementarity, etc.
    """


def validate_bracket_structure(
    teams: Dict[str, Any],
    bracket_matchups: Optional[Dict[str, str]] = None,
    hard_tolerance_pct: float = 1.0,
    soft_tolerance_pct: float = 0.50,
) -> List[str]:
    """Validate cross-team bracket-structural constraints on public picks.

    These are constraints that ``PublicPicksSchema`` *cannot* enforce
    because they span multiple teams:

    * **Round-sum constraints** — if 64 teams enter R64 and 32 advance,
      the sum of R64 pick percentages should be ~3200% (each advancing
      slot = 100%, 32 slots).  Similarly R32→S16 yields sum ~1600%,
      and so on through CHAMP ~100%.
    * **First-round complementarity** — two teams that meet in R64
      should have R64 picks summing to ~100% (one of them advances).

    Tolerances are *relative* to the expected sum (not fixed pp) because
    expected sums range from 100% (CHAMP) to 3200% (R64).

    Args:
        teams: Validated team dict from ``validate_consensus_data``,
            keyed by team_id.  Each value has ``round_of_64_pct`` … ``champion_pct``.
        bracket_matchups: Optional mapping of team_id → opponent_team_id
            for first-round games.  When provided, enables the
            complementarity check.
        hard_tolerance_pct: Fraction of expected sum that triggers a
            hard failure (default 1.0 = 100%, i.e. sum must be within
            [0, 2×expected]).
        soft_tolerance_pct: Fraction of expected sum that triggers a
            soft warning (default 0.50 = 50%).

    Returns:
        List of human-readable warning strings for soft violations.

    Raises:
        BracketStructureError: When a hard constraint is violated
            (sum is so far off that the data is structurally broken).
    """
    warnings: List[str] = []
    if not teams:
        return warnings

    n_teams = len(teams)

    # Round-sum constraints only make statistical sense with a
    # representative portion of the bracket.  With < 32 teams the
    # per-seed distribution is too sparse for the sums to be meaningful.
    if n_teams < 32:
        return warnings

    # -- Collect per-round sums -------------------------------------------
    round_attrs = [
        ("R64", "round_of_64_pct"),
        ("R32", "round_of_32_pct"),
        ("S16", "sweet_16_pct"),
        ("E8", "elite_8_pct"),
        ("F4", "final_four_pct"),
        ("CHAMP", "champion_pct"),
    ]

    # Expected sums for a 64-team bracket (percentages, not fractions):
    #   R64: 32 teams advance  → sum ≈ 3200%
    #   R32: 16 advance        → sum ≈ 1600%
    #   S16: 8 advance         → sum ≈ 800%
    #   E8:  4 advance         → sum ≈ 400%
    #   F4:  2 advance         → sum ≈ 200%   (reach championship game)
    #   CHAMP: 1 wins          → sum ≈ 100%
    expected_sums = {
        "R64": 3200.0,
        "R32": 1600.0,
        "S16": 800.0,
        "E8": 400.0,
        "F4": 200.0,
        "CHAMP": 100.0,
    }

    # Scale expectations if we have fewer teams (e.g. partial data)
    scale = n_teams / 64.0

    for round_name, attr in round_attrs:
        total = 0.0
        for team_data in teams.values():
            if isinstance(team_data, dict):
                total += float(team_data.get(attr, 0.0))
            else:
                total += float(getattr(team_data, attr, 0.0))

        expected = expected_sums[round_name] * scale
        if expected <= 0:
            continue

        deviation = abs(total - expected)
        hard_threshold = expected * hard_tolerance_pct
        soft_threshold = expected * soft_tolerance_pct

        if deviation >= hard_threshold:
            raise BracketStructureError(
                f"{round_name} picks sum to {total:.1f}% "
                f"(expected ~{expected:.0f}% for {n_teams} teams, "
                f"deviation {deviation:.1f}pp exceeds "
                f"{hard_tolerance_pct:.0%} tolerance)"
            )
        elif deviation > soft_threshold:
            warnings.append(
                f"{round_name} picks sum to {total:.1f}% "
                f"(expected ~{expected:.0f}%, deviation {deviation:.1f}pp)"
            )

    # -- First-round matchup complementarity ------------------------------
    if bracket_matchups:
        checked = set()
        for team_a, team_b in bracket_matchups.items():
            pair = tuple(sorted([team_a, team_b]))
            if pair in checked:
                continue
            checked.add(pair)

            if team_a not in teams or team_b not in teams:
                continue

            data_a = teams[team_a]
            data_b = teams[team_b]
            if isinstance(data_a, dict):
                pct_a = float(data_a.get("round_of_64_pct", 0.0))
                pct_b = float(data_b.get("round_of_64_pct", 0.0))
            else:
                pct_a = float(getattr(data_a, "round_of_64_pct", 0.0))
                pct_b = float(getattr(data_b, "round_of_64_pct", 0.0))

            pair_sum = pct_a + pct_b
            if abs(pair_sum - 100.0) > 10.0:
                warnings.append(
                    f"R64 matchup {team_a} vs {team_b}: "
                    f"pick sum {pair_sum:.1f}% (expected ~100%)"
                )

    return warnings


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
