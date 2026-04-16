"""
Shared contract for the march-madness MCP servers.

Both dev and prod servers expose tools with these exact return shapes.
TypedDicts are used instead of Pydantic so the schemas carry zero runtime
overhead — FastMCP serialises them to JSON automatically.
"""

from typing import Optional
from typing_extensions import TypedDict


class TeamRating(TypedDict):
    team_id: str
    team_name: str
    adj_off_eff: float
    adj_def_eff: float
    adj_tempo: float
    barthag: float
    seed: Optional[int]
    conference: str


class PublicPickEntry(TypedDict):
    team_id: str
    team_name: str
    seed: int
    region: str
    R64: float
    R32: float
    S16: float
    E8: float
    F4: float
    CHAMP: float


class BracketTeam(TypedDict):
    team_id: str
    team_name: str
    seed: int
    region: str
    conference: str
    school_slug: str


class MarketOddsEntry(TypedDict):
    team_id: str
    team_name: str
    source: str
    championship_odds: float  # American odds (e.g. +500)
    implied_probability: float  # vig-removed, 0-1


# CLI commands that run_cli will execute. Read-only pipeline queries only.
SAFE_CLI_COMMANDS: frozenset[str] = frozenset({"status", "pool-report", "eval"})
