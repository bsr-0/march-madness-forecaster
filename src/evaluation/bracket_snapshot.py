"""Bracket snapshot for Selection Sunday backtesting.

Captures the exact bracket structure as it would have been known on
Selection Sunday: teams, seeds, regions, first-round matchups, and
play-in games.

Play-in (First Four) games are always modeled as probabilistic — even
in historical backtests — because they had not yet been played at the
moment the bracket was announced.

Bracket data is loaded from Selection Sunday bracket files, NOT
reconstructed from tournament results (which would leak information).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TeamEntry:
    """A team in the bracket snapshot."""

    team_id: str
    seed: int
    region: str
    name: str = ""
    is_play_in: bool = False
    play_in_slot: int = -1  # -1 = not a play-in team


@dataclass(frozen=True)
class PlayInMatchup:
    """A First Four play-in game.

    On Selection Sunday the bracket is announced with play-in matchups
    but the games have not been played yet. These must be modeled
    probabilistically.
    """

    team_a: str  # team_id of first play-in team
    team_b: str  # team_id of second play-in team
    seed: int  # The shared seed (e.g., 16, 16, 11, 11)
    region: str
    slot_index: int  # Position in the 64-team bracket where the winner goes

    def teams(self) -> Tuple[str, str]:
        return (self.team_a, self.team_b)


@dataclass
class BracketSnapshot:
    """Complete bracket structure as known on Selection Sunday.

    Contains 68 teams (64 direct + 4 play-in pairs) organized by
    region and seed, plus the play-in matchups that must be resolved
    probabilistically.
    """

    year: int
    teams: List[TeamEntry] = field(default_factory=list)
    seeds: Dict[str, int] = field(default_factory=dict)
    regions: Dict[str, str] = field(default_factory=dict)
    first_round_matchups: List[str] = field(default_factory=list)  # 64 team IDs
    play_in_matchups: List[PlayInMatchup] = field(default_factory=list)

    @property
    def team_ids(self) -> List[str]:
        return [t.team_id for t in self.teams]

    @property
    def n_teams(self) -> int:
        return len(self.teams)

    @property
    def has_play_ins(self) -> bool:
        return len(self.play_in_matchups) > 0

    def get_teams_by_region(self) -> Dict[str, List[TeamEntry]]:
        """Group teams by region."""
        by_region: Dict[str, List[TeamEntry]] = {}
        for t in self.teams:
            by_region.setdefault(t.region, []).append(t)
        for region in by_region:
            by_region[region].sort(key=lambda t: t.seed)
        return by_region

    def get_play_in_teams(self) -> List[str]:
        """Return all team_ids involved in play-in games."""
        teams = []
        for pim in self.play_in_matchups:
            teams.extend([pim.team_a, pim.team_b])
        return teams

    def to_dict(self) -> Dict[str, Any]:
        return {
            "year": self.year,
            "teams": [
                {
                    "team_id": t.team_id,
                    "seed": t.seed,
                    "region": t.region,
                    "name": t.name,
                    "is_play_in": t.is_play_in,
                    "play_in_slot": t.play_in_slot,
                }
                for t in self.teams
            ],
            "seeds": self.seeds,
            "regions": self.regions,
            "first_round_matchups": self.first_round_matchups,
            "play_in_matchups": [
                {
                    "team_a": pim.team_a,
                    "team_b": pim.team_b,
                    "seed": pim.seed,
                    "region": pim.region,
                    "slot_index": pim.slot_index,
                }
                for pim in self.play_in_matchups
            ],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> BracketSnapshot:
        teams = [
            TeamEntry(
                team_id=t["team_id"],
                seed=t["seed"],
                region=t["region"],
                name=t.get("name", ""),
                is_play_in=t.get("is_play_in", False),
                play_in_slot=t.get("play_in_slot", -1),
            )
            for t in data.get("teams", [])
        ]
        play_ins = [
            PlayInMatchup(
                team_a=pim["team_a"],
                team_b=pim["team_b"],
                seed=pim["seed"],
                region=pim["region"],
                slot_index=pim["slot_index"],
            )
            for pim in data.get("play_in_matchups", [])
        ]
        snapshot = cls(
            year=data["year"],
            teams=teams,
            seeds=data.get("seeds", {}),
            regions=data.get("regions", {}),
            first_round_matchups=data.get("first_round_matchups", []),
            play_in_matchups=play_ins,
        )
        return snapshot


# ---------------------------------------------------------------------------
# Play-in probability marginalisation
# ---------------------------------------------------------------------------

def marginalise_play_in(
    predict_fn: Callable[[str, str], float],
    play_in: PlayInMatchup,
    opponent_id: str,
) -> float:
    """Compute P(opponent beats play-in winner) by marginalising over play-in outcome.

    P(opponent beats winner) =
        P(team_a wins play-in) * P(opponent beats team_a)
      + P(team_b wins play-in) * P(opponent beats team_b)

    Args:
        predict_fn: Function (team1, team2) -> P(team1 wins).
        play_in: The play-in matchup to marginalise over.
        opponent_id: The team_id of the R64 opponent.

    Returns:
        Probability that opponent_id wins the R64 game.
    """
    p_a_wins_playin = predict_fn(play_in.team_a, play_in.team_b)
    p_b_wins_playin = 1.0 - p_a_wins_playin

    p_opp_beats_a = predict_fn(opponent_id, play_in.team_a)
    p_opp_beats_b = predict_fn(opponent_id, play_in.team_b)

    return p_a_wins_playin * p_opp_beats_a + p_b_wins_playin * p_opp_beats_b


# ---------------------------------------------------------------------------
# Bracket loading from historical data
# ---------------------------------------------------------------------------

def load_bracket_snapshot(
    year: int,
    data_dir: str = "data/raw/historical",
    bracket_dir: str = "data/raw",
) -> BracketSnapshot:
    """Load a bracket snapshot from historical data files.

    Attempts to load from:
    1. ``bracket_{year}.json`` in bracket_dir (preferred — Selection Sunday data)
    2. ``tournament_seeds_{year}.json`` in data_dir (fallback — seed/region data)

    Args:
        year: Tournament year.
        data_dir: Directory with historical data files.
        bracket_dir: Directory with bracket JSON files.

    Returns:
        BracketSnapshot for the given year.
    """
    # Try bracket file first (contains full bracket structure)
    bracket_path = Path(bracket_dir) / f"bracket_{year}.json"
    if bracket_path.exists():
        return _load_from_bracket_json(year, bracket_path)

    # Fallback to seeds file
    seeds_path = Path(data_dir) / f"tournament_seeds_{year}.json"
    if seeds_path.exists():
        return _load_from_seeds_json(year, seeds_path)

    raise FileNotFoundError(
        f"No bracket data found for {year}. "
        f"Checked: {bracket_path}, {seeds_path}"
    )


def _load_from_bracket_json(year: int, path: Path) -> BracketSnapshot:
    """Load bracket snapshot from a bracket_{year}.json file."""
    with open(path) as f:
        data = json.load(f)

    teams: List[TeamEntry] = []
    seeds: Dict[str, int] = {}
    regions: Dict[str, str] = {}
    first_round_matchups: List[str] = []
    play_in_matchups: List[PlayInMatchup] = []

    # Handle various bracket JSON formats
    bracket_teams = data.get("teams", data.get("bracket", {}).get("teams", []))
    if isinstance(bracket_teams, dict):
        # Format: {region: [{team_id, seed, ...}, ...]}
        for region_name, region_teams in bracket_teams.items():
            if not isinstance(region_teams, list):
                continue
            for t in region_teams:
                team_id = t.get("team_id", t.get("id", t.get("name", "")))
                seed = int(t.get("seed", 0))
                is_play_in = t.get("is_play_in", t.get("play_in", False))
                entry = TeamEntry(
                    team_id=team_id,
                    seed=seed,
                    region=region_name,
                    name=t.get("name", team_id),
                    is_play_in=bool(is_play_in),
                )
                teams.append(entry)
                seeds[team_id] = seed
                regions[team_id] = region_name
    elif isinstance(bracket_teams, list):
        # Format: [{team_id, seed, region, ...}, ...]
        for t in bracket_teams:
            team_id = t.get("team_id", t.get("id", t.get("name", "")))
            seed = int(t.get("seed", 0))
            region = t.get("region", "")
            is_play_in = t.get("is_play_in", t.get("play_in", False))
            entry = TeamEntry(
                team_id=team_id,
                seed=seed,
                region=region,
                name=t.get("name", team_id),
                is_play_in=bool(is_play_in),
            )
            teams.append(entry)
            seeds[team_id] = seed
            regions[team_id] = region

    # Load play-in matchups if present
    play_in_data = data.get("play_in_matchups", data.get("first_four", []))
    for pim in play_in_data:
        play_in_matchups.append(PlayInMatchup(
            team_a=pim.get("team_a", pim.get("team1", "")),
            team_b=pim.get("team_b", pim.get("team2", "")),
            seed=int(pim.get("seed", 0)),
            region=pim.get("region", ""),
            slot_index=int(pim.get("slot_index", pim.get("slot", 0))),
        ))

    # Load first-round matchups if present
    first_round_matchups = data.get(
        "first_round_matchups",
        data.get("bracket", {}).get("first_round_matchups", []),
    )

    return BracketSnapshot(
        year=year,
        teams=teams,
        seeds=seeds,
        regions=regions,
        first_round_matchups=first_round_matchups,
        play_in_matchups=play_in_matchups,
    )


def _load_from_seeds_json(year: int, path: Path) -> BracketSnapshot:
    """Load bracket snapshot from a tournament_seeds_{year}.json file.

    This is the fallback loader when no bracket JSON is available.
    Seeds files typically have format: {team_id: {seed, region}} or
    [{team_id, seed, region}].
    """
    with open(path) as f:
        data = json.load(f)

    teams: List[TeamEntry] = []
    seeds: Dict[str, int] = {}
    regions: Dict[str, str] = {}

    if isinstance(data, dict):
        # Format: {team_id: {seed: int, region: str}}
        for team_id, info in data.items():
            if isinstance(info, dict):
                seed = int(info.get("seed", 0))
                region = info.get("region", "")
            elif isinstance(info, (int, float)):
                seed = int(info)
                region = ""
            else:
                continue
            teams.append(TeamEntry(
                team_id=team_id, seed=seed, region=region, name=team_id,
            ))
            seeds[team_id] = seed
            regions[team_id] = region
    elif isinstance(data, list):
        for t in data:
            team_id = t.get("team_id", t.get("id", t.get("name", "")))
            seed = int(t.get("seed", 0))
            region = t.get("region", "")
            teams.append(TeamEntry(
                team_id=team_id, seed=seed, region=region, name=team_id,
            ))
            seeds[team_id] = seed
            regions[team_id] = region

    # Build standard first-round matchups from seeds
    first_round_matchups = _build_standard_matchups(teams)

    return BracketSnapshot(
        year=year,
        teams=teams,
        seeds=seeds,
        regions=regions,
        first_round_matchups=first_round_matchups,
        play_in_matchups=[],  # No play-in info in seeds-only file
    )


def _build_standard_matchups(teams: List[TeamEntry]) -> List[str]:
    """Build standard NCAA bracket first-round matchups from team list.

    Standard seed matchup order per region:
    (1,16), (8,9), (5,12), (4,13), (6,11), (3,14), (7,10), (2,15)
    """
    seed_order = [(1, 16), (8, 9), (5, 12), (4, 13), (6, 11), (3, 14), (7, 10), (2, 15)]
    region_order = ["East", "West", "South", "Midwest"]

    by_region: Dict[str, Dict[int, str]] = {}
    for t in teams:
        if t.region not in by_region:
            by_region[t.region] = {}
        by_region[t.region][t.seed] = t.team_id

    matchups: List[str] = []
    for region in region_order:
        seed_map = by_region.get(region, {})
        for high_seed, low_seed in seed_order:
            if high_seed in seed_map:
                matchups.append(seed_map[high_seed])
            if low_seed in seed_map:
                matchups.append(seed_map[low_seed])

    return matchups
