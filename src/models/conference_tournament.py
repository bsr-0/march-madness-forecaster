"""Conference tournament bracket model.

Represents single-elimination conference tournament brackets with
variable sizes (4-18 teams), multi-level byes for top seeds, and
conference-specific seeding based on regular-season conference standings.

Unlike the NCAA tournament Bracket model (which enforces 64/68 teams
across 4 regions), this supports the diverse formats used by the 32
D1 conferences.  Bracket formats are defined in
``src.conference_tournament.bracket_formats`` and handle conferences
that use more rounds than the mathematical minimum (e.g. the ACC with
15 teams uses 5 rounds, not 4).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
import math

if TYPE_CHECKING:
    from ..conference_tournament.bracket_formats import BracketFormat


@dataclass
class ConferenceTeam:
    """A team in a conference tournament.

    Attributes:
        team_id: Normalized team identifier (e.g. "duke").
        name: Display name (e.g. "Duke").
        conf_seed: Conference tournament seed (1 = regular-season champ).
        conference: Conference abbreviation (e.g. "ACC").
        t_rank: Overall Torvik T-Rank (national ranking).
        adj_em: Adjusted efficiency margin.
        adj_o: Adjusted offensive efficiency (per 100 possessions).
        adj_d: Adjusted defensive efficiency (per 100 possessions).
        tempo: Adjusted tempo (possessions per 40 minutes).
        stats: Additional statistics (Four Factors, shooting, etc.).
    """

    team_id: str
    name: str
    conf_seed: int
    conference: str
    t_rank: int = 999
    adj_em: float = 0.0
    adj_o: float = 0.0
    adj_d: float = 0.0
    tempo: float = 0.0
    stats: Dict[str, float] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"[{self.conf_seed}] {self.name}"


@dataclass
class ConferenceTournamentGame:
    """A single game in a conference tournament bracket."""

    game_id: int
    round: int  # 1 = first round, 2 = quarterfinals, etc.
    round_name: str  # Human-readable round label
    team1: Optional[ConferenceTeam] = None
    team2: Optional[ConferenceTeam] = None
    winner: Optional[ConferenceTeam] = None
    win_probability: float = 0.5
    is_bye: bool = False  # True if one team has a bye (auto-advances)

    def set_prediction(self, winner: ConferenceTeam, probability: float):
        if not 0 <= probability <= 1:
            raise ValueError(f"Probability must be between 0 and 1, got {probability}")
        self.winner = winner
        self.win_probability = probability

    @property
    def is_upset(self) -> bool:
        if not self.winner or not self.team1 or not self.team2:
            return False
        favored = self.team1 if self.team1.conf_seed < self.team2.conf_seed else self.team2
        return self.winner.team_id != favored.team_id

    def to_dict(self) -> dict:
        return {
            "game_id": self.game_id,
            "round": self.round,
            "round_name": self.round_name,
            "team1": str(self.team1) if self.team1 else "TBD",
            "team2": str(self.team2) if self.team2 else "TBD",
            "winner": str(self.winner) if self.winner else None,
            "win_probability": self.win_probability,
            "is_upset": self.is_upset,
            "is_bye": self.is_bye,
        }


# Standard round names by bracket depth
_ROUND_NAMES = {
    1: {1: "Championship"},
    2: {1: "Semifinals", 2: "Championship"},
    3: {1: "Quarterfinals", 2: "Semifinals", 3: "Championship"},
    4: {1: "First Round", 2: "Quarterfinals", 3: "Semifinals", 4: "Championship"},
    5: {1: "Play-in", 2: "First Round", 3: "Quarterfinals", 4: "Semifinals", 5: "Championship"},
    6: {1: "Play-in", 2: "First Round", 3: "Second Round", 4: "Quarterfinals", 5: "Semifinals", 6: "Championship"},
}


def bracket_seed_order(n: int) -> List[int]:
    """Return indices that reorder seed-sorted teams into proper bracket positions.

    Given *n* teams sorted by seed (best first), returns a list of indices
    such that ``[teams[i] for i in bracket_seed_order(n)]`` places them in
    standard tournament bracket order.  When these reordered teams are
    paired adjacently ``(pos 0 vs 1, 2 vs 3, …)`` the resulting bracket
    guarantees that the #1 and #2 seeds are on opposite halves and can
    only meet in the final.

    *n* must be a power of 2.

    Examples (index lists, 0-based into a seed-sorted array):
        n=2  → [0, 1]              → matchups: 1v2
        n=4  → [0, 3, 1, 2]        → matchups: 1v4, 2v3
        n=8  → [0, 7, 3, 4, 1, 6, 2, 5]
               → matchups: 1v8, 4v5, 2v7, 3v6
    """
    if n <= 0:
        return []
    if n == 1:
        return [0]
    if n == 2:
        return [0, 1]
    half = bracket_seed_order(n // 2)
    result = []
    for p in half:
        result.append(p)
        result.append(n - 1 - p)
    return result


class ConferenceTournamentBracket:
    """Single-elimination conference tournament bracket.

    Supports 4-18 team brackets with multi-level byes for top seeds.
    Uses ``BracketFormat`` definitions to determine the correct number
    of rounds and bye structure for each conference.

    The bracket is built as a list of rounds, each containing games.
    Winners advance to fill subsequent rounds.
    """

    def __init__(
        self,
        conference: str,
        teams: List[ConferenceTeam],
        num_byes: int = 0,
        bracket_format: Optional["BracketFormat"] = None,
    ):
        """Build a conference tournament bracket.

        Args:
            conference: Conference abbreviation (e.g. "ACC").
            teams: Teams sorted by conference seed (ascending).
            num_byes: Ignored when bracket_format is provided.  Kept for
                backward compatibility.
            bracket_format: Explicit bracket format.  If None, looked up
                from the bracket_formats registry.
        """
        if len(teams) < 2:
            raise ValueError(f"Need at least 2 teams, got {len(teams)}")

        self.conference = conference
        self.teams = sorted(teams, key=lambda t: t.conf_seed)
        self.num_teams = len(teams)

        # Look up bracket format
        from ..conference_tournament.bracket_formats import get_bracket_format
        self.bracket_format = bracket_format or get_bracket_format(
            conference, self.num_teams
        )

        self.total_rounds = self.bracket_format.total_rounds
        # num_byes = total teams NOT playing in round 1
        r1_seeds = self.bracket_format.round_entry.get(1, ())
        self.num_byes = self.num_teams - len(r1_seeds)

        # Map seed -> team for quick lookup
        self._seed_to_team: Dict[int, ConferenceTeam] = {
            t.conf_seed: t for t in self.teams
        }

        self.games: List[List[ConferenceTournamentGame]] = []
        self.champion: Optional[ConferenceTeam] = None
        self._build_bracket()

    def _round_name(self, round_num: int) -> str:
        names = _ROUND_NAMES.get(self.total_rounds, {})
        return names.get(round_num, f"Round {round_num}")

    def _build_bracket(self):
        """Build the bracket structure using the BracketFormat definition.

        Round 1 teams are paired using first_round_matchups (if defined)
        or top-vs-bottom within the R1 seed group.

        Subsequent rounds merge new entrants (from round_entry) with
        previous-round winners.  Games are created as empty slots to be
        filled during simulation.
        """
        fmt = self.bracket_format
        game_id = 0

        # ── Round 1 ──────────────────────────────────────────────────
        r1_seeds = list(fmt.round_entry.get(1, ()))
        r1_teams = [self._seed_to_team[s] for s in r1_seeds if s in self._seed_to_team]

        first_round_games = []
        if fmt.first_round_matchups:
            # Explicit matchups defined
            for seed_a, seed_b in fmt.first_round_matchups:
                team_a = self._seed_to_team.get(seed_a)
                team_b = self._seed_to_team.get(seed_b)
                if team_a and team_b:
                    game_id += 1
                    first_round_games.append(ConferenceTournamentGame(
                        game_id=game_id,
                        round=1,
                        round_name=self._round_name(1),
                        team1=team_a,
                        team2=team_b,
                    ))
        else:
            # No explicit matchups: pair top-vs-bottom within R1 seeds
            # or use full bracket ordering when there are no byes
            r1_teams_sorted = sorted(r1_teams, key=lambda t: t.conf_seed)
            n_games = len(r1_teams_sorted) // 2

            has_byes = any(
                rnd > 1 and seeds
                for rnd, seeds in fmt.round_entry.items()
            )

            if not has_byes and len(r1_teams_sorted) >= 4:
                # No byes: apply bracket ordering so #1 and #2 are on
                # opposite halves
                positions = bracket_seed_order(len(r1_teams_sorted))
                ordered = [r1_teams_sorted[p] for p in positions]
            else:
                # With byes: top-vs-bottom within first-round group
                ordered = []
                for i in range(n_games):
                    ordered.append(r1_teams_sorted[i])
                    ordered.append(r1_teams_sorted[len(r1_teams_sorted) - 1 - i])

            for i in range(n_games):
                game_id += 1
                first_round_games.append(ConferenceTournamentGame(
                    game_id=game_id,
                    round=1,
                    round_name=self._round_name(1),
                    team1=ordered[2 * i],
                    team2=ordered[2 * i + 1],
                ))

        self.games.append(first_round_games)

        # ── Rounds 2+ ────────────────────────────────────────────────
        # Track how many winners come from the previous round
        prev_winner_count = len(first_round_games)

        for round_num in range(2, self.total_rounds + 1):
            new_seeds = fmt.round_entry.get(round_num, ())
            total_entrants = prev_winner_count + len(new_seeds)
            n_games = total_entrants // 2

            round_games = []
            for _ in range(n_games):
                game_id += 1
                round_games.append(ConferenceTournamentGame(
                    game_id=game_id,
                    round=round_num,
                    round_name=self._round_name(round_num),
                ))
            self.games.append(round_games)
            prev_winner_count = n_games

    def get_all_games(self) -> List[ConferenceTournamentGame]:
        """Return all games flattened across rounds."""
        return [g for rnd in self.games for g in rnd]

    def summary(self) -> str:
        """Human-readable bracket summary."""
        lines = [
            f"{'='*60}",
            f"  {_CONFERENCE_FULL_NAMES.get(self.conference, self.conference)} Conference Tournament",
            f"  {self.num_teams} teams, {self.num_byes} first-round byes",
            f"{'='*60}",
        ]

        for round_idx, round_games in enumerate(self.games):
            round_num = round_idx + 1
            lines.append(f"\n  {self._round_name(round_num)}")
            lines.append(f"  {'-'*40}")
            for game in round_games:
                if game.team1 and game.team2:
                    prob_str = f"{game.win_probability:.1%}" if game.winner else ""
                    winner_str = f" -> {game.winner}" if game.winner else ""
                    lines.append(
                        f"    {game.team1}  vs  {game.team2}  {prob_str}{winner_str}"
                    )
                else:
                    lines.append(f"    TBD  vs  TBD")

        if self.champion:
            lines.append(f"\n  CHAMPION: {self.champion}")

        lines.append(f"{'='*60}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "conference": self.conference,
            "conference_name": _CONFERENCE_FULL_NAMES.get(self.conference, self.conference),
            "num_teams": self.num_teams,
            "num_byes": self.num_byes,
            "total_rounds": self.total_rounds,
            "rounds": [
                [g.to_dict() for g in rnd]
                for rnd in self.games
            ],
            "champion": str(self.champion) if self.champion else None,
        }


# Conference abbreviation -> full name mapping (Torvik abbreviations)
_CONFERENCE_FULL_NAMES: Dict[str, str] = {
    "A10": "Atlantic 10",
    "ACC": "ACC",
    "AE": "America East",
    "ASun": "ASUN",
    "Amer": "American Athletic",
    "B10": "Big Ten",
    "B12": "Big 12",
    "BE": "Big East",
    "BSky": "Big Sky",
    "BSth": "Big South",
    "BW": "Big West",
    "CAA": "CAA",
    "CUSA": "Conference USA",
    "Horz": "Horizon League",
    "Ivy": "Ivy League",
    "MAAC": "MAAC",
    "MAC": "MAC",
    "MEAC": "MEAC",
    "MVC": "Missouri Valley",
    "MWC": "Mountain West",
    "NEC": "NEC",
    "OVC": "Ohio Valley",
    "Pat": "Patriot League",
    "SB": "Sun Belt",
    "SC": "Southern",
    "SEC": "SEC",
    "SWAC": "SWAC",
    "Slnd": "Southland",
    "Sum": "Summit League",
    "WAC": "WAC",
    "WCC": "WCC",
}
