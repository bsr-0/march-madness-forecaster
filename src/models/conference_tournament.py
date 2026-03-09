"""Conference tournament bracket model.

Represents single-elimination conference tournament brackets with
variable sizes (8-16 teams), byes for top seeds, and conference-specific
seeding based on regular-season conference standings.

Unlike the NCAA tournament Bracket model (which enforces 64/68 teams
across 4 regions), this supports the diverse formats used by the 32
D1 conferences.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math


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
}


class ConferenceTournamentBracket:
    """Single-elimination conference tournament bracket.

    Supports 4-16 team brackets with optional first-round byes for
    top seeds (standard in most conference tournaments).

    The bracket is built as a list of rounds, each containing games.
    Winners advance to fill subsequent rounds.
    """

    def __init__(
        self,
        conference: str,
        teams: List[ConferenceTeam],
        num_byes: int = 0,
    ):
        """Build a conference tournament bracket.

        Args:
            conference: Conference abbreviation (e.g. "ACC").
            teams: Teams sorted by conference seed (ascending).
            num_byes: Number of top seeds that receive first-round byes.
                If 0, auto-calculated to fill bracket to a power of 2.
        """
        if len(teams) < 2:
            raise ValueError(f"Need at least 2 teams, got {len(teams)}")

        self.conference = conference
        self.teams = sorted(teams, key=lambda t: t.conf_seed)
        self.num_teams = len(teams)

        # Calculate bracket structure
        self.total_rounds = math.ceil(math.log2(self.num_teams))
        full_bracket_size = 2 ** self.total_rounds

        if num_byes == 0:
            # Auto-calculate: top seeds get byes to fill the bracket
            self.num_byes = full_bracket_size - self.num_teams
        else:
            self.num_byes = num_byes

        self.games: List[List[ConferenceTournamentGame]] = []
        self.champion: Optional[ConferenceTeam] = None
        self._build_bracket()

    def _round_name(self, round_num: int) -> str:
        names = _ROUND_NAMES.get(self.total_rounds, {})
        return names.get(round_num, f"Round {round_num}")

    def _build_bracket(self):
        """Build the bracket structure with proper seeding and byes."""
        game_id = 0

        # First round: pair bottom seeds (those without byes)
        first_round_teams = self.teams[self.num_byes:]
        bye_teams = self.teams[:self.num_byes]

        first_round_games = []
        n_first_round = len(first_round_teams) // 2

        for i in range(n_first_round):
            # Standard 1-vs-N seeding: top remaining vs bottom remaining
            top_idx = i
            bottom_idx = len(first_round_teams) - 1 - i
            game_id += 1
            game = ConferenceTournamentGame(
                game_id=game_id,
                round=1,
                round_name=self._round_name(1),
                team1=first_round_teams[top_idx],
                team2=first_round_teams[bottom_idx],
            )
            first_round_games.append(game)

        self.games.append(first_round_games)

        # Subsequent rounds: winners from previous round + bye teams
        for round_num in range(2, self.total_rounds + 1):
            round_games = []
            if round_num == 2 and bye_teams:
                # Second round pairs bye teams with first-round winners
                # Bye teams are matched against the winner whose opponent
                # was the lowest seed (standard bracket convention)
                prev_winners_slots = len(first_round_games)
                total_slots = prev_winners_slots + len(bye_teams)
                n_games = total_slots // 2

                for i in range(n_games):
                    game_id += 1
                    game = ConferenceTournamentGame(
                        game_id=game_id,
                        round=round_num,
                        round_name=self._round_name(round_num),
                    )
                    round_games.append(game)
            else:
                prev_games = len(self.games[-1])
                n_games = prev_games // 2
                for i in range(n_games):
                    game_id += 1
                    game = ConferenceTournamentGame(
                        game_id=game_id,
                        round=round_num,
                        round_name=self._round_name(round_num),
                    )
                    round_games.append(game)

            self.games.append(round_games)

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
