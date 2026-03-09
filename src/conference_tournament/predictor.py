"""Conference tournament prediction engine.

Reuses the SOTA pipeline's trained models and feature engineering to
predict conference tournament matchups.  This module does NOT train
new models — it leverages the existing pairwise prediction capability
(SOTAPipeline.predict_probability) to forecast conference tournament
brackets.

Usage:
    from src.conference_tournament.predictor import ConferenceTournamentPredictor

    predictor = ConferenceTournamentPredictor.from_torvik_json(
        "data/raw/torvik_2026.json"
    )

    # Predict a single conference tournament
    bracket = predictor.predict_conference("ACC")
    print(bracket.summary())

    # Predict all 31 conference tournaments
    results = predictor.predict_all()
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..models.conference_tournament import (
    ConferenceTournamentBracket,
    ConferenceTournamentGame,
    ConferenceTeam,
    _CONFERENCE_FULL_NAMES,
)

logger = logging.getLogger(__name__)


# Default conference tournament sizes (teams invited).
# Most conferences invite all members; some use a subset.
# This can be overridden via the conf_sizes parameter.
_DEFAULT_CONF_TOURNAMENT_SIZES: Dict[str, int] = {
    # Power conferences typically invite all members
    "ACC": 18,
    "B10": 18,
    "B12": 16,
    "SEC": 16,
    "BE": 11,
    # Mid-majors — most invite all
    "Amer": 13,
    "MWC": 12,
    "WCC": 12,
    "A10": 14,
    # Smaller conferences — all teams
    "Ivy": 8,
    "MEAC": 8,
    "WAC": 7,
}


class ConferenceTournamentPredictor:
    """Predicts conference tournament outcomes using existing team data.

    Operates in two modes:
    1. **Standalone mode** (no SOTA pipeline): Uses Torvik adjusted efficiency
       margins with a logistic model for pairwise predictions.  Fast, no
       ML dependencies, good for quick estimates.

    2. **Pipeline mode** (with trained SOTAPipeline): Uses the full ensemble
       (LightGBM + XGBoost + Logistic + calibration) for predictions.
       More accurate, requires a trained pipeline instance.
    """

    def __init__(
        self,
        teams_by_conference: Dict[str, List[ConferenceTeam]],
        pipeline=None,
        conf_tournament_sizes: Optional[Dict[str, int]] = None,
    ):
        """Initialize predictor.

        Args:
            teams_by_conference: Conference abbrev -> list of teams, sorted
                by conference standing (conf_seed).
            pipeline: Optional trained SOTAPipeline instance.  If None, uses
                standalone Torvik-based predictions.
            conf_tournament_sizes: Optional override for how many teams each
                conference invites to its tournament.
        """
        self.teams_by_conference = teams_by_conference
        self.pipeline = pipeline
        self.conf_sizes = conf_tournament_sizes or _DEFAULT_CONF_TOURNAMENT_SIZES

    @classmethod
    def from_torvik_json(
        cls,
        torvik_path: str,
        pipeline=None,
        conf_tournament_sizes: Optional[Dict[str, int]] = None,
    ) -> "ConferenceTournamentPredictor":
        """Build predictor from a Torvik JSON data file.

        Args:
            torvik_path: Path to torvik_YYYY.json file.
            pipeline: Optional trained SOTAPipeline instance.
            conf_tournament_sizes: Optional override for tournament sizes.

        Returns:
            Configured ConferenceTournamentPredictor.
        """
        with open(torvik_path) as f:
            data = json.load(f)

        teams_by_conf: Dict[str, List[ConferenceTeam]] = {}

        for team_data in data.get("teams", []):
            conf = team_data.get("conference", "")
            if not conf:
                continue

            team = ConferenceTeam(
                team_id=team_data.get("team_id", ""),
                name=team_data.get("team_name", "") or team_data.get("name", ""),
                conf_seed=0,  # Will be assigned by conference standing
                conference=conf,
                t_rank=team_data.get("t_rank", 999),
                adj_em=(
                    team_data.get("adj_offensive_efficiency", 100.0)
                    - team_data.get("adj_defensive_efficiency", 100.0)
                ),
            )
            teams_by_conf.setdefault(conf, []).append(team)

        # Sort each conference by adjusted efficiency margin (best first)
        # and assign conference seeds
        for conf, teams in teams_by_conf.items():
            teams.sort(key=lambda t: -t.adj_em)
            for i, team in enumerate(teams):
                team.conf_seed = i + 1

        return cls(
            teams_by_conference=teams_by_conf,
            pipeline=pipeline,
            conf_tournament_sizes=conf_tournament_sizes,
        )

    def list_conferences(self) -> List[str]:
        """Return sorted list of available conference abbreviations."""
        return sorted(self.teams_by_conference.keys())

    def get_conference_teams(self, conference: str) -> List[ConferenceTeam]:
        """Get teams for a conference, sorted by seed."""
        teams = self.teams_by_conference.get(conference, [])
        return sorted(teams, key=lambda t: t.conf_seed)

    def predict_matchup(self, team1: ConferenceTeam, team2: ConferenceTeam) -> float:
        """Predict P(team1 wins) for a single matchup.

        Uses the SOTA pipeline if available, otherwise falls back to
        a logistic model based on Torvik AdjEM difference.

        Args:
            team1: First team.
            team2: Second team.

        Returns:
            Win probability for team1 (0.0 to 1.0).
        """
        if self.pipeline is not None:
            try:
                return self.pipeline.predict_probability(team1.team_id, team2.team_id)
            except Exception as e:
                logger.warning(
                    "Pipeline prediction failed for %s vs %s: %s. "
                    "Falling back to standalone model.",
                    team1.team_id, team2.team_id, e,
                )

        # Standalone: logistic model on AdjEM difference.
        # Calibrated from historical college basketball data:
        # ~75% of games are won by the team with higher AdjEM,
        # and each point of AdjEM difference ≈ 3% win probability shift.
        import math
        em_diff = team1.adj_em - team2.adj_em
        # Slope of 0.15 in logit space: 10-point AdjEM gap ≈ 82% win prob
        prob = 1.0 / (1.0 + math.exp(-0.15 * em_diff))
        return float(max(0.02, min(0.98, prob)))

    def predict_conference(
        self,
        conference: str,
        num_teams: Optional[int] = None,
    ) -> ConferenceTournamentBracket:
        """Predict an entire conference tournament bracket.

        Args:
            conference: Conference abbreviation (e.g. "ACC", "B12").
            num_teams: Number of teams in the tournament.  If None, uses
                the default for that conference (or all teams).

        Returns:
            ConferenceTournamentBracket with predictions filled in.

        Raises:
            ValueError: If conference not found or invalid team count.
        """
        all_teams = self.get_conference_teams(conference)
        if not all_teams:
            raise ValueError(
                f"Conference '{conference}' not found. "
                f"Available: {', '.join(self.list_conferences())}"
            )

        # Determine tournament size
        if num_teams is None:
            num_teams = self.conf_sizes.get(conference, len(all_teams))

        num_teams = min(num_teams, len(all_teams))
        tournament_teams = all_teams[:num_teams]

        bracket = ConferenceTournamentBracket(
            conference=conference,
            teams=tournament_teams,
        )

        # Simulate the tournament round by round
        self._simulate_bracket(bracket)

        return bracket

    def predict_all(
        self,
        conferences: Optional[List[str]] = None,
    ) -> Dict[str, ConferenceTournamentBracket]:
        """Predict all (or selected) conference tournaments.

        Args:
            conferences: List of conference abbreviations to predict.
                If None, predicts all available conferences.

        Returns:
            Dict mapping conference abbreviation to predicted bracket.
        """
        if conferences is None:
            conferences = self.list_conferences()

        results = {}
        for conf in conferences:
            try:
                results[conf] = self.predict_conference(conf)
                logger.info(
                    "Predicted %s tournament: champion = %s",
                    conf,
                    results[conf].champion,
                )
            except Exception as e:
                logger.warning("Failed to predict %s tournament: %s", conf, e)

        return results

    def _simulate_bracket(self, bracket: ConferenceTournamentBracket):
        """Simulate all rounds of a bracket, advancing winners."""
        bye_teams = bracket.teams[:bracket.num_byes]

        # Round 1: predict first-round games
        for game in bracket.games[0]:
            if game.team1 and game.team2:
                prob = self.predict_matchup(game.team1, game.team2)
                winner = game.team1 if prob >= 0.5 else game.team2
                game.set_prediction(winner, prob if winner == game.team1 else 1 - prob)

        # Round 2+: fill in teams from previous round winners + byes
        for round_idx in range(1, len(bracket.games)):
            round_games = bracket.games[round_idx]
            prev_round_games = bracket.games[round_idx - 1]
            prev_winners = [g.winner for g in prev_round_games]

            if round_idx == 1 and bye_teams:
                # Merge bye teams with first-round winners
                # Convention: 1-seed plays winner of lowest-seed matchup, etc.
                entrants = list(bye_teams) + list(prev_winners)
                # Re-sort by seed to maintain bracket integrity
                entrants.sort(key=lambda t: t.conf_seed if t else 999)
            else:
                entrants = list(prev_winners)

            # Pair up: first vs last, second vs second-to-last, etc.
            for i, game in enumerate(round_games):
                if 2 * i < len(entrants) and 2 * i + 1 < len(entrants):
                    game.team1 = entrants[2 * i]
                    game.team2 = entrants[2 * i + 1]
                elif 2 * i < len(entrants):
                    # Odd number of teams — auto-advance
                    game.team1 = entrants[2 * i]
                    game.winner = entrants[2 * i]
                    game.win_probability = 1.0
                    game.is_bye = True
                    continue

                if game.team1 and game.team2:
                    prob = self.predict_matchup(game.team1, game.team2)
                    winner = game.team1 if prob >= 0.5 else game.team2
                    game.set_prediction(
                        winner, prob if winner == game.team1 else 1 - prob,
                    )

        # Set champion from final game
        final_round = bracket.games[-1]
        if final_round and final_round[0].winner:
            bracket.champion = final_round[0].winner

    def generate_report(
        self,
        conferences: Optional[List[str]] = None,
    ) -> str:
        """Generate a human-readable prediction report.

        Args:
            conferences: Conferences to include.  None = all.

        Returns:
            Formatted string report.
        """
        results = self.predict_all(conferences)

        lines = [
            "=" * 70,
            "  CONFERENCE TOURNAMENT PREDICTIONS",
            "  Pre-NCAA Tournament Validation",
            "=" * 70,
        ]

        # Summary table
        lines.append(f"\n{'Conference':<25} {'Champion':<25} {'Seed':>4}  {'T-Rank':>6}")
        lines.append("-" * 70)

        for conf in sorted(results.keys()):
            bracket = results[conf]
            if bracket.champion:
                conf_name = _CONFERENCE_FULL_NAMES.get(conf, conf)
                lines.append(
                    f"{conf_name:<25} {bracket.champion.name:<25} "
                    f"{bracket.champion.conf_seed:>4}  {bracket.champion.t_rank:>6}"
                )

        # Upset alerts
        lines.append(f"\n{'='*70}")
        lines.append("  UPSET ALERTS (lower seed predicted to win)")
        lines.append(f"{'='*70}")

        upset_count = 0
        for conf in sorted(results.keys()):
            bracket = results[conf]
            for game in bracket.get_all_games():
                if game.is_upset and not game.is_bye:
                    conf_name = _CONFERENCE_FULL_NAMES.get(conf, conf)
                    lines.append(
                        f"  {conf_name}: {game.winner} over {game.team1 if game.winner != game.team1 else game.team2} "
                        f"({game.win_probability:.1%})"
                    )
                    upset_count += 1

        if upset_count == 0:
            lines.append("  No upsets predicted.")
        else:
            lines.append(f"\n  Total upsets predicted: {upset_count}")

        # Detailed brackets for power conferences
        power_confs = ["ACC", "B10", "B12", "SEC", "BE"]
        lines.append(f"\n{'='*70}")
        lines.append("  DETAILED BRACKETS (Power Conferences)")
        lines.append(f"{'='*70}")

        for conf in power_confs:
            if conf in results:
                lines.append(results[conf].summary())

        return "\n".join(lines)

    def to_json(
        self,
        conferences: Optional[List[str]] = None,
    ) -> str:
        """Export predictions as JSON.

        Args:
            conferences: Conferences to include.  None = all.

        Returns:
            JSON string of all predictions.
        """
        results = self.predict_all(conferences)
        return json.dumps(
            {conf: bracket.to_dict() for conf, bracket in results.items()},
            indent=2,
        )
