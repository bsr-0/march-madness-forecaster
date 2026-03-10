"""Monte Carlo conference tournament simulator.

Adapts the logit-space noise approach from src/simulation/monte_carlo.py
for variable-size conference tournament brackets.  Produces championship
probability distributions instead of single-bracket predictions.
"""

from __future__ import annotations

import math
import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .predictor import ConferenceTournamentPredictor
    from ..models.conference_tournament import ConferenceTeam, ConferenceTournamentBracket

logger = logging.getLogger(__name__)


@dataclass
class ConferenceTournamentResult:
    """Results of Monte Carlo simulation for a conference tournament."""

    conference: str
    num_simulations: int
    championship_probs: Dict[str, float]  # team_id -> P(champion)
    semifinal_probs: Dict[str, float]     # team_id -> P(reach semis)
    most_likely_champion: str
    most_likely_champion_prob: float
    upset_alerts: List[Dict]  # Games with high upset probability

    def summary(self) -> str:
        """Human-readable summary of championship odds."""
        lines = [
            f"  Championship Probabilities ({self.num_simulations:,} simulations):",
        ]
        # Sort by probability descending
        sorted_probs = sorted(
            self.championship_probs.items(), key=lambda x: -x[1]
        )
        for team_id, prob in sorted_probs:
            if prob >= 0.005:  # Only show teams with >= 0.5%
                bar = "#" * int(prob * 40)
                lines.append(f"    {team_id:<25s} {prob:6.1%}  {bar}")

        if self.upset_alerts:
            lines.append(f"\n  Upset Alerts (underdog wins >{30}% of sims):")
            for alert in self.upset_alerts[:10]:
                lines.append(
                    f"    {alert['underdog']} over {alert['favorite']}: "
                    f"{alert['upset_pct']:.0%} of sims "
                    f"(base P={alert['base_prob']:.1%})"
                )

        return "\n".join(lines)


class ConferenceTournamentSimulator:
    """Monte Carlo simulator for conference tournament brackets.

    Uses the same logit-space noise approach as the main tournament
    simulator (noise_std=0.12 by default, calibrated from Lopez &
    Matthews, JQAS 2015).
    """

    def __init__(
        self,
        predictor: "ConferenceTournamentPredictor",
        num_simulations: int = 50000,
        noise_std: float = 0.12,
        random_seed: int = 2026,
    ):
        self.predictor = predictor
        self.num_simulations = num_simulations
        self.noise_std = noise_std
        self.rng = np.random.default_rng(random_seed)

    def simulate_conference(
        self,
        conference: str,
        num_teams: Optional[int] = None,
    ) -> ConferenceTournamentResult:
        """Run Monte Carlo simulation for a conference tournament.

        Args:
            conference: Conference abbreviation.
            num_teams: Tournament size override.

        Returns:
            ConferenceTournamentResult with championship probabilities.
        """
        # Build bracket structure (without predictions — we'll simulate)
        all_teams = self.predictor.get_conference_teams(conference)
        if not all_teams:
            raise ValueError(f"Conference '{conference}' not found")

        conf_sizes = self.predictor.conf_sizes
        if num_teams is None:
            num_teams = conf_sizes.get(conference, len(all_teams))
        num_teams = min(num_teams, len(all_teams))
        tournament_teams = all_teams[:num_teams]

        # Pre-compute base matchup probabilities for all possible pairs
        base_probs = self._precompute_matchup_probs(tournament_teams)

        # Run simulations
        champion_counts: Counter = Counter()
        semifinal_counts: Counter = Counter()
        # Track (favorite, underdog) -> {upset_count, total_count}
        matchup_stats: Dict[Tuple[str, str], List[int]] = {}  # [upsets, total]

        from ..models.conference_tournament import ConferenceTournamentBracket

        for sim_idx in range(self.num_simulations):
            bracket = ConferenceTournamentBracket(
                conference=conference, teams=tournament_teams
            )
            champion, semi_teams, game_results = self._simulate_single(
                bracket, base_probs
            )
            if champion:
                champion_counts[champion] += 1
            for t in semi_teams:
                semifinal_counts[t] += 1
            for fav, dog, was_upset in game_results:
                key = (fav, dog)
                if key not in matchup_stats:
                    matchup_stats[key] = [0, 0]
                matchup_stats[key][1] += 1  # total
                if was_upset:
                    matchup_stats[key][0] += 1  # upset count

        # Compute probabilities
        n = self.num_simulations
        championship_probs = {
            t.team_id: champion_counts.get(t.team_id, 0) / n
            for t in tournament_teams
        }
        semifinal_probs = {
            t.team_id: semifinal_counts.get(t.team_id, 0) / n
            for t in tournament_teams
        }

        # Find most likely champion
        most_likely = max(championship_probs, key=championship_probs.get)
        most_likely_prob = championship_probs[most_likely]

        # Identify upset alerts (underdog wins >30% of matchups)
        upset_alerts = []
        for (fav, dog), (upset_count, total) in sorted(
            matchup_stats.items(), key=lambda x: -x[1][0] / max(x[1][1], 1)
        ):
            upset_pct = upset_count / max(total, 1)
            if upset_pct > 0.30 and total >= n * 0.01:  # At least 1% of sims
                base_p = base_probs.get((fav, dog), 0.5)
                upset_alerts.append({
                    "favorite": fav,
                    "underdog": dog,
                    "upset_pct": upset_pct,
                    "base_prob": base_p,
                })

        return ConferenceTournamentResult(
            conference=conference,
            num_simulations=n,
            championship_probs=championship_probs,
            semifinal_probs=semifinal_probs,
            most_likely_champion=most_likely,
            most_likely_champion_prob=most_likely_prob,
            upset_alerts=upset_alerts[:10],
        )

    def simulate_all(
        self,
        conferences: Optional[List[str]] = None,
    ) -> Dict[str, ConferenceTournamentResult]:
        """Simulate all (or selected) conference tournaments.

        Args:
            conferences: List of conference abbreviations. None = all.

        Returns:
            Dict mapping conference to simulation results.
        """
        if conferences is None:
            conferences = self.predictor.list_conferences()

        results = {}
        for conf in conferences:
            try:
                results[conf] = self.simulate_conference(conf)
                logger.info(
                    "Simulated %s: champion=%s (%.1f%%)",
                    conf,
                    results[conf].most_likely_champion,
                    results[conf].most_likely_champion_prob * 100,
                )
            except Exception as e:
                logger.warning("Failed to simulate %s: %s", conf, e)

        return results

    def _precompute_matchup_probs(
        self, teams: List["ConferenceTeam"]
    ) -> Dict[Tuple[str, str], float]:
        """Pre-compute pairwise win probabilities for all team pairs."""
        probs = {}
        for i, t1 in enumerate(teams):
            for j, t2 in enumerate(teams):
                if i != j:
                    key = (t1.team_id, t2.team_id)
                    if key not in probs:
                        p = self.predictor.predict_matchup(t1, t2)
                        probs[key] = p
                        probs[(t2.team_id, t1.team_id)] = 1.0 - p
        return probs

    def _simulate_single(
        self,
        bracket: "ConferenceTournamentBracket",
        base_probs: Dict[Tuple[str, str], float],
    ) -> Tuple[Optional[str], List[str], List[Tuple[str, str, bool]]]:
        """Simulate a single tournament run with noise.

        Returns:
            (champion_id, semifinal_team_ids,
             [(favorite_id, underdog_id, was_upset) for each game])
        """
        bye_teams = bracket.teams[:bracket.num_byes]
        semi_teams: List[str] = []
        game_results: List[Tuple[str, str, bool]] = []

        # Round 1: simulate first-round games with noise
        for game in bracket.games[0]:
            if game.team1 and game.team2:
                winner = self._noisy_game(
                    game.team1, game.team2, base_probs, game_results
                )
                game.winner = winner

        # Subsequent rounds
        for round_idx in range(1, len(bracket.games)):
            round_games = bracket.games[round_idx]
            prev_round_games = bracket.games[round_idx - 1]
            prev_winners = [g.winner for g in prev_round_games]

            if round_idx == 1 and bye_teams:
                entrants = list(bye_teams) + list(prev_winners)
                # Sort by seed then reorder into proper bracket positions
                # so that #1 and #2 are on opposite sides of the bracket
                entrants.sort(key=lambda t: t.conf_seed if t else 999)
                from ..models.conference_tournament import bracket_seed_order
                positions = bracket_seed_order(len(entrants))
                entrants = [entrants[p] for p in positions]
            else:
                entrants = list(prev_winners)

            # Track semifinalists
            is_semifinal = (round_idx == len(bracket.games) - 2)
            if is_semifinal:
                semi_teams = [t.team_id for t in entrants if t]

            for i, game in enumerate(round_games):
                if 2 * i < len(entrants) and 2 * i + 1 < len(entrants):
                    t1 = entrants[2 * i]
                    t2 = entrants[2 * i + 1]
                    if t1 and t2:
                        game.team1 = t1
                        game.team2 = t2
                        winner = self._noisy_game(t1, t2, base_probs, game_results)
                        game.winner = winner
                elif 2 * i < len(entrants):
                    game.winner = entrants[2 * i]

        # Champion
        final = bracket.games[-1]
        champion = final[0].winner.team_id if final and final[0].winner else None
        return champion, semi_teams, game_results

    def _noisy_game(
        self,
        team1: "ConferenceTeam",
        team2: "ConferenceTeam",
        base_probs: Dict[Tuple[str, str], float],
        game_results: List[Tuple[str, str, bool]],
    ) -> "ConferenceTeam":
        """Simulate a single game with logit-space noise.

        Args:
            team1: First team.
            team2: Second team.
            base_probs: Pre-computed matchup probabilities.
            game_results: List to append (favorite, underdog, was_upset) to.

        Returns:
            Winning team.
        """
        key = (team1.team_id, team2.team_id)
        base_p = base_probs.get(key, 0.5)

        # Apply noise in logit space
        logit = math.log(max(base_p, 1e-6) / max(1.0 - base_p, 1e-6))
        noisy_logit = logit + self.rng.normal(0, self.noise_std)
        noisy_p = 1.0 / (1.0 + math.exp(-noisy_logit))
        noisy_p = max(0.01, min(0.99, noisy_p))

        # Simulate outcome
        if self.rng.random() < noisy_p:
            winner = team1
        else:
            winner = team2

        # Track game result (favorite = lower seed number)
        favorite = team1 if team1.conf_seed < team2.conf_seed else team2
        underdog = team2 if favorite is team1 else team1
        was_upset = (winner is underdog)
        game_results.append((favorite.team_id, underdog.team_id, was_upset))

        return winner
