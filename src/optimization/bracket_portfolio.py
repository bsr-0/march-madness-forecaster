"""
Bracket portfolio optimization for Kaggle March Mania (2024+ format).

The competition allows submitting 1-100,000 brackets. This is fundamentally
different from pairwise probability prediction — the goal is to maximize the
probability of having at least one bracket in the top positions.

Key strategies:
1. Diversity: Generate brackets that collectively cover many outcomes
2. Anti-correlation: Over-represent scenarios the public under-weights
3. Champion stratification: Ensure portfolio represents all viable champions
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BracketPick:
    """A single game pick within a bracket."""
    round_num: int          # 0=R64, 1=R32, 2=S16, 3=E8, 4=F4, 5=CHAMP
    game_idx: int           # Game index within the round
    winner_id: str          # Team predicted to win
    loser_id: str           # Team predicted to lose
    win_probability: float  # Model's probability for this pick


@dataclass
class GeneratedBracket:
    """A complete tournament bracket (all 63 game picks)."""
    bracket_id: int
    picks: List[BracketPick] = field(default_factory=list)
    champion: str = ""
    final_four: List[str] = field(default_factory=list)
    expected_score: float = 0.0       # Expected Brier/bracket score
    log_probability: float = 0.0      # Log probability of this exact outcome
    strategy: str = "balanced"        # "chalk", "balanced", "contrarian", "targeted"

    def to_submission_dict(self) -> Dict[str, str]:
        """Convert to Kaggle submission format."""
        result = {}
        for pick in self.picks:
            key = f"R{pick.round_num}G{pick.game_idx}"
            result[key] = pick.winner_id
        return result


class BracketPortfolioGenerator:
    """Generate a diverse portfolio of tournament brackets.

    Uses Monte Carlo simulation to sample tournament outcomes, then
    selects a diverse subset that maximizes coverage.

    The key insight: in a portfolio competition, you want brackets that
    are collectively unlikely to ALL be wrong. Diverse brackets achieve
    this better than many copies of the "best" bracket.
    """

    def __init__(
        self,
        predict_fn: Callable[[str, str], float],
        public_pick_pcts: Optional[Dict[str, float]] = None,
    ):
        """
        Args:
            predict_fn: (team1_id, team2_id) -> P(team1 wins)
            public_pick_pcts: team_id -> championship pick percentage
        """
        self.predict_fn = predict_fn
        self.public_picks = public_pick_pcts or {}

    def generate_portfolio(
        self,
        teams_by_region: Dict[str, List[Dict]],
        n_brackets: int = 1000,
        n_simulations: int = 50000,
        seed: int = 42,
        strategy_mix: Optional[Dict[str, float]] = None,
    ) -> List[GeneratedBracket]:
        """Generate a diverse bracket portfolio.

        Args:
            teams_by_region: {region: [{team_id, seed}, ...]}
            n_brackets: Number of brackets to generate
            n_simulations: Monte Carlo simulations for sampling
            seed: Random seed
            strategy_mix: Fraction of brackets per strategy type

        Returns:
            List of GeneratedBracket objects
        """
        if strategy_mix is None:
            strategy_mix = {
                "chalk": 0.10,       # Follow favorites
                "balanced": 0.40,    # Model-probability sampling
                "contrarian": 0.30,  # Anti-correlated with public
                "targeted": 0.20,    # Champion-targeted brackets
            }

        rng = np.random.default_rng(seed)

        # Build team lists and matchup order
        all_teams, first_round = self._build_bracket_structure(teams_by_region)

        # Pre-compute all pairwise probabilities
        matchup_cache = self._precompute_matchups(all_teams)

        # Generate simulation pool
        sim_results = self._run_simulations(
            first_round, all_teams, matchup_cache, n_simulations, rng
        )

        # Select diverse brackets from simulation pool
        brackets = []
        bracket_id = 0

        for strategy, fraction in strategy_mix.items():
            n_strategy = max(1, int(n_brackets * fraction))

            if strategy == "chalk":
                new_brackets = self._generate_chalk_brackets(
                    first_round, matchup_cache, n_strategy, rng
                )
            elif strategy == "contrarian":
                new_brackets = self._generate_contrarian_brackets(
                    sim_results, n_strategy, rng
                )
            elif strategy == "targeted":
                new_brackets = self._generate_targeted_brackets(
                    sim_results, all_teams, n_strategy, rng
                )
            else:  # balanced
                new_brackets = self._select_diverse_brackets(
                    sim_results, n_strategy, rng
                )

            for b in new_brackets:
                b.bracket_id = bracket_id
                b.strategy = strategy
                bracket_id += 1
                brackets.append(b)

        logger.info(
            "Generated portfolio of %d brackets (strategies: %s)",
            len(brackets),
            {s: int(n_brackets * f) for s, f in strategy_mix.items()},
        )
        return brackets

    def _build_bracket_structure(
        self, teams_by_region: Dict[str, List[Dict]]
    ) -> Tuple[Dict[str, Dict], List[str]]:
        """Build bracket structure from teams."""
        all_teams = {}
        first_round = []

        seed_order = [(1, 16), (8, 9), (5, 12), (4, 13),
                       (6, 11), (3, 14), (7, 10), (2, 15)]

        for region in ["East", "West", "South", "Midwest"]:
            region_teams = teams_by_region.get(region, [])
            seed_to_team = {}
            for t in region_teams:
                tid = t.get("team_id", t.get("name", ""))
                seed = t.get("seed", 8)
                all_teams[tid] = {"seed": seed, "region": region}
                seed_to_team[seed] = tid

            for hi_seed, lo_seed in seed_order:
                if hi_seed in seed_to_team:
                    first_round.append(seed_to_team[hi_seed])
                if lo_seed in seed_to_team:
                    first_round.append(seed_to_team[lo_seed])

        return all_teams, first_round

    def _precompute_matchups(
        self, all_teams: Dict[str, Dict]
    ) -> Dict[Tuple[str, str], float]:
        """Pre-compute all pairwise matchup probabilities."""
        cache = {}
        team_ids = list(all_teams.keys())
        for i, t1 in enumerate(team_ids):
            for j, t2 in enumerate(team_ids):
                if i < j:
                    p = self.predict_fn(t1, t2)
                    cache[(t1, t2)] = p
                    cache[(t2, t1)] = 1.0 - p
        return cache

    def _run_simulations(
        self,
        first_round: List[str],
        all_teams: Dict[str, Dict],
        matchup_cache: Dict[Tuple[str, str], float],
        n_sims: int,
        rng: np.random.Generator,
    ) -> List[GeneratedBracket]:
        """Run MC simulations and return bracket results."""
        results = []
        noise_std = 0.12  # Match pipeline's MC noise

        for sim in range(n_sims):
            current = list(first_round)
            picks = []
            round_num = 0

            while len(current) > 1:
                winners = []
                for g_idx in range(0, len(current), 2):
                    if g_idx + 1 >= len(current):
                        winners.append(current[g_idx])
                        continue

                    t1, t2 = current[g_idx], current[g_idx + 1]
                    base_p = matchup_cache.get((t1, t2), 0.5)

                    # Add logit noise
                    safe_p = np.clip(base_p, 0.001, 0.999)
                    logit = np.log(safe_p / (1 - safe_p))
                    logit += rng.normal(0, noise_std)
                    p = 1.0 / (1.0 + np.exp(-logit))

                    if rng.random() < p:
                        winner, loser = t1, t2
                        win_p = base_p
                    else:
                        winner, loser = t2, t1
                        win_p = 1.0 - base_p

                    picks.append(BracketPick(
                        round_num=round_num,
                        game_idx=g_idx // 2,
                        winner_id=winner,
                        loser_id=loser,
                        win_probability=win_p,
                    ))
                    winners.append(winner)

                current = winners
                round_num += 1

            champion = current[0] if current else ""
            final_four = [p.winner_id for p in picks if p.round_num == 4]

            # Log probability of this exact bracket
            log_prob = sum(
                math.log(max(p.win_probability, 1e-10)) for p in picks
            )

            results.append(GeneratedBracket(
                bracket_id=sim,
                picks=picks,
                champion=champion,
                final_four=final_four,
                log_probability=log_prob,
            ))

        return results

    def _select_diverse_brackets(
        self,
        sim_results: List[GeneratedBracket],
        n_select: int,
        rng: np.random.Generator,
    ) -> List[GeneratedBracket]:
        """Select diverse brackets using champion-stratified sampling."""
        # Group by champion
        by_champion: Dict[str, List[GeneratedBracket]] = {}
        for b in sim_results:
            if b.champion not in by_champion:
                by_champion[b.champion] = []
            by_champion[b.champion].append(b)

        # Allocate brackets proportional to championship probability
        total_sims = len(sim_results)
        selected = []

        for champ, champ_brackets in by_champion.items():
            champ_prob = len(champ_brackets) / total_sims
            n_for_champ = max(1, int(n_select * champ_prob))

            # Sort by log probability (most likely brackets first)
            champ_brackets.sort(key=lambda b: b.log_probability, reverse=True)

            # Take top brackets for this champion
            selected.extend(champ_brackets[:n_for_champ])

        # If we have too many, trim; if too few, add more from best
        if len(selected) > n_select:
            selected = selected[:n_select]
        elif len(selected) < n_select:
            remaining = n_select - len(selected)
            all_sorted = sorted(sim_results, key=lambda b: b.log_probability, reverse=True)
            selected_ids = {id(b) for b in selected}
            for b in all_sorted:
                if id(b) not in selected_ids and remaining > 0:
                    selected.append(b)
                    remaining -= 1

        return selected

    def _generate_chalk_brackets(
        self,
        first_round: List[str],
        matchup_cache: Dict[Tuple[str, str], float],
        n_brackets: int,
        rng: np.random.Generator,
    ) -> List[GeneratedBracket]:
        """Generate chalk (favorites-always-win) brackets with small noise."""
        results = []
        for _ in range(n_brackets):
            current = list(first_round)
            picks = []
            round_num = 0

            while len(current) > 1:
                winners = []
                for g_idx in range(0, len(current), 2):
                    if g_idx + 1 >= len(current):
                        winners.append(current[g_idx])
                        continue

                    t1, t2 = current[g_idx], current[g_idx + 1]
                    p = matchup_cache.get((t1, t2), 0.5)

                    # Small noise for chalk brackets (more deterministic)
                    noise = rng.normal(0, 0.03)
                    safe_p = np.clip(p, 0.01, 0.99)
                    logit = np.log(safe_p / (1 - safe_p)) + noise
                    p_noisy = 1.0 / (1.0 + np.exp(-logit))

                    if p_noisy >= 0.5:
                        winner, loser = t1, t2
                        win_p = p
                    else:
                        winner, loser = t2, t1
                        win_p = 1.0 - p

                    picks.append(BracketPick(
                        round_num=round_num,
                        game_idx=g_idx // 2,
                        winner_id=winner,
                        loser_id=loser,
                        win_probability=win_p,
                    ))
                    winners.append(winner)

                current = winners
                round_num += 1

            champion = current[0] if current else ""
            log_prob = sum(math.log(max(p.win_probability, 1e-10)) for p in picks)
            results.append(GeneratedBracket(
                bracket_id=0, picks=picks, champion=champion,
                log_probability=log_prob, strategy="chalk",
            ))

        return results

    def _generate_contrarian_brackets(
        self,
        sim_results: List[GeneratedBracket],
        n_brackets: int,
        rng: np.random.Generator,
    ) -> List[GeneratedBracket]:
        """Generate brackets with contrarian champions (anti-correlated to public).

        Over-represent champions where model_prob >> public_prob.
        """
        if not self.public_picks:
            # Without public data, fall back to diverse selection
            return self._select_diverse_brackets(sim_results, n_brackets, rng)

        # Group by champion
        by_champion: Dict[str, List[GeneratedBracket]] = {}
        for b in sim_results:
            if b.champion not in by_champion:
                by_champion[b.champion] = []
            by_champion[b.champion].append(b)

        total_sims = len(sim_results)

        # Compute leverage ratio per champion
        champion_leverage = {}
        for champ, brackets in by_champion.items():
            model_p = len(brackets) / total_sims
            public_p = self.public_picks.get(champ, 0.01)
            leverage = model_p / max(public_p, 0.001)
            champion_leverage[champ] = leverage

        # Weight selection toward high-leverage champions
        total_leverage = sum(champion_leverage.values())
        selected = []

        for champ, leverage in champion_leverage.items():
            n_for_champ = max(1, int(n_brackets * leverage / total_leverage))
            brackets = by_champion.get(champ, [])
            brackets.sort(key=lambda b: b.log_probability, reverse=True)
            selected.extend(brackets[:n_for_champ])

        return selected[:n_brackets]

    def _generate_targeted_brackets(
        self,
        sim_results: List[GeneratedBracket],
        all_teams: Dict[str, Dict],
        n_brackets: int,
        rng: np.random.Generator,
    ) -> List[GeneratedBracket]:
        """Generate brackets each targeting a specific champion.

        Ensures the portfolio includes at least one bracket for each
        viable champion (>1% model probability).
        """
        by_champion: Dict[str, List[GeneratedBracket]] = {}
        for b in sim_results:
            if b.champion not in by_champion:
                by_champion[b.champion] = []
            by_champion[b.champion].append(b)

        total = len(sim_results)
        viable = {c: bs for c, bs in by_champion.items()
                   if len(bs) / total >= 0.01}

        selected = []
        per_champ = max(1, n_brackets // max(len(viable), 1))

        for champ, brackets in viable.items():
            brackets.sort(key=lambda b: b.log_probability, reverse=True)
            selected.extend(brackets[:per_champ])

        return selected[:n_brackets]
