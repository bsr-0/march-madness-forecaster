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
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from .leverage import PoolStrategyProfile

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Kaggle Effective Pool Size Estimation
# ---------------------------------------------------------------------------

def estimate_kaggle_effective_pool_size(
    total_entries: int = 3000,
    bracket_format: bool = True,
    year: int = 2025,
) -> int:
    """Estimate the effective competitive pool size for Kaggle allocation tuning.

    Kaggle March Mania has ~3000-6000 total entries, but many are baseline
    submissions (seed-only, all-0.5, etc.).  The *effective* pool is the
    number of submissions that are genuinely competitive — i.e., would
    beat a seed-baseline model.

    Empirical estimates from historical Kaggle leaderboards:
    - ~30-40% of entries are "competitive" (better than seed baseline)
    - ~10-15% are "strong" (using real models, not just seeds)
    - The bracket format (2024+) allows up to 100K entries per user,
      concentrating quality entries from fewer participants

    The effective pool size drives bracket portfolio allocation: a larger
    effective pool pushes the optimal strategy toward more contrarian
    and champion-targeted brackets.

    Args:
        total_entries: Total number of Kaggle submissions.
        bracket_format: Whether the competition uses the bracket portfolio
            format (2024+) rather than pairwise probabilities.
        year: Competition year (for trend adjustments).

    Returns:
        Estimated number of competitive entries to calibrate strategy against.
    """
    # Base competitive fraction from historical data
    if bracket_format:
        # Bracket format: fewer participants but each submits many brackets.
        # The effective pool is larger because each competitive participant
        # submits O(1000) diverse brackets.
        competitive_fraction = 0.50
        # Post-2024 trend: more participants discover optimal strategies
        year_adjustment = min(0.10, max(0, (year - 2024)) * 0.03)
    else:
        # Pairwise format: each person submits 1-2 entries
        competitive_fraction = 0.35
        year_adjustment = min(0.05, max(0, (year - 2020)) * 0.01)

    effective = int(total_entries * (competitive_fraction + year_adjustment))
    # Floor at 100 (always some competitive entries)
    return max(100, effective)


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

    Cross-strategy synergy: when a ``PoolStrategyProfile`` is provided,
    the generator uses its ``contrarian_strength`` to modulate how
    aggressively contrarian brackets over-weight leverage picks.  When
    ``leverage_picks`` are supplied (from the EV mode's pool analysis),
    contrarian and targeted strategies use per-round leverage data to
    make smarter pick decisions, not just champion-level selection.
    """

    def __init__(
        self,
        predict_fn: Callable[[str, str], float],
        public_pick_pcts: Optional[Dict[str, float]] = None,
        round_public_picks: Optional[Dict[str, Dict[str, float]]] = None,
        leverage_picks: Optional[List[Dict]] = None,
    ):
        """
        Args:
            predict_fn: (team1_id, team2_id) -> P(team1 wins)
            public_pick_pcts: team_id -> championship pick percentage
            round_public_picks: team_id -> {round: pick_pct} for per-round
                leverage.  When provided, contrarian brackets use per-round
                leverage instead of champion-only.
            leverage_picks: Serialized LeveragePick dicts from pool analysis.
                When provided, contrarian and targeted strategies use these
                pre-computed leverage opportunities.
        """
        self.predict_fn = predict_fn
        self.public_picks = public_pick_pcts or {}
        self.round_public_picks = round_public_picks or {}
        self.leverage_picks = leverage_picks or []
        self._contrarian_strength = 1.0  # Set from profile in generate_portfolio
        self._variance_target = 0.5  # Set from profile in generate_portfolio

    def generate_portfolio(
        self,
        teams_by_region: Dict[str, List[Dict]],
        n_brackets: int = 1000,
        n_simulations: int = 50000,
        seed: int = 42,
        strategy_mix: Optional[Dict[str, float]] = None,
        enable_search: bool = False,
        pool_strategy_profile: Optional["PoolStrategyProfile"] = None,
        kaggle_effective_pool_size: Optional[int] = None,
    ) -> List[GeneratedBracket]:
        """Generate a diverse bracket portfolio.

        Args:
            teams_by_region: {region: [{team_id, seed}, ...]}
            n_brackets: Number of brackets to generate
            n_simulations: Monte Carlo simulations for sampling
            seed: Random seed
            strategy_mix: Fraction of brackets per strategy type.  Takes
                precedence over ``pool_strategy_profile`` if both are provided.
            enable_search: If True, refine top brackets using Simulated
                Annealing search after initial MC sampling.
            pool_strategy_profile: Optional :class:`PoolStrategyProfile` from
                :mod:`~src.optimization.leverage`.  When provided and
                ``strategy_mix`` is not, uses the profile's pool-size-adaptive
                strategy allocation instead of hardcoded defaults.
            kaggle_effective_pool_size: Estimated number of competitive
                entries in the Kaggle field.  When provided and no
                ``pool_strategy_profile`` is given, automatically generates
                a profile using :func:`~src.optimization.leverage.get_strategy_profile`
                calibrated to this pool size.  This produces smarter
                allocations than the hardcoded defaults, especially for
                the bracket portfolio format where the field is 3000+
                expert entries.

        Returns:
            List of GeneratedBracket objects
        """
        # Auto-generate a pool strategy profile from Kaggle effective pool size
        if (
            strategy_mix is None
            and pool_strategy_profile is None
            and kaggle_effective_pool_size is not None
        ):
            from .leverage import get_strategy_profile
            pool_strategy_profile = get_strategy_profile(
                pool_size=kaggle_effective_pool_size,
                scoring_system="standard",
                payout_structure="top_3",  # Kaggle pays top ~3 spots
            )
            logger.info(
                "Auto-generated pool strategy profile from Kaggle effective "
                "pool size=%d: %s (contrarian_strength=%.1f)",
                kaggle_effective_pool_size,
                pool_strategy_profile.strategy_mix,
                pool_strategy_profile.contrarian_strength,
            )

        if strategy_mix is None and pool_strategy_profile is not None:
            # Cross-strategy synergy: reuse pool-size-adaptive allocations
            # from the EV mode's game-theoretic analysis.
            strategy_mix = pool_strategy_profile.strategy_mix.copy()
            self._contrarian_strength = pool_strategy_profile.contrarian_strength
            self._variance_target = getattr(
                pool_strategy_profile, "variance_target", 0.5,
            )
            logger.info(
                "Using pool strategy profile (pool_size=%d, payout=%s, "
                "contrarian_strength=%.1f, variance_target=%.2f) "
                "for bracket portfolio allocation: %s",
                pool_strategy_profile.pool_size,
                pool_strategy_profile.payout_structure,
                pool_strategy_profile.contrarian_strength,
                self._variance_target,
                strategy_mix,
            )
        elif strategy_mix is None:
            # Gap #5: Optimized for bracket portfolio format (2024+).
            # In large-field competitions, contrarian and champion-targeted
            # strategies dominate because chalk brackets are duplicated
            # across thousands of entrants.  Maximize the chance of
            # having a uniquely-good bracket rather than a consensus-average one.
            strategy_mix = {
                "chalk": 0.05,       # Minimal: only 1 or 2 "safe" brackets
                "balanced": 0.30,    # Model-probability sampling
                "contrarian": 0.35,  # Anti-correlated with public picks
                "targeted": 0.30,    # Champion-diversification brackets
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

        # Post-sampling search refinement
        if enable_search and brackets:
            brackets = self._refine_with_search(brackets, seed)

        return brackets

    def _refine_with_search(
        self,
        brackets: List[GeneratedBracket],
        seed: int = 42,
    ) -> List[GeneratedBracket]:
        """Refine top brackets using Simulated Annealing search.

        Takes the top 10% of brackets by log_probability, converts them
        to SearchBracket format, optimizes with SA, and replaces the
        originals if the search finds better fitness.

        Args:
            brackets: Portfolio of generated brackets.
            seed: Random seed for SA optimizer.

        Returns:
            Portfolio with top brackets refined by search.
        """
        try:
            from .bracket_search import (
                SAConfig,
                SearchBracket,
                SimulatedAnnealingOptimizer,
                BracketPick as SearchBracketPick,
            )
        except ImportError:
            logger.warning("bracket_search module unavailable; skipping refinement")
            return brackets

        n_refine = max(1, len(brackets) // 10)

        # Sort by log probability to find top candidates
        ranked = sorted(
            enumerate(brackets),
            key=lambda ib: ib[1].log_probability,
            reverse=True,
        )
        top_indices = [idx for idx, _ in ranked[:n_refine]]

        sa_config = SAConfig(
            max_iterations=500,
            max_no_improve=100,
            random_seed=seed,
            initial_temperature=0.5,
            cooling_rate=0.99,
            min_temperature=0.005,
            leverage_weight=0.6,
            robustness_weight=0.3,
            diversity_weight=0.1,
        )

        optimizer = SimulatedAnnealingOptimizer(
            predict_fn=self.predict_fn,
            config=sa_config,
            public_picks=self.public_picks,
        )

        refined_count = 0
        for idx in top_indices:
            original = brackets[idx]

            # Convert GeneratedBracket -> SearchBracket
            search_picks = [
                SearchBracketPick(
                    round_num=p.round_num,
                    game_idx=p.game_idx,
                    winner_id=p.winner_id,
                    loser_id=p.loser_id,
                    win_probability=p.win_probability,
                )
                for p in original.picks
            ]
            search_bracket = SearchBracket(
                picks=search_picks,
                champion=original.champion,
                log_probability=original.log_probability,
            )

            # Run SA optimization
            optimized = optimizer.optimize(search_bracket)

            # Convert back and replace if improved
            if optimized.fitness > optimizer.evaluate_bracket(search_bracket):
                new_picks = [
                    BracketPick(
                        round_num=p.round_num,
                        game_idx=p.game_idx,
                        winner_id=p.winner_id,
                        loser_id=p.loser_id,
                        win_probability=p.win_probability,
                    )
                    for p in optimized.picks
                ]
                new_log_prob = sum(
                    math.log(max(p.win_probability, 1e-10)) for p in new_picks
                )
                brackets[idx] = GeneratedBracket(
                    bracket_id=original.bracket_id,
                    picks=new_picks,
                    champion=optimized.champion,
                    final_four=[
                        p.winner_id for p in optimized.picks
                        if p.round_num == 4
                    ],
                    expected_score=original.expected_score,
                    log_probability=new_log_prob,
                    strategy=f"{original.strategy}_sa_refined",
                )
                refined_count += 1

        logger.info(
            "Search refinement: refined %d/%d top brackets via SA",
            refined_count, n_refine,
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
        # Scale MC noise by variance_target: higher variance_target → more noise
        # → more diverse/upset-heavy brackets in the simulation pool.
        # Base noise 0.12 (pipeline default); winner-take-all (vt=0.95) → 0.17;
        # top-25% (vt=0.15) → 0.10.
        noise_std = 0.10 + 0.08 * self._variance_target

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

        Over-represent champions where model_prob >> public_prob.  When
        per-round leverage data is available, also scores individual bracket
        simulations by how many per-round leverage picks they contain —
        preferring brackets that capture multiple leverage opportunities
        across rounds, not just the champion pick.

        The ``contrarian_strength`` from the pool strategy profile scales
        how aggressively leverage ratios are amplified:
        - 0.5: mild (small pools where accuracy matters more)
        - 1.0: neutral
        - 2.0+: aggressive (large pools where differentiation is mandatory)
        """
        if not self.public_picks and not self.round_public_picks:
            # Without public data, fall back to diverse selection
            return self._select_diverse_brackets(sim_results, n_brackets, rng)

        # Group by champion
        by_champion: Dict[str, List[GeneratedBracket]] = {}
        for b in sim_results:
            if b.champion not in by_champion:
                by_champion[b.champion] = []
            by_champion[b.champion].append(b)

        total_sims = len(sim_results)

        # Compute leverage ratio per champion, amplified by contrarian_strength
        champion_leverage = {}
        for champ, brackets in by_champion.items():
            model_p = len(brackets) / total_sims
            public_p = self.public_picks.get(champ, 0.01)
            raw_leverage = model_p / max(public_p, 0.001)
            # Apply contrarian_strength: raise leverage to the power of strength.
            # strength=1 is neutral; >1 amplifies high-leverage champions;
            # <1 dampens, pulling allocation back toward model probabilities.
            champion_leverage[champ] = raw_leverage ** self._contrarian_strength
            if self._contrarian_strength > 1.0:
                logger.debug(
                    "Contrarian champion %s: raw_lev=%.2f, amplified=%.2f "
                    "(strength=%.1f)",
                    champ, raw_leverage, champion_leverage[champ],
                    self._contrarian_strength,
                )

        # Build per-round leverage lookup for scoring individual brackets
        leverage_teams_by_round = self._build_leverage_lookup()

        # Weight selection toward high-leverage champions
        total_leverage = sum(champion_leverage.values())
        selected = []

        for champ, leverage in champion_leverage.items():
            n_for_champ = max(1, int(n_brackets * leverage / total_leverage))
            brackets = by_champion.get(champ, [])

            if leverage_teams_by_round:
                # Score each bracket by how many per-round leverage picks it
                # contains, then sort by (leverage_score, log_probability).
                for b in brackets:
                    b._leverage_score = self._score_bracket_leverage(
                        b, leverage_teams_by_round,
                    )
                brackets.sort(
                    key=lambda b: (b._leverage_score, b.log_probability),
                    reverse=True,
                )
            else:
                brackets.sort(key=lambda b: b.log_probability, reverse=True)

            selected.extend(brackets[:n_for_champ])

        return selected[:n_brackets]

    def _build_leverage_lookup(self) -> Dict[int, set]:
        """Build round -> set of leverage team IDs from leverage_picks."""
        if not self.leverage_picks:
            return {}
        round_to_num = {"R64": 0, "R32": 1, "S16": 2, "E8": 3, "F4": 4, "CHAMP": 5}
        lookup: Dict[int, set] = {}
        for lp in self.leverage_picks:
            rnd = lp.get("round", "")
            rn = round_to_num.get(rnd)
            if rn is not None:
                lookup.setdefault(rn, set()).add(lp.get("team_id", ""))
        return lookup

    def _score_bracket_leverage(
        self,
        bracket: GeneratedBracket,
        leverage_teams_by_round: Dict[int, set],
    ) -> float:
        """Score a bracket by how many leverage picks it contains.

        Each pick that matches a leverage opportunity contributes the
        pick's win_probability as its score (so higher-probability
        leverage picks are worth more).
        """
        score = 0.0
        for pick in bracket.picks:
            leverage_teams = leverage_teams_by_round.get(pick.round_num, set())
            if pick.winner_id in leverage_teams:
                score += pick.win_probability
        return score

    def _generate_targeted_brackets(
        self,
        sim_results: List[GeneratedBracket],
        all_teams: Dict[str, Dict],
        n_brackets: int,
        rng: np.random.Generator,
    ) -> List[GeneratedBracket]:
        """Generate brackets each targeting a specific champion.

        Ensures the portfolio includes at least one bracket for each
        viable champion (>1% model probability).  When public pick data
        is available, allocates more brackets to high-leverage champions
        (undervalued by the public) while still guaranteeing coverage
        of every viable champion.
        """
        by_champion: Dict[str, List[GeneratedBracket]] = {}
        for b in sim_results:
            if b.champion not in by_champion:
                by_champion[b.champion] = []
            by_champion[b.champion].append(b)

        total = len(sim_results)
        viable = {c: bs for c, bs in by_champion.items()
                   if len(bs) / total >= 0.01}

        if not viable:
            return self._select_diverse_brackets(sim_results, n_brackets, rng)

        # Leverage-weighted allocation: give more brackets to undervalued
        # champions while guaranteeing at least 1 per viable champion.
        leverage_teams_by_round = self._build_leverage_lookup()
        if self.public_picks:
            champion_weights = {}
            for champ in viable:
                model_p = len(viable[champ]) / total
                public_p = self.public_picks.get(champ, model_p)
                leverage = model_p / max(public_p, 0.001)
                # Blend leverage with model probability to avoid allocating
                # brackets to champions with high leverage but negligible
                # probability.  Weight: 60% leverage, 40% model probability.
                champion_weights[champ] = 0.6 * leverage + 0.4 * (model_p * 100)
            total_weight = sum(champion_weights.values())
        else:
            champion_weights = {c: len(bs) / total for c, bs in viable.items()}
            total_weight = sum(champion_weights.values())

        selected = []
        guaranteed = max(1, n_brackets // max(len(viable), 1) // 2)  # Floor guarantee
        remaining = n_brackets - guaranteed * len(viable)

        for champ, brackets in viable.items():
            weight = champion_weights.get(champ, 1.0) / max(total_weight, 1e-9)
            n_for_champ = guaranteed + max(0, int(remaining * weight))

            if leverage_teams_by_round:
                for b in brackets:
                    b._leverage_score = self._score_bracket_leverage(
                        b, leverage_teams_by_round,
                    )
                brackets.sort(
                    key=lambda b: (b._leverage_score, b.log_probability),
                    reverse=True,
                )
            else:
                brackets.sort(key=lambda b: b.log_probability, reverse=True)

            selected.extend(brackets[:n_for_champ])

        return selected[:n_brackets]
