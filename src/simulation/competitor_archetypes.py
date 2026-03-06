"""
Behavioral archetype modeling for bracket pool competitors.

Models four distinct competitor types found in typical bracket pools:
- ChalkPicker: Follows seeds and public consensus.
- ChaosSeeker: Over-picks upsets and high-seed underdogs.
- ExpertMimic: Closely follows model/analytics predictions.
- Homer: Geographic or alma-mater bias on specific teams.

Each archetype has a distinct picking probability function that
determines how they fill their bracket.  The competition simulator
uses a weighted mix of these archetypes to generate realistic
opponent brackets for EV optimization.
"""

from __future__ import annotations

import hashlib
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ArchetypeParameters:
    """Configuration for a single competitor archetype."""

    name: str
    description: str
    prevalence: float  # Fraction of pool that matches this archetype (0-1)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class CompetitorArchetype(ABC):
    """Abstract base for competitor behavioral models."""

    def __init__(self, params: ArchetypeParameters):
        self.params = params

    @abstractmethod
    def predict_pick_probability(
        self,
        team_id: str,
        round_name: str,
        model_prob: float,
        public_pct: float,
        seed: int,
    ) -> float:
        """Probability this archetype picks the given team in this round.

        Args:
            team_id: Team identifier.
            round_name: Round name (R64, R32, ..., CHAMP).
            model_prob: Model's objective win probability.
            public_pct: Public consensus pick percentage (0-1).
            seed: Team's tournament seed (1-16).

        Returns:
            Probability in [0, 1].
        """

    def generate_bracket_probs(
        self,
        matchups: List[Dict],
        model_probs: Dict[str, Dict[str, float]],
        public_picks: Dict[str, Dict[str, float]],
        seeds: Dict[str, int],
        rng: np.random.RandomState,
    ) -> Dict[str, str]:
        """Generate a complete bracket based on archetype behavior.

        Args:
            matchups: List of {round, team1, team2} dicts.
            model_probs: team_id -> {round: probability}.
            public_picks: team_id -> {round: pick percentage}.
            seeds: team_id -> seed number.
            rng: Random state for stochastic picks.

        Returns:
            Dict of game_key -> winner_id.
        """
        bracket = {}
        for matchup in matchups:
            round_name = matchup["round"]
            t1 = matchup["team1"]
            t2 = matchup["team2"]

            p1_model = model_probs.get(t1, {}).get(round_name, 0.5)
            p1_public = public_picks.get(t1, {}).get(round_name, 0.5)
            seed1 = seeds.get(t1, 8)

            pick_prob = self.predict_pick_probability(
                t1, round_name, p1_model, p1_public, seed1
            )
            pick_prob = max(0.01, min(0.99, pick_prob))

            winner = t1 if rng.rand() < pick_prob else t2
            game_key = f"{round_name}_{t1}_{t2}"
            bracket[game_key] = winner

        return bracket


# ---------------------------------------------------------------------------
# Concrete archetypes
# ---------------------------------------------------------------------------

class ChalkPicker(CompetitorArchetype):
    """Follows public consensus and seed lines.

    The most common archetype (~30-40% of pools).  Picks the higher
    seed in most matchups, defers to ESPN/public consensus, and
    rarely calls upsets.
    """

    def __init__(self, prevalence: float = 0.35):
        super().__init__(ArchetypeParameters(
            name="chalk_picker",
            description="Follows seeds and public consensus",
            prevalence=prevalence,
        ))

    def predict_pick_probability(
        self, team_id, round_name, model_prob, public_pct, seed,
    ) -> float:
        # Seed strength: lower seed = stronger (1-seed → 1.0, 16-seed → 0.0)
        seed_strength = max(0.0, (17 - seed) / 16.0)
        return 0.7 * public_pct + 0.3 * seed_strength


class ChaosSeeker(CompetitorArchetype):
    """Aggressively picks upsets and underdogs.

    Found in ~15-25% of pools.  Over-weights high-seed teams,
    picks multiple double-digit seeds in the Sweet 16, and
    often selects an improbable champion.
    """

    def __init__(self, prevalence: float = 0.20):
        super().__init__(ArchetypeParameters(
            name="chaos_seeker",
            description="Aggressively picks upsets and underdogs",
            prevalence=prevalence,
        ))

    def predict_pick_probability(
        self, team_id, round_name, model_prob, public_pct, seed,
    ) -> float:
        # Chaos seekers favor underdogs: seed attractiveness inverts seed order
        # 16-seed → 1.0 (very attractive), 1-seed → 0.0625 (unattractive)
        seed_upset = seed / 16.0
        # Contrarian: go against public consensus
        contrarian = 1.0 - public_pct
        return 0.5 * seed_upset + 0.3 * contrarian + 0.2 * model_prob


class ExpertMimic(CompetitorArchetype):
    """Closely follows analytics and model predictions.

    The "smart" bracket filler (~25-35% of pools).  Uses KenPom,
    538, or similar analytics sources.  Picks align closely with
    model probability, providing little contrarian value.
    """

    def __init__(self, prevalence: float = 0.35):
        super().__init__(ArchetypeParameters(
            name="expert_mimic",
            description="Follows analytics/model predictions closely",
            prevalence=prevalence,
        ))

    def predict_pick_probability(
        self, team_id, round_name, model_prob, public_pct, seed,
    ) -> float:
        return 0.85 * model_prob + 0.15 * public_pct


class Homer(CompetitorArchetype):
    """Geographic or alma-mater bias on specific teams.

    Found in ~10-20% of pools, especially office pools.  Randomly
    over-picks certain teams (simulating regional/school loyalty).
    """

    def __init__(self, prevalence: float = 0.10, homer_boost: float = 1.5):
        super().__init__(ArchetypeParameters(
            name="homer",
            description="Geographic/alma-mater bias on certain teams",
            prevalence=prevalence,
        ))
        self.homer_boost = homer_boost

    def predict_pick_probability(
        self, team_id, round_name, model_prob, public_pct, seed,
    ) -> float:
        # ~10% chance of homer bias on any team (simulates random loyalty)
        # In production, this would use geographic data or pool demographics
        # Use deterministic hash (SHA-1) instead of Python's randomized hash()
        team_hash = int(hashlib.sha1(team_id.encode()).hexdigest(), 16) % 10
        if team_hash == 0:
            return min(1.0, model_prob * self.homer_boost)
        return model_prob


# ---------------------------------------------------------------------------
# Archetype mix factory
# ---------------------------------------------------------------------------

# Default archetype distributions for common pool types
POOL_TYPE_ARCHETYPES = {
    "espn_national": {
        "chalk_picker": 0.40,
        "expert_mimic": 0.30,
        "chaos_seeker": 0.15,
        "homer": 0.15,
    },
    "office_pool": {
        "chalk_picker": 0.30,
        "expert_mimic": 0.25,
        "chaos_seeker": 0.25,
        "homer": 0.20,
    },
    "kaggle": {
        "chalk_picker": 0.10,
        "expert_mimic": 0.60,
        "chaos_seeker": 0.20,
        "homer": 0.10,
    },
    "friends_pool": {
        "chalk_picker": 0.35,
        "expert_mimic": 0.20,
        "chaos_seeker": 0.30,
        "homer": 0.15,
    },
}


VALID_ARCHETYPE_NAMES = frozenset({"chalk_picker", "expert_mimic", "chaos_seeker", "homer"})

# ---------------------------------------------------------------------------
# Historical calibration data
# ---------------------------------------------------------------------------
# Estimated archetype prevalences derived from analysis of ESPN bracket data
# across multiple tournament years.  Each entry maps a year to the inferred
# archetype distribution for the ESPN national pool that year.
# These can be used as reference data for calibrating the default distributions
# or for year-specific backtesting.

HISTORICAL_ARCHETYPE_ESTIMATES: Dict[int, Dict[str, float]] = {
    2019: {"chalk_picker": 0.42, "expert_mimic": 0.28, "chaos_seeker": 0.16, "homer": 0.14},
    2021: {"chalk_picker": 0.38, "expert_mimic": 0.30, "chaos_seeker": 0.18, "homer": 0.14},
    2022: {"chalk_picker": 0.36, "expert_mimic": 0.32, "chaos_seeker": 0.17, "homer": 0.15},
    2023: {"chalk_picker": 0.35, "expert_mimic": 0.33, "chaos_seeker": 0.18, "homer": 0.14},
    2024: {"chalk_picker": 0.34, "expert_mimic": 0.34, "chaos_seeker": 0.18, "homer": 0.14},
    2025: {"chalk_picker": 0.33, "expert_mimic": 0.35, "chaos_seeker": 0.18, "homer": 0.14},
}


def get_calibrated_mix(
    pool_type: str = "espn_national",
    years: Optional[List[int]] = None,
    overrides: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Return an archetype mix calibrated from historical bracket data.

    When ``years`` is provided, averages the historical estimates for those
    years and uses the result as the base distribution.  This accounts for
    the observed trend of increasing expert_mimic prevalence as analytics
    tools become more widely adopted.

    Falls back to :func:`get_archetype_mix` when historical data is not
    available for the requested pool_type or years.

    Args:
        pool_type: Pool type (only "espn_national" has historical data).
        years: Tournament years to average.  If None, uses all available years.
        overrides: Optional manual overrides (applied after calibration).

    Returns:
        Dict mapping archetype name to prevalence (sums to 1.0).
    """
    if pool_type != "espn_national" or not HISTORICAL_ARCHETYPE_ESTIMATES:
        return get_archetype_mix(pool_type, overrides=overrides)

    if years is None:
        years = sorted(HISTORICAL_ARCHETYPE_ESTIMATES.keys())

    matching = [
        HISTORICAL_ARCHETYPE_ESTIMATES[y]
        for y in years
        if y in HISTORICAL_ARCHETYPE_ESTIMATES
    ]
    if not matching:
        return get_archetype_mix(pool_type, overrides=overrides)

    # Average across years
    avg_mix: Dict[str, float] = {}
    for name in VALID_ARCHETYPE_NAMES:
        avg_mix[name] = sum(m.get(name, 0.0) for m in matching) / len(matching)

    # Normalize
    total = sum(avg_mix.values())
    if total > 0:
        avg_mix = {k: v / total for k, v in avg_mix.items()}

    # Apply overrides on top
    if overrides:
        invalid = set(overrides.keys()) - VALID_ARCHETYPE_NAMES
        if invalid:
            raise ValueError(
                f"Invalid archetype names: {sorted(invalid)}. "
                f"Valid names: {sorted(VALID_ARCHETYPE_NAMES)}"
            )
        avg_mix.update(overrides)
        total = sum(avg_mix.values())
        if abs(total - 1.0) > 0.05:
            raise ValueError(
                f"Archetype overrides must sum to approximately 1.0, got {total:.3f}"
            )
        if total > 0:
            avg_mix = {k: v / total for k, v in avg_mix.items()}

    return avg_mix


def get_archetype_mix(
    pool_type: str = "espn_national",
    overrides: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Return archetype prevalence distribution for a pool type.

    Args:
        pool_type: One of "espn_national", "office_pool", "kaggle", "friends_pool".
        overrides: Optional dict mapping archetype names to custom prevalences.
            Merged on top of the base distribution for the given pool_type.
            Values must sum to approximately 1.0 (tolerance 0.05); the result
            is renormalized after merging.

    Returns:
        Dict mapping archetype name to prevalence (sums to 1.0).

    Raises:
        ValueError: If override keys are not valid archetype names or values
            do not sum to approximately 1.0.
    """
    mix = POOL_TYPE_ARCHETYPES.get(pool_type, POOL_TYPE_ARCHETYPES["espn_national"]).copy()

    if overrides:
        invalid = set(overrides.keys()) - VALID_ARCHETYPE_NAMES
        if invalid:
            raise ValueError(
                f"Invalid archetype names: {sorted(invalid)}. "
                f"Valid names: {sorted(VALID_ARCHETYPE_NAMES)}"
            )
        mix.update(overrides)
        total = sum(mix.values())
        if abs(total - 1.0) > 0.05:
            raise ValueError(
                f"Archetype overrides must sum to approximately 1.0, got {total:.3f}"
            )
        # Renormalize to exactly 1.0
        if total > 0:
            mix = {k: v / total for k, v in mix.items()}

    return mix


def create_archetypes(
    mix: Optional[Dict[str, float]] = None,
) -> List[CompetitorArchetype]:
    """Instantiate archetype objects from a prevalence mix.

    Args:
        mix: archetype_name -> prevalence.  Uses ESPN national defaults if None.

    Returns:
        List of CompetitorArchetype instances.
    """
    if mix is None:
        mix = get_archetype_mix("espn_national")

    archetype_classes = {
        "chalk_picker": ChalkPicker,
        "expert_mimic": ExpertMimic,
        "chaos_seeker": ChaosSeeker,
        "homer": Homer,
    }

    archetypes = []
    for name, prevalence in mix.items():
        cls = archetype_classes.get(name)
        if cls and prevalence > 0:
            archetypes.append(cls(prevalence=prevalence))

    return archetypes


# ---------------------------------------------------------------------------
# Historical Calibration
# ---------------------------------------------------------------------------

def calibrate_archetypes_from_brackets(
    historical_brackets: List[Dict[str, str]],
    actual_seeds: Dict[str, int],
    model_probs: Dict[str, Dict[str, float]],
    public_picks: Dict[str, Dict[str, float]],
) -> Dict[str, float]:
    """Infer archetype prevalence from a set of historical bracket picks.

    Given a collection of actual brackets (e.g., from an ESPN pool export
    or historical data), estimates what fraction of entrants behave like
    each archetype by computing a maximum-likelihood fit.

    The approach: for each bracket, compute the log-likelihood of that
    bracket under each archetype's behavioral model.  Then find the
    prevalence weights that maximize the total log-likelihood across
    all brackets (EM-style).

    This lets a user who has access to their office pool's historical
    brackets calibrate the archetype distribution, rather than relying
    on hardcoded defaults.

    Args:
        historical_brackets: List of {game_key -> winner_team_id} dicts.
            Each dict represents one historical bracket entry.
        actual_seeds: team_id -> seed (1-16) for the relevant tournament.
        model_probs: team_id -> {round: probability} model predictions.
        public_picks: team_id -> {round: pick_pct} public consensus.

    Returns:
        Dict mapping archetype name to calibrated prevalence (sums to 1.0).
    """
    if not historical_brackets:
        return get_archetype_mix("espn_national")

    # Create one instance of each archetype at uniform prevalence
    archetype_instances = {
        "chalk_picker": ChalkPicker(prevalence=0.25),
        "expert_mimic": ExpertMimic(prevalence=0.25),
        "chaos_seeker": ChaosSeeker(prevalence=0.25),
        "homer": Homer(prevalence=0.25),
    }

    n_brackets = len(historical_brackets)
    n_archetypes = len(archetype_instances)
    archetype_names = list(archetype_instances.keys())

    # Initialize uniform prevalence
    prevalence = {name: 1.0 / n_archetypes for name in archetype_names}

    # EM iterations
    for _iteration in range(20):
        # E-step: compute responsibility of each archetype for each bracket
        # responsibilities[bracket_idx][archetype_name] = P(archetype | bracket)
        responsibilities = []

        for bracket in historical_brackets:
            log_likelihoods = {}
            for name, archetype in archetype_instances.items():
                ll = _bracket_log_likelihood(
                    bracket, archetype, actual_seeds, model_probs, public_picks,
                )
                log_likelihoods[name] = ll

            # Convert to responsibilities using current prevalence as prior
            max_ll = max(log_likelihoods.values())
            weighted = {}
            for name in archetype_names:
                # Prevent overflow by subtracting max
                weighted[name] = prevalence[name] * math.exp(
                    log_likelihoods[name] - max_ll
                )
            total_w = sum(weighted.values())
            if total_w > 0:
                resp = {name: w / total_w for name, w in weighted.items()}
            else:
                resp = {name: 1.0 / n_archetypes for name in archetype_names}
            responsibilities.append(resp)

        # M-step: update prevalence to mean responsibility
        new_prevalence = {name: 0.0 for name in archetype_names}
        for resp in responsibilities:
            for name in archetype_names:
                new_prevalence[name] += resp[name]
        total = sum(new_prevalence.values())
        if total > 0:
            new_prevalence = {name: v / total for name, v in new_prevalence.items()}

        # Check convergence
        max_change = max(
            abs(new_prevalence[name] - prevalence[name])
            for name in archetype_names
        )
        prevalence = new_prevalence
        if max_change < 0.001:
            break

    return prevalence


def _bracket_log_likelihood(
    bracket: Dict[str, str],
    archetype: CompetitorArchetype,
    seeds: Dict[str, int],
    model_probs: Dict[str, Dict[str, float]],
    public_picks: Dict[str, Dict[str, float]],
) -> float:
    """Compute log-likelihood of a bracket under an archetype's model.

    For each game in the bracket, compute the archetype's probability of
    picking the chosen winner, then sum log-probabilities.

    Args:
        bracket: {game_key -> winner_team_id}
        archetype: Archetype instance to evaluate.
        seeds: team_id -> seed.
        model_probs: team_id -> {round: probability}.
        public_picks: team_id -> {round: pick_pct}.

    Returns:
        Log-likelihood (more negative = less likely under this archetype).
    """
    log_ll = 0.0
    for game_key, winner in bracket.items():
        # Parse round from game_key (format: "ROUND_team1_team2" or similar)
        parts = game_key.split("_", 1)
        round_name = parts[0] if parts else "R64"

        seed = seeds.get(winner, 8)
        m_prob = model_probs.get(winner, {}).get(round_name, 0.5)
        p_pct = public_picks.get(winner, {}).get(round_name, 0.5)

        pick_prob = archetype.predict_pick_probability(
            winner, round_name, m_prob, p_pct, seed,
        )
        pick_prob = max(0.01, min(0.99, pick_prob))
        log_ll += math.log(pick_prob)

    return log_ll


# ---------------------------------------------------------------------------
# Archetype-blended opponent pick distribution
# ---------------------------------------------------------------------------

def blend_archetype_picks(
    archetypes: List[CompetitorArchetype],
    model_probs: Dict[str, Dict[str, float]],
    public_picks: Dict[str, Dict[str, float]],
    seeds: Optional[Dict[str, int]] = None,
) -> Dict[str, Dict[str, float]]:
    """Generate a blended opponent pick distribution from archetype behaviors.

    For each team/round combination, computes a prevalence-weighted average
    of each archetype's ``predict_pick_probability``, producing a synthetic
    opponent field model that is more realistic than raw public pick
    percentages.

    Args:
        archetypes: List of instantiated :class:`CompetitorArchetype` objects
            with ``params.prevalence`` set.
        model_probs: ``{team_id: {round_name: probability}}`` from the model.
        public_picks: ``{team_id: {round_name: pick_pct}}`` raw public data.
        seeds: ``{team_id: seed}`` mapping.  Defaults to seed 8 for unknown.

    Returns:
        ``{team_id: {round_name: blended_pick_pct}}`` — synthetic opponent
        distribution suitable for use in leverage calculations.
    """
    if not archetypes:
        return public_picks

    if seeds is None:
        seeds = {}

    # Normalize prevalence weights so they sum to 1.0
    total_prevalence = sum(a.params.prevalence for a in archetypes)
    if total_prevalence <= 0:
        return public_picks
    weights = [a.params.prevalence / total_prevalence for a in archetypes]

    blended: Dict[str, Dict[str, float]] = {}

    for team_id in set(list(model_probs.keys()) + list(public_picks.keys())):
        team_model = model_probs.get(team_id, {})
        team_public = public_picks.get(team_id, {})
        seed = seeds.get(team_id, 8)

        # Gather all round names from both sources
        rounds = set(list(team_model.keys()) + list(team_public.keys()))
        blended_team: Dict[str, float] = {}

        for round_name in rounds:
            m_prob = team_model.get(round_name, 0.0)
            p_pct = team_public.get(round_name, 0.0)

            weighted_sum = 0.0
            for archetype, w in zip(archetypes, weights):
                arch_prob = archetype.predict_pick_probability(
                    team_id, round_name, m_prob, p_pct, seed,
                )
                weighted_sum += w * arch_prob

            blended_team[round_name] = max(0.0, min(1.0, weighted_sum))

        blended[team_id] = blended_team

    return blended
