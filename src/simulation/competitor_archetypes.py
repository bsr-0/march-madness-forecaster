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
        # Boost for high-seed underdogs
        upset_factor = 1.3 if seed >= 9 else 1.0
        # Less reliance on public, more on raw probability boosted by chaos factor
        raw = model_prob * upset_factor
        return min(1.0, 0.8 * raw + 0.2 * public_pct)


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
        team_hash = hash(team_id) % 10
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


def get_archetype_mix(
    pool_type: str = "espn_national",
) -> Dict[str, float]:
    """Return archetype prevalence distribution for a pool type.

    Args:
        pool_type: One of "espn_national", "office_pool", "kaggle", "friends_pool".

    Returns:
        Dict mapping archetype name to prevalence (sums to 1.0).
    """
    return POOL_TYPE_ARCHETYPES.get(pool_type, POOL_TYPE_ARCHETYPES["espn_national"]).copy()


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
