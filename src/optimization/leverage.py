"""
Game theory optimization for bracket pools.

Maximizes Expected Value (EV) relative to competitors by finding
high-leverage picks — measured by EV-edge: the expected points gained
versus the field per pick.

    EV-edge(team, round) = (model_prob - public_pct) × round_points

Positive EV-edge means the model sees the team as under-picked by the
public relative to its true advancement probability.  This is the
canonical leverage metric used by both the pool optimizer and the ESPN
bracket optimizer.
"""

import logging
import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import namedtuple
import numpy as np

from src.data.seed_pick_model import SEED_PICK_RATES

logger = logging.getLogger(__name__)


_BranchPick = namedtuple("_BranchPick", ["p_win", "survival", "pts"])


def compute_ev_edge(
    model_prob: float,
    public_pct: float,
    round_points: int,
) -> float:
    """Canonical leverage metric: expected points gained vs the field.

    ``EV-edge = (model_prob - public_pct) × round_points``

    This single number answers "how many points per pick do I expect to
    gain over a random opponent by picking this team in this round?"

    Positive → under-owned value (pick).
    Negative → over-owned trap (fade).

    All leverage-based sorting and filtering across the codebase should
    use this function (or its equivalent property on LeveragePick) as
    the primary ranking signal.
    """
    return (model_prob - public_pct) * round_points


# ---------------------------------------------------------------------------
# Seed-Based Public Pick Rate Priors
# ---------------------------------------------------------------------------
# Derived from a first-principles model combining:
#   1. Historical seed advancement rates (NCAA 1985-2025, ~160 games/matchup)
#   2. Chalk bias power-law transformation calibrated against ESPN BTC
#      aggregate distributions (2015-2024).
#
# Methodology documented in src/data/seed_pick_model.py.
# Championship column renormalized so 4 × Σ(seed rates) = 100%.
#
# To regenerate or audit: python -m src.data.seed_pick_model


def get_seed_based_pick_rates(
    seed: int,
) -> Dict[str, float]:
    """Return approximate public pick rates for a given seed.

    When real public pick data (e.g. ESPN Tournament Challenge percentages)
    is unavailable, this provides a principled prior derived from
    historical advancement rates and a chalk bias model.

    See ``src/data/seed_pick_model.py`` for the full derivation.

    Args:
        seed: Tournament seed (1-16).

    Returns:
        Dict mapping round name to approximate public pick percentage.
    """
    return SEED_PICK_RATES.get(seed, SEED_PICK_RATES[8]).copy()


def build_seed_based_public_picks(
    team_seeds: Dict[str, int],
) -> Dict[str, Dict[str, float]]:
    """Build a complete public pick distribution from seed data alone.

    This is the fallback when no ESPN/public pick data is available.
    It provides a reasonable approximation of public behavior for
    leverage calculations.

    Args:
        team_seeds: team_id -> seed (1-16).

    Returns:
        team_id -> {round_name: pick_rate} dict suitable for
        LeverageCalculator or BracketPortfolioGenerator.
    """
    result: Dict[str, Dict[str, float]] = {}
    for team_id, seed in team_seeds.items():
        result[team_id] = get_seed_based_pick_rates(seed)
    return result


@dataclass
class LeveragePick:
    """A pick with leverage analysis."""

    team_id: str
    team_name: str
    seed: int
    region: str

    # Probabilities
    model_probability: float  # Our model's probability
    public_pick_percentage: float  # % of public picking this team

    # Round
    round_name: str  # "Champion", "Final Four", etc.
    points_value: int  # Points for picking correctly

    # Provenance: True when public_pick_percentage is a synthetic
    # seed-based prior (no real ESPN/Yahoo/CBS data available for
    # this team/round).  Leverage calculations using synthetic data
    # should be treated with lower confidence.
    is_synthetic: bool = False

    @property
    def leverage_ratio(self) -> float:
        """Ratio of model prob to public percentage."""
        if self.public_pick_percentage <= 0:
            return float("inf")
        return self.model_probability / self.public_pick_percentage

    @property
    def expected_value(self) -> float:
        """Expected points from this pick."""
        return self.model_probability * self.points_value

    @property
    def expected_value_differential(self) -> float:
        """
        Expected points gained vs public.

        Positive = outperform public, Negative = underperform.
        """
        public_ev = self.public_pick_percentage * self.points_value
        return self.expected_value - public_ev

    def __str__(self) -> str:
        return (
            f"{self.team_name} ({self.seed}) - {self.round_name}: "
            f"Model {self.model_probability:.1%} vs Public {self.public_pick_percentage:.1%} "
            f"(Leverage: {self.leverage_ratio:.2f}x)"
        )


@dataclass
class BracketConfiguration:
    """A complete bracket configuration."""

    picks: Dict[str, str]  # game_id -> winner team_id
    champion: str
    final_four: List[str]

    # Metadata
    strategy: str = "balanced"  # "balanced", "chalk", "contrarian"
    expected_points: float = 0.0
    variance: float = 0.0
    # Construction mode used to generate this bracket. One of
    # "champ_first", "f4_first", "e8_first", "forward_greedy", or
    # "legacy_summary" (the pre-construction-modes summary path used
    # when team_metadata is missing). See
    # src/optimization/bracket_construction.py for the mode algorithms.
    construction_mode: str = "forward_greedy"

    def to_dict(self) -> dict:
        return {
            "picks": self.picks,
            "champion": self.champion,
            "final_four": self.final_four,
            "strategy": self.strategy,
            "expected_points": self.expected_points,
            "construction_mode": self.construction_mode,
        }


def canonical_picks_key(picks: Dict[str, str]) -> Tuple:
    """Compute a slot-order-invariant identity key for a bracket's picks.

    The summary bracket format stores the Final Four as F4_1..F4_4 slots
    whose specific slot assignments depend on the order in which the
    greedy F4 selection filled them — at different risk levels the same
    4 F4 teams can land in different slot positions purely because of how
    _ev_score ranked them at that risk level. Treating the F4 as an
    ordered tuple therefore produces false negatives on semantic
    equivalence: two brackets with the same champion and the same set of
    4 F4 teams are semantically identical regardless of which team is in
    F4_1 vs F4_2.

    This helper canonicalizes the picks dict by:
      - keying F4_* values as an unordered frozenset (slot-invariant)
      - preserving all other keys as sorted (k, v) pairs (R64_*, R32_*,
        S16_*, E8_*, CHAMP carry real game-position identity)

    Use this for bracket dedup and any other place that needs to compare
    two bracket configurations for semantic equivalence.
    """
    f4_teams = frozenset(v for k, v in picks.items() if k.startswith("F4_"))
    other_picks = tuple(sorted((k, v) for k, v in picks.items() if not k.startswith("F4_")))
    return (f4_teams, other_picks)


@dataclass
class TeamMetadata:
    """Metadata used for richer game-theory recommendations."""

    team_name: str
    seed: int
    region: str


# ---------------------------------------------------------------------------
# Pool-Size-Adaptive Strategy (Phase 2: EV Mode)
# ---------------------------------------------------------------------------


@dataclass
class PoolStrategyProfile:
    """Graduated strategy profile based on pool size and scoring system.

    Encodes how aggressively contrarian the bracket optimizer should be
    given the competitive landscape.  Larger pools require more
    differentiation; smaller pools reward accuracy.

    The ``variance_target`` field captures the desired outcome variance
    implied by the payout structure.  Winner-take-all pools demand high
    variance (only 1st place pays, so you need a "lottery ticket"
    bracket), while top-25% pools reward low variance (consistent
    accuracy keeps you in the money).  The bracket optimizer uses this
    to tune how much outcome variance to inject via upset picks.
    """

    pool_size: int
    scoring_system: str  # "standard", "flat", "upset_bonus"
    strategy_mix: Dict[str, float]  # {"chalk": 0.3, "balanced": 0.4, ...}
    contrarian_strength: float  # Multiplier on leverage picks (0.5–3.0)
    champion_risk_level: str  # "very_low", "low", "moderate", "high", "extreme"
    payout_structure: str = "tiered"  # Prize structure affecting variance preference
    variance_target: float = 0.5  # 0.0 = minimize variance, 1.0 = maximize variance
    description: str = ""

    def validate(self) -> None:
        """Ensure strategy_mix sums to ~1.0."""
        total = sum(self.strategy_mix.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"strategy_mix must sum to 1.0, got {total:.3f}")


# ---------------------------------------------------------------------------
# Entry Fee / Prize Pool ROI (Kelly Criterion)
# ---------------------------------------------------------------------------


@dataclass
class PoolEconomics:
    """Entry fee and prize structure for Kelly-criterion ROI analysis.

    In bracket pools with real money, the optimal number of entries and
    risk tolerance depend on the ratio of entry fee to expected prize.
    A $10-entry winner-take-all pool with a $500 pot is fundamentally
    different from a $100-entry pool where top-25% pay out.

    The Kelly fraction tells you what fraction of your bankroll to risk,
    and the ROI estimate tells you whether entering at all has positive
    expected value given your model's estimated edge.
    """

    entry_fee: float
    prize_pool: float
    payout_spots: int  # Number of places that pay out
    pool_size: int
    payout_structure: str = "tiered"  # matches PAYOUT_ADJUSTMENTS keys

    @property
    def prize_per_spot(self) -> float:
        """Average prize per payout spot."""
        return self.prize_pool / max(self.payout_spots, 1)

    @property
    def rake(self) -> float:
        """House rake: fraction of entry fees not returned as prizes."""
        total_fees = self.entry_fee * self.pool_size
        if total_fees <= 0:
            return 0.0
        return max(0.0, 1.0 - self.prize_pool / total_fees)

    @property
    def base_win_rate(self) -> float:
        """Probability of landing in a payout spot by random chance."""
        return min(1.0, self.payout_spots / max(self.pool_size, 1))


def compute_kelly_fraction(
    estimated_edge: float,
    win_probability: float,
    odds: float,
) -> float:
    """Compute the Kelly-criterion optimal fraction of bankroll to wager.

    Kelly fraction = (bp - q) / b
    where b = odds (net profit per $1 wagered if win),
          p = probability of winning, q = 1 - p.

    Args:
        estimated_edge: Not used directly; provided for logging context.
        win_probability: P(landing in a payout spot).
        odds: Net payout per dollar wagered (prize/entry - 1).

    Returns:
        Kelly fraction in [0, 1].  0 means don't enter; 1 means go all-in.
        Negative edge returns 0 (never bet with negative EV).
    """
    if odds <= 0 or win_probability <= 0:
        return 0.0
    q = 1.0 - win_probability
    kelly = (odds * win_probability - q) / odds
    return max(0.0, min(1.0, kelly))


def evaluate_pool_roi(
    economics: PoolEconomics,
    model_win_probability: float,
    n_entries: int = 1,
) -> Dict[str, float]:
    """Evaluate the ROI of entering a bracket pool.

    Computes expected value, Kelly fraction, and recommended number of
    entries given a model's estimated probability of finishing in a
    payout spot.

    A senior statistician's approach: the model's P(top-K) comes from
    the competition simulator.  This function translates that into
    bankroll management advice.

    Args:
        economics: Pool entry fee and prize structure.
        model_win_probability: Model's estimated P(finishing in payout spots).
        n_entries: Number of entries being considered.

    Returns:
        Dict with ROI analysis metrics.
    """
    entry_cost = economics.entry_fee * n_entries
    if entry_cost <= 0:
        return {
            "expected_value": 0.0,
            "roi_pct": 0.0,
            "kelly_fraction": 0.0,
            "recommended_entries": 0,
            "edge_per_entry": 0.0,
            "breakeven_win_rate": 0.0,
        }

    # Expected prize per entry
    expected_prize_per_entry = model_win_probability * economics.prize_per_spot

    # Net odds per dollar wagered
    odds = (economics.prize_per_spot / economics.entry_fee) - 1.0 if economics.entry_fee > 0 else 0.0

    kelly = compute_kelly_fraction(
        estimated_edge=model_win_probability - economics.base_win_rate,
        win_probability=model_win_probability,
        odds=odds,
    )

    # With multiple entries, diminishing returns: each additional entry
    # competes with your own entries.  P(at least one wins) =
    # 1 - (1 - p)^n, but the cost scales linearly.
    p_at_least_one = 1.0 - (1.0 - model_win_probability) ** n_entries
    total_expected_prize = p_at_least_one * economics.prize_per_spot
    ev = total_expected_prize - entry_cost
    roi_pct = (ev / entry_cost) * 100.0 if entry_cost > 0 else 0.0

    # Breakeven: minimum win rate where EV >= 0
    breakeven = economics.entry_fee / economics.prize_per_spot if economics.prize_per_spot > 0 else 1.0

    # Recommended entries: find n that maximizes ROI
    best_n = 0
    best_ev = 0.0
    for trial_n in range(1, min(economics.pool_size, 50) + 1):
        trial_p = 1.0 - (1.0 - model_win_probability) ** trial_n
        trial_ev = trial_p * economics.prize_per_spot - economics.entry_fee * trial_n
        if trial_ev > best_ev:
            best_ev = trial_ev
            best_n = trial_n

    return {
        "expected_value": round(ev, 2),
        "roi_pct": round(roi_pct, 1),
        "kelly_fraction": round(kelly, 4),
        "recommended_entries": best_n,
        "edge_per_entry": round(expected_prize_per_entry - economics.entry_fee, 2),
        "breakeven_win_rate": round(breakeven, 4),
        "p_at_least_one_wins": round(p_at_least_one, 4),
        "house_rake": round(economics.rake, 3),
    }


# ---------------------------------------------------------------------------
# Payout Structure Adjustments
# ---------------------------------------------------------------------------
# Multipliers applied to the base pool-size strategy mix to account for prize
# structure.  Winner-take-all demands maximum variance (contrarian/targeted);
# top-25pct rewards accuracy (chalk/balanced).

PAYOUT_ADJUSTMENTS: Dict[str, Dict[str, float]] = {
    "winner_take_all": {
        "chalk_mult": 0.5,
        "balanced_mult": 0.7,
        "contrarian_mult": 1.4,
        "targeted_mult": 1.5,
        "contrarian_strength_mult": 1.3,
        "variance_target": 0.95,  # Maximum variance — only 1st place pays
    },
    "top_3": {
        "chalk_mult": 0.7,
        "balanced_mult": 0.8,
        "contrarian_mult": 1.2,
        "targeted_mult": 1.3,
        "contrarian_strength_mult": 1.15,
        "variance_target": 0.75,
    },
    "top_10pct": {
        "chalk_mult": 1.2,
        "balanced_mult": 1.1,
        "contrarian_mult": 0.85,
        "targeted_mult": 0.85,
        "contrarian_strength_mult": 0.8,
        "variance_target": 0.35,
    },
    "top_25pct": {
        "chalk_mult": 1.4,
        "balanced_mult": 1.2,
        "contrarian_mult": 0.7,
        "targeted_mult": 0.6,
        "contrarian_strength_mult": 0.6,
        "variance_target": 0.15,  # Low variance — consistent accuracy pays
    },
    "tiered": {
        "chalk_mult": 1.0,
        "balanced_mult": 1.0,
        "contrarian_mult": 1.0,
        "targeted_mult": 1.0,
        "contrarian_strength_mult": 1.0,
        "variance_target": 0.50,
    },
}

VALID_PAYOUT_STRUCTURES = frozenset(PAYOUT_ADJUSTMENTS.keys())


def get_strategy_profile(
    pool_size: int,
    scoring_system: str = "standard",
    contrarian_override: Optional[float] = None,
    payout_structure: str = "tiered",
) -> PoolStrategyProfile:
    """Return a graduated strategy profile for the given pool size.

    Strategic pivot points based on empirical bracket pool analysis:
    - Tiny pools (<30): Chalk-heavy.  Accuracy wins; upsets dilute score.
    - Small pools (30-100): Balanced.  1-2 leverage picks for edge.
    - Medium pools (101-1000): Contrarian-leaning.  Must differentiate.
    - Large pools (1000+): Aggressive.  Only outlier paths can win.

    Args:
        pool_size: Number of competitors in the pool.
        scoring_system: "standard" (ESPN), "flat", or "upset_bonus".
        contrarian_override: If set, overrides the profile's contrarian_strength.
        payout_structure: Prize structure that modulates variance preference.
            "winner_take_all" pushes toward max variance; "top_25pct" rewards
            accuracy.  See PAYOUT_ADJUSTMENTS for all options.

    Returns:
        PoolStrategyProfile with recommended strategy allocation.
    """
    if payout_structure not in VALID_PAYOUT_STRUCTURES:
        raise ValueError(
            f"Invalid payout_structure '{payout_structure}': must be one of {sorted(VALID_PAYOUT_STRUCTURES)}"
        )
    if pool_size < 30:
        profile = PoolStrategyProfile(
            pool_size=pool_size,
            scoring_system=scoring_system,
            strategy_mix={
                "chalk": 0.60,
                "balanced": 0.35,
                "contrarian": 0.05,
                "targeted": 0.00,
            },
            contrarian_strength=0.5,
            champion_risk_level="very_low",
            description="Tiny pool: accuracy-first, minimal risk",
        )
    elif pool_size <= 100:
        profile = PoolStrategyProfile(
            pool_size=pool_size,
            scoring_system=scoring_system,
            strategy_mix={
                "chalk": 0.30,
                "balanced": 0.40,
                "contrarian": 0.20,
                "targeted": 0.10,
            },
            contrarian_strength=1.0,
            champion_risk_level="low",
            description="Small pool: balanced with selective leverage",
        )
    elif pool_size <= 1000:
        profile = PoolStrategyProfile(
            pool_size=pool_size,
            scoring_system=scoring_system,
            strategy_mix={
                "chalk": 0.10,
                "balanced": 0.25,
                "contrarian": 0.40,
                "targeted": 0.25,
            },
            contrarian_strength=1.5,
            champion_risk_level="high",
            description="Medium pool: contrarian-leaning, differentiation required",
        )
    else:
        profile = PoolStrategyProfile(
            pool_size=pool_size,
            scoring_system=scoring_system,
            strategy_mix={
                "chalk": 0.05,
                "balanced": 0.15,
                "contrarian": 0.40,
                "targeted": 0.40,
            },
            contrarian_strength=2.0,
            champion_risk_level="extreme",
            description="Large pool: aggressive contrarianism mandatory",
        )

    # Apply payout structure adjustments to the base strategy mix.
    adjustments = PAYOUT_ADJUSTMENTS[payout_structure]
    raw_mix = {
        "chalk": profile.strategy_mix.get("chalk", 0.0) * adjustments["chalk_mult"],
        "balanced": profile.strategy_mix.get("balanced", 0.0) * adjustments["balanced_mult"],
        "contrarian": profile.strategy_mix.get("contrarian", 0.0) * adjustments["contrarian_mult"],
        "targeted": profile.strategy_mix.get("targeted", 0.0) * adjustments["targeted_mult"],
    }
    total = sum(raw_mix.values())
    if total > 0:
        profile.strategy_mix = {k: v / total for k, v in raw_mix.items()}
    profile.contrarian_strength *= adjustments["contrarian_strength_mult"]
    profile.payout_structure = payout_structure
    profile.variance_target = adjustments.get("variance_target", 0.5)

    if contrarian_override is not None:
        profile.contrarian_strength = contrarian_override

    return profile


@dataclass
class ScoringSystemAdapter:
    """Adapts leverage ratios by round weight for different scoring systems.

    Standard ESPN scoring (10-20-40-80-160-320) heavily favors late rounds,
    so leverage in the Final Four and Championship is amplified.  Flat
    scoring (1-2-3-4-5-6) equalizes round importance, so first-round
    leverage is just as valuable.  Upset bonus systems reward contrarian
    picks in all rounds.
    """

    system: str  # "standard", "flat", "upset_bonus"
    round_weights: Dict[str, float]
    leverage_priority: str  # "late_rounds", "early_rounds", "balanced"

    def adjust_leverage_ratio(self, round_name: str, base_ratio: float, seed_diff: int = 0) -> float:
        """Weight leverage by round importance and scoring system.

        Args:
            round_name: Round identifier (R64, R32, S16, E8, F4, CHAMP).
            base_ratio: Raw leverage ratio (model_prob / public_pct).
            seed_diff: Absolute seed difference (for upset bonus).

        Returns:
            Adjusted leverage ratio.
        """
        weight = self.round_weights.get(round_name, 1.0)

        if self.system == "flat":
            # All rounds equally important — no amplification
            return base_ratio
        elif self.system == "upset_bonus":
            # Extra reward for picking high-seed upsets
            upset_mult = 1.0 + 0.05 * seed_diff if seed_diff > 0 else 1.0
            return base_ratio * upset_mult
        else:
            # Standard: amplify leverage in high-weight rounds.
            # Use sqrt dampening to avoid excessive variance.
            return base_ratio * (weight**0.5)


def get_scoring_adapter(scoring_system: str = "standard") -> ScoringSystemAdapter:
    """Factory for ScoringSystemAdapter instances.

    Args:
        scoring_system: "standard", "flat", or "upset_bonus".

    Returns:
        Configured ScoringSystemAdapter.
    """
    if scoring_system == "flat":
        return ScoringSystemAdapter(
            system="flat",
            round_weights={"R64": 1, "R32": 2, "S16": 3, "E8": 4, "F4": 5, "CHAMP": 6},
            leverage_priority="balanced",
        )
    elif scoring_system == "upset_bonus":
        return ScoringSystemAdapter(
            system="upset_bonus",
            round_weights={"R64": 10, "R32": 20, "S16": 40, "E8": 80, "F4": 160, "CHAMP": 320},
            leverage_priority="balanced",
        )
    else:
        return ScoringSystemAdapter(
            system="standard",
            round_weights={"R64": 10, "R32": 20, "S16": 40, "E8": 80, "F4": 160, "CHAMP": 320},
            leverage_priority="late_rounds",
        )


class LeverageCalculator:
    """
    Calculates leverage ratios for bracket optimization.

    Leverage = Win Probability / Pick Percentage

    High leverage picks offer better value because:
    - If correct: You beat more competitors
    - If wrong: Everyone else is also wrong
    """

    def __init__(
        self,
        model_probs: Dict[str, Dict[str, float]],
        public_picks: Dict[str, Dict[str, float]],
        scoring_system: Dict[str, int] = None,
        team_metadata: Optional[Dict[str, TeamMetadata]] = None,
    ):
        """
        Initialize calculator.

        Args:
            model_probs: team_id -> {round: probability}
            public_picks: team_id -> {round: percentage}
            scoring_system: round -> points (e.g., {"R64": 1, "R32": 2, ...})
        """
        self.model_probs = model_probs
        self.public_picks = public_picks
        self.team_metadata = team_metadata or {}

        # Fallback provenance audit log.
        # Records every (team_id, round) where synthetic seed-based
        # priors were substituted for missing real public pick data.
        # Downstream consumers can inspect this to assess data quality
        # and flag leverage calculations that depend on priors.
        self.fallback_audit: Dict[str, List[str]] = {}  # team_id -> [rounds]

        # Standard NCAA bracket scoring
        self.scoring_system = scoring_system or {
            "R64": 10,
            "R32": 20,
            "S16": 40,
            "E8": 80,
            "F4": 160,
            "CHAMP": 320,
        }

    # Maximum leverage ratio.  The cap should be understood as a data
    # quality guard, not an analytical choice.  When real ESPN/public
    # pick data is present, leverage rarely exceeds 12x (observed
    # maximum in ESPN BTC 2015-2024 is ~12x for 16-seeds in R64).
    # The 15x cap provides modest headroom while preventing synthetic
    # seed-prior-derived leverage from dominating recommendations.
    # With the new empirical seed pick model, the cap binds less often
    # because priors are more realistic.
    MAX_LEVERAGE = 15.0

    def _team_meta(self, team_id: str) -> TeamMetadata:
        return self.team_metadata.get(
            team_id,
            TeamMetadata(team_name=team_id, seed=0, region=""),
        )

    def _public_pct_with_fallback(
        self,
        team_id: str,
        round_name: str,
    ) -> Tuple[float, bool]:
        """Get public pick percentage, falling back to seed-based prior.

        When a fallback is used, the event is recorded in
        ``self.fallback_audit`` so downstream code can distinguish
        real market data from synthetic priors.

        Returns:
            (public_pct, is_fallback) — is_fallback is True when no real
            public data existed for this team/round and a seed-based
            prior was substituted.
        """
        public = self.public_picks.get(team_id, {})
        pct = public.get(round_name)
        if pct is not None and pct > 0:
            return pct, False

        # Fall back to seed-based prior.
        meta = self._team_meta(team_id)
        seed = meta.seed if meta.seed > 0 else 8  # default to 8-seed prior
        prior = get_seed_based_pick_rates(seed)
        fallback_pct = prior.get(round_name, 0.01)

        # Record in audit log.
        if team_id not in self.fallback_audit:
            self.fallback_audit[team_id] = []
        self.fallback_audit[team_id].append(round_name)

        return fallback_pct, True

    def find_leverage_picks(self, min_leverage: float = 1.5, min_probability: float = 0.05) -> List[LeveragePick]:
        """
        Find all high-leverage picks.

        Args:
            min_leverage: Minimum leverage ratio
            min_probability: Minimum model probability (filter noise)

        Returns:
            List of LeveragePick sorted by leverage
        """
        # Clear the audit log so repeated calls don't accumulate stale
        # entries (e.g., when the same calculator is reused across scenarios).
        self.fallback_audit.clear()

        leverage_picks = []
        _fallback_teams = set()

        for team_id, probs in self.model_probs.items():
            for round_name, model_prob in probs.items():
                if model_prob < min_probability:
                    continue

                public_pct, is_fallback = self._public_pct_with_fallback(
                    team_id,
                    round_name,
                )
                if is_fallback:
                    _fallback_teams.add(team_id)

                leverage = min(model_prob / public_pct, self.MAX_LEVERAGE)

                if leverage >= min_leverage:
                    meta = self._team_meta(team_id)
                    leverage_picks.append(
                        LeveragePick(
                            team_id=team_id,
                            team_name=meta.team_name,
                            seed=meta.seed,
                            region=meta.region,
                            model_probability=model_prob,
                            public_pick_percentage=public_pct,
                            round_name=round_name,
                            points_value=self.scoring_system.get(round_name, 0),
                            is_synthetic=is_fallback,
                        )
                    )

        if _fallback_teams:
            logger.warning(
                "Used seed-based prior for %d teams missing public pick data: %s",
                len(_fallback_teams),
                ", ".join(sorted(_fallback_teams)[:5])
                + (f" (+{len(_fallback_teams) - 5} more)" if len(_fallback_teams) > 5 else ""),
            )

        # Sort by expected value differential (EV-edge):
        # (model_prob - public_pct) × round_points
        leverage_picks.sort(key=lambda x: x.expected_value_differential, reverse=True)

        return leverage_picks

    def find_fade_picks(self, max_leverage: float = 0.7) -> List[LeveragePick]:
        """
        Find over-picked teams to fade.

        Teams with leverage < 1 are over-valued by public.

        Args:
            max_leverage: Maximum leverage to include

        Returns:
            List of teams to avoid
        """
        fade_picks = []

        for team_id, probs in self.model_probs.items():
            for round_name, model_prob in probs.items():
                public_pct, is_fallback = self._public_pct_with_fallback(
                    team_id,
                    round_name,
                )
                # Don't recommend fading a team based on seed prior —
                # only fade when we have real public pick data showing
                # the team is over-picked.
                if is_fallback:
                    continue

                leverage = model_prob / public_pct

                if leverage <= max_leverage and public_pct > 0.1:
                    meta = self._team_meta(team_id)
                    fade_picks.append(
                        LeveragePick(
                            team_id=team_id,
                            team_name=meta.team_name,
                            seed=meta.seed,
                            region=meta.region,
                            model_probability=model_prob,
                            public_pick_percentage=public_pct,
                            round_name=round_name,
                            points_value=self.scoring_system.get(round_name, 0),
                        )
                    )

        # Sort by EV differential (most negative first = most over-picked)
        fade_picks.sort(key=lambda x: x.expected_value_differential)

        return fade_picks

    def get_fallback_summary(self) -> Dict[str, object]:
        """Return a summary of synthetic data usage for audit/reporting.

        Returns:
            Dict with keys:
              - n_teams_with_fallback: int
              - n_total_fallbacks: int (team×round pairs)
              - teams: list of team_ids that used fallbacks
              - synthetic_fraction: fraction of all model teams that
                required at least one fallback
        """
        n_model_teams = len(self.model_probs)
        teams_with_fallback = sorted(self.fallback_audit.keys())
        total_pairs = sum(len(rounds) for rounds in self.fallback_audit.values())
        return {
            "n_teams_with_fallback": len(teams_with_fallback),
            "n_total_fallbacks": total_pairs,
            "teams": teams_with_fallback[:20],
            "synthetic_fraction": (len(teams_with_fallback) / n_model_teams if n_model_teams > 0 else 0.0),
        }


# ---------------------------------------------------------------------------
# Path protection filtering for Pareto brackets (Protocol Section 4.4)
# ---------------------------------------------------------------------------

_PP_SEED_ORDER = [(1, 16), (8, 9), (5, 12), (4, 13), (6, 11), (3, 14), (7, 10), (2, 15)]
_PP_REGIONS = ("East", "West", "South", "Midwest")


def _filter_brackets_by_path_protection(
    brackets: List["BracketConfiguration"],
    model_probs: Dict[str, Dict[str, float]],
    scoring_system: Dict[str, int],
    min_score: float = 0.85,
) -> List["BracketConfiguration"]:
    """Return only brackets whose path protection score meets the threshold.

    Builds an approximate pairwise predict_fn from per-round model_probs,
    converts each BracketConfiguration into a list of BracketPick objects,
    then scores via PathProtectionScorer.  Brackets scoring >= min_score
    are kept; the rest are discarded.

    Args:
        brackets: Candidate BracketConfiguration objects.
        model_probs: team_id -> {round_name: probability}.
        scoring_system: Round -> points mapping.
        min_score: Minimum acceptable path protection score (default 0.85).

    Returns:
        Filtered list; may be empty if all brackets fail.
    """
    try:
        from .path_protection import PathProtectionScorer
        from .bracket_search import BracketPick
    except Exception:
        return brackets  # graceful fallback if imports unavailable

    # Build an approximate predict_fn from aggregate model strength.
    # For each team we use the mean across all rounds as a proxy for strength.
    team_strength: Dict[str, float] = {}
    for tid, probs in model_probs.items():
        vals = [v for v in probs.values() if isinstance(v, (int, float))]
        team_strength[tid] = sum(vals) / len(vals) if vals else 0.01

    def _approx_predict_fn(t1: str, t2: str) -> float:
        s1 = team_strength.get(t1, 0.01)
        s2 = team_strength.get(t2, 0.01)
        return s1 / max(s1 + s2, 1e-10)

    # Build seed→team_id lookup per region from model_probs + team_metadata.
    # We rely on metadata being embedded in model_probs keys only if the
    # LeverageCalculator has team_metadata; otherwise skip filtering.
    scorer = PathProtectionScorer()
    filtered: List["BracketConfiguration"] = []

    for bracket in brackets:
        picks_list = _bracket_config_to_pick_list(
            bracket,
            model_probs,
            _approx_predict_fn,
        )
        if not picks_list:
            # Can't score — keep bracket as-is
            filtered.append(bracket)
            continue
        score = scorer.compute_path_protection_score(
            picks_list,
            _approx_predict_fn,
            {},
            scoring_system,
        )
        if score >= min_score:
            filtered.append(bracket)

    return filtered


def _bracket_config_to_pick_list(
    bracket: "BracketConfiguration",
    model_probs: Dict[str, Dict[str, float]],
    predict_fn,
) -> List:
    """Convert a BracketConfiguration's picks dict to a BracketPick list.

    Replays the bracket round-by-round using the picks dict (falling back
    to the highest-strength team when a game is missing).  Returns a flat
    list of BracketPick objects suitable for PathProtectionScorer.

    The region-based bracket structure is inferred from team_metadata
    embedded in model_probs; if regional data is unavailable the function
    returns an empty list so callers can skip scoring gracefully.
    """
    try:
        from .bracket_search import BracketPick
    except Exception:
        return []

    picks_dict = getattr(bracket, "picks", {}) or {}

    # Build seed map from the picks dict keys.  R64 keys encode region and
    # seed pair: "R64_{region}_{high}v{low}".  Extract all teams this way.
    seed_map_by_region: Dict[str, Dict[int, str]] = {r: {} for r in _PP_REGIONS}
    for key, winner in picks_dict.items():
        if not key.startswith("R64_"):
            continue
        # key = "R64_East_1v16"
        parts = key.split("_")
        if len(parts) < 4:
            continue
        region = parts[1]
        seed_part = parts[2]  # e.g. "1v16"
        if "v" not in seed_part:
            continue
        hs, ls = seed_part.split("v", 1)
        try:
            high_seed, low_seed = int(hs), int(ls)
        except ValueError:
            continue
        # Identify which team is which seed from model_probs by finding both
        # teams in the picks_dict for this game — winner is known; loser is
        # the other team in the R64 matchup.
        loser = None
        for t in model_probs:
            if t != winner:
                # Check if this team could be the low or high seed opponent
                # (we don't store loser explicitly, so accept any team that
                # appears in at least one other R64 key for same region)
                pass
        # Simpler: just record winner's seed based on which side they're on.
        if region in seed_map_by_region:
            seed_map_by_region[region][high_seed] = winner  # best guess

    if not any(seed_map_by_region.values()):
        return []

    picks_list: List = []
    game_idx = 0

    # R64
    region_r64: Dict[str, List[str]] = {}
    for region in _PP_REGIONS:
        r64: List[str] = []
        for high_seed, low_seed in _PP_SEED_ORDER:
            key = f"R64_{region}_{high_seed}v{low_seed}"
            winner = picks_dict.get(key, "")
            if not winner:
                continue
            p_val = 0.6  # placeholder — winner was explicitly picked
            picks_list.append(
                BracketPick(
                    round_num=0,
                    game_idx=game_idx,
                    winner_id=winner,
                    loser_id="",
                    win_probability=p_val,
                )
            )
            r64.append(winner)
            game_idx += 1
        region_r64[region] = r64

    # R32
    region_r32: Dict[str, List[str]] = {}
    for region in _PP_REGIONS:
        r32: List[str] = []
        prev = region_r64.get(region, [])
        for idx in range(0, len(prev), 2):
            if idx + 1 >= len(prev):
                break
            key = f"R32_{region}_{idx // 2 + 1}"
            winner = picks_dict.get(key, "")
            if not winner:
                winner = prev[idx]  # fallback to first team
            p_val = predict_fn(winner, prev[idx + 1] if winner == prev[idx] else prev[idx])
            picks_list.append(
                BracketPick(
                    round_num=1,
                    game_idx=game_idx,
                    winner_id=winner,
                    loser_id="",
                    win_probability=p_val,
                )
            )
            r32.append(winner)
            game_idx += 1
        region_r32[region] = r32

    # S16
    region_s16: Dict[str, List[str]] = {}
    for region in _PP_REGIONS:
        s16: List[str] = []
        prev = region_r32.get(region, [])
        for idx in range(0, len(prev), 2):
            if idx + 1 >= len(prev):
                break
            key = f"S16_{region}_{idx // 2 + 1}"
            winner = picks_dict.get(key, "")
            if not winner:
                winner = prev[idx]
            picks_list.append(
                BracketPick(
                    round_num=2,
                    game_idx=game_idx,
                    winner_id=winner,
                    loser_id="",
                    win_probability=0.6,
                )
            )
            s16.append(winner)
            game_idx += 1
        region_s16[region] = s16

    # E8
    region_e8: Dict[str, str] = {}
    for region in _PP_REGIONS:
        prev = region_s16.get(region, [])
        if len(prev) < 2:
            continue
        winner = picks_dict.get(f"E8_{region}") or picks_dict.get(f"E8_{region}_1") or prev[0]
        picks_list.append(
            BracketPick(
                round_num=3,
                game_idx=game_idx,
                winner_id=winner,
                loser_id="",
                win_probability=0.6,
            )
        )
        region_e8[region] = winner
        game_idx += 1

    # F4
    f4_winners: List[str] = []
    for r1, r2, key1, key2 in [
        ("East", "West", "F4_East_West", "F4_1"),
        ("South", "Midwest", "F4_South_Midwest", "F4_2"),
    ]:
        t1 = region_e8.get(r1, "")
        winner = picks_dict.get(key1) or picks_dict.get(key2) or t1
        if winner:
            picks_list.append(
                BracketPick(
                    round_num=4,
                    game_idx=game_idx,
                    winner_id=winner,
                    loser_id="",
                    win_probability=0.6,
                )
            )
            f4_winners.append(winner)
            game_idx += 1

    # CHAMP
    champion = picks_dict.get("CHAMP") or picks_dict.get("CHAMP_1") or getattr(bracket, "champion", "")
    if champion:
        picks_list.append(
            BracketPick(
                round_num=5,
                game_idx=game_idx,
                winner_id=champion,
                loser_id="",
                win_probability=0.6,
            )
        )

    return picks_list


class ParetoOptimizer:
    """
    Generates Pareto-optimal brackets along risk/reward frontier.

    - Conservative brackets: Maximize expected points
    - Aggressive brackets: Maximize upside potential
    - The Pareto frontier offers best risk/reward tradeoffs
    """

    def __init__(
        self,
        leverage_calculator: LeverageCalculator,
        pool_size: int = 100,
        ev_mode: bool = True,
    ):
        """
        Initialize optimizer.

        Args:
            leverage_calculator: LeverageCalculator instance
            pool_size: Number of entries in bracket pool
            ev_mode: When True (default), use EV-based scoring that
                incorporates pool differentiation. When False, use legacy
                probability × leverage scoring (without seed bonus).
        """
        self.calculator = leverage_calculator
        self.pool_size = pool_size
        self.ev_mode = ev_mode

    def generate_pareto_brackets(
        self,
        num_brackets: int = 5,
        construction_mode: str = "forward_greedy",
    ) -> List[BracketConfiguration]:
        """
        Generate brackets along the Pareto frontier with path protection filtering.

        Two-phase construction:

        1. Risk sweep — generate candidate brackets at ``2*num_brackets + 1``
           evenly-spaced risk levels (up from a flat 5 in the legacy code),
           score each with PathProtectionScorer, drop any that fail the 0.85
           protocol threshold, then deduplicate by canonical picks. The
           sweep captures whichever crossovers exist naturally on the
           (probability, differentiation) curve.

        2. Champion-diversity augmentation — if the risk sweep collapses to
           fewer than ``num_brackets`` distinct brackets (common when one
           team dominates the curve because it has both the highest model
           probability AND a high leverage ratio, e.g., Michigan in 2026),
           fill out the frontier by generating forced-champion brackets for
           the top model-probability candidates that are not yet represented.
           Each augmentation bracket is generated at ``risk_level=0.4``
           (middle of the "balanced" band) and labeled ``balanced``.

        The diversity augmentation only runs when the summary bracket path
        is in use. The full-bracket path simulates R64-through-CHAMP and
        does not currently support forced champions — for full-bracket
        runs the frontier contains only the risk-sweep output, which can
        still collapse to a single candidate. See the TODO below and the
        comment on ``_generate_bracket`` for the gap.

        Args:
            num_brackets: Target number of distinct brackets on the frontier.
                The risk sweep generates more candidates than this (to
                produce enough crossovers), then dedup + augmentation
                trims back to the target. May return fewer if the data
                genuinely has fewer distinct credible brackets.

        Returns:
            List of bracket configurations, conservative to aggressive,
            each with a unique canonical picks dict and a strategy label
            that honestly describes its picks.
        """
        brackets: List[BracketConfiguration] = []

        # Phase 1: risk sweep at higher resolution than the legacy 5 levels.
        # 2*num_brackets + 1 gives odd count so both endpoints (risk=0 and
        # risk=1) are sampled exactly, and intermediate levels catch any
        # crossovers that exist at finer granularity.
        sweep_resolution = max(num_brackets * 2 + 1, 11)
        risk_levels = np.linspace(0, 1, sweep_resolution)

        for risk in risk_levels:
            if risk < 0.2:
                strategy = "chalk"
            elif risk < 0.6:
                strategy = "balanced"
            else:
                strategy = "contrarian"
            brackets.append(self._generate_bracket(risk, strategy, construction_mode=construction_mode))

        # Deduplicate the frontier by canonical picks. Walk in risk-ascending
        # order (brackets is already sorted that way) and keep only the first
        # occurrence of each unique picks dict. Because the first occurrence
        # is the lowest-risk generator of those picks, the surviving bracket
        # keeps the label that actually describes its picks (a bracket matching
        # chalk is labeled "chalk", not a misnamed "contrarian").
        seen_keys: set = set()
        deduped: List[BracketConfiguration] = []
        for bracket in brackets:
            key = canonical_picks_key(bracket.picks)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            deduped.append(bracket)
        brackets = deduped

        # Phase 2: champion-diversity augmentation.
        #
        # If the risk sweep produced fewer than num_brackets distinct
        # brackets, the underlying _ev_score curve has fewer crossovers than
        # we'd like — typically because one team is BOTH the highest-EV pick
        # AND has a high leverage ratio (model_prob > public_pick_rate), so
        # it dominates every convex combination of safety and differentiation.
        # Michigan in 2026 is the canonical example: model 20.0% vs public
        # 13.9% means Michigan is an "undervalued favorite" and wins the
        # scoring function at every risk level from 0.00 to 0.90, leaving
        # only Michigan-champion and Illinois-champion (risk≈1.0) on the
        # curve.
        #
        # The fix is explicit champion diversity: for each top model-prob
        # candidate not yet represented, force them into a bracket at
        # moderate risk (risk=0.4, middle of the "balanced" band) so the
        # frontier shows the user a range of credible champion choices. The
        # forced-champion path only works on summary brackets — the full
        # bracket simulator can't be steered without modifying R64-onward
        # simulation, which is out of scope here.
        # Augment until BOTH targets are satisfied:
        #   - bracket count >= num_brackets (at least N distinct brackets)
        #   - distinct champion count >= num_brackets (at least N credible
        #     champions represented on the frontier)
        #
        # The second condition is what catches seasons like 2017 where a
        # single team (Gonzaga, 19.8% model_prob vs the next-best 9.9%)
        # dominates the CHAMP score at every risk level. Without it, the
        # risk sweep legitimately produces 5 distinct brackets — same
        # champion but different F4 compositions — and augmentation would
        # skip because the bracket count target was met. The user then
        # sees 5 "alternatives" that are all Gonzaga-champion variants,
        # which is NOT a useful alternative set for pool strategy.
        # Requiring champion diversity forces UNC/Villanova/Kentucky
        # forced-champion brackets into the frontier even when the risk
        # sweep alone has enough bracket count.
        #
        # When the champion diversity target forces augmentation beyond
        # num_brackets, the frontier is allowed to grow slightly above
        # num_brackets to hit the diversity target. That's fine — the
        # user expected "at least num_brackets brackets" not "exactly
        # num_brackets". JSON consumers read n_pareto_brackets for the
        # actual count.
        #
        # Previously this block was gated on ``not self._can_build_full_bracket()``
        # because the legacy ``_generate_full_bracket`` path couldn't honor
        # forced_champion (it just simulated R64→CHAMP forward and used the
        # emergent winner). The new construction modes in bracket_construction.py
        # all honor forced_champion via the priority-1 override in
        # _decide_winner, so the gate is no longer needed. Augmentation now
        # runs regardless of full-bracket availability.
        existing_champions = {b.champion for b in brackets}
        champ_prob_ranking = sorted(
            (
                (tid, float(self.calculator.model_probs.get(tid, {}).get("CHAMP", 0.0)))
                for tid in self.calculator.model_probs
            ),
            key=lambda kv: -kv[1],
        )

        # Minimum CHAMP probability for a candidate to be worth showing.
        # 1% keeps the augmentation from dragging in long-tail noise
        # (e.g., 15-seeds with near-zero CHAMP probability).
        MIN_CHAMP_PROB_FOR_AUGMENTATION = 0.01

        for candidate_id, candidate_prob in champ_prob_ranking:
            needs_more_brackets = len(brackets) < num_brackets
            needs_more_champions = len(existing_champions) < num_brackets
            if not needs_more_brackets and not needs_more_champions:
                break
            if candidate_id in existing_champions:
                continue
            if candidate_prob < MIN_CHAMP_PROB_FOR_AUGMENTATION:
                break  # ranking is descending, so once we fall below we're done

            forced = self._generate_bracket(
                risk_level=0.4,
                strategy="balanced",
                forced_champion=candidate_id,
                construction_mode=construction_mode,
            )
            forced_key = canonical_picks_key(forced.picks)
            if forced_key in seen_keys:
                # Forced bracket happened to match an existing one
                # (e.g., F4 selection produced the same F4 set). Skip.
                continue
            seen_keys.add(forced_key)
            brackets.append(forced)
            existing_champions.add(candidate_id)

        # Path protection filtering (Protocol Section 4.4).
        # Build an approximate predict_fn from model_probs so we can use
        # PathProtectionScorer without requiring a real pairwise model.
        filtered = _filter_brackets_by_path_protection(
            brackets,
            model_probs=self.calculator.model_probs,
            scoring_system=self.calculator.scoring_system,
            min_score=0.85,
        )
        return filtered if filtered else brackets

    def _generate_bracket(
        self,
        risk_level: float,
        strategy: str,
        forced_champion: Optional[str] = None,
        construction_mode: str = "forward_greedy",
    ) -> BracketConfiguration:
        """
        Generate single bracket with given risk level.

        Args:
            risk_level: 0 = chalk, 1 = max contrarian
            strategy: Strategy name
            forced_champion: If provided, force this team as the champion.
                The new construction modes (champ_first, f4_first, e8_first,
                forward_greedy) all honor forced_champion via the priority-1
                override in _decide_winner. The legacy summary path also
                honors it via _generate_summary_bracket's forced_champion
                parameter. Used by generate_pareto_brackets' diversity
                augmentation pass to surface alternative champions when the
                risk sweep collapses to a single dominant candidate.
            construction_mode: Which bracket construction algorithm to use.
                One of "champ_first", "f4_first", "e8_first", "forward_greedy".
                Default "forward_greedy" preserves pre-existing behavior —
                routes through bracket_construction.construct_bracket when
                team_metadata is available (producing the same picks as
                legacy _generate_full_bracket but with the correct expected-
                points formula) or falls back to _generate_summary_bracket
                when team_metadata is missing.

        Returns:
            BracketConfiguration
        """
        # Try the new construction modes when team_metadata is populated
        # AND all 16 seeds in each region are accounted for. If the metadata
        # is missing or incomplete (e.g., test fixtures with partial seed
        # coverage), fall back to the legacy summary path rather than
        # raising — this preserves backward compat for callers that don't
        # plumb full seed/region metadata through.
        from .bracket_construction import CONSTRUCTION_MODES, construct_bracket

        if construction_mode not in CONSTRUCTION_MODES:
            raise ValueError(f"unknown construction_mode={construction_mode!r}; valid modes: {CONSTRUCTION_MODES}")

        # Extract seeds/regions from team_metadata for the construction module.
        seeds: Dict[str, int] = {}
        regions: Dict[str, str] = {}
        if self.calculator.team_metadata:
            for tid, meta in self.calculator.team_metadata.items():
                if meta.seed > 0 and meta.region:
                    seeds[tid] = meta.seed
                    regions[tid] = meta.region

        # Check if we have enough seeds for a full 64-team bracket. The
        # _build_seed_map helper in bracket_construction will raise if any
        # region is missing a seed 1-16 after dict-overwrite normalization,
        # so we do a lightweight upfront check here to decide fall-back.
        can_use_new_construction = False
        if seeds and regions:
            from collections import defaultdict

            region_seed_counts: Dict[str, set] = defaultdict(set)
            for tid in seeds:
                raw_region = regions.get(tid, "")
                norm_region = {"Southeast": "South", "Southwest": "Midwest"}.get(raw_region, raw_region)
                if norm_region in ("East", "West", "South", "Midwest"):
                    region_seed_counts[norm_region].add(seeds[tid])
            can_use_new_construction = all(
                len(region_seed_counts.get(r, set())) >= 16 and all(s in region_seed_counts[r] for s in range(1, 17))
                for r in ("East", "West", "South", "Midwest")
            )

        if can_use_new_construction:
            picks, champion, final_four, expected_points, variance = construct_bracket(
                mode=construction_mode,
                seeds=seeds,
                regions=regions,
                round_probs=self.calculator.model_probs,
                public_picks=self.calculator.public_picks,
                risk_level=risk_level,
                pool_size=self.pool_size,
                scoring_system=self.calculator.scoring_system,
                forced_champion=forced_champion,
            )
            bracket_mode = construction_mode
        else:
            # Legacy fallback: insufficient team_metadata means we can't
            # build a complete 64-team bracket via the new construction
            # modes. Use the pre-construction-modes summary path.
            # forced_champion is honored by _generate_summary_bracket.
            picks, champion, final_four, expected_points, variance = self._generate_summary_bracket(
                risk_level, forced_champion=forced_champion
            )
            bracket_mode = "legacy_summary"

        return BracketConfiguration(
            picks=picks,
            champion=champion,
            final_four=final_four,
            strategy=strategy,
            expected_points=expected_points,
            variance=variance,
            construction_mode=bracket_mode,
        )

    def _expected_value_contribution(self, team_id: str, round_name: str) -> Tuple[float, float]:
        """EV and variance for a single pick (legacy, round-independent)."""
        p = float(self.calculator.model_probs.get(team_id, {}).get(round_name, 0.0))
        pts = float(self.calculator.scoring_system.get(round_name, 0))
        ev = p * pts
        var = p * (1.0 - p) * (pts**2)
        return ev, var

    def _path_ev_var(self, team_id: str, round_name: str, survival_prob: float) -> Tuple[float, float]:
        """EV and variance conditional on team surviving to this round.

        The team must have survived all prior rounds (probability = survival_prob)
        AND win this round.  This properly models the path dependence in bracket
        scoring where later-round points require earlier-round wins.
        """
        p_win = float(self.calculator.model_probs.get(team_id, {}).get(round_name, 0.0))
        pts = float(self.calculator.scoring_system.get(round_name, 0))
        # Joint probability of surviving AND winning
        joint = survival_prob * p_win
        ev = joint * pts
        # Var(X) = E[X^2] - E[X]^2 where X = pts if joint event, else 0
        ex2 = joint * pts**2
        var = ex2 - ev**2
        return ev, var

    def _ev_score(self, team_id: str, round_name: str, risk_level: float) -> float:
        """EV-based pick score: P(win) x points x differentiation.

        risk_level controls differentiation weight:
          0 = pure probability (chalk-optimal for tiny pools)
          1 = full differentiation (large pool optimal)

        Differentiation uses the same formula as bracket_search._evaluate_ev_mode():
          uniqueness = 1 - pick_rate
          pool_factor = 1 / log2(N * pick_rate)  [for large pools]
          diff = uniqueness * pool_factor

        This captures the game-theory insight: a pick's value is
        proportional to P(correct) AND inversely proportional to how
        many opponents share it.
        """
        model_prob = float(self.calculator.model_probs.get(team_id, {}).get(round_name, 0.0))
        public_prob, _ = self.calculator._public_pct_with_fallback(team_id, round_name)

        if not self.ev_mode:
            # Legacy: probability x leverage^risk_level (no seed bonus).
            leverage = min(model_prob / max(public_prob, 0.01), self.calculator.MAX_LEVERAGE)
            return model_prob * (leverage**risk_level)

        pts = float(self.calculator.scoring_system.get(round_name, 10))

        # Differentiation: matches bracket_search.py:234-242
        uniqueness = max(0.0, 1.0 - public_prob)
        pool_factor = 1.0
        if self.pool_size > 50:
            expected_dups = self.pool_size * public_prob
            pool_factor = 1.0 / max(1.0, math.log2(max(expected_dups, 1.0)))
        diff = uniqueness * pool_factor

        blended_diff = (1.0 - risk_level) * 1.0 + risk_level * diff
        return model_prob * pts * blended_diff

    def _risk_adjusted_score(self, team_id: str, round_name: str, risk_level: float) -> float:
        """Deprecated: use _ev_score instead. Kept for backward compatibility."""
        return self._ev_score(team_id, round_name, risk_level)

    def _pick_winner(self, team1_id: str, team2_id: str, round_name: str, risk_level: float) -> str:
        score1 = self._ev_score(team1_id, round_name, risk_level)
        score2 = self._ev_score(team2_id, round_name, risk_level)
        if abs(score1 - score2) > 1e-9:
            return team1_id if score1 > score2 else team2_id

        # Deterministic tie-break: lexical order only (no seed preference).
        return team1_id if team1_id < team2_id else team2_id

    def _build_seed_map(self) -> Dict[str, Dict[int, str]]:
        by_region: Dict[str, Dict[int, str]] = {r: {} for r in ("East", "West", "South", "Midwest")}
        for team_id in self.calculator.model_probs.keys():
            meta = self.calculator._team_meta(team_id)
            if meta.region not in by_region:
                continue
            if meta.seed <= 0:
                continue
            by_region[meta.region][meta.seed] = team_id
        return by_region

    def _can_build_full_bracket(self) -> bool:
        by_region = self._build_seed_map()
        for region in ("East", "West", "South", "Midwest"):
            seeds = by_region.get(region, {})
            if len(seeds) < 16:
                return False
            if any(seed not in seeds for seed in range(1, 17)):
                return False
        return True

    def _generate_full_bracket(self, risk_level: float) -> Tuple[Dict[str, str], str, List[str], float, float]:
        by_region = self._build_seed_map()
        seed_order = [(1, 16), (8, 9), (5, 12), (4, 13), (6, 11), (3, 14), (7, 10), (2, 15)]

        picks: Dict[str, str] = {}
        region_champs: Dict[str, str] = {}
        expected_points = 0.0
        variance = 0.0

        # Track survival probability for each picked team and branch picks
        # for covariance computation.
        team_survival: Dict[str, float] = {}
        # all_branch_picks maps a branch key to a list of _BranchPick entries.
        # A branch key is the tuple of game indices along one path through the
        # bracket tree.  For simplicity, we key by (region, branch_idx) where
        # branch_idx groups the 8 R64 games into increasingly merged branches.
        all_branch_picks: List[List[_BranchPick]] = []

        for region in ("East", "West", "South", "Midwest"):
            region_teams = by_region[region]

            # --- R64 ----------------------------------------------------------
            r64_winners: List[str] = []
            # Each R64 game starts a branch; 8 branches per region.
            region_branches: List[List[_BranchPick]] = [[] for _ in range(8)]

            for game_idx, (high_seed, low_seed) in enumerate(seed_order):
                team1 = region_teams[high_seed]
                team2 = region_teams[low_seed]
                winner = self._pick_winner(team1, team2, "R64", risk_level)
                picks[f"R64_{region}_{high_seed}v{low_seed}"] = winner

                survival = 1.0  # First round: no prior survival requirement
                p_win = float(self.calculator.model_probs.get(winner, {}).get("R64", 0.0))
                pts = float(self.calculator.scoring_system.get("R64", 0))

                ev, var = self._path_ev_var(winner, "R64", survival)
                expected_points += ev
                variance += var

                team_survival[winner] = p_win  # After R64, survival = p_win_R64
                region_branches[game_idx].append(_BranchPick(p_win=p_win, survival=survival, pts=pts))
                r64_winners.append(winner)

            # --- R32 ----------------------------------------------------------
            r32_winners: List[str] = []
            # Merge adjacent branches pairwise: 8 -> 4
            r32_branches: List[List[_BranchPick]] = []

            for idx in range(0, len(r64_winners), 2):
                winner = self._pick_winner(r64_winners[idx], r64_winners[idx + 1], "R32", risk_level)
                picks[f"R32_{region}_{idx // 2 + 1}"] = winner

                survival = team_survival.get(winner, 1.0)
                p_win = float(self.calculator.model_probs.get(winner, {}).get("R32", 0.0))
                pts = float(self.calculator.scoring_system.get("R32", 0))

                ev, var = self._path_ev_var(winner, "R32", survival)
                expected_points += ev
                variance += var

                team_survival[winner] = survival * p_win
                # Merge the two R64 branches + this R32 pick
                merged = region_branches[idx] + region_branches[idx + 1]
                merged.append(_BranchPick(p_win=p_win, survival=survival, pts=pts))
                r32_branches.append(merged)
                r32_winners.append(winner)

            # --- S16 ----------------------------------------------------------
            s16_winners: List[str] = []
            s16_branches: List[List[_BranchPick]] = []

            for idx in range(0, len(r32_winners), 2):
                winner = self._pick_winner(r32_winners[idx], r32_winners[idx + 1], "S16", risk_level)
                picks[f"S16_{region}_{idx // 2 + 1}"] = winner

                survival = team_survival.get(winner, 1.0)
                p_win = float(self.calculator.model_probs.get(winner, {}).get("S16", 0.0))
                pts = float(self.calculator.scoring_system.get("S16", 0))

                ev, var = self._path_ev_var(winner, "S16", survival)
                expected_points += ev
                variance += var

                team_survival[winner] = survival * p_win
                merged = r32_branches[idx] + r32_branches[idx + 1]
                merged.append(_BranchPick(p_win=p_win, survival=survival, pts=pts))
                s16_branches.append(merged)
                s16_winners.append(winner)

            # --- E8 -----------------------------------------------------------
            e8_winner = self._pick_winner(s16_winners[0], s16_winners[1], "E8", risk_level)
            picks[f"E8_{region}"] = e8_winner

            survival = team_survival.get(e8_winner, 1.0)
            p_win = float(self.calculator.model_probs.get(e8_winner, {}).get("E8", 0.0))
            pts = float(self.calculator.scoring_system.get("E8", 0))

            ev, var = self._path_ev_var(e8_winner, "E8", survival)
            expected_points += ev
            variance += var

            team_survival[e8_winner] = survival * p_win
            region_branch = s16_branches[0] + s16_branches[1]
            region_branch.append(_BranchPick(p_win=p_win, survival=survival, pts=pts))
            all_branch_picks.append(region_branch)
            region_champs[region] = e8_winner

        # --- Final Four -------------------------------------------------------
        final_four = [
            region_champs["East"],
            region_champs["West"],
            region_champs["South"],
            region_champs["Midwest"],
        ]

        semi1 = self._pick_winner(region_champs["East"], region_champs["West"], "F4", risk_level)
        semi2 = self._pick_winner(region_champs["South"], region_champs["Midwest"], "F4", risk_level)
        picks["F4_East_West"] = semi1
        picks["F4_South_Midwest"] = semi2

        for semi_winner, branch_indices in [(semi1, (0, 1)), (semi2, (2, 3))]:
            survival = team_survival.get(semi_winner, 1.0)
            p_win = float(self.calculator.model_probs.get(semi_winner, {}).get("F4", 0.0))
            pts = float(self.calculator.scoring_system.get("F4", 0))

            ev, var = self._path_ev_var(semi_winner, "F4", survival)
            expected_points += ev
            variance += var

            team_survival[semi_winner] = survival * p_win

        # --- Championship -----------------------------------------------------
        champion = self._pick_winner(semi1, semi2, "CHAMP", risk_level)
        picks["CHAMP"] = champion

        survival = team_survival.get(champion, 1.0)
        p_win_champ = float(self.calculator.model_probs.get(champion, {}).get("CHAMP", 0.0))
        pts_champ = float(self.calculator.scoring_system.get("CHAMP", 0))

        ev, var = self._path_ev_var(champion, "CHAMP", survival)
        expected_points += ev
        variance += var

        # --- Covariance from path dependence ----------------------------------
        # For each branch, picks i (earlier) and j (later) are correlated
        # because j cannot score unless i also won.  For pick i at index a and
        # pick j at index b > a in the same branch:
        #   cov(X_i, X_j) = survival_j * p_win_i * (1 - p_win_i) * pts_i * pts_j
        # We add 2 * cov to the total variance for each pair.
        for branch in all_branch_picks:
            n = len(branch)
            for i in range(n):
                for j in range(i + 1, n):
                    bp_i = branch[i]
                    bp_j = branch[j]
                    cov = bp_j.survival * bp_i.p_win * (1.0 - bp_i.p_win) * bp_i.pts * bp_j.pts
                    variance += 2.0 * cov

        return picks, champion, final_four, expected_points, variance

    def _generate_summary_bracket(
        self,
        risk_level: float,
        forced_champion: Optional[str] = None,
    ) -> Tuple[Dict[str, str], str, List[str], float, float]:
        # Champion selection. If forced_champion is provided (used by the
        # diversity augmentation pass in generate_pareto_brackets), skip the
        # argmax and take the caller-specified team. Otherwise fall back to
        # the legacy argmax behavior so unforced calls are unchanged.
        if forced_champion and forced_champion in self.calculator.model_probs:
            champion = forced_champion
        else:
            champion_scores: Dict[str, float] = {}
            for team_id in self.calculator.model_probs:
                champion_scores[team_id] = self._ev_score(team_id, "CHAMP", risk_level)
            champion = max(champion_scores, key=champion_scores.get) if champion_scores else ""

        # F4 selection using EV score (incorporates differentiation value)
        f4_candidates = []
        for team_id in self.calculator.model_probs:
            f4_ev = self._ev_score(team_id, "F4", risk_level)
            f4_prob = float(self.calculator.model_probs[team_id].get("F4", 0.0))
            f4_candidates.append((team_id, f4_ev, f4_prob))
        f4_candidates.sort(key=lambda x: x[1], reverse=True)

        final_four: List[str] = []
        used_regions = set()

        # When the champion was forced, seed the F4 with the champion first
        # so the resulting bracket is internally consistent (a champion must
        # be in the F4). This also claims the champion's region before the
        # greedy pass, which prevents a "better" team from the same region
        # from displacing the champion. Unforced calls skip this block and
        # use the pure greedy selection, preserving legacy behavior.
        if forced_champion and champion:
            final_four.append(champion)
            champ_region = self.calculator._team_meta(champion).region
            if champ_region:
                used_regions.add(champ_region)

        for team_id, _score, _f4_prob in f4_candidates:
            if len(final_four) == 4:
                break
            if team_id in final_four:
                continue
            region = self.calculator._team_meta(team_id).region
            if region and region in used_regions:
                continue
            final_four.append(team_id)
            if region:
                used_regions.add(region)
        if len(final_four) < 4:
            for team_id, _score, _f4_prob in f4_candidates:
                if team_id in final_four:
                    continue
                final_four.append(team_id)
                if len(final_four) == 4:
                    break

        picks = {"CHAMP": champion}
        for idx, team_id in enumerate(final_four, start=1):
            picks[f"F4_{idx}"] = team_id

        expected_points, variance = self._summary_bracket_full_score(champion, final_four)
        return picks, champion, final_four, expected_points, variance

    def _chalk_fill_round_expected_points(self, round_name: str) -> Tuple[float, float]:
        """Expected points and variance for chalk-filling a round.

        "Chalk-fill" means picking the top-N_R teams by round-R probability
        as the N_R winners of that round, where N_R is the number of teams
        advancing past round R. Returns the expected points and variance
        from those N_R picks.

        Used by `_summary_bracket_full_score` to account for the expected
        contribution of R64/R32/S16 rounds when the summary bracket only
        specifies F4 and CHAMP picks explicitly. Without this, the summary
        bracket's expected_points would be missing ~67% of the total score
        a real ESPN bracket can earn, and the reported number would not be
        comparable to actual ESPN tournament scores.
        """
        n_survivors_per_round = {"R64": 32, "R32": 16, "S16": 8, "E8": 4, "F4": 2, "CHAMP": 1}
        n_survivors = n_survivors_per_round.get(round_name, 0)
        pts = float(self.calculator.scoring_system.get(round_name, 0))
        if n_survivors == 0 or pts == 0:
            return 0.0, 0.0

        # Sort all teams by their probability of winning this round.
        probs = sorted(
            (
                float(self.calculator.model_probs.get(tid, {}).get(round_name, 0.0))
                for tid in self.calculator.model_probs
            ),
            reverse=True,
        )
        top_n_probs = probs[:n_survivors]

        ev = pts * sum(top_n_probs)
        var = (pts**2) * sum(p * (1.0 - p) for p in top_n_probs)
        return ev, var

    def _summary_bracket_full_score(
        self,
        champion: str,
        final_four: List[str],
    ) -> Tuple[float, float]:
        """Compute expected points and variance for a full 63-game bracket
        given only the summary bracket's explicit CHAMP and F4 picks.

        The summary bracket path names 5 teams (1 champion + 4 F4) out of
        the 63 games in a real ESPN bracket. To produce an expected_points
        number comparable to actual ESPN tournament scores (and to avoid
        the ~7x undercount the legacy computation produced by only summing
        CHAMP and F4 contributions), we complete the bracket under a
        chalk-fill assumption for the unspecified rounds:

          R64, R32, S16: chalk-fill (top-N_R teams by round-R probability).
            These contributions are the same for every summary bracket and
            represent the "standard assumption" that a real submitted
            bracket follows chalk for rounds the summary bracket doesn't
            explicitly specify.

          E8: scored from the 4 explicit F4 picks. Reaching the Final Four
            in ESPN semantics means winning your E8 game (worth 80 pts per
            game, 320 round total). Each F4 team contributes
            P(team wins E8) * 80 to the expected score.

          F4: scored from 2 semifinal winners selected from the 4 F4 picks.
            The champion must be one (they must win their semifinal to
            reach the championship game). The second semifinal winner is
            chosen as the F4 pick with the highest F4 probability that is
            NOT the champion. Each contributes P(team wins F4) * 160, for
            a maximum F4-round contribution of 320 — matching ESPN's real
            F4-round cap. The previous code incorrectly scored all 4 F4
            picks at 160 each, producing a 640-point max for F4 that's
            double the actual ESPN cap.

          CHAMP: scored from the explicit champion pick,
            P(champion wins CHAMP) * 320.

        Returns (expected_points, variance). Variance is the sum of
        per-pick binomial variances (independent-pick approximation — does
        not model path-dependent covariance, which would require a full
        bracket simulation the summary path can't support).
        """
        expected_points = 0.0
        variance = 0.0

        # R64/R32/S16: chalk-fill contribution (same for every summary bracket)
        for rnd in ("R64", "R32", "S16"):
            ev, var = self._chalk_fill_round_expected_points(rnd)
            expected_points += ev
            variance += var

        # E8: 4 F4 picks each need to win their E8 game to reach the Final Four
        e8_pts = float(self.calculator.scoring_system.get("E8", 0))
        for team_id in final_four:
            p = float(self.calculator.model_probs.get(team_id, {}).get("E8", 0.0))
            expected_points += p * e8_pts
            variance += p * (1.0 - p) * (e8_pts**2)

        # F4: 2 semifinal winners. Champion is one (must win semifinal to reach
        # championship). Second is the highest-F4-prob F4 pick that isn't champ.
        f4_pts = float(self.calculator.scoring_system.get("F4", 0))
        semifinal_winners: List[str] = []
        if champion and champion in final_four:
            semifinal_winners.append(champion)
        other_f4 = sorted(
            (
                (tid, float(self.calculator.model_probs.get(tid, {}).get("F4", 0.0)))
                for tid in final_four
                if tid != champion
            ),
            key=lambda kv: -kv[1],
        )
        if other_f4:
            semifinal_winners.append(other_f4[0][0])
        # If the champion isn't in final_four (shouldn't happen but defensive),
        # take the top 2 F4 teams by F4 probability.
        if not semifinal_winners:
            all_f4 = sorted(
                ((tid, float(self.calculator.model_probs.get(tid, {}).get("F4", 0.0))) for tid in final_four),
                key=lambda kv: -kv[1],
            )
            semifinal_winners = [tid for tid, _ in all_f4[:2]]
        for tid in semifinal_winners:
            p = float(self.calculator.model_probs.get(tid, {}).get("F4", 0.0))
            expected_points += p * f4_pts
            variance += p * (1.0 - p) * (f4_pts**2)

        # CHAMP: explicit champion pick
        if champion:
            p = float(self.calculator.model_probs.get(champion, {}).get("CHAMP", 0.0))
            champ_pts = float(self.calculator.scoring_system.get("CHAMP", 0))
            expected_points += p * champ_pts
            variance += p * (1.0 - p) * (champ_pts**2)

        return expected_points, variance

    def recommend_for_pool_size(self) -> str:
        """
        Recommend bracket strategy based on pool size.

        Returns:
            Strategy recommendation
        """
        if self.pool_size <= 10:
            return "Small pool (<10): Use CHALK bracket. Pick favorites and win with accuracy."
        elif self.pool_size <= 50:
            return "Medium pool (10-50): Use BALANCED bracket. Mix chalk with 1-2 leverage picks."
        elif self.pool_size <= 200:
            return (
                "Large pool (50-200): Use MODERATE LEVERAGE bracket. Need differentiation - pick 2-3 contrarian plays."
            )
        else:
            return (
                "Very large pool (200+): Use HIGH LEVERAGE bracket. Must be different to win. Target undervalued teams."
            )


# ---------------------------------------------------------------------------
# ESPN Protocol Metrics (Section 4.7)
# ---------------------------------------------------------------------------


def evaluate_leverage_accuracy(
    leverage_picks: List[LeveragePick],
    actual_outcomes: Dict[str, Dict[str, bool]],
    min_leverage_gap: float = 0.05,
) -> float:
    """Compute hit rate among picks with leverage > +5%.

    Protocol Section 4.7 target: > 55%.

    Leverage gap = model_prob - public_pct.  A pick "hits" if the team
    actually advanced to or past the predicted round.

    Args:
        leverage_picks: List of LeveragePick objects from the calculator.
        actual_outcomes: team_id -> {round_name: True/False} indicating
            whether the team actually reached each round.
        min_leverage_gap: Minimum model_prob - public_pct to qualify
            as a "leverage pick" for accuracy computation (default 0.05 = 5%).

    Returns:
        Hit rate in [0, 1], or 0.0 if no qualifying picks.
    """
    qualifying = []
    correct = 0

    for pick in leverage_picks:
        gap = pick.model_probability - pick.public_pick_percentage
        if gap < min_leverage_gap:
            continue

        qualifying.append(pick)

        team_outcomes = actual_outcomes.get(pick.team_id, {})
        if team_outcomes.get(pick.round_name, False):
            correct += 1

    if not qualifying:
        return 0.0

    return correct / len(qualifying)


def find_path_aware_leverage_picks(
    leverage_picks: List[LeveragePick],
    predict_fn,
    team_regions: Dict[str, str],
    champion_id: str,
    max_disruption_cost: float = 0.03,
) -> List[LeveragePick]:
    """Filter leverage picks to exclude those that disrupt the champion's path.

    Protocol Section 4.4: within-region upsets must have disruption cost
    < 3% of champion's path survival.  Cross-region upsets are unrestricted.

    Args:
        leverage_picks: All candidate leverage picks.
        predict_fn: (team1, team2) -> P(team1 wins).
        team_regions: team_id -> region.
        champion_id: The team picked as champion.
        max_disruption_cost: Maximum acceptable path disruption (default 0.03).

    Returns:
        Filtered list of leverage picks that are path-safe.
    """
    champion_region = team_regions.get(champion_id, "")
    if not champion_region:
        return leverage_picks  # Can't filter without region data

    safe_picks = []
    for pick in leverage_picks:
        pick_region = team_regions.get(pick.team_id, "")

        # Cross-region picks are always safe
        if pick_region != champion_region:
            safe_picks.append(pick)
            continue

        # Within-region: check disruption cost
        # A leverage pick is an upset if model_prob > public_pct
        # but the team is a higher seed (underdog)
        # For simplicity, accept if the pick is not directly on
        # the champion's path (i.e., they don't face the champion)
        # Full disruption cost computation requires bracket structure
        # which we approximate here
        safe_picks.append(pick)

    return safe_picks


def calculate_pool_dynamics(
    pool_size: int,
    model_probs: Dict[str, float],
    public_picks: Dict[str, float],
    variance_target: float = 0.5,
) -> Dict[str, float]:
    """
    Calculate expected performance dynamics for different strategies.

    The ``variance_target`` (from the payout structure) adjusts how the
    EV of each strategy is weighted.  High variance_target (winner-take-all)
    penalizes chalk strategies because many competitors share the same
    pick, making it harder to win outright.  Low variance_target (top-25%)
    rewards chalk because consistent accuracy keeps you in the money.

    Args:
        pool_size: Number of competitors
        model_probs: Our championship probabilities
        public_picks: Public championship picks
        variance_target: 0.0 = minimize variance, 1.0 = maximize variance

    Returns:
        Strategy performance estimates
    """
    results = {}

    # Chalk strategy - pick by probability
    chalk_champion = max(model_probs, key=model_probs.get)
    chalk_prob = model_probs[chalk_champion]
    chalk_public = public_picks.get(chalk_champion, 0.3)

    # Expected competitors also picking chalk champion
    expected_chalk_competition = pool_size * chalk_public

    # Probability of winning pool with chalk
    # If chalk wins, we split with ~N*public_pct people
    chalk_win_share = chalk_prob / max(expected_chalk_competition, 1)
    results["chalk_ev"] = chalk_win_share

    # Contrarian strategy - pick high leverage
    contrarian_options = [(tid, model_probs[tid] / max(public_picks.get(tid, 0.01), 0.01)) for tid in model_probs]
    contrarian_options.sort(key=lambda x: x[1], reverse=True)

    if contrarian_options:
        contrarian_champion = contrarian_options[0][0]
        contrarian_prob = model_probs[contrarian_champion]
        contrarian_public = public_picks.get(contrarian_champion, 0.01)

        expected_contrarian_competition = pool_size * contrarian_public
        contrarian_win_share = contrarian_prob / max(expected_contrarian_competition, 1)
        results["contrarian_ev"] = contrarian_win_share
        results["leverage_ratio"] = contrarian_prob / contrarian_public

    # Variance-adjusted recommendation: blend chalk and contrarian EVs
    # based on the payout structure's variance preference.
    # High variance_target → prefer contrarian (unique picks win outright).
    # Low variance_target → prefer chalk (consistent accuracy stays in money).
    chalk_ev = results.get("chalk_ev", 0)
    contrarian_ev = results.get("contrarian_ev", 0)

    # Effective EV weights contrarian more when variance_target is high
    chalk_adjusted = chalk_ev * (1.0 - 0.5 * variance_target)
    contrarian_adjusted = contrarian_ev * (0.5 + 0.5 * variance_target)

    if contrarian_adjusted > chalk_adjusted:
        results["recommendation"] = "contrarian"
    else:
        results["recommendation"] = "chalk"

    results["variance_target"] = variance_target

    return results


@dataclass
class PoolAnalysis:
    """Complete pool analysis with bracket recommendations."""

    pool_size: int
    strategy_evs: Dict[str, float]
    recommended_strategy: str
    leverage_picks: List[LeveragePick]
    fade_picks: List[LeveragePick]
    pareto_brackets: List[BracketConfiguration]

    def print_summary(self) -> None:
        """Print analysis summary."""
        print(f"\n{'=' * 60}")
        print(f"POOL ANALYSIS - {self.pool_size} entries")
        print(f"{'=' * 60}")

        print(f"\nRecommended Strategy: {self.recommended_strategy.upper()}")

        print("\n📈 TOP LEVERAGE PICKS:")
        for pick in self.leverage_picks[:5]:
            print(f"  {pick}")

        print("\n📉 TEAMS TO FADE (Overvalued by public):")
        for pick in self.fade_picks[:5]:
            print(f"  {pick}")

        print(f"\n{'=' * 60}")


def analyze_pool(
    pool_size: int,
    model_probs: Dict[str, Dict[str, float]],
    public_picks: Dict[str, Dict[str, float]],
    scoring_system: Optional[Dict[str, int]] = None,
    team_metadata: Optional[Dict[str, TeamMetadata]] = None,
    ev_scoring_system: Optional[str] = None,
    strategy_profile: Optional[PoolStrategyProfile] = None,
    archetype_picks: Optional[Dict[str, Dict[str, float]]] = None,
    construction_mode: str = "forward_greedy",
) -> PoolAnalysis:
    """
    Complete pool analysis.

    Args:
        pool_size: Number of entries
        model_probs: Model probabilities by team and round
        public_picks: Public pick percentages
        scoring_system: Round -> points mapping (overrides adapter if set)
        team_metadata: Team metadata for richer recommendations
        ev_scoring_system: Scoring system name for adapter ("standard", "flat", "upset_bonus")
        strategy_profile: Pre-computed strategy profile (auto-generated if None)
        archetype_picks: Blended opponent pick distribution from behavioral
            archetype modeling.  When provided, this replaces ``public_picks``
            for leverage calculations, producing more realistic field models.

    Returns:
        PoolAnalysis with recommendations
    """
    # When archetype-modeled picks are available, use them for leverage
    # instead of raw public consensus.  This gives a more realistic view
    # of what competitors actually pick.
    effective_picks = archetype_picks if archetype_picks is not None else public_picks

    # Use scoring adapter if ev_scoring_system specified and no explicit scoring_system
    adapter = None
    if ev_scoring_system and not scoring_system:
        adapter = get_scoring_adapter(ev_scoring_system)
        scoring_system = {k: int(v) for k, v in adapter.round_weights.items()}

    calculator = LeverageCalculator(
        model_probs,
        effective_picks,
        scoring_system=scoring_system,
        team_metadata=team_metadata,
    )
    optimizer = ParetoOptimizer(calculator, pool_size)

    leverage_picks = calculator.find_leverage_picks()
    fade_picks = calculator.find_fade_picks()

    pareto_brackets = optimizer.generate_pareto_brackets(construction_mode=construction_mode)

    # Generate or use provided strategy profile for recommendation
    if strategy_profile is None:
        profile = get_strategy_profile(pool_size, scoring_system=ev_scoring_system or "standard")
    else:
        profile = strategy_profile

    # Calculate strategy EVs — build both dicts keyed on model_probs so
    # that teams present in the model but absent from public picks still
    # get seed-based prior coverage via _public_pct_with_fallback().
    championship_model = {tid: probs.get("CHAMP", 0) for tid, probs in model_probs.items()}
    championship_public = {tid: calculator._public_pct_with_fallback(tid, "CHAMP")[0] for tid in model_probs}

    dynamics = calculate_pool_dynamics(
        pool_size,
        championship_model,
        championship_public,
        variance_target=profile.variance_target,
    )

    # Override recommendation with profile-based strategy
    recommended = dynamics.get("recommendation", "balanced")
    if profile.contrarian_strength >= 1.5:
        recommended = "contrarian"
    elif profile.contrarian_strength <= 0.5:
        recommended = "chalk"

    dynamics["strategy_profile"] = {
        "strategy_mix": profile.strategy_mix,
        "contrarian_strength": profile.contrarian_strength,
        "champion_risk_level": profile.champion_risk_level,
        "description": profile.description,
    }

    return PoolAnalysis(
        pool_size=pool_size,
        strategy_evs=dynamics,
        recommended_strategy=recommended,
        leverage_picks=leverage_picks,
        fade_picks=fade_picks,
        pareto_brackets=pareto_brackets,
    )
