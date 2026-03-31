"""Seed-based public pick rate model.

Derives expected public pick percentages from first principles using:

1. **Historical seed advancement rates** computed from round-by-round
   win probabilities sourced from ``HISTORICAL_SEED_WIN_RATES``
   (1985-2025, ~160 games per first-round matchup).

2. **Chalk bias adjustment** — the public systematically over-picks
   favorites relative to true win rates.  This is one of the most
   well-documented phenomena in bracket pool analytics:

   - Metrick (1996), *March Madness: The Survival of Bias in the NCAA
     Basketball Tournament Bracket*: public brackets over-weight
     seeds 1-4 by 5-15 percentage points relative to historical rates.
   - ESPN's own "Who Picked Whom" data (2015-2024) shows that seed-1
     teams are picked to win the championship ~18% of the time (per
     team) despite historical championship rates of ~12% per team.
   - Chalk bias is strongest in early rounds (R64, R32) where the
     public picks nearly all 1- and 2-seeds, and in the championship
     round where the public concentrates on top seeds.

   We model this as: pick_pct = clip(adv_rate^(1/α), floor)
   where α ≈ 1.15 (favorites) to 1.35 (underdogs).

3. **Structural constraint**: Championship percentages across all 64
   teams must sum to ~100% (each bracket picks exactly one champion).
   R64 percentages for each 1v16 pairing must sum to ~100%.

Derivation methodology
----------------------
For each seed s in {1..16}:

  R64:   P(s advances) = P(s beats opponent) from HISTORICAL_SEED_WIN_RATES
  R32:   P(s in S16)   = P(R64) × P(win R32 | typical R32 opponent for s)
  S16:   P(s in E8)    = P(R32) × P(win S16 | typical S16 opponent for s)
  ...and so on through CHAMP.

The "typical opponent" at each round is determined by the bracket
structure: a 1-seed plays 8/9 in R64, then likely 4/5 in R32, then
likely 3/6 in S16, then likely 2/7 in E8, then likely 1/2 in F4/CHAMP.

Public pick percentages are then derived by applying the chalk bias
function to these advancement rates.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

# ============================================================================
# Historical seed win rates (1985-2025 NCAA Men's Tournament)
# Source: src/data/features/tournament_features.py HISTORICAL_SEED_WIN_RATES
# Cross-referenced with src/ml/calibration/brier_optimal.py MENS_SEED_WIN_RATES
# N ≈ 160 games per first-round matchup through 2025.
# ============================================================================

# (lower_seed, higher_seed) -> P(lower_seed wins)
_HISTORICAL_WIN_RATES: Dict[Tuple[int, int], float] = {
    # Round of 64 (canonical matchups)
    (1, 16): 0.990,   # 155/157 through 2025 (avg of two codebase sources)
    (2, 15): 0.944,
    (3, 14): 0.851,
    (4, 13): 0.791,
    (5, 12): 0.642,
    (6, 11): 0.622,
    (7, 10): 0.609,
    (8, 9):  0.515,
    # Round of 32 common matchups
    (1, 8):  0.798,
    (1, 9):  0.825,
    (2, 7):  0.670,
    (2, 10): 0.700,
    (3, 6):  0.580,
    (3, 11): 0.615,
    (4, 5):  0.539,
    (4, 12): 0.620,
    # Sweet 16 common matchups
    (1, 4):  0.650,
    (1, 5):  0.700,
    (2, 3):  0.555,
    (2, 6):  0.605,
    (1, 3):  0.610,
    # Elite 8 / Final Four
    (1, 2):  0.535,
    (1, 1):  0.500,
    (2, 2):  0.500,
}


def _win_rate(seed_a: int, seed_b: int) -> float:
    """P(seed_a beats seed_b) from historical data.

    Uses direct lookup when available, falls back to a logistic model
    calibrated from the historical data for unseen matchups.
    """
    if seed_a == seed_b:
        return 0.500
    lower, higher = min(seed_a, seed_b), max(seed_a, seed_b)
    rate = _HISTORICAL_WIN_RATES.get((lower, higher))
    if rate is not None:
        return rate if seed_a == lower else 1.0 - rate
    # Logistic fallback for unseen matchups.
    # β = 0.175, consistent with tournament_features.py (line 88) and
    # pipeline config.  Fitted to historical seed-vs-seed outcomes via
    # logistic regression on seed difference.
    diff = seed_b - seed_a
    return 1.0 / (1.0 + math.exp(-0.175 * diff))


# ============================================================================
# Bracket structure: expected opponents at each round by seed
# ============================================================================
# In a standard 64-team bracket, seeds face predictable opponents:
# R64: 1v16, 2v15, 3v14, 4v13, 5v12, 6v11, 7v10, 8v9
# R32: Winner of (1/16) vs winner of (8/9), etc.
# We model the "expected opponent" as a probability-weighted mixture
# of the likely opponents, not just the chalk opponent.

# For each seed, the likely opponents at each round, with weights
# representing the probability of facing that opponent (based on
# historical seed advancement).
# Format: {my_seed: [(opponent_seed, probability), ...]}

def _r64_opponent(seed: int) -> int:
    """The R64 opponent for a given seed (standard bracket pairing)."""
    return 17 - seed


def _compute_advancement_rates() -> Dict[int, Dict[str, float]]:
    """Compute P(seed s reaches round R) for all seeds and rounds.

    Uses the bracket structure and historical win rates to compute
    cumulative advancement probabilities.  These represent the TRUE
    advancement rates, not public pick percentages.

    Method: For each seed, compute the probability of surviving each
    round by considering the probability-weighted set of possible
    opponents.

    Returns:
        {seed: {"R64": p, "R32": p, "S16": p, "E8": p, "F4": p, "CHAMP": p}}
        where p is in [0, 1].
    """
    # Step 1: R64 advancement = P(beat R64 opponent)
    r64 = {}
    for seed in range(1, 17):
        opp = _r64_opponent(seed)
        r64[seed] = _win_rate(seed, opp)

    # Step 2: R32 advancement.
    # After R64, the bracket pairs up:
    #   1/16 winner vs 8/9 winner
    #   2/15 winner vs 7/10 winner
    #   3/14 winner vs 6/11 winner
    #   4/13 winner vs 5/12 winner
    # For a given seed s, we compute the weighted-average win prob
    # against each possible R32 opponent.

    # R32 bracket pairings: (group_a, group_b) where group = (fav, dog)
    r32_pairings = [
        ((1, 16), (8, 9)),
        ((2, 15), (7, 10)),
        ((3, 14), (6, 11)),
        ((4, 13), (5, 12)),
    ]

    r32 = {}
    for group_a, group_b in r32_pairings:
        for seed in group_a + group_b:
            # P(seed reaches R32) = r64[seed]
            if seed not in [g for pair in [group_a, group_b] for g in pair]:
                continue
            # Determine which group this seed is in and who the opponents are
            if seed in group_a:
                opp_group = group_b
            else:
                opp_group = group_a

            # Weighted win prob against each possible R32 opponent
            weighted_win = 0.0
            for opp_seed in opp_group:
                opp_weight = r64[opp_seed]
                weighted_win += opp_weight * _win_rate(seed, opp_seed)

            # P(reach S16) = P(reach R32) × P(win R32)
            r32[seed] = r64[seed] * weighted_win

    # Step 3: S16 advancement.
    # S16 pairings (within each region):
    #   Winner of (1/16 vs 8/9) vs Winner of (4/13 vs 5/12)
    #   Winner of (2/15 vs 7/10) vs Winner of (3/14 vs 6/11)
    s16_pairings = [
        ((1, 16, 8, 9), (4, 13, 5, 12)),
        ((2, 15, 7, 10), (3, 14, 6, 11)),
    ]

    s16 = {}
    for group_a, group_b in s16_pairings:
        all_seeds = group_a + group_b
        for seed in all_seeds:
            if seed in group_a:
                opp_seeds = group_b
            else:
                opp_seeds = group_a

            # Weighted win prob — opponent weight is P(opp reaches S16)
            total_opp_weight = sum(r32[o] for o in opp_seeds)
            if total_opp_weight < 1e-12:
                s16[seed] = r32[seed] * 0.5
                continue
            weighted_win = 0.0
            for opp_seed in opp_seeds:
                opp_weight = r32[opp_seed] / total_opp_weight
                weighted_win += opp_weight * _win_rate(seed, opp_seed)

            s16[seed] = r32[seed] * weighted_win

    # Step 4: E8 advancement.
    # E8 pairings: top half of region vs bottom half
    #   Winner of (1/16/8/9/4/13/5/12 bracket) vs
    #   Winner of (2/15/7/10/3/14/6/11 bracket)
    e8_top = (1, 16, 8, 9, 4, 13, 5, 12)
    e8_bot = (2, 15, 7, 10, 3, 14, 6, 11)

    e8 = {}
    for seed in range(1, 17):
        if seed in e8_top:
            opp_seeds = e8_bot
        else:
            opp_seeds = e8_top

        total_opp_weight = sum(s16[o] for o in opp_seeds)
        if total_opp_weight < 1e-12:
            e8[seed] = s16[seed] * 0.5
            continue
        weighted_win = 0.0
        for opp_seed in opp_seeds:
            opp_weight = s16[opp_seed] / total_opp_weight
            weighted_win += opp_weight * _win_rate(seed, opp_seed)

        e8[seed] = s16[seed] * weighted_win

    # Step 5: F4 advancement.
    # At this point we're in the national bracket — opponent could be
    # any seed from another region.  We approximate by assuming the
    # opponent distribution mirrors the E8 advancement distribution
    # across all seeds (i.e., any seed that made it to the F4).
    f4 = {}
    total_e8 = sum(e8.values())
    for seed in range(1, 17):
        weighted_win = 0.0
        for opp_seed in range(1, 17):
            if opp_seed == seed:
                # Same seed from another region
                opp_weight = e8[opp_seed] / total_e8
                weighted_win += opp_weight * 0.50
            else:
                opp_weight = e8[opp_seed] / total_e8
                weighted_win += opp_weight * _win_rate(seed, opp_seed)

        f4[seed] = e8[seed] * weighted_win

    # Step 6: Championship.
    # Same logic as F4 — opponent drawn from F4 distribution.
    champ = {}
    total_f4 = sum(f4.values())
    for seed in range(1, 17):
        weighted_win = 0.0
        for opp_seed in range(1, 17):
            if opp_seed == seed:
                opp_weight = f4[opp_seed] / total_f4
                weighted_win += opp_weight * 0.50
            else:
                opp_weight = f4[opp_seed] / total_f4
                weighted_win += opp_weight * _win_rate(seed, opp_seed)

        champ[seed] = f4[seed] * weighted_win

    # Assemble
    result: Dict[int, Dict[str, float]] = {}
    for seed in range(1, 17):
        result[seed] = {
            "R64": r64[seed],
            "R32": r32[seed],
            "S16": s16[seed],
            "E8": e8[seed],
            "F4": f4[seed],
            "CHAMP": champ[seed],
        }

    return result


# ============================================================================
# Chalk bias model — empirical multiplier table
# ============================================================================
# The public over-picks favorites.  Rather than a parametric power-law
# (which cannot fit the data — see discussion below), we use an empirical
# multiplier table: pick_rate = true_advancement × multiplier(seed, round).
#
# The multipliers are calibrated against ESPN "Who Picked Whom" BTC
# aggregate data from 2015–2024.  Documented anchor points:
#
#   Seed 1, R64:   ESPN ~97%, true ~99.0% → mult ≈ 0.98
#   Seed 1, F4:    ESPN ~42%, true ~21.0% → mult ≈ 2.00
#   Seed 1, CHAMP: ESPN ~18%, true ~11.9% → mult ≈ 1.51
#   Seed 2, F4:    ESPN ~22%, true ~11.8% → mult ≈ 1.87
#   Seed 2, CHAMP: ESPN ~6%,  true ~6.1%  → mult ≈ 1.00
#   Seed 5, R64:   ESPN ~64%, true ~64.2% → mult ≈ 1.00
#   Seed 8, R64:   ESPN ~52%, true ~51.5% → mult ≈ 1.01
#   Seed 12, R64:  ESPN ~36%, true ~35.8% → mult ≈ 1.01
#   Seed 16, R64:  ESPN ~2%,  true ~1.0%  → mult ≈ 2.00 (tiny absolute base)
#
# Key insights from the ESPN data:
#   1. R64: public tracks true rates closely (mult ≈ 1.0 for all seeds)
#   2. F4: public massively concentrates on 1–2 seeds (mult up to 2.0x)
#   3. CHAMP: 1-seeds specifically over-picked (1.51x), 2-seeds track truth
#   4. The power-law model failed because chalk bias is NON-UNIFORM —
#      it concentrates on 1-seeds specifically, not "all favorites equally"
#
# References:
#   - Metrick (1996), "March Madness: The Survival of Bias"
#   - Clair & Letscher (2007), "Optimal strategies for sports betting pools"
#   - ESPN BTC "Who Picked Whom" 2015–2024 aggregate
#
# IMPORTANT: These priors are a FALLBACK for when real ESPN/Yahoo/CBS
# public pick data is unavailable.  When real data exists, it should
# always be preferred.  The fallback_audit mechanism in
# LeverageCalculator tracks when these priors are used.

_ROUND_DEPTH = {"R64": 0, "R32": 1, "S16": 2, "E8": 3, "F4": 4, "CHAMP": 5}


def _chalk_multiplier(seed: int, round_name: str) -> float:
    """Return empirical chalk multiplier: ESPN_pick / true_advancement.

    Anchored to ESPN BTC "Who Picked Whom" aggregate 2015–2024.
    Cells without direct ESPN anchors are interpolated from the
    documented pattern: minimal R64 bias, strong F4/CHAMP concentration
    on 1–2 seeds, and declining bias for mid/low seeds.
    """
    # --- R64: minimal bias, public tracks historical rates ---
    if round_name == "R64":
        if seed <= 4:
            return 0.98   # slight under-pick (favorites already near 100%)
        if seed <= 8:
            return 1.00
        if seed <= 12:
            return 1.01
        return 1.50       # over-pick from tiny base (not meaningful for EV)

    # --- R32: slight chalk concentration begins ---
    if round_name == "R32":
        if seed <= 2:
            return 1.15
        if seed <= 4:
            return 1.05
        if seed <= 8:
            return 0.95
        return 0.90

    # --- S16: moderate concentration ---
    if round_name == "S16":
        if seed <= 2:
            return 1.35
        if seed <= 4:
            return 1.15
        if seed <= 8:
            return 0.90
        return 0.80

    # --- E8: strong concentration ---
    if round_name == "E8":
        if seed <= 2:
            return 1.65
        if seed <= 4:
            return 1.25
        if seed <= 8:
            return 0.85
        return 0.75

    # --- F4: very strong — anchored to ESPN data ---
    if round_name == "F4":
        if seed == 1:
            return 2.00   # ESPN anchor: 42% / 21%
        if seed == 2:
            return 1.87   # ESPN anchor: 22% / 11.8%
        if seed <= 4:
            return 1.40
        if seed <= 8:
            return 0.80
        return 0.65

    # --- CHAMP: use direct ESPN rates (see _ESPN_CHAMP_RATES) ---
    # Multiplier approach doesn't work for CHAMP because the
    # renormalization constraint (sum=100%) distorts the results.
    # CHAMP rates are set directly in compute_seed_pick_rates().
    return 1.0  # placeholder; overridden by direct rates


def _apply_chalk_bias(true_rate: float, seed: int, round_name: str) -> float:
    """Convert true advancement rate to expected public pick percentage.

    Applies empirical chalk multiplier calibrated from ESPN BTC
    "Who Picked Whom" aggregate data (2015–2024).
    """
    if true_rate <= 0:
        return 0.0
    if true_rate >= 1.0:
        return 1.0
    mult = _chalk_multiplier(seed, round_name)
    return min(true_rate * mult, 1.0)


# ============================================================================
# Assemble the final SEED_PICK_RATES table
# ============================================================================

# Empirical per-team championship pick rates from ESPN BTC aggregate
# (2015–2024).  The public concentrates champion picks on 1–2 seeds:
#   4 × 18% (1-seeds) + 4 × 6% (2-seeds) = 96% of all champion picks.
#   Seeds 3–16 share the remaining ~4%.
#
# These are per-team rates (4 teams of each seed in the bracket).
# They bypass the multiplier model because CHAMP renormalization
# (sum-to-100% constraint) distorts multiplier-based estimates.
# Seeds 3–16 rates are proportional to historical advancement,
# normalized to fill the remaining 4% budget.
_ESPN_CHAMP_RATES: Dict[int, float] = {
    1: 0.1800,    # ESPN anchor: ~18% per 1-seed team
    2: 0.0600,    # ESPN anchor: ~6% per 2-seed team
    # Seeds 3-16: ~4% total budget (0.04/4 = 0.01 per seed-team average).
    # Distributed proportionally to historical advancement rates, then
    # rescaled to sum to the remaining budget.
}


def compute_seed_pick_rates() -> Dict[int, Dict[str, float]]:
    """Compute seed-based public pick rate priors.

    Combines historical advancement rates with empirical chalk bias
    (multiplier table for R64–F4, direct ESPN rates for CHAMP) to
    produce expected public pick percentages by seed and round.

    Returns:
        {seed: {"R64": pct, "R32": pct, ..., "CHAMP": pct}}
        where pct is in (0, 1).
    """
    advancement = _compute_advancement_rates()
    rates: Dict[int, Dict[str, float]] = {}

    for seed in range(1, 17):
        rates[seed] = {}
        for rnd in ("R64", "R32", "S16", "E8", "F4"):
            true_rate = advancement[seed][rnd]
            rates[seed][rnd] = _apply_chalk_bias(true_rate, seed, rnd)

    # --- CHAMP: use direct ESPN rates for seeds 1-2, ---
    # --- proportional advancement for seeds 3-16.    ---
    top_seed_budget = sum(_ESPN_CHAMP_RATES.values())      # 0.24
    remaining_budget = 0.25 - top_seed_budget               # 0.01
    # 0.25 = total per-seed CHAMP budget (64 teams, 4 per seed, sum = 1.0)

    # Seeds 3-16: distribute remaining_budget proportionally to advancement
    adv_3_16 = {s: advancement[s]["CHAMP"] for s in range(3, 17)}
    adv_sum = sum(adv_3_16.values())

    for seed in range(1, 17):
        if seed in _ESPN_CHAMP_RATES:
            rates[seed]["CHAMP"] = _ESPN_CHAMP_RATES[seed]
        elif adv_sum > 0:
            rates[seed]["CHAMP"] = remaining_budget * adv_3_16[seed] / adv_sum
        else:
            rates[seed]["CHAMP"] = remaining_budget / 14.0

    # Apply minimum floor for numerical stability.
    for seed in range(1, 17):
        for rnd in rates[seed]:
            rates[seed][rnd] = max(rates[seed][rnd], 1e-5)

    return rates


def _round_to_sig(val: float, sig: int = 3) -> float:
    """Round to N significant figures."""
    if val == 0:
        return 0.0
    magnitude = math.floor(math.log10(abs(val)))
    factor = 10 ** (sig - 1 - magnitude)
    return round(val * factor) / factor


# Module-level computation — runs once at import time.
SEED_PICK_RATES: Dict[int, Dict[str, float]] = compute_seed_pick_rates()


# ============================================================================
# Diagnostic: print the table (useful for code review)
# ============================================================================

def print_comparison_table() -> None:
    """Print derived pick rates alongside advancement rates for review."""
    advancement = _compute_advancement_rates()
    pick_rates = compute_seed_pick_rates()
    rounds = ["R64", "R32", "S16", "E8", "F4", "CHAMP"]

    print("\n=== Historical Advancement Rates (true probabilities) ===")
    header = f"{'Seed':>4}" + "".join(f"  {r:>7}" for r in rounds)
    print(header)
    for seed in range(1, 17):
        vals = "".join(f"  {advancement[seed][r]:7.4f}" for r in rounds)
        print(f"{seed:4d}{vals}")

    print("\n=== Derived Public Pick Rates (with chalk bias) ===")
    print(header)
    for seed in range(1, 17):
        vals = "".join(f"  {pick_rates[seed][r]:7.4f}" for r in rounds)
        print(f"{seed:4d}{vals}")

    # Verify championship sum
    champ_total = sum(pick_rates[s]["CHAMP"] for s in range(1, 17)) * 4
    print(f"\nChampionship total (should ≈ 1.0): {champ_total:.4f}")

    # Verify R64 pairing sums ≈ 1.0
    for s in range(1, 9):
        opp = 17 - s
        r64_sum = pick_rates[s]["R64"] + pick_rates[opp]["R64"]
        print(f"R64 pairing {s:2d} vs {opp:2d}: sum = {r64_sum:.4f}")


if __name__ == "__main__":
    print_comparison_table()
