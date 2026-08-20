"""Diagnostic: measure how much noseed probabilities diverge from SEED_PICK_RATES.

Council Session 5 identified that all modes perform near-random because
the model's brackets may not be differentiated from the opponent field.
This script quantifies the divergence per round.

If noseed and SEED_PICK_RATES agree on most games, no opponent model
change or contrarian sampling will help — the model IS the crowd.
"""

import sys
from collections import defaultdict
from pathlib import Path
from scripts._common import _load_torvik_ff, load_seeds_and_regions  # noqa: F401

import numpy as np

from src.data.seed_pick_model import SEED_PICK_RATES
from src.prediction.noseed_model import (
    train_noseed_model,
    build_noseed_probabilities,
)
from src.prediction.pairwise import PairwiseProbabilities
from src.prediction.seed_probabilities import build_seed_probabilities
from src.simulation.pool_competition import ROUND_NAMES

HIST_DIR = Path("data/raw/historical")
BACKTEST_YEARS = [y for y in range(2008, 2026) if y != 2020]
SEED_MATCHUP_ORDER = [(1, 16), (8, 9), (5, 12), (4, 13), (6, 11), (3, 14), (7, 10), (2, 15)]
REGION_ORDER = ["East", "West", "South", "Midwest"]
ESPN_SCORING = {"R64": 10, "R32": 20, "S16": 40, "E8": 80, "F4": 160, "CHAMP": 320}


def _load_team_stats(year):
    return _load_torvik_ff(year) or {}


def build_first_round_matchups(seeds, regions):
    matchups = []
    teams_by_region = defaultdict(dict)
    for tid, seed in seeds.items():
        teams_by_region[regions.get(tid, "")][seed] = tid
    for region in REGION_ORDER:
        rt = teams_by_region.get(region, {})
        for hs, ls in SEED_MATCHUP_ORDER:
            matchups.extend([rt.get(hs, f"unk_{region}_{hs}"), rt.get(ls, f"unk_{region}_{ls}")])
    return matchups


def field_pick_share(s1, s2, round_name):
    """Share of brackets picking seed *s1* over seed *s2* at *round_name*.

    This normalizes two OWNERSHIP percentages, not two model probabilities.
    SEED_PICK_RATES[seed][round] is the fraction of public brackets advancing
    that seed, so the ratio answers "of brackets that advance one of these two,
    what share take s1" — a genuine ratio of two shares of the same population.

    Contrast with model advancement probabilities, where the same arithmetic is
    invalid (see src/prediction/pairwise.py). Kept in its own function so the
    contract scanner can allowlist this and only this.
    """
    p1 = SEED_PICK_RATES.get(s1, {}).get(round_name, 0.0)
    p2 = SEED_PICK_RATES.get(s2, {}).get(round_name, 0.0)
    return p1 / (p1 + p2) if (p1 + p2) > 1e-8 else 0.5


def compute_divergence(first_round, seed_pw, noseed_pw, seeds):
    """Compute per-game probability divergence between noseed and seed/field.

    For each game, computes the head-to-head win probability under both the
    noseed and seed models, and the field's pick share under SEED_PICK_RATES.

    CORRECTED 2026-08-19: this used to derive the model head-to-head numbers by
    normalizing two marginal round-advancement probabilities
    (``p1 / (p1 + p2)``), which is invalid from R32 onward and biased toward the
    favorite by 7-14pp. Since the divergence being measured here is *between*
    two such numbers, the bias partly cancelled — but not exactly, because seed
    and noseed have different marginal profiles. Any divergence figure produced
    by this script before that date should be treated as unreliable.

    Args:
        seed_pw / noseed_pw: :class:`PairwiseProbabilities` for each model.
    """
    results = []

    # All models see the same first-round matchups; later rounds depend on the
    # path, so walk with the seed model for consistent matchup tracking.
    current_teams = list(first_round)

    for round_idx in range(6):
        round_name = ROUND_NAMES[round_idx]
        next_round = []

        for g in range(0, len(current_teams), 2):
            if g + 1 >= len(current_teams):
                next_round.append(current_teams[g])
                continue

            t1, t2 = current_teams[g], current_teams[g + 1]
            s1 = seeds.get(t1, 8)
            s2 = seeds.get(t2, 8)

            p_ns = noseed_pw.p(t1, t2)
            p_sd = seed_pw.p(t1, t2)
            p_f = field_pick_share(s1, s2, round_name)

            results.append(
                {
                    "round": round_name,
                    "t1": t1,
                    "t2": t2,
                    "s1": s1,
                    "s2": s2,
                    "p_noseed": p_ns,
                    "p_seed": p_sd,
                    "p_field": p_f,
                    "delta_ns_field": p_ns - p_f,
                    "delta_ns_seed": p_ns - p_sd,
                    "abs_delta_ns_field": abs(p_ns - p_f),
                    "abs_delta_ns_seed": abs(p_ns - p_sd),
                }
            )

            # Advance using seed model (deterministic, for consistent matchup tracking)
            winner = t1 if p_sd >= 0.5 else t2
            next_round.append(winner)

        current_teams = next_round

    return results


def main():
    print("=" * 100)
    print("DIVERGENCE DIAGNOSTIC: Noseed vs Field (SEED_PICK_RATES)")
    print("=" * 100)
    print()

    all_games = []

    for year in BACKTEST_YEARS:
        seeds, regions = load_seeds_and_regions(year)
        if not seeds:
            continue
        stats = _load_team_stats(year)
        first_round = build_first_round_matchups(seeds, regions)
        if len(first_round) != 64:
            continue

        model = train_noseed_model(max_year=year)
        seed_pw = PairwiseProbabilities.from_dict(
            build_seed_probabilities(seeds), "historical_seed_h2h"
        )
        noseed_pw = PairwiseProbabilities.from_dict(
            build_noseed_probabilities(model, seeds, stats), "noseed_model"
        )

        games = compute_divergence(first_round, seed_pw, noseed_pw, seeds)
        for g in games:
            g["year"] = year
        all_games.extend(games)

    if not all_games:
        print("No data.")
        return 1

    # --- Per-round divergence summary ---
    print(f"  Per-Round Divergence (noseed vs field pick probability)")
    print(
        f"  {'Round':<8} {'N':>4} {'MeanAbsDelta':>13} {'MaxDelta':>10} {'StdDelta':>10} "
        f"{'ScoringWt':>10} {'WeightedDiv':>12}"
    )
    print(f"  {'-' * 75}")

    for rnd in ROUND_NAMES:
        rnd_games = [g for g in all_games if g["round"] == rnd]
        if not rnd_games:
            continue
        deltas = [g["abs_delta_ns_field"] for g in rnd_games]
        signed = [g["delta_ns_field"] for g in rnd_games]
        wt = ESPN_SCORING[rnd]
        print(
            f"  {rnd:<8} {len(rnd_games):>4} {np.mean(deltas):13.4f} "
            f"{np.max(deltas):10.4f} {np.std(signed):10.4f} "
            f"{wt:10d} {np.mean(deltas) * wt:12.2f}"
        )

    # --- Overall ---
    all_deltas = [g["abs_delta_ns_field"] for g in all_games]
    print(f"\n  Overall mean |delta|: {np.mean(all_deltas):.4f}")
    print(f"  Overall max  |delta|: {np.max(all_deltas):.4f}")

    # --- Same comparison: noseed vs seed model ---
    print(f"\n  Per-Round Divergence (noseed vs SEED model)")
    print(f"  {'Round':<8} {'N':>4} {'MeanAbsDelta':>13} {'MaxDelta':>10}")
    print(f"  {'-' * 40}")

    for rnd in ROUND_NAMES:
        rnd_games = [g for g in all_games if g["round"] == rnd]
        if not rnd_games:
            continue
        deltas = [g["abs_delta_ns_seed"] for g in rnd_games]
        print(f"  {rnd:<8} {len(rnd_games):>4} {np.mean(deltas):13.4f} {np.max(deltas):10.4f}")

    # --- Largest divergence games (noseed vs field) ---
    print(f"\n  Top 20 Largest Divergences (noseed vs field):")
    print(f"  {'Year':>4} {'Round':<6} {'Matchup':<35} {'P_noseed':>9} {'P_field':>9} {'Delta':>8} {'Pts':>5}")
    print(f"  {'-' * 85}")

    sorted_games = sorted(all_games, key=lambda g: g["abs_delta_ns_field"], reverse=True)
    for g in sorted_games[:20]:
        matchup = f"({g['s1']}){g['t1'][:12]} v ({g['s2']}){g['t2'][:12]}"
        print(
            f"  {g['year']:>4} {g['round']:<6} {matchup:<35} "
            f"{g['p_noseed']:9.3f} {g['p_field']:9.3f} "
            f"{g['delta_ns_field']:+8.3f} {ESPN_SCORING[g['round']]:5d}"
        )

    # --- Correlation: noseed vs seed vs field ---
    ns_probs = np.array([g["p_noseed"] for g in all_games])
    sd_probs = np.array([g["p_seed"] for g in all_games])
    f_probs = np.array([g["p_field"] for g in all_games])

    print(f"\n  Correlation Matrix:")
    print(f"    noseed vs seed:  r = {np.corrcoef(ns_probs, sd_probs)[0, 1]:.4f}")
    print(f"    noseed vs field: r = {np.corrcoef(ns_probs, f_probs)[0, 1]:.4f}")
    print(f"    seed vs field:   r = {np.corrcoef(sd_probs, f_probs)[0, 1]:.4f}")

    # --- Key question: does noseed pick more upsets than the field? ---
    print(f"\n  Upset Tendency (P(underdog) > 0.5 in model but < 0.5 in field):")
    upset_picks = [g for g in all_games if g["p_noseed"] < 0.5 and g["p_field"] > 0.5]
    anti_upset = [g for g in all_games if g["p_noseed"] > 0.5 and g["p_field"] < 0.5]
    print(f"    Noseed picks underdog when field picks favorite: {len(upset_picks)} games")
    print(f"    Noseed picks favorite when field picks underdog: {len(anti_upset)} games")
    print(f"    Total games: {len(all_games)}")

    # By round
    print(f"\n  Upset Picks by Round:")
    for rnd in ROUND_NAMES:
        rnd_upsets = [g for g in upset_picks if g["round"] == rnd]
        rnd_total = len([g for g in all_games if g["round"] == rnd])
        if rnd_total > 0:
            print(
                f"    {rnd:<8}: {len(rnd_upsets):3d} / {rnd_total:4d} games ({len(rnd_upsets) / rnd_total * 100:5.1f}%)"
            )

    print(f"\n{'=' * 100}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
