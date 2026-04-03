"""Diagnostic: measure how much noseed probabilities diverge from SEED_PICK_RATES.

Council Session 5 identified that all modes perform near-random because
the model's brackets may not be differentiated from the opponent field.
This script quantifies the divergence per round.

If noseed and SEED_PICK_RATES agree on most games, no opponent model
change or contrarian sampling will help — the model IS the crowd.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.data.seed_pick_model import SEED_PICK_RATES
from src.prediction.noseed_model import (
    train_noseed_model,
    build_noseed_round_probabilities,
)
from src.prediction.seed_probabilities import build_seed_round_probabilities
from src.simulation.pool_competition import ROUND_NAMES

HIST_DIR = Path("data/raw/historical")
BACKTEST_YEARS = [y for y in range(2008, 2026) if y != 2020]
SEED_MATCHUP_ORDER = [(1, 16), (8, 9), (5, 12), (4, 13), (6, 11), (3, 14), (7, 10), (2, 15)]
REGION_ORDER = ["East", "West", "South", "Midwest"]
ESPN_SCORING = {"R64": 10, "R32": 20, "S16": 40, "E8": 80, "F4": 160, "CHAMP": 320}


def load_seeds_and_regions(year):
    path = HIST_DIR / f"tournament_seeds_{year}.json"
    if not path.exists():
        return {}, {}
    with open(path) as f:
        data = json.load(f)
    seeds, regions = {}, {}
    if isinstance(data, dict) and "teams" in data:
        for t in data["teams"]:
            seeds[t["team_id"]] = t["seed"]
            regions[t["team_id"]] = t.get("region", "")
    return seeds, regions


def _load_team_stats(year):
    path = HIST_DIR / f"torvik_four_factors_{year}.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


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


def compute_divergence(first_round, seed_rp, noseed_rp, seeds):
    """Compute per-game probability divergence between noseed and seed/field.

    For each game, computes the head-to-head pick probability under both
    noseed and seed models, and under the SEED_PICK_RATES field model.
    """
    results = []
    current_teams_seed = list(first_round)
    current_teams_noseed = list(first_round)
    current_teams_field = list(first_round)

    # For simplicity, walk with seed model to determine matchups
    # (all models see the same first-round matchups; later rounds depend on path)
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

            # Noseed head-to-head probability
            p1_ns = noseed_rp.get(t1, {}).get(round_name, 0.0)
            p2_ns = noseed_rp.get(t2, {}).get(round_name, 0.0)
            p_ns = p1_ns / (p1_ns + p2_ns) if (p1_ns + p2_ns) > 1e-8 else 0.5

            # Seed head-to-head probability
            p1_sd = seed_rp.get(t1, {}).get(round_name, 0.0)
            p2_sd = seed_rp.get(t2, {}).get(round_name, 0.0)
            p_sd = p1_sd / (p1_sd + p2_sd) if (p1_sd + p2_sd) > 1e-8 else 0.5

            # Field (SEED_PICK_RATES) head-to-head probability
            p1_f = SEED_PICK_RATES.get(s1, {}).get(round_name, 0.0)
            p2_f = SEED_PICK_RATES.get(s2, {}).get(round_name, 0.0)
            p_f = p1_f / (p1_f + p2_f) if (p1_f + p2_f) > 1e-8 else 0.5

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
        seed_rp = build_seed_round_probabilities(seeds)
        noseed_rp = build_noseed_round_probabilities(model, seeds, stats)

        games = compute_divergence(first_round, seed_rp, noseed_rp, seeds)
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
