"""Diagnose catastrophic opt_torvik years (2009, 2016, 2025).

For each year, reports:
  1. Major upsets vs Torvik barthag predictions
  2. Where the optimizer's contrarian picks went wrong
  3. How many points the optimizer left on the table vs seed baseline
  4. Whether the year was genuinely high-upset or Torvik-specific failure

Usage:
    python scripts/diagnose_catastrophic_years.py [--years 2009 2016 2025]
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.mc_pool_backtest import (
    HIST_DIR,
    ESPN_SCORING,
    SEED_MATCHUP_ORDER,
    REGION_ORDER,
    load_seeds_and_regions,
    load_tournament_results,
    _load_torvik_barthag,
    _log5,
    build_first_round_matchups,
    build_actual_outcome,
    build_seed_round_probabilities,
    build_torvik_round_probabilities,
    build_seed_pick_distribution,
    build_optimized_brackets,
    derive_f4_region_pairing,
    resolve_first_four,
    sample_model_brackets,
)
from src.simulation.pool_competition import (
    score_brackets_against_outcome,
    build_scoring_vector,
    ROUND_NAMES,
)

CATASTROPHIC_YEARS = [2009, 2016, 2025]
ROUND_POINTS = {"R64": 10, "R32": 20, "S16": 40, "E8": 80, "F4": 160, "CHAMP": 320}


def count_upsets_by_seed(games):
    """Count upsets (lower seed beats higher seed) per round."""
    upsets = defaultdict(list)
    for g in games:
        if g.get("round_name") == "FF":
            continue
        s1, s2 = g["team1_seed"], g["team2_seed"]
        winner_seed = s1 if g["team1_won"] else s2
        loser_seed = s2 if g["team1_won"] else s1
        winner_id = g["team1_id"] if g["team1_won"] else g["team2_id"]
        loser_id = g["team2_id"] if g["team1_won"] else g["team1_id"]
        if winner_seed > loser_seed:
            upsets[g["round_name"]].append(
                {
                    "winner": winner_id,
                    "winner_seed": winner_seed,
                    "loser": loser_id,
                    "loser_seed": loser_seed,
                    "seed_diff": winner_seed - loser_seed,
                }
            )
    return upsets


def compute_upset_rate(games):
    """Fraction of games where the higher seed lost."""
    total = 0
    upsets = 0
    for g in games:
        if g.get("round_name") == "FF":
            continue
        s1, s2 = g["team1_seed"], g["team2_seed"]
        if s1 == s2:
            continue
        total += 1
        winner_seed = s1 if g["team1_won"] else s2
        loser_seed = s2 if g["team1_won"] else s1
        if winner_seed > loser_seed:
            upsets += 1
    return upsets / total if total > 0 else 0.0


def torvik_surprise_games(games, barthag):
    """Find games where Torvik's Log5 probability was most wrong.

    Returns list of games sorted by how surprising the outcome was
    (highest P(loser) = most surprising).
    """
    surprises = []
    for g in games:
        if g.get("round_name") == "FF":
            continue
        t1, t2 = g["team1_id"], g["team2_id"]
        b1 = barthag.get(t1, 0.5)
        b2 = barthag.get(t2, 0.5)
        p1 = _log5(b1, b2)
        actual_winner = t1 if g["team1_won"] else t2
        actual_loser = t2 if g["team1_won"] else t1
        p_winner = p1 if g["team1_won"] else (1 - p1)
        surprises.append(
            {
                "round": g["round_name"],
                "winner": actual_winner,
                "winner_seed": g["team1_seed"] if g["team1_won"] else g["team2_seed"],
                "loser": actual_loser,
                "loser_seed": g["team2_seed"] if g["team1_won"] else g["team1_seed"],
                "p_winner": p_winner,
                "barthag_winner": barthag.get(actual_winner, 0.5),
                "barthag_loser": barthag.get(actual_loser, 0.5),
            }
        )
    surprises.sort(key=lambda x: x["p_winner"])
    return surprises


def analyze_year(year):
    """Full diagnostic for a single catastrophic year."""
    print(f"\n{'=' * 80}")
    print(f"  YEAR {year} — CATASTROPHIC YEAR DIAGNOSIS")
    print(f"{'=' * 80}")

    seeds, regions = load_seeds_and_regions(year)
    games = load_tournament_results(year)
    resolve_first_four(games, seeds, regions)
    barthag = _load_torvik_barthag(year, seeds)
    region_order = derive_f4_region_pairing(games, regions)
    first_round = build_first_round_matchups(seeds, regions, region_order=region_order)
    actual = build_actual_outcome(first_round, games)
    scoring_vector = build_scoring_vector(ESPN_SCORING)

    # --- 1. Upset profile ---
    upsets = count_upsets_by_seed(games)
    upset_rate = compute_upset_rate(games)
    total_upsets = sum(len(v) for v in upsets.values())
    print(f"\n  1. UPSET PROFILE")
    print(f"     Overall upset rate: {upset_rate:.1%} ({total_upsets} upsets)")
    for rnd in ["R64", "R32", "S16", "E8", "F4", "CHAMP"]:
        if rnd in upsets:
            for u in upsets[rnd]:
                print(f"       {rnd}: #{u['winner_seed']} {u['winner']:<25} over #{u['loser_seed']} {u['loser']}")

    # --- 2. Torvik's biggest misses ---
    surprises = torvik_surprise_games(games, barthag)
    print(f"\n  2. TORVIK'S BIGGEST SURPRISES (top 10 by P(winner))")
    for s in surprises[:10]:
        print(
            f"       {s['round']}: #{s['winner_seed']} {s['winner']:<25} beat "
            f"#{s['loser_seed']} {s['loser']:<25} "
            f"(Torvik gave winner {s['p_winner']:.1%}, barthag {s['barthag_winner']:.3f} vs {s['barthag_loser']:.3f})"
        )

    # --- 3. Score comparison: seed vs torvik vs opt_torvik ---
    seed_rp = build_seed_round_probabilities(seeds)
    torvik_rp = build_torvik_round_probabilities(seeds, regions, barthag)
    pick_dist = build_seed_pick_distribution(seeds)
    pool_size = 1000

    rng = np.random.default_rng(42 + year)

    # Seed baseline brackets
    seed_brackets = sample_model_brackets(first_round, seed_rp, 50, rng)
    seed_scores = score_brackets_against_outcome(seed_brackets, actual, scoring_vector)

    # Torvik stochastic brackets
    torvik_brackets = sample_model_brackets(first_round, torvik_rp, 50, rng)
    torvik_scores = score_brackets_against_outcome(torvik_brackets, actual, scoring_vector)

    # Optimized torvik brackets
    opt_brackets = build_optimized_brackets(
        first_round, seeds, regions, torvik_rp, pick_dist, pool_size, n_target=50, rng=rng
    )
    opt_scores = (
        score_brackets_against_outcome(opt_brackets, actual, scoring_vector)
        if opt_brackets.shape[0] > 0
        else np.array([0])
    )

    print(f"\n  3. BRACKET SCORE COMPARISON")
    print(f"     {'Mode':<15} {'MeanScr':>8} {'BestScr':>8} {'WorstScr':>8}")
    print(f"     {'-' * 45}")
    for name, scores in [("seed", seed_scores), ("torvik", torvik_scores), ("opt_torvik", opt_scores)]:
        print(f"     {name:<15} {scores.mean():8.0f} {scores.max():8.0f} {scores.min():8.0f}")

    # --- 4. Perfect bracket score for reference ---
    perfect_score = int(actual.sum() * 0 + sum(ROUND_POINTS[r] * n for r, n in zip(ROUND_NAMES, [32, 16, 8, 4, 2, 1])))
    # Actually compute it properly
    max_possible = sum(
        ESPN_SCORING[r] * n for r, n in zip(["R64", "R32", "S16", "E8", "F4", "CHAMP"], [32, 16, 8, 4, 2, 1])
    )
    print(f"\n     Max possible score: {max_possible}")

    # --- 5. Champion analysis ---
    champion = None
    for g in games:
        if g.get("round_name") in ("CHAMP", "NCG"):
            champion = g["team1_id"] if g["team1_won"] else g["team2_id"]
            champ_seed = g["team1_seed"] if g["team1_won"] else g["team2_seed"]
            break

    if champion:
        champ_barthag = barthag.get(champion, 0.0)
        champ_pick_pct = pick_dist.get(champion, {}).get("CHAMP", 0.0)
        champ_torvik_prob = torvik_rp.get(champion, {}).get("CHAMP", 0.0)
        champ_seed_prob = seed_rp.get(champion, {}).get("CHAMP", 0.0)
        print(f"\n  4. CHAMPION: #{champ_seed} {champion}")
        print(f"     Barthag: {champ_barthag:.4f}")
        print(f"     Torvik P(champ): {champ_torvik_prob:.3f}")
        print(f"     Seed P(champ):   {champ_seed_prob:.3f}")
        print(f"     Public pick %:   {champ_pick_pct:.3f}")

        leverage = champ_torvik_prob / champ_pick_pct if champ_pick_pct > 0.001 else float("inf")
        print(f"     Leverage ratio:  {leverage:.2f}")
        if leverage < 1.0:
            print(f"     >>> OPTIMIZER FADED THE ACTUAL CHAMPION (leverage < 1)")
        elif leverage > 2.0:
            print(f"     >>> Optimizer heavily backed the champion (leverage > 2)")

    # --- 6. Torvik barthag ranking vs seed ranking for Final Four teams ---
    print(f"\n  5. FINAL FOUR TEAMS — Torvik vs Seed expectations")
    f4_teams = set()
    for g in games:
        if g["round_name"] in ("F4", "CHAMP"):
            f4_teams.add(g["team1_id"])
            f4_teams.add(g["team2_id"])
    for tid in sorted(f4_teams):
        seed = seeds.get(tid, "?")
        b = barthag.get(tid, 0.0)
        torvik_f4 = torvik_rp.get(tid, {}).get("F4", 0.0)
        seed_f4 = seed_rp.get(tid, {}).get("F4", 0.0)
        print(f"     #{seed:<3} {tid:<25} barthag={b:.4f}  Torvik_P(F4)={torvik_f4:.3f}  Seed_P(F4)={seed_f4:.3f}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Diagnose catastrophic opt_torvik years")
    parser.add_argument("--years", type=int, nargs="+", default=CATASTROPHIC_YEARS)
    args = parser.parse_args()

    # Also compute upset rates across all years for comparison
    all_years = [y for y in range(2008, 2026) if y != 2020]
    print("BASELINE UPSET RATES (all years):")
    print(f"  {'Year':<6} {'Rate':>6} {'Upsets':>7}")
    rates = {}
    for y in all_years:
        _, _ = load_seeds_and_regions(y)
        g = load_tournament_results(y)
        if g:
            r = compute_upset_rate(g)
            n = sum(
                1
                for gg in g
                if gg.get("round_name") != "FF"
                and gg["team1_seed"] != gg["team2_seed"]
                and (gg["team1_seed"] > gg["team2_seed"]) == gg["team1_won"]
                or (gg["team2_seed"] > gg["team1_seed"]) == (not gg["team1_won"])
            )
            rates[y] = r
            marker = " <<<" if y in args.years else ""
            print(f"  {y:<6} {r:5.1%}  {marker}")

    mean_rate = np.mean(list(rates.values()))
    print(f"  {'Mean':<6} {mean_rate:5.1%}")

    for year in args.years:
        analyze_year(year)

    print(f"\n{'=' * 80}")
    print("DIAGNOSIS COMPLETE")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
