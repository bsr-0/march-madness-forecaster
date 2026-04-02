#!/usr/bin/env python3
"""Validate Elite 8 matchup interaction signals against historical data.

With only 68 E8 games (2008-2025), we cannot train/test split. Instead,
validate DIRECTIONAL CORRECTNESS:
1. Compute E8 interaction composite for all historical E8 games
2. Split into tertiles by composite score
3. Check: does high-composite tertile have higher upset rate?
4. Compare BSS against existing SI signals (target: BSS > -0.010)

Uses the same data loading patterns as backtest_by_round.py.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.optimization.e8_matchup_scorer import E8MatchupInteractionScorer

DATA_DIR = Path("data/raw")
HIST_DIR = DATA_DIR / "historical"

BACKTEST_YEARS = [2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025]


def load_tournament_results(year):
    path = HIST_DIR / f"tournament_results_{year}.json"
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f).get("games", [])


def load_team_features(year):
    """Load Torvik four-factor data for E8 interaction signals."""
    features = {}

    # Try torvik_{year}.json (has team-level stats)
    torvik_path = HIST_DIR / f"torvik_{year}.json"
    if not torvik_path.exists():
        torvik_path = DATA_DIR / f"torvik_{year}.json"
    torvik = {}
    if torvik_path.exists():
        with open(torvik_path) as f:
            data = json.load(f)
        torvik = {t["team_id"]: t for t in data.get("teams", [])}

    # Try four factors file
    ff_path = HIST_DIR / f"torvik_four_factors_{year}.json"
    if not ff_path.exists():
        ff_path = DATA_DIR / f"torvik_four_factors_{year}.json"
    four_factors = {}
    if ff_path.exists():
        with open(ff_path) as f:
            ff_data = json.load(f)
        if isinstance(ff_data, dict) and "teams" not in ff_data:
            four_factors = ff_data
        else:
            four_factors = {t.get("team_id", ""): t for t in ff_data.get("teams", [])}

    all_teams = set(torvik.keys()) | set(four_factors.keys())
    for tid in all_teams:
        tv = torvik.get(tid, {})
        ff = four_factors.get(tid, {})
        enriched = tv.get("enriched_stats", {})

        def sf(val, default):
            try:
                v = float(val)
                return v if np.isfinite(v) else default
            except (TypeError, ValueError):
                return default

        features[tid] = SimpleNamespace(
            opp_turnover_rate=sf(
                ff.get("opp_turnover_rate") or enriched.get("opp_turnover_rate") or tv.get("opp_turnover_rate"), 0.18
            ),
            turnover_rate=sf(ff.get("turnover_rate") or enriched.get("turnover_rate") or tv.get("turnover_rate"), 0.18),
            offensive_reb_rate=sf(
                ff.get("offensive_reb_rate") or enriched.get("offensive_reb_rate") or tv.get("offensive_reb_rate"), 0.28
            ),
            defensive_reb_rate=sf(
                ff.get("defensive_reb_rate") or enriched.get("defensive_reb_rate") or tv.get("defensive_reb_rate"), 0.72
            ),
            adj_tempo=sf(tv.get("adj_tempo"), 68.0),
            three_pt_pct=sf(ff.get("three_pt_pct") or tv.get("three_pt_pct"), 0.34),
            opp_effective_fg_pct=sf(
                ff.get("opp_effective_fg_pct")
                or enriched.get("opp_effective_fg_pct")
                or tv.get("opp_effective_fg_pct"),
                0.48,
            ),
            coach_e8_appearances=int(sf(tv.get("coach_e8_appearances"), 0)),
            coach_deep_run_rate=sf(tv.get("coach_deep_run_rate"), 0.0),
        )

    return features


def main():
    scorer = E8MatchupInteractionScorer()
    e8_games = []

    print("Loading Elite 8 games with Torvik four-factor data...\n")

    for year in BACKTEST_YEARS:
        games = load_tournament_results(year)
        features = load_team_features(year)

        for g in games:
            round_name = g.get("round", "")
            if round_name != "E8":
                continue

            t1 = g.get("team1_id", "")
            t2 = g.get("team2_id", "")
            s1 = g.get("seed1", 0)
            s2 = g.get("seed2", 0)
            winner = g.get("winner_id", "")

            if s1 == s2:
                continue

            # Identify favorite (lower seed) and underdog
            if s1 < s2:
                fav_id, dog_id, fav_seed, dog_seed = t1, t2, s1, s2
            else:
                fav_id, dog_id, fav_seed, dog_seed = t2, t1, s2, s1

            was_upset = winner == dog_id

            # Get features for both teams
            fav_feat = features.get(fav_id)
            dog_feat = features.get(dog_id)

            if fav_feat is None or dog_feat is None:
                continue

            # Score: underdog (team_a) vs favorite (team_b)
            # Positive composite → underdog has interaction advantage
            result = scorer.score(dog_feat, fav_feat)

            e8_games.append(
                {
                    "year": year,
                    "fav_id": fav_id,
                    "dog_id": dog_id,
                    "fav_seed": fav_seed,
                    "dog_seed": dog_seed,
                    "was_upset": was_upset,
                    "composite": result.composite,
                    "adjustment": result.adjustment,
                    "signals": result.signals,
                    "seed_gap": dog_seed - fav_seed,
                }
            )

    if not e8_games:
        print("ERROR: No E8 games with features found.")
        return 1

    print(f"Found {len(e8_games)} E8 games with complete Torvik data\n")

    # ========================================================================
    # 1. Tertile analysis
    # ========================================================================
    composites = [g["composite"] for g in e8_games]
    sorted_games = sorted(e8_games, key=lambda g: g["composite"])
    n = len(sorted_games)
    t1_end = n // 3
    t2_end = 2 * n // 3

    tertiles = [
        ("Low composite", sorted_games[:t1_end]),
        ("Mid composite", sorted_games[t1_end:t2_end]),
        ("High composite", sorted_games[t2_end:]),
    ]

    print("=" * 70)
    print("1. UPSET RATE BY E8 INTERACTION COMPOSITE TERTILE")
    print("   High composite = underdog has pairwise interaction advantage")
    print("=" * 70)
    print(f"\n  {'Tertile':<18} {'N':>5} {'Upsets':>7} {'Rate':>7}  {'Avg Comp':>9}  {'Avg Gap':>8}")
    print(f"  {'-' * 16} {'-' * 5} {'-' * 7} {'-' * 7}  {'-' * 9}  {'-' * 8}")

    for label, games in tertiles:
        n_games = len(games)
        upsets = sum(1 for g in games if g["was_upset"])
        rate = upsets / n_games if n_games else 0
        avg_comp = np.mean([g["composite"] for g in games]) if games else 0
        avg_gap = np.mean([g["seed_gap"] for g in games]) if games else 0
        print(f"  {label:<18} {n_games:5d} {upsets:7d} {rate:7.1%}  {avg_comp:9.3f}  {avg_gap:8.1f}")

    # ========================================================================
    # 2. BSS comparison
    # ========================================================================
    print(f"\n{'=' * 70}")
    print("2. BRIER SCORE COMPARISON: Seed baseline vs E8-interaction-adjusted")
    print("=" * 70)

    seed_brier_scores = []
    e8_brier_scores = []

    for g in e8_games:
        # Seed baseline: P(upset) from logistic on seed gap
        from src.data.seed_pick_model import _win_rate

        seed_upset_prob = 1.0 - _win_rate(g["fav_seed"], g["dog_seed"])
        outcome = 1.0 if g["was_upset"] else 0.0

        seed_brier_scores.append((seed_upset_prob - outcome) ** 2)

        # E8-adjusted: shift seed prob by adjustment
        e8_prob = max(0.01, min(0.99, seed_upset_prob + g["adjustment"]))
        e8_brier_scores.append((e8_prob - outcome) ** 2)

    seed_brier = np.mean(seed_brier_scores)
    e8_brier = np.mean(e8_brier_scores)
    bss = 1.0 - (e8_brier / seed_brier) if seed_brier > 0 else 0.0

    print(f"\n  Seed Brier:     {seed_brier:.4f}")
    print(f"  E8-adj Brier:   {e8_brier:.4f}")
    print(f"  BSS:            {bss:+.4f}")
    print(f"  Verdict:        {'E8 HELPS' if bss > 0.005 else 'NO DIFF' if bss > -0.005 else 'E8 HURTS'}")

    # ========================================================================
    # 3. Per-signal breakdown
    # ========================================================================
    print(f"\n{'=' * 70}")
    print("3. PER-SIGNAL MEAN VALUES: Upset games vs Non-upset games")
    print("=" * 70)

    signal_names = list(e8_games[0]["signals"].keys())
    upset_games = [g for g in e8_games if g["was_upset"]]
    non_upset_games = [g for g in e8_games if not g["was_upset"]]

    print(f"\n  {'Signal':<26} {'Upset Mean':>11} {'Non-upset':>11} {'Diff':>8}")
    print(f"  {'-' * 24} {'-' * 11} {'-' * 11} {'-' * 8}")

    for name in signal_names:
        upset_mean = np.mean([g["signals"][name] for g in upset_games]) if upset_games else 0
        non_upset_mean = np.mean([g["signals"][name] for g in non_upset_games]) if non_upset_games else 0
        diff = upset_mean - non_upset_mean
        print(f"  {name:<26} {upset_mean:11.3f} {non_upset_mean:11.3f} {diff:+8.3f}")

    composite_upset = np.mean([g["composite"] for g in upset_games]) if upset_games else 0
    composite_non = np.mean([g["composite"] for g in non_upset_games]) if non_upset_games else 0
    print(f"  {'COMPOSITE':<26} {composite_upset:11.3f} {composite_non:11.3f} {composite_upset - composite_non:+8.3f}")

    # ========================================================================
    # 4. Year-by-year breakdown
    # ========================================================================
    print(f"\n{'=' * 70}")
    print("4. YEAR-BY-YEAR E8 INTERACTION SCORES")
    print("=" * 70)

    by_year = defaultdict(list)
    for g in e8_games:
        by_year[g["year"]].append(g)

    print(f"\n  {'Year':>6}  {'Games':>5}  {'Upsets':>6}  {'Rate':>6}  {'Avg Comp':>9}  {'High-comp upset':>16}")
    print(f"  {'-' * 4:>6}  {'-' * 5}  {'-' * 6}  {'-' * 6}  {'-' * 9}  {'-' * 16}")

    for year in sorted(by_year.keys()):
        games = by_year[year]
        n_games = len(games)
        upsets = sum(1 for g in games if g["was_upset"])
        rate = upsets / n_games if n_games else 0
        avg_comp = np.mean([g["composite"] for g in games])

        # Did high-composite games upset more?
        median_comp = np.median([g["composite"] for g in games])
        high_games = [g for g in games if g["composite"] >= median_comp]
        high_upsets = sum(1 for g in high_games if g["was_upset"])
        high_rate = high_upsets / len(high_games) if high_games else 0

        print(f"  {year:6d}  {n_games:5d}  {upsets:6d}  {rate:6.0%}  {avg_comp:9.3f}  {high_rate:16.0%}")

    # ========================================================================
    # Summary
    # ========================================================================
    total_upsets = sum(1 for g in e8_games if g["was_upset"])
    total_games = len(e8_games)
    print(f"\n{'=' * 70}")
    print(f"SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Total E8 games with data: {total_games}")
    print(f"  Total upsets: {total_upsets} ({total_upsets / total_games:.1%})")
    print(f"  BSS vs seed baseline: {bss:+.4f}")
    print(f"  Composite mean (upsets): {composite_upset:.3f}")
    print(f"  Composite mean (non-upsets): {composite_non:.3f}")

    high_tertile_upsets = sum(1 for g in tertiles[2][1] if g["was_upset"])
    low_tertile_upsets = sum(1 for g in tertiles[0][1] if g["was_upset"])
    high_n = len(tertiles[2][1])
    low_n = len(tertiles[0][1])
    print(f"  High tertile upset rate: {high_tertile_upsets}/{high_n} = {high_tertile_upsets / high_n:.1%}")
    print(f"  Low tertile upset rate:  {low_tertile_upsets}/{low_n} = {low_tertile_upsets / low_n:.1%}")

    directional = "YES" if high_tertile_upsets / high_n > low_tertile_upsets / low_n else "NO"
    print(f"  Directional signal: {directional}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
