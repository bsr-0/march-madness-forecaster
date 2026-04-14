#!/usr/bin/env python3
"""Historical backtest: do high-pool-value upset picks actually hit?

For each tournament year (2008-2025, excluding 2020):
1. Load actual matchups from tournament_results
2. Load team features (Torvik + four factors) for seed-independent signals
3. Run MatchupVulnerabilityScorer to rank matchups by pool value
4. Check which upsets actually happened
5. Compare hit rate of top-pool-value picks vs random upsets vs seed-baseline

Key question: does pool_value (driven by seed-independent features like
tempo, rebounding, turnovers) predict which upsets hit better than chance?
"""

import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from scripts._common import (  # noqa: F401
    load_seeds,
    load_torvik_and_ff,
    load_tournament_results,
    sf,
)

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ml.ensemble.upset_detector import UpsetDetector
from src.optimization.matchup_vulnerability import (
    MatchupVulnerabilityScorer,
    _seed_based_public_default,
)

DATA_DIR = Path("data/raw")
HIST_DIR = DATA_DIR / "historical"

# Years with both tournament results and Torvik data
BACKTEST_YEARS = [2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025]

# Only count R64 upsets where seed gap >= 3 (e.g., 5v12 but not 8v9)
# 8v9 is a coin flip, not a real "upset" for pool purposes
MIN_SEED_GAP = 3


@dataclass
class TeamFeatures:
    """Features needed by UpsetDetector's amplifier signals."""

    adj_tempo: float = 68.0
    offensive_reb_rate: float = 0.28
    defensive_reb_rate: float = 0.72
    opp_turnover_rate: float = 0.18
    turnover_rate: float = 0.18
    three_pt_variance: float = 0.095
    pace_adjusted_variance: float = 0.0
    momentum: float = 0.0
    avg_experience: float = 2.0
    bench_depth_score: float = 0.0
    adj_efficiency_margin: float = 0.0


def load_team_features(year: int) -> Dict[str, TeamFeatures]:
    """Load Torvik + four factors into TeamFeatures objects."""
    torvik, four_factors = load_torvik_and_ff(year)
    features = {}
    for tid in set(torvik) | set(four_factors):
        tv = torvik.get(tid, {})
        ff = four_factors.get(tid, {})
        enriched = tv.get("enriched_stats", {})
        adj_off = sf(tv.get("adj_offensive_efficiency"), 100.0)
        adj_def = sf(tv.get("adj_defensive_efficiency"), 100.0)
        features[tid] = TeamFeatures(
            adj_tempo=sf(tv.get("adj_tempo"), 68.0),
            offensive_reb_rate=sf(
                ff.get("offensive_reb_rate") or enriched.get("offensive_reb_rate") or tv.get("offensive_reb_rate"),
                0.28,
            ),
            defensive_reb_rate=sf(
                ff.get("defensive_reb_rate") or enriched.get("defensive_reb_rate") or tv.get("defensive_reb_rate"),
                0.72,
            ),
            opp_turnover_rate=sf(
                ff.get("opp_turnover_rate") or enriched.get("opp_turnover_rate") or tv.get("opp_turnover_rate"),
                0.18,
            ),
            turnover_rate=sf(
                ff.get("turnover_rate") or enriched.get("turnover_rate") or tv.get("turnover_rate"),
                0.18,
            ),
            adj_efficiency_margin=adj_off - adj_def,
        )
    return features


def seed_based_model_prob(seed1: int, seed2: int) -> float:
    """Estimate P(team1 wins) from seeds alone.

    Uses a logistic model fit to historical seed-matchup outcomes.
    This serves as the 'model_prob_team1' since we're backtesting the
    vulnerability scorer, not the ensemble model.
    """
    from src.ml.ensemble.upset_detector import HISTORICAL_UPSET_RATES

    # Determine favorite/underdog
    fav_seed = min(seed1, seed2)
    dog_seed = max(seed1, seed2)

    # Look up historical upset rate
    key = (fav_seed, dog_seed)
    upset_rate = HISTORICAL_UPSET_RATES.get(key)

    if upset_rate is not None:
        fav_win_prob = 1.0 - upset_rate
    else:
        # Fallback: logistic based on seed gap
        seed_diff = dog_seed - fav_seed
        fav_win_prob = 1.0 / (1.0 + np.exp(-0.15 * seed_diff))

    # Return P(team1 wins)
    if seed1 <= seed2:
        return fav_win_prob
    else:
        return 1.0 - fav_win_prob


def run_backtest():
    detector = UpsetDetector()
    scorer = MatchupVulnerabilityScorer(detector)

    # Aggregate stats across all years
    all_matchups = []  # (year, pool_value, si_score, upset_prob, actually_upset, seed_gap, round_name)
    year_summaries = []

    print("=" * 80)
    print("POOL VALUE BACKTEST: Do seed-independent signals predict upsets?")
    print("=" * 80)

    for year in BACKTEST_YEARS:
        games = load_tournament_results(year)
        seeds = load_seeds(year)
        features = load_team_features(year)

        if not games:
            print(f"  {year}: no tournament results, skipping")
            continue

        # Filter to meaningful upset-possible matchups (seed gap >= MIN_SEED_GAP)
        # Include R64, R32, S16, E8 — not First Four
        eligible_games = []
        for g in games:
            if g["round_name"] == "FF":
                continue
            s1 = g.get("team1_seed", 0)
            s2 = g.get("team2_seed", 0)
            if abs(s1 - s2) < MIN_SEED_GAP:
                continue
            eligible_games.append(g)

        if not eligible_games:
            print(f"  {year}: no eligible games")
            continue

        # Build matchup dicts for scorer
        matchup_dicts = []
        game_outcomes = {}  # (team1_id, team2_id) -> was_upset

        for g in eligible_games:
            s1, s2 = g["team1_seed"], g["team2_seed"]
            t1, t2 = g["team1_id"], g["team2_id"]
            prob_t1 = seed_based_model_prob(s1, s2)

            matchup_dicts.append(
                {
                    "team1_id": t1,
                    "team2_id": t2,
                    "seed1": s1,
                    "seed2": s2,
                    "model_prob_team1": prob_t1,
                    "round_name": g["round_name"],
                }
            )

            # Was this an upset? (higher seed beat lower seed)
            fav_seed = min(s1, s2)
            if s1 == fav_seed:
                was_upset = not g["team1_won"]
            else:
                was_upset = g["team1_won"]

            game_outcomes[(t1, t2)] = was_upset

        # Score all matchups
        report = scorer.score_matchups(matchup_dicts, team_features=features)

        # Record results
        year_results = []
        for m in report.matchups:
            was_upset = game_outcomes.get((m.team1_id, m.team2_id), False)
            seed_gap = abs(m.seed2 - m.seed1)
            entry = (
                year,
                m.pool_value,
                m.seed_independent_score,
                m.upset_prob,
                was_upset,
                seed_gap,
                m.round_name,
                m.underdog_id,
                m.favorite_id,
            )
            all_matchups.append(entry)
            year_results.append(entry)

        n_upsets = sum(1 for e in year_results if e[4])
        n_games = len(year_results)
        year_summaries.append((year, n_games, n_upsets))
        print(f"  {year}: {n_games} eligible games, {n_upsets} upsets ({n_upsets / n_games * 100:.0f}%)")

    if not all_matchups:
        print("ERROR: No matchups collected")
        sys.exit(1)

    # === ANALYSIS ===
    print(f"\n{'=' * 80}")
    print(f"TOTAL: {len(all_matchups)} matchups across {len(year_summaries)} years")
    total_upsets = sum(1 for m in all_matchups if m[4])
    print(f"Total upsets: {total_upsets} ({total_upsets / len(all_matchups) * 100:.1f}%)")
    print(f"{'=' * 80}")

    # Sort all matchups by pool_value descending
    sorted_by_pv = sorted(all_matchups, key=lambda x: x[1], reverse=True)

    # Sort by seed_independent_score descending
    sorted_by_si = sorted(all_matchups, key=lambda x: x[2], reverse=True)

    # Sort by upset_prob descending (seed baseline)
    sorted_by_prob = sorted(all_matchups, key=lambda x: x[3], reverse=True)

    # === Hit rate by pool_value quintile ===
    print(f"\n{'=' * 80}")
    print("1. UPSET HIT RATE BY POOL VALUE QUINTILE")
    print("   (Does higher pool_value predict actual upsets?)")
    print(f"{'=' * 80}")
    n = len(sorted_by_pv)
    quintile_size = n // 5
    for q in range(5):
        start = q * quintile_size
        end = start + quintile_size if q < 4 else n
        subset = sorted_by_pv[start:end]
        hits = sum(1 for m in subset if m[4])
        avg_pv = np.mean([m[1] for m in subset])
        avg_si = np.mean([m[2] for m in subset])
        avg_prob = np.mean([m[3] for m in subset])
        hit_rate = hits / len(subset) * 100
        label = "TOP" if q == 0 else f"Q{q + 1}"
        print(
            f"  {label:>4} (n={len(subset):>3}, avg_pv={avg_pv:.3f}, avg_si={avg_si:.3f}, "
            f"avg_prob={avg_prob:.3f}): {hits:>3}/{len(subset)} upsets = {hit_rate:.1f}%"
        )

    # === Hit rate: top-N pool value picks ===
    print(f"\n{'=' * 80}")
    print("2. TOP-N POOL VALUE PICKS: HIT RATE")
    print("   (If you picked the top N upsets by pool value each year)")
    print(f"{'=' * 80}")
    for top_n in [5, 10, 15, 20, 30]:
        if top_n > len(sorted_by_pv):
            continue
        subset = sorted_by_pv[:top_n]
        hits = sum(1 for m in subset if m[4])
        print(f"  Top {top_n:>2}: {hits}/{top_n} hit = {hits / top_n * 100:.1f}%")

    # === Comparison: pool_value vs seed_independent vs upset_prob ===
    print(f"\n{'=' * 80}")
    print("3. RANKING COMPARISON: Which signal best predicts upsets?")
    print("   Hit rate of top-20% by each ranking method")
    print(f"{'=' * 80}")
    top_20_n = max(1, n // 5)

    for name, sorted_list in [
        ("pool_value", sorted_by_pv),
        ("seed_independent_score", sorted_by_si),
        ("upset_prob (seed baseline)", sorted_by_prob),
    ]:
        subset = sorted_list[:top_20_n]
        hits = sum(1 for m in subset if m[4])
        hit_rate = hits / len(subset) * 100
        print(f"  {name:<30}: {hits}/{len(subset)} = {hit_rate:.1f}%")

    # Overall upset rate for reference
    base_rate = total_upsets / len(all_matchups) * 100
    print(f"  {'baseline (all matchups)':<30}: {total_upsets}/{len(all_matchups)} = {base_rate:.1f}%")

    # === Per-year top-5 pool value analysis ===
    print(f"\n{'=' * 80}")
    print("4. PER-YEAR TOP-5 POOL VALUE PICKS")
    print("   (Did the scorer identify the right upsets each year?)")
    print(f"{'=' * 80}")

    yearly_hits = []
    for year in BACKTEST_YEARS:
        year_matchups = [
            (pv, si, prob, upset, gap, rnd, dog, fav)
            for (y, pv, si, prob, upset, gap, rnd, dog, fav) in all_matchups
            if y == year
        ]
        if not year_matchups:
            continue
        year_matchups.sort(key=lambda x: x[0], reverse=True)
        top5 = year_matchups[:5]
        hits = sum(1 for m in top5 if m[3])  # m[3] is upset boolean
        yearly_hits.append((year, hits, len(top5)))

        print(f"\n  {year} (top-5 by pool_value, {hits}/5 hit):")
        for pv, si, prob, upset, gap, rnd, dog, fav in top5:
            marker = "HIT" if upset else "   "
            print(f"    [{marker}] {fav} vs {dog} ({rnd}, gap={gap}, pv={pv:.3f}, si={si:.3f}, prob={prob:.3f})")

    # Summary
    avg_yearly_hits = np.mean([h for _, h, _ in yearly_hits])
    print(f"\n  Average: {avg_yearly_hits:.1f}/5 hits per year")

    # === Seed-independent signal analysis ===
    print(f"\n{'=' * 80}")
    print("5. SEED-INDEPENDENT SCORE: Does it add value over seed baseline?")
    print("   Comparing upsets where SI score was above/below median")
    print(f"{'=' * 80}")

    si_scores = [m[2] for m in all_matchups]
    median_si = np.median(si_scores)

    high_si = [m for m in all_matchups if m[2] >= median_si]
    low_si = [m for m in all_matchups if m[2] < median_si]

    high_si_hits = sum(1 for m in high_si if m[4])
    low_si_hits = sum(1 for m in low_si if m[4])

    print(f"  Median SI score: {median_si:.3f}")
    print(f"  High SI (n={len(high_si)}): {high_si_hits} upsets = {high_si_hits / max(1, len(high_si)) * 100:.1f}%")
    print(f"  Low SI  (n={len(low_si)}):  {low_si_hits} upsets = {low_si_hits / max(1, len(low_si)) * 100:.1f}%")
    diff = high_si_hits / max(1, len(high_si)) - low_si_hits / max(1, len(low_si))
    print(f"  Difference: {diff * 100:+.1f} percentage points")

    # === Bracket pool simulation ===
    print(f"\n{'=' * 80}")
    print("6. BRACKET POOL SIMULATION")
    print("   Contrarian (top pool_value upsets) vs Chalk (seed favorites)")
    print("   Scoring: R64=10, R32=20, S16=40, E8=80, F4=160, CHAMP=320")
    print(f"{'=' * 80}")

    round_points = {"R64": 10, "R32": 20, "S16": 40, "E8": 80, "F4": 160, "CHAMP": 320}
    contrarian_total = 0
    chalk_total = 0
    random_total = 0
    n_sims = 0

    for year in BACKTEST_YEARS:
        year_data = [
            (pv, si, prob, upset, gap, rnd, dog, fav)
            for (y, pv, si, prob, upset, gap, rnd, dog, fav) in all_matchups
            if y == year
        ]
        if not year_data:
            continue

        year_data.sort(key=lambda x: x[0], reverse=True)

        # Contrarian: pick underdog for top-5 pool_value matchups, chalk for rest
        # Chalk: always pick the favorite
        c_pts = 0
        ch_pts = 0
        top5_ids = set()
        for i, (pv, si, prob, upset, gap, rnd, dog, fav) in enumerate(year_data):
            pts = round_points.get(rnd, 10)
            if i < 5:
                top5_ids.add(dog)
                # Contrarian picks the underdog
                if upset:
                    c_pts += pts
                # Chalk picks the favorite
                if not upset:
                    ch_pts += pts
            else:
                # Both pick the favorite for non-top-5
                if not upset:
                    c_pts += pts
                    ch_pts += pts

        contrarian_total += c_pts
        chalk_total += ch_pts

        # Random: pick 5 random underdogs
        rng = np.random.RandomState(year)
        indices = rng.choice(len(year_data), size=min(5, len(year_data)), replace=False)
        r_pts = 0
        for i, (pv, si, prob, upset, gap, rnd, dog, fav) in enumerate(year_data):
            pts = round_points.get(rnd, 10)
            if i in indices:
                if upset:
                    r_pts += pts
            else:
                if not upset:
                    r_pts += pts
        random_total += r_pts
        n_sims += 1

        print(f"  {year}: contrarian={c_pts:>4}pts, chalk={ch_pts:>4}pts, random={r_pts:>4}pts")

    print(f"\n  TOTALS ({n_sims} years):")
    print(f"    Contrarian (pool_value top-5): {contrarian_total} pts")
    print(f"    Chalk (always favorite):       {chalk_total} pts")
    print(f"    Random (5 random underdogs):   {random_total} pts")
    print(
        f"    Contrarian vs Chalk edge:      {contrarian_total - chalk_total:+d} pts ({(contrarian_total / max(1, chalk_total) - 1) * 100:+.1f}%)"
    )

    # === Final verdict ===
    print(f"\n{'=' * 80}")
    print("VERDICT")
    print(f"{'=' * 80}")

    top20_pv_hits = sum(1 for m in sorted_by_pv[:top_20_n] if m[4])
    top20_pv_rate = top20_pv_hits / top_20_n

    top20_prob_hits = sum(1 for m in sorted_by_prob[:top_20_n] if m[4])
    top20_prob_rate = top20_prob_hits / top_20_n

    if top20_pv_rate > base_rate / 100 + 0.05:
        print("POSITIVE: Pool value ranking identifies upsets significantly better")
        print(f"than the base rate ({top20_pv_rate * 100:.1f}% vs {base_rate:.1f}%).")
    elif top20_pv_rate > base_rate / 100:
        print("MARGINAL: Pool value ranking shows a small edge over base rate")
        print(f"({top20_pv_rate * 100:.1f}% vs {base_rate:.1f}%).")
    else:
        print("NEGATIVE: Pool value ranking does not beat base rate at identifying upsets.")

    if top20_pv_rate > top20_prob_rate:
        print(f"Pool value ({top20_pv_rate * 100:.1f}%) beats pure upset probability ({top20_prob_rate * 100:.1f}%) —")
        print("seed-independent signals add value beyond seed-based priors.")
    elif abs(top20_pv_rate - top20_prob_rate) < 0.02:
        print(f"Pool value ({top20_pv_rate * 100:.1f}%) ≈ upset probability ({top20_prob_rate * 100:.1f}%) —")
        print("seed-independent signals don't clearly improve upset targeting.")
    else:
        print(f"Pool value ({top20_pv_rate * 100:.1f}%) < upset probability ({top20_prob_rate * 100:.1f}%) —")
        print("seed-independent signals may be adding noise, not signal.")

    if contrarian_total > chalk_total:
        edge_pct = (contrarian_total / max(1, chalk_total) - 1) * 100
        print(f"\nContrarian bracket strategy outperforms chalk by {edge_pct:.1f}%.")
        print("The pool value framework generates positive ROI in bracket pools.")
    else:
        print("\nChalk outperforms contrarian — the pool value formula needs recalibration.")


if __name__ == "__main__":
    run_backtest()
