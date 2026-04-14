#!/usr/bin/env python3
"""Backtest segmented by round: does seed dominance break down in later rounds?

The council's #1 recommendation: before building anything new, check if
"seeds beat everything" holds uniformly across all tournament rounds, or
if signal exists in later rounds where seed gaps narrow.

For each round (R64, R32, S16, E8, F4, NCG):
- Upset rate by seed gap
- Whether seed-independent signals (tempo, rebounding, turnovers) add
  any predictive value
- Whether the ML model's upset probability beats the seed baseline

If seeds dominate rounds 1-3 but break down in E8/F4/NCG, that's a
bounded problem worth attacking with different features.
"""

import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ml.ensemble.upset_detector import HISTORICAL_UPSET_RATES, UpsetDetector

from scripts._common import load_tournament_results  # noqa: F401  (re-exported for back-compat)

DATA_DIR = Path("data/raw")
HIST_DIR = DATA_DIR / "historical"

BACKTEST_YEARS = [
    2008,
    2009,
    2010,
    2011,
    2012,
    2013,
    2014,
    2015,
    2016,
    2017,
    2018,
    2019,
    2021,
    2022,
    2023,
    2024,
    2025,
]

ROUND_ORDER = ["R64", "R32", "S16", "E8", "F4", "NCG"]
ROUND_LABELS = {
    "R64": "Round of 64",
    "R32": "Round of 32",
    "S16": "Sweet 16",
    "E8": "Elite 8",
    "F4": "Final Four",
    "NCG": "Championship",
}


@dataclass
class TeamFeatures:
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


@dataclass
class GameRecord:
    year: int
    round_name: str
    team1_id: str
    team2_id: str
    seed1: int
    seed2: int
    team1_won: bool
    fav_seed: int
    dog_seed: int
    fav_id: str
    dog_id: str
    was_upset: bool
    seed_gap: int
    # Signals
    seed_upset_prob: float  # Historical seed-based P(upset)
    si_score: float  # Seed-independent score from UpsetDetector


def load_team_features(year: int) -> Dict[str, TeamFeatures]:
    features = {}

    torvik_path = HIST_DIR / f"torvik_{year}.json"
    if not torvik_path.exists():
        torvik_path = DATA_DIR / f"torvik_{year}.json"
    torvik = {}
    if torvik_path.exists():
        with open(torvik_path) as f:
            data = json.load(f)
        torvik = {t["team_id"]: t for t in data.get("teams", [])}

    ff_path = HIST_DIR / f"torvik_four_factors_{year}.json"
    if not ff_path.exists():
        ff_path = DATA_DIR / f"torvik_four_factors_{year}.json"
    four_factors = {}
    if ff_path.exists():
        with open(ff_path) as f:
            ff_data = json.load(f)
        if isinstance(ff_data, dict) and not ff_data.get("teams"):
            # Top-level dict mixes metadata keys (e.g. "data_type", "cutoff_date",
            # "n_teams") with team-keyed dicts. Keep only the team entries.
            four_factors = {k: v for k, v in ff_data.items() if isinstance(v, dict)}
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


def seed_upset_prob(fav_seed: int, dog_seed: int) -> float:
    """Historical P(upset) for this seed matchup."""
    key = (fav_seed, dog_seed)
    rate = HISTORICAL_UPSET_RATES.get(key)
    if rate is not None:
        return rate
    # Fallback: logistic on seed gap
    gap = dog_seed - fav_seed
    if gap <= 0:
        return 0.5
    return 1.0 / (1.0 + math.exp(0.15 * gap))


def sigmoid(x: float, center: float = 0.0, scale: float = 1.0) -> float:
    z = (x - center) / max(scale, 1e-9)
    return 1.0 / (1.0 + math.exp(-z))


def compute_si_score(dog_features: TeamFeatures, fav_features: TeamFeatures) -> float:
    """Compute seed-independent score from tempo, rebounding, turnovers."""
    # Tempo mismatch
    tempo_diff = abs(dog_features.adj_tempo - fav_features.adj_tempo)
    tempo_sig = sigmoid(tempo_diff, center=4.0, scale=2.5)

    # Rebounding edge: underdog ORB vs favorite DRB weakness
    reb_edge = dog_features.offensive_reb_rate - (1.0 - fav_features.defensive_reb_rate)
    reb_sig = sigmoid(reb_edge, center=0.0, scale=0.04)

    # Turnover pressure
    to_pressure = (dog_features.opp_turnover_rate - 0.18) + (fav_features.turnover_rate - 0.18)
    to_sig = sigmoid(to_pressure, center=0.0, scale=0.03)

    # Equal weight
    return (tempo_sig + reb_sig + to_sig) / 3.0


def collect_all_games() -> List[GameRecord]:
    """Load all tournament games across all years with features and signals."""
    detector = UpsetDetector()
    all_records = []

    for year in BACKTEST_YEARS:
        games = load_tournament_results(year)
        features = load_team_features(year)
        if not games:
            continue

        for g in games:
            if g["round_name"] == "FF":
                continue

            s1, s2 = g["team1_seed"], g["team2_seed"]
            t1, t2 = g["team1_id"], g["team2_id"]

            if s1 <= s2:
                fav_seed, dog_seed = s1, s2
                fav_id, dog_id = t1, t2
                was_upset = not g["team1_won"]
            else:
                fav_seed, dog_seed = s2, s1
                fav_id, dog_id = t2, t1
                was_upset = g["team1_won"]

            seed_gap = dog_seed - fav_seed

            # Seed-based upset probability
            s_prob = seed_upset_prob(fav_seed, dog_seed)

            # Seed-independent score
            dog_feat = features.get(dog_id, TeamFeatures())
            fav_feat = features.get(fav_id, TeamFeatures())
            si = compute_si_score(dog_feat, fav_feat)

            all_records.append(
                GameRecord(
                    year=year,
                    round_name=g["round_name"],
                    team1_id=t1,
                    team2_id=t2,
                    seed1=s1,
                    seed2=s2,
                    team1_won=g["team1_won"],
                    fav_seed=fav_seed,
                    dog_seed=dog_seed,
                    fav_id=fav_id,
                    dog_id=dog_id,
                    was_upset=was_upset,
                    seed_gap=seed_gap,
                    seed_upset_prob=s_prob,
                    si_score=si,
                )
            )

    return all_records


def brier_score(predictions: List[float], outcomes: List[bool]) -> float:
    """Mean squared error between predicted prob and binary outcome."""
    if not predictions:
        return float("nan")
    return np.mean([(p - float(o)) ** 2 for p, o in zip(predictions, outcomes)])


def run_round_segmentation():
    print("Loading all tournament games...")
    records = collect_all_games()
    print(f"Loaded {len(records)} games across {len(BACKTEST_YEARS)} years\n")

    # =========================================================================
    # SECTION 1: Basic stats by round
    # =========================================================================
    print("=" * 90)
    print("1. UPSET RATE BY ROUND")
    print("   Does seed dominance hold uniformly, or break down in later rounds?")
    print("=" * 90)

    print(f"\n  {'Round':<16} {'Games':>6} {'Upsets':>7} {'Rate':>7}  {'Avg Seed Gap':>13}  {'Avg Fav Seed':>13}")
    print(f"  {'-' * 16} {'-' * 6} {'-' * 7} {'-' * 7}  {'-' * 13}  {'-' * 13}")

    for rnd in ROUND_ORDER:
        rnd_games = [r for r in records if r.round_name == rnd]
        if not rnd_games:
            continue
        n = len(rnd_games)
        upsets = sum(1 for r in rnd_games if r.was_upset)
        rate = upsets / n * 100
        avg_gap = np.mean([r.seed_gap for r in rnd_games])
        avg_fav = np.mean([r.fav_seed for r in rnd_games])
        label = ROUND_LABELS.get(rnd, rnd)
        print(f"  {label:<16} {n:>6} {upsets:>7} {rate:>6.1f}%  {avg_gap:>13.1f}  {avg_fav:>13.1f}")

    total_upsets = sum(1 for r in records if r.was_upset)
    print(f"  {'TOTAL':<16} {len(records):>6} {total_upsets:>7} {total_upsets / len(records) * 100:>6.1f}%")

    # =========================================================================
    # SECTION 2: Brier score by round — seed baseline vs SI-adjusted
    # =========================================================================
    print(f"\n{'=' * 90}")
    print("2. BRIER SCORE BY ROUND: Seed baseline vs SI-adjusted prediction")
    print("   Lower = better. BSS > 0 means SI signals improve on seeds.")
    print("=" * 90)

    print(f"\n  {'Round':<16} {'N':>5}  {'Seed Brier':>11}  {'SI-adj Brier':>13}  {'BSS':>7}  {'Verdict':>12}")
    print(f"  {'-' * 16} {'-' * 5}  {'-' * 11}  {'-' * 13}  {'-' * 7}  {'-' * 12}")

    for rnd in ROUND_ORDER:
        rnd_games = [r for r in records if r.round_name == rnd]
        if not rnd_games:
            continue

        # Seed baseline: predict P(upset) = historical seed rate
        # Then Brier = mean((p - outcome)^2) where outcome=1 if upset
        seed_preds = [r.seed_upset_prob for r in rnd_games]
        outcomes = [r.was_upset for r in rnd_games]
        seed_brier = brier_score(seed_preds, outcomes)

        # SI-adjusted: blend seed prob with SI score
        # P_adj = seed_prob * (0.7 + 0.6 * si_score)  clamped to [0.01, 0.99]
        # This is a simple test of whether SI adds information
        si_preds = []
        for r in rnd_games:
            adj = r.seed_upset_prob * (0.7 + 0.6 * r.si_score)
            adj = max(0.01, min(0.99, adj))
            si_preds.append(adj)
        si_brier = brier_score(si_preds, outcomes)

        bss = 1.0 - si_brier / seed_brier if seed_brier > 0 else 0.0
        label = ROUND_LABELS.get(rnd, rnd)

        if bss > 0.02:
            verdict = "SI HELPS"
        elif bss < -0.02:
            verdict = "SI HURTS"
        else:
            verdict = "NO DIFF"

        print(f"  {label:<16} {len(rnd_games):>5}  {seed_brier:>11.4f}  {si_brier:>13.4f}  {bss:>+7.3f}  {verdict:>12}")

    # =========================================================================
    # SECTION 3: SI score vs upset rate by round
    # =========================================================================
    print(f"\n{'=' * 90}")
    print("3. SEED-INDEPENDENT SCORE: Upset rate by SI quintile, per round")
    print("   If SI signals work, higher SI quintile should have higher upset rate.")
    print("=" * 90)

    for rnd in ROUND_ORDER:
        rnd_games = [r for r in records if r.round_name == rnd]
        if len(rnd_games) < 10:
            continue

        # Sort by SI score, split into terciles (quintiles too small for some rounds)
        sorted_games = sorted(rnd_games, key=lambda r: r.si_score, reverse=True)
        n = len(sorted_games)
        third = n // 3

        slices = [
            ("High SI", sorted_games[:third]),
            ("Mid SI", sorted_games[third : 2 * third]),
            ("Low SI", sorted_games[2 * third :]),
        ]

        label = ROUND_LABELS.get(rnd, rnd)
        print(f"\n  {label} (n={n}):")
        for name, subset in slices:
            hits = sum(1 for r in subset if r.was_upset)
            rate = hits / len(subset) * 100 if subset else 0
            avg_si = np.mean([r.si_score for r in subset]) if subset else 0
            avg_gap = np.mean([r.seed_gap for r in subset]) if subset else 0
            print(
                f"    {name:<8} (n={len(subset):>3}, avg_si={avg_si:.3f}, avg_gap={avg_gap:.1f}): "
                f"{hits:>3} upsets = {rate:.1f}%"
            )

    # =========================================================================
    # SECTION 4: Later rounds deep dive — where seeds break down
    # =========================================================================
    print(f"\n{'=' * 90}")
    print("4. LATER ROUNDS DEEP DIVE: S16 / E8 / F4 / NCG")
    print("   These rounds have smaller seed gaps. Do any signals help here?")
    print("=" * 90)

    later_games = [r for r in records if r.round_name in ("S16", "E8", "F4", "NCG")]
    print(f"\n  Total later-round games: {len(later_games)}")
    later_upsets = sum(1 for r in later_games if r.was_upset)
    print(f"  Later-round upsets: {later_upsets} ({later_upsets / max(1, len(later_games)) * 100:.1f}%)")

    # Seed gap distribution in later rounds
    print(f"\n  Upset rate by seed gap in later rounds:")
    gap_groups = defaultdict(list)
    for r in later_games:
        gap_groups[r.seed_gap].append(r)

    for gap in sorted(gap_groups.keys()):
        games = gap_groups[gap]
        upsets = sum(1 for r in games if r.was_upset)
        rate = upsets / len(games) * 100
        print(f"    Gap={gap:>2}: {len(games):>4} games, {upsets:>3} upsets = {rate:.1f}%")

    # SI score predictive power in later rounds
    if later_games:
        sorted_later = sorted(later_games, key=lambda r: r.si_score, reverse=True)
        n = len(sorted_later)
        half = n // 2
        high_half = sorted_later[:half]
        low_half = sorted_later[half:]

        high_upsets = sum(1 for r in high_half if r.was_upset)
        low_upsets = sum(1 for r in low_half if r.was_upset)

        print(f"\n  SI score split in later rounds:")
        print(
            f"    High SI (n={len(high_half)}): {high_upsets} upsets = {high_upsets / max(1, len(high_half)) * 100:.1f}%"
        )
        print(
            f"    Low SI  (n={len(low_half)}):  {low_upsets} upsets = {low_upsets / max(1, len(low_half)) * 100:.1f}%"
        )
        diff = high_upsets / max(1, len(high_half)) - low_upsets / max(1, len(low_half))
        print(f"    Difference: {diff * 100:+.1f}pp")

    # =========================================================================
    # SECTION 5: Early rounds (R64/R32) — confirming seed dominance
    # =========================================================================
    print(f"\n{'=' * 90}")
    print("5. EARLY ROUNDS (R64 + R32): Confirming seed dominance")
    print("=" * 90)

    early_games = [r for r in records if r.round_name in ("R64", "R32")]
    print(f"\n  Total early-round games: {len(early_games)}")
    early_upsets = sum(1 for r in early_games if r.was_upset)
    print(f"  Early-round upsets: {early_upsets} ({early_upsets / max(1, len(early_games)) * 100:.1f}%)")

    # Brier comparison for early rounds
    seed_preds = [r.seed_upset_prob for r in early_games]
    outcomes = [r.was_upset for r in early_games]
    seed_brier = brier_score(seed_preds, outcomes)

    si_preds = [max(0.01, min(0.99, r.seed_upset_prob * (0.7 + 0.6 * r.si_score))) for r in early_games]
    si_brier = brier_score(si_preds, outcomes)
    bss_early = 1.0 - si_brier / seed_brier if seed_brier > 0 else 0.0

    print(f"  Seed Brier: {seed_brier:.4f}")
    print(f"  SI-adj Brier: {si_brier:.4f}")
    print(f"  BSS: {bss_early:+.4f}")

    # Same for later rounds
    if later_games:
        seed_preds_l = [r.seed_upset_prob for r in later_games]
        outcomes_l = [r.was_upset for r in later_games]
        seed_brier_l = brier_score(seed_preds_l, outcomes_l)

        si_preds_l = [max(0.01, min(0.99, r.seed_upset_prob * (0.7 + 0.6 * r.si_score))) for r in later_games]
        si_brier_l = brier_score(si_preds_l, outcomes_l)
        bss_later = 1.0 - si_brier_l / seed_brier_l if seed_brier_l > 0 else 0.0

        print(f"\n  Later rounds for comparison:")
        print(f"  Seed Brier: {seed_brier_l:.4f}")
        print(f"  SI-adj Brier: {si_brier_l:.4f}")
        print(f"  BSS: {bss_later:+.4f}")

    # =========================================================================
    # SECTION 6: Year-over-year stability
    # =========================================================================
    print(f"\n{'=' * 90}")
    print("6. YEAR-OVER-YEAR UPSET RATES BY ROUND")
    print("   Checking for regime change / instability")
    print("=" * 90)

    # Header
    header = f"  {'Year':>6}"
    for rnd in ROUND_ORDER:
        header += f"  {rnd:>6}"
    header += "  TOTAL"
    print(header)
    print(f"  {'-' * 6}" + f"  {'-' * 6}" * (len(ROUND_ORDER) + 1))

    for year in BACKTEST_YEARS:
        row = f"  {year:>6}"
        year_games = [r for r in records if r.year == year]
        for rnd in ROUND_ORDER:
            rnd_games = [r for r in year_games if r.round_name == rnd]
            if not rnd_games:
                row += f"  {'--':>6}"
            else:
                upsets = sum(1 for r in rnd_games if r.was_upset)
                rate = upsets / len(rnd_games) * 100
                row += f"  {rate:>5.0f}%"
        total_ups = sum(1 for r in year_games if r.was_upset)
        total_rate = total_ups / max(1, len(year_games)) * 100
        row += f"  {total_rate:>5.0f}%"
        print(row)

    # Averages
    row = f"  {'AVG':>6}"
    for rnd in ROUND_ORDER:
        rnd_games = [r for r in records if r.round_name == rnd]
        if rnd_games:
            rate = sum(1 for r in rnd_games if r.was_upset) / len(rnd_games) * 100
            row += f"  {rate:>5.1f}"
        else:
            row += f"  {'--':>6}"
    row += f"  {total_upsets / len(records) * 100:>5.1f}"
    print(row)

    # =========================================================================
    # VERDICT
    # =========================================================================
    print(f"\n{'=' * 90}")
    print("VERDICT: Does seed dominance break down in later rounds?")
    print("=" * 90)

    # Compare early vs late upset rates and SI effectiveness
    early_upset_rate = early_upsets / max(1, len(early_games))
    later_upset_rate = later_upsets / max(1, len(later_games)) if later_games else 0

    print(f"\n  Early rounds (R64+R32) upset rate: {early_upset_rate * 100:.1f}%")
    print(f"  Later rounds (S16+)   upset rate: {later_upset_rate * 100:.1f}%")

    if later_upset_rate > early_upset_rate + 0.05:
        print(
            f"\n  YES — Later rounds have significantly higher upset rate ({later_upset_rate * 100:.1f}% vs {early_upset_rate * 100:.1f}%)."
        )
        print("  Seed dominance weakens as the tournament progresses.")
        print("  This is where alternative signals might add value.")
    elif later_upset_rate > early_upset_rate:
        print(
            f"\n  MARGINAL — Later rounds slightly higher ({later_upset_rate * 100:.1f}% vs {early_upset_rate * 100:.1f}%),"
        )
        print("  but the difference may not be significant given sample sizes.")
    else:
        print(f"\n  NO — Seeds dominate uniformly across all rounds.")

    if later_games:
        if bss_later > 0.02:
            print(f"\n  SI signals show positive BSS in later rounds ({bss_later:+.4f}).")
            print("  There may be an exploitable niche in later rounds.")
        elif bss_later < -0.02:
            print(f"\n  SI signals actually HURT in later rounds (BSS={bss_later:+.4f}).")
            print("  Seed-independent features add noise, not signal, even where seeds converge.")
        else:
            print(f"\n  SI signals show no meaningful effect in later rounds (BSS={bss_later:+.4f}).")

    print()


if __name__ == "__main__":
    run_round_segmentation()
