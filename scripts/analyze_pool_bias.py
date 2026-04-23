"""Analyze pick bias in the actual pool vs ESPN national averages.

Identifies teams that the pool over-picks or under-picks relative to
the national field, creating exploitable leverage for contrarian strategy.

Usage:
    python scripts/analyze_pool_bias.py
    python scripts/analyze_pool_bias.py --year 2026
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

POOL_DATA = "data/pool_history/pool_hist_results.json"


def load_pool_data():
    with open(POOL_DATA) as f:
        return json.load(f)


def analyze_year(year_data: dict, year: str) -> dict:
    """Analyze pick distribution for one year."""
    brackets = year_data.get("brackets", [])
    n = len(brackets)
    if n == 0:
        return {}

    # Count picks per team per round
    rounds = ["r64", "r32", "s16", "e8", "f4", "champ"]
    round_labels = {
        "r64": "Round of 64",
        "r32": "Round of 32",
        "s16": "Sweet 16",
        "e8": "Elite 8",
        "f4": "Final Four",
        "champ": "Champion",
    }

    team_picks: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for b in brackets:
        for rnd in rounds:
            picks = b.get(rnd, [])
            if isinstance(picks, list):
                for team in picks:
                    team_picks[team][rnd] += 1
            elif isinstance(picks, str):
                team_picks[picks][rnd] += 1

    # Compute pick percentages
    analysis = {
        "year": year,
        "n_brackets": n,
        "teams": {},
    }

    # Sort by champion pick rate
    sorted_teams = sorted(
        team_picks.items(),
        key=lambda x: x[1].get("champ", 0),
        reverse=True,
    )

    for team, picks in sorted_teams:
        team_analysis = {}
        for rnd in rounds:
            count = picks.get(rnd, 0)
            expected = {"r64": 32, "r32": 16, "s16": 8, "e8": 4, "f4": 2, "champ": 1}
            pct = (count / n) * 100
            team_analysis[rnd] = {"count": count, "pct": round(pct, 1)}
        analysis["teams"][team] = team_analysis

    return analysis


def find_pool_bias(analysis: dict) -> list[dict]:
    """Identify over/under-picked teams in late rounds."""
    biases = []
    n = analysis["n_brackets"]

    for team, rounds in analysis["teams"].items():
        champ_pct = rounds.get("champ", {}).get("pct", 0)
        f4_pct = rounds.get("f4", {}).get("pct", 0)
        e8_pct = rounds.get("e8", {}).get("pct", 0)

        # A team picked as champion by >20% of a 30-person pool is heavily concentrated
        if champ_pct > 15 or f4_pct > 30:
            biases.append({
                "team": team,
                "champ_pct": champ_pct,
                "f4_pct": f4_pct,
                "e8_pct": e8_pct,
                "signal": "OVER-PICKED",
                "strategy": "FADE — pool is concentrated here, low leverage",
            })

    return biases


def print_report(pool_data: dict, year_filter: str = None):
    years = pool_data.get("years", {})

    for year in sorted(years.keys()):
        if year_filter and year != year_filter:
            continue

        analysis = analyze_year(years[year], year)
        if not analysis:
            continue

        n = analysis["n_brackets"]
        print(f"\n{'='*60}")
        print(f"  {year} Pool Analysis ({n} brackets)")
        print(f"{'='*60}")

        # Champion concentration
        print(f"\n  Champion picks:")
        for team, rounds in analysis["teams"].items():
            champ = rounds.get("champ", {})
            if champ.get("count", 0) > 0:
                print(f"    {team:6s}: {champ['count']:2d}/{n} ({champ['pct']:5.1f}%)")

        # Final Four concentration
        print(f"\n  Final Four picks:")
        f4_teams = sorted(
            [(t, r["f4"]["count"]) for t, r in analysis["teams"].items() if r.get("f4", {}).get("count", 0) > 0],
            key=lambda x: -x[1],
        )
        for team, count in f4_teams[:10]:
            pct = (count / n) * 100
            print(f"    {team:6s}: {count:2d}/{n} ({pct:5.1f}%)")

        # Bias analysis
        biases = find_pool_bias(analysis)
        if biases:
            print(f"\n  Pool Bias Signals:")
            for b in biases:
                print(f"    {b['signal']:12s} {b['team']:6s} — champ={b['champ_pct']:.1f}%, F4={b['f4_pct']:.1f}%")
                print(f"                    → {b['strategy']}")

        # Concentration metric: HHI of champion picks
        champ_counts = [r.get("champ", {}).get("count", 0) for r in analysis["teams"].values()]
        champ_counts = [c for c in champ_counts if c > 0]
        if champ_counts:
            total = sum(champ_counts)
            hhi = sum((c / total) ** 2 for c in champ_counts)
            n_effective = 1 / hhi if hhi > 0 else 0
            print(f"\n  Champion concentration: HHI={hhi:.3f}, effective choices={n_effective:.1f}")
            if n_effective < 3:
                print(f"    → HIGHLY CONCENTRATED — strong contrarian opportunity")
            elif n_effective < 5:
                print(f"    → MODERATELY CONCENTRATED — some contrarian value")
            else:
                print(f"    → DISPERSED — less contrarian opportunity")

    # Cross-year patterns
    if not year_filter:
        print(f"\n{'='*60}")
        print(f"  Cross-Year Patterns")
        print(f"{'='*60}")

        all_champs = Counter()
        all_f4 = Counter()
        for year, yr_data in years.items():
            for b in yr_data.get("brackets", []):
                all_champs[b.get("champ", "")] += 1
                f4 = b.get("f4", [])
                if isinstance(f4, list):
                    for t in f4:
                        all_f4[t] += 1

        print(f"\n  Most-picked champions (all years):")
        for team, count in all_champs.most_common(10):
            print(f"    {team:6s}: {count} times")

        print(f"\n  Most-picked Final Four (all years):")
        for team, count in all_f4.most_common(10):
            print(f"    {team:6s}: {count} times")


def main():
    parser = argparse.ArgumentParser(description="Analyze pool pick bias")
    parser.add_argument("--year", type=str, help="Single year to analyze")
    args = parser.parse_args()

    pool_data = load_pool_data()
    print_report(pool_data, args.year)


if __name__ == "__main__":
    main()
