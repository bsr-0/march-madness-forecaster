#!/usr/bin/env python3
"""Validate box-score-computed Four Factors against Torvik's published values.

Council gate: r >= 0.99 per feature, no systematic residual bias by team
profile (seed range, conference tier, tempo quartile). This is the
foundational check before any backtest using Torvik FF data.

Usage:
    python scripts/validate_four_factors.py --year 2024
    python scripts/validate_four_factors.py                # all LOYO years
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.features.proprietary_metrics import (
    ProprietaryMetricsEngine,
    team_games_to_game_records,
)
from src.data.team_name_resolver import TeamNameResolver

_resolver = TeamNameResolver()

FF_FIELDS = [
    "effective_fg_pct",
    "turnover_rate",
    "offensive_reb_rate",
    "free_throw_rate",
    "opp_effective_fg_pct",
    "opp_turnover_rate",
    "defensive_reb_rate",
    "opp_free_throw_rate",
]

FF_LABELS = [
    "eFG%",
    "TO%",
    "ORB%",
    "FTR",
    "Opp eFG%",
    "Opp TO%",
    "DRB%",
    "Opp FTR",
]

POWER_CONFERENCES = {"ACC", "B10", "B12", "BE", "SEC", "P12", "MWC"}

HIST_DIR = ROOT / "data" / "raw" / "historical"


def _pearson_r(xs: List[float], ys: List[float]) -> float:
    n = len(xs)
    if n < 3:
        return float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx < 1e-12 or sy < 1e-12:
        return float("nan")
    return cov / (sx * sy)


def _mae(xs: List[float], ys: List[float]) -> float:
    return sum(abs(x - y) for x, y in zip(xs, ys)) / len(xs)


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def load_torvik_ff(year: int) -> Dict[str, dict]:
    """Load pre-tournament Torvik FF + metadata per team."""
    path = HIST_DIR / f"torvik_{year}.json"
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    result = {}
    for t in data.get("teams", []):
        tid = t.get("team_id", "")
        if not tid:
            continue
        ff = {field: t.get(field, 0.0) for field in FF_FIELDS}
        # Skip teams with all-zero FF (data quality issue)
        if all(abs(ff[f]) < 1e-6 for f in FF_FIELDS[:4]):
            continue
        ff["conference"] = t.get("conference", "")
        ff["adj_tempo"] = t.get("adj_tempo", 68.0)
        result[tid] = ff
    return result


def compute_boxscore_ff(year: int, cutoff_date: str) -> Dict[str, dict]:
    """Compute box-score FF from historical games at the given cutoff."""
    games_path = HIST_DIR / f"historical_games_{year}.json"
    if not games_path.exists():
        return {}
    with open(games_path) as f:
        payload = json.load(f)
    team_games = payload.get("team_games", [])
    if not team_games:
        return {}

    game_records = team_games_to_game_records(team_games, year)
    engine = ProprietaryMetricsEngine(require_cutoff_date=True)
    metrics = engine.compute(game_records, cutoff_date=cutoff_date)

    # Count games per raw team_id so we can break resolver collisions in
    # favour of the team with the most games — the "real" NCAA Division-I
    # entry, not a small D3/affiliate team that happens to share a name
    # prefix.
    from collections import defaultdict

    games_per_tid: Dict[str, int] = defaultdict(int)
    for rec in game_records:
        if getattr(rec, "has_box_score", True):
            games_per_tid[rec.team_id] += 1

    # Iterate in descending game-count order so that when two raw ids
    # resolve to the same canonical, the larger-sample team wins.
    # Otherwise: e.g. `vermont_state_lyndon_hornets` (1 game) was
    # overwriting `vermont_catamounts` (35 games) because both resolve
    # to `vermont` via the resolver's `containment` method. See
    # COUNCIL_LESSONS.md §2 O2 / O2a.
    sorted_tids = sorted(
        metrics.keys(),
        key=lambda t: (-games_per_tid.get(t, 0), t),
    )

    result: Dict[str, dict] = {}
    for tid in sorted_tids:
        m = metrics[tid]
        ff = {field: getattr(m, field, 0.0) for field in FF_FIELDS}
        if all(abs(ff[f]) < 1e-6 for f in FF_FIELDS[:4]):
            continue
        # Resolve mascot-suffixed IDs (e.g. "akron_zips" -> "akron") ONLY
        # via deterministic high-confidence methods. The resolver's
        # fuzzy `containment` method (conf ≈ 0.9) collapses unrelated
        # teams: `tennessee_wesleyan_bulldogs` -> `tennessee` would
        # silently overwrite the real Tennessee Volunteers otherwise.
        match = _resolver.resolve(tid.replace("_", " "))
        if match is not None and match.method in ("exact_id", "prefix_strip"):
            canonical = match.canonical_id
        else:
            canonical = tid
        # Belt-and-suspenders: first writer wins within this
        # game-count-sorted iteration. A subsequent smaller-sample team
        # whose canonical collides with one already placed is ignored.
        if canonical not in result:
            result[canonical] = ff
    return result


def get_tournament_seeds(year: int) -> Dict[str, int]:
    """Load tournament seeds from teams file."""
    path = HIST_DIR / f"teams_{year}.json"
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    return {t["team_id"]: t.get("seed", 0) for t in data.get("teams", []) if t.get("team_id")}


def get_cutoff_date(year: int) -> Optional[str]:
    """Get the pre-tournament cutoff date from Torvik metadata."""
    path = HIST_DIR / f"torvik_{year}.json"
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    return data.get("cutoff_date")


def validate_year(year: int) -> Optional[dict]:
    """Run the full four-factor validation for one year."""
    cutoff = get_cutoff_date(year)
    if not cutoff:
        print(f"  {year}: No cutoff_date in torvik_{year}.json — skipping")
        return None

    torvik_ff = load_torvik_ff(year)
    if not torvik_ff:
        print(f"  {year}: No Torvik FF data — skipping")
        return None

    boxscore_ff = compute_boxscore_ff(year, cutoff)
    if not boxscore_ff:
        print(f"  {year}: No box-score FF data — skipping")
        return None

    # Inner join on team_id
    common = sorted(set(torvik_ff) & set(boxscore_ff))
    seeds = get_tournament_seeds(year)
    tournament_teams = set(seeds.keys())

    # Prefer tournament teams, fall back to all common teams
    if len(tournament_teams & set(common)) >= 50:
        common = sorted(tournament_teams & set(common))

    if len(common) < 30:
        print(f"  {year}: Only {len(common)} common teams — skipping")
        return None

    # Compute per-feature correlations
    feature_results = {}
    for field, label in zip(FF_FIELDS, FF_LABELS):
        tv = [torvik_ff[t][field] for t in common]
        bs = [boxscore_ff[t][field] for t in common]
        r = _pearson_r(tv, bs)
        r2 = r**2 if not math.isnan(r) else float("nan")
        mae = _mae(tv, bs)
        residuals = [bs[i] - tv[i] for i in range(len(common))]
        mean_resid = _mean(residuals)
        passed = not math.isnan(r) and r >= 0.99
        feature_results[field] = {
            "label": label,
            "r": r,
            "r2": r2,
            "mae": mae,
            "mean_residual": mean_resid,
            "passed": passed,
        }

    # Stratified analysis
    strata_results = {}

    # By seed range
    seed_bins = {"1-4": (1, 4), "5-8": (5, 8), "9-12": (9, 12), "13-16": (13, 16)}
    for bin_name, (lo, hi) in seed_bins.items():
        subset = [t for t in common if lo <= seeds.get(t, 0) <= hi]
        if len(subset) < 5:
            continue
        stratum_data = {}
        for field, label in zip(FF_FIELDS, FF_LABELS):
            residuals = [boxscore_ff[t][field] - torvik_ff[t][field] for t in subset]
            stratum_data[field] = {
                "n": len(subset),
                "mean_residual": _mean(residuals),
                "mae": _mean([abs(r) for r in residuals]),
            }
        strata_results[f"seed_{bin_name}"] = stratum_data

    # By conference tier
    for tier_name, tier_set in [("power", POWER_CONFERENCES), ("mid", None)]:
        if tier_set:
            subset = [t for t in common if torvik_ff[t]["conference"] in tier_set]
        else:
            subset = [t for t in common if torvik_ff[t]["conference"] not in POWER_CONFERENCES]
        if len(subset) < 5:
            continue
        stratum_data = {}
        for field, label in zip(FF_FIELDS, FF_LABELS):
            residuals = [boxscore_ff[t][field] - torvik_ff[t][field] for t in subset]
            stratum_data[field] = {
                "n": len(subset),
                "mean_residual": _mean(residuals),
                "mae": _mean([abs(r) for r in residuals]),
            }
        strata_results[f"conf_{tier_name}"] = stratum_data

    # By tempo quartile
    tempos = sorted([(torvik_ff[t]["adj_tempo"], t) for t in common])
    q_size = len(tempos) // 4
    if q_size >= 5:
        for qi, q_label in enumerate(["slow", "med_slow", "med_fast", "fast"]):
            start = qi * q_size
            end = len(tempos) if qi == 3 else (qi + 1) * q_size
            subset = [t for _, t in tempos[start:end]]
            stratum_data = {}
            for field, label in zip(FF_FIELDS, FF_LABELS):
                residuals = [boxscore_ff[t][field] - torvik_ff[t][field] for t in subset]
                stratum_data[field] = {
                    "n": len(subset),
                    "mean_residual": _mean(residuals),
                    "mae": _mean([abs(r) for r in residuals]),
                }
            strata_results[f"tempo_{q_label}"] = stratum_data

    # Check for systematic bias in any stratum
    bias_flags = []
    for stratum_name, stratum_data in strata_results.items():
        for field, metrics in stratum_data.items():
            if abs(metrics["mean_residual"]) > 0.01:
                label = FF_LABELS[FF_FIELDS.index(field)]
                bias_flags.append(f"{stratum_name}/{label}: bias={metrics['mean_residual']:+.4f}")

    return {
        "year": year,
        "n_teams": len(common),
        "cutoff_date": cutoff,
        "features": feature_results,
        "strata": strata_results,
        "bias_flags": bias_flags,
    }


def print_year_report(result: dict) -> bool:
    """Print formatted report for one year. Returns True if all gates passed."""
    year = result["year"]
    n = result["n_teams"]
    cutoff = result["cutoff_date"]
    features = result["features"]
    bias_flags = result["bias_flags"]

    all_passed = all(f["passed"] for f in features.values())
    no_bias = len(bias_flags) == 0

    print(f"\n{'=' * 70}")
    print(f"  YEAR {year}  |  {n} teams  |  cutoff={cutoff}")
    print(f"{'=' * 70}")
    print(f"  {'Feature':<12s} {'r':>8s} {'R²':>8s} {'MAE':>8s} {'Bias':>9s} {'Gate':>6s}")
    print(f"  {'-' * 12} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 9} {'-' * 6}")

    for field in FF_FIELDS:
        f = features[field]
        gate_str = "PASS" if f["passed"] else "FAIL"
        print(
            f"  {f['label']:<12s} {f['r']:>8.5f} {f['r2']:>8.5f} "
            f"{f['mae']:>8.5f} {f['mean_residual']:>+9.5f} {gate_str:>6s}"
        )

    # Stratified summary
    strata = result["strata"]
    if strata:
        print(f"\n  Stratified residuals (bias > 0.01 = FLAG):")
        for stratum_name, stratum_data in sorted(strata.items()):
            flagged_fields = []
            for field, metrics in stratum_data.items():
                if abs(metrics["mean_residual"]) > 0.01:
                    label = FF_LABELS[FF_FIELDS.index(field)]
                    flagged_fields.append(f"{label}={metrics['mean_residual']:+.4f}")
            n_teams = next(iter(stratum_data.values()))["n"]
            status = ", ".join(flagged_fields) if flagged_fields else "clean"
            print(f"    {stratum_name:<18s} (n={n_teams:>3d}): {status}")

    # Overall verdict
    if all_passed and no_bias:
        print(f"\n  VERDICT: ALL GATES PASSED")
    else:
        if not all_passed:
            failed = [f["label"] for f in features.values() if not f["passed"]]
            print(f"\n  VERDICT: CORRELATION GATE FAILED for: {', '.join(failed)}")
        if not no_bias:
            print(f"  BIAS FLAGS ({len(bias_flags)}):")
            for flag in bias_flags[:10]:
                print(f"    - {flag}")

    return all_passed and no_bias


def main():
    parser = argparse.ArgumentParser(description="Validate box-score FF against Torvik FF")
    parser.add_argument("--year", type=int, help="Single year (default: all LOYO years)")
    args = parser.parse_args()

    if args.year:
        years = [args.year]
    else:
        years = [
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

    print("Four-Factor Correlation Validation")
    print(f"Council gate: r >= 0.99 per feature, |bias| < 0.01 per stratum")
    print(f"Years: {years}")

    results = []
    for year in years:
        result = validate_year(year)
        if result:
            results.append(result)

    if not results:
        print("\nNo valid results. Check data availability.")
        return 1

    all_ok = True
    for result in results:
        passed = print_year_report(result)
        if not passed:
            all_ok = False

    # Summary table across years
    if len(results) > 1:
        print(f"\n{'=' * 70}")
        print(f"  SUMMARY ACROSS {len(results)} YEARS")
        print(f"{'=' * 70}")
        print(f"  {'Feature':<12s} {'Mean r':>8s} {'Min r':>8s} {'Mean MAE':>9s} {'Pass%':>7s}")
        print(f"  {'-' * 12} {'-' * 8} {'-' * 8} {'-' * 9} {'-' * 7}")
        for field, label in zip(FF_FIELDS, FF_LABELS):
            rs = [r["features"][field]["r"] for r in results]
            maes = [r["features"][field]["mae"] for r in results]
            passes = sum(1 for r in results if r["features"][field]["passed"])
            print(
                f"  {label:<12s} {_mean(rs):>8.5f} {min(rs):>8.5f} {_mean(maes):>9.5f} {passes:>3d}/{len(results):<3d}"
            )

    if all_ok:
        print(f"\nALL GATES PASSED across {len(results)} years.")
    else:
        print(f"\nSOME GATES FAILED. Review per-year reports above.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
