#!/usr/bin/env python3
"""O4 — re-serialize the opponent-independence test for lock-testing.

COUNCIL_LESSONS.md §2 O4 was closed 2026-04-13 with the narrative
`ANALYSIS_O4_OPPONENT_CORRELATION.md`, but the underlying numbers
live only in that markdown file. This script re-runs the
`independence_test` over every year in `pool_hist_results.json` and
serializes the z-score, observed mean ρ, and simulated null mean
into `artifacts/o4_opponent_independence_2026-04-14.json` so
`tests/test_opponent_independence_lock.py` can assert durable
invariants (pooled z < 0, per-year independence or less-clustered).

Usage:
    python scripts/o4_independence_lock.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.pool_multiyear_analysis import (
    analyze_year,
    load_actual_results,
)


def main() -> int:
    pool_path = PROJECT_ROOT / "pool_hist_results.json"
    with open(pool_path) as f:
        pool = json.load(f)

    years = sorted(pool["years"].keys())
    per_year: dict[str, dict] = {}

    for yr_str in years:
        year = int(yr_str)
        yr_data = pool["years"][yr_str]
        try:
            actual = load_actual_results(year)
        except FileNotFoundError:
            print(f"  {year}: no results, skipping")
            continue
        summary = analyze_year(year, yr_data, actual)
        indep = summary["independence"]
        per_year[yr_str] = {
            "year": year,
            "n_brackets": summary["n_brackets"],
            "observed_mean": indep["observed_mean"],
            "simulated_mean": indep["simulated_mean"],
            "excess": indep["excess"],
            "z_score": indep["z_score"],
            "verdict": (
                "less_correlated_than_independence"
                if indep["z_score"] < -1.96
                else "independence_holds"
                if indep["z_score"] < 1.96
                else "more_correlated_than_independence"
            ),
        }

    # Pooled z = sum(z_i) / sqrt(n)  (Stouffer)
    zs = [r["z_score"] for r in per_year.values()]
    pooled_z = sum(zs) / math.sqrt(len(zs)) if zs else float("nan")

    out = {
        "per_year": per_year,
        "pooled_stouffer_z": pooled_z,
        "n_years": len(per_year),
        "source_file": str(pool_path.relative_to(PROJECT_ROOT)),
        "pool": pool.get("pool"),
        "group_id": pool.get("groupId"),
        "rng_seed": 42,
        "n_sim_per_year": 2000,
        "note": (
            "Re-serialization of the independence test behind O4. "
            "Stouffer pooled z < 0 means brackets are LESS correlated "
            "than IID draws from the empirical marginals — the core "
            "O4 closure invariant."
        ),
    }
    out_path = PROJECT_ROOT / "artifacts" / "o4_opponent_independence_2026-04-14.json"
    out_path.write_text(json.dumps(out, indent=2, default=float))

    print(f"\n{'Year':<6} {'K':<4} {'obs ρ':>8} {'sim ρ':>8} {'excess':>9} {'z':>7} verdict")
    print("-" * 72)
    for yr_str, r in sorted(per_year.items()):
        print(
            f"{yr_str:<6} {r['n_brackets']:<4} {r['observed_mean']:8.4f} "
            f"{r['simulated_mean']:8.4f} {r['excess']:+9.4f} {r['z_score']:+7.2f} "
            f"{r['verdict']}"
        )
    print(f"\nPooled Stouffer z = {pooled_z:+.2f} (n={len(zs)} years)")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
