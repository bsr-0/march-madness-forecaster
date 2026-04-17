"""Compute candidate regular-season chaos indicators per year and z-score 2026.

Reads pre-tournament Torvik snapshots from data/raw/historical/torvik_<year>.json
and computes:
  - top1_barthag        : best team's barthag (low = no juggernaut)
  - top10_mean          : mean barthag of top 10
  - top10_spread        : barthag range across top 10 (low = compressed elite)
  - top25_std           : std of barthag across top 25 (low = parity)
  - gap_1_to_25         : barthag(#1) - barthag(#25) (low = compressed)
  - elite_count_p90     : number of teams with barthag >= 0.90
  - top10_aem_mean      : mean adj_off - adj_def among top 10
  - top10_aem_std       : std of top-10 adj efficiency margin

Then z-scores 2026 against 2008-2025 (ex. 2020).
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

DATA_DIR = Path("data/raw/historical")
YEARS = list(range(2008, 2027))
EXCLUDE = {2020}  # no tournament


def load_year(year: int) -> list[dict]:
    p = DATA_DIR / f"torvik_{year}.json"
    if not p.exists():
        return []
    with p.open() as f:
        doc = json.load(f)
    return doc.get("teams", [])


def metrics(teams: list[dict]) -> dict:
    barthag = sorted(
        (t["barthag"] for t in teams if t.get("barthag") is not None),
        reverse=True,
    )
    aem = []
    teams_by_bt = sorted(
        (t for t in teams if t.get("barthag") is not None),
        key=lambda t: t["barthag"],
        reverse=True,
    )
    for t in teams_by_bt[:10]:
        o = t.get("adj_offensive_efficiency")
        d = t.get("adj_defensive_efficiency")
        if o is not None and d is not None:
            aem.append(o - d)

    if len(barthag) < 25:
        return {}

    top10 = barthag[:10]
    top25 = barthag[:25]
    return {
        "n_teams": len(barthag),
        "top1_barthag": top10[0],
        "top10_mean": statistics.fmean(top10),
        "top10_spread": top10[0] - top10[-1],
        "top25_std": statistics.pstdev(top25),
        "gap_1_to_25": top25[0] - top25[-1],
        "elite_count_p90": sum(1 for b in barthag if b >= 0.90),
        "top10_aem_mean": statistics.fmean(aem) if aem else None,
        "top10_aem_std": statistics.pstdev(aem) if len(aem) >= 2 else None,
    }


def zscore(x: float, sample: list[float]) -> float:
    mu = statistics.fmean(sample)
    sd = statistics.pstdev(sample)
    if sd == 0:
        return float("nan")
    return (x - mu) / sd


def main() -> None:
    rows: dict[int, dict] = {}
    for y in YEARS:
        if y in EXCLUDE:
            continue
        teams = load_year(y)
        m = metrics(teams)
        if m:
            rows[y] = m

    if 2026 not in rows:
        raise SystemExit("2026 metrics not computed (missing data?)")

    keys = [
        "top1_barthag",
        "top10_mean",
        "top10_spread",
        "top25_std",
        "gap_1_to_25",
        "elite_count_p90",
        "top10_aem_mean",
        "top10_aem_std",
    ]

    hist_years = [y for y in rows if y != 2026]
    print(f"Years in baseline: {sorted(hist_years)} (n={len(hist_years)})")
    print()
    header = f"{'year':>4} | " + " | ".join(f"{k:>16}" for k in keys)
    print(header)
    print("-" * len(header))
    for y in sorted(rows):
        vals = " | ".join(
            f"{rows[y][k]:>16.4f}" if isinstance(rows[y][k], float) else f"{rows[y][k]:>16d}" for k in keys
        )
        marker = "  <-- 2026" if y == 2026 else ""
        print(f"{y:>4} | {vals}{marker}")

    print()
    print("=== 2026 vs 2008-2025 baseline (z-scores; |z|>=1.5 flagged) ===")
    for k in keys:
        sample = [rows[y][k] for y in hist_years if rows[y].get(k) is not None]
        x = rows[2026][k]
        z = zscore(x, sample)
        mu = statistics.fmean(sample)
        sd = statistics.pstdev(sample)
        flag = "  *** FLAG" if abs(z) >= 1.5 else ""
        # rank percentile
        rank = sum(1 for v in sample if v < x)
        pct = 100 * rank / len(sample)
        print(f"{k:>18} : 2026={x:8.4f}  hist mean={mu:7.4f} std={sd:6.4f}  z={z:+.2f}  pct={pct:5.1f}{flag}")

    print()
    print("Interpretation guide:")
    print("  upset-favorable signal = LOW top1_barthag, LOW top10_mean, LOW top10_spread,")
    print("    LOW top25_std, LOW gap_1_to_25, LOW elite_count_p90, LOW top10_aem_mean")


if __name__ == "__main__":
    main()
