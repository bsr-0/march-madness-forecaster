"""Make ``tournament_context_{year}.json["teams"]`` agree with its ``["seeds"]`` block.

    python -m scripts.reconcile_tournament_context --years 2005,2008,2009,2019,2023
    python -m scripts.reconcile_tournament_context --years ... --write

Each context file carries the field twice: ``seeds.teams`` (the bracket as
scraped, with team_name/school_slug/seed/region) and ``teams.teams`` (the
loader's copy, with name/rating/conference). The loader reads the second.
Found 2026-09-11: in five seasons the two disagree, and in every case the
``seeds`` block is the real bracket --

  2005  Stanford/Mississippi State 8-9 swapped
  2008  Memphis (South #1) filed in USC's 6-Midwest slot; USC absent
  2009  BYU/Texas A&M 8-9 swapped
  2019  FDU and Prairie View (West 16 play-in) filed under East
  2023  College of Charleston seeded 13, not 12

2019 killed a nine-season walk-forward run four folds in ("Region West has
15 teams"): after the First Four losers were removed, West had no 16-seed.

This script takes seed/region from ``seeds.teams`` by team_id, adds any team
present only there (name from team_name, rating 1500.0, conference ""), and
reports what changed. 2026 uses a different id namespace in the two blocks
and is refused: it needs an id bridge, not a seed/region copy.

Dry-run by default; ``--write`` rewrites the file in place.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HIST = PROJECT_ROOT / "data" / "raw" / "historical"
REFUSED_YEARS = {2026}


def diff_blocks(ctx: Dict) -> Tuple[List[Tuple], List[Dict]]:
    """(changed, added): changed = (team_id, old (seed, region), new); added = seeds entries with no teams row."""
    teams = ctx["teams"]["teams"]
    seeds = ctx["seeds"]["teams"]
    by_id = {t["team_id"]: t for t in teams}
    changed, added = [], []
    for s in seeds:
        tid = s["team_id"]
        want = (int(s["seed"]), s["region"])
        if tid not in by_id:
            added.append(s)
            continue
        have = (int(by_id[tid]["seed"]), by_id[tid]["region"])
        if have != want:
            changed.append((tid, have, want))
    return changed, added


def reconcile(ctx: Dict) -> Tuple[List[Tuple], List[Dict]]:
    """Mutate ctx["teams"]["teams"] to agree with ctx["seeds"]["teams"]. Returns what changed."""
    changed, added = diff_blocks(ctx)
    by_id = {t["team_id"]: t for t in ctx["teams"]["teams"]}
    for tid, _have, (seed, region) in changed:
        by_id[tid]["seed"] = seed
        by_id[tid]["region"] = region
    for s in added:
        ctx["teams"]["teams"].append(
            {
                "name": s.get("team_name", s["team_id"]),
                "team_id": s["team_id"],
                "seed": int(s["seed"]),
                "region": s["region"],
                "conference": "",
                "rating": 1500.0,
                "school_slug": s.get("school_slug", ""),
            }
        )
    return changed, added


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", required=True, help="comma-separated seasons")
    ap.add_argument("--historical-dir", default=str(HIST))
    ap.add_argument("--write", action="store_true", help="rewrite the context files in place")
    args = ap.parse_args()

    hist = Path(args.historical_dir)
    rc = 0
    for year in (int(y) for y in args.years.split(",")):
        if year in REFUSED_YEARS:
            print(f"{year}: refused -- teams and seeds blocks use different team_id namespaces")
            rc = 1
            continue
        path = hist / f"tournament_context_{year}.json"
        ctx = json.loads(path.read_text())
        if not isinstance(ctx.get("teams"), dict) or not isinstance(ctx.get("seeds"), dict):
            print(f"{year}: no teams/seeds blocks, skipped")
            continue
        changed, added = reconcile(ctx)
        if not changed and not added:
            print(f"{year}: consistent")
            continue
        for tid, have, want in changed:
            print(f"{year}: {tid}: {have} -> {want}")
        for s in added:
            print(f"{year}: + {s['team_id']} ({s['seed']} {s['region']})")
        if args.write:
            path.write_text(json.dumps(ctx, indent=2))
            print(f"{year}: written")
        else:
            print(f"{year}: dry run, pass --write to apply")
    return rc


if __name__ == "__main__":
    sys.exit(main())
