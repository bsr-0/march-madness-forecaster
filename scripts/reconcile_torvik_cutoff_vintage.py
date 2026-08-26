#!/usr/bin/env python3
"""Bring teams[] into agreement with the season-cutoff snapshot in torvik_*.json.

WHY THIS IS NEEDED ONCE. rescrape_pretournament_torvik._write_monthly_ff used
to update only four_factors_snapshots, never teams[]. A later re-scrape
therefore rewrote the dated snapshots with a fresh pull while leaving teams[]
on whatever vintage the original base scrape captured. Torvik revises ratings
for past date windows, so the two surfaces drifted apart while continuing to
carry the same date label -- the divergence audit_snapshot_boundary check B
reports. The writer now updates both surfaces from one fetch, which prevents
this going forward but does not repair files already on disk. This script does
that repair, applying exactly the rule the writer now applies.

The cutoff snapshot is treated as authoritative because it is the fresher pull
and because it is the surface the dated series ends on; leaving teams[] as the
odd one out would put a discontinuity at the season boundary, where
regular-season rows read snapshots and tournament rows read teams[].

Zero values are skipped for the same reason as in the writer: none of these
fields is ever legitimately 0.0, so a zero means the column was missing from
that CSV layout rather than measured as zero.

Run: python3 scripts/reconcile_torvik_cutoff_vintage.py          # dry run
     python3 scripts/reconcile_torvik_cutoff_vintage.py --apply
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
HIST = REPO / "data" / "raw" / "historical"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true", help="write changes (default: dry run)")
    args = ap.parse_args()

    paths = sorted(HIST.glob("torvik_*.json"))
    if not paths:
        print(f"no torvik_*.json under {HIST}")
        return 1

    grand_fields = 0
    grand_teams = 0
    changed_files = 0

    for path in paths:
        data = json.loads(path.read_text())
        cutoff = data.get("cutoff_date")
        snaps = data.get("four_factors_snapshots") or []
        if not cutoff or not snaps:
            continue
        final = next((s for s in snaps if s["date"] == cutoff), None)
        if final is None:
            print(f"  {path.name}: no snapshot at cutoff {cutoff} -- skipped")
            continue

        payload = final["data"]
        touched_fields = 0
        touched_teams = 0
        for team in data.get("teams", []):
            snap = payload.get(team.get("team_id"))
            if not isinstance(snap, dict):
                continue
            hit = False
            for field, value in snap.items():
                if not isinstance(value, (int, float)) or value == 0.0:
                    continue
                if not isinstance(team.get(field), (int, float)) or abs(team[field] - value) > 1e-9:
                    team[field] = value
                    touched_fields += 1
                    hit = True
            touched_teams += hit

        if touched_fields:
            changed_files += 1
            grand_fields += touched_fields
            grand_teams += touched_teams
            print(f"  {path.name}: {touched_fields:5} fields across {touched_teams:3} teams")
            if args.apply:
                path.write_text(json.dumps(data, indent=2))

    verb = "updated" if args.apply else "would update"
    print(f"\n{verb} {grand_fields:,} field values across {grand_teams:,} team entries in {changed_files} files")
    if not args.apply:
        print("dry run -- re-run with --apply to write")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
