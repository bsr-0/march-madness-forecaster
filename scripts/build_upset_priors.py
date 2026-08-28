#!/usr/bin/env python3
"""Historical upset base rates by seed matchup, for the UI's prior blend.

WHAT THIS IS. For each (round, better seed, worse seed) the share of games the
WORSE seed won -- 5v12 at .397, 6v11 at .517 across 2010-2025. A user can blend
these into the model's probability, so the board can be pulled toward what
history says about a seed pairing rather than only what the variables say about
the two teams.

COMPUTED WALK-FORWARD, WHICH IS THE WHOLE DIFFICULTY. A base rate built from
all seasons and then applied to 2024 has 2024's own results inside it. That is
the same leak as fitting on the season you are predicting, just wearing
different clothes, and it would flatter the blend exactly where a user would be
looking. So the table for season Y is built from seasons STRICTLY BEFORE Y, and
the earliest displayed seasons therefore rest on fewer years.

SPARSE CELLS ARE SHRUNK, NOT TRUSTED. There are 96 (round, seed-pair) cells and
only 24 carry 8 or more games; a 1v16 in the Elite 8 has never happened and
several cells hold a single game. A raw rate there is noise presented as
history. Each cell is therefore shrunk toward its ROUND's overall upset rate
with weight n/(n + PRIOR_STRENGTH), so a thin cell reports approximately the
round rate and a well-observed one reports itself. The same shape as the
calibration shrinkage in model_baseline.

Seeds come from MNCAATourneySeeds; rounds from the Kaggle day numbers. First
Four games (days 134-135) are excluded -- they are play-in games between teams
on the same seed line, so "upset" is undefined for them.

Run: python3 scripts/build_upset_priors.py
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

KAGGLE = REPO / "data" / "kaggle"
OUT = REPO / "docs" / "data" / "upset_priors.json"

# Seasons the UI can display. Matches build_ui_payload.SEASONS.
UI_SEASONS = [2024, 2025, 2026, 2027]

# Games-equivalent weight given to the round rate when shrinking a cell. At 8,
# a cell with 8 games sits halfway between its own rate and its round's.
PRIOR_STRENGTH = 8

DAY_TO_ROUND = {
    136: "R64", 137: "R64",
    138: "R32", 139: "R32",
    143: "S16", 144: "S16",
    145: "E8", 146: "E8",
    152: "F4",
    154: "NCG",
}


def load_seeds() -> dict:
    out = {}
    with open(KAGGLE / "MNCAATourneySeeds.csv") as f:
        for r in csv.DictReader(f):
            try:
                out[(int(r["Season"]), int(r["TeamID"]))] = int(r["Seed"][1:3])
            except (ValueError, IndexError):
                continue
    return out


def load_games(seeds: dict):
    """(season, round, better_seed, worse_seed, worse_seed_won)."""
    out = []
    with open(KAGGLE / "MNCAATourneyCompactResults.csv") as f:
        for r in csv.DictReader(f):
            rnd = DAY_TO_ROUND.get(int(r["DayNum"]))
            if rnd is None:
                continue  # First Four: same seed line, "upset" undefined
            s = int(r["Season"])
            w, l = int(r["WTeamID"]), int(r["LTeamID"])
            sw, sl = seeds.get((s, w)), seeds.get((s, l))
            if sw is None or sl is None or sw == sl:
                continue
            better, worse = min(sw, sl), max(sw, sl)
            out.append((s, rnd, better, worse, 1 if sw == worse else 0))
    return out


def main() -> int:
    seeds = load_seeds()
    games = load_games(seeds)
    if not games:
        print("no tournament games found")
        return 1

    payload = {}
    for season in UI_SEASONS:
        prior_games = [g for g in games if g[0] < season]
        if not prior_games:
            continue

        cell = defaultdict(lambda: [0, 0])
        rnd = defaultdict(lambda: [0, 0])
        for _s, r, b, w, up in prior_games:
            cell[(r, b, w)][0] += up
            cell[(r, b, w)][1] += 1
            rnd[r][0] += up
            rnd[r][1] += 1

        round_rate = {r: u / n for r, (u, n) in rnd.items() if n}
        table = {}
        for (r, b, w), (u, n) in cell.items():
            base = round_rate.get(r, 0.5)
            weight = n / (n + PRIOR_STRENGTH)
            table.setdefault(r, {})[f"{b}-{w}"] = {
                "p": round(weight * (u / n) + (1 - weight) * base, 4),
                "n": n,
            }

        payload[str(season)] = {
            "rounds": {r: round(v, 4) for r, v in round_rate.items()},
            "cells": table,
            "seasons_used": sorted({g[0] for g in prior_games}),
        }

    OUT.write_text(json.dumps(payload, separators=(",", ":")))
    print(f"\nwrote {OUT.relative_to(REPO)}  ({OUT.stat().st_size / 1024:.1f} KB)")
    for s in sorted(payload):
        p = payload[s]
        yrs = p["seasons_used"]
        n_cells = sum(len(v) for v in p["cells"].values())
        print(f"  {s}: {len(yrs)} prior seasons ({min(yrs)}-{max(yrs)}), {n_cells} cells")
    ex = payload.get("2026")
    if ex:
        print("\n  2026 R64, shrunk (raw n in brackets):")
        for k, v in sorted(ex["cells"].get("R64", {}).items(), key=lambda kv: int(kv[0].split("-")[0])):
            print(f"    {k:<6} {v['p']:.3f}  [n={v['n']}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
