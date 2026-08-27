#!/usr/bin/env python3
"""Assert the invariants a prediction must satisfy, on the real feature matrix.

WHY THIS EXISTS ALONGSIDE tests/test_calibration.js. That file already asserts
antisymmetry of the LINK: winProbFromMargin(m) + winProbFromMargin(-m) == 1.
That is necessary and not sufficient. The link can be perfectly antisymmetric
while the FEATURES fed to it are not, and then the board still contradicts
itself -- A beats B with probability 0.61 and B beats A with probability 0.44.

Every differenced column is antisymmetric for free: x = z(A) - z(B) negates
when the teams swap, so there is nothing to check. The venue columns are the
exception and the reason this file exists. Venue is a property of the GAME, not
of either team, so it is appended to x rather than differenced -- which means
its antisymmetry is a property of how it was CODED, not of the arithmetic. A
one-hot "is_home" column would break the invariant silently: swapping the teams
would leave it unchanged, the mirrored row would not be the negation of the
original, and the zero-intercept guarantee that makes margin(A,B) = -margin(B,A)
would quietly stop holding.

CHECKS
  1  every venue column is SIGNED, taking both +v and -v, never one-hot. A
     column that only ever takes 0 and 1 cannot negate under a team swap.
  2  the two teams' venue terms are exact negations of each other, checked by
     re-deriving them in both orientations on every real game.

     THE OBVIOUS VERSION OF THIS CHECK IS VACUOUS, which is worth recording
     because it looked convincing. "Refit on (-x, -m) and confirm the
     coefficients match" holds for ANY matrix by algebra --
     ((-X)'(-X))^-1 (-X)'(-m) reduces to (X'X)^-1 X'm -- so it passes a
     deliberately one-hot venue column without complaint. A check that cannot
     fail is worse than no check: it manufactures confidence. The real
     invariant has to re-derive the features in the swapped orientation, which
     is what this does.
  3  the matrix contains no NCAA tournament rows. Those are held out for
     evaluation, and a stray one would be trained on.
  4  tournament_venue() is zero and assert_neutral_for_prediction rejects a
     non-zero term. The venue term must contribute nothing on a neutral court,
     and the dangerous failure is OMISSION -- an absent column looks identical
     to a correctly-zeroed one in review and differs by about three points in
     every prediction.

Run: python3 scripts/assert_prediction_invariants.py
Exit 1 on any failure, so it can gate CI.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.data.features.venue import (  # noqa: E402
    assert_neutral_for_prediction,
    derive_home_cities,
    load_city_coords,
    load_game_cities,
    split_states,
    team_location,
    tournament_venue,
    travel_advantage,
    venue_for,
)

MATRIX = REPO / "docs" / "data" / "training_pit.json"
KAGGLE = REPO / "data" / "kaggle"
# venue_home / venue_host_city are the COURT terms: signed indicators that must
# be exactly zero on a neutral court. venue_travel is the PROXIMITY term, which
# is signed and antisymmetric like them but is legitimately non-zero on neutral
# courts -- an NCAA pod can land in a participant's back yard. Kept in separate
# tuples so a check never demands the wrong property of the wrong column.
COURT_KEYS = ("venue_home", "venue_host_city")
VENUE_KEYS = COURT_KEYS + ("venue_travel",)

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}")
    if detail:
        print(f"          {detail}")
    if not ok:
        failures.append(name)


def main() -> int:
    if not MATRIX.exists():
        print(f"missing {MATRIX}; run scripts/build_pit_training_matrix.py")
        return 1
    payload = json.loads(MATRIX.read_text())
    keys = payload["keys"]
    games = payload["games"]
    X = np.array([g["x"] for g in games], dtype=float)
    m = np.array([g["m"] for g in games], dtype=float)

    print(f"\n{len(games):,} rows x {len(keys)} features from {MATRIX.name}")

    # ---------------------------------------------------------------- 1
    print("\n1. venue columns are signed, not one-hot")
    missing = [k for k in VENUE_KEYS if k not in keys]
    if missing:
        check("venue columns present in keys", False, f"absent: {', '.join(missing)}")
    else:
        for k in VENUE_KEYS:
            col = X[:, keys.index(k)]
            pos, neg = int((col > 0).sum()), int((col < 0).sum())
            vals = sorted(set(col.tolist()))
            # A categorical column is worth printing in full; a continuous one
            # has tens of thousands of levels and printing them buries the
            # result it is supposed to show.
            shape = (
                f"values {vals}"
                if len(vals) <= 8
                else f"continuous, range [{col.min():.3f}, {col.max():.3f}]"
            )
            check(
                f"{k} takes both signs",
                pos > 0 and neg > 0,
                f"{pos:,} positive, {neg:,} negative, {shape}",
            )

    # ---------------------------------------------------------------- 2
    print("\n2. the two teams' venue terms negate under a swap")
    cities = load_game_cities()
    homes = derive_home_cities()
    coords = load_city_coords()
    _lc: dict = {}

    def loc(season, team):
        if (season, team) not in _lc:
            _lc[(season, team)] = team_location(season, team, homes, coords)
        return _lc[(season, team)]
    ct = set()
    ct_path = KAGGLE / "MConferenceTourneyGames.csv"
    if ct_path.exists():
        with open(ct_path) as f:
            for r in csv.DictReader(f):
                ct.add(
                    (int(r["Season"]), int(r["DayNum"]), int(r["WTeamID"]), int(r["LTeamID"]))
                )
    bad = 0
    tested = 0
    reg = KAGGLE / "MRegularSeasonCompactResults.csv"
    with open(reg) as f:
        for r in csv.DictReader(f):
            season = int(r["Season"])
            if season < 2010:
                continue
            key = (season, int(r["DayNum"]), int(r["WTeamID"]), int(r["LTeamID"]))
            ws, ls = venue_for(
                season, key[1], key[2], key[3], r["WLoc"], cities, homes, key in ct
            )
            wh, wc = split_states(ws)
            lh, lc = split_states(ls)
            # The travel term is appended to x alongside the court terms, so it
            # needs the same guarantee. It negates by construction --
            # (opp - team) flips when the teams swap -- but "by construction"
            # is exactly the assumption that has failed repeatedly here, so it
            # is asserted rather than argued.
            wt = travel_advantage(cities.get(key), loc(season, key[2]), loc(season, key[3]), coords)
            lt = travel_advantage(cities.get(key), loc(season, key[3]), loc(season, key[2]), coords)
            if abs(wt + lt) > 1e-9:
                bad += 1
            tested += 1
            # the loser's terms must be the exact negation of the winner's;
            # a one-hot coding gives (1, 0) and (0, 0), which is not.
            if (wh, wc) != (-lh, -lc):
                bad += 1
    check(
        "winner and loser venue terms are exact negations",
        bad == 0 and tested > 0,
        f"{tested:,} games re-derived in both orientations, {bad} asymmetric",
    )

    # ---------------------------------------------------------------- 3
    print("\n3. no NCAA tournament rows in the training matrix")
    ncaa = set()
    path = KAGGLE / "MNCAATourneyCompactResults.csv"
    if path.exists():
        with open(path) as f:
            for r in csv.DictReader(f):
                ncaa.add((int(r["Season"]), int(r["DayNum"])))
    stray = sum(1 for g in games if (g["y"], g.get("d")) in ncaa)
    max_day = max(g["d"] for g in games)
    check(
        "tournament games are held out for evaluation",
        stray == 0 and max_day < 133,
        f"{stray} stray rows; last boundary day {max_day} (Selection Sunday is 133)",
    )

    # ---------------------------------------------------------------- 4
    print("\n4. the venue term is zero on a neutral court")
    check("tournament_venue() is zero", tournament_venue() == 0)
    ok_zero = True
    try:
        assert_neutral_for_prediction([0, 0, 0])
    except AssertionError:
        ok_zero = False
    ok_raise = False
    try:
        assert_neutral_for_prediction([0, 1, 0])
    except AssertionError:
        ok_raise = True
    check(
        "assert_neutral_for_prediction accepts zeros and rejects non-zeros",
        ok_zero and ok_raise,
    )

    print()
    if failures:
        print(f"{len(failures)} INVARIANT(S) VIOLATED: {', '.join(failures)}")
        return 1
    print("all prediction invariants hold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
