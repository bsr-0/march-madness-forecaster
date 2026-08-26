#!/usr/bin/env python3
"""Build a per-date training matrix from regular-season games.

WHAT THIS ADDS. build_training_matrix emits one row per tournament game, all
predicted from the Selection Sunday boundary -- 1,008 rows total. That is
correct but small. Every regular-season game is also a labelled observation of
"these two teams, this quality gap, this margin", and there are roughly 5,000
per season. This builds those rows, each predicted from features as they stood
BEFORE the game was played.

THE RULE EVERY ROW OBEYS. A game played on day D is predicted from the most
recent week boundary strictly before D, and every feature at that boundary is
computed only from games strictly before it. No feature in a row has seen the
game it is predicting, or any game after it. That is the whole point; the
alternative is a matrix that looks excellent and means nothing.

WHERE THE JOINS CAN LEAK, AND HOW EACH IS HANDLED
  Torvik snapshots are calendar dates, Kaggle games are day numbers. They are
  joined through MSeasons.DayZero, and only a snapshot dated STRICTLY BEFORE
  the boundary is eligible. Taking the nearest snapshot in either direction --
  the obvious implementation -- would pull March ratings into a February row.

  Opponent-dependent features (sos_avg_opp_barthag, losses_to_weaker_rate) use
  the dated barthag from that same eligible snapshot, never a final rating.
  point_in_time_kaggle requires these as arguments precisely so this cannot be
  skipped by accident.

  Standardisation is done within (season, boundary) over the teams present at
  that boundary. Standardising within a season as a whole would leak the
  season's final distribution into a November row.

THE CONNECTIVITY GATE IS A FILTER HERE, NOT A WARNING. Before any boundary is
used, the game graph must be connected enough for opponent-adjusted ratings to
mean anything. In a normal season this passes by late November. In 2021 it
does not pass until January, and rows before that are DROPPED rather than
down-weighted: an SRS on a disconnected graph is not noisy, it is undefined,
and no amount of shrinkage repairs an undefined quantity.

Rows are oriented by canonical team id, which is a pre-game fact independent of
the result, so the target cannot encode the winner. The caller must still drop
the season being predicted before fitting -- same rule as training.json.

Run: python3 scripts/build_pit_training_matrix.py [--out PATH]
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.build_ui_payload import VARIABLES, zscores  # noqa: E402
from src.data.features.custom_ratings import ratings_to_canonical  # noqa: E402
from src.data.features.point_in_time_kaggle import (  # noqa: E402
    ADJUSTABLE_RATES,
    adjusted_rates,
    box_profile,
    detailed_before,
    form_stats,
    load_detailed_games,
    losses_to_weaker_rate,
    rank_from_rating,
    season_is_complete,
    strength_of_schedule,
)
from src.data.features.venue import (  # noqa: E402
    derive_home_cities,
    load_game_cities,
    split_states,
    venue_for,
)
from src.data.features.point_in_time_ratings import (  # noqa: E402
    SELECTION_SUNDAY_DAY,
    game_counts,
    games_before,
    largest_component_share,
    load_season_games,
    shrink_to_prior,
    srs,
)

KAGGLE = REPO / "data" / "kaggle"
HIST = REPO / "data" / "raw" / "historical"
DEFAULT_OUT = REPO / "docs" / "data" / "training_pit.json"

# Week boundaries, in Kaggle day numbers. Ends before Selection Sunday so no
# tournament game is ever a row here.
#
# STARTS AT 28, NOT 21, AND THE REASON IS A MEASURED UNCERTAINTY. Torvik's
# adjusted ratings are demonstrably bounded by their own date from day 28
# onward: they correlate with a known-bounded same-date SRS at 0.9498 against
# 0.9126 for full-season SRS, on 44 of 47 snapshots tested. All three failures
# were the single EARLIEST snapshot (day ~21) of 2015, 2016 and 2017, where
# the ratings resembled the full season slightly more (by <= 0.035). The benign
# explanation -- Torvik blending a preseason prior, which is legitimately known
# in advance -- was tested and rejected: a prior-blended estimate correlates
# worse (0.8580) than either alternative. In the six most recent seasons the
# day-21 snapshot is clearly bounded, so this is a small anomaly in old seasons
# rather than a demonstrated leak, and it is not currently distinguishable from
# noise at n=3.
#
# Dropping the boundary costs 1,813 rows, 4.2% of the matrix, and those are the
# rows where SRS is noisiest and shrinkage is carrying the most weight -- the
# least informative rows in the set. Cheap insurance against the one date this
# pipeline cannot certify. Independently justified: day 21 is roughly week 3,
# and point-in-time SRS does not beat a prior-season rating until week 5
# (validate_point_in_time_srs.py).
FIRST_BOUNDARY = 28
BOUNDARY_STEP = 7

# Minimum share of the season's D1 field in the largest connected component.
# Normal seasons clear this by late November on 3-5 games per team; 2021 does
# not clear it until January. Set from the measured distribution, not tuned.
CONNECTIVITY_FLOOR = 0.90

# Variables reconstructible point-in-time today. Mirrors audit_snapshot_boundary
# check E; anything outside this set has no dated value and would have to be
# filled with a season-final number, which is the leak this file exists to avoid.
TORVIK_DATED = {
    "barthag",
    "adj_offensive_efficiency",
    "adj_defensive_efficiency",
    "adj_tempo",
    "effective_fg_pct",
    "turnover_rate",
    "offensive_reb_rate",
    "free_throw_rate",
    "opp_effective_fg_pct",
    "opp_turnover_rate",
    "defensive_reb_rate",
    "opp_free_throw_rate",
}
KAGGLE_DERIVED = {
    "three_pt_rate",
    "three_pt_pct",
    "opp_three_pt_pct",
    "ast_to_ratio",
    "havoc_rate",
    "reg_season_margin_avg",
    "reg_season_margin_std",
    "close_game_win_rate",
    "true_road_win_pct",
}
OPPONENT_DEPENDENT = {"sos_avg_opp_barthag", "losses_to_weaker_rate"}

# Not a UI variable, but the measured strongest single signal available at an
# arbitrary date (see validate_point_in_time_srs.py), so it is carried as an
# extra column rather than discarded.
EXTRA_KEYS = ["srs_blend"]

# Venue terms live IN the design matrix, not beside it. Stored as a sibling
# field they were invisible to any consumer that builds its matrix from `keys`,
# which is the omission failure: with no intercept (mirrored rows force it to
# zero) the home effect does not surface as a constant offset, it is absorbed
# into the correlated strength coefficients -- inflating srs_blend 13% and
# barthag 44%, because strong teams buy home games so strength and home-ness
# correlate at r = 0.13. Being in `keys` means a consumer gets them by default
# and must zero them DELIBERATELY for tournament prediction, which is the safe
# direction for the mistake to run.
#
# These are appended to x rather than differenced like every other column,
# because venue is a property of the GAME, not of either team.
VENUE_KEYS = ["venue_home", "venue_host_city"]


def load_dayzero() -> dict[int, dt.date]:
    out = {}
    with open(KAGGLE / "MSeasons.csv") as f:
        for r in csv.DictReader(f):
            out[int(r["Season"])] = dt.datetime.strptime(r["DayZero"], "%m/%d/%Y").date()
    return out


def load_conferences() -> dict[int, dict[int, str]]:
    """Season -> {team_id: conference}. Carried onto each row so evaluation can
    separate inter-conference games, which is where a conference confound can
    actually bite: in a differential the two teams' conference effects cancel
    when both sit in the same league."""
    out: dict[int, dict[int, str]] = {}
    with open(KAGGLE / "MTeamConferences.csv") as f:
        for r in csv.DictReader(f):
            out.setdefault(int(r["Season"]), {})[int(r["TeamID"])] = r["ConfAbbrev"]
    return out


def load_universe() -> dict[int, set[int]]:
    """Season -> set of D1 team ids, the denominator for the connectivity gate."""
    out: dict[int, set[int]] = {}
    with open(KAGGLE / "MTeamConferences.csv") as f:
        for r in csv.DictReader(f):
            out.setdefault(int(r["Season"]), set()).add(int(r["TeamID"]))
    return out


def load_torvik_snapshots(year: int, dayzero: dt.date) -> list[tuple[int, dict]]:
    """Dated Torvik snapshots as (day_number, {canonical_id: fields}), ascending.

    Converting to day numbers here means the point-in-time comparison at the
    call site is a plain integer comparison against the boundary, rather than a
    date/day-number mix that is easy to get backwards.
    """
    path = HIST / f"torvik_{year}.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    out = []
    for snap in data.get("four_factors_snapshots") or []:
        day = (dt.date.fromisoformat(snap["date"]) - dayzero).days
        teams = {k: v for k, v in snap["data"].items() if isinstance(v, dict)}
        if teams:
            out.append((day, teams))
    out.sort(key=lambda kv: kv[0])
    return out


def snapshot_before(snaps: list[tuple[int, dict]], boundary: int) -> dict | None:
    """Latest snapshot STRICTLY before the boundary, or None.

    Strictly before, not nearest: a snapshot dated on or after the boundary
    reflects games the row is not allowed to have seen.
    """
    eligible = [teams for day, teams in snaps if day < boundary]
    return eligible[-1] if eligible else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--seasons", type=int, nargs="*", help="limit to these seasons")
    ap.add_argument(
        "--raw-rates",
        action="store_true",
        help="use the unadjusted rates (the pre-fix matrix, for A/B comparison)",
    )
    args = ap.parse_args()

    use_adjusted = not args.raw_rates
    dayzero = load_dayzero()
    conf_of = load_conferences()
    # Venue is a property of where the game was played, so it is knowable
    # before tip-off and carries no point-in-time risk -- unlike every other
    # feature here it needs no cutoff. Derived home cities come from the whole
    # season because "which arena does this team host in" is a fact about the
    # programme, not a running measurement of its form.
    game_cities = load_game_cities()
    home_cities = derive_home_cities()
    universe = load_universe()
    conf_tourney = set()
    ct_path = KAGGLE / "MConferenceTourneyGames.csv"
    if ct_path.exists():
        with open(ct_path) as f:
            for r in csv.DictReader(f):
                conf_tourney.add(
                    (int(r["Season"]), int(r["DayNum"]), int(r["WTeamID"]), int(r["LTeamID"]))
                )
    keys = [k for k, *_ in VARIABLES if k in TORVIK_DATED | KAGGLE_DERIVED | OPPONENT_DEPENDENT]
    higher_better = {k: hb for k, _l, _g, hb, _d in VARIABLES}
    diff_keys = keys + EXTRA_KEYS
    all_keys = diff_keys + VENUE_KEYS

    years = args.seasons or sorted(y for y in universe if y >= 2010)
    rows: list[dict] = []
    per_year: dict[str, int] = {}
    skipped_boundaries: dict[str, int] = {}
    skipped_seasons: list[str] = []

    for year in years:
        compact = load_season_games(year)
        if not compact:
            continue
        if not season_is_complete(compact):
            skipped_seasons.append(f"{year} (kaggle ends day {max(g.day for g in compact)})")
            continue
        detailed = load_detailed_games(year)
        prev = load_season_games(year - 1)
        prior = srs(games_before(prev, SELECTION_SUNDAY_DAY)) if prev else {}
        snaps = load_torvik_snapshots(year, dayzero[year])
        field = universe.get(year, set())

        # canonical id per kaggle id, resolved once per season
        kaggle_ids = sorted({g.winner for g in compact} | {g.loser for g in compact})
        canon_of_kid = {}
        for canon, kid in ratings_to_canonical({k: float(k) for k in kaggle_ids}).items():
            canon_of_kid[int(kid)] = canon

        n_before = len(rows)
        for boundary in range(FIRST_BOUNDARY, SELECTION_SUNDAY_DAY, BOUNDARY_STEP):
            past = games_before(compact, boundary)
            if not past:
                continue
            share = largest_component_share(past, field)
            if share < CONNECTIVITY_FLOOR:
                skipped_boundaries[str(year)] = skipped_boundaries.get(str(year), 0) + 1
                continue

            future = [g for g in compact if boundary <= g.day < boundary + BOUNDARY_STEP]
            if not future:
                continue

            tv = snapshot_before(snaps, boundary)
            if tv is None:
                skipped_boundaries[str(year)] = skipped_boundaries.get(str(year), 0) + 1
                continue

            pit = srs(past)
            blend = shrink_to_prior(pit, prior, game_counts(past))
            form = form_stats(past)
            past_detailed = detailed_before(detailed, boundary) if detailed else []
            box = box_profile(past_detailed) if past_detailed else {}
            # Re-solved from THIS slice. The opponent effects that correct the
            # rates come from the same window as the rates, so a December row
            # is corrected by December opponents -- reusing a season-final
            # adjustment here would be the same hindsight in a subtler place.
            adjr = adjusted_rates(past_detailed) if (use_adjusted and past_detailed) else {}

            # Opponent-dependent features need DATED opponent ratings. barthag
            # from the eligible snapshot is the dated stand-in for both the SOS
            # weight and the ranking that defines a "bad loss".
            dated_barthag = {
                kid: tv[canon]["barthag"]
                for kid, canon in canon_of_kid.items()
                if canon in tv and isinstance(tv[canon].get("barthag"), (int, float))
            }
            sos = strength_of_schedule(past, dated_barthag)
            bad = losses_to_weaker_rate(past, rank_from_rating(dated_barthag))

            # assemble raw per-team values over the teams present at this boundary
            present = [kid for kid in pit if canon_of_kid.get(kid) in tv]
            if len(present) < 50:
                continue
            raw: dict[str, list] = {k: [] for k in diff_keys}
            for kid in present:
                canon = canon_of_kid[kid]
                tvals = tv[canon]
                f = form.get(kid, {})
                b = box.get(kid, {})
                for k in keys:
                    if use_adjusted and k in ADJUSTABLE_RATES:
                        # Opponent-adjusted, and therefore NOT the Torvik
                        # snapshot value for the same name. audit_opponent_
                        # adjustment measures these four raw rates carrying
                        # conference strength at |partial r| 0.13-0.33 against
                        # every instrument tested. This row set is 41x the
                        # tournament matrix and spans the full D1 range of
                        # conference strength, so it is the worst place to
                        # leave the confound in.
                        v = adjr.get(kid, {}).get(k)
                    elif k in TORVIK_DATED:
                        v = tvals.get(k)
                    elif k in KAGGLE_DERIVED:
                        v = f.get(k, b.get(k))
                    elif k == "sos_avg_opp_barthag":
                        v = sos.get(kid)
                    else:
                        v = bad.get(kid)
                    raw[k].append(v if isinstance(v, (int, float)) else None)
                raw["srs_blend"].append(blend.get(kid))

            z = {k: zscores(raw[k], higher_better.get(k, True)) for k in keys}
            z["srs_blend"] = zscores(raw["srs_blend"], True)
            idx = {kid: i for i, kid in enumerate(present)}

            for g in future:
                if g.winner not in idx or g.loser not in idx:
                    continue
                a_kid, b_kid = g.winner, g.loser
                ca, cb = canon_of_kid[a_kid], canon_of_kid[b_kid]
                # orient by canonical id: a pre-game fact, never the result
                if cb < ca:
                    a_kid, b_kid, ca, cb = b_kid, a_kid, cb, ca
                won = 1 if a_kid == g.winner else 0
                margin = g.winner_score - g.loser_score if a_kid == g.winner else g.loser_score - g.winner_score
                ia, ib = idx[a_kid], idx[b_kid]
                rows.append(
                    {
                        "y": year,
                        "d": boundary,
                        "w": won,
                        "m": margin,
                        "ic": int(
                            conf_of.get(year, {}).get(a_kid)
                            != conf_of.get(year, {}).get(b_kid)
                        ),
                        "x": [round(z[k][ia] - z[k][ib], 4) for k in diff_keys]
                        + list(
                            split_states(
                                venue_for(
                                    year, g.day, g.winner, g.loser, g.winner_loc,
                                    game_cities, home_cities,
                                    (year, g.day, g.winner, g.loser) in conf_tourney,
                                )[0 if a_kid == g.winner else 1]
                            )
                        ),
                    }
                )
        per_year[str(year)] = len(rows) - n_before

    payload = {
        "keys": all_keys,
        "games": rows,
        "n_games": len(rows),
        "years": [int(y) for y in per_year if per_year[y]],
        "per_year": per_year,
        "orientation": (
            "x = z(team1) - z(team2) at boundary d; w = 1 if team1 won; "
            "m = team1 score - team2 score; team1 is the lexicographically "
            "smaller canonical id, which is independent of the result"
        ),
        "rates": "opponent-adjusted" if use_adjusted else "raw",
        "point_in_time": (
            "every feature computed from games strictly before d; torvik "
            "snapshot strictly before d; standardised within (season, d)"
        ),
    }
    args.out.write_text(json.dumps(payload, separators=(",", ":")))

    print(f"\n{len(rows):,} rows across {len(payload['years'])} seasons -> {args.out}")
    print(f"  {len(all_keys)} features: {len(keys)} UI variables + "
          f"{len(EXTRA_KEYS)} extra + {len(VENUE_KEYS)} venue")
    print(f"  size {args.out.stat().st_size / 1e6:.1f} MB")
    if skipped_seasons:
        print(f"  seasons skipped: {', '.join(skipped_seasons)}")
    if skipped_boundaries:
        print(f"  boundaries dropped (connectivity/no snapshot): {dict(sorted(skipped_boundaries.items()))}")
    print("\n  rows per season:")
    for y, n in sorted(per_year.items()):
        print(f"    {y}: {n:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
