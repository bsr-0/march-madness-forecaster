#!/usr/bin/env python3
"""Audit the point-in-time boundary shared by the training and prediction paths.

WHY THIS IS AN AUDIT AND NOT A TEST OF THE MODEL. Everything downstream --
wider training populations, new features, a recalibrated link -- assumes the
features a game is predicted from were knowable before that game was played,
and that the training path and the prediction path construct them the same way.
If either assumption is false, every metric measured afterwards is measuring
the wrong thing while looking completely healthy. A silent failure here
invalidates the results of every later step, which is why it runs before them
and not after.

CHECKS
  A  cutoff construction is single-sourced: every season's Torvik cutoff is
     exactly the day before that season's tournament start, from
     TOURNAMENT_START_DATES, with no per-season special cases.
  B  the season-end snapshot agrees exactly with the ratings actually used.
     Same date must mean same numbers; if it does not, one of the two is not
     the thing it claims to be.
  C  snapshots are genuine as-of-date pulls rather than copies of one scrape,
     AND each one is bounded by its own date. The second half is the load-
     bearing part: values that merely differ from each other would still pass
     if the source ignored the `end` bound, so the boundary is verified
     against an independent source (Kaggle box scores) by recomputing raw
     eFG% and confirming an early snapshot resembles games-before-that-date
     far more than the full season.
  D  train/serve feature parity, in two forms. D1 compares the SHIPPED season
     payload against the training path -- two artefacts, two scripts, so a
     disagreement means the browser predicts from numbers the model was not
     fit on; it runs only where a payload exists. D2 compares the two code
     paths across every season in the training set, covering the seasons the
     UI does not ship. Neither alone answers the question.
  E  inventory of which fields the dated snapshots actually carry. This is the
     feasibility gate for training on regular-season games: a variable with no
     dated snapshot cannot be reconstructed point-in-time for a December game,
     and using its March value there would be leakage.
  F  the same question as C, for ratings computed rather than stored: an SRS
     solve must depend on nothing after its cutoff, must actually differ
     between December and March, and the connectivity gate that decides when
     a solve is meaningful must discriminate 2021 from a normal season.
  H  join-sourced columns resolve for the whole field. Every column is a join
     onto canonical team ids, and this repo has hit the same defect three
     times -- two mapping paths for one join, producing plausible nulls rather
     than an error. All three were found by chance; this makes looking
     automatic.
  G  venue is coded from the host venue, not from the game type. The NCAA
     tournament term must be unconditionally zero -- a participant in the host
     city there is proximity, not home court -- while conference tournaments
     must NOT be blanket-neutral, because 28% of them are played in a
     participant's own home city and a game-type flag miscodes every one.

Run: python3 scripts/audit_snapshot_boundary.py
"""

from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.build_ui_payload import VARIABLES, zscores  # noqa: E402
from scripts.build_training_matrix import season_z  # noqa: E402
from src.data.features.point_in_time_ratings import (  # noqa: E402
    SELECTION_SUNDAY_DAY,
    games_before,
    largest_component_share,
    load_season_games,
    srs,
    srs_asof,
)
from scripts.validate_torvik_snapshot_bounds import (  # noqa: E402
    EARLY_DAY_MAX,
    efg_through,
    load_shooting_rows,
)
from src.data.features.custom_ratings import ratings_to_canonical  # noqa: E402
from src.data.features.venue import (  # noqa: E402
    derive_home_cities,
    load_game_cities,
    tournament_venue,
    venue_for,
)
from src.pipeline.config import TOURNAMENT_START_DATES  # noqa: E402

HIST = REPO / "data" / "raw" / "historical"
STATS_PATH = REPO / "docs" / "data" / "team_stats_by_year.json"
CANDIDATES = REPO / "artifacts" / "candidates"
UI_DATA = REPO / "docs" / "data"

# Kaggle DayZero per season, for converting snapshot dates to day numbers.
DAYZERO: dict[int, date] = {}
with open(REPO / "data" / "kaggle" / "MSeasons.csv") as _f:
    import csv as _csv

    for _r in _csv.DictReader(_f):
        _m, _d, _y = _r["DayZero"].split("/")
        DAYZERO[int(_r["Season"])] = date(int(_y), int(_m), int(_d))

failures: list[str] = []
notes: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}")
    if detail:
        print(f"          {detail}")
    if not ok:
        failures.append(name)


def torvik(year: int) -> dict | None:
    p = HIST / f"torvik_{year}.json"
    return json.loads(p.read_text()) if p.exists() else None


def main() -> int:
    stats = json.loads(STATS_PATH.read_text())["stats_by_year"]
    years = sorted(int(y) for y in stats)

    # ---------------------------------------------------------------- A
    print("\nA. cutoff construction is single-sourced")
    bad = []
    for y in years:
        d = torvik(y)
        if not d or not d.get("cutoff_date"):
            continue
        expected = TOURNAMENT_START_DATES.get(y)
        if expected is None:
            bad.append(f"{y}: no TOURNAMENT_START_DATES entry")
            continue
        want = expected - timedelta(days=1)
        got = date.fromisoformat(d["cutoff_date"])
        if got != want:
            bad.append(f"{y}: cutoff {got} != tournament_start-1 {want}")
    check(
        "every cutoff is exactly tournament_start - 1 day",
        not bad,
        "; ".join(bad) if bad else f"{len(years)} seasons, one rule, no exceptions",
    )

    # ---------------------------------------------------------------- B
    print("\nB. season-end snapshot agrees with the ratings actually used")
    mismatches, compared = [], 0
    for y in years:
        d = torvik(y)
        if not d:
            continue
        snaps = d.get("four_factors_snapshots") or []
        if not snaps:
            continue
        final = max(snaps, key=lambda s: s["date"])
        if final["date"] != d.get("cutoff_date"):
            mismatches.append(f"{y}: last snapshot {final['date']} != cutoff {d['cutoff_date']}")
            continue
        payload = final["data"]
        by_id = {t["team_id"]: t for t in d.get("teams", [])}
        for tid, snap in payload.items():
            if not isinstance(snap, dict) or tid not in by_id:
                continue
            for field, sv in snap.items():
                mv = by_id[tid].get(field)
                if isinstance(sv, (int, float)) and isinstance(mv, (int, float)):
                    compared += 1
                    if abs(sv - mv) > 1e-9:
                        mismatches.append(f"{y} {tid}.{field}: snapshot {sv} vs used {mv}")
    check(
        "same date yields identical numbers on both surfaces",
        not mismatches,
        f"{compared:,} field comparisons across {len(years)} seasons" if not mismatches else "; ".join(mismatches[:4]),
    )

    # ---------------------------------------------------------------- C
    print("\nC. snapshots are genuine as-of-date pulls, not copies")
    static, checked = [], 0
    for y in years:
        d = torvik(y)
        if not d:
            continue
        snaps = sorted(d.get("four_factors_snapshots") or [], key=lambda s: s["date"])
        if len(snaps) < 2:
            continue
        first, last = snaps[0]["data"], snaps[-1]["data"]
        moved = 0
        for tid, sv in last.items():
            if not isinstance(sv, dict) or tid not in first or not isinstance(first[tid], dict):
                continue
            for field, v in sv.items():
                fv = first[tid].get(field)
                if isinstance(v, (int, float)) and isinstance(fv, (int, float)):
                    checked += 1
                    if abs(v - fv) > 1e-9:
                        moved += 1
        if moved == 0:
            static.append(f"{y}: first and last snapshot identical")
    check(
        "ratings move between the first and last snapshot",
        not static,
        f"{checked:,} field comparisons; values evolve as games are played" if not static else "; ".join(static),
    )

    # "The values move" rules out one scrape copied under many labels, but not
    # the failure that matters: a snapshot dated in November carrying
    # end-of-season information. A source that ignored the `end` parameter
    # would still produce snapshots that differ from each other. So check the
    # boundary directly against an INDEPENDENT source -- Torvik's
    # effective_fg_pct is a raw rate, recomputable from Kaggle box scores as
    # (FGM + 0.5*FGM3)/FGA. A correctly bounded early snapshot must resemble
    # the games-before-that-date computation far more than the full-season one.
    # Driven by early snapshots because the two computations converge by March
    # no matter what, which would dilute the verdict.
    shooting = load_shooting_rows()
    bounded_err, full_err, tested, wrong = [], [], 0, []
    for y in years:
        d = torvik(y)
        if not d or y not in shooting:
            continue
        rows = shooting[y]
        full = efg_through(rows, None)
        canon_of_kid = {int(kid): canon for canon, kid in ratings_to_canonical({k: float(k) for k in full}).items()}
        for snap in d.get("four_factors_snapshots") or []:
            day = (date.fromisoformat(snap["date"]) - DAYZERO[y]).days
            if day > EARLY_DAY_MAX:
                continue
            bounded = efg_through(rows, day)
            tv = {
                k: v["effective_fg_pct"]
                for k, v in snap["data"].items()
                if isinstance(v, dict) and isinstance(v.get("effective_fg_pct"), (int, float))
            }
            db = [abs(tv[c] - bounded[k]) for k, c in canon_of_kid.items() if c in tv and k in bounded]
            df = [abs(tv[c] - full[k]) for k, c in canon_of_kid.items() if c in tv and k in full]
            if len(db) < 50:
                continue
            tested += 1
            mb, mf = sum(db) / len(db), sum(df) / len(df)
            bounded_err.append(mb)
            full_err.append(mf)
            if mb >= mf:
                wrong.append(f"{y} {snap['date']}: bounded {mb:.4f} >= full {mf:.4f}")

    ratio = (sum(full_err) / len(full_err)) / (sum(bounded_err) / len(bounded_err)) if bounded_err else 0
    check(
        "each early snapshot is bounded by its own date, not the full season",
        bool(bounded_err) and not wrong,
        f"{tested} early snapshots; {ratio:.1f}x closer to games-before-date than to season total"
        if not wrong
        else f"{len(wrong)} resemble the full season: " + "; ".join(wrong[:3]),
    )

    # ---------------------------------------------------------------- D
    # Two forms, because neither alone covers the question.
    #
    # D1 compares the SHIPPED season payload against the training path. That is
    # the real train/serve test: two artefacts produced by two scripts, and a
    # disagreement means the browser predicts from numbers the model was not
    # fit on. It can only run where a season payload exists (build_ui_payload
    # ships four seasons), so it is deep but narrow.
    #
    # D2 compares the two CODE PATHS over every season in the training set.
    # It cannot catch a stale shipped artefact -- nothing is read from disk --
    # but it does cover all 16 seasons rather than the handful the UI ships,
    # so a season-specific divergence in the standardisation cannot hide in
    # the 13 seasons D1 never looks at.
    print("\nD. train/serve feature parity")

    worst_shipped = (0.0, None, None, None)
    shipped_years, shipped_comparisons = 0, 0
    for y in years:
        payload_path = UI_DATA / f"season_{y}.json"
        if not payload_path.exists():
            continue
        payload = json.loads(payload_path.read_text())
        if payload.get("status") != "ready" or "z" not in payload:
            continue
        shipped_years += 1
        z_train = season_z(stats[str(y)])
        team_ids = [t["id"] for t in payload["teams"]]
        for key, served in payload["z"].items():
            for i, tid in enumerate(team_ids):
                if tid not in z_train or key not in z_train[tid]:
                    continue
                shipped_comparisons += 1
                delta = abs(z_train[tid][key] - served[i])
                if delta > worst_shipped[0]:
                    worst_shipped = (delta, y, tid, key)

    check(
        "shipped season payload matches the training path exactly",
        worst_shipped[0] == 0.0 and shipped_comparisons > 0,
        f"{shipped_comparisons:,} comparisons across {shipped_years} shipped seasons"
        if worst_shipped[0] == 0.0
        else f"differs by {worst_shipped[0]:g} at {worst_shipped[1]} {worst_shipped[2]}.{worst_shipped[3]}",
    )

    worst_fn = (0.0, None, None, None)
    fn_comparisons = 0
    for y in years:
        rows = stats[str(y)]
        by_id = {r["team_id"]: r for r in rows}
        z_train = season_z(rows)
        order = [r["team_id"] for r in rows]
        for key, _l, _g, higher_better, _d in VARIABLES:
            vals = [by_id[t].get(key) for t in order]
            vals = [v if isinstance(v, (int, float)) else None for v in vals]
            z_serve = zscores(vals, higher_better)
            for i, tid in enumerate(order):
                if key not in z_train.get(tid, {}):
                    continue
                fn_comparisons += 1
                delta = abs(z_train[tid][key] - z_serve[i])
                if delta > worst_fn[0]:
                    worst_fn = (delta, y, tid, key)

    check(
        "both standardisation paths agree on every season in the training set",
        worst_fn[0] == 0.0,
        f"{fn_comparisons:,} comparisons across {len(years)} seasons x {len(VARIABLES)} variables"
        if worst_fn[0] == 0.0
        else f"differs by {worst_fn[0]:g} at {worst_fn[1]} {worst_fn[2]}.{worst_fn[3]}",
    )

    # ---------------------------------------------------------------- E
    print("\nE. which variables have dated snapshots (gate for regular-season training)")
    d = torvik(2025)
    snaps = d.get("four_factors_snapshots") or []
    snap_fields = set()
    for tid, v in snaps[-1]["data"].items():
        if isinstance(v, dict):
            snap_fields |= set(v)
    ui_keys = [k for k, *_ in VARIABLES]
    dated = sorted(k for k in ui_keys if k in snap_fields)

    # A Torvik snapshot is not the only way to be dated. Every Kaggle results
    # row carries a DayNum, so these are reconstructible at any cutoff via
    # src/data/features/point_in_time_kaggle, which mirrors the season-final
    # formulas exactly (asserted by validate_point_in_time_kaggle.py). Listing
    # them as UNDATED here would understate what is available and wrongly make
    # regular-season training look infeasible.
    kaggle_derived = {
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
    # These two are derivable, but ONLY when handed dated opponent ratings;
    # the season-final builders use final barthag / t_rank, which is hindsight
    # anywhere but Selection Sunday. Tracked separately so the requirement
    # stays visible rather than being assumed satisfied.
    needs_dated_opponents = {"sos_avg_opp_barthag", "losses_to_weaker_rate"}

    derivable = sorted(k for k in ui_keys if k in kaggle_derived)
    conditional = sorted(k for k in ui_keys if k in needs_dated_opponents)
    undated = [
        k for k in ui_keys if k not in snap_fields and k not in kaggle_derived and k not in needs_dated_opponents
    ]
    print(f"          dated snapshot   ({len(dated)}/{len(ui_keys)}): {', '.join(dated)}")
    print(f"          kaggle-derivable ({len(derivable)}/{len(ui_keys)}): {', '.join(derivable)}")
    print(f"          needs dated opp  ({len(conditional)}/{len(ui_keys)}): {', '.join(conditional)}")
    print(f"          UNDATED          ({len(undated)}/{len(ui_keys)}): {', '.join(undated)}")
    notes.append(
        f"{len(dated) + len(derivable)} of {len(ui_keys)} UI variables are reconstructible "
        f"point-in-time today ({len(dated)} dated snapshots + {len(derivable)} kaggle-derived); "
        f"{len(conditional)} more need dated opponent ratings; {len(undated)} remain undated."
    )
    print(f"          snapshot dates for 2025: {', '.join(s['date'] for s in snaps)}")

    # ---------------------------------------------------------------- F
    # Same failure mode as C, one layer down. C asks whether a stored snapshot
    # is a genuine as-of-date pull; F asks whether a *computed* rating is. The
    # trap is specific and easy to fall into: compute_srs_ratings defaults to
    # cutoff_day=133, which is right for tournament rows and is leakage for a
    # December row. A March rating used to predict a December game is hindsight
    # in a point-in-time wrapper and every metric downstream stays healthy.
    print("\nF. point-in-time ratings are solved per date, not reused from March")
    season = 2024
    games = load_season_games(season)
    if not games:
        check("kaggle results available for the ratings check", False, f"no games for {season}")
    else:
        dec_cut = 45  # roughly mid-December in the Kaggle day calendar

        # F1: the solver cannot see past what it is handed. Physically truncate
        # the game list and confirm the answer is identical -- if srs reached
        # for the CSV or any global state, these would diverge.
        from_full = srs_asof(games, dec_cut)
        from_truncated = srs(list(games_before(games, dec_cut)))
        drift = max(
            (abs(from_full[t] - from_truncated.get(t, 0.0)) for t in from_full),
            default=0.0,
        )
        check(
            "a dated solve depends on nothing after its cutoff",
            drift == 0.0,
            f"{len(from_full)} teams identical from full and truncated inputs"
            if drift == 0.0
            else f"max drift {drift:g}",
        )

        # F2: dated solves must actually move between dates. If a caller wires
        # the March cutoff into every row, this is what catches it.
        march = srs_asof(games, SELECTION_SUNDAY_DAY)
        common = sorted(set(from_full) & set(march))
        moved = sum(1 for t in common if abs(from_full[t] - march[t]) > 1e-9)
        check(
            "December and March solves are different numbers",
            common and moved == len(common),
            f"all {len(common)} shared teams differ between the two dates"
            if common and moved == len(common)
            else f"only {moved}/{len(common)} teams moved",
        )

        # F3: the connectivity gate has to discriminate, not just return ~1.0.
        # A gate that passes everything would wave through 2021, where the
        # graph is genuinely disconnected and cross-team SRS comparisons are
        # undefined rather than merely noisy.
        conf_2021 = set()
        conf_path = REPO / "data" / "kaggle" / "MTeamConferences.csv"
        if conf_path.exists():
            import csv as _csv

            with open(conf_path) as fh:
                for row in _csv.DictReader(fh):
                    if int(row["Season"]) == 2021:
                        conf_2021.add(int(row["TeamID"]))
        g21 = load_season_games(2021)
        if g21 and conf_2021:
            early_21 = largest_component_share(games_before(g21, 30), conf_2021)
            early_24 = largest_component_share(games_before(games, 30), set(from_full) | set(march))
            check(
                "connectivity gate separates 2021 from a normal season",
                early_21 < 0.5 <= early_24,
                f"2021 share {early_21:.2f} vs {season} share {early_24:.2f} at the same boundary",
            )
            notes.append(
                f"2021 largest-component share is {early_21:.2f} at day 30; the SRS floor must be "
                "a per-season connectivity gate, not a calendar date."
            )

    # ---------------------------------------------------------------- G
    print("\nG. venue coding is a host check, not a game-type flag")
    cities = load_game_cities()
    home_cities = derive_home_cities()
    if not cities:
        check("kaggle game-city data available", False, "MGameCities.csv missing")
    else:
        # G1: the NCAA tournament venue term must be exactly zero, always. The
        # NCAA bans true home games, so a participant playing in the host city
        # is a PROXIMITY effect (travel_distance) and not a home-court one.
        # Routing tournament games through the host check would code 4 of them
        # as home/away and credit the wrong mechanism.
        ncaa_path = REPO / "data" / "kaggle" / "MNCAATourneyCompactResults.csv"
        non_neutral = 0
        total_ncaa = 0
        if ncaa_path.exists():
            import csv as _c

            with open(ncaa_path) as fh:
                for r in _c.DictReader(fh):
                    if int(r["Season"]) < 2010:
                        continue
                    total_ncaa += 1
                    ws, ls = venue_for(
                        int(r["Season"]), int(r["DayNum"]), int(r["WTeamID"]),
                        int(r["LTeamID"]), r.get("WLoc", "N"), cities, home_cities,
                    )
                    if (ws, ls) != (0, 0):
                        non_neutral += 1
        check(
            "tournament_venue() is unconditionally zero",
            tournament_venue() == 0,
            f"host check would code {non_neutral} of {total_ncaa:,} NCAA games non-neutral, "
            f"so the zero must not be derived from it",
        )

        # G2: conference tournaments must NOT be blanket-coded neutral. If this
        # ever reads ~0 the host check has silently stopped working, because
        # conference tournaments are routinely held in a member's home city.
        ct_path = REPO / "data" / "kaggle" / "MConferenceTourneyGames.csv"
        hosted = 0
        total_ct = 0
        if ct_path.exists():
            import csv as _c

            with open(ct_path) as fh:
                for r in _c.DictReader(fh):
                    s_ = int(r["Season"])
                    if s_ < 2010:
                        continue
                    key = (s_, int(r["DayNum"]), int(r["WTeamID"]), int(r["LTeamID"]))
                    if key not in cities:
                        continue
                    total_ct += 1
                    ws, ls = venue_for(
                        s_, key[1], key[2], key[3], "N", cities, home_cities
                    )
                    if (ws, ls) != (0, 0):
                        hosted += 1
        share = hosted / total_ct if total_ct else 0.0
        check(
            "conference tournaments are not blanket-neutral",
            share > 0.05,
            f"{hosted:,} of {total_ct:,} ({share * 100:.1f}%) have a participant in its own "
            f"home city; a game-type flag would miscode every one",
        )
        notes.append(
            f"conference-tournament games with a home participant: {share * 100:.1f}%; "
            "measured home-court effect there is +1.2 points against +2.9 generally, "
            "so a home CITY at a neutral arena is a weaker effect than a true home game."
        )

    # ---------------------------------------------------------------- H
    # Every column here is produced by joining an external source onto
    # canonical team ids, and this repo has now hit the same defect three
    # times: TWO MAPPING PATHS FOR ONE JOIN, where a builder reaches for
    # whichever mapper was nearest instead of the one that handles awkward
    # names. Each instance produced plausible-looking nulls or zeros rather
    # than an error:
    #   torvik vintage split     two surfaces, one date label      (check B)
    #   cbbpy roster bridge      142 of 1,084 team-seasons zero    (83d982a)
    #   massey via KaggleLoader  113 of 1,085 null                 (3036497)
    # All three were found by someone happening to look. This makes looking
    # automatic: a join-sourced column that silently stops resolving for a
    # slice of the field fails here instead of shipping.
    print("\nH. join-sourced columns resolve for the whole field")
    rows_all = [r for y in years for r in stats[str(y)]]
    n_rows = len(rows_all)
    # Tolerance is deliberately loose. The failures worth catching were 10.4%
    # and 13% of the field; the surviving legitimate gaps are single teams with
    # source-side slug drift (new_orleans 2017 is `new_orleans_privateers` in
    # the rosters and `lsu_new_orleans_privateers` in the box scores). A
    # threshold that chased those would cry wolf without catching anything the
    # eye would not.
    max_null_share = 0.01
    structural = {
        # A team's first tournament appearance has no prior history to
        # residualise against; the dataset starts in 2010.
        "hist_residual",
        # The Ivy League ran no conference tournament before 2017, and Kaggle's
        # MConferenceTourneyGames stops at 2025 so 2026 is upstream-missing.
        "conf_tourney_wins",
    }
    offenders = []
    checked = 0
    for key in sorted({k for r in rows_all for k in r}):
        if key in structural:
            continue
        vals = [r.get(key) for r in rows_all]
        if not any(isinstance(v, (int, float)) for v in vals):
            continue  # string/bool column, not a join-sourced measurement
        checked += 1
        nulls = sum(1 for v in vals if v is None)
        if nulls > n_rows * max_null_share:
            offenders.append(f"{key}: {nulls}/{n_rows} null ({nulls / n_rows:.1%})")
    check(
        "no join-sourced column is missing for more than 1% of the field",
        not offenders,
        f"{checked} numeric columns over {n_rows:,} team-seasons; "
        f"{len(structural)} structural exemptions"
        if not offenders
        else "; ".join(offenders),
    )

    print()
    if failures:
        print(f"{len(failures)} CHECK(S) FAILED: {', '.join(failures)}")
        return 1
    print("all boundary checks passed")
    for n in notes:
        print(f"NOTE: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
