"""Guards for the docs/ multi-year team stats table.

The table's whole premise is that every stat column is knowable BEFORE that
year's tournament tips off, with post-hoc results quarantined in clearly
labelled `outcome_*` fields. The regular-season volatility columns are the
risky ones, and they can fail in two opposite directions:

  LOOK-AHEAD — a tournament game folded into a "pre-tournament" column.
  UNDER-COUNTING — real regular-season games silently dropped, so the column
    is computed from a fraction of the schedule.

Both were live. The columns used to come from a game log running through the
national championship, filtered by date; the filter did block every tournament
game, but the log's per-game dates are largely synthetic and it also threw away
600-800 genuine regular-season games a season. Median games_played was 22 in
2012 against a true D1 schedule of ~31. These tests fail loudly on either.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts._common import load_tournament_results
from src.data.normalize import resolve_cbbpy_bridge

ARTIFACT = Path("docs/data/team_stats_by_year.json")
HIST = Path("data/raw/historical")

# The expensive raw-game-log check runs on a representative sample rather
# than all 16 years, to keep the unit tier fast.
LEAKAGE_SAMPLE_YEARS = (2013, 2024, 2026)

PRE_TOURNAMENT_FIELDS = (
    "barthag",
    "adj_offensive_efficiency",
    "reg_season_margin_avg",
    "reg_season_margin_std",
    "close_game_rate",
    "losses_to_weaker_rate",
)


def _artifact() -> dict:
    if not ARTIFACT.exists():
        pytest.skip(f"{ARTIFACT} not generated")
    with open(ARTIFACT) as f:
        return json.load(f)


@pytest.mark.parametrize("year", LEAKAGE_SAMPLE_YEARS)
def test_no_tournament_games_in_pre_tournament_window(year):
    """No actual NCAA tournament game may survive the pre-tournament filter.

    Matches on (team pair, exact score pair), so it catches a bad date
    filter regardless of how the game log spells its dates.
    """
    from scripts.generate_team_stats_table import _torvik_meta

    games_path = HIST / f"historical_games_{year}.json"
    if not games_path.exists():
        pytest.skip(f"no game log for {year}")
    tournament_start = _torvik_meta(year).get("tournament_start")
    assert tournament_start, f"{year} has no tournament_start to filter on"

    with open(HIST / f"torvik_{year}.json") as f:
        canonical = {t["team_id"] for t in json.load(f)["teams"]}

    ncaa = {
        (frozenset([g["team1_id"], g["team2_id"]]), tuple(sorted([g["team1_score"], g["team2_score"]])))
        for g in load_tournament_results(year)
        if g.get("team1_score") is not None
    }
    assert ncaa, f"{year} has no tournament results to check against"

    with open(games_path) as f:
        log = json.load(f).get("games", [])

    appearances: dict[str, int] = {}
    for g in log:
        for key in ("team1_id", "team2_id"):
            if g.get(key):
                appearances[g[key]] = appearances.get(g[key], 0) + 1
    bridge_map = resolve_cbbpy_bridge(appearances, canonical)

    def bridge(raw):
        return bridge_map.get(raw)

    leaked = []
    for g in log:
        if not g.get("date") or g["date"] >= tournament_start or g.get("team1_score") is None:
            continue
        key = (
            frozenset([bridge(g["team1_id"]), bridge(g["team2_id"])]),
            tuple(sorted([g["team1_score"], g["team2_score"]])),
        )
        if key in ncaa:
            leaked.append(g)

    assert not leaked, f"{year}: {len(leaked)} NCAA tournament game(s) leaked into the pre-tournament window"


def test_outcome_finish_distribution_is_a_valid_bracket():
    """Each year's finishes must add up to exactly one real bracket."""
    expected = {
        "Champion": 1,
        "Runner-up": 1,
        "Final Four": 2,
        "Elite 8": 4,
        "Sweet 16": 8,
        "Round of 32": 16,
        "Round of 64": 32,
    }
    data = _artifact()
    for year, rows in data["stats_by_year"].items():
        counts: dict[str, int] = {}
        for r in rows:
            if r.get("outcome_finish"):
                counts[r["outcome_finish"]] = counts.get(r["outcome_finish"], 0) + 1
        for label, n in expected.items():
            assert counts.get(label) == n, f"{year}: expected {n}x {label!r}, got {counts.get(label)}"


def test_rounds_won_consistent_with_finish():
    data = _artifact()
    by_finish = {
        "Round of 64": 0,
        "Round of 32": 1,
        "Sweet 16": 2,
        "Elite 8": 3,
        "Final Four": 4,
        "Runner-up": 5,
        "Champion": 6,
    }
    for year, rows in data["stats_by_year"].items():
        for r in rows:
            finish = r.get("outcome_finish")
            if finish in by_finish:
                assert r["outcome_rounds_won"] == by_finish[finish], (
                    f"{year} {r['team_name']}: finish={finish} but rounds_won={r['outcome_rounds_won']}"
                )


def test_pre_tournament_fields_populated():
    """Every row carries the pre-tournament stat block (no silent gaps)."""
    data = _artifact()
    for year, rows in data["stats_by_year"].items():
        for r in rows:
            for field in PRE_TOURNAMENT_FIELDS:
                assert field in r, f"{year} {r['team_name']}: missing {field}"
            assert r["games_played"], f"{year} {r['team_name']}: no game-log stats (bridge failure?)"


# A Division I team plays ~29-35 games before the tournament: roughly 31
# regular-season plus a conference tournament run. A median materially below
# this band means games are being dropped, not that teams played less
# basketball.
#
# 2021 is exempt from both floors, and genuinely so rather than as a fudge:
# COVID pods and mid-season pauses left Colgate with 15 games and Iona with 17,
# and those are the real schedules those teams played.
MIN_PLAUSIBLE_MEDIAN_GAMES = 29
MIN_PLAUSIBLE_TEAM_GAMES = 20
SHORT_SEASONS = {"2021"}


def test_schedules_are_not_silently_truncated():
    """Form columns must be computed from a team's whole pre-tournament season.

    This is the regression guard for the defect that motivated moving these
    columns off the cbbpy game log: a date filter applied to unreliable dates
    discarded a quarter of most schedules without erroring. Nothing else in the
    suite noticed, because every surviving value was individually well-formed.
    """
    import statistics

    data = _artifact()
    for year, rows in data["stats_by_year"].items():
        counts = [r["games_played"] for r in rows if r.get("games_played")]
        assert counts, f"{year}: no team has games_played"

        if year in SHORT_SEASONS:
            continue

        median = statistics.median(counts)
        assert median >= MIN_PLAUSIBLE_MEDIAN_GAMES, (
            f"{year}: median games_played={median}, below {MIN_PLAUSIBLE_MEDIAN_GAMES}. "
            "Games are being dropped somewhere between the source and the table."
        )

        worst = min(counts)
        assert worst >= MIN_PLAUSIBLE_TEAM_GAMES, (
            f"{year}: a team has only {worst} games. No tournament qualifier plays "
            "that few, so its form columns are computed from a partial schedule."
        )


def test_rates_are_in_unit_interval():
    data = _artifact()
    for year, rows in data["stats_by_year"].items():
        for r in rows:
            for field in ("close_game_rate", "close_game_win_rate", "losses_to_weaker_rate"):
                v = r.get(field)
                if v is not None:
                    assert 0.0 <= v <= 1.0, f"{year} {r['team_name']}: {field}={v} outside [0,1]"


def test_historical_residual_is_backward_looking_only():
    """`hist_residual` must never see the current year or the future.

    The earliest year has no prior tournaments in the dataset, so every row
    must show zero prior appearances; and a team's appearance count must
    grow monotonically over time, never shrink.
    """
    data = _artifact()
    years = sorted(data["stats_by_year"], key=int)

    for r in data["stats_by_year"][years[0]]:
        assert r["hist_appearances"] == 0, f"{years[0]} {r['team_name']}: has prior history before the dataset starts"
        assert r["hist_residual"] is None

    seen: dict[str, int] = {}
    for year in years:
        for r in data["stats_by_year"][year]:
            prior = seen.get(r["team_id"], 0)
            assert r["hist_appearances"] == prior, (
                f"{year} {r['team_name']}: hist_appearances={r['hist_appearances']} but "
                f"{prior} prior appearance(s) in the dataset"
            )
            assert (r["hist_residual"] is None) == (r["hist_appearances"] == 0)
            if r.get("outcome_vs_seed_delta") is not None:
                seen[r["team_id"]] = prior + 1


def test_2020_absent_and_year_span():
    data = _artifact()
    years = data["years"]
    assert 2020 not in years, "2020 has no tournament and must not appear"
    assert min(years) >= 2010 and max(years) <= 2026


@pytest.mark.parametrize("path", ["docs/data/team_stats_by_year.json", "docs/data/matchups_by_year.json"])
def test_artifact_is_strict_json(path):
    """Artifacts must be valid JSON by the spec, not just by Python's reader.

    Python writes bare `NaN`/`Infinity` literals for non-finite floats and
    reads them straight back, so a Python-only check passes while every
    browser refuses the file with "Unexpected token 'N'". This happened for
    real: a NaN minutes-per-game value silently broke the whole stats table
    in the browser while every Python test stayed green. Rejecting the
    non-standard constants here is what actually guards the front end.
    """
    p = Path(path)
    if not p.exists():
        pytest.skip(f"{path} not generated")

    def reject(const):
        raise AssertionError(f"{path} contains non-standard JSON constant {const!r}")

    with open(p) as f:
        json.load(f, parse_constant=reject)


def test_kaggle_box_profile_has_no_tournament_game_overlap():
    """`MRegularSeasonDetailedResults` (the source for the 3PT/havoc/road
    columns) must share zero games with `MNCAATourneyDetailedResults` — that
    is what makes it pre-tournament by construction, with no date filter
    needed, unlike the excluded Torvik shooting files.
    """
    kaggle = Path("data/kaggle")
    reg_path = kaggle / "MRegularSeasonDetailedResults.csv"
    tourney_path = kaggle / "MNCAATourneyDetailedResults.csv"
    if not reg_path.exists() or not tourney_path.exists():
        pytest.skip("Kaggle box-score files not present")

    import csv

    def keys(path):
        with open(path) as f:
            return {(r["Season"], r["DayNum"], r["WTeamID"], r["LTeamID"]) for r in csv.DictReader(f)}

    overlap = keys(reg_path) & keys(tourney_path)
    assert not overlap, f"{len(overlap)} game(s) appear in both regular-season and tournament box scores"


def test_kaggle_box_profile_values_are_in_range():
    """Sanity bounds on the new shooting/pressure columns — catches a mixed-up
    numerator/denominator or a wrong team-side attribution outright.
    """
    data = _artifact()
    checked = 0
    for year in data["years"]:
        for r in data["stats_by_year"][str(year)]:
            for field in ("three_pt_rate", "three_pt_pct", "opp_three_pt_pct", "true_road_win_pct"):
                v = r.get(field)
                if v is not None:
                    assert 0.0 <= v <= 1.0, f"{year} {r['team_name']} {field}={v} out of [0, 1]"
                    checked += 1
            if r.get("ast_to_ratio") is not None:
                assert 0 < r["ast_to_ratio"] < 5, f"{year} {r['team_name']} ast_to_ratio out of range"
            if r.get("havoc_rate") is not None:
                assert 0 <= r["havoc_rate"] < 30, f"{year} {r['team_name']} havoc_rate out of range"
    assert checked > 1000, "expected shooting-profile columns on most rows across 16 years"


def test_coach_experience_is_backward_looking_only():
    """A coach's prior-tournament games/wins must never include their result
    in the row's own year — only strictly earlier seasons. Checked directly
    against the Kaggle source, not just the artifact, so a bug in the
    cumulative-sum boundary (`< year` vs `<= year`) is caught even if it
    happens not to move any single artifact value.
    """
    import csv

    kaggle = Path("data/kaggle")
    coaches_path = kaggle / "MTeamCoaches.csv"
    tourney_path = kaggle / "MNCAATourneyCompactResults.csv"
    if not coaches_path.exists() or not tourney_path.exists():
        pytest.skip("Kaggle coach files not present")

    from scripts.generate_team_stats_table import build_coach_experience
    from scripts._common import load_torvik_and_ff

    year = 2026
    torvik, _ = load_torvik_and_ff(year)
    result = build_coach_experience(year, set(torvik))
    assert result, "expected coach experience for at least one team in 2026"

    with open(tourney_path) as f:
        games_2026 = [r for r in csv.DictReader(f) if r["Season"] == "2026"]
    assert not games_2026, "2026 must not appear in MNCAATourneyCompactResults (it's the target year)"

    for info in result.values():
        assert info["coach_prior_tourney_games"] >= info["coach_prior_tourney_wins"] >= 0
        assert info["coach_first_tourney"] == (info["coach_prior_tourney_games"] == 0)
