"""Guards for the docs/ multi-year team stats table.

The table's whole premise is that every stat column is knowable BEFORE that
year's tournament tips off, with post-hoc results quarantined in clearly
labelled `outcome_*` fields. The regular-season volatility columns are
computed from a game log that (in most years) runs through the national
championship, so the date filter in
`scripts/generate_team_stats_table.py::build_regular_season_stats` is the
only thing standing between "pre-tournament" and silent look-ahead bias.
These tests fail loudly if that filter regresses.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts._common import load_tournament_results
from src.data.normalize import bridge_cbbpy_id

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
    from scripts.generate_team_stats_table import _EXTRA_CBBPY_ALIASES, _torvik_meta

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

    cache: dict[str, str | None] = {}

    def bridge(raw):
        if raw not in cache:
            alias = _EXTRA_CBBPY_ALIASES.get(raw)
            cache[raw] = alias if (alias and alias in canonical) else bridge_cbbpy_id(raw, canonical)
        return cache[raw]

    with open(games_path) as f:
        log = json.load(f).get("games", [])

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
