"""Tests for the ESPN boxscore scraper and its minutes features.

Everything runs against a fixture captured from a real 2022 game
(401482889, Boston College vs Cornell, 2022-11-07) — a date well before the
2025-02-11 substitution cutover that makes the play-by-play route useless for
historical seasons. No network access.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.data.features.boxscore_player_minutes import (
    MinutesCoverageError,
    build_season_minutes_features,
)
from src.data.scrapers.espn_boxscore import (
    _extract_bxscr,
    _slugify_team_name,
    parse_boxscore,
    validate_boxscore_minutes,
)

FIXTURE = Path(__file__).parent / "fixtures" / "espn_boxscore" / "bxscr_401482889.json"


@pytest.fixture(scope="module")
def bxscr_json() -> str:
    return FIXTURE.read_text()


@pytest.fixture(scope="module")
def page_html(bxscr_json: str) -> str:
    """A stub page carrying BOTH bxscr keys, in the order ESPN emits them.

    The real page has a column-schema config under "bxscr" *before* the data
    array. A naive ``html.find('"bxscr"')`` grabs the config and finds no
    players, so the ordering here is the point of the fixture, not decoration.
    """
    config = (
        '{"bxscr":{"grp":{"types":["starters","bench",{"totals":"team"}],"keys":["minutes","points"]},"tabbed":false}}'
    )
    return f'<html><body><script>window.x={config};window.y={{"sbpg":"boxscore","bxscr":{bxscr_json}}};</script></body></html>'


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def test_extract_bxscr_skips_the_config_block(page_html):
    """Must return the data array, not the schema config that precedes it."""
    bx = _extract_bxscr(page_html)
    assert isinstance(bx, list)
    assert len(bx) == 2
    assert all("tm" in team for team in bx)


def test_extract_bxscr_returns_none_when_absent():
    assert _extract_bxscr("<html><body>no boxscore here</body></html>") is None


def test_extract_bxscr_survives_brackets_inside_strings():
    """A ']' inside a string value must not terminate the bracket scan."""
    payload = '[{"tm":{"dspNm":"Weird ] Team"},"stats":[]}]'
    html = f'<html>{{"bxscr":{payload}}}</html>'.replace('"bxscr":[{"tm"', '"bxscr":[{"tm"')
    bx = _extract_bxscr(html)
    assert bx is not None and bx[0]["tm"]["dspNm"] == "Weird ] Team"


def test_slugify_matches_pbp_team_ids():
    """Team ids must line up with athlete_team values already on disk."""
    assert _slugify_team_name("Cornell Big Red") == "cornell_big_red"
    assert _slugify_team_name("Boston College Eagles") == "boston_college_eagles"
    assert _slugify_team_name("Texas A&M Aggies") == "texas_a_m_aggies"
    assert _slugify_team_name("") == ""


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def test_parse_boxscore_extracts_both_teams(page_html):
    game = parse_boxscore(page_html, "401482889", game_date="2022-11-07")
    assert game["game_id"] == "401482889"
    assert game["game_date"] == "2022-11-07"
    assert {t["team_id"] for t in game["teams"]} == {"cornell_big_red", "boston_college_eagles"}


def test_parse_boxscore_minutes_sum_to_regulation(page_html):
    """Real published data: each team's minutes must total exactly 200."""
    game = parse_boxscore(page_html, "401482889")
    for team in game["teams"]:
        total = sum(p["minutes"] for p in team["players"] if p["minutes"] is not None)
        assert total == 200, f"{team['team_id']} totalled {total}, expected 200"


def test_parse_boxscore_labels_exactly_five_starters(page_html):
    """Starters are labelled by ESPN, not inferred — so there are exactly 5."""
    game = parse_boxscore(page_html, "401482889")
    for team in game["teams"]:
        assert sum(1 for p in team["players"] if p["started"]) == 5


def test_parse_boxscore_keeps_the_full_stat_line(page_html):
    """Scraping is the expensive step; discarding non-minutes stats would
    force a re-scrape to get points/rebounds later."""
    game = parse_boxscore(page_html, "401482889")
    player = game["teams"][0]["players"][0]
    assert player["stats"]["minutes"]
    assert "points" in player["stats"]
    assert player["athlete_id"] and player["athlete_name"]


def test_parse_boxscore_returns_none_without_data():
    assert parse_boxscore("<html>nothing</html>", "1") is None


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_validate_accepts_regulation(page_html):
    game = parse_boxscore(page_html, "401482889")
    assert all(validate_boxscore_minutes(game).values())


@pytest.mark.parametrize("total,ok", [(200, True), (225, True), (250, True), (140, False), (0, False)])
def test_validate_accepts_only_legal_minute_budgets(total, ok):
    """200 regulation, +25 per overtime; anything else is a parse failure."""
    game = {"teams": [{"team_id": "t", "players": [{"minutes": float(total)}]}]}
    assert validate_boxscore_minutes(game)["t"] is ok


def test_validate_ignores_dnp_rows():
    game = {"teams": [{"team_id": "t", "players": [{"minutes": 200.0}, {"minutes": None}]}]}
    assert validate_boxscore_minutes(game)["t"] is True


# ---------------------------------------------------------------------------
# Season features
# ---------------------------------------------------------------------------


# The fixture game is 2022-11-07, which belongs to the 2023 season (seasons
# run Nov Y-1 -> Mar Y). Labelling it 2022 trips the pre-tournament leakage
# guard, correctly — 2022's tournament started 2022-03-15.
_FIXTURE_SEASON = 2023


def _season_payload(game, n_games, year=_FIXTURE_SEASON):
    """Repeat one parsed game n times as distinct game_ids."""
    games = []
    for i in range(n_games):
        g = json.loads(json.dumps(game))
        g["game_id"] = f"g{i}"
        g["game_date"] = "2022-11-07"
        games.append(g)
    return {"season": year, "games": games}


def test_season_features_match_the_pbp_output_schema(page_html):
    game = parse_boxscore(page_html, "401482889")
    out = build_season_minutes_features(_FIXTURE_SEASON, "data", boxscore_payload=_season_payload(game, 3), strict=False)
    assert set(out) == {"season", "players", "generated_at", "source", "metadata"}
    assert set(out["metadata"]) == {"games_used", "games_rejected"}
    assert set(out["players"][0]) == {
        "team_id",
        "athlete_id",
        "athlete_name",
        "games_played",
        "games_started",
        "total_minutes",
        "minutes_per_game",
    }


def test_season_features_aggregate_across_games(page_html):
    game = parse_boxscore(page_html, "401482889")
    out = build_season_minutes_features(_FIXTURE_SEASON, "data", boxscore_payload=_season_payload(game, 3), strict=False)
    assert out["metadata"]["games_used"] == 3
    assert out["metadata"]["games_rejected"] == 0

    # Every team's season minutes = 3 games x 200.
    by_team: dict = {}
    for p in out["players"]:
        by_team[p["team_id"]] = by_team.get(p["team_id"], 0.0) + p["total_minutes"]
    assert all(total == 600 for total in by_team.values())

    starter = max(out["players"], key=lambda p: p["total_minutes"])
    assert starter["games_played"] == 3
    assert starter["minutes_per_game"] == pytest.approx(starter["total_minutes"] / 3)


def test_season_features_reject_games_failing_validation(page_html):
    game = parse_boxscore(page_html, "401482889")
    broken = json.loads(json.dumps(game))
    for team in broken["teams"]:
        for p in team["players"]:
            p["minutes"] = 1.0  # nowhere near a legal budget
    payload = {"season": _FIXTURE_SEASON, "games": [broken]}
    out = build_season_minutes_features(_FIXTURE_SEASON, "data", boxscore_payload=payload, strict=False)
    assert out["metadata"]["games_rejected"] == 1
    assert out["metadata"]["games_used"] == 0


def test_thin_season_raises_loudly(page_html):
    """The failure that went unnoticed for hours in the PBP backfill.

    2024 logged 'player_minutes produced nothing' and 2023 wrote 26 players,
    and the run sailed past both. A season this thin must now fail.
    """
    game = parse_boxscore(page_html, "401482889")
    with pytest.raises(MinutesCoverageError, match="signature of a parse failure"):
        build_season_minutes_features(_FIXTURE_SEASON, "data", boxscore_payload=_season_payload(game, 2), strict=True)


def test_missing_payload_raises_in_strict_mode(tmp_path):
    with pytest.raises(MinutesCoverageError, match="No boxscores_2022.json"):
        build_season_minutes_features(2022, tmp_path, strict=True)


def test_missing_payload_returns_empty_when_not_strict(tmp_path):
    assert build_season_minutes_features(2022, tmp_path, strict=False) == {}
