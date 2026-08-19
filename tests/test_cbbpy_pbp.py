"""Tests for the ESPN-HTML-based play-by-play scraper.

Uses a fixture built from the *confirmed live* schema (game 401714261,
2025-02-10, verified against real bytes — see cbbpy_pbp.py's module
docstring) rather than a guess, so a real ESPN schema drift shows up here
before it silently breaks aggregation.
"""

import json

from src.data.scrapers.cbbpy_pbp import (
    CBBpyPbpScraper,
    _extract_json_array,
    _slugify_team_name,
)

# Trimmed down real play objects (fields we don't use omitted), confirmed
# live against https://www.espn.com/mens-college-basketball/playbyplay/_/gameId/401714261
_REAL_PLAYS_FIXTURE = [
    {
        "id": "401714261101806001",
        "period": {"number": 1},
        "text": "Jahnathan Lamothe missed Three Point Jumper.",
        "homeAway": "away",
        "athlete": {
            "id": "4711264",
            "name": "Jahnathan Lamothe",
            "team": "North Carolina A&T Aggies",
        },
        "coordinate": {"x": 25, "y": 0},
        "shootingPlay": True,
        "pointsAttempted": 3,
        "type": {"categoryId": "1006", "id": "558", "txt": "JumpShot"},
        "title": "Missed 3PT",
        "favoredTeam": {"isAway": False, "winProbability": "88.2"},
        "clock": {"value": 1179, "displayValue": "19:39"},
        "awScr": 0,
        "hmScr": 0,
    },
    {
        "id": "401714261101807901",
        "period": {"number": 1},
        "text": "Colby Duggan made Layup.",
        "homeAway": "home",
        "scoringPlay": True,
        "athlete": {"team": "Campbell Fighting Camels"},
        "clock": {"value": 1160, "displayValue": "19:20"},
        "awScr": 0,
        "hmScr": 2,
    },
    {
        "id": "401714261102999905",
        "period": {"number": 2},
        "text": "End of Game",
        "clock": {"value": 0, "displayValue": "0:00"},
        "awScr": 62,
        "hmScr": 66,
        # No 'homeAway'/'athlete' -- confirmed real plays like "End of Game"
        # lack team attribution; normalization must not crash on this.
    },
]


class TestSlugifyTeamName:
    def test_matches_cbbpy_id_convention(self):
        assert _slugify_team_name("North Carolina A&T Aggies") == "north_carolina_a_t_aggies"
        assert _slugify_team_name("Campbell Fighting Camels") == "campbell_fighting_camels"

    def test_collapses_repeated_separators(self):
        assert _slugify_team_name("St. John's  (NY)") == "st_john_s_ny"


class TestExtractJsonArray:
    def test_extracts_valid_array_embedded_in_html(self):
        html = f'<script>window.__data = {{"plays":{json.dumps(_REAL_PLAYS_FIXTURE)},"other":1}}</script>'
        result = _extract_json_array(html, "plays")
        assert result == _REAL_PLAYS_FIXTURE

    def test_bracket_matching_ignores_brackets_inside_strings(self):
        tricky = [{"text": "Team [scores] a basket [somehow]"}]
        html = f'"plays":{json.dumps(tricky)}'
        result = _extract_json_array(html, "plays")
        assert result == tricky

    def test_missing_key_returns_none(self):
        assert _extract_json_array("<html>no data here</html>", "plays") is None

    def test_malformed_json_returns_none(self):
        assert _extract_json_array('"plays":[{"a": }]', "plays") is None


class TestNormalizePlayRow:
    def test_maps_confirmed_espn_fields(self):
        normalized = CBBpyPbpScraper._normalize_play_row("g1", _REAL_PLAYS_FIXTURE[0])
        assert normalized == {
            "game_id": "g1",
            "period": 1,
            "seconds_remaining": 1179.0,
            "home_score": 0,
            "away_score": 0,
            "home_away": "away",
            "scoring_play": False,
            "shooting_play": True,
            "points_attempted": 3,
            "play_type": "JumpShot",
            "play_type_category_id": "1006",
            "text": "Jahnathan Lamothe missed Three Point Jumper.",
            "athlete_id": "4711264",
            "athlete_name": "Jahnathan Lamothe",
            "athlete_team": "north_carolina_a_t_aggies",
            "win_probability": 88.2,
            "favored_is_away": False,
            "coordinate_x": 25,
            "coordinate_y": 0,
        }

    def test_end_of_game_row_without_team_still_normalizes(self):
        # 'End of Game' has no homeAway/athlete but does have clock/period/scores.
        normalized = CBBpyPbpScraper._normalize_play_row("g1", _REAL_PLAYS_FIXTURE[2])
        assert normalized["home_score"] == 66
        assert normalized["away_score"] == 62
        assert normalized["athlete_team"] is None
        assert normalized["home_away"] is None

    def test_uses_title_as_text_fallback_when_text_absent(self):
        row = {
            "period": {"number": 1},
            "clock": {"value": 500, "displayValue": "8:20"},
            "awScr": 10,
            "hmScr": 12,
            "title": "Timeout",
        }
        normalized = CBBpyPbpScraper._normalize_play_row("g1", row)
        assert normalized["text"] == "Timeout"

    def test_row_missing_clock_returns_none(self):
        row = {"period": {"number": 1}, "awScr": 0, "hmScr": 0}
        assert CBBpyPbpScraper._normalize_play_row("g1", row) is None


class TestBuildGamePayload:
    def test_derives_home_away_team_from_first_matching_play(self):
        scraper = CBBpyPbpScraper()
        payload = scraper._build_game_payload("401714261", "2025-02-10", _REAL_PLAYS_FIXTURE)
        assert payload["home_team_raw"] == "campbell_fighting_camels"
        assert payload["away_team_raw"] == "north_carolina_a_t_aggies"
        # All 3 fixture rows have clock+period+scores, so all 3 normalize.
        assert len(payload["plays"]) == 3
        assert payload["plays"][-1]["home_score"] == 66


class TestFetchSeasonPbpCheckpointing:
    """A full historical backfill runs for hours/days at a deliberately slow,
    respectful pace -- these lock in that an interruption loses at most one
    day's worth of work, and a resumed run doesn't re-scrape from scratch.
    """

    def _patch_network(self, monkeypatch, day_to_ids, game_pbp):
        def fake_scrape_ids(day_str, http_timeout=15):
            return day_to_ids.get(day_str, [])

        def fake_fetch_pbp(self, game_id, delay=0.0):
            return game_pbp.get(game_id, [])

        monkeypatch.setattr(CBBpyPbpScraper, "_scrape_game_ids_for_date", staticmethod(fake_scrape_ids))
        monkeypatch.setattr(CBBpyPbpScraper, "_fetch_game_pbp", fake_fetch_pbp)
        monkeypatch.setattr("time.sleep", lambda *_a, **_kw: None)

    def test_writes_checkpoint_after_each_day(self, monkeypatch, tmp_path):
        from datetime import date as _date

        from src.pipeline import config as pipeline_config

        monkeypatch.setattr(pipeline_config, "TOURNAMENT_START_DATES", {2025: _date(2024, 11, 4)})

        day_to_ids = {"2024-11-01": ["g1"], "2024-11-02": ["g2"], "2024-11-03": ["g3"]}
        game_pbp = {gid: _REAL_PLAYS_FIXTURE for gid in ["g1", "g2", "g3"]}
        self._patch_network(monkeypatch, day_to_ids, game_pbp)

        scraper = CBBpyPbpScraper(cache_dir=str(tmp_path))
        result = scraper.fetch_season_pbp(2025)

        assert result["metadata"]["complete"] is True
        assert {g["game_id"] for g in result["games"]} == {"g1", "g2", "g3"}

    def test_resumes_from_last_completed_date_without_rescraping(self, monkeypatch, tmp_path):
        from datetime import date as _date

        from src.pipeline import config as pipeline_config

        monkeypatch.setattr(pipeline_config, "TOURNAMENT_START_DATES", {2025: _date(2024, 11, 4)})

        day_to_ids = {"2024-11-01": ["g1"], "2024-11-02": ["g2"], "2024-11-03": ["g3"]}
        game_pbp = {gid: _REAL_PLAYS_FIXTURE for gid in ["g1", "g2", "g3"]}

        calls = {"g2": 0, "g3": 0}
        orig_pbp = dict(game_pbp)

        def counting_fetch(self, game_id, delay=0.0):
            if game_id in calls:
                calls[game_id] += 1
            return orig_pbp.get(game_id, [])

        monkeypatch.setattr(
            CBBpyPbpScraper, "_scrape_game_ids_for_date", staticmethod(lambda d, http_timeout=15: day_to_ids.get(d, []))
        )
        monkeypatch.setattr(CBBpyPbpScraper, "_fetch_game_pbp", counting_fetch)
        monkeypatch.setattr("time.sleep", lambda *_a, **_kw: None)

        cache_dir = str(tmp_path)
        # Simulate an interrupted first run that only got through day 1 by
        # writing a partial (incomplete) cache directly.
        scraper = CBBpyPbpScraper(cache_dir=cache_dir)
        partial_game = scraper._build_game_payload("g1", "2024-11-01", _REAL_PLAYS_FIXTURE)
        scraper._save_cache(
            "pbp_2025.json",
            {
                "season": 2025,
                "source": "espn_playbyplay_html",
                "cutoff_date": "2024-11-04",
                "games": [partial_game],
                "metadata": {
                    "raw_game_count": 1,
                    "date_window": ["2024-11-01", "2024-11-03"],
                    "include_tournament": False,
                    "complete": False,
                    "last_completed_date": "2024-11-01",
                },
            },
        )

        result = scraper.fetch_season_pbp(2025)

        # Day 1's game must not be re-fetched -- it was already cached.
        assert calls["g2"] == 1
        assert calls["g3"] == 1
        assert {g["game_id"] for g in result["games"]} == {"g1", "g2", "g3"}
        assert result["metadata"]["complete"] is True

    def test_already_complete_cache_is_returned_without_any_fetch(self, monkeypatch, tmp_path):
        def boom(*_a, **_kw):
            raise AssertionError("should not fetch when cache is already complete")

        monkeypatch.setattr(CBBpyPbpScraper, "_scrape_game_ids_for_date", staticmethod(boom))
        monkeypatch.setattr(CBBpyPbpScraper, "_fetch_game_pbp", boom)

        scraper = CBBpyPbpScraper(cache_dir=str(tmp_path))
        scraper._save_cache(
            "pbp_2025.json",
            {
                "season": 2025,
                "games": [{"game_id": "g1", "plays": []}],
                "metadata": {"complete": True, "last_completed_date": "2025-03-17"},
            },
        )
        result = scraper.fetch_season_pbp(2025)
        assert result["metadata"]["complete"] is True
