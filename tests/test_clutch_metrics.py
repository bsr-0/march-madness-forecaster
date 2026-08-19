"""Tests for blown-lead/clutch feature aggregation from play-by-play.

Uses synthetic PBP fixtures (no live network) — the real cbbpy column
mapping in cbbpy_pbp.py is exercised separately by a live pilot run per
the plan; these tests only cover the aggregation math, which is decoupled
from raw source columns by cbbpy_pbp._normalize_play_row.
"""

import json
from datetime import date

import pytest

from src.data.features.clutch_metrics import (
    GameClutchRecord,
    aggregate_team_season_clutch,
    build_game_clutch_records,
    build_season_clutch_features,
    compute_game_margin_trajectory,
)


def _play(period, seconds_remaining, home_score, away_score):
    return {
        "period": period,
        "seconds_remaining": seconds_remaining,
        "home_score": home_score,
        "away_score": away_score,
    }


class TestMarginTrajectory:
    def test_blown_20_point_lead(self):
        # Home team builds a 20-point lead in the 2nd half, then loses by 2.
        plays = [
            _play(1, 1200, 0, 0),
            _play(2, 1200, 45, 25),  # home +20
            _play(2, 600, 50, 40),  # home +10 at 10min mark
            _play(2, 300, 55, 52),  # home +3 at 5min mark
            _play(2, 60, 58, 60),  # home -2 at 1min mark
            _play(2, 0, 60, 62),  # final: home loses by 2
        ]
        result = compute_game_margin_trajectory(plays, "home", "away")
        assert result is not None
        home, away = result

        assert home["max_lead"] == 20
        assert home["final_margin"] == -2
        # Blew the full 20-point lead (final margin capped at 0 for this calc).
        assert home["largest_lead_blown"] == 20
        assert home["margin_at"]["10min"] == 10
        assert home["margin_at"]["5min"] == 3
        assert home["margin_at"]["1min"] == -2

        assert away["max_deficit"] == 20
        assert away["final_margin"] == 2
        assert away["largest_lead_blown"] == 0  # away never led, nothing to blow

    def test_no_plays_returns_none(self):
        assert compute_game_margin_trajectory([], "home", "away") is None

    def test_wire_to_wire_win_has_zero_blown_lead(self):
        plays = [
            _play(1, 1200, 0, 0),
            _play(2, 1200, 40, 20),
            _play(2, 0, 70, 50),
        ]
        home, _ = compute_game_margin_trajectory(plays, "home", "away")
        assert home["largest_lead_blown"] == 0.0
        assert home["final_margin"] == 20


class TestGameClutchRecords:
    def test_builds_both_sides(self):
        game_payload = {
            "game_id": "g1",
            "home_team_raw": "duke_blue_devils",
            "away_team_raw": "unc_tar_heels",
            "plays": [
                _play(1, 1200, 0, 0),
                _play(2, 1200, 40, 20),
                _play(2, 0, 65, 60),
            ],
        }
        records = build_game_clutch_records(game_payload)
        assert len(records) == 2
        ids = {r.team_id for r in records}
        assert ids == {"duke_blue_devils", "unc_tar_heels"}
        home_rec = next(r for r in records if r.team_id == "duke_blue_devils")
        assert home_rec.won is True
        assert home_rec.opponent_id == "unc_tar_heels"

    def test_missing_team_refs_returns_empty(self):
        game_payload = {"game_id": "g1", "plays": [_play(1, 1200, 0, 0)]}
        assert build_game_clutch_records(game_payload) == []


class TestSeasonAggregation:
    def test_blown_lead_rate_and_close_game_rate(self):
        # Team A: blew a 10pt lead in game 1, protected a lead in game 2.
        records = [
            GameClutchRecord(
                game_id="g1",
                team_id="team_a",
                opponent_id="team_b",
                won=False,
                final_margin=-2,
                max_lead=15,
                max_deficit=2,
                largest_lead_blown=15,
                margin_at={"10min": 12, "5min": 5, "2min": 2, "1min": -2},
            ),
            GameClutchRecord(
                game_id="g2",
                team_id="team_a",
                opponent_id="team_c",
                won=True,
                final_margin=10,
                max_lead=12,
                max_deficit=0,
                largest_lead_blown=0,
                margin_at={"10min": 8, "5min": 8, "2min": 10, "1min": 10},
            ),
        ]
        stats = aggregate_team_season_clutch(records)
        team_a = stats["team_a"]
        assert team_a["games_with_clutch_data"] == 2
        # Both games had a lead >= 10; one blew it.
        assert team_a["blown_10pt_lead_rate"] == 0.5
        # game 1 was a 2pt loss (close game), game 2 wasn't close (10pt win).
        assert team_a["close_game_win_rate"] == 0.0
        # Led at 5min in both games; won only game 2.
        assert team_a["win_rate_when_leading_at_5min"] == 0.5

    def test_teams_never_leading_get_none_not_zero(self):
        records = [
            GameClutchRecord(
                game_id="g1",
                team_id="team_a",
                opponent_id="team_b",
                won=False,
                final_margin=-30,
                max_lead=0,
                max_deficit=30,
                largest_lead_blown=0,
                margin_at={"10min": -20, "5min": -25, "2min": -28, "1min": -30},
            ),
        ]
        stats = aggregate_team_season_clutch(records)
        # Never held any lead >= 10 -> rate is undefined, not 0.
        assert stats["team_a"]["blown_10pt_lead_rate"] is None
        assert stats["team_a"]["win_rate_when_leading_at_5min"] is None


class TestBuildSeasonClutchFeatures:
    def test_rejects_post_cutoff_games(self, monkeypatch, tmp_path):
        # Fake a 2024 cutoff and a game on/after it — must raise, not silently drop.
        fake_cutoff = date(2024, 3, 19)
        monkeypatch.setattr("src.pipeline.config.TOURNAMENT_START_DATES", {2024: fake_cutoff}, raising=False)
        payload = {
            "season": 2024,
            "games": [
                {
                    "game_id": "g1",
                    "game_date": "2024-03-19",  # on the cutoff -> leakage
                    "home_team_raw": "duke_blue_devils",
                    "away_team_raw": "unc_tar_heels",
                    "plays": [_play(2, 0, 60, 50)],
                }
            ],
        }
        from src.exceptions import LeakageError

        with pytest.raises(LeakageError):
            build_season_clutch_features(2024, tmp_path, pbp_payload=payload)

    def test_empty_games_returns_empty_dict(self, tmp_path):
        assert build_season_clutch_features(2024, tmp_path, pbp_payload={"games": []}) == {}

    def test_bridges_against_real_tournament_context_shape(self, tmp_path):
        # Regression test: tournament_context_{year}.json's "seeds" field is
        # {"season": ..., "teams": [{"team_id": ..., "seed": ...}, ...]} --
        # NOT a flat {team_id: seed} dict. An earlier version of
        # _load_canonical_ids assumed the flat shape, silently treated
        # ctx["seeds"].keys() (== {"season", "teams"}) as the canonical ID
        # universe, and produced zero bridged teams for every real payload
        # (only caught by a live pilot run against real data, not by tests
        # that only exercised the load_d1_team_ids fallback path).
        historical = tmp_path / "raw" / "historical"
        historical.mkdir(parents=True)
        (historical / "tournament_context_2025.json").write_text(
            json.dumps(
                {
                    "seeds": {
                        "season": 2025,
                        "teams": [
                            {"team_name": "Duke", "team_id": "duke", "seed": 1},
                            {"team_name": "Houston", "team_id": "houston", "seed": 1},
                        ],
                    }
                }
            )
        )

        payload = {
            "season": 2025,
            "games": [
                {
                    "game_id": "g1",
                    "game_date": "2024-11-05",
                    "home_team_raw": "duke_blue_devils",
                    "away_team_raw": "maine_black_bears",
                    "plays": [
                        _play(1, 1200, 0, 0),
                        _play(2, 1200, 40, 20),
                        _play(2, 0, 70, 50),
                    ],
                }
            ],
        }
        result = build_season_clutch_features(2025, tmp_path, pbp_payload=payload)
        team_ids = {t["team_id"] for t in result["teams"]}
        assert "duke" in team_ids
