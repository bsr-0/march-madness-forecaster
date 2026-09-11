"""Rebuilt roster features must not encode tournament outcomes.

The cbbpy roster files carry each team's tournament run inside
``games_played`` (audit H5). The rebuild aggregates only games dated before
the season's tournament start, so the one number that exposed the leak --
the correlation between a team's roster ``games_played`` and the tournament
rounds it went on to win -- must be ~0 on the rebuilt file, where it was
+0.49 to +0.83 on the originals.
"""

import json
from datetime import date
from pathlib import Path

import pytest

from src.data.features import boxscore_rosters as br

ROOT = Path(__file__).resolve().parents[1]


def _game(game_id, game_date, team, players):
    return {
        "game_id": game_id,
        "game_date": game_date,
        "teams": [
            {
                "team_id": team,
                "team_display": team,
                "espn_team_id": "1",
                "home": True,
                "players": [
                    {
                        "athlete_id": f"{team}-{i}",
                        "athlete_name": f"P{i}",
                        "position": "G",
                        "started": i < 5,
                        "minutes": 40.0 if i < 5 else 0.0,
                        "stats": {
                            "minutes": "40" if i < 5 else "0",
                            "points": "10",
                            "fieldGoalsMade-fieldGoalsAttempted": "4-8",
                            "threePointFieldGoalsMade-threePointFieldGoalsAttempted": "1-2",
                            "freeThrowsMade-freeThrowsAttempted": "1-2",
                            "rebounds": "5",
                            "offensiveRebounds": "2",
                            "defensiveRebounds": "3",
                            "assists": "2",
                            "turnovers": "1",
                            "steals": "1",
                            "blocks": "0",
                            "fouls": "2",
                        },
                    }
                    for i in range(players)
                ],
            }
        ],
    }


def test_games_on_or_after_cutoff_are_excluded():
    cutoff = date(2030, 3, 19)
    payload = {
        "games": [
            _game("g1", "2030-03-01", "alpha_hawks", 5),
            _game("g2", "2030-03-18", "alpha_hawks", 5),  # day before: kept
            _game("g3", "2030-03-19", "alpha_hawks", 5),  # cutoff day: dropped
            _game("g4", "2030-04-01", "alpha_hawks", 5),  # tournament: dropped
        ]
    }
    rows, games_by_team, stats = br.boxscore_rows_for_season(payload, cutoff)
    assert games_by_team == {"alpha_hawks": 2}
    assert stats["games_used"] == 2
    assert stats["games_on_or_after_cutoff"] == 2
    assert stats["max_game_date"] == "2030-03-18"
    assert {r["game_id"] for r in rows} == {"g1", "g2"}


def test_made_attempted_strings_are_split():
    assert br._split_made_attempted("4-8") == (4.0, 8.0)
    assert br._split_made_attempted("--") == (0.0, 0.0)
    assert br._split_made_attempted(None) == (0.0, 0.0)


def test_teams_failing_the_minutes_check_are_rejected():
    cutoff = date(2030, 3, 19)
    bad = _game("g1", "2030-03-01", "alpha_hawks", 5)
    bad["teams"][0]["players"][0]["minutes"] = 400.0  # impossible total
    rows, games_by_team, stats = br.boxscore_rows_for_season({"games": [bad]}, cutoff)
    assert rows == []
    assert stats["team_games_rejected_minutes"] == 1


@pytest.mark.integration
def test_rebuilt_2024_file_does_not_encode_tournament_advancement():
    """Real data, real outcomes: the leak signature must be gone."""
    from scipy import stats as sp

    from scripts._common import load_tournament_results

    path = ROOT / "data" / "raw" / "historical" / "rosters_boxscore_2024.json"
    if not path.exists():
        pytest.skip("rosters_boxscore_2024.json not built (python -m scripts.build_boxscore_rosters)")
    d = json.loads(path.read_text())
    assert d["data_type"] == "pre_tournament_rosters"
    assert d["max_game_date"] < d["cutoff_date"]

    wins = {}
    for g in load_tournament_results(2024):
        if g.get("round_name") not in ("R64", "R32", "S16", "E8", "F4", "NCG"):
            continue
        w = g["team1_id"] if g["team1_won"] else g["team2_id"]
        loser = g["team2_id"] if g["team1_won"] else g["team1_id"]
        wins[w] = wins.get(w, 0) + 1
        wins.setdefault(loser, 0)
    games = {t["team_id"]: max(p["games_played"] for p in t["players"]) for t in d["teams"] if t["players"]}
    common = [t for t in wins if t in games]
    assert len(common) >= 55, f"only {len(common)} tournament teams bridged"
    r = sp.pearsonr([games[t] for t in common], [wins[t] for t in common])[0]
    assert abs(r) < 0.25, f"games_played still tracks rounds won: r={r:+.3f} (cbbpy file was +0.686)"


@pytest.mark.integration
def test_rebuilt_file_is_accepted_by_the_contamination_guard():
    from src.pipeline.stages.data_loader import load_roster_overlay

    path = ROOT / "data" / "raw" / "historical" / "rosters_boxscore_2024.json"
    if not path.exists():
        pytest.skip("rosters_boxscore_2024.json not built")
    overlay = load_roster_overlay(str(path), year=2024, strict=True)
    assert len(overlay) > 300
