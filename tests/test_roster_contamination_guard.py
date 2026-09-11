"""Roster overlays must not carry tournament outcomes into training rows.

`cbbpy_rosters_{year}.json` holds per-player `games_played`, and
`cbbpy_rosters.py` computes `warp = bpm * minute_share * games_played / 300`.
So if a season's roster file was scraped after that season's tournament, every
roster-derived feature is partly a measure of how far the team advanced —
including `diff_total_warp`, which `SIMPLE_FEATURE_SET` annotates as the
"largest coefficient".

Measured on the shipped files: the correlation between a team's roster
`games_played` and the tournament rounds it won is +0.49 to +0.83 in every
season 2011-2025, and the team with the most games in each file is that
season's champion or runner-up. 2026, whose snapshot is genuinely
mid-February, is the control at -0.05.

The guard that was supposed to catch this skipped precisely the contaminated
case (it bailed out when the file's `year` matched the season, which is true
of every per-season file on disk) and was warning-only.
"""

import json
import sys
import types
from datetime import date

import pytest


def _import_data_loader():
    """Import the module under test.

    `src/pipeline/stages/data_loader.py` imports
    `src.conference_tournament.data_enrichment`, a package deleted in commit
    44b048f (2026-04-21) and never restored, so the module is unimportable and
    the ML pipeline fails at runtime the first time it loads data. That is a
    separate defect, deliberately not papered over in the source; this stub
    exists only so the leakage guard can be tested at all, and it will become
    unnecessary the moment that import is resolved.
    """
    name = "src.conference_tournament.data_enrichment"
    if name not in sys.modules:
        pkg = types.ModuleType("src.conference_tournament")
        pkg.__path__ = []
        mod = types.ModuleType(name)
        mod.enrich_torvik_teams = lambda *a, **k: (a[0] if a else {})
        sys.modules.setdefault("src.conference_tournament", pkg)
        sys.modules[name] = mod
    from src.pipeline.stages import data_loader

    return data_loader


dl = _import_data_loader()
TOURNAMENT_START = dl.TOURNAMENT_START_DATES


def _roster_file(tmp_path, year, timestamp, games_played=39):
    payload = {
        "year": year,
        "timestamp": timestamp,
        "teams": [
            {
                "team_id": "purdue",
                "players": [
                    {
                        "name": f"p{i}",
                        "games_played": games_played,
                        "minutes_per_game": 30.0,
                        "warp": 2.0,
                        "rapm_total": 1.5,
                        "bpm": 5.0,
                    }
                    for i in range(8)
                ],
            }
        ],
    }
    path = tmp_path / f"cbbpy_rosters_{year}.json"
    path.write_text(json.dumps(payload))
    return str(path)


YEAR = 2024
AFTER_TOURNAMENT = str(TOURNAMENT_START[YEAR] + __import__("datetime").timedelta(days=30))
BEFORE_TOURNAMENT = str(TOURNAMENT_START[YEAR] - __import__("datetime").timedelta(days=20))


def test_pre_tournament_snapshot_is_used(tmp_path):
    overlay = dl.load_roster_overlay(_roster_file(tmp_path, YEAR, BEFORE_TOURNAMENT), year=YEAR)
    assert overlay, "a clean, genuinely pre-tournament roster file must still be used"
    assert "purdue" in overlay


def test_post_tournament_snapshot_is_dropped(tmp_path):
    """The real files are all like this: scraped 2026-02-21, i.e. years after
    the season they describe."""
    overlay = dl.load_roster_overlay(_roster_file(tmp_path, YEAR, AFTER_TOURNAMENT), year=YEAR)
    assert overlay == {}, "a contaminated roster file must not reach the feature vector"


def test_matching_file_year_is_not_an_exemption(tmp_path):
    """The old guard skipped when `file_year == year`, calling it
    'season-specific historical data scraped retroactively'. That is the
    contaminated case, not an exemption — and it is true of every file on disk."""
    path = _roster_file(tmp_path, YEAR, AFTER_TOURNAMENT)
    assert json.loads(open(path).read())["year"] == YEAR
    assert dl.load_roster_overlay(path, year=YEAR) == {}


def test_strict_mode_raises_instead_of_dropping(tmp_path):
    with pytest.raises(dl.RosterContaminationError) as exc:
        dl.load_roster_overlay(_roster_file(tmp_path, YEAR, AFTER_TOURNAMENT), year=YEAR, strict=True)
    assert "games_played" in str(exc.value)


def test_no_year_means_no_guard(tmp_path):
    """Callers that do not say which season the file is for get no check —
    there is nothing to compare the timestamp against."""
    overlay = dl.load_roster_overlay(_roster_file(tmp_path, YEAR, AFTER_TOURNAMENT))
    assert overlay


def test_the_real_shipped_files_are_contaminated():
    """Not a synthetic case: this is the state of the repo's own data."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    checked = 0
    for year in (2018, 2022, 2024):
        path = root / "data" / "raw" / "historical" / f"cbbpy_rosters_{year}.json"
        if not path.exists():
            continue
        checked += 1
        assert dl.load_roster_overlay(str(path), year=year) == {}, (
            f"cbbpy_rosters_{year}.json is scraped post-tournament and must be dropped"
        )
    if checked == 0:
        pytest.skip("no historical roster files present")
