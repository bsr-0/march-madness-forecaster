"""The two copies of the bracket inside each tournament_context file must agree.

Found 2026-09-11: ``teams.teams`` (what the loader reads) disagreed with
``seeds.teams`` (the scraped bracket) in five seasons. 2019 filed the West
16-seed play-in pair under East, so after First Four resolution the West had
15 teams and the walk-forward harness died four folds in. 2023 seeded College
of Charleston 13 instead of 12. Nothing checked, because the loader only
reads one copy.

Two guards: every evaluation season's blocks agree on seed and region for
every team, and every region of the loader's copy contains seeds 1..16.
Plus unit coverage for the reconcile script's transformation.
"""

import json
from pathlib import Path

import pytest

from scripts.mc_pool_backtest import EVALUATION_YEARS
from scripts.reconcile_tournament_context import diff_blocks, reconcile

HIST = Path(__file__).resolve().parents[1] / "data" / "raw" / "historical"


def _ctx(year):
    p = HIST / f"tournament_context_{year}.json"
    if not p.exists():
        pytest.skip(f"no context file for {year}")
    return json.loads(p.read_text())


@pytest.mark.parametrize("year", sorted(EVALUATION_YEARS))
def test_teams_block_agrees_with_seeds_block(year):
    ctx = _ctx(year)
    changed, added = diff_blocks(ctx)
    assert not changed and not added, f"{year}: changed={changed} missing={[(s['team_id'], s['seed'], s['region']) for s in added]}"


@pytest.mark.parametrize("year", sorted(EVALUATION_YEARS))
def test_every_region_has_all_sixteen_seeds(year):
    ctx = _ctx(year)
    by_region = {}
    for t in ctx["teams"]["teams"]:
        by_region.setdefault(t["region"], set()).add(int(t["seed"]))
    assert len(by_region) == 4, f"{year}: regions {sorted(by_region)}"
    for region, seeds in by_region.items():
        missing = set(range(1, 17)) - seeds
        assert not missing, f"{year} {region}: missing seeds {sorted(missing)}"


def test_reconcile_copies_seed_and_region_and_adds_missing():
    ctx = {
        "teams": {"teams": [
            {"name": "Memphis", "team_id": "memphis", "seed": 6, "region": "Midwest", "rating": 1500.0},
        ]},
        "seeds": {"teams": [
            {"team_name": "Memphis", "team_id": "memphis", "seed": 1, "region": "South", "school_slug": "memphis"},
            {"team_name": "USC", "team_id": "southern_california", "seed": 6, "region": "Midwest", "school_slug": "southern-california"},
        ]},
    }
    changed, added = reconcile(ctx)
    assert changed == [("memphis", (6, "Midwest"), (1, "South"))]
    assert [s["team_id"] for s in added] == ["southern_california"]
    rows = {t["team_id"]: t for t in ctx["teams"]["teams"]}
    assert (rows["memphis"]["seed"], rows["memphis"]["region"]) == (1, "South")
    assert rows["memphis"]["rating"] == 1500.0, "other fields untouched"
    assert rows["southern_california"]["name"] == "USC"
    assert diff_blocks(ctx) == ([], []), "idempotent after reconcile"
