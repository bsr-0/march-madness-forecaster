"""The pipeline's team list must be the Round-of-64 field when results exist.

`load_teams` returned every entered team (68), both play-in teams sharing a
(region, seed) slot. `bracket_construction._build_seed_map` refuses such a
field since 2026-09-06 -- correctly -- so every pipeline run for every
historical season died in the pool/leverage section after 16 minutes of
training. Play-in games finish before brackets lock; their results are
pre-tournament information, and the backtest and the 2027 artifact path
already resolve the field from them. Now the pipeline does too.
"""

import pytest

from src.models.team import Team
from src.pipeline.stages import data_loader as dl


def _teams():
    return [
        Team(name="Alpha", seed=1, region="East"),
        Team(name="Beta A", seed=16, region="East"),
        Team(name="Beta B", seed=16, region="East"),
        Team(name="Gamma", seed=2, region="West"),
    ]


def test_first_four_loser_is_dropped_when_results_exist(monkeypatch):
    import src.prediction.noseed_model as nm

    monkeypatch.setattr(
        nm, "_load_tournament_results",
        lambda year: [{"round_name": "FF", "team1_id": "beta_a", "team2_id": "beta_b", "team1_won": False}],
    )
    kept, n = dl.resolve_first_four_teams(_teams(), 2099)
    assert n == 1
    assert {t.name for t in kept} == {"Alpha", "Beta B", "Gamma"}
    assert dl._duplicate_slots(kept) == []


def test_no_results_leaves_the_field_alone(monkeypatch):
    import src.prediction.noseed_model as nm

    monkeypatch.setattr(nm, "_load_tournament_results", lambda year: [])
    kept, n = dl.resolve_first_four_teams(_teams(), 2099)
    assert n == 0 and len(kept) == 4
    assert dl._duplicate_slots(kept) == [("East", 16)]


def test_no_year_is_a_noop():
    kept, n = dl.resolve_first_four_teams(_teams(), None)
    assert n == 0 and len(kept) == 4


@pytest.mark.integration
def test_real_2024_field_resolves_to_64():
    """Real data: the 2024 seeds file lists 68 teams; four First Four games
    were played; the pipeline must see exactly the 64 that played the R64."""
    from src.pipeline.config import ForecastConfig

    cfg = ForecastConfig(year=2024, teams_json="data/raw/historical/teams_2024.json")
    teams = dl.load_teams(cfg, bracket_pipeline=None)
    assert len(teams) == 64, f"expected 64, got {len(teams)}"
    assert dl._duplicate_slots(teams) == []
