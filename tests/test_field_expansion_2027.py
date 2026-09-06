"""The 2027 field is 76 teams and 12 play-in games. The pipeline assumed 68/4.

FINDINGS §8 recorded this as open: "NCAA tournament expands 68->76 teams (12
play-in games producing duplicate seeds in the seeds file) -- pipeline currently
assumes 64+4 (First Four) and needs structural changes."

The main draw is still 64 (76 - 12 = 64), so the bracket maths and the (1, 63)
encoding survive. What breaks is INGESTION: both teams in a play-in game share a
(region, seed) slot, and ``build_bracket_order`` resolved that collision by dict
insertion order -- the order of lines in the seeds file.

That was already wrong at four slots: the shipped 2026 artifact put lehigh in
the South 16 slot when prairie_view won that game and played the Round of 64.
Three of the four matched the real winner by luck. At twelve slots, luck is not
a plan.

These tests use synthetic fields because 2027's does not exist yet, and because
the point is the SHAPE of the field, not its teams.
"""

from __future__ import annotations

import pytest

REGIONS = ("East", "West", "South", "Midwest")


def _field(play_in_slots: int):
    """A tournament field with ``play_in_slots`` contested (region, seed) slots.

    64 main-draw slots, plus one extra team in each contested slot -- so 68
    teams at 4 (the format through 2026) and 76 at 12 (2027).
    """
    seeds, regions, contested = {}, {}, []
    for region in REGIONS:
        for seed in range(1, 17):
            tid = f"{region.lower()}_{seed}"
            seeds[tid] = seed
            regions[tid] = region

    slots = [(r, s) for r in REGIONS for s in (16, 11, 12, 15)][:play_in_slots]
    for region, seed in slots:
        tid = f"{region.lower()}_{seed}_challenger"
        seeds[tid] = seed
        regions[tid] = region
        contested.append((region, seed, f"{region.lower()}_{seed}", tid))
    return seeds, regions, contested


def _play_in_games(contested, winner_is_challenger=True):
    """Results in the shape resolve_first_four reads."""
    games = []
    for _region, _seed, incumbent, challenger in contested:
        games.append(
            {
                "round_name": "FF",
                "team1_id": challenger if winner_is_challenger else incumbent,
                "team2_id": incumbent if winner_is_challenger else challenger,
                "team1_won": True,
            }
        )
    return games


class TestAnUndeterminedDrawIsRefused:
    """The failure that shipped: a contested slot settled by file order."""

    @pytest.mark.parametrize("play_in_slots", [4, 12])
    def test_unresolved_slots_raise_rather_than_guess(self, play_in_slots):
        from scripts.mc_pool_backtest import build_bracket_order

        seeds, regions, _ = _field(play_in_slots)
        with pytest.raises(ValueError, match="more than one team"):
            build_bracket_order(seeds, regions)

    def test_the_error_names_the_contested_slots(self):
        from scripts.mc_pool_backtest import build_bracket_order

        seeds, regions, contested = _field(12)
        with pytest.raises(ValueError) as exc:
            build_bracket_order(seeds, regions)
        assert "12 bracket slot(s)" in str(exc.value)
        region, seed, incumbent, challenger = contested[0]
        assert incumbent in str(exc.value) and challenger in str(exc.value)


class TestTwelvePlayInGamesResolveToSixtyFour:
    @pytest.mark.parametrize("play_in_slots,entered", [(4, 68), (12, 76)])
    def test_the_draw_comes_out_at_64(self, play_in_slots, entered):
        from scripts.mc_pool_backtest import build_bracket_order, resolve_first_four

        seeds, regions, contested = _field(play_in_slots)
        assert len(seeds) == entered

        replaced = resolve_first_four(_play_in_games(contested), seeds, regions)
        assert replaced == play_in_slots
        assert len(seeds) == 64

        order = build_bracket_order(seeds, regions)
        assert len(order) == 64
        assert not [t for t in order if t.startswith("unknown_")], "placeholder slots in the draw"
        assert len(set(order)) == 64, "a team appears twice in the draw"

    def test_the_winner_is_the_team_that_won(self):
        """Not the one that happened to be later in the file."""
        from scripts.mc_pool_backtest import build_bracket_order, resolve_first_four

        seeds, regions, contested = _field(12)
        resolve_first_four(_play_in_games(contested, winner_is_challenger=False), seeds, regions)
        order = set(build_bracket_order(seeds, regions))
        for _region, _seed, incumbent, challenger in contested:
            assert incumbent in order, f"{incumbent} won its play-in game and is not in the draw"
            assert challenger not in order, f"{challenger} lost its play-in game and is in the draw"


class TestTheArtifactPathResolvesTheField:
    def test_resolve_field_reports_what_it_did(self):
        from scripts._common import load_seeds_and_regions
        from scripts.experiments.build_candidate_artifact import resolve_field

        seeds, regions = load_seeds_and_regions(2026)
        prov = resolve_field(2026, seeds, regions)
        assert prov["teams_entered"] == 68
        assert prov["play_in_games"] == 4
        assert prov["slots_resolved"] == 4
        assert prov["main_draw"] == 64

    def test_the_real_play_in_winner_reaches_the_draw(self):
        """2026 shipped with lehigh; prairie_view won that game."""
        from scripts._common import load_seeds_and_regions
        from scripts.experiments.build_candidate_artifact import resolve_field

        seeds, regions = load_seeds_and_regions(2026)
        resolve_field(2026, seeds, regions)
        assert "prairie_view" in seeds
        assert "lehigh" not in seeds

    def test_a_field_that_cannot_reach_64_is_refused(self, monkeypatch):
        """Building before the play-in games are played must fail, not guess."""
        import scripts.experiments.build_candidate_artifact as bca
        from scripts._common import load_seeds_and_regions

        monkeypatch.setattr(
            "src.prediction.noseed_model._load_tournament_results", lambda year: []
        )
        seeds, regions = load_seeds_and_regions(2026)
        with pytest.raises(RuntimeError, match="undetermined|needs exactly 64"):
            bca.resolve_field(2026, seeds, regions)
