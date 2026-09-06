"""2011's regions were named Southeast/Southwest, and one loader didn't know.

Bracket layout keys on the canonical four region names (East/West/South/
Midwest). ``mc_pool_backtest`` normalized 2011 through a private alias map;
``scripts/_common`` -- the loader ``build_candidate_artifact`` actually uses --
did not, and the divergence was documented in ``_common``'s header as a
harmless simplification.

It was not harmless. With 2011 unnormalized, every Southeast and Southwest team
failed the region lookup in ``build_bracket_order``, half the field was replaced
by ``unknown_South_*`` / ``unknown_Midwest_*`` placeholders, and the build died
looking one of them up by team index -- so 2011 could not be shown in the UI at
all. The map now has one definition that both loaders import.
"""

from __future__ import annotations

import pytest

from scripts._common import REGION_ALIASES, load_seeds_and_regions

CANONICAL = {"East", "West", "South", "Midwest"}
ALIASED_SEASON = 2011


def _seasons_on_disk():
    from src.data.season_calendar import SELECTION_SUNDAY_DATES

    return [y for y in sorted(SELECTION_SUNDAY_DATES) if load_seeds_and_regions(y)[0]]


class TestAliasesAreApplied:
    def test_2011_resolves_to_canonical_regions(self):
        _seeds, regions = load_seeds_and_regions(ALIASED_SEASON)
        assert regions, "2011 seed data should be on disk"
        assert set(regions.values()) == CANONICAL

    def test_no_season_leaks_a_non_canonical_region(self):
        """Any season whose regions are not the canonical four breaks the layout.

        Stated over every season on disk rather than 2011 alone: the failure is
        silent (placeholder slots, not an error) so a second oddly-named season
        would not announce itself.
        """
        offenders = {}
        for year in _seasons_on_disk():
            _seeds, regions = load_seeds_and_regions(year)
            extra = set(regions.values()) - CANONICAL
            if extra:
                offenders[year] = sorted(extra)
        assert not offenders, f"non-canonical region names survive normalization: {offenders}"


class TestTheTwoLoadersCannotDisagree:
    def test_both_import_the_same_map(self):
        from scripts.mc_pool_backtest import _REGION_ALIASES

        assert _REGION_ALIASES is REGION_ALIASES

    @pytest.mark.parametrize("year", [ALIASED_SEASON, 2013, 2026])
    def test_both_loaders_agree_on_regions(self, year):
        from scripts.mc_pool_backtest import load_seeds_and_regions as backtest_loader

        assert load_seeds_and_regions(year)[1] == backtest_loader(year)[1]


class TestBracketOrderHasNoPlaceholders:
    def test_2011_bracket_order_is_all_real_teams(self):
        """The symptom, pinned at the place it actually surfaced."""
        from scripts.mc_pool_backtest import build_bracket_order

        seeds, regions = load_seeds_and_regions(ALIASED_SEASON)
        order = build_bracket_order(seeds, regions)
        placeholders = [t for t in order if t.startswith("unknown_")]
        assert not placeholders, f"2011 bracket has placeholder slots: {sorted(set(placeholders))}"
