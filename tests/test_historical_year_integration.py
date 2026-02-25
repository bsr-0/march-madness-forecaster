import json
import os

import numpy as np
import pytest

from src.pipeline.sota import SOTAPipeline, SOTAPipelineConfig


def test_load_year_samples_is_tombstoned():
    """_load_year_samples() must raise NotImplementedError (tombstoned to prevent
    accidental use of season-end data in training, which causes temporal leakage)."""
    pipeline = SOTAPipeline(SOTAPipelineConfig(year=2026, data_cache_dir="data/raw"))
    with pytest.raises(NotImplementedError, match="season-end"):
        pipeline._load_year_samples("games.json", "metrics.json", feature_dim=75, year=2015)


# ── Backfilled data validation tests ──────────────────────────────────


def test_four_factors_2005_2009_loaded():
    """Verify four factors cache files for 2005-2009 are populated and valid."""
    from src.data.scrapers.torvik import BartTorvikScraper

    expected_keys = {
        "effective_fg_pct", "turnover_rate", "offensive_reb_rate", "free_throw_rate",
        "opp_effective_fg_pct", "opp_turnover_rate", "defensive_reb_rate", "opp_free_throw_rate",
    }

    for year in range(2005, 2010):
        scraper = BartTorvikScraper(cache_dir="data/raw")
        ff = scraper.fetch_four_factors(year)

        assert len(ff) >= 300, f"{year}: only {len(ff)} teams (expected >= 300)"

        for tid, vals in ff.items():
            assert set(vals.keys()) == expected_keys, f"{year}/{tid}: missing keys"

        efg_values = [v["effective_fg_pct"] for v in ff.values()]
        assert all(0.25 <= e <= 0.70 for e in efg_values), (
            f"{year}: eFG% out of [0.25, 0.70] range"
        )

        opp_efg_nonzero = sum(1 for v in ff.values() if v["opp_effective_fg_pct"] > 0)
        assert opp_efg_nonzero / len(ff) >= 0.95, (
            f"{year}: only {opp_efg_nonzero}/{len(ff)} teams have opp_eFG%"
        )

    # 2009 specifically should have opp_turnover_rate populated
    ff_2009 = BartTorvikScraper(cache_dir="data/raw").fetch_four_factors(2009)
    opp_tor_nonzero = sum(1 for v in ff_2009.values() if v["opp_turnover_rate"] > 0)
    assert opp_tor_nonzero / len(ff_2009) >= 0.90, (
        f"2009: only {opp_tor_nonzero}/{len(ff_2009)} have opp_TO% (expected >= 90%)"
    )


def test_shooting_stats_2005_2009_loaded():
    """Verify shooting stats cache files for 2005-2009 are populated and valid."""
    from src.data.scrapers.torvik import BartTorvikScraper

    for year in range(2005, 2010):
        scraper = BartTorvikScraper(cache_dir="data/raw")
        sh = scraper.fetch_shooting_stats(year)

        assert len(sh) >= 300, f"{year}: only {len(sh)} teams (expected >= 300)"

        for tid, vals in sh.items():
            assert "ft_pct" in vals, f"{year}/{tid}: missing ft_pct"
            assert "three_pt_pct" in vals, f"{year}/{tid}: missing three_pt_pct"

        ft_values = [v["ft_pct"] for v in sh.values()]
        valid_ft = sum(1 for f in ft_values if 0.50 <= f <= 0.90)
        assert valid_ft / len(sh) >= 0.95, (
            f"{year}: only {valid_ft}/{len(sh)} teams have FT% in [0.50, 0.90]"
        )


def test_tournament_seeds_2005_2006_loaded():
    """Verify tournament seed files for 2005 and 2006 exist and are valid."""
    base_dir = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "historical")

    expected_one_seeds = {
        2005: {"UNC", "Duke", "Illinois", "Washington"},
        2006: {"UConn", "Duke", "Villanova", "Memphis"},
    }

    for year in [2005, 2006]:
        path = os.path.abspath(os.path.join(base_dir, f"tournament_seeds_{year}.json"))
        assert os.path.isfile(path), f"Missing tournament_seeds_{year}.json"

        with open(path, "r") as f:
            payload = json.load(f)

        assert payload["season"] == year
        teams = payload["teams"]
        assert len(teams) == 65, f"{year}: {len(teams)} teams (expected 65)"

        regions = {t["region"] for t in teams}
        assert regions == {"East", "South", "Midwest", "West"}, f"{year}: unexpected regions {regions}"

        for t in teams:
            assert 1 <= t["seed"] <= 16, f"{year}/{t['team_name']}: seed {t['seed']} out of range"
            assert t.get("team_id"), f"{year}/{t['team_name']}: missing team_id"

        one_seeds = {t["team_name"] for t in teams if t["seed"] == 1}
        assert one_seeds == expected_one_seeds[year], (
            f"{year}: 1-seeds {one_seeds} != expected {expected_one_seeds[year]}"
        )
