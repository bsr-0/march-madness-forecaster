"""Tests for extended historical data integration.

Validates:
- TOURNAMENT_START_DATES covers 1996-2026
- DATA_QUALITY_ERA_WEIGHTS covers historical years
- CANONICAL_DEV_YEARS includes 2003-2024 (minus 2020)
- RULE_REGIME_BREAKS are defined correctly
- Era-aware feature masking produces correct subsets
- No overlap between dev/holdout/prospective year sets
- Production config is NOT modified (locked)
- Extended ingestion config has correct defaults
"""

import json
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from src.pipeline.config import (
    DATA_QUALITY_ERA_WEIGHTS,
    RULE_REGIME_BREAKS,
    TOURNAMENT_START_DATES,
)
from src.ml.evaluation.evaluation_integrity import (
    CANONICAL_DEV_YEARS,
    CANONICAL_HOLDOUT_YEARS,
    PROSPECTIVE_CANDIDATE_YEARS,
    YearSplitPolicy,
)
from src.data.features.feature_engineering import (
    TEAM_FEATURE_DIM,
    TeamFeatures,
    era_available_features,
    era_exclude_mask,
)


class TestTournamentStartDates:
    """Verify TOURNAMENT_START_DATES covers the extended range."""

    def test_covers_1996_to_2026(self):
        """All years 1996-2026 (except 2020) should have entries."""
        for year in range(1996, 2027):
            if year == 2020:
                continue
            assert year in TOURNAMENT_START_DATES, (
                f"TOURNAMENT_START_DATES missing year {year}"
            )

    def test_dates_are_in_march(self):
        """Tournament start dates should be in March."""
        for year, d in TOURNAMENT_START_DATES.items():
            assert d.month == 3, f"Year {year} start date not in March: {d}"

    def test_dates_are_before_april(self):
        """Tournament starts before April."""
        for year, d in TOURNAMENT_START_DATES.items():
            assert d < date(year, 4, 1), f"Year {year} start date too late: {d}"

    def test_dates_are_correct_year(self):
        """Each date must be in its keyed year."""
        for year, d in TOURNAMENT_START_DATES.items():
            assert d.year == year, f"Year {year} date is in wrong year: {d}"

    def test_covid_year_excluded(self):
        """2020 should not be in the tournament dates (COVID)."""
        assert 2020 not in TOURNAMENT_START_DATES

    def test_original_dates_unchanged(self):
        """Verify that existing 2008-2026 dates were not accidentally modified."""
        expected_originals = {
            2008: date(2008, 3, 18),
            2016: date(2016, 3, 15),
            2019: date(2019, 3, 19),
            2025: date(2025, 3, 18),
            2026: date(2026, 3, 17),
        }
        for year, expected_date in expected_originals.items():
            assert TOURNAMENT_START_DATES[year] == expected_date, (
                f"Original date for {year} was modified!"
            )


class TestDataQualityEraWeights:
    """Verify era weights cover the extended range."""

    def test_covers_1996_onwards(self):
        """Pre-digital years should have zero or near-zero weights."""
        for year in range(1996, 2000):
            assert year in DATA_QUALITY_ERA_WEIGHTS, f"Missing era weight for {year}"
            assert DATA_QUALITY_ERA_WEIGHTS[year] == 0.0, (
                f"Pre-digital year {year} should have weight 0.0"
            )

    def test_early_digital_low_weights(self):
        """Early digital era (2000-2004) should have zero weights."""
        for year in range(2000, 2005):
            assert year in DATA_QUALITY_ERA_WEIGHTS, f"Missing era weight for {year}"
            assert DATA_QUALITY_ERA_WEIGHTS[year] == 0.0, (
                f"Year {year} weight should be 0.0: {DATA_QUALITY_ERA_WEIGHTS[year]}"
            )

    def test_modern_era_higher_weights(self):
        """Modern era (2010+) should have significantly higher weights."""
        for year in range(2010, 2015):
            assert year in DATA_QUALITY_ERA_WEIGHTS
            assert DATA_QUALITY_ERA_WEIGHTS[year] >= 0.5, (
                f"Modern year {year} weight too low: {DATA_QUALITY_ERA_WEIGHTS[year]}"
            )

    def test_weights_monotonically_non_decreasing(self):
        """Era weights should generally increase over time."""
        sorted_years = sorted(DATA_QUALITY_ERA_WEIGHTS.keys())
        for i in range(1, len(sorted_years)):
            curr = DATA_QUALITY_ERA_WEIGHTS[sorted_years[i]]
            prev = DATA_QUALITY_ERA_WEIGHTS[sorted_years[i - 1]]
            assert curr >= prev, (
                f"Weight decreased from {sorted_years[i-1]} ({prev}) to "
                f"{sorted_years[i]} ({curr})"
            )

    def test_original_weights_unchanged(self):
        """Verify that existing 2005-2014 weights were not modified."""
        original_weights = {
            2008: 0.20, 2009: 0.30, 2010: 0.55, 2014: 0.85,
        }
        for year, expected in original_weights.items():
            assert DATA_QUALITY_ERA_WEIGHTS[year] == expected, (
                f"Original weight for {year} was modified!"
            )


class TestRuleRegimeBreaks:
    """Verify NCAA rule regime breaks are defined."""

    def test_shot_clock_change(self):
        assert 2016 in RULE_REGIME_BREAKS
        assert "shot_clock" in RULE_REGIME_BREAKS[2016]

    def test_three_point_line(self):
        assert 2020 in RULE_REGIME_BREAKS
        assert "3pt" in RULE_REGIME_BREAKS[2020]

    def test_tournament_expansion(self):
        assert 2011 in RULE_REGIME_BREAKS
        assert "tournament" in RULE_REGIME_BREAKS[2011]


class TestCanonicalDevYearsExpanded:
    """Verify that CANONICAL_DEV_YEARS includes the extended range."""

    def test_includes_2003_to_2007(self):
        """Extended dev years should include 2003-2007."""
        for year in range(2003, 2008):
            assert year in CANONICAL_DEV_YEARS, f"Year {year} missing from CANONICAL_DEV_YEARS"

    def test_includes_original_years(self):
        """Original 2008-2024 years should still be present."""
        for year in range(2008, 2020):
            assert year in CANONICAL_DEV_YEARS
        for year in range(2021, 2025):
            assert year in CANONICAL_DEV_YEARS

    def test_excludes_covid_year(self):
        assert 2020 not in CANONICAL_DEV_YEARS

    def test_no_overlap_with_holdout(self):
        assert set(CANONICAL_DEV_YEARS).isdisjoint(set(CANONICAL_HOLDOUT_YEARS))

    def test_no_overlap_with_prospective(self):
        assert set(CANONICAL_DEV_YEARS).isdisjoint(set(PROSPECTIVE_CANDIDATE_YEARS))

    def test_year_split_policy_accepts_extended_years(self):
        """YearSplitPolicy should accept the expanded CANONICAL_DEV_YEARS."""
        policy = YearSplitPolicy()
        # Should not raise — 2003-2007 are valid dev years now
        policy.assert_dev_only([2003, 2004, 2005, 2006, 2007], context="extended training")


class TestEraAvailableFeatures:
    """Verify era-aware feature masking works correctly."""

    def test_modern_year_has_all_features(self):
        """A modern year (2020+) should have all 79 features."""
        features = era_available_features(2024)
        assert len(features) == TEAM_FEATURE_DIM

    def test_old_year_has_fewer_features(self):
        """A pre-2008 year should have fewer features (no Torvik, etc)."""
        features = era_available_features(2005)
        assert len(features) < TEAM_FEATURE_DIM
        # Core basketball features should still be available
        assert 'win_pct' in features
        assert 'elo_rating' in features
        assert 'seed_strength' in features

    def test_torvik_features_excluded_pre_2008(self):
        """Torvik features should not appear for years before 2008."""
        features = era_available_features(2005)
        assert 'adj_off_eff' not in features
        assert 'adj_def_eff' not in features
        assert 'adj_tempo' not in features

    def test_torvik_features_included_2008_plus(self):
        """Torvik features should appear for 2008+."""
        features = era_available_features(2008)
        assert 'adj_off_eff' in features
        assert 'adj_def_eff' in features

    def test_exclude_mask_matches_available(self):
        """exclude mask should be inverse of available features."""
        year = 2005
        available = set(era_available_features(year))
        mask = era_exclude_mask(year)
        all_names = TeamFeatures.get_feature_names(include_embeddings=False)

        for i, name in enumerate(all_names):
            if name in available:
                assert not mask[i], f"Feature {name} is available but mask says exclude"
            else:
                assert mask[i], f"Feature {name} is unavailable but mask says include"

    def test_mask_length_matches_feature_dim(self):
        """Exclude mask length must equal TEAM_FEATURE_DIM."""
        mask = era_exclude_mask(2005)
        assert len(mask) == TEAM_FEATURE_DIM

    def test_modern_year_mask_all_false(self):
        """Modern year should have no exclusions."""
        mask = era_exclude_mask(2024)
        assert not any(mask)


class TestProductionConfigUnmodified:
    """Verify production config was NOT modified."""

    def test_production_training_years_locked(self):
        """Production training years must remain [2016-2019, 2021-2024]."""
        config_path = Path("configs/production_2026.json")
        if not config_path.exists():
            pytest.skip("production_2026.json not found")

        with open(config_path) as f:
            config = json.load(f)

        expected_training = [2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024]
        assert config["training_years"] == expected_training
        assert config["holdout_years"] == [2025]
        assert config["strict_leakage_mode"] is True

    def test_production_validator_unchanged(self):
        """Production validator expected years should not have changed."""
        from src.governance.production_validator import (
            EXPECTED_TRAINING_YEARS,
            EXPECTED_HOLDOUT_YEARS,
        )
        assert EXPECTED_TRAINING_YEARS == [2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024]
        assert EXPECTED_HOLDOUT_YEARS == [2025]


class TestExtendedIngestionConfig:
    """Verify the extended ingestion config defaults."""

    def test_default_start_season(self):
        from src.data.ingestion.extended_historical_ingest import ExtendedIngestionConfig
        config = ExtendedIngestionConfig()
        assert config.start_season == 2003

    def test_default_end_season(self):
        from src.data.ingestion.extended_historical_ingest import ExtendedIngestionConfig
        config = ExtendedIngestionConfig()
        assert config.end_season == 2025

    def test_skip_existing_default_true(self):
        from src.data.ingestion.extended_historical_ingest import ExtendedIngestionConfig
        config = ExtendedIngestionConfig()
        assert config.skip_existing is True


class TestHistoricalIngestionConfigDefaults:
    """Verify that HistoricalIngestionConfig defaults were updated."""

    def test_default_start_season_is_2003(self):
        from src.data.ingestion.historical_pipeline import HistoricalIngestionConfig
        config = HistoricalIngestionConfig()
        assert config.start_season == 2003


class TestRuntimeStateConfigImmutability:
    """Verify that _runtime_state pattern prevents config mutation.

    The production runner hashes config before and after pipeline.run().
    If any code mutates self.config during execution, the production
    run fails with a hash mismatch.  The _runtime_state dict on
    SOTAPipeline holds mutable overrides (MC calibration, budget
    degradation, path auto-resolution) so self.config stays frozen.
    """

    def test_pipeline_has_runtime_state(self):
        """SOTAPipeline initializes _runtime_state from config."""
        from src.pipeline.config import SOTAPipelineConfig
        from src.pipeline.sota import SOTAPipeline

        config = SOTAPipelineConfig()
        pipeline = SOTAPipeline(config)
        assert hasattr(pipeline, "_runtime_state")
        assert pipeline._runtime_state["mc_noise_std"] == config.mc_noise_std
        assert pipeline._runtime_state["num_simulations"] == config.num_simulations

    def test_mc_calibration_does_not_mutate_config(self):
        """_load_mc_calibration writes to _runtime_state, not config."""
        import json
        import tempfile
        import os
        from src.pipeline.config import SOTAPipelineConfig
        from src.pipeline.sota import SOTAPipeline

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"best_params": {"noise_std": 0.12, "regional_correlation": 0.05}}, f)
            mc_path = f.name

        try:
            config = SOTAPipelineConfig(mc_calibration_json=mc_path)
            original_noise = config.mc_noise_std
            original_corr = config.mc_regional_correlation
            pipeline = SOTAPipeline(config)

            pipeline._load_mc_calibration()

            # Config must be unchanged
            assert config.mc_noise_std == original_noise
            assert config.mc_regional_correlation == original_corr
            # Runtime state must have the calibrated values
            assert pipeline._runtime_state["mc_noise_std"] == 0.12
            assert pipeline._runtime_state["mc_regional_correlation"] == 0.05
        finally:
            os.unlink(mc_path)

    def test_budget_degradation_does_not_mutate_config(self):
        """Budget degradation writes to _runtime_state, not config."""
        from src.pipeline.config import SOTAPipelineConfig
        from src.pipeline.sota import SOTAPipeline

        config = SOTAPipelineConfig(
            num_simulations=10000,
            enable_gnn=False,
            enable_transformer=False,
        )
        pipeline = SOTAPipeline(config)

        # Simulate budget degradation logic
        pipeline._runtime_state["num_simulations"] = max(
            1000, int(pipeline._runtime_state["num_simulations"]) // 2
        )
        pipeline._runtime_state["enable_gnn"] = False
        pipeline._runtime_state["enable_transformer"] = False

        # Config must be unchanged
        assert config.num_simulations == 10000
        assert pipeline._runtime_state["num_simulations"] == 5000

    def test_config_hash_stable_after_runtime_state_changes(self):
        """Config hash remains identical after _runtime_state mutations."""
        import hashlib
        import dataclasses
        from src.pipeline.config import SOTAPipelineConfig
        from src.pipeline.sota import SOTAPipeline

        config = SOTAPipelineConfig()
        pipeline = SOTAPipeline(config)

        def _hash_config(c):
            h = hashlib.sha256()
            for f in dataclasses.fields(c):
                h.update(f"{f.name}={getattr(c, f.name)!r}".encode("utf-8"))
            return h.hexdigest()

        hash_before = _hash_config(config)

        # Simulate all runtime state changes
        pipeline._runtime_state["mc_noise_std"] = 0.12
        pipeline._runtime_state["num_simulations"] = 25000
        pipeline._runtime_state["multi_year_games_dir"] = "/tmp/resolved"

        hash_after = _hash_config(config)
        assert hash_before == hash_after, "Config hash must not change when _runtime_state is mutated"
