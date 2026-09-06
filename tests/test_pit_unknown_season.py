"""An unknown season must fail the PIT fold, not pass it.

Before this, PITValidator.validate_fold() responded to a season with no
Selection Sunday on record by appending a warning and returning a result with
passed=True -- skipping every Tier 2/3 temporal check below it. The fold that
enforced nothing was indistinguishable from the fold that enforced everything,
which is the worst possible outcome for a leakage guard.
"""

from __future__ import annotations

import pytest

from src.pipeline.stages.pit_validation import (
    PITValidator,
    PITViolationError,
    SELECTION_SUNDAY_DATES,
)

UNKNOWN_SEASON = 2099


@pytest.fixture
def validator():
    return PITValidator()


class TestUnknownSeasonIsLoud:
    def test_strict_mode_raises(self, validator):
        with pytest.raises(PITViolationError) as exc:
            validator.validate_fold(
                year=UNKNOWN_SEASON,
                feature_names=["adj_off_eff"],
                feature_metadata={"adj_off_eff": {"latest_game_date": "2099-03-10"}},
                strict=True,
            )
        assert str(UNKNOWN_SEASON) in str(exc.value)

    def test_non_strict_mode_reports_failure_rather_than_passing(self, validator):
        result = validator.validate_fold(
            year=UNKNOWN_SEASON,
            feature_names=["adj_off_eff"],
            feature_metadata={"adj_off_eff": {"latest_game_date": "2099-03-10"}},
            strict=False,
        )
        assert result.passed is False
        assert result.violations, "Unknown season must record a violation"

    def test_cancelled_2020_season_also_fails(self, validator):
        assert 2020 not in SELECTION_SUNDAY_DATES
        result = validator.validate_fold(
            year=2020,
            feature_names=["adj_off_eff"],
            feature_metadata={},
            strict=False,
        )
        assert result.passed is False


class TestKnownSeasonStillWorks:
    def test_clean_fold_passes(self, validator):
        result = validator.validate_fold(
            year=2025,
            feature_names=["adj_off_eff"],
            feature_metadata={"adj_off_eff": {"latest_game_date": "2025-03-10"}},
            strict=True,
        )
        assert result.passed is True

    def test_post_selection_sunday_data_still_flagged(self, validator):
        """The check that was being skipped must still fire for known seasons."""
        tier2 = validator.get_tier2_features()
        if not tier2:
            pytest.skip("MANIFEST.yaml unavailable; no Tier 2 features to check")

        feature = tier2[0]
        result = validator.validate_fold(
            year=2025,
            feature_names=[feature],
            # 2025 Selection Sunday is Mar 16; this is a week of tournament games.
            feature_metadata={feature: {"latest_game_date": "2025-03-23"}},
            strict=False,
        )
        assert result.passed is False
        assert any("Tier 2" in v for v in result.violations)
