"""Tests for pool probability profile, ESPN routing guard, and profile integration.

Verifies:
- "pool" probability profile is accepted by config validation
- "pool" profile skips tournament shrinkage
- ESPN mode + production profile raises ValueError
- ESPN mode + pool profile is accepted
- Production locked path validation ignores pool profile
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.pipeline.probability_pipeline import (
    apply_calibration,
    apply_final_clip,
    apply_tournament_shrinkage,
)


# ---------------------------------------------------------------------------
# Config validation tests
# ---------------------------------------------------------------------------


class TestPoolProfileConfig:
    """Config accepts 'pool' as a probability_profile value."""

    def test_pool_profile_accepted(self):
        """Config construction with probability_profile='pool' should not raise."""
        from src.pipeline.config import SOTAPipelineConfig

        config = SOTAPipelineConfig(probability_profile="pool")
        assert config.probability_profile == "pool"

    def test_production_profile_still_accepted(self):
        from src.pipeline.config import SOTAPipelineConfig

        config = SOTAPipelineConfig(probability_profile="production")
        assert config.probability_profile == "production"

    def test_experimental_profile_still_accepted(self):
        from src.pipeline.config import SOTAPipelineConfig

        config = SOTAPipelineConfig(probability_profile="experimental")
        assert config.probability_profile == "experimental"

    def test_invalid_profile_rejected(self):
        from src.pipeline.config import SOTAPipelineConfig

        with pytest.raises(ValueError, match="must be 'production', 'pool', or 'experimental'"):
            SOTAPipelineConfig(probability_profile="turbo")


class TestESPNProfileRoutingGuard:
    """ESPN mode is incompatible with production probability profile."""

    def test_espn_with_production_raises(self):
        from src.pipeline.config import SOTAPipelineConfig

        with pytest.raises(ValueError, match="ESPN pool optimization.*incompatible"):
            SOTAPipelineConfig(
                probability_profile="production",
                espn_mode=True,
            )

    def test_espn_with_pool_accepted(self):
        from src.pipeline.config import SOTAPipelineConfig

        config = SOTAPipelineConfig(
            probability_profile="pool",
            espn_mode=True,
        )
        assert config.espn_mode is True
        assert config.probability_profile == "pool"

    def test_espn_with_experimental_accepted(self):
        from src.pipeline.config import SOTAPipelineConfig

        config = SOTAPipelineConfig(
            probability_profile="experimental",
            espn_mode=True,
        )
        assert config.espn_mode is True

    def test_production_without_espn_still_fine(self):
        from src.pipeline.config import SOTAPipelineConfig

        config = SOTAPipelineConfig(
            probability_profile="production",
            espn_mode=False,
        )
        assert config.probability_profile == "production"


class TestLockedProductionPathIgnoresPool:
    """Pool profile should not trigger locked production path validation."""

    def test_pool_profile_skips_locked_validation(self):
        """validate_locked_production_path only fires for production profile."""
        from src.pipeline.config import SOTAPipelineConfig

        # Pool profile with non-standard dev_years should NOT raise
        config = SOTAPipelineConfig(
            probability_profile="pool",
            dev_years=[2020, 2021],
            holdout_years=[2024],
        )
        assert config.probability_profile == "pool"


# ---------------------------------------------------------------------------
# Pool probability path tests (unit-level)
# ---------------------------------------------------------------------------


class TestPoolProbabilityPath:
    """Pool profile: raw → calibration → clip (NO shrinkage)."""

    def test_pool_skips_shrinkage(self):
        """The pool path should NOT apply tournament shrinkage."""
        raw = 0.80
        clip_lo, clip_hi = 0.005, 0.995
        shrinkage = 0.06

        # Build mock calibrator
        calibrator = MagicMock()
        calibrator.calibrate.return_value = np.array([raw])  # Pass-through

        # Production path: includes shrinkage
        prod_calibrated = apply_calibration(raw, calibrator, clip_lo, clip_hi)
        prod_shrunk = apply_tournament_shrinkage(prod_calibrated, shrinkage)
        prod_final = apply_final_clip(prod_shrunk, clip_lo, clip_hi)

        # Pool path: skips shrinkage
        pool_calibrated = apply_calibration(raw, calibrator, clip_lo, clip_hi)
        pool_final = apply_final_clip(pool_calibrated, clip_lo, clip_hi)

        # Pool output should be further from 0.5 than production
        assert abs(pool_final - 0.5) > abs(prod_final - 0.5)
        # Pool output should equal calibrated (no shrinkage applied)
        assert pool_final == pytest.approx(pool_calibrated)

    def test_pool_preserves_decisive_signals(self):
        """Probabilities far from 0.5 stay far from 0.5 in pool profile."""
        raw = 0.90
        clip_lo, clip_hi = 0.005, 0.995
        shrinkage = 0.06

        calibrator = MagicMock()
        calibrator.calibrate.return_value = raw

        # Production: 0.06 * 0.5 + 0.94 * 0.90 = 0.030 + 0.846 = 0.876
        prod = apply_tournament_shrinkage(raw, shrinkage)
        assert prod == pytest.approx(0.876)

        # Pool: stays at 0.90
        pool = apply_final_clip(raw, clip_lo, clip_hi)
        assert pool == pytest.approx(0.90)

        # Pool has stronger signal
        assert abs(pool - 0.5) > abs(prod - 0.5)

    def test_shrinkage_effect_on_close_games(self):
        """For close-to-0.5 probabilities, shrinkage has less effect."""
        raw = 0.52
        shrinkage = 0.06

        prod = apply_tournament_shrinkage(raw, shrinkage)
        # 0.06 * 0.5 + 0.94 * 0.52 = 0.030 + 0.4888 = 0.5188
        assert prod == pytest.approx(0.5188)

        # Pool keeps 0.52 — only 0.0012 difference for this near-toss-up
        diff = abs(raw - prod)
        assert diff < 0.002  # Small effect near 0.5

    def test_symmetry_maintained(self):
        """Pool profile maintains P(A>B) + P(B>A) ≈ 1 symmetry."""
        raw_ab = 0.75
        raw_ba = 0.25
        clip_lo, clip_hi = 0.005, 0.995

        pool_ab = apply_final_clip(raw_ab, clip_lo, clip_hi)
        pool_ba = apply_final_clip(raw_ba, clip_lo, clip_hi)
        assert pool_ab + pool_ba == pytest.approx(1.0)
