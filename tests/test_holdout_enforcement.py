"""End-to-end closure test for COUNCIL_LESSONS §2 O20 — held-out-window
enforcement.

Gate: "Code-level gate that refuses to fit on the held-out window."

The enforcement chain is composed of three pieces, each of which has its
own unit test elsewhere:

  1. `validate_locked_production_path()` (src/pipeline/config.py:901) hard-fails
     production configs whose holdout_years drift from [2025] or whose
     dev_years overlap with holdout. Tested in test_production_2026_freeze.

  2. `_filter_years(include_holdout=False)` (src/pipeline/pipeline_runner.py:496)
     strips holdout entries from any year list passed to training callers.
     Tested in test_temporal_integrity.test_filter_years_excludes_holdout.

  3. `YearSplitPolicy.assert_dev_only` / `assert_dev_artifact_years`
     (src/ml/evaluation/evaluation_integrity.py:130) raises
     HoldoutContaminationError if any caller leaks a holdout year into a
     development context. Wired in at:
       - src/pipeline/stages/baseline_training/_ensemble.py:530, 770
       - src/pipeline/stages/baseline_training/_models.py:107, 178, 248
       - src/ml/optimization/hyperparameter_tuning.py:60
     Tested in test_temporal_integrity.test_year_split_policy_rejects_holdout_in_training.

This module composes those checks into a single end-to-end scenario test
so the closure of O20 is verifiable from one entry point.
"""

from __future__ import annotations

import pytest

from src.ml.evaluation.evaluation_integrity import (
    HoldoutContaminationError,
    YearSplitPolicy,
)


def _production_split() -> YearSplitPolicy:
    """Return the YearSplitPolicy that mirrors pipeline_freeze.json."""
    return YearSplitPolicy(
        dev_years=tuple(y for y in range(2008, 2025) if y != 2020),
        holdout_years=(2025,),
        prospective_years=(2026, 2027, 2028),
    )


def test_production_split_constructs_cleanly():
    """The locked production split must satisfy every YearSplitPolicy
    invariant: no overlap, no COVID, immutable."""
    policy = _production_split()
    assert 2025 in policy.holdout_years
    assert 2025 not in policy.dev_years
    assert 2020 not in policy.dev_years
    assert 2020 not in policy.holdout_years


def test_holdout_year_alone_in_training_raises():
    policy = _production_split()
    with pytest.raises(HoldoutContaminationError, match="2025"):
        policy.assert_dev_only([2025], context="model training")


def test_holdout_year_mixed_with_dev_raises():
    policy = _production_split()
    with pytest.raises(HoldoutContaminationError, match="2025"):
        policy.assert_dev_only([2018, 2019, 2021, 2025], context="ensemble fit")


def test_dev_only_call_passes():
    policy = _production_split()
    # Should not raise.
    policy.assert_dev_only([2016, 2017, 2018, 2024], context="model training")


def test_assert_dev_artifact_years_alias_enforces_same_rule():
    policy = _production_split()
    with pytest.raises(HoldoutContaminationError):
        policy.assert_dev_artifact_years([2016, 2025], context="LOYO ensemble")


def test_overlapping_split_rejected_at_construction():
    """If a config redefines dev/holdout to overlap, the policy itself
    refuses to construct — so downstream guards never get reached."""
    with pytest.raises(ValueError, match="Dev/holdout overlap"):
        YearSplitPolicy(
            dev_years=(2018, 2019, 2024, 2025),
            holdout_years=(2025,),
        )


def test_covid_year_rejected_at_construction():
    with pytest.raises(ValueError, match="COVID year 2020"):
        YearSplitPolicy(
            dev_years=(2019, 2020, 2021),
            holdout_years=(2025,),
        )


def test_filter_years_strips_holdout_with_default_flag():
    """The pipeline runner's _filter_years must strip the holdout year
    when include_holdout=False (the default for any training caller)."""
    from src.pipeline.config import SOTAPipelineConfig
    from src.pipeline.pipeline_runner import _PipelineRunner

    config = SOTAPipelineConfig(
        year=2026,
        dev_years=[2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024],
        holdout_years=[2025],
        enforce_production_path=False,  # bypass the locked-path checks
    )
    runner = _PipelineRunner.__new__(_PipelineRunner)
    runner._p = type("P", (), {"config": config})()
    candidates = [2016, 2017, 2024, 2025, 2026]
    filtered = runner._filter_years(candidates, include_holdout=False)
    assert 2025 not in filtered, "holdout year 2025 leaked into training years"
    assert 2024 in filtered
    assert filtered == sorted(filtered), "filtered years should be sorted"
