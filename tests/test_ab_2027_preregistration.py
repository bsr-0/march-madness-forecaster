"""The 2027 A/B pre-registration must stay pre-registered (recommendation 16).

A pre-registration protects against exactly one thing: deciding what counts as
evidence after seeing the evidence. Every test here guards a way that could
happen — the protocol being edited once data exists, a season being re-recorded
until it reads better, the inconvenient metric being dropped, or a single
season being reported as though it settled something that needs 181.

The power result is itself pinned. "181 seasons at 80% power" is the finding
that made this a ledger rather than a test, and if a future edit makes the A/B
look feasible, that should fail loudly rather than quietly licensing a
conclusion.
"""

from __future__ import annotations

import json

import pytest

from src.governance.ab_2027 import (
    ALPHA,
    ARM_CONTROL,
    ARM_TREATMENT,
    BARRED_CONTROLS,
    FIRST_SEASON,
    HISTORICAL_ESTIMATE,
    METRIC_DIRECTION,
    PRIMARY_METRICS,
    PreregistrationViolation,
    freeze,
    record_season,
    spec,
    spec_hash,
    stopping_rule,
    tally,
    verify,
)


# ---------------------------------------------------------------------------
# The arms
# ---------------------------------------------------------------------------


def test_the_control_arm_is_not_the_circular_one():
    """The trap recommendation 16 explicitly set.

    `fixed_blendA100_r35` tops the P(1st) table, so it is the obvious control —
    and it is built from pure seed probabilities while the referee IS the seed
    model, so it would be graded by its own source. Picking it would measure
    finding C2's circularity and report it as a strategy comparison.
    """
    assert ARM_CONTROL != "fixed_blendA100_r35"
    assert "fixed_blendA100_r35" in BARRED_CONTROLS
    assert "seed" in BARRED_CONTROLS["fixed_blendA100_r35"], "the bar must record WHY"


def test_both_arms_are_real_modes():
    from scripts.mc_pool_backtest import ALL_MODES

    assert ARM_TREATMENT in ALL_MODES
    assert ARM_CONTROL in ALL_MODES
    for barred in BARRED_CONTROLS:
        assert barred in ALL_MODES, f"{barred} is barred but no longer exists; the bar is stale"


def test_both_metrics_are_primary_and_they_disagree():
    """Declaring both in advance is the point.

    On the historical window the treatment is ahead on one metric and behind on
    the other. If only one had been pre-declared, the choice of which would
    have decided the answer.
    """
    assert set(PRIMARY_METRICS) == {"p_first", "mean_rank"}

    p1 = HISTORICAL_ESTIMATE["p_first"]["paired_diff_raw"] * METRIC_DIRECTION["p_first"]
    mr = HISTORICAL_ESTIMATE["mean_rank"]["paired_diff_raw"] * METRIC_DIRECTION["mean_rank"]
    assert p1 > 0 > mr, (
        "the two primary metrics no longer disagree in sign. That is a substantive change to "
        "finding H11 and should be re-measured, not absorbed."
    )


# ---------------------------------------------------------------------------
# The power result
# ---------------------------------------------------------------------------


def test_the_ab_is_not_resolvable_on_the_decision_relevant_metric():
    """181 seasons. Pinned, because it is the reason this is a ledger."""
    needed = HISTORICAL_ESTIMATE["p_first"]["seasons_for_80pct_power"]
    assert needed > 100, (
        f"p_first now claims to need only {needed} seasons. If the effect size genuinely "
        "changed this is a real finding; if a definition changed, the pre-registration is "
        "measuring something else than it was frozen on."
    )
    assert FIRST_SEASON + needed - 1 > 2200, "sanity: resolution is beyond any planning horizon"


def test_the_power_artifact_agrees_with_the_frozen_estimate():
    """The frozen numbers must match what the script actually measured."""
    from pathlib import Path

    path = Path("artifacts/headline_measurement/ab_2027_power.json")
    if not path.exists():  # pragma: no cover - artifact is committed, but don't hard-fail a fresh clone
        pytest.skip("power artifact not present")
    with open(path) as f:
        measured = json.load(f)

    assert measured["arm_treatment"] == ARM_TREATMENT
    assert measured["arm_control"] == ARM_CONTROL
    assert measured["verdict"] == "not_resolvable_prospectively"
    for metric in PRIMARY_METRICS:
        frozen = HISTORICAL_ESTIMATE[metric]
        live = measured["per_metric"][metric]
        assert live["seasons_for_80pct_power"] == frozen["seasons_for_80pct_power"], (
            f"{metric}: frozen estimate says {frozen['seasons_for_80pct_power']} seasons, the "
            f"measurement says {live['seasons_for_80pct_power']}"
        )
        assert live["treatment_mean"] == pytest.approx(frozen["treatment_mean"], abs=1e-3)
        assert live["control_mean"] == pytest.approx(frozen["control_mean"], abs=1e-3)


# ---------------------------------------------------------------------------
# The protocol cannot drift
# ---------------------------------------------------------------------------


def test_the_frozen_protocol_matches_the_live_one():
    result = verify()
    assert result["frozen"], "configs/frozen/prospective_2027_ab.json is missing; run --freeze"
    assert result["matches"], (
        f"the protocol has drifted from what was frozen: {result['drifted_fields']}. A "
        "pre-registration is not editable in place -- bump SPEC_VERSION."
    )


def test_freeze_refuses_to_overwrite_a_different_protocol(tmp_path, monkeypatch):
    """Silent editing is the whole failure mode."""
    import src.governance.ab_2027 as mod

    path = tmp_path / "frozen.json"
    freeze(path)

    monkeypatch.setattr(mod, "ARM_CONTROL", "some_other_mode")
    with pytest.raises(PreregistrationViolation, match="not editable in place"):
        freeze(path)


def test_spec_hash_changes_when_the_protocol_changes(monkeypatch):
    import src.governance.ab_2027 as mod

    before = spec_hash()
    monkeypatch.setattr(mod, "ALPHA", 0.10)
    assert spec_hash() != before, "the hash must cover the significance level"


def test_stopping_rule_forbids_interim_looks():
    rule = stopping_rule()
    assert rule["no_interim_analysis"] is True
    assert rule["no_early_stopping"] is True
    assert rule["analyse_at_seasons"] == [21, 181]
    assert rule["if_effect_absent_at_21"], "the null outcome must be pre-specified too"


def test_the_spec_records_that_it_does_not_expect_to_conclude():
    text = spec()["honest_expectation"]
    assert "will not conclude" in text


# ---------------------------------------------------------------------------
# The ledger
# ---------------------------------------------------------------------------


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    import src.governance.ab_2027 as mod

    monkeypatch.setattr(mod, "LEDGER_PATH", tmp_path / "ledger.json")
    return mod


def _obs(p_first, mean_rank):
    return {"p_first": p_first, "mean_rank": mean_rank}


def test_recording_a_season_round_trips(ledger):
    entry = record_season(2027, _obs(0.13, 9.5), _obs(0.11, 9.9))
    assert entry["year"] == 2027
    assert entry["spec_hash"] == spec_hash()

    summary = tally()
    assert summary["n_seasons"] == 1
    assert summary["per_metric"]["p_first"]["treatment_better_seasons"] == 1
    assert summary["per_metric"]["mean_rank"]["treatment_better_seasons"] == 1


def test_a_season_cannot_be_recorded_twice(ledger):
    record_season(2027, _obs(0.13, 9.5), _obs(0.11, 9.9))
    with pytest.raises(PreregistrationViolation, match="already recorded"):
        record_season(2027, _obs(0.20, 8.0), _obs(0.05, 12.0))


def test_retrospective_seasons_cannot_be_added(ledger):
    """Back-filling the prospective series makes it retrospective."""
    with pytest.raises(PreregistrationViolation, match="predates this pre-registration"):
        record_season(2025, _obs(0.21, 10.0), _obs(0.18, 10.1))


def test_both_metrics_must_be_reported(ledger):
    """Reporting one metric is reporting the convenient half."""
    with pytest.raises(PreregistrationViolation, match="missing"):
        record_season(2027, {"p_first": 0.13}, {"p_first": 0.11})


def test_tally_is_not_a_test(ledger):
    """No p-value before a scheduled analysis point, and it says so."""
    for i, year in enumerate((2027, 2028, 2029)):
        record_season(year, _obs(0.13 + i * 0.01, 9.5), _obs(0.11, 9.9))

    summary = tally()
    assert summary["is_a_test"] is False
    assert summary["next_scheduled_analysis"] == 21
    assert "No significance claim" in summary["reminder"]
    blob = json.dumps(summary)
    assert "p_value" not in blob, "the running tally must not surface a p-value"


def test_tally_counts_both_directions(ledger):
    record_season(2027, _obs(0.13, 9.5), _obs(0.11, 9.9))  # treatment better on both
    record_season(2028, _obs(0.09, 11.0), _obs(0.12, 9.0))  # control better on both

    per = tally()["per_metric"]
    for metric in PRIMARY_METRICS:
        assert per[metric]["treatment_better_seasons"] == 1
        assert per[metric]["control_better_seasons"] == 1


def test_empty_ledger_reports_zero_not_an_error(ledger):
    summary = tally()
    assert summary["n_seasons"] == 0
    assert summary["per_metric"]["p_first"]["mean_signed_diff"] is None


def test_2027_is_still_the_sequestered_holdout():
    """The A/B's first season must be the untouched one, or it is not prospective."""
    from src.governance.pool_rdof_audit import SEQUESTERED_YEARS

    assert FIRST_SEASON in SEQUESTERED_YEARS


def test_alpha_is_conventional():
    assert ALPHA == 0.05
