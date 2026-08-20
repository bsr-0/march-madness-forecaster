"""Regression tests for the noseed train/serve feature skew (FINDINGS.md 6c).

The defect: ``mc_pool_backtest._load_team_stats`` returned the four-factors
sub-dict (8 keys) to a model whose feature vector needs 12. The other four --
``adj_offensive_efficiency``, ``adj_defensive_efficiency``, ``adj_tempo`` and
``barthag`` -- fell through to their per-key defaults on *both* sides of every
differential, so those dimensions were identically 0.0 for every matchup.

It was invisible because a zero differential is a perfectly plausible feature
value ("these two teams are equal"), so the model degraded to a coin flip with
nothing in the output to indicate it: 17/32 agreement with the seed favourite,
and 1-seeds over 16-seeds at 0.474-0.540.

Two things are pinned here: that the serving loader supplies the full payload,
and that a skewed payload now fails loudly instead of silently.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.prediction.noseed_model import (
    REQUIRED_FEATURE_KEYS,
    FeatureSkewError,
    _build_feature_vector,
    _get_stat,
    validate_stats_payload,
)

# Indices into _build_feature_vector's output for the four dimensions the
# four-factors-only payload could not populate.
_SKEWED_DIMS = {"adj_offensive_efficiency": 0, "adj_defensive_efficiency": 1, "adj_tempo": 2, "barthag": 11}

_FOUR_FACTOR_KEYS = (
    "effective_fg_pct",
    "turnover_rate",
    "offensive_reb_rate",
    "free_throw_rate",
    "opp_effective_fg_pct",
    "opp_turnover_rate",
    "defensive_reb_rate",
    "opp_free_throw_rate",
)


def _full_team(**over) -> dict:
    base = {
        "adj_offensive_efficiency": 118.0,
        "adj_defensive_efficiency": 92.0,
        "adj_tempo": 67.0,
        "effective_fg_pct": 0.55,
        "turnover_rate": 0.16,
        "offensive_reb_rate": 0.33,
        "free_throw_rate": 0.34,
        "opp_effective_fg_pct": 0.45,
        "opp_turnover_rate": 0.20,
        "defensive_reb_rate": 0.75,
        "opp_free_throw_rate": 0.28,
        "barthag": 0.95,
    }
    base.update(over)
    return base


def _four_factors_only(team: dict) -> dict:
    return {k: v for k, v in team.items() if k in _FOUR_FACTOR_KEYS}


# ---------------------------------------------------------------------------
# The defect itself
# ---------------------------------------------------------------------------


def test_four_factors_only_payload_zeroes_the_four_predictive_dims():
    """Demonstrates the failure mode, so its signature stays recognisable.

    This is what the model was being served: the differential is *exactly* zero
    on the four strongest dimensions, which is indistinguishable from two
    genuinely equal teams.
    """
    strong = _full_team()
    weak = _full_team(adj_offensive_efficiency=99.0, adj_defensive_efficiency=108.0, adj_tempo=70.0, barthag=0.35)

    skewed = _build_feature_vector(_four_factors_only(strong), _four_factors_only(weak))
    for name, idx in _SKEWED_DIMS.items():
        assert skewed[idx] == 0.0, f"{name} should be zeroed by a four-factors-only payload"

    # With the full payload those same dimensions carry real signal.
    full = _build_feature_vector(strong, weak)
    for name, idx in _SKEWED_DIMS.items():
        assert full[idx] != 0.0, f"{name} should be non-zero with the full payload"
    assert full[_SKEWED_DIMS["barthag"]] == pytest.approx(0.60)


def test_the_serving_loader_supplies_every_required_key():
    """The fix: mc_pool_backtest must hand over the FULL team-stats payload.

    Guards the exact regression -- if this loader is ever pointed back at
    ``_load_torvik_ff``, four features silently flatline again.
    """
    from scripts.mc_pool_backtest import _load_team_stats

    stats = _load_team_stats(2024)
    assert stats, "2024 team stats should load"

    teams = [t for t in stats.values() if isinstance(t, dict)]
    for key in REQUIRED_FEATURE_KEYS:
        coverage = sum(1 for t in teams if t.get(key) is not None) / len(teams)
        assert coverage > 0.9, f"{key} covered for only {coverage:.0%} of teams"


# ---------------------------------------------------------------------------
# The guard
# ---------------------------------------------------------------------------


def test_validate_rejects_a_four_factors_only_payload():
    payload = {f"team{i}": _four_factors_only(_full_team()) for i in range(20)}
    with pytest.raises(FeatureSkewError) as exc:
        validate_stats_payload(payload, context="unit")

    msg = str(exc.value)
    assert "train/serve skew" in msg
    for missing in _SKEWED_DIMS:
        assert missing in msg, f"error should name the missing key {missing}"
    assert "unit" in msg


def test_validate_accepts_a_full_payload():
    validate_stats_payload({f"team{i}": _full_team() for i in range(20)})


def test_validate_tolerates_one_team_missing_one_stat():
    """A per-key default exists for exactly this case; it must not raise."""
    payload = {f"team{i}": _full_team() for i in range(20)}
    del payload["team0"]["barthag"]
    validate_stats_payload(payload)


def test_validate_rejects_when_a_key_is_mostly_absent():
    payload = {f"team{i}": _full_team() for i in range(20)}
    for i in range(15):
        del payload[f"team{i}"]["barthag"]
    with pytest.raises(FeatureSkewError, match="barthag"):
        validate_stats_payload(payload)


def test_validate_reads_enriched_stats_fallback():
    payload = {}
    for i in range(20):
        t = _full_team()
        barthag = t.pop("barthag")
        t["enriched_stats"] = {"barthag": barthag}
        payload[f"team{i}"] = t
    validate_stats_payload(payload)


@pytest.mark.parametrize("bad", [{}, {"a": "not a dict"}])
def test_validate_rejects_degenerate_payloads(bad):
    with pytest.raises(FeatureSkewError):
        validate_stats_payload(bad)


def test_builders_reject_skewed_payloads():
    """Both serving entry points are guarded, not just one."""
    from src.prediction.noseed_model import (
        build_noseed_probabilities,
        build_noseed_round_probabilities,
    )

    seeds = {f"team{i}": (i % 16) + 1 for i in range(8)}
    payload = {t: _four_factors_only(_full_team()) for t in seeds}

    with pytest.raises(FeatureSkewError):
        build_noseed_probabilities(None, seeds, payload)
    with pytest.raises(FeatureSkewError):
        build_noseed_round_probabilities(None, seeds, payload)


# ---------------------------------------------------------------------------
# The falsy-coalescing bug in _get_stat
# ---------------------------------------------------------------------------


def test_get_stat_preserves_a_legitimate_zero():
    """``stats.get(k) or fallback`` treated 0.0 as missing. ``is None`` does not."""
    assert _get_stat({"turnover_rate": 0.0}, "turnover_rate", 0.18) == 0.0
    assert _get_stat({}, "turnover_rate", 0.18) == 0.18
    assert _get_stat({"turnover_rate": None}, "turnover_rate", 0.18) == 0.18


def test_get_stat_falls_back_to_enriched_then_default():
    assert _get_stat({"enriched_stats": {"barthag": 0.8}}, "barthag", 0.5) == 0.8
    assert _get_stat({"enriched_stats": {}}, "barthag", 0.5) == 0.5
    assert _get_stat({"barthag": float("nan")}, "barthag", 0.5) == 0.5


def test_feature_vector_is_antisymmetric():
    """All 12 dims are pure differentials, so swap(x) == -x.

    train_noseed_model relies on this for its symmetric augmentation.
    """
    a, b = _full_team(), _full_team(barthag=0.4, adj_tempo=71.0)
    np.testing.assert_allclose(_build_feature_vector(a, b), -_build_feature_vector(b, a))
