"""The Python port must keep reproducing the frozen browser baseline.

src/prediction/pit_production_model.py is a port of docs/fit.js, and it is only
useful while it is faithful. The failure mode it guards against is silent: a
port that drifts does not raise, it returns slightly different probabilities,
and every pool result computed from it describes a model that is not shipped.

These tests are the same check scripts/verify_pit_port.py performs, wired into
the suite so that editing either side of the language boundary -- the JS model
or the Python port -- fails here rather than months later in a pool number
nobody can reconcile.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.prediction.pit_production_model import (
    CANONICAL_KEYS,
    WARM_PRIOR_N,
    load_training,
    resolve_cols,
    score,
    walk_forward,
    walk_forward_calibration,
)

REPO = Path(__file__).resolve().parent.parent
BASELINE = REPO / "artifacts" / "model_baseline.json"


@pytest.fixture(scope="module")
def frozen():
    return json.loads(BASELINE.read_text())


@pytest.fixture(scope="module")
def ported():
    keys, X, m, y = load_training()
    cols = resolve_cols(keys, CANONICAL_KEYS)
    py, pm, pp, ps = walk_forward(X, m, y, cols)
    cal = walk_forward_calibration(py, pm, pp, ps)
    warm_years = {yr for yr, c in cal.items() if c["priorN"] >= WARM_PRIOR_N}
    warm = [i for i, yr in enumerate(py.tolist()) if yr in warm_years]
    return {
        "all": score(py, pm, pp, ps, cal),
        "warm": score(py[warm], pm[warm], pp[warm], ps[warm], cal),
        "cal": cal,
    }


def test_canonical_keys_match_the_frozen_baseline(frozen):
    """The baseline is defined by key. A renamed variable must be deliberate."""
    assert tuple(frozen["canonicalKeys"]) == CANONICAL_KEYS


@pytest.mark.parametrize(
    "bucket, field",
    [
        ("all", "calibratedWalkForward"),
        ("warm", "calibratedWalkForwardWarm"),
    ],
)
def test_log_loss_reproduces_frozen_baseline(frozen, ported, bucket, field):
    assert ported[bucket]["n"] == frozen[field]["n"]
    assert ported[bucket]["logLoss"] == pytest.approx(frozen[field]["logLoss"], abs=5e-4)


def test_warm_secondary_metrics_reproduce(frozen, ported):
    want, got = frozen["calibratedWalkForwardWarm"], ported["warm"]
    assert got["brier"] == pytest.approx(want["brier"], abs=5e-3)
    assert got["accuracy"] == pytest.approx(want["accuracy"], abs=5e-3)
    assert got["rmse"] == pytest.approx(want["rmse"], abs=5e-3)


def test_per_year_link_parameters_reproduce(frozen, ported):
    """Aggregates can match while folds drift in compensating directions."""
    for yr, want in frozen["walkForwardCalibration"].items():
        got = ported["cal"][int(yr)]
        want_nu = want["nu"] if isinstance(want["nu"], (int, float)) else np.inf
        assert got["a"] == pytest.approx(want["a"], abs=5e-3), f"a drifted in {yr}"
        assert got["nu"] == want_nu, f"nu drifted in {yr}"


def test_pairwise_probabilities_are_antisymmetric():
    """p(a,b) + p(b,a) = 1 is a structural property of an intercept-free model.

    Checked on real teams rather than asserted, because the property survives
    only while the model has no intercept and the link stays symmetric about
    zero -- both of which are edits someone could plausibly make.
    """
    from src.prediction.pit_production_model import pairwise_for_year

    stats = json.loads((REPO / "docs" / "data" / "team_stats_by_year.json").read_text())
    teams = sorted(r["team_id"] for r in stats["stats_by_year"]["2019"])[:12]
    pw = pairwise_for_year(2019, teams)
    for a in teams:
        for b in teams:
            if a != b:
                assert pw[(a, b)] + pw[(b, a)] == pytest.approx(1.0, abs=1e-12)


def test_pairwise_refuses_a_year_it_cannot_fit():
    """No silent fallback: too little history must raise, not return a guess."""
    from src.prediction.pit_production_model import pairwise_for_year

    with pytest.raises((ValueError, KeyError)):
        pairwise_for_year(2010, ["duke", "kansas"])
