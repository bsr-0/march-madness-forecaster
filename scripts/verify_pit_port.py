#!/usr/bin/env python3
"""Gate: does the Python port reproduce the frozen JavaScript baseline?

src/prediction/pit_production_model.py exists so the pool optimizer can consume
the probabilities the UI actually ships. That is only true if it computes the
same numbers. This script recomputes the baseline from the port and diffs it
against artifacts/model_baseline.json, which scripts/model_baseline.js froze.

WHY A HARD GATE RATHER THAN A SPOT CHECK. A port that is subtly wrong -- an
intercept that should not be there, a sigma taken held-out instead of
in-sample, a nu grid off by one entry -- does not crash. It produces plausible
probabilities that describe a model nobody ships, and every pool result built
on it would be measuring the wrong thing while looking fine. This session has
already found six defects of exactly that shape, so the port is presumed broken
until it reproduces the frozen numbers.

TOLERANCE. 5e-4 on log loss, which is far looser than the ~1e-10 agreement the
arithmetic should give and far tighter than any effect anyone would act on. The
remaining slack is the two implementations' incomplete-beta routines: the JS
uses a hand-rolled continued fraction, this uses scipy.

Run: python3 scripts/verify_pit_port.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.prediction.pit_production_model import (  # noqa: E402
    CANONICAL_KEYS,
    WARM_PRIOR_N,
    load_training,
    resolve_cols,
    score,
    walk_forward,
    walk_forward_calibration,
)

BASELINE = REPO / "artifacts" / "model_baseline.json"
TOL_LL = 5e-4
TOL_A = 5e-3


def main() -> int:
    frozen = json.loads(BASELINE.read_text())

    if tuple(frozen["canonicalKeys"]) != CANONICAL_KEYS:
        print("FAIL: canonical key set differs from the frozen baseline.")
        print(f"  frozen: {frozen['canonicalKeys']}")
        print(f"  port  : {list(CANONICAL_KEYS)}")
        return 1

    keys, X, m, y = load_training()
    cols = resolve_cols(keys, CANONICAL_KEYS)
    py, pm, pp, ps = walk_forward(X, m, y, cols)
    cal = walk_forward_calibration(py, pm, pp, ps)

    warm_years = {yr for yr, c in cal.items() if c["priorN"] >= WARM_PRIOR_N}
    warm = [i for i, yr in enumerate(py.tolist()) if yr in warm_years]

    got_all = score(py, pm, pp, ps, cal)
    got_warm = score(py[warm], pm[warm], pp[warm], ps[warm], cal)

    ok = True
    print(f"\n  {'metric':<34}{'frozen':>12}{'port':>12}{'delta':>12}")
    for label, want, got in (
        ("calibratedWalkForward.logLoss", frozen["calibratedWalkForward"], got_all),
        ("calibratedWalkForwardWarm.logLoss", frozen["calibratedWalkForwardWarm"], got_warm),
    ):
        d = got["logLoss"] - want["logLoss"]
        flag = "" if abs(d) <= TOL_LL else "   <-- FAIL"
        if flag:
            ok = False
        print(f"  {label:<34}{want['logLoss']:>12.5f}{got['logLoss']:>12.5f}{d:>+12.6f}{flag}")
        if want["n"] != got["n"]:
            print(f"    n differs: frozen {want['n']} vs port {got['n']}   <-- FAIL")
            ok = False

    for label, want, got in (
        ("warm brier", frozen["calibratedWalkForwardWarm"]["brier"], got_warm["brier"]),
        ("warm accuracy", frozen["calibratedWalkForwardWarm"]["accuracy"], got_warm["accuracy"]),
        ("warm rmse", frozen["calibratedWalkForwardWarm"]["rmse"], got_warm["rmse"]),
    ):
        d = got - want
        flag = "" if abs(d) <= 5e-3 else "   <-- FAIL"
        if flag:
            ok = False
        print(f"  {label:<34}{want:>12.5f}{got:>12.5f}{d:>+12.6f}{flag}")

    # Per-year link parameters. The aggregate can match while individual folds
    # drift in compensating directions, so the folds are checked too.
    print(f"\n  {'year':<8}{'frozen a':>10}{'port a':>10}{'frozen nu':>11}{'port nu':>9}")
    for yr in sorted(cal):
        f = frozen["walkForwardCalibration"].get(str(yr))
        if not f:
            continue
        c = cal[yr]
        fnu = f["nu"] if isinstance(f["nu"], (int, float)) else float("inf")
        da = c["a"] - f["a"]
        bad = abs(da) > TOL_A or fnu != c["nu"]
        if bad:
            ok = False
        print(f"  {yr:<8}{f['a']:>10.4f}{c['a']:>10.4f}{fnu:>11}{c['nu']:>9}{'   <-- FAIL' if bad else ''}")

    print(
        "\n  PASS: the port reproduces the frozen baseline."
        if ok
        else "\n  FAIL: the port does not reproduce the frozen baseline. Do not build on it."
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
