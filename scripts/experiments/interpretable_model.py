#!/usr/bin/env python3
"""Build a small, readable companion to the 24-variable forecaster.

WHY A SECOND MODEL RATHER THAN A TUNED FIRST ONE
The shipped model answers "which team wins". It answers it well -- 78.2%
walk-forward -- and its coefficients are unreadable, because near-substitutes
(overall rating and national rank) split credit between them and land at -39 and
+36. More than half its coefficients change sign between folds.

Detuning it to fix that costs about four accuracy points. So this builds a
SEPARATE model whose only job is to be interpretable, leaving the forecaster
alone. The two answer different questions and there is no reason one artefact
must do both.

SELECTION RULE
Greedy forward selection, where a candidate is admitted only if it BOTH improves
walk-forward error and keeps a stable sign across every fold. Stability is a
hard gate, not a tiebreak: a coefficient that reads +0.8, -0.2, +1.4 across folds
is not an effect, however much it improves the average.

This deliberately rejects variables that only matter in combination with others.
That is a real cost and the reason this model is not the forecaster: it will
score lower, and the gap is the price of being able to read it.

STOPPING RULE
Stop when no remaining candidate passes both gates. Two guards on top:

  MIN_IMPROVEMENT   an RMSE gain below this is noise at 63 games per fold, and
                    admitting it buys an unreadable extra term for nothing.
  MAX_VARS          a hard ceiling, because "interpretable" has a size limit
                    regardless of what the arithmetic says.

EVALUATION IS THE SAME AS THE FORECASTER'S
Walk-forward: fit on strictly earlier seasons, score the held-out one. Selection
itself runs inside that loop's aggregate, which means the reported figure is
optimistic -- the variable set was chosen with knowledge of all folds. That is
disclosed rather than hidden, and the honest comparison is at the bottom: the
selected set is re-scored under a nested walk-forward where selection is redone
from scratch inside each fold, using only that fold's training seasons.

WHAT THIS PRODUCES, AND ITS PRICE
Three variables, all positive, all sign-stable: national rank at +7.6 points per
standard deviation, offensive rebounding at +0.8, road wins at +0.8. That is a
readable model. It costs 3.2 points of accuracy against the 24-variable
forecaster (75.0% vs 78.2%), which is roughly the price quoted before building
it.

Two honest caveats on reading it. The equation leans almost entirely on one
composite rating -- `t_rank` already contains offense and defense, which is
precisely why they cannot be admitted alongside it without one going negative.
And the nested check shows folds picking anywhere from 1 to 6 variables, with
only `t_rank` surviving in all of them. The procedure is more stable than the
24-variable fit, not stable in absolute terms.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

MATRIX = REPO / "docs" / "data" / "training.json"

# NOT under docs/. Everything in docs/data/ is published with the site, and this
# is an inference artefact: it exists to answer "what is this variable worth",
# which is a question the product deliberately does not put in front of a user.
# The UI ships predictions -- a bracket, and how often that bracket has been
# right on seasons it never saw. Analysis output lives here instead.
OUT = REPO / "artifacts" / "interpretable_model.json"

MIN_TEST_YEAR = 2014
RIDGE_PER_1000 = 1.0
MIN_IMPROVEMENT = 0.02  # points of RMSE
MAX_VARS = 8

# Ridge shrinkage can push a genuinely-zero coefficient a hair below zero
# without it being a suppressor. This separates "numerically ~0" from
# "cancelling a collinear partner", which is orders of magnitude larger.
SIGN_TOL = 0.05


def fit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    lam = RIDGE_PER_1000 * (len(y) / 1000)
    return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)


def walk_forward(
    X: np.ndarray, m: np.ndarray, yr: np.ndarray, cols: Sequence[int], years: Sequence[int]
) -> Dict[str, object]:
    """Fit on strictly earlier seasons, score the held-out one, pool the results."""
    sse = sae = sst = 0.0
    correct = n = 0
    betas: List[np.ndarray] = []
    for ty in years:
        if ty < MIN_TEST_YEAR:
            continue
        tr, te = yr < ty, yr == ty
        if tr.sum() < len(cols) * 5 or not te.any():
            continue
        beta = fit(X[tr][:, cols], m[tr])
        betas.append(beta)
        pred = X[te][:, cols] @ beta
        sse += float(((m[te] - pred) ** 2).sum())
        sae += float(np.abs(m[te] - pred).sum())
        sst += float((m[te] ** 2).sum())
        correct += int(((pred > 0) == (m[te] > 0)).sum())
        n += int(te.sum())
    if not n:
        return {}
    B = np.array(betas)
    return {
        "rmse": (sse / n) ** 0.5,
        "mae": sae / n,
        "r2": 1 - sse / sst,
        "accuracy": correct / n,
        "n": n,
        "folds": len(betas),
        "betas": B,
        "stable": [bool((B[:, j] > 0).all() or (B[:, j] < 0).all()) for j in range(B.shape[1])],
        # Every z-score is sign-corrected upstream so that HIGHER IS BETTER.
        # A negative coefficient therefore says "being better at this loses
        # games", which is never the finding -- it means the variable is acting
        # as a suppressor, cancelling part of a collinear partner. See SIGN_TOL.
        "signed_correctly": [bool((B[:, j] > -SIGN_TOL).all()) for j in range(B.shape[1])],
    }


def select(
    X: np.ndarray,
    m: np.ndarray,
    yr: np.ndarray,
    years: Sequence[int],
    keys: Sequence[str],
    verbose: bool = True,
    pool: Sequence[int] | None = None,
) -> List[int]:
    """Greedy forward selection under two hard gates.

    GATE 1, sign stability across folds. Rejects a coefficient that reads
    +0.8, -0.2, +1.4 -- not an effect, whatever its average.

    GATE 2, correct sign. Rejects a coefficient that is stable but negative.
    This gate exists because gate 1 alone does not work, which is worth stating
    plainly: run selection on stability alone and it happily admits `barthag` at
    -39.79 alongside `t_rank` at +36.37. Those two are near-substitutes, so the
    fit cancels them against each other, and it does so REPRODUCIBLY -- barthag
    lands between -43 and -38 in every single fold. Perfectly stable, perfectly
    meaningless: "a better overall rating costs you 40 points" is not a finding,
    it is an artefact of two columns measuring the same thing.

    Because every z-score is sign-corrected upstream so higher is better, the
    honest form of "readable" is simply that no coefficient may be negative.
    That makes each term a contribution in points that can be quoted on its own.
    """
    chosen: List[int] = []
    best_rmse = float("inf")

    while len(chosen) < MAX_VARS:
        candidate = None
        for j in pool if pool is not None else range(X.shape[1]):
            if j in chosen:
                continue
            r = walk_forward(X, m, yr, chosen + [j], years)
            if not r or not all(r["stable"]) or not all(r["signed_correctly"]):
                continue
            if r["rmse"] < best_rmse - MIN_IMPROVEMENT and (candidate is None or r["rmse"] < candidate[1]):
                candidate = (j, r["rmse"], r)
        if candidate is None:
            break
        j, rmse, r = candidate
        chosen.append(j)
        best_rmse = rmse
        if verbose:
            print(f"  + {keys[j]:28} RMSE {rmse:6.3f}  acc {100 * r['accuracy']:.1f}%  ({len(chosen)} vars)")
    return chosen


def nested_score(
    X: np.ndarray,
    m: np.ndarray,
    yr: np.ndarray,
    years: Sequence[int],
    keys: Sequence[str],
    pool: Sequence[int] | None = None,
) -> Dict[str, object]:
    """Redo selection inside every fold, using only that fold's training seasons.

    This is the number that is actually honest about how the PROCEDURE
    generalises. The headline figure above reuses one variable set chosen with
    sight of every fold, so it flatters itself; the gap between the two is the
    size of that selection bias.
    """
    sse = sst = 0.0
    correct = n = 0
    picked: List[List[str]] = []
    for ty in years:
        if ty < MIN_TEST_YEAR:
            continue
        tr, te = yr < ty, yr == ty
        if not te.any() or tr.sum() < 50:
            continue
        inner_years = [y for y in years if y < ty]
        if len(inner_years) < 4:
            continue
        cols = select(X[tr], m[tr], yr[tr], inner_years, keys, verbose=False, pool=pool)
        if not cols:
            continue
        beta = fit(X[tr][:, cols], m[tr])
        pred = X[te][:, cols] @ beta
        sse += float(((m[te] - pred) ** 2).sum())
        sst += float((m[te] ** 2).sum())
        correct += int(((pred > 0) == (m[te] > 0)).sum())
        n += int(te.sum())
        picked.append([keys[c] for c in cols])
    if not n:
        return {}
    return {
        "rmse": (sse / n) ** 0.5,
        "r2": 1 - sse / sst,
        "accuracy": correct / n,
        "n": n,
        "picked": picked,
    }


# Composite team-quality summaries. `barthag` and `t_rank` are not variables
# alongside the others so much as summaries OF them, which is why admitting one
# blocks offense and defense from entering: they are already inside it.
#
# Excluding them is available as `--no-composites` to answer a specific
# question -- "what can the component variables be measured at once nothing is
# summarising them?" -- and NOT because they are illegitimate. They are the
# single most informative thing in the matrix. Measured walk-forward:
#
#   rank pair alone            R2 0.463
#   the other 22 alone         R2 0.418
#   all 24                     R2 0.538
#
# so the pair contributes +0.119 of R2 that the other 22 cannot reconstruct,
# while the other 22 contribute +0.075 beyond the pair. Dropping them costs 4.7
# points of accuracy. That is a measurement instrument, not a forecasting one.
COMPOSITES = ("t_rank", "barthag")

# Withholding the rank pair does NOT free the component variables -- offense and
# defense simply inherit the role, because adjusted efficiency is itself a
# summary of possessions rather than a component. The matrix has a hierarchy:
#
#   tier 1  t_rank, barthag                    a summary of the whole team
#   tier 2  adj_offensive/defensive_efficiency a summary of each half of it
#   tier 3  four factors, shooting, form, ...  the things being summarised
#
# `--components-only` withholds tiers 1 and 2 together, which is the only way to
# ask what tier 3 measures on its own. `adj_tempo` stays: it describes style,
# not quality, and summarises nothing.
COMPOSITE_TIERS = COMPOSITES + ("adj_offensive_efficiency", "adj_defensive_efficiency")

# READ THIS BEFORE QUOTING ANY COEFFICIENT FROM A WITHHELD RUN.
#
# Withholding the composites makes the remaining coefficients large, stable and
# readable. It does NOT make them more valid -- it makes them worse, and the
# mechanism is ordinary omitted-variable bias. Team quality is the dominant
# cause of margin; remove every column that measures it and the surviving
# variables absorb it, because they correlate with it.
#
# The same variable, fitted alone / with t_rank / with offense and defense:
#
#     offensive_reb_rate         3.09    0.83    0.72
#     reg_season_margin_avg      4.88    0.94   -0.21
#     turnover_rate              2.37    0.01   -0.27
#     coach_prior_tourney_wins   2.94    0.41    0.12
#     true_road_win_pct          3.01    0.80    0.19
#     effective_fg_pct           2.46    0.18   -0.58
#
# `reg_season_margin_avg` at +3.40 in a components-only run is not "average
# margin is worth 3.4 points a standard deviation". It is average margin
# standing in for the overall quality that was withheld. Control for quality and
# the honest partial effects are the right-hand columns: small, and several of
# them negative or indistinguishable from zero.
#
# So there are two failure modes and no clean escape:
#
#   INCLUDE the composites -> coefficients are partial effects that control for
#     quality, but collinearity makes them unstable and unreadable.
#   EXCLUDE them -> coefficients are stable and readable, and confounded.
#
# Neither is a causal effect. The genuine finding is the unglamorous one: once
# team quality is accounted for, the component variables are worth tenths of a
# point, not points. That is a result about basketball, not an artefact to be
# tuned away.
CONFOUNDING_WARNING = (
    "Composites were withheld. The coefficients below are inflated by omitted-variable "
    "bias -- they absorb the team quality that was removed -- and must not be quoted as "
    "the effect of the named variable. See COMPOSITE_TIERS in the source."
)


def main() -> int:
    d = json.loads(MATRIX.read_text())
    keys = d["keys"]
    drop_composites = "--no-composites" in sys.argv
    components_only = "--components-only" in sys.argv
    X = np.array([g["x"] for g in d["games"]], dtype=float)
    m = np.array([g["m"] for g in d["games"]], dtype=float)
    yr = np.array([g["y"] for g in d["games"]])
    years = sorted(set(int(y) for y in yr))

    full = walk_forward(X, m, yr, list(range(len(keys))), years)
    print(f"Forecaster, all {len(keys)} variables (for reference)")
    print(
        f"  acc {100 * full['accuracy']:.1f}%   RMSE {full['rmse']:.2f}   "
        f"R2 {full['r2']:.3f}   "
        f"{sum(1 for s in full['stable'] if not s)} of {len(keys)} coefficients flip sign\n"
    )

    pool = list(range(len(keys)))
    withheld = COMPOSITE_TIERS if components_only else (COMPOSITES if drop_composites else ())
    if withheld:
        pool = [i for i in pool if keys[i] not in withheld]
        flag = "--components-only" if components_only else "--no-composites"
        print(f"{flag}: withheld from selection — {', '.join(withheld)}")
        print("  WARNING: " + CONFOUNDING_WARNING + "\n")

    print("Forward selection (admit only if RMSE improves AND sign is stable in every fold)")
    cols = select(X, m, yr, years, keys, pool=pool)
    if not cols:
        print("  no variable passed both gates")
        return 1

    res = walk_forward(X, m, yr, cols, years)
    beta = fit(X[:, cols], m)

    print(f"\nInterpretable model: {len(cols)} variables")
    print(f"  acc {100 * res['accuracy']:.1f}%   RMSE {res['rmse']:.2f}   MAE {res['mae']:.2f}   R2 {res['r2']:.3f}")
    print(
        f"  vs forecaster: {100 * (res['accuracy'] - full['accuracy']):+.1f} pts accuracy, "
        f"{res['rmse'] - full['rmse']:+.2f} RMSE\n"
    )

    print("  Equation (points of margin per standard deviation of edge):")
    order = sorted(range(len(cols)), key=lambda i: -abs(beta[i]))
    for i in order:
        series = res["betas"][:, i]
        print(f"    {beta[i]:+7.2f}  {keys[cols[i]]:28} across folds {series.min():+.2f} to {series.max():+.2f}")

    print("\nNested check — selection redone from scratch inside every fold")
    nested = nested_score(X, m, yr, years, keys, pool=pool)
    if nested:
        print(f"  acc {100 * nested['accuracy']:.1f}%   RMSE {nested['rmse']:.2f}   R2 {nested['r2']:.3f}")
        gap = 100 * (res["accuracy"] - nested["accuracy"])
        direction = "optimistic" if gap > 0 else "pessimistic"
        print(
            f"  headline above is {abs(gap):.1f} pts {direction} — that gap is the "
            "selection bias, whose sign is not guaranteed at 63 games a fold"
        )
        sizes = [len(p) for p in nested["picked"]]
        always = sorted(set.intersection(*(set(p) for p in nested["picked"])))
        print(f"  per-fold set size {min(sizes)}-{max(sizes)}; chosen in EVERY fold: {always}")
        if max(sizes) - min(sizes) > 2:
            print(
                "  NOTE: the selection procedure is itself unstable — folds disagree on "
                "which variables to keep. Read the equation as one plausible account of "
                "the data, not the account."
            )

    OUT.write_text(
        json.dumps(
            {
                "purpose": (
                    "Companion model built for interpretation, not forecasting. The "
                    "24-variable ridge model in fit.js is the predictor; this exists "
                    "to give a coefficient vector that can be read."
                ),
                "withheld_variables": list(withheld),
                "confounding_warning": CONFOUNDING_WARNING if withheld else None,
                "keys": [keys[c] for c in cols],
                "beta": [round(float(b), 4) for b in beta],
                "units": "points of scoring margin per standard deviation of edge",
                "selection": {
                    "method": "greedy forward, hard sign-stability gate across walk-forward folds",
                    "min_rmse_improvement": MIN_IMPROVEMENT,
                    "max_vars": MAX_VARS,
                    "ridge_per_1000_rows": RIDGE_PER_1000,
                },
                "walk_forward": {
                    "accuracy": round(res["accuracy"], 4),
                    "rmse": round(res["rmse"], 3),
                    "mae": round(res["mae"], 3),
                    "r2": round(res["r2"], 4),
                    "n": res["n"],
                    "folds": res["folds"],
                    "note": (
                        "Optimistic: the variable set was chosen with sight of every "
                        "fold. See nested_walk_forward for the unbiased figure."
                    ),
                },
                "nested_walk_forward": (
                    {
                        "accuracy": round(nested["accuracy"], 4),
                        "rmse": round(nested["rmse"], 3),
                        "r2": round(nested["r2"], 4),
                        "n": nested["n"],
                        "note": "Selection redone inside each fold. This is the honest number.",
                    }
                    if nested
                    else None
                ),
                "coefficient_range_across_folds": {
                    keys[cols[i]]: [
                        round(float(res["betas"][:, i].min()), 4),
                        round(float(res["betas"][:, i].max()), 4),
                    ]
                    for i in range(len(cols))
                },
                "forecaster_for_comparison": {
                    "n_variables": len(keys),
                    "accuracy": round(full["accuracy"], 4),
                    "rmse": round(full["rmse"], 3),
                    "sign_flipping_coefficients": sum(1 for s in full["stable"] if not s),
                },
            },
            indent=2,
        )
    )
    print(f"\nwrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
