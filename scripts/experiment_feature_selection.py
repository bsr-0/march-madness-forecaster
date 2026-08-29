#!/usr/bin/env python3
"""How many features does the regular-season model actually need?

METHOD. Greedy forward selection: start empty, and at each step add whichever
remaining feature most improves WALK-FORWARD log loss on held-out NCAA
tournament games. Stop when nothing improves it.

SELECTION IS SCORED THE SAME WAY THE MODEL IS, which is the point. Ranking
features by in-sample coefficient size or by correlation with the target would
pick the ones that fit the training rows, not the ones that predict unseen
seasons. Every candidate here is judged by the same held-out metric the model
is finally reported on.

THE HONEST CAVEAT, stated because it is easy to forget. The selection path
itself sees every test season, so the log loss AT the selected size is
optimistically biased -- it is the best of many peeks. What the curve supports
is the SHAPE (how many features before it flattens) rather than the level. A
clean number would need an outer split, which 12 test seasons cannot spare.
This is a screening tool, not a benchmark.

Venue is always included and never selected: it is fitted so the home effect
does not contaminate the strength coefficients, and it is zero at prediction
because NCAA games are neutral court.

Run: python3 scripts/experiment_feature_selection.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.experiment_model_families import (  # noqa: E402
    CAL_PRIOR,
    MIN_TEST_YEAR,
    VENUE_KEYS,
    boot_ci,
    fit_ridge,
    load,
)


def score(cols, Xtr, mtr, ytr, Xev, mev, yev, years):
    wev = (mev > 0).astype(int)
    per = []
    for y in years:
        tr = ytr < y
        te = yev == y
        if tr.sum() < 500 or te.sum() == 0:
            continue
        predict = fit_ridge(Xtr[tr][:, cols], mtr[tr])
        resid = mtr[tr] - predict(Xtr[tr][:, cols])
        sigma = max(float(np.sqrt((resid**2).mean())), 1e-6)
        pm = yev < y
        a = 1.0
        if pm.sum() >= 100:
            ppr = predict(Xev[pm][:, cols])
            best = (9e9, 1.0)
            for aa in np.arange(0.05, 3.0, 0.01):
                q = np.clip(norm.cdf(aa * ppr / sigma), 1e-6, 1 - 1e-6)
                ll = -(wev[pm] * np.log(q) + (1 - wev[pm]) * np.log(1 - q)).mean()
                if ll < best[0]:
                    best = (ll, aa)
            n = pm.sum()
            a = (n * best[1] + CAL_PRIOR) / (n + CAL_PRIOR)
        p = np.clip(norm.cdf(a * predict(Xev[te][:, cols]) / sigma), 1e-6, 1 - 1e-6)
        per.append(-(wev[te] * np.log(p) + (1 - wev[te]) * np.log(1 - p)))
    return np.concatenate(per)


def main() -> int:
    keys, Xtr, mtr, ytr, Xev, mev, yev = load()
    years = [y for y in sorted(set(yev.tolist())) if y >= MIN_TEST_YEAR]

    venue = [i for i, k in enumerate(keys) if k in VENUE_KEYS]
    pool = [i for i, k in enumerate(keys) if k not in VENUE_KEYS]

    print(
        f"\n{Xtr.shape[0]:,} train rows | {len(mev)} test games | "
        f"{len(pool)} selectable features + {len(venue)} venue (always in)"
    )
    print("greedy forward selection on walk-forward log loss\n")

    chosen, history = [], []
    best_so_far = None
    while pool:
        cand = []
        for j in pool:
            per = score(venue + chosen + [j], Xtr, mtr, ytr, Xev, mev, yev, years)
            cand.append((per.mean(), j, per))
        cand.sort(key=lambda t: t[0])
        val, j, per = cand[0]
        if best_so_far is not None and val >= best_so_far:
            print(f"  stop: no remaining feature improves on {best_so_far:.5f}")
            break
        best_so_far = val
        chosen.append(j)
        pool.remove(j)
        history.append((len(chosen), keys[j], val, per))
        print(f"  {len(chosen):>2}. +{keys[j]:<28} {val:.5f}")

    print(f"\n  selected {len(chosen)}: {', '.join(keys[i] for i in chosen)}")

    full = score(venue + [i for i, k in enumerate(keys) if k not in VENUE_KEYS], Xtr, mtr, ytr, Xev, mev, yev, years)
    best = history[-1][3]
    rng = np.random.default_rng(0)
    lo, hi = boot_ci(full, best, rng)
    print(f"\n  all features      {full.mean():.5f}")
    print(f"  selected subset   {best.mean():.5f}")
    print(
        f"  difference        {full.mean() - best.mean():+.5f}  "
        f"95% CI [{lo:+.5f}, {hi:+.5f}]  "
        f"{'FINDING' if lo > 0 else 'not a finding'}"
    )
    print(
        "\n  Remember the selection path saw every test season, so the level "
        "at the chosen size\n  is optimistic. The shape of the curve is the "
        "usable part."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
