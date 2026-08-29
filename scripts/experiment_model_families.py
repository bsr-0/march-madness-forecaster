#!/usr/bin/env python3
"""Compare model families on the regular-season training population.

THE QUESTION. The browser fits a ridge spread regression. Does a different
model family do better on the same rows -- k-nearest neighbours, locally
weighted linear regression, or gradient boosting? And how few features does it
actually need?

THE SETUP, and its limits stated up front.
  train   training_pit.json, 41,321 regular-season and conference-tournament
          rows, standardised within the D1 field at each week boundary
  test    eval_pit_tournament.json, held-out NCAA tournament games, built by
          the same code path at the Selection Sunday boundary so the two are
          on one scale
  split   WALK-FORWARD, not leave-one-year-out: season Y is predicted from
          seasons strictly before it. LOYO would let 2024 train on 2025, which
          nobody could have done at the time.

Every model predicts a MARGIN and is carried to a probability by the same
Student-t link, with the scale parameter fitted on prior years' held-out
predictions only and shrunk toward 1 while residuals are scarce. Holding the
link fixed is what makes the families comparable: a model that looked better
because it got a better-tuned link would be measuring the link.

WHAT THIS CANNOT TELL YOU. This population is measurably worse than the
tournament-only one the shipped model uses -- +0.0823 for lacking t_rank, which
has no dated snapshot, and +0.0420 for the wider standardisation. So absolute
numbers here sit near 0.57 against production's 0.45296 and are NOT comparable
to it. The comparison that means something is BETWEEN the families below, all
of which pay those same two costs.

THE STOPPING RULE APPLIES. A difference smaller than its paired bootstrap CI is
not a finding. Three results dissolved under that test earlier in this work,
including two that had already been written up as wins.

Run: python3 scripts/experiment_model_families.py [--quick]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import norm

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

TRAIN = REPO / "docs" / "data" / "training_pit.json"
EVAL = REPO / "docs" / "data" / "eval_pit_tournament.json"

MIN_TEST_YEAR = 2014
CAL_PRIOR = 63  # games-equivalent shrinkage for the link's scale
BOOT = 4000

# Venue is fitted but contributes nothing at prediction: NCAA games are neutral
# court, and eval_pit already carries 0 for the court terms. Excluding it from
# the fit would push the home effect into the correlated strength coefficients.
VENUE_KEYS = ("venue_home", "venue_host_city", "venue_travel")


def load():
    tr = json.loads(TRAIN.read_text())
    ev = json.loads(EVAL.read_text())
    keys = [k for k in tr["keys"] if k in ev["keys"]]
    ti = [tr["keys"].index(k) for k in keys]
    ei = [ev["keys"].index(k) for k in keys]
    Xtr = np.array([g["x"] for g in tr["games"]], dtype=float)[:, ti]
    mtr = np.array([g["m"] for g in tr["games"]], dtype=float)
    ytr = np.array([g["y"] for g in tr["games"]])
    Xev = np.array([g["x"] for g in ev["games"]], dtype=float)[:, ei]
    mev = np.array([g["m"] for g in ev["games"]], dtype=float)
    yev = np.array([g["y"] for g in ev["games"]])
    return keys, Xtr, mtr, ytr, Xev, mev, yev


# ---------------------------------------------------------------- models
def fit_ridge(X, m, lam=1.0):
    n, k = X.shape
    A = np.column_stack([np.ones(n), X])
    G = A.T @ A
    G[np.arange(1, k + 1), np.arange(1, k + 1)] += lam * (n / 1000.0)
    beta = np.linalg.solve(G, A.T @ m)
    return lambda Q: np.column_stack([np.ones(len(Q)), Q]) @ beta


def fit_knn(X, m, k=25):
    from sklearn.neighbors import NearestNeighbors

    # Both orientations, so the neighbour set of a query is the mirror of the
    # set for the swapped query and the predictions negate -- the antisymmetry
    # the board depends on. Storing one orientation does not give it for free.
    XX = np.vstack([X, -X])
    mm = np.concatenate([m, -m])
    nn = NearestNeighbors(n_neighbors=min(k, len(XX))).fit(XX)

    def predict(Q):
        _, idx = nn.kneighbors(Q)
        return mm[idx].mean(axis=1)

    return predict


def fit_local_linear(X, m, k=200):
    """Locally weighted linear regression: a ridge fit per query on its k
    nearest rows, tricube-weighted by distance. Bends where the global fit
    cannot, at the cost of one solve per prediction."""
    from sklearn.neighbors import NearestNeighbors

    XX = np.vstack([X, -X])
    mm = np.concatenate([m, -m])
    nn = NearestNeighbors(n_neighbors=min(k, len(XX))).fit(XX)

    def predict(Q):
        dist, idx = nn.kneighbors(Q)
        out = np.empty(len(Q))
        for i in range(len(Q)):
            d = dist[i]
            h = max(d[-1], 1e-9)
            w = (1 - (d / h) ** 3) ** 3  # tricube
            Xi = XX[idx[i]]
            yi = mm[idx[i]]
            A = np.column_stack([np.ones(len(Xi)), Xi]) * np.sqrt(w)[:, None]
            b = yi * np.sqrt(w)
            G = A.T @ A + 1e-3 * np.eye(A.shape[1])
            beta = np.linalg.solve(G, A.T @ b)
            out[i] = np.concatenate([[1.0], Q[i]]) @ beta
        return out

    return predict


def fit_lgbm(X, m, **kw):
    import lightgbm as lgb

    params = dict(
        objective="regression",
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=40,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        verbose=-1,
        n_jobs=-1,
    )
    params.update(kw)
    # Mirrored rows again: a tree model has no built-in antisymmetry, and
    # training on both orientations is what supplies it.
    mdl = lgb.LGBMRegressor(**params).fit(np.vstack([X, -X]), np.concatenate([m, -m]))
    return lambda Q: mdl.predict(Q)


# ---------------------------------------------------------------- evaluation
def walk_forward(make, Xtr, mtr, ytr, Xev, mev, yev, years):
    """Per-game held-out log loss, with the link's scale fitted on prior years."""
    wev = (mev > 0).astype(int)
    per, keep = [], []
    for y in years:
        tr = ytr < y
        te = yev == y
        if tr.sum() < 500 or te.sum() == 0:
            continue
        predict = make(Xtr[tr], mtr[tr])
        resid = mtr[tr] - predict(Xtr[tr])
        sigma = max(float(np.sqrt((resid**2).mean())), 1e-6)

        pm = yev < y
        a = 1.0
        if pm.sum() >= 100:
            ppr = predict(Xev[pm])
            grid = np.arange(0.05, 3.0, 0.01)
            best = (9e9, 1.0)
            for aa in grid:
                q = np.clip(norm.cdf(aa * ppr / sigma), 1e-6, 1 - 1e-6)
                ll = -(wev[pm] * np.log(q) + (1 - wev[pm]) * np.log(1 - q)).mean()
                if ll < best[0]:
                    best = (ll, aa)
            n = pm.sum()
            a = (n * best[1] + CAL_PRIOR * 1.0) / (n + CAL_PRIOR)

        p = np.clip(norm.cdf(a * predict(Xev[te]) / sigma), 1e-6, 1 - 1e-6)
        per.append(-(wev[te] * np.log(p) + (1 - wev[te]) * np.log(1 - p)))
        keep.append(np.where(te)[0])
    return np.concatenate(per), np.concatenate(keep)


def boot_ci(a, b, rng, n=BOOT):
    """Paired bootstrap on the difference in mean log loss."""
    m = len(a)
    d = np.empty(n)
    for i in range(n):
        j = rng.integers(0, m, m)
        d[i] = a[j].mean() - b[j].mean()
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="fewer years and models")
    args = ap.parse_args()

    keys, Xtr, mtr, ytr, Xev, mev, yev = load()
    years = [y for y in sorted(set(yev.tolist())) if y >= MIN_TEST_YEAR]
    if args.quick:
        years = years[-4:]
    print(
        f"\ntrain {Xtr.shape[0]:,} rows x {Xtr.shape[1]} features | "
        f"test {len(mev)} tournament games | walk-forward over {len(years)} seasons"
    )
    print(f"venue in the fit, zero at prediction: {[k for k in VENUE_KEYS if k in keys]}\n")

    models = [
        ("ridge", lambda X, m: fit_ridge(X, m)),
        ("knn k=25", lambda X, m: fit_knn(X, m, 25)),
        ("knn k=100", lambda X, m: fit_knn(X, m, 100)),
        ("knn k=500", lambda X, m: fit_knn(X, m, 500)),
        ("local linear k=200", lambda X, m: fit_local_linear(X, m, 200)),
        ("lightgbm", lambda X, m: fit_lgbm(X, m)),
    ]
    if args.quick:
        models = [m for m in models if "local" not in m[0]]

    results = {}
    print(f"  {'model':<22}{'log loss':>10}{'Brier':>9}{'acc':>8}{'sec':>7}")
    for name, make in models:
        t0 = time.time()
        per, idx = walk_forward(make, Xtr, mtr, ytr, Xev, mev, yev, years)
        w = (mev[idx] > 0).astype(int)
        p = np.exp(-per) * w + (1 - np.exp(-per)) * (1 - w)  # recover p from loss
        results[name] = per
        print(
            f"  {name:<22}{per.mean():>10.5f}{((p - w) ** 2).mean():>9.5f}"
            f"{((p > 0.5) == (w == 1)).mean() * 100:>7.2f}%{time.time() - t0:>7.0f}"
        )

    rng = np.random.default_rng(0)
    base = results["ridge"]
    print(f"\n  paired bootstrap vs ridge (positive = ridge worse, i.e. the challenger wins):")
    for name, per in results.items():
        if name == "ridge":
            continue
        lo, hi = boot_ci(base, per, rng)
        verdict = "FINDING" if lo > 0 else ("ridge better" if hi < 0 else "not a finding")
        print(f"    {name:<22}{base.mean() - per.mean():>+9.5f}  95% CI [{lo:+.5f}, {hi:+.5f}]  {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
