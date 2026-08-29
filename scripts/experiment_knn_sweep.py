#!/usr/bin/env python3
"""Sweep k for kNN, and try to rescue local linear regression.

TWO QUESTIONS THE FAMILY COMPARISON LEFT OPEN.

k was spot-checked at 25, 100 and 500, which is enough to see kNN lose but not
enough to say it was given its best shot. The optimum could sit anywhere,
and on 1,008 tournament rows a large k is a very different model from a small
one -- at k = 2,000 (both orientations of every row) kNN degenerates to the
global mean.

Local linear scored 0.777 on the tournament matrix and 0.852 on the
regular-season one. Both are worse than always saying 50-50 (log loss ln 2 =
0.693), which is not a result about local methods, it is a broken
specification. The suspect is dimensionality: fitting 28+ parameters on 200
tricube-weighted neighbours with a 1e-3 ridge is near-unregularised, so the
local fits have enormous variance and extrapolate wildly at the query point.
Three fixes are tried here -- more neighbours, a stronger local ridge, and
fewer dimensions -- and if none rescues it, that is worth knowing plainly
rather than leaving a broken model in the table.

The stable-core subset comes from the nested selection run: the six features
chosen in every single fold.

Run: python3 scripts/experiment_knn_sweep.py [--matrix tournament|regular]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.experiment_model_families import (  # noqa: E402
    MIN_TEST_YEAR,
    boot_ci,
    fit_knn,
    fit_local_linear,
    fit_ridge,
    load,
    walk_forward,
)

# Chosen in all nine folds of the nested selection run.
STABLE_CORE = (
    "t_rank",
    "barthag",
    "srs",
    "adj_defensive_efficiency",
    "turnover_rate",
    "n_returning_players",
)


def local_linear_ridged(k, alpha):
    """Local linear with a tunable ridge on the local fit."""

    def make(X, m):
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
                w = (1 - (d / h) ** 3) ** 3
                Xi, yi = XX[idx[i]], mm[idx[i]]
                A = np.column_stack([np.ones(len(Xi)), Xi]) * np.sqrt(w)[:, None]
                G = A.T @ A + alpha * np.eye(A.shape[1])
                beta = np.linalg.solve(G, A.T @ (yi * np.sqrt(w)))
                out[i] = np.concatenate([[1.0], Q[i]]) @ beta
            return out

        return predict

    return make


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matrix", choices=["regular", "tournament"], default="tournament")
    args = ap.parse_args()

    keys, Xtr, mtr, ytr, Xev, mev, yev = load(args.matrix)
    years = [y for y in sorted(set(yev.tolist())) if y >= MIN_TEST_YEAR]
    core = [i for i, k in enumerate(keys) if k in STABLE_CORE]

    print(f"\nmatrix={args.matrix}  {Xtr.shape[0]:,} train rows x {Xtr.shape[1]} features")
    print(f"coin flip (always 0.5) = {np.log(2):.5f}\n")

    base, _ = walk_forward(fit_ridge, Xtr, mtr, ytr, Xev, mev, yev, years)
    print(f"  {'model':<34}{'log loss':>10}{'vs ridge':>11}")
    print(f"  {'ridge (all features)':<34}{base.mean():>10.5f}{'--':>11}")

    rng = np.random.default_rng(0)
    results = {}

    print("\n  kNN, k swept:")
    grid = [3, 5, 10, 25, 50, 100, 200, 400, 800, 1600]
    grid = [k for k in grid if k <= 2 * Xtr.shape[0]]
    for k in grid:
        per, _ = walk_forward(lambda X, m, k=k: fit_knn(X, m, k), Xtr, mtr, ytr, Xev, mev, yev, years)
        results[f"knn k={k}"] = per
        print(f"  {'  k=' + str(k):<34}{per.mean():>10.5f}{base.mean() - per.mean():>+11.5f}")

    print("\n  local linear, rescue attempts:")
    attempts = [
        ("k=200 alpha=1e-3 (original)", local_linear_ridged(200, 1e-3), None),
        ("k=800 alpha=1.0", local_linear_ridged(800, 1.0), None),
        ("k=800 alpha=50", local_linear_ridged(800, 50.0), None),
        ("k=1600 alpha=50", local_linear_ridged(1600, 50.0), None),
        ("k=800 alpha=50, core features", local_linear_ridged(800, 50.0), core),
    ]
    for name, make, cols in attempts:
        if cols is not None and not cols:
            continue
        A, E = (Xtr[:, cols], Xev[:, cols]) if cols else (Xtr, Xev)
        per, _ = walk_forward(make, A, mtr, ytr, E, mev, yev, years)
        results[f"local {name}"] = per
        print(f"  {'  ' + name:<34}{per.mean():>10.5f}{base.mean() - per.mean():>+11.5f}")

    if core:
        per, _ = walk_forward(fit_ridge, Xtr[:, core], mtr, ytr, Xev[:, core], mev, yev, years)
        results["ridge core"] = per
        lo, hi = boot_ci(base, per, rng)
        print(
            f"\n  ridge on the 6 stable-core features only: {per.mean():.5f}"
            f"   diff {base.mean() - per.mean():+.5f}  95% CI [{lo:+.5f}, {hi:+.5f}]"
            f"  {'FINDING' if lo > 0 else 'not a finding'}"
        )

    best = min(results, key=lambda k: results[k].mean())
    lo, hi = boot_ci(base, results[best], rng)
    print(f"\n  best challenger: {best} at {results[best].mean():.5f}")
    print(
        f"  vs ridge {base.mean() - results[best].mean():+.5f}  "
        f"95% CI [{lo:+.5f}, {hi:+.5f}]  "
        f"{'FINDING' if lo > 0 else 'not a finding'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
