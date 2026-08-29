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
    MIN_TEST_YEAR,
    VENUE_KEYS,
    boot_ci,
    fit_ridge,
    load,
    walk_forward,
)


def score(cols, Xtr, mtr, ytr, Xev, mev, yev, years):
    """Held-out per-game log loss for one column subset.

    Delegates to the shared two-pass walk_forward so selection is scored by
    exactly the evaluator the models are reported on -- including its fix for
    calibrating on rows the model was trained on, which mattered most for the
    flexible families but applies to every subset scored here.
    """
    per, _ = walk_forward(
        lambda X, m: fit_ridge(X, m),
        Xtr[:, cols], mtr, ytr, Xev[:, cols], mev, yev, years,
    )
    return per


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matrix", choices=["regular", "tournament"], default="regular")
    ap.add_argument("--nested", action="store_true",
                    help="re-run selection inside each fold; the honest version")
    args = ap.parse_args()
    keys, Xtr, mtr, ytr, Xev, mev, yev = load(args.matrix)
    years = [y for y in sorted(set(yev.tolist())) if y >= MIN_TEST_YEAR]

    venue = [i for i, k in enumerate(keys) if k in VENUE_KEYS]
    pool = [i for i, k in enumerate(keys) if k not in VENUE_KEYS]

    print(
        f"\n{Xtr.shape[0]:,} train rows | {len(mev)} test games | "
        f"{len(pool)} selectable features + {len(venue)} venue (always in)"
    )
    print("greedy forward selection on walk-forward log loss\n")

    if args.nested:
        pa, ps, picks = nested(keys, Xtr, mtr, ytr, Xev, mev, yev, years, venue, pool)
        rng = np.random.default_rng(0)
        lo, hi = boot_ci(pa, ps, rng)
        print("  per-season selections (differences between them are the tell):")
        for y in sorted(picks):
            print(f"    {y}: {len(picks[y]):>2}  {', '.join(picks[y][:6])}"
                  f"{' ...' if len(picks[y]) > 6 else ''}")
        counts = {}
        for v in picks.values():
            for k in v:
                counts[k] = counts.get(k, 0) + 1
        print(f"\n  chosen in every fold: "
              f"{[k for k, c in counts.items() if c == len(picks)] or 'none'}")
        print(f"\n  all features      {pa.mean():.5f}")
        print(f"  nested selection  {ps.mean():.5f}")
        print(f"  difference        {pa.mean() - ps.mean():+.5f}  "
              f"95% CI [{lo:+.5f}, {hi:+.5f}]  "
              f"{'FINDING' if lo > 0 else 'not a finding'}")
        return 0

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


def nested(keys, Xtr, mtr, ytr, Xev, mev, yev, years, venue, pool):
    """Selection re-run inside each fold, so no season sees its own selection.

    WHY THE FLAT VERSION IS NOT ENOUGH. Selecting features by walk-forward score
    over every test season and then reporting that score is circular: the subset
    was chosen because it happened to suit those seasons. A paired bootstrap
    does not rescue it, because the bias is in which columns were picked, not in
    sampling noise -- so the naive run prints "FINDING" for a comparison that
    cannot support one.

    Here, for test season Y the selection runs on seasons strictly before Y and
    is judged on Y alone. The chosen subset differs year to year, which is
    itself the useful output: a set that keeps changing is a set that was fitting
    the fold.
    """
    wev = (mev > 0).astype(int)
    per_all, per_sel, picks = [], [], {}
    for y in years:
        inner = [t for t in years if t < y]
        if len(inner) < 3:
            continue
        chosen, best = [], None
        remaining = list(pool)
        while remaining:
            cand = []
            for j in remaining:
                p = score(venue + chosen + [j], Xtr, mtr, ytr, Xev, mev, yev, inner)
                cand.append((p.mean(), j))
            cand.sort()
            val, j = cand[0]
            if best is not None and val >= best:
                break
            best = val
            chosen.append(j)
            remaining.remove(j)
        picks[y] = [keys[i] for i in chosen]
        per_sel.append(score(venue + chosen, Xtr, mtr, ytr, Xev, mev, yev, [y]))
        per_all.append(score(venue + pool, Xtr, mtr, ytr, Xev, mev, yev, [y]))
    return np.concatenate(per_all), np.concatenate(per_sel), picks


if __name__ == "__main__":
    raise SystemExit(main())
