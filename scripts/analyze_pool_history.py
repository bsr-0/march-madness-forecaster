#!/usr/bin/env python3
"""Is this pool different from the ESPN national field, and how does it score?

WHY THIS EXISTS. Pool edge is the gap between two beliefs:

    leverage = P_outcome(team wins) - P_public(team picked)

Almost all modelling effort in this repo has gone into the first term. The
second is supplied by ESPN's national pick distribution, which is an assumption
about who you are playing against, not a measurement of it. pool_hist_results.json
holds 105 real brackets from the actual pool across 2023-2026, so the assumption
can be checked rather than trusted.

THE ANSWER IS THAT THE ASSUMPTION IS FINE, which is worth recording precisely
because the opposite conclusion is so tempting. Raw champion shares look wildly
divergent -- St. John's at 21.9% locally against 4.8% nationally in 2025,
Arizona 36.7% against 21.9% in 2026 -- and it is easy to read those as an
exploitable local bias. They are not. A 25-32 person pool drawn from the
national distribution produces gaps that size routinely, and the Monte-Carlo
test below cannot distinguish this pool from ESPN in any year (p = 0.97, 0.95,
0.81). The aggregate chalk rate agrees to within half a percentage point.

So: do not re-open the opponent model expecting to find a local tendency. It
was looked for, with the data, and it is not there. Fitting the chalk-bias
exponent to this pool is likewise unjustified.

WHAT THE SCORING CHECK IS FOR. Every expected-points number in the product
assumes ESPN's 10/20/40/80/160/320 with no upset bonus. ESPN does offer an
upset bonus in some contexts, and if this pool used one, the "maximise expected
points" strategy would be optimising the wrong function entirely. Recomputing
real brackets from their picks and comparing to their reported scores settles
it against this pool rather than against documentation: 2026 reproduces exactly
for 28 of 30 brackets, and an upset bonus large enough to matter would fail far
more loudly than that.

NOTE THAT THE POOL IS WINNER-TAKE-ALL, so expected points is the wrong
objective for it regardless. Second place pays what thirtieth pays. The scoring
check matters anyway, because the ev figures are shown in the UI and would be
wrong if the rule were wrong.

A KNOWN DISCREPANCY IN 2024-2025, recorded rather than smoothed over. 2026
recomputes to the point for 28 of 30 brackets (mean error -0.3). 2024 and 2025
manage only 4 of 25 and 3 of 32, and the errors are SYSTEMATICALLY POSITIVE:
mean +37.6 and +31.6 against totals of ~730 and ~1,024, with 19 of 25 and 27 of
32 brackets over-credited and only two negative in each season.

The sign is the informative part. Recomputing HIGH means ESPN awarded fewer
points than plain scoring implies for those picks. An upset bonus -- the rule
most likely to be missing here -- would push ESPN's reported total the other
way, so the direction rules it out rather than merely failing to detect it.

It is not the actual results (checked against the real Final Fours) and not
name resolution (1 unresolved pick in 1,572 for 2024, none at all for 2025 or
2026). The leading explanation is the scrape rather than the rules:
pool_hist_results.json was captured on 2026-04-12, long after those seasons
finished, and a bracket view rendered post-hoc can show advanced or actual
teams in place of a user's eliminated picks. That would inflate a
recomputation in exactly one direction and would not affect 2026, which was
scraped close to live. Unconfirmed, so treat 2024-2025 pool SCORES as
approximate. The PICKS still reproduce and resolve cleanly, so the
pick-distribution results above are unaffected.

TWO EARLIER READINGS OF THIS WERE WRONG, both from looking at too little. The
first inspected only the top six brackets per season; the second quoted the
min/max range and called the residuals bidirectional without looking at their
distribution. Sampling the leaderboard is not sampling the pool, and a range is
not a distribution.

Run: python3 scripts/analyze_pool_history.py
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.experiments.build_candidate_artifact import (  # noqa: E402
    build_espn_pick_distribution,
)
from scripts.mc_pool_backtest import (  # noqa: E402
    build_bracket_order,
    load_seeds_and_regions,
)
from src.simulation.pool_history_opponent_model import resolve_abbrev  # noqa: E402

POOL_HIST = REPO / "pool_hist_results.json"
PAYLOADS = REPO / "docs" / "data"

# The scoring the product assumes. Checked, not trusted -- see module docstring.
ESPN_SCORING = {"r64": 10, "r32": 20, "s16": 40, "e8": 80, "f4": 160, "champ": 320}
ROUND_KEYS = ("r64", "r32", "s16", "e8", "f4", "champ")


def champion_distribution_test(year, brackets, seeds, rng):
    """Monte-Carlo exact test: are these champion picks drawn from ESPN's?

    Uses the multinomial log-likelihood as the statistic rather than chi-square,
    because most teams are picked by nobody and chi-square's large-sample
    approximation is not usable on a 30-bracket pool with 64 cells.
    """
    espn = build_espn_pick_distribution(year, seeds)
    obs = collections.Counter(resolve_abbrev(b["champ"], seeds) for b in brackets)
    teams = [t for t in seeds if espn.get(t, {}).get("CHAMP", 0) > 0 or obs.get(t, 0) > 0]
    if not teams:
        return None
    p = np.array([max(espn.get(t, {}).get("CHAMP", 0), 1e-4) for t in teams])
    p = p / p.sum()
    o = np.array([obs.get(t, 0) for t in teams])

    def loglik(counts):
        return float((counts * np.log(p)).sum())

    sims = rng.multinomial(len(brackets), p, size=20000)
    return float((np.array([loglik(s) for s in sims]) <= loglik(o)).mean())


def chalk_rate(brackets, seeds):
    """Share of R64 picks that take the better seed."""
    fav = tot = 0
    for b in brackets:
        for ab in b["r64"]:
            tid = resolve_abbrev(ab, seeds)
            s = seeds.get(tid) if tid else None
            if s:
                tot += 1
                fav += 1 if s <= 8 else 0
    return 100 * fav / tot if tot else float("nan")


def espn_chalk_rate(year, seeds, regions):
    """The same statistic for ESPN's national field, game by game."""
    espn = build_espn_pick_distribution(year, seeds)
    order = build_bracket_order(seeds, regions)
    shares = []
    for g in range(0, 64, 2):
        a, b = order[g], order[g + 1]
        if a.startswith("unknown_") or b.startswith("unknown_"):
            continue
        lower = a if seeds.get(a, 99) <= seeds.get(b, 99) else b
        p = espn.get(lower, {}).get("R64")
        if p is not None:
            shares.append(p)
    return 100 * float(np.mean(shares)) if shares else float("nan")


def verify_scoring(year, brackets):
    """Recompute reported scores from picks under plain ESPN scoring."""
    path = PAYLOADS / f"season_{year}.json"
    if not path.exists():
        return None
    season = json.loads(path.read_text())
    if not season.get("actual"):
        return None
    team_ids = [t["id"] for t in season["teams"]]
    won = [set(team_ids[i] for i in r) for r in season["actual"]]
    seeds, _ = load_seeds_and_regions(year)

    exact = 0
    deltas = []
    for b in brackets:
        total = 0
        for idx, key in enumerate(ROUND_KEYS):
            picks = b[key] if isinstance(b[key], list) else [b[key]]
            for ab in picks:
                tid = resolve_abbrev(ab, seeds)
                if tid and tid in won[idx]:
                    total += ESPN_SCORING[key]
        deltas.append(total - b["pts"])
        exact += total == b["pts"]
    return exact, len(brackets), deltas


def main() -> int:
    hist = json.loads(POOL_HIST.read_text())["years"]
    rng = np.random.default_rng(0)
    years = sorted(hist)

    print(f"\n  pool history: {sum(len(hist[y]['brackets']) for y in years)} brackets over {len(years)} seasons\n")

    print(
        f"  {'year':<7}{'entrants':>9}{'winner pts':>12}{'chalk R64':>11}"
        f"{'ESPN chalk':>12}{'gap':>8}{'champ dist p':>14}"
    )
    for y in years:
        entry = hist[y]
        brackets = entry["brackets"]
        seeds, regions = load_seeds_and_regions(int(y))
        mine = chalk_rate(brackets, seeds)
        try:
            theirs = espn_chalk_rate(int(y), seeds, regions)
        except Exception:
            theirs = float("nan")
        try:
            pval = champion_distribution_test(int(y), brackets, seeds, rng)
        except Exception:
            pval = None
        best = max(b["pts"] for b in brackets)
        pv = f"{pval:.3f}" if pval is not None else "n/a"
        print(
            f"  {y:<7}{entry.get('groupSize', len(brackets)):>9}{best:>12}"
            f"{mine:>10.1f}%{theirs:>11.1f}%{mine - theirs:>+8.1f}{pv:>14}"
        )

    print("\n  champion-distribution p is a Monte-Carlo test against ESPN's national")
    print("  picks. High p = indistinguishable. Every season is indistinguishable,")
    print("  so the national distribution is a fair opponent model for this pool.")

    print("\n  scoring check (plain ESPN 10/20/40/80/160/320, no upset bonus):")
    for y in years:
        res = verify_scoring(int(y), hist[y]["brackets"])
        if res is None:
            print(f"    {y}: no actual results in the payload; skipped")
            continue
        exact, n, deltas = res
        arr = np.array(deltas)
        # Report the mean and the sign split, not the range. The range alone
        # once led to calling these residuals bidirectional when they are
        # strongly one-sided, which inverted the diagnosis.
        note = ""
        if exact != n:
            note = f"   mean {arr.mean():+.1f}  ({(arr > 0).sum()} over / {(arr < 0).sum()} under)"
        print(f"    {y}: {exact}/{n} brackets reproduce exactly{note}")
    print("\n  A pool using an upset bonus would report MORE than this recomputes,")
    print("  i.e. negative residuals. 2024-2025 are systematically POSITIVE, so no")
    print("  bonus is in play; see the docstring for what those residuals likely are.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
