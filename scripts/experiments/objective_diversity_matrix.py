"""Objective x diversity matrix over the scenario bank.

Answers two product questions left open by FINDINGS.md 6g:

  A. Does scoring-aware diversity fix the champion collapse? Raw Hamming gives
     R64 50.8% of the distance budget for 16.7% of the points, and the champion
     pick 1.6% for 16.7%, so maximising Hamming buys diversity in the cheapest
     games and every returned bracket shared one champion.

  B. Do "highest expected score" and "most likely to win the pool" actually
     select different brackets, or merely score the same bracket differently?

Three diversity methods:
    hamming        raw game-count distance (the 6g baseline)
    weighted       distance weighted by points at stake in each game
    hierarchical   distinct champion first, then distinct Final Four, then
                   weighted distance -- prevents a clever selector satisfying a
                   scalar diversity target while returning five brackets that
                   all have the same title game

Two objectives:
    ev             maximise exact E[bracket score]
    p1             maximise P(1st) against a simulated pool

WHY THE CANDIDATE POOL IS STRATIFIED
------------------------------------
Pre-filtering to the top-EV brackets would make the objective comparison
circular: a low-EV / high-P(1st) bracket could never be found. The pool is
therefore sampled across the whole E[score] range, so the two objectives are
free to disagree.

COMMON RANDOM NUMBERS
---------------------
Every candidate is scored against one shared set of (opponent field, tournament)
draws. That is also what makes P(1st) ranking affordable: the opponent field is
identical across candidates, so the per-trial opponent maximum is computed once
rather than per candidate -- 31x fewer bracket-scorings.

Usage:
    python3 scripts/experiments/objective_diversity_matrix.py --year 2024
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts._common import load_seeds_and_regions  # noqa: E402
from scripts.experiments.conditional_bracket_engine import (  # noqa: E402
    _REACHES,
    at_least_n_double_digit,
    at_least_n_of_seeds,
    expected_scores,
    null_constraint,
    round_marginals,
)
from scripts.mc_pool_backtest import (  # noqa: E402
    ESPN_SCORING,
    _load_torvik_barthag,
    build_bracket_order,
    build_espn_pick_distribution,
    draw_selection_trials,
)
from src.prediction.pairwise import PairwiseProbabilities, simulate_bracket_outcomes  # noqa: E402
from src.prediction.seed_probabilities import build_seed_probabilities  # noqa: E402
from src.simulation.pool_competition import (  # noqa: E402
    build_scoring_vector,
    score_brackets_team_identity,
)


# ---------------------------------------------------------------------------
# P(1st) for a whole pool, under common random numbers
# ---------------------------------------------------------------------------


def pool_p_first(candidates: np.ndarray, trials, first_round) -> np.ndarray:
    """P(1st) for every candidate against a shared trial set.

    The opponent field is identical across candidates by construction, so the
    per-trial opponent maximum is computed once and reused. That turns
    ``P x T x (1 + n_opponents)`` bracket-scorings into ``P x T``.
    """
    n = len(candidates)
    wins = np.zeros(n, dtype=np.int64)
    for opp, sim_winners in trials:
        opp_max = score_brackets_team_identity(opp, sim_winners, first_round, ESPN_SCORING).max()
        cand = score_brackets_team_identity(candidates, sim_winners, first_round, ESPN_SCORING)
        wins += cand >= opp_max
    return wins / max(len(trials), 1)


# ---------------------------------------------------------------------------
# Diversity
# ---------------------------------------------------------------------------


def weighted_distance(a: np.ndarray, bank: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Points-at-stake distance between one bracket and many."""
    return ((bank != a).astype(np.float64) * w).sum(axis=1)


def _greedy_farthest(bank, pool_idx, scores, k, w=None):
    """Max-min distance selection over a quality pool. w=None -> raw Hamming."""
    if len(pool_idx) == 0:
        return []
    order = pool_idx[np.argsort(scores[pool_idx])[::-1]]
    chosen = [int(order[0])]
    if k <= 1:
        return chosen
    weights = np.ones(63) if w is None else w
    pool_bits = bank[order]
    min_d = weighted_distance(bank[chosen[0]], pool_bits, weights)
    for _ in range(k - 1):
        nxt = int(np.argmax(min_d))
        if min_d[nxt] <= 0:
            break
        chosen.append(int(order[nxt]))
        min_d = np.minimum(min_d, weighted_distance(bank[chosen[-1]], pool_bits, weights))
    return chosen


def _greedy_hierarchical(bank, pool_idx, scores, rounds, k, w):
    """Distinct champion first, then distinct Final Four, then weighted distance.

    A scalar diversity target can be satisfied while every bracket keeps the same
    title game. Making composition an explicit, ordered requirement removes that
    freedom instead of hoping the metric discourages it.
    """
    if len(pool_idx) == 0:
        return []
    order = pool_idx[np.argsort(scores[pool_idx])[::-1]]
    chosen: List[int] = []
    used_champs, used_f4 = set(), set()

    def champ_of(i):
        r = rounds[i][_REACHES["CHAMP"]]
        return r[0] if r else None

    def f4_of(i):
        return frozenset(rounds[i][_REACHES["F4"]])

    # Tier 1: one bracket per distinct champion, best-scoring first.
    for i in order:
        if len(chosen) >= k:
            break
        c = champ_of(int(i))
        if c is not None and c not in used_champs:
            chosen.append(int(i))
            used_champs.add(c)
            used_f4.add(f4_of(int(i)))

    # Tier 2: distinct Final Four composition.
    if len(chosen) < k:
        for i in order:
            if len(chosen) >= k:
                break
            f = f4_of(int(i))
            if f not in used_f4:
                chosen.append(int(i))
                used_f4.add(f)

    # Tier 3: weighted distance for anything still missing.
    if len(chosen) < k:
        remaining = [int(i) for i in order if int(i) not in chosen]
        while remaining and len(chosen) < k:
            d = [min(weighted_distance(bank[c], bank[[i]], w)[0] for c in chosen) for i in remaining]
            chosen.append(remaining.pop(int(np.argmax(d))))
    return chosen


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------


def describe(sel, bank, rounds, ev, p1, seeds, w, best_ev, best_p1) -> Dict:
    champs = [rounds[i][_REACHES["CHAMP"]][0] for i in sel if rounds[i][_REACHES["CHAMP"]]]
    f4s = [tuple(sorted(rounds[i][_REACHES["F4"]])) for i in sel]
    pairs = [(a, b) for a in range(len(sel)) for b in range(a + 1, len(sel))]
    wd = [float(weighted_distance(bank[sel[a]], bank[[sel[b]]], w)[0]) for a, b in pairs]
    ham = [int((bank[sel[a]] != bank[sel[b]]).sum()) for a, b in pairs]
    # "Materially distinct" = differs in champion or Final Four composition.
    material = len({(c, f) for c, f in zip(champs, f4s)})
    return {
        "n_returned": len(sel),
        "mean_expected_score": round(float(np.mean([ev[i] for i in sel])), 1),
        "ev_retention": round(float(np.mean([ev[i] for i in sel])) / best_ev, 4) if best_ev else 0,
        "mean_p_first": round(float(np.mean([p1[i] for i in sel])), 4),
        "p1_retention": round(float(np.mean([p1[i] for i in sel])) / best_p1, 4) if best_p1 else 0,
        "distinct_champions": len(set(champs)),
        "champions": [f"{c}({seeds.get(c)})" for c in champs],
        "distinct_final_fours": len(set(f4s)),
        "final_four_seeds": [sorted(seeds.get(t, 0) for t in f) for f in f4s],
        "materially_distinct": material,
        "weighted_distance_mean": round(float(np.mean(wd)), 1) if wd else 0.0,
        "hamming_mean": round(float(np.mean(ham)), 1) if ham else 0.0,
    }


def run(year: int, n_sims: int, pool_size: int, trials_n: int, k: int, seed: int) -> Dict:
    seeds, regions = load_seeds_and_regions(year)
    first_round = build_bracket_order(seeds, regions)
    barthag = _load_torvik_barthag(year, seeds)
    pw = PairwiseProbabilities.from_ratings(barthag, source=f"log5(torvik_{year})")
    w = build_scoring_vector(ESPN_SCORING)

    rng = np.random.default_rng(seed)
    print(f"bank: {n_sims:,} sims ...")
    bank, rounds = simulate_bracket_outcomes(pw, first_round, n_sims, rng, noise_std=0.0)
    ev_all = expected_scores(rounds, round_marginals(rounds), ESPN_SCORING)

    seed_pw = build_seed_probabilities(seeds)
    try:
        pick_dist = build_espn_pick_distribution(year, seeds)
    except Exception:
        pick_dist = {}
    print(f"drawing {trials_n:,} shared trials ...")
    trials = draw_selection_trials(
        trials_n,
        n_opponents=30,
        first_round=first_round,
        pick_dist=pick_dist,
        matchup_probs=seed_pw,
        seeds=seeds,
        rng=np.random.default_rng(seed + 7),
    )

    constraints = [
        null_constraint(),
        at_least_n_of_seeds(2, (2, 3), "F4", seeds),
        at_least_n_double_digit(2, "S16", seeds),
    ]

    out = []
    for con in constraints:
        mask = con.mask(rounds)
        idx = np.flatnonzero(mask)
        if len(idx) < pool_size:
            pool = idx
        else:
            # Stratified across the E[score] range so the objectives can diverge.
            order = idx[np.argsort(ev_all[idx])[::-1]]
            pool = order[np.linspace(0, len(order) - 1, pool_size).astype(int)]

        print(f"  {con.name}: {len(idx):,} surviving, evaluating P(1st) for {len(pool):,} ...")
        p1_pool = pool_p_first(bank[pool], trials, first_round)

        # Local arrays indexed 0..len(pool)-1
        b, r = bank[pool], [rounds[i] for i in pool]
        ev, p1 = ev_all[pool], p1_pool
        best_ev, best_p1 = float(ev.max()), float(p1.max())
        all_idx = np.arange(len(pool))

        entry = {
            "constraint": con.name,
            "n_surviving": int(mask.sum()),
            "pool_size": len(pool),
            "unconstrained_best_ev": round(best_ev, 1),
            "unconstrained_best_p1": round(best_p1, 4),
            "ev_p1_spearman": round(
                float(np.corrcoef(np.argsort(np.argsort(ev)), np.argsort(np.argsort(p1)))[0, 1]), 3
            ),
            "cells": {},
        }
        for obj_name, obj in (("ev", ev), ("p1", p1)):
            for div_name in ("hamming", "weighted", "hierarchical"):
                if div_name == "hamming":
                    sel = _greedy_farthest(b, all_idx, obj, k, None)
                elif div_name == "weighted":
                    sel = _greedy_farthest(b, all_idx, obj, k, w)
                else:
                    sel = _greedy_hierarchical(b, all_idx, obj, r, k, w)
                entry["cells"][f"{obj_name}/{div_name}"] = describe(sel, b, r, ev, p1, seeds, w, best_ev, best_p1)
        out.append(entry)

    return {
        "year": year,
        "n_sims": n_sims,
        "pool_size": pool_size,
        "trials": trials_n,
        "k": k,
        "p1_se_estimate": round(float(np.sqrt(0.035 * 0.965 / trials_n)), 5),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "results": out,
    }


def report(res: Dict) -> None:
    print(f"\n{'=' * 112}")
    print(
        f"OBJECTIVE x DIVERSITY — {res['year']}   bank={res['n_sims']:,}  pool={res['pool_size']}  "
        f"trials={res['trials']:,} (P(1st) SE ~{res['p1_se_estimate'] * 100:.2f}pp)  K={res['k']}"
    )
    print("=" * 112)
    for r in res["results"]:
        print(
            f"\n--- {r['constraint']}   ({r['n_surviving']:,} surviving; "
            f"best EV {r['unconstrained_best_ev']}, best P1 {r['unconstrained_best_p1']}; "
            f"EV-vs-P1 rank corr {r['ev_p1_spearman']}) ---"
        )
        hdr = (
            f"{'objective/diversity':22} {'meanEV':>8} {'EVret':>7} {'P(1st)':>8} {'P1ret':>7} "
            f"{'champs':>7} {'F4s':>5} {'distinct':>9} {'wDist':>7} {'Ham':>6}"
        )
        print(hdr)
        print("-" * len(hdr))
        for cell, v in r["cells"].items():
            print(
                f"{cell:22} {v['mean_expected_score']:8.1f} {v['ev_retention']:7.3f} "
                f"{v['mean_p_first']:8.4f} {v['p1_retention']:7.3f} {v['distinct_champions']:7d} "
                f"{v['distinct_final_fours']:5d} {v['materially_distinct']:9d} "
                f"{v['weighted_distance_mean']:7.0f} {v['hamming_mean']:6.1f}"
            )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--n-sims", type=int, default=60_000)
    ap.add_argument("--pool-size", type=int, default=1200)
    ap.add_argument("--trials", type=int, default=3000)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20260820)
    ap.add_argument("--out", type=str, default="artifacts/objective_diversity")
    args = ap.parse_args()

    res = run(args.year, args.n_sims, args.pool_size, args.trials, args.k, args.seed)
    report(res)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"objective_diversity_{args.year}.json"
    with open(path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
