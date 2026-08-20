"""Conditional bracket engine — smallest thing that answers the product question.

    Given a user preference, can we reliably produce 3-5 materially different,
    high-quality brackets from the existing model distribution?

Context: FINDINGS.md 6e established that production's ~25.5 candidates collapse
to ~5 distinct brackets at mean Hamming 6/63, and 6f established that the model
itself carries a rich distribution (Hamming ~23/63, every simulated tournament a
distinct bracket). The bottleneck is generation, not prediction. This builds the
smallest engine that exploits that.

DESIGN
------
    pairwise probabilities -> simulate N tournaments  (the candidate universe)
                           -> filter by a Constraint
                           -> rank by expected bracket score
                           -> greedily select K *diverse* high scorers

No new optimizer, no search: candidates are simulated tournament rows, which are
path-consistent and internally coherent by construction.

RANKING IS EXACT, NOT SAMPLED
-----------------------------
E[score] of a bracket decomposes by linearity of expectation:

    E[score] = sum_R pts_R * sum_{t in picked_R} P(t wins round R)

with P taken from the UNCONDITIONAL bank — a preference does not change which
tournament actually happens. So ranking needs no evaluation slice at all: it is
O(63) per bracket and carries no Monte Carlo error. This is the same fact that
made ``_make_ev_scorer`` correct during the pairwise-contract audit — marginals
are the right quantity for expected score, they are only wrong as a substitute
for pairwise probabilities.

DIVERSITY SELECTION
-------------------
Returning the top K by score returns K one-pick variations of the same bracket —
exactly the collapse this work exists to avoid. Instead: take the top
``quality_pool`` by score, then greedily pick the bracket maximising the minimum
Hamming distance to those already selected (max-min / farthest-point). Quality is
bounded by pool membership, diversity is maximised within it. Deliberately simple.

READING THE NUMBERS
-------------------
EV retention above 1.0 does NOT mean a preference improves the model. Requiring
a strong team in the Final Four selects scenarios nearer the model's centre of
mass; that is conditioning on an event, not predictive skill. The meaningful
quantity is the **tradeoff**: how much diversity is obtained per unit of expected
utility given up.

Usage:
    python3 scripts/experiments/conditional_bracket_engine.py --year 2024
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts._common import load_seeds_and_regions  # noqa: E402
from scripts.mc_pool_backtest import (  # noqa: E402
    ESPN_SCORING,
    _load_torvik_barthag,
    build_bracket_order,
    build_espn_pick_distribution,
    draw_selection_trials,
    score_candidate_p1,
)
from src.prediction.pairwise import PairwiseProbabilities, simulate_bracket_outcomes  # noqa: E402
from src.prediction.seed_probabilities import build_seed_probabilities  # noqa: E402

ROUND_NAMES = ("R64", "R32", "S16", "E8", "F4", "CHAMP")
# rounds[sim][i] = winners of round i = the teams REACHING round i+1.
_REACHES = {"R32": 0, "S16": 1, "E8": 2, "F4": 3, "FINAL": 4, "CHAMP": 5}


# ---------------------------------------------------------------------------
# Constraint interface
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Constraint:
    """A user preference, expressed as a predicate over one simulated tournament.

    ``predicate`` receives that tournament's per-round winner lists and returns
    True when the tournament satisfies the preference. Everything the product
    needs to express is a composition of counting and membership over those
    lists, which is why this stays a one-method interface.
    """

    name: str
    predicate: Callable[[List[List[str]]], bool]

    def mask(self, rounds: List[List[List[str]]]) -> np.ndarray:
        return np.fromiter((self.predicate(r) for r in rounds), dtype=bool, count=len(rounds))


def null_constraint() -> Constraint:
    return Constraint("null (unconditional)", lambda r: True)


def at_least_n_of_seeds(n: int, seed_set: Sequence[int], round_key: str, seeds: Dict[str, int]) -> Constraint:
    label = "/".join(str(s) for s in seed_set)
    return Constraint(
        f">={n} {label}-seed in {round_key}",
        lambda r: sum(1 for t in r[_REACHES[round_key]] if seeds.get(t) in seed_set) >= n,
    )


def at_least_n_double_digit(n: int, round_key: str, seeds: Dict[str, int]) -> Constraint:
    return Constraint(
        f">={n} double-digit seed in {round_key}",
        lambda r: sum(1 for t in r[_REACHES[round_key]] if seeds.get(t, 0) >= 10) >= n,
    )


def exactly_n_of_seed(n: int, seed: int, round_key: str, seeds: Dict[str, int]) -> Constraint:
    return Constraint(
        f"exactly {n} {seed}-seeds in {round_key}",
        lambda r: sum(1 for t in r[_REACHES[round_key]] if seeds.get(t) == seed) == n,
    )


def team_reaches(team: str, round_key: str) -> Constraint:
    return Constraint(f"{team} reaches {round_key}", lambda r: team in r[_REACHES[round_key]])


def champion_is(team: str) -> Constraint:
    return Constraint(
        f"champion = {team}",
        lambda r: bool(r[_REACHES["CHAMP"]]) and r[_REACHES["CHAMP"]][0] == team,
    )


# ---------------------------------------------------------------------------
# Exact expected score
# ---------------------------------------------------------------------------


def round_marginals(rounds: List[List[List[str]]]) -> List[Dict[str, float]]:
    """P(team wins round R) for each round, from the UNCONDITIONAL bank.

    The bank is the model's view of reality; a user's preference does not change
    it. These marginals stay fixed across every constraint, which is what makes
    conditional brackets comparable to unconditional ones.
    """
    n = len(rounds)
    out = []
    for ri in range(6):
        c = Counter()
        for r in rounds:
            c.update(r[ri])
        out.append({t: v / n for t, v in c.items()})
    return out


def expected_scores(rounds: List[List[List[str]]], marginals, scoring: Dict[str, int]) -> np.ndarray:
    """Exact E[ESPN score] per candidate: sum_R pts_R * sum_{t in picked_R} P(t wins R)."""
    pts = [float(scoring[r]) for r in ROUND_NAMES]
    out = np.empty(len(rounds), dtype=np.float64)
    for i, r in enumerate(rounds):
        total = 0.0
        for ri in range(6):
            m = marginals[ri]
            total += pts[ri] * sum(m.get(t, 0.0) for t in r[ri])
        out[i] = total
    return out


# ---------------------------------------------------------------------------
# Diverse selection
# ---------------------------------------------------------------------------


def select_diverse(
    bank: np.ndarray, scores: np.ndarray, k: int, quality_pool: int = 2000
) -> List[int]:
    """Top-scoring but mutually distinct brackets (greedy farthest-point).

    Plain top-K returns K one-pick variations of the same bracket. Instead:
    restrict to the ``quality_pool`` highest scorers, seed with the best, then
    repeatedly add whichever candidate maximises the MINIMUM Hamming distance to
    everything already chosen. Quality is bounded by pool membership; diversity
    is maximised inside it.

    Returns indices into the arrays passed in.
    """
    if len(scores) == 0:
        return []
    pool = np.argsort(scores)[::-1][: min(quality_pool, len(scores))]
    chosen = [int(pool[0])]
    if k <= 1:
        return chosen

    pool_bits = bank[pool]
    min_dist = (pool_bits != bank[chosen[0]]).sum(axis=1).astype(np.int32)
    for _ in range(k - 1):
        nxt = int(np.argmax(min_dist))
        if min_dist[nxt] == 0:
            break  # pool exhausted of anything distinct
        chosen.append(int(pool[nxt]))
        d = (pool_bits != bank[chosen[-1]]).sum(axis=1).astype(np.int32)
        min_dist = np.minimum(min_dist, d)
    return chosen


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------


def evaluate(
    constraint: Constraint,
    bank: np.ndarray,
    rounds: List,
    scores: np.ndarray,
    k: int,
    p1_trials: List,
    first_round: List[str],
    seeds: Dict[str, int],
    baseline_ev: Optional[float],
) -> Dict:
    mask = constraint.mask(rounds)
    n_surv = int(mask.sum())
    res = {"constraint": constraint.name, "p_constraint": round(n_surv / len(rounds), 5), "n_surviving": n_surv}
    if n_surv < k * 2:
        res["note"] = "infeasible at this bank size"
        return res

    idx = np.flatnonzero(mask)
    sub_bank, sub_scores = bank[idx], scores[idx]
    sub_rounds = [rounds[i] for i in idx]

    picked = select_diverse(sub_bank, sub_scores, k)
    sel_bank = sub_bank[picked]
    sel_rounds = [sub_rounds[i] for i in picked]
    sel_scores = sub_scores[picked]

    # Pairwise Hamming among the RETURNED brackets — the product-facing number.
    d = [
        int((sel_bank[a] != sel_bank[b]).sum())
        for a in range(len(sel_bank))
        for b in range(a + 1, len(sel_bank))
    ]
    # P(1st) against a shared opponent field (common random numbers).
    p1 = [round(score_candidate_p1(v, p1_trials, first_round, ESPN_SCORING), 4) for v in sel_bank]

    res.update(
        {
            "best_ev_in_pool": round(float(sub_scores.max()), 1),
            "returned": [
                {
                    "expected_score": round(float(s), 1),
                    "p_first": p,
                    "champion": r[_REACHES["CHAMP"]][0] if r[_REACHES["CHAMP"]] else None,
                    "champion_seed": seeds.get(r[_REACHES["CHAMP"]][0]) if r[_REACHES["CHAMP"]] else None,
                    "final_four_seeds": sorted(seeds.get(t, 0) for t in r[_REACHES["F4"]]),
                    "dd_seeds_in_s16": sum(1 for t in r[_REACHES["S16"]] if seeds.get(t, 0) >= 10),
                }
                for s, p, r in zip(sel_scores, p1, sel_rounds)
            ],
            "returned_hamming": {
                "mean": round(float(np.mean(d)), 1) if d else 0.0,
                "min": int(np.min(d)) if d else 0,
                "max": int(np.max(d)) if d else 0,
            },
            "mean_expected_score": round(float(sel_scores.mean()), 1),
            "mean_p_first": round(float(np.mean(p1)), 4),
            "distinct_champions_returned": len({r[_REACHES["CHAMP"]][0] for r in sel_rounds if r[_REACHES["CHAMP"]]}),
        }
    )
    if baseline_ev:
        res["ev_retention"] = round(res["mean_expected_score"] / baseline_ev, 4)
    return res


def run(year: int, n_sims: int, k: int, noise_std: float, seed: int, p1_trials_n: int) -> Dict:
    seeds, regions = load_seeds_and_regions(year)
    first_round = build_bracket_order(seeds, regions)
    barthag = _load_torvik_barthag(year, seeds)
    pw = PairwiseProbabilities.from_ratings(barthag, source=f"log5(torvik_{year})")

    rng = np.random.default_rng(seed)
    bank, rounds = simulate_bracket_outcomes(pw, first_round, n_sims, rng, noise_std=noise_std)
    marg = round_marginals(rounds)
    scores = expected_scores(rounds, marg, ESPN_SCORING)

    # P(1st) uses production's evaluator setup (seed-based opponents and
    # tournaments) so the numbers are comparable to the documented baselines,
    # and ONE shared trial set so every bracket faces identical draws.
    seed_pw = build_seed_probabilities(seeds)
    try:
        pick_dist = build_espn_pick_distribution(year, seeds)
    except Exception:
        pick_dist = {}
    p1_trials = draw_selection_trials(
        p1_trials_n,
        n_opponents=30,
        first_round=first_round,
        pick_dist=pick_dist,
        matchup_probs=seed_pw,
        seeds=seeds,
        rng=np.random.default_rng(seed + 7),
    )

    top = max(barthag, key=barthag.get)
    constraints = [
        null_constraint(),
        at_least_n_of_seeds(1, (2, 3), "F4", seeds),
        at_least_n_of_seeds(2, (2, 3), "F4", seeds),
        at_least_n_double_digit(1, "S16", seeds),
        at_least_n_double_digit(2, "S16", seeds),
        exactly_n_of_seed(2, 1, "F4", seeds),
        team_reaches(top, "F4"),
        champion_is(top),
    ]

    baseline = evaluate(constraints[0], bank, rounds, scores, k, p1_trials, first_round, seeds, None)
    base_ev = baseline["mean_expected_score"]
    baseline["ev_retention"] = 1.0

    out = [baseline]
    for c in constraints[1:]:
        out.append(evaluate(c, bank, rounds, scores, k, p1_trials, first_round, seeds, base_ev))

    return {
        "year": year,
        "n_sims": n_sims,
        "noise_std": noise_std,
        "k_returned": k,
        "source": pw.source,
        "unconditional_mean_expected_score": base_ev,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "results": out,
    }


def report(res: Dict) -> None:
    print(f"\n{'=' * 104}")
    print(f"CONDITIONAL BRACKET ENGINE — {res['year']}   noise_std={res['noise_std']}   "
          f"bank={res['n_sims']:,}   returning K={res['k_returned']}")
    print("=" * 104)
    hdr = (f"{'constraint':34} {'P(cond)':>8} {'surviving':>10} {'meanEV':>8} {'EVret':>7} "
           f"{'P(1st)':>7} {'Hamming(min/mean)':>18} {'champs':>7}")
    print(hdr)
    print("-" * len(hdr))
    for r in res["results"]:
        if "returned" not in r:
            print(f"{r['constraint']:34} {r['p_constraint']:8.4f} {r['n_surviving']:10,}   {r.get('note','')}")
            continue
        h = r["returned_hamming"]
        print(f"{r['constraint']:34} {r['p_constraint']:8.4f} {r['n_surviving']:10,} "
              f"{r['mean_expected_score']:8.1f} {r['ev_retention']:7.3f} {r['mean_p_first']:7.4f} "
              f"{str(h['min']) + '/' + str(h['mean']):>18} {r['distinct_champions_returned']:7d}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--n-sims", type=int, default=100_000)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--p1-trials", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20260820)
    ap.add_argument("--out", type=str, default="artifacts/conditional_engine")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for noise in (0.0, 0.16):
        res = run(args.year, args.n_sims, args.k, noise, args.seed, args.p1_trials)
        report(res)
        path = out_dir / f"conditional_{args.year}_noise{noise}.json"
        with open(path, "w") as f:
            json.dump(res, f, indent=2)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
