"""Scenario-bank experiment: is there enough bracket diversity to build on?

THE QUESTION
------------
Production generates ~25.5 candidate brackets per year but they collapse to ~5
distinct ones with a mean pairwise Hamming distance of 6 games out of 63. The
optimizer is choosing between near-duplicates, which is why selection precision
turned out not to be the binding constraint (FINDINGS.md 6e).

This asks whether the *probability model* is the limitation or the *candidate
generation machinery* is:

    Does the existing pairwise model contain a rich enough distribution of
    plausible, materially different brackets to support a preference-driven
    product, once we stop collapsing candidates into near-duplicates?

METHOD
------
Every row of a Monte Carlo tournament simulation is already a complete,
path-consistent, internally coherent bracket drawn from the model. So instead of
constructing candidates with a greedy optimizer, use the simulated tournaments
themselves. No constrained optimizer is built here -- that is deliberately the
next step, gated on this result.

    pairwise probabilities -> simulate N tournaments -> that IS the candidate bank
                           -> filter by a preference predicate
                           -> measure what survives

WHAT IS NOT DONE HERE
---------------------
No UI changes, no ML features, no model tuning. The pairwise/simulation
machinery is used as-is and is the source of truth throughout.

COMMON RANDOM NUMBERS
---------------------
Every condition is evaluated against the SAME bank and the SAME held-out
evaluation slice. Conditions are therefore compared on identical draws, so
differences between them reflect the conditions rather than sampling noise --
the same discipline applied to candidate selection in FINDINGS.md 6e.

Usage:
    python3 scripts/experiments/scenario_bank_diversity.py --year 2024 --n-sims 200000
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts._common import load_seeds_and_regions  # noqa: E402
from scripts.mc_pool_backtest import (  # noqa: E402
    ESPN_SCORING,
    _load_torvik_barthag,
    build_bracket_order,
)
from src.prediction.pairwise import (  # noqa: E402
    PairwiseProbabilities,
    simulate_bracket_outcomes,
)
from src.simulation.pool_competition import score_brackets_team_identity  # noqa: E402

ROUND_NAMES = ("R64", "R32", "S16", "E8", "F4", "CHAMP")
# outcomes_by_round[sim][i] holds the winners of round i, i.e. the teams that
# REACH round i+1. Index 3 therefore holds the Final Four.
_REACHES = {"R32": 0, "S16": 1, "E8": 2, "F4": 3, "FINAL": 4, "CHAMP": 5}


# ---------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------


def make_conditions(seeds: Dict[str, int], named_team: str) -> List[Tuple[str, Callable]]:
    """Preference predicates over one simulated tournament's per-round winners.

    Each takes ``rounds`` (the per-round winner lists for a single simulation)
    and returns True when that tournament satisfies the preference.
    """

    def reaching(rounds, key):
        return rounds[_REACHES[key]]

    def null_control(rounds):
        return True

    def one_23_seed_f4(rounds):
        return sum(1 for t in reaching(rounds, "F4") if seeds.get(t) in (2, 3)) >= 1

    def two_23_seeds_f4(rounds):
        return sum(1 for t in reaching(rounds, "F4") if seeds.get(t) in (2, 3)) >= 2

    def dd_seed_s16(rounds):
        return any(seeds.get(t, 0) >= 10 for t in reaching(rounds, "S16"))

    def two_dd_seeds_s16(rounds):
        return sum(1 for t in reaching(rounds, "S16") if seeds.get(t, 0) >= 10) >= 2

    def named_in_f4(rounds):
        return named_team in reaching(rounds, "F4")

    def exactly_two_1seeds_f4(rounds):
        return sum(1 for t in reaching(rounds, "F4") if seeds.get(t) == 1) == 2

    return [
        ("null (unconditional control)", null_control),
        (">=1 2/3-seed in Final Four", one_23_seed_f4),
        (">=2 2/3-seeds in Final Four", two_23_seeds_f4),
        (">=1 double-digit seed in S16", dd_seed_s16),
        (">=2 double-digit seeds in S16", two_dd_seeds_s16),
        ("exactly two 1-seeds in Final Four", exactly_two_1seeds_f4),
        (f"{named_team} reaches Final Four", named_in_f4),
    ]


# ---------------------------------------------------------------------------
# Diversity metrics
# ---------------------------------------------------------------------------


def hamming_stats(bank: np.ndarray, rng: np.random.Generator, n_pairs: int = 20000) -> Dict:
    """Pairwise Hamming distance over a random sample of bracket pairs.

    Sampled rather than exhaustive: a 200k bank has 2e10 pairs. The comparison
    that matters is against production's mean of 6.0 / 63.
    """
    n = len(bank)
    if n < 2:
        return {"mean": 0.0, "p05": 0.0, "p50": 0.0, "p95": 0.0}
    i = rng.integers(0, n, size=min(n_pairs, n * 4))
    j = rng.integers(0, n, size=len(i))
    keep = i != j
    i, j = i[keep], j[keep]
    d = (bank[i] != bank[j]).sum(axis=1)
    return {
        "mean": float(d.mean()),
        "p05": float(np.percentile(d, 5)),
        "p50": float(np.percentile(d, 50)),
        "p95": float(np.percentile(d, 95)),
    }


def seed_profile(rounds_list: List, seeds: Dict[str, int]) -> Dict[str, float]:
    """Mean seed of the teams reaching each round, across the surviving set."""
    out = {}
    for key in ("S16", "E8", "F4", "CHAMP"):
        idx = _REACHES[key]
        vals = [seeds.get(t, 0) for r in rounds_list for t in r[idx]]
        out[key] = round(float(np.mean(vals)), 2) if vals else 0.0
    return out


def champion_entropy(rounds_list: List) -> Tuple[float, int, str, float]:
    """Shannon entropy (bits) of the champion distribution, plus the modal champ."""
    champs = Counter(r[_REACHES["CHAMP"]][0] for r in rounds_list if r[_REACHES["CHAMP"]])
    total = sum(champs.values())
    if not total:
        return 0.0, 0, "", 0.0
    p = np.array([c / total for c in champs.values()])
    ent = float(-(p * np.log2(p)).sum())
    top, top_n = champs.most_common(1)[0]
    return ent, len(champs), top, top_n / total


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------


def run(year: int, n_sims: int, eval_slice: int, seed: int) -> Dict:
    seeds, regions = load_seeds_and_regions(year)
    first_round = build_bracket_order(seeds, regions)
    barthag = _load_torvik_barthag(year, seeds)
    pw = PairwiseProbabilities.from_ratings(barthag, source=f"log5(torvik_{year})")

    rng = np.random.default_rng(seed)
    print(f"Simulating {n_sims:,} tournaments for {year} from {pw.source} ...")
    bank, rounds = simulate_bracket_outcomes(pw, first_round, n_sims, rng)

    # COMMON RANDOM NUMBERS: one held-out evaluation slice, shared by every
    # condition. Conditions are scored against identical tournaments.
    eval_idx = np.arange(len(bank) - eval_slice, len(bank))
    eval_rounds = [rounds[i] for i in eval_idx]
    cand_idx = np.arange(0, len(bank) - eval_slice)

    # Team-identity scoring, NOT the shape-encoded variant. Shape encoding
    # credits a positional bool match at slot X; it is only valid when the
    # bracket and the outcome come from the SAME stochastic realization, since
    # otherwise the teams occupying slot X diverge after R64. Here every
    # candidate is scored against a *different* simulated tournament, so shape
    # would credit brackets for advancing teams that never won. See
    # pool_competition.score_brackets_against_outcome's own warning.
    eval_winners = [
        {rnd: set(r[ri]) for ri, rnd in enumerate(ROUND_NAMES)} for r in eval_rounds
    ]

    def expected_score(candidates: np.ndarray, n_cand: int = 300, n_eval: int = 250) -> float:
        """Mean ESPN score of candidates against the shared evaluation slice.

        The evaluation slice is drawn from the UNCONDITIONAL bank on purpose: a
        user's preference does not change which tournament actually happens, so
        conditional brackets must be judged against the real distribution.
        """
        if len(candidates) == 0:
            return 0.0
        pick = candidates[
            rng.choice(len(candidates), size=min(n_cand, len(candidates)), replace=False)
        ]
        totals = [
            score_brackets_team_identity(pick, w, first_round, ESPN_SCORING).mean()
            for w in eval_winners[:n_eval]
        ]
        return float(np.mean(totals))

    named_team = max(barthag, key=barthag.get)
    conditions = make_conditions(seeds, named_team)

    cand_bank = bank[cand_idx]
    cand_rounds = [rounds[i] for i in cand_idx]
    base_ev = expected_score(cand_bank)

    results = []
    for label, pred in conditions:
        mask = np.fromiter((pred(r) for r in cand_rounds), dtype=bool, count=len(cand_rounds))
        n_surv = int(mask.sum())
        p_cond = n_surv / len(cand_rounds)
        entry = {
            "condition": label,
            "p_condition": round(p_cond, 5),
            "n_surviving": n_surv,
            "n_unique": int(len(np.unique(cand_bank[mask], axis=0))) if n_surv else 0,
        }
        if n_surv >= 50:
            sub = cand_bank[mask]
            sub_rounds = [r for r, m in zip(cand_rounds, mask) if m]
            ev = expected_score(sub)
            ent, n_champs, top_champ, top_share = champion_entropy(sub_rounds)
            entry.update(
                {
                    "hamming": hamming_stats(sub, rng),
                    "expected_score": round(ev, 1),
                    "ev_retention": round(ev / base_ev, 4) if base_ev else 0.0,
                    "champion_entropy_bits": round(ent, 2),
                    "distinct_champions": n_champs,
                    "modal_champion": top_champ,
                    "modal_champion_share": round(top_share, 3),
                    "mean_seed_by_round": seed_profile(sub_rounds, seeds),
                }
            )
        else:
            entry["note"] = "infeasible at this bank size (<50 surviving)"
        results.append(entry)

    return {
        "year": year,
        "n_sims": n_sims,
        "eval_slice": eval_slice,
        "candidate_pool": len(cand_idx),
        "source": pw.source,
        "named_team": named_team,
        "baseline_expected_score": round(base_ev, 1),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "conditions": results,
    }


def report(res: Dict) -> None:
    print(f"\n{'=' * 100}")
    print(
        f"SCENARIO BANK — {res['year']}  ({res['candidate_pool']:,} candidates, "
        f"{res['eval_slice']:,} held out for scoring)"
    )
    print(f"source: {res['source']}   baseline E[score]: {res['baseline_expected_score']}")
    print("=" * 100)
    hdr = f"{'condition':34} {'P(cond)':>8} {'surviving':>10} {'unique':>9} {'Hamming':>8} {'E[score]':>9} {'EV ret':>7} {'champs':>7}"
    print(hdr)
    print("-" * len(hdr))
    for c in res["conditions"]:
        if "hamming" not in c:
            print(
                f"{c['condition']:34} {c['p_condition']:8.4f} {c['n_surviving']:10,} "
                f"{c['n_unique']:9,}   {c.get('note', '')}"
            )
            continue
        print(
            f"{c['condition']:34} {c['p_condition']:8.4f} {c['n_surviving']:10,} "
            f"{c['n_unique']:9,} {c['hamming']['mean']:8.1f} {c['expected_score']:9.1f} "
            f"{c['ev_retention']:7.3f} {c['distinct_champions']:7,}"
        )

    print(f"\n{'PRODUCTION REFERENCE':34} {'—':>8} {'~25.5':>10} {'~5':>9} {'6.0':>8}     (FINDINGS.md 6e)")
    print("\nMean seed by round, per condition:")
    for c in res["conditions"]:
        if "mean_seed_by_round" in c:
            s = c["mean_seed_by_round"]
            print(
                f"  {c['condition']:34} S16 {s['S16']:5.2f}  E8 {s['E8']:5.2f}  "
                f"F4 {s['F4']:5.2f}  CHAMP {s['CHAMP']:5.2f}"
            )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--n-sims", type=int, default=200_000)
    ap.add_argument("--eval-slice", type=int, default=2_000)
    ap.add_argument("--seed", type=int, default=20260820)
    ap.add_argument("--out", type=str, default="artifacts/scenario_bank")
    args = ap.parse_args()

    res = run(args.year, args.n_sims, args.eval_slice, args.seed)
    report(res)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"scenario_bank_{args.year}.json"
    with open(path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
