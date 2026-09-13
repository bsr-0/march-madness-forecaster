"""Is the simulated opponent field shaped like the real pool? (audit H3 / recommendation 13)

WHY THIS EXISTS
---------------
Audit recommendation 13 asks for "chalk clustering / correlated picks" on the
grounds (finding H3) that the opponent field is drawn as independent brackets
with ``chalk_noise_std=0.0`` and therefore has "no opponent-outcome coupling, no
chalk clustering". The machinery to fix that already exists and is correct:
``src.simulation.pool_competition.generate_opponent_brackets`` implements a
proper two-level hierarchical draw -- one shared pool-narrative shift, one
per-opponent shift around it, applied in logit space scaled by seed gap -- and
its docstring recommends "0.3-0.6 for realistic N=31 pool correlation".

Nothing in the repo has ever checked that recommendation against the real pool.
Before wiring correlated picks into the production path, the question has to be
asked in the other direction: **is the real pool actually more correlated than
independent draws?** If it is not, turning this on would move the simulation
away from the thing it is meant to imitate.

One prior attempt exists -- ``artifacts/o4_opponent_independence_2026-04-14.json``
-- and it reported negative excess correlation in all four years (pooled
Stouffer z = -4.15). It cannot be relied on:

  * It is a **re-serialization**. No script in the repo produces it, so the
    measurement cannot be reproduced, inspected or corrected.
  * It predates ``b73d351`` (2026-09-06), the play-in resolution fix. Bracket
    vectors are positionally aligned to the Round-of-64 field, and that fix
    changed the field in every season, so every pre-fix bracket vector was
    built against a partly wrong R64 -- the same reason every pre-fix backtest
    number is void rather than merely stale.

So it is measured here again, reproducibly, on current code.

THE F4 ORDERING TRAP
--------------------
``pool_entry_to_bracket_vector`` maps an entry's picks into a 63-slot vector
positionally aligned with ``first_round``, so the bracket-tree topology has to
match the season's real Final Four pairing. The NCAA rotates which regions meet,
and ``build_first_round_matchups``'s default ``REGION_ORDER`` matches the real
pairing in **zero of the 15 seasons**. Building the vectors against the default
would silently compare brackets in the wrong coordinate system. This script uses
``derive_f4_region_pairing`` -- the same fix as
``scripts/independent_referee_check.py``, where omitting it produced a bracket
that could not be reconstructed at all.

PRE-REGISTERED DECISION RULE
----------------------------
Fixed here before the script was first run. The statistic is mean pairwise
agreement: over all pairs of brackets in a field, the fraction of the 63 games
where the two picks match. The null is an independent field drawn from the same
per-game marginals, which preserves "everyone picks the 1-seeds" consensus and
removes only the correlation.

  pooled Stouffer z > +2   real pool IS more correlated than independent draws.
                           Adopt chalk clustering, calibrated to the value whose
                           simulated agreement best matches the real field.
  |pooled z| <= 2          independence is adequate at this sample size. Do not
                           adopt; record the recommendation as unsupported.
  pooled z < -2            the real pool is MORE diverse than independent draws.
                           Adopting chalk clustering would move the simulation
                           away from reality. Do not adopt, and correct the
                           docstring that recommends 0.3-0.6.

Only 2023-2026 have real pool brackets (105 entries from one pool), so this is
n=4 pool-years however many brackets it contains -- the entrants overlap year to
year. A null result here is weak evidence of absence, and the report says so.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OUT_PATH = Path("artifacts/headline_measurement/opponent_realism.json")

# The only seasons with real pool brackets (data/pool_history/pool_hist_results.json).
POOL_YEARS = (2023, 2024, 2025, 2026)

# Chalk values to compare against reality. 0.0 is production; 0.3/0.4/0.6 are
# the range generate_opponent_brackets' own docstring recommends; 0.15 is the
# pool_history_opponent_model fallback default.
CHALK_GRID = (0.0, 0.15, 0.3, 0.4, 0.6)

N_NULL_DRAWS = 2000
SEED = 42


def _mean_pairwise_agreement(matrix: np.ndarray) -> float:
    """Mean over all bracket pairs of the fraction of the 63 games they agree on.

    Computed in closed form rather than by looping over pairs: for each game,
    the number of agreeing pairs is C(k,2) + C(n-k,2) where k is how many
    brackets picked the first-listed team. Summing over games and dividing by
    n_games * C(n,2) gives the mean agreement exactly.
    """
    n = matrix.shape[0]
    if n < 2:
        raise ValueError(f"need at least 2 brackets to have a pair, got {n}")
    k = matrix.sum(axis=0).astype(float)  # per-game count picking team1
    agreeing = k * (k - 1.0) / 2.0 + (n - k) * (n - k - 1.0) / 2.0
    total_pairs = n * (n - 1.0) / 2.0
    return float(agreeing.sum() / (matrix.shape[1] * total_pairs))


def _independent_null(matrix: np.ndarray, n_draws: int, rng: np.random.Generator) -> np.ndarray:
    """Mean pairwise agreement of fields drawn independently from the same marginals.

    Preserves each game's observed pick rate and destroys every dependence
    between games and between brackets. The gap between the real field and this
    null is exactly "correlation beyond what consensus alone explains".
    """
    n, n_games = matrix.shape
    rates = matrix.mean(axis=0)
    out = np.empty(n_draws, dtype=float)
    for i in range(n_draws):
        sim = rng.random((n, n_games)) < rates[None, :]
        out[i] = _mean_pairwise_agreement(sim)
    return out


def _year_inputs(year: int):
    """Seeds, regions, the real F4-ordered first round, and the real bracket matrix."""
    from scripts.mc_pool_backtest import (
        POOL_HIST_PATH,
        build_first_round_matchups,
        derive_f4_region_pairing,
        load_seeds_and_regions,
        load_tournament_results,
    )
    from src.simulation.pool_history_opponent_model import load_pool_bracket_vectors

    seeds, regions = load_seeds_and_regions(year)
    games = load_tournament_results(year)
    # See "THE F4 ORDERING TRAP" above. Never call this without region_order.
    region_order = derive_f4_region_pairing(games, regions)
    first_round = build_first_round_matchups(seeds, regions, region_order=region_order)
    matrix, group_size = load_pool_bracket_vectors(POOL_HIST_PATH, year, first_round, seeds)
    return seeds, regions, first_round, matrix, group_size, region_order


def _simulated_field(year: int, seeds, first_round, n_opponents: int, chalk: float, rng) -> np.ndarray:
    """One synthetic opponent field at a given chalk level, as production builds it."""
    from scripts.mc_pool_backtest import POOL_HIST_PATH, build_seed_probabilities
    from src.simulation.pool_competition import generate_opponent_brackets
    from src.simulation.pool_history_opponent_model import build_pool_pick_distribution, load_pool_brackets

    brackets, _ = load_pool_brackets(POOL_HIST_PATH, year)
    pick_dist = build_pool_pick_distribution(brackets, seeds)
    seed_pw = build_seed_probabilities(seeds)
    return generate_opponent_brackets(
        n_opponents=n_opponents,
        first_round_matchups=first_round,
        matchup_probs=seed_pw,
        pick_distribution=pick_dist,
        seeds=seeds,
        rng=rng,
        chalk_noise_std=chalk,
    )


def measure(years: Sequence[int] = POOL_YEARS, n_null: int = N_NULL_DRAWS) -> Dict[str, Any]:
    from scipy import stats as sp_stats

    per_year: Dict[str, Any] = {}
    z_scores: List[float] = []

    for year in years:
        seeds, _regions, first_round, matrix, group_size, region_order = _year_inputs(year)
        n_brackets = matrix.shape[0]
        observed = _mean_pairwise_agreement(matrix)

        rng = np.random.default_rng(SEED + year)
        null = _independent_null(matrix, n_null, rng)
        null_mean = float(null.mean())
        null_sd = float(null.std(ddof=1))
        z = (observed - null_mean) / max(null_sd, 1e-12)
        z_scores.append(z)

        # How does the production simulator compare, at each chalk level?
        sim_rows = []
        for chalk in CHALK_GRID:
            sim_rng = np.random.default_rng(SEED + year)
            # Average over repeats: one call re-draws the shared pool narrative,
            # so a single field is a sample of one from the chalk distribution
            # and would be far too noisy to calibrate against.
            agreements = []
            for rep in range(50):
                field = _simulated_field(
                    year, seeds, first_round, n_brackets, chalk, np.random.default_rng(SEED + year + 1000 * rep)
                )
                agreements.append(_mean_pairwise_agreement(field))
            del sim_rng
            sim_mean = float(np.mean(agreements))
            sim_rows.append(
                {
                    "chalk_noise_std": chalk,
                    "simulated_agreement": sim_mean,
                    "simulated_sd_over_repeats": float(np.std(agreements, ddof=1)),
                    "abs_error_vs_real": abs(sim_mean - observed),
                }
            )

        best = min(sim_rows, key=lambda r: r["abs_error_vs_real"])
        per_year[str(year)] = {
            "n_brackets": n_brackets,
            "group_size": group_size,
            "f4_region_order": list(region_order),
            "observed_agreement": observed,
            "independent_null_mean": null_mean,
            "independent_null_sd": null_sd,
            "excess_agreement": observed - null_mean,
            "z_score": z,
            "verdict": (
                "more_correlated_than_independent"
                if z > 2
                else ("less_correlated_than_independent" if z < -2 else "independence_holds")
            ),
            "simulator_comparison": sim_rows,
            "best_matching_chalk": best["chalk_noise_std"],
        }

    # Stouffer pooling. The years share a pool and largely share entrants, so
    # this overstates independence between years; it is reported because the
    # prior artifact reported it, and hedged in the notes.
    pooled_z = float(np.sum(z_scores) / np.sqrt(len(z_scores)))
    pooled_p_two_sided = float(2.0 * (1.0 - sp_stats.norm.cdf(abs(pooled_z))))

    if pooled_z > 2:
        verdict = "adopt_chalk_clustering"
    elif pooled_z < -2:
        verdict = "reject_chalk_clustering_real_pool_is_more_diverse"
    else:
        verdict = "independence_adequate_do_not_adopt"

    best_chalks = [per_year[str(y)]["best_matching_chalk"] for y in years]
    return {
        "measurement": "inter-bracket agreement in the real pool vs an independent field",
        "statistic": "mean pairwise agreement over the 63 games",
        "years": list(years),
        "n_pool_years": len(years),
        "per_year": per_year,
        "pooled_stouffer_z": pooled_z,
        "pooled_p_two_sided": pooled_p_two_sided,
        "verdict": verdict,
        "decision_rule": (
            "Pre-registered in the module docstring before the first run: pooled z > +2 adopt "
            "chalk clustering; |z| <= 2 independence adequate, do not adopt; z < -2 the real "
            "pool is more diverse than independent draws and adopting would move the "
            "simulation away from reality."
        ),
        "best_matching_chalk_per_year": dict(zip(map(str, years), best_chalks)),
        "chalk_grid": list(CHALK_GRID),
        "limitations": (
            "Only 2023-2026 have real pool brackets, all from one pool (groupId "
            "0b6a2bbe-...), and the entrants overlap year to year -- so the effective sample "
            "is nearer 4 pool-years than 105 brackets, and a null result is weak evidence of "
            "absence rather than evidence of independence. The statistic is symmetric "
            "agreement, which is insensitive to WHICH games drive the correlation; a field "
            "could match on total agreement while clustering on entirely different games."
        ),
        "supersedes": (
            "artifacts/o4_opponent_independence_2026-04-14.json, which no script produces "
            "(it is a re-serialization) and which predates the b73d351 play-in fix, so its "
            "bracket vectors were built against a partly wrong Round-of-64 field."
        ),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--years", type=int, nargs="+", default=list(POOL_YEARS))
    parser.add_argument("--n-null", type=int, default=N_NULL_DRAWS)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args(argv)

    result = measure(args.years, args.n_null)
    result["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    print(f"\n{'year':>6} {'n':>4} {'observed':>10} {'indep null':>11} {'excess':>9} {'z':>7}  verdict")
    print("-" * 78)
    for year in args.years:
        r = result["per_year"][str(year)]
        print(
            f"{year:>6} {r['n_brackets']:>4} {r['observed_agreement']:>10.4f} "
            f"{r['independent_null_mean']:>11.4f} {r['excess_agreement']:>+9.4f} {r['z_score']:>+7.2f}  {r['verdict']}"
        )
    print(f"\npooled Stouffer z = {result['pooled_stouffer_z']:+.3f} (p={result['pooled_p_two_sided']:.4f})")
    print(f"VERDICT: {result['verdict']}")

    print(f"\n{'year':>6}  simulated agreement by chalk_noise_std (real field in brackets)")
    print(f"{'':>6}  " + "  ".join(f"{c:>7.2f}" for c in CHALK_GRID) + "     real")
    for year in args.years:
        r = result["per_year"][str(year)]
        row = "  ".join(f"{s['simulated_agreement']:>7.4f}" for s in r["simulator_comparison"])
        print(f"{year:>6}  {row}   [{r['observed_agreement']:.4f}]  best={r['best_matching_chalk']}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    tmp.replace(args.out)
    print(f"\n  [artifact] {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
