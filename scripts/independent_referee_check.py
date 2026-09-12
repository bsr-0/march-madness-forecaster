"""Does the headline P(1st) hold up against a referee it wasn't selected on?

    python -m scripts.independent_referee_check

Audit finding C2 (CRITICAL) / recommendation 11: `meta_region_poolaware` is
both *chosen* and *scored* against the same simulated-outcome distribution,
`seed_pw` -- the empirical seed-vs-seed head-to-head table, fit on
2010-2025, which overlaps every backtested season. That is in-sample by
construction, and conflates two questions: "did the selector pick the
candidate that wins most under this referee" with "does the strategy win
real pools."

This script does NOT touch selection. It takes the bracket
`meta_region_poolaware` already selected for each canonical evaluation
season (from `artifacts/backtest_brackets/`, built by
`mc_pool_backtest.py --save-brackets` under the unmodified, current
production code) and rescoring that FIXED bracket against a genuinely
independent referee: Torvik barthag + log5 (`build_torvik_probabilities`),
which never saw seed-advancement history and is not the model anything was
selected against.

Opponent PICK BEHAVIOUR is left on seed_pw, unchanged from production --
that model concerns how real people pick, a separate question from which
referee grades the outcome. Only the "true outcome" draw
(`simulate_tournament_outcomes`) changes. Both referees draw from the SAME
opponent field per repeat (common random numbers), so the paired
seed-vs-torvik difference is the estimand, not two independently noisy
absolute numbers.

Decision rule, fixed before running: "the headline is referee-sensitive" if
the paired 95% CI on the per-season P(1st) difference (torvik - seed_pw)
excludes zero.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts._common import load_tournament_results  # noqa: E402
from scripts.mc_pool_backtest import (  # noqa: E402
    ESPN_SCORING,
    REFEREE_NOISE_STD,
    ROUND_NAMES,
    build_first_round_matchups,
    build_seed_probabilities,
    derive_f4_region_pairing,
    load_seeds_and_regions,
    resolve_first_four,
    resolve_opponent_pick_distribution,
)
from src.prediction.torvik_probabilities import (  # noqa: E402
    build_torvik_probabilities,
    load_torvik_barthag,
)
from src.simulation.pool_competition import (  # noqa: E402
    generate_opponent_brackets,
    score_brackets_team_identity,
    simulate_tournament_outcomes,
)

EVAL_YEARS = [2011, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025]
N_OPPONENTS = 29
N_REPEATS = 500  # higher than the production default (100): this script runs one
# mode for one year at a time, so the extra precision is cheap, and a smaller
# season-level effect needs it to be detectable at n=13 seasons.
BRACKETS_DIR = PROJECT_ROOT / "artifacts" / "backtest_brackets"


def _round_winners_to_bool_vector(
    first_round: List[str], picks_by_round: Dict[str, List[str]], label: str = ""
) -> np.ndarray:
    """(63,) bool vector from a {round_name: [teams advancing]} dict.

    `picks_by_round[round]` lists who WON that round -- i.e. who advances --
    not a game-keyed winner dict, so this is a direct membership walk rather
    than a reuse of `_picks_dict_to_bool_array` (which expects the other
    shape). Requires `first_round`'s bracket-tree topology (region_order) to
    match whatever topology produced `picks_by_round`, or a round's winner
    set won't contain either team of some game it's checked against --
    concretely, the default REGION_ORDER vs the actual year's real F4
    pairing (NCAA rotates which regions meet); see `derive_f4_region_pairing`.
    A mismatch raises here rather than silently mis-scoring.
    """
    result = np.zeros(63, dtype=bool)
    current = list(first_round)
    game_idx = 0
    for round_name in ROUND_NAMES:
        winners = set(picks_by_round[round_name])
        next_round = []
        for g in range(0, len(current), 2):
            t1, t2 = current[g], current[g + 1]
            if t1 in winners:
                result[game_idx] = True
                next_round.append(t1)
            elif t2 in winners:
                result[game_idx] = False
                next_round.append(t2)
            else:
                raise ValueError(f"{label}{round_name} game {g}: neither {t1} nor {t2} in {winners}")
            game_idx += 1
        current = next_round
    return result


def _selected_bracket_bool_vector(year: int, first_round: List[str]) -> Tuple[np.ndarray, str]:
    """The meta_region_poolaware pick for `year`, as a (63,) bool vector.

    Reads artifacts/backtest_brackets/backtest_brackets_{year}.json, built by
    `mc_pool_backtest.py --save-brackets` (unmodified selection code).

    DO NOT regenerate this file with `--modes meta_region_poolaware` alone.
    `--save-brackets` overwrites the whole file with only the modes named on
    that run, and the file already carries torvik/f4_first_tv/e8_first_tv --
    a locked ledger `tests/test_oracle_drift_guard.py` pins to specific
    numbers (found 2026-09-12, the hard way: a meta_region_poolaware-only
    regeneration during this script's own development silently dropped
    those three modes and failed 39 tests; recovered with
    `git checkout HEAD -- artifacts/backtest_brackets/`). Regenerate with
    every mode the file currently has, meta_region_poolaware included, in
    ONE invocation, or don't regenerate at all -- for most uses the
    committed file already has what this script needs.
    """
    path = BRACKETS_DIR / f"backtest_brackets_{year}.json"
    _REGEN_WARNING = (
        "regenerate in ONE run with EVERY mode this file already has, meta_region_poolaware "
        "included (--save-brackets overwrites the whole file with only the modes you name -- "
        "see this function's docstring) -- do not run --modes meta_region_poolaware alone: "
        f"python -m scripts.mc_pool_backtest --team-identity --opponent pool "
        f"--n-opponents {N_OPPONENTS} --n-repeats 100 --years {year} --save-brackets "
        f"--modes torvik f4_first_tv e8_first_tv meta_region_poolaware  # + any other modes present"
    )
    if not path.exists():
        raise FileNotFoundError(f"{path} missing -- {_REGEN_WARNING}")
    data = json.loads(path.read_text())
    modes = {m["mode"]: m for m in data["modes"]}
    if "meta_region_poolaware" not in modes:
        raise KeyError(f"{path}: no meta_region_poolaware entry -- {_REGEN_WARNING}")
    bracket = modes["meta_region_poolaware"]["brackets"][0]
    result = _round_winners_to_bool_vector(first_round, bracket["picks"], label=f"{year} ")
    return result, bracket["champion"]


def _score_year(year: int, n_repeats: int, seed: int) -> Dict:
    seeds, regions = load_seeds_and_regions(year)
    games = load_tournament_results(year)
    resolve_first_four(games, seeds, regions)
    # The synthetic bracket tree's F4 pairing must match the REAL one for
    # this year (NCAA rotates which regions meet in the Final Four), or the
    # round-winner sets in the saved bracket won't line up with adjacent-pair
    # game walks in the tree built here -- exactly the corruption
    # derive_f4_region_pairing's own docstring documents.
    region_order = derive_f4_region_pairing(games, regions)
    first_round = build_first_round_matchups(seeds, regions, region_order=region_order)
    if len(first_round) != 64:
        raise ValueError(f"{year}: {len(first_round)} teams in first round, expected 64")

    model_bracket, champion = _selected_bracket_bool_vector(year, first_round)
    model_brackets = model_bracket.reshape(1, 63)

    pick_dist, n_opp, _chalk = resolve_opponent_pick_distribution(
        year, seeds, N_OPPONENTS, "pool"
    )
    seed_pw = build_seed_probabilities(seeds)
    barthag = load_torvik_barthag(year, seeds)
    torvik_pw = build_torvik_probabilities(seeds, barthag)

    rng = np.random.default_rng(seed)
    ranks_seed = np.zeros(n_repeats)
    ranks_torvik = np.zeros(n_repeats)

    for rep in range(n_repeats):
        opp = generate_opponent_brackets(n_opp, first_round, seed_pw, pick_dist, seeds, rng)

        _out_s, by_round_s = simulate_tournament_outcomes(
            n_tournaments=1, first_round_matchups=first_round, matchup_probs=seed_pw,
            seeds=seeds, noise_std=REFEREE_NOISE_STD, rng=rng,
        )
        winners_seed = {rnd: set(by_round_s[0][ri]) for ri, rnd in enumerate(ROUND_NAMES)}
        opp_scores_s = score_brackets_team_identity(opp, winners_seed, first_round, ESPN_SCORING)
        model_score_s = score_brackets_team_identity(model_brackets, winners_seed, first_round, ESPN_SCORING)[0]
        better = np.sum(opp_scores_s > model_score_s)
        tied = np.sum(opp_scores_s == model_score_s)
        ranks_seed[rep] = better + 1 + tied / 2.0

        _out_t, by_round_t = simulate_tournament_outcomes(
            n_tournaments=1, first_round_matchups=first_round, matchup_probs=torvik_pw,
            seeds=seeds, noise_std=REFEREE_NOISE_STD, rng=rng,
        )
        winners_torvik = {rnd: set(by_round_t[0][ri]) for ri, rnd in enumerate(ROUND_NAMES)}
        opp_scores_t = score_brackets_team_identity(opp, winners_torvik, first_round, ESPN_SCORING)
        model_score_t = score_brackets_team_identity(model_brackets, winners_torvik, first_round, ESPN_SCORING)[0]
        better = np.sum(opp_scores_t > model_score_t)
        tied = np.sum(opp_scores_t == model_score_t)
        ranks_torvik[rep] = better + 1 + tied / 2.0

    return {
        "year": year,
        "champion": champion,
        "p1_seed": float((ranks_seed == 1.0).mean()),
        "p1_torvik": float((ranks_torvik == 1.0).mean()),
    }


def paired_bootstrap(diff: np.ndarray, n: int = 5000, seed: int = 42) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    means = np.empty(n)
    for k in range(n):
        idx = rng.integers(0, len(diff), len(diff))
        means[k] = diff[idx].mean()
    return float(diff.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-repeats", type=int, default=N_REPEATS)
    ap.add_argument("--seed", type=int, default=20260912)
    ap.add_argument("--years", default=None, help="comma-separated; default the 13 canonical evaluation years")
    args = ap.parse_args()

    years = [int(y) for y in args.years.split(",")] if args.years else EVAL_YEARS

    rows = []
    for year in years:
        try:
            r = _score_year(year, args.n_repeats, args.seed + year)
        except FileNotFoundError as exc:
            print(f"  {year}: SKIP -- {exc}")
            continue
        rows.append(r)
        print(f"  {year}  champ={r['champion']:<20}  P1(seed)={r['p1_seed']:.4f}  P1(torvik)={r['p1_torvik']:.4f}")

    if not rows:
        raise SystemExit("no seasons scored")

    p1_seed = np.array([r["p1_seed"] for r in rows])
    p1_torvik = np.array([r["p1_torvik"] for r in rows])
    diff = p1_torvik - p1_seed

    print()
    print("=" * 88)
    print(f"pooled P(1st): seed_pw referee {p1_seed.mean():.4f}   torvik referee {p1_torvik.mean():.4f}")
    m, lo, hi = paired_bootstrap(diff)
    print(f"paired per-season difference (torvik - seed_pw): {m:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  n={len(rows)}")
    referee_sensitive = not (lo <= 0.0 <= hi)
    print()
    if referee_sensitive:
        print("VERDICT: referee-sensitive -- the paired CI excludes zero. The headline P(1st) is")
        print("materially different under an independent referee; C2's circularity concern is live.")
    else:
        print("VERDICT: not referee-sensitive at this sample size -- the paired CI includes zero.")
        print("The headline P(1st) is not detectably an artifact of grading against the selection referee.")
    print("Rule fixed in advance: referee-sensitive iff the paired 95% CI excludes zero.")

    out = {
        "years": years,
        "n_repeats": args.n_repeats,
        "rows": rows,
        "pooled_p1_seed": float(p1_seed.mean()),
        "pooled_p1_torvik": float(p1_torvik.mean()),
        "paired_diff_mean": m,
        "paired_diff_ci95": [lo, hi],
        "referee_sensitive": referee_sensitive,
    }
    out_path = PROJECT_ROOT / "artifacts" / "headline_measurement" / "independent_referee_check.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
