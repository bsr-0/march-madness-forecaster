"""Phase-3: build seed=42 reproducible bracket cache from Phase-2 probabilities.

For each (construction_mode, year), reads the cached torvik probability tensor,
seeds the RNG to 42, calls the production sampler, converts the bracket array
into the (50, 63) uint16 alphabetical-within-round format used by Phase-1, and
writes one NPZ per (strategy_id, year) into the strategy cache.

Each (mode, year) job is independent — run with --workers N to parallelize via
ProcessPoolExecutor. Manifest writes are serialized in the main process to
avoid races.

Run:
    python -m scripts.build_bracket_cache                    # sequential, all
    python -m scripts.build_bracket_cache --workers 8        # 8-way parallel
    python -m scripts.build_bracket_cache --modes f4_first_tv --years 2026
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.mc_pool_backtest import (  # noqa: E402
    BACKTEST_YEARS,
    build_first_round_matchups,
    derive_f4_region_pairing,
    load_seeds_and_regions,
    load_tournament_results,
    resolve_first_four,
    sample_e8_first_brackets,
    sample_f4_first_brackets,
    sample_model_brackets,
)
from src.data.strategy_cache import (  # noqa: E402
    Manifest,
    compute_cache_key,
    load_probabilities,
    load_team_lookup,
    save_brackets,
)
from src.simulation.pool_competition import picks_by_round  # noqa: E402

ROUND_ORDER = ("R64", "R32", "S16", "E8", "F4", "CHAMP")
ROUND_SIZES = {"R64": 32, "R32": 16, "S16": 8, "E8": 4, "F4": 2, "CHAMP": 1}
RNG_SEED = 42
N_BRACKETS = 50
PROBABILITY_SOURCE = "torvik"
PROBABILITY_MODEL_VERSION = "torvik-barthag-v1"
PROBABILITY_DATA_VERSION = "tournament_seeds+torvik_barthag"
BRACKET_DATA_VERSION = "torvik+seed=42"
BRACKET_MODEL_VERSION = "torvik-barthag-v1"

# (strategy_id, sampler-builder). Builder takes (seeds, regions) and returns
# the callable signature (first_round, round_probs, n_brackets, rng).
SAMPLER_BUILDERS = {
    "torvik": lambda seeds, regions: lambda fr, rp, n, r: sample_model_brackets(fr, rp, n, r),
    "f4_first_tv": lambda seeds, regions: lambda fr, rp, n, r: sample_f4_first_brackets(fr, rp, n, r, seeds, regions),
    "e8_first_tv": lambda seeds, regions: lambda fr, rp, n, r: sample_e8_first_brackets(fr, rp, n, r, seeds, regions),
}
DEFAULT_MODES = list(SAMPLER_BUILDERS.keys())


def git_sha_short() -> str:
    return subprocess.check_output(["git", "rev-parse", "--short=12", "HEAD"], cwd=PROJECT_ROOT).decode().strip()


def load_round_probs_from_cache(year: int, code_version: str) -> dict:
    """Reverse the Phase-2 (n_teams, 6) array back into Dict[team, Dict[round, prob]]."""
    prob_key = compute_cache_key(
        strategy_id=PROBABILITY_SOURCE,
        year=year,
        rng_seed=RNG_SEED,
        code_version=code_version,
        data_version=PROBABILITY_DATA_VERSION,
        model_version=PROBABILITY_MODEL_VERSION,
    )
    probs, team_ids = load_probabilities(prob_key)
    lookup = load_team_lookup()
    inverse = {v: k for k, v in lookup.items()}
    out = {}
    for i, team_id in enumerate(team_ids):
        team_name = inverse[int(team_id)]
        out[team_name] = {round_name: float(probs[i, j]) for j, round_name in enumerate(ROUND_ORDER)}
    return out


def brackets_to_array(
    model_brackets: np.ndarray,
    first_round: list,
    lookup: dict[str, int],
) -> np.ndarray:
    """Convert sampler output into (n_brackets, 63) uint16 alphabetical-within-round."""
    n = model_brackets.shape[0]
    out = np.zeros((n, 63), dtype=np.uint16)
    for m in range(n):
        picks = picks_by_round(model_brackets[m], first_round)
        cursor = 0
        for round_name in ROUND_ORDER:
            size = ROUND_SIZES[round_name]
            teams = sorted(picks[round_name])
            if len(teams) != size:
                raise ValueError(f"bracket {m} round {round_name}: expected {size}, got {len(teams)}")
            for i, team in enumerate(teams):
                out[m, cursor + i] = lookup[team]
            cursor += size
    return out


def build_one(args: tuple[str, int, str]) -> dict:
    """Worker function — top-level so it's picklable for ProcessPoolExecutor."""
    mode, year, code_version = args
    seeds, regions = load_seeds_and_regions(year)
    seeds = resolve_first_four(year, seeds)
    games = load_tournament_results(year)
    region_order = derive_f4_region_pairing(games, regions)
    first_round = build_first_round_matchups(seeds, regions, region_order=region_order)
    round_probs = load_round_probs_from_cache(year, code_version)

    sampler = SAMPLER_BUILDERS[mode](seeds, regions)
    rng = np.random.default_rng(RNG_SEED)
    model_brackets = sampler(first_round, round_probs, N_BRACKETS, rng)

    lookup = load_team_lookup()
    bracket_array = brackets_to_array(model_brackets, first_round, lookup)

    cache_key = compute_cache_key(
        strategy_id=mode,
        year=year,
        rng_seed=RNG_SEED,
        code_version=code_version,
        data_version=BRACKET_DATA_VERSION,
        model_version=BRACKET_MODEL_VERSION,
    )
    entry = save_brackets(
        cache_key,
        bracket_array,
        strategy_id=mode,
        year=year,
        rng_seed=RNG_SEED,
        code_version=code_version,
        data_version=BRACKET_DATA_VERSION,
        model_version=BRACKET_MODEL_VERSION,
        provenance={
            "sampler": mode,
            "n_brackets": N_BRACKETS,
            "from_probability_cache": True,
        },
    )
    return {"mode": mode, "year": year, "cache_key": cache_key, "entry": entry}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="*", default=DEFAULT_MODES)
    ap.add_argument("--years", type=int, nargs="*", default=BACKTEST_YEARS)
    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        help="parallel workers (1 = sequential, deterministic ordering)",
    )
    args = ap.parse_args()

    code_version = git_sha_short()
    jobs = [(mode, year, code_version) for mode in args.modes for year in args.years]
    print(f"building {len(jobs)} bracket entries ({len(args.modes)} modes x {len(args.years)} years)")

    results: list[dict] = []
    if args.workers <= 1:
        for j in jobs:
            results.append(build_one(j))
            print(f"  built {results[-1]['mode']} {results[-1]['year']} key={results[-1]['cache_key']}")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(build_one, j): j for j in jobs}
            for fut in as_completed(futures):
                r = fut.result()
                results.append(r)
                print(f"  built {r['mode']} {r['year']} key={r['cache_key']}")

    manifest = Manifest.load()
    for r in results:
        manifest.append(r["entry"])
    manifest.save()
    print(f"\nwrote {len(results)} bracket entries; manifest updated")


if __name__ == "__main__":
    main()
