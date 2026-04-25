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
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

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
    BRACKET_DATA_VERSION,
    BRACKET_MODEL_VERSION,
    CACHE_PARENT_SEED,
    Manifest,
    PROBABILITY_DATA_VERSION,
    PROBABILITY_MODEL_VERSION,
    STRATEGY_CACHE_VERSION,
    brackets_to_array,
    compute_cache_key,
    load_probabilities,
    load_team_lookup,
    make_mode_rng,
    save_brackets,
)

ROUND_ORDER = ("R64", "R32", "S16", "E8", "F4", "CHAMP")
N_BRACKETS = 50
PROBABILITY_SOURCE = "torvik"

# (strategy_id, sampler-builder). Builder takes (seeds, regions) and returns
# the callable signature (first_round, round_probs, n_brackets, rng).
SAMPLER_BUILDERS = {
    "torvik": lambda seeds, regions: lambda fr, rp, n, r: sample_model_brackets(fr, rp, n, r),
    "f4_first_tv": lambda seeds, regions: lambda fr, rp, n, r: sample_f4_first_brackets(fr, rp, n, r, seeds, regions),
    "e8_first_tv": lambda seeds, regions: lambda fr, rp, n, r: sample_e8_first_brackets(fr, rp, n, r, seeds, regions),
}
DEFAULT_MODES = list(SAMPLER_BUILDERS.keys())


def load_round_probs_from_cache(year: int) -> dict:
    """Reverse the Phase-2 (n_teams, 6) array back into Dict[team, Dict[round, prob]]."""
    prob_key = compute_cache_key(
        strategy_id=PROBABILITY_SOURCE,
        year=year,
        rng_seed=CACHE_PARENT_SEED,
        code_version=str(STRATEGY_CACHE_VERSION),
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


def build_one(args: tuple[str, int]) -> dict:
    """Worker function — top-level so it's picklable for ProcessPoolExecutor."""
    mode, year = args
    seeds, regions = load_seeds_and_regions(year)
    seeds = resolve_first_four(year, seeds)
    games = load_tournament_results(year)
    region_order = derive_f4_region_pairing(games, regions)
    first_round = build_first_round_matchups(seeds, regions, region_order=region_order)
    round_probs = load_round_probs_from_cache(year)

    sampler = SAMPLER_BUILDERS[mode](seeds, regions)
    # Per-(mode, year) child rng — matches what the live experiment loop
    # in scripts/mc_pool_backtest.run_backtest passes the same sampler, so
    # cached brackets are bit-exact reproducible from the loop side.
    bracket_rng = make_mode_rng(mode, year, parent_seed=CACHE_PARENT_SEED)
    model_brackets = sampler(first_round, round_probs, N_BRACKETS, bracket_rng)

    lookup = load_team_lookup()
    bracket_array = brackets_to_array(model_brackets, first_round, lookup)

    cache_key = compute_cache_key(
        strategy_id=mode,
        year=year,
        rng_seed=CACHE_PARENT_SEED,
        code_version=str(STRATEGY_CACHE_VERSION),
        data_version=BRACKET_DATA_VERSION,
        model_version=BRACKET_MODEL_VERSION,
    )
    entry = save_brackets(
        cache_key,
        bracket_array,
        strategy_id=mode,
        year=year,
        rng_seed=CACHE_PARENT_SEED,
        code_version=str(STRATEGY_CACHE_VERSION),
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

    jobs = [(mode, year) for mode in args.modes for year in args.years]
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
