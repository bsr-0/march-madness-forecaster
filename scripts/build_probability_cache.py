"""Phase-2: build per-(strategy, year) round-advancement probability cache.

Calls the production probability builder (`build_torvik_round_probabilities`,
which is internally seeded with rng=42) per backtest year, converts the
Dict[team_id, Dict[round, prob]] output into a (n_teams, 6) float32 array
with a parallel uint16 team_ids column, and writes one NPZ per (source, year)
into the strategy cache.

Phase-2 scope: only the `torvik` probability source — covers the 3 production
bracket modes (torvik, f4_first_tv, e8_first_tv), which all consume the same
torvik round_probs and differ only in their construction sampler.

Run:
    python -m scripts.build_probability_cache
    python -m scripts.build_probability_cache --years 2025 2026
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.mc_pool_backtest import (  # noqa: E402
    BACKTEST_YEARS,
    _load_torvik_barthag,
    build_torvik_round_probabilities,
    load_seeds_and_regions,
)
from src.data.strategy_cache import (  # noqa: E402
    CACHE_PARENT_SEED,
    Manifest,
    PROBABILITY_DATA_VERSION,
    PROBABILITY_MODEL_VERSION,
    STRATEGY_CACHE_VERSION,
    compute_cache_key,
    load_team_lookup,
    save_probabilities,
    save_team_lookup,
)

ROUND_ORDER = ("R64", "R32", "S16", "E8", "F4", "CHAMP")
PROBABILITY_SOURCE = "torvik"


def round_probs_to_array(round_probs: dict, lookup: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
    """Return (probabilities, team_ids) as (n_teams, 6) float32 + (n_teams,) uint16."""
    teams = sorted(round_probs.keys())
    team_ids = np.array([lookup[t] for t in teams], dtype=np.uint16)
    probs = np.zeros((len(teams), 6), dtype=np.float32)
    for i, team in enumerate(teams):
        for j, round_name in enumerate(ROUND_ORDER):
            probs[i, j] = round_probs[team][round_name]
    return probs, team_ids


def ensure_team_lookup_covers(team_names: set[str]) -> dict[str, int]:
    """Extend the team_id lookup if a year introduces new team names.

    Phase-1 migration seeded the lookup from cached brackets and historical
    results; new strategies / years can introduce names that weren't on disk
    yet (e.g., NIT-only teams that won't appear in tournament_results).
    """
    lookup = load_team_lookup()
    missing = sorted(team_names - lookup.keys())
    if missing:
        next_id = max(lookup.values(), default=-1) + 1
        for name in missing:
            lookup[name] = next_id
            next_id += 1
        save_team_lookup(lookup)
    return lookup


def build_one_year(year: int) -> int:
    """Compute and cache probabilities for one year. Returns cache hits added."""
    seeds, regions = load_seeds_and_regions(year)
    barthag = _load_torvik_barthag(year, seeds)
    if barthag is None:
        print(f"  year={year}: barthag unavailable, skipping")
        return 0
    round_probs = build_torvik_round_probabilities(seeds, regions, barthag)

    lookup = ensure_team_lookup_covers(set(round_probs.keys()))
    probabilities, team_ids = round_probs_to_array(round_probs, lookup)

    cache_key = compute_cache_key(
        strategy_id=PROBABILITY_SOURCE,
        year=year,
        rng_seed=CACHE_PARENT_SEED,
        code_version=str(STRATEGY_CACHE_VERSION),
        data_version=PROBABILITY_DATA_VERSION,
        model_version=PROBABILITY_MODEL_VERSION,
    )
    entry = save_probabilities(
        cache_key,
        probabilities,
        team_ids,
        strategy_id=PROBABILITY_SOURCE,
        year=year,
        rng_seed=CACHE_PARENT_SEED,
        code_version=str(STRATEGY_CACHE_VERSION),
        data_version=PROBABILITY_DATA_VERSION,
        model_version=PROBABILITY_MODEL_VERSION,
        provenance={"builder": "build_torvik_round_probabilities", "n_sims": 10000},
    )

    manifest = Manifest.load()
    manifest.append(entry)
    manifest.save()
    print(f"  year={year} key={cache_key} shape={probabilities.shape} n_teams={len(team_ids)}")
    return 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--years",
        type=int,
        nargs="*",
        default=BACKTEST_YEARS,
        help="years to build (default: all 14 backtest years)",
    )
    args = ap.parse_args()

    n_added = 0
    for year in args.years:
        n_added += build_one_year(year)
    print(f"\nwrote {n_added} probability entries")
    print("manifest at artifacts/strategy_cache/manifest.json")


if __name__ == "__main__":
    main()
