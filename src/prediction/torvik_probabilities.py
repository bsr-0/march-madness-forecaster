"""Torvik barthag + Log5 + Monte Carlo probabilities for pool optimization.

Produces pairwise and round-advancement probabilities using Torvik's barthag
rating (expected win % vs an average opponent) as the only input. No ML
ensemble, no calibration stage — Log5 for pairwise and a bracket-structure
Monte Carlo for round advancement.

This module was deleted in commit 44b048f (2026-04-21), which silently broke
`optimize-pool --mode torvik` — the CLI's own default mode. It is restored
here, but as a thin wrapper rather than a reimplementation: pairwise
probabilities delegate to `PairwiseProbabilities.from_ratings` (the one
sanctioned log5 implementation, per `src/prediction/pairwise.py`) and round
probabilities delegate to `scripts.mc_pool_backtest.build_torvik_round_probabilities`
(the same function the production backtest uses) rather than each keeping its
own copy of the Monte Carlo loop. Only barthag *loading* is implemented here,
because it is the one piece with a CLI-specific concern (`--data-dir`) that
the backtest's loader does not take.

The function signatures mirror `src.prediction.seed_probabilities` so the
PoolOptimizer can consume torvik output with no changes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

from .noseed_model import _validate_pretournament
from .pairwise import PairwiseProbabilities

HIST_DIR = Path("data/raw/historical")
DATA_DIR = Path("data/raw")


def load_torvik_barthag(
    year: int,
    seeds: Dict[str, int],
    data_dir: Path | str | None = None,
) -> Dict[str, float]:
    """Load barthag ratings for tournament teams.

    Searches `data/raw/historical/torvik_{year}.json` first, then
    `{data_dir}/torvik_{year}.json` (defaults to `data/raw/`). The file is
    validated for pre-tournament provenance — a missing or non-pre_tournament
    `data_type` field raises `LeakageError` rather than silently ingesting
    potentially look-ahead-biased data.

    Teams that exist in `seeds` but are absent from the Torvik file fall back
    to a seed-based estimate: `max(0.10, 1 - seed * 0.04)`. This is a crude
    floor that keeps MC simulation well-defined when scraping misses a team.

    Returns: dict of team_id -> barthag in [0.0, 1.0].
    """
    barthag: Dict[str, float] = {}
    search_dirs = [HIST_DIR, Path(data_dir) if data_dir else DATA_DIR]

    for prefix in search_dirs:
        path = prefix / f"torvik_{year}.json"
        if not path.exists():
            continue
        with open(path) as f:
            data = json.load(f)
        _validate_pretournament(data, path)
        for t in data.get("teams", []):
            tid = t.get("team_id", "")
            b = t.get("barthag")
            if tid in seeds and b is not None:
                barthag[tid] = float(b)
        break

    for tid, seed in seeds.items():
        if tid not in barthag:
            barthag[tid] = max(0.10, 1.0 - seed * 0.04)

    return barthag


def build_torvik_probabilities(
    seeds: Dict[str, int],
    barthag: Dict[str, float],
) -> Dict[Tuple[str, str], float]:
    """Pairwise win probabilities for every team pair, via the canonical log5.

    Return format matches `src.prediction.seed_probabilities.build_seed_probabilities`:
    both orientations of every pair are included, with `probs[(a, b)] + probs[(b, a)] == 1`.
    """
    return PairwiseProbabilities.from_ratings(
        barthag, source=f"log5(torvik_barthag), {len(seeds)} teams"
    ).as_dict()


def build_torvik_round_probabilities(
    seeds: Dict[str, int],
    regions: Dict[str, str],
    barthag: Dict[str, float],
    n_sims: int = 10000,
):
    """Per-round advancement probabilities via bracket Monte Carlo.

    Delegates to `scripts.mc_pool_backtest.build_torvik_round_probabilities`
    — the same function the production 15-year backtest uses — so this
    module has no independent copy of the bracket-simulation loop to drift
    from it. That function's RNG seed is fixed internally (42), not
    parameterized.
    """
    from scripts.mc_pool_backtest import (
        build_torvik_round_probabilities as _backtest_build_torvik_round_probabilities,
    )

    return _backtest_build_torvik_round_probabilities(seeds, regions, barthag, n_sims=n_sims)
