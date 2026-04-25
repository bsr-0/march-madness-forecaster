"""Phase-1 migration: convert artifacts/backtest_brackets/*.json into the
strategy-cache NPZ schema.

Demonstrates the cache primitives end-to-end against known artifacts (3 production
modes x 4 years = 12 entries). No model runs. Picks are stored as (50, 63) uint16
arrays in alphabetical-within-round order, matching the source JSON convention.

Source brackets were generated at an unknown RNG seed on 2026-04-18, so they're
catalogued with rng_seed='frozen-2026-04-18'. Phase 2+ caches will use seed=42
and be bit-exact reproducible.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.strategy_cache import (
    Manifest,
    compute_cache_key,
    load_team_lookup,
    save_brackets,
    save_team_lookup,
)

SOURCE_DIR = PROJECT_ROOT / "artifacts" / "backtest_brackets"
ROUND_ORDER = ["R64", "R32", "S16", "E8", "F4", "CHAMP"]
ROUND_SIZES = {"R64": 32, "R32": 16, "S16": 8, "E8": 4, "F4": 2, "CHAMP": 1}
RNG_SEED_LABEL = "frozen-2026-04-18"


def git_sha_short() -> str:
    return subprocess.check_output(["git", "rev-parse", "--short=12", "HEAD"], cwd=PROJECT_ROOT).decode().strip()


def collect_team_names(source_files: list[Path]) -> list[str]:
    names: set[str] = set()
    for path in source_files:
        data = json.loads(path.read_text())
        for mode_block in data["modes"]:
            for bracket in mode_block["brackets"]:
                for round_name in ROUND_ORDER:
                    names.update(bracket["picks"][round_name])
    return sorted(names)


def build_team_lookup(source_files: list[Path]) -> dict[str, int]:
    existing = load_team_lookup()
    if existing:
        return existing
    names = collect_team_names(source_files)
    lookup = {name: i for i, name in enumerate(names)}
    save_team_lookup(lookup)
    return lookup


def bracket_to_array(bracket: dict, lookup: dict[str, int]) -> np.ndarray:
    out = np.zeros(63, dtype=np.uint16)
    cursor = 0
    for round_name in ROUND_ORDER:
        size = ROUND_SIZES[round_name]
        picks = sorted(bracket["picks"][round_name])
        if len(picks) != size:
            raise ValueError(f"round {round_name} expected {size} picks, got {len(picks)}")
        for i, name in enumerate(picks):
            out[cursor + i] = lookup[name]
        cursor += size
    return out


def main() -> None:
    source_files = sorted(SOURCE_DIR.glob("backtest_brackets_*.json"))
    if not source_files:
        raise SystemExit(f"no source files in {SOURCE_DIR}")

    code_version = git_sha_short()
    data_version = "backtest_brackets_2026-04-18"
    model_version = "shape-baseline-pre-O27"

    lookup = build_team_lookup(source_files)
    print(f"team-id lookup: {len(lookup)} unique teams")

    manifest = Manifest.load()
    n_added = 0
    for source in source_files:
        data = json.loads(source.read_text())
        year = int(data["year"])
        for mode_block in data["modes"]:
            strategy_id = mode_block["mode"]
            brackets_2d = np.stack([bracket_to_array(b, lookup) for b in mode_block["brackets"]])
            cache_key = compute_cache_key(
                strategy_id=strategy_id,
                year=year,
                rng_seed=RNG_SEED_LABEL,
                code_version=code_version,
                data_version=data_version,
                model_version=model_version,
            )
            entry = save_brackets(
                cache_key,
                brackets_2d,
                strategy_id=strategy_id,
                year=year,
                rng_seed=RNG_SEED_LABEL,
                code_version=code_version,
                data_version=data_version,
                model_version=model_version,
                provenance={"snapshot_source": str(source.relative_to(PROJECT_ROOT))},
            )
            manifest.append(entry)
            n_added += 1
            print(f"  cached {strategy_id} year={year} key={cache_key}")
    manifest.save()
    print(f"\nwrote {n_added} entries to {manifest.entries[0].artifact_path.split('/')[0]}/")
    print(f"manifest at artifacts/strategy_cache/manifest.json")


if __name__ == "__main__":
    main()
