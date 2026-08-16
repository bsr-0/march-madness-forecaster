"""Phase-1 migration: convert artifacts/backtest_brackets/*.json AND
data/raw/historical/tournament_results_*.json into the strategy-cache NPZ schema.

Two artifact kinds populated:
  - brackets/{cache_key}.npz: 3 production modes x 4 years (2023-2026) from
    the existing backtest_brackets snapshots. Catalogued with
    rng_seed='frozen-2026-04-18' since the original RNG seed isn't recorded.
  - outcomes/{year}.npz: ground-truth winners for every year that has a
    tournament_results file (2005-2026 ex 2020).

Pick / winner ordering is alphabetical within each round, so per-round
validation reduces to (cached_brackets == cached_outcomes[None, :]).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.strategy_cache import (
    Manifest,
    compute_cache_key,
    load_team_lookup,
    save_brackets,
    save_outcomes,
    save_team_lookup,
)

SOURCE_DIR = PROJECT_ROOT / "artifacts" / "backtest_brackets"
RESULTS_DIR = PROJECT_ROOT / "data" / "raw" / "historical"
ROUND_ORDER = ["R64", "R32", "S16", "E8", "F4", "CHAMP"]
ROUND_SIZES = {"R64": 32, "R32": 16, "S16": 8, "E8": 4, "F4": 2, "CHAMP": 1}
# Tournament-results schema uses "NCG" for the championship game; everything
# else aligns with ROUND_ORDER. "FF" entries are First Four play-ins, skipped.
RESULTS_ROUND_TO_BRACKET = {
    "R64": "R64",
    "R32": "R32",
    "S16": "S16",
    "E8": "E8",
    "F4": "F4",
    "NCG": "CHAMP",
}
RNG_SEED_LABEL = "frozen-2026-04-18"


def git_sha_short() -> str:
    return subprocess.check_output(["git", "rev-parse", "--short=12", "HEAD"], cwd=PROJECT_ROOT).decode().strip()


def load_results(path: Path) -> dict:
    """Load a tournament-results document from `path`.

    `path` is a `tournament_context_{year}.json` (consolidated format,
    "results" sub-key extracted) or an old-style `tournament_results_{year}.json`
    (top-level `{"year", "games"}` shape) — both return the same shape."""
    data = json.loads(path.read_text())
    if path.name.startswith("tournament_context_"):
        return data["results"]
    return data


def collect_team_names(bracket_files: list[Path], result_files: list[Path]) -> list[str]:
    names: set[str] = set()
    for path in bracket_files:
        data = json.loads(path.read_text())
        for mode_block in data["modes"]:
            for bracket in mode_block["brackets"]:
                for round_name in ROUND_ORDER:
                    names.update(bracket["picks"][round_name])
    for path in result_files:
        data = load_results(path)
        for game in data["games"]:
            if game.get("round_name") == "FF":
                continue
            names.add(game["team1_id"])
            names.add(game["team2_id"])
    return sorted(names)


def build_team_lookup(bracket_files: list[Path], result_files: list[Path]) -> dict[str, int]:
    existing = load_team_lookup()
    if existing:
        return existing
    names = collect_team_names(bracket_files, result_files)
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


def winners_array(games: list[dict], lookup: dict[str, int]) -> np.ndarray:
    """Extract per-round winners as alphabetically-sorted (63,) uint16."""
    by_round: dict[str, list[str]] = {r: [] for r in ROUND_ORDER}
    for g in games:
        bracket_round = RESULTS_ROUND_TO_BRACKET.get(g["round_name"])
        if bracket_round is None:  # FF play-ins
            continue
        winner = g["team1_id"] if g["team1_won"] else g["team2_id"]
        by_round[bracket_round].append(winner)
    out = np.zeros(63, dtype=np.uint16)
    cursor = 0
    for round_name in ROUND_ORDER:
        size = ROUND_SIZES[round_name]
        winners = sorted(by_round[round_name])
        if len(winners) != size:
            raise ValueError(f"round {round_name} expected {size} winners, got {len(winners)}")
        for i, name in enumerate(winners):
            out[cursor + i] = lookup[name]
        cursor += size
    return out


def main() -> None:
    bracket_files = sorted(SOURCE_DIR.glob("backtest_brackets_*.json"))
    result_files = [
        p for p in sorted(RESULTS_DIR.glob("tournament_context_*.json")) if "results" in json.loads(p.read_text())
    ]
    result_files += sorted(RESULTS_DIR.glob("tournament_results_*.json"))
    if not bracket_files:
        raise SystemExit(f"no bracket files in {SOURCE_DIR}")
    if not result_files:
        raise SystemExit(f"no result files in {RESULTS_DIR}")

    code_version = git_sha_short()
    bracket_data_version = "backtest_brackets_2026-04-18"
    results_data_version = "historical_tournament_results"
    model_version = "shape-baseline-pre-O27"

    lookup = build_team_lookup(bracket_files, result_files)
    print(f"team-id lookup: {len(lookup)} unique teams")

    manifest = Manifest.load()

    # --- brackets -------------------------------------------------------
    n_brackets = 0
    for source in bracket_files:
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
                data_version=bracket_data_version,
                model_version=model_version,
            )
            entry = save_brackets(
                cache_key,
                brackets_2d,
                strategy_id=strategy_id,
                year=year,
                rng_seed=RNG_SEED_LABEL,
                code_version=code_version,
                data_version=bracket_data_version,
                model_version=model_version,
                provenance={"snapshot_source": str(source.relative_to(PROJECT_ROOT))},
            )
            manifest.append(entry)
            n_brackets += 1
            print(f"  brackets: {strategy_id} year={year} key={cache_key}")

    # --- outcomes -------------------------------------------------------
    n_outcomes = 0
    for source in result_files:
        data = load_results(source)
        year = int(data["year"])
        winners = winners_array(data["games"], lookup)
        entry = save_outcomes(
            year,
            winners,
            code_version=code_version,
            data_version=results_data_version,
            provenance={"source": str(source.relative_to(PROJECT_ROOT))},
        )
        manifest.append(entry)
        n_outcomes += 1
        print(f"  outcomes: year={year}")

    manifest.save()
    print(f"\nwrote {n_brackets} bracket entries + {n_outcomes} outcome entries")
    print("manifest at artifacts/strategy_cache/manifest.json")


if __name__ == "__main__":
    main()
