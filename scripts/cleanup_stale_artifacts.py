#!/usr/bin/env python3
"""Clean up stale, empty, and broken backtest/validation artifacts.

This script addresses the CRITICAL issue where 16+ backtest artifact files
contain empty ``results: []`` arrays due to silent pipeline failures, and
the main ``unified_backtest_2025.json`` is a stale empty run while valid
data exists in the ``_fresh`` / ``_pipeline`` variants.

Actions performed:
  1. Delete all ``backtest_unified_smoke_2025*.json`` files (empty results).
  2. Replace ``unified_backtest_2025.json`` with data from the best valid
     variant (``_fresh`` or ``_pipeline``).
  3. Delete leftover diagnostic/intermediate backtest variants.
  4. Report on ``market_validation_2026.json`` status.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = REPO_ROOT / "artifacts"


def _load_json(path: Path) -> dict | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def _has_results(data: dict | None) -> bool:
    """Return True if the artifact contains non-empty backtest results."""
    if data is None:
        return False
    results = data.get("results", [])
    return isinstance(results, list) and len(results) > 0


def _count_calibration_games(data: dict) -> int:
    """Sum n_games across calibration results to gauge quality."""
    return sum(
        r.get("n_games", 0)
        for r in data.get("results", [])
        if r.get("mode") == "calibration"
    )


def cleanup_smoke_backtests() -> list[str]:
    """Delete all backtest_unified_smoke_2025*.json files with empty results."""
    deleted = []
    for path in sorted(ARTIFACTS_DIR.glob("backtest_unified_smoke_2025*.json")):
        data = _load_json(path)
        if not _has_results(data):
            path.unlink()
            deleted.append(path.name)
            print(f"  DELETED (empty results): {path.name}")
        else:
            print(f"  KEPT (has results): {path.name}")
    return deleted


def fix_unified_backtest_2025() -> str | None:
    """Replace empty unified_backtest_2025.json with best valid variant."""
    main_path = ARTIFACTS_DIR / "unified_backtest_2025.json"
    main_data = _load_json(main_path)

    if _has_results(main_data):
        n_games = _count_calibration_games(main_data)
        print(f"  unified_backtest_2025.json already has results ({n_games} calibration games). No action needed.")
        return None

    # Find the best valid variant to promote
    candidates = [
        ("unified_backtest_2025_fresh.json", ARTIFACTS_DIR / "unified_backtest_2025_fresh.json"),
        ("unified_backtest_2025_pipeline.json", ARTIFACTS_DIR / "unified_backtest_2025_pipeline.json"),
        ("unified_backtest_2025_pipeline_progress.json", ARTIFACTS_DIR / "unified_backtest_2025_pipeline_progress.json"),
    ]

    best_source = None
    best_games = 0
    for name, path in candidates:
        data = _load_json(path)
        if _has_results(data):
            games = _count_calibration_games(data)
            if games > best_games:
                best_games = games
                best_source = (name, path, data)

    if best_source is None:
        print("  WARNING: No valid variant found to replace empty unified_backtest_2025.json")
        return None

    source_name, source_path, _ = best_source
    shutil.copy2(source_path, main_path)
    print(f"  REPLACED unified_backtest_2025.json with {source_name} ({best_games} calibration games)")
    return source_name


def cleanup_intermediate_variants() -> list[str]:
    """Delete intermediate/diagnostic backtest variants that are redundant."""
    # These are iterative debug runs that are no longer needed
    patterns_to_clean = [
        "unified_backtest_2025_diagnostic*.json",
    ]
    deleted = []
    for pattern in patterns_to_clean:
        for path in sorted(ARTIFACTS_DIR.glob(pattern)):
            data = _load_json(path)
            if not _has_results(data):
                path.unlink()
                deleted.append(path.name)
                print(f"  DELETED (empty diagnostic): {path.name}")
            else:
                print(f"  KEPT (has results): {path.name}")
    return deleted


def report_market_validation():
    """Report on market_validation_2026.json status."""
    path = ARTIFACTS_DIR / "market_validation_2026.json"
    data = _load_json(path)
    if data is None:
        print("  market_validation_2026.json: MISSING")
        return

    status = data.get("status", "unknown")
    reason = data.get("reason", "")
    if status == "skipped":
        print(f"  market_validation_2026.json: SKIPPED - {reason}")
        print("  Root cause: No betting odds cache at data/raw/betting_odds/ and/or")
        print("  no model championship odds in the production report.")
        print("  To fix: Run the betting odds scraper before the production pipeline,")
        print("  or provide betting_odds_2026.json in data/raw/betting_odds/.")
    elif status == "completed":
        print(f"  market_validation_2026.json: OK (completed)")
    else:
        print(f"  market_validation_2026.json: {status} - {reason}")


def main():
    print("=" * 70)
    print("Stale Artifact Cleanup")
    print("=" * 70)

    if not ARTIFACTS_DIR.exists():
        print(f"Artifacts directory not found: {ARTIFACTS_DIR}")
        return 1

    # 1. Clean up empty smoke backtests
    print("\n[1/4] Cleaning empty smoke backtest files...")
    smoke_deleted = cleanup_smoke_backtests()
    if not smoke_deleted:
        print("  No empty smoke backtests found.")

    # 2. Fix the main unified_backtest_2025.json
    print("\n[2/4] Fixing unified_backtest_2025.json...")
    promoted = fix_unified_backtest_2025()

    # 3. Clean up diagnostic/intermediate variants
    print("\n[3/4] Cleaning intermediate diagnostic variants...")
    diag_deleted = cleanup_intermediate_variants()
    if not diag_deleted:
        print("  No empty diagnostic variants found.")

    # 4. Report on market validation
    print("\n[4/4] Market validation status...")
    report_market_validation()

    # Summary
    total_deleted = len(smoke_deleted) + len(diag_deleted)
    print("\n" + "=" * 70)
    print(f"Summary: {total_deleted} stale files deleted" +
          (f", unified_backtest_2025.json promoted from {promoted}" if promoted else ""))
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
