"""Consolidate per-year ESPN points for pool/exhaustive/stat/chalk into
docs/data/loyo_points.json for the UI.

Reads:
  - artifacts/backtest_runs/mc_pool_backtest_*_full_capture.txt: full stdout
    from `python -m scripts.mc_pool_backtest --modes meta_region_poolaware
    meta_exhaustive meta_region --team-identity --n-opponents 29 --opponent
    pool`, ending in a printed Python list of per-(year, mode) result dicts
    (run_backtest()'s return value).

    NOTE: the auto-log file mc_pool_backtest.py normally writes to
    artifacts/backtest_runs/mc_pool_backtest_<ts>.txt does NOT contain this
    line — main() restores sys.stdout (closing the log Tee) before whatever
    prints the return value, so that line only exists in a full terminal
    capture. Re-running this pipeline means saving the full stdout of the
    mc_pool_backtest invocation (not just its own auto-log) to a
    *_full_capture.txt file here first. The proper fix is to have
    mc_pool_backtest.py's main() itself write a results JSON before
    restoring stdout — worth doing if this becomes a recurring need.
  - artifacts/backtest_runs/chalk_loyo_points.json: chalk's per-year points
    from scripts/compute_chalk_loyo_points.py (deterministic, no backtest
    run needed).
"""

import ast
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RUN_LOG_DIR = PROJECT_ROOT / "artifacts" / "backtest_runs"
CHALK_JSON = RUN_LOG_DIR / "chalk_loyo_points.json"
OUT_PATH = PROJECT_ROOT / "docs" / "data" / "loyo_points.json"

MODE_TO_UI_KEY = {
    "meta_region_poolaware": "pool",
    "meta_exhaustive": "exhaustive",
    "meta_region": "stat",
}


def latest_full_capture() -> Path:
    candidates = sorted(RUN_LOG_DIR.glob("mc_pool_backtest_*_full_capture.txt"))
    if not candidates:
        print(f"ERROR: no *_full_capture.txt files found in {RUN_LOG_DIR} — see module docstring.")
        sys.exit(1)
    return candidates[-1]


def parse_results_list(text: str):
    """Find the printed results list — scan from the end, skip non-parsing lines."""
    for line in reversed(text.splitlines()):
        line = line.strip()
        if not line.startswith("["):
            continue
        try:
            return ast.literal_eval(line)
        except (ValueError, SyntaxError):
            continue
    return None


def main():
    capture_path = latest_full_capture()
    print(f"  Reading {capture_path}")
    records = parse_results_list(capture_path.read_text())
    if records is None:
        print("ERROR: could not find a parseable results list in the capture file")
        sys.exit(1)

    points_by_key: dict[str, dict[str, float]] = {"pool": {}, "exhaustive": {}, "stat": {}, "chalk": {}}
    for r in records:
        key = MODE_TO_UI_KEY.get(r["mode"])
        if key is None:
            continue
        points_by_key[key][str(r["year"])] = round(r["best_score"], 1)

    chalk_data = json.loads(CHALK_JSON.read_text())
    points_by_key["chalk"] = {yr: round(v, 1) for yr, v in chalk_data["points_by_year"].items()}

    years = sorted({y for mode_points in points_by_key.values() for y in mode_points}, key=int)

    output = {
        "scoring": "ESPN team-identity (R64=10,R32=20,S16=40,E8=80,F4=160,CHAMP=320)",
        "opponent_pool": "N=29 (real pool history where available, ESPN public picks otherwise)",
        "years": years,
        "points_by_strategy": points_by_key,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    for key, pts in points_by_key.items():
        vals = list(pts.values())
        print(f"  {key:12s} n={len(vals):2d}  mean={sum(vals) / len(vals):7.1f}  min={min(vals):6.1f}  max={max(vals):6.1f}")
    print(f"\nWritten to {OUT_PATH}")


if __name__ == "__main__":
    main()
