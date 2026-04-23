"""Compare LOYO Brier scores: baseline vs all-features-wired.

Now that conference maps, coach experience, and market features are
properly wired, compare the new results against the saved April 10
baseline (backtest_result.json) which had these features as zeros.

Also runs a with-market vs without-market A/B within the new code.
"""
import json
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline.config import ForecastConfig
from src.ml.evaluation.rdof_audit import HoldoutEvaluator

HISTORICAL_DIR = "data/raw/historical"
DEV_YEARS = [2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025]


def run_holdout(label: str, use_market: bool):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    config = ForecastConfig(use_market_features=use_market)
    evaluator = HoldoutEvaluator(
        historical_dir=HISTORICAL_DIR,
        all_years=DEV_YEARS,
        dev_years=DEV_YEARS,
    )

    results = {}
    for year in DEV_YEARS:
        if year not in evaluator.all_years:
            print(f"  Skipping {year} (no data)")
            continue
        t0 = time.time()
        try:
            report = evaluator.evaluate_holdout_year(year, config)
            elapsed = time.time() - t0
            brier = report.brier_score
            n = report.n_games
            print(f"  {year}: Brier={brier:.4f}  (n={n}, {elapsed:.1f}s)")
            results[year] = {"brier": brier, "n_games": n}
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  {year}: FAILED ({e.__class__.__name__}: {e}) {elapsed:.1f}s")

    years = sorted(results.keys())
    briers = [results[y]["brier"] for y in years]
    avg = sum(briers) / len(briers) if briers else 0
    print(f"\n  Mean Brier: {avg:.5f}  ({len(years)} years)")
    return results, avg


def load_old_baseline():
    """Load pre-April-21 baseline from saved backtest_result.json."""
    path = Path("artifacts/backtest_result.json")
    if not path.exists():
        return None, None
    with open(path) as f:
        data = json.load(f)
    results = {}
    for yr in data.get("aggregate_report", {}).get("year_reports", []):
        year = yr["year"]
        brier = yr["global_metrics"]["brier_score"]["estimate"]
        n = yr["global_metrics"]["brier_score"]["n_samples"]
        results[year] = {"brier": brier, "n_games": n}
    briers = [r["brier"] for r in results.values()]
    avg = sum(briers) / len(briers) if briers else 0
    return results, avg


if __name__ == "__main__":
    t_start = time.time()

    # Load saved baseline (pre-Vegas, features 46-55 all zeros)
    old_results, old_avg = load_old_baseline()
    if old_results:
        print(f"\n{'='*60}")
        print(f"  OLD BASELINE (saved April 10, features 46-55 = zeros)")
        print(f"{'='*60}")
        print(f"  Mean Brier: {old_avg:.5f}  ({len(old_results)} years)")

    # Run with conf+coach but NO market features
    no_mkt_results, no_mkt_avg = run_holdout("NEW: conf + coach (no market)", use_market=False)

    # Run with conf+coach+market features
    full_results, full_avg = run_holdout("NEW: conf + coach + market", use_market=True)

    # Comparison table
    print(f"\n{'='*60}")
    print(f"  COMPARISON")
    print(f"{'='*60}")
    if old_avg:
        print(f"  Old baseline (zeros):      {old_avg:.5f}")
    print(f"  New (conf+coach):          {no_mkt_avg:.5f}")
    print(f"  New (conf+coach+market):   {full_avg:.5f}")
    if old_avg:
        print(f"  Improvement (new vs old):  {no_mkt_avg - old_avg:+.5f} ({'BETTER' if no_mkt_avg < old_avg else 'WORSE'})")
        print(f"  Improvement (full vs old): {full_avg - old_avg:+.5f} ({'BETTER' if full_avg < old_avg else 'WORSE'})")
    print(f"  Market delta:              {full_avg - no_mkt_avg:+.5f} ({'BETTER' if full_avg < no_mkt_avg else 'WORSE'})")

    # Per-year breakdown
    common_years = sorted(set(no_mkt_results) & set(full_results))
    print(f"\n  {'Year':>6}  {'Old':>8}  {'NoMkt':>8}  {'Full':>8}  {'Δ full-old':>10}")
    for year in common_years:
        old_b = old_results.get(year, {}).get("brier", float("nan")) if old_results else float("nan")
        nm_b = no_mkt_results[year]["brier"]
        f_b = full_results[year]["brier"]
        d = f_b - old_b if old_results and year in old_results else float("nan")
        print(f"  {year:>6}  {old_b:>8.4f}  {nm_b:>8.4f}  {f_b:>8.4f}  {d:>+10.4f}")

    elapsed = time.time() - t_start
    print(f"\n  Total elapsed: {elapsed:.0f}s")

    out = {
        "old_baseline_avg": old_avg,
        "no_market_avg": no_mkt_avg,
        "full_avg": full_avg,
        "no_market_per_year": no_mkt_results,
        "full_per_year": full_results,
    }
    out_path = "artifacts/market_features_comparison.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Saved to {out_path}")
