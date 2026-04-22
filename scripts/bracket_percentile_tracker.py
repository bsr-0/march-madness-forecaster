"""Bracket percentile tracker: track how each bracket archetype performs across years.

For each year, generate 500 brackets, tag each by its expected_score percentile
(model confidence), and score against reality. Then aggregate by percentile bin
across all 15 years to answer: "does the 90th-percentile-confidence bracket
consistently outscore the 50th-percentile bracket?"

Additional metrics per bracket:
  - expected_score: model confidence (sum of pick_prob × round_pts)
  - leverage: total EV-edge vs ESPN public picks (contrarian-ness)
  - f4_correct: how many Final Four teams the bracket got right
  - champ_correct: did the bracket pick the actual champion (0/1)

Bins: deciles (0-10%, 10-20%, ..., 90-100%) by expected_score.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts._common import load_tournament_results  # noqa: E402
from scripts.o25_g3_diversity_diagnostic import (  # noqa: E402
    build_first_round_matchups,
    build_seed_probabilities,
    build_torvik_round_probabilities,
    derive_f4_region_pairing,
    load_seeds_and_regions,
    resolve_first_four,
    _load_barthag,
)
from scripts.mc_pool_backtest import (  # noqa: E402
    sample_model_brackets,
    sample_f4_first_brackets,
    sample_e8_first_brackets,
)
from src.simulation.pool_competition import (  # noqa: E402
    actual_winners_by_round,
    picks_by_round,
    score_brackets_team_identity,
)

YEARS = list(range(2011, 2027))
YEARS = [y for y in YEARS if y != 2020]
N_BRACKETS = 500
ESPN_SCORING = {"R64": 10, "R32": 20, "S16": 40, "E8": 80, "F4": 160, "CHAMP": 320}
GEN_SEED = 12345
DECAY = 0.85
N_BINS = 10  # deciles

# ESPN public pick rates for leverage computation
ESPN_PICK_RATES = {
    "R64": {1: 0.99, 2: 0.94, 3: 0.85, 4: 0.79, 5: 0.64, 6: 0.63,
            7: 0.61, 8: 0.49, 9: 0.51, 10: 0.39, 11: 0.37, 12: 0.36,
            13: 0.21, 14: 0.15, 15: 0.06, 16: 0.01},
    "R32": {1: 0.82, 2: 0.65, 3: 0.51, 4: 0.43, 5: 0.28, 6: 0.24,
            7: 0.18, 8: 0.16, 9: 0.10, 10: 0.08, 11: 0.08, 12: 0.06,
            13: 0.03, 14: 0.02, 15: 0.01, 16: 0.00},
    "S16": {1: 0.55, 2: 0.35, 3: 0.23, 4: 0.16, 5: 0.10, 6: 0.08,
            7: 0.06, 8: 0.04, 9: 0.03, 10: 0.02, 11: 0.02, 12: 0.01,
            13: 0.00, 14: 0.00, 15: 0.00, 16: 0.00},
    "E8":  {1: 0.34, 2: 0.17, 3: 0.10, 4: 0.07, 5: 0.04, 6: 0.03,
            7: 0.02, 8: 0.01, 9: 0.01, 10: 0.01, 11: 0.00, 12: 0.00,
            13: 0.00, 14: 0.00, 15: 0.00, 16: 0.00},
    "F4":  {1: 0.21, 2: 0.08, 3: 0.05, 4: 0.03, 5: 0.02, 6: 0.01,
            7: 0.01, 8: 0.00, 9: 0.00, 10: 0.00, 11: 0.00, 12: 0.00,
            13: 0.00, 14: 0.00, 15: 0.00, 16: 0.00},
    "CHAMP": {1: 0.11, 2: 0.04, 3: 0.02, 4: 0.01, 5: 0.01, 6: 0.00,
              7: 0.00, 8: 0.00, 9: 0.00, 10: 0.00, 11: 0.00, 12: 0.00,
              13: 0.00, 14: 0.00, 15: 0.00, 16: 0.00},
}

MODES = {
    "torvik": lambda fr, rp, rng, s, r: sample_model_brackets(fr, rp, N_BRACKETS, rng),
    "f4_first_tv": lambda fr, rp, rng, s, r: sample_f4_first_brackets(fr, rp, N_BRACKETS, rng, s, r),
    "e8_first_tv": lambda fr, rp, rng, s, r: sample_e8_first_brackets(fr, rp, N_BRACKETS, rng, s, r),
}


def _compute_metrics(candidates, first_round, seeds, round_probs, real_winners):
    """Compute per-bracket metrics."""
    n = candidates.shape[0]

    actual_scores = score_brackets_team_identity(
        candidates, real_winners, first_round, ESPN_SCORING,
    )

    expected_scores = np.zeros(n)
    leverages = np.zeros(n)
    f4_correct = np.zeros(n)
    champ_correct = np.zeros(n)

    actual_f4 = real_winners.get("F4", set())
    actual_champ = real_winners.get("CHAMP", set())

    for b in range(n):
        bp = picks_by_round(candidates[b], first_round)

        # Expected score
        exp = 0.0
        for rnd, pts in ESPN_SCORING.items():
            for team in bp.get(rnd, set()):
                if team in round_probs:
                    exp += round_probs[team].get(rnd, 0.0) * pts
        expected_scores[b] = exp

        # Leverage: sum of (model_prob - espn_public_prob) × pts for each pick
        lev = 0.0
        for rnd, pts in ESPN_SCORING.items():
            for team in bp.get(rnd, set()):
                model_prob = round_probs.get(team, {}).get(rnd, 0.0)
                seed = seeds.get(team, 16)
                public_prob = ESPN_PICK_RATES.get(rnd, {}).get(seed, 0.0)
                lev += (model_prob - public_prob) * pts
        leverages[b] = lev

        # F4 correct
        bracket_f4 = bp.get("F4", set())
        f4_correct[b] = len(bracket_f4 & actual_f4)

        # Champion correct
        bracket_champ = bp.get("CHAMP", set())
        champ_correct[b] = 1.0 if bracket_champ & actual_champ else 0.0

    return {
        "actual_score": actual_scores,
        "expected_score": expected_scores,
        "leverage": leverages,
        "f4_correct": f4_correct,
        "champ_correct": champ_correct,
    }


def _bin_by_percentile(values, metric_to_bin_by, n_bins=N_BINS):
    """Bin brackets into percentile groups by a metric, return per-bin stats."""
    percentile_edges = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(metric_to_bin_by, percentile_edges)

    bins = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (metric_to_bin_by >= lo) & (metric_to_bin_by <= hi)
        else:
            mask = (metric_to_bin_by >= lo) & (metric_to_bin_by < hi)
        if mask.sum() == 0:
            continue
        bins.append({
            "pct_lo": int(percentile_edges[i]),
            "pct_hi": int(percentile_edges[i + 1]),
            "n": int(mask.sum()),
            "mean": float(values[mask].mean()),
            "median": float(np.median(values[mask])),
            "max": float(values[mask].max()),
            "min": float(values[mask].min()),
        })
    return bins


def _run_year(year, mode_name, mode_sampler):
    games = load_tournament_results(year)
    if not games:
        return None

    seeds, regions = load_seeds_and_regions(year)
    resolve_first_four(games, seeds, regions)
    try:
        region_order = derive_f4_region_pairing(games, regions)
    except ValueError:
        return None

    first_round = build_first_round_matchups(seeds, regions, region_order=region_order)
    if len(first_round) != 64:
        return None

    barthag = _load_barthag(year, seeds)
    round_probs = build_torvik_round_probabilities(seeds, regions, barthag)
    real_winners = actual_winners_by_round(games)

    rng = np.random.default_rng(GEN_SEED)
    candidates = mode_sampler(first_round, round_probs, rng, seeds, regions)

    metrics = _compute_metrics(candidates, first_round, seeds, round_probs, real_winners)

    # Bin actual_score by expected_score percentile
    score_by_exp = _bin_by_percentile(
        metrics["actual_score"], metrics["expected_score"],
    )

    # Bin actual_score by leverage percentile
    score_by_lev = _bin_by_percentile(
        metrics["actual_score"], metrics["leverage"],
    )

    # Bin by expected_score — also show leverage, f4_correct, champ_correct per bin
    exp_bins_detail = []
    percentile_edges = np.linspace(0, 100, N_BINS + 1)
    bin_edges = np.percentile(metrics["expected_score"], percentile_edges)
    for i in range(N_BINS):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == N_BINS - 1:
            mask = (metrics["expected_score"] >= lo) & (metrics["expected_score"] <= hi)
        else:
            mask = (metrics["expected_score"] >= lo) & (metrics["expected_score"] < hi)
        if mask.sum() == 0:
            continue
        exp_bins_detail.append({
            "pct_lo": int(percentile_edges[i]),
            "pct_hi": int(percentile_edges[i + 1]),
            "n": int(mask.sum()),
            "actual_score_mean": float(metrics["actual_score"][mask].mean()),
            "actual_score_median": float(np.median(metrics["actual_score"][mask])),
            "actual_score_max": float(metrics["actual_score"][mask].max()),
            "leverage_mean": float(metrics["leverage"][mask].mean()),
            "f4_correct_mean": float(metrics["f4_correct"][mask].mean()),
            "champ_correct_pct": float(metrics["champ_correct"][mask].mean() * 100),
            "expected_score_range": f"[{lo:.0f}, {hi:.0f}]",
        })

    return {
        "score_by_expected": score_by_exp,
        "score_by_leverage": score_by_lev,
        "detail_bins": exp_bins_detail,
        "overall_mean": float(metrics["actual_score"].mean()),
        "overall_best": float(metrics["actual_score"].max()),
        "overall_champ_correct_pct": float(metrics["champ_correct"].mean() * 100),
        "overall_f4_correct_mean": float(metrics["f4_correct"].mean()),
    }


def main() -> int:
    print("Bracket Percentile Tracker")
    print(f"  {N_BRACKETS} brackets/year, {N_BINS} decile bins by expected_score")
    print(f"  Metrics: actual_score, leverage, f4_correct, champ_correct")

    years_sorted = sorted(YEARS)
    most_recent = max(years_sorted)
    weights = {y: DECAY ** (most_recent - y) for y in years_sorted}
    total_weight = sum(weights.values())

    all_output = {}

    for mode_name, mode_sampler in MODES.items():
        print(f"\n{'=' * 80}")
        print(f"MODE: {mode_name}")
        print(f"{'=' * 80}")

        year_results = {}
        for year in years_sorted:
            t0 = time.time()
            r = _run_year(year, mode_name, mode_sampler)
            if r is None:
                print(f"  {year}: SKIP")
                continue
            year_results[year] = r
            print(f"  {year}: best={r['overall_best']:.0f}  mean={r['overall_mean']:.0f}  "
                  f"champ%={r['overall_champ_correct_pct']:.1f}  "
                  f"f4={r['overall_f4_correct_mean']:.2f}  ({time.time()-t0:.1f}s)")

        # Aggregate: weighted average actual_score per decile across years
        print(f"\n  --- Decile Performance (expected_score bins, weighted avg) ---")
        print(f"  {'Decile':<10} {'Actual':>8} {'Actual':>8} {'Actual':>8} "
              f"{'Leverage':>10} {'F4 Corr':>8} {'Champ%':>8}")
        print(f"  {'':10} {'Mean':>8} {'Median':>8} {'Max':>8} "
              f"{'Mean':>10} {'Mean':>8} {'':>8}")

        for bin_idx in range(N_BINS):
            w_mean = 0.0
            w_median = 0.0
            w_max = 0.0
            w_lev = 0.0
            w_f4 = 0.0
            w_champ = 0.0
            w_total = 0.0

            for year in year_results:
                bins = year_results[year]["detail_bins"]
                if bin_idx < len(bins):
                    b = bins[bin_idx]
                    w = weights[year]
                    w_mean += b["actual_score_mean"] * w
                    w_median += b["actual_score_median"] * w
                    w_max += b["actual_score_max"] * w
                    w_lev += b["leverage_mean"] * w
                    w_f4 += b["f4_correct_mean"] * w
                    w_champ += b["champ_correct_pct"] * w
                    w_total += w

            if w_total > 0:
                label = f"{bin_idx * 10}-{(bin_idx + 1) * 10}%"
                print(f"  {label:<10} {w_mean/w_total:>8.0f} {w_median/w_total:>8.0f} "
                      f"{w_max/w_total:>8.0f} {w_lev/w_total:>+10.1f} "
                      f"{w_f4/w_total:>8.2f} {w_champ/w_total:>7.1f}%")

        # Per-year breakdown for recent years
        print(f"\n  --- Per-Year Decile Scores (actual_score mean per decile) ---")
        header = f"  {'Decile':<10}"
        for year in years_sorted[-6:]:
            if year in year_results:
                header += f"  {year:>6}"
        header += "  WtAvg"
        print(header)

        for bin_idx in range(N_BINS):
            label = f"{bin_idx * 10}-{(bin_idx + 1) * 10}%"
            row = f"  {label:<10}"
            for year in years_sorted[-6:]:
                if year in year_results:
                    bins = year_results[year]["detail_bins"]
                    if bin_idx < len(bins):
                        row += f"  {bins[bin_idx]['actual_score_mean']:>6.0f}"
                    else:
                        row += f"  {'N/A':>6}"

            # Weighted average
            w_sum = 0.0
            w_total = 0.0
            for year in year_results:
                bins = year_results[year]["detail_bins"]
                if bin_idx < len(bins):
                    w = weights[year]
                    w_sum += bins[bin_idx]["actual_score_mean"] * w
                    w_total += w
            if w_total > 0:
                row += f"  {w_sum/w_total:>6.0f}"
            print(row)

        all_output[mode_name] = {str(k): v for k, v in year_results.items()}

    out_path = PROJECT_ROOT / "artifacts" / "bracket_percentile_tracker_2026-04-19.json"
    out = {
        "description": "Bracket percentile tracker: decile performance across years",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {
            "n_brackets": N_BRACKETS,
            "n_bins": N_BINS,
            "gen_seed": GEN_SEED,
            "decay": DECAY,
            "years": years_sorted,
        },
        "modes": all_output,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Artifact: {out_path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
