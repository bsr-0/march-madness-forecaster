"""Historical mode scoring: which bracket construction mode produces
the highest-scoring brackets against actual tournament outcomes?

No opponent model, no pool simulation — just score brackets against reality.
For each year 2011-2026 (ex 2020) and each mode, generate 50 brackets,
score all against the actual tournament result (team-identity), and report:
  - Best-of-50 score (ceiling)
  - Median bracket score (typical quality)
  - Score the median_score selector would pick (best practical pick)

Apply exponential recency weighting: recent years count more.

Modes tested:
  - seed:           pure seed probability sampling
  - torvik:         torvik barthag-based round probabilities
  - champ_first_tv: lock champion, sample rest from torvik
  - f4_first_tv:    lock F4 per region, sample rest from torvik
  - e8_first_tv:    lock E8 per quadrant, sample rest from torvik
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
    sample_champ_first_brackets,
    sample_e8_first_brackets,
    sample_f4_first_brackets,
    sample_model_brackets,
)
from src.simulation.pool_competition import (  # noqa: E402
    actual_winners_by_round,
    score_brackets_team_identity,
)

YEARS = list(range(2011, 2027))
YEARS = [y for y in YEARS if y != 2020]  # no tournament in 2020
N_BRACKETS = 50
ESPN_SCORING = {"R64": 10, "R32": 20, "S16": 40, "E8": 80, "F4": 160, "CHAMP": 320}
GEN_SEED = 12345
DECAY = 0.85  # exponential decay per year (most recent = 1.0)


def _run_year(year):
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
    torvik_rp = build_torvik_round_probabilities(seeds, regions, barthag)
    seed_rp = {}
    seed_pw = build_seed_probabilities(seeds)
    # Build seed-based round probs (simple: just use seed matchup probs as marginals)
    for tid in first_round:
        s = seeds.get(tid, 16)
        # Approximate round advancement from seed
        seed_rp[tid] = {
            "R64": max(0.01, 1.0 - s * 0.055),
            "R32": max(0.01, 1.0 - s * 0.065) ** 2,
            "S16": max(0.01, 1.0 - s * 0.065) ** 3,
            "E8": max(0.01, 1.0 - s * 0.065) ** 4,
            "F4": max(0.01, 1.0 - s * 0.065) ** 5,
            "CHAMP": max(0.001, 1.0 - s * 0.065) ** 6,
        }

    real_winners = actual_winners_by_round(games)

    modes = {}

    # Seed-based sampling
    rng = np.random.default_rng(GEN_SEED)
    cands = sample_model_brackets(first_round, seed_rp, N_BRACKETS, rng)
    scores = score_brackets_team_identity(cands, real_winners, first_round, ESPN_SCORING)
    modes["seed"] = scores

    # Torvik-based sampling (plain stochastic)
    rng = np.random.default_rng(GEN_SEED)
    cands = sample_model_brackets(first_round, torvik_rp, N_BRACKETS, rng)
    scores = score_brackets_team_identity(cands, real_winners, first_round, ESPN_SCORING)
    modes["torvik"] = scores

    # Champ-first (torvik)
    rng = np.random.default_rng(GEN_SEED)
    cands = sample_champ_first_brackets(first_round, torvik_rp, N_BRACKETS, rng)
    scores = score_brackets_team_identity(cands, real_winners, first_round, ESPN_SCORING)
    modes["champ_first_tv"] = scores

    # F4-first (torvik)
    rng = np.random.default_rng(GEN_SEED)
    cands = sample_f4_first_brackets(first_round, torvik_rp, N_BRACKETS, rng, seeds, regions)
    scores = score_brackets_team_identity(cands, real_winners, first_round, ESPN_SCORING)
    modes["f4_first_tv"] = scores

    # E8-first (torvik)
    rng = np.random.default_rng(GEN_SEED)
    cands = sample_e8_first_brackets(first_round, torvik_rp, N_BRACKETS, rng, seeds, regions)
    scores = score_brackets_team_identity(cands, real_winners, first_round, ESPN_SCORING)
    modes["e8_first_tv"] = scores

    return modes


def main() -> int:
    print("Historical Mode Scoring: which mode produces the best brackets?")
    print(f"  Years: {YEARS[0]}-{YEARS[-1]} ({len(YEARS)} years)")
    print(f"  {N_BRACKETS} brackets per mode per year, team-identity scoring")
    print(f"  Recency decay: {DECAY} per year (most recent = 1.0)")

    mode_names = ["seed", "torvik", "champ_first_tv", "f4_first_tv", "e8_first_tv"]

    # Collect per-year results
    all_results = {}
    for year in YEARS:
        t0 = time.time()
        modes = _run_year(year)
        if modes is None:
            print(f"  {year}: SKIP")
            continue
        all_results[year] = {}
        row_parts = []
        for mode in mode_names:
            s = modes[mode]
            best = float(s.max())
            median = float(np.median(s))
            mean = float(s.mean())
            # Median-score selector: pick the bracket with highest median
            # Since we only have one sim (actual), use mean across hypothetical
            # But here we just have actual scores — pick the bracket with
            # highest score (which IS the best selector against actual)
            median_pick = float(s[np.argmax(s)])  # same as best for single outcome
            all_results[year][mode] = {
                "best_of_50": best,
                "median_bracket": median,
                "mean_bracket": mean,
                "std": float(s.std()),
                "all_scores": s.tolist(),
            }
            row_parts.append(f"{best:>6.0f}/{median:>5.0f}")
        elapsed = time.time() - t0
        print(f"  {year}: " + "  ".join(row_parts) + f"  ({elapsed:.1f}s)")

    # Compute weighted averages
    years_sorted = sorted(all_results.keys())
    most_recent = max(years_sorted)
    weights = {y: DECAY ** (most_recent - y) for y in years_sorted}
    total_weight = sum(weights.values())

    print(f"\n{'=' * 80}")
    print("RESULTS: Best-of-50 / Median bracket score")
    print(f"{'=' * 80}")

    # Header
    header = f"  {'Year':<6} {'Wt':>5}"
    for mode in mode_names:
        header += f"  {mode:>15}"
    print(header)

    for year in years_sorted:
        w = weights[year]
        row = f"  {year:<6} {w:>5.2f}"
        for mode in mode_names:
            r = all_results[year][mode]
            row += f"  {r['best_of_50']:>6.0f}/{r['median_bracket']:>5.0f}"
        print(row)

    # Weighted averages
    print(f"  {'':─<6} {'':─>5}" + "  " + "─" * (17 * len(mode_names)))

    print(f"\n  Weighted averages (decay={DECAY}, most_recent={most_recent}):")
    print(f"  {'Metric':<20}", end="")
    for mode in mode_names:
        print(f"  {mode:>15}", end="")
    print()

    for metric_name in ["best_of_50", "median_bracket", "mean_bracket"]:
        print(f"  {metric_name:<20}", end="")
        for mode in mode_names:
            weighted_avg = sum(
                weights[y] * all_results[y][mode][metric_name]
                for y in years_sorted
            ) / total_weight
            print(f"  {weighted_avg:>15.1f}", end="")
        print()

    # Rank modes per year by best-of-50 and by median
    print(f"\n  Per-year ranks (1=best mode that year):")
    print(f"  {'Year':<6}", end="")
    for mode in mode_names:
        print(f"  {mode:>15}", end="")
    print()

    mode_rank_sums_best = {m: 0.0 for m in mode_names}
    mode_rank_sums_median = {m: 0.0 for m in mode_names}
    mode_wins_best = {m: 0 for m in mode_names}
    mode_wins_median = {m: 0 for m in mode_names}

    for year in years_sorted:
        bests = [(all_results[year][m]["best_of_50"], m) for m in mode_names]
        medians = [(all_results[year][m]["median_bracket"], m) for m in mode_names]
        bests.sort(reverse=True)
        medians.sort(reverse=True)
        best_ranks = {m: i + 1 for i, (_, m) in enumerate(bests)}
        median_ranks = {m: i + 1 for i, (_, m) in enumerate(medians)}

        row = f"  {year:<6}"
        for mode in mode_names:
            row += f"  {best_ranks[mode]:>6}b/{median_ranks[mode]:>5}m"
        print(row)

        for mode in mode_names:
            w = weights[year]
            mode_rank_sums_best[mode] += best_ranks[mode] * w
            mode_rank_sums_median[mode] += median_ranks[mode] * w
            if best_ranks[mode] == 1:
                mode_wins_best[mode] += 1
            if median_ranks[mode] == 1:
                mode_wins_median[mode] += 1

    print(f"\n  Weighted avg rank (lower=better):")
    print(f"  {'':20}", end="")
    for mode in mode_names:
        print(f"  {mode:>15}", end="")
    print()
    print(f"  {'best-of-50 rank':<20}", end="")
    for mode in mode_names:
        print(f"  {mode_rank_sums_best[mode] / total_weight:>15.2f}", end="")
    print()
    print(f"  {'median rank':<20}", end="")
    for mode in mode_names:
        print(f"  {mode_rank_sums_median[mode] / total_weight:>15.2f}", end="")
    print()
    print(f"  {'#1 finishes (best)':<20}", end="")
    for mode in mode_names:
        print(f"  {mode_wins_best[mode]:>15}", end="")
    print()
    print(f"  {'#1 finishes (med)':<20}", end="")
    for mode in mode_names:
        print(f"  {mode_wins_median[mode]:>15}", end="")
    print()

    out_path = PROJECT_ROOT / "artifacts" / "mode_historical_scoring_2026-04-19.json"
    out = {
        "description": "Historical mode scoring: best/median bracket score vs actual outcomes",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {
            "years": years_sorted,
            "n_brackets": N_BRACKETS,
            "gen_seed": GEN_SEED,
            "decay": DECAY,
            "scoring": "team_identity",
            "modes": mode_names,
        },
        "per_year": {str(k): v for k, v in all_results.items()},
        "weights": {str(k): v for k, v in weights.items()},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Artifact: {out_path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
