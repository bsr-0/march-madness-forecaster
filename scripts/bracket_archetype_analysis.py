"""Bracket archetype analysis: which bracket profile scores best historically?

Instead of generating 50 random brackets and trying to pick one, this asks:
what measurable property of a bracket predicts its actual tournament score?

For each year (2011-2026 ex 2020), generate 500 brackets across the full
spectrum of chalkiness, measure several properties, score against reality,
and find which property values correlate with high scores.

Properties measured per bracket:
  - chalk_fraction: % of picks matching the higher seed (1.0 = all chalk)
  - r64_upsets: number of R64 upsets (lower seed beating higher)
  - late_upsets: upsets in S16/E8/F4/CHAMP rounds
  - champ_seed: tournament seed of the picked champion
  - expected_score: model's predicted total points (sum of round_prob × pts)

If a property has a stable, strong correlation with actual score across years,
we can use it to GENERATE a single bracket at the optimal setting — no
selection problem at all.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

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
)
from src.simulation.pool_competition import (  # noqa: E402
    actual_winners_by_round,
    picks_by_round,
    score_brackets_team_identity,
)

YEARS = list(range(2011, 2027))
YEARS = [y for y in YEARS if y != 2020]
N_BRACKETS = 500  # wide spread to cover the property range
ESPN_SCORING = {"R64": 10, "R32": 20, "S16": 40, "E8": 80, "F4": 160, "CHAMP": 320}
GEN_SEED = 12345
DECAY = 0.85


def _compute_bracket_properties(candidates, first_round, seeds, round_probs):
    """Compute measurable properties for each bracket."""
    n = candidates.shape[0]

    props = {
        "chalk_fraction": np.zeros(n),
        "r64_upsets": np.zeros(n),
        "late_upsets": np.zeros(n),
        "champ_seed": np.zeros(n),
        "expected_score": np.zeros(n),
    }

    # Build seed lookup for first-round pairs
    n_r64 = len(first_round) // 2
    r64_pairs = [(first_round[2 * i], first_round[2 * i + 1]) for i in range(n_r64)]
    r64_favorites = []
    for t1, t2 in r64_pairs:
        s1, s2 = seeds.get(t1, 16), seeds.get(t2, 16)
        r64_favorites.append(t1 if s1 <= s2 else t2)

    for b in range(n):
        bracket_picks = picks_by_round(candidates[b], first_round)

        # Chalk fraction: across all 63 games, how often did the higher seed win?
        chalk_count = 0
        total_games = 0
        current_teams = list(first_round)
        gi = 0
        for round_idx in range(6):
            for g in range(0, len(current_teams), 2):
                if g + 1 >= len(current_teams):
                    gi += 1
                    continue
                t1, t2 = current_teams[g], current_teams[g + 1]
                s1 = seeds.get(t1, 16)
                s2 = seeds.get(t2, 16)
                winner = t1 if candidates[b, gi] else t2
                favorite = t1 if s1 <= s2 else t2
                if winner == favorite:
                    chalk_count += 1
                total_games += 1
                gi += 1
            # Advance
            next_teams = []
            for g in range(0, len(current_teams), 2):
                if g + 1 >= len(current_teams):
                    next_teams.append(current_teams[g])
                    continue
                t1, t2 = current_teams[g], current_teams[g + 1]
                game_gi = sum(len(current_teams) // 2 for _ in range(round_idx))  # approximate
                # Just use picks_by_round result
            break  # chalk fraction from full bracket_picks instead

        # Simpler chalk fraction using picks_by_round
        all_picks_flat = set()
        for rnd_picks in bracket_picks.values():
            all_picks_flat |= rnd_picks
        # Count R64 upsets
        r64_upset_count = 0
        for i, (t1, t2) in enumerate(r64_pairs):
            s1, s2 = seeds.get(t1, 16), seeds.get(t2, 16)
            favorite = t1 if s1 <= s2 else t2
            underdog = t2 if favorite == t1 else t1
            r64_winners = bracket_picks.get("R64", set())
            if underdog in r64_winners:
                r64_upset_count += 1
        props["r64_upsets"][b] = r64_upset_count

        # Late upsets (S16, E8, F4, CHAMP): count picks where a lower seed
        # advances past a higher seed
        late_upset_count = 0
        for rnd in ("S16", "E8", "F4", "CHAMP"):
            rnd_picks = bracket_picks.get(rnd, set())
            for team in rnd_picks:
                s = seeds.get(team, 16)
                if s > 4:  # seeds 5+ advancing past S16/E8/F4/CHAMP = upset
                    late_upset_count += 1
        props["late_upsets"][b] = late_upset_count

        # Champion seed
        champ_picks = bracket_picks.get("CHAMP", set())
        if champ_picks:
            champ_team = next(iter(champ_picks))
            props["champ_seed"][b] = seeds.get(champ_team, 16)
        else:
            props["champ_seed"][b] = 16

        # Chalk fraction: simple measure — what fraction of R64 picks are favorites?
        r64_chalk = sum(1 for fav in r64_favorites if fav in bracket_picks.get("R64", set()))
        props["chalk_fraction"][b] = r64_chalk / max(1, len(r64_favorites))

        # Expected score: sum of round advancement probability × points
        exp_score = 0.0
        for rnd, pts in ESPN_SCORING.items():
            rnd_picks = bracket_picks.get(rnd, set())
            for team in rnd_picks:
                if team in round_probs:
                    exp_score += round_probs[team].get(rnd, 0.0) * pts
        props["expected_score"][b] = exp_score

    return props


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
    round_probs = build_torvik_round_probabilities(seeds, regions, barthag)
    real_winners = actual_winners_by_round(games)

    # Generate 500 brackets (torvik stochastic — covers wide range)
    rng = np.random.default_rng(GEN_SEED)
    candidates = sample_model_brackets(first_round, round_probs, N_BRACKETS, rng)

    # Score against reality
    actual_scores = score_brackets_team_identity(
        candidates, real_winners, first_round, ESPN_SCORING,
    )

    # Compute properties
    props = _compute_bracket_properties(candidates, first_round, seeds, round_probs)

    # Correlations
    correlations = {}
    for prop_name, prop_values in props.items():
        rho, pval = spearmanr(prop_values, actual_scores)
        correlations[prop_name] = {"rho": float(rho), "pval": float(pval)}

    # Bin analysis for key properties
    bins = {}
    for prop_name in ["chalk_fraction", "r64_upsets", "champ_seed", "expected_score"]:
        vals = props[prop_name]
        scores = actual_scores

        if prop_name == "chalk_fraction":
            edges = [0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
        elif prop_name == "r64_upsets":
            edges = [-0.5, 2.5, 4.5, 6.5, 8.5, 12.5, 20.5]
        elif prop_name == "champ_seed":
            edges = [0.5, 1.5, 2.5, 3.5, 5.5, 16.5]
        elif prop_name == "expected_score":
            p10 = np.percentile(vals, 10)
            p30 = np.percentile(vals, 30)
            p50 = np.percentile(vals, 50)
            p70 = np.percentile(vals, 70)
            p90 = np.percentile(vals, 90)
            edges = [vals.min() - 1, p10, p30, p50, p70, p90, vals.max() + 1]
        else:
            continue

        bin_results = []
        for i in range(len(edges) - 1):
            mask = (vals >= edges[i]) & (vals < edges[i + 1])
            n_in_bin = int(mask.sum())
            if n_in_bin > 0:
                bin_results.append({
                    "range": f"[{edges[i]:.1f}, {edges[i+1]:.1f})",
                    "n": n_in_bin,
                    "mean_score": float(scores[mask].mean()),
                    "median_score": float(np.median(scores[mask])),
                    "max_score": float(scores[mask].max()),
                })
        bins[prop_name] = bin_results

    return {
        "correlations": correlations,
        "bins": bins,
        "best_of_500": float(actual_scores.max()),
        "median_bracket": float(np.median(actual_scores)),
        "n_brackets": int(len(actual_scores)),
    }


def main() -> int:
    print("Bracket Archetype Analysis")
    print(f"  {N_BRACKETS} torvik brackets per year, 5 measured properties")
    print(f"  Question: which bracket property predicts actual tournament score?")

    all_results = {}
    for year in YEARS:
        t0 = time.time()
        r = _run_year(year)
        if r is None:
            print(f"  {year}: SKIP")
            continue
        all_results[year] = r
        corrs = r["correlations"]
        parts = [f"{name}: {c['rho']:+.2f}" for name, c in corrs.items()]
        print(f"  {year}: {', '.join(parts)}  ({time.time()-t0:.1f}s)")

    # Aggregate correlations across years
    print(f"\n{'=' * 80}")
    print("AGGREGATE: Spearman ρ(property, actual_score) per year")
    print(f"{'=' * 80}")

    prop_names = list(all_results[YEARS[0]]["correlations"].keys()) if YEARS[0] in all_results else []
    years_sorted = sorted(all_results.keys())
    most_recent = max(years_sorted)
    weights = {y: DECAY ** (most_recent - y) for y in years_sorted}
    total_weight = sum(weights.values())

    print(f"\n  {'Year':<6}", end="")
    for prop in prop_names:
        print(f"  {prop:>16}", end="")
    print()
    for year in years_sorted:
        print(f"  {year:<6}", end="")
        for prop in prop_names:
            rho = all_results[year]["correlations"][prop]["rho"]
            sig = "*" if all_results[year]["correlations"][prop]["pval"] < 0.05 else " "
            print(f"  {rho:>+14.3f}{sig}", end="")
        print()

    print(f"\n  Weighted avg ρ:")
    print(f"  {'':6}", end="")
    for prop in prop_names:
        wavg = sum(
            weights[y] * all_results[y]["correlations"][prop]["rho"]
            for y in years_sorted
        ) / total_weight
        # Count how many years the sign is consistent
        signs = [all_results[y]["correlations"][prop]["rho"] > 0 for y in years_sorted]
        consistency = max(sum(signs), len(signs) - sum(signs))
        print(f"  {wavg:>+10.3f} ({consistency}/{len(signs)})", end="")
    print()

    # Show binned analysis for the best property
    print(f"\n{'=' * 80}")
    print("BINNED ANALYSIS: expected_score bins (model confidence)")
    print(f"{'=' * 80}")
    for year in years_sorted[-5:]:  # last 5 years
        print(f"\n  {year}:")
        for b in all_results[year]["bins"].get("expected_score", []):
            print(f"    {b['range']:<25} n={b['n']:>4}  "
                  f"mean={b['mean_score']:>6.0f}  median={b['median_score']:>6.0f}  "
                  f"max={b['max_score']:>6.0f}")

    print(f"\n{'=' * 80}")
    print("BINNED ANALYSIS: champ_seed bins")
    print(f"{'=' * 80}")
    for year in years_sorted[-5:]:
        print(f"\n  {year}:")
        for b in all_results[year]["bins"].get("champ_seed", []):
            print(f"    seed {b['range']:<15} n={b['n']:>4}  "
                  f"mean={b['mean_score']:>6.0f}  median={b['median_score']:>6.0f}  "
                  f"max={b['max_score']:>6.0f}")

    out_path = PROJECT_ROOT / "artifacts" / "bracket_archetype_analysis_2026-04-19.json"
    out = {
        "description": "Bracket archetype analysis: property → actual score correlation",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {
            "n_brackets": N_BRACKETS,
            "gen_seed": GEN_SEED,
            "decay": DECAY,
            "scoring": "team_identity",
            "years": years_sorted,
        },
        "per_year": {str(k): v for k, v in all_results.items()},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Artifact: {out_path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
