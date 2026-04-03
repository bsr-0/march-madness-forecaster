"""Monte Carlo pool backtest: measures P(rank=1) for noseed vs seed vs blend.

Uses the existing PoolCompetitionSimulator infrastructure to generate opponent
brackets and score them against actual historical tournament outcomes.

For each year (2008-2025, excluding 2020):
  1. Build a deterministic model bracket for each mode (seed/noseed/blend)
  2. Generate N opponent brackets from seed-based pick distributions
  3. Score all brackets against the actual tournament outcome
  4. Record finish rank of each model bracket

This answers the council's question: does better Brier accuracy translate
to higher P(rank=1) in a simulated pool?
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from src.data.seed_pick_model import SEED_PICK_RATES
from src.prediction.noseed_model import (
    train_noseed_model,
    build_noseed_probabilities,
    build_noseed_round_probabilities,
    build_blend_probabilities,
    build_blend_round_probabilities,
)
from src.prediction.seed_probabilities import (
    build_seed_probabilities,
    build_seed_round_probabilities,
)
from src.simulation.pool_competition import (
    generate_opponent_brackets,
    score_brackets_against_outcome,
    build_scoring_vector,
    ROUND_NAMES,
    GAMES_PER_ROUND,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HIST_DIR = Path("data/raw/historical")
BACKTEST_YEARS = [y for y in range(2008, 2026) if y != 2020]  # 17 years
ESPN_SCORING = {"R64": 10, "R32": 20, "S16": 40, "E8": 80, "F4": 160, "CHAMP": 320}
N_OPPONENTS = 999  # 1000-person pool
N_REPEATS = 50  # Repeat opponent sampling to reduce variance
SEED_MATCHUP_ORDER = [(1, 16), (8, 9), (5, 12), (4, 13), (6, 11), (3, 14), (7, 10), (2, 15)]
REGION_ORDER = ["East", "West", "South", "Midwest"]


# ---------------------------------------------------------------------------
# Data loading (reused patterns from unified_mode_evaluation.py)
# ---------------------------------------------------------------------------


def load_seeds_and_regions(year):
    """Load seeds and regions from tournament_seeds_{year}.json."""
    path = HIST_DIR / f"tournament_seeds_{year}.json"
    if not path.exists():
        return {}, {}
    with open(path) as f:
        data = json.load(f)
    seeds = {}
    regions = {}
    if isinstance(data, dict) and "teams" in data:
        for t in data["teams"]:
            seeds[t["team_id"]] = t["seed"]
            regions[t["team_id"]] = t.get("region", "")
    return seeds, regions


def load_tournament_results(year):
    """Load tournament game results."""
    path = HIST_DIR / f"tournament_results_{year}.json"
    if not path.exists():
        return []
    with open(path) as f:
        data = json.load(f)
    return data.get("games", data) if isinstance(data, dict) else data


def _load_team_stats(year):
    """Load Torvik four-factors for noseed model."""
    path = HIST_DIR / f"torvik_four_factors_{year}.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Bracket structure helpers
# ---------------------------------------------------------------------------


def build_first_round_matchups(seeds, regions):
    """Build ordered 64-team first-round matchup list from seeds and regions."""
    matchups = []
    teams_by_region = defaultdict(dict)
    for tid, seed in seeds.items():
        region = regions.get(tid, "")
        teams_by_region[region][seed] = tid

    for region in REGION_ORDER:
        region_teams = teams_by_region.get(region, {})
        for high_seed, low_seed in SEED_MATCHUP_ORDER:
            t_high = region_teams.get(high_seed, f"unknown_{region}_{high_seed}")
            t_low = region_teams.get(low_seed, f"unknown_{region}_{low_seed}")
            matchups.extend([t_high, t_low])

    return matchups


def build_model_bracket(first_round_matchups, round_probs):
    """Convert round probabilities into a 63-winner bracket.

    Walks the bracket structure deterministically, picking the team with
    higher advancement probability at each game.

    Returns list of 63 team_id winners in standard bracket order.
    """
    winners = []
    current_teams = list(first_round_matchups)

    for round_idx in range(6):
        round_name = ROUND_NAMES[round_idx]
        next_round = []
        for g in range(0, len(current_teams), 2):
            if g + 1 >= len(current_teams):
                next_round.append(current_teams[g])
                continue
            t1, t2 = current_teams[g], current_teams[g + 1]
            p1 = round_probs.get(t1, {}).get(round_name, 0.0)
            p2 = round_probs.get(t2, {}).get(round_name, 0.0)
            winner = t1 if p1 >= p2 else t2
            winners.append(winner)
            next_round.append(winner)
        current_teams = next_round

    return winners


def build_actual_outcome(first_round_matchups, games):
    """Convert actual tournament results into a (63,) boolean vector.

    True means the first-listed team in the bracket slot won.
    Matches the convention used by generate_opponent_brackets.
    """
    # Index games by team pair for lookup
    game_results = {}
    for g in games:
        if g.get("round_name") == "FF":
            continue
        t1, t2 = g["team1_id"], g["team2_id"]
        game_results[(t1, t2)] = g["team1_won"]
        game_results[(t2, t1)] = not g["team1_won"]

    outcome = np.zeros(63, dtype=bool)
    current_teams = list(first_round_matchups)
    game_idx = 0

    for round_idx in range(6):
        next_round = []
        for g in range(0, len(current_teams), 2):
            if g + 1 >= len(current_teams):
                next_round.append(current_teams[g])
                continue
            t1, t2 = current_teams[g], current_teams[g + 1]

            # Look up actual result
            t1_won = game_results.get((t1, t2))
            if t1_won is None:
                # Fallback: check if either team won in this round's results
                t1_won = True  # default

            outcome[game_idx] = t1_won
            winner = t1 if t1_won else t2
            next_round.append(winner)
            game_idx += 1

        current_teams = next_round

    return outcome


def build_seed_pick_distribution(seeds):
    """Build opponent pick distribution from SEED_PICK_RATES."""
    return {tid: dict(SEED_PICK_RATES.get(seed, SEED_PICK_RATES[8])) for tid, seed in seeds.items()}


# ---------------------------------------------------------------------------
# Main backtest
# ---------------------------------------------------------------------------


def run_backtest(years=None, n_opponents=N_OPPONENTS, n_repeats=N_REPEATS):
    """Run MC pool backtest across historical years."""
    if years is None:
        years = BACKTEST_YEARS

    scoring_vector = build_scoring_vector(ESPN_SCORING)
    pool_size = n_opponents + 1  # 1 model bracket + N opponents

    print("=" * 100)
    print("MC POOL BACKTEST: P(rank=1) Evaluation")
    print("=" * 100)
    print(f"  Pool size: {pool_size} (1 model + {n_opponents} opponents)")
    print(f"  Opponent model: seed-based pick rates (independent draws)")
    print(f"  Repeats per year: {n_repeats} (reduces opponent sampling variance)")
    print(f"  Years: {len(years)}")
    print()

    header = f"  {'Year':<6} {'Mode':<8} {'Score':>6} {'Rank':>8} {'P(1st)':>8} {'P(top5)':>8} {'P(top25)':>9}"
    print(header)
    print(f"  {'-' * 80}")

    results = []

    for year in years:
        seeds, regions = load_seeds_and_regions(year)
        if not seeds or not regions:
            print(f"  {year:<6} SKIP — no seeds/regions")
            continue

        games = load_tournament_results(year)
        if not games:
            print(f"  {year:<6} SKIP — no games")
            continue

        stats = _load_team_stats(year)

        # Build first-round matchups
        first_round = build_first_round_matchups(seeds, regions)
        if len(first_round) != 64:
            print(f"  {year:<6} SKIP — {len(first_round)} teams (need 64)")
            continue

        # Build actual outcome
        actual = build_actual_outcome(first_round, games)

        # Build pairwise probs for opponent bracket generation
        seed_pw = build_seed_probabilities(seeds)

        # Train noseed model
        model = train_noseed_model(max_year=year)

        # Build round probs for each mode
        seed_rp = build_seed_round_probabilities(seeds)
        noseed_rp = build_noseed_round_probabilities(model, seeds, stats)
        blend_rp = build_blend_round_probabilities(seed_rp, noseed_rp, alpha=0.5)

        # Build model brackets
        mode_brackets = {
            "seed": build_model_bracket(first_round, seed_rp),
            "noseed": build_model_bracket(first_round, noseed_rp),
            "blend": build_model_bracket(first_round, blend_rp),
        }

        # Build opponent distribution
        pick_dist = build_seed_pick_distribution(seeds)

        # Run repeated simulations to reduce opponent sampling variance
        rng = np.random.default_rng(42 + year)

        for mode_name, bracket_winners in mode_brackets.items():
            # Convert model bracket to boolean array
            model_bool = np.zeros(63, dtype=bool)
            current_teams = list(first_round)
            game_idx = 0
            winner_idx = 0
            replay_current = list(first_round)
            for round_idx in range(6):
                next_round = []
                for g in range(0, len(replay_current), 2):
                    if g + 1 >= len(replay_current):
                        next_round.append(replay_current[g])
                        continue
                    t1 = replay_current[g]
                    picked = bracket_winners[game_idx]
                    model_bool[game_idx] = picked == t1
                    next_round.append(picked)
                    game_idx += 1
                replay_current = next_round

            # Score model bracket against actual outcome
            model_score = float(score_brackets_against_outcome(model_bool.reshape(1, -1), actual, scoring_vector)[0])

            # Repeat opponent generation to get stable rank estimates
            ranks = []
            for rep in range(n_repeats):
                opp = generate_opponent_brackets(
                    n_opponents,
                    first_round,
                    seed_pw,
                    pick_dist,
                    seeds,
                    rng,
                )
                all_brackets = np.vstack([model_bool.reshape(1, -1), opp])
                scores = score_brackets_against_outcome(all_brackets, actual, scoring_vector)

                # Rank (1 = best)
                order = np.argsort(-scores)
                rank_arr = np.empty_like(order, dtype=np.float64)
                rank_arr[order] = np.arange(1, pool_size + 1)
                # Average ties
                for s in np.unique(scores):
                    mask = scores == s
                    if mask.sum() > 1:
                        rank_arr[mask] = rank_arr[mask].mean()
                ranks.append(rank_arr[0])  # Model bracket is index 0

            mean_rank = np.mean(ranks)
            p_first = np.mean([r == 1.0 for r in ranks])
            p_top5 = np.mean([r <= max(1, pool_size * 0.05) for r in ranks])
            p_top25 = np.mean([r <= max(1, pool_size * 0.25) for r in ranks])

            print(
                f"  {year:<6} {mode_name:<8} {model_score:6.0f} "
                f"{mean_rank:7.1f} {p_first:8.3f} {p_top5:8.3f} {p_top25:9.3f}"
            )

            results.append(
                {
                    "year": year,
                    "mode": mode_name,
                    "score": model_score,
                    "mean_rank": mean_rank,
                    "p_first": p_first,
                    "p_top5": p_top5,
                    "p_top25": p_top25,
                }
            )

    if not results:
        print("\nNo results.")
        return 1

    # --- Aggregates ---
    print(f"\n{'=' * 100}")
    print("AGGREGATE")
    print(f"{'=' * 100}")
    print(f"\n  {'Mode':<8} {'Mean Rank':>10} {'P(1st)':>8} {'P(top5%)':>10} {'P(top25%)':>10} {'Mean Score':>11}")
    print(f"  {'-' * 60}")

    for mode in ["seed", "noseed", "blend"]:
        mode_results = [r for r in results if r["mode"] == mode]
        if not mode_results:
            continue
        print(
            f"  {mode:<8} "
            f"{np.mean([r['mean_rank'] for r in mode_results]):10.1f} "
            f"{np.mean([r['p_first'] for r in mode_results]):8.4f} "
            f"{np.mean([r['p_top5'] for r in mode_results]):10.4f} "
            f"{np.mean([r['p_top25'] for r in mode_results]):10.4f} "
            f"{np.mean([r['score'] for r in mode_results]):11.0f}"
        )

    # --- Statistical tests ---
    print(f"\n  Statistical Tests (paired across {len(BACKTEST_YEARS)} years):")

    seed_ranks = []
    noseed_ranks = []
    blend_ranks = []
    for year in years:
        sr = [r for r in results if r["year"] == year and r["mode"] == "seed"]
        nr = [r for r in results if r["year"] == year and r["mode"] == "noseed"]
        br = [r for r in results if r["year"] == year and r["mode"] == "blend"]
        if sr and nr and br:
            seed_ranks.append(sr[0]["mean_rank"])
            noseed_ranks.append(nr[0]["mean_rank"])
            blend_ranks.append(br[0]["mean_rank"])

    if len(seed_ranks) >= 5:
        seed_arr = np.array(seed_ranks)
        noseed_arr = np.array(noseed_ranks)
        blend_arr = np.array(blend_ranks)

        # Paired t-test on ranks (lower = better, so test seed - noseed > 0)
        t_ns, p_ns = sp_stats.ttest_rel(seed_arr, noseed_arr)
        t_bl, p_bl = sp_stats.ttest_rel(seed_arr, blend_arr)
        print(f"    Rank t-test (seed vs noseed): t={t_ns:.3f}, p={p_ns:.4f}")
        print(f"    Rank t-test (seed vs blend):  t={t_bl:.3f}, p={p_bl:.4f}")

        # Wilcoxon signed-rank
        try:
            w_ns, pw_ns = sp_stats.wilcoxon(seed_arr - noseed_arr, alternative="greater")
            print(f"    Wilcoxon (seed rank > noseed rank): W={w_ns:.0f}, p={pw_ns:.4f}")
        except ValueError:
            pass

        # Win count
        ns_wins = np.sum(noseed_arr < seed_arr)
        bl_wins = np.sum(blend_arr < seed_arr)
        n_years = len(seed_arr)
        print(f"    Noseed beats seed: {ns_wins}/{n_years} years")
        print(f"    Blend beats seed:  {bl_wins}/{n_years} years")

        # Mean rank improvement
        print(f"\n    Mean rank improvement (lower = better):")
        print(f"      Noseed vs seed: {np.mean(seed_arr - noseed_arr):+.1f} positions")
        print(f"      Blend vs seed:  {np.mean(seed_arr - blend_arr):+.1f} positions")

    print(f"\n{'=' * 100}")
    return 0


def main():
    import argparse

    parser = argparse.ArgumentParser(description="MC pool backtest")
    parser.add_argument("--years", type=int, nargs="+", default=None, help="Specific years to test (default: all 17)")
    parser.add_argument("--n-opponents", type=int, default=N_OPPONENTS)
    parser.add_argument("--n-repeats", type=int, default=N_REPEATS)
    args = parser.parse_args()
    return run_backtest(years=args.years, n_opponents=args.n_opponents, n_repeats=args.n_repeats)


if __name__ == "__main__":
    sys.exit(main())
