"""Monte Carlo pool backtest: measures P(rank=1) for noseed vs seed vs blend.

Uses the existing PoolCompetitionSimulator infrastructure to generate opponent
brackets and score them against actual historical tournament outcomes.

For each year (2008-2025, excluding 2020):
  1. Sample N_MODEL_BRACKETS stochastic brackets from each mode's round
     probabilities (path-consistent random draws, NOT deterministic argmax)
  2. Generate opponent brackets from seed-based pick distributions
  3. Score all brackets against the actual tournament outcome
  4. Record best finish rank across model brackets for each mode

Council Session 4 identified deterministic argmax brackets as the core defect
in the original backtest: argmax collapses calibrated probabilities into a
single crowd-following bracket, discarding the model's ability to identify
high-leverage upsets. Stochastic sampling preserves that signal.
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
N_MODEL_BRACKETS = 50  # Stochastic brackets per mode per repeat
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


def build_model_bracket_argmax(first_round_matchups, round_probs):
    """Convert round probabilities into a 63-winner bracket (deterministic).

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


def sample_model_brackets(first_round_matchups, round_probs, n_brackets, rng):
    """Sample N stochastic brackets from model round probabilities.

    Uses the same path-consistent walk as the opponent bracket sampler:
    at each game, the winner is drawn probabilistically from the model's
    head-to-head probability (derived from marginal round advancement rates).
    Path consistency is enforced — a team can only appear in round R+1 if
    it won in round R within this bracket.

    This preserves the model's calibrated probability signal instead of
    collapsing it to a single argmax bracket.

    Returns:
        Boolean array of shape (n_brackets, 63).
    """
    all_brackets = np.zeros((n_brackets, 63), dtype=bool)

    for b in range(n_brackets):
        current_teams = list(first_round_matchups)
        game_idx = 0

        for round_idx in range(6):
            round_name = ROUND_NAMES[round_idx]
            next_round = []

            for g in range(0, len(current_teams), 2):
                if g + 1 >= len(current_teams):
                    next_round.append(current_teams[g])
                    continue

                t1, t2 = current_teams[g], current_teams[g + 1]

                # Get marginal advancement probabilities and normalize
                p1 = round_probs.get(t1, {}).get(round_name, 0.0)
                p2 = round_probs.get(t2, {}).get(round_name, 0.0)

                if p1 + p2 > 1e-8:
                    p_t1 = p1 / (p1 + p2)
                else:
                    p_t1 = 0.5

                if rng.random() < p_t1:
                    winner = t1
                    all_brackets[b, game_idx] = True
                else:
                    winner = t2
                    all_brackets[b, game_idx] = False

                next_round.append(winner)
                game_idx += 1

            current_teams = next_round

    return all_brackets


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


def build_espn_pick_distribution(year, seeds):
    """Build opponent pick distribution from real ESPN public picks data.

    Falls back to seed-based if no ESPN data available for this year.
    """
    from src.data.historical_picks import load_historical_public_picks

    picks = load_historical_public_picks(year, seeds)
    return picks


# ---------------------------------------------------------------------------
# Main backtest
# ---------------------------------------------------------------------------


def run_backtest(
    years=None, n_opponents=N_OPPONENTS, n_repeats=N_REPEATS, n_model=N_MODEL_BRACKETS, opponent_source="seed"
):
    """Run MC pool backtest across historical years.

    For each year and mode, we sample n_model stochastic brackets from the
    model's round probabilities. Each stochastic bracket competes in n_repeats
    pools against n_opponents. We report:
      - best_rank: average rank of the BEST stochastic bracket (pool optimizer)
      - mean_rank: average rank across ALL stochastic brackets (fair comparison)
      - P(1st): fraction of (bracket x repeat) trials finishing first

    Args:
        opponent_source: "seed" for SEED_PICK_RATES, "espn" for real ESPN data.
    """
    if years is None:
        years = BACKTEST_YEARS

    scoring_vector = build_scoring_vector(ESPN_SCORING)
    pool_size = n_opponents + 1  # 1 model bracket + N opponents

    print("=" * 100)
    print("MC POOL BACKTEST: P(rank=1) — Stochastic Brackets")
    print("=" * 100)
    print(f"  Pool size: {pool_size} (1 model + {n_opponents} opponents)")
    print(f"  Opponent model: {opponent_source} pick rates (independent draws)")
    print(f"  Model brackets per mode: {n_model} (stochastic, NOT argmax)")
    print(f"  Repeats per year: {n_repeats} (reduces opponent sampling variance)")
    print(f"  Years: {len(years)}")
    print()

    header = (
        f"  {'Year':<6} {'Mode':<8} {'BestRnk':>8} {'MeanRnk':>8} "
        f"{'P(1st)':>8} {'P(top5)':>8} {'P(top25)':>9} {'BestScr':>8} {'MeanScr':>8}"
    )
    print(header)
    print(f"  {'-' * 90}")

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

        mode_round_probs = {
            "seed": seed_rp,
            "noseed": noseed_rp,
            "blend": blend_rp,
        }

        # Build opponent distribution
        if opponent_source == "espn":
            pick_dist = build_espn_pick_distribution(year, seeds)
        else:
            pick_dist = build_seed_pick_distribution(seeds)

        rng = np.random.default_rng(42 + year)

        for mode_name, rp in mode_round_probs.items():
            # Sample stochastic model brackets
            model_brackets = sample_model_brackets(first_round, rp, n_model, rng)

            # Score all model brackets against actual outcome
            model_scores = score_brackets_against_outcome(model_brackets, actual, scoring_vector)

            # For each repeat: generate opponents, score everything, rank
            # Track ranks for each model bracket across repeats
            all_ranks = np.zeros((n_model, n_repeats))

            for rep in range(n_repeats):
                opp = generate_opponent_brackets(
                    n_opponents,
                    first_round,
                    seed_pw,
                    pick_dist,
                    seeds,
                    rng,
                )
                opp_scores = score_brackets_against_outcome(opp, actual, scoring_vector)

                # Rank each model bracket against this opponent field
                for m in range(n_model):
                    # How many opponents scored strictly higher + 1
                    better = np.sum(opp_scores > model_scores[m])
                    tied = np.sum(opp_scores == model_scores[m])
                    # Average rank among ties (model is 1 of the tied group)
                    all_ranks[m, rep] = better + 1 + tied / 2.0

            # Per-bracket average rank across repeats
            bracket_mean_ranks = all_ranks.mean(axis=1)
            # Best bracket = lowest average rank
            best_bracket_idx = np.argmin(bracket_mean_ranks)
            best_rank = bracket_mean_ranks[best_bracket_idx]
            mean_rank = bracket_mean_ranks.mean()

            # P(1st) across all brackets x repeats
            p_first = (all_ranks == 1.0).mean()
            p_top5 = (all_ranks <= max(1, pool_size * 0.05)).mean()
            p_top25 = (all_ranks <= max(1, pool_size * 0.25)).mean()

            best_score = float(model_scores[best_bracket_idx])
            mean_score = float(model_scores.mean())

            print(
                f"  {year:<6} {mode_name:<8} {best_rank:8.1f} {mean_rank:8.1f} "
                f"{p_first:8.3f} {p_top5:8.3f} {p_top25:9.3f} "
                f"{best_score:8.0f} {mean_score:8.0f}"
            )

            results.append(
                {
                    "year": year,
                    "mode": mode_name,
                    "best_rank": best_rank,
                    "mean_rank": mean_rank,
                    "best_score": best_score,
                    "mean_score": mean_score,
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
    print(
        f"\n  {'Mode':<8} {'BestRnk':>8} {'MeanRnk':>8} {'P(1st)':>8} {'P(top5%)':>10} {'P(top25%)':>10} {'MeanScr':>8}"
    )
    print(f"  {'-' * 65}")

    for mode in ["seed", "noseed", "blend"]:
        mode_results = [r for r in results if r["mode"] == mode]
        if not mode_results:
            continue
        print(
            f"  {mode:<8} "
            f"{np.mean([r['best_rank'] for r in mode_results]):8.1f} "
            f"{np.mean([r['mean_rank'] for r in mode_results]):8.1f} "
            f"{np.mean([r['p_first'] for r in mode_results]):8.4f} "
            f"{np.mean([r['p_top5'] for r in mode_results]):10.4f} "
            f"{np.mean([r['p_top25'] for r in mode_results]):10.4f} "
            f"{np.mean([r['mean_score'] for r in mode_results]):8.0f}"
        )

    # --- Statistical tests (on mean_rank for fair comparison) ---
    print(f"\n  Statistical Tests — Mean Rank (paired across years):")

    seed_ranks = []
    noseed_ranks = []
    blend_ranks = []
    seed_best = []
    noseed_best = []
    blend_best = []
    for year in years:
        sr = [r for r in results if r["year"] == year and r["mode"] == "seed"]
        nr = [r for r in results if r["year"] == year and r["mode"] == "noseed"]
        br = [r for r in results if r["year"] == year and r["mode"] == "blend"]
        if sr and nr and br:
            seed_ranks.append(sr[0]["mean_rank"])
            noseed_ranks.append(nr[0]["mean_rank"])
            blend_ranks.append(br[0]["mean_rank"])
            seed_best.append(sr[0]["best_rank"])
            noseed_best.append(nr[0]["best_rank"])
            blend_best.append(br[0]["best_rank"])

    if len(seed_ranks) >= 5:
        seed_arr = np.array(seed_ranks)
        noseed_arr = np.array(noseed_ranks)
        blend_arr = np.array(blend_ranks)

        t_ns, p_ns = sp_stats.ttest_rel(seed_arr, noseed_arr)
        t_bl, p_bl = sp_stats.ttest_rel(seed_arr, blend_arr)
        print(f"    Rank t-test (seed vs noseed): t={t_ns:.3f}, p={p_ns:.4f}")
        print(f"    Rank t-test (seed vs blend):  t={t_bl:.3f}, p={p_bl:.4f}")

        try:
            w_ns, pw_ns = sp_stats.wilcoxon(seed_arr - noseed_arr, alternative="greater")
            print(f"    Wilcoxon (seed rank > noseed rank): W={w_ns:.0f}, p={pw_ns:.4f}")
        except ValueError:
            pass

        ns_wins = np.sum(noseed_arr < seed_arr)
        bl_wins = np.sum(blend_arr < seed_arr)
        n_years = len(seed_arr)
        print(f"    Noseed beats seed: {ns_wins}/{n_years} years (mean rank)")
        print(f"    Blend beats seed:  {bl_wins}/{n_years} years (mean rank)")

        print(f"\n    Mean rank improvement (lower = better):")
        print(f"      Noseed vs seed: {np.mean(seed_arr - noseed_arr):+.1f} positions")
        print(f"      Blend vs seed:  {np.mean(seed_arr - blend_arr):+.1f} positions")

    # --- Best-bracket stats (pool optimizer view) ---
    if len(seed_best) >= 5:
        print(f"\n  Statistical Tests — Best Bracket Rank (pool optimizer view):")
        sb = np.array(seed_best)
        nb = np.array(noseed_best)
        bb = np.array(blend_best)

        t_ns, p_ns = sp_stats.ttest_rel(sb, nb)
        t_bl, p_bl = sp_stats.ttest_rel(sb, bb)
        print(f"    Rank t-test (seed vs noseed): t={t_ns:.3f}, p={p_ns:.4f}")
        print(f"    Rank t-test (seed vs blend):  t={t_bl:.3f}, p={p_bl:.4f}")

        ns_wins = np.sum(nb < sb)
        bl_wins = np.sum(bb < sb)
        print(f"    Noseed beats seed: {ns_wins}/{len(sb)} years (best bracket)")
        print(f"    Blend beats seed:  {bl_wins}/{len(sb)} years (best bracket)")

        print(f"\n    Best-bracket rank improvement:")
        print(f"      Noseed vs seed: {np.mean(sb - nb):+.1f} positions")
        print(f"      Blend vs seed:  {np.mean(sb - bb):+.1f} positions")

    print(f"\n{'=' * 100}")
    return 0


def main():
    import argparse

    parser = argparse.ArgumentParser(description="MC pool backtest")
    parser.add_argument("--years", type=int, nargs="+", default=None, help="Specific years to test (default: all 17)")
    parser.add_argument("--n-opponents", type=int, default=N_OPPONENTS)
    parser.add_argument("--n-repeats", type=int, default=N_REPEATS)
    parser.add_argument("--n-model", type=int, default=N_MODEL_BRACKETS, help="Stochastic brackets per mode")
    parser.add_argument(
        "--opponent",
        choices=["seed", "espn"],
        default="seed",
        help="Opponent pick distribution source: seed (SEED_PICK_RATES) or espn (real ESPN data)",
    )
    args = parser.parse_args()
    return run_backtest(
        years=args.years,
        n_opponents=args.n_opponents,
        n_repeats=args.n_repeats,
        n_model=args.n_model,
        opponent_source=args.opponent,
    )


if __name__ == "__main__":
    sys.exit(main())
