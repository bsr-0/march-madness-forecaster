"""O24: Does P(score >= 97th-pct threshold) produce same bracket rankings as P(rank=1)?

Council verdict (2026-04-16): if changing opponent-model marginals doesn't change
rankings (O21), the opponent distribution is effectively a constant backdrop.
Replacing it with a dynamic threshold (97th percentile of simulated winner scores)
should produce equivalent rankings.

Gate:
  - Spearman rho >= 0.95 on full ranking across all tested years
  - Top-decile concordance >= 0.97 across all tested years

Methodology:
  For each historical year (2023-2025):
  1. Build seed-based round probabilities and opponent pick distribution
  2. Generate N_BRACKETS candidate brackets using champ_first_tv sampler
  3. Run N_TOURN MC trials: simulate tournament -> score all brackets -> rank
  4. From trial data compute:
       P(rank=1)[b]         = fraction of trials where bracket b is rank 1
       winner_score[t]      = max score across all brackets in trial t
       threshold            = np.percentile(winner_scores, 97)
       P(score>=thresh)[b]  = fraction of trials where b's score >= threshold
  5. Rank brackets by each metric, compute Spearman rho and top-decile concordance

Note: functions from mc_pool_backtest are inlined here to avoid its module-level
sklearn import (noseed_model dependency). The inlined functions are pure Python
with only numpy/json/pathlib dependencies.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts._common import load_tournament_results
from src.prediction.seed_probabilities import (
    build_seed_probabilities,
    build_seed_round_probabilities,
)
from src.data.seed_pick_model import SEED_PICK_RATES
from src.simulation.pool_competition import (
    build_scoring_vector,
    generate_opponent_brackets,
    score_brackets_against_outcome,
    simulate_tournament_outcomes,
)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

YEARS = [2023, 2024, 2025]
N_BRACKETS = 30  # candidate brackets per year
N_TOURN = 5000  # MC tournament simulations per year
N_OPPONENTS = 30  # opponent pool size
NOISE_STD = 0.16  # from constant_registry (Lopez & Matthews 2015)
WINNER_PERCENTILE = 97  # "first-place score" threshold
FULL_RHO_GATE = 0.95
TOP_DECILE_GATE = 0.97

ESPN_SCORING = {"R64": 10, "R32": 20, "S16": 40, "E8": 80, "F4": 160, "NCG": 320}

# ---------------------------------------------------------------------------
# Inlined helpers from mc_pool_backtest (sklearn-free subset)
# ---------------------------------------------------------------------------

HIST_DIR = Path("data/raw/historical")
SEED_MATCHUP_ORDER = [(1, 16), (8, 9), (5, 12), (4, 13), (6, 11), (3, 14), (7, 10), (2, 15)]
REGION_ORDER = ["East", "West", "South", "Midwest"]
ROUND_NAMES = ["R64", "R32", "S16", "E8", "F4", "CHAMP"]

_REGION_ALIASES = {
    "Southeast": "South",
    "Southwest": "Midwest",
}

_LOCK_ROUND_INDEX = {"R64": 0, "R32": 1, "S16": 2, "E8": 3, "F4": 4, "CHAMP": 5}


def load_seeds_and_regions(year):
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
            raw_region = t.get("region", "")
            regions[t["team_id"]] = _REGION_ALIASES.get(raw_region, raw_region)
    return seeds, regions


def resolve_first_four(games, seeds, regions) -> int:
    ff_games = [g for g in games if g.get("round_name") == "FF"]
    n_replaced = 0
    for g in ff_games:
        loser = g["team2_id"] if g["team1_won"] else g["team1_id"]
        if loser in seeds:
            seeds.pop(loser)
            regions.pop(loser, "")
            n_replaced += 1
    return n_replaced


def derive_f4_region_pairing(games, regions):
    f4_games = [g for g in games if g.get("round_name") == "F4"]
    if len(f4_games) < 2:
        raise ValueError(f"expected 2 F4 games to derive region pairing, got {len(f4_games)}")

    pairs = []
    for g in f4_games[:2]:
        t1, t2 = g["team1_id"], g["team2_id"]
        r1 = regions.get(t1)
        r2 = regions.get(t2)
        if not r1 or not r2:
            raise ValueError(f"could not resolve regions for F4 game {t1} vs {t2}: {r1!r} vs {r2!r}")
        if r1 == r2:
            raise ValueError(f"F4 game has two teams from the same region ({r1}): {t1} vs {t2}")
        pairs.append((r1, r2))

    all_regions = {r for pair in pairs for r in pair}
    if len(all_regions) != 4:
        raise ValueError(f"F4 pairs do not cover 4 distinct regions: pairs={pairs}")

    return (pairs[0][0], pairs[0][1], pairs[1][0], pairs[1][1])


def build_first_round_matchups(seeds, regions, region_order: Sequence[str] = REGION_ORDER):
    matchups = []
    teams_by_region = defaultdict(dict)
    for tid, seed in seeds.items():
        region = regions.get(tid, "")
        teams_by_region[region][seed] = tid

    for region in region_order:
        region_teams = teams_by_region.get(region, {})
        for high_seed, low_seed in SEED_MATCHUP_ORDER:
            t_high = region_teams.get(high_seed, f"unknown_{region}_{high_seed}")
            t_low = region_teams.get(low_seed, f"unknown_{region}_{low_seed}")
            matchups.extend([t_high, t_low])

    return matchups


def _draw_categorical(rng, items, weights):
    w = np.asarray(weights, dtype=float)
    total = w.sum()
    if total > 1e-12:
        w = w / total
    else:
        w = np.ones(len(items)) / len(items)
    idx = rng.choice(len(items), p=w)
    return items[idx]


def _sample_with_locks(first_round_matchups, round_probs, n_brackets, rng, locked_teams_per_sample, lock_through_round):
    all_brackets = np.zeros((n_brackets, 63), dtype=bool)
    lock_round_idx = _LOCK_ROUND_INDEX.get(lock_through_round, -1) if lock_through_round else -1

    for b in range(n_brackets):
        locked = locked_teams_per_sample[b]
        current_teams = list(first_round_matchups)
        game_idx = 0

        for round_idx in range(6):
            round_name = ROUND_NAMES[round_idx]
            next_round = []
            within_lock_range = lock_round_idx >= 0 and round_idx <= lock_round_idx

            for g in range(0, len(current_teams), 2):
                if g + 1 >= len(current_teams):
                    next_round.append(current_teams[g])
                    continue
                t1, t2 = current_teams[g], current_teams[g + 1]

                locked_winner = None
                if within_lock_range and locked:
                    if t1 in locked:
                        locked_winner = t1
                    elif t2 in locked:
                        locked_winner = t2

                if locked_winner is not None:
                    winner = locked_winner
                    all_brackets[b, game_idx] = winner == t1
                else:
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


def sample_champ_first_brackets(first_round_matchups, round_probs, n_brackets, rng):
    teams = list(round_probs.keys())
    champ_weights = [round_probs[t].get("CHAMP", 0.0) for t in teams]

    locked_teams_per_sample = []
    for _ in range(n_brackets):
        champion = _draw_categorical(rng, teams, champ_weights)
        locked_teams_per_sample.append({champion})

    return _sample_with_locks(
        first_round_matchups,
        round_probs,
        n_brackets,
        rng,
        locked_teams_per_sample,
        lock_through_round="CHAMP",
    )


# ---------------------------------------------------------------------------
# Seed pick distribution helper
# ---------------------------------------------------------------------------


def _build_seed_pick_distribution(seeds: dict, pick_rates: dict) -> dict:
    """Build a pick distribution from seed-based SEED_PICK_RATES."""
    dist = {}
    for team_id, seed in seeds.items():
        seed_int = int(seed) if not isinstance(seed, int) else seed
        rates = pick_rates.get(seed_int, {})
        dist[team_id] = {
            "R64": rates.get("R64", 0.5),
            "R32": rates.get("R32", 0.3),
            "S16": rates.get("S16", 0.15),
            "E8": rates.get("E8", 0.08),
            "F4": rates.get("F4", 0.04),
            "CHAMP": rates.get("CHAMP", 0.02),
        }
    return dist


# ---------------------------------------------------------------------------
# Per-year test
# ---------------------------------------------------------------------------


def _run_year(year: int, rng: np.random.Generator) -> dict | None:
    """Run the O24 test for a single year. Returns dict with metrics or None on skip."""
    games = load_tournament_results(year)
    if not games:
        print(f"  {year}: SKIP — no tournament results")
        return None

    seeds, regions = load_seeds_and_regions(year)
    resolve_first_four(games, seeds, regions)

    try:
        region_order = derive_f4_region_pairing(games, regions)
    except ValueError as exc:
        print(f"  {year}: SKIP — {exc}")
        return None

    first_round = build_first_round_matchups(seeds, regions, region_order=region_order)
    if len(first_round) != 64:
        print(f"  {year}: SKIP — {len(first_round)} teams (need 64)")
        return None

    # Game probabilities (seed-based, no ML model needed)
    seed_pw = build_seed_probabilities(seeds)
    seed_rp = build_seed_round_probabilities(seeds)

    # Opponent pick distribution (seed-based)
    pick_dist = _build_seed_pick_distribution(seeds, SEED_PICK_RATES)

    # Scoring vector
    scoring_vector = build_scoring_vector(ESPN_SCORING)

    # --- Generate candidate brackets ---
    model_brackets = sample_champ_first_brackets(first_round, seed_rp, N_BRACKETS, rng)
    # shape: (N_BRACKETS, 63) bool

    # --- Generate opponent brackets (fixed for this year's test) ---
    opp_brackets = generate_opponent_brackets(N_OPPONENTS, first_round, seed_pw, pick_dist, seeds, rng)
    # shape: (N_OPPONENTS, 63) bool

    # --- Combine all brackets ---
    # Convention: model brackets first, opponents last
    all_brackets = np.vstack([model_brackets, opp_brackets])
    pool_size = N_BRACKETS + N_OPPONENTS

    # --- Simulate tournaments and score ---
    outcomes, _ = simulate_tournament_outcomes(N_TOURN, first_round, seed_pw, seeds, NOISE_STD, rng)

    all_scores = np.zeros((N_TOURN, pool_size), dtype=np.float64)
    for t in range(N_TOURN):
        all_scores[t] = score_brackets_against_outcome(all_brackets, outcomes[t], scoring_vector)

    # Model bracket scores only: (N_TOURN, N_BRACKETS)
    model_scores = all_scores[:, :N_BRACKETS]

    # --- Method 1: P(rank=1) ---
    # For each trial, count how often each model bracket ties for 1st in the
    # full pool (including opponent brackets).
    p_rank1 = np.zeros(N_BRACKETS)
    for t in range(N_TOURN):
        scores_t = all_scores[t]
        max_score = scores_t.max()
        for b in range(N_BRACKETS):
            if model_scores[t, b] == max_score:
                p_rank1[b] += 1.0 / N_TOURN

    # --- Method 2: P(score >= 97th-pct winner threshold) ---
    # Winner score per trial = max across the full pool
    winner_scores = all_scores.max(axis=1)  # shape: (N_TOURN,)
    threshold = np.percentile(winner_scores, WINNER_PERCENTILE)
    p_threshold = (model_scores >= threshold).mean(axis=0)  # shape: (N_BRACKETS,)

    # --- Comparison metrics ---
    rho, pval = spearmanr(p_rank1, p_threshold)

    # Top-decile concordance: do the top 10% of brackets agree?
    n_top = max(1, int(np.ceil(N_BRACKETS * 0.10)))
    top_by_rank1 = set(np.argsort(-p_rank1)[:n_top])
    top_by_threshold = set(np.argsort(-p_threshold)[:n_top])
    top_decile_concordance = len(top_by_rank1 & top_by_threshold) / n_top

    print(
        f"  {year}: rho={rho:.4f} (p={pval:.4f})  "
        f"top-decile concordance={top_decile_concordance:.3f}  "
        f"threshold={threshold:.0f}pts  "
        f"p_rank1 range=[{p_rank1.min():.4f},{p_rank1.max():.4f}]  "
        f"p_thresh range=[{p_threshold.min():.4f},{p_threshold.max():.4f}]"
    )

    return {
        "year": year,
        "spearman_rho": float(rho),
        "spearman_pval": float(pval),
        "top_decile_concordance": float(top_decile_concordance),
        "dynamic_threshold_pts": float(threshold),
        "p_rank1": p_rank1.tolist(),
        "p_threshold": p_threshold.tolist(),
        "gate_rho_pass": bool(rho >= FULL_RHO_GATE),
        "gate_top_decile_pass": bool(top_decile_concordance >= TOP_DECILE_GATE),
    }


def main() -> int:
    print("O24: P(rank=1) vs P(score >= 97th-pct threshold) ranking comparison")
    print(f"Parameters: {N_BRACKETS} brackets, {N_TOURN} tournaments, {N_OPPONENTS} opponents")
    print(f"Gate: rho >= {FULL_RHO_GATE}, top-decile concordance >= {TOP_DECILE_GATE}")
    print()

    rng = np.random.default_rng(2026)
    results = []

    for year in YEARS:
        r = _run_year(year, rng)
        if r is not None:
            results.append(r)

    if not results:
        print("No results.")
        return 1

    print()
    print("=== SUMMARY ===")
    all_pass = True
    for r in results:
        rho_ok = "PASS" if r["gate_rho_pass"] else "FAIL"
        td_ok = "PASS" if r["gate_top_decile_pass"] else "FAIL"
        print(
            f"  {r['year']}: rho={r['spearman_rho']:.4f} [{rho_ok}]  "
            f"top-decile={r['top_decile_concordance']:.3f} [{td_ok}]  "
            f"threshold={r['dynamic_threshold_pts']:.0f}pts"
        )
        if not (r["gate_rho_pass"] and r["gate_top_decile_pass"]):
            all_pass = False

    print()
    if all_pass:
        print("VERDICT: PASS — P(score>=threshold) produces equivalent rankings to P(rank=1).")
        print("The opponent simulation is NOT load-bearing for bracket selection.")
        print("Dynamic threshold approach is viable.")
    else:
        print("VERDICT: FAIL — Rankings diverge. Opponent simulation is load-bearing.")
        print("Do NOT replace the opponent simulation with a score threshold.")

    out_path = Path("artifacts/o24_threshold_ranking_2026-04-16.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(
            {
                "council_item": "O24",
                "date": "2026-04-16",
                "parameters": {
                    "n_brackets": N_BRACKETS,
                    "n_tournaments": N_TOURN,
                    "n_opponents": N_OPPONENTS,
                    "winner_percentile": WINNER_PERCENTILE,
                    "rho_gate": FULL_RHO_GATE,
                    "top_decile_gate": TOP_DECILE_GATE,
                },
                "overall_pass": all_pass,
                "per_year": results,
            },
            f,
            indent=2,
        )
    print(f"\nArtifact saved: {out_path}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
