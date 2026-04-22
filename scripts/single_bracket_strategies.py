"""Single bracket strategy test: no selection problem.

Instead of generating 50 random brackets and trying to pick one, construct
exactly 1 bracket per strategy per year. Score against reality.

Strategies tested:

Construction methods:
  1. argmax         — pick higher-probability team at every game
  2. champ_lock     — lock the model's #1 champion, argmax the rest
  3. f4_lock        — lock the model's #1 F4 per region, argmax the rest
  4. e8_lock        — lock the model's #1 E8 per quadrant, argmax the rest

Risk-tuned (temperature on logit probabilities):
  5. temp_0.5       — sharpen probabilities (more chalky than argmax on close games)
  6. temp_0.8       — slightly conservative
  7. temp_1.5       — add mild randomness (single draw)
  8. temp_2.0       — add moderate randomness

Contrarian:
  9. champ_2nd      — lock the 2nd-most-likely champion, argmax the rest
  10. champ_3rd     — lock the 3rd-most-likely champion
  11. f4_2nd        — each region uses 2nd-most-likely F4 team

Hybrid:
  12. top5_median   — generate 5 brackets (argmax + 4 temp variations),
                      pick the one with highest expected_score

Each strategy produces exactly 1 deterministic bracket (or 1 pick from a
tiny pool). Scored across 15 years with team-identity scoring.
"""

from __future__ import annotations

import json
import math
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
from src.simulation.pool_competition import (  # noqa: E402
    actual_winners_by_round,
    picks_by_round,
    score_brackets_team_identity,
)
from src.simulation.pool_history_opponent_model import (  # noqa: E402
    load_pool_bracket_vectors,
)

POOL_HIST_PATH = PROJECT_ROOT / "pool_hist_results.json"
POOL_YEARS = [2023, 2024, 2025, 2026]

YEARS = list(range(2011, 2027))
YEARS = [y for y in YEARS if y != 2020]
ESPN_SCORING = {"R64": 10, "R32": 20, "S16": 40, "E8": 80, "F4": 160, "CHAMP": 320}
GEN_SEED = 12345
DECAY = 0.85

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

REGION_ALIASES = {"Southeast": "South", "Southwest": "Midwest"}


def _get_matchup_prob(t1, t2, round_probs):
    """Head-to-head probability from marginal round probs (log5 approximation)."""
    # Use the round advancement rates as a strength proxy
    p1 = sum(round_probs.get(t1, {}).values())
    p2 = sum(round_probs.get(t2, {}).values())
    if p1 + p2 == 0:
        return 0.5
    return p1 / (p1 + p2)


def _build_argmax_bracket(first_round, round_probs, locked_teams=None, temperature=0.0, rng=None):
    """Build a single bracket by walking the bracket tree.

    Args:
        locked_teams: set of team_ids forced to win every game they play
        temperature: 0 = argmax, >0 = sample from softened probabilities
        rng: required if temperature > 0
    """
    if locked_teams is None:
        locked_teams = set()

    bracket = np.zeros(63, dtype=bool)
    current_teams = list(first_round)
    gi = 0

    for round_idx in range(6):
        next_round = []
        for g in range(0, len(current_teams), 2):
            if g + 1 >= len(current_teams):
                next_round.append(current_teams[g])
                gi += 1
                continue

            t1, t2 = current_teams[g], current_teams[g + 1]

            # If one team is locked, they win
            if t1 in locked_teams and t2 not in locked_teams:
                bracket[gi] = True
                next_round.append(t1)
            elif t2 in locked_teams and t1 not in locked_teams:
                bracket[gi] = False
                next_round.append(t2)
            elif t1 in locked_teams and t2 in locked_teams:
                # Both locked (shouldn't happen often) — use probability
                p = _get_matchup_prob(t1, t2, round_probs)
                bracket[gi] = True if p >= 0.5 else False
                next_round.append(t1 if bracket[gi] else t2)
            else:
                p = _get_matchup_prob(t1, t2, round_probs)

                if temperature == 0.0:
                    # Pure argmax
                    bracket[gi] = p >= 0.5
                else:
                    # Temperature-scaled sampling
                    safe_p = max(0.001, min(0.999, p))
                    logit = math.log(safe_p / (1 - safe_p))
                    scaled_logit = logit / temperature
                    final_p = 1.0 / (1.0 + math.exp(-scaled_logit))
                    bracket[gi] = rng.random() < final_p

                next_round.append(t1 if bracket[gi] else t2)

            gi += 1
        current_teams = next_round

    return bracket


def _get_top_n_teams(round_probs, regions, round_name, n, region_filter=None):
    """Get top N teams by round advancement probability, optionally filtered by region."""
    teams = []
    for tid, probs in round_probs.items():
        if region_filter:
            raw_region = regions.get(tid, "")
            region = REGION_ALIASES.get(raw_region, raw_region)
            if region != region_filter:
                continue
        teams.append((tid, probs.get(round_name, 0.0)))
    teams.sort(key=lambda x: -x[1])
    return [t[0] for t in teams[:n]]


def _compute_expected_score(bracket, first_round, round_probs):
    """Sum of pick_probability × round_points for each pick."""
    bp = picks_by_round(bracket, first_round)
    exp = 0.0
    for rnd, pts in ESPN_SCORING.items():
        for team in bp.get(rnd, set()):
            exp += round_probs.get(team, {}).get(rnd, 0.0) * pts
    return exp


def _compute_leverage(bracket, first_round, seeds, round_probs):
    """Sum of (model_prob - espn_public_prob) × round_points."""
    bp = picks_by_round(bracket, first_round)
    lev = 0.0
    for rnd, pts in ESPN_SCORING.items():
        for team in bp.get(rnd, set()):
            model_prob = round_probs.get(team, {}).get(rnd, 0.0)
            seed = seeds.get(team, 16)
            public_prob = ESPN_PICK_RATES.get(rnd, {}).get(seed, 0.0)
            lev += (model_prob - public_prob) * pts
    return lev


def _build_strategies(first_round, round_probs, seeds, regions):
    """Build all strategy brackets. Returns dict of name -> (63,) bool array."""
    strategies = {}

    # 1. Argmax
    strategies["argmax"] = _build_argmax_bracket(first_round, round_probs)

    # 2. Champion lock (model's #1 champion)
    top_champs = _get_top_n_teams(round_probs, regions, "CHAMP", 3)
    if top_champs:
        strategies["champ_lock"] = _build_argmax_bracket(
            first_round, round_probs, locked_teams={top_champs[0]},
        )

    # 3. F4 lock (model's #1 F4 per region)
    f4_locked = set()
    for region in ("East", "West", "South", "Midwest"):
        top = _get_top_n_teams(round_probs, regions, "F4", 1, region_filter=region)
        if top:
            f4_locked.add(top[0])
    strategies["f4_lock"] = _build_argmax_bracket(
        first_round, round_probs, locked_teams=f4_locked,
    )

    # 4. E8 lock (model's #1 E8 per quadrant — top 2 per region)
    e8_locked = set()
    for region in ("East", "West", "South", "Midwest"):
        top = _get_top_n_teams(round_probs, regions, "E8", 2, region_filter=region)
        for t in top:
            e8_locked.add(t)
    strategies["e8_lock"] = _build_argmax_bracket(
        first_round, round_probs, locked_teams=e8_locked,
    )

    # 5-8. Temperature variants
    for temp in [0.5, 0.8, 1.5, 2.0]:
        rng = np.random.default_rng(GEN_SEED)
        strategies[f"temp_{temp}"] = _build_argmax_bracket(
            first_round, round_probs, temperature=temp, rng=rng,
        )

    # 9-10. Contrarian champion
    if len(top_champs) >= 2:
        strategies["champ_2nd"] = _build_argmax_bracket(
            first_round, round_probs, locked_teams={top_champs[1]},
        )
    if len(top_champs) >= 3:
        strategies["champ_3rd"] = _build_argmax_bracket(
            first_round, round_probs, locked_teams={top_champs[2]},
        )

    # 11. F4 contrarian (2nd-most-likely per region)
    f4_2nd = set()
    for region in ("East", "West", "South", "Midwest"):
        top = _get_top_n_teams(round_probs, regions, "F4", 2, region_filter=region)
        if len(top) >= 2:
            f4_2nd.add(top[1])
        elif top:
            f4_2nd.add(top[0])
    strategies["f4_2nd"] = _build_argmax_bracket(
        first_round, round_probs, locked_teams=f4_2nd,
    )

    # 12. Top-5 median: generate 5 brackets, pick highest expected_score
    pool = [strategies["argmax"]]
    for temp in [0.3, 0.6, 1.0, 1.5]:
        rng = np.random.default_rng(GEN_SEED + int(temp * 100))
        pool.append(_build_argmax_bracket(
            first_round, round_probs, temperature=temp, rng=rng,
        ))
    exp_scores = [_compute_expected_score(b, first_round, round_probs) for b in pool]
    best_idx = int(np.argmax(exp_scores))
    strategies["top5_expmax"] = pool[best_idx]

    return strategies


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
    actual_f4 = real_winners.get("F4", set())
    actual_champ = real_winners.get("CHAMP", set())

    # Load actual pool opponents for pool years
    pool_winner_score = None
    if year in POOL_YEARS:
        try:
            opp_actual, _ = load_pool_bracket_vectors(
                POOL_HIST_PATH, year, first_round, seeds,
            )
            opp_scores = score_brackets_team_identity(
                opp_actual, real_winners, first_round, ESPN_SCORING,
            )
            pool_winner_score = float(opp_scores.max())
        except Exception:
            pass

    strategies = _build_strategies(first_round, round_probs, seeds, regions)

    results = {}
    for name, bracket in strategies.items():
        bracket_2d = bracket.reshape(1, -1)
        actual_score = float(score_brackets_team_identity(
            bracket_2d, real_winners, first_round, ESPN_SCORING,
        )[0])

        bp = picks_by_round(bracket, first_round)
        exp_score = _compute_expected_score(bracket, first_round, round_probs)
        leverage = _compute_leverage(bracket, first_round, seeds, round_probs)
        f4_corr = len(bp.get("F4", set()) & actual_f4)
        champ_corr = 1 if bp.get("CHAMP", set()) & actual_champ else 0

        beats_pool = False
        ties_pool = False
        if pool_winner_score is not None:
            beats_pool = actual_score > pool_winner_score
            ties_pool = actual_score == pool_winner_score

        results[name] = {
            "actual_score": actual_score,
            "expected_score": exp_score,
            "leverage": leverage,
            "f4_correct": f4_corr,
            "champ_correct": champ_corr,
            "beats_pool_winner": beats_pool,
            "ties_pool_winner": ties_pool,
            "pool_winner_score": pool_winner_score,
        }

    return results


def main() -> int:
    print("Single Bracket Strategy Test")
    print(f"  1 bracket per strategy per year, 15 years, team-identity scoring")
    print(f"  12 strategies: construction × risk × contrarian × hybrid")

    years_sorted = sorted(YEARS)
    most_recent = max(years_sorted)
    weights = {y: DECAY ** (most_recent - y) for y in years_sorted}
    total_weight = sum(weights.values())

    all_results = {}
    for year in years_sorted:
        t0 = time.time()
        r = _run_year(year)
        if r is None:
            print(f"  {year}: SKIP")
            continue
        all_results[year] = r
        # Print top 3 strategies this year
        ranked = sorted(r.items(), key=lambda x: -x[1]["actual_score"])
        top3 = ", ".join(f"{n}={v['actual_score']:.0f}" for n, v in ranked[:3])
        print(f"  {year}: best={ranked[0][1]['actual_score']:.0f} ({ranked[0][0]})  "
              f"top3: {top3}  ({time.time()-t0:.1f}s)")

    # Collect all strategy names
    strat_names = list(all_results[years_sorted[0]].keys()) if years_sorted[0] in all_results else []

    # Aggregate table
    print(f"\n{'=' * 110}")
    print("AGGREGATE RESULTS (weighted avg across 15 years)")
    print(f"{'=' * 110}")
    print(f"  {'Strategy':<16} {'Actual':>8} {'ExpScr':>8} {'Leverage':>10} "
          f"{'F4 Corr':>8} {'Champ%':>8} {'PoolWins':>9} {'Top3 Yr':>8}")

    strat_stats = []
    for name in strat_names:
        w_actual = 0.0
        w_exp = 0.0
        w_lev = 0.0
        w_f4 = 0.0
        w_champ = 0.0
        w_total = 0.0
        pool_wins = 0
        top3_count = 0

        for year in years_sorted:
            if year not in all_results or name not in all_results[year]:
                continue
            r = all_results[year][name]
            w = weights[year]
            w_actual += r["actual_score"] * w
            w_exp += r["expected_score"] * w
            w_lev += r["leverage"] * w
            w_f4 += r["f4_correct"] * w
            w_champ += r["champ_correct"] * w
            w_total += w

            # Pool wins (2023-2026 only)
            if r.get("beats_pool_winner") or r.get("ties_pool_winner"):
                pool_wins += 1

            # Top-3 among strategies this year
            yr_scores = [(n, all_results[year][n]["actual_score"])
                         for n in strat_names if n in all_results[year]]
            yr_scores.sort(key=lambda x: -x[1])
            yr_rank = next((i + 1 for i, (n, _) in enumerate(yr_scores) if n == name), 99)
            if yr_rank <= 3:
                top3_count += 1

        if w_total > 0:
            avg_actual = w_actual / w_total
            avg_exp = w_exp / w_total
            avg_lev = w_lev / w_total
            avg_f4 = w_f4 / w_total
            avg_champ = w_champ / w_total * 100
            strat_stats.append((name, avg_actual, avg_exp, avg_lev, avg_f4, avg_champ, pool_wins, top3_count))

    strat_stats.sort(key=lambda x: -x[1])
    for name, avg_actual, avg_exp, avg_lev, avg_f4, avg_champ, pool_wins, top3 in strat_stats:
        print(f"  {name:<16} {avg_actual:>8.0f} {avg_exp:>8.0f} {avg_lev:>+10.1f} "
              f"{avg_f4:>8.2f} {avg_champ:>7.1f}% {pool_wins:>5}/4   {top3:>8}")

    # Pool detail
    print(f"\n  --- Pool Results (2023-2026) ---")
    for year in POOL_YEARS:
        if year not in all_results:
            continue
        pw = all_results[year][strat_names[0]].get("pool_winner_score")
        if pw is None:
            continue
        print(f"\n  {year} (pool winner: {pw:.0f} pts):")
        yr_ranked = sorted(
            [(n, all_results[year][n]["actual_score"]) for n in strat_names if n in all_results[year]],
            key=lambda x: -x[1],
        )
        for name, score in yr_ranked:
            beat = "WIN" if all_results[year][name]["beats_pool_winner"] else \
                   "TIE" if all_results[year][name]["ties_pool_winner"] else ""
            print(f"    {name:<16} {score:>6.0f} pts  {beat}")

    # Per-year detail
    print(f"\n{'=' * 100}")
    print("PER-YEAR: actual_score by strategy")
    print(f"{'=' * 100}")

    header = f"  {'Strategy':<16}"
    for year in years_sorted:
        if year in all_results:
            header += f" {year:>5}"
    header += "  WtAvg"
    print(header)

    for name, avg_actual, *_ in strat_stats:
        row = f"  {name:<16}"
        for year in years_sorted:
            if year in all_results and name in all_results[year]:
                row += f" {all_results[year][name]['actual_score']:>5.0f}"
            else:
                row += f" {'N/A':>5}"
        row += f"  {avg_actual:>5.0f}"
        print(row)

    out_path = PROJECT_ROOT / "artifacts" / "single_bracket_strategies_2026-04-19.json"
    out = {
        "description": "Single bracket strategy comparison: 12 strategies × 15 years",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {
            "years": years_sorted,
            "gen_seed": GEN_SEED,
            "decay": DECAY,
            "scoring": "team_identity",
        },
        "per_year": {str(k): v for k, v in all_results.items()},
        "aggregate_ranking": [
            {"strategy": name, "weighted_avg_score": avg, "pool_wins": pw, "top3_years": t3}
            for name, avg, _, _, _, _, pw, t3 in strat_stats
        ],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Artifact: {out_path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
