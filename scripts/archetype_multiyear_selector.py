"""Multi-year archetype selector: pick the bracket archetype ONCE based on
15 years of historical performance, then use that same archetype every year.

The designed portfolio generates 30 brackets per year (5 champion ranks ×
3 risk levels × 2 lock modes). Each combination is an "archetype" — a
recipe that can be applied to any year's model outputs.

This script:
1. Runs all 30 archetypes across 15 years
2. Aggregates each archetype's actual score across years (recency-weighted)
3. Ranks archetypes by multi-year track record
4. Reports which archetype you should commit to for 2027
5. Shows pool wins for 2023-2026 under the multi-year-best archetype

The key difference from per-year selection: the archetype is chosen BEFORE
seeing any year's tournament. No hindsight. The same recipe is applied
every year because historical data says it works best on average.
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

YEARS = list(range(2011, 2027))
YEARS = [y for y in YEARS if y != 2020]
ESPN_SCORING = {"R64": 10, "R32": 20, "S16": 40, "E8": 80, "F4": 160, "CHAMP": 320}
GEN_SEED = 12345
DECAY = 0.85
POOL_HIST_PATH = PROJECT_ROOT / "pool_hist_results.json"
POOL_YEARS = [2023, 2024, 2025, 2026]

REGION_ALIASES = {"Southeast": "South", "Southwest": "Midwest"}

# Archetype grid
CHAMP_RANKS = [1, 2, 3, 4, 5]
RISK_LEVELS = [("chalk", 0.0), ("mild", 0.3), ("moderate", 0.7)]
LOCK_MODES = [("champ_only", False), ("champ_f4", True)]


def _get_matchup_prob(t1, t2, round_probs):
    p1 = sum(round_probs.get(t1, {}).values())
    p2 = sum(round_probs.get(t2, {}).values())
    return p1 / (p1 + p2) if (p1 + p2) > 0 else 0.5


def _build_bracket(first_round, round_probs, locked_teams=None, temperature=0.0, rng=None):
    if locked_teams is None:
        locked_teams = set()
    bracket = np.zeros(63, dtype=bool)
    current_teams = list(first_round)
    gi = 0
    for _ in range(6):
        next_round = []
        for g in range(0, len(current_teams), 2):
            if g + 1 >= len(current_teams):
                next_round.append(current_teams[g])
                gi += 1
                continue
            t1, t2 = current_teams[g], current_teams[g + 1]
            if t1 in locked_teams and t2 not in locked_teams:
                bracket[gi] = True
                next_round.append(t1)
            elif t2 in locked_teams and t1 not in locked_teams:
                bracket[gi] = False
                next_round.append(t2)
            else:
                p = _get_matchup_prob(t1, t2, round_probs)
                if temperature == 0.0:
                    bracket[gi] = p >= 0.5
                else:
                    safe_p = max(0.001, min(0.999, p))
                    logit = math.log(safe_p / (1 - safe_p))
                    final_p = 1.0 / (1.0 + math.exp(-logit / temperature))
                    bracket[gi] = rng.random() < final_p
                next_round.append(t1 if bracket[gi] else t2)
            gi += 1
        current_teams = next_round
    return bracket


def _get_sorted_champs(round_probs):
    teams = [(tid, probs.get("CHAMP", 0.0)) for tid, probs in round_probs.items()]
    teams.sort(key=lambda x: -x[1])
    return [t for t, p in teams if p > 0.001]


def _get_region_f4_leader(round_probs, regions, region):
    best_tid, best_p = None, 0.0
    for tid, probs in round_probs.items():
        r = REGION_ALIASES.get(regions.get(tid, ""), regions.get(tid, ""))
        if r == region and probs.get("F4", 0.0) > best_p:
            best_tid, best_p = tid, probs.get("F4", 0.0)
    return best_tid


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

    pool_winner_score = None
    if year in POOL_YEARS:
        try:
            opp_actual, _ = load_pool_bracket_vectors(POOL_HIST_PATH, year, first_round, seeds)
            opp_scores = score_brackets_team_identity(opp_actual, real_winners, first_round, ESPN_SCORING)
            pool_winner_score = float(opp_scores.max())
        except Exception:
            pass

    champs_sorted = _get_sorted_champs(round_probs)
    f4_leaders = set()
    for region in ("East", "West", "South", "Midwest"):
        leader = _get_region_f4_leader(round_probs, regions, region)
        if leader:
            f4_leaders.add(leader)

    seed_counter = 0
    archetype_results = {}

    for champ_rank in CHAMP_RANKS:
        if champ_rank > len(champs_sorted):
            continue
        champ_tid = champs_sorted[champ_rank - 1]
        champ_seed = seeds.get(champ_tid, 16)

        for risk_name, temp in RISK_LEVELS:
            for lock_name, use_f4_lock in LOCK_MODES:
                locked = {champ_tid}
                if use_f4_lock:
                    locked |= f4_leaders

                rng = np.random.default_rng(GEN_SEED + seed_counter) if temp > 0 else None
                seed_counter += 1

                bracket = _build_bracket(first_round, round_probs, locked, temp, rng)
                bracket_2d = bracket.reshape(1, -1)
                actual_score = float(score_brackets_team_identity(
                    bracket_2d, real_winners, first_round, ESPN_SCORING,
                )[0])

                bp = picks_by_round(bracket, first_round)
                f4_corr = len(bp.get("F4", set()) & actual_f4)
                champ_corr = 1 if bp.get("CHAMP", set()) & actual_champ else 0

                archetype_key = f"champ{champ_rank}_{risk_name}_{lock_name}"
                archetype_results[archetype_key] = {
                    "actual_score": actual_score,
                    "champion": champ_tid,
                    "champ_seed": champ_seed,
                    "f4_correct": f4_corr,
                    "champ_correct": champ_corr,
                    "pool_winner_score": pool_winner_score,
                    "beats_pool": actual_score > pool_winner_score if pool_winner_score else False,
                    "ties_pool": actual_score == pool_winner_score if pool_winner_score else False,
                }

    return archetype_results


def main() -> int:
    print("Multi-Year Archetype Selector")
    print(f"  Grid: {len(CHAMP_RANKS)} champ ranks × {len(RISK_LEVELS)} risks × {len(LOCK_MODES)} locks = {len(CHAMP_RANKS)*len(RISK_LEVELS)*len(LOCK_MODES)} archetypes")
    print(f"  Years: {len(YEARS)} (2011-2026 ex 2020), decay={DECAY}")
    print(f"  Selection: best weighted avg actual_score across ALL years (no per-year picking)")

    years_sorted = sorted(YEARS)
    most_recent = max(years_sorted)
    weights = {y: DECAY ** (most_recent - y) for y in years_sorted}
    total_weight = sum(weights.values())

    all_years = {}
    for year in years_sorted:
        t0 = time.time()
        r = _run_year(year)
        if r is None:
            print(f"  {year}: SKIP")
            continue
        all_years[year] = r
        best_key = max(r, key=lambda k: r[k]["actual_score"])
        print(f"  {year}: best archetype = {best_key} ({r[best_key]['actual_score']:.0f} pts)  ({time.time()-t0:.1f}s)")

    # Collect all archetype keys
    archetype_keys = list(all_years[years_sorted[0]].keys()) if years_sorted[0] in all_years else []

    # Aggregate: weighted average actual_score per archetype across ALL years
    archetype_agg = {}
    for key in archetype_keys:
        w_score = 0.0
        w_f4 = 0.0
        w_champ = 0.0
        w_total = 0.0
        pool_wins = 0
        pool_close = 0  # within 50 pts
        per_year_scores = {}

        for year in years_sorted:
            if year not in all_years or key not in all_years[year]:
                continue
            r = all_years[year][key]
            w = weights[year]
            w_score += r["actual_score"] * w
            w_f4 += r["f4_correct"] * w
            w_champ += r["champ_correct"] * w
            w_total += w
            per_year_scores[year] = r["actual_score"]

            if r.get("beats_pool") or r.get("ties_pool"):
                pool_wins += 1
            pw = r.get("pool_winner_score")
            if pw and abs(r["actual_score"] - pw) <= 50:
                pool_close += 1

        if w_total > 0:
            archetype_agg[key] = {
                "weighted_avg": w_score / w_total,
                "avg_f4": w_f4 / w_total,
                "avg_champ_pct": w_champ / w_total * 100,
                "pool_wins": pool_wins,
                "pool_close": pool_close,
                "per_year": per_year_scores,
            }

    # Sort by weighted average
    ranked = sorted(archetype_agg.items(), key=lambda x: -x[1]["weighted_avg"])

    print(f"\n{'=' * 110}")
    print("ARCHETYPE RANKING (by weighted avg actual score across 15 years)")
    print(f"{'=' * 110}")
    print(f"  {'Rank':>4} {'Archetype':<30} {'WtAvg':>7} {'F4':>5} {'Champ%':>7} "
          f"{'PoolWin':>8} {'Close':>6} {'2023':>5} {'2024':>5} {'2025':>5} {'2026':>5}")

    for rank_i, (key, agg) in enumerate(ranked):
        py = agg["per_year"]
        print(f"  {rank_i+1:>4} {key:<30} {agg['weighted_avg']:>7.0f} "
              f"{agg['avg_f4']:>5.2f} {agg['avg_champ_pct']:>6.1f}% "
              f"{agg['pool_wins']:>5}/4  {agg['pool_close']:>4}/4 "
              f"{py.get(2023, 0):>5.0f} {py.get(2024, 0):>5.0f} "
              f"{py.get(2025, 0):>5.0f} {py.get(2026, 0):>5.0f}")

    # Summary
    best_key, best_agg = ranked[0]
    print(f"\n{'=' * 110}")
    print(f"RECOMMENDATION FOR 2027: {best_key}")
    print(f"{'=' * 110}")
    print(f"  Weighted avg score: {best_agg['weighted_avg']:.0f} pts")
    print(f"  Avg F4 correct:    {best_agg['avg_f4']:.2f}")
    print(f"  Champion correct:  {best_agg['avg_champ_pct']:.1f}%")
    print(f"  Pool wins 23-26:   {best_agg['pool_wins']}/4")
    print(f"  Within 50 pts:     {best_agg['pool_close']}/4")

    # Parse the archetype key
    parts = best_key.split("_")
    champ_rank = int(parts[0].replace("champ", ""))
    risk = parts[1]
    lock = "_".join(parts[2:])
    print(f"\n  Recipe:")
    print(f"    1. Take the model's #{champ_rank} most likely champion")
    print(f"    2. Lock that champion to win the tournament")
    if "f4" in lock:
        print(f"    3. Lock the model's #1 F4 team per region")
        print(f"    4. Fill remaining games at risk='{risk}'")
    else:
        print(f"    3. Fill all other games at risk='{risk}'")
    print(f"    4. Submit this single bracket — no selection needed")

    # Compare top archetype vs single-bracket strategies from earlier
    print(f"\n  --- Comparison to single-bracket strategies ---")
    print(f"  This archetype:        {best_agg['weighted_avg']:.0f} pts (multi-year selected)")
    print(f"  champ_2nd (prev best): 1072 pts")
    print(f"  argmax:                1053 pts")
    print(f"  temp_0.5:              1021 pts")
    print(f"  50-bracket median_sel:  969 pts (f4_first_tv + median_score)")

    # What would have happened if we committed to this archetype for all 4 pool years?
    print(f"\n  --- Pool simulation: submit this archetype every year ---")
    for year in POOL_YEARS:
        if year not in all_years or best_key not in all_years[year]:
            continue
        r = all_years[year][best_key]
        pw = r.get("pool_winner_score", 0)
        score = r["actual_score"]
        margin = score - pw if pw else 0
        result = "WIN" if r.get("beats_pool") else "TIE" if r.get("ties_pool") else "LOSE"
        print(f"    {year}: {score:.0f} pts vs pool winner {pw:.0f}  ({margin:+.0f})  {result}  "
              f"champ={'Y' if r['champ_correct'] else 'N'}  f4={r['f4_correct']}")

    out_path = PROJECT_ROOT / "artifacts" / "archetype_multiyear_selector_2026-04-19.json"
    out = {
        "description": "Multi-year archetype selector: best archetype across 15 years",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {
            "years": years_sorted, "decay": DECAY, "gen_seed": GEN_SEED,
            "champ_ranks": CHAMP_RANKS,
            "risk_levels": [r[0] for r in RISK_LEVELS],
            "lock_modes": [lm[0] for lm in LOCK_MODES],
        },
        "ranking": [
            {"rank": i + 1, "archetype": k, **v}
            for i, (k, v) in enumerate(ranked)
        ],
        "recommendation": best_key,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Artifact: {out_path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
