"""Designed portfolio: intentionally diverse brackets along meaningful dimensions.

Instead of 50 random draws from the same distribution, construct a small
portfolio where each bracket is a deliberate variation along stable,
measurable dimensions:

Dimension 1: Champion (model's top N most likely champions)
  - Each bracket locks a different champion. Getting champion right is worth
    320 pts and is the single biggest differentiator.

Dimension 2: Risk level (temperature on non-locked games)
  - temp=0.0 (argmax): most chalky, highest expected score
  - temp=0.3: slightly loosened, allows near-toss-up upsets
  - temp=0.7: moderate risk, more upsets in uncertain games

Dimension 3: Late-round confidence (F4 lock vs no lock)
  - locked: champion + their regional F4 path forced
  - open: only champion locked, F4 sampled from probabilities

The portfolio is: top_champs × risk_levels × lock_modes, producing
a compact grid of ~12-20 brackets, each with a clear identity.

Selection: pick by expected_score (the best criterion, ρ = +0.44).

Also tested: momentum adjustment — boost teams whose conference
tournament seed implies recent strong performance (proxy for hot
streaks not captured in season-long Torvik ratings).
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


def _get_matchup_prob(t1, t2, round_probs):
    p1 = sum(round_probs.get(t1, {}).values())
    p2 = sum(round_probs.get(t2, {}).values())
    if p1 + p2 == 0:
        return 0.5
    return p1 / (p1 + p2)


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
                    scaled = logit / temperature
                    final_p = 1.0 / (1.0 + math.exp(-scaled))
                    bracket[gi] = rng.random() < final_p
                next_round.append(t1 if bracket[gi] else t2)
            gi += 1
        current_teams = next_round
    return bracket


def _get_top_champs(round_probs, n):
    teams = [(tid, probs.get("CHAMP", 0.0)) for tid, probs in round_probs.items()]
    teams.sort(key=lambda x: -x[1])
    return [(t, p) for t, p in teams[:n] if p > 0.001]


def _get_region_f4_leader(round_probs, regions, region):
    best_tid, best_p = None, 0.0
    for tid, probs in round_probs.items():
        raw = regions.get(tid, "")
        r = REGION_ALIASES.get(raw, raw)
        if r == region and probs.get("F4", 0.0) > best_p:
            best_tid = tid
            best_p = probs.get("F4", 0.0)
    return best_tid


def _compute_expected_score(bracket, first_round, round_probs):
    bp = picks_by_round(bracket, first_round)
    exp = 0.0
    for rnd, pts in ESPN_SCORING.items():
        for team in bp.get(rnd, set()):
            exp += round_probs.get(team, {}).get(rnd, 0.0) * pts
    return exp


def _compute_leverage(bracket, first_round, seeds, round_probs):
    bp = picks_by_round(bracket, first_round)
    lev = 0.0
    for rnd, pts in ESPN_SCORING.items():
        for team in bp.get(rnd, set()):
            model_prob = round_probs.get(team, {}).get(rnd, 0.0)
            seed = seeds.get(team, 16)
            public_prob = ESPN_PICK_RATES.get(rnd, {}).get(seed, 0.0)
            lev += (model_prob - public_prob) * pts
    return lev


def _build_designed_portfolio(first_round, round_probs, seeds, regions):
    """Build the designed portfolio: champions × risk × lock mode."""
    top_champs = _get_top_champs(round_probs, 5)
    risk_levels = [
        ("chalk", 0.0),
        ("mild", 0.3),
        ("moderate", 0.7),
    ]
    lock_modes = [
        ("champ_only", False),
        ("champ_f4", True),
    ]

    # Build F4 lock set (model's #1 F4 per region)
    f4_leaders = set()
    for region in ("East", "West", "South", "Midwest"):
        leader = _get_region_f4_leader(round_probs, regions, region)
        if leader:
            f4_leaders.add(leader)

    portfolio = []
    seed_counter = 0

    for champ_tid, champ_prob in top_champs:
        for risk_name, temp in risk_levels:
            for lock_name, use_f4_lock in lock_modes:
                locked = {champ_tid}
                if use_f4_lock:
                    # Lock all F4 leaders (champion is already locked)
                    locked |= f4_leaders

                rng = np.random.default_rng(GEN_SEED + seed_counter) if temp > 0 else None
                seed_counter += 1

                bracket = _build_bracket(
                    first_round, round_probs, locked_teams=locked,
                    temperature=temp, rng=rng,
                )

                champ_seed = seeds.get(champ_tid, 16)
                name = f"{champ_tid[:12]}_s{champ_seed}_{risk_name}_{lock_name}"

                portfolio.append({
                    "name": name,
                    "bracket": bracket,
                    "champion": champ_tid,
                    "champ_seed": champ_seed,
                    "champ_prob": champ_prob,
                    "risk_level": risk_name,
                    "temperature": temp,
                    "lock_mode": lock_name,
                    "expected_score": _compute_expected_score(bracket, first_round, round_probs),
                    "leverage": _compute_leverage(bracket, first_round, seeds, round_probs),
                })

    return portfolio


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

    # Pool data
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

    portfolio = _build_designed_portfolio(first_round, round_probs, seeds, regions)

    # Score all brackets against reality
    brackets_2d = np.stack([b["bracket"] for b in portfolio])
    actual_scores = score_brackets_team_identity(
        brackets_2d, real_winners, first_round, ESPN_SCORING,
    )

    results = []
    for i, entry in enumerate(portfolio):
        bp = picks_by_round(entry["bracket"], first_round)
        f4_corr = len(bp.get("F4", set()) & actual_f4)
        champ_corr = 1 if bp.get("CHAMP", set()) & actual_champ else 0

        results.append({
            "name": entry["name"],
            "champion": entry["champion"],
            "champ_seed": entry["champ_seed"],
            "champ_prob": entry["champ_prob"],
            "risk_level": entry["risk_level"],
            "lock_mode": entry["lock_mode"],
            "expected_score": entry["expected_score"],
            "leverage": entry["leverage"],
            "actual_score": float(actual_scores[i]),
            "f4_correct": f4_corr,
            "champ_correct": champ_corr,
        })

    # Selection: pick by expected_score
    exp_scores = [r["expected_score"] for r in results]
    selected_idx = int(np.argmax(exp_scores))
    selected = results[selected_idx]

    # Best possible
    best_idx = int(np.argmax(actual_scores))
    best = results[best_idx]

    return {
        "n_brackets": len(portfolio),
        "pool_winner_score": pool_winner_score,
        "selected": selected,
        "best": best,
        "all_brackets": results,
    }


def main() -> int:
    print("Designed Portfolio Test")
    print(f"  Grid: top-5 champions × 3 risk levels × 2 lock modes = up to 30 brackets")
    print(f"  Selection: highest expected_score")

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
        sel = r["selected"]
        bst = r["best"]
        pw = r["pool_winner_score"]
        pw_str = f"  pool={pw:.0f}" if pw else ""
        beat = ""
        if pw and sel["actual_score"] > pw:
            beat = " POOL WIN"
        elif pw and sel["actual_score"] == pw:
            beat = " POOL TIE"
        print(f"  {year}: selected={sel['actual_score']:.0f} ({sel['name']})  "
              f"best={bst['actual_score']:.0f} ({bst['name']}){pw_str}{beat}  "
              f"n={r['n_brackets']}  ({time.time()-t0:.1f}s)")

    # Aggregate
    print(f"\n{'=' * 100}")
    print("AGGREGATE")
    print(f"{'=' * 100}")

    w_selected = sum(weights[y] * all_results[y]["selected"]["actual_score"] for y in all_results) / total_weight
    w_best = sum(weights[y] * all_results[y]["best"]["actual_score"] for y in all_results) / total_weight
    pool_wins = sum(
        1 for y in POOL_YEARS
        if y in all_results and all_results[y]["pool_winner_score"]
        and all_results[y]["selected"]["actual_score"] >= all_results[y]["pool_winner_score"]
    )
    print(f"  Weighted avg selected score:  {w_selected:.0f}")
    print(f"  Weighted avg best-of-N score: {w_best:.0f}")
    print(f"  Pool wins (2023-2026):        {pool_wins}/4")
    print(f"  Selection captures {w_selected/w_best*100:.0f}% of ceiling")

    # Dimension analysis: which champion rank works best?
    print(f"\n  --- By Champion Rank (1st = model favorite, 5th = long shot) ---")
    for champ_rank in range(1, 6):
        w_score = 0.0
        w_total = 0.0
        champ_wins = 0
        f4_sum = 0.0
        n_years = 0
        for y in all_results:
            # Find brackets with this champion rank
            champs_sorted = sorted(
                set(b["champion"] for b in all_results[y]["all_brackets"]),
                key=lambda c: -max(b["champ_prob"] for b in all_results[y]["all_brackets"] if b["champion"] == c),
            )
            if champ_rank > len(champs_sorted):
                continue
            target_champ = champs_sorted[champ_rank - 1]
            # Best bracket with this champion (by actual score)
            champ_brackets = [b for b in all_results[y]["all_brackets"] if b["champion"] == target_champ]
            if not champ_brackets:
                continue
            best_score = max(b["actual_score"] for b in champ_brackets)
            best_bracket = max(champ_brackets, key=lambda b: b["actual_score"])
            w = weights[y]
            w_score += best_score * w
            f4_sum += best_bracket["f4_correct"] * w
            champ_wins += best_bracket["champ_correct"]
            w_total += w
            n_years += 1
        if w_total > 0:
            print(f"    Champ #{champ_rank}: avg_score={w_score/w_total:>7.0f}  "
                  f"champ_correct={champ_wins}/{n_years}  "
                  f"f4_avg={f4_sum/w_total:.2f}")

    # Dimension analysis: which risk level works best?
    print(f"\n  --- By Risk Level ---")
    for risk in ["chalk", "mild", "moderate"]:
        w_score = 0.0
        w_total = 0.0
        for y in all_results:
            risk_brackets = [b for b in all_results[y]["all_brackets"] if b["risk_level"] == risk]
            if not risk_brackets:
                continue
            best_score = max(b["actual_score"] for b in risk_brackets)
            w = weights[y]
            w_score += best_score * w
            w_total += w
        if w_total > 0:
            print(f"    {risk:<12} best-of-group avg: {w_score/w_total:>7.0f}")

    # Dimension analysis: lock mode
    print(f"\n  --- By Lock Mode ---")
    for lock in ["champ_only", "champ_f4"]:
        w_score = 0.0
        w_total = 0.0
        for y in all_results:
            lock_brackets = [b for b in all_results[y]["all_brackets"] if b["lock_mode"] == lock]
            if not lock_brackets:
                continue
            best_score = max(b["actual_score"] for b in lock_brackets)
            w = weights[y]
            w_score += best_score * w
            w_total += w
        if w_total > 0:
            print(f"    {lock:<14} best-of-group avg: {w_score/w_total:>7.0f}")

    # Pool year detail
    print(f"\n{'=' * 100}")
    print("POOL YEAR DETAIL (2023-2026)")
    print(f"{'=' * 100}")
    for year in POOL_YEARS:
        if year not in all_results:
            continue
        r = all_results[year]
        pw = r["pool_winner_score"]
        print(f"\n  {year} — pool winner: {pw:.0f} pts, portfolio size: {r['n_brackets']}")
        print(f"  Selected: {r['selected']['name']} = {r['selected']['actual_score']:.0f} pts "
              f"(exp={r['selected']['expected_score']:.0f}, f4={r['selected']['f4_correct']}, "
              f"champ={'Y' if r['selected']['champ_correct'] else 'N'})")
        print(f"  Best:     {r['best']['name']} = {r['best']['actual_score']:.0f} pts "
              f"(f4={r['best']['f4_correct']}, champ={'Y' if r['best']['champ_correct'] else 'N'})")

        # Show top 5 brackets
        sorted_brackets = sorted(r["all_brackets"], key=lambda b: -b["actual_score"])
        print(f"  Top 5:")
        for b in sorted_brackets[:5]:
            beat = " *WIN*" if pw and b["actual_score"] > pw else \
                   " *TIE*" if pw and b["actual_score"] == pw else ""
            print(f"    {b['name']:<40} {b['actual_score']:>5.0f} pts  "
                  f"exp={b['expected_score']:>4.0f}  f4={b['f4_correct']}  "
                  f"champ={'Y' if b['champ_correct'] else 'N'}{beat}")

        # How many beat the pool?
        n_beat = sum(1 for b in r["all_brackets"] if pw and b["actual_score"] > pw)
        n_tie = sum(1 for b in r["all_brackets"] if pw and b["actual_score"] == pw)
        print(f"  Brackets beating pool: {n_beat} (+{n_tie} tied) / {r['n_brackets']}")

    out_path = PROJECT_ROOT / "artifacts" / "designed_portfolio_test_2026-04-19.json"
    out = {
        "description": "Designed portfolio: champions × risk × lock mode, selected by expected_score",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {
            "years": years_sorted,
            "gen_seed": GEN_SEED,
            "decay": DECAY,
            "champions": "top 5 by model CHAMP probability",
            "risk_levels": ["chalk (0.0)", "mild (0.3)", "moderate (0.7)"],
            "lock_modes": ["champ_only", "champ_f4"],
        },
        "per_year": {
            str(k): {
                "n_brackets": v["n_brackets"],
                "pool_winner_score": v["pool_winner_score"],
                "selected": v["selected"],
                "best": v["best"],
                "all_brackets": v["all_brackets"],
            }
            for k, v in all_results.items()
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Artifact: {out_path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
