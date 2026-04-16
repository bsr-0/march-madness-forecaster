"""G3 portfolio diversity diagnostic for §2 O25.

Measures pairwise score correlation across the 50-bracket torvik portfolio
per year (2023-2026) to determine whether over-correlation explains the
2025 coverage failure (no bracket scored >= 1620 against pool winner).

Algorithm
---------
For each year:
  1. Build torvik round probabilities (barthag-based Log5).
  2. Sample N_BRACKETS=50 champ_first_tv stochastic brackets.
  3. Run N_TOURN=5000 MC tournament outcomes.
  4. Score each bracket against each outcome -> (N_TOURN, N_BRACKETS) matrix.
  5. Compute pairwise Pearson r for all (50 choose 2)=1225 bracket pairs.
  6. Report mean pairwise r, distribution stats.
  7. Compute R64 upset-pick counts per bracket.
  8. Report whether any bracket is in top decile of upset-count distribution.

Special 2025 analysis
---------------------
For 2025, also score each bracket against the actual tournament outcome
to confirm whether coverage (max achievable = 1620?) is the binding
constraint, or whether any bracket in the portfolio could have scored >=1620
had it been generated with different upset picks.

Gates (O25 G3)
--------------
  PASS: mean pairwise correlation < 0.60 AND at least one bracket in top
        decile of upset-count distribution across all 4 years.
  FAIL: mean pairwise correlation >= 0.60 for any year (portfolio too
        homogeneous — diversity injection required) OR top-decile upset
        coverage is absent (model doesn't generate enough upset brackets).
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.stats import pearsonr, spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts._common import load_tournament_results
from src.simulation.pool_competition import (
    build_scoring_vector,
    simulate_tournament_outcomes,
    score_brackets_against_outcome,
    ROUND_NAMES,
    GAMES_PER_ROUND,
)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

YEARS = [2023, 2024, 2025, 2026]
N_BRACKETS = 50
N_TOURN = 5000
NOISE_STD = 0.16
ESPN_SCORING = {"R64": 10, "R32": 20, "S16": 40, "E8": 80, "F4": 160, "CHAMP": 320}
HIST_DIR = PROJECT_ROOT / "data" / "raw" / "historical"
SEED_MATCHUP_ORDER = [(1, 16), (8, 9), (5, 12), (4, 13), (6, 11), (3, 14), (7, 10), (2, 15)]
REGION_ORDER = ["East", "West", "South", "Midwest"]
_REGION_ALIASES = {"Southeast": "South", "Southwest": "Midwest"}
_LOCK_ROUND_INDEX = {"R64": 0, "R32": 1, "S16": 2, "E8": 3, "F4": 4, "CHAMP": 5}

# Actual pool winner scores from pool_hist_results.json backtest
# (2025 pool winner scored 1620; other years provided for reference)
POOL_WINNER_SCORES = {2023: None, 2024: None, 2025: 1620, 2026: None}


# ---------------------------------------------------------------------------
# Inlined helpers (sklearn-free subset of mc_pool_backtest)
# ---------------------------------------------------------------------------


def load_seeds_and_regions(year):
    path = HIST_DIR / f"tournament_seeds_{year}.json"
    if not path.exists():
        return {}, {}
    with open(path) as f:
        data = json.load(f)
    seeds, regions = {}, {}
    if isinstance(data, dict) and "teams" in data:
        for t in data["teams"]:
            seeds[t["team_id"]] = t["seed"]
            raw = t.get("region", "")
            regions[t["team_id"]] = _REGION_ALIASES.get(raw, raw)
    return seeds, regions


def resolve_first_four(games, seeds, regions):
    for g in [g for g in games if g.get("round_name") == "FF"]:
        loser = g["team2_id"] if g["team1_won"] else g["team1_id"]
        seeds.pop(loser, None)
        regions.pop(loser, None)


def derive_f4_region_pairing(games, regions):
    f4 = [g for g in games if g.get("round_name") == "F4"]
    if len(f4) < 2:
        raise ValueError(f"need 2 F4 games, got {len(f4)}")
    pairs = []
    for g in f4[:2]:
        r1 = regions.get(g["team1_id"])
        r2 = regions.get(g["team2_id"])
        if not r1 or not r2:
            raise ValueError(f"missing region for F4 game {g['team1_id']} vs {g['team2_id']}")
        pairs.append((r1, r2))
    all_r = {r for p in pairs for r in p}
    if len(all_r) != 4:
        raise ValueError(f"F4 pairs don't cover 4 distinct regions: {pairs}")
    return (pairs[0][0], pairs[0][1], pairs[1][0], pairs[1][1])


def build_first_round_matchups(seeds, regions, region_order=REGION_ORDER):
    matchups = []
    by_region = defaultdict(dict)
    for tid, seed in seeds.items():
        by_region[regions.get(tid, "")][seed] = tid
    for region in region_order:
        rt = by_region.get(region, {})
        for hi, lo in SEED_MATCHUP_ORDER:
            matchups.extend([rt.get(hi, f"unk_{region}_{hi}"), rt.get(lo, f"unk_{region}_{lo}")])
    return matchups


def _load_barthag(year, seeds):
    """Load Torvik barthag ratings; fall back to seed-based estimate."""
    barthag = {}
    for prefix in [HIST_DIR, PROJECT_ROOT / "data" / "raw"]:
        path = prefix / f"torvik_{year}.json"
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            for t in data.get("teams", []):
                tid = t.get("team_id", "")
                b = t.get("barthag")
                if tid in seeds and b is not None:
                    barthag[tid] = float(b)
            break
    for tid, seed in seeds.items():
        if tid not in barthag:
            barthag[tid] = max(0.10, 1.0 - seed * 0.04)
    return barthag


def _log5(pa, pb):
    num = pa * (1 - pb)
    denom = pa * (1 - pb) + pb * (1 - pa)
    return num / denom if denom > 1e-12 else 0.5


def build_torvik_round_probabilities(seeds, regions, barthag, n_sims=10_000):
    """Monte Carlo round advancement probabilities from Torvik barthag."""
    rng = np.random.default_rng(42)
    by_region = {r: {} for r in REGION_ORDER}
    for tid, seed in seeds.items():
        r = regions.get(tid, "")
        if r in by_region:
            by_region[r][seed] = tid

    bracket_order = []
    for region in REGION_ORDER:
        rt = by_region[region]
        for hi, lo in SEED_MATCHUP_ORDER:
            bracket_order.extend([rt.get(hi, f"unk_{region}_{hi}"), rt.get(lo, f"unk_{region}_{lo}")])

    counts = {tid: {rnd: 0 for rnd in ROUND_NAMES} for tid in seeds}
    for _ in range(n_sims):
        current = list(bracket_order)
        for rnd in ROUND_NAMES:
            nxt = []
            for g in range(0, len(current), 2):
                t1, t2 = current[g], current[g + 1]
                p1 = _log5(barthag.get(t1, 0.5), barthag.get(t2, 0.5))
                winner = t1 if rng.random() < p1 else t2
                if winner in counts:
                    counts[winner][rnd] += 1
                nxt.append(winner)
            current = nxt

    return {tid: {rnd: max(0.001, counts[tid][rnd] / n_sims) for rnd in ROUND_NAMES} for tid in seeds}


def build_seed_probabilities(seeds):
    """Pairwise win probability lookup keyed by (team_id_a, team_id_b)."""
    from src.prediction.seed_probabilities import build_seed_probabilities as _bsp

    return _bsp(seeds)


def _draw_categorical(rng, items, weights):
    w = np.asarray(weights, dtype=float)
    total = w.sum()
    w = w / total if total > 1e-12 else np.ones(len(items)) / len(items)
    return items[rng.choice(len(items), p=w)]


def _sample_with_locks(first_round, round_probs, n, rng, locked_per_sample, lock_through):
    all_b = np.zeros((n, 63), dtype=bool)
    lock_idx = _LOCK_ROUND_INDEX.get(lock_through, -1) if lock_through else -1
    for b in range(n):
        locked = locked_per_sample[b]
        current = list(first_round)
        gi = 0
        for ri, rnd in enumerate(ROUND_NAMES):
            nxt = []
            in_lock = lock_idx >= 0 and ri <= lock_idx
            for g in range(0, len(current), 2):
                t1, t2 = current[g], current[g + 1]
                forced = None
                if in_lock and locked:
                    if t1 in locked:
                        forced = t1
                    elif t2 in locked:
                        forced = t2
                if forced is not None:
                    winner = forced
                    all_b[b, gi] = winner == t1
                else:
                    p1 = round_probs.get(t1, {}).get(rnd, 0.0)
                    p2 = round_probs.get(t2, {}).get(rnd, 0.0)
                    p_t1 = p1 / (p1 + p2) if p1 + p2 > 1e-8 else 0.5
                    if rng.random() < p_t1:
                        winner = t1
                        all_b[b, gi] = True
                    else:
                        winner = t2
                        all_b[b, gi] = False
                nxt.append(winner)
                gi += 1
            current = nxt
    return all_b


def sample_champ_first_brackets(first_round, round_probs, n, rng):
    teams = list(round_probs.keys())
    weights = [round_probs[t].get("CHAMP", 0.0) for t in teams]
    locks = [{_draw_categorical(rng, teams, weights)} for _ in range(n)]
    return _sample_with_locks(first_round, round_probs, n, rng, locks, "CHAMP")


def build_actual_outcome(first_round, games):
    """Convert game results into (63,) boolean outcome vector."""
    results = {}
    for g in games:
        if g.get("round_name") == "FF":
            continue
        t1, t2 = g["team1_id"], g["team2_id"]
        results[(t1, t2)] = g["team1_won"]
        results[(t2, t1)] = not g["team1_won"]

    outcome = np.zeros(63, dtype=bool)
    current = list(first_round)
    gi = 0
    for rnd in ROUND_NAMES:
        nxt = []
        for g in range(0, len(current), 2):
            t1, t2 = current[g], current[g + 1]
            t1_won = results.get((t1, t2), True)  # default: higher seed wins
            outcome[gi] = t1_won
            nxt.append(t1 if t1_won else t2)
            gi += 1
        current = nxt
    return outcome


def count_r64_upsets(brackets):
    """Count R64 upset picks per bracket (picks where lower seed wins).

    The first 32 games in the bracket vector are R64.
    For each game, team1 is the higher seed (lower seed number).
    An upset = bracket[b, game_idx] = False (team2/lower-seed wins).
    """
    r64_slice = brackets[:, :32]  # (n_brackets, 32)
    # upset = team2 wins = False
    return (~r64_slice).sum(axis=1)  # (n_brackets,)


# ---------------------------------------------------------------------------
# Main per-year analysis
# ---------------------------------------------------------------------------


def _run_year(year, rng):
    print(f"\n{'=' * 60}")
    print(f"Year {year}")
    print(f"{'=' * 60}")

    games = load_tournament_results(year)
    if not games:
        print(f"  SKIP — no tournament results")
        return None

    seeds, regions = load_seeds_and_regions(year)
    resolve_first_four(games, seeds, regions)

    try:
        region_order = derive_f4_region_pairing(games, regions)
    except ValueError as exc:
        print(f"  SKIP — {exc}")
        return None

    first_round = build_first_round_matchups(seeds, regions, region_order=region_order)
    if len(first_round) != 64:
        print(f"  SKIP — {len(first_round)} teams in first round (need 64)")
        return None

    barthag = _load_barthag(year, seeds)
    round_probs = build_torvik_round_probabilities(seeds, regions, barthag)
    seed_pw = build_seed_probabilities(seeds)
    scoring_vector = build_scoring_vector(ESPN_SCORING)

    # --- Generate 50 torvik (champ_first_tv) brackets ---
    brackets = sample_champ_first_brackets(first_round, round_probs, N_BRACKETS, rng)
    # shape: (N_BRACKETS, 63) bool

    # --- Actual tournament outcome ---
    actual_outcome = build_actual_outcome(first_round, games)
    actual_scores = score_brackets_against_outcome(brackets, actual_outcome, scoring_vector)

    # --- MC tournament simulations -> score matrix ---
    outcomes, _ = simulate_tournament_outcomes(N_TOURN, first_round, seed_pw, seeds, NOISE_STD, rng)
    score_matrix = np.zeros((N_TOURN, N_BRACKETS), dtype=np.float64)
    for t in range(N_TOURN):
        score_matrix[t] = score_brackets_against_outcome(brackets, outcomes[t], scoring_vector)
    # shape: (N_TOURN, N_BRACKETS)

    # --- Pairwise score correlations ---
    # For each pair (i, j), Pearson r between score_matrix[:, i] and score_matrix[:, j]
    n = N_BRACKETS
    pairwise_r = []
    for i in range(n):
        for j in range(i + 1, n):
            r, _ = pearsonr(score_matrix[:, i], score_matrix[:, j])
            pairwise_r.append(r)
    pairwise_r = np.array(pairwise_r)

    mean_r = float(pairwise_r.mean())
    median_r = float(np.median(pairwise_r))
    p10_r = float(np.percentile(pairwise_r, 10))
    p90_r = float(np.percentile(pairwise_r, 90))

    # --- Upset-count distribution ---
    upset_counts = count_r64_upsets(brackets)
    p90_upset = float(np.percentile(upset_counts, 90))
    top_decile_threshold = np.percentile(upset_counts, 90)
    n_top_decile = int((upset_counts >= top_decile_threshold).sum())

    # --- Score analysis against actual outcome ---
    max_actual = float(actual_scores.max())
    min_actual = float(actual_scores.min())
    mean_actual = float(actual_scores.mean())
    p90_actual = float(np.percentile(actual_scores, 90))
    top5_actual = sorted(actual_scores.tolist(), reverse=True)[:5]

    # --- Pool winner comparison ---
    pool_winner_score = POOL_WINNER_SCORES.get(year)
    beats_pool = None
    if pool_winner_score is not None:
        beats_pool = int((actual_scores >= pool_winner_score).sum())

    # --- 2025 coverage deep-dive: what upset profile would clear 1620? ---
    high_score_brackets = None
    if year == 2025 and pool_winner_score is not None:
        threshold = pool_winner_score
        above = np.where(actual_scores >= threshold)[0]
        if len(above) > 0:
            high_score_brackets = {
                "n": len(above),
                "scores": actual_scores[above].tolist(),
                "upset_counts": upset_counts[above].tolist(),
            }
        else:
            # Find what the closest brackets look like
            top_by_score = np.argsort(-actual_scores)[:5]
            high_score_brackets = {
                "n": 0,
                "note": f"No bracket reached {threshold}. Top-5 scores: {actual_scores[top_by_score].tolist()}",
                "top5_upset_counts": upset_counts[top_by_score].tolist(),
                "top5_actual_scores": actual_scores[top_by_score].tolist(),
            }

    # --- Print summary ---
    print(f"  Portfolio size: {N_BRACKETS} brackets")
    print(f"\n  Pairwise score correlation:")
    print(f"    mean r = {mean_r:.4f}  (gate < 0.60: {'PASS' if mean_r < 0.60 else 'FAIL'})")
    print(f"    median r = {median_r:.4f},  p10={p10_r:.4f},  p90={p90_r:.4f}")
    print(f"\n  R64 upset picks per bracket:")
    print(f"    mean = {upset_counts.mean():.1f},  min = {upset_counts.min()},  max = {upset_counts.max()}")
    print(f"    p90 threshold = {p90_upset:.1f},  brackets in top decile = {n_top_decile}")
    print(f"    Top-decile gate (>= 1 bracket): {'PASS' if n_top_decile >= 1 else 'FAIL'}")
    print(f"\n  Scores against actual {year} tournament outcome:")
    print(f"    max = {max_actual:.0f},  mean = {mean_actual:.0f},  min = {min_actual:.0f}")
    print(f"    p90 = {p90_actual:.0f}")
    print(f"    Top-5 actual scores: {[f'{s:.0f}' for s in top5_actual]}")
    if pool_winner_score is not None:
        print(f"    Pool winner score = {pool_winner_score:.0f}")
        print(f"    Brackets that would beat pool winner = {beats_pool}")
    if high_score_brackets is not None:
        print(f"\n  2025 coverage deep-dive:")
        if high_score_brackets["n"] == 0:
            print(f"    {high_score_brackets['note']}")
            print(f"    Top-5 bracket upset counts: {high_score_brackets['top5_upset_counts']}")
        else:
            print(f"    {high_score_brackets['n']} brackets scored >= {pool_winner_score}")
            print(f"    Their scores: {high_score_brackets['scores']}")
            print(f"    Their upset counts: {high_score_brackets['upset_counts']}")

    result = {
        "year": year,
        "n_brackets": N_BRACKETS,
        "mean_pairwise_r": mean_r,
        "median_pairwise_r": median_r,
        "p10_pairwise_r": p10_r,
        "p90_pairwise_r": p90_r,
        "gate_corr_pass": mean_r < 0.60,
        "upset_counts": {
            "mean": float(upset_counts.mean()),
            "min": int(upset_counts.min()),
            "max": int(upset_counts.max()),
            "p90_threshold": float(p90_upset),
            "n_top_decile": n_top_decile,
            "gate_diversity_pass": n_top_decile >= 1,
        },
        "actual_scores": {
            "max": max_actual,
            "mean": mean_actual,
            "min": min_actual,
            "p90": p90_actual,
            "top5": top5_actual,
        },
        "g3_pass": mean_r < 0.60 and n_top_decile >= 1,
    }
    if pool_winner_score is not None:
        result["pool_winner_score"] = pool_winner_score
        result["brackets_beat_pool"] = beats_pool
    if high_score_brackets is not None:
        result["coverage_2025"] = high_score_brackets

    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    print("G3 Portfolio Diversity Diagnostic — §2 O25")
    print(f"Generating {N_BRACKETS} torvik brackets × {N_TOURN} MC tournaments per year\n")

    rng = np.random.default_rng(42)
    results = {}

    for year in YEARS:
        r = _run_year(year, rng)
        if r is not None:
            results[year] = r

    # --- Overall gate verdict ---
    print(f"\n{'=' * 60}")
    print("G3 SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'Year':<6}  {'MeanR':<8}  {'CorrGate':<10}  {'TopDecile':<12}  {'DivGate':<10}  {'G3':<6}")
    all_pass = True
    for year, r in results.items():
        corr_gate = "PASS" if r["gate_corr_pass"] else "FAIL"
        div_gate = "PASS" if r["upset_counts"]["gate_diversity_pass"] else "FAIL"
        g3 = "PASS" if r["g3_pass"] else "FAIL"
        if not r["g3_pass"]:
            all_pass = False
        print(
            f"{year:<6}  {r['mean_pairwise_r']:.4f}    {corr_gate:<10}  "
            f"{r['upset_counts']['n_top_decile']:<12}  {div_gate:<10}  {g3:<6}"
        )

    print(f"\nOverall G3: {'PASS' if all_pass else 'FAIL'}")

    # --- Save artifact ---
    artifact_dir = PROJECT_ROOT / "artifacts"
    artifact_dir.mkdir(exist_ok=True)
    artifact_path = artifact_dir / "o25_g3_diversity_2026-04-16.json"
    with open(artifact_path, "w") as f:
        json.dump(
            {
                "generated": "2026-04-16",
                "parameters": {
                    "n_brackets": N_BRACKETS,
                    "n_tourn": N_TOURN,
                    "years": YEARS,
                    "mode": "champ_first_tv (torvik round probs)",
                    "corr_gate": 0.60,
                },
                "results": {str(y): r for y, r in results.items()},
                "overall_g3_pass": all_pass,
            },
            f,
            indent=2,
        )
    print(f"\nArtifact saved to {artifact_path}")


if __name__ == "__main__":
    main()
