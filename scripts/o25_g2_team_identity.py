"""G2 empirical-supremum diagnostic under TEAM-IDENTITY (ESPN) scoring — §2 O25.

Companion to `scripts/o25_g2_empirical_supremum.py`, which runs the same
test under the pipeline's default shape-encoded scoring. The pool
actually pays out by team-identity scoring (what ESPN displays), so the
shape-encoded result is scoring-methodology-suspect on its own. This
script closes that loophole by re-running the G2 primary gate under
real team-identity scoring and comparing.

Algorithm mirrors the shape version; the only change is how brackets
are scored against outcomes:

  shape:         (bracket_vec == outcome_vec) @ scoring_vector
                 — credits positional bool match, agnostic to team
                 identity; over-credits when bracket and outcome
                 agree on "team1 won" for different team1s.
  team-identity: for each round r with pts p, score += p * |picks_r ∩
                 winners_r|
                 — credits only when the picked team is in the round's
                 actual winner set.

Under team-identity scoring, my candidates DO contain a pool-winning
bracket in all 4 years (coverage check run 2026-04-16; cand max beats
stored pool max in all of 2023/2024/2025/2026). The G2-A primary gate
therefore tests the right question: ranking ability on brackets that
are at least capable of winning.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts._common import load_tournament_results  # noqa: E402
from scripts.o25_g3_diversity_diagnostic import (  # noqa: E402
    _load_barthag,
    build_first_round_matchups,
    build_seed_probabilities,
    build_torvik_round_probabilities,
    derive_f4_region_pairing,
    load_seeds_and_regions,
    resolve_first_four,
    sample_champ_first_brackets,
)
from src.simulation.pool_competition import ROUND_NAMES, simulate_tournament_outcomes  # noqa: E402
from src.simulation.pool_history_opponent_model import load_pool_bracket_vectors  # noqa: E402

YEARS = [2023, 2024, 2025, 2026]
N_BRACKETS = 50
N_TOURN = 5000
NOISE_STD = 0.16
RNG_SEED = 42
POOL_HIST_PATH = PROJECT_ROOT / "pool_hist_results.json"

ROUND_PTS = {"R64": 10, "R32": 20, "S16": 40, "E8": 80, "F4": 160, "CHAMP": 320}
# Map actual-tournament round names (ESPN uses NCG for the championship
# game) to our internal round names.
_ACTUAL_NCG_ALIASES = {"NCG": "CHAMP"}

GATE_A_SPEARMAN_THRESHOLD = 0.50


def _picks_by_round(vec: np.ndarray, first_round: list) -> dict[str, set[str]]:
    """Walk a (63,) bracket vector and return per-round advancing-team sets."""
    by_round: dict[str, set[str]] = {}
    current = list(first_round)
    gi = 0
    for rnd in ROUND_NAMES:
        nxt = []
        for g in range(0, len(current), 2):
            t1, t2 = current[g], current[g + 1]
            winner = t1 if vec[gi] else t2
            nxt.append(winner)
            gi += 1
        by_round[rnd] = set(nxt)
        current = nxt
    return by_round


def _score_team_identity(picks_by_round: dict[str, set[str]], winners_by_round: dict[str, set[str]]) -> int:
    total = 0
    for rnd, pts in ROUND_PTS.items():
        total += pts * len(picks_by_round[rnd] & winners_by_round.get(rnd, set()))
    return total


def _actual_winners_by_round(games: list[dict]) -> dict[str, set[str]]:
    """Winners per round from raw tournament_results, cascade-consistent.

    Identical normalization to tests/test_pool_hist_ev_validation.py._build_winners
    (ensures First-Four resolution phantoms don't leak into later rounds)."""
    winners: dict[str, set[str]] = defaultdict(set)
    for g in games:
        w = g["team1_id"] if g["team1_won"] else g["team2_id"]
        winners[g["round_name"]].add(w)
    prev: set[str] | None = None
    for rnd in ["R64", "R32", "S16", "E8", "F4", "NCG"]:
        if rnd in winners and prev is not None:
            winners[rnd] &= prev
        prev = winners.get(rnd, set())
    # Relabel NCG → CHAMP so downstream matches ROUND_NAMES.
    return {
        "R64": winners["R64"],
        "R32": winners["R32"],
        "S16": winners["S16"],
        "E8": winners["E8"],
        "F4": winners["F4"],
        "CHAMP": winners.get("NCG", set()),
    }


def _precompute_picks_per_bracket(brackets: np.ndarray, first_round: list) -> list[dict[str, set[str]]]:
    return [_picks_by_round(brackets[b], first_round) for b in range(brackets.shape[0])]


def _run_year(year: int) -> dict | None:
    print(f"\n{'=' * 60}")
    print(f"Year {year} (team-identity scoring)")
    print(f"{'=' * 60}")

    games = load_tournament_results(year)
    if not games:
        print("  SKIP — no tournament results")
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
        print(f"  SKIP — first_round has {len(first_round)} (need 64)")
        return None

    barthag = _load_barthag(year, seeds)
    round_probs = build_torvik_round_probabilities(seeds, regions, barthag)
    seed_pw = build_seed_probabilities(seeds)

    rng = np.random.default_rng(RNG_SEED)
    candidates = sample_champ_first_brackets(first_round, round_probs, N_BRACKETS, rng)
    opp_matrix, group_size = load_pool_bracket_vectors(POOL_HIST_PATH, year, first_round, seeds)
    n_opp = opp_matrix.shape[0]
    print(f"  n_candidates={N_BRACKETS}  n_opp_actual={n_opp} (groupSize={group_size})")

    outcomes, _ = simulate_tournament_outcomes(N_TOURN, first_round, seed_pw, seeds, NOISE_STD, rng)

    cand_picks = _precompute_picks_per_bracket(candidates, first_round)
    opp_picks = _precompute_picks_per_bracket(opp_matrix, first_round)

    # Score every bracket against every simulated tournament.
    s_cand = np.zeros((N_TOURN, N_BRACKETS), dtype=np.int32)
    s_opp = np.zeros((N_TOURN, n_opp), dtype=np.int32)
    for t in range(N_TOURN):
        outcome_winners = _picks_by_round(outcomes[t], first_round)
        for b in range(N_BRACKETS):
            s_cand[t, b] = _score_team_identity(cand_picks[b], outcome_winners)
        for o in range(n_opp):
            s_opp[t, o] = _score_team_identity(opp_picks[o], outcome_winners)

    max_opp = s_opp.max(axis=1)
    p_sup = (s_cand > max_opp[:, None]).mean(axis=0)

    cand_winner_idx = s_cand.argmax(axis=1)
    p_rank1 = np.array([(cand_winner_idx == i).mean() for i in range(N_BRACKETS)])

    # Actual-outcome team-identity scores (ground truth for Spearman).
    actual_wr = _actual_winners_by_round(games)
    actual_scores = np.array([_score_team_identity(cand_picks[b], actual_wr) for b in range(N_BRACKETS)])
    actual_opp_scores = np.array([_score_team_identity(opp_picks[o], actual_wr) for o in range(n_opp)])

    rho_new = float(spearmanr(p_sup, actual_scores).statistic)
    rho_old = float(spearmanr(p_rank1, actual_scores).statistic)

    actual_order = np.argsort(-actual_scores)
    actual_best_idx = int(actual_order[0])
    new_order = np.argsort(-p_sup)
    old_order = np.argsort(-p_rank1)
    best_rank_new = int(np.where(new_order == actual_best_idx)[0][0]) + 1
    best_rank_old = int(np.where(old_order == actual_best_idx)[0][0]) + 1
    g2b_flipped = bool(best_rank_new <= 5 and best_rank_old > 5)

    # Coverage: does cand's actual-best beat the stored pool winner pts?
    # Stored pts is the authoritative team-identity score (what ESPN paid).
    from src.simulation.pool_history_opponent_model import load_pool_brackets

    pool_entries, _ = load_pool_brackets(POOL_HIST_PATH, year)
    stored_max = max(b["pts"] for b in pool_entries)
    cand_best_score = int(actual_scores[actual_best_idx])
    coverage_beats_stored = cand_best_score > stored_max

    print(f"  ρ(p_sup, actual_scores)   = {rho_new:+.4f}  (G2-A gate > {GATE_A_SPEARMAN_THRESHOLD})")
    print(f"  ρ(p_rank1, actual_scores) = {rho_old:+.4f}  (baseline)")
    print(f"  actual-#1 of 50 (team-identity score = {cand_best_score})")
    print(f"    rank under new ranker = #{best_rank_new}, under old = #{best_rank_old}, G2-B flip: {g2b_flipped}")
    print(
        f"  Coverage: cand_best({cand_best_score}) vs stored_pool_max({stored_max})  -> "
        f"{'BEATS' if coverage_beats_stored else 'LOSES'}"
    )

    return {
        "year": year,
        "n_candidates": N_BRACKETS,
        "n_opp_actual": n_opp,
        "group_size": group_size,
        "spearman_new_vs_actual": rho_new,
        "spearman_old_vs_actual": rho_old,
        "actual_best_idx": actual_best_idx,
        "actual_best_score": cand_best_score,
        "actual_best_new_rank": best_rank_new,
        "actual_best_old_rank": best_rank_old,
        "g2b_flipped_into_top5": g2b_flipped,
        "stored_pool_max_pts": int(stored_max),
        "coverage_beats_stored": coverage_beats_stored,
        "actual_scores_summary": {
            "max": float(actual_scores.max()),
            "min": float(actual_scores.min()),
            "mean": float(actual_scores.mean()),
        },
        "actual_opp_scores_summary": {
            "max": float(actual_opp_scores.max()),
            "min": float(actual_opp_scores.min()),
            "mean": float(actual_opp_scores.mean()),
        },
    }


def main() -> int:
    print("G2 Team-Identity Scoring Diagnostic — §2 O25 (companion to shape-encoded G2)")
    print(f"  n_candidates={N_BRACKETS}, n_tourn={N_TOURN}, rng_seed={RNG_SEED}")

    results: dict[int, dict] = {}
    for year in YEARS:
        r = _run_year(year)
        if r is not None:
            results[year] = r

    rho_new_vals = [r["spearman_new_vs_actual"] for r in results.values()]
    rho_old_vals = [r["spearman_old_vs_actual"] for r in results.values()]
    mean_new = float(np.mean(rho_new_vals))
    mean_old = float(np.mean(rho_old_vals))
    any_flip = any(r["g2b_flipped_into_top5"] for r in results.values())
    all_coverage = all(r["coverage_beats_stored"] for r in results.values())
    g2a_pass = mean_new > GATE_A_SPEARMAN_THRESHOLD

    print(f"\n{'=' * 60}")
    print("G2 TEAM-IDENTITY SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'Year':<6}  {'ρ_new':>8}  {'ρ_old':>8}  {'cand-beats-pool?':>18}")
    for y, r in sorted(results.items()):
        print(
            f"{y:<6}  {r['spearman_new_vs_actual']:+.4f}  {r['spearman_old_vs_actual']:+.4f}  "
            f"{('YES' if r['coverage_beats_stored'] else 'NO'):>18}"
        )
    print()
    print(
        f"  G2-A primary (team-identity): mean ρ(p_sup, actual) = {mean_new:+.4f}  "
        f"(gate > {GATE_A_SPEARMAN_THRESHOLD}): {'PASS' if g2a_pass else 'FAIL'}"
    )
    print(f"           baseline mean ρ(p_rank1, actual) = {mean_old:+.4f}")
    print(f"  G2-B (team-identity): actual-#1 flip into top-5 anywhere? {'YES' if any_flip else 'NO'}")
    print(f"  Coverage (all 4 yrs cand beats stored pool max): {'YES' if all_coverage else 'NO'}")

    artifact_dir = PROJECT_ROOT / "artifacts"
    artifact_dir.mkdir(exist_ok=True)
    artifact_path = artifact_dir / "o25_g2_team_identity_2026-04-16.json"
    payload = {
        "generated": "2026-04-16",
        "purpose": "Team-identity (real ESPN) companion to o25_g2_empirical_supremum_2026-04-16.json",
        "parameters": {
            "n_candidates": N_BRACKETS,
            "n_tourn": N_TOURN,
            "noise_std": NOISE_STD,
            "rng_seed": RNG_SEED,
            "years": YEARS,
            "pool_corpus": str(POOL_HIST_PATH.relative_to(PROJECT_ROOT)),
            "candidate_mode": "champ_first_tv (torvik round probs)",
            "scoring": "team_identity (per-round |picks ∩ winners| × pts)",
            "gate_a_threshold": GATE_A_SPEARMAN_THRESHOLD,
        },
        "results": {str(y): r for y, r in results.items()},
        "aggregate": {
            "mean_spearman_new_vs_actual": mean_new,
            "mean_spearman_old_vs_actual": mean_old,
            "g2a_pass": g2a_pass,
            "g2b_any_flip": any_flip,
            "coverage_all_years": all_coverage,
        },
    }
    with open(artifact_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nArtifact: {artifact_path.relative_to(PROJECT_ROOT)}")
    return 0 if g2a_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
