"""G1 spread diagnostic + 2026 post-mortem for §2 O25.

REAL G1 — replaces the fabricated 2026-04-16 narrative in
COUNCIL_LESSONS row 229.

The original O25 row claimed: "Ran 50 torvik brackets × 200 repeats ×
pool-history opponents for 2023–2026. Ranker correctly identifies the
winning bracket as #1 in all 4 years. P(rank=1) spread between #1 and
#11: 2023=0.835, 2024=0.315, 2025=0.045, 2026=0.310 — all > 3%,
mechanically triggering G2." Investigation 2026-04-16 (this branch)
found:
  - No script for this run exists in git history.
  - No artifact for this run exists in git history.
  - The 0.835 / 0.315 / 0.045 / 0.310 numbers appear nowhere else
    in the repo and are inconsistent with the pre-existing
    rank_correlation_diagnostic.json (which has real p1 values in
    [0, 0.13] range and #1-to-#11 spreads of 0.028-0.092).

This script runs G1 properly so the record is honest.

Spec (per §2 row 229)
---------------------
For each of the 50 brackets in {2023, 2024, 2025, 2026}, extract the
P(rank=1) assigned by the current ranker and the actual rank outcome;
compute the spread between #1 and #11; check whether #11 had a
structurally distinct upset profile.

Gate (pre-committed per the council spec):
  spread < 1%  → selection optimization is noise trading; pivot to G3
  spread > 3%  → systematic error; proceed to G2
  1% ≤ spread ≤ 3%  → ambiguous (council spec is silent; report both)

Three candidate definitions of "the current ranker" exist in the repo:
  (a) p_rank1_within  — within-portfolio: P(candidate is rank-1 of 50)
  (b) p_rank1_actual  — vs actual pool opponents: P(cand > max(real opp))
  (c) p_rank1_iid     — vs IID opponents from pool marginals
We report all three since the council spec doesn't disambiguate.

Setup matches G2/G3 (champ_first_tv torvik, rng_seed=42, n_brackets=50,
n_tourn=5000, noise_std=0.16) so direct comparison is possible.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts._common import load_tournament_results  # noqa: E402
from scripts.o25_g3_diversity_diagnostic import (  # noqa: E402
    build_actual_outcome,
    build_first_round_matchups,
    build_seed_probabilities,
    build_torvik_round_probabilities,
    count_r64_upsets,
    derive_f4_region_pairing,
    load_seeds_and_regions,
    resolve_first_four,
    sample_champ_first_brackets,
    _load_barthag,
)
from src.simulation.pool_competition import (  # noqa: E402
    build_scoring_vector,
    score_brackets_against_outcome,
    simulate_tournament_outcomes,
)
from src.simulation.pool_history_opponent_model import (  # noqa: E402
    build_pool_pick_distribution,
    load_pool_bracket_vectors,
    load_pool_brackets,
)

YEARS = [2023, 2024, 2025, 2026]
N_BRACKETS = 50
N_TOURN = 5000
NOISE_STD = 0.16
ESPN_SCORING = {"R64": 10, "R32": 20, "S16": 40, "E8": 80, "F4": 160, "CHAMP": 320}
POOL_HIST_PATH = PROJECT_ROOT / "pool_hist_results.json"
RNG_SEED = 42

GATE_NOISE_THRESHOLD = 0.01  # spread < 1% → noise trading (pivot to G3)
GATE_FIXABLE_THRESHOLD = 0.03  # spread > 3% → systematic error (proceed to G2)


def _spread(values: np.ndarray, top: int = 1, bottom: int = 11) -> float:
    s = sorted(values, reverse=True)
    if len(s) < bottom:
        return float("nan")
    return float(s[top - 1] - s[bottom - 1])


def _rank_of(idx: int, scores: np.ndarray) -> int:
    """1-indexed rank of `idx` when `scores` is sorted descending."""
    order = np.argsort(-scores)
    return int(np.where(order == idx)[0][0]) + 1


def _gate_verdict(spread: float) -> str:
    if spread < GATE_NOISE_THRESHOLD:
        return "NOISE_TRADING (pivot to G3)"
    if spread > GATE_FIXABLE_THRESHOLD:
        return "FIXABLE_SYSTEMATIC (proceed to G2)"
    return "AMBIGUOUS (1%-3%)"


def _run_year(year: int) -> dict | None:
    print(f"\n{'=' * 60}")
    print(f"Year {year}")
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
    scoring_vector = build_scoring_vector(ESPN_SCORING)

    rng = np.random.default_rng(RNG_SEED)
    candidates = sample_champ_first_brackets(first_round, round_probs, N_BRACKETS, rng)

    # --- Three opponent setups for the three ranker definitions ---
    opp_actual, group_size = load_pool_bracket_vectors(POOL_HIST_PATH, year, first_round, seeds)
    n_opp = opp_actual.shape[0]

    pool_entries, _ = load_pool_brackets(POOL_HIST_PATH, year)
    pool_marg = build_pool_pick_distribution(pool_entries, seeds, floor_strategy="laplace", laplace_alpha=0.5)
    iid_rng = np.random.default_rng(RNG_SEED + 1)
    opp_iid = sample_champ_first_brackets(first_round, pool_marg, n_opp, iid_rng)

    outcomes, _ = simulate_tournament_outcomes(N_TOURN, first_round, seed_pw, seeds, NOISE_STD, rng)

    s_cand = np.stack([score_brackets_against_outcome(candidates, outcomes[t], scoring_vector) for t in range(N_TOURN)])
    s_opp_actual = np.stack(
        [score_brackets_against_outcome(opp_actual, outcomes[t], scoring_vector) for t in range(N_TOURN)]
    )
    s_opp_iid = np.stack([score_brackets_against_outcome(opp_iid, outcomes[t], scoring_vector) for t in range(N_TOURN)])

    max_actual = s_opp_actual.max(axis=1)
    max_iid = s_opp_iid.max(axis=1)

    p_rank1_within = np.array([(s_cand.argmax(axis=1) == i).mean() for i in range(N_BRACKETS)])
    p_rank1_actual = (s_cand > max_actual[:, None]).mean(axis=0)
    p_rank1_iid = (s_cand > max_iid[:, None]).mean(axis=0)

    # --- Actual tournament outcome ---
    actual_outcome = build_actual_outcome(first_round, games)
    actual_scores = score_brackets_against_outcome(candidates, actual_outcome, scoring_vector)
    actual_best_idx = int(np.argmax(actual_scores))
    upset_counts = count_r64_upsets(candidates)
    actual_best_upsets = int(upset_counts[actual_best_idx])
    median_upsets = float(np.median(upset_counts))
    mean_upsets = float(upset_counts.mean())

    # --- Spreads + per-ranker placement of actual-best ---
    spread_within = _spread(p_rank1_within)
    spread_actual = _spread(p_rank1_actual)
    spread_iid = _spread(p_rank1_iid)

    rank_within = _rank_of(actual_best_idx, p_rank1_within)
    rank_actual = _rank_of(actual_best_idx, p_rank1_actual)
    rank_iid = _rank_of(actual_best_idx, p_rank1_iid)

    print(f"  n_candidates={N_BRACKETS}  n_opp_actual={n_opp} (groupSize={group_size})")
    print()
    print(
        f"  Ranker (a) within-portfolio P(rank=1)    : spread #1↔#11 = {spread_within:.4f}  "
        f"verdict = {_gate_verdict(spread_within)}"
    )
    print(
        f"  Ranker (b) vs actual pool opponents      : spread #1↔#11 = {spread_actual:.4f}  "
        f"verdict = {_gate_verdict(spread_actual)}"
    )
    print(
        f"  Ranker (c) vs IID from pool marginals    : spread #1↔#11 = {spread_iid:.4f}  "
        f"verdict = {_gate_verdict(spread_iid)}"
    )
    print()
    print(
        f"  Actual-best bracket (within these 50):  idx={actual_best_idx}, score={actual_scores[actual_best_idx]:.0f}"
    )
    print(f"    rank under (a) = #{rank_within} of {N_BRACKETS}")
    print(f"    rank under (b) = #{rank_actual} of {N_BRACKETS}")
    print(f"    rank under (c) = #{rank_iid} of {N_BRACKETS}")
    print(
        f"  Upset profile of actual-best: R64 upset count = {actual_best_upsets}  "
        f"(portfolio median={median_upsets:.1f}, mean={mean_upsets:.2f})"
    )

    return {
        "year": year,
        "n_candidates": N_BRACKETS,
        "n_opp_actual": n_opp,
        "group_size": group_size,
        "spread_within_portfolio": spread_within,
        "spread_vs_actual_opp": spread_actual,
        "spread_vs_iid_opp": spread_iid,
        "verdict_within_portfolio": _gate_verdict(spread_within),
        "verdict_vs_actual_opp": _gate_verdict(spread_actual),
        "verdict_vs_iid_opp": _gate_verdict(spread_iid),
        "actual_best_idx": actual_best_idx,
        "actual_best_score": float(actual_scores[actual_best_idx]),
        "actual_best_rank_within": rank_within,
        "actual_best_rank_vs_actual_opp": rank_actual,
        "actual_best_rank_vs_iid_opp": rank_iid,
        "actual_best_r64_upsets": actual_best_upsets,
        "portfolio_median_upsets": median_upsets,
        "portfolio_mean_upsets": mean_upsets,
        "p_rank1_within_top11_descending": sorted(p_rank1_within.tolist(), reverse=True)[:11],
        "p_rank1_vs_actual_top11_descending": sorted(p_rank1_actual.tolist(), reverse=True)[:11],
        "p_rank1_vs_iid_top11_descending": sorted(p_rank1_iid.tolist(), reverse=True)[:11],
    }


def _crossref_existing_data() -> dict:
    """Compute spreads from the pre-existing rank_correlation_diagnostic.json
    so the G1 record cross-references the only other extant P(rank=1) data
    in the repo."""
    path = PROJECT_ROOT / "artifacts" / "rank_correlation_diagnostic.json"
    if not path.exists():
        return {"available": False, "reason": "rank_correlation_diagnostic.json not found"}
    with open(path) as f:
        d = json.load(f)
    out = {"available": True, "source": str(path.relative_to(PROJECT_ROOT)), "per_year": {}}
    for r in d.get("results", []):
        y = r.get("year")
        if y not in YEARS:
            continue
        details = r.get("details", [])
        p1s = sorted([b["p1"] for b in details if "p1" in b], reverse=True)
        if len(p1s) < 11:
            spread = float("nan")
        else:
            spread = p1s[0] - p1s[10]
        out["per_year"][str(y)] = {
            "n": len(details),
            "p1_max": (p1s[0] if p1s else None),
            "p1_at_rank_11": (p1s[10] if len(p1s) >= 11 else None),
            "spread_1_to_11": spread,
            "verdict": _gate_verdict(spread) if not np.isnan(spread) else "INSUFFICIENT_DATA",
        }
    return out


def main() -> int:
    print("G1 Spread Diagnostic — §2 O25 (REAL run)")
    print(f"  n_candidates={N_BRACKETS}, n_tourn={N_TOURN}, rng_seed={RNG_SEED}")
    print(f"  Pool corpus: {POOL_HIST_PATH.relative_to(PROJECT_ROOT)}")
    print()
    print("Council G1 spread thresholds:")
    print(f"  spread < {GATE_NOISE_THRESHOLD} → NOISE_TRADING (pivot to G3)")
    print(f"  spread > {GATE_FIXABLE_THRESHOLD} → FIXABLE_SYSTEMATIC (proceed to G2)")
    print(f"  in between → AMBIGUOUS")

    results: dict[int, dict] = {}
    for year in YEARS:
        r = _run_year(year)
        if r is not None:
            results[year] = r

    crossref = _crossref_existing_data()

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print("G1 SUMMARY (this run)")
    print(f"{'=' * 60}")
    print(f"{'Year':<6}  {'within':>10}  {'vs_actual':>10}  {'vs_IID':>10}  {'best_rank_within':>18}")
    for year, r in sorted(results.items()):
        print(
            f"{year:<6}  {r['spread_within_portfolio']:>10.4f}  "
            f"{r['spread_vs_actual_opp']:>10.4f}  {r['spread_vs_iid_opp']:>10.4f}  "
            f"{r['actual_best_rank_within']:>18}"
        )

    print()
    print("Cross-reference: pre-existing rank_correlation_diagnostic.json")
    if crossref.get("available"):
        for ystr, d in crossref["per_year"].items():
            print(
                f"  {ystr}: n={d['n']}  p1_max={d['p1_max']}  spread_1_to_11={d['spread_1_to_11']:.4f}  "
                f"verdict={d['verdict']}"
            )
    else:
        print(f"  not available: {crossref.get('reason')}")

    print()
    print("COUNCIL_LESSONS row 229 G1 numbers (FOR COMPARISON; provenance: none)")
    fabricated = {2023: 0.835, 2024: 0.315, 2025: 0.045, 2026: 0.310}
    for year, claim in fabricated.items():
        if year in results:
            real = results[year]["spread_within_portfolio"]
            ratio = claim / real if real > 0 else float("inf")
            print(f"  {year}: claimed={claim:.4f}  vs  real(within)={real:.4f}  ratio={ratio:.1f}x")

    artifact_dir = PROJECT_ROOT / "artifacts"
    artifact_dir.mkdir(exist_ok=True)
    artifact_path = artifact_dir / "o25_g1_real_2026-04-16.json"
    payload = {
        "generated": "2026-04-16",
        "purpose": "Real G1 measurement; replaces fabricated COUNCIL_LESSONS row 229 narrative.",
        "parameters": {
            "n_candidates": N_BRACKETS,
            "n_tourn": N_TOURN,
            "noise_std": NOISE_STD,
            "rng_seed": RNG_SEED,
            "years": YEARS,
            "pool_corpus": str(POOL_HIST_PATH.relative_to(PROJECT_ROOT)),
            "candidate_mode": "champ_first_tv (torvik round probs)",
            "gate_noise_threshold": GATE_NOISE_THRESHOLD,
            "gate_fixable_threshold": GATE_FIXABLE_THRESHOLD,
        },
        "results": {str(y): r for y, r in results.items()},
        "crossref_existing_data": crossref,
        "council_lessons_row_229_g1_numbers": {
            "values": fabricated,
            "note": (
                "These spread values appear in COUNCIL_LESSONS.md row 229 and nowhere else "
                "in the repo. No corresponding script or artifact exists in git history. "
                "Investigation 2026-04-16 concluded these numbers are unsubstantiated."
            ),
        },
    }
    with open(artifact_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nArtifact: {artifact_path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
