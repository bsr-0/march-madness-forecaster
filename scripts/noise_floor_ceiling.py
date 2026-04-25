"""Noise-floor ceiling — opponent-model side closure for council 64 action #3.

Council 64 (`council-report-2026-04-17-2307-next-11-months.md`, action 3):
  > Compute the irreducible noise floor: given perfect opponent knowledge
  > and n=3-4 pool-years, what is the best achievable BestRank? If the
  > ceiling gap between "current system" and "perfect opponent model" is
  > <3 rank positions, kill the opponent-calibration track.

Operationalization:
  - "Perfect opponent knowledge" => oracle picks the bracket from the 50-bracket
    portfolio that achieves the best score against the realized outcome.
  - "Current system" => MC ranker picks the bracket with lowest `mean_rank`
    (its predicted within-pool placement against simulated opponents).
  - Insert each pick into the actual pool-history field, score by ESPN
    team-identity points, compute its rank, take ranker_rank - best_rank.

For each year (2023-2026) x each cached mode (torvik, f4_first_tv, e8_first_tv):
  - best_score   = max(score_team_identity) across 50 portfolio brackets
  - ranker_score = score_team_identity of the bracket with min(mean_rank)
  - best_rank    = rank of best_score inserted into pool field
  - ranker_rank  = rank of ranker_score inserted into pool field
  - gap          = ranker_rank - best_rank

Council 71 (2026-04-19) already closed the *selection-side* noise floor
(D18, D19) — within-portfolio ranking is irreducible noise. This script is
the explicit opponent-side ceiling that the council 64 plan asked for.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PORTFOLIO_DIR = PROJECT_ROOT / "artifacts" / "backtest_brackets"
POOL_PATH = PROJECT_ROOT / "data" / "pool_history" / "pool_hist_results.json"
OUTPUT_PATH = PROJECT_ROOT / "artifacts" / f"noise_floor_ceiling_{datetime.now():%Y-%m-%d}.json"
YEARS = [2023, 2024, 2025, 2026]
KILL_THRESHOLD = 3


def rank_in_pool(my_score: float, pool_pts: list[int]) -> int:
    """1-based rank of my_score inserted into the pool field (ties broken in our favor)."""
    return 1 + sum(1 for p in pool_pts if p > my_score)


def main() -> None:
    pool_data = json.loads(POOL_PATH.read_text())
    pool_by_year = {
        int(year): [b["pts"] for b in payload["brackets"] if b["pts"] is not None]
        for year, payload in pool_data["years"].items()
    }

    rows: list[dict] = []
    for year in YEARS:
        portfolio = json.loads((PORTFOLIO_DIR / f"backtest_brackets_{year}.json").read_text())
        pool_pts = pool_by_year[year]
        for mode_block in portfolio["modes"]:
            brackets = mode_block["brackets"]
            scores = [b["score_team_identity"] for b in brackets]
            mean_ranks = [b["mean_rank"] for b in brackets]

            best_score = max(scores)
            best_idx = scores.index(best_score)
            ranker_idx = mean_ranks.index(min(mean_ranks))
            ranker_score = scores[ranker_idx]

            best_rank = rank_in_pool(best_score, pool_pts)
            ranker_rank = rank_in_pool(ranker_score, pool_pts)

            rows.append(
                {
                    "year": year,
                    "mode": mode_block["mode"],
                    "n_pool": len(pool_pts),
                    "n_portfolio": len(brackets),
                    "pool_winner_pts": max(pool_pts),
                    "best_score": best_score,
                    "best_bracket_idx": brackets[best_idx]["bracket_idx"],
                    "best_rank_in_pool": best_rank,
                    "ranker_score": ranker_score,
                    "ranker_bracket_idx": brackets[ranker_idx]["bracket_idx"],
                    "ranker_rank_in_pool": ranker_rank,
                    "gap": ranker_rank - best_rank,
                }
            )

    by_mode: dict[str, list[int]] = {}
    for r in rows:
        by_mode.setdefault(r["mode"], []).append(r["gap"])
    by_mode_summary = {
        mode: {
            "mean_gap": sum(gaps) / len(gaps),
            "max_gap": max(gaps),
            "n_years": len(gaps),
        }
        for mode, gaps in by_mode.items()
    }

    overall_gaps = [r["gap"] for r in rows]
    overall = {
        "mean_gap": sum(overall_gaps) / len(overall_gaps),
        "max_gap": max(overall_gaps),
        "n_obs": len(overall_gaps),
        "kill_threshold": KILL_THRESHOLD,
        "kill_opponent_calibration_track": (sum(overall_gaps) / len(overall_gaps)) < KILL_THRESHOLD,
    }

    payload = {
        "purpose": "Council 64 action #3 — opponent-model noise-floor ceiling",
        "kill_gate": (f"if mean(gap) < {KILL_THRESHOLD} rank positions, kill opponent-calibration track"),
        "definitions": {
            "best_score": "oracle pick from 50 — highest score_team_identity",
            "ranker_score": "MC ranker pick — bracket with lowest mean_rank",
            "gap": "ranker_rank_in_pool - best_rank_in_pool",
        },
        "years": YEARS,
        "rows": rows,
        "by_mode": by_mode_summary,
        "overall": overall,
        "generated": datetime.now().isoformat(timespec="seconds"),
    }
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2))

    # Human-readable summary
    print("\n=== Noise-Floor Ceiling — Council 64 Action #3 ===\n")
    header = (
        f"{'Year':<6}{'Mode':<16}{'PoolN':<7}"
        f"{'BestScore':<11}{'RankerScore':<13}"
        f"{'BestRank':<10}{'RankerRank':<12}{'Gap':<5}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['year']:<6}{r['mode']:<16}{r['n_pool']:<7}"
            f"{r['best_score']:<11.0f}{r['ranker_score']:<13.0f}"
            f"{r['best_rank_in_pool']:<10}{r['ranker_rank_in_pool']:<12}"
            f"{r['gap']:<5}"
        )
    print(f"\nBy mode (mean gap across {len(YEARS)} years):")
    for mode, agg in by_mode_summary.items():
        print(f"  {mode:<16} mean_gap={agg['mean_gap']:.2f}  max_gap={agg['max_gap']}  n={agg['n_years']}")
    print(f"\nOverall mean_gap = {overall['mean_gap']:.2f}  (kill threshold: < {KILL_THRESHOLD})")
    verdict = (
        "KILL opponent-calibration track"
        if overall["kill_opponent_calibration_track"]
        else "KEEP opponent-calibration track"
    )
    print(f"Verdict: {verdict}")
    print(f"\nArtifact: {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
