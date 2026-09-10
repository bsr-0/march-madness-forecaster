"""Real-outcome, real-opponent placement of meta_region_poolaware, 2023-2026.

Every other number this project reports for P(1st) is measured against a
*simulated* tournament and *simulated* opponents (see AUDIT_INDEPENDENT_EVALUATOR_2027.md
finding C1). This script computes the one honest complement: where the
production bracket the backtest actually selected would have placed against
the REAL tournament outcome and the REAL brackets of the 30-person pool, for
the four seasons real pool data exists (2023-2026).

It does not reimplement candidate construction or selection — it reads the
picks written by `mc_pool_backtest.py --save-brackets`, which already scores
each saved bracket with `score_brackets_team_identity` against
`actual_winners_by_round`, i.e. the real result. This script's only job is to
rank that real score against the real pool's real `pts`.

Usage:
    python -m scripts.mc_pool_backtest --team-identity --opponent pool \\
        --years 2023 2024 2025 2026 --modes meta_region_poolaware seed \\
        --save-brackets
    python -m scripts.real_pool_placement

n=4. This is not a statistic; it is four individual outcomes, reported as
such (no mean is asserted to be more than descriptive, no CI is computed).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.simulation.pool_history_opponent_model import load_pool_brackets

ROOT = Path(__file__).resolve().parent.parent
YEARS = [2023, 2024, 2025, 2026]
BRACKET_DIR = ROOT / "artifacts" / "backtest_brackets"
POOL_HIST_PATH = ROOT / "pool_hist_results.json"


def _load_saved_mode(year: int, mode: str) -> Optional[Dict[str, Any]]:
    path = BRACKET_DIR / f"backtest_brackets_{year}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    for m in data.get("modes", []):
        if m["mode"] == mode:
            return m
    return None


def _real_pool_scores(year: int) -> List[float]:
    brackets, _group_size = load_pool_brackets(str(POOL_HIST_PATH), year)
    return sorted((float(b["pts"]) for b in brackets), reverse=True)


def _rank_against_pool(model_score: float, pool_scores: List[float]) -> Dict[str, Any]:
    n_pool = len(pool_scores)
    better = sum(1 for s in pool_scores if s > model_score)
    tied = sum(1 for s in pool_scores if s == model_score)
    # Model bracket is not itself one of the n_pool real entries; rank it
    # as if inserted into the field (ties broken conservatively, i.e. the
    # model is placed last among ties rather than first).
    rank = better + tied + 1
    return {
        "rank": rank,
        "n_pool_entries": n_pool,
        "percentile": 100.0 * (n_pool - better) / n_pool if n_pool else None,
    }


def placement_for_year(year: int, mode: str = "meta_region_poolaware") -> Optional[Dict[str, Any]]:
    """Real-outcome score and real-pool rank for one year's bracket(s).

    ``meta_region_poolaware`` writes exactly one bracket per year (it is
    deterministic) — that single bracket is the production pick, and is
    reported as-is.

    Baseline modes such as ``seed`` are stochastic: the file holds up to
    ``n_model`` brackets, already sorted descending by their real-outcome
    score. Taking ``brackets[0]`` would silently report the best-of-N
    bracket chosen WITH HINDSIGHT of the real result — nobody submits 50
    brackets and keeps only the winner. For any mode with more than one
    saved bracket this function instead reports the MEAN real score across
    all of them: an honest answer to "if you had to submit one bracket
    from this strategy, not knowing which would score best."
    """
    saved = _load_saved_mode(year, mode)
    if saved is None or not saved.get("brackets"):
        return None
    brackets = saved["brackets"]
    pool_scores = _real_pool_scores(year)
    is_single_pick = len(brackets) == 1

    if is_single_pick:
        model_score = brackets[0]["score_team_identity"]
        champion = brackets[0].get("champion")
    else:
        model_score = sum(b["score_team_identity"] for b in brackets) / len(brackets)
        champions = {b.get("champion") for b in brackets}
        champion = next(iter(champions)) if len(champions) == 1 else f"{len(champions)} different"

    placement = _rank_against_pool(model_score, pool_scores)
    return {
        "year": year,
        "mode": mode,
        "is_single_pick": is_single_pick,
        "n_brackets": len(brackets),
        "model_score": model_score,
        "model_champion": champion,
        "pool_top_score": pool_scores[0] if pool_scores else None,
        "pool_median_score": pool_scores[len(pool_scores) // 2] if pool_scores else None,
        **placement,
    }


def main() -> None:
    print("=" * 100)
    print("REAL-OUTCOME, REAL-OPPONENT PLACEMENT — meta_region_poolaware vs seed, 2023-2026")
    print("=" * 100)
    print(
        "Every row below scores the production bracket against the REAL tournament result\n"
        "and ranks it against the REAL pool's REAL scores (pool_hist_results.json). This is\n"
        "the complement to every simulated-tournament P(1st) figure reported elsewhere.\n"
        "n=4 seasons. No mean, CI, or significance test is computed or implied below — four\n"
        "outcomes is not a distribution.\n"
    )

    header = f"  {'Year':<6}{'Mode':<24}{'Score':>7}{'Rank':>7}{'/Pool':>7}{'%ile':>7}  Champion picked"
    print(header)
    print("  " + "-" * (len(header) - 2))

    rows = []
    for year in YEARS:
        for mode in ("meta_region_poolaware", "seed"):
            r = placement_for_year(year, mode)
            if r is None:
                print(f"  {year:<6}{mode:<24}  (no saved bracket — run with --save-brackets first)")
                continue
            rows.append(r)
            pct = f"{r['percentile']:.0f}%" if r["percentile"] is not None else "?"
            score_str = f"{r['model_score']:.0f}"
            note = "" if r["is_single_pick"] else f" (mean of {r['n_brackets']})"
            print(
                f"  {r['year']:<6}{r['mode']:<24}{score_str:>7}"
                f"{r['rank']:>7}{r['n_pool_entries']:>7}{pct:>7}  {r['model_champion']}{note}"
            )
    print(
        "\n  seed's score/rank above is the MEAN across its saved brackets (no single seed-mode\n"
        "  pick exists), not the best — a best-of-N read would be hindsight-biased. Only\n"
        "  meta_region_poolaware has a single production pick to report as-is."
    )

    print()
    poolaware_rows = [r for r in rows if r["mode"] == "meta_region_poolaware"]
    wins = sum(1 for r in poolaware_rows if r["rank"] == 1)
    top3 = sum(1 for r in poolaware_rows if r["rank"] <= 3)
    print(
        f"meta_region_poolaware, real outcomes, real opponents, n={len(poolaware_rows)}: "
        f"{wins}/{len(poolaware_rows)} finished 1st, {top3}/{len(poolaware_rows)} finished top 3."
    )
    print(
        "This is a small-sample descriptive fact, not an estimate of a rate: with n=4, one\n"
        "additional or one fewer win changes the observed fraction by 25 points. It does not\n"
        "confirm or refute the simulated P(1st) figure; it is the only number in this\n"
        "project measured against reality rather than a model of reality."
    )


if __name__ == "__main__":
    main()
