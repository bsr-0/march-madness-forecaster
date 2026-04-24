"""Testing Budget primitives for the tiered experiment loop.

Pure logic, no backtest imports. This is the code path that enforces
STRATEGY_CATALOG.md § "Testing Budget: Comprehensive but Time-Bounded":
T1 screens every candidate cheaply and kills obvious losers, T2 ranks
survivors at mid fidelity, T3 validates the top 5 at full rigor with
the significance gate.

The catalog contract and this module must stay in lockstep — see
``tests/test_experiment_budget.py`` for the lock.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

T1_RECENT_YEARS = 8


@dataclass(frozen=True)
class TierConfig:
    """Parameters for a single evaluation tier.

    Attributes:
        name: "T1", "T2", or "T3".
        n_repeats: Repeats per year for the backtest.
        n_model: Model brackets per strategy per year.
        year_mode: "recent8" (most recent N years) or "full" (all years).
        promote_top_n: How many strategies advance to the next tier
            (None = final tier, no promotion).
        kill_min_wins_vs_baseline: Minimum years where P(1st) > baseline
            to survive (None = no wins-based kill).
        kill_on_zero_year: If True, kill any strategy with P(1st)=0 in any year.
        wall_time_target_hours: Informational wall-time target used to
            flag overruns in the run summary.
    """

    name: str
    n_repeats: int
    n_model: int
    year_mode: str
    promote_top_n: Optional[int]
    kill_min_wins_vs_baseline: Optional[int]
    kill_on_zero_year: bool
    wall_time_target_hours: float


TIER_CONFIGS: Dict[str, TierConfig] = {
    "T1": TierConfig(
        name="T1",
        n_repeats=25,
        n_model=50,
        year_mode="recent8",
        promote_top_n=10,
        kill_min_wins_vs_baseline=3,
        kill_on_zero_year=True,
        wall_time_target_hours=1.0,
    ),
    "T2": TierConfig(
        name="T2",
        n_repeats=50,
        n_model=50,
        year_mode="full",
        promote_top_n=5,
        kill_min_wins_vs_baseline=5,
        kill_on_zero_year=False,
        wall_time_target_hours=3.0,
    ),
    "T3": TierConfig(
        name="T3",
        n_repeats=100,
        n_model=50,
        year_mode="full",
        promote_top_n=None,
        kill_min_wins_vs_baseline=8,
        kill_on_zero_year=False,
        wall_time_target_hours=6.0,
    ),
}


def select_tier_years(
    all_years: Iterable[int],
    tier_cfg: TierConfig,
    n_recent: int = T1_RECENT_YEARS,
) -> List[int]:
    """Return the years a tier should run on."""
    sorted_years = sorted(all_years)
    if tier_cfg.year_mode == "recent8":
        return sorted_years[-n_recent:]
    if tier_cfg.year_mode == "full":
        return sorted_years
    raise ValueError(f"Unknown year_mode: {tier_cfg.year_mode!r}")


def aggregate_strategy_stats(
    results: Sequence[dict],
    baseline_key: str,
) -> Dict[str, dict]:
    """Pivot flat backtest results into per-strategy stats.

    For each strategy records mean P(1st), wins vs baseline per year,
    years covered, and zero-year count. Downstream kill/promote
    functions consume this dict exclusively.
    """
    by_strategy: Dict[str, List[dict]] = {}
    for r in results:
        by_strategy.setdefault(r["mode"], []).append(r)

    baseline_by_year = {r["year"]: r["p_first"] for r in results if r["mode"] == baseline_key}

    stats: Dict[str, dict] = {}
    for s, rows in by_strategy.items():
        p1_by_year = {r["year"]: r["p_first"] for r in rows}
        wins = sum(1 for y, p in p1_by_year.items() if y in baseline_by_year and p > baseline_by_year[y])
        zero_years = sum(1 for p in p1_by_year.values() if p == 0.0)
        stats[s] = {
            "mean_p_first": float(np.mean(list(p1_by_year.values()))) if p1_by_year else 0.0,
            "wins_vs_baseline": wins,
            "n_years": len(p1_by_year),
            "zero_years": zero_years,
            "p1_by_year": p1_by_year,
        }
    return stats


def apply_kill_threshold(
    stats: Dict[str, dict],
    tier_cfg: TierConfig,
    baseline_key: str,
) -> Tuple[Dict[str, dict], List[Tuple[str, str]]]:
    """Split strategies into survivors and killed by the tier's rules.

    The baseline is always a survivor — it's needed as the comparator
    for later tiers.
    """
    survivors: Dict[str, dict] = {}
    killed: List[Tuple[str, str]] = []
    for s, st in stats.items():
        if s == baseline_key:
            survivors[s] = st
            continue
        reasons = []
        if tier_cfg.kill_on_zero_year and st["zero_years"] > 0:
            reasons.append(f"P(1st)=0 in {st['zero_years']} year(s)")
        if (
            tier_cfg.kill_min_wins_vs_baseline is not None
            and st["wins_vs_baseline"] < tier_cfg.kill_min_wins_vs_baseline
        ):
            reasons.append(
                f"{st['wins_vs_baseline']}/{st['n_years']} wins vs baseline "
                f"(need >= {tier_cfg.kill_min_wins_vs_baseline})"
            )
        if reasons:
            killed.append((s, "; ".join(reasons)))
        else:
            survivors[s] = st
    return survivors, killed


def promote_top_n(
    survivors: Dict[str, dict],
    baseline_key: str,
    n: int,
) -> List[str]:
    """Rank surviving strategies by mean P(1st); return top N plus baseline."""
    ranked = sorted(
        ((s, st) for s, st in survivors.items() if s != baseline_key),
        key=lambda kv: -kv[1]["mean_p_first"],
    )
    promoted = [s for s, _ in ranked[:n]]
    if baseline_key not in promoted:
        promoted.append(baseline_key)
    return promoted


def best_baseline_delta(
    stats: Dict[str, dict],
    baseline_key: str,
) -> Optional[float]:
    """Largest mean-P(1st) improvement any non-baseline strategy has over baseline."""
    baseline = stats.get(baseline_key)
    if baseline is None:
        return None
    deltas = [st["mean_p_first"] - baseline["mean_p_first"] for s, st in stats.items() if s != baseline_key]
    return max(deltas) if deltas else None


def stats_for_artifact(stats: Dict[str, dict]) -> Dict[str, dict]:
    """Trim per-year dicts so JSON artifacts stay readable."""
    return {
        s: {
            "mean_p_first": st["mean_p_first"],
            "wins_vs_baseline": st["wins_vs_baseline"],
            "n_years": st["n_years"],
            "zero_years": st["zero_years"],
        }
        for s, st in stats.items()
    }
