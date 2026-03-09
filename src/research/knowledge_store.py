"""Knowledge retention store for experiment insights and patterns.

Indexes experiment results by hypothesis, regime, and parameter space to
enable learning across runs. Answers questions like "what works for
upset-heavy years?" and "which parameters have we not yet explored?"

Implements Agent Directive V7 S14 (knowledge retention).

Usage:
    from src.research.knowledge_store import KnowledgeStore

    store = KnowledgeStore(registry)
    insights = store.insights_by_regime("upset_heavy")
    unexplored = store.unexplored_parameters()
    report = store.research_summary()
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class KnowledgeStore:
    """Searchable knowledge index over experiment history.

    Provides structured queries across the experiment registry to identify
    patterns, underexplored regions, and actionable insights.
    """

    def __init__(self, registry: Any = None) -> None:
        """Initialize with an ExperimentRegistry instance."""
        self._registry = registry

    def insights_by_regime(self, regime: str) -> Dict[str, Any]:
        """Get performance insights for a specific regime type.

        Args:
            regime: Regime name (e.g., 'upset_heavy', 'chalk').

        Returns:
            Dict with best config, mean Brier, and recommendations.
        """
        experiments = self._get_experiments()
        matching: List[Dict[str, Any]] = []

        for exp in experiments:
            regime_data = exp.regime_analysis or {}
            regimes = regime_data.get("regimes", {})
            if regime in regimes:
                matching.append(
                    {
                        "experiment_id": exp.experiment_id,
                        "brier": regimes[regime].get("mean_brier")
                        if isinstance(regimes[regime], dict)
                        else None,
                        "config_hash": exp.config_hash,
                        "model_family": exp.model_family,
                    }
                )

        if not matching:
            return {"regime": regime, "experiments": 0, "insights": "No data available."}

        briers = [m["brier"] for m in matching if m["brier"] is not None]
        best = min(matching, key=lambda m: m["brier"] or float("inf"))

        return {
            "regime": regime,
            "experiments": len(matching),
            "mean_brier": sum(briers) / len(briers) if briers else None,
            "best_experiment": best["experiment_id"],
            "best_brier": best["brier"],
            "best_config": best["config_hash"],
        }

    def insights_by_year(self) -> Dict[int, Dict[str, Any]]:
        """Get per-year performance insights across all experiments.

        Returns:
            Dict mapping year to {best_brier, worst_brier, mean_brier, n_experiments}.
        """
        experiments = self._get_experiments()
        year_data: Dict[int, List[float]] = defaultdict(list)

        for exp in experiments:
            for year, brier in (exp.loyo_year_briers or {}).items():
                year_data[year].append(brier)

        result: Dict[int, Dict[str, Any]] = {}
        for year in sorted(year_data.keys()):
            briers = year_data[year]
            result[year] = {
                "best_brier": min(briers),
                "worst_brier": max(briers),
                "mean_brier": sum(briers) / len(briers),
                "n_experiments": len(briers),
            }
        return result

    def unexplored_parameters(
        self,
        param_ranges: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Identify parameter regions not yet explored in experiments.

        Args:
            param_ranges: Expected parameter ranges. If None, uses defaults.

        Returns:
            List of unexplored parameter regions with suggestions.
        """
        if param_ranges is None:
            from src.research.experiment_scheduler import DEFAULT_PARAM_RANGES

            param_ranges = DEFAULT_PARAM_RANGES

        experiments = self._get_experiments()
        explored: Dict[str, List[Any]] = defaultdict(list)

        for exp in experiments:
            for param, val in (exp.hyperparameters or {}).items():
                if param in param_ranges:
                    explored[param].append(val)

        suggestions: List[Dict[str, Any]] = []
        for param, spec in param_ranges.items():
            vals = explored.get(param, [])
            if not vals:
                suggestions.append(
                    {
                        "parameter": param,
                        "status": "never_explored",
                        "range": [spec["min"], spec["max"]],
                        "suggestion": f"Try {param} across range [{spec['min']}, {spec['max']}]",
                    }
                )
            else:
                # Check if extremes are explored
                explored_min = min(vals)
                explored_max = max(vals)
                if explored_min > spec["min"] + spec["step"]:
                    suggestions.append(
                        {
                            "parameter": param,
                            "status": "low_end_unexplored",
                            "explored_min": explored_min,
                            "range_min": spec["min"],
                            "suggestion": f"Try lower {param}: [{spec['min']}, {explored_min}]",
                        }
                    )
                if explored_max < spec["max"] - spec["step"]:
                    suggestions.append(
                        {
                            "parameter": param,
                            "status": "high_end_unexplored",
                            "explored_max": explored_max,
                            "range_max": spec["max"],
                            "suggestion": f"Try higher {param}: [{explored_max}, {spec['max']}]",
                        }
                    )

        return suggestions

    def what_works(self, top_n: int = 5) -> Dict[str, Any]:
        """Identify patterns in the best-performing experiments.

        Returns:
            Summary of common traits among top experiments.
        """
        experiments = self._get_experiments()
        if not experiments:
            return {"status": "no_experiments"}

        # Sort by Brier (lower is better)
        sorted_exps = sorted(
            [e for e in experiments if e.loyo_mean_brier > 0],
            key=lambda e: e.loyo_mean_brier,
        )

        top = sorted_exps[:top_n]
        if not top:
            return {"status": "no_valid_experiments"}

        # Collect common traits
        model_families: Dict[str, int] = defaultdict(int)
        calibration_methods: Dict[str, int] = defaultdict(int)

        for exp in top:
            if exp.model_family:
                model_families[exp.model_family] += 1
            if exp.calibration_method:
                calibration_methods[exp.calibration_method] += 1

        return {
            "top_n": top_n,
            "brier_range": [top[0].loyo_mean_brier, top[-1].loyo_mean_brier],
            "common_model_families": dict(model_families),
            "common_calibration": dict(calibration_methods),
            "top_experiments": [
                {
                    "id": e.experiment_id,
                    "brier": e.loyo_mean_brier,
                    "model": e.model_family,
                }
                for e in top
            ],
        }

    def research_summary(self) -> str:
        """Generate a human-readable research summary."""
        experiments = self._get_experiments()
        if not experiments:
            return "No experiments available for analysis."

        lines = [
            "Research Knowledge Summary",
            "=" * 60,
            f"Total experiments analyzed: {len(experiments)}",
            "",
        ]

        # Best/worst
        valid = [e for e in experiments if e.loyo_mean_brier > 0]
        if valid:
            best = min(valid, key=lambda e: e.loyo_mean_brier)
            worst = max(valid, key=lambda e: e.loyo_mean_brier)
            lines.append(f"Best Brier:  {best.loyo_mean_brier:.4f} (id={best.experiment_id})")
            lines.append(f"Worst Brier: {worst.loyo_mean_brier:.4f} (id={worst.experiment_id})")
            lines.append("")

        # Per-year insights
        year_insights = self.insights_by_year()
        if year_insights:
            lines.append("Per-Year Performance (best across experiments):")
            for year, data in year_insights.items():
                lines.append(
                    f"  {year}: best={data['best_brier']:.4f}, "
                    f"mean={data['mean_brier']:.4f} (n={data['n_experiments']})"
                )
            lines.append("")

        # Unexplored areas
        unexplored = self.unexplored_parameters()
        if unexplored:
            lines.append(f"Unexplored parameter regions: {len(unexplored)}")
            for u in unexplored[:5]:
                lines.append(f"  - {u['suggestion']}")

        return "\n".join(lines)

    def update_from_registry(self) -> Dict[str, Any]:
        """Refresh insights from the latest experiments in the registry.

        Returns a summary of what was learned, including weak areas and
        unexplored parameters that can guide the next scheduling round.
        """
        experiments = self._get_experiments()
        if not experiments:
            return {"status": "no_experiments", "weak_years": [], "unexplored": []}

        # Identify weak years
        year_insights = self.insights_by_year()
        weak_years: List[Dict[str, Any]] = []
        if year_insights:
            all_briers = [d["best_brier"] for d in year_insights.values()]
            if all_briers:
                mean_best = sum(all_briers) / len(all_briers)
                for year, data in year_insights.items():
                    if data["best_brier"] > mean_best * 1.15:
                        weak_years.append({
                            "year": year,
                            "best_brier": data["best_brier"],
                            "gap": data["best_brier"] - mean_best,
                        })

        # Identify unexplored parameter regions
        unexplored = self.unexplored_parameters()

        # Get what works patterns
        patterns = self.what_works()

        return {
            "status": "updated",
            "n_experiments": len(experiments),
            "weak_years": weak_years,
            "unexplored": unexplored,
            "patterns": patterns,
        }

    def _get_experiments(self) -> List[Any]:
        """Load experiments from registry."""
        if self._registry is None:
            return []
        return self._registry.list(n=1000)
