"""Unified quant report — consolidated output from both layers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, Optional

from .kaggle_layer import KaggleResult
from .espn_layer import ESPNResult


@dataclass
class QuantReport:
    """Consolidated report from the quant decision engine.

    Answers the key questions:
    - Kaggle: "What is my expected log loss?" / "Am I competitive?"
    - ESPN: "What is my probability of winning?" / "Am I contrarian enough?"
    """

    kaggle: KaggleResult
    espn: ESPNResult
    shared_core_stats: Dict = field(default_factory=dict)

    def summary(self) -> str:
        """Human-readable summary answering the key decision questions."""
        lines = [
            "=" * 60,
            "QUANT DECISION SYSTEM REPORT",
            "=" * 60,
            "",
            self.kaggle.summary(),
            "",
            self.espn.summary(),
            "",
            "-" * 60,
            "KEY ANSWERS",
            "-" * 60,
            f"  Expected log loss:   {self.kaggle.log_loss:.4f}",
            f"  Competitive tier:    {self.kaggle.tier}",
        ]

        for br in self.espn.bracket_results:
            lines.extend([
                f"  Pool {br.pool_size} win prob:  {br.win_probability:.4f}",
                f"  Pool {br.pool_size} strategy:  {br.strategy} "
                f"(contrarian={br.contrarian_index:.3f})",
            ])

        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> Dict:
        """Serialize to dictionary for JSON export."""
        return {
            "kaggle": {
                "log_loss": self.kaggle.log_loss,
                "tier": self.kaggle.tier,
                "brier_score": self.kaggle.brier_score,
                "calibration_ece": self.kaggle.calibration_ece,
                "n_matchups": self.kaggle.n_matchups,
                "brier_skill_score": self.kaggle.brier_skill_score,
                "calibration_uncertainty_band": self.kaggle.calibration_uncertainty_band,
                "calibration_n_samples": self.kaggle.calibration_n_samples,
            },
            "espn": {
                "brackets": [
                    {
                        "pool_size": br.pool_size,
                        "strategy": br.strategy,
                        "win_probability": br.win_probability,
                        "top_1_percent_probability": br.top_1_percent_probability,
                        "average_finish": br.average_finish,
                        "expected_score": br.expected_score,
                        "contrarian_index": br.contrarian_index,
                    }
                    for br in self.espn.bracket_results
                ],
            },
            "shared_core": self.shared_core_stats,
        }

    def save(self, path: str) -> None:
        """Save report as JSON."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
