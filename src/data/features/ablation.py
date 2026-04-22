"""Feature-group ablation for A/B testing.

Zeros out feature groups in team vectors based on config flags.
Called after all overlays and before matchup vector construction.
"""

from __future__ import annotations

import numpy as np

ABLATION_GROUPS = {
    "ablate_elo": [29],
    "ablate_conf_tourney": [46, 47, 48],
    "ablate_late_season": [49, 50, 51],
    "ablate_market": [52, 53],
    "ablate_injury": [54],
}


def apply_ablation(v: np.ndarray, config) -> None:
    """Zero out ablated feature groups in a team vector (in-place)."""
    for flag, indices in ABLATION_GROUPS.items():
        if getattr(config, flag, False):
            for idx in indices:
                if idx < len(v):
                    v[idx] = 0.0
