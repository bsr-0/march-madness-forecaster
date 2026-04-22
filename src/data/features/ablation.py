"""Feature-group ablation for A/B testing.

Zeros out feature groups in team vectors based on config flags.
Called after all overlays (roster, market) and before matchup vector construction.
"""

from __future__ import annotations

import numpy as np


# Feature index ranges for each ablation group.
# These must match the ordering in TeamFeatures.to_vector() / metrics_to_team_vector().
ABLATION_GROUPS = {
    "ablate_elo": [29],                    # elo_rating
    "ablate_conf_tourney": [46, 47, 48],   # conf_tourney_champion, games, margin
    "ablate_late_season": [49, 50, 51],    # late_season_games, margin, win_pct
    "ablate_market": [52, 53],             # market_implied_prob, market_spread
    "ablate_injury": [54],                 # injury_risk
}

# Massey features are passed via external_rating_composite/spread and massey_features
# params to metrics_to_team_vector. The indices depend on the massey_systems layout,
# but the composite goes to a known slot. We zero the massey-related indices.
# These are NOT in a fixed position — they're NaN by default and only populated
# when Massey data is available. The ablation approach for Massey is to skip
# loading rather than zeroing indices. See sample_loading.py.


def apply_ablation(v: np.ndarray, config) -> None:
    """Zero out ablated feature groups in a team vector (in-place)."""
    for flag, indices in ABLATION_GROUPS.items():
        if getattr(config, flag, False):
            for idx in indices:
                if idx < len(v):
                    v[idx] = 0.0
