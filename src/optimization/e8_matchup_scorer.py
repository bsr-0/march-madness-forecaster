"""Elite 8 matchup interaction scoring for bracket pool optimization.

The E8 has a 42.6% upset rate (2008-2025) — far higher than earlier rounds.
Existing seed-independent signals (raw team-level differentials) show
BSS=-0.010 in E8: they're worse than seeds. The problem is that raw
differentials between two elite teams flatten to noise.

This module uses PAIRWISE INTERACTION features instead: team A's offensive
strength × team B's defensive weakness. A pressing defense against a
turnover-prone offense is categorically different from two high-TO teams,
even if the raw TO differential is the same.

With only 68 E8 games in sample (4/year × 17 years), no ML model can be
trained without massive overfitting. All weights are fixed by domain
knowledge with equal weighting across signals.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Signal weight configuration (equal — no data to estimate differential)
# ---------------------------------------------------------------------------

_SIGNAL_WEIGHTS = {
    "turnover_interaction": 0.22,
    "rebounding_interaction": 0.17,
    "tempo_exploitation": 0.17,
    "three_point_interaction": 0.12,
    "coach_e8_experience": 0.17,
    "clutch_composure_interaction": 0.15,
}

# Maximum probability adjustment (±8pp).  The 42.6% upset rate implies
# favorites win ~57.4%; an 8pp swing gives [49.4%, 65.4%].
_MAX_ADJUSTMENT = 0.08

# Dead zone: halve adjustment when composite is close to neutral (0.5).
_DEAD_ZONE_THRESHOLD = 0.10

# D1 population averages for centering interactions.
_AVG_TURNOVER_RATE = 0.18
_AVG_OFFENSIVE_REB_RATE = 0.28
_AVG_DEFENSIVE_REB_RATE = 0.72
_AVG_TEMPO = 68.0
_AVG_THREE_PT_PCT = 0.34
_AVG_OPP_EFG_PCT = 0.48

# Clutch/blown-lead population averages (src/data/features/clutch_metrics.py).
# Neutral defaults (0.5 / 0.0) so a team missing clutch data — the common
# case until historical backfill runs — contributes zero asymmetry rather
# than a spurious adjustment.
_AVG_BLOWN_10PT_LEAD_RATE = 0.5
_AVG_CLOSE_GAME_WIN_RATE = 0.5


def _sigmoid(x: float, center: float = 0.0, scale: float = 1.0) -> float:
    """Logistic sigmoid mapped to [0, 1]."""
    z = (x - center) / scale if scale != 0 else 0.0
    z = max(-20.0, min(20.0, z))  # clamp to avoid overflow
    return 1.0 / (1.0 + math.exp(-z))


# ---------------------------------------------------------------------------
# Data container for E8 matchup scores
# ---------------------------------------------------------------------------


@dataclass
class E8MatchupScore:
    """Result of scoring an E8 matchup interaction."""

    composite: float  # [0, 1], centered at 0.5 = neutral
    adjustment: float  # [-0.08, +0.08], probability shift for team_a
    signals: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Core scorer
# ---------------------------------------------------------------------------


class E8MatchupInteractionScorer:
    """Score Elite 8 matchups using pairwise interaction features.

    Each signal computes a bidirectional asymmetry:
        (team_a attack × team_b defense weakness)
      - (team_b attack × team_a defense weakness)

    Positive composite → team_a has the interaction advantage.
    """

    def score(self, team_a, team_b) -> E8MatchupScore:
        """Score an E8 matchup between two teams.

        Args:
            team_a: Object with TeamFeatures-compatible attributes.
            team_b: Object with TeamFeatures-compatible attributes.

        Returns:
            E8MatchupScore with composite, adjustment, and per-signal values.
        """
        signals = {
            "turnover_interaction": self._turnover_interaction(team_a, team_b),
            "rebounding_interaction": self._rebounding_interaction(team_a, team_b),
            "tempo_exploitation": self._tempo_exploitation(team_a, team_b),
            "three_point_interaction": self._three_point_interaction(team_a, team_b),
            "coach_e8_experience": self._coach_experience(team_a, team_b),
            "clutch_composure_interaction": self._clutch_composure_interaction(team_a, team_b),
        }

        composite = sum(_SIGNAL_WEIGHTS[name] * value for name, value in signals.items())

        # Map [0, 1] composite to [-MAX, +MAX] adjustment.
        raw_adjustment = 2 * _MAX_ADJUSTMENT * (composite - 0.5)

        # Dead zone: halve small adjustments to avoid noise.
        if abs(composite - 0.5) < _DEAD_ZONE_THRESHOLD:
            raw_adjustment *= 0.5

        adjustment = max(-_MAX_ADJUSTMENT, min(_MAX_ADJUSTMENT, raw_adjustment))

        return E8MatchupScore(
            composite=composite,
            adjustment=adjustment,
            signals=signals,
        )

    def _turnover_interaction(self, a, b) -> float:
        """Pressing defense × turnover-prone offense, bidirectional."""
        # a's defense forcing TOs × b's offense committing TOs
        a_pressure = getattr(a, "opp_turnover_rate", _AVG_TURNOVER_RATE) - _AVG_TURNOVER_RATE
        b_prone = getattr(b, "turnover_rate", _AVG_TURNOVER_RATE) - _AVG_TURNOVER_RATE
        forward = a_pressure * b_prone

        # Reverse: b's defense × a's offense
        b_pressure = getattr(b, "opp_turnover_rate", _AVG_TURNOVER_RATE) - _AVG_TURNOVER_RATE
        a_prone = getattr(a, "turnover_rate", _AVG_TURNOVER_RATE) - _AVG_TURNOVER_RATE
        reverse = b_pressure * a_prone

        return _sigmoid(forward - reverse, center=0.0, scale=0.001)

    def _rebounding_interaction(self, a, b) -> float:
        """Offensive rebounding strength × defensive rebounding weakness."""
        a_orb = getattr(a, "offensive_reb_rate", _AVG_OFFENSIVE_REB_RATE) - _AVG_OFFENSIVE_REB_RATE
        b_drb_weakness = _AVG_DEFENSIVE_REB_RATE - getattr(b, "defensive_reb_rate", _AVG_DEFENSIVE_REB_RATE)
        forward = a_orb * b_drb_weakness

        b_orb = getattr(b, "offensive_reb_rate", _AVG_OFFENSIVE_REB_RATE) - _AVG_OFFENSIVE_REB_RATE
        a_drb_weakness = _AVG_DEFENSIVE_REB_RATE - getattr(a, "defensive_reb_rate", _AVG_DEFENSIVE_REB_RATE)
        reverse = b_orb * a_drb_weakness

        return _sigmoid(forward - reverse, center=0.0, scale=0.002)

    def _tempo_exploitation(self, a, b) -> float:
        """Does one team's preferred tempo hurt the other?

        A fast team forces a slow, rigid opponent out of its comfort zone.
        Rigidity = how far the team's tempo is from average (extreme tempo
        teams are more disrupted when forced to change pace).
        """
        a_tempo = getattr(a, "adj_tempo", _AVG_TEMPO)
        b_tempo = getattr(b, "adj_tempo", _AVG_TEMPO)

        tempo_gap = a_tempo - b_tempo
        b_rigidity = abs(b_tempo - _AVG_TEMPO)
        a_rigidity = abs(a_tempo - _AVG_TEMPO)

        # a forces pace on b: positive gap × b's rigidity
        forward = tempo_gap * b_rigidity / 100.0
        # b forces pace on a: negative gap × a's rigidity
        reverse = -tempo_gap * a_rigidity / 100.0

        # Net exploitation advantage for team_a
        net = forward - reverse
        return _sigmoid(net, center=0.0, scale=0.015)

    def _three_point_interaction(self, a, b) -> float:
        """Three-point shooting strength × perimeter defense weakness."""
        a_three = getattr(a, "three_pt_pct", _AVG_THREE_PT_PCT) - _AVG_THREE_PT_PCT
        b_def_weakness = getattr(b, "opp_effective_fg_pct", _AVG_OPP_EFG_PCT) - _AVG_OPP_EFG_PCT
        forward = a_three * b_def_weakness

        b_three = getattr(b, "three_pt_pct", _AVG_THREE_PT_PCT) - _AVG_THREE_PT_PCT
        a_def_weakness = getattr(a, "opp_effective_fg_pct", _AVG_OPP_EFG_PCT) - _AVG_OPP_EFG_PCT
        reverse = b_three * a_def_weakness

        return _sigmoid(forward - reverse, center=0.0, scale=0.003)

    def _coach_experience(self, a, b) -> float:
        """Coach E8 experience asymmetry."""
        a_e8 = getattr(a, "coach_e8_appearances", 0)
        a_deep = getattr(a, "coach_deep_run_rate", 0.0)
        a_score = math.log1p(a_e8) + 0.5 * a_deep

        b_e8 = getattr(b, "coach_e8_appearances", 0)
        b_deep = getattr(b, "coach_deep_run_rate", 0.0)
        b_score = math.log1p(b_e8) + 0.5 * b_deep

        return _sigmoid(a_score - b_score, center=0.0, scale=1.0)

    def _clutch_composure_interaction(self, a, b) -> float:
        """Composure asymmetry: blown-lead tendency and close-game record.

        Fields come from src/data/features/clutch_metrics.py's per-team
        aggregation (blown_10pt_lead_rate, close_game_win_rate). Not a
        pairwise interaction like the other signals (composure is a team
        trait, not a matchup-dependent one) — scored as a direct asymmetry
        so a team that reliably protects leads and wins close games gets an
        edge over one that doesn't, independent of the opponent's profile.
        """
        a_blown = getattr(a, "blown_10pt_lead_rate", _AVG_BLOWN_10PT_LEAD_RATE)
        b_blown = getattr(b, "blown_10pt_lead_rate", _AVG_BLOWN_10PT_LEAD_RATE)
        a_close = getattr(a, "close_game_win_rate", _AVG_CLOSE_GAME_WIN_RATE)
        b_close = getattr(b, "close_game_win_rate", _AVG_CLOSE_GAME_WIN_RATE)

        # Lower blown-lead rate and higher close-game win rate = more composed.
        a_composure = (1.0 - a_blown) + a_close
        b_composure = (1.0 - b_blown) + b_close

        return _sigmoid(a_composure - b_composure, center=0.0, scale=0.5)


# ---------------------------------------------------------------------------
# Pipeline integration: adjust round probabilities
# ---------------------------------------------------------------------------


def apply_e8_adjustments(
    round_probs: Dict[str, Dict[str, float]],
    e8_matchups: List[Tuple[str, str]],
    team_features: Dict[str, object],
) -> Dict[str, Dict[str, float]]:
    """Adjust E8/F4/CHAMP round probabilities based on matchup interactions.

    For each predicted E8 matchup, computes the interaction score and shifts
    both teams' E8 advancement probability. F4 and CHAMP scale proportionally.

    Args:
        round_probs: team_id -> {round_name: probability}. NOT mutated.
        e8_matchups: List of (team_a_id, team_b_id) predicted E8 matchups.
        team_features: team_id -> object with TeamFeatures-compatible attrs.

    Returns:
        New round_probs dict with E8/F4/CHAMP adjusted.
    """
    adjusted = copy.deepcopy(round_probs)
    scorer = E8MatchupInteractionScorer()

    for team_a_id, team_b_id in e8_matchups:
        feat_a = team_features.get(team_a_id)
        feat_b = team_features.get(team_b_id)
        if feat_a is None or feat_b is None:
            continue

        result = scorer.score(feat_a, feat_b)

        # Apply adjustments: team_a gets +adjustment, team_b gets -adjustment.
        for team_id, adj in [(team_a_id, result.adjustment), (team_b_id, -result.adjustment)]:
            if team_id not in adjusted or "E8" not in adjusted[team_id]:
                continue

            original_e8 = adjusted[team_id]["E8"]
            new_e8 = max(0.01, min(0.99, original_e8 + adj))
            adjusted[team_id]["E8"] = new_e8

            # Cascade: scale F4 and CHAMP proportionally.
            if original_e8 > 0:
                ratio = new_e8 / original_e8
                for later_round in ("F4", "CHAMP"):
                    if later_round in adjusted[team_id]:
                        adjusted[team_id][later_round] = max(
                            0.001,
                            min(0.99, adjusted[team_id][later_round] * ratio),
                        )

    return adjusted


def predict_e8_matchups(
    seeds: Dict[str, int],
) -> List[Tuple[str, str]]:
    """Predict likely E8 matchups from bracket structure.

    Uses the standard bracket pairing: top half (1/16/8/9/4/13/5/12) vs
    bottom half (2/15/7/10/3/14/6/11) of each region. Returns the chalk
    E8 matchup (lowest seed from each half) for each region.

    Args:
        seeds: team_id -> seed for tournament teams.

    Returns:
        List of (team_a_id, team_b_id) representing predicted E8 matchups.
    """
    # Group teams by seed for lookup.
    seed_to_teams: Dict[int, List[str]] = {}
    for team_id, seed in seeds.items():
        seed_to_teams.setdefault(seed, []).append(team_id)

    # E8 bracket halves (standard 64-team bracket).
    top_half_seeds = {1, 16, 8, 9, 4, 13, 5, 12}
    bot_half_seeds = {2, 15, 7, 10, 3, 14, 6, 11}

    # For each region, the chalk E8 matchup is the best seed from each half.
    # With 4 regions and 4 teams per seed, we pair them by position.
    top_teams = []
    for s in sorted(top_half_seeds):
        top_teams.extend(seed_to_teams.get(s, []))

    bot_teams = []
    for s in sorted(bot_half_seeds):
        bot_teams.extend(seed_to_teams.get(s, []))

    # Pair by position (each region contributes one from each half).
    # With a standard 64-team bracket: 4 regions × 1 E8 game = 4 matchups.
    matchups = []
    n_regions = min(len(top_teams), len(bot_teams), 4)
    for i in range(n_regions):
        matchups.append((top_teams[i], bot_teams[i]))

    return matchups
