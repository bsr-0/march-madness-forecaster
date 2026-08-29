"""The outcome referee and the public-pick model must not share a window.

WHY THIS TEST EXISTS. Pool edge is the gap between two beliefs:

    leverage = P_outcome(team wins) - P_public(team picked)

Both were once read from a single table, which silently assumed the crowd is
perfectly calibrated to long-run seed history and, worse, made any update to
that table move both sides together and cancel its own effect exactly. A
recency change would have measured as doing nothing, and the natural conclusion
would have been that recency does not matter.

The failure mode is invisible: nothing raises, no number looks wrong, the
optimiser simply cannot see an edge that is really there. So the separation is
asserted rather than assumed, and the specific matchup that motivated it is
pinned with it.
"""

from __future__ import annotations

import pytest

from src.data.seed_pick_model import (
    _compute_advancement_rates,
    _logistic_rate,
    _recent_win_rates,
    _win_rate,
)
from src.prediction.seed_probabilities import OUTCOME_WINDOW, seed_matchup_probability


def test_outcome_path_uses_the_recent_window():
    """The referee must opt in explicitly, not inherit the default."""
    assert OUTCOME_WINDOW == "recent"


def test_the_two_windows_actually_disagree():
    """A split that returns the same numbers would be decoration.

    6-11 is the matchup that motivated the change: a clear favourite across the
    full history, close to a coin flip since 2010.
    """
    full = _win_rate(6, 11, "full")
    recent = _win_rate(6, 11, "recent")
    assert full > 0.60, f"expected the long window to favour the 6 seed, got {full}"
    assert recent < 0.55, f"expected the recent window to be near even, got {recent}"
    assert full - recent > 0.05


def test_referee_reads_the_recent_window():
    """seed_matchup_probability is the referee's entry point."""
    assert seed_matchup_probability(6, 11) == pytest.approx(_win_rate(6, 11, "recent"))
    assert seed_matchup_probability(6, 11) != pytest.approx(_win_rate(6, 11, "full"))


def test_public_pick_path_still_uses_the_full_window():
    """SEED_PICK_RATES models crowd behaviour and must NOT follow the referee.

    Crowd beliefs are anchored on decades of watching brackets and move slowly.
    Aligning this to the recent window would re-close the gap the split opened.
    """
    assert _compute_advancement_rates() == _compute_advancement_rates("full")
    assert _compute_advancement_rates() != _compute_advancement_rates("recent")


@pytest.mark.parametrize("window", ["full", "recent"])
def test_win_rates_are_antisymmetric(window):
    """p(a,b) + p(b,a) = 1 for every seed pair, in both windows."""
    for a in range(1, 17):
        for b in range(1, 17):
            if a == b:
                continue
            total = _win_rate(a, b, window) + _win_rate(b, a, window)
            assert total == pytest.approx(1.0, abs=1e-12)


def test_unknown_window_raises():
    """No silent fallback: a typo must not quietly return the long window."""
    with pytest.raises(ValueError):
        _win_rate(1, 16, "recennt")


def test_thin_recent_cells_are_excluded_rather_than_trusted():
    """A matchup with a handful of games must fall through to the curve.

    Without this, a 2-of-3 deep-round cell would enter the referee as 0.667 and
    be indistinguishable from a rate measured on sixty games.
    """
    recent = _recent_win_rates()
    assert recent, "expected a non-empty recent table; are the Kaggle files present?"
    # 1 vs 15 has essentially never happened, so it cannot be in the table.
    assert (1, 15) not in recent
    assert _win_rate(1, 15, "recent") == pytest.approx(_logistic_rate(1, 15))


def test_recent_table_only_holds_well_sampled_cells():
    """Every published cell should be one the window can actually support."""
    from src.data.seed_pick_model import _RECENT_MIN_GAMES

    assert _RECENT_MIN_GAMES >= 8
    # The eight canonical R64 matchups all have ~60 games since 2010 and must
    # therefore be present; if they are not, the loader is broken.
    for lo, hi in [(1, 16), (2, 15), (3, 14), (4, 13), (5, 12), (6, 11), (7, 10), (8, 9)]:
        assert (lo, hi) in _recent_win_rates()
