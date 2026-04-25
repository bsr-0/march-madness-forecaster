"""Tests for src.pipeline.stages.ev_mode — EV mode utility functions."""

from src.pipeline.stages.ev_mode import (
    classify_pick_strategy,
    compute_leverage,
    format_leverage_table,
)


def test_compute_leverage_positive():
    # Model favours more than public
    lev = compute_leverage(model_prob=0.8, public_pick_rate=0.4)
    assert lev > 0


def test_compute_leverage_negative():
    # Model less confident than public
    lev = compute_leverage(model_prob=0.2, public_pick_rate=0.6)
    assert lev < 0


def test_compute_leverage_zero_public():
    assert compute_leverage(0.5, 0.0) == 0.0


def test_compute_leverage_weighted():
    lev1 = compute_leverage(0.8, 0.4, scoring_weight=1.0)
    lev2 = compute_leverage(0.8, 0.4, scoring_weight=2.0)
    assert abs(lev2 - 2 * lev1) < 1e-9


def test_classify_neutral():
    assert classify_pick_strategy(0.05, 0.5, 0.48) == "neutral"


def test_classify_contrarian():
    assert classify_pick_strategy(0.5, 0.6, 0.2) == "contrarian"


def test_classify_chalk():
    assert classify_pick_strategy(0.3, 0.7, 0.5) == "chalk"


def test_classify_fade():
    assert classify_pick_strategy(-0.3, 0.3, 0.7) == "fade"


def test_format_leverage_table():
    teams = [
        {"team": "Duke", "model_prob": 0.7, "public_rate": 0.4, "leverage": 0.5, "strategy": "contrarian"},
        {"team": "UNC", "model_prob": 0.3, "public_rate": 0.5, "leverage": -0.2, "strategy": "fade"},
    ]
    table = format_leverage_table(teams, top_n=5)
    assert "Duke" in table
    assert "UNC" in table
    assert "contrarian" in table
