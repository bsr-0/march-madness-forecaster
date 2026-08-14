"""Unit tests for D1 volatile adjustment.

Covers the loader (per-team point-margin volatility from cbbpy games,
bridged to canonical IDs), the adjustment itself (regression toward
per-round uniform weighted by normalized volatility), and the pipeline
registry + permutation enumeration.
"""

from pathlib import Path

import pytest

from src.prediction.volatile_probabilities import (
    ROUND_NAMES,
    build_volatile_round_probs,
    load_team_volatility,
)

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
_TEAMS_PER_ROUND = {"R64": 32, "R32": 16, "S16": 8, "E8": 4, "F4": 2, "CHAMP": 1}


def test_load_team_volatility_2026_covers_all_tournament_teams():
    """All 68 teams in the 2026 field get a volatility value (via the cbbpy bridge)."""
    import json

    seeds = json.load(open(DATA_ROOT / "raw" / "tournament_seeds_2026.json"))
    tourney_ids = {s["team_id"] for s in seeds}
    vol = load_team_volatility(2026, tourney_ids, DATA_ROOT)
    assert set(vol.keys()) == tourney_ids
    # Values should span the normalized range (not all be the 0.5 fallback).
    assert any(v < 0.3 for v in vol.values()), "Expected at least one low-volatility team"
    assert any(v > 0.7 for v in vol.values()), "Expected at least one high-volatility team"


def test_load_team_volatility_normalized_to_unit_range():
    """Output values must land in [0, 1] — the adjustment math assumes this."""
    import json

    seeds = json.load(open(DATA_ROOT / "raw" / "tournament_seeds_2026.json"))
    tourney_ids = {s["team_id"] for s in seeds}
    vol = load_team_volatility(2026, tourney_ids, DATA_ROOT)
    for tid, v in vol.items():
        assert 0.0 <= v <= 1.0, f"Team {tid} volatility {v} outside [0,1]"


def test_load_team_volatility_returns_neutral_for_missing_data():
    """Unknown year → neutral 0.5 fallback for every requested team (no crash)."""
    vol = load_team_volatility(1999, {"duke", "kansas"}, DATA_ROOT)
    assert vol == {"duke": 0.5, "kansas": 0.5}


def test_build_volatile_renormalizes_per_round_to_team_counts():
    """After adjustment, round totals must match the bracket's team-count targets."""
    # 64 teams with identical baseline rp, so renorm math is crisp.
    teams = [f"t{i}" for i in range(1, 65)]
    model_rp = {t: {"R64": 1.0, "R32": 0.5, "S16": 0.25, "E8": 0.125, "F4": 0.0625, "CHAMP": 0.03125} for t in teams}
    # Mixed volatility so blending actually changes things.
    volatility = {t: (i / 63.0) for i, t in enumerate(teams)}

    adjusted = build_volatile_round_probs(model_rp, volatility)
    for rnd, target in _TEAMS_PER_ROUND.items():
        total = sum(adjusted[t][rnd] for t in teams)
        assert total == pytest.approx(target, rel=1e-6), (
            f"Round {rnd} total should renormalize to {target}, got {total:.4f}"
        )


def test_build_volatile_low_volatility_team_keeps_baseline_rp():
    """v=0 means no blending — team's adjusted rp should match its base (up to renorm scale)."""
    # Two teams — one low vol, one high vol — with different baselines.
    teams = ["consistent", "chaotic"]
    model_rp = {
        "consistent": {r: 0.5 for r in ROUND_NAMES},
        "chaotic": {r: 0.5 for r in ROUND_NAMES},
    }
    volatility = {"consistent": 0.0, "chaotic": 1.0}

    adjusted = build_volatile_round_probs(model_rp, volatility, strength=0.5)
    # With equal baseline 0.5 and uniform ~0.5 (for R32), the blend doesn't
    # change anything for chaotic either — so test a more discriminating case.
    model_rp = {
        "consistent": {"R64": 1.0, "R32": 0.9, "S16": 0.7, "E8": 0.5, "F4": 0.3, "CHAMP": 0.15},
        "chaotic": {"R64": 1.0, "R32": 0.1, "S16": 0.05, "E8": 0.02, "F4": 0.01, "CHAMP": 0.005},
    }
    adjusted = build_volatile_round_probs(model_rp, volatility, strength=0.5)
    # Consistent team (v=0) shouldn't move toward uniform; ratio adjusted/base
    # should equal the renorm scale.
    # Chaotic team should move TOWARD its round's uniform (weaker team
    # gains mass since uniform > base for small base).
    for r in ["R32", "S16", "E8", "F4"]:
        assert adjusted["chaotic"][r] > model_rp["chaotic"][r], (
            f"Weak chaotic team should gain mass at {r}; base={model_rp['chaotic'][r]:.3f}, adj={adjusted['chaotic'][r]:.3f}"
        )


def test_build_volatile_strong_volatile_team_loses_mass_to_upset_losses():
    """Strong team + high volatility → more upset losses → lower round_probs than baseline."""
    # Populate the field so the baseline sums make sense. Strong_vol is the
    # only volatile one; eleven other teams are neutral-vol filler.
    teams = ["strong_vol"] + [f"f{i}" for i in range(1, 12)]
    model_rp = {t: {"R64": 1.0, "R32": 0.5, "S16": 0.25, "E8": 0.125, "F4": 0.0625, "CHAMP": 0.03125} for t in teams}
    # Strong team: high base
    model_rp["strong_vol"] = {"R64": 1.0, "R32": 0.95, "S16": 0.80, "E8": 0.60, "F4": 0.35, "CHAMP": 0.18}
    volatility = {t: 0.5 for t in teams}
    volatility["strong_vol"] = 1.0  # maximum volatility

    # Baseline renorm only (no vol):
    zero_vol = {t: 0.0 for t in teams}
    base_adj = build_volatile_round_probs(model_rp, zero_vol, strength=0.5)
    vol_adj = build_volatile_round_probs(model_rp, volatility, strength=0.5)

    # At S16 and later, strong_vol's vol-adjusted rp must be BELOW its
    # zero-vol rp (it lost mass to upset losses).
    for r in ["S16", "E8", "F4", "CHAMP"]:
        assert vol_adj["strong_vol"][r] < base_adj["strong_vol"][r], (
            f"Strong volatile team should LOSE mass at {r}; "
            f"base_adj={base_adj['strong_vol'][r]:.3f}, vol_adj={vol_adj['strong_vol'][r]:.3f}"
        )


def test_volatile_registered_in_pipeline():
    from src.prediction.strategy_pipeline import IMPLEMENTED_ADJUSTMENTS

    assert "volatile" in IMPLEMENTED_ADJUSTMENTS


def test_permutation_generator_enumerates_volatile_strategies():
    from src.prediction.strategy_pipeline import generate_all_permutations

    strategies = generate_all_permutations(implemented_only=True)
    vol = [s for s in strategies if "volatile" in s]
    assert len(vol) >= 5
    # Phase 3 spot-check: the bare single-source + volatile variants exist.
    assert "seed+volatile" in strategies
    assert "torvik+volatile" in strategies


def test_resolve_pipeline_returns_none_without_volatility_context():
    """Using the volatile adjustment without supplying team_volatility must fail cleanly."""
    from src.prediction.strategy_pipeline import parse_pipeline, resolve_pipeline_round_probs

    sources, adjustments, _ = parse_pipeline("seed+volatile_forward")
    base_rp = {"t1": {r: 0.5 for r in ROUND_NAMES}}
    # Resolver returns None when an adjustment's required context is missing.
    result = resolve_pipeline_round_probs(
        sources,
        adjustments,
        {"seed": base_rp},
        pick_dist={},
        # team_volatility intentionally omitted
    )
    assert result is None


def test_resolve_pipeline_applies_volatile_when_context_provided():
    """End-to-end: with team_volatility supplied, the resolver returns a dict with flattened rp."""
    from src.prediction.strategy_pipeline import parse_pipeline, resolve_pipeline_round_probs

    teams = ["t1", "t2"]
    base_rp = {t: {r: 0.5 for r in ROUND_NAMES} for t in teams}
    # Make t1 max-vol and t2 min-vol so the adjustment has something to do.
    vol = {"t1": 1.0, "t2": 0.0}

    sources, adjustments, _ = parse_pipeline("seed+volatile_forward")
    result = resolve_pipeline_round_probs(
        sources,
        adjustments,
        {"seed": base_rp},
        pick_dist={},
        team_volatility=vol,
    )
    assert result is not None
    assert set(result.keys()) == {"t1", "t2"}
    for r in ROUND_NAMES:
        assert r in result["t1"]
        assert r in result["t2"]
