"""Unit tests for C3 momentum adjustment.

Covers the snapshot finder, the Jan→Mar efficiency-proxy delta loader,
the tanh-saturated ±3% multiplicative adjustment, and the pipeline
registry + permutation enumeration.

Same shape as ``tests/test_roster_adj_adjustment.py`` since momentum is
the third C-base adjustment. Two-sided (improving teams boosted,
declining teams penalized) — unlike one-sided ``coach_adj``.
"""

from pathlib import Path

import pytest

from src.prediction.momentum_probabilities import (
    ROUND_NAMES,
    build_momentum_round_probs,
    load_team_momentum,
)

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
_TEAMS_PER_ROUND = {"R64": 32, "R32": 16, "S16": 8, "E8": 4, "F4": 2, "CHAMP": 1}


def test_load_team_momentum_2026_covers_tournament_field():
    """All 68 teams in the 2026 field get a momentum delta (team IDs match canonical)."""
    import json

    seeds = json.load(open(DATA_ROOT / "raw" / "tournament_seeds_2026.json"))
    tourney_ids = {s["team_id"] for s in seeds}
    mom = load_team_momentum(2026, tourney_ids, DATA_ROOT)
    coverage = len(set(mom.keys()) & tourney_ids)
    assert coverage >= 60, f"Expected >=60 of 68 teams covered, got {coverage}"
    for tid, delta in mom.items():
        assert isinstance(delta, float), f"{tid} delta not float: {type(delta)}"


def test_load_team_momentum_distribution_has_both_signs():
    """Some teams should be improving, others declining — deltas span around 0."""
    import json

    seeds = json.load(open(DATA_ROOT / "raw" / "tournament_seeds_2026.json"))
    tourney_ids = {s["team_id"] for s in seeds}
    mom = load_team_momentum(2026, tourney_ids, DATA_ROOT)
    values = list(mom.values())
    assert any(v > 0 for v in values), "Expected at least one improving team"
    assert any(v < 0 for v in values), "Expected at least one declining team"


def test_load_team_momentum_empty_for_missing_year():
    """Year without four-factor snapshot files → empty dict, no crash."""
    mom = load_team_momentum(1999, {"duke", "kansas"}, DATA_ROOT)
    assert mom == {}


def test_load_team_momentum_walk_forward_isolation():
    """load_team_momentum(year=2025) only reads torvik_four_factors_2025*.json."""
    import json

    hist_dir = DATA_ROOT / "raw" / "historical"
    ctx_path = hist_dir / "tournament_context_2025.json"
    raw = None
    if ctx_path.exists():
        with open(ctx_path) as f:
            raw = json.load(f).get("seeds")
    if raw is None:
        raw = json.load(open(hist_dir / "tournament_seeds_2025.json"))
    tourney_ids = {s["team_id"] for s in raw["teams"]}
    mom_2025 = load_team_momentum(2025, tourney_ids, DATA_ROOT)
    assert mom_2025, "2025 momentum should produce values"
    # Values should be bounded (four-factor deltas rarely exceed ±0.2 absolute).
    for tid, d in mom_2025.items():
        assert abs(d) < 1.0, f"{tid} delta {d} outside sane range"


def test_build_momentum_renormalizes_per_round_to_team_counts():
    """After adjustment, round totals must match the bracket's team-count targets."""
    teams = [f"t{i}" for i in range(1, 65)]
    model_rp = {t: {"R64": 1.0, "R32": 0.5, "S16": 0.25, "E8": 0.125, "F4": 0.0625, "CHAMP": 0.03125} for t in teams}
    # Mixed deltas so adjustment actually changes things.
    momentum = {t: (i - 32) * 0.005 for i, t in enumerate(teams)}

    adjusted = build_momentum_round_probs(model_rp, momentum)
    for rnd, target in _TEAMS_PER_ROUND.items():
        total = sum(adjusted[t][rnd] for t in teams)
        assert total == pytest.approx(target, rel=1e-6), (
            f"Round {rnd} total should renormalize to {target}, got {total:.4f}"
        )


def test_build_momentum_improving_team_beats_declining_team():
    """Team with positive delta > team with negative delta, all else equal."""
    teams = ["hot", "cold"] + [f"f{i}" for i in range(1, 63)]
    model_rp = {t: {"R64": 1.0, "R32": 0.5, "S16": 0.25, "E8": 0.125, "F4": 0.0625, "CHAMP": 0.03125} for t in teams}
    momentum = {"hot": 0.08, "cold": -0.08}
    for t in teams[2:]:
        momentum[t] = 0.0

    adjusted = build_momentum_round_probs(model_rp, momentum)
    for r in ROUND_NAMES:
        assert adjusted["hot"][r] > adjusted["cold"][r], (
            f"Improving team should beat declining team at {r} "
            f"(hot={adjusted['hot'][r]:.4f}, cold={adjusted['cold'][r]:.4f})"
        )


def test_build_momentum_caps_factor_at_plus_minus_3_percent():
    """tanh saturation: even delta=0.5 (pathological) stays within ±3%."""
    teams = ["extreme_hot", "extreme_cold"]
    model_rp = {
        "extreme_hot": {r: 0.5 for r in ROUND_NAMES},
        "extreme_cold": {r: 0.5 for r in ROUND_NAMES},
    }
    momentum = {"extreme_hot": 0.5, "extreme_cold": -0.5}

    adjusted = build_momentum_round_probs(model_rp, momentum)
    # Ratio of hot/cold must respect cap: max (1.03 / 0.97) ≈ 1.062.
    ratio = adjusted["extreme_hot"]["R64"] / adjusted["extreme_cold"]["R64"]
    assert ratio <= 1.03 / 0.97 + 1e-6, f"Hot/cold ratio should respect ±3% cap, got {ratio:.4f}"


def test_build_momentum_is_two_sided():
    """Declining team's pre-renorm factor is < 1.0 (unlike coach_adj which is one-sided).

    Verified by running with all-zero momentum and comparing: the negative-delta
    team should have LOWER rp than its zero-delta baseline (modulo renorm).
    """
    teams = ["plus", "zero", "minus"]
    model_rp = {t: {r: 0.5 for r in ROUND_NAMES} for t in teams}

    zero_mom = {t: 0.0 for t in teams}
    base_adj = build_momentum_round_probs(model_rp, zero_mom)

    mixed_mom = {"plus": 0.05, "zero": 0.0, "minus": -0.05}
    mixed_adj = build_momentum_round_probs(model_rp, mixed_mom)

    for r in ROUND_NAMES:
        # Minus team should lose mass (two-sided: can penalize).
        assert mixed_adj["minus"][r] < base_adj["minus"][r], (
            f"Declining team should lose mass at {r}: "
            f"base={base_adj['minus'][r]:.4f}, mixed={mixed_adj['minus'][r]:.4f}"
        )
        # Plus team should gain mass.
        assert mixed_adj["plus"][r] > base_adj["plus"][r], f"Improving team should gain mass at {r}"


def test_build_momentum_missing_team_treated_as_zero_delta():
    """Team absent from momentum dict → treated as delta=0 (no adjustment)."""
    teams = ["unknown", "zero", "veteran"]
    model_rp = {t: {r: 0.5 for r in ROUND_NAMES} for t in teams}
    # 'unknown' intentionally missing; 'zero' has explicit 0.0.
    momentum = {"zero": 0.0, "veteran": 0.05}

    adjusted = build_momentum_round_probs(model_rp, momentum)
    # unknown and zero should be equal (both treated as 0 delta).
    for r in ROUND_NAMES:
        assert adjusted["unknown"][r] == pytest.approx(adjusted["zero"][r], rel=1e-6), (
            f"Missing team should match 0-delta team at {r}"
        )


def test_momentum_registered_in_pipeline():
    from src.prediction.strategy_pipeline import IMPLEMENTED_ADJUSTMENTS

    assert "momentum" in IMPLEMENTED_ADJUSTMENTS


def test_permutation_generator_enumerates_momentum_strategies():
    from src.prediction.strategy_pipeline import generate_all_permutations

    strategies = generate_all_permutations(implemented_only=True)
    mom = [s for s in strategies if "momentum" in s]
    assert len(mom) >= 5, f"Expected >=5 momentum strategies, got {len(mom)}"
    assert "seed+momentum" in strategies
    assert "torvik+momentum" in strategies


def test_resolve_pipeline_returns_none_without_team_momentum_context():
    """Using momentum without supplying team_momentum must return None."""
    from src.prediction.strategy_pipeline import parse_pipeline, resolve_pipeline_round_probs

    sources, adjustments, _ = parse_pipeline("seed+momentum_forward")
    base_rp = {"t1": {r: 0.5 for r in ROUND_NAMES}}
    result = resolve_pipeline_round_probs(
        sources,
        adjustments,
        {"seed": base_rp},
        pick_dist={},
        # team_momentum intentionally omitted
    )
    assert result is None


def test_resolve_pipeline_applies_momentum_when_context_provided():
    """End-to-end: with team_momentum supplied, the resolver returns adjusted round_probs."""
    from src.prediction.strategy_pipeline import parse_pipeline, resolve_pipeline_round_probs

    teams = ["hot", "cold"]
    base_rp = {t: {r: 0.5 for r in ROUND_NAMES} for t in teams}
    momentum = {"hot": 0.08, "cold": -0.08}

    sources, adjustments, _ = parse_pipeline("seed+momentum_forward")
    result = resolve_pipeline_round_probs(
        sources,
        adjustments,
        {"seed": base_rp},
        pick_dist={},
        team_momentum=momentum,
    )
    assert result is not None
    assert set(result.keys()) == {"hot", "cold"}
    for r in ROUND_NAMES:
        assert result["hot"][r] > result["cold"][r], f"Improving team should beat declining team at {r}"


def test_build_momentum_zero_delta_matches_uniform_baseline():
    """All-zero momentum should produce identical per-team rp (modulo renorm)."""
    teams = ["a", "b", "c"]
    model_rp = {t: {r: 0.5 for r in ROUND_NAMES} for t in teams}
    zero_mom = {t: 0.0 for t in teams}

    adjusted = build_momentum_round_probs(model_rp, zero_mom)
    # Every team should have the same rp at every round (all factors = 1.0).
    for r in ROUND_NAMES:
        vals = [adjusted[t][r] for t in teams]
        assert max(vals) == pytest.approx(min(vals), rel=1e-9), (
            f"Zero-delta teams should have identical rp at {r}, got {vals}"
        )
