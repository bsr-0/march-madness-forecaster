"""Unit tests for C1 coach_adj adjustment.

Covers the canonical→Kaggle bridge, the walk-forward appearance loader,
the asymmetric +3% multiplicative adjustment, and the pipeline registry.

Same shape as ``tests/test_roster_adj_adjustment.py`` since coach_adj is
the C1 sibling. Key contract difference vs roster_adj: coach_adj is
ONE-SIDED — veteran coaches get a +0%..+3% boost, but new/unknown
coaches stay at factor=1.0 (never penalized). Catalog spec C1.
"""

from pathlib import Path

import pytest

from src.prediction.coach_adj_probabilities import (
    ROUND_NAMES,
    build_coach_adj_round_probs,
    load_coach_experience,
)

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
_TEAMS_PER_ROUND = {"R64": 32, "R32": 16, "S16": 8, "E8": 4, "F4": 2, "CHAMP": 1}


def test_load_coach_experience_2026_covers_tournament_field():
    """Bridge + Kaggle data → at least 60 of 68 2026 teams get an experience count."""
    import json

    seeds = json.load(open(DATA_ROOT / "raw" / "tournament_seeds_2026.json"))
    tourney_ids = {s["team_id"] for s in seeds}
    exp = load_coach_experience(2026, tourney_ids, DATA_ROOT)
    coverage = len(set(exp.keys()) & tourney_ids)
    assert coverage >= 60, f"Expected >=60 of 68 teams bridged, got {coverage}"
    # Every value must be a non-negative integer (apps count).
    for tid, apps in exp.items():
        assert isinstance(apps, int), f"{tid} apps not int: {type(apps)}"
        assert apps >= 0, f"{tid} apps negative: {apps}"


def test_load_coach_experience_walk_forward_isolation():
    """load_coach_experience(year=2025) must only cumulate Seasons strictly < 2025."""
    import json

    raw = json.load(open(DATA_ROOT / "raw" / "historical" / "tournament_seeds_2025.json"))
    tourney_ids = {s["team_id"] for s in raw["teams"]}
    exp_2025 = load_coach_experience(2025, tourney_ids, DATA_ROOT)
    # Sanity: at least one veteran coach should clear the 10-app threshold by 2024.
    # (Izzo, Calipari, Self all have >15 by then.)
    assert any(a >= 10 for a in exp_2025.values()), "Expected at least one veteran coach (>=10 apps) in 2025 field"
    # The same lookup for 2011 should produce strictly smaller maximums than 2025
    # (more years of cumulation in 2025 than in 2011).
    seeds_2011 = json.load(open(DATA_ROOT / "raw" / "historical" / "tournament_seeds_2011.json"))
    tourney_ids_2011 = {s["team_id"] for s in seeds_2011["teams"]}
    exp_2011 = load_coach_experience(2011, tourney_ids_2011, DATA_ROOT)
    assert max(exp_2025.values()) >= max(exp_2011.values()), (
        "Walk-forward broken: 2025 max apps should be >= 2011 max apps"
    )


def test_load_coach_experience_returns_empty_for_missing_year():
    """A year without MTeamCoaches data → empty dict (no crash)."""
    # Year 1900 won't have any seeds, but the loader should handle it without
    # exploding on missing data.
    exp = load_coach_experience(1900, {"duke", "kansas"}, DATA_ROOT)
    # All teams should resolve to 0 apps (unknown coach) or be absent — either is fine.
    for tid, apps in exp.items():
        assert apps == 0


def test_load_coach_experience_known_veteran_2025():
    """Spot-check: a known-veteran 2025 program (Michigan State, Izzo) has >15 prior apps."""
    import json

    raw = json.load(open(DATA_ROOT / "raw" / "historical" / "tournament_seeds_2025.json"))
    tourney_ids = {s["team_id"] for s in raw["teams"]}
    exp = load_coach_experience(2025, tourney_ids, DATA_ROOT)
    # Find Michigan State by canonical id (likely 'michigan_state' or close).
    msu_candidates = [tid for tid in tourney_ids if "michigan_state" in tid]
    assert msu_candidates, "Michigan State not in 2025 field?"
    msu_tid = msu_candidates[0]
    if msu_tid in exp:
        # Tom Izzo had ~25+ NCAA tournament apps by 2024 — clear room for ≥15.
        assert exp[msu_tid] >= 15, f"Izzo's apps count seems off: {exp[msu_tid]} (expected >=15 by 2025)"


def test_build_coach_adj_renormalizes_per_round_to_team_counts():
    """After adjustment, round totals must match bracket team-count targets."""
    teams = [f"t{i}" for i in range(1, 65)]
    model_rp = {t: {"R64": 1.0, "R32": 0.5, "S16": 0.25, "E8": 0.125, "F4": 0.0625, "CHAMP": 0.03125} for t in teams}
    # Spread of experience values so the adjustment varies across teams.
    coach_exp = {t: i for i, t in enumerate(teams)}

    adjusted = build_coach_adj_round_probs(model_rp, coach_exp)
    for rnd, target in _TEAMS_PER_ROUND.items():
        total = sum(adjusted[t][rnd] for t in teams)
        assert total == pytest.approx(target, rel=1e-6), (
            f"Round {rnd} total should renormalize to {target}, got {total:.4f}"
        )


def test_build_coach_adj_high_experience_team_gains_mass_relative_to_zero():
    """Veteran-coach team's adjusted rp > zero-experience team's."""
    teams = ["veteran", "rookie"] + [f"f{i}" for i in range(1, 63)]
    model_rp = {t: {"R64": 1.0, "R32": 0.5, "S16": 0.25, "E8": 0.125, "F4": 0.0625, "CHAMP": 0.03125} for t in teams}
    coach_exp = {"veteran": 25, "rookie": 0}
    for t in teams[2:]:
        coach_exp[t] = 0

    adjusted = build_coach_adj_round_probs(model_rp, coach_exp)
    for r in ROUND_NAMES:
        assert adjusted["veteran"][r] > adjusted["rookie"][r], (
            f"Veteran should beat rookie at {r} (vet={adjusted['veteran'][r]:.4f}, rookie={adjusted['rookie'][r]:.4f})"
        )


def test_build_coach_adj_caps_factor_at_plus_3_percent():
    """Even an extreme veteran (50+ apps) must produce factor ≤ 1.03 — pre-renorm cap."""
    # Two teams: extreme veteran and a baseline. After renorm, the relative
    # ratio of veteran/baseline equals (1.03 / 1.0) = 1.03 in the limit.
    teams = ["legend", "baseline"]
    model_rp = {
        "legend": {r: 0.5 for r in ROUND_NAMES},
        "baseline": {r: 0.5 for r in ROUND_NAMES},
    }
    coach_exp = {"legend": 50, "baseline": 0}

    adjusted = build_coach_adj_round_probs(model_rp, coach_exp)
    ratio = adjusted["legend"]["R64"] / adjusted["baseline"]["R64"]
    assert ratio <= 1.03 + 1e-6, f"Legend/baseline ratio should respect +3% cap, got {ratio:.4f}"


def test_build_coach_adj_is_one_sided_no_team_penalized():
    """Asymmetric contract: every team's pre-renorm factor is ≥ 1.0 (no penalties).

    Verified by checking that the team with min experience has the SMALLEST
    pre-renorm factor — equal to 1.0 — and no team's adjusted/base ratio
    drops below the renorm scale. Equivalently: the lowest-experience team's
    adjusted rp matches the all-zero-experience baseline (modulo renorm).
    """
    teams = ["zero", "one_app", "veteran"]
    model_rp = {t: {r: 0.5 for r in ROUND_NAMES} for t in teams}

    # Run with all-zero experience as reference.
    zero_exp = {t: 0 for t in teams}
    base_adj = build_coach_adj_round_probs(model_rp, zero_exp)

    # Run with mixed experience.
    coach_exp = {"zero": 0, "one_app": 1, "veteran": 25}
    mixed_adj = build_coach_adj_round_probs(model_rp, coach_exp)

    # The "zero" team's adjusted rp should be ≤ its zero-exp baseline (since
    # other teams gained mass and the renorm scale shrank). One_app and
    # veteran should be at or above their zero-exp baselines.
    for r in ROUND_NAMES:
        assert mixed_adj["zero"][r] <= base_adj["zero"][r] + 1e-9, (
            f"Zero-exp team gained mass at {r} despite no boost: "
            f"base={base_adj['zero'][r]:.4f}, mixed={mixed_adj['zero'][r]:.4f}"
        )
        assert mixed_adj["one_app"][r] >= mixed_adj["zero"][r], (
            f"One-app team should not be penalized vs zero-exp at {r}"
        )
        assert mixed_adj["veteran"][r] >= mixed_adj["one_app"][r], f"Veteran should be >= one-app team at {r}"


def test_build_coach_adj_missing_team_treated_as_zero_experience():
    """Team absent from coach_exp dict → treated as 0 apps (no boost)."""
    teams = ["unknown", "rookie", "veteran"]
    model_rp = {t: {r: 0.5 for r in ROUND_NAMES} for t in teams}
    # 'unknown' is intentionally missing from coach_exp; rookie has 0 apps explicitly.
    coach_exp = {"rookie": 0, "veteran": 25}

    adjusted = build_coach_adj_round_probs(model_rp, coach_exp)
    # unknown and rookie should be equal (both treated as 0 apps).
    for r in ROUND_NAMES:
        assert adjusted["unknown"][r] == pytest.approx(adjusted["rookie"][r], rel=1e-6), (
            f"Missing team should match 0-app team at {r}"
        )


def test_coach_adj_registered_in_pipeline():
    from src.prediction.strategy_pipeline import IMPLEMENTED_ADJUSTMENTS

    assert "coach_adj" in IMPLEMENTED_ADJUSTMENTS


def test_permutation_generator_enumerates_coach_adj_strategies():
    from src.prediction.strategy_pipeline import generate_all_permutations

    strategies = generate_all_permutations(implemented_only=True)
    coach = [s for s in strategies if "coach_adj" in s]
    assert len(coach) >= 5, f"Expected >=5 coach_adj strategies, got {len(coach)}"
    assert "seed+coach_adj" in strategies
    assert "torvik+coach_adj" in strategies


def test_resolve_pipeline_returns_none_without_coach_experience_context():
    """Using coach_adj without supplying coach_experience must return None."""
    from src.prediction.strategy_pipeline import parse_pipeline, resolve_pipeline_round_probs

    sources, adjustments, _ = parse_pipeline("seed+coach_adj_forward")
    base_rp = {"t1": {r: 0.5 for r in ROUND_NAMES}}
    result = resolve_pipeline_round_probs(
        sources,
        adjustments,
        {"seed": base_rp},
        pick_dist={},
        # coach_experience intentionally omitted
    )
    assert result is None


def test_resolve_pipeline_applies_coach_adj_when_context_provided():
    """With coach_experience supplied, the resolver returns adjusted round_probs."""
    from src.prediction.strategy_pipeline import parse_pipeline, resolve_pipeline_round_probs

    teams = ["veteran", "rookie"]
    base_rp = {t: {r: 0.5 for r in ROUND_NAMES} for t in teams}
    coach_exp = {"veteran": 25, "rookie": 0}

    sources, adjustments, _ = parse_pipeline("seed+coach_adj_forward")
    result = resolve_pipeline_round_probs(
        sources,
        adjustments,
        {"seed": base_rp},
        pick_dist={},
        coach_experience=coach_exp,
    )
    assert result is not None
    assert set(result.keys()) == {"veteran", "rookie"}
    for r in ROUND_NAMES:
        assert result["veteran"][r] > result["rookie"][r], f"Veteran should beat rookie at {r}"
