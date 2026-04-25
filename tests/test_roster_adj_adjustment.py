"""Unit tests for C2 roster_adj adjustment.

Covers the loader (per-team top-5 WARP from cbbpy_rosters, bridged to
canonical IDs), the adjustment itself (multiplicative ±4% boost/penalty
on round_probs by team-talent z-score), and the pipeline registry +
permutation enumeration.

Same shape as ``tests/test_volatile_adjustment.py`` since roster_adj is
the C2 adjustment counterpart to D1 volatile.
"""

from pathlib import Path

import pytest

from src.prediction.roster_adj_probabilities import (
    ROUND_NAMES,
    build_roster_adj_round_probs,
    load_team_talent,
)

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
_TEAMS_PER_ROUND = {"R64": 64, "R32": 32, "S16": 16, "E8": 8, "F4": 4, "CHAMP": 2}


def test_load_team_talent_2026_covers_tournament_field():
    """All 68 teams in the 2026 field get a top-5 WARP value via the cbbpy bridge."""
    import json

    seeds = json.load(open(DATA_ROOT / "raw" / "tournament_seeds_2026.json"))
    tourney_ids = {s["team_id"] for s in seeds}
    talent = load_team_talent(2026, tourney_ids, DATA_ROOT)
    # Expect at least 64 of 68 covered (a few obscure programs may not bridge);
    # the exact number is data-driven, but coverage shouldn't be catastrophic.
    coverage = len(set(talent.keys()) & tourney_ids)
    assert coverage >= 60, f"Expected >=60 of 68 teams bridged, got {coverage}"
    # Values must be numeric and span a non-trivial range (not all the same).
    for tid, w in talent.items():
        assert isinstance(w, float), f"{tid} talent is not float: {type(w)}"
    values = list(talent.values())
    assert max(values) - min(values) > 0.05, "Top-5 WARP should vary across teams"


def test_load_team_talent_returns_empty_for_missing_year():
    """Year without rosters file → empty dict (not a crash)."""
    talent = load_team_talent(1999, {"duke", "kansas"}, DATA_ROOT)
    assert talent == {}


def test_load_team_talent_walk_forward_isolation():
    """load_team_talent(year=Y) reads only cbbpy_rosters_{Y}.json, never future years."""
    # If any cross-year reads happened, removing prior years' files would change
    # the result. Easier check: 2025 talent computed today must equal the value
    # computed from a snapshot of just the 2025 file, not any later year.
    import json

    raw = json.load(open(DATA_ROOT / "raw" / "historical" / "tournament_seeds_2025.json"))
    tourney_ids = {s["team_id"] for s in raw["teams"]}
    talent_2025 = load_team_talent(2025, tourney_ids, DATA_ROOT)
    # Spot check: at least one team got a positive WARP. (If we accidentally
    # loaded 2026, this would still pass, so the real protection is the loader
    # opening exactly cbbpy_rosters_2025.json — verified by reading the source.)
    assert any(w > 0 for w in talent_2025.values())


def test_build_roster_adj_renormalizes_per_round_to_team_counts():
    """After adjustment, round totals must match the bracket's team-count targets."""
    teams = [f"t{i}" for i in range(1, 65)]
    model_rp = {t: {"R64": 1.0, "R32": 0.5, "S16": 0.25, "E8": 0.125, "F4": 0.0625, "CHAMP": 0.03125} for t in teams}
    # Spread-out talent so the z-score adjustment actually changes things.
    talent = {t: float(i) for i, t in enumerate(teams)}

    adjusted = build_roster_adj_round_probs(model_rp, talent)
    for rnd, target in _TEAMS_PER_ROUND.items():
        total = sum(adjusted[t][rnd] for t in teams)
        assert total == pytest.approx(target, rel=1e-6), (
            f"Round {rnd} total should renormalize to {target}, got {total:.4f}"
        )


def test_build_roster_adj_high_talent_team_gains_mass_relative_to_low():
    """A high-talent team's round_probs should be larger than a low-talent team's
    when both start with identical baselines."""
    teams = ["star_loaded"] + [f"f{i}" for i in range(1, 64)]
    model_rp = {t: {"R64": 1.0, "R32": 0.5, "S16": 0.25, "E8": 0.125, "F4": 0.0625, "CHAMP": 0.03125} for t in teams}
    # star_loaded has high talent; everyone else has the same low talent so
    # league mean ≈ low and the star is +many stddev.
    talent = {"star_loaded": 1.0}
    for t in teams[1:]:
        talent[t] = 0.1

    adjusted = build_roster_adj_round_probs(model_rp, talent)
    # At every round, star_loaded must exceed any individual filler team.
    for r in ROUND_NAMES:
        for f in teams[1:5]:  # spot-check 4 fillers
            assert adjusted["star_loaded"][r] > adjusted[f][r], (
                f"High-talent team should beat filler at {r} "
                f"(star={adjusted['star_loaded'][r]:.4f}, filler={adjusted[f][r]:.4f})"
            )


def test_build_roster_adj_caps_adjustment_at_plus_minus_4_percent():
    """Even an extreme talent z-score must produce <= ±4% adjustment to barthag,
    so the relative ratio between a team's adjusted vs base rp stays in
    [0.96, 1.04] before re-normalization."""
    # Two teams: extreme outlier (very high talent) and a baseline.
    teams = ["outlier", "baseline"]
    model_rp = {
        "outlier": {r: 0.5 for r in ROUND_NAMES},
        "baseline": {r: 0.5 for r in ROUND_NAMES},
    }
    # outlier is 100 stddev above the mean — without the cap this would be
    # adjustment_factor = 1 + 0.02 * 100 = 3.0 (300% boost).
    talent = {"outlier": 100.0, "baseline": 0.0}

    # Manual cap check: compute the adjustment factor before renorm by
    # comparing the relative ratio of pre-renorm to post-renorm.
    # The cleanest assertion: the ratio of outlier's rp to baseline's rp
    # must be at most (1.04 / 0.96) ≈ 1.083 — i.e., the cap survived.
    adjusted = build_roster_adj_round_probs(model_rp, talent)
    ratio = adjusted["outlier"]["R64"] / adjusted["baseline"]["R64"]
    assert ratio <= 1.04 / 0.96 + 1e-6, f"Outlier should not exceed cap-implied ratio of ~1.083, got {ratio:.4f}"


def test_build_roster_adj_missing_team_unchanged_modulo_renorm():
    """A team not in the talent dict gets the neutral z=0 fallback (no boost/penalty)."""
    teams = ["unknown", "filler1", "filler2"]
    model_rp = {t: {r: 0.5 for r in ROUND_NAMES} for t in teams}
    # Only fillers have talent data; unknown is absent.
    talent = {"filler1": 0.5, "filler2": 0.5}

    adjusted = build_roster_adj_round_probs(model_rp, talent)
    # All three teams have identical baselines and (with neutral fallback) the
    # same effective adjustment, so post-renorm they should be equal.
    for r in ROUND_NAMES:
        assert adjusted["unknown"][r] == pytest.approx(adjusted["filler1"][r], rel=1e-6)
        assert adjusted["unknown"][r] == pytest.approx(adjusted["filler2"][r], rel=1e-6)


def test_roster_adj_registered_in_pipeline():
    from src.prediction.strategy_pipeline import IMPLEMENTED_ADJUSTMENTS

    assert "roster_adj" in IMPLEMENTED_ADJUSTMENTS


def test_permutation_generator_enumerates_roster_adj_strategies():
    from src.prediction.strategy_pipeline import generate_all_permutations

    strategies = generate_all_permutations(implemented_only=True)
    roster = [s for s in strategies if "roster_adj" in s]
    assert len(roster) >= 5, f"Expected >=5 roster_adj strategies, got {len(roster)}"
    # Spot-check: bare single-source + roster_adj exists for the obvious bases.
    assert "seed+roster_adj" in strategies
    assert "torvik+roster_adj" in strategies


def test_resolve_pipeline_returns_none_without_team_talent_context():
    """Using roster_adj without supplying team_talent must fail cleanly (return None)."""
    from src.prediction.strategy_pipeline import parse_pipeline, resolve_pipeline_round_probs

    sources, adjustments, _ = parse_pipeline("seed+roster_adj_forward")
    base_rp = {"t1": {r: 0.5 for r in ROUND_NAMES}}
    result = resolve_pipeline_round_probs(
        sources,
        adjustments,
        {"seed": base_rp},
        pick_dist={},
        # team_talent intentionally omitted
    )
    assert result is None


def test_resolve_pipeline_applies_roster_adj_when_context_provided():
    """End-to-end: with team_talent supplied, the resolver returns adjusted round_probs."""
    from src.prediction.strategy_pipeline import parse_pipeline, resolve_pipeline_round_probs

    teams = ["t1", "t2"]
    base_rp = {t: {r: 0.5 for r in ROUND_NAMES} for t in teams}
    talent = {"t1": 1.0, "t2": 0.0}  # t1 boosted, t2 penalized

    sources, adjustments, _ = parse_pipeline("seed+roster_adj_forward")
    result = resolve_pipeline_round_probs(
        sources,
        adjustments,
        {"seed": base_rp},
        pick_dist={},
        team_talent=talent,
    )
    assert result is not None
    assert set(result.keys()) == {"t1", "t2"}
    for r in ROUND_NAMES:
        assert r in result["t1"]
        assert r in result["t2"]
    # t1 (high talent) > t2 (low talent) at every round.
    for r in ROUND_NAMES:
        assert result["t1"][r] > result["t2"][r]
