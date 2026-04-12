"""Unit tests for src/optimization/bracket_construction.py.

Each construction mode is exercised with a synthetic 64-team fixture that
has monotonically-decreasing probabilities by seed. At risk=0.0 (pure chalk),
all four modes should converge to the same bracket. At higher risk levels,
we verify the forced_champion path works for all four modes.
"""

from __future__ import annotations

import pytest

from src.optimization.bracket_construction import (
    CONSTRUCTION_MODES,
    construct_bracket,
)

_REGIONS = ("East", "West", "South", "Midwest")


def _synthetic_fixture():
    """Build a 64-team synthetic fixture with chalk-dominant probabilities.

    Team IDs follow the pattern ``{region_lower}_{seed}`` so tests can
    predict the chalk outcome (1-seeds win everything).

    Round probabilities are generated from the seed line using a simple
    ``1 / (seed + 0.5)`` decay so that 1-seeds dominate every round.
    Public picks are identical to model probs (so leverage = 1.0 for
    every team, isolating the construction-order signal from any
    public-pick asymmetry).
    """
    seeds = {}
    regions = {}
    round_probs = {}

    rounds = ("R64", "R32", "S16", "E8", "F4", "CHAMP")

    for region in _REGIONS:
        for seed in range(1, 17):
            tid = f"{region.lower()}_{seed}"
            seeds[tid] = seed
            regions[tid] = region
            # Monotonic decay: 1-seed gets 1/1.5, 16-seed gets 1/16.5
            base = 1.0 / (seed + 0.5)
            round_probs[tid] = {
                "R64": min(0.99, base * 1.6),
                "R32": base * 0.9,
                "S16": base * 0.5,
                "E8": base * 0.3,
                "F4": base * 0.15,
                "CHAMP": base * 0.06,
            }

    public_picks = {tid: dict(rp) for tid, rp in round_probs.items()}
    return seeds, regions, round_probs, public_picks


def _assert_complete_bracket(picks, final_four, champion):
    """Verify the picks dict has all 63 expected keys and is internally consistent."""
    round_counts = {}
    for key in picks:
        rnd = key.split("_", 1)[0]
        round_counts[rnd] = round_counts.get(rnd, 0) + 1
    assert round_counts == {"R64": 32, "R32": 16, "S16": 8, "E8": 4, "F4": 2, "CHAMP": 1}, (
        f"wrong round counts: {round_counts}"
    )
    assert len(picks) == 63
    assert len(final_four) == 4
    assert len(set(final_four)) == 4, "final_four has duplicates"
    assert champion in final_four, f"champion {champion} not in F4 {final_four}"


@pytest.mark.parametrize("mode", CONSTRUCTION_MODES)
def test_mode_produces_complete_bracket(mode):
    seeds, regions, round_probs, public_picks = _synthetic_fixture()
    picks, champion, final_four, exp, var = construct_bracket(
        mode=mode,
        seeds=seeds,
        regions=regions,
        round_probs=round_probs,
        public_picks=public_picks,
        risk_level=0.0,
        pool_size=100,
    )
    _assert_complete_bracket(picks, final_four, champion)
    assert exp > 0, f"expected_points={exp} should be positive"
    assert var > 0, f"variance={var} should be positive"


@pytest.mark.parametrize("mode", CONSTRUCTION_MODES)
def test_chalk_picks_one_seed_champion(mode):
    """At risk=0.0 with monotonic-by-seed probabilities, the champion must be a 1-seed."""
    seeds, regions, round_probs, public_picks = _synthetic_fixture()
    _, champion, _, _, _ = construct_bracket(
        mode=mode,
        seeds=seeds,
        regions=regions,
        round_probs=round_probs,
        public_picks=public_picks,
        risk_level=0.0,
        pool_size=100,
    )
    assert seeds[champion] == 1, f"{mode} chalk champion {champion} has seed {seeds[champion]}, expected 1"


@pytest.mark.parametrize("mode", CONSTRUCTION_MODES)
def test_chalk_final_four_respects_one_seed_cap(mode):
    """At risk=0.0 with default max_one_seeds_f4=2, f4_first caps one-seeds at 2.

    Other modes (forward_greedy, champ_first, e8_first) don't apply the
    cap so they still produce 4 one-seeds with monotonic-by-seed probs.
    """
    seeds, regions, round_probs, public_picks = _synthetic_fixture()
    _, _, final_four, _, _ = construct_bracket(
        mode=mode,
        seeds=seeds,
        regions=regions,
        round_probs=round_probs,
        public_picks=public_picks,
        risk_level=0.0,
        pool_size=100,
    )
    f4_seeds = sorted(seeds[t] for t in final_four)
    one_seed_count = sum(1 for s in f4_seeds if s == 1)
    if mode == "f4_first":
        assert one_seed_count <= 2, f"f4_first chalk F4 has {one_seed_count} one-seeds, expected <= 2"
    else:
        assert f4_seeds == [1, 1, 1, 1], f"{mode} chalk F4 seeds are {f4_seeds}, expected [1,1,1,1]"


@pytest.mark.parametrize("mode", CONSTRUCTION_MODES)
def test_uncapped_final_four_is_four_one_seeds(mode):
    """With max_one_seeds_f4=4 (no cap), all modes produce 4 one-seeds at chalk."""
    seeds, regions, round_probs, public_picks = _synthetic_fixture()
    _, _, final_four, _, _ = construct_bracket(
        mode=mode,
        seeds=seeds,
        regions=regions,
        round_probs=round_probs,
        public_picks=public_picks,
        risk_level=0.0,
        pool_size=100,
        max_one_seeds_f4=4,
    )
    f4_seeds = sorted(seeds[t] for t in final_four)
    assert f4_seeds == [1, 1, 1, 1], f"{mode} uncapped chalk F4 seeds are {f4_seeds}, expected [1,1,1,1]"


@pytest.mark.parametrize("mode", CONSTRUCTION_MODES)
@pytest.mark.parametrize("forced_seed", [2, 3, 4, 5])
def test_forced_champion_honored(mode, forced_seed):
    """Forcing a non-chalk champion should produce that exact champion in every mode."""
    seeds, regions, round_probs, public_picks = _synthetic_fixture()
    forced = f"east_{forced_seed}"
    _, champion, final_four, _, _ = construct_bracket(
        mode=mode,
        seeds=seeds,
        regions=regions,
        round_probs=round_probs,
        public_picks=public_picks,
        risk_level=0.4,
        pool_size=100,
        forced_champion=forced,
    )
    assert champion == forced, f"{mode} forced_champion={forced} produced champion={champion}"
    assert forced in final_four, f"{mode} forced champion {forced} not in F4 {final_four}"


def test_unknown_mode_raises():
    seeds, regions, round_probs, public_picks = _synthetic_fixture()
    with pytest.raises(ValueError, match="Unknown construction mode"):
        construct_bracket(
            mode="bogus",
            seeds=seeds,
            regions=regions,
            round_probs=round_probs,
            public_picks=public_picks,
            risk_level=0.0,
        )


def test_forced_champion_not_in_seeds_raises():
    seeds, regions, round_probs, public_picks = _synthetic_fixture()
    with pytest.raises(ValueError, match="not in seeds"):
        construct_bracket(
            mode="champ_first",
            seeds=seeds,
            regions=regions,
            round_probs=round_probs,
            public_picks=public_picks,
            risk_level=0.0,
            forced_champion="nonexistent_team",
        )


def test_forward_greedy_with_forced_champion_falls_back():
    """forward_greedy mode should transparently delegate to champ_first when
    a forced_champion is provided (see the design doc decision)."""
    seeds, regions, round_probs, public_picks = _synthetic_fixture()
    forced = "south_3"
    _, champion, _, _, _ = construct_bracket(
        mode="forward_greedy",
        seeds=seeds,
        regions=regions,
        round_probs=round_probs,
        public_picks=public_picks,
        risk_level=0.5,
        forced_champion=forced,
    )
    assert champion == forced


def test_missing_region_seed_raises():
    """If the seed map is incomplete after normalization, construction must fail loudly."""
    seeds, regions, round_probs, public_picks = _synthetic_fixture()
    # Drop the east_16 team from seeds so East is missing seed 16
    del seeds["east_16"]
    del regions["east_16"]
    del round_probs["east_16"]
    del public_picks["east_16"]
    with pytest.raises(ValueError, match="missing seed"):
        construct_bracket(
            mode="champ_first",
            seeds=seeds,
            regions=regions,
            round_probs=round_probs,
            public_picks=public_picks,
            risk_level=0.0,
        )


@pytest.mark.parametrize("mode", CONSTRUCTION_MODES)
def test_chalk_convergence_across_modes(mode):
    """All modes should produce the same bracket at risk=0.0 given
    strictly-chalk-dominant probabilities, EXCEPT f4_first which diverges
    due to the one-seed cap (max_one_seeds_f4=2 default).

    With cap disabled (max_one_seeds_f4=4), all modes converge."""
    seeds, regions, round_probs, public_picks = _synthetic_fixture()
    # Compute the forward_greedy chalk bracket as the canonical reference.
    ref_picks, ref_champion, _, _, _ = construct_bracket(
        mode="forward_greedy",
        seeds=seeds,
        regions=regions,
        round_probs=round_probs,
        public_picks=public_picks,
        risk_level=0.0,
        pool_size=100,
    )
    # Use uncapped so convergence holds across all modes
    picks, champion, _, _, _ = construct_bracket(
        mode=mode,
        seeds=seeds,
        regions=regions,
        round_probs=round_probs,
        public_picks=public_picks,
        risk_level=0.0,
        pool_size=100,
        max_one_seeds_f4=4,
    )
    assert champion == ref_champion, f"{mode} chalk champion {champion} != forward_greedy {ref_champion}"
    # Picks should match — all modes converge to same chalk bracket
    assert picks == ref_picks, f"{mode} chalk picks differ from forward_greedy"
