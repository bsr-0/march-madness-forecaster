"""What `chalk_noise_std` actually does, pinned (audit H3 / recommendation 13).

For about a year `generate_opponent_brackets` documented `chalk_noise_std` as
creating "the correlation structure observed in real pools", with a recommended
range of "0.3-0.6 for realistic N=31 pool correlation". Measured on 2026-09-13,
both halves of that are wrong: within-field bracket agreement is flat across the
whole range, and the real pool is *less* correlated than independent draws, so
the recommended direction was backwards as well.

Nothing caught it because nothing tested the parameter's effect -- the two
existing tests that touch it (`test_pool_behavioral_blend.py`,
`test_selection_common_random_numbers.py`) check that a float comes back and
that a kwarg is forwarded. Plumbing, not behaviour. These tests pin the
behaviour, so a future reader who trusts the name rather than the measurement
gets a failure rather than a quiet wrong model.
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from scripts.pool_opponent_realism import _mean_pairwise_agreement  # noqa: E402


def _toy_field(n_opponents: int, chalk: float, seed: int):
    """A synthetic 64-team bracket with a wide seed spread, so the seed-gap
    weighting that gates the chalk shift is actually exercised."""
    from src.simulation.pool_competition import generate_opponent_brackets

    teams = [f"t{i}" for i in range(64)]
    # Seeds 1..16 repeated per region, matching the real first_round layout:
    # pairs are (1,16), (8,9), (5,12), ... so adjacent entries are a matchup.
    pair_seeds = [(1, 16), (8, 9), (5, 12), (4, 13), (6, 11), (3, 14), (7, 10), (2, 15)]
    seeds = {}
    idx = 0
    for _region in range(4):
        for hi, lo in pair_seeds:
            seeds[teams[idx]] = hi
            seeds[teams[idx + 1]] = lo
            idx += 2
    # Pick distribution: higher seeds picked more often, so there is real
    # consensus for the correlation to be measured against.
    pick_dist = {}
    for t, s in seeds.items():
        base = max(0.02, min(0.98, 1.0 - (s - 1) / 17.0))
        pick_dist[t] = {r: base ** (i + 1) for i, r in enumerate(["R64", "R32", "S16", "E8", "F4", "CHAMP"])}
    matchup_probs = {}
    return generate_opponent_brackets(
        n_opponents=n_opponents,
        first_round_matchups=teams,
        matchup_probs=matchup_probs,
        pick_distribution=pick_dist,
        seeds=seeds,
        rng=np.random.default_rng(seed),
        chalk_noise_std=chalk,
    )


def test_chalk_noise_std_does_not_create_within_field_correlation():
    """The claim the old docstring made, and the measurement that refutes it.

    If `chalk_noise_std` clustered opponents, mean pairwise agreement would rise
    with it. It does not: across 0.0 -> 0.6 the change is far smaller than the
    repeat-to-repeat spread, and if anything it goes down.
    """
    reps = 60
    means = {}
    for chalk in (0.0, 0.6):
        vals = [_mean_pairwise_agreement(_toy_field(30, chalk, 4000 + r)) for r in range(reps)]
        means[chalk] = (float(np.mean(vals)), float(np.std(vals, ddof=1)))

    low_mean, low_sd = means[0.0]
    high_mean, _ = means[0.6]
    # The effect must be small relative to noise -- emphatically not the
    # "realistic pool correlation" the docstring promised.
    assert abs(high_mean - low_mean) < low_sd, (
        f"chalk_noise_std moved within-field agreement from {low_mean:.5f} to {high_mean:.5f}, "
        f"more than one repeat SD ({low_sd:.5f}). If this is now a real clustering effect, the "
        "docstring measurement in pool_competition.generate_opponent_brackets is stale and "
        "scripts/pool_opponent_realism.py should be re-run."
    )
    assert high_mean <= low_mean + 1e-9, (
        "raising chalk_noise_std increased agreement; the measured behaviour was the opposite "
        "(the per-opponent spread term makes brackets slightly MORE different)"
    )


def test_chalk_noise_std_does_control_between_pool_chalkiness():
    """What the parameter genuinely does, so the fix is not just a deletion.

    It is a real knob -- it models uncertainty about whether this year's field
    will be chalky or contrarian. That shows up ACROSS simulated pools, not
    within one.
    """
    reps = 60
    spread = {}
    for chalk in (0.0, 0.6):
        chalkiness = [float(_toy_field(30, chalk, 5000 + r).mean()) for r in range(reps)]
        spread[chalk] = float(np.std(chalkiness, ddof=1))

    assert spread[0.6] > 1.5 * spread[0.0], (
        f"between-pool chalkiness SD was {spread[0.0]:.5f} at chalk=0.0 and {spread[0.6]:.5f} at "
        "0.6; the parameter is supposed to widen this even though it does not cluster brackets"
    )


def test_production_paths_use_independent_opponents():
    """Every live path draws opponents independently, and that is now a choice.

    The real 2023-2026 fields are LESS correlated than independent draws
    (pooled Stouffer z = -2.87), so 0.0 is defensible rather than merely
    inherited. This test exists so that turning it on becomes a deliberate,
    visible act.
    """
    import inspect

    from src.simulation.pool_competition import generate_opponent_brackets

    default = inspect.signature(generate_opponent_brackets).parameters["chalk_noise_std"].default
    assert default == 0.0, (
        "generate_opponent_brackets' chalk_noise_std default changed. The real pool is less "
        "correlated than independent draws, so a non-zero default moves the simulation away "
        "from reality -- see scripts/pool_opponent_realism.py and audit finding H3."
    )


def test_mean_pairwise_agreement_matches_a_brute_force_count():
    """The closed form used by the measurement, checked against the naive loop."""
    rng = np.random.default_rng(11)
    m = rng.random((9, 63)) < 0.6

    n = m.shape[0]
    total = 0.0
    pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += float((m[i] == m[j]).mean())
            pairs += 1

    assert _mean_pairwise_agreement(m) == pytest.approx(total / pairs)


def test_identical_brackets_agree_completely_and_opposites_not_at_all():
    """Endpoints, so a sign error in the closed form cannot pass."""
    ones = np.ones((4, 63), dtype=bool)
    assert _mean_pairwise_agreement(ones) == pytest.approx(1.0)

    half = np.zeros((2, 63), dtype=bool)
    half[1] = True
    assert _mean_pairwise_agreement(half) == pytest.approx(0.0)


def test_agreement_needs_at_least_two_brackets():
    with pytest.raises(ValueError, match="at least 2 brackets"):
        _mean_pairwise_agreement(np.ones((1, 63), dtype=bool))
