"""Tests for MC-derived round probabilities in PoolOptimizer and unified EV-edge leverage."""

import warnings

import pytest
import numpy as np

from src.optimization.pool_optimizer import PoolEnvironment, PoolOptimizer
from src.optimization.leverage import compute_ev_edge, LeveragePick


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TEAMS = ["duke", "unc", "kansas", "gonzaga"]

SCORING_RULES = {
    "R64": 10, "R32": 20, "S16": 40, "E8": 80, "F4": 160, "CHAMP": 320,
}


def _make_pairwise_probs(teams):
    """Generate synthetic pairwise probabilities for test teams."""
    probs = {}
    strengths = {t: 0.5 + 0.1 * i for i, t in enumerate(teams)}
    for i, t1 in enumerate(teams):
        for j, t2 in enumerate(teams):
            if i < j:
                s1, s2 = strengths[t1], strengths[t2]
                p = s1 / (s1 + s2)
                probs[(t1, t2)] = p
    return probs


def _make_mc_round_probs(teams):
    """Simulate realistic MC-derived round advancement probabilities."""
    return {
        "duke":    {"R64": 1.0, "R32": 0.82, "S16": 0.54, "E8": 0.28, "F4": 0.11, "CHAMP": 0.04},
        "unc":     {"R64": 1.0, "R32": 0.88, "S16": 0.61, "E8": 0.35, "F4": 0.16, "CHAMP": 0.07},
        "kansas":  {"R64": 1.0, "R32": 0.91, "S16": 0.68, "E8": 0.42, "F4": 0.22, "CHAMP": 0.11},
        "gonzaga": {"R64": 1.0, "R32": 0.94, "S16": 0.74, "E8": 0.50, "F4": 0.29, "CHAMP": 0.15},
    }


def _make_public_picks(teams):
    """Synthetic public pick distribution."""
    return {
        "duke":    {"R64": 0.95, "R32": 0.70, "S16": 0.40, "E8": 0.20, "F4": 0.08, "CHAMP": 0.03},
        "unc":     {"R64": 0.96, "R32": 0.80, "S16": 0.55, "E8": 0.30, "F4": 0.12, "CHAMP": 0.05},
        "kansas":  {"R64": 0.97, "R32": 0.85, "S16": 0.60, "E8": 0.35, "F4": 0.18, "CHAMP": 0.09},
        "gonzaga": {"R64": 0.98, "R32": 0.90, "S16": 0.70, "E8": 0.45, "F4": 0.25, "CHAMP": 0.13},
    }


def _make_environment(teams):
    return PoolEnvironment(
        pool_size=500,
        scoring_rules=SCORING_RULES,
        payout_structure="top_10pct",
        public_pick_distribution=_make_public_picks(teams),
    )


# ---------------------------------------------------------------------------
# Tests: MC round probs injection
# ---------------------------------------------------------------------------


class TestMCRoundProbsInjection:
    """PoolOptimizer uses MC-derived round probs when provided."""

    def test_mc_probs_bypass_heuristic(self):
        """When model_round_probs is passed, no deprecation warning fires."""
        probs = _make_pairwise_probs(TEAMS)
        mc_probs = _make_mc_round_probs(TEAMS)
        env = _make_environment(TEAMS)

        optimizer = PoolOptimizer(probs, env, model_round_probs=mc_probs)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = optimizer.optimize()
            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) == 0, "MC probs should bypass the heuristic path"

        assert result.manifest is not None
        assert result.manifest.pool_size == 500

    def test_heuristic_fallback_emits_warning(self):
        """Without model_round_probs, the heuristic fires a DeprecationWarning."""
        probs = _make_pairwise_probs(TEAMS)
        env = _make_environment(TEAMS)

        optimizer = PoolOptimizer(probs, env)  # No MC probs

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = optimizer.optimize()
            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) >= 1, "Heuristic fallback should warn"

        assert result.manifest is not None

    def test_mc_probs_are_deep_copied(self):
        """Mutating the original dict after construction has no effect."""
        probs = _make_pairwise_probs(TEAMS)
        mc_probs = _make_mc_round_probs(TEAMS)
        env = _make_environment(TEAMS)

        optimizer = PoolOptimizer(probs, env, model_round_probs=mc_probs)

        # Mutate the original
        mc_probs["duke"]["CHAMP"] = 0.99

        # Optimizer's internal copy should be unchanged
        assert optimizer._mc_round_probs["duke"]["CHAMP"] == pytest.approx(0.04)

    def test_sensitivity_analysis_propagates_mc_probs(self):
        """Sensitivity analysis passes MC probs to shifted sub-optimizers."""
        probs = _make_pairwise_probs(TEAMS)
        mc_probs = _make_mc_round_probs(TEAMS)
        env = _make_environment(TEAMS)

        optimizer = PoolOptimizer(probs, env, model_round_probs=mc_probs)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            report = optimizer.sensitivity_analysis(pick_shift_pct=0.05)
            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) == 0, (
                "Shifted optimizers should inherit MC probs, not fall back to heuristic"
            )

        assert report.flag in ("STABLE", "HIGH_STRATEGY_UNCERTAINTY")


# ---------------------------------------------------------------------------
# Tests: Unified EV-edge leverage
# ---------------------------------------------------------------------------


class TestComputeEvEdge:
    """Canonical compute_ev_edge function."""

    def test_positive_edge(self):
        """Model sees team as under-picked → positive EV edge."""
        edge = compute_ev_edge(model_prob=0.30, public_pct=0.10, round_points=160)
        assert edge == pytest.approx(0.20 * 160)

    def test_negative_edge(self):
        """Model sees team as over-picked → negative EV edge."""
        edge = compute_ev_edge(model_prob=0.10, public_pct=0.30, round_points=160)
        assert edge == pytest.approx(-0.20 * 160)

    def test_zero_edge(self):
        """Equal model and public → zero edge."""
        edge = compute_ev_edge(model_prob=0.50, public_pct=0.50, round_points=320)
        assert edge == pytest.approx(0.0)

    def test_round_weighting(self):
        """Same probability gap produces higher edge in later rounds."""
        gap = 0.10
        edge_r64 = compute_ev_edge(0.50, 0.40, 10)
        edge_champ = compute_ev_edge(0.50, 0.40, 320)
        assert edge_champ > edge_r64
        assert edge_r64 == pytest.approx(gap * 10)
        assert edge_champ == pytest.approx(gap * 320)


class TestLeveragePickEvEdge:
    """LeveragePick.expected_value_differential matches compute_ev_edge."""

    def test_ev_differential_matches_canonical(self):
        pick = LeveragePick(
            team_id="duke",
            team_name="Duke",
            seed=1,
            region="East",
            model_probability=0.30,
            public_pick_percentage=0.10,
            round_name="F4",
            points_value=160,
        )
        canonical = compute_ev_edge(0.30, 0.10, 160)
        assert pick.expected_value_differential == pytest.approx(canonical)

    def test_leverage_ratio_still_available(self):
        """leverage_ratio property still works for display purposes."""
        pick = LeveragePick(
            team_id="unc",
            team_name="UNC",
            seed=2,
            region="South",
            model_probability=0.20,
            public_pick_percentage=0.05,
            round_name="E8",
            points_value=80,
        )
        assert pick.leverage_ratio == pytest.approx(4.0)
