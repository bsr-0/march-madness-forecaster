"""Phase 5 tests: dual_submission seed table completeness + end-to-end leverage integration."""

import pytest

from src.optimization.dual_submission import ChampionBoostStrategy
from src.optimization.leverage import (
    LeverageCalculator,
    TeamMetadata,
    SEED_PICK_RATES,
    get_seed_based_pick_rates,
)
from src.simulation.monte_carlo import AggregatedResults


# ---------------------------------------------------------------------------
# dual_submission: seed tables cover all 16 seeds
# ---------------------------------------------------------------------------


class TestSeedTableCompleteness:

    def test_champ_rates_cover_all_seeds(self):
        """_estimate_championship_probs handles seeds 1-16 without 0.001 fallback."""
        strategy = ChampionBoostStrategy()
        seeds = {f"team_{s}": s for s in range(1, 17)}
        probs = strategy._estimate_championship_probs(seeds)

        for seed in range(1, 17):
            team = f"team_{seed}"
            assert team in probs
            # No seed should have the old 0.001 fallback as its raw rate
            # (before normalization, seed 10 has 0.0005 which is < 0.001)
            assert probs[team] > 0

        # Still normalizes to 1.0
        assert abs(sum(probs.values()) - 1.0) < 1e-6

    def test_champ_rates_monotonic(self):
        """Higher seeds get lower championship rates."""
        strategy = ChampionBoostStrategy()
        seeds = {f"team_{s}": s for s in range(1, 17)}
        probs = strategy._estimate_championship_probs(seeds)

        for s in range(1, 15):
            assert probs[f"team_{s}"] >= probs[f"team_{s + 1}"], (
                f"Seed {s} should have >= prob than seed {s + 1}"
            )

    def test_crowd_prob_covers_all_seeds(self):
        """_estimate_crowd_championship_prob handles seeds 1-16."""
        strategy = ChampionBoostStrategy()
        for seed in range(1, 17):
            prob = strategy._estimate_crowd_championship_prob(seed)
            assert prob > 0
            # Must not be the old 0.001 flat fallback for seeds 9-16
            if seed >= 10:
                assert prob < 0.001, (
                    f"Seed {seed} crowd prob should be < 0.001"
                )

    def test_crowd_prob_monotonic(self):
        """Higher seeds get lower crowd championship estimates."""
        strategy = ChampionBoostStrategy()
        for s in range(1, 15):
            assert strategy._estimate_crowd_championship_prob(s) >= (
                strategy._estimate_crowd_championship_prob(s + 1)
            ), f"Seed {s} should have >= crowd prob than seed {s + 1}"

    def test_unknown_seed_uses_seed8_not_hardcoded(self):
        """Unknown seed falls back to seed-8 rate, not a hardcoded 0.001."""
        strategy = ChampionBoostStrategy()
        prob = strategy._estimate_crowd_championship_prob(99)
        seed_8_prob = strategy._estimate_crowd_championship_prob(8)
        assert prob == seed_8_prob


# ---------------------------------------------------------------------------
# End-to-end integration: data → leverage → ranking
# ---------------------------------------------------------------------------


class TestEndToEndLeveragePipeline:
    """Integration tests simulating the full public-picks-to-leverage path."""

    @pytest.fixture
    def tournament_setup(self):
        """16-team single-region bracket with model probs and partial public data."""
        seeds = {f"team_{s}": s for s in range(1, 17)}
        meta = {
            f"team_{s}": TeamMetadata(
                team_name=f"Team {s}", seed=s, region="East",
            )
            for s in range(1, 17)
        }
        model_probs = {}
        for s in range(1, 17):
            team = f"team_{s}"
            # Model probs roughly track seed (1-seed strongest)
            base = max(0.01, 0.50 - 0.03 * s)
            model_probs[team] = {"CHAMP": base, "F4": min(base * 2, 0.95)}

        # Public data exists for seeds 1-8, missing for 9-16
        public_picks = {}
        for s in range(1, 9):
            team = f"team_{s}"
            rates = get_seed_based_pick_rates(s)
            public_picks[team] = {"CHAMP": rates["CHAMP"], "F4": rates["F4"]}

        return seeds, meta, model_probs, public_picks

    def test_no_flat_fallback(self, tournament_setup):
        """Missing teams get seed-specific priors, not a uniform 0.001."""
        _, meta, model_probs, public_picks = tournament_setup
        calc = LeverageCalculator(
            model_probs=model_probs,
            public_picks=public_picks,
            team_metadata=meta,
        )
        picks = calc.find_leverage_picks(min_leverage=0.0, min_probability=0.0)

        # Seeds 9-16 are missing from public data.  Under the old code,
        # they would ALL get public_pct=0.001 regardless of seed.
        # With the fix, each gets its own seed prior (which varies).
        missing_champ_pcts = {
            pick.seed: pick.public_pick_percentage
            for pick in picks
            if pick.seed >= 9 and pick.round_name == "CHAMP"
        }
        # Verify seed-differentiated priors (not all identical)
        unique_pcts = set(missing_champ_pcts.values())
        assert len(unique_pcts) > 1, (
            f"All missing-data seeds got identical public_pct={unique_pcts} — "
            "likely still using flat fallback"
        )

    def test_missing_teams_use_seed_priors(self, tournament_setup):
        """Seeds 9-16 missing from public data get seed-appropriate priors."""
        _, meta, model_probs, public_picks = tournament_setup
        calc = LeverageCalculator(
            model_probs=model_probs,
            public_picks=public_picks,
            team_metadata=meta,
        )
        picks = calc.find_leverage_picks(min_leverage=0.0, min_probability=0.0)

        for pick in picks:
            seed = pick.seed
            if seed >= 9:
                expected_prior = get_seed_based_pick_rates(seed)
                round_prior = expected_prior.get(pick.round_name, 0.01)
                assert pick.public_pick_percentage == pytest.approx(
                    round_prior, rel=0.01
                ), (
                    f"Seed {seed} {pick.round_name}: expected prior "
                    f"{round_prior}, got {pick.public_pick_percentage}"
                )

    def test_leverage_ranking_sensible(self, tournament_setup):
        """Top leverage picks should be mid-to-low seeds, not data artifacts."""
        _, meta, model_probs, public_picks = tournament_setup
        calc = LeverageCalculator(
            model_probs=model_probs,
            public_picks=public_picks,
            team_metadata=meta,
        )
        picks = calc.find_leverage_picks(min_leverage=1.5, min_probability=0.05)

        if picks:
            # The #1 pick should NOT be a 16-seed with artificial leverage
            top = picks[0]
            assert top.seed < 16, (
                f"Top leverage pick is seed {top.seed} — likely data artifact"
            )

    def test_aggregated_results_leverage_no_infinite(self, tournament_setup):
        """AggregatedResults.get_leverage_picks also caps properly."""
        seeds, _, model_probs, _ = tournament_setup
        champ_odds = {t: p["CHAMP"] for t, p in model_probs.items()}

        result = AggregatedResults(
            championship_odds=champ_odds,
            num_simulations=10000,
        )
        # Only provide public odds for seeds 1-4
        public = {f"team_{s}": get_seed_based_pick_rates(s)["CHAMP"] for s in range(1, 5)}

        picks = result.get_leverage_picks(public, min_leverage=1.0)
        for _, _, _, leverage in picks:
            assert leverage <= 20.0, f"Leverage {leverage:.1f}x exceeds cap"


# ---------------------------------------------------------------------------
# Regression: no remaining 0.001 hardcoded fallbacks in leverage paths
# ---------------------------------------------------------------------------


class TestNoHardcodedFallbacks:
    """Verify the 0.001 fallback pattern is no longer in critical code paths."""

    def test_leverage_calculator_uses_seed_prior(self):
        """Completely empty public_picks → all teams get their seed's prior."""
        model = {
            "alpha": {"CHAMP": 0.25},
            "beta": {"CHAMP": 0.10},
        }
        meta = {
            "alpha": TeamMetadata(team_name="Alpha", seed=2, region="East"),
            "beta": TeamMetadata(team_name="Beta", seed=5, region="West"),
        }
        calc = LeverageCalculator(
            model_probs=model, public_picks={}, team_metadata=meta,
        )
        picks = calc.find_leverage_picks(min_leverage=0.0, min_probability=0.0)

        alpha_pick = [p for p in picks if p.team_id == "alpha"][0]
        beta_pick = [p for p in picks if p.team_id == "beta"][0]

        # Should match their seed-specific priors
        assert alpha_pick.public_pick_percentage == pytest.approx(
            SEED_PICK_RATES[2]["CHAMP"]
        )
        assert beta_pick.public_pick_percentage == pytest.approx(
            SEED_PICK_RATES[5]["CHAMP"]
        )
        # And they should be different from each other
        assert alpha_pick.public_pick_percentage != beta_pick.public_pick_percentage

    def test_simulation_result_uses_floor_not_fallback(self):
        """AggregatedResults uses 0.01 floor, not 0.001."""
        result = AggregatedResults(
            championship_odds={"x": 0.05},
            num_simulations=10000,
        )
        picks = result.get_leverage_picks({}, min_leverage=0.0)
        assert len(picks) == 1
        _, _, public_used, _ = picks[0]
        assert public_used == pytest.approx(0.01), (
            f"Expected 0.01 floor, got {public_used}"
        )
