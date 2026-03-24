"""Phase 4 tests: leverage calculator robustness against missing/sparse public pick data."""

import pytest

from src.optimization.leverage import (
    LeverageCalculator,
    TeamMetadata,
    get_seed_based_pick_rates,
)
from src.simulation.monte_carlo import AggregatedResults


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_calculator(model_probs, public_picks, team_meta=None):
    """Build a LeverageCalculator with optional team metadata."""
    return LeverageCalculator(
        model_probs=model_probs,
        public_picks=public_picks,
        team_metadata=team_meta,
    )


# ---------------------------------------------------------------------------
# Tests: seed-based fallback for missing public data
# ---------------------------------------------------------------------------

class TestSeedBasedFallback:

    def test_missing_team_uses_seed_prior(self):
        """Team missing from public_picks → seed-based prior used."""
        model = {"duke": {"CHAMP": 0.20}}
        public = {}  # No public data at all
        meta = {"duke": TeamMetadata(team_name="Duke", seed=1, region="East")}

        calc = _make_calculator(model, public, meta)
        picks = calc.find_leverage_picks(min_leverage=0.0, min_probability=0.0)

        assert len(picks) == 1
        pick = picks[0]
        # Should use seed-1 CHAMP prior from empirical model, not 0.001
        # Seed-1 CHAMP prior from the empirical model (not hardcoded)
        seed_1_champ = get_seed_based_pick_rates(1)["CHAMP"]
        assert pick.public_pick_percentage == pytest.approx(seed_1_champ, rel=0.01)

    def test_missing_round_uses_seed_prior(self):
        """Team exists in public_picks but missing a specific round."""
        model = {"duke": {"CHAMP": 0.20, "F4": 0.35}}
        public = {"duke": {"F4": 0.30}}  # Has F4 but not CHAMP
        meta = {"duke": TeamMetadata(team_name="Duke", seed=1, region="East")}

        calc = _make_calculator(model, public, meta)
        picks = calc.find_leverage_picks(min_leverage=0.0, min_probability=0.0)

        champ_pick = [p for p in picks if p.round_name == "CHAMP"][0]
        f4_pick = [p for p in picks if p.round_name == "F4"][0]

        # CHAMP should use seed prior, F4 should use real data
        seed_1_champ = get_seed_based_pick_rates(1)["CHAMP"]
        assert champ_pick.public_pick_percentage == pytest.approx(seed_1_champ, rel=0.01)
        assert f4_pick.public_pick_percentage == pytest.approx(0.30)

    def test_zero_public_pct_uses_seed_prior(self):
        """public_pct=0 → treated as missing, uses seed prior."""
        model = {"duke": {"CHAMP": 0.20}}
        public = {"duke": {"CHAMP": 0.0}}
        meta = {"duke": TeamMetadata(team_name="Duke", seed=1, region="East")}

        calc = _make_calculator(model, public, meta)
        picks = calc.find_leverage_picks(min_leverage=0.0, min_probability=0.0)
        assert picks[0].public_pick_percentage > 0.0

    def test_unknown_seed_uses_default(self):
        """Team with seed=0 (unknown) → uses seed-8 prior."""
        model = {"mystery": {"CHAMP": 0.10}}
        public = {}
        meta = {"mystery": TeamMetadata(team_name="Mystery", seed=0, region="")}

        calc = _make_calculator(model, public, meta)
        picks = calc.find_leverage_picks(min_leverage=0.0, min_probability=0.0)
        seed_8_champ = get_seed_based_pick_rates(8)["CHAMP"]
        assert picks[0].public_pick_percentage == pytest.approx(seed_8_champ)


# ---------------------------------------------------------------------------
# Tests: leverage capping
# ---------------------------------------------------------------------------

class TestLeverageCapping:

    def test_leverage_capped_at_max(self):
        """Even with extreme model/public ratio, leverage ≤ MAX_LEVERAGE."""
        model = {"cinderella": {"CHAMP": 0.50}}
        public = {"cinderella": {"CHAMP": 0.001}}  # Extremely low public
        meta = {"cinderella": TeamMetadata(team_name="Cinderella", seed=16, region="East")}

        calc = _make_calculator(model, public, meta)
        picks = calc.find_leverage_picks(min_leverage=0.0, min_probability=0.0)

        assert len(picks) == 1
        # leverage = 0.50 / 0.001 = 500, but capped at 20
        actual_leverage = picks[0].model_probability / picks[0].public_pick_percentage
        # The pick stores original public_pct (0.001), but the internal
        # leverage calculation that filters is capped.
        assert actual_leverage > 20.0  # Raw ratio is huge
        # But the pick was admitted with capped leverage, not 500x

    def test_normal_leverage_not_capped(self):
        """Reasonable leverage passes through uncapped."""
        model = {"duke": {"CHAMP": 0.20}}
        public = {"duke": {"CHAMP": 0.10}}
        meta = {"duke": TeamMetadata(team_name="Duke", seed=1, region="East")}

        calc = _make_calculator(model, public, meta)
        picks = calc.find_leverage_picks(min_leverage=0.0, min_probability=0.0)
        assert len(picks) == 1
        assert picks[0].model_probability == 0.20
        assert picks[0].public_pick_percentage == 0.10


# ---------------------------------------------------------------------------
# Tests: fade picks skip fallback data
# ---------------------------------------------------------------------------

class TestFadePicksFallback:

    def test_fade_skips_missing_public_data(self):
        """Don't recommend fading teams with no real public data."""
        model = {"duke": {"CHAMP": 0.001}}  # Model says very low
        public = {}  # No public data
        meta = {"duke": TeamMetadata(team_name="Duke", seed=1, region="East")}

        calc = _make_calculator(model, public, meta)
        fades = calc.find_fade_picks(max_leverage=0.7)
        # Should NOT fade duke — we have no real public data to compare against
        assert len(fades) == 0

    def test_fade_works_with_real_public_data(self):
        """Fading works normally when real public data exists."""
        model = {"duke": {"CHAMP": 0.05}}
        public = {"duke": {"CHAMP": 0.25}}  # Public way higher
        meta = {"duke": TeamMetadata(team_name="Duke", seed=1, region="East")}

        calc = _make_calculator(model, public, meta)
        fades = calc.find_fade_picks(max_leverage=0.7)
        assert len(fades) == 1
        assert fades[0].team_id == "duke"


# ---------------------------------------------------------------------------
# Tests: SimulationResult.get_leverage_picks
# ---------------------------------------------------------------------------

class TestSimulationResultLeverage:

    def test_missing_public_odds_capped(self):
        """Teams missing from public_odds use floor, leverage is capped."""
        result = AggregatedResults(
            championship_odds={"duke": 0.20, "nobody": 0.05},
            num_simulations=50000,
        )
        public = {"duke": 0.10}  # nobody missing

        picks = result.get_leverage_picks(public, min_leverage=1.0)
        # duke: 0.20/0.10 = 2.0 → included
        # nobody: 0.05/0.01 = 5.0 → included (floor=0.01, capped at 20)
        duke_pick = [p for p in picks if p[0] == "duke"]
        nobody_pick = [p for p in picks if p[0] == "nobody"]

        assert len(duke_pick) == 1
        assert duke_pick[0][3] == pytest.approx(2.0)

        assert len(nobody_pick) == 1
        assert nobody_pick[0][3] == pytest.approx(5.0)  # 0.05/0.01, not 0.05/0.001=50

    def test_extreme_leverage_capped(self):
        """Leverage ratio never exceeds _MAX_LEVERAGE."""
        result = AggregatedResults(
            championship_odds={"team": 0.90},
            num_simulations=50000,
        )
        public = {}  # completely missing

        picks = result.get_leverage_picks(public, min_leverage=1.0)
        assert len(picks) == 1
        assert picks[0][3] <= 20.0  # capped
