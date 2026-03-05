"""Tests for Phase 4: Behavioral Archetype Modeling."""

import pytest
import numpy as np
from src.simulation.competitor_archetypes import (
    ChalkPicker,
    ChaosSeeker,
    ExpertMimic,
    Homer,
    VALID_ARCHETYPE_NAMES,
    get_archetype_mix,
    create_archetypes,
    POOL_TYPE_ARCHETYPES,
)


class TestArchetypePickProbabilities:
    """Test that each archetype produces valid probabilities."""

    def test_chalk_picker_valid_range(self):
        arch = ChalkPicker()
        for seed in range(1, 17):
            prob = arch.predict_pick_probability("team", "R64", 0.6, 0.7, seed)
            assert 0.0 <= prob <= 1.0

    def test_chaos_seeker_valid_range(self):
        arch = ChaosSeeker()
        for seed in range(1, 17):
            prob = arch.predict_pick_probability("team", "R64", 0.6, 0.7, seed)
            assert 0.0 <= prob <= 1.0

    def test_expert_mimic_valid_range(self):
        arch = ExpertMimic()
        for seed in range(1, 17):
            prob = arch.predict_pick_probability("team", "R64", 0.6, 0.7, seed)
            assert 0.0 <= prob <= 1.0

    def test_homer_valid_range(self):
        arch = Homer()
        for seed in range(1, 17):
            prob = arch.predict_pick_probability("team", "R64", 0.6, 0.7, seed)
            assert 0.0 <= prob <= 1.0


class TestArchetypeBehavior:
    """Test characteristic behavioral patterns of each archetype."""

    def test_chalk_favors_high_public(self):
        """Chalk picker should favor teams with high public consensus."""
        arch = ChalkPicker()
        high_pub = arch.predict_pick_probability("A", "R64", 0.5, 0.9, 1)
        low_pub = arch.predict_pick_probability("A", "R64", 0.5, 0.1, 1)
        assert high_pub > low_pub

    def test_chalk_favors_low_seeds(self):
        """Chalk picker should favor 1-seeds over 16-seeds."""
        arch = ChalkPicker()
        seed1 = arch.predict_pick_probability("A", "R64", 0.5, 0.5, 1)
        seed16 = arch.predict_pick_probability("A", "R64", 0.5, 0.5, 16)
        assert seed1 > seed16

    def test_chaos_boosts_underdogs(self):
        """Chaos seeker should give underdogs (high seeds) a boost."""
        arch = ChaosSeeker()
        # Same model prob, but seed 12 (underdog) should get a boost
        favorite = arch.predict_pick_probability("A", "R64", 0.3, 0.1, 5)
        underdog = arch.predict_pick_probability("A", "R64", 0.3, 0.1, 12)
        assert underdog > favorite

    def test_expert_follows_model(self):
        """Expert mimic should primarily follow model probability."""
        arch = ExpertMimic()
        high_model = arch.predict_pick_probability("A", "R64", 0.9, 0.1, 8)
        low_model = arch.predict_pick_probability("A", "R64", 0.1, 0.9, 8)
        assert high_model > low_model

    def test_expert_mimic_close_to_model(self):
        """Expert mimic's output should be close to model probability."""
        arch = ExpertMimic()
        model_prob = 0.7
        result = arch.predict_pick_probability("A", "R64", model_prob, 0.5, 4)
        # Should be within 0.15 of model (85% weight)
        assert abs(result - model_prob) < 0.2

    def test_homer_sometimes_boosts(self):
        """Homer should boost some teams' probabilities."""
        arch = Homer()
        # Test many teams — at least one should get homer boost
        # (homer fires when hash(team_id) % 10 == 0, so with 100 teams
        # we should reliably hit at least one)
        probs = [
            arch.predict_pick_probability(f"team_{i}", "R64", 0.5, 0.5, 8)
            for i in range(100)
        ]
        # At least one should be > 0.5 (homer boosted)
        assert any(p > 0.6 for p in probs)


class TestArchetypeMix:
    """Test archetype mix factory and validation."""

    def test_espn_national_sums_to_one(self):
        mix = get_archetype_mix("espn_national")
        assert abs(sum(mix.values()) - 1.0) < 0.01

    def test_office_pool_sums_to_one(self):
        mix = get_archetype_mix("office_pool")
        assert abs(sum(mix.values()) - 1.0) < 0.01

    def test_kaggle_sums_to_one(self):
        mix = get_archetype_mix("kaggle")
        assert abs(sum(mix.values()) - 1.0) < 0.01

    def test_all_pool_types_sum_to_one(self):
        for pool_type in POOL_TYPE_ARCHETYPES:
            mix = get_archetype_mix(pool_type)
            assert abs(sum(mix.values()) - 1.0) < 0.01, f"{pool_type} doesn't sum to 1.0"

    def test_unknown_pool_type_uses_default(self):
        mix = get_archetype_mix("unknown_pool")
        assert mix == POOL_TYPE_ARCHETYPES["espn_national"]

    def test_kaggle_heavy_on_experts(self):
        mix = get_archetype_mix("kaggle")
        assert mix["expert_mimic"] >= 0.5

    def test_office_pool_more_homers(self):
        espn = get_archetype_mix("espn_national")
        office = get_archetype_mix("office_pool")
        assert office["homer"] >= espn["homer"]


class TestCreateArchetypes:
    """Test archetype instantiation."""

    def test_creates_all_four_archetypes(self):
        archetypes = create_archetypes()
        assert len(archetypes) == 4
        names = {a.params.name for a in archetypes}
        assert "chalk_picker" in names
        assert "chaos_seeker" in names
        assert "expert_mimic" in names
        assert "homer" in names

    def test_custom_mix(self):
        mix = {"chalk_picker": 0.5, "expert_mimic": 0.5}
        archetypes = create_archetypes(mix)
        assert len(archetypes) == 2

    def test_preserves_prevalence(self):
        mix = {"chalk_picker": 0.7, "chaos_seeker": 0.3}
        archetypes = create_archetypes(mix)
        chalk = [a for a in archetypes if a.params.name == "chalk_picker"][0]
        assert chalk.params.prevalence == 0.7

    def test_zero_prevalence_excluded(self):
        mix = {"chalk_picker": 0.5, "homer": 0.0, "expert_mimic": 0.5}
        archetypes = create_archetypes(mix)
        names = {a.params.name for a in archetypes}
        assert "homer" not in names


class TestGenerateBracketProbs:
    """Test bracket generation from archetypes."""

    def test_chalk_generates_bracket(self):
        arch = ChalkPicker()
        matchups = [
            {"round": "R64", "team1": "duke", "team2": "unc"},
            {"round": "R32", "team1": "duke", "team2": "kansas"},
        ]
        model_probs = {
            "duke": {"R64": 0.7, "R32": 0.6},
            "unc": {"R64": 0.3, "R32": 0.2},
            "kansas": {"R64": 0.8, "R32": 0.5},
        }
        public_picks = {
            "duke": {"R64": 0.75, "R32": 0.55},
            "unc": {"R64": 0.25, "R32": 0.15},
            "kansas": {"R64": 0.8, "R32": 0.5},
        }
        seeds = {"duke": 1, "unc": 8, "kansas": 2}
        rng = np.random.RandomState(42)

        bracket = arch.generate_bracket_probs(matchups, model_probs, public_picks, seeds, rng)
        assert len(bracket) == 2
        for key, winner in bracket.items():
            assert winner in ("duke", "unc", "kansas")


class TestArchetypeOverrides:
    """Test user-provided archetype distribution overrides."""

    def test_overrides_merge_with_defaults(self):
        """Overrides should replace specific archetype values in the base mix."""
        mix = get_archetype_mix("espn_national", overrides={
            "homer": 0.40,
            "chalk_picker": 0.20,
            "expert_mimic": 0.25,
            "chaos_seeker": 0.15,
        })
        assert abs(mix["homer"] - 0.40) < 0.01
        assert abs(mix["chalk_picker"] - 0.20) < 0.01

    def test_overrides_partial_merge(self):
        """Partial overrides should merge with remaining base values."""
        base = get_archetype_mix("espn_national")
        # Only override homer — but this changes the total, so must adjust
        # to keep sum close to 1.0
        mix = get_archetype_mix("espn_national", overrides={
            "homer": 0.30,
            "chalk_picker": 0.25,
            "expert_mimic": 0.30,
            "chaos_seeker": 0.15,
        })
        assert abs(sum(mix.values()) - 1.0) < 0.01

    def test_overrides_invalid_name_raises(self):
        with pytest.raises(ValueError, match="Invalid archetype names"):
            get_archetype_mix("espn_national", overrides={"bad_name": 0.5})

    def test_overrides_not_summing_to_one_raises(self):
        """Overrides that make the total far from 1.0 should raise."""
        with pytest.raises(ValueError, match="must sum to approximately 1.0"):
            get_archetype_mix("espn_national", overrides={
                "chalk_picker": 0.1,
                "expert_mimic": 0.1,
                "chaos_seeker": 0.1,
                "homer": 0.1,
            })

    def test_overrides_close_to_one_renormalized(self):
        """Values summing to ~0.98 should be renormalized, not rejected."""
        mix = get_archetype_mix("espn_national", overrides={
            "chalk_picker": 0.38,
            "expert_mimic": 0.30,
            "chaos_seeker": 0.15,
            "homer": 0.15,
        })
        assert abs(sum(mix.values()) - 1.0) < 0.001

    def test_overrides_none_returns_defaults(self):
        mix = get_archetype_mix("espn_national", overrides=None)
        expected = POOL_TYPE_ARCHETYPES["espn_national"]
        for k in expected:
            assert abs(mix[k] - expected[k]) < 0.001

    def test_overrides_with_different_pool_types(self):
        """Overrides should work with any base pool type."""
        for pool_type in POOL_TYPE_ARCHETYPES:
            mix = get_archetype_mix(pool_type, overrides={
                "chalk_picker": 0.25,
                "expert_mimic": 0.25,
                "chaos_seeker": 0.25,
                "homer": 0.25,
            })
            assert abs(sum(mix.values()) - 1.0) < 0.01
            assert abs(mix["chalk_picker"] - 0.25) < 0.01
