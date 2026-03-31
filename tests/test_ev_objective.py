"""Tests for the EV-based bracket optimizer objective function.

Validates that ParetoOptimizer uses expected pool-winning value
(P(win) x points x differentiation) rather than raw probability,
eliminating the systematic chalk/favorite bias that caused all-#1-seed
Final Four predictions.
"""

import pytest

from src.optimization.leverage import (
    LeverageCalculator,
    ParetoOptimizer,
    TeamMetadata,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_calculator(
    model_probs,
    public_picks,
    metadata,
    scoring_system=None,
):
    """Build a LeverageCalculator with standard ESPN scoring."""
    return LeverageCalculator(
        model_probs=model_probs,
        public_picks=public_picks,
        team_metadata=metadata,
        scoring_system=scoring_system,
    )


def _round_probs(champ, f4, e8, s16, r32, r64):
    """Shorthand for a full round probability dict."""
    return {"CHAMP": champ, "F4": f4, "E8": e8, "S16": s16, "R32": r32, "R64": r64}


def _full_region_teams(region, prob_scale=1.0, pick_scale=1.0):
    """Generate 16 teams for a region with seed-proportional probs/picks.

    Higher seeds (1-seed) get higher probabilities.  pick_scale controls
    how closely public picks track model probability (1.0 = identical,
    >1.0 = public over-picks favorites).
    """
    model_probs = {}
    public_picks = {}
    metadata = {}
    for seed in range(1, 17):
        tid = f"{region.lower()}_{seed}"
        # Probability decreases with seed number
        base_p = max(0.02, (17 - seed) / 16.0 * prob_scale)
        model_probs[tid] = _round_probs(
            champ=base_p * 0.15,
            f4=base_p * 0.35,
            e8=base_p * 0.55,
            s16=base_p * 0.70,
            r32=base_p * 0.85,
            r64=base_p * 0.95,
        )
        # Public picks: scale so favorites are over-picked
        pub_base = base_p * pick_scale
        public_picks[tid] = _round_probs(
            champ=pub_base * 0.15,
            f4=pub_base * 0.35,
            e8=pub_base * 0.55,
            s16=pub_base * 0.70,
            r32=pub_base * 0.85,
            r64=pub_base * 0.95,
        )
        metadata[tid] = TeamMetadata(team_name=f"{region}{seed}", seed=seed, region=region)
    return model_probs, public_picks, metadata


# ---------------------------------------------------------------------------
# Test: No seed bonus
# ---------------------------------------------------------------------------


class TestEvScoreNoSeedBonus:
    def test_same_prob_different_seeds_equal_score(self):
        """Two teams with identical prob and pick_rate but different seeds
        must receive equal EV scores — proving the seed bonus is gone."""
        model_probs = {
            "team_a": _round_probs(0.10, 0.25, 0.40, 0.55, 0.70, 0.90),
            "team_b": _round_probs(0.10, 0.25, 0.40, 0.55, 0.70, 0.90),
        }
        public_picks = {
            "team_a": _round_probs(0.08, 0.20, 0.35, 0.50, 0.65, 0.85),
            "team_b": _round_probs(0.08, 0.20, 0.35, 0.50, 0.65, 0.85),
        }
        metadata = {
            "team_a": TeamMetadata(team_name="One Seed", seed=1, region="East"),
            "team_b": TeamMetadata(team_name="Twelve Seed", seed=12, region="West"),
        }
        calc = _make_calculator(model_probs, public_picks, metadata)
        opt = ParetoOptimizer(calc, pool_size=200)

        for round_name in ("R64", "R32", "S16", "E8", "F4", "CHAMP"):
            score_a = opt._ev_score("team_a", round_name, risk_level=0.5)
            score_b = opt._ev_score("team_b", round_name, risk_level=0.5)
            assert score_a == pytest.approx(score_b, abs=1e-9), (
                f"Seed bonus detected in {round_name}: {score_a} != {score_b}"
            )


# ---------------------------------------------------------------------------
# Test: Underowned team preferred at high risk
# ---------------------------------------------------------------------------


class TestEvScorePrefersUnderowned:
    def test_underowned_beats_overowned_at_full_risk(self):
        """Team B has lower probability but much lower public ownership.
        At risk_level=1.0 (full differentiation), B should score higher."""
        model_probs = {
            "favorite": _round_probs(0.12, 0.30, 0.50, 0.65, 0.80, 0.95),
            "underdog": _round_probs(0.08, 0.22, 0.38, 0.52, 0.68, 0.85),
        }
        public_picks = {
            "favorite": _round_probs(0.20, 0.45, 0.60, 0.75, 0.85, 0.95),
            "underdog": _round_probs(0.02, 0.05, 0.10, 0.15, 0.20, 0.30),
        }
        metadata = {
            "favorite": TeamMetadata(team_name="Fav", seed=1, region="East"),
            "underdog": TeamMetadata(team_name="Dog", seed=5, region="East"),
        }
        calc = _make_calculator(model_probs, public_picks, metadata)
        opt = ParetoOptimizer(calc, pool_size=200)

        # At risk_level=1.0 (full differentiation), underdog should win on EV
        # in at least some rounds where the ownership gap is extreme
        score_fav = opt._ev_score("favorite", "F4", risk_level=1.0)
        score_dog = opt._ev_score("underdog", "F4", risk_level=1.0)
        assert score_dog > score_fav, (
            f"Underowned team should beat overowned at risk=1.0: dog={score_dog}, fav={score_fav}"
        )


# ---------------------------------------------------------------------------
# Test: Chalk at zero risk
# ---------------------------------------------------------------------------


class TestEvScoreChalkAtZeroRisk:
    def test_higher_prob_wins_at_zero_risk(self):
        """At risk_level=0, differentiation is disabled — the higher-probability
        team must always win (chalk-optimal behavior)."""
        model_probs = {
            "strong": _round_probs(0.15, 0.35, 0.55, 0.70, 0.85, 0.95),
            "weak": _round_probs(0.05, 0.15, 0.30, 0.45, 0.60, 0.80),
        }
        public_picks = {
            "strong": _round_probs(0.10, 0.30, 0.50, 0.65, 0.80, 0.90),
            "weak": _round_probs(0.01, 0.03, 0.08, 0.12, 0.20, 0.40),
        }
        metadata = {
            "strong": TeamMetadata(team_name="Strong", seed=1, region="East"),
            "weak": TeamMetadata(team_name="Weak", seed=12, region="East"),
        }
        calc = _make_calculator(model_probs, public_picks, metadata)
        opt = ParetoOptimizer(calc, pool_size=200)

        for round_name in ("R64", "R32", "S16", "E8", "F4", "CHAMP"):
            score_s = opt._ev_score("strong", round_name, risk_level=0.0)
            score_w = opt._ev_score("weak", round_name, risk_level=0.0)
            assert score_s > score_w, (
                f"At risk=0, stronger team should always win in {round_name}"
            )


# ---------------------------------------------------------------------------
# Test: No seed-based tie-break
# ---------------------------------------------------------------------------


class TestPickWinnerNoSeedTiebreak:
    def test_lexical_tiebreak_not_seed(self):
        """When EV scores are exactly equal, tie-break should be lexical,
        not seed-based. A 12-seed with ID 'aaa' should beat a 1-seed with ID 'zzz'."""
        model_probs = {
            "aaa": _round_probs(0.10, 0.25, 0.40, 0.55, 0.70, 0.90),
            "zzz": _round_probs(0.10, 0.25, 0.40, 0.55, 0.70, 0.90),
        }
        public_picks = {
            "aaa": _round_probs(0.10, 0.25, 0.40, 0.55, 0.70, 0.90),
            "zzz": _round_probs(0.10, 0.25, 0.40, 0.55, 0.70, 0.90),
        }
        metadata = {
            "aaa": TeamMetadata(team_name="Underdog", seed=12, region="East"),
            "zzz": TeamMetadata(team_name="Favorite", seed=1, region="East"),
        }
        calc = _make_calculator(model_probs, public_picks, metadata)
        opt = ParetoOptimizer(calc, pool_size=100)

        winner = opt._pick_winner("aaa", "zzz", "R64", risk_level=0.5)
        assert winner == "aaa", (
            f"Lexical tie-break should pick 'aaa' over 'zzz', got '{winner}'"
        )


# ---------------------------------------------------------------------------
# Test: Full bracket does NOT always produce all-#1 F4
# ---------------------------------------------------------------------------


class TestFullBracketNotAllOnes:
    def test_not_all_one_seeds_at_high_risk(self):
        """With realistic ESPN-like ownership (public massively over-picks
        top seeds), a full bracket at risk_level=0.8 in a large pool should
        NOT have all four #1 seeds in the F4.

        This mimics real ESPN BTC data where ~45% of brackets pick all
        #1 seeds to the F4, making those picks poor EV choices in pools.
        """
        model_probs = {}
        public_picks = {}
        metadata = {}

        for region in ("East", "West", "South", "Midwest"):
            for seed in range(1, 17):
                tid = f"{region.lower()}_{seed}"
                # Flatter model probabilities — 1-seeds not dramatically better
                # than 2-4 seeds (realistic: models see small quality gaps)
                base_p = max(0.05, 1.0 - (seed - 1) * 0.055)
                model_probs[tid] = _round_probs(
                    champ=base_p * 0.12,
                    f4=base_p * 0.30,
                    e8=base_p * 0.50,
                    s16=base_p * 0.65,
                    r32=base_p * 0.80,
                    r64=base_p * 0.92,
                )
                # Public ownership: heavily concentrated on top seeds
                # 1-seed gets 3-4x more public picks than model probability
                # Lower seeds get 0.3-0.5x (under-picked)
                if seed <= 2:
                    pub_mult = 3.5
                elif seed <= 4:
                    pub_mult = 1.8
                elif seed <= 8:
                    pub_mult = 0.6
                else:
                    pub_mult = 0.3
                public_picks[tid] = _round_probs(
                    champ=base_p * 0.12 * pub_mult,
                    f4=base_p * 0.30 * pub_mult,
                    e8=base_p * 0.50 * pub_mult,
                    s16=base_p * 0.65 * pub_mult,
                    r32=base_p * 0.80 * pub_mult,
                    r64=base_p * 0.92 * pub_mult,
                )
                metadata[tid] = TeamMetadata(
                    team_name=f"{region}{seed}", seed=seed, region=region,
                )

        calc = _make_calculator(model_probs, public_picks, metadata)
        opt = ParetoOptimizer(calc, pool_size=500)

        if not opt._can_build_full_bracket():
            pytest.skip("Cannot build full bracket with test data")

        _picks, _champ, final_four, _ev, _var = opt._generate_full_bracket(risk_level=0.8)

        f4_seeds = [metadata[tid].seed for tid in final_four]

        # Not ALL should be 1-seeds when public massively over-picks them
        assert not all(s == 1 for s in f4_seeds), (
            f"All-#1-seed F4 at risk=0.8, pool=500 with heavy chalk ownership: {f4_seeds}"
        )


# ---------------------------------------------------------------------------
# Test: Summary bracket uses EV score
# ---------------------------------------------------------------------------


class TestSummaryBracketUsesEvScore:
    def test_summary_not_all_ones_at_high_risk(self):
        """Summary bracket at risk_level=0.8 with over-picked favorites
        should pick at least one non-#1 F4 team."""
        model_probs = {}
        public_picks = {}
        metadata = {}

        for region in ("East", "West", "South", "Midwest"):
            for seed in range(1, 17):
                tid = f"{region.lower()}_{seed}"
                base_p = max(0.02, (17 - seed) / 16.0)
                model_probs[tid] = _round_probs(
                    champ=base_p * 0.15,
                    f4=base_p * 0.35,
                    e8=base_p * 0.55,
                    s16=base_p * 0.70,
                    r32=base_p * 0.85,
                    r64=base_p * 0.95,
                )
                # Heavily over-pick #1 seeds
                pub_scale = 2.0 if seed <= 2 else 0.5
                public_picks[tid] = _round_probs(
                    champ=base_p * 0.15 * pub_scale,
                    f4=base_p * 0.35 * pub_scale,
                    e8=base_p * 0.55 * pub_scale,
                    s16=base_p * 0.70 * pub_scale,
                    r32=base_p * 0.85 * pub_scale,
                    r64=base_p * 0.95 * pub_scale,
                )
                metadata[tid] = TeamMetadata(
                    team_name=f"{region}{seed}", seed=seed, region=region,
                )

        calc = _make_calculator(model_probs, public_picks, metadata)
        opt = ParetoOptimizer(calc, pool_size=200)

        _picks, _champ, final_four, _ev, _var = opt._generate_summary_bracket(risk_level=0.8)

        f4_seeds = [metadata[tid].seed for tid in final_four]
        assert not all(s == 1 for s in f4_seeds), (
            f"Summary bracket all-#1 at risk=0.8: {f4_seeds}"
        )


# ---------------------------------------------------------------------------
# Test: ev_mode=False backward compatibility
# ---------------------------------------------------------------------------


class TestEvModeFalseBackwardCompat:
    def test_legacy_mode_uses_leverage_formula(self):
        """With ev_mode=False, _ev_score should use the legacy formula:
        model_prob * (leverage ^ risk_level), without seed bonus."""
        model_probs = {
            "team_a": _round_probs(0.10, 0.25, 0.40, 0.55, 0.70, 0.90),
        }
        public_picks = {
            "team_a": _round_probs(0.05, 0.12, 0.20, 0.30, 0.40, 0.60),
        }
        metadata = {
            "team_a": TeamMetadata(team_name="A", seed=5, region="East"),
        }
        calc = _make_calculator(model_probs, public_picks, metadata)
        opt = ParetoOptimizer(calc, pool_size=100, ev_mode=False)

        # At risk_level=0, legacy returns model_prob * leverage^0 = model_prob
        score_r0 = opt._ev_score("team_a", "CHAMP", risk_level=0.0)
        assert score_r0 == pytest.approx(0.10, abs=1e-6)

        # At risk_level=1, legacy returns model_prob * leverage^1
        # leverage = 0.10 / max(0.05, 0.01) = 2.0
        score_r1 = opt._ev_score("team_a", "CHAMP", risk_level=1.0)
        expected = 0.10 * 2.0  # model_prob * leverage
        assert score_r1 == pytest.approx(expected, abs=1e-6)


# ---------------------------------------------------------------------------
# Test: Pool size affects differentiation
# ---------------------------------------------------------------------------


class TestPoolSizeAffectsDifferentiation:
    def test_large_pool_more_differentiation(self):
        """In a large pool, a heavily-picked team's EV score should be
        lower than in a small pool (more differentiation pressure)."""
        model_probs = {
            "popular": _round_probs(0.10, 0.25, 0.40, 0.55, 0.70, 0.90),
        }
        public_picks = {
            "popular": _round_probs(0.15, 0.35, 0.50, 0.65, 0.80, 0.92),
        }
        metadata = {
            "popular": TeamMetadata(team_name="Popular", seed=1, region="East"),
        }

        calc = _make_calculator(model_probs, public_picks, metadata)
        opt_small = ParetoOptimizer(calc, pool_size=20)
        opt_large = ParetoOptimizer(calc, pool_size=1000)

        score_small = opt_small._ev_score("popular", "F4", risk_level=1.0)
        score_large = opt_large._ev_score("popular", "F4", risk_level=1.0)

        assert score_large < score_small, (
            f"Large pool should penalize popular picks more: "
            f"small={score_small}, large={score_large}"
        )


# ---------------------------------------------------------------------------
# Test: Full bracket _generate_full_bracket uses _ev_score for every game
# ---------------------------------------------------------------------------


class TestFullBracketUsesEvScore:
    """Verify _generate_full_bracket routes through _ev_score, not a hidden
    chalk formula. We do this by checking that a contrarian bracket at
    risk_level=1.0 produces different picks than a chalk bracket at
    risk_level=0.0 when ownership is skewed."""

    def test_contrarian_differs_from_chalk(self):
        model_probs = {}
        public_picks = {}
        metadata = {}

        for region in ("East", "West", "South", "Midwest"):
            for seed in range(1, 17):
                tid = f"{region.lower()}_{seed}"
                base_p = max(0.05, 1.0 - (seed - 1) * 0.055)
                model_probs[tid] = _round_probs(
                    champ=base_p * 0.12,
                    f4=base_p * 0.30,
                    e8=base_p * 0.50,
                    s16=base_p * 0.65,
                    r32=base_p * 0.80,
                    r64=base_p * 0.92,
                )
                if seed <= 2:
                    pub_mult = 4.0
                elif seed <= 4:
                    pub_mult = 2.0
                elif seed <= 8:
                    pub_mult = 0.5
                else:
                    pub_mult = 0.2
                public_picks[tid] = _round_probs(
                    champ=base_p * 0.12 * pub_mult,
                    f4=base_p * 0.30 * pub_mult,
                    e8=base_p * 0.50 * pub_mult,
                    s16=base_p * 0.65 * pub_mult,
                    r32=base_p * 0.80 * pub_mult,
                    r64=base_p * 0.92 * pub_mult,
                )
                metadata[tid] = TeamMetadata(
                    team_name=f"{region}{seed}", seed=seed, region=region,
                )

        calc = _make_calculator(model_probs, public_picks, metadata)
        opt = ParetoOptimizer(calc, pool_size=500)

        if not opt._can_build_full_bracket():
            pytest.skip("Cannot build full bracket with test data")

        chalk_picks, _, chalk_f4, _, _ = opt._generate_full_bracket(0.0)
        contra_picks, _, contra_f4, _, _ = opt._generate_full_bracket(1.0)

        # At least some picks must differ
        diff_count = sum(
            1 for k in chalk_picks
            if chalk_picks[k] != contra_picks.get(k)
        )
        assert diff_count > 0, (
            "risk_level=0 and risk_level=1 produced identical brackets — "
            "_ev_score differentiation is not working"
        )

    def test_full_bracket_f4_seed_diversity(self):
        """At high risk in a large pool, F4 should include non-#1 seeds."""
        model_probs = {}
        public_picks = {}
        metadata = {}

        for region in ("East", "West", "South", "Midwest"):
            for seed in range(1, 17):
                tid = f"{region.lower()}_{seed}"
                # Make 3-4 seeds nearly as strong as 1-2 seeds
                if seed <= 4:
                    base_p = 0.90 - (seed - 1) * 0.03
                else:
                    base_p = max(0.05, 0.75 - (seed - 4) * 0.06)
                model_probs[tid] = _round_probs(
                    champ=base_p * 0.12,
                    f4=base_p * 0.30,
                    e8=base_p * 0.50,
                    s16=base_p * 0.65,
                    r32=base_p * 0.80,
                    r64=base_p * 0.92,
                )
                # Public heavily concentrates on 1-seeds
                if seed == 1:
                    pub_mult = 5.0
                elif seed == 2:
                    pub_mult = 3.0
                elif seed <= 4:
                    pub_mult = 1.0
                else:
                    pub_mult = 0.3
                public_picks[tid] = _round_probs(
                    champ=base_p * 0.12 * pub_mult,
                    f4=base_p * 0.30 * pub_mult,
                    e8=base_p * 0.50 * pub_mult,
                    s16=base_p * 0.65 * pub_mult,
                    r32=base_p * 0.80 * pub_mult,
                    r64=base_p * 0.92 * pub_mult,
                )
                metadata[tid] = TeamMetadata(
                    team_name=f"{region}{seed}", seed=seed, region=region,
                )

        calc = _make_calculator(model_probs, public_picks, metadata)
        opt = ParetoOptimizer(calc, pool_size=1000)

        if not opt._can_build_full_bracket():
            pytest.skip("Cannot build full bracket with test data")

        _, _, f4, _, _ = opt._generate_full_bracket(risk_level=0.9)
        f4_seeds = [metadata[tid].seed for tid in f4]

        assert not all(s == 1 for s in f4_seeds), (
            f"Full bracket F4 all #1 seeds at risk=0.9, pool=1000: {f4_seeds}"
        )


# ---------------------------------------------------------------------------
# Test: Edge cases
# ---------------------------------------------------------------------------


class TestEvScoreEdgeCases:
    def test_zero_model_prob_returns_zero(self):
        """A team with model_prob=0 should have EV score = 0."""
        model_probs = {"ghost": _round_probs(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)}
        public_picks = {"ghost": _round_probs(0.10, 0.20, 0.30, 0.40, 0.50, 0.60)}
        metadata = {"ghost": TeamMetadata(team_name="Ghost", seed=8, region="East")}
        calc = _make_calculator(model_probs, public_picks, metadata)
        opt = ParetoOptimizer(calc, pool_size=100)

        for rn in ("R64", "CHAMP"):
            assert opt._ev_score("ghost", rn, 0.5) == 0.0

    def test_unknown_team_returns_zero(self):
        """A team not in model_probs should score 0."""
        model_probs = {"real": _round_probs(0.10, 0.25, 0.40, 0.55, 0.70, 0.90)}
        public_picks = {"real": _round_probs(0.08, 0.20, 0.35, 0.50, 0.65, 0.85)}
        metadata = {"real": TeamMetadata(team_name="Real", seed=1, region="East")}
        calc = _make_calculator(model_probs, public_picks, metadata)
        opt = ParetoOptimizer(calc, pool_size=100)

        assert opt._ev_score("nonexistent", "CHAMP", 0.5) == 0.0

    def test_pool_size_one_no_crash(self):
        """Pool size of 1 should not crash or produce NaN."""
        model_probs = {"team": _round_probs(0.10, 0.25, 0.40, 0.55, 0.70, 0.90)}
        public_picks = {"team": _round_probs(0.08, 0.20, 0.35, 0.50, 0.65, 0.85)}
        metadata = {"team": TeamMetadata(team_name="Solo", seed=3, region="East")}
        calc = _make_calculator(model_probs, public_picks, metadata)
        opt = ParetoOptimizer(calc, pool_size=1)

        score = opt._ev_score("team", "CHAMP", 1.0)
        assert score > 0
        assert score == score  # not NaN


# ---------------------------------------------------------------------------
# Test: QuadrantCorrelationConstraint defaults
# ---------------------------------------------------------------------------


class TestPathProtectionDefaults:
    def test_require_safe_f4_defaults_false(self):
        """require_safe_f4 should default to False to avoid chalk bias."""
        from src.optimization.path_protection import QuadrantCorrelationConstraint
        constraint = QuadrantCorrelationConstraint()
        assert constraint.require_safe_f4 is False, (
            "require_safe_f4 should default to False"
        )


# ---------------------------------------------------------------------------
# Test: Chalk fallback uses stochastic sampling
# ---------------------------------------------------------------------------


class TestChalkFallbackStochastic:
    def test_fallback_not_deterministic_chalk(self):
        """_generate_chalk_winners should use stochastic sampling,
        not always pick the favorite (p >= 0.5)."""
        from src.pipeline.stages.ev_analysis import _generate_chalk_winners

        # Set up 32-team bracket where underdogs have ~45% win prob
        teams = [f"team_{i}" for i in range(32)]
        matchup_probs = {}
        for i in range(0, 32, 2):
            # Slight favorite: 55% vs 45%
            matchup_probs[(teams[i], teams[i + 1])] = 0.55

        winners = _generate_chalk_winners(teams, matchup_probs)
        # With p=0.55 and ~16 first-round games, deterministic chalk would
        # always pick teams[0], teams[2], etc. Stochastic should sometimes
        # pick the other team. With seed=42, at least one upset should occur.
        r64_winners = winners[:16]
        favorites = {teams[i] for i in range(0, 32, 2)}
        non_favorites = [w for w in r64_winners if w not in favorites]
        assert len(non_favorites) > 0, (
            "Fallback bracket is deterministic chalk — should be stochastic"
        )
