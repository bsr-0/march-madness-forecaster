"""
Tests for Poisson Binomial SOR and WAB metrics.

Validates:
  1. Poisson Binomial CDF correctness against known distributions
  2. SOR monotonicity: better records → higher SOR
  3. SOR schedule sensitivity: same record against harder schedule → higher SOR
  4. WAB_poisson consistency with per-game WAB
  5. Edge cases: undefeated, winless, single-game seasons
  6. Feature vector integration: dimensions match after adding SOR/WAB_pb
  7. Numerical stability: extreme probabilities don't produce NaN/inf
"""

import math
import numpy as np
import pytest

from src.data.features.proprietary_metrics import (
    GameRecord,
    ProprietaryMetricsEngine,
    ProprietaryTeamMetrics,
)
from src.data.features.feature_engineering import (
    TeamFeatures,
    TEAM_FEATURE_DIM,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_game(
    team_id: str,
    opp_id: str,
    team_pts: float,
    opp_pts: float,
    is_home: bool = False,
    is_neutral: bool = True,
    game_date: str = "2025-02-01",
    game_id: str = "",
) -> GameRecord:
    """Create a minimal GameRecord for testing."""
    return GameRecord(
        game_id=game_id or f"{team_id}_vs_{opp_id}_{game_date}",
        game_date=game_date,
        team_id=team_id,
        team_name=team_id,
        opponent_id=opp_id,
        points=team_pts,
        opp_points=opp_pts,
        possessions=70.0,
        fga=60.0, fgm=25.0, fg3a=20.0, fg3m=7.0,
        fta=15.0, ftm=11.0, tov=12.0, orb=10.0, drb=25.0,
        opp_fga=60.0, opp_fgm=24.0, opp_fg3a=20.0, opp_fg3m=6.0,
        opp_fta=14.0, opp_ftm=10.0, opp_tov=13.0, opp_orb=9.0, opp_drb=24.0,
        ast=14.0, stl=7.0, blk=4.0, pf=16.0,
        opp_ast=13.0, opp_stl=6.0, opp_blk=3.0, opp_pf=15.0,
        is_home=is_home,
        is_neutral=is_neutral,
    )


def _build_season(
    team_id: str,
    opponents: list,
    results: list,
    is_neutral: bool = True,
) -> list:
    """
    Build a list of GameRecords for a team.

    Args:
        team_id: The team's ID
        opponents: list of opponent IDs
        results: list of (team_pts, opp_pts) tuples
        is_neutral: whether games are neutral-site
    """
    games = []
    for i, (opp_id, (pts, opp_pts)) in enumerate(zip(opponents, results)):
        date = f"2025-01-{(i+1):02d}"
        games.append(_make_game(
            team_id=team_id,
            opp_id=opp_id,
            team_pts=pts,
            opp_pts=opp_pts,
            is_neutral=is_neutral,
            game_date=date,
        ))
        # Mirror game for opponent
        games.append(_make_game(
            team_id=opp_id,
            opp_id=team_id,
            team_pts=opp_pts,
            opp_pts=pts,
            is_neutral=is_neutral,
            game_date=date,
        ))
    return games


# ---------------------------------------------------------------------------
# Test: Poisson Binomial CDF correctness
# ---------------------------------------------------------------------------

class TestPoissonBinomialCDF:
    """Test the _poisson_binomial_cdf static method."""

    def setup_method(self):
        self.engine = ProprietaryMetricsEngine()

    def test_identical_probs_matches_binomial(self):
        """When all p_i are equal, PoissonBinomial reduces to Binomial."""
        from scipy.stats import binom

        p = 0.6
        n = 20
        probs = [p] * n

        for k in range(n + 1):
            pb_cdf = self.engine._poisson_binomial_cdf(k, probs)
            binom_cdf = binom.cdf(k, n, p)
            assert abs(pb_cdf - binom_cdf) < 1e-10, (
                f"k={k}: PoissonBinomial CDF={pb_cdf:.10f}, Binomial CDF={binom_cdf:.10f}"
            )

    def test_empty_schedule(self):
        """Empty schedule → CDF(any k) = 1.0."""
        assert self.engine._poisson_binomial_cdf(0, []) == 1.0
        assert self.engine._poisson_binomial_cdf(5, []) == 1.0

    def test_single_game(self):
        """Single game: P(X ≤ 0) = 1-p, P(X ≤ 1) = 1.0."""
        p = 0.7
        cdf_0 = self.engine._poisson_binomial_cdf(0, [p])
        cdf_1 = self.engine._poisson_binomial_cdf(1, [p])
        assert abs(cdf_0 - (1 - p)) < 1e-10
        assert abs(cdf_1 - 1.0) < 1e-10

    def test_two_different_probs(self):
        """Two games with different probabilities."""
        p1, p2 = 0.8, 0.3
        # P(X=0) = (1-p1)(1-p2) = 0.2*0.7 = 0.14
        # P(X=1) = p1*(1-p2) + (1-p1)*p2 = 0.8*0.7 + 0.2*0.3 = 0.62
        # P(X=2) = p1*p2 = 0.24
        # CDF(0) = 0.14, CDF(1) = 0.76, CDF(2) = 1.0
        cdf_0 = self.engine._poisson_binomial_cdf(0, [p1, p2])
        cdf_1 = self.engine._poisson_binomial_cdf(1, [p1, p2])
        cdf_2 = self.engine._poisson_binomial_cdf(2, [p1, p2])
        assert abs(cdf_0 - 0.14) < 1e-10
        assert abs(cdf_1 - 0.76) < 1e-10
        assert abs(cdf_2 - 1.0) < 1e-10

    def test_monotonicity(self):
        """CDF must be non-decreasing in k."""
        probs = [0.3, 0.5, 0.7, 0.9, 0.1, 0.6, 0.4, 0.8]
        prev = 0.0
        for k in range(len(probs) + 1):
            val = self.engine._poisson_binomial_cdf(k, probs)
            assert val >= prev - 1e-12, f"CDF not monotone at k={k}"
            prev = val
        assert abs(prev - 1.0) < 1e-10

    def test_numerical_stability_extreme_probs(self):
        """Extreme probabilities (near 0 and 1) should not produce NaN."""
        probs = [0.001, 0.999, 0.5, 0.001, 0.999] * 6  # 30 games
        for k in range(len(probs) + 1):
            val = self.engine._poisson_binomial_cdf(k, probs)
            assert not math.isnan(val), f"NaN at k={k}"
            assert not math.isinf(val), f"Inf at k={k}"
            assert 0.0 <= val <= 1.0 + 1e-10, f"CDF out of [0,1] at k={k}: {val}"


# ---------------------------------------------------------------------------
# Test: SOR properties
# ---------------------------------------------------------------------------

class TestSOR:
    """Test SOR computation via the full engine."""

    def _run_engine(self, games):
        engine = ProprietaryMetricsEngine()
        results = engine.compute(games)
        return results

    def test_better_record_higher_sor(self):
        """A team with more wins against the same opponents should have higher SOR."""
        # Team A: beats everyone (5-0)
        # Team B: beats some (3-2)
        # Same opponents (C, D, E, F, G) — all neutral-site
        opps = ["C", "D", "E", "F", "G"]

        games_a = _build_season("A", opps, [(80, 60)] * 5)
        games_b = _build_season("B", opps, [(80, 60)] * 3 + [(60, 80)] * 2)

        # Also need dummy games for opponents to establish their strength
        all_games = games_a + games_b
        results = self._run_engine(all_games)

        assert results["A"].sor > results["B"].sor, (
            f"5-0 team SOR ({results['A'].sor}) should exceed 3-2 team SOR ({results['B'].sor})"
        )

    def test_harder_schedule_higher_sor(self):
        """Same record against harder opponents should yield higher SOR."""
        # Team A: 3-2 against strong opponents (who beat each other)
        # Team B: 3-2 against weak opponents (who lose to each other)
        # We construct this by having "strong" opponents also play and win games
        # and "weak" opponents also play and lose games.

        # Strong opponents: S1-S5 (will have positive AdjEM)
        # Weak opponents: W1-W5 (will have negative AdjEM)

        games = []

        # Team A plays strong opponents, goes 3-2
        a_opps = [f"S{i}" for i in range(1, 6)]
        a_results = [(80, 65)] * 3 + [(65, 80)] * 2
        games += _build_season("A", a_opps, a_results)

        # Team B plays weak opponents, goes 3-2
        b_opps = [f"W{i}" for i in range(1, 6)]
        b_results = [(80, 65)] * 3 + [(65, 80)] * 2
        games += _build_season("B", b_opps, b_results)

        # Strong opponents beat weak opponents to establish quality gap
        for i in range(1, 6):
            date = f"2025-02-{i:02d}"
            games.append(_make_game(f"S{i}", f"W{i}", 85, 60, game_date=date))
            games.append(_make_game(f"W{i}", f"S{i}", 60, 85, game_date=date))

        results = self._run_engine(games)
        assert results["A"].sor > results["B"].sor, (
            f"Hard-schedule SOR ({results['A'].sor}) should exceed "
            f"easy-schedule SOR ({results['B'].sor})"
        )

    def test_sor_range(self):
        """SOR must be in [0, 1]."""
        opps = [f"T{i}" for i in range(1, 11)]
        games = _build_season("X", opps, [(75, 65)] * 10)
        results = self._run_engine(games)
        assert 0.0 <= results["X"].sor <= 1.0

    def test_undefeated_team_high_sor(self):
        """Undefeated team should have SOR near 1.0."""
        opps = [f"T{i}" for i in range(1, 16)]
        games = _build_season("CHAMP", opps, [(85, 65)] * 15)
        results = self._run_engine(games)
        assert results["CHAMP"].sor > 0.85, (
            f"15-0 team should have high SOR, got {results['CHAMP'].sor}"
        )

    def test_winless_team_low_sor(self):
        """Winless team against decent opponents should have low SOR."""
        # Create a league where opponents have good records against each other
        # so they have positive AdjEM, making the winless record unimpressive.
        games = []
        opps = [f"T{i}" for i in range(1, 6)]

        # LAST loses to all opponents
        games += _build_season("LAST", opps, [(55, 80)] * 5)

        # Opponents also play each other (round robin) to establish quality
        for i, t1 in enumerate(opps):
            for t2 in opps[i+1:]:
                date = f"2025-02-{(i*5+int(t2[1:])):02d}"
                games.append(_make_game(t1, t2, 75, 70, game_date=date))
                games.append(_make_game(t2, t1, 70, 75, game_date=date))

        results = self._run_engine(games)
        assert results["LAST"].sor < 0.25, (
            f"0-5 team vs decent opponents should have low SOR, got {results['LAST'].sor}"
        )


# ---------------------------------------------------------------------------
# Test: WAB Poisson properties
# ---------------------------------------------------------------------------

class TestWABPoisson:
    """Test WAB_poisson metric properties."""

    def _run_engine(self, games):
        engine = ProprietaryMetricsEngine()
        return engine.compute(games)

    def test_wab_poisson_sign(self):
        """Good teams should have positive WAB_pb, bad teams negative.

        Need opponents with established quality (via games against each other)
        so that the bubble team's expected wins are reasonable.
        """
        games = []
        opps = [f"T{i}" for i in range(1, 6)]

        # STRONG: beats all 5 opponents
        games += _build_season("STRONG", opps, [(80, 65)] * 5)
        # WEAK: loses to all 5 opponents
        games += _build_season("WEAK", opps, [(55, 75)] * 5)

        # Opponents play each other (round robin) — establishes moderate quality
        for i, t1 in enumerate(opps):
            for t2 in opps[i+1:]:
                date = f"2025-02-{(i*5+int(t2[1:])):02d}"
                games.append(_make_game(t1, t2, 72, 70, game_date=date))
                games.append(_make_game(t2, t1, 70, 72, game_date=date))

        results = self._run_engine(games)

        assert results["STRONG"].wab_poisson > results["WEAK"].wab_poisson, (
            f"5-0 WAB_pb ({results['STRONG'].wab_poisson}) should exceed "
            f"0-5 WAB_pb ({results['WEAK'].wab_poisson})"
        )

    def test_wab_poisson_consistent_with_pergame_wab(self):
        """WAB_poisson should correlate with per-game WAB (same sign, similar magnitude)."""
        opps = [f"T{i}" for i in range(1, 11)]
        games = _build_season("MID", opps, [(75, 70)] * 6 + [(65, 75)] * 4)
        results = self._run_engine(games)

        wab_original = results["MID"].wab
        wab_pb = results["MID"].wab_poisson

        # They should have the same sign
        if abs(wab_original) > 0.5 and abs(wab_pb) > 0.5:
            assert (wab_original > 0) == (wab_pb > 0), (
                f"WAB ({wab_original}) and WAB_pb ({wab_pb}) should have same sign"
            )

    def test_wab_poisson_monotone_in_wins(self):
        """More wins against the same schedule → higher WAB_pb."""
        opps = [f"T{i}" for i in range(1, 6)]

        games_3_2 = _build_season("A", opps, [(80, 65)] * 3 + [(60, 75)] * 2)
        games_4_1 = _build_season("B", opps, [(80, 65)] * 4 + [(60, 75)] * 1)
        all_games = games_3_2 + games_4_1
        results = self._run_engine(all_games)

        assert results["B"].wab_poisson > results["A"].wab_poisson, (
            f"4-1 WAB_pb ({results['B'].wab_poisson}) should exceed "
            f"3-2 WAB_pb ({results['A'].wab_poisson})"
        )


# ---------------------------------------------------------------------------
# Test: Feature vector integration
# ---------------------------------------------------------------------------

class TestFeatureVectorIntegration:
    """Ensure SOR and WAB_pb are in the feature vector at correct positions."""

    def test_feature_dim_updated(self):
        """TEAM_FEATURE_DIM should be 79 (71 base + 2 graph SOS + 3 win quality + 3 per-stage coaching)."""
        assert TEAM_FEATURE_DIM == 79

    def test_to_vector_length(self):
        """to_vector() output length should match TEAM_FEATURE_DIM."""
        tf = TeamFeatures(team_id="test", team_name="Test", seed=8, region="East")
        vec = tf.to_vector()
        assert len(vec) == TEAM_FEATURE_DIM, (
            f"to_vector() produced {len(vec)} features, expected {TEAM_FEATURE_DIM}"
        )

    def test_feature_names_length(self):
        """get_feature_names() should match TEAM_FEATURE_DIM."""
        names = TeamFeatures.get_feature_names()
        assert len(names) == TEAM_FEATURE_DIM

    def test_sor_in_feature_names(self):
        """'sor' and 'wab_poisson' should be in feature names."""
        names = TeamFeatures.get_feature_names()
        assert 'sor' in names, "Missing 'sor' in feature names"
        assert 'wab_poisson' in names, "Missing 'wab_poisson' in feature names"

    def test_sor_value_in_vector(self):
        """SOR and WAB_pb values should appear in the vector."""
        tf = TeamFeatures(
            team_id="test", team_name="Test", seed=1, region="East",
            sor=0.85, wab_poisson=3.5,
        )
        vec = tf.to_vector()
        names = TeamFeatures.get_feature_names()

        sor_idx = names.index('sor')
        wab_pb_idx = names.index('wab_poisson')

        assert abs(vec[sor_idx] - 0.85) < 1e-10, f"SOR value mismatch: {vec[sor_idx]}"
        assert abs(vec[wab_pb_idx] - 3.5) < 1e-10, f"WAB_pb value mismatch: {vec[wab_pb_idx]}"

    def test_vector_names_alignment(self):
        """Every feature name should correspond to the correct position in to_vector()."""
        tf = TeamFeatures(
            team_id="test", team_name="Test", seed=5, region="West",
            wab=2.0, sor=0.7, wab_poisson=1.5,
        )
        vec = tf.to_vector()
        names = TeamFeatures.get_feature_names()

        # Check wab, sor, wab_poisson are consecutive (sor/wab_pb right after wab)
        wab_idx = names.index('wab')
        sor_idx = names.index('sor')
        wab_pb_idx = names.index('wab_poisson')
        assert sor_idx == wab_idx + 1, "SOR should immediately follow WAB"
        assert wab_pb_idx == sor_idx + 1, "WAB_poisson should immediately follow SOR"


# ---------------------------------------------------------------------------
# Test: Poisson Binomial mean
# ---------------------------------------------------------------------------

class TestPoissonBinomialMean:
    """Test the _poisson_binomial_mean helper."""

    def test_mean_equals_sum(self):
        probs = [0.3, 0.5, 0.7, 0.9]
        expected = sum(probs)
        result = ProprietaryMetricsEngine._poisson_binomial_mean(probs)
        assert abs(result - expected) < 1e-10

    def test_empty_mean(self):
        assert ProprietaryMetricsEngine._poisson_binomial_mean([]) == 0.0


# ---------------------------------------------------------------------------
# Test: End-to-end with engine
# ---------------------------------------------------------------------------

class TestEndToEnd:
    """Integration test: full engine produces valid SOR/WAB_pb values."""

    def test_full_engine_produces_sor_wab_pb(self):
        """Verify that engine.compute() populates sor and wab_poisson."""
        games = []
        teams = [f"T{i}" for i in range(1, 9)]

        # Round-robin: each team plays every other team once
        for i, t1 in enumerate(teams):
            for t2 in teams[i+1:]:
                date = f"2025-01-{(i+1):02d}"
                # Higher-indexed teams generally win (creates a hierarchy)
                t1_pts = 70 + int(t1[1:]) * 2
                t2_pts = 70 + int(t2[1:]) * 2
                games.append(_make_game(t1, t2, t1_pts, t2_pts, game_date=date))
                games.append(_make_game(t2, t1, t2_pts, t1_pts, game_date=date))

        engine = ProprietaryMetricsEngine()
        results = engine.compute(games)

        for tid in teams:
            assert tid in results
            r = results[tid]
            assert 0.0 <= r.sor <= 1.0, f"Team {tid} SOR={r.sor} out of [0,1]"
            assert not math.isnan(r.wab_poisson), f"Team {tid} WAB_pb is NaN"
            assert not math.isinf(r.wab_poisson), f"Team {tid} WAB_pb is inf"

    def test_sor_ordering_matches_record_quality(self):
        """In a round-robin, the best team should have highest SOR."""
        games = []
        # 6 teams in round-robin
        teams = [f"T{i}" for i in range(1, 7)]
        for i, t1 in enumerate(teams):
            for j, t2 in enumerate(teams):
                if i >= j:
                    continue
                date = f"2025-01-{(i*5+j):02d}"
                # Team with higher index wins by larger margin
                t1_pts = 65 + i * 3
                t2_pts = 65 + j * 3
                games.append(_make_game(t1, t2, t1_pts, t2_pts, game_date=date))
                games.append(_make_game(t2, t1, t2_pts, t1_pts, game_date=date))

        engine = ProprietaryMetricsEngine()
        results = engine.compute(games)

        # T6 should have highest SOR (most wins), T1 lowest
        sor_values = [(tid, results[tid].sor) for tid in teams]
        # T6 wins all, T5 wins 4 of 5, etc.
        assert results["T6"].sor > results["T1"].sor
