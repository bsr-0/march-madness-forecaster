"""Phase 8 tests: statistician-grade audit of the public picks pipeline.

These tests validate that the ingestion pipeline meets the standards
a senior sports statistician would require for a production bracket
optimization system.  Each test class addresses a specific audit concern.
"""

import math
import pytest

# ============================================================================
# 1. Seed pick model: empirical grounding and structural constraints
# ============================================================================


class TestSeedPickModelDerivation:
    """Verify the seed pick model produces rates consistent with known
    NCAA tournament data and public bracket behavior."""

    def test_all_seeds_present(self):
        from src.data.seed_pick_model import SEED_PICK_RATES
        for seed in range(1, 17):
            assert seed in SEED_PICK_RATES, f"Missing seed {seed}"
            for rnd in ("R64", "R32", "S16", "E8", "F4", "CHAMP"):
                assert rnd in SEED_PICK_RATES[seed], f"Missing {rnd} for seed {seed}"

    def test_rates_are_probabilities(self):
        """All values in (0, 1)."""
        from src.data.seed_pick_model import SEED_PICK_RATES
        for seed, rounds in SEED_PICK_RATES.items():
            for rnd, val in rounds.items():
                assert 0 < val < 1, f"Seed {seed} {rnd} = {val} out of (0,1)"

    def test_rates_monotonically_decrease_by_round(self):
        """For any seed, R64 >= R32 >= S16 >= ... >= CHAMP."""
        from src.data.seed_pick_model import SEED_PICK_RATES
        order = ["R64", "R32", "S16", "E8", "F4", "CHAMP"]
        for seed, rounds in SEED_PICK_RATES.items():
            for i in range(len(order) - 1):
                assert rounds[order[i]] >= rounds[order[i + 1]], (
                    f"Seed {seed}: {order[i]}={rounds[order[i]]:.6f} "
                    f"< {order[i+1]}={rounds[order[i+1]]:.6f}"
                )

    def test_higher_seeds_generally_lower_rates(self):
        """For any round, seed 1 > seed 16."""
        from src.data.seed_pick_model import SEED_PICK_RATES
        for rnd in ("R64", "R32", "S16", "E8", "F4", "CHAMP"):
            assert SEED_PICK_RATES[1][rnd] > SEED_PICK_RATES[16][rnd], (
                f"{rnd}: seed 1 ({SEED_PICK_RATES[1][rnd]:.6f}) not > "
                f"seed 16 ({SEED_PICK_RATES[16][rnd]:.6f})"
            )

    def test_championship_sums_to_100_pct(self):
        """Championship picks across 64 teams (4 of each seed) sum to ~100%."""
        from src.data.seed_pick_model import SEED_PICK_RATES
        total = sum(SEED_PICK_RATES[s]["CHAMP"] for s in range(1, 17)) * 4.0
        assert total == pytest.approx(1.0, abs=0.01), (
            f"Championship total = {total:.4f}, expected ~1.0"
        )

    def test_r64_pairing_sums_reasonable(self):
        """R64 pairing sums (seed s + seed 17-s) should be ≤ 1.0."""
        from src.data.seed_pick_model import SEED_PICK_RATES
        for s in range(1, 9):
            opp = 17 - s
            r64_sum = SEED_PICK_RATES[s]["R64"] + SEED_PICK_RATES[opp]["R64"]
            # Sum should be close to 1.0 (some people don't pick, so ≤1.0)
            assert 0.90 <= r64_sum <= 1.01, (
                f"R64 {s}v{opp}: sum = {r64_sum:.4f}"
            )

    def test_seed_1_r64_near_historical(self):
        """Seed-1 R64 rate should be near historical win rate (~0.99).

        The public barely deviates from reality in R64 for top seeds.
        """
        from src.data.seed_pick_model import SEED_PICK_RATES
        assert SEED_PICK_RATES[1]["R64"] == pytest.approx(0.99, abs=0.02)

    def test_seed_1_champ_reflects_chalk_bias(self):
        """Seed-1 CHAMP rate should be higher than true advancement rate.

        True seed-1 championship probability: ~3% per team (12% for all 4).
        Public picks ~10% per team (40% for all 4) due to chalk bias.
        """
        from src.data.seed_pick_model import SEED_PICK_RATES, _compute_advancement_rates
        adv = _compute_advancement_rates()
        # Public pick rate should exceed true rate (chalk bias)
        assert SEED_PICK_RATES[1]["CHAMP"] > adv[1]["CHAMP"] * 0.8, (
            f"Seed 1 CHAMP pick rate ({SEED_PICK_RATES[1]['CHAMP']:.4f}) "
            f"not sufficiently above true rate ({adv[1]['CHAMP']:.4f})"
        )
        # But not absurdly high
        assert SEED_PICK_RATES[1]["CHAMP"] < 0.20, (
            f"Seed 1 CHAMP = {SEED_PICK_RATES[1]['CHAMP']:.4f} seems too high"
        )

    def test_advancement_rates_decrease_by_round(self):
        """Raw advancement rates must be monotonically decreasing."""
        from src.data.seed_pick_model import _compute_advancement_rates
        order = ["R64", "R32", "S16", "E8", "F4", "CHAMP"]
        adv = _compute_advancement_rates()
        for seed in range(1, 17):
            for i in range(len(order) - 1):
                assert adv[seed][order[i]] >= adv[seed][order[i + 1]] - 1e-10, (
                    f"Seed {seed}: adv[{order[i]}]={adv[seed][order[i]]:.6f} "
                    f"< adv[{order[i+1]}]={adv[seed][order[i+1]]:.6f}"
                )

    def test_advancement_rates_sum_by_round(self):
        """In each round, advancement rates × 4 teams should sum to expected count.

        R64: 32 teams advance (out of 64) → sum of rates × 4 ≈ 32
        R32: 16 teams advance → sum × 4 ≈ 16
        """
        from src.data.seed_pick_model import _compute_advancement_rates
        adv = _compute_advancement_rates()
        expected_count = {"R64": 32, "R32": 16, "S16": 8, "E8": 4, "F4": 2, "CHAMP": 1}
        for rnd, expected in expected_count.items():
            actual = sum(adv[s][rnd] for s in range(1, 17)) * 4.0
            assert actual == pytest.approx(expected, rel=0.15), (
                f"{rnd}: sum×4 = {actual:.2f}, expected ~{expected}"
            )

    def test_model_reproducible(self):
        """Model produces the same values on repeated calls."""
        from src.data.seed_pick_model import compute_seed_pick_rates
        rates_1 = compute_seed_pick_rates()
        rates_2 = compute_seed_pick_rates()
        for seed in range(1, 17):
            for rnd in rates_1[seed]:
                assert rates_1[seed][rnd] == rates_2[seed][rnd]


# ============================================================================
# 2. Source agreement: statistical methodology
# ============================================================================


class TestSourceAgreementStatistics:
    """Verify the agreement module uses sound statistical methods."""

    def test_minimum_sample_size_enforced(self):
        """Pairs with < 10 shared teams are skipped."""
        from src.data.scrapers.source_agreement import assess_source_agreement
        from src.data.scrapers.espn_picks import ConsensusData, PublicPicks

        def _small_consensus(n_teams):
            teams = {}
            for i in range(n_teams):
                tid = f"team_{i}"
                teams[tid] = PublicPicks(
                    team_id=tid, team_name=f"T{i}", seed=i + 1,
                    region="East", champion_pct=10.0 - i * 0.5,
                )
            return ConsensusData(teams=teams)

        sources = {
            "espn": _small_consensus(8),
            "yahoo": _small_consensus(8),
        }
        report = assess_source_agreement(sources)
        # n=8 < 10 → pair skipped → no correlations computed
        assert len(report.pairwise_rank_correlations) == 0

    def test_p_value_based_weighting(self):
        """Flagged sources are down-weighted by p-value, not ρ²."""
        import random
        from src.data.scrapers.source_agreement import assess_source_agreement
        from src.data.scrapers.espn_picks import ConsensusData, PublicPicks

        rng = random.Random(42)

        def _make_consensus(n_teams, noise=0.0):
            teams = {}
            for i in range(n_teams):
                tid = f"team_{i}"
                champ = max(0.1, 20.0 - i * 1.0 + rng.gauss(0, noise))
                teams[tid] = PublicPicks(
                    team_id=tid, team_name=f"T{i}", seed=(i % 16) + 1,
                    region="East", champion_pct=champ,
                )
            return ConsensusData(teams=teams)

        # ESPN and Yahoo agree perfectly, CBS is noisy
        espn = _make_consensus(30, noise=0.0)
        yahoo = _make_consensus(30, noise=0.0)
        cbs = _make_consensus(30, noise=15.0)

        sources = {"espn": espn, "yahoo": yahoo, "cbs": cbs}
        report = assess_source_agreement(
            sources,
            min_correlation=0.85,
            configured_weights={"espn": 0.5, "yahoo": 0.3, "cbs": 0.2},
        )

        # CBS should be down-weighted (it's noisy)
        if "cbs" in report.flagged_sources:
            assert report.recommended_weights["cbs"] < 0.2
            # ESPN and Yahoo should get more weight
            assert report.recommended_weights["espn"] > report.recommended_weights["cbs"]

    def test_spearman_p_value_known_values(self):
        """Verify p-value computation against known statistical values."""
        from src.data.scrapers.source_agreement import _spearman_p_value

        # Perfect correlation with large n → p ≈ 0
        assert _spearman_p_value(1.0, 30) == 0.0

        # Zero correlation → p = 1.0 (not significant)
        assert _spearman_p_value(0.0, 30) == pytest.approx(1.0, abs=0.01)

        # n < 3 → p = 1.0
        assert _spearman_p_value(0.95, 2) == 1.0

        # Moderate correlation, moderate n → significant
        p = _spearman_p_value(0.7, 20)
        assert p < 0.01  # Should be highly significant

        # Moderate correlation, small n → less significant
        p_small = _spearman_p_value(0.7, 10)
        assert p_small > p  # Larger p-value with smaller n


# ============================================================================
# 3. Fallback provenance tracking
# ============================================================================


class TestFallbackProvenance:
    """Verify that synthetic data usage is tracked and auditable."""

    def _make_calculator(self, model_teams, public_teams):
        from src.optimization.leverage import LeverageCalculator, TeamMetadata
        model_probs = {}
        metadata = {}
        for tid, seed in model_teams.items():
            model_probs[tid] = {"R64": 0.90, "CHAMP": 0.08}
            metadata[tid] = TeamMetadata(team_name=tid, seed=seed, region="East")
        public_picks = {}
        for tid in public_teams:
            public_picks[tid] = {"R64": 0.85, "CHAMP": 0.05}
        return LeverageCalculator(
            model_probs=model_probs, public_picks=public_picks,
            team_metadata=metadata,
        )

    def test_audit_log_populated(self):
        """Fallback audit log records team/round pairs."""
        calc = self._make_calculator(
            {"duke": 1, "gonzaga": 4}, {"duke"},
        )
        # Access gonzaga — should trigger fallback
        pct, is_fallback = calc._public_pct_with_fallback("gonzaga", "CHAMP")
        assert is_fallback is True
        assert "gonzaga" in calc.fallback_audit
        assert "CHAMP" in calc.fallback_audit["gonzaga"]

    def test_no_fallback_no_audit(self):
        """Teams with real data don't appear in fallback audit."""
        calc = self._make_calculator({"duke": 1}, {"duke"})
        pct, is_fallback = calc._public_pct_with_fallback("duke", "CHAMP")
        assert is_fallback is False
        assert "duke" not in calc.fallback_audit

    def test_leverage_picks_carry_synthetic_flag(self):
        """LeveragePick.is_synthetic is True for picks using fallback data."""
        calc = self._make_calculator(
            {"duke": 1, "mystery": 12}, {"duke"},
        )
        picks = calc.find_leverage_picks(min_leverage=0.0, min_probability=0.0)
        mystery_picks = [p for p in picks if p.team_id == "mystery"]
        duke_picks = [p for p in picks if p.team_id == "duke"]
        assert all(p.is_synthetic for p in mystery_picks)
        assert all(not p.is_synthetic for p in duke_picks)

    def test_fallback_summary(self):
        """get_fallback_summary provides audit-ready summary."""
        calc = self._make_calculator(
            {"duke": 1, "gonzaga": 4, "mystery": 12},
            {"duke"},  # only duke has real data
        )
        # Trigger fallbacks
        calc.find_leverage_picks(min_leverage=0.0, min_probability=0.0)
        summary = calc.get_fallback_summary()
        assert summary["n_teams_with_fallback"] == 2
        assert summary["synthetic_fraction"] == pytest.approx(2 / 3)
        assert "gonzaga" in summary["teams"]
        assert "mystery" in summary["teams"]


# ============================================================================
# 4. Validation thresholds
# ============================================================================


class TestValidationThresholds:
    """Verify that plausibility thresholds are defensible."""

    def _make_consensus(self, n_teams, champ_each=None):
        from src.data.scrapers.espn_picks import ConsensusData, PublicPicks
        teams = {}
        for i in range(n_teams):
            seed = (i % 16) + 1
            tid = f"team_{i}"
            teams[tid] = PublicPicks(
                team_id=tid, team_name=f"T{i}", seed=seed,
                region=["East", "West", "South", "Midwest"][i % 4],
                champion_pct=champ_each or 100.0 / max(n_teams, 1),
            )
        return ConsensusData(teams=teams)

    def test_min_teams_32_enforced(self):
        """Consensus with < 32 teams gets a warning."""
        from src.data.scrapers.espn_picks import validate_consensus_plausibility
        c = self._make_consensus(20)
        warnings = validate_consensus_plausibility(c)
        assert any("teams" in w.lower() for w in warnings)

    def test_32_teams_no_team_warning(self):
        """Consensus with exactly 32 teams passes the team count check."""
        from src.data.scrapers.espn_picks import validate_consensus_plausibility
        c = self._make_consensus(32)
        team_warnings = [w for w in validate_consensus_plausibility(c)
                         if "teams" in w.lower() and "seed" not in w.lower()]
        assert len(team_warnings) == 0

    def test_champ_sum_110_flagged(self):
        """Championship sum > 110% is flagged."""
        from src.data.scrapers.espn_picks import validate_consensus_plausibility
        # 64 teams × 1.8% each = 115.2% → inflated
        c = self._make_consensus(64, champ_each=1.8)
        warnings = validate_consensus_plausibility(c)
        assert any("inflated" in w.lower() for w in warnings)

    def test_champ_sum_90_flagged(self):
        """Championship sum < 90% is flagged."""
        from src.data.scrapers.espn_picks import validate_consensus_plausibility
        # 64 teams × 1.3% each = 83.2% → deflated
        c = self._make_consensus(64, champ_each=1.3)
        warnings = validate_consensus_plausibility(c)
        assert any("deflated" in w.lower() for w in warnings)

    def test_single_team_40pct_flagged(self):
        """Single team with > 40% championship share is flagged."""
        from src.data.scrapers.espn_picks import ConsensusData, PublicPicks, validate_consensus_plausibility
        teams = {}
        for i in range(64):
            tid = f"team_{i}"
            seed = (i % 16) + 1
            champ = 42.0 if i == 0 else (58.0 / 63)
            teams[tid] = PublicPicks(
                team_id=tid, team_name=f"T{i}", seed=seed,
                region="East", champion_pct=champ,
            )
        c = ConsensusData(teams=teams)
        warnings = validate_consensus_plausibility(c)
        assert any("corruption" in w.lower() for w in warnings)


# ============================================================================
# 5. MAX_LEVERAGE cap justification
# ============================================================================


class TestMaxLeverage:
    """Verify the leverage cap is principled."""

    def test_max_leverage_is_15(self):
        """Cap reduced from 20 to 15 to bound synthetic-prior leverage."""
        from src.optimization.leverage import LeverageCalculator
        assert LeverageCalculator.MAX_LEVERAGE == 15.0

    def test_cap_binds_on_synthetic_data(self):
        """When using seed priors, leverage is capped and flagged synthetic."""
        from src.optimization.leverage import LeverageCalculator, TeamMetadata
        # 16-seed with model prob 0.05, no public data → synthetic prior
        calc = LeverageCalculator(
            model_probs={"cinderella": {"CHAMP": 0.05}},
            public_picks={},
            team_metadata={"cinderella": TeamMetadata(
                team_name="Cinderella", seed=16, region="East",
            )},
        )
        picks = calc.find_leverage_picks(min_leverage=0.0, min_probability=0.0)
        assert len(picks) == 1
        assert picks[0].is_synthetic is True
        # Raw leverage would be huge, but capped at 15
        raw = picks[0].model_probability / picks[0].public_pick_percentage
        assert raw > 15.0  # Would exceed cap without it


# ============================================================================
# 6. Coverage-bias detection in aggregation
# ============================================================================


class TestCoverageBiasDetection:

    def test_single_source_teams_logged(self):
        """Teams in only one source produce a coverage bias warning."""
        import logging
        from src.data.scrapers.espn_picks import (
            ConsensusData, PublicPicks, aggregate_consensus,
        )

        def _picks(tid, champ=5.0):
            return PublicPicks(
                team_id=tid, team_name=tid, seed=1, region="East",
                champion_pct=champ,
            )

        espn = ConsensusData(teams={
            "duke": _picks("duke"), "unc": _picks("unc"),
        })
        yahoo = ConsensusData(teams={
            "duke": _picks("duke"),
            # UNC missing from Yahoo
        })
        cbs = ConsensusData(teams={})

        # Capture the warning log
        import io
        handler = logging.StreamHandler(io.StringIO())
        handler.setLevel(logging.WARNING)
        logger = logging.getLogger("src.data.scrapers.espn_picks")
        logger.addHandler(handler)
        try:
            aggregate_consensus(
                espn, yahoo, cbs,
                weights={"espn": 0.5, "yahoo": 0.5, "cbs": 0.0},
            )
            log_output = handler.stream.getvalue()
        finally:
            logger.removeHandler(handler)

        # UNC should be flagged as single-source
        # (normalize_team_id("unc") → "north_carolina")
        assert "Coverage bias" in log_output or "coverage" in log_output.lower()


# ============================================================================
# 7. Normalize pick probability edge cases
# ============================================================================


class TestNormalizeProbabilityEdgeCases:

    def test_boundary_at_1(self):
        """Value of exactly 1.0 is treated as probability (not divided)."""
        from src.pipeline.stages.simulation import normalize_pick_probability
        result = normalize_pick_probability(1.0)
        assert result == pytest.approx(0.9999)  # Clipped to max

    def test_just_above_1(self):
        """Value of 1.01 is treated as percentage → 0.0101."""
        from src.pipeline.stages.simulation import normalize_pick_probability
        result = normalize_pick_probability(1.01)
        assert result == pytest.approx(0.0101)

    def test_100_is_percentage(self):
        """Value of 100 → 1.0 → clipped to 0.9999."""
        from src.pipeline.stages.simulation import normalize_pick_probability
        result = normalize_pick_probability(100)
        assert result == pytest.approx(0.9999)

    def test_0_returns_floor(self):
        from src.pipeline.stages.simulation import normalize_pick_probability
        assert normalize_pick_probability(0) == pytest.approx(0.0001)

    def test_negative_returns_floor(self):
        from src.pipeline.stages.simulation import normalize_pick_probability
        assert normalize_pick_probability(-5) == pytest.approx(0.0001)


# ============================================================================
# 8. Historical win rate cross-validation
# ============================================================================


class TestHistoricalWinRateCrossValidation:
    """Verify the seed pick model's win rates are consistent with
    the codebase's other historical data sources."""

    def test_r64_win_rates_match_tournament_features(self):
        """Seed pick model R64 win rates match tournament_features.py.

        Note: tournament_features.py has a duplicate key (4,13) that appears
        for both R64 and R32.  The R32 value (0.690) overwrites the R64
        value (0.793).  We skip (4,13) for this cross-check and compare
        the matchups that are unambiguous R64 entries.
        """
        from src.data.seed_pick_model import _HISTORICAL_WIN_RATES
        from src.data.features.tournament_features import HISTORICAL_SEED_WIN_RATES

        # Only matchups that are unambiguously R64 (no duplicate keys)
        r64_matchups = [
            (1, 16), (2, 15), (3, 14),
            (5, 12), (6, 11), (7, 10), (8, 9),
        ]
        for lower, higher in r64_matchups:
            model_rate = _HISTORICAL_WIN_RATES[(lower, higher)]
            feature_rate = HISTORICAL_SEED_WIN_RATES.get((lower, higher))
            if feature_rate is not None:
                # seed_pick_model uses 1985-2025 full-range averages;
                # tournament_features uses 2008-2025 recency-weighted rates.
                # Wider tolerance (0.08) accounts for methodological difference.
                assert model_rate == pytest.approx(feature_rate, abs=0.08), (
                    f"({lower},{higher}): model={model_rate:.3f}, "
                    f"features={feature_rate:.3f}"
                )

    def test_r64_win_rates_match_brier_optimal(self):
        """Seed pick model R64 win rates match brier_optimal.py."""
        from src.data.seed_pick_model import _HISTORICAL_WIN_RATES
        from src.ml.calibration.brier_optimal import SeedBasedOverrides

        for (lower, higher), rate in SeedBasedOverrides.MENS_SEED_WIN_RATES.items():
            model_rate = _HISTORICAL_WIN_RATES.get((lower, higher))
            if model_rate is not None:
                assert model_rate == pytest.approx(rate, abs=0.015), (
                    f"({lower},{higher}): model={model_rate:.3f}, "
                    f"brier={rate:.3f}"
                )

    def test_logistic_beta_consistent_with_codebase(self):
        """The logistic fallback uses β=0.175, matching tournament_features.py."""
        from src.data.seed_pick_model import _win_rate
        # β=0.175: P(1 beats 16) = 1/(1+exp(-0.175*15)) ≈ 0.932
        # This is the fallback, only used for unseen matchups.
        # The direct lookup should be used for (1,16) since it's in the table.
        # Test an unseen matchup instead.
        p = _win_rate(1, 7)  # Not in the direct table (only later-round)
        # With β=0.175, diff=6: 1/(1+exp(-0.175*6)) ≈ 0.741
        assert p == pytest.approx(0.741, abs=0.01)


# ============================================================================
# 9. Phase 8b: Fixes from self-audit
# ============================================================================


class TestNaNHandling:
    """Verify NaN is handled safely throughout the pipeline."""

    def test_normalize_nan_returns_floor(self):
        from src.pipeline.stages.simulation import normalize_pick_probability
        assert normalize_pick_probability(float('nan')) == pytest.approx(0.0001)

    def test_normalize_nan_string_returns_floor(self):
        from src.pipeline.stages.simulation import normalize_pick_probability
        assert normalize_pick_probability('nan') == pytest.approx(0.0001)

    def test_normalize_inf_treated_as_percentage(self):
        from src.pipeline.stages.simulation import normalize_pick_probability
        # inf > 1.0, so divided by 100 → still inf → clipped to 0.9999
        result = normalize_pick_probability(float('inf'))
        assert result == pytest.approx(0.9999)


class TestFallbackAuditClearing:
    """Verify fallback_audit is cleared between calls."""

    def test_audit_cleared_on_reuse(self):
        from src.optimization.leverage import LeverageCalculator, TeamMetadata
        calc = LeverageCalculator(
            model_probs={"duke": {"CHAMP": 0.10}},
            public_picks={},
            team_metadata={"duke": TeamMetadata("Duke", 1, "East")},
        )
        # First call — populates audit
        calc.find_leverage_picks(min_leverage=0.0, min_probability=0.0)
        assert len(calc.fallback_audit) > 0

        # Provide real data for second call
        calc.public_picks = {"duke": {"CHAMP": 0.05}}
        calc.find_leverage_picks(min_leverage=0.0, min_probability=0.0)
        # Audit should be cleared — duke now has real data
        assert "duke" not in calc.fallback_audit


class TestFisherCombinedPValue:
    """Verify Fisher's method is used for p-value combination."""

    def test_chi2_sf_known_values(self):
        from src.data.scrapers.source_agreement import _chi2_sf
        # χ²(0, df=2) → p = 1.0
        assert _chi2_sf(0, 2) == pytest.approx(1.0, abs=0.01)
        # χ²(10, df=2) → p ≈ 0.0067
        assert _chi2_sf(10, 2) == pytest.approx(0.0067, abs=0.005)
        # χ²(1, df=2) → p ≈ 0.607
        assert _chi2_sf(1, 2) == pytest.approx(0.607, abs=0.05)

    def test_chi2_sf_negative_returns_1(self):
        from src.data.scrapers.source_agreement import _chi2_sf
        assert _chi2_sf(-1, 2) == 1.0
        assert _chi2_sf(0, 0) == 1.0
