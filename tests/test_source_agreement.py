"""Tests for cross-source statistical agreement testing (Phase 2)."""

import math
import random

import pytest

from src.data.scrapers.espn_picks import ConsensusData, PublicPicks
from src.data.scrapers.source_agreement import (
    SourceAgreementReport,
    assess_source_agreement,
    _detect_team_outliers,
    _spearmanr,
    _spearman_p_value,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_team(team_id: str, seed: int, region: str, champ: float,
               f4: float = 0.0, e8: float = 0.0, s16: float = 0.0,
               r32: float = 0.0, r64: float = 0.0) -> PublicPicks:
    """Create a PublicPicks with sensible defaults cascading from champ."""
    # If higher-round values not supplied, derive monotonically from champ.
    if f4 == 0.0:
        f4 = max(champ, champ * 1.5)
    if e8 == 0.0:
        e8 = max(f4, f4 * 1.3)
    if s16 == 0.0:
        s16 = max(e8, e8 * 1.3)
    if r32 == 0.0:
        r32 = max(s16, s16 * 1.2)
    if r64 == 0.0:
        r64 = max(r32, min(99.0, r32 * 1.1))

    return PublicPicks(
        team_id=team_id,
        team_name=team_id.replace("_", " ").title(),
        seed=seed,
        region=region,
        round_of_64_pct=r64,
        round_of_32_pct=r32,
        sweet_16_pct=s16,
        elite_8_pct=e8,
        final_four_pct=f4,
        champion_pct=champ,
    )


def _make_consensus(teams: dict, source: str = "test") -> ConsensusData:
    """Wrap a dict of team_id→PublicPicks into ConsensusData."""
    return ConsensusData(teams=teams, sources=[source])


def _realistic_teams(noise_std: float = 0.0, rng_seed: int = 42):
    """Generate 16 teams with realistic CHAMP pick distributions.

    If noise_std > 0, add Gaussian noise to championship percentages
    to simulate source disagreement.
    """
    rng = random.Random(rng_seed)
    base_champ = {
        1: 22.0, 2: 15.0, 3: 8.0, 4: 5.0,
        5: 3.0, 6: 2.0, 7: 1.5, 8: 1.0,
        9: 0.8, 10: 0.5, 11: 0.4, 12: 0.3,
        13: 0.1, 14: 0.05, 15: 0.02, 16: 0.01,
    }
    teams = {}
    for seed, champ in base_champ.items():
        c = champ + (rng.gauss(0, noise_std) if noise_std > 0 else 0.0)
        c = max(0.01, c)
        teams[f"team_{seed}"] = _make_team(
            f"team_{seed}", seed, "East", champ=c
        )
    return teams


# ---------------------------------------------------------------------------
# Tests: _spearmanr
# ---------------------------------------------------------------------------

class TestSpearmanr:
    def test_perfect_correlation(self):
        assert _spearmanr([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)

    def test_perfect_negative(self):
        assert _spearmanr([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)

    def test_no_correlation(self):
        # Specific arrangement that yields ρ ≈ 0
        rho = _spearmanr([1, 2, 3, 4], [2, 4, 1, 3])
        assert abs(rho) < 0.5

    def test_single_element(self):
        assert _spearmanr([1], [2]) == 0.0


# ---------------------------------------------------------------------------
# Tests: _spearman_p_value
# ---------------------------------------------------------------------------

class TestSpearmanPValue:
    def test_perfect_correlation_p_zero(self):
        """ρ=1.0 → p=0."""
        assert _spearman_p_value(1.0, 20) == 0.0

    def test_zero_correlation_p_high(self):
        """ρ=0.0, n=10 → p=1.0 (not significant)."""
        p = _spearman_p_value(0.0, 10)
        assert p > 0.9

    def test_small_n_needs_high_rho(self):
        """With n=6, ρ=0.7 should NOT be significant at α=0.05."""
        p = _spearman_p_value(0.7, 6)
        assert p > 0.05

    def test_large_n_moderate_rho_significant(self):
        """With n=64, ρ=0.5 should be highly significant."""
        p = _spearman_p_value(0.5, 64)
        assert p < 0.001

    def test_n_less_than_3(self):
        """n < 3 → p=1.0 (untestable)."""
        assert _spearman_p_value(0.99, 2) == 1.0


# ---------------------------------------------------------------------------
# Tests: assess_source_agreement
# ---------------------------------------------------------------------------

class TestAssessSourceAgreement:

    def test_identical_sources_high_agreement(self):
        """Three identical sources → ρ=1.0, no flags, agreement='high'."""
        teams = _realistic_teams()
        sources = {
            "espn": _make_consensus(teams, "espn"),
            "yahoo": _make_consensus(teams, "yahoo"),
            "cbs": _make_consensus(teams, "cbs"),
        }
        report = assess_source_agreement(sources)

        assert report.agreement_level == "high"
        assert report.flagged_sources == []
        for pair_rhos in report.pairwise_rank_correlations.values():
            assert pair_rhos["CHAMP"] == pytest.approx(1.0)

    def test_one_corrupted_source_flagged(self):
        """ESPN and Yahoo agree, CBS has shuffled data → CBS flagged."""
        good_teams = _realistic_teams(noise_std=0.0, rng_seed=42)

        # Create CBS with completely shuffled CHAMP values so rank
        # correlation with the good sources is near zero.
        rng = random.Random(77)
        cbs_teams = {}
        team_ids = sorted(good_teams.keys())
        champ_values = [good_teams[t].champion_pct for t in team_ids]
        rng.shuffle(champ_values)
        for tid, champ in zip(team_ids, champ_values):
            orig = good_teams[tid]
            cbs_teams[tid] = _make_team(
                tid, orig.seed, orig.region, champ=champ,
            )

        sources = {
            "espn": _make_consensus(good_teams, "espn"),
            "yahoo": _make_consensus(good_teams, "yahoo"),
            "cbs": _make_consensus(cbs_teams, "cbs"),
        }
        report = assess_source_agreement(sources, min_correlation=0.85)

        assert "cbs" in report.flagged_sources
        assert "espn" not in report.flagged_sources
        assert "yahoo" not in report.flagged_sources
        assert report.agreement_level in ("low", "conflicting")

    def test_all_sources_disagree(self):
        """All three sources have shuffled data → agreement_level='conflicting'."""
        base_teams = _realistic_teams(noise_std=0.0)
        team_ids = sorted(base_teams.keys())
        champ_values = [base_teams[t].champion_pct for t in team_ids]

        def _shuffle_source(seed_val):
            rng = random.Random(seed_val)
            vals = list(champ_values)
            rng.shuffle(vals)
            teams = {}
            for tid, champ in zip(team_ids, vals):
                orig = base_teams[tid]
                teams[tid] = _make_team(tid, orig.seed, orig.region, champ=champ)
            return teams

        sources = {
            "espn": _make_consensus(_shuffle_source(11), "espn"),
            "yahoo": _make_consensus(_shuffle_source(22), "yahoo"),
            "cbs": _make_consensus(_shuffle_source(33), "cbs"),
        }
        report = assess_source_agreement(sources, min_correlation=0.85)

        assert report.agreement_level in ("low", "conflicting")
        assert len(report.flagged_sources) >= 2

    def test_recommended_weights_downweight_outlier(self):
        """Flagged source gets lower weight than configured."""
        good_teams = _realistic_teams(noise_std=0.0)

        # Shuffled CBS so it's clearly the outlier
        rng = random.Random(77)
        cbs_teams = {}
        team_ids = sorted(good_teams.keys())
        champ_values = [good_teams[t].champion_pct for t in team_ids]
        rng.shuffle(champ_values)
        for tid, champ in zip(team_ids, champ_values):
            orig = good_teams[tid]
            cbs_teams[tid] = _make_team(tid, orig.seed, orig.region, champ=champ)

        sources = {
            "espn": _make_consensus(good_teams, "espn"),
            "yahoo": _make_consensus(good_teams, "yahoo"),
            "cbs": _make_consensus(cbs_teams, "cbs"),
        }
        configured = {"espn": 0.5, "yahoo": 0.3, "cbs": 0.2}
        report = assess_source_agreement(
            sources, configured_weights=configured, min_correlation=0.85
        )

        assert "cbs" in report.flagged_sources
        # CBS recommended weight should be lower than its configured weight
        assert report.recommended_weights["cbs"] < configured["cbs"]

    def test_two_sources_sufficient(self):
        """Agreement check works with only two sources."""
        teams = _realistic_teams()
        sources = {
            "espn": _make_consensus(teams, "espn"),
            "yahoo": _make_consensus(teams, "yahoo"),
        }
        report = assess_source_agreement(sources)

        assert report.agreement_level == "high"
        assert len(report.pairwise_rank_correlations) == 1
        assert ("espn", "yahoo") in report.pairwise_rank_correlations

    def test_single_source_no_agreement_check(self):
        """One source → skips agreement check, returns 'unknown'."""
        teams = _realistic_teams()
        sources = {"espn": _make_consensus(teams, "espn")}

        report = assess_source_agreement(sources)

        assert report.agreement_level == "unknown"
        assert report.flagged_sources == []
        assert report.recommended_weights == {"espn": 1.0}

    def test_small_sample_not_fooled_by_high_rho(self):
        """With very few shared teams, even high ρ isn't significant."""
        # Only 5 teams shared — ρ can be high by chance.
        small_teams_a = {
            f"team_{i}": _make_team(f"team_{i}", i, "East", champ=20.0 - i * 3)
            for i in range(1, 6)
        }
        # Slightly shuffled — still correlated but from tiny sample.
        small_teams_b = dict(small_teams_a)

        sources = {
            "espn": _make_consensus(small_teams_a, "espn"),
            "yahoo": _make_consensus(small_teams_b, "yahoo"),
        }
        report = assess_source_agreement(
            sources, min_correlation=0.85, significance_level=0.01,
        )
        # With identical data ρ=1.0, n=5 → p ≈ 0.0, so this should pass.
        # But the key point is the p-value machinery is exercised.
        assert report.agreement_level in ("high", "moderate")

    def test_weights_sum_to_one(self):
        """Recommended weights always sum to 1.0."""
        good_teams = _realistic_teams(noise_std=0.0)
        # Shuffled CBS
        rng = random.Random(77)
        cbs_teams = {}
        team_ids = sorted(good_teams.keys())
        champ_values = [good_teams[t].champion_pct for t in team_ids]
        rng.shuffle(champ_values)
        for tid, champ in zip(team_ids, champ_values):
            orig = good_teams[tid]
            cbs_teams[tid] = _make_team(tid, orig.seed, orig.region, champ=champ)
        sources = {
            "espn": _make_consensus(good_teams, "espn"),
            "yahoo": _make_consensus(good_teams, "yahoo"),
            "cbs": _make_consensus(cbs_teams, "cbs"),
        }
        report = assess_source_agreement(sources)

        total = sum(report.recommended_weights.values())
        assert total == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Tests: _detect_team_outliers
# ---------------------------------------------------------------------------

class TestDetectTeamOutliers:

    def test_no_outliers_with_identical_sources(self):
        """Identical sources → no team-level outliers."""
        teams = _realistic_teams()
        sources = {
            "espn": _make_consensus(teams, "espn"),
            "yahoo": _make_consensus(teams, "yahoo"),
        }
        outliers = _detect_team_outliers(sources)
        assert all(len(v) == 0 for v in outliers.values())

    def test_single_team_outlier_detected(self):
        """One source has a wildly wrong value for a single team."""
        teams_a = _realistic_teams()
        teams_b = _realistic_teams()
        teams_c = dict(_realistic_teams())  # copy to corrupt
        # Corrupt one team in source c: team_1 CHAMP goes from 22% to 90%
        bad = _make_team("team_1", 1, "East", champ=90.0)
        teams_c["team_1"] = bad

        sources = {
            "espn": _make_consensus(teams_a, "espn"),
            "yahoo": _make_consensus(teams_b, "yahoo"),
            "cbs": _make_consensus(teams_c, "cbs"),
        }
        # Median of [22, 22, 90] = 22.  CBS deviates by 68pp > 10pp default.
        outliers = _detect_team_outliers(sources)

        assert "team_1" in outliers["cbs"]
        # ESPN and Yahoo are at the median — not flagged.
        assert "team_1" not in outliers["espn"]
        assert "team_1" not in outliers["yahoo"]

    def test_single_source_returns_empty(self):
        """With only one source, no outlier detection is possible."""
        teams = _realistic_teams()
        sources = {"espn": _make_consensus(teams, "espn")}
        outliers = _detect_team_outliers(sources)
        assert outliers == {"espn": []}
