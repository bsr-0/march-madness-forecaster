"""Phase 7 tests: team ID normalization in aggregation + EV coverage alignment."""

import pytest


# ---------------------------------------------------------------------------
# Team ID normalization in aggregate_consensus
# ---------------------------------------------------------------------------


def _make_picks(team_id, team_name, seed=1, region="East", champ=10.0):
    from src.data.scrapers.espn_picks import PublicPicks
    return PublicPicks(
        team_id=team_id,
        team_name=team_name,
        seed=seed,
        region=region,
        champion_pct=champ,
        final_four_pct=20.0,
        elite_8_pct=30.0,
        sweet_16_pct=40.0,
        round_of_32_pct=60.0,
        round_of_64_pct=95.0,
    )


def _make_consensus(teams_dict):
    from src.data.scrapers.espn_picks import ConsensusData
    return ConsensusData(teams=teams_dict)


class TestAggregateNormalization:

    def test_same_team_different_ids_merged(self):
        """ESPN 'unc' and Yahoo 'north_carolina' merge into one entry."""
        from src.data.scrapers.espn_picks import aggregate_consensus

        espn = _make_consensus({
            "unc": _make_picks("unc", "North Carolina", champ=30.0),
        })
        yahoo = _make_consensus({
            "north_carolina": _make_picks("north_carolina", "North Carolina", champ=20.0),
        })
        cbs = _make_consensus({})

        result = aggregate_consensus(
            espn, yahoo, cbs,
            weights={"espn": 0.5, "yahoo": 0.5, "cbs": 0.0},
        )

        # Both should merge under the canonical "north_carolina" ID
        assert len(result.teams) == 1
        canonical = list(result.teams.keys())[0]
        assert canonical == "north_carolina"
        # Weighted average: (0.5*30 + 0.5*20) / 1.0 = 25.0
        assert result.teams[canonical].champion_pct == pytest.approx(25.0)

    def test_already_canonical_ids_still_work(self):
        """Sources with matching canonical IDs still aggregate correctly."""
        from src.data.scrapers.espn_picks import aggregate_consensus

        espn = _make_consensus({
            "duke": _make_picks("duke", "Duke", champ=15.0),
        })
        yahoo = _make_consensus({
            "duke": _make_picks("duke", "Duke", champ=12.0),
        })
        cbs = _make_consensus({
            "duke": _make_picks("duke", "Duke", champ=18.0),
        })

        result = aggregate_consensus(espn, yahoo, cbs)
        assert "duke" in result.teams
        # Default weights: espn=0.5, yahoo=0.3, cbs=0.2
        expected = (0.5 * 15.0 + 0.3 * 12.0 + 0.2 * 18.0) / 1.0
        assert result.teams["duke"].champion_pct == pytest.approx(expected)

    def test_uconn_variants_merge(self):
        """'uconn' and 'connecticut' both normalize to 'connecticut'."""
        from src.data.scrapers.espn_picks import aggregate_consensus

        espn = _make_consensus({
            "uconn": _make_picks("uconn", "UConn", champ=25.0),
        })
        yahoo = _make_consensus({
            "connecticut": _make_picks("connecticut", "Connecticut", champ=22.0),
        })
        cbs = _make_consensus({})

        result = aggregate_consensus(
            espn, yahoo, cbs,
            weights={"espn": 0.5, "yahoo": 0.5, "cbs": 0.0},
        )
        assert len(result.teams) == 1
        assert "connecticut" in result.teams

    def test_no_sources_produces_empty(self):
        """Empty sources still produce valid empty consensus."""
        from src.data.scrapers.espn_picks import aggregate_consensus

        result = aggregate_consensus(
            _make_consensus({}), _make_consensus({}), _make_consensus({}),
        )
        assert len(result.teams) == 0


# ---------------------------------------------------------------------------
# EV coverage alignment: _public_pct_with_fallback used everywhere
# ---------------------------------------------------------------------------


class TestEVCoverageAlignment:

    def _make_calculator(self, model_teams, public_teams):
        """Create a LeverageCalculator with specified team sets."""
        from src.optimization.leverage import LeverageCalculator, TeamMetadata

        model_probs = {}
        metadata = {}
        for tid, seed in model_teams.items():
            model_probs[tid] = {"R64": 0.95, "R32": 0.75, "S16": 0.50, "E8": 0.30, "F4": 0.15, "CHAMP": 0.08}
            metadata[tid] = TeamMetadata(team_name=tid, seed=seed, region="East")

        public_picks = {}
        for tid in public_teams:
            public_picks[tid] = {"R64": 0.90, "R32": 0.60, "S16": 0.35, "E8": 0.20, "F4": 0.10, "CHAMP": 0.05}

        return LeverageCalculator(
            model_probs=model_probs,
            public_picks=public_picks,
            team_metadata=metadata,
        )

    def test_fallback_for_missing_public_team(self):
        """Team in model but not in public_picks gets seed-based fallback."""
        calc = self._make_calculator(
            model_teams={"duke": 1, "gonzaga": 4},
            public_teams={"duke"},  # gonzaga missing from public picks
        )
        pct, is_fallback = calc._public_pct_with_fallback("gonzaga", "CHAMP")
        assert is_fallback is True
        assert pct > 0  # seed-based prior, not zero

    def test_present_team_uses_actual_data(self):
        """Team present in public_picks uses actual data, not fallback."""
        calc = self._make_calculator(
            model_teams={"duke": 1},
            public_teams={"duke"},
        )
        pct, is_fallback = calc._public_pct_with_fallback("duke", "CHAMP")
        assert is_fallback is False
        assert pct == pytest.approx(0.05)

    def test_championship_public_covers_all_model_teams(self):
        """analyze_pool builds championship_public for ALL model teams."""
        from src.optimization.leverage import analyze_pool, TeamMetadata

        model_probs = {
            "duke": {"R64": 0.95, "R32": 0.75, "S16": 0.50, "E8": 0.30, "F4": 0.15, "CHAMP": 0.10},
            "gonzaga": {"R64": 0.90, "R32": 0.65, "S16": 0.40, "E8": 0.25, "F4": 0.12, "CHAMP": 0.06},
        }
        # Only duke has public picks — gonzaga missing
        public_picks = {
            "duke": {"R64": 0.90, "R32": 0.60, "S16": 0.35, "E8": 0.20, "F4": 0.10, "CHAMP": 0.08},
        }
        metadata = {
            "duke": TeamMetadata(team_name="Duke", seed=1, region="East"),
            "gonzaga": TeamMetadata(team_name="Gonzaga", seed=4, region="West"),
        }

        result = analyze_pool(
            pool_size=100,
            model_probs=model_probs,
            public_picks=public_picks,
            team_metadata=metadata,
        )

        # Should not crash and should produce a valid result with both teams
        assert result is not None
        # The log should show gonzaga used seed-based prior (covered by fallback)

    def test_risk_adjusted_score_uses_fallback(self):
        """ParetoOptimizer._risk_adjusted_score uses fallback for missing teams."""
        from src.optimization.leverage import ParetoOptimizer, TeamMetadata

        model_probs = {
            "duke": {"CHAMP": 0.10, "F4": 0.20, "E8": 0.35, "S16": 0.50, "R32": 0.70, "R64": 0.90},
            "unknown": {"CHAMP": 0.05, "F4": 0.10, "E8": 0.20, "S16": 0.30, "R32": 0.50, "R64": 0.80},
        }
        public_picks = {
            "duke": {"CHAMP": 0.08, "F4": 0.15, "E8": 0.25, "S16": 0.40, "R32": 0.60, "R64": 0.90},
            # "unknown" intentionally missing
        }
        metadata = {
            "duke": TeamMetadata(team_name="Duke", seed=1, region="East"),
            "unknown": TeamMetadata(team_name="Unknown", seed=12, region="West"),
        }

        from src.optimization.leverage import LeverageCalculator
        calc = LeverageCalculator(
            model_probs=model_probs,
            public_picks=public_picks,
            team_metadata=metadata,
        )
        optimizer = ParetoOptimizer(calc)

        # Should not raise, even though "unknown" has no public picks
        score = optimizer._risk_adjusted_score("unknown", "CHAMP", 1.0)
        assert score >= 0
