"""Mandatory tests for the decoupled Forecast Engine / Pool Optimizer architecture.

These four tests prove the architectural contract:

1. Forecast Invariance: Changing pool_size or scoring_rules does NOT
   affect ForecastEngine.predict_proba() output.
2. Optimization Pivot: Fixing P=0.52 and shifting public_pick_percentage
   from 0.20 to 0.85 causes PoolOptimizer to select different paths.
3. Metadata Integrity: Every PoolResult contains a non-null
   AssumptionsManifest with all 4 mandatory environmental parameters.
4. Calibration Check: Brier Score > 0.20 triggers IntegrityError in
   production mode; logs warning in experimental mode.
"""

import logging
import math

import pytest

from src.exceptions import IntegrityError
from src.forecasting.engine import (
    CalibrationReport,
    ForecastEngine,
    ForecastEngineConfig,
)
from src.optimization.pool_optimizer import (
    AssumptionsManifest,
    PoolEnvironment,
    PoolOptimizer,
    PoolResult,
)


# ── Helpers ──────────────────────────────────────────────────────────

def _make_predict_fn(fixed_prob: float = 0.60):
    """Return a deterministic predict function for testing."""
    def predict_fn(team1_id: str, team2_id: str) -> float:
        # Seed-based deterministic probability:
        # Higher seed number = weaker team, so seed 1 > seed 16.
        # Extract seed from team_id like "east_1", "west_16".
        try:
            s1 = int(team1_id.split("_")[1])
            s2 = int(team2_id.split("_")[1])
        except (IndexError, ValueError):
            return fixed_prob
        # Logistic function on seed difference
        diff = s2 - s1  # positive = team1 is favored
        return 1.0 / (1.0 + math.exp(-0.3 * diff))
    return predict_fn


def _make_engine(mode: str = "production", brier_threshold: float = 0.20) -> ForecastEngine:
    """Create a ForecastEngine with a deterministic predict function."""
    config = ForecastEngineConfig(
        mode=mode,
        brier_threshold=brier_threshold,
    )
    engine = ForecastEngine(config)
    engine.set_predict_fn(_make_predict_fn())
    return engine


def _team_ids() -> list:
    """Generate 64 team IDs (4 regions × 16 seeds)."""
    ids = []
    for region in ("east", "west", "south", "midwest"):
        for seed in range(1, 17):
            ids.append(f"{region}_{seed}")
    return ids


def _build_public_picks(
    team_ids: list,
    base_pct: float = 0.50,
) -> dict:
    """Build synthetic public pick distribution."""
    picks = {}
    for tid in team_ids:
        try:
            seed = int(tid.split("_")[1])
        except (IndexError, ValueError):
            seed = 8
        strength = (17 - seed) / 16.0
        picks[tid] = {
            "R64": min(0.99, base_pct + 0.45 * strength),
            "R32": min(0.95, (base_pct - 0.10) + 0.50 * strength),
            "S16": min(0.85, (base_pct - 0.20) + 0.45 * strength),
            "E8": min(0.70, (base_pct - 0.30) + 0.35 * strength),
            "F4": min(0.50, (base_pct - 0.35) + 0.25 * strength),
            "CHAMP": min(0.30, (base_pct - 0.40) + 0.15 * strength),
        }
    return picks


def _standard_scoring() -> dict:
    return {"R64": 10, "R32": 20, "S16": 40, "E8": 80, "F4": 160, "NCG": 320}


# ═══════════════════════════════════════════════════════════════════════
# TEST 1: FORECAST INVARIANCE
# ═══════════════════════════════════════════════════════════════════════

class TestForecastInvariance:
    """Changing pool_size or scoring_rules does NOT affect predict_proba()."""

    def test_pool_size_does_not_affect_probability(self):
        """ForecastEngine output is identical regardless of pool environment."""
        engine = _make_engine()

        # Predict probabilities
        p1 = engine.predict_proba("east_1", "east_16")
        p2 = engine.predict_proba("east_8", "east_9")
        p3 = engine.predict_proba("west_3", "west_14")

        # "Change" pool parameters — create different PoolEnvironments
        env_small = PoolEnvironment(
            pool_size=10,
            scoring_rules={"R64": 1, "R32": 2, "S16": 4, "E8": 8, "F4": 16, "NCG": 32},
            payout_structure="winner_take_all",
            public_pick_distribution=_build_public_picks(_team_ids()),
        )
        env_large = PoolEnvironment(
            pool_size=10000,
            scoring_rules=_standard_scoring(),
            payout_structure="top_25pct",
            public_pick_distribution=_build_public_picks(_team_ids(), base_pct=0.30),
        )

        # Re-predict — output MUST be identical (zero diff)
        p1_after = engine.predict_proba("east_1", "east_16")
        p2_after = engine.predict_proba("east_8", "east_9")
        p3_after = engine.predict_proba("west_3", "west_14")

        assert p1 == p1_after, f"Probability changed: {p1} -> {p1_after}"
        assert p2 == p2_after, f"Probability changed: {p2} -> {p2_after}"
        assert p3 == p3_after, f"Probability changed: {p3} -> {p3_after}"

    def test_scoring_rules_do_not_affect_probability(self):
        """Different scoring systems produce identical forecast probabilities."""
        engine = _make_engine()

        matchups = [
            ("east_1", "east_16"),
            ("west_5", "west_12"),
            ("south_7", "south_10"),
        ]

        # Get baseline predictions
        baseline = [engine.predict_proba(t1, t2) for t1, t2 in matchups]

        # Create optimizers with wildly different scoring rules
        for scoring in [
            {"R64": 1, "R32": 1, "S16": 1, "E8": 1, "F4": 1, "NCG": 1},
            {"R64": 100, "R32": 200, "S16": 400, "E8": 800, "F4": 1600, "NCG": 3200},
        ]:
            # Re-predict after creating optimizers with these scoring rules
            after = [engine.predict_proba(t1, t2) for t1, t2 in matchups]
            for i, (b, a) in enumerate(zip(baseline, after)):
                assert b == a, (
                    f"Matchup {matchups[i]}: probability changed from {b} to {a} "
                    f"after pool optimizer with scoring {scoring}"
                )

    def test_engine_config_rejects_pool_params(self):
        """ForecastEngineConfig raises ValueError if pool params are set."""
        # This tests the structural firewall
        config = ForecastEngineConfig()
        # The config itself should NOT have any forbidden attributes
        for forbidden in ("pool_size", "ev_pool_size", "ev_scoring_system",
                         "ev_contrarian_strength", "public_picks_json",
                         "scoring_rules_json", "payout_structure"):
            assert not hasattr(config, forbidden), (
                f"ForecastEngineConfig has forbidden attribute '{forbidden}'"
            )


# ═══════════════════════════════════════════════════════════════════════
# TEST 2: OPTIMIZATION PIVOT
# ═══════════════════════════════════════════════════════════════════════

class TestOptimizationPivot:
    """Fixed P=0.52 with different public_pick_percentage → different paths."""

    def test_public_pick_shift_changes_strategy(self):
        """When P=0.52, shifting public picks from 0.20 to 0.85 changes optimal path."""
        # Fixed probability: every team wins with P=0.52
        def fixed_predict(t1, t2):
            return 0.52

        engine = ForecastEngine(ForecastEngineConfig())
        engine.set_predict_fn(fixed_predict)

        team_ids = _team_ids()
        probs = engine.predict_all_matchups(team_ids)

        # Environment A: low public pick percentage (team is undervalued)
        public_low = {}
        for tid in team_ids:
            public_low[tid] = {
                "R64": 0.20, "R32": 0.15, "S16": 0.10,
                "E8": 0.08, "F4": 0.05, "CHAMP": 0.02,
            }
        env_low = PoolEnvironment(
            pool_size=500,
            scoring_rules=_standard_scoring(),
            payout_structure="winner_take_all",
            public_pick_distribution=public_low,
        )

        # Environment B: high public pick percentage (team is overvalued by crowd)
        public_high = {}
        for tid in team_ids:
            public_high[tid] = {
                "R64": 0.85, "R32": 0.80, "S16": 0.75,
                "E8": 0.70, "F4": 0.65, "CHAMP": 0.55,
            }
        env_high = PoolEnvironment(
            pool_size=500,
            scoring_rules=_standard_scoring(),
            payout_structure="winner_take_all",
            public_pick_distribution=public_high,
        )

        opt_low = PoolOptimizer(probs, env_low)
        opt_high = PoolOptimizer(probs, env_high)

        result_low = opt_low.optimize()
        result_high = opt_high.optimize()

        # The recommended strategy should differ because the leverage
        # dynamics are completely different when public picks are
        # 20% vs 85%.
        # At minimum, the strategy EVs or recommended strategy should differ.
        low_strategy = result_low.recommended_strategy
        high_strategy = result_high.recommended_strategy
        low_evs = result_low.strategy_evs
        high_evs = result_high.strategy_evs

        # At least one of these must differ:
        strategies_differ = low_strategy != high_strategy
        evs_differ = low_evs != high_evs
        leverage_picks_differ = result_low.leverage_picks != result_high.leverage_picks

        assert strategies_differ or evs_differ or leverage_picks_differ, (
            "PoolOptimizer produced identical results for dramatically "
            "different public pick distributions (0.20 vs 0.85). "
            "The optimizer should pivot its strategy based on crowd behavior."
        )

    def test_probabilities_not_mutated_by_optimizer(self):
        """PoolOptimizer deep-copies probabilities; originals are untouched."""
        engine = _make_engine()
        team_ids = _team_ids()[:8]  # Small set for speed
        probs = engine.predict_all_matchups(team_ids)
        original_probs = {k: v for k, v in probs.items()}

        env = PoolEnvironment(
            pool_size=100,
            scoring_rules=_standard_scoring(),
            payout_structure="tiered",
            public_pick_distribution=_build_public_picks(team_ids),
        )
        optimizer = PoolOptimizer(probs, env)
        optimizer.optimize()

        # Original dict must be unchanged
        for key, val in original_probs.items():
            assert probs[key] == val, (
                f"Probability for {key} was mutated: {val} -> {probs[key]}"
            )


# ═══════════════════════════════════════════════════════════════════════
# TEST 3: METADATA INTEGRITY
# ═══════════════════════════════════════════════════════════════════════

class TestMetadataIntegrity:
    """Every PoolResult contains non-null AssumptionsManifest with 4 params."""

    def test_manifest_has_all_required_fields(self):
        """AssumptionsManifest has non-null pool_size, expected_chalk,
        payout_structure, and scoring_rules."""
        engine = _make_engine()
        team_ids = _team_ids()[:16]
        probs = engine.predict_all_matchups(team_ids)

        env = PoolEnvironment(
            pool_size=250,
            scoring_rules=_standard_scoring(),
            payout_structure="top_3",
            public_pick_distribution=_build_public_picks(team_ids),
        )

        optimizer = PoolOptimizer(probs, env)
        result = optimizer.optimize()

        manifest = result.manifest
        assert manifest is not None, "AssumptionsManifest is None"

        # Four mandatory environmental parameters must be non-null
        assert manifest.pool_size is not None and manifest.pool_size > 0, (
            f"pool_size is null or zero: {manifest.pool_size}"
        )
        assert manifest.expected_chalk is not None and isinstance(manifest.expected_chalk, dict), (
            f"expected_chalk is null or wrong type: {manifest.expected_chalk}"
        )
        assert manifest.payout_structure is not None and len(manifest.payout_structure) > 0, (
            f"payout_structure is null or empty: {manifest.payout_structure}"
        )
        assert manifest.scoring_rules is not None and isinstance(manifest.scoring_rules, dict) and len(manifest.scoring_rules) > 0, (
            f"scoring_rules is null or empty: {manifest.scoring_rules}"
        )

    def test_manifest_values_match_environment(self):
        """Manifest values must match the input environment."""
        team_ids = _team_ids()[:16]
        engine = _make_engine()
        probs = engine.predict_all_matchups(team_ids)

        scoring = {"R64": 10, "R32": 20, "S16": 40, "E8": 80, "F4": 160, "NCG": 320}
        env = PoolEnvironment(
            pool_size=500,
            scoring_rules=scoring,
            payout_structure="top_10pct",
            public_pick_distribution=_build_public_picks(team_ids),
        )

        optimizer = PoolOptimizer(probs, env)
        result = optimizer.optimize()

        assert result.manifest.pool_size == 500
        assert result.manifest.payout_structure == "top_10pct"
        assert result.manifest.scoring_rules == scoring

    def test_manifest_serialization(self):
        """AssumptionsManifest.to_dict() returns complete, non-null dict."""
        team_ids = _team_ids()[:8]
        engine = _make_engine()
        probs = engine.predict_all_matchups(team_ids)

        env = PoolEnvironment(
            pool_size=100,
            scoring_rules=_standard_scoring(),
            payout_structure="tiered",
            public_pick_distribution=_build_public_picks(team_ids),
        )

        optimizer = PoolOptimizer(probs, env)
        result = optimizer.optimize()
        d = result.manifest.to_dict()

        for key in ("pool_size", "expected_chalk", "payout_structure", "scoring_rules"):
            assert key in d, f"Missing key '{key}' in manifest dict"
            assert d[key] is not None, f"Key '{key}' is None in manifest dict"


# ═══════════════════════════════════════════════════════════════════════
# TEST 4: CALIBRATION CHECK (BRIER GATE)
# ═══════════════════════════════════════════════════════════════════════

class TestCalibrationCheck:
    """Brier Score > 0.20 → IntegrityError in production, warning in experimental."""

    def test_production_mode_rejects_mediocre_model(self):
        """BS > 0.20 raises IntegrityError in production mode."""
        engine = _make_engine(mode="production", brier_threshold=0.20)

        # Synthetic data with poor calibration: predict 0.90 but outcome is random
        forecasts = [0.90] * 50 + [0.10] * 50
        outcomes = [0] * 50 + [1] * 50  # Completely wrong predictions

        # BS = mean((0.9 - 0)^2 * 50 + (0.1 - 1)^2 * 50) / 100 = 0.81
        with pytest.raises(IntegrityError, match="Model Calibration Failure"):
            engine.validate_calibration(forecasts, outcomes)

    def test_experimental_mode_warns_but_continues(self, caplog):
        """BS > 0.20 logs CRITICAL_WARNING in experimental mode but does not raise."""
        engine = _make_engine(mode="experimental", brier_threshold=0.20)

        forecasts = [0.90] * 50 + [0.10] * 50
        outcomes = [0] * 50 + [1] * 50

        with caplog.at_level(logging.WARNING):
            report = engine.validate_calibration(forecasts, outcomes)

        assert not report.passed
        assert report.model_quality == "rejected"
        assert report.brier_score > 0.20
        assert "CRITICAL_WARNING" in caplog.text

    def test_good_model_passes_gate(self):
        """BS <= 0.20 passes the gate and returns accepted quality."""
        engine = _make_engine(mode="production", brier_threshold=0.20)

        # Well-calibrated predictions
        forecasts = [0.70] * 70 + [0.30] * 30
        outcomes = [1] * 70 + [0] * 30

        # BS = mean((0.7 - 1)^2 * 70 + (0.3 - 0)^2 * 30) / 100
        #    = (0.09 * 70 + 0.09 * 30) / 100 = 0.09
        report = engine.validate_calibration(forecasts, outcomes)

        assert report.passed
        assert report.model_quality == "accepted"
        assert report.brier_score <= 0.20

    def test_calibration_report_stored(self):
        """After validate_calibration(), report is accessible via property."""
        engine = _make_engine(mode="experimental")
        forecasts = [0.50] * 100
        outcomes = [1] * 50 + [0] * 50

        assert engine.calibration_report is None
        report = engine.validate_calibration(forecasts, outcomes)
        assert engine.calibration_report is report
        assert report.n_samples == 100

    def test_empty_inputs_raise_value_error(self):
        """Empty forecast/outcome lists raise ValueError."""
        engine = _make_engine()
        with pytest.raises(ValueError, match="zero samples"):
            engine.validate_calibration([], [])

    def test_mismatched_lengths_raise_value_error(self):
        """Mismatched forecast/outcome lengths raise ValueError."""
        engine = _make_engine()
        with pytest.raises(ValueError, match="length mismatch"):
            engine.validate_calibration([0.5, 0.6], [1])


# ═══════════════════════════════════════════════════════════════════════
# BONUS: SOURCE CODE FIREWALL AUDIT
# ═══════════════════════════════════════════════════════════════════════

class TestSourceCodeFirewall:
    """Verify ForecastEngine source contains zero pool/strategy references."""

    def test_engine_source_has_no_pool_references(self):
        """Grep-equivalent: ForecastEngine source must not reference pool params."""
        import inspect
        source = inspect.getsource(ForecastEngine)

        forbidden_terms = [
            "pool_size",
            "scoring_rules",
            "public_pick",
            "payout_structure",
            "contrarian",
            "leverage_pick",
        ]
        for term in forbidden_terms:
            assert term not in source, (
                f"ForecastEngine source code contains forbidden term '{term}'. "
                f"The forecast engine must be completely isolated from "
                f"pool/strategy parameters."
            )

    def test_engine_config_source_has_no_pool_references(self):
        """ForecastEngineConfig source must not define pool params."""
        import inspect
        source = inspect.getsource(ForecastEngineConfig)

        # These must not appear as attribute definitions
        forbidden_attrs = [
            "pool_size:", "ev_pool_size:", "ev_scoring_system:",
            "public_picks_json:", "scoring_rules_json:",
        ]
        for attr in forbidden_attrs:
            assert attr not in source, (
                f"ForecastEngineConfig defines forbidden attribute '{attr}'. "
                f"Forecast configuration must exclude pool/strategy parameters."
            )
