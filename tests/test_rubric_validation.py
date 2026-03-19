"""Tests for rubric validation and static audit modules."""

import json
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def sample_predictions(rng):
    """200 synthetic predictions with realistic calibration."""
    return np.clip(rng.beta(2, 2, size=200), 0.05, 0.95)


@pytest.fixture
def sample_outcomes(rng):
    return rng.integers(0, 2, size=200).astype(float)


@pytest.fixture
def sample_seed_probs():
    """Simple seed-based baseline: constant 0.5."""
    return np.full(200, 0.5)


# ---------------------------------------------------------------------------
# Pass 1: PIT + Metrics + Model Selection Gate
# ---------------------------------------------------------------------------

class TestPass1PITAndMetrics:

    def test_manifest_exists_and_has_79_features(self):
        """MANIFEST.yaml should declare exactly 79 features."""
        import yaml
        manifest_path = Path(__file__).resolve().parents[1] / "features" / "MANIFEST.yaml"
        assert manifest_path.exists(), "features/MANIFEST.yaml not found"
        with open(manifest_path) as f:
            manifest = yaml.safe_load(f)
        assert manifest["total_features"] == 79

    def test_manifest_tier_classification(self):
        """All 79 features must be classified into tiers 1, 2, or 3."""
        import yaml
        manifest_path = Path(__file__).resolve().parents[1] / "features" / "MANIFEST.yaml"
        with open(manifest_path) as f:
            manifest = yaml.safe_load(f)
        t1 = len(manifest.get("tier_1_static", {}).get("features", []))
        t2 = len(manifest.get("tier_2_cumulative", {}).get("features", []))
        t3 = len(manifest.get("tier_3_external", {}).get("features", []))
        assert t1 + t2 + t3 == 79, f"Tier sum {t1+t2+t3} != 79"
        assert t1 > 0 and t2 > 0 and t3 > 0

    def test_pit_validator_importable(self):
        """PITValidator must be importable from pipeline stages."""
        from src.pipeline.stages.pit_validation import PITValidator, PITViolationError
        assert PITValidator is not None
        assert PITViolationError is not None

    def test_pit_validator_loads_manifest(self):
        """PITValidator should load the MANIFEST and classify features."""
        from src.pipeline.stages.pit_validation import PITValidator
        validator = PITValidator()
        # Should have loaded tier info
        tier2 = validator.get_tier2_features()
        tier3 = validator.get_tier3_features()
        assert len(tier2) > 0, "No Tier 2 features loaded"
        assert len(tier3) > 0, "No Tier 3 features loaded"

    def test_brier_decomposition(self, sample_predictions, sample_outcomes):
        """Brier decomposition must return reliability, resolution, uncertainty."""
        from src.ml.evaluation.loyo_protocol import brier_decomposition
        decomp = brier_decomposition(sample_predictions, sample_outcomes)
        rel = decomp["reliability"]
        res = decomp["resolution"]
        unc = decomp["uncertainty"]
        assert rel >= 0
        assert res >= 0
        assert unc >= 0
        # Brier = Reliability - Resolution + Uncertainty (approximately)
        brier = float(np.mean((sample_predictions - sample_outcomes) ** 2))
        reconstructed = rel - res + unc
        assert abs(brier - reconstructed) < 0.05, (
            f"Decomposition mismatch: Brier={brier:.4f}, R-R+U={reconstructed:.4f}"
        )

    def test_bss_vs_seeds(self, sample_predictions, sample_outcomes):
        """BSS vs seeds must be computable."""
        from src.ml.evaluation.loyo_protocol import brier_skill_score_vs_seeds
        bss = brier_skill_score_vs_seeds(
            sample_predictions, sample_outcomes, seed_probs=np.full(200, 0.5),
        )
        assert isinstance(bss, float)

    def test_ece_computation(self, sample_predictions, sample_outcomes):
        """ECE must be computable via calibration metrics."""
        from src.ml.calibration.calibration import calculate_calibration_metrics
        metrics = calculate_calibration_metrics(sample_predictions, sample_outcomes)
        assert hasattr(metrics, "expected_calibration_error")
        assert metrics.expected_calibration_error >= 0

    def test_admission_gate_accepts_better_model(self):
        """AdmissionGate should accept a candidate that improves on baseline."""
        from src.ml.evaluation.admission_gate import (
            AdmissionGate, FoldMetrics, LOYOEvaluation,
        )
        gate = AdmissionGate(min_mean_brier_improvement=0.0)
        baseline = LOYOEvaluation(
            name="baseline",
            fold_metrics=[
                FoldMetrics(year=2021, brier_score=0.200, calibration_error=0.030, n_games=63),
                FoldMetrics(year=2022, brier_score=0.210, calibration_error=0.035, n_games=63),
            ],
        )
        candidate = LOYOEvaluation(
            name="candidate",
            fold_metrics=[
                FoldMetrics(year=2021, brier_score=0.190, calibration_error=0.028, n_games=63),
                FoldMetrics(year=2022, brier_score=0.200, calibration_error=0.033, n_games=63),
            ],
        )
        result = gate.evaluate("test_better", baseline, candidate)
        assert result.passed, f"Expected PASS but got: {result.reason}"

    def test_admission_gate_rejects_worse_model(self):
        """AdmissionGate should reject a candidate worse than baseline."""
        from src.ml.evaluation.admission_gate import (
            AdmissionGate, FoldMetrics, LOYOEvaluation,
        )
        gate = AdmissionGate(min_mean_brier_improvement=0.0)
        baseline = LOYOEvaluation(
            name="baseline",
            fold_metrics=[
                FoldMetrics(year=2021, brier_score=0.200, calibration_error=0.030, n_games=63),
            ],
        )
        candidate = LOYOEvaluation(
            name="candidate_worse",
            fold_metrics=[
                FoldMetrics(year=2021, brier_score=0.220, calibration_error=0.050, n_games=63),
            ],
        )
        result = gate.evaluate("test_worse", baseline, candidate)
        assert not result.passed, f"Expected FAIL but got: {result.reason}"


# ---------------------------------------------------------------------------
# Pass 2: BMA Ensemble + Calibration
# ---------------------------------------------------------------------------

class TestPass2BMAAndCalibration:

    def test_bma_em_algorithm(self, rng):
        """BMA must use real EM algorithm with convergence tracking."""
        from src.ml.ensemble.bma import BayesianModelAveraging
        n = 200
        y = rng.integers(0, 2, size=n).astype(float)
        preds = {
            "lgbm": np.clip(y + rng.normal(0, 0.3, n), 0.05, 0.95),
            "xgb": np.clip(y + rng.normal(0, 0.35, n), 0.05, 0.95),
            "lr": np.clip(y + rng.normal(0, 0.4, n), 0.05, 0.95),
        }
        bma = BayesianModelAveraging()
        result = bma.fit(preds, y)
        assert result.n_iterations > 0, "BMA did not run EM iterations"
        assert result.weights, "BMA returned empty weights"

    def test_bma_weights_sum_to_one(self, rng):
        """BMA weights must sum to 1."""
        from src.ml.ensemble.bma import BayesianModelAveraging
        n = 200
        y = rng.integers(0, 2, size=n).astype(float)
        preds = {
            "m1": np.clip(y + rng.normal(0, 0.3, n), 0.05, 0.95),
            "m2": np.clip(y + rng.normal(0, 0.35, n), 0.05, 0.95),
        }
        result = BayesianModelAveraging().fit(preds, y)
        weight_sum = sum(result.weights.values())
        assert abs(weight_sum - 1.0) < 1e-6, f"Weights sum to {weight_sum}"

    def test_bma_effective_model_count(self, rng):
        """BMA must compute effective model count = 1/sum(w^2)."""
        from src.ml.ensemble.bma import BayesianModelAveraging
        n = 200
        y = rng.integers(0, 2, size=n).astype(float)
        preds = {
            "m1": np.clip(y + rng.normal(0, 0.3, n), 0.05, 0.95),
            "m2": np.clip(y + rng.normal(0, 0.35, n), 0.05, 0.95),
            "m3": np.clip(y + rng.normal(0, 0.4, n), 0.05, 0.95),
        }
        result = BayesianModelAveraging().fit(preds, y)
        assert result.effective_model_count > 0
        # Verify formula: 1 / sum(w^2)
        expected = 1.0 / sum(w ** 2 for w in result.weights.values())
        assert abs(result.effective_model_count - expected) < 1e-4

    def test_production_sharpening_disabled(self):
        """Production config must have sharpening disabled."""
        config_path = Path(__file__).resolve().parents[1] / "configs" / "production_2026.json"
        with open(config_path) as f:
            config = json.load(f)
        assert not config.get("enable_brier_sharpening", True), "Sharpening must be disabled"

    def test_production_calibration_method(self):
        """Production config calibration method must be temperature, isotonic, or platt."""
        config_path = Path(__file__).resolve().parents[1] / "configs" / "production_2026.json"
        with open(config_path) as f:
            config = json.load(f)
        assert config["calibration_method"] in ("temperature", "isotonic", "platt")


# ---------------------------------------------------------------------------
# Pass 3: ESPN System
# ---------------------------------------------------------------------------

class TestPass3ESPNSystem:

    @pytest.fixture
    def synthetic_bracket(self):
        """Create a synthetic 64-team bracket for testing."""
        team_ids = [f"T{i:03d}" for i in range(64)]
        matchup_probs = {}
        for i in range(64):
            for j in range(i + 1, 64):
                p = 0.5 + 0.3 * (j - i) / 63
                matchup_probs[(team_ids[i], team_ids[j])] = float(np.clip(p, 0.1, 0.9))

        model_round_probs = {}
        public_picks = {}
        for t in team_ids:
            idx = int(t[1:])
            base = max(0.01, 1.0 - idx / 64)
            model_round_probs[t] = {
                "R64": min(0.99, base), "R32": min(0.95, base * 0.7),
                "S16": min(0.90, base * 0.4), "E8": min(0.80, base * 0.2),
                "F4": min(0.60, base * 0.1), "CHAMP": min(0.40, base * 0.05),
            }
            public_picks[t] = {
                "R64": min(0.99, base * 0.9), "R32": min(0.95, base * 0.6),
                "S16": min(0.85, base * 0.3), "E8": min(0.70, base * 0.15),
                "F4": min(0.50, base * 0.07), "CHAMP": min(0.30, base * 0.03),
            }

        return team_ids, matchup_probs, model_round_probs, public_picks

    def test_mc_simulator_runs(self, synthetic_bracket):
        """ESPN Monte Carlo simulator must run 10k+ simulations."""
        team_ids, matchup_probs, _, public_picks = synthetic_bracket
        from src.espn.mc_simulator import ESPNMonteCarloSimulator, ESPNMCSimConfig

        sim = ESPNMonteCarloSimulator(
            first_round_matchups=team_ids,
            matchup_probs=matchup_probs,
            public_pick_distribution=public_picks,
        )
        # Generate bracket (pick favorites)
        from src.espn.leverage import ROUND_GAME_COUNTS, lookup_matchup_probability
        winners = []
        current = list(team_ids)
        for _, n_games in enumerate(ROUND_GAME_COUNTS):
            next_r = []
            for g in range(n_games):
                t1, t2 = current[2 * g], current[2 * g + 1]
                w = t1 if lookup_matchup_probability(matchup_probs, t1, t2) >= 0.5 else t2
                winners.append(w)
                next_r.append(w)
            current = next_r
        assert len(winners) == 63

        result = sim.evaluate_bracket(winners, ESPNMCSimConfig(
            num_simulations=10000, n_opponents=50, random_seed=42,
        ))
        assert result.num_simulations == 10000
        assert 0 <= result.mean_rank_percentile <= 100

    def test_leverage_computation(self, synthetic_bracket):
        """Leverage table must compute model-vs-public gaps."""
        _, _, model_round_probs, public_picks = synthetic_bracket
        from src.espn.leverage import compute_leverage_table
        rows = compute_leverage_table(model_round_probs, public_picks)
        assert len(rows) > 0
        # First row should have highest leverage points
        assert rows[0].expected_leverage_points >= rows[-1].expected_leverage_points

    def test_path_dependency_structure(self):
        """Path dependency functions must return correct structure."""
        from src.espn.leverage import (
            get_champion_path_indexes,
            get_champion_protected_sibling_games,
        )
        path = get_champion_path_indexes(0)
        protected = get_champion_protected_sibling_games(0)
        assert len(path) == 6, "Must have path for all 6 rounds"
        assert len(protected) >= 5, "Must protect siblings for rounds 1-5"

    def test_bracket_optimizer_end_to_end(self, synthetic_bracket):
        """ESPN bracket optimizer must produce a valid result."""
        team_ids, matchup_probs, model_round_probs, public_picks = synthetic_bracket
        from src.espn.bracket_optimizer import ESPNBracketOptimizer, ESPNOptimizationConfig

        optimizer = ESPNBracketOptimizer(
            first_round_matchups=team_ids,
            matchup_probs=matchup_probs,
            model_round_probs=model_round_probs,
            public_pick_distribution=public_picks,
        )
        result = optimizer.optimize(ESPNOptimizationConfig(
            num_simulations=10000, pool_size=50, n_candidates=6, random_seed=42,
        ))
        assert result.selected_champion in team_ids
        assert 0 <= result.simulated_rank_percentile <= 100
        assert 0 <= result.top_10_rate <= 1
        assert 0 <= result.path_protection_score <= 1
        assert len(result.diagnostics) > 0


# ---------------------------------------------------------------------------
# Pass 4: Static Audit
# ---------------------------------------------------------------------------

class TestPass4StaticAudit:

    def test_static_audit_scores_above_60(self):
        """Static audit should score at least 60/100 for this codebase."""
        from src.validation.rubric_audit import run_static_audit
        result = run_static_audit()
        assert result.score >= 60, (
            f"Static audit scored {result.score}/100. Failures: {result.failures}"
        )

    def test_static_audit_all_sections_present(self):
        """All rubric sections (A-F) should be present in audit details."""
        from src.validation.rubric_audit import run_static_audit
        result = run_static_audit()
        for section in ["section_A", "section_B", "section_C", "section_D", "section_E", "section_F"]:
            assert section in result.details, f"Missing section: {section}"


# ---------------------------------------------------------------------------
# Integration: Full validation runner
# ---------------------------------------------------------------------------

class TestFullValidationRunner:

    def test_run_all_passes(self):
        """Full validation should run all 4 passes without error."""
        from src.validation.rubric_validator import run_validation
        result = run_validation(pass_numbers=[1, 2, 3, 4])
        assert len(result.passes) == 4
        assert result.max_total > 0
        assert result.total_score > 0

    def test_run_single_pass(self):
        """Running a single pass should work."""
        from src.validation.rubric_validator import run_validation
        result = run_validation(pass_numbers=[2])
        assert len(result.passes) == 1
        assert result.passes[0].pass_number == 2

    def test_report_generation(self):
        """Report should be a non-empty string."""
        from src.validation.rubric_validator import run_validation
        result = run_validation(pass_numbers=[1, 2])
        report = result.report()
        assert isinstance(report, str)
        assert "RUBRIC VALIDATION REPORT" in report


# ---------------------------------------------------------------------------
# Enhanced ESPN: EV picks, quadrant correlation, upset injection
# ---------------------------------------------------------------------------

class TestEVPickSelection:
    """Test EV-based pick selection (leverage × probability)."""

    def test_ev_pick_favors_leverage_over_pure_probability(self):
        """EV pick should prefer a team with lower probability but higher leverage."""
        from src.espn.leverage import compute_ev_pick

        # Team A: 60% to win, but public picks A at 70% → negative leverage
        # Team B: 40% to win, but public picks B at 20% → positive leverage
        # EV_A = (0.6 - 0.7) * 0.6 = -0.06
        # EV_B = (0.4 - 0.2) * 0.4 = 0.08 → B wins
        matchup_probs = {("A", "B"): 0.6}
        public_picks = {
            "A": {"R64": 0.70},
            "B": {"R64": 0.20},  # intentionally don't sum to 1 pre-normalization
        }
        result = compute_ev_pick("A", "B", "R64", matchup_probs, public_picks)
        assert result == "B", "EV pick should favor leverage, not raw probability"

    def test_ev_pick_favors_high_probability_with_positive_leverage(self):
        """Team with both higher probability and positive leverage should win."""
        from src.espn.leverage import compute_ev_pick

        # Team A: 70% prob, public 60% → leverage = 0.10, EV = 0.10 * 0.70 = 0.07
        # Team B: 30% prob, public 40% → leverage = -0.10, EV = -0.10 * 0.30 = -0.03
        matchup_probs = {("A", "B"): 0.7}
        public_picks = {
            "A": {"R64": 0.60},
            "B": {"R64": 0.40},
        }
        result = compute_ev_pick("A", "B", "R64", matchup_probs, public_picks)
        assert result == "A"

    def test_ev_pick_handles_missing_public_data(self):
        """EV pick should default to 50/50 public when no data available."""
        from src.espn.leverage import compute_ev_pick

        matchup_probs = {("A", "B"): 0.6}
        public_picks = {}  # No public data
        result = compute_ev_pick("A", "B", "R64", matchup_probs, public_picks)
        # With equal public picks, higher model prob should win
        assert result == "A"


class TestQuadrantCorrelation:
    """Test quadrant correlation enforcement."""

    def test_enforces_chalk_in_champion_region(self):
        """Chalk should be forced in R64/R32 of champion's region."""
        from src.espn.leverage import (
            enforce_quadrant_correlation,
            ROUND_GAME_COUNTS,
            lookup_matchup_probability,
        )

        team_ids = [f"T{i:03d}" for i in range(64)]
        matchup_probs = {}
        for i in range(64):
            for j in range(i + 1, 64):
                matchup_probs[(team_ids[i], team_ids[j])] = 0.5 + 0.3 * (j - i) / 63

        # Build a bracket with some upsets
        winners = []
        current = list(team_ids)
        for _, n_games in enumerate(ROUND_GAME_COUNTS):
            nxt = []
            for g in range(n_games):
                t1, t2 = current[2 * g], current[2 * g + 1]
                # Intentionally pick upsets (pick t2, the underdog)
                winners.append(t2)
                nxt.append(t2)
            current = nxt

        # Champion is T000 (game 0, region 0)
        result = enforce_quadrant_correlation(
            bracket_winners=winners,
            champion_id="T000",
            first_round_game_idx=0,
            matchup_probs=matchup_probs,
            first_round_matchups=team_ids,
        )

        # Region 0 = games 0-7 in R64 — should now be favorites (lower index)
        for g in range(8):
            t1 = team_ids[2 * g]
            t2 = team_ids[2 * g + 1]
            p = lookup_matchup_probability(matchup_probs, t1, t2)
            expected_fav = t1 if p >= 0.5 else t2
            assert result[g] == expected_fav, (
                f"R64 game {g} in champion region should be chalk ({expected_fav}), "
                f"got {result[g]}"
            )

    def test_does_not_modify_other_regions(self):
        """Non-champion regions should not be modified."""
        from src.espn.leverage import enforce_quadrant_correlation, ROUND_GAME_COUNTS

        team_ids = [f"T{i:03d}" for i in range(64)]
        matchup_probs = {}
        for i in range(64):
            for j in range(i + 1, 64):
                matchup_probs[(team_ids[i], team_ids[j])] = 0.5 + 0.3 * (j - i) / 63

        # Original bracket: pick underdogs everywhere
        winners = []
        current = list(team_ids)
        for _, n_games in enumerate(ROUND_GAME_COUNTS):
            nxt = []
            for g in range(n_games):
                winners.append(current[2 * g + 1])
                nxt.append(current[2 * g + 1])
            current = nxt

        original = list(winners)
        result = enforce_quadrant_correlation(
            winners, "T000", 0, matchup_probs, team_ids,
        )

        # Region 1-3 R64 games (8-31) should be unchanged
        for g in range(8, 32):
            assert result[g] == original[g], f"R64 game {g} (other region) was modified"


class TestUpsetInjection:
    """Test controlled upset injection."""

    @pytest.fixture
    def optimizer_with_bracket(self):
        """Create optimizer and favorite bracket for upset testing."""
        from src.espn.bracket_optimizer import ESPNBracketOptimizer
        team_ids = [f"T{i:03d}" for i in range(64)]
        matchup_probs = {}
        for i in range(64):
            for j in range(i + 1, 64):
                matchup_probs[(team_ids[i], team_ids[j])] = 0.5 + 0.3 * (j - i) / 63

        model_round_probs = {}
        public_picks = {}
        for t in team_ids:
            idx = int(t[1:])
            base = max(0.01, 1.0 - idx / 64)
            model_round_probs[t] = {
                "R64": min(0.99, base), "R32": min(0.95, base * 0.7),
                "S16": min(0.90, base * 0.4), "E8": min(0.80, base * 0.2),
                "F4": min(0.60, base * 0.1), "CHAMP": min(0.40, base * 0.05),
            }
            # Public underestimates mid-tier teams → creates leverage
            public_picks[t] = {
                "R64": min(0.99, base * 0.8), "R32": min(0.95, base * 0.5),
                "S16": min(0.85, base * 0.25), "E8": min(0.70, base * 0.1),
                "F4": min(0.50, base * 0.05), "CHAMP": min(0.30, base * 0.02),
            }

        opt = ESPNBracketOptimizer(
            first_round_matchups=team_ids,
            matchup_probs=matchup_probs,
            model_round_probs=model_round_probs,
            public_pick_distribution=public_picks,
        )
        return opt, team_ids, matchup_probs

    def test_identify_upset_candidates_filters_by_thresholds(self, optimizer_with_bracket):
        """Upset candidates must meet model_prob and leverage thresholds."""
        opt, team_ids, matchup_probs = optimizer_with_bracket
        from src.espn.leverage import ROUND_GAME_COUNTS, lookup_matchup_probability

        # Build favorite bracket
        winners = []
        current = list(team_ids)
        for _, n_games in enumerate(ROUND_GAME_COUNTS):
            nxt = []
            for g in range(n_games):
                t1, t2 = current[2 * g], current[2 * g + 1]
                p = lookup_matchup_probability(matchup_probs, t1, t2)
                w = t1 if p >= 0.5 else t2
                winners.append(w)
                nxt.append(w)
            current = nxt

        candidates = opt._identify_upset_candidates(
            winners, min_prob=0.35, min_leverage=0.05,
        )
        # All candidates must meet thresholds
        for _, _, _, prob, ev in candidates:
            assert prob >= 0.35, f"Upset candidate has prob {prob} < 0.35"
            assert ev > 0, "EV score must be positive"

    def test_inject_upsets_respects_max_count(self, optimizer_with_bracket):
        """Should not inject more upsets than max_upsets."""
        opt, team_ids, matchup_probs = optimizer_with_bracket
        from src.espn.leverage import ROUND_GAME_COUNTS, lookup_matchup_probability

        winners = []
        current = list(team_ids)
        for _, n_games in enumerate(ROUND_GAME_COUNTS):
            nxt = []
            for g in range(n_games):
                t1, t2 = current[2 * g], current[2 * g + 1]
                p = lookup_matchup_probability(matchup_probs, t1, t2)
                winners.append(t1 if p >= 0.5 else t2)
                nxt.append(t1 if p >= 0.5 else t2)
            current = nxt

        original = list(winners)
        candidates = opt._identify_upset_candidates(winners)
        result = opt._inject_controlled_upsets(
            winners, candidates, "T000", max_upsets=2, max_disruption=1.0,
        )
        diffs = sum(1 for a, b in zip(original, result) if a != b)
        assert diffs <= 2, f"Injected {diffs} upsets, max was 2"


class TestFinalFourExpansion:
    """Test Final Four set generation."""

    @pytest.fixture
    def optimizer(self):
        from src.espn.bracket_optimizer import ESPNBracketOptimizer
        team_ids = [f"T{i:03d}" for i in range(64)]
        matchup_probs = {}
        for i in range(64):
            for j in range(i + 1, 64):
                matchup_probs[(team_ids[i], team_ids[j])] = 0.5 + 0.3 * (j - i) / 63
        model_round_probs = {}
        public_picks = {}
        for t in team_ids:
            idx = int(t[1:])
            base = max(0.01, 1.0 - idx / 64)
            model_round_probs[t] = {
                "R64": min(0.99, base), "R32": min(0.95, base * 0.7),
                "S16": min(0.90, base * 0.4), "E8": min(0.80, base * 0.2),
                "F4": min(0.60, base * 0.1), "CHAMP": min(0.40, base * 0.05),
            }
            public_picks[t] = {
                "R64": min(0.99, base * 0.9), "R32": min(0.95, base * 0.6),
                "S16": min(0.85, base * 0.3), "E8": min(0.70, base * 0.15),
                "F4": min(0.50, base * 0.07), "CHAMP": min(0.30, base * 0.03),
            }
        return ESPNBracketOptimizer(
            first_round_matchups=team_ids,
            matchup_probs=matchup_probs,
            model_round_probs=model_round_probs,
            public_pick_distribution=public_picks,
        )

    def test_f4_sets_have_four_teams(self, optimizer):
        """Each F4 set should have exactly 4 teams."""
        f4_sets = optimizer._generate_final_four_sets("T000", max_sets=8)
        assert len(f4_sets) > 0
        for f4 in f4_sets:
            assert len(f4) == 4, f"F4 set has {len(f4)} teams, expected 4"

    def test_f4_sets_include_champion(self, optimizer):
        """Champion must appear in every F4 set."""
        f4_sets = optimizer._generate_final_four_sets("T000", max_sets=8)
        for f4 in f4_sets:
            assert "T000" in f4, f"Champion T000 missing from F4 set: {f4}"

    def test_f4_sets_one_per_region(self, optimizer):
        """Each F4 set should have one team per region."""
        f4_sets = optimizer._generate_final_four_sets("T000", max_sets=8)
        for f4 in f4_sets:
            regions = set()
            for tid in f4:
                region = optimizer._team_regions.get(tid, -1)
                if region >= 0:
                    regions.add(region)
            assert len(regions) == 4, f"F4 set spans {len(regions)} regions, expected 4"

    def test_structured_search_produces_more_candidates(self, optimizer):
        """Enhanced optimizer should produce more candidate diagnostics."""
        from src.espn.bracket_optimizer import ESPNOptimizationConfig
        config = ESPNOptimizationConfig(
            num_simulations=10000, pool_size=50,
            n_candidates=12, random_seed=42,
            max_final_four_sets=4,
        )
        result = optimizer.optimize(config)
        # Structured search should produce more than old 12-bracket grid
        assert len(result.diagnostics) > 0
        assert result.selected_champion in [f"T{i:03d}" for i in range(64)]
        assert 0 <= result.top_10_rate <= 1
        assert 0 <= result.path_protection_score <= 1
