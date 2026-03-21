"""Tests for the quant decision system."""

import pytest


# ---------------------------------------------------------------------------
# QuantConfig tests
# ---------------------------------------------------------------------------


class TestQuantConfig:
    def test_default_config(self):
        from src.quant.config import QuantConfig

        config = QuantConfig()
        assert config.year == 2026
        assert config.pool_sizes == [20, 100, 1000]
        assert config.simulations == 50000
        assert config.calibration_method == "isotonic"
        assert config.alpha == 1.0
        assert config.beta == 1.0

    def test_beta_scaling_small_pool(self):
        from src.quant.config import QuantConfig

        config = QuantConfig(beta=1.0)
        assert config.get_beta_for_pool_size(10) == pytest.approx(0.3)
        assert config.get_beta_for_pool_size(20) == pytest.approx(0.3)

    def test_beta_scaling_medium_pool(self):
        from src.quant.config import QuantConfig

        config = QuantConfig(beta=1.0)
        assert config.get_beta_for_pool_size(50) == pytest.approx(1.0)
        assert config.get_beta_for_pool_size(200) == pytest.approx(1.0)

    def test_beta_scaling_large_pool(self):
        from src.quant.config import QuantConfig

        config = QuantConfig(beta=1.0)
        assert config.get_beta_for_pool_size(1000) == pytest.approx(2.0)
        assert config.get_beta_for_pool_size(5000) == pytest.approx(2.0)

    def test_beta_scaling_custom_base(self):
        from src.quant.config import QuantConfig

        config = QuantConfig(beta=2.0)
        assert config.get_beta_for_pool_size(10) == pytest.approx(0.6)
        assert config.get_beta_for_pool_size(100) == pytest.approx(2.0)
        assert config.get_beta_for_pool_size(1000) == pytest.approx(4.0)

    def test_strategy_labels(self):
        from src.quant.config import QuantConfig

        config = QuantConfig()
        assert config.get_strategy_label(10) == "conservative"
        assert config.get_strategy_label(20) == "conservative"
        assert config.get_strategy_label(100) == "balanced"
        assert config.get_strategy_label(200) == "balanced"
        assert config.get_strategy_label(1000) == "aggressive"


# ---------------------------------------------------------------------------
# Kaggle layer tests
# ---------------------------------------------------------------------------


class TestKaggleTiering:
    def test_elite_tier(self):
        from src.quant.kaggle_layer import assign_tier

        assert assign_tier(0.50) == "elite"
        assert assign_tier(0.53) == "elite"

    def test_strong_tier(self):
        from src.quant.kaggle_layer import assign_tier

        assert assign_tier(0.54) == "strong"
        assert assign_tier(0.55) == "strong"

    def test_average_tier(self):
        from src.quant.kaggle_layer import assign_tier

        assert assign_tier(0.56) == "average"
        assert assign_tier(0.58) == "average"

    def test_below_average_tier(self):
        from src.quant.kaggle_layer import assign_tier

        assert assign_tier(0.59) == "below_average"
        assert assign_tier(0.65) == "below_average"

    def test_evaluate_kaggle_with_report(self):
        from src.quant.kaggle_layer import evaluate_kaggle

        report = {
            "pipeline_diagnostics": {
                "loyo_mean_log_loss": 0.54,
                "loyo_mean_brier": 0.18,
                "loyo_brier_skill_score": 0.28,
            },
            "calibration_metrics": {
                "expected_calibration_error": 0.02,
            },
            "predictions": {f"p_{i}": 0.5 for i in range(100)},
        }
        result = evaluate_kaggle(report)
        assert result.log_loss == pytest.approx(0.54)
        assert result.tier == "strong"
        assert result.brier_score == pytest.approx(0.18)
        assert result.calibration_ece == pytest.approx(0.02)
        assert result.n_matchups == 100
        assert result.brier_skill_score == pytest.approx(0.28)

    def test_evaluate_kaggle_empty_report(self):
        from src.quant.kaggle_layer import evaluate_kaggle

        result = evaluate_kaggle({})
        assert result.tier == "below_average"
        assert result.n_matchups == 0


# ---------------------------------------------------------------------------
# Contrarian formula tests
# ---------------------------------------------------------------------------


class TestContrarianFormula:
    def test_basic_score(self):
        from src.quant.espn_layer import contrarian_pick_score

        # With alpha=1, beta=0: pure model probability
        assert contrarian_pick_score(0.7, 0.5, alpha=1.0, beta=0.0) == pytest.approx(0.7)

    def test_contrarian_boost(self):
        from src.quant.espn_layer import contrarian_pick_score

        # Low ownership boosts score
        high_own = contrarian_pick_score(0.6, 0.8, alpha=1.0, beta=1.0)
        low_own = contrarian_pick_score(0.6, 0.2, alpha=1.0, beta=1.0)
        assert low_own > high_own

    def test_beta_increases_contrarianism(self):
        from src.quant.espn_layer import contrarian_pick_score

        # Higher beta amplifies ownership penalty
        score_b1 = contrarian_pick_score(0.6, 0.7, alpha=1.0, beta=1.0)
        score_b2 = contrarian_pick_score(0.6, 0.7, alpha=1.0, beta=2.0)
        assert score_b1 > score_b2  # Higher beta penalizes high ownership more

    def test_formula_values(self):
        from src.quant.espn_layer import contrarian_pick_score

        # P(pick) = model_prob^alpha * (1 - ownership)^beta
        result = contrarian_pick_score(0.8, 0.5, alpha=1.0, beta=1.0)
        expected = 0.8 * 0.5  # 0.4
        assert result == pytest.approx(expected)

    def test_contrarian_ev_pick_favors_underowned(self):
        from src.quant.espn_layer import contrarian_ev_pick

        # Team 1 is 60% to win but 80% owned
        # Team 2 is 40% to win but 20% owned
        matchup_probs = {("A", "B"): 0.6, ("B", "A"): 0.4}
        public_picks = {
            "A": {"R64": 0.8},
            "B": {"R64": 0.2},
        }

        # With high beta, should pick the underowned team
        pick = contrarian_ev_pick(
            "A", "B", "R64",
            matchup_probs, public_picks,
            alpha=1.0, beta=3.0,
        )
        # score_A = 0.6^1 * (1-0.8)^3 = 0.6 * 0.008 = 0.0048
        # score_B = 0.4^1 * (1-0.2)^3 = 0.4 * 0.512 = 0.2048
        assert pick == "B"

    def test_contrarian_ev_pick_favors_favorite_low_beta(self):
        from src.quant.espn_layer import contrarian_ev_pick

        matchup_probs = {("A", "B"): 0.7, ("B", "A"): 0.3}
        public_picks = {
            "A": {"R64": 0.6},
            "B": {"R64": 0.4},
        }

        # With low beta, model probability dominates
        pick = contrarian_ev_pick(
            "A", "B", "R64",
            matchup_probs, public_picks,
            alpha=1.0, beta=0.3,
        )
        assert pick == "A"


# ---------------------------------------------------------------------------
# ESPN result tests
# ---------------------------------------------------------------------------


class TestESPNResult:
    def test_bracket_result_has_required_metrics(self):
        from src.quant.espn_layer import ESPNBracketResult

        result = ESPNBracketResult(
            pool_size=100,
            strategy="balanced",
            win_probability=0.05,
            top_1_percent_probability=0.02,
            average_finish=0.3,
            expected_score=120.5,
        )
        # All four required metrics present
        assert hasattr(result, "win_probability")
        assert hasattr(result, "top_1_percent_probability")
        assert hasattr(result, "average_finish")
        assert hasattr(result, "expected_score")

    def test_espn_result_summary(self):
        from src.quant.espn_layer import ESPNBracketResult, ESPNResult

        br = ESPNBracketResult(
            pool_size=100,
            strategy="balanced",
            win_probability=0.05,
            top_1_percent_probability=0.02,
            average_finish=0.3,
            expected_score=120.5,
        )
        result = ESPNResult(bracket_results=[br])
        summary = result.summary()
        assert "ESPN LAYER" in summary
        assert "100" in summary
        assert "balanced" in summary


# ---------------------------------------------------------------------------
# Report tests
# ---------------------------------------------------------------------------


class TestQuantReport:
    def test_report_summary(self):
        from src.quant.kaggle_layer import KaggleResult
        from src.quant.espn_layer import ESPNBracketResult, ESPNResult
        from src.quant.report import QuantReport

        kaggle = KaggleResult(
            log_loss=0.54,
            tier="strong",
            brier_score=0.18,
            calibration_ece=0.02,
            n_matchups=2278,
        )
        espn = ESPNResult(bracket_results=[
            ESPNBracketResult(
                pool_size=100,
                strategy="balanced",
                win_probability=0.05,
                top_1_percent_probability=0.02,
                average_finish=0.3,
                expected_score=120.5,
            ),
        ])
        report = QuantReport(kaggle=kaggle, espn=espn)
        summary = report.summary()
        assert "QUANT DECISION SYSTEM REPORT" in summary
        assert "0.5400" in summary
        assert "strong" in summary
        assert "KEY ANSWERS" in summary

    def test_report_to_dict(self):
        from src.quant.kaggle_layer import KaggleResult
        from src.quant.espn_layer import ESPNBracketResult, ESPNResult
        from src.quant.report import QuantReport

        kaggle = KaggleResult(
            log_loss=0.54, tier="strong", brier_score=0.18,
            calibration_ece=0.02, n_matchups=100,
        )
        espn = ESPNResult(bracket_results=[
            ESPNBracketResult(
                pool_size=100, strategy="balanced",
                win_probability=0.05, top_1_percent_probability=0.02,
                average_finish=0.3, expected_score=120.5,
            ),
        ])
        report = QuantReport(kaggle=kaggle, espn=espn)
        d = report.to_dict()
        assert d["kaggle"]["log_loss"] == 0.54
        assert d["kaggle"]["tier"] == "strong"
        assert len(d["espn"]["brackets"]) == 1
        assert d["espn"]["brackets"][0]["pool_size"] == 100
        assert d["espn"]["brackets"][0]["win_probability"] == 0.05

    def test_report_save(self, tmp_path):
        import json
        from src.quant.kaggle_layer import KaggleResult
        from src.quant.espn_layer import ESPNResult
        from src.quant.report import QuantReport

        report = QuantReport(
            kaggle=KaggleResult(
                log_loss=0.55, tier="strong", brier_score=0.19,
                calibration_ece=0.03, n_matchups=50,
            ),
            espn=ESPNResult(),
        )
        path = str(tmp_path / "report.json")
        report.save(path)
        with open(path) as f:
            data = json.load(f)
        assert data["kaggle"]["tier"] == "strong"


# ---------------------------------------------------------------------------
# Bracket optimizer contrarian integration tests
# ---------------------------------------------------------------------------


class TestBracketOptimizerContrarian:
    def test_config_has_contrarian_fields(self):
        from src.espn.bracket_optimizer import ESPNOptimizationConfig

        config = ESPNOptimizationConfig()
        assert hasattr(config, "alpha")
        assert hasattr(config, "beta")
        assert hasattr(config, "use_contrarian_formula")
        assert config.alpha == 1.0
        assert config.beta == 1.0
        assert config.use_contrarian_formula is False

    def test_config_contrarian_enabled(self):
        from src.espn.bracket_optimizer import ESPNOptimizationConfig

        config = ESPNOptimizationConfig(
            use_contrarian_formula=True,
            alpha=1.0,
            beta=2.0,
        )
        assert config.use_contrarian_formula is True
        assert config.beta == 2.0
