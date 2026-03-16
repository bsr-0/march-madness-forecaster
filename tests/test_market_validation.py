"""Tests for governance market validation module."""

import json
import math
import tempfile
from pathlib import Path

import pytest

from src.governance.market_validation import (
    MarketValidationResult,
    _classify_divergence,
    load_market_probs_from_cache,
    validate_model_vs_market,
)


class TestClassifyDivergence:
    def test_strong_agreement(self):
        assert _classify_divergence(0.02, 0.95) == "strong_agreement"

    def test_moderate_divergence_by_rmsd(self):
        assert _classify_divergence(0.04, 0.95) == "moderate_divergence"

    def test_moderate_divergence_by_spearman(self):
        assert _classify_divergence(0.02, 0.85) == "moderate_divergence"

    def test_significant_divergence_by_rmsd(self):
        assert _classify_divergence(0.08, 0.95) == "significant_divergence"

    def test_significant_divergence_by_spearman(self):
        assert _classify_divergence(0.02, 0.75) == "significant_divergence"

    def test_major_disagreement_by_rmsd(self):
        assert _classify_divergence(0.12, 0.95) == "major_disagreement"

    def test_major_disagreement_by_spearman(self):
        assert _classify_divergence(0.02, 0.60) == "major_disagreement"

    def test_spearman_none_uses_rmsd_only(self):
        assert _classify_divergence(0.02, None) == "strong_agreement"
        assert _classify_divergence(0.12, None) == "major_disagreement"


class TestValidateModelVsMarket:
    def test_no_overlap_returns_none(self):
        result = validate_model_vs_market(
            {"team_a": 0.1}, {"team_b": 0.1}, adjust_vig=False
        )
        assert result is None

    def test_identical_probs(self):
        probs = {"team_a": 0.5, "team_b": 0.3, "team_c": 0.2}
        result = validate_model_vs_market(probs, dict(probs), adjust_vig=False)
        assert result is not None
        assert result.rmsd == 0.0
        assert result.n_common_teams == 3
        assert result.interpretation == "strong_agreement"

    def test_rmsd_computation(self):
        model = {"a": 0.3, "b": 0.2}
        market = {"a": 0.2, "b": 0.1}
        result = validate_model_vs_market(model, market, adjust_vig=False)
        assert result is not None
        expected_rmsd = math.sqrt((0.1**2 + 0.1**2) / 2)
        assert abs(result.rmsd - expected_rmsd) < 1e-5

    def test_top_disagreements_ordered(self):
        model = {"a": 0.5, "b": 0.3, "c": 0.2}
        market = {"a": 0.1, "b": 0.25, "c": 0.15}
        result = validate_model_vs_market(model, market, adjust_vig=False)
        assert result is not None
        assert result.top_disagreements[0]["team_id"] == "a"
        assert result.top_disagreements[0]["abs_diff"] >= result.top_disagreements[1]["abs_diff"]

    def test_partial_overlap(self):
        model = {"a": 0.3, "b": 0.2, "c": 0.1}
        market = {"b": 0.2, "c": 0.15, "d": 0.1}
        result = validate_model_vs_market(model, market, adjust_vig=False)
        assert result is not None
        assert result.n_common_teams == 2
        assert set(r["team_id"] for r in result.top_disagreements) == {"b", "c"}

    def test_n_top_disagreements_limit(self):
        model = {f"t{i}": 0.01 * i for i in range(20)}
        market = {f"t{i}": 0.01 * (i + 1) for i in range(20)}
        result = validate_model_vs_market(model, market, adjust_vig=False, n_top_disagreements=5)
        assert result is not None
        assert len(result.top_disagreements) == 5

    def test_vig_adjustment_applied(self):
        model = {"a": 0.5, "b": 0.3}
        # Market probs that sum to > 1 (has vig)
        market = {"a": 0.6, "b": 0.5}
        result_adj = validate_model_vs_market(model, market, adjust_vig=True)
        result_raw = validate_model_vs_market(model, market, adjust_vig=False)
        assert result_adj is not None
        assert result_raw is not None
        assert result_adj.vig_adjusted is True
        assert result_raw.vig_adjusted is False
        # After vig adjustment, market probs should be closer to model
        assert result_adj.rmsd != result_raw.rmsd


class TestLoadMarketProbsFromCache:
    def test_missing_file_returns_none(self, tmp_path):
        result = load_market_probs_from_cache(season=2026, cache_dir=tmp_path)
        assert result is None

    def test_valid_json_with_teams(self, tmp_path):
        data = {
            "teams": {
                "houston": {"implied_probability": 0.20, "championship_odds": 400},
                "duke": {"implied_probability": 0.15, "championship_odds": 600},
            },
            "source": "test",
        }
        cache_file = tmp_path / "betting_odds_2026.json"
        cache_file.write_text(json.dumps(data))
        result = load_market_probs_from_cache(season=2026, cache_dir=tmp_path)
        assert result is not None
        assert "houston" in result
        assert "duke" in result
        assert abs(result["houston"] - 0.20) < 0.01
        assert abs(result["duke"] - 0.15) < 0.01

    def test_american_odds_fallback(self, tmp_path):
        data = {
            "teams": {
                "houston": {"championship_odds": 400},
            },
        }
        cache_file = tmp_path / "betting_odds_2026.json"
        cache_file.write_text(json.dumps(data))
        result = load_market_probs_from_cache(season=2026, cache_dir=tmp_path)
        assert result is not None
        assert "houston" in result
        # +400 → 100/500 = 0.20
        assert abs(result["houston"] - 0.20) < 0.01

    def test_invalid_json_returns_none(self, tmp_path):
        cache_file = tmp_path / "betting_odds_2026.json"
        cache_file.write_text("not valid json{{{")
        result = load_market_probs_from_cache(season=2026, cache_dir=tmp_path)
        assert result is None

    def test_empty_teams_returns_none(self, tmp_path):
        data = {"teams": {}}
        cache_file = tmp_path / "betting_odds_2026.json"
        cache_file.write_text(json.dumps(data))
        result = load_market_probs_from_cache(season=2026, cache_dir=tmp_path)
        assert result is None

    def test_flat_format(self, tmp_path):
        data = {"houston": 0.20, "duke": 0.15}
        cache_file = tmp_path / "betting_odds_2026.json"
        cache_file.write_text(json.dumps(data))
        result = load_market_probs_from_cache(season=2026, cache_dir=tmp_path)
        assert result is not None
        assert abs(result["houston"] - 0.20) < 0.01
