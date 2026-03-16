"""Comprehensive unit tests for remaining modules to maximize coverage.

Covers:
  - src/data/normalize.py
  - src/data/versioning.py
  - src/data/team_name_resolver.py
  - src/data/schemas.py
  - src/data/loader.py
  - src/data/manifest_generator.py
  - src/data/historical_tournament_results.py
  - src/models/team.py
  - src/models/game.py
  - src/models/bracket.py
  - src/forecasting/engine.py
  - src/reproducibility/artifact_store.py
  - src/reproducibility/run_hasher.py
  - src/exports/kaggle.py
  - src/exports/deliverables_manager.py
"""

import hashlib
import json
import os
import pickle
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pandas as pd
import pytest

# =====================================================================
# normalize.py
# =====================================================================
from src.data.normalize import (
    _raw_normalize,
    normalize_team_id,
    normalize_team_name,
    strip_ncaa_suffix,
    strip_ncaa_suffix_name,
)


class TestRawNormalize:
    def test_empty_string(self):
        assert _raw_normalize("") == ""

    def test_basic_lowercase(self):
        assert _raw_normalize("Duke") == "duke"

    def test_html_entity(self):
        assert _raw_normalize("Texas A&amp;M") == "texas_a_m"

    def test_unicode_accents(self):
        result = _raw_normalize("San Jose\u0301 State")
        assert result == "san_jose_state"

    def test_multiple_underscores_collapsed(self):
        assert _raw_normalize("a   b") == "a_b"

    def test_special_chars_replaced(self):
        result = _raw_normalize("St. Mary's")
        assert "." not in result
        assert "'" not in result

    def test_leading_trailing_underscores_stripped(self):
        assert _raw_normalize("  Duke  ") == "duke"


class TestNormalizeTeamId:
    def test_empty_string(self):
        assert normalize_team_id("") == ""

    def test_basic_pass_through(self):
        assert normalize_team_id("Duke") == "duke"

    def test_alias_resolution_uconn(self):
        assert normalize_team_id("UConn") == "connecticut"

    def test_alias_resolution_byu(self):
        assert normalize_team_id("BYU") == "brigham_young"

    def test_mascot_suffix_stripped(self):
        assert normalize_team_id("Duke Blue Devils") == "duke"

    def test_ncaa_suffix_stripped(self):
        assert normalize_team_id("DukeNCAA") == "duke"

    def test_ncaa_suffix_with_alias(self):
        assert normalize_team_id("AlabamaNCCA") != ""  # doesn't match ncaa pattern

    def test_html_entity_ampersand(self):
        assert normalize_team_id("Texas A&amp;M") == "texas_a_m"

    def test_lsu_alias(self):
        assert normalize_team_id("LSU") == "louisiana_state"

    def test_usc_alias(self):
        assert normalize_team_id("USC") == "southern_california"

    def test_pitt_alias(self):
        assert normalize_team_id("Pitt") == "pittsburgh"

    def test_ole_miss_alias(self):
        assert normalize_team_id("Ole Miss") == "mississippi"

    def test_miami_fl_alias(self):
        assert normalize_team_id("Miami FL") == "miami__fl"

    def test_unknown_team_passthrough(self):
        result = normalize_team_id("Totally Unknown University")
        assert result == "totally_unknown_university"

    def test_ncaa_suffix_with_underscore(self):
        assert normalize_team_id("PurdueNCAA") == "purdue"

    def test_non_aliased_ncaa_suffix(self):
        # A team not in alias table but with NCAA suffix
        result = normalize_team_id("SomeRandomTeamNCAA")
        assert result == "somerandomteam"


class TestNormalizeTeamName:
    def test_empty_string(self):
        assert normalize_team_name("") == ""

    def test_basic(self):
        assert normalize_team_name("Duke") == "duke"

    def test_html_entity(self):
        result = normalize_team_name("Texas A&amp;M")
        assert "amp" not in result

    def test_stop_words_removed(self):
        result = normalize_team_name("The University of Texas")
        assert "the" not in result.split()
        assert "university" not in result.split()
        assert "of" not in result.split()
        assert "texas" in result.split()

    def test_ampersand_replaced(self):
        result = normalize_team_name("William & Mary")
        assert "and" in result

    def test_unicode_normalized(self):
        result = normalize_team_name("San Jose\u0301 State")
        assert "jose" in result


class TestStripNcaaSuffix:
    def test_with_suffix(self):
        assert strip_ncaa_suffix("alabamancaa") == "alabama"

    def test_without_suffix(self):
        assert strip_ncaa_suffix("alabama") == "alabama"

    def test_empty_string(self):
        assert strip_ncaa_suffix("") == ""

    def test_none_like(self):
        assert strip_ncaa_suffix("") == ""

    def test_case_insensitive(self):
        assert strip_ncaa_suffix("dukeNCAA") == "duke"


class TestStripNcaaSuffixName:
    def test_with_suffix(self):
        assert strip_ncaa_suffix_name("AlabamaNCAA") == "Alabama"

    def test_without_suffix(self):
        assert strip_ncaa_suffix_name("Alabama") == "Alabama"

    def test_empty(self):
        assert strip_ncaa_suffix_name("") == ""

    def test_none_returns_none(self):
        assert strip_ncaa_suffix_name(None) is None

    def test_trailing_space_stripped(self):
        result = strip_ncaa_suffix_name("Duke NCAA")
        # "NCAA" is at end, so it gets stripped; trailing space is also stripped
        assert result == "Duke"


# =====================================================================
# team_name_resolver.py
# =====================================================================
from src.data.team_name_resolver import (
    MatchResult,
    TeamNameResolver,
    _normalize_str,
    _to_canonical_id,
)


class TestNormalizeStr:
    def test_basic(self):
        assert _normalize_str("Duke") == "duke"

    def test_html_unescape(self):
        assert "amp" not in _normalize_str("A&amp;M")

    def test_ampersand_replaced(self):
        result = _normalize_str("A&M")
        assert "and" in result

    def test_whitespace_collapsed(self):
        assert _normalize_str("a   b") == "a b"


class TestToCanonicalId:
    def test_basic(self):
        assert _to_canonical_id("Duke") == "duke"

    def test_ampersand_becomes_underscore(self):
        result = _to_canonical_id("A&M")
        assert "&" not in result

    def test_special_chars_underscored(self):
        result = _to_canonical_id("St. Mary's")
        assert "." not in result
        assert "'" not in result


class TestMatchResult:
    def test_creation(self):
        mr = MatchResult("duke", "Duke", 1.0, "exact_id")
        assert mr.canonical_id == "duke"
        assert mr.display_name == "Duke"
        assert mr.confidence == 1.0
        assert mr.method == "exact_id"


class TestTeamNameResolver:
    def test_exact_id_match(self):
        r = TeamNameResolver()
        result = r.resolve("duke")
        assert result.canonical_id == "duke"
        assert result.confidence == 1.0

    def test_alias_match_uconn(self):
        r = TeamNameResolver()
        result = r.resolve("UConn")
        assert result.canonical_id == "connecticut"

    def test_alias_match_byu(self):
        r = TeamNameResolver()
        result = r.resolve("BYU")
        assert result.canonical_id == "brigham_young"

    def test_empty_name(self):
        r = TeamNameResolver()
        result = r.resolve("")
        assert result.canonical_id == ""
        assert result.method == "empty"

    def test_whitespace_name(self):
        r = TeamNameResolver()
        result = r.resolve("   ")
        assert result.method == "empty"

    def test_extra_aliases(self):
        r = TeamNameResolver(extra_aliases={"duke": ["Blue D"]})
        result = r.resolve("Blue D")
        assert result.canonical_id == "duke"

    def test_extra_aliases_new_team(self):
        r = TeamNameResolver(extra_aliases={"new_team": ["New Team U"]})
        result = r.resolve("New Team U")
        assert result.canonical_id == "new_team"

    def test_slug_match(self):
        r = TeamNameResolver()
        result = r.resolve("gonzaga")
        assert result.canonical_id == "gonzaga"

    def test_get_display_name(self):
        r = TeamNameResolver()
        assert r.get_display_name("duke") == "Duke"

    def test_get_display_name_unknown(self):
        r = TeamNameResolver()
        assert r.get_display_name("zzz_unknown") == "zzz_unknown"

    def test_add_alias(self):
        r = TeamNameResolver()
        r.add_alias("duke", "The Dukies")
        result = r.resolve("The Dukies")
        assert result.canonical_id == "duke"

    def test_add_alias_new_team(self):
        r = TeamNameResolver()
        r.add_alias("brand_new_team", "Brand New Team")
        result = r.resolve("Brand New Team")
        assert result.canonical_id == "brand_new_team"

    def test_known_teams_property(self):
        r = TeamNameResolver()
        teams = r.known_teams
        assert "duke" in teams
        assert "kansas" in teams
        assert len(teams) > 300

    def test_resolve_batch(self):
        r = TeamNameResolver()
        results = r.resolve_batch(["Duke", "Kansas", "UConn"])
        assert len(results) == 3
        assert results[0].canonical_id == "duke"
        assert results[2].canonical_id == "connecticut"

    def test_prefix_strip_mascot(self):
        r = TeamNameResolver()
        result = r.resolve("Kansas Jayhawks")
        assert result.canonical_id == "kansas"

    def test_unresolved_team(self):
        r = TeamNameResolver()
        result = r.resolve("XYZZYPLUGH University Of Nowhere")
        assert result.method in ("unresolved", "containment", "containment_best", "fuzzy")


# =====================================================================
# schemas.py
# =====================================================================
from src.data.schemas import (
    ValidationResult,
    validate_calibration_data,
    validate_engineered_features,
    validate_ensemble_weights,
    validate_feature_matrix,
    validate_loaded_data,
    validate_loyo_fold,
    validate_matchup_vector,
    validate_predictions,
    validate_trained_models,
)


class TestValidationResult:
    def test_to_dict(self):
        vr = ValidationResult(passed=True, warnings=["w1"], errors=[])
        d = vr.to_dict()
        assert d["passed"] is True
        assert d["warnings"] == ["w1"]
        assert d["errors"] == []


class TestValidateFeatureMatrix:
    def test_valid_matrix(self):
        X = np.random.randn(100, 10)
        result = validate_feature_matrix(X)
        assert result.passed is True

    def test_1d_matrix(self):
        result = validate_feature_matrix(np.array([1, 2, 3]))
        assert result.passed is False

    def test_too_few_samples(self):
        X = np.random.randn(5, 10)
        result = validate_feature_matrix(X, min_samples=10)
        assert result.passed is False

    def test_too_few_features(self):
        X = np.random.randn(100, 0).reshape(100, 0)
        result = validate_feature_matrix(X, min_features=1)
        assert result.passed is False

    def test_nan_warnings(self):
        X = np.random.randn(100, 5)
        X[:80, 0] = np.nan  # 80% NaN in first column
        result = validate_feature_matrix(X, max_nan_fraction=0.5)
        assert len(result.warnings) > 0

    def test_inf_values(self):
        X = np.random.randn(100, 5)
        X[0, 0] = np.inf
        result = validate_feature_matrix(X)
        assert result.passed is False

    def test_constant_columns_warning(self):
        X = np.random.randn(100, 5)
        X[:, 0] = 7.0  # constant column
        result = validate_feature_matrix(X)
        assert any("constant" in w.lower() for w in result.warnings)

    def test_feature_names_in_warnings(self):
        X = np.random.randn(100, 3)
        X[:80, 0] = np.nan
        result = validate_feature_matrix(X, feature_names=["a", "b", "c"], max_nan_fraction=0.5)
        assert any("a" in w for w in result.warnings)


class TestValidatePredictions:
    def test_valid(self):
        preds = np.array([0.1, 0.5, 0.9, 0.3])
        result = validate_predictions(preds)
        assert result.passed is True

    def test_2d_fails(self):
        preds = np.array([[0.1, 0.2]])
        result = validate_predictions(preds)
        assert result.passed is False

    def test_nan_fails(self):
        preds = np.array([0.5, np.nan])
        result = validate_predictions(preds)
        assert result.passed is False

    def test_out_of_range(self):
        preds = np.array([0.5, 1.5])
        result = validate_predictions(preds)
        assert result.passed is False

    def test_near_zero_variance_warning(self):
        preds = np.array([0.5, 0.5, 0.5, 0.5])
        result = validate_predictions(preds)
        assert len(result.warnings) > 0

    def test_inf_fails(self):
        preds = np.array([0.5, np.inf])
        result = validate_predictions(preds)
        assert result.passed is False


class TestValidateLoyoFold:
    def test_valid_fold(self):
        preds = np.array([0.3, 0.7, 0.5, 0.6] * 10)
        actuals = np.array([0, 1, 0, 1] * 10)
        result = validate_loyo_fold(2024, 0.18, 40, preds, actuals)
        assert result.passed is True

    def test_brier_out_of_range(self):
        preds = np.array([0.3, 0.7] * 10)
        actuals = np.array([0, 1] * 10)
        result = validate_loyo_fold(2024, 1.5, 20, preds, actuals)
        assert result.passed is False

    def test_very_high_brier_warning(self):
        preds = np.array([0.3, 0.7] * 10)
        actuals = np.array([0, 1] * 10)
        result = validate_loyo_fold(2024, 0.36, 20, preds, actuals)
        assert any("very high" in w.lower() for w in result.warnings)

    def test_few_games_warning(self):
        preds = np.array([0.3, 0.7, 0.5])
        actuals = np.array([0, 1, 0])
        result = validate_loyo_fold(2024, 0.18, 3, preds, actuals)
        assert any("few" in w.lower() for w in result.warnings)

    def test_length_mismatch(self):
        preds = np.array([0.3, 0.7])
        actuals = np.array([0, 1, 0])
        result = validate_loyo_fold(2024, 0.18, 3, preds, actuals)
        assert result.passed is False


class TestValidateEnsembleWeights:
    def test_valid(self):
        result = validate_ensemble_weights({"a": 0.6, "b": 0.4})
        assert result.passed is True

    def test_empty_weights(self):
        result = validate_ensemble_weights({})
        assert result.passed is False

    def test_negative_weight(self):
        result = validate_ensemble_weights({"a": -0.1, "b": 1.1})
        assert result.passed is False

    def test_sum_not_one(self):
        result = validate_ensemble_weights({"a": 0.3, "b": 0.3})
        assert result.passed is False

    def test_expected_components_missing(self):
        result = validate_ensemble_weights(
            {"a": 0.5, "b": 0.5}, expected_components=["a", "b", "c"]
        )
        assert any("missing" in w.lower() for w in result.warnings)

    def test_unexpected_components(self):
        result = validate_ensemble_weights(
            {"a": 0.5, "b": 0.5}, expected_components=["a"]
        )
        assert any("unexpected" in w.lower() for w in result.warnings)


class TestValidateCalibrationData:
    def test_valid(self):
        preds = np.array([0.3, 0.7, 0.5, 0.6] * 10)
        outcomes = np.array([0.0, 1.0, 0.0, 1.0] * 10)
        result = validate_calibration_data(preds, outcomes)
        assert result.passed is True

    def test_length_mismatch(self):
        result = validate_calibration_data(np.array([0.5]), np.array([0.0, 1.0]))
        assert result.passed is False

    def test_non_binary_outcomes(self):
        preds = np.array([0.5] * 40)
        outcomes = np.array([0.5] * 40)
        result = validate_calibration_data(preds, outcomes)
        assert result.passed is False

    def test_single_class(self):
        preds = np.array([0.5] * 40)
        outcomes = np.array([1.0] * 40)
        result = validate_calibration_data(preds, outcomes)
        assert result.passed is False

    def test_few_samples_warning(self):
        preds = np.array([0.3, 0.7, 0.5])
        outcomes = np.array([0.0, 1.0, 0.0])
        result = validate_calibration_data(preds, outcomes, min_samples=30)
        assert any("sample" in w.lower() for w in result.warnings)


class TestValidateLoadedData:
    def test_empty_teams(self):
        mock_data = MagicMock()
        mock_data.teams = []
        result = validate_loaded_data(mock_data)
        assert result.passed is False

    def test_few_teams_warning(self):
        mock_data = MagicMock()
        mock_data.teams = list(range(30))
        mock_data.torvik_map = {"a": 1}
        mock_data.rosters = {"a": 1}
        result = validate_loaded_data(mock_data)
        assert any("30" in w for w in result.warnings)

    def test_empty_torvik_warning(self):
        mock_data = MagicMock()
        mock_data.teams = list(range(100))
        mock_data.torvik_map = {}
        mock_data.rosters = {"a": 1}
        result = validate_loaded_data(mock_data)
        assert any("torvik" in w.lower() for w in result.warnings)


class TestValidateEngineeredFeatures:
    def test_empty_features(self):
        mock_f = MagicMock()
        mock_f.team_features = {}
        result = validate_engineered_features(mock_f)
        assert result.passed is False

    def test_inconsistent_dims(self):
        mock_f = MagicMock()
        mock_f.team_features = {
            "a": np.array([1, 2]),
            "b": np.array([1, 2, 3]),
        }
        mock_f.team_struct = {}
        result = validate_engineered_features(mock_f)
        assert result.passed is False

    def test_valid_features(self):
        mock_f = MagicMock()
        mock_f.team_features = {
            "a": np.array([1, 2, 3]),
            "b": np.array([4, 5, 6]),
        }
        mock_f.team_struct = {"a": 1}
        result = validate_engineered_features(mock_f)
        assert result.passed is True


class TestValidateTrainedModels:
    def test_no_baseline(self):
        mock_m = MagicMock()
        mock_m.baseline_model = None
        result = validate_trained_models(mock_m)
        assert result.passed is False

    def test_oof_length_mismatch(self):
        mock_m = MagicMock()
        mock_m.baseline_model = "something"
        mock_m.oof_predictions = [1, 2, 3]
        mock_m.oof_actuals = [1, 2]
        result = validate_trained_models(mock_m)
        assert result.passed is False


class TestValidateMatchupVector:
    def test_valid(self):
        v = np.array([1.0, 2.0, 3.0])
        result = validate_matchup_vector(v)
        assert result.passed is True

    def test_2d_fails(self):
        v = np.array([[1.0, 2.0]])
        result = validate_matchup_vector(v)
        assert result.passed is False

    def test_wrong_dimension(self):
        v = np.array([1.0, 2.0, 3.0])
        result = validate_matchup_vector(v, expected_dim=5)
        assert result.passed is False

    def test_inf_fails(self):
        v = np.array([1.0, np.inf])
        result = validate_matchup_vector(v)
        assert result.passed is False

    def test_high_nan_fraction_fails(self):
        v = np.array([np.nan] * 8 + [1.0, 2.0])
        result = validate_matchup_vector(v)
        assert result.passed is False

    def test_moderate_nan_warning(self):
        v = np.array([np.nan] * 3 + [1.0] * 7)
        result = validate_matchup_vector(v)
        assert result.passed is True
        assert len(result.warnings) > 0


# =====================================================================
# models/team.py
# =====================================================================
from src.models.team import Team


class TestTeam:
    def test_creation(self):
        t = Team(name="Duke", seed=1, region="East")
        assert t.name == "Duke"
        assert t.seed == 1
        assert t.elo_rating == 1500.0

    def test_invalid_seed_low(self):
        with pytest.raises(ValueError):
            Team(name="Duke", seed=0, region="East")

    def test_invalid_seed_high(self):
        with pytest.raises(ValueError):
            Team(name="Duke", seed=17, region="East")

    def test_invalid_region(self):
        with pytest.raises(ValueError):
            Team(name="Duke", seed=1, region="North")

    def test_to_dict(self):
        t = Team(name="Duke", seed=1, region="East", stats={"ppg": 80.0})
        d = t.to_dict()
        assert d["name"] == "Duke"
        assert d["stats"]["ppg"] == 80.0

    def test_from_dict(self):
        d = {"name": "Duke", "seed": 1, "region": "East", "elo_rating": 1600.0}
        t = Team.from_dict(d)
        assert t.elo_rating == 1600.0

    def test_from_dict_defaults(self):
        d = {"name": "Duke", "seed": 1, "region": "East"}
        t = Team.from_dict(d)
        assert t.elo_rating == 1500.0
        assert t.stats == {}

    def test_expected_score_equal_rating(self):
        t = Team(name="Duke", seed=1, region="East")
        assert abs(t.expected_score(1500.0) - 0.5) < 1e-6

    def test_expected_score_higher_opponent(self):
        t = Team(name="Duke", seed=1, region="East")
        assert t.expected_score(1700.0) < 0.5

    def test_expected_score_lower_opponent(self):
        t = Team(name="Duke", seed=1, region="East")
        assert t.expected_score(1300.0) > 0.5

    def test_update_elo_win(self):
        t = Team(name="Duke", seed=1, region="East", elo_rating=1500.0)
        t.update_elo(1500.0, 1.0)
        assert t.elo_rating > 1500.0

    def test_update_elo_loss(self):
        t = Team(name="Duke", seed=1, region="East", elo_rating=1500.0)
        t.update_elo(1500.0, 0.0)
        assert t.elo_rating < 1500.0


# =====================================================================
# models/game.py
# =====================================================================
from src.models.game import Game


class TestGame:
    def _team1(self):
        return Team(name="Duke", seed=1, region="East")

    def _team2(self):
        return Team(name="Kansas", seed=2, region="East")

    def test_creation(self):
        g = Game(game_id=1, round=1, team1=self._team1(), team2=self._team2())
        assert g.game_id == 1
        assert g.model_scores == {}
        assert g.predicted_winner is None

    def test_set_prediction(self):
        t1, t2 = self._team1(), self._team2()
        g = Game(game_id=1, round=1, team1=t1, team2=t2)
        g.set_prediction(t1, 0.75)
        assert g.predicted_winner is t1
        assert g.win_probability == 0.75

    def test_set_prediction_invalid_probability(self):
        t1, t2 = self._team1(), self._team2()
        g = Game(game_id=1, round=1, team1=t1, team2=t2)
        with pytest.raises(ValueError):
            g.set_prediction(t1, 1.5)

    def test_set_prediction_invalid_winner(self):
        t1, t2 = self._team1(), self._team2()
        g = Game(game_id=1, round=1, team1=t1, team2=t2)
        other = Team(name="UNC", seed=3, region="East")
        with pytest.raises(ValueError):
            g.set_prediction(other, 0.5)

    def test_set_prediction_with_model_scores(self):
        t1, t2 = self._team1(), self._team2()
        g = Game(game_id=1, round=1, team1=t1, team2=t2)
        g.set_prediction(t1, 0.7, {"elo": 0.65, "ml": 0.75})
        assert g.model_scores == {"elo": 0.65, "ml": 0.75}

    def test_to_dict(self):
        t1, t2 = self._team1(), self._team2()
        g = Game(game_id=1, round=1, team1=t1, team2=t2)
        g.set_prediction(t1, 0.75)
        d = g.to_dict()
        assert d["game_id"] == 1
        assert d["predicted_winner"] == "Duke"
        assert d["team1"] == "Duke"

    def test_to_dict_no_prediction(self):
        g = Game(game_id=1, round=1, team1=self._team1(), team2=self._team2())
        d = g.to_dict()
        assert d["predicted_winner"] is None

    def test_is_upset_higher_seed_wins(self):
        t1 = Team(name="Duke", seed=1, region="East")
        t2 = Team(name="Kansas", seed=16, region="East")
        g = Game(game_id=1, round=1, team1=t1, team2=t2)
        g.set_prediction(t2, 0.6)  # 16 seed predicted to win
        assert g.is_upset is True

    def test_is_upset_lower_seed_wins(self):
        t1 = Team(name="Duke", seed=1, region="East")
        t2 = Team(name="Kansas", seed=16, region="East")
        g = Game(game_id=1, round=1, team1=t1, team2=t2)
        g.set_prediction(t1, 0.9)  # 1 seed wins = no upset
        assert g.is_upset is False

    def test_is_upset_no_prediction(self):
        g = Game(game_id=1, round=1, team1=self._team1(), team2=self._team2())
        assert g.is_upset is False


# =====================================================================
# models/bracket.py
# =====================================================================
from src.models.bracket import Bracket


def _make_64_teams():
    """Create 64 teams for a valid bracket."""
    teams = []
    for region in Bracket.REGIONS:
        for seed in range(1, 17):
            teams.append(Team(name=f"{region}_{seed}", seed=seed, region=region))
    return teams


class TestBracket:
    def test_valid_64_teams(self):
        teams = _make_64_teams()
        b = Bracket(teams)
        assert len(b.teams) == 64

    def test_invalid_team_count(self):
        teams = [Team(name="Duke", seed=1, region="East")]
        with pytest.raises(ValueError):
            Bracket(teams)

    def test_add_game(self):
        teams = _make_64_teams()
        b = Bracket(teams)
        t1 = b.get_team("East_1")
        t2 = b.get_team("East_16")
        g = Game(game_id=1, round=1, team1=t1, team2=t2)
        b.add_game(g)
        assert len(b.games) == 1

    def test_get_games_by_round(self):
        teams = _make_64_teams()
        b = Bracket(teams)
        t1 = b.get_team("East_1")
        t2 = b.get_team("East_16")
        g = Game(game_id=1, round=1, team1=t1, team2=t2)
        b.add_game(g)
        assert len(b.get_games_by_round(1)) == 1
        assert len(b.get_games_by_round(2)) == 0

    def test_get_team(self):
        teams = _make_64_teams()
        b = Bracket(teams)
        assert b.get_team("East_1") is not None
        assert b.get_team("Nonexistent") is None

    def test_set_champion(self):
        teams = _make_64_teams()
        b = Bracket(teams)
        t = b.get_team("East_1")
        b.set_champion(t)
        assert b.champion is t

    def test_set_final_four(self):
        teams = _make_64_teams()
        b = Bracket(teams)
        ff = [b.get_team(f"{r}_1") for r in Bracket.REGIONS]
        b.set_final_four(ff)
        assert len(b.final_four) == 4

    def test_set_final_four_invalid(self):
        teams = _make_64_teams()
        b = Bracket(teams)
        with pytest.raises(ValueError):
            b.set_final_four([b.get_team("East_1")])

    def test_to_dict(self):
        teams = _make_64_teams()
        b = Bracket(teams)
        d = b.to_dict()
        assert len(d["teams"]) == 64
        assert d["champion"] is None

    def test_from_dict_round_trip(self):
        teams = _make_64_teams()
        b = Bracket(teams)
        d = b.to_dict()
        b2 = Bracket.from_dict(d)
        assert len(b2.teams) == 64

    def test_get_matchup_path_same_region(self):
        teams = _make_64_teams()
        b = Bracket(teams)
        t1 = b.get_team("East_1")
        t16 = b.get_team("East_16")
        result = b.get_matchup_path(t1, t16)
        assert result == 1  # 1 vs 16 is round 1

    def test_get_matchup_path_different_region(self):
        teams = _make_64_teams()
        b = Bracket(teams)
        t1 = b.get_team("East_1")
        t2 = b.get_team("West_1")
        result = b.get_matchup_path(t1, t2)
        assert result is None


# =====================================================================
# data/loader.py
# =====================================================================
from src.data.loader import DataLoader


class TestDataLoader:
    def test_load_teams_from_json(self, tmp_path):
        data = {
            "teams": [
                {"name": "Duke", "seed": 1, "region": "East"},
                {"name": "Kansas", "seed": 1, "region": "West"},
            ]
        }
        f = tmp_path / "teams.json"
        f.write_text(json.dumps(data))
        teams = DataLoader.load_teams_from_json(str(f))
        assert len(teams) == 2
        assert teams[0].name == "Duke"

    def test_load_matchups(self, tmp_path):
        data = {
            "matchups": [
                {"team1": "Duke", "team2": "UNC", "round": 1, "game_id": 1},
            ]
        }
        f = tmp_path / "matchups.json"
        f.write_text(json.dumps(data))
        result = DataLoader.load_matchups(str(f))
        assert len(result) == 1
        assert result[0] == ("Duke", "UNC", 1, 1)

    def test_hash_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello world")
        h = DataLoader.hash_file(str(f))
        expected = hashlib.sha256(b"hello world").hexdigest()
        assert h == expected

    def test_hash_dataset(self, tmp_path):
        f1 = tmp_path / "a.txt"
        f1.write_text("a")
        f2 = tmp_path / "b.txt"
        f2.write_text("b")
        result = DataLoader.hash_dataset([str(f1), str(f2)], ["file_a", "file_b"])
        assert "file_a" in result
        assert "file_b" in result
        assert "combined" in result

    def test_hash_dataset_missing_file(self, tmp_path):
        result = DataLoader.hash_dataset(["/nonexistent/file.txt"])
        assert result["file.txt"] == "MISSING"

    def test_hash_dataset_no_labels(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("col1,col2")
        result = DataLoader.hash_dataset([str(f)])
        assert "data.csv" in result

    def test_load_bracket_from_json(self, tmp_path):
        teams_data = [
            {"name": f"{r}_{s}", "seed": s, "region": r}
            for r in Bracket.REGIONS
            for s in range(1, 17)
        ]
        data = {"teams": teams_data, "games": [], "champion": None, "final_four": []}
        f = tmp_path / "bracket.json"
        f.write_text(json.dumps(data))
        b = DataLoader.load_bracket_from_json(str(f))
        assert len(b.teams) == 64

    def test_save_bracket_to_json(self, tmp_path):
        teams = _make_64_teams()
        b = Bracket(teams)
        f = tmp_path / "out.json"
        DataLoader.save_bracket_to_json(b, str(f))
        assert f.exists()
        loaded = json.loads(f.read_text())
        assert len(loaded["teams"]) == 64


# =====================================================================
# data/versioning.py
# =====================================================================
from src.data.versioning import (
    SnapshotManifest,
    list_snapshots,
    restore_snapshot,
    snapshot_data_dir,
)


class TestSnapshotManifest:
    def test_to_dict(self):
        m = SnapshotManifest(snapshot_id="test", timestamp="2024-01-01")
        d = m.to_dict()
        assert d["snapshot_id"] == "test"
        assert d["file_count"] == 0


class TestVersioning:
    def test_snapshot_and_restore(self, tmp_path):
        src = tmp_path / "data"
        src.mkdir()
        (src / "file.txt").write_text("hello")
        snap_dir = tmp_path / "snaps"

        sid = snapshot_data_dir(str(src), label="test", snapshot_base=str(snap_dir))
        assert "test" in sid

        target = tmp_path / "restored"
        count = restore_snapshot(sid, str(target), str(snap_dir))
        assert count == 1
        assert (target / "file.txt").read_text() == "hello"

    def test_snapshot_nonexistent_dir(self):
        with pytest.raises(FileNotFoundError):
            snapshot_data_dir("/nonexistent/path/xyz")

    def test_list_snapshots_empty(self, tmp_path):
        result = list_snapshots(str(tmp_path / "empty"))
        assert result == []

    def test_list_snapshots(self, tmp_path):
        src = tmp_path / "data"
        src.mkdir()
        (src / "f.txt").write_text("x")
        snap_dir = tmp_path / "snaps"
        snapshot_data_dir(str(src), label="s1", snapshot_base=str(snap_dir))
        snapshot_data_dir(str(src), label="s2", snapshot_base=str(snap_dir))
        results = list_snapshots(str(snap_dir))
        assert len(results) >= 2

    def test_restore_nonexistent_snapshot(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            restore_snapshot("nonexistent", str(tmp_path), str(tmp_path))

    def test_snapshot_skips_hidden_files(self, tmp_path):
        src = tmp_path / "data"
        src.mkdir()
        (src / ".hidden").write_text("secret")
        (src / "visible.txt").write_text("public")
        snap_dir = tmp_path / "snaps"
        sid = snapshot_data_dir(str(src), snapshot_base=str(snap_dir))
        target = tmp_path / "restored"
        count = restore_snapshot(sid, str(target), str(snap_dir))
        assert count == 1

    def test_restore_with_verify_failure(self, tmp_path):
        src = tmp_path / "data"
        src.mkdir()
        (src / "f.txt").write_text("hello")
        snap_dir = tmp_path / "snaps"
        sid = snapshot_data_dir(str(src), snapshot_base=str(snap_dir))

        # Corrupt the snapshot file
        snap_file = snap_dir / sid / "f.txt"
        snap_file.write_text("corrupted")

        target = tmp_path / "restored"
        with pytest.raises(ValueError, match="Hash mismatch"):
            restore_snapshot(sid, str(target), str(snap_dir), verify=True)

    def test_restore_without_verify(self, tmp_path):
        src = tmp_path / "data"
        src.mkdir()
        (src / "f.txt").write_text("hello")
        snap_dir = tmp_path / "snaps"
        sid = snapshot_data_dir(str(src), snapshot_base=str(snap_dir))

        snap_file = snap_dir / sid / "f.txt"
        snap_file.write_text("changed")

        target = tmp_path / "restored"
        count = restore_snapshot(sid, str(target), str(snap_dir), verify=False)
        assert count == 1


# =====================================================================
# data/historical_tournament_results.py
# =====================================================================
from src.data.historical_tournament_results import (
    TournamentGame,
    get_available_years,
    get_tournament_games_for_eval,
    load_tournament_results,
    save_tournament_results,
)


class TestTournamentGame:
    def _game(self, team1_won=True, seed1=1, seed2=16):
        return TournamentGame(
            year=2024, round_name="R64",
            team1_id="duke", team2_id="unc",
            team1_seed=seed1, team2_seed=seed2,
            team1_score=80, team2_score=70,
            team1_won=team1_won,
        )

    def test_winner_id(self):
        assert self._game(True).winner_id == "duke"
        assert self._game(False).winner_id == "unc"

    def test_loser_id(self):
        assert self._game(True).loser_id == "unc"
        assert self._game(False).loser_id == "duke"

    def test_upset_higher_seed_wins(self):
        # seed 16 beating seed 1 is an upset
        g = self._game(team1_won=False, seed1=1, seed2=16)
        assert g.upset is True

    def test_no_upset(self):
        g = self._game(team1_won=True, seed1=1, seed2=16)
        assert g.upset is False

    def test_to_eval_dict(self):
        d = self._game().to_eval_dict()
        assert d["team1_id"] == "duke"
        assert d["team1_won"] is True
        assert d["round"] == "R64"


class TestHistoricalTournamentIO:
    def test_save_and_load(self, tmp_path):
        games = [
            TournamentGame(
                year=2024, round_name="R64",
                team1_id="duke", team2_id="unc",
                team1_seed=1, team2_seed=16,
                team1_score=80, team2_score=70,
                team1_won=True,
            )
        ]
        save_tournament_results(games, 2024, str(tmp_path))
        loaded = load_tournament_results(2024, str(tmp_path))
        assert len(loaded) == 1
        assert loaded[0].team1_id == "duke"

    def test_load_nonexistent_year(self, tmp_path):
        result = load_tournament_results(1900, str(tmp_path))
        assert result == []

    def test_get_available_years(self, tmp_path):
        games = [
            TournamentGame(
                year=2023, round_name="R64",
                team1_id="a", team2_id="b",
                team1_seed=1, team2_seed=16,
                team1_score=80, team2_score=70,
                team1_won=True,
            )
        ]
        save_tournament_results(games, 2023, str(tmp_path))
        years = get_available_years(str(tmp_path))
        assert 2023 in years

    def test_get_tournament_games_for_eval(self, tmp_path):
        games = [
            TournamentGame(
                year=2024, round_name="R64",
                team1_id="duke", team2_id="unc",
                team1_seed=1, team2_seed=16,
                team1_score=80, team2_score=70,
                team1_won=True,
            )
        ]
        save_tournament_results(games, 2024, str(tmp_path))
        result = get_tournament_games_for_eval(2024, str(tmp_path))
        assert len(result) == 1
        assert result[0]["team1_id"] == "duke"


# =====================================================================
# data/manifest_generator.py
# =====================================================================
from src.data.manifest_generator import ManifestGenerator, _sha256_bytes, _sha256_file


class TestManifestGeneratorHelpers:
    def test_sha256_bytes(self):
        result = _sha256_bytes(b"hello")
        assert result == hashlib.sha256(b"hello").hexdigest()

    def test_sha256_file_missing(self):
        assert _sha256_file("/nonexistent/file.txt") == "FILE_NOT_FOUND"

    def test_sha256_file_valid(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello")
        result = _sha256_file(str(f))
        assert len(result) == 64  # SHA-256 hex length


class TestManifestGenerator:
    @patch("src.data.manifest_generator._git_commit_sha", return_value="abc123")
    @patch("src.data.manifest_generator._git_diff_hash", return_value="clean")
    @patch("src.data.manifest_generator._lockfile_hash", return_value="lock_hash")
    def test_generate(self, mock_lock, mock_diff, mock_sha):
        gen = ManifestGenerator(config_path="configs/test.json")
        manifest = gen.generate(
            run_id="test_run",
            config_hash="cfg_hash",
            raw_inputs=[{"provider": "test", "path_or_uri": "/tmp/x", "snapshot_timestamp": "now", "checksum": "abc"}],
            training_data_hash="td_hash",
            model_artifact_hash="ma_hash",
            random_seed=42,
            model_artifact_path="model.pkl",
            training_data_path="train.csv",
        )
        assert manifest["run_id"] == "test_run"
        assert manifest["random_seed"] == 42
        assert manifest["code"]["commit_sha"] == "abc123"

    def test_save(self, tmp_path):
        gen = ManifestGenerator()
        manifest = {"run_id": "test", "data": 123}
        path = gen.save(manifest, str(tmp_path / "m.json"))
        assert Path(path).exists()
        loaded = json.loads(Path(path).read_text())
        assert loaded["run_id"] == "test"

    def test_validate_level_a_full(self):
        manifest = {
            "run_id": "test",
            "created_at": "2024-01-01",
            "code": {"commit_sha": "abc", "diff_hash": "clean"},
            "config": {"path": "p", "hash": "h"},
            "raw_inputs": [{"provider": "p", "path_or_uri": "u", "snapshot_timestamp": "t", "checksum": "c"}],
            "feature_contracts": {"path": "p", "hash": "h"},
            "training_data": {"path_or_uri": "p", "hash": "h"},
            "model": {"artifact_path_or_uri": "p", "artifact_hash": "h"},
            "random_seed": 42,
            "environment": {"python_version": "3.11", "os": "Linux", "lockfile_hash": "h"},
        }
        violations = ManifestGenerator.validate_level_a(manifest)
        assert violations == []

    def test_validate_level_a_missing_fields(self):
        violations = ManifestGenerator.validate_level_a({})
        assert len(violations) > 0

    def test_validate_level_a_empty_field(self):
        manifest = {
            "run_id": "",
            "created_at": "2024",
            "code": {"commit_sha": "a", "diff_hash": "b"},
            "config": {"path": "p", "hash": "h"},
            "raw_inputs": [],
            "feature_contracts": {"path": "p", "hash": "h"},
            "training_data": {"path_or_uri": "p", "hash": "h"},
            "model": {"artifact_path_or_uri": "p", "artifact_hash": "h"},
            "random_seed": 42,
            "environment": {"python_version": "3.11", "os": "Linux", "lockfile_hash": "h"},
        }
        violations = ManifestGenerator.validate_level_a(manifest)
        assert any("Empty" in v for v in violations)

    def test_validate_level_b_match(self):
        manifest = {
            "run_id": "test",
            "created_at": "2024-01-01",
            "code": {"commit_sha": "abc", "diff_hash": "clean"},
            "config": {"path": "p", "hash": "h"},
            "raw_inputs": [{"provider": "p", "path_or_uri": "u", "snapshot_timestamp": "t", "checksum": "c"}],
            "feature_contracts": {"path": "p", "hash": "h"},
            "training_data": {"path_or_uri": "p", "hash": "EXPECTED"},
            "model": {"artifact_path_or_uri": "p", "artifact_hash": "h"},
            "random_seed": 42,
            "environment": {"python_version": "3.11", "os": "Linux", "lockfile_hash": "h"},
        }
        violations = ManifestGenerator.validate_level_b(manifest, "EXPECTED")
        assert not any("Level B" in v for v in violations)

    def test_validate_level_b_mismatch(self):
        manifest = {
            "run_id": "test",
            "created_at": "2024-01-01",
            "code": {"commit_sha": "abc", "diff_hash": "clean"},
            "config": {"path": "p", "hash": "h"},
            "raw_inputs": [{"provider": "p", "path_or_uri": "u", "snapshot_timestamp": "t", "checksum": "c"}],
            "feature_contracts": {"path": "p", "hash": "h"},
            "training_data": {"path_or_uri": "p", "hash": "EXPECTED"},
            "model": {"artifact_path_or_uri": "p", "artifact_hash": "h"},
            "random_seed": 42,
            "environment": {"python_version": "3.11", "os": "Linux", "lockfile_hash": "h"},
        }
        violations = ManifestGenerator.validate_level_b(manifest, "DIFFERENT")
        assert any("Level B" in v for v in violations)


# =====================================================================
# forecasting/engine.py
# =====================================================================
from src.forecasting.engine import (
    CalibrationReport,
    ForecastEngine,
    ForecastEngineConfig,
)
from src.exceptions import IntegrityError


class TestForecastEngineConfig:
    def test_defaults(self):
        cfg = ForecastEngineConfig()
        assert cfg.year == 2026
        assert cfg.brier_threshold == 0.20

    def test_custom(self):
        cfg = ForecastEngineConfig(year=2025, random_seed=99)
        assert cfg.year == 2025

    def test_firewall_rejects_forbidden(self):
        # Dynamically add a forbidden param
        with pytest.raises(ValueError, match="forbidden"):
            cfg = ForecastEngineConfig()
            cfg.pool_size = 100
            cfg._enforce_firewall()


class TestCalibrationReport:
    def test_to_dict(self):
        cr = CalibrationReport(
            brier_score=0.18, n_samples=100, threshold=0.20,
            passed=True, model_quality="accepted",
        )
        d = cr.to_dict()
        assert d["passed"] is True
        assert d["brier_score"] == 0.18


class TestForecastEngine:
    def test_default_config(self):
        engine = ForecastEngine()
        assert engine.config.year == 2026

    def test_custom_config(self):
        cfg = ForecastEngineConfig(year=2025)
        engine = ForecastEngine(cfg)
        assert engine.config.year == 2025

    def test_predict_proba_no_source_raises(self):
        engine = ForecastEngine()
        with pytest.raises(RuntimeError):
            engine.predict_proba("duke", "unc")

    def test_predict_proba_with_fn(self):
        engine = ForecastEngine()
        engine.set_predict_fn(lambda t1, t2: 0.75)
        assert abs(engine.predict_proba("duke", "unc") - 0.75) < 1e-6

    def test_predict_proba_clipping(self):
        engine = ForecastEngine()
        engine.set_predict_fn(lambda t1, t2: 1.5)
        result = engine.predict_proba("duke", "unc")
        assert result <= engine.config.pre_calibration_clip_hi

    def test_predict_proba_with_pipeline(self):
        pipeline = MagicMock()
        pipeline.predict_probability.return_value = 0.65
        engine = ForecastEngine()
        engine.set_pipeline(pipeline)
        assert abs(engine.predict_proba("duke", "unc") - 0.65) < 1e-6

    def test_set_pipeline_invalid(self):
        engine = ForecastEngine()
        with pytest.raises(TypeError):
            engine.set_pipeline(object())

    def test_predict_all_matchups(self):
        engine = ForecastEngine()
        engine.set_predict_fn(lambda t1, t2: 0.6)
        result = engine.predict_all_matchups(["a", "b", "c"])
        assert ("a", "b") in result
        assert ("b", "a") in result
        assert abs(result[("a", "b")] + result[("b", "a")] - 1.0) < 1e-6

    def test_validate_calibration_pass(self):
        cfg = ForecastEngineConfig(brier_threshold=0.30, mode="experimental")
        engine = ForecastEngine(cfg)
        report = engine.validate_calibration([0.6, 0.4, 0.7], [1, 0, 1])
        assert report.passed is True
        assert engine.calibration_report is not None

    def test_validate_calibration_fail_experimental(self):
        cfg = ForecastEngineConfig(brier_threshold=0.01, mode="experimental", strict_brier_gate=False)
        engine = ForecastEngine(cfg)
        report = engine.validate_calibration([0.6, 0.4], [1, 0])
        assert report.passed is False
        assert report.model_quality == "rejected"

    def test_validate_calibration_fail_production_raises(self):
        cfg = ForecastEngineConfig(brier_threshold=0.01, mode="production", strict_brier_gate=True)
        engine = ForecastEngine(cfg)
        with pytest.raises(IntegrityError):
            engine.validate_calibration([0.6, 0.4], [1, 0])

    def test_validate_calibration_length_mismatch(self):
        engine = ForecastEngine()
        with pytest.raises(ValueError):
            engine.validate_calibration([0.5], [1, 0])

    def test_validate_calibration_empty(self):
        engine = ForecastEngine()
        with pytest.raises(ValueError):
            engine.validate_calibration([], [])

    def test_predict_fn_takes_priority_over_pipeline(self):
        pipeline = MagicMock()
        pipeline.predict_probability.return_value = 0.1
        engine = ForecastEngine()
        engine.set_pipeline(pipeline)
        engine.set_predict_fn(lambda t1, t2: 0.9)
        assert engine.predict_proba("a", "b") > 0.85


# =====================================================================
# reproducibility/artifact_store.py
# =====================================================================
from src.reproducibility.artifact_store import ArtifactManifest, ArtifactStore


class TestArtifactManifest:
    def test_to_dict(self):
        m = ArtifactManifest(
            artifact_id="abc",
            artifact_type="model",
            created_at="2024-01-01",
            experiment_id="exp1",
            file_path="model/abc.pkl",
            size_bytes=1024,
        )
        d = m.to_dict()
        assert d["artifact_id"] == "abc"

    def test_from_dict(self):
        d = {
            "artifact_id": "abc",
            "artifact_type": "model",
            "created_at": "2024-01-01",
            "experiment_id": "exp1",
            "file_path": "model/abc.pkl",
            "size_bytes": 1024,
            "extra_field": "ignored",
        }
        m = ArtifactManifest.from_dict(d)
        assert m.artifact_id == "abc"


class TestArtifactStore:
    def test_save_and_load_model(self, tmp_path):
        store = ArtifactStore(str(tmp_path / "store"))
        model = {"weights": [1, 2, 3]}
        aid = store.save_model(model, "exp1")
        assert len(aid) == 8
        loaded = store.load_artifact(aid)
        assert loaded == model

    def test_dedup_model(self, tmp_path):
        store = ArtifactStore(str(tmp_path / "store"))
        model = {"weights": [1, 2, 3]}
        aid1 = store.save_model(model, "exp1")
        aid2 = store.save_model(model, "exp2")
        assert aid1 == aid2

    def test_save_predictions(self, tmp_path):
        store = ArtifactStore(str(tmp_path / "store"))
        preds = {"duke_unc": 0.75}
        aid = store.save_predictions(preds, "exp1")
        loaded = store.load_artifact(aid)
        assert loaded == preds

    def test_save_config(self, tmp_path):
        @dataclass
        class TestConfig:
            lr: float = 0.01
            epochs: int = 10

        store = ArtifactStore(str(tmp_path / "store"))
        aid = store.save_config(TestConfig(), "exp1")
        loaded = store.load_artifact(aid)
        assert loaded["lr"] == 0.01

    def test_load_nonexistent(self, tmp_path):
        store = ArtifactStore(str(tmp_path / "store"))
        with pytest.raises(FileNotFoundError):
            store.load_artifact("nonexistent")

    def test_list_artifacts(self, tmp_path):
        store = ArtifactStore(str(tmp_path / "store"))
        store.save_predictions({"a": 0.5}, "exp1")
        store.save_predictions({"b": 0.6}, "exp2")
        all_arts = store.list_artifacts()
        assert len(all_arts) == 2

    def test_list_artifacts_filter_experiment(self, tmp_path):
        store = ArtifactStore(str(tmp_path / "store"))
        store.save_predictions({"a": 0.5}, "exp1")
        store.save_predictions({"b": 0.6}, "exp2")
        filtered = store.list_artifacts(experiment_id="exp1")
        assert len(filtered) == 1

    def test_list_artifacts_filter_type(self, tmp_path):
        store = ArtifactStore(str(tmp_path / "store"))
        store.save_predictions({"a": 0.5}, "exp1")
        store.save_model({"w": 1}, "exp1")
        filtered = store.list_artifacts(artifact_type="predictions")
        assert len(filtered) == 1

    def test_list_artifacts_empty_store(self, tmp_path):
        store = ArtifactStore(str(tmp_path / "store"))
        assert store.list_artifacts() == []


# =====================================================================
# reproducibility/run_hasher.py
# =====================================================================
from src.reproducibility.run_hasher import RunHasher


class TestRunHasher:
    def _make_config(self, tmp_path):
        @dataclass
        class FakeConfig:
            teams_json: Optional[str] = None
            torvik_json: Optional[str] = None
            historical_games_json: Optional[str] = None
            sports_reference_json: Optional[str] = None
            public_picks_json: Optional[str] = None
            scoring_rules_json: Optional[str] = None
            roster_json: Optional[str] = None
            transfer_portal_json: Optional[str] = None
            preseason_ap_json: Optional[str] = None
            coach_tournament_json: Optional[str] = None
            conf_champions_json: Optional[str] = None
            injury_report_json: Optional[str] = None
            venue_locations_json: Optional[str] = None
            team_locations_json: Optional[str] = None
            bracket_json: Optional[str] = None

        f = tmp_path / "teams.json"
        f.write_text("{}")
        return FakeConfig(teams_json=str(f))

    def test_hash_all_inputs(self, tmp_path):
        cfg = self._make_config(tmp_path)
        hasher = RunHasher(cfg)
        hashes = hasher.hash_all_inputs()
        assert len(hashes) == 1
        assert len(list(hashes.values())[0]) == 64

    def test_hash_all_inputs_skips_none(self, tmp_path):
        cfg = self._make_config(tmp_path)
        cfg.teams_json = None
        hasher = RunHasher(cfg)
        assert hasher.hash_all_inputs() == {}

    def test_hash_config(self, tmp_path):
        cfg = self._make_config(tmp_path)
        hasher = RunHasher(cfg)
        h = hasher.hash_config()
        assert len(h) == 64

    def test_compute_reproducibility_hash(self, tmp_path):
        cfg = self._make_config(tmp_path)
        hasher = RunHasher(cfg)
        h = hasher.compute_reproducibility_hash("v1.0")
        assert len(h) == 64

    def test_verify_against_match(self, tmp_path):
        cfg = self._make_config(tmp_path)
        hasher = RunHasher(cfg)
        current = hasher.hash_all_inputs()
        record = MagicMock()
        record.dataset_hashes = current
        assert hasher.verify_against(record) is True

    def test_verify_against_mismatch(self, tmp_path):
        cfg = self._make_config(tmp_path)
        hasher = RunHasher(cfg)
        record = MagicMock()
        record.dataset_hashes = {str(tmp_path / "teams.json"): "wrong_hash"}
        assert hasher.verify_against(record) is False

    def test_verify_against_empty(self, tmp_path):
        cfg = self._make_config(tmp_path)
        hasher = RunHasher(cfg)
        record = MagicMock()
        record.dataset_hashes = {}
        assert hasher.verify_against(record) is False


# =====================================================================
# exports/kaggle.py
# =====================================================================
from src.exports.kaggle import (
    KaggleExportStats,
    build_team_id_map,
    generate_predictions,
    is_womens_team,
    load_kaggle_teams,
    parse_kaggle_id,
)


class TestKaggleHelpers:
    def test_parse_kaggle_id_valid(self):
        season, t1, t2 = parse_kaggle_id("2024_1101_1102")
        assert season == 2024
        assert t1 == 1101
        assert t2 == 1102

    def test_parse_kaggle_id_invalid(self):
        with pytest.raises(ValueError):
            parse_kaggle_id("invalid")

    def test_parse_kaggle_id_none(self):
        with pytest.raises(ValueError):
            parse_kaggle_id(None)

    def test_is_womens_team(self):
        assert is_womens_team(3001) is True
        assert is_womens_team(2999) is False
        assert is_womens_team(3000) is True

    def test_kaggle_export_stats_to_dict(self):
        s = KaggleExportStats(total_rows=10, mapped_rows=8)
        d = s.to_dict()
        assert d["total_rows"] == 10
        assert d["mapped_rows"] == 8


class TestLoadKaggleTeams:
    def test_load_valid(self, tmp_path):
        f = tmp_path / "MTeams.csv"
        f.write_text("TeamID,TeamName\n1101,Duke\n1102,Kansas\n")
        result = load_kaggle_teams(str(f))
        assert result[1101] == "Duke"
        assert result[1102] == "Kansas"

    def test_load_missing_columns(self, tmp_path):
        f = tmp_path / "bad.csv"
        f.write_text("ID,Name\n1,Duke\n")
        with pytest.raises(ValueError):
            load_kaggle_teams(str(f))

    def test_load_invalid_id(self, tmp_path):
        f = tmp_path / "MTeams.csv"
        f.write_text("TeamID,TeamName\nabc,Duke\n1102,Kansas\n")
        result = load_kaggle_teams(str(f))
        assert 1102 in result
        assert len(result) == 1


class TestBuildTeamIdMap:
    def test_basic(self):
        resolver = TeamNameResolver()
        team_map = {1101: "Duke", 1102: "Kansas"}
        result = build_team_id_map(team_map, resolver)
        assert result[1101] == "duke"
        assert result[1102] == "kansas"

    def test_unmappable(self):
        resolver = TeamNameResolver()
        team_map = {9999: "XYZZY Completely Unknown"}
        result = build_team_id_map(team_map, resolver)
        # Low confidence may still map, or may be excluded
        # The point is it doesn't crash


class TestGeneratePredictions:
    def test_basic(self):
        df = pd.DataFrame({"ID": ["2024_1101_1102"], "Pred": [0.5]})
        id_map = {1101: "duke", 1102: "kansas"}
        predict_fn = lambda t1, t2: 0.75
        result = generate_predictions(df, id_map, predict_fn)
        assert result["Pred"].iloc[0] == 0.75

    def test_bad_id(self):
        df = pd.DataFrame({"ID": ["bad_id"], "Pred": [0.5]})
        result = generate_predictions(df, {}, lambda t1, t2: 0.5)
        assert result["Pred"].iloc[0] == 0.5

    def test_season_filter(self):
        df = pd.DataFrame({"ID": ["2024_1101_1102", "2025_1101_1102"], "Pred": [0.5, 0.5]})
        id_map = {1101: "duke", 1102: "kansas"}
        result = generate_predictions(df, id_map, lambda t1, t2: 0.8, season_filter=2024)
        assert result["Pred"].iloc[0] == 0.8
        assert result["Pred"].iloc[1] == 0.5  # filtered out

    def test_unmapped_team(self):
        df = pd.DataFrame({"ID": ["2024_9999_9998"], "Pred": [0.5]})
        result = generate_predictions(df, {}, lambda t1, t2: 0.5)
        assert result["Pred"].iloc[0] == 0.5

    def test_predict_failure(self):
        df = pd.DataFrame({"ID": ["2024_1101_1102"], "Pred": [0.5]})
        id_map = {1101: "duke", 1102: "kansas"}
        def bad_fn(t1, t2):
            raise RuntimeError("boom")
        result = generate_predictions(df, id_map, bad_fn)
        assert result["Pred"].iloc[0] == 0.5

    def test_missing_id_column(self):
        df = pd.DataFrame({"bad_col": ["x"]})
        with pytest.raises(ValueError):
            generate_predictions(df, {}, lambda t1, t2: 0.5)

    def test_clipping(self):
        df = pd.DataFrame({"ID": ["2024_1101_1102"], "Pred": [0.5]})
        id_map = {1101: "duke", 1102: "kansas"}
        result = generate_predictions(df, id_map, lambda t1, t2: -0.5)
        assert result["Pred"].iloc[0] == 0.0

    def test_womens_routing(self):
        df = pd.DataFrame({"ID": ["2024_3001_3002"], "Pred": [0.5]})
        mens_map = {}
        womens_map = {3001: "w_team_a", 3002: "w_team_b"}
        mens_fn = lambda t1, t2: 0.1
        womens_fn = lambda t1, t2: 0.9
        result = generate_predictions(
            df, mens_map, mens_fn,
            womens_id_map=womens_map, womens_predict_fn=womens_fn,
        )
        assert result["Pred"].iloc[0] == 0.9


# =====================================================================
# exports/deliverables_manager.py
# =====================================================================
from src.exports.deliverables_manager import (
    DecisionRecord,
    DeliverableManifest,
    DeliverablesManager,
)


class TestDecisionRecord:
    def test_defaults(self):
        dr = DecisionRecord()
        assert dr.timestamp != ""
        assert dr.ensemble_weights == {}

    def test_to_dict(self):
        dr = DecisionRecord(ensemble_rationale="test")
        d = dr.to_dict()
        assert d["ensemble_rationale"] == "test"


class TestDeliverableManifest:
    def test_defaults(self):
        dm = DeliverableManifest()
        assert dm.timestamp != ""
        assert dm.year == 2026

    def test_to_dict(self):
        dm = DeliverableManifest(year=2025, mode="production")
        d = dm.to_dict()
        assert d["year"] == 2025


class TestDeliverablesManager:
    def test_init_creates_dirs(self, tmp_path):
        mgr = DeliverablesManager(year=2025, mode="test", output_base=str(tmp_path), timestamp="20250101_000000")
        assert mgr.output_dir.exists()
        assert (mgr.output_dir / "predictions").is_dir()
        assert (mgr.output_dir / "reports").is_dir()
        assert (mgr.output_dir / "audit").is_dir()
        assert (mgr.output_dir / "metadata").is_dir()

    def test_export_predictions(self, tmp_path):
        mgr = DeliverablesManager(output_base=str(tmp_path), timestamp="t")
        path = mgr.export_predictions({"duke_unc": 0.75})
        assert Path(path).exists()
        data = json.loads(Path(path).read_text())
        assert data["predictions"]["duke_unc"] == 0.75

    def test_export_predictions_with_ci(self, tmp_path):
        mgr = DeliverablesManager(output_base=str(tmp_path), timestamp="t")
        ci = {"duke_unc": {"lower": 0.6, "upper": 0.9, "mean": 0.75}}
        path = mgr.export_predictions({"duke_unc": 0.75}, confidence_intervals=ci)
        ci_path = mgr.output_dir / "predictions" / "confidence_intervals.json"
        assert ci_path.exists()

    def test_export_risk_report(self, tmp_path):
        mgr = DeliverablesManager(output_base=str(tmp_path), timestamp="t")
        path = mgr.export_risk_report({"drawdown": 0.05, "tail_loss": 0.02})
        assert Path(path).exists()
        summary_path = mgr.output_dir / "reports" / "risk_summary.txt"
        assert summary_path.exists()

    def test_export_regime_analysis(self, tmp_path):
        mgr = DeliverablesManager(output_base=str(tmp_path), timestamp="t")
        path = mgr.export_regime_analysis({"regime": "normal"})
        assert Path(path).exists()

    def test_export_scenario_analysis(self, tmp_path):
        mgr = DeliverablesManager(output_base=str(tmp_path), timestamp="t")
        path = mgr.export_scenario_analysis({"scenario": "base"})
        assert Path(path).exists()

    def test_export_decision_record(self, tmp_path):
        mgr = DeliverablesManager(output_base=str(tmp_path), timestamp="t")
        dr = DecisionRecord(ensemble_rationale="equal weights")
        path = mgr.export_decision_record(dr)
        assert Path(path).exists()

    def test_export_config_snapshot(self, tmp_path):
        mgr = DeliverablesManager(output_base=str(tmp_path), timestamp="t")
        path = mgr.export_config_snapshot({"year": 2026})
        assert Path(path).exists()

    def test_export_evaluation_matrix(self, tmp_path):
        mgr = DeliverablesManager(output_base=str(tmp_path), timestamp="t")
        path = mgr.export_evaluation_matrix({"brier": 0.18})
        assert Path(path).exists()

    def test_set_provenance(self, tmp_path):
        mgr = DeliverablesManager(output_base=str(tmp_path), timestamp="t")
        mgr.set_provenance(config_hash="abc", code_version="v1", data_hashes={"f": "h"})
        assert mgr._manifest.config_hash == "abc"
        assert mgr._manifest.code_version == "v1"

    def test_finalize(self, tmp_path):
        mgr = DeliverablesManager(output_base=str(tmp_path), timestamp="t")
        mgr.export_predictions({"a": 0.5})
        path = mgr.finalize()
        assert Path(path).exists()
        assert mgr._finalized is True

    def test_summary(self, tmp_path):
        mgr = DeliverablesManager(output_base=str(tmp_path), timestamp="t")
        mgr.export_predictions({"a": 0.5})
        s = mgr.summary()
        assert "In progress" in s

    def test_summary_finalized(self, tmp_path):
        mgr = DeliverablesManager(output_base=str(tmp_path), timestamp="t")
        mgr.finalize()
        s = mgr.summary()
        assert "FINALIZED" in s

    def test_risk_report_with_nested_dict(self, tmp_path):
        mgr = DeliverablesManager(output_base=str(tmp_path), timestamp="t")
        path = mgr.export_risk_report({"metrics": {"a": 1, "b": 2}, "count": 5})
        summary_path = mgr.output_dir / "reports" / "risk_summary.txt"
        content = summary_path.read_text()
        assert "count" in content
