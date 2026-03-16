"""Extra coverage tests for pipeline stage modules with low coverage.

Covers:
- orchestration.py: pre-checks, validation functions
- context.py: PipelineContext, get_code_version
- game_utils.py: edge cases in utility functions
- ev_mode.py: leverage, strategy classification, pool rank
- inference.py: embedding probability, symmetrize, seed prior
- robustness_stage.py: RobustnessReport, RobustnessStage, _noop_ctx
- probability_pipeline.py: production + experimental functions, ablation
- production_baseline.py: spec validation, sanctioning checks
"""

from __future__ import annotations

import math
import logging
from datetime import date, datetime, timezone
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# 1. orchestration.py
# ---------------------------------------------------------------------------

class TestOrchestrationPreChecks:
    """Test orchestration pre-check functions with mocked pipeline."""

    def _make_pipeline(self, **overrides):
        """Build a MagicMock pipeline with sensible defaults."""
        p = MagicMock()
        p.config = MagicMock()
        p.config.year = 2025
        p.config.strict_leakage_mode = False
        p.config.require_freeze_file = False
        p.config.freeze_file = None
        p.config.multi_year_games_dir = None
        p.config.kaggle_dir = None
        p.config.data_cache_dir = "/tmp/cache"
        p.config.training_years = []
        p._experiment_registry = MagicMock()
        p._experiment_registry.best.return_value = None
        p._load_mc_calibration = MagicMock(return_value=None)
        for k, v in overrides.items():
            setattr(p, k, v) if not k.startswith("config.") else setattr(p.config, k.split(".", 1)[1], v)
        return p

    def test_run_pre_checks_basic_2025(self):
        """run_pre_checks succeeds for year < 2026 without freeze."""
        from src.pipeline.stages.orchestration import run_pre_checks

        p = self._make_pipeline()
        # Patch internal helpers to isolate run_pre_checks logic
        with patch("src.pipeline.stages.orchestration._check_dataset_hashes"), \
             patch("src.pipeline.stages.orchestration._check_holdout_contamination"), \
             patch("src.pipeline.stages.orchestration._validate_year_split_policy"), \
             patch("src.pipeline.stages.orchestration._check_freeze_requirement", return_value=None), \
             patch("src.pipeline.stages.orchestration._check_freeze_for_season"), \
             patch("src.pipeline.stages.orchestration._auto_detect_kaggle_dir"):
            result = run_pre_checks(p)
        assert result is None
        p._load_mc_calibration.assert_called_once()

    def test_run_pre_checks_2026_requires_mc_calibration(self):
        """year >= 2026 without MC calibration raises DataRequirementError."""
        from src.pipeline.stages.orchestration import run_pre_checks
        from src.pipeline.config import DataRequirementError

        p = self._make_pipeline()
        p.config.year = 2026
        p.config.require_freeze_file = True
        p._load_mc_calibration.return_value = None

        with patch("src.pipeline.stages.orchestration._check_dataset_hashes"), \
             patch("src.pipeline.stages.orchestration._check_holdout_contamination"), \
             patch("src.pipeline.stages.orchestration._validate_year_split_policy"):
            with pytest.raises(DataRequirementError, match="MC calibration"):
                run_pre_checks(p)

    def test_run_pre_checks_2026_requires_freeze(self):
        """year >= 2026 without require_freeze_file raises DataRequirementError."""
        from src.pipeline.stages.orchestration import run_pre_checks
        from src.pipeline.config import DataRequirementError

        p = self._make_pipeline()
        p.config.year = 2026
        p.config.require_freeze_file = False
        p._load_mc_calibration.return_value = MagicMock()  # MC cal present

        with patch("src.pipeline.stages.orchestration._check_dataset_hashes"), \
             patch("src.pipeline.stages.orchestration._check_holdout_contamination"), \
             patch("src.pipeline.stages.orchestration._validate_year_split_policy"):
            with pytest.raises(DataRequirementError, match="freeze required"):
                run_pre_checks(p)

    def test_check_dataset_hashes_no_incumbent(self):
        """_check_dataset_hashes succeeds when no incumbent exists."""
        from src.pipeline.stages.orchestration import _check_dataset_hashes

        p = self._make_pipeline()
        mock_hasher = MagicMock()
        mock_hasher.hash_all_inputs.return_value = {"file1": "abc123"}

        with patch("src.reproducibility.run_hasher.RunHasher", return_value=mock_hasher):
            _check_dataset_hashes(p)

        assert p._dataset_hashes == {"file1": "abc123"}

    def test_check_dataset_hashes_incumbent_mismatch_strict(self):
        """_check_dataset_hashes raises LeakageError in strict mode on mismatch."""
        from src.pipeline.stages.orchestration import _check_dataset_hashes
        from src.exceptions import LeakageError

        p = self._make_pipeline()
        p.config.strict_leakage_mode = True
        incumbent = MagicMock()
        incumbent.dataset_hashes = {"file1": "old_hash"}
        incumbent.experiment_id = "exp-001"
        p._experiment_registry.best.return_value = incumbent

        mock_hasher = MagicMock()
        mock_hasher.hash_all_inputs.return_value = {"file1": "new_hash"}
        mock_hasher.verify_against.return_value = False

        with patch("src.reproducibility.run_hasher.RunHasher", return_value=mock_hasher):
            with pytest.raises(LeakageError):
                _check_dataset_hashes(p)

    def test_check_dataset_hashes_incumbent_match(self):
        """_check_dataset_hashes passes when hashes match incumbent."""
        from src.pipeline.stages.orchestration import _check_dataset_hashes

        p = self._make_pipeline()
        incumbent = MagicMock()
        incumbent.dataset_hashes = {"file1": "abc123"}
        incumbent.experiment_id = "exp-001"
        p._experiment_registry.best.return_value = incumbent

        mock_hasher = MagicMock()
        mock_hasher.hash_all_inputs.return_value = {"file1": "abc123"}
        mock_hasher.verify_against.return_value = True

        with patch("src.reproducibility.run_hasher.RunHasher", return_value=mock_hasher):
            _check_dataset_hashes(p)

    def test_check_dataset_hashes_exception_fallback(self):
        """_check_dataset_hashes gracefully handles import errors."""
        from src.pipeline.stages.orchestration import _check_dataset_hashes

        p = self._make_pipeline()
        # Patch at the source so the lazy import inside the function gets the mock
        with patch.dict("sys.modules", {"src.reproducibility.run_hasher": MagicMock(RunHasher=MagicMock(side_effect=RuntimeError("fail")))}):
            _check_dataset_hashes(p)
        assert p._run_hasher is None
        assert p._dataset_hashes == {}

    def test_check_holdout_contamination_clean(self):
        """_check_holdout_contamination passes when no contamination found."""
        from src.pipeline.stages.orchestration import _check_holdout_contamination

        p = self._make_pipeline()
        mock_module = MagicMock()
        mock_module.check_holdout_contamination.return_value = None
        with patch.dict("sys.modules", {"src.ml.evaluation.rdof_audit": mock_module}):
            _check_holdout_contamination(p)

    def test_check_holdout_contamination_with_contamination(self):
        """_check_holdout_contamination logs warning when contamination found.

        Note: The function catches all exceptions (including LeakageError)
        in its broad except clause, so it never propagates.
        """
        from src.pipeline.stages.orchestration import _check_holdout_contamination

        p = self._make_pipeline()
        p.config.strict_leakage_mode = True

        mock_module = MagicMock()
        mock_module.check_holdout_contamination.return_value = {"message": "overlap detected"}
        with patch.dict("sys.modules", {"src.ml.evaluation.rdof_audit": mock_module}):
            # The LeakageError raised inside is caught by the broad except clause
            _check_holdout_contamination(p)
        mock_module.check_holdout_contamination.assert_called_once()

    def test_validate_year_split_policy_success(self):
        """_validate_year_split_policy stores policy on pipeline."""
        from src.pipeline.stages.orchestration import _validate_year_split_policy

        p = self._make_pipeline()
        p.config.training_years = [2020, 2021]

        # Use a simple namespace instead of MagicMock to avoid the
        # assert_* method name conflict with MagicMock internals.
        dev_only_calls = []

        class FakePolicy:
            dev_years = {2020, 2021, 2022}
            holdout_years = {2023}
            def assert_dev_only(self, years, context=""):
                dev_only_calls.append((years, context))

        fake_policy = FakePolicy()
        mock_module = MagicMock()
        mock_module.YearSplitPolicy.from_config.return_value = fake_policy
        with patch.dict("sys.modules", {"src.ml.evaluation.evaluation_integrity": mock_module}):
            _validate_year_split_policy(p)

        assert p._year_split_policy is fake_policy
        assert len(dev_only_calls) == 1

    def test_validate_year_split_policy_strict_failure(self):
        """_validate_year_split_policy raises in strict mode on failure."""
        from src.pipeline.stages.orchestration import _validate_year_split_policy
        from src.pipeline.config import DataRequirementError

        p = self._make_pipeline()
        p.config.strict_leakage_mode = True

        mock_module = MagicMock()
        mock_module.YearSplitPolicy.from_config.side_effect = ValueError("bad config")
        with patch.dict("sys.modules", {"src.ml.evaluation.evaluation_integrity": mock_module}):
            with pytest.raises(DataRequirementError, match="Year-split policy"):
                _validate_year_split_policy(p)

    def test_validate_year_split_policy_no_training_years(self):
        """_validate_year_split_policy skips assert_dev_only when no training years."""
        from src.pipeline.stages.orchestration import _validate_year_split_policy

        p = self._make_pipeline()
        p.config.training_years = []

        dev_only_calls = []

        class FakePolicy:
            dev_years = {2020}
            holdout_years = {2023}
            def assert_dev_only(self, years, context=""):
                dev_only_calls.append((years, context))

        fake_policy = FakePolicy()
        mock_module = MagicMock()
        mock_module.YearSplitPolicy.from_config.return_value = fake_policy
        with patch.dict("sys.modules", {"src.ml.evaluation.evaluation_integrity": mock_module}):
            _validate_year_split_policy(p)

        assert len(dev_only_calls) == 0

    def test_check_freeze_requirement_no_freeze_required(self):
        """_check_freeze_requirement returns None when freeze not required."""
        from src.pipeline.stages.orchestration import _check_freeze_requirement

        p = self._make_pipeline()
        p.config.require_freeze_file = False
        assert _check_freeze_requirement(p) is None

    def test_check_freeze_requirement_no_file(self):
        """_check_freeze_requirement raises when freeze required but no file."""
        from src.pipeline.stages.orchestration import _check_freeze_requirement
        from src.pipeline.config import DataRequirementError

        p = self._make_pipeline()
        p.config.require_freeze_file = True
        p.config.freeze_file = None

        with pytest.raises(DataRequirementError, match="no --freeze-file"):
            _check_freeze_requirement(p)

    def test_check_freeze_requirement_valid(self):
        """_check_freeze_requirement returns verification dict on success."""
        from src.pipeline.stages.orchestration import _check_freeze_requirement

        p = self._make_pipeline()
        p.config.require_freeze_file = True
        p.config.freeze_file = "/tmp/freeze.json"

        mock_module = MagicMock()
        mock_module.verify_freeze.return_value = {"matches": True}
        with patch.dict("sys.modules", {"src.ml.evaluation.rdof_audit": mock_module}):
            result = _check_freeze_requirement(p)
        assert result == {"matches": True}

    def test_check_freeze_requirement_mismatch(self):
        """_check_freeze_requirement raises on verification mismatch."""
        from src.pipeline.stages.orchestration import _check_freeze_requirement
        from src.pipeline.config import DataRequirementError

        p = self._make_pipeline()
        p.config.require_freeze_file = True
        p.config.freeze_file = "/tmp/freeze.json"

        mock_module = MagicMock()
        mock_module.verify_freeze.return_value = {"matches": False, "mismatches": ["hash differs"]}
        with patch.dict("sys.modules", {"src.ml.evaluation.rdof_audit": mock_module}):
            with pytest.raises(DataRequirementError, match="Freeze verification failed"):
                _check_freeze_requirement(p)

    def test_auto_detect_kaggle_dir_already_set(self):
        """_auto_detect_kaggle_dir is a no-op when kaggle_dir is already set."""
        from src.pipeline.stages.orchestration import _auto_detect_kaggle_dir

        p = self._make_pipeline()
        p.config.kaggle_dir = "/existing/kaggle"
        _auto_detect_kaggle_dir(p)
        assert p.config.kaggle_dir == "/existing/kaggle"

    def test_auto_detect_kaggle_dir_fallback(self):
        """_auto_detect_kaggle_dir tries ensure_kaggle_data then directory scan."""
        from src.pipeline.stages.orchestration import _auto_detect_kaggle_dir

        p = self._make_pipeline()
        p.config.kaggle_dir = None

        mock_module = MagicMock()
        mock_module.ensure_kaggle_data.return_value = "/resolved/kaggle"
        with patch.dict("sys.modules", {"src.data.kaggle_downloader": mock_module}):
            _auto_detect_kaggle_dir(p)
        assert p.config.kaggle_dir == "/resolved/kaggle"

    def test_check_freeze_for_season_pre2026_no_freeze(self):
        """_check_freeze_for_season logs info for pre-2026 without freeze."""
        from src.pipeline.stages.orchestration import _check_freeze_for_season

        p = self._make_pipeline()
        p.config.year = 2024
        p.config.require_freeze_file = False

        mock_module = MagicMock()
        mock_module.require_freeze_for_season.return_value = {"verified": False}
        with patch.dict("sys.modules", {"src.ml.evaluation.evaluation_integrity": mock_module}):
            _check_freeze_for_season(p, None)  # Should not raise


class TestModeGatedResult:
    """Test ModeGatedResult dataclass."""

    def test_defaults(self):
        from src.pipeline.stages.orchestration import ModeGatedResult
        r = ModeGatedResult()
        assert r.public_picks == {}
        assert r.scoring_system is None
        assert r.pool_analysis is None
        assert r.leverage_preview == []
        assert r.bracket_portfolio_stats == {}


# ---------------------------------------------------------------------------
# 2. context.py
# ---------------------------------------------------------------------------

class TestPipelineContext:
    """Test PipelineContext dataclass and helpers."""

    def test_basic_construction(self):
        from src.pipeline.stages.context import PipelineContext
        ctx = PipelineContext(config=MagicMock())
        assert ctx.phase_timer is None
        assert ctx.resource_tracker is None
        assert ctx.experiment_registry is None
        assert ctx.logger is not None

    def test_from_pipeline(self):
        from src.pipeline.stages.context import PipelineContext

        pipeline = MagicMock()
        pipeline.__class__.__name__ = "TestPipeline"
        pipeline.config = MagicMock()
        pipeline._phase_timer = MagicMock()
        pipeline._resource_tracker = MagicMock()
        pipeline._experiment_registry = MagicMock()

        ctx = PipelineContext.from_pipeline(pipeline)
        assert ctx.config is pipeline.config
        assert ctx.phase_timer is pipeline._phase_timer
        assert ctx.resource_tracker is pipeline._resource_tracker
        assert ctx.experiment_registry is pipeline._experiment_registry

    def test_from_pipeline_missing_attrs(self):
        """from_pipeline handles missing optional attributes gracefully."""
        from src.pipeline.stages.context import PipelineContext

        pipeline = MagicMock(spec=[])  # No attributes
        pipeline.config = MagicMock()
        pipeline.__class__.__name__ = "Bare"

        ctx = PipelineContext.from_pipeline(pipeline)
        assert ctx.phase_timer is None
        assert ctx.resource_tracker is None

    def test_get_code_version_returns_string(self):
        from src.pipeline.stages.context import get_code_version
        result = get_code_version()
        assert isinstance(result, str)
        assert len(result) <= 12

    def test_get_code_version_fallback(self):
        """get_code_version returns 'unknown' when git fails."""
        from src.pipeline.stages.context import get_code_version
        with patch("subprocess.check_output", side_effect=FileNotFoundError):
            assert get_code_version() == "unknown"


# ---------------------------------------------------------------------------
# 3. game_utils.py
# ---------------------------------------------------------------------------

class TestGameUtils:
    """Test game utility functions."""

    def test_normalize_team_key(self):
        from src.pipeline.stages.game_utils import normalize_team_key
        assert normalize_team_key("  Duke  ") == "duke"
        assert normalize_team_key("Michigan St") == "michigan"
        assert normalize_team_key("Ohio State") == "ohio"
        assert normalize_team_key("Harvard Univ") == "harvard"
        assert normalize_team_key("Stanford University") == "stanford"

    def test_normalize_key(self):
        from src.pipeline.stages.game_utils import normalize_key
        assert normalize_key("A & B") == "a_and_b"
        assert normalize_key("foo-bar") == "foo_bar"
        assert normalize_key("  hello world  ") == "hello_world"

    def test_is_tournament_game_known_year(self):
        from src.pipeline.stages.game_utils import is_tournament_game
        # 2024 tournament starts March 19
        assert is_tournament_game(date(2024, 3, 20), 2024) is True
        assert is_tournament_game(date(2024, 1, 15), 2024) is False

    def test_is_tournament_game_unknown_year(self):
        from src.pipeline.stages.game_utils import is_tournament_game
        # Unknown year defaults to March 15
        assert is_tournament_game(date(2099, 3, 16), 2099) is True
        assert is_tournament_game(date(2099, 3, 14), 2099) is False

    def test_detect_tournament_game(self):
        from src.pipeline.stages.game_utils import detect_tournament_game
        assert detect_tournament_game("2024-03-25", 2024) is True
        assert detect_tournament_game("2024-01-10", 2024) is False
        # After tournament end
        assert detect_tournament_game("2024-04-20", 2024) is False

    def test_detect_tournament_game_bad_date(self):
        from src.pipeline.stages.game_utils import detect_tournament_game
        # Unparseable date returns False
        assert detect_tournament_game("not-a-date", 2024) is False

    def test_game_sort_key(self):
        from src.pipeline.stages.game_utils import game_sort_key
        g = {"date": "2024-03-15", "opponent": "Duke"}
        assert game_sort_key(g) == ("2024-03-15", "Duke")
        assert game_sort_key({}) == ("", "")

    def test_compute_game_sort_key(self):
        from src.pipeline.stages.game_utils import compute_game_sort_key
        assert compute_game_sort_key("2024-03-15", 2024) == 20240315
        # Fallback for bad date
        assert compute_game_sort_key("", 2024) == 20240101

    def test_parse_game_date(self):
        from src.pipeline.stages.game_utils import parse_game_date
        assert parse_game_date("2024-03-15") == date(2024, 3, 15)
        assert parse_game_date("03/15/2024") == date(2024, 3, 15)
        assert parse_game_date("2024/03/15") == date(2024, 3, 15)
        assert parse_game_date("03-15-2024") == date(2024, 3, 15)
        assert parse_game_date("garbage") is None

    def test_parse_timestamp(self):
        from src.pipeline.stages.game_utils import parse_timestamp
        # ISO format
        ts = parse_timestamp("2024-03-15T10:30:00")
        assert ts is not None
        assert ts.year == 2024

        # Z suffix
        ts = parse_timestamp("2024-03-15T10:30:00Z")
        assert ts is not None

        # Empty / None
        assert parse_timestamp("") is None
        assert parse_timestamp(None) is None

        # Bad value
        assert parse_timestamp("not-a-timestamp") is None

    def test_game_outcome_scores(self):
        from src.pipeline.stages.game_utils import game_outcome
        g = MagicMock()
        g.team1_score = 80
        g.team2_score = 70
        g.lead_history = None
        assert game_outcome(g) == 1

        g.team1_score = 60
        g.team2_score = 75
        assert game_outcome(g) == 0

    def test_game_outcome_zero_scores(self):
        from src.pipeline.stages.game_utils import game_outcome
        g = MagicMock()
        g.team1_score = 0
        g.team2_score = 0
        g.lead_history = [5, 3, -2]
        # total == 0, so falls to lead_history; last is negative
        assert game_outcome(g) == 0

    def test_game_outcome_lead_history_only(self):
        from src.pipeline.stages.game_utils import game_outcome
        g = MagicMock()
        g.team1_score = None
        g.team2_score = None
        g.lead_history = [1, 2, 5]
        assert game_outcome(g) == 1

    def test_game_outcome_none(self):
        from src.pipeline.stages.game_utils import game_outcome
        g = MagicMock()
        g.team1_score = None
        g.team2_score = None
        g.lead_history = []
        assert game_outcome(g) is None

    def test_compute_game_outcome(self):
        from src.pipeline.stages.game_utils import compute_game_outcome
        win, margin = compute_game_outcome(80, 70)
        assert win is True
        assert margin == 10

        win, margin = compute_game_outcome(60, 75)
        assert win is False
        assert margin == -15

    def test_coerce_game_date_various_formats(self):
        from src.pipeline.stages.game_utils import coerce_game_date
        assert coerce_game_date("2024-03-15") == "2024-03-15"
        assert coerce_game_date("2024/03/15") == "2024-03-15"
        assert coerce_game_date("03/15/2024") == "2024-03-15"
        assert coerce_game_date("2024-03-15T10:30:00") == "2024-03-15"
        assert coerce_game_date("2024-03-15T10:30:00Z") == "2024-03-15"

    def test_coerce_game_date_with_T_fallback(self):
        from src.pipeline.stages.game_utils import coerce_game_date
        # Unparseable but contains T
        assert coerce_game_date("badTvalue") == "bad"

    def test_coerce_game_date_empty(self):
        from src.pipeline.stages.game_utils import coerce_game_date
        assert coerce_game_date(None, fallback_year=2024) == "2024-01-01"
        assert coerce_game_date("", fallback_year=2024) == "2024-01-01"
        # Default fallback year
        assert coerce_game_date(None) == "2025-01-01"

    def test_coerce_game_date_with_logging(self):
        from src.pipeline.stages.game_utils import coerce_game_date
        # game_id triggers logging path
        result = coerce_game_date("", fallback_year=2024, game_id="G001", source="test")
        assert result == "2024-01-01"

    def test_is_target_season_game(self):
        from src.pipeline.stages.game_utils import is_target_season_game
        # Within season window
        assert is_target_season_game("2024-01-15", 2024) is True
        assert is_target_season_game("2023-11-15", 2024) is True
        # Outside window
        assert is_target_season_game("2023-06-15", 2024) is False
        assert is_target_season_game("2024-05-15", 2024) is False

    def test_is_target_season_game_bad_date(self):
        from src.pipeline.stages.game_utils import is_target_season_game
        # Bad date falls back, coerce returns raw string, strptime fails -> True
        assert is_target_season_game("invalid", 2024) is True

    def test_validate_source_coverage_ok(self):
        from src.pipeline.stages.game_utils import validate_source_coverage
        coverage = {"t1": object(), "t2": object(), "t3": object()}
        validate_source_coverage("TestSource", coverage, n_teams=4, min_ratio=0.5)

    def test_validate_source_coverage_too_low(self):
        from src.pipeline.stages.game_utils import validate_source_coverage
        with pytest.raises(ValueError, match="coverage is too low"):
            validate_source_coverage("TestSource", {"t1": 1}, n_teams=10, min_ratio=0.5)

    def test_validate_source_coverage_no_teams(self):
        from src.pipeline.stages.game_utils import validate_source_coverage
        with pytest.raises(ValueError, match="No tournament teams"):
            validate_source_coverage("TestSource", {}, n_teams=0, min_ratio=0.5)


# ---------------------------------------------------------------------------
# 4. ev_mode.py
# ---------------------------------------------------------------------------

class TestEvMode:
    """Test EV mode utility functions."""

    def test_compute_leverage_basic(self):
        from src.pipeline.stages.ev_mode import compute_leverage
        lev = compute_leverage(0.7, 0.3, scoring_weight=1.0)
        assert lev == pytest.approx((0.7 - 0.3) / 0.3)

    def test_compute_leverage_zero_public(self):
        from src.pipeline.stages.ev_mode import compute_leverage
        assert compute_leverage(0.5, 0.0) == 0.0

    def test_compute_leverage_small_public(self):
        from src.pipeline.stages.ev_mode import compute_leverage
        # public_pick_rate < 0.01 gets clamped to 0.01
        lev = compute_leverage(0.5, 0.005, scoring_weight=2.0)
        assert lev == pytest.approx(2.0 * (0.5 - 0.005) / 0.01)

    def test_classify_pick_strategy_neutral(self):
        from src.pipeline.stages.ev_mode import classify_pick_strategy
        assert classify_pick_strategy(0.05, 0.5, 0.5) == "neutral"

    def test_classify_pick_strategy_contrarian(self):
        from src.pipeline.stages.ev_mode import classify_pick_strategy
        # leverage > 0.1, model_prob > public, public < 0.3
        assert classify_pick_strategy(0.5, 0.6, 0.2) == "contrarian"

    def test_classify_pick_strategy_chalk(self):
        from src.pipeline.stages.ev_mode import classify_pick_strategy
        # leverage > 0.1, model_prob > public, public >= 0.3
        assert classify_pick_strategy(0.5, 0.8, 0.6) == "chalk"

    def test_classify_pick_strategy_fade(self):
        from src.pipeline.stages.ev_mode import classify_pick_strategy
        assert classify_pick_strategy(-0.5, 0.3, 0.7) == "fade"

    def test_expected_pool_rank_single(self):
        from src.pipeline.stages.ev_mode import expected_pool_rank
        assert expected_pool_rank(100, 1) == 1.0
        assert expected_pool_rank(100, 0) == 1.0

    def test_expected_pool_rank_zero_std(self):
        from src.pipeline.stages.ev_mode import expected_pool_rank
        assert expected_pool_rank(100, 50, score_std=0) == 1.0

    def test_expected_pool_rank_normal(self):
        from src.pipeline.stages.ev_mode import expected_pool_rank
        # With z=0, prob_beat=0.5, rank = pool_size * 0.5
        rank = expected_pool_rank(100, 100, score_std=15.0)
        assert rank == pytest.approx(50.0)

    def test_format_leverage_table(self):
        from src.pipeline.stages.ev_mode import format_leverage_table
        teams = [
            {"team": "Duke", "model_prob": 0.8, "public_rate": 0.5, "leverage": 0.6, "strategy": "chalk"},
            {"team": "UNC", "model_prob": 0.3, "public_rate": 0.6, "leverage": -0.5, "strategy": "fade"},
        ]
        table = format_leverage_table(teams, top_n=2)
        assert "Duke" in table
        assert "UNC" in table
        assert "Team" in table  # header

    def test_format_leverage_table_empty(self):
        from src.pipeline.stages.ev_mode import format_leverage_table
        table = format_leverage_table([], top_n=5)
        assert "Team" in table  # header still present


# ---------------------------------------------------------------------------
# 5. inference.py
# ---------------------------------------------------------------------------

class TestInference:
    """Test inference utility functions."""

    def test_embedding_probability_none_vectors(self):
        from src.pipeline.stages.inference import embedding_probability
        assert embedding_probability(None, np.array([1, 2])) == 0.5
        assert embedding_probability(np.array([1, 2]), None) == 0.5
        assert embedding_probability(None, None) == 0.5

    def test_embedding_probability_with_projection(self):
        from src.pipeline.stages.inference import embedding_probability
        model = MagicMock()
        model.predict_proba.return_value = np.array([[0.3, 0.7]])

        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.0, 1.0])
        prob = embedding_probability(v1, v2, projection_model=model)
        assert prob == pytest.approx(0.7)

    def test_embedding_probability_with_projection_clipping(self):
        from src.pipeline.stages.inference import embedding_probability
        model = MagicMock()
        # Returns extreme value that should be clipped
        model.predict_proba.return_value = np.array([[0.001, 0.999]])

        v1 = np.array([1.0])
        v2 = np.array([0.0])
        prob = embedding_probability(v1, v2, projection_model=model)
        assert 0.02 <= prob <= 0.98

    def test_embedding_probability_fallback(self):
        from src.pipeline.stages.inference import embedding_probability
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])
        prob = embedding_probability(v1, v2, projection_model=None)
        assert 0.0 < prob < 1.0

    def test_symmetrize_probability(self):
        from src.pipeline.stages.inference import symmetrize_probability
        assert symmetrize_probability(0.7, 0.3) == pytest.approx(0.7)
        assert symmetrize_probability(0.6, 0.5) == pytest.approx(0.55)
        # Perfect symmetry
        assert symmetrize_probability(0.5, 0.5) == pytest.approx(0.5)

    def test_apply_seed_prior(self):
        from src.pipeline.stages.inference import apply_seed_prior
        # 1-seed vs 16-seed: should increase probability
        prob = apply_seed_prior(0.5, seed1=1, seed2=16, weight=0.1, slope=0.15)
        assert prob > 0.5

        # 16-seed vs 1-seed: should decrease
        prob = apply_seed_prior(0.5, seed1=16, seed2=1, weight=0.1, slope=0.15)
        assert prob < 0.5

    def test_apply_seed_prior_zero_weight(self):
        from src.pipeline.stages.inference import apply_seed_prior
        prob = apply_seed_prior(0.65, seed1=1, seed2=16, weight=0.0)
        assert prob == pytest.approx(0.65)


# ---------------------------------------------------------------------------
# 6. robustness_stage.py
# ---------------------------------------------------------------------------

class TestRobustnessStage:
    """Test RobustnessReport and RobustnessStage."""

    def test_robustness_report_defaults(self):
        from src.pipeline.stages.robustness_stage import RobustnessReport
        r = RobustnessReport()
        assert r.feature_dropout == {}
        assert r.distribution_shift == {}
        assert r.feature_stability == {}
        assert r.deployment_gate_passed is True
        assert r.gate_failures == []

    def test_noop_ctx(self):
        from src.pipeline.stages.robustness_stage import _noop_ctx
        ctx = _noop_ctx()
        with ctx:
            pass  # Should not raise

    def test_robustness_stage_name(self):
        from src.pipeline.stages.robustness_stage import RobustnessStage
        stage = RobustnessStage()
        assert stage.name == "robustness_audit"

    def _mock_robustness_module(self, dropout_result=None, shift_result=None, stability_result=None):
        """Create a mock for src.ml.evaluation.robustness with given results."""
        mock_module = MagicMock()
        if dropout_result is not None:
            mock_module.FeatureDropoutTest.return_value.run.return_value = dropout_result
        if shift_result is not None:
            mock_module.DistributionShiftDetector.return_value.detect.return_value = shift_result
        if stability_result is not None:
            mock_module.FeatureStabilityReport.from_importance_rankings.return_value = stability_result
        return mock_module

    def test_robustness_stage_run_all_pass(self):
        """RobustnessStage.run with all checks passing."""
        from src.pipeline.stages.robustness_stage import RobustnessStage
        from src.pipeline.stages.context import PipelineContext

        stage = RobustnessStage()
        ctx = PipelineContext(config=MagicMock())
        ctx.resource_tracker = None
        ctx.phase_timer = None

        X = np.random.randn(50, 5)
        y = np.random.randint(0, 2, 50).astype(float)
        predict_fn = lambda x: np.full(x.shape[0], 0.5)
        feature_names = [f"f{i}" for i in range(5)]

        mock_dropout_result = MagicMock()
        mock_dropout_result.fragile_features = []
        mock_dropout_result.max_degradation = 0.001
        mock_dropout_result.to_dict.return_value = {"fragile_features": []}

        mock_shift_result = MagicMock()
        mock_shift_result.significant_features = []
        mock_shift_result.to_dict.return_value = {"significant": []}

        mock_stability = MagicMock()
        mock_stability.to_dict.return_value = {"tau": 0.9}

        mock_mod = self._mock_robustness_module(mock_dropout_result, mock_shift_result, mock_stability)
        with patch.dict("sys.modules", {"src.ml.evaluation.robustness": mock_mod}):
            report = stage.run(
                ctx, X, y, predict_fn, feature_names,
                X_predict=X,
                fold_importances={0: {"f0": 1.0}, 1: {"f0": 0.9}},
            )

        assert report.deployment_gate_passed is True
        assert len(report.gate_failures) == 0

    def test_robustness_stage_gate_fails_fragile(self):
        """Gate fails with too many fragile features."""
        from src.pipeline.stages.robustness_stage import RobustnessStage
        from src.pipeline.stages.context import PipelineContext

        stage = RobustnessStage()
        ctx = PipelineContext(config=MagicMock())

        X = np.random.randn(20, 3)
        y = np.random.randint(0, 2, 20).astype(float)

        mock_dropout_result = MagicMock()
        mock_dropout_result.fragile_features = ["f0", "f1", "f2", "f3", "f4"]  # > MAX_FRAGILE_FEATURES
        mock_dropout_result.max_degradation = 0.001
        mock_dropout_result.to_dict.return_value = {}

        mock_mod = self._mock_robustness_module(mock_dropout_result)
        with patch.dict("sys.modules", {"src.ml.evaluation.robustness": mock_mod}):
            report = stage.run(ctx, X, y, lambda x: np.full(x.shape[0], 0.5), ["f0", "f1", "f2"])

        assert report.deployment_gate_passed is False
        assert any("fragile" in f for f in report.gate_failures)

    def test_robustness_stage_dropout_exception(self):
        """Feature dropout exception is caught gracefully."""
        from src.pipeline.stages.robustness_stage import RobustnessStage
        from src.pipeline.stages.context import PipelineContext

        stage = RobustnessStage()
        ctx = PipelineContext(config=MagicMock())

        X = np.random.randn(10, 2)
        y = np.ones(10)

        mock_mod = MagicMock()
        mock_mod.FeatureDropoutTest.return_value.run.side_effect = RuntimeError("dropout failed")
        with patch.dict("sys.modules", {"src.ml.evaluation.robustness": mock_mod}):
            report = stage.run(ctx, X, y, lambda x: x[:, 0], ["f0", "f1"])

        assert "error" in report.feature_dropout

    def test_robustness_stage_with_timer(self):
        """RobustnessStage uses resource_tracker.phase when available."""
        from src.pipeline.stages.robustness_stage import RobustnessStage
        from src.pipeline.stages.context import PipelineContext

        stage = RobustnessStage()
        mock_tracker = MagicMock()
        mock_tracker.phase.return_value = MagicMock(__enter__=MagicMock(), __exit__=MagicMock(return_value=False))
        ctx = PipelineContext(config=MagicMock(), resource_tracker=mock_tracker)

        X = np.random.randn(10, 2)
        y = np.ones(10)

        mock_dropout_result = MagicMock()
        mock_dropout_result.fragile_features = []
        mock_dropout_result.max_degradation = 0.0
        mock_dropout_result.to_dict.return_value = {}

        mock_mod = self._mock_robustness_module(mock_dropout_result)
        with patch.dict("sys.modules", {"src.ml.evaluation.robustness": mock_mod}):
            report = stage.run(ctx, X, y, lambda x: np.full(x.shape[0], 0.5), ["f0", "f1"])

        mock_tracker.phase.assert_called_once_with("robustness_audit")


# ---------------------------------------------------------------------------
# 7. probability_pipeline.py
# ---------------------------------------------------------------------------

class TestProbabilityPipeline:
    """Test probability pipeline functions."""

    def test_compute_raw_probability_finite(self):
        from src.pipeline.probability_pipeline import compute_raw_probability
        assert compute_raw_probability(0.7) == 0.7
        assert compute_raw_probability(0.0) == 0.0

    def test_compute_raw_probability_nan(self):
        from src.pipeline.probability_pipeline import compute_raw_probability
        assert compute_raw_probability(float("nan")) == 0.5

    def test_compute_raw_probability_inf(self):
        from src.pipeline.probability_pipeline import compute_raw_probability
        assert compute_raw_probability(float("inf")) == 0.5
        assert compute_raw_probability(float("-inf")) == 0.5

    def test_apply_calibration_none(self):
        from src.pipeline.probability_pipeline import apply_calibration
        assert apply_calibration(0.7, None) == 0.7

    def test_apply_calibration_with_calibrate_method(self):
        from src.pipeline.probability_pipeline import apply_calibration
        cal = MagicMock()
        cal.calibrate.return_value = np.array([0.65])
        assert apply_calibration(0.7, cal) == pytest.approx(0.65)

    def test_apply_calibration_with_brier_calibrator(self):
        from src.pipeline.probability_pipeline import apply_calibration
        cal = MagicMock(spec=[])
        cal.fitted = True
        cal.temperature = 1.5
        # Remove calibrate method to trigger second branch
        del cal.calibrate
        result = apply_calibration(0.7, cal)
        assert 0.005 <= result <= 0.995

    def test_apply_calibration_unfitted(self):
        from src.pipeline.probability_pipeline import apply_calibration
        cal = MagicMock(spec=[])
        del cal.calibrate
        cal.fitted = False
        assert apply_calibration(0.7, cal) == 0.7

    def test_apply_tournament_shrinkage(self):
        from src.pipeline.probability_pipeline import apply_tournament_shrinkage
        # shrinkage=0 -> no change
        assert apply_tournament_shrinkage(0.8, 0.0) == pytest.approx(0.8)
        # shrinkage=1 -> fully shrunk to 0.5
        assert apply_tournament_shrinkage(0.8, 1.0) == pytest.approx(0.5)
        # shrinkage=0.1
        assert apply_tournament_shrinkage(0.8, 0.1) == pytest.approx(0.1 * 0.5 + 0.9 * 0.8)

    def test_apply_final_clip(self):
        from src.pipeline.probability_pipeline import apply_final_clip
        assert apply_final_clip(0.5, 0.005, 0.995) == 0.5
        assert apply_final_clip(0.001, 0.005, 0.995) == 0.005
        assert apply_final_clip(0.999, 0.005, 0.995) == 0.995

    def test_validate_stage_output_in_bounds(self):
        from src.pipeline.probability_pipeline import validate_stage_output
        assert validate_stage_output(0.5, 0.0, 1.0, "test") == 0.5

    def test_validate_stage_output_out_of_bounds(self):
        from src.pipeline.probability_pipeline import validate_stage_output
        assert validate_stage_output(1.5, 0.0, 1.0, "test") == 1.0
        assert validate_stage_output(-0.1, 0.0, 1.0, "test") == 0.0

    def test_apply_experimental_seed_override_none(self):
        from src.pipeline.probability_pipeline import apply_experimental_seed_override
        assert apply_experimental_seed_override(0.7, 1, 16, None) == 0.7
        assert apply_experimental_seed_override(0.7, 0, 16, MagicMock()) == 0.7
        assert apply_experimental_seed_override(0.7, 1, 0, MagicMock()) == 0.7

    def test_apply_experimental_seed_override_applied(self):
        from src.pipeline.probability_pipeline import apply_experimental_seed_override
        overrides = MagicMock()
        overrides.apply.return_value = 0.85
        assert apply_experimental_seed_override(0.7, 1, 16, overrides) == pytest.approx(0.85)

    def test_apply_experimental_sharpening_none(self):
        from src.pipeline.probability_pipeline import apply_experimental_sharpening
        assert apply_experimental_sharpening(0.7, None) == 0.7

    def test_apply_experimental_sharpening_unfitted(self):
        from src.pipeline.probability_pipeline import apply_experimental_sharpening
        sharpener = MagicMock()
        sharpener.fitted = False
        assert apply_experimental_sharpening(0.7, sharpener) == 0.7

    def test_apply_experimental_sharpening_fitted(self):
        from src.pipeline.probability_pipeline import apply_experimental_sharpening
        sharpener = MagicMock()
        sharpener.fitted = True
        sharpener.sharpen.return_value = 0.9
        assert apply_experimental_sharpening(0.7, sharpener) == pytest.approx(0.9)

    def test_apply_experimental_goto_none(self):
        from src.pipeline.probability_pipeline import apply_experimental_goto
        assert apply_experimental_goto(0.7, None) == 0.7

    def test_apply_experimental_goto_unfitted(self):
        from src.pipeline.probability_pipeline import apply_experimental_goto
        conv = MagicMock()
        conv.fitted = False
        assert apply_experimental_goto(0.7, conv) == 0.7

    def test_apply_experimental_goto_fitted(self):
        from src.pipeline.probability_pipeline import apply_experimental_goto
        conv = MagicMock()
        conv.fitted = True
        conv.correct_single.return_value = 0.72
        assert apply_experimental_goto(0.7, conv) == pytest.approx(0.72)

    def test_apply_experimental_seed_prior_zero_weight(self):
        from src.pipeline.probability_pipeline import apply_experimental_seed_prior
        assert apply_experimental_seed_prior(0.7, 1, 16, weight=0.0, slope=0.15) == 0.7

    def test_apply_experimental_seed_prior_bad_seeds(self):
        from src.pipeline.probability_pipeline import apply_experimental_seed_prior
        assert apply_experimental_seed_prior(0.7, 0, 16, weight=0.1, slope=0.15) == 0.7
        assert apply_experimental_seed_prior(0.7, 1, 0, weight=0.1, slope=0.15) == 0.7

    def test_apply_experimental_seed_prior_applied(self):
        from src.pipeline.probability_pipeline import apply_experimental_seed_prior
        result = apply_experimental_seed_prior(0.5, 1, 16, weight=0.1, slope=0.15)
        assert result > 0.5  # 1-seed should be favoured over 16-seed

    def test_run_ablation_ladder(self):
        from src.pipeline.probability_pipeline import run_ablation_ladder
        raw = np.array([0.7, 0.3, 0.5, 0.9])
        outcomes = np.array([1.0, 0.0, 1.0, 1.0])
        results = run_ablation_ladder(raw, outcomes, calibrator=None, shrinkage=0.1)
        assert "raw" in results
        assert "calibrated" in results
        assert "shrunk" in results
        assert "final" in results
        for stage_name, metrics in results.items():
            assert "brier" in metrics
            assert "ece" in metrics
            assert "log_loss" in metrics
            assert "mean_movement" in metrics
            assert "max_movement" in metrics

    def test_run_ablation_ladder_with_calibrator(self):
        from src.pipeline.probability_pipeline import run_ablation_ladder
        cal = MagicMock()
        cal.calibrate.side_effect = lambda x: x * 0.95
        raw = np.array([0.6, 0.4])
        outcomes = np.array([1.0, 0.0])
        results = run_ablation_ladder(raw, outcomes, calibrator=cal, shrinkage=0.05)
        assert results["calibrated"]["mean_movement"] >= 0

    def test_validate_probability_stage_clean(self):
        from src.pipeline.probability_pipeline import validate_probability_stage
        before = np.array([0.5, 0.6, 0.7])
        after = np.array([0.51, 0.61, 0.71])
        result = validate_probability_stage(before, after, "test_stage")
        assert result["stage"] == "test_stage"
        assert len(result["warnings"]) == 0

    def test_validate_probability_stage_out_of_bounds(self):
        from src.pipeline.probability_pipeline import validate_probability_stage
        before = np.array([0.5])
        after = np.array([-0.1])
        result = validate_probability_stage(before, after, "bad_stage")
        assert any("outside [0, 1]" in w for w in result["warnings"])

    def test_validate_probability_stage_large_movement(self):
        from src.pipeline.probability_pipeline import validate_probability_stage
        before = np.array([0.3, 0.4, 0.5])
        after = np.array([0.8, 0.9, 0.95])  # Big jumps
        result = validate_probability_stage(
            before, after, "jump_stage",
            max_movement_pct=0.05,
            max_large_move_frac=0.1,
            large_move_threshold=0.05,
        )
        assert len(result["warnings"]) > 0

    def test_expected_calibration_error(self):
        from src.pipeline.probability_pipeline import _expected_calibration_error
        # Perfect calibration
        probs = np.array([0.1, 0.9])
        outcomes = np.array([0.0, 1.0])
        ece = _expected_calibration_error(probs, outcomes, n_bins=10)
        assert ece < 0.15  # Should be close to 0

        # Worst calibration
        probs = np.array([0.9, 0.9, 0.9])
        outcomes = np.array([0.0, 0.0, 0.0])
        ece = _expected_calibration_error(probs, outcomes, n_bins=10)
        assert ece > 0.5


# ---------------------------------------------------------------------------
# 8. production_baseline.py
# ---------------------------------------------------------------------------

class TestProductionBaseline:
    """Test ProductionBaselineSpec and AdmissionGateThresholds."""

    def test_admission_gate_defaults(self):
        from src.pipeline.production_baseline import AdmissionGateThresholds
        gate = AdmissionGateThresholds()
        assert gate.min_mean_brier_improvement == 0.0
        assert gate.min_fold_improvement_rate == 0.60
        assert gate.max_calibration_degradation == 0.01

    def test_production_baseline_spec_defaults(self):
        from src.pipeline.production_baseline import ProductionBaselineSpec, PRODUCTION_BASELINE
        spec = ProductionBaselineSpec()
        assert spec.name == "spread_logistic_temp_v1"
        assert "spread_regressor" in spec.models
        assert spec.calibration == "temperature"
        assert spec.ensemble_policy == "four_model_fixed_weight_blend"

    def test_is_model_sanctioned(self):
        from src.pipeline.production_baseline import PRODUCTION_BASELINE
        assert PRODUCTION_BASELINE.is_model_sanctioned("spread_regressor") is True
        assert PRODUCTION_BASELINE.is_model_sanctioned("logistic_regression") is True
        assert PRODUCTION_BASELINE.is_model_sanctioned("random_forest") is False

    def test_is_calibrator_sanctioned(self):
        from src.pipeline.production_baseline import PRODUCTION_BASELINE
        assert PRODUCTION_BASELINE.is_calibrator_sanctioned("temperature") is True
        assert PRODUCTION_BASELINE.is_calibrator_sanctioned("round_specific_calibrator") is False

    def test_validate_default_spec(self):
        from src.pipeline.production_baseline import PRODUCTION_BASELINE
        violations = PRODUCTION_BASELINE.validate()
        assert violations == []

    def test_validate_no_models(self):
        from src.pipeline.production_baseline import ProductionBaselineSpec
        spec = ProductionBaselineSpec(models=())
        violations = spec.validate()
        assert any("No models" in v for v in violations)

    def test_validate_no_calibration(self):
        from src.pipeline.production_baseline import ProductionBaselineSpec
        spec = ProductionBaselineSpec(calibration="")
        violations = spec.validate()
        assert any("No calibration" in v for v in violations)

    def test_validate_bad_weights(self):
        from src.pipeline.production_baseline import ProductionBaselineSpec
        spec = ProductionBaselineSpec(default_weights={"spread": 0.5, "logistic": 0.5, "lgb": 0.5, "xgb": 0.5})
        violations = spec.validate()
        assert any("weights sum" in v for v in violations)

    def test_validate_unknown_weight_key(self):
        from src.pipeline.production_baseline import ProductionBaselineSpec
        spec = ProductionBaselineSpec(default_weights={"spread": 0.5, "logistic": 0.2, "lgb": 0.2, "unknown_model": 0.1})
        violations = spec.validate()
        assert any("unknown_model" in v for v in violations)

    def test_frozen_dataclass(self):
        from src.pipeline.production_baseline import PRODUCTION_BASELINE
        with pytest.raises(AttributeError):
            PRODUCTION_BASELINE.name = "new_name"

    def test_deprecated_models_empty(self):
        from src.pipeline.production_baseline import PRODUCTION_BASELINE
        assert PRODUCTION_BASELINE.deprecated_production_models == ()

    def test_deprecated_calibrators(self):
        from src.pipeline.production_baseline import PRODUCTION_BASELINE
        assert "round_specific_calibrator" in PRODUCTION_BASELINE.deprecated_production_calibrators
