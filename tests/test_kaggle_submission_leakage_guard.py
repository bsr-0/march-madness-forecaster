"""The Kaggle submission entry point refuses to build a non-blind submission.

Every other leakage guard in this repo protects a MODEL'S inputs (training
rows strictly before the target year, torvik snapshots marked pre-tournament).
None of them stop a human from running scripts/kaggle_submission.py long
after a season's tournament concluded and calling the output a "prediction" --
scripts/kaggle_submission.py::_assert_not_post_selection_sunday is the gate
that catches exactly that, and _assert_mens_torvik_is_pretournament re-checks
the men's feature source's own provenance at submission time rather than
trusting that whatever built docs/data/team_stats_by_year.json checked it.
"""

import json
import sys
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import scripts.kaggle_submission as ks  # noqa: E402


class TestSelectionSundayGuard:
    def test_before_selection_sunday_is_unblocked(self):
        with patch("scripts.kaggle_submission.date") as mock_date:
            mock_date.today.return_value = date(2026, 1, 1)
            warning = ks._assert_not_post_selection_sunday(2026, allow_override=False)
        assert warning is None

    def test_after_selection_sunday_blocks_without_override(self):
        with patch("scripts.kaggle_submission.date") as mock_date:
            mock_date.today.return_value = date(2026, 9, 19)
            with pytest.raises(ks.LeakageGuardError):
                ks._assert_not_post_selection_sunday(2026, allow_override=False)

    def test_after_selection_sunday_warns_with_override(self):
        with patch("scripts.kaggle_submission.date") as mock_date:
            mock_date.today.return_value = date(2026, 9, 19)
            warning = ks._assert_not_post_selection_sunday(2026, allow_override=True)
        assert warning is not None
        assert "2026" in warning

    def test_on_selection_sunday_itself_blocks(self):
        with patch("scripts.kaggle_submission.date") as mock_date:
            mock_date.today.return_value = date(2026, 3, 15)  # 2026 Selection Sunday
            with pytest.raises(ks.LeakageGuardError):
                ks._assert_not_post_selection_sunday(2026, allow_override=False)

    def test_unknown_season_does_not_block(self):
        with patch("scripts.kaggle_submission.date") as mock_date:
            mock_date.today.return_value = date(2026, 9, 19)
            warning = ks._assert_not_post_selection_sunday(1900, allow_override=False)
        assert warning is None


class TestTorvikProvenanceGuard:
    def test_pre_tournament_snapshot_passes(self, tmp_path):
        hist = tmp_path / "raw" / "historical"
        hist.mkdir(parents=True)
        (hist / "torvik_2099.json").write_text(json.dumps({"data_type": "pre_tournament"}))
        ks._assert_mens_torvik_is_pretournament(2099, tmp_path)  # must not raise

    def test_post_season_snapshot_blocks(self, tmp_path):
        hist = tmp_path / "raw" / "historical"
        hist.mkdir(parents=True)
        (hist / "torvik_2099.json").write_text(json.dumps({"data_type": "final"}))
        with pytest.raises(ks.LeakageGuardError):
            ks._assert_mens_torvik_is_pretournament(2099, tmp_path)

    def test_missing_snapshot_does_not_block(self, tmp_path):
        ks._assert_mens_torvik_is_pretournament(2099, tmp_path)  # nothing to check, must not raise
