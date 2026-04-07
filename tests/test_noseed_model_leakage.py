"""Regression tests for noseed_model leakage guard.

These tests enforce that _load_team_stats() rejects any Torvik data file that
lacks confirmed pre-tournament provenance. If this guard ever regresses, the
noseed model silently ingests post-tournament data and all backtest results
become invalid.
"""

import json
import pytest

from src.exceptions import LeakageError
from src.prediction.noseed_model import _load_team_stats, _validate_pretournament


class TestValidatePretournament:
    def test_rejects_pre_tournament_computed(self, tmp_path):
        data = {"data_type": "pre_tournament_computed", "teams": []}
        f = tmp_path / "torvik_2024.json"
        with pytest.raises(LeakageError, match="pre_tournament_computed"):
            _validate_pretournament(data, f)

    def test_accepts_pre_tournament(self, tmp_path):
        data = {"data_type": "pre_tournament", "teams": []}
        f = tmp_path / "torvik_2024.json"
        _validate_pretournament(data, f)  # must not raise

    def test_rejects_post_tournament(self, tmp_path):
        data = {"data_type": "post_tournament", "teams": []}
        f = tmp_path / "torvik_2024.json"
        with pytest.raises(LeakageError, match="post_tournament"):
            _validate_pretournament(data, f)

    def test_rejects_missing_data_type(self, tmp_path):
        data = {"teams": []}  # no data_type key
        f = tmp_path / "torvik_2024.json"
        with pytest.raises(LeakageError, match="None"):
            _validate_pretournament(data, f)

    def test_rejects_unknown_data_type(self, tmp_path):
        data = {"data_type": "full_season", "teams": []}
        f = tmp_path / "torvik_2024.json"
        with pytest.raises(LeakageError, match="full_season"):
            _validate_pretournament(data, f)


class TestLoadTeamStatsLeakageGuard:
    """Integration tests: _load_team_stats() must reject contaminated files."""

    def _write_torvik_file(self, path, data_type, teams=None):
        payload = {"data_type": data_type, "teams": teams or []}
        path.write_text(json.dumps(payload))

    def _write_four_factors_file(self, path, data_type, team_data=None):
        payload = {"data_type": data_type}
        if team_data:
            payload.update(team_data)
        path.write_text(json.dumps(payload))

    def test_rejects_post_tournament_torvik_file(self, tmp_path, monkeypatch):
        import src.prediction.noseed_model as mod

        monkeypatch.setattr(mod, "HIST_DIR", tmp_path / "hist")
        monkeypatch.setattr(mod, "DATA_DIR", tmp_path)

        self._write_torvik_file(tmp_path / "torvik_2024.json", "post_tournament")

        with pytest.raises(LeakageError, match="post_tournament"):
            _load_team_stats(2024)

    def test_rejects_post_tournament_four_factors_file(self, tmp_path, monkeypatch):
        import src.prediction.noseed_model as mod

        monkeypatch.setattr(mod, "HIST_DIR", tmp_path / "hist")
        monkeypatch.setattr(mod, "DATA_DIR", tmp_path)

        # torvik_{year}.json is missing (no file), but four_factors is contaminated
        self._write_four_factors_file(tmp_path / "torvik_four_factors_2024.json", "post_tournament")

        with pytest.raises(LeakageError, match="post_tournament"):
            _load_team_stats(2024)

    def test_accepts_clean_torvik_file(self, tmp_path, monkeypatch):
        import src.prediction.noseed_model as mod

        monkeypatch.setattr(mod, "HIST_DIR", tmp_path / "hist")
        monkeypatch.setattr(mod, "DATA_DIR", tmp_path)

        self._write_torvik_file(
            tmp_path / "torvik_2024.json",
            "pre_tournament",
            teams=[{"team_id": "duke", "barthag": 0.92}],
        )

        result = _load_team_stats(2024)
        assert "duke" in result

    def test_accepts_clean_four_factors_file(self, tmp_path, monkeypatch):
        import src.prediction.noseed_model as mod

        monkeypatch.setattr(mod, "HIST_DIR", tmp_path / "hist")
        monkeypatch.setattr(mod, "DATA_DIR", tmp_path)

        self._write_four_factors_file(
            tmp_path / "torvik_four_factors_2024.json",
            "pre_tournament",
            team_data={"duke": {"effective_fg_pct": 0.55}},
        )

        result = _load_team_stats(2024)
        assert "duke" in result

    def test_returns_empty_when_no_files_exist(self, tmp_path, monkeypatch):
        import src.prediction.noseed_model as mod

        monkeypatch.setattr(mod, "HIST_DIR", tmp_path / "hist")
        monkeypatch.setattr(mod, "DATA_DIR", tmp_path / "data")

        result = _load_team_stats(2024)
        assert result == {}
