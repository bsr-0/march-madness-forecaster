"""Tests for the experiment registry (C4 fix)."""

import json
import os
import tempfile

import pytest

from src.ml.evaluation.experiment_registry import ExperimentRecord, ExperimentRegistry


@pytest.fixture
def tmp_ledger(tmp_path):
    """Provide a temporary ledger path."""
    return str(tmp_path / "test_ledger.jsonl")


class TestExperimentRecord:
    def test_to_dict_roundtrip(self):
        rec = ExperimentRecord(
            experiment_id="abc123",
            config_hash="deadbeef",
            loyo_mean_brier=0.195,
            loyo_year_briers={2023: 0.19, 2024: 0.20},
            model_components=["lgb", "xgb"],
        )
        d = rec.to_dict()
        rec2 = ExperimentRecord.from_dict(d)
        assert rec2.experiment_id == "abc123"
        assert rec2.loyo_mean_brier == 0.195
        assert rec2.loyo_year_briers == {2023: 0.19, 2024: 0.20}

    def test_from_dict_handles_string_keys(self):
        """JSON serialization converts int keys to strings."""
        d = {"loyo_year_briers": {"2023": 0.19, "2024": 0.20}}
        rec = ExperimentRecord.from_dict(d)
        assert 2023 in rec.loyo_year_briers
        assert 2024 in rec.loyo_year_briers

    def test_from_dict_ignores_unknown_fields(self):
        d = {"experiment_id": "x", "unknown_field": "value"}
        rec = ExperimentRecord.from_dict(d)
        assert rec.experiment_id == "x"


class TestExperimentRegistry:
    def test_log_creates_file(self, tmp_ledger):
        registry = ExperimentRegistry(tmp_ledger)
        rec = ExperimentRecord(loyo_mean_brier=0.195, config_hash="abc")
        exp_id = registry.log(rec)
        assert exp_id
        assert os.path.exists(tmp_ledger)

    def test_log_assigns_id_and_timestamp(self, tmp_ledger):
        registry = ExperimentRegistry(tmp_ledger)
        rec = ExperimentRecord(loyo_mean_brier=0.195)
        exp_id = registry.log(rec)
        assert len(exp_id) == 8
        assert rec.timestamp  # Should be filled in

    def test_list_returns_most_recent_first(self, tmp_ledger):
        registry = ExperimentRegistry(tmp_ledger)
        for i in range(5):
            registry.log(ExperimentRecord(
                experiment_id=f"exp{i}",
                loyo_mean_brier=0.2 - i * 0.01,
            ))
        results = registry.list(n=3)
        assert len(results) == 3
        assert results[0].experiment_id == "exp4"  # Most recent first

    def test_best_returns_lowest_brier(self, tmp_ledger):
        registry = ExperimentRegistry(tmp_ledger)
        registry.log(ExperimentRecord(experiment_id="bad", loyo_mean_brier=0.25))
        registry.log(ExperimentRecord(experiment_id="good", loyo_mean_brier=0.18))
        registry.log(ExperimentRecord(experiment_id="mid", loyo_mean_brier=0.20))
        best = registry.best()
        assert best.experiment_id == "good"

    def test_best_empty_ledger(self, tmp_ledger):
        registry = ExperimentRegistry(tmp_ledger)
        assert registry.best() is None

    def test_compare_shows_differences(self, tmp_ledger):
        registry = ExperimentRegistry(tmp_ledger)
        registry.log(ExperimentRecord(
            experiment_id="aaa",
            loyo_mean_brier=0.20,
            config_hash="hash1",
        ))
        registry.log(ExperimentRecord(
            experiment_id="bbb",
            loyo_mean_brier=0.19,
            config_hash="hash2",
        ))
        diff = registry.compare("aaa", "bbb")
        assert diff["brier_delta"] == pytest.approx(-0.01)
        assert "config_hash" in diff["differences"]

    def test_compare_missing_id(self, tmp_ledger):
        registry = ExperimentRegistry(tmp_ledger)
        diff = registry.compare("nonexistent_a", "nonexistent_b")
        assert "error" in diff

    def test_append_only(self, tmp_ledger):
        """Multiple logs must append, not overwrite."""
        registry = ExperimentRegistry(tmp_ledger)
        registry.log(ExperimentRecord(experiment_id="first", loyo_mean_brier=0.20))
        registry.log(ExperimentRecord(experiment_id="second", loyo_mean_brier=0.19))

        with open(tmp_ledger) as f:
            lines = f.readlines()
        assert len(lines) == 2

    def test_duplicate_config_hash_warns(self, tmp_ledger, caplog):
        """Logging same config_hash twice should warn."""
        import logging

        registry = ExperimentRegistry(tmp_ledger)
        registry.log(ExperimentRecord(config_hash="same_hash", loyo_mean_brier=0.20))

        with caplog.at_level(logging.WARNING):
            registry.log(ExperimentRecord(config_hash="same_hash", loyo_mean_brier=0.19))

        assert any("already exists" in msg for msg in caplog.messages)

    def test_summary_output(self, tmp_ledger):
        registry = ExperimentRegistry(tmp_ledger)
        assert "No experiments" in registry.summary()

        registry.log(ExperimentRecord(
            experiment_id="test1",
            loyo_mean_brier=0.195,
            config_hash="abcdef12",
        ))
        summary = registry.summary()
        assert "1 experiments" in summary
        assert "0.195" in summary

    def test_malformed_line_skipped(self, tmp_ledger):
        """Malformed JSON lines should be skipped gracefully."""
        with open(tmp_ledger, "w") as f:
            f.write('{"experiment_id": "good", "loyo_mean_brier": 0.2}\n')
            f.write("not valid json\n")
            f.write('{"experiment_id": "also_good", "loyo_mean_brier": 0.19}\n')

        registry = ExperimentRegistry(tmp_ledger)
        records = registry.list()
        assert len(records) == 2
