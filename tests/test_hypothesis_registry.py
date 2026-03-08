"""Tests for hypothesis registry and experiment scheduler."""

import json
import tempfile

import pytest

from src.ml.research.hypothesis_registry import (
    Hypothesis,
    HypothesisRegistry,
    HypothesisStatus,
)


@pytest.fixture
def tmp_registry(tmp_path):
    """Registry backed by a temp file."""
    path = str(tmp_path / "hypotheses.jsonl")
    return HypothesisRegistry(registry_path=path)


class TestHypothesisRegistry:
    def test_add_and_get(self, tmp_registry):
        h = tmp_registry.add(
            title="Test hypothesis",
            rationale="Because reasons",
            expected_impact=0.005,
            priority="high",
        )
        assert h.hypothesis_id
        assert h.status == "proposed"

        retrieved = tmp_registry.get(h.hypothesis_id)
        assert retrieved is not None
        assert retrieved.title == "Test hypothesis"

    def test_schedule(self, tmp_registry):
        h = tmp_registry.add(title="Schedulable hypothesis")
        schedule = tmp_registry.schedule(
            h.hypothesis_id,
            experiment_config={"model": "lgb", "features": ["elo"]},
        )
        assert schedule.schedule_id

        updated = tmp_registry.get(h.hypothesis_id)
        assert updated.status == "scheduled"

    def test_resolve(self, tmp_registry):
        h = tmp_registry.add(title="Resolvable hypothesis")
        tmp_registry.resolve(
            h.hypothesis_id,
            outcome="confirmed",
            actual_impact=-0.003,
            notes="Feature improved Brier by 0.003",
        )
        resolved = tmp_registry.get(h.hypothesis_id)
        assert resolved.status == "confirmed"
        assert resolved.actual_impact == -0.003

    def test_list_filter(self, tmp_registry):
        tmp_registry.add(title="H1", priority="high")
        tmp_registry.add(title="H2", priority="low")
        tmp_registry.add(title="H3", priority="high")

        high = tmp_registry.list(priority="high")
        assert len(high) == 2

    def test_generate_from_diagnostics(self, tmp_registry):
        generated = tmp_registry.generate_from_diagnostics(
            dropout_report={
                "fragile_features": ["feat_1", "feat_5"],
            },
            shift_report={
                "significant_features": ["adj_eff"],
            },
            loyo_year_briers={2022: 0.20, 2023: 0.18, 2024: 0.30},
        )
        assert len(generated) == 3
        tags = [t for h in generated for t in h.tags]
        assert "auto_generated" in tags

    def test_summary(self, tmp_registry):
        tmp_registry.add(title="Test")
        s = tmp_registry.summary()
        assert "1 hypotheses" in s

    def test_persistence(self, tmp_path):
        path = str(tmp_path / "persist.jsonl")
        reg1 = HypothesisRegistry(registry_path=path)
        reg1.add(title="Persisted hypothesis")

        reg2 = HypothesisRegistry(registry_path=path)
        all_h = reg2.list()
        assert len(all_h) == 1
        assert all_h[0].title == "Persisted hypothesis"

    def test_next_experiments(self, tmp_registry):
        h = tmp_registry.add(title="Scheduled", priority="high")
        tmp_registry.schedule(h.hypothesis_id, {"test": True})
        experiments = tmp_registry.next_experiments()
        assert len(experiments) == 1


class TestHypothesis:
    def test_roundtrip(self):
        h = Hypothesis(
            hypothesis_id="abc123",
            title="Test",
            priority="high",
            tags=["test"],
        )
        d = h.to_dict()
        restored = Hypothesis.from_dict(d)
        assert restored.hypothesis_id == "abc123"
        assert restored.tags == ["test"]
