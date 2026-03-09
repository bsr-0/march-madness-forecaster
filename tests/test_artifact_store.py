"""Tests for src.reproducibility.artifact_store."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import pytest

from src.reproducibility.artifact_store import ArtifactStore


# ---------------------------------------------------------------------------
# Minimal config stub for save_config tests.
# ---------------------------------------------------------------------------

@dataclass
class _StubConfig:
    year: int = 2026
    random_seed: int = 42


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSaveLoadConfigRoundtrip:
    """save_config -> load_artifact should return the same dict."""

    def test_roundtrip(self, tmp_path: Path) -> None:
        store = ArtifactStore(store_dir=str(tmp_path / "artifacts"))
        cfg = _StubConfig(year=2025, random_seed=99)

        artifact_id = store.save_config(cfg, experiment_id="exp-001")
        loaded = store.load_artifact(artifact_id)

        assert loaded["year"] == 2025
        assert loaded["random_seed"] == 99


class TestSaveLoadPredictionsRoundtrip:
    """save_predictions -> load_artifact should return the same dict."""

    def test_roundtrip(self, tmp_path: Path) -> None:
        store = ArtifactStore(store_dir=str(tmp_path / "artifacts"))
        preds = {"teamA_vs_teamB": 0.75, "teamC_vs_teamD": 0.42}

        artifact_id = store.save_predictions(preds, experiment_id="exp-002")
        loaded = store.load_artifact(artifact_id)

        assert loaded == preds


class TestSaveLoadModelRoundtrip:
    """save_model -> load_artifact should return the same object."""

    def test_roundtrip(self, tmp_path: Path) -> None:
        store = ArtifactStore(store_dir=str(tmp_path / "artifacts"))
        model = {"weights": [0.1, 0.9], "bias": 0.05}

        artifact_id = store.save_model(model, experiment_id="exp-003")
        loaded = store.load_artifact(artifact_id)

        assert loaded == model

    def test_with_metadata(self, tmp_path: Path) -> None:
        store = ArtifactStore(store_dir=str(tmp_path / "artifacts"))
        model = {"w": 1}
        meta = {"framework": "lightgbm", "n_estimators": 100}

        artifact_id = store.save_model(model, "exp-004", metadata=meta)
        manifests = store.list_artifacts(artifact_type="model")
        assert len(manifests) == 1
        assert manifests[0].metadata["framework"] == "lightgbm"


class TestContentAddressableDedup:
    """Saving identical content twice should return the same artifact_id."""

    def test_dedup_predictions(self, tmp_path: Path) -> None:
        store = ArtifactStore(store_dir=str(tmp_path / "artifacts"))
        preds = {"game1": 0.6}

        id1 = store.save_predictions(preds, experiment_id="exp-a")
        id2 = store.save_predictions(preds, experiment_id="exp-b")
        assert id1 == id2

        # Manifest should have only one entry (the first write).
        manifests = store.list_artifacts(artifact_type="predictions")
        assert len(manifests) == 1

    def test_dedup_model(self, tmp_path: Path) -> None:
        store = ArtifactStore(store_dir=str(tmp_path / "artifacts"))
        model = {"w": [1, 2, 3]}

        id1 = store.save_model(model, "exp-a")
        id2 = store.save_model(model, "exp-b")
        assert id1 == id2

    def test_different_content_different_id(self, tmp_path: Path) -> None:
        store = ArtifactStore(store_dir=str(tmp_path / "artifacts"))
        id1 = store.save_predictions({"g": 0.5}, "exp-a")
        id2 = store.save_predictions({"g": 0.9}, "exp-b")
        assert id1 != id2


class TestListArtifactsFiltering:
    """list_artifacts should filter by experiment_id and artifact_type."""

    def test_filter_by_experiment_id(self, tmp_path: Path) -> None:
        store = ArtifactStore(store_dir=str(tmp_path / "artifacts"))
        store.save_predictions({"a": 0.1}, "exp-1")
        store.save_predictions({"b": 0.2}, "exp-2")

        result = store.list_artifacts(experiment_id="exp-1")
        assert len(result) == 1
        assert result[0].experiment_id == "exp-1"

    def test_filter_by_type(self, tmp_path: Path) -> None:
        store = ArtifactStore(store_dir=str(tmp_path / "artifacts"))
        store.save_predictions({"a": 0.1}, "exp-1")
        store.save_model({"w": 1}, "exp-1")
        store.save_config(_StubConfig(), "exp-1")

        models = store.list_artifacts(artifact_type="model")
        assert len(models) == 1
        assert models[0].artifact_type == "model"

        configs = store.list_artifacts(artifact_type="config")
        assert len(configs) == 1

    def test_filter_combined(self, tmp_path: Path) -> None:
        store = ArtifactStore(store_dir=str(tmp_path / "artifacts"))
        store.save_predictions({"a": 0.1}, "exp-1")
        store.save_model({"w": 1}, "exp-1")
        store.save_predictions({"b": 0.2}, "exp-2")

        result = store.list_artifacts(experiment_id="exp-1", artifact_type="predictions")
        assert len(result) == 1
        assert result[0].experiment_id == "exp-1"
        assert result[0].artifact_type == "predictions"

    def test_no_filter_returns_all(self, tmp_path: Path) -> None:
        store = ArtifactStore(store_dir=str(tmp_path / "artifacts"))
        store.save_predictions({"a": 0.1}, "exp-1")
        store.save_model({"w": 1}, "exp-2")

        result = store.list_artifacts()
        assert len(result) == 2


class TestLoadArtifactNotFound:
    """load_artifact should raise FileNotFoundError for unknown IDs."""

    def test_raises(self, tmp_path: Path) -> None:
        store = ArtifactStore(store_dir=str(tmp_path / "artifacts"))
        with pytest.raises(FileNotFoundError):
            store.load_artifact("nonexistent")
