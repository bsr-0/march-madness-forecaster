"""Tests for model card, dataset catalog, and feature catalog."""

import json
import pytest

from src.ml.evaluation.model_card import (
    DatasetCatalog,
    FeatureCatalog,
    ModelCard,
    ModelCardGenerator,
)


class TestModelCard:
    def test_default_card(self):
        card = ModelCard()
        assert card.model_name == "NCAA Tournament Prediction Ensemble"
        assert len(card.limitations) > 0
        assert len(card.ethical_considerations) > 0

    def test_to_dict(self):
        card = ModelCard(model_version="2025", feature_count=42)
        d = card.to_dict()
        assert d["model_version"] == "2025"
        assert d["feature_count"] == 42

    def test_save_and_load(self, tmp_path):
        card = ModelCard(model_version="2025")
        path = str(tmp_path / "card.json")
        card.save(path)

        with open(path) as f:
            loaded = json.load(f)
        assert loaded["model_version"] == "2025"
        assert "limitations" in loaded

    def test_summary(self):
        card = ModelCard(
            model_version="2025",
            primary_metric_value=0.185,
            feature_count=42,
        )
        s = card.summary()
        assert "2025" in s
        assert "0.185" in s


class TestModelCardGenerator:
    def test_generate_from_result(self):
        result = {
            "year": "2025",
            "model_components": ["lgb", "xgb", "logistic"],
            "calibration_method": "isotonic",
            "n_features": 42,
            "dataset_version": "2016-2025",
            "loyo_cv": {
                "mean_brier": 0.185,
                "std_brier": 0.012,
            },
        }
        card = ModelCardGenerator.generate(result, code_version="abc123")
        assert card.model_version == "2025"
        assert card.code_version == "abc123"
        assert len(card.components) == 3
        assert card.primary_metric_value == 0.185

    def test_generate_minimal(self):
        card = ModelCardGenerator.generate({})
        assert card.model_name == "NCAA Tournament Prediction Ensemble"


class TestDatasetCatalog:
    def test_build_default(self):
        catalog = DatasetCatalog.build_default()
        assert len(catalog.entries) >= 5
        assert catalog.entries[0].name  # Has a name

    def test_save(self, tmp_path):
        catalog = DatasetCatalog.build_default()
        path = str(tmp_path / "datasets.json")
        catalog.save(path)

        with open(path) as f:
            loaded = json.load(f)
        assert "datasets" in loaded
        assert len(loaded["datasets"]) >= 5

    def test_to_dict(self):
        catalog = DatasetCatalog.build_default()
        d = catalog.to_dict()
        assert "datasets" in d
        assert all("name" in ds for ds in d["datasets"])


class TestFeatureCatalog:
    def test_build_default(self):
        catalog = FeatureCatalog.build_default()
        assert len(catalog.entries) >= 5
        assert catalog.total_features == len(catalog.entries)

    def test_by_category(self):
        catalog = FeatureCatalog.build_default()
        d = catalog.to_dict()
        assert "by_category" in d
        assert sum(d["by_category"].values()) == len(catalog.entries)

    def test_save(self, tmp_path):
        catalog = FeatureCatalog.build_default()
        path = str(tmp_path / "features.json")
        catalog.save(path)

        with open(path) as f:
            loaded = json.load(f)
        assert loaded["total_features"] >= 5
