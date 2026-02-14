"""Integration tests for the SOTA 2026 pipeline."""

import json

from src.pipeline.sota import SOTAPipeline, SOTAPipelineConfig


def test_sota_pipeline_produces_rubric_artifacts():
    config = SOTAPipelineConfig(num_simulations=120, pool_size=64, calibration_method="isotonic")
    pipeline = SOTAPipeline(config)

    report = pipeline.run()

    assert "rubric_evaluation" in report
    assert "artifacts" in report

    adjacency = report["artifacts"]["adjacency_matrix"]
    assert len(adjacency) == 64
    assert len(adjacency[0]) == 64

    sim = report["artifacts"]["simulation"]
    assert sim["num_simulations"] == 120

    ev_bracket = report["artifacts"]["ev_max_bracket"]
    assert "champion" in ev_bracket

    baseline = report["artifacts"]["baseline_training"]
    assert baseline["model"] in {"lightgbm", "logistic_regression", "none"}


def test_sota_pipeline_output_file(tmp_path):
    output_path = tmp_path / "sota_report.json"

    config = SOTAPipelineConfig(num_simulations=80, pool_size=20)
    pipeline = SOTAPipeline(config)
    report = pipeline.run()

    with open(output_path, "w") as f:
        json.dump(report, f)

    with open(output_path, "r") as f:
        restored = json.load(f)

    assert restored["artifacts"]["simulation"]["num_simulations"] == 80
