import importlib.util
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_admission_module():
    script_path = REPO_ROOT / "scripts" / "admit_kaggle_candidate.py"
    spec = importlib.util.spec_from_file_location("admit_kaggle_candidate_test", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_candidate_experiments_expands_search_space():
    module = _load_admission_module()
    spec_data = {
        "candidate_search_spaces": [
            {
                "modes": ["ensemble", "torvik_corrected"],
                "params": {
                    "recent_year_start": [2021],
                    "recent_year_weight": [2.0, 3.0],
                },
            }
        ]
    }

    experiments = module.build_candidate_experiments(spec_data)

    names = sorted(spec.name for spec in experiments)
    assert len(experiments) == 4
    assert any(name.startswith("ensemble__") for name in names)
    assert any(name.startswith("torvik_corrected__") for name in names)


def test_evaluate_admission_gate_flags_final_year_regression():
    module = _load_admission_module()

    incumbent_spec = module.ModelSpec(name="incumbent", mode="ensemble", params={})
    candidate_spec = module.ModelSpec(name="candidate", mode="torvik_corrected", params={})

    incumbent_rows = [
        module.YearMetrics(year=2023, brier=0.18, seed_brier=0.20, bss=0.10, ece=0.02, metadata={}),
        module.YearMetrics(year=2024, brier=0.18, seed_brier=0.20, bss=0.10, ece=0.02, metadata={}),
        module.YearMetrics(year=2025, brier=0.18, seed_brier=0.20, bss=0.10, ece=0.02, metadata={}),
        module.YearMetrics(year=2026, brier=0.18, seed_brier=0.20, bss=0.10, ece=0.02, metadata={}),
    ]
    candidate_rows = [
        module.YearMetrics(year=2023, brier=0.17, seed_brier=0.20, bss=0.15, ece=0.02, metadata={"coef": [0, 1, 1, 1]}),
        module.YearMetrics(year=2024, brier=0.17, seed_brier=0.20, bss=0.15, ece=0.02, metadata={"coef": [0, 1, 1, 1]}),
        module.YearMetrics(year=2025, brier=0.17, seed_brier=0.20, bss=0.15, ece=0.02, metadata={"coef": [0, 1, 1, 1]}),
        module.YearMetrics(year=2026, brier=0.19, seed_brier=0.20, bss=0.05, ece=0.02, metadata={"coef": [0, 1, 1, 1]}),
    ]

    incumbent_holdout = module.EvaluationSummary(spec=incumbent_spec, per_year=incumbent_rows)
    candidate_holdout = module.EvaluationSummary(spec=candidate_spec, per_year=candidate_rows)
    incumbent_final = module.EvaluationSummary(spec=incumbent_spec, per_year=incumbent_rows[-2:])
    candidate_final = module.EvaluationSummary(spec=candidate_spec, per_year=candidate_rows[-2:])

    result = module.evaluate_admission_gate(
        incumbent_holdout,
        candidate_holdout,
        incumbent_final,
        candidate_final,
        min_final_mean_improvement=0.002,
        max_final_year_regression=0.003,
        min_holdout_win_rate=0.75,
        max_calibration_degradation=0.01,
        min_coef_sign_stability=0.75,
    )

    assert not result["passed"]
    assert not result["checks"]["per_year_regression"]["passed"]


def test_evaluate_admission_gate_passes_clean_candidate():
    module = _load_admission_module()

    incumbent_spec = module.ModelSpec(name="incumbent", mode="ensemble", params={})
    candidate_spec = module.ModelSpec(name="candidate", mode="torvik_corrected", params={})

    incumbent_rows = [
        module.YearMetrics(year=2023, brier=0.18, seed_brier=0.20, bss=0.10, ece=0.03, metadata={}),
        module.YearMetrics(year=2024, brier=0.18, seed_brier=0.20, bss=0.10, ece=0.03, metadata={}),
        module.YearMetrics(year=2025, brier=0.18, seed_brier=0.20, bss=0.10, ece=0.03, metadata={}),
        module.YearMetrics(year=2026, brier=0.18, seed_brier=0.20, bss=0.10, ece=0.03, metadata={}),
    ]
    candidate_rows = [
        module.YearMetrics(year=2023, brier=0.17, seed_brier=0.20, bss=0.15, ece=0.025, metadata={"coef": [0, 1, 1, -1]}),
        module.YearMetrics(year=2024, brier=0.17, seed_brier=0.20, bss=0.15, ece=0.025, metadata={"coef": [0, 1, 1, -1]}),
        module.YearMetrics(year=2025, brier=0.175, seed_brier=0.20, bss=0.125, ece=0.025, metadata={"coef": [0, 1, 1, -1]}),
        module.YearMetrics(year=2026, brier=0.175, seed_brier=0.20, bss=0.125, ece=0.025, metadata={"coef": [0, 1, 1, -1]}),
    ]

    incumbent_holdout = module.EvaluationSummary(spec=incumbent_spec, per_year=incumbent_rows)
    candidate_holdout = module.EvaluationSummary(spec=candidate_spec, per_year=candidate_rows)
    incumbent_final = module.EvaluationSummary(spec=incumbent_spec, per_year=incumbent_rows[-2:])
    candidate_final = module.EvaluationSummary(spec=candidate_spec, per_year=candidate_rows[-2:])

    result = module.evaluate_admission_gate(
        incumbent_holdout,
        candidate_holdout,
        incumbent_final,
        candidate_final,
        min_final_mean_improvement=0.002,
        max_final_year_regression=0.003,
        min_holdout_win_rate=0.75,
        max_calibration_degradation=0.01,
        min_coef_sign_stability=0.75,
    )

    assert result["passed"]


def test_evaluate_spec_uses_explicit_fit_years(monkeypatch):
    module = _load_admission_module()

    seen_train_years = []

    def fake_evaluate_ensemble(train_year_records, test_rows, params):
        seen_train_years.append(sorted(train_year_records))
        year = int(test_rows[0]["year"])
        return module.YearMetrics(year=year, brier=0.18, seed_brier=0.20, bss=0.10, ece=0.03, metadata={})

    monkeypatch.setattr(module, "_evaluate_ensemble", fake_evaluate_ensemble)

    data = {
        2018: [{"torvik": 0.6, "pipeline": 0.55, "seed": 0.52, "outcome": 1.0}],
        2019: [{"torvik": 0.6, "pipeline": 0.55, "seed": 0.52, "outcome": 1.0}],
        2020: [{"torvik": 0.6, "pipeline": 0.55, "seed": 0.52, "outcome": 1.0}],
        2023: [{"torvik": 0.6, "pipeline": 0.55, "seed": 0.52, "outcome": 1.0}],
        2024: [{"torvik": 0.6, "pipeline": 0.55, "seed": 0.52, "outcome": 1.0}],
    }
    spec = module.ModelSpec(name="ensemble", mode="ensemble", params={})

    summary = module.evaluate_spec(data, spec, eval_years=[2023, 2024], fit_years=[2018, 2019, 2020])

    assert [row.year for row in summary.per_year] == [2023, 2024]
    assert seen_train_years == [[2018, 2019, 2020], [2018, 2019, 2020]]


def test_validate_split_rejects_overlapping_years():
    module = _load_admission_module()

    with pytest.raises(ValueError, match="appears in both"):
        module._validate_split([2019, 2021], [2021, 2022], [2025, 2026])
