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


def _build_summary(module, name, years_to_brier):
    spec = module.ModelSpec(name=name, mode="ensemble", params={})
    rows = [
        module.YearMetrics(year=year, brier=brier, seed_brier=0.20, bss=1.0 - brier / 0.20, ece=0.02, metadata={})
        for year, brier in years_to_brier
    ]
    return module.EvaluationSummary(spec=spec, per_year=rows)


def test_build_candidate_experiments_expands_search_space():
    module = _load_admission_module()
    spec_data = {
        "candidate_search_spaces": [
            {
                "modes": ["ensemble", "torvik_corrected", "torvik_closing", "closing_market_blend"],
                "params": {
                    "recent_year_start": [2021],
                    "recent_year_weight": [2.0, 3.0],
                },
            }
        ]
    }

    experiments = module.build_candidate_experiments(spec_data)

    names = sorted(spec.name for spec in experiments)
    assert len(experiments) == 8
    assert any(name.startswith("ensemble__") for name in names)
    assert any(name.startswith("torvik_corrected__") for name in names)
    assert any(name.startswith("torvik_closing__") for name in names)
    assert any(name.startswith("closing_market_blend__") for name in names)


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


def test_summarize_recent_weighted_uses_recent_block_policy():
    module = _load_admission_module()
    summary = _build_summary(
        module,
        "weighted",
        [
            (2019, 0.10),
            (2021, 0.20),
            (2022, 0.30),
            (2023, 0.40),
            (2024, 0.50),
            (2025, 0.60),
            (2026, 0.70),
        ],
    )

    recent_weighted = module.summarize_recent_weighted(summary)

    # Recent-5 policy on 7 years => 2 older years at weight 1.0, 5 recent years at 0.4 each.
    assert recent_weighted["weighting"]["recent_year_count"] == 5
    assert recent_weighted["weighting"]["recent_total_ratio"] == pytest.approx(1.0)
    assert recent_weighted["weighting"]["recent_year_weight"] == pytest.approx(0.4)
    assert recent_weighted["weighting"]["per_year_weights"]["2019"] == pytest.approx(1.0)
    assert recent_weighted["weighting"]["per_year_weights"]["2023"] == pytest.approx(0.4)
    assert recent_weighted["mean_brier"] == pytest.approx(0.325)


def test_summarize_priority_window_uses_requested_years_when_all_present():
    module = _load_admission_module()
    summary = _build_summary(
        module,
        "priority",
        [(2022, 0.19), (2023, 0.18), (2024, 0.17), (2025, 0.16), (2026, 0.15)],
    )

    priority_window = module.summarize_year_slice(summary, module.PRIORITY_WINDOW_YEARS)

    assert priority_window["requested_years"] == [2023, 2024, 2025]
    assert priority_window["used_years"] == [2023, 2024, 2025]
    assert priority_window["mean_brier"] == pytest.approx((0.18 + 0.17 + 0.16) / 3.0)


def test_summarize_priority_window_omits_missing_years():
    module = _load_admission_module()
    summary = _build_summary(
        module,
        "priority-missing",
        [(2022, 0.19), (2023, 0.18), (2025, 0.16), (2026, 0.15)],
    )

    priority_window = module.summarize_year_slice(summary, module.PRIORITY_WINDOW_YEARS)

    assert priority_window["requested_years"] == [2023, 2024, 2025]
    assert priority_window["used_years"] == [2023, 2025]
    assert priority_window["mean_brier"] == pytest.approx((0.18 + 0.16) / 2.0)


def test_build_report_payload_includes_new_summary_blocks_without_changing_gate_result():
    module = _load_admission_module()

    incumbent_shadow = _build_summary(module, "incumbent", [(2023, 0.18), (2024, 0.18)])
    incumbent_final = _build_summary(module, "incumbent", [(2025, 0.18), (2026, 0.18)])
    incumbent_holdout = module._merge_summaries(incumbent_shadow.spec, incumbent_shadow, incumbent_final)

    candidate_shadow = _build_summary(module, "candidate", [(2023, 0.17), (2024, 0.17)])
    candidate_final = _build_summary(module, "candidate", [(2025, 0.175), (2026, 0.175)])
    candidate_holdout = module._merge_summaries(candidate_shadow.spec, candidate_shadow, candidate_final)

    admission = module.evaluate_admission_gate(
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

    report_payload = module.build_report_payload(
        dev_years=[2019, 2021, 2022],
        shadow_years=[2023, 2024],
        final_years=[2025, 2026],
        shadow_incumbent=incumbent_shadow,
        final_incumbent=incumbent_final,
        holdout_incumbent=incumbent_holdout,
        candidate_shadow_evals=[candidate_shadow],
        selected_shadow=candidate_shadow,
        final_candidate=candidate_final,
        holdout_candidate=candidate_holdout,
        admission=admission,
    )

    assert admission["passed"]
    assert report_payload["admission_gate"]["passed"]
    assert report_payload["reporting_policy"]["priority_window_years"] == [2023, 2024, 2025]
    for side in ("incumbent", "candidate_selected"):
        assert "recent_weighted" in report_payload[side]
        assert "priority_window" in report_payload[side]
        assert "final_holdout" in report_payload[side]
        assert report_payload[side]["final"]["mean_brier"] == pytest.approx(report_payload[side]["final_holdout"]["mean_brier"])


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
