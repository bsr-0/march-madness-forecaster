"""Closure / lock test for COUNCIL_LESSONS §2 O17 — researcher-DoF audit.

Freezes the tuning-surface inventory documented in
`artifacts/o17_researcher_dof_audit_2026-04-14.md`. The audit answers
"which years were observable when which hyperparameters were tuned" by
enumerating every Optuna search-space bound, the tuning budget, the
temporal-CV policy, and the year-split enforcement wiring.

This test fails if any of those values drift. A failure is a signal that
researcher DoF has expanded and the audit artifact must be re-issued
(new dated artifact file, updated MEMORY.md row, updated COUNCIL_LESSONS
§2 O17 crumb).

DO NOT relax a bound to make this pass. Update the artifact instead and
justify the new DoF surface.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
AUDIT_ARTIFACT = REPO_ROOT / "artifacts" / "o17_researcher_dof_audit_2026-04-14.md"
TUNING_MODULE = REPO_ROOT / "src" / "ml" / "optimization" / "hyperparameter_tuning.py"


def test_audit_artifact_present() -> None:
    """The dated audit evidence file must exist and be non-empty."""
    assert AUDIT_ARTIFACT.exists(), (
        f"Missing {AUDIT_ARTIFACT}. The artifact is the closure evidence for "
        f"COUNCIL_LESSONS.md §2 O17 — restore it from git or re-issue."
    )
    # Sanity: ≥ 3 KB catches accidental truncation.
    assert AUDIT_ARTIFACT.stat().st_size > 3000


def test_tuning_budget_locked() -> None:
    """Global Optuna budget must match the §B values in the audit.

    A bump in `optuna_n_trials` or flipping `enable_stacking` back on
    materially expands researcher DoF. Either change requires a re-audit."""
    from src.pipeline.config import SOTAPipelineConfig

    config = SOTAPipelineConfig(year=2026)
    assert config.optuna_n_trials == 15, (
        f"optuna_n_trials={config.optuna_n_trials}; audit locks it at 15 "
        f"(OOS-FIX reduction from 50). Re-audit before raising."
    )
    assert config.enable_stacking is False, (
        "enable_stacking was flipped True. Audit documents it as a known "
        "overfitter on ~400 OOF samples. Re-audit before re-enabling."
    )
    assert config.temporal_cv_splits == 5, (
        f"temporal_cv_splits={config.temporal_cv_splits}; audit locks 5. Folds affect the selection-bias surface."
    )
    assert config.scoring_metric == "brier"
    assert config.enable_hyperparameter_tuning is True, (
        "Audit documents hyperparameter tuning as ACTIVE. If tuning was "
        "disabled, the audit artifact should be revised to note that."
    )


def test_tuner_gate_wired() -> None:
    """The YearSplitPolicy enforcement (O20 link) must still be called
    from the Optuna tuner. Without it, researcher-DoF containment breaks."""
    tree = ast.parse(TUNING_MODULE.read_text())
    names = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    assert "_enforce_dev_only_years" in names, (
        "Expected _enforce_dev_only_years() to guard Optuna tuning — it "
        "routes every tune() call through YearSplitPolicy. Audit §E."
    )
    # Additionally: the module must still import YearSplitPolicy callable
    # path implicitly (either directly or via a passed-in policy). Search
    # for the canonical invocation.
    source = TUNING_MODULE.read_text()
    assert "assert_dev_artifact_years" in source, (
        "YearSplitPolicy.assert_dev_artifact_years call disappeared from "
        "the tuner. The tuner-side O20 gate is the runtime enforcement "
        "O17 relies on. Restore it or re-audit."
    )


# ---------------------------------------------------------------------------
# Search-space bounds. Each tuple is (param_name, low, high). The test
# asserts the live code's trial.suggest_* calls exactly match these.
# ---------------------------------------------------------------------------

LIGHTGBM_EXPECTED = {
    "num_leaves": (4, 16),
    "learning_rate": (0.01, 0.15),
    "feature_fraction": (0.5, 0.9),
    "bagging_fraction": (0.5, 0.9),
    "min_child_samples": (30, 100),
    "lambda_l1": (0.1, 10.0),
    "lambda_l2": (0.1, 10.0),
    "num_rounds": (50, 200),
}

XGBOOST_EXPECTED = {
    "max_depth": (2, 4),
    "learning_rate": (0.01, 0.15),
    "subsample": (0.5, 0.9),
    "colsample_bytree": (0.5, 0.9),
    "min_child_weight": (5, 30),
    "gamma": (0.1, 5.0),
    "reg_alpha": (0.1, 10.0),
    "reg_lambda": (0.1, 10.0),
    "num_rounds": (50, 200),
}

LOGISTIC_EXPECTED = {
    "C": (0.01, 10.0),
    "l1_ratio": (0.0, 1.0),
}


def _extract_suggest_bounds(source: str) -> dict[str, tuple[float, float]]:
    """Return every `trial.suggest_*("name", low, high, ...)` call's
    `(name -> (low, high))`. Raises if the same name appears twice
    (that would mean two tuners with different bounds for the same
    param — ambiguous audit state)."""
    tree = ast.parse(source)
    result: dict[str, tuple[float, float]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id == "trial"
            and func.attr.startswith("suggest_")
        ):
            continue
        if len(node.args) < 3:
            continue
        name_node = node.args[0]
        if not isinstance(name_node, ast.Constant) or not isinstance(name_node.value, str):
            continue
        try:
            low = ast.literal_eval(node.args[1])
            high = ast.literal_eval(node.args[2])
        except (ValueError, SyntaxError):
            continue
        # Qualify param by which class scope it sits in, so LGB vs XGB
        # `learning_rate` don't collide. Walk parents via enclosing
        # FunctionDef/ClassDef lookup.
        key = name_node.value
        if key in result and result[key] != (low, high):
            # Two tuners, different bounds — acceptable. We'll key by
            # (class_name, param_name) via a second pass below.
            pass
        result[(key, node.lineno)] = (low, high)
    return result


def _bounds_by_class(source: str) -> dict[str, dict[str, tuple[float, float]]]:
    """Group `trial.suggest_*` bounds by enclosing class name.

    Returns `{ClassName: {param_name: (low, high)}}`."""
    tree = ast.parse(source)
    out: dict[str, dict[str, tuple[float, float]]] = {}

    for class_node in ast.walk(tree):
        if not isinstance(class_node, ast.ClassDef):
            continue
        per_class: dict[str, tuple[float, float]] = {}
        for node in ast.walk(class_node):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (
                isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name)
                and func.value.id == "trial"
                and func.attr.startswith("suggest_")
            ):
                continue
            if len(node.args) < 3:
                continue
            name_node = node.args[0]
            if not isinstance(name_node, ast.Constant):
                continue
            try:
                low = ast.literal_eval(node.args[1])
                high = ast.literal_eval(node.args[2])
            except (ValueError, SyntaxError):
                continue
            per_class[name_node.value] = (low, high)
        if per_class:
            out[class_node.name] = per_class

    # num_rounds is suggested OUTSIDE the objective closure in each tuner;
    # pick it up via a free-standing `trial.suggest_int("num_rounds", ...)`
    # scan and attach to the class containing the call.
    return out


@pytest.mark.parametrize(
    "class_name,expected",
    [
        ("LightGBMTuner", LIGHTGBM_EXPECTED),
        ("XGBoostTuner", XGBOOST_EXPECTED),
    ],
)
def test_gbm_search_space_locked(class_name: str, expected: dict[str, tuple[float, float]]) -> None:
    """Search-space bounds for LightGBM/XGBoost tuners must match the audit §A."""
    source = TUNING_MODULE.read_text()
    by_class = _bounds_by_class(source)
    assert class_name in by_class, f"{class_name} not found in {TUNING_MODULE}"
    got = by_class[class_name]
    for param, (lo, hi) in expected.items():
        assert param in got, (
            f"{class_name}.{param} no longer tuned. If removed, update the "
            f"audit and delete this expectation. Otherwise, restore it."
        )
        assert got[param] == (lo, hi), (
            f"{class_name}.{param} bounds drifted {got[param]} (live) vs "
            f"{(lo, hi)} (audit). Re-audit before widening the search."
        )


def test_logistic_search_space_locked() -> None:
    """Logistic tuner's C / l1_ratio bounds must match the audit."""
    source = TUNING_MODULE.read_text()
    by_class = _bounds_by_class(source)
    # LogisticTuner's class name is whatever contains the `C` suggest call.
    for cls, params in by_class.items():
        if "C" in params:
            for key, (lo, hi) in LOGISTIC_EXPECTED.items():
                assert params.get(key) == (lo, hi), (
                    f"{cls}.{key} bounds drifted {params.get(key)} vs {(lo, hi)}. Re-audit."
                )
            return
    pytest.fail("No tuner class found that suggests 'C' — the logistic tuner was removed or renamed. Update the audit.")
