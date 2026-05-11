#!/usr/bin/env python3
"""Strict Kaggle admission gate for incumbent vs candidate experiments.

This script enforces a stricter selection process than a pooled historical
backtest:

1. Tune/select on a development + shadow split.
2. Freeze the chosen candidate configuration.
3. Admit only if the frozen candidate beats the incumbent on a final holdout.

It supports either:
- a single incumbent/candidate from CLI flags, or
- an experiment spec JSON with multiple model types and parameter grids.

Objective policy for this script:

- Optimize Kaggle candidates on held-out tournament Brier.
- Use BSS as a guardrail, not as the primary ranking target.
- Prefer strong recency weighting over pooled all-history means.

See ``docs/kaggle-objective-policy.md`` for the current repo policy.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.bootstrap_metrics import expected_calibration_error
from src.ml.evaluation.loyo_protocol import compute_ablation_threshold
from src.prediction.torvik_correction import (
    TorvikCorrectionConfig,
    fit_torvik_correction_from_year_records,
    fit_torvik_isotonic_from_year_records,
)
from src.prediction.torvik_kaggle import _select_weighted_alpha

PERGAME_ARTIFACT = REPO_ROOT / "artifacts" / "loyo_pergame_predictions.json"
DEFAULT_DEV_YEARS = [year for year in range(2009, 2023) if year != 2020]
DEFAULT_SHADOW_YEARS = [2023, 2024]
DEFAULT_FINAL_YEARS = [2025, 2026]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    mode: str
    params: Dict[str, Any]


@dataclass
class YearMetrics:
    year: int
    brier: float
    seed_brier: float
    bss: float
    ece: float
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "year": self.year,
            "brier": self.brier,
            "seed_brier": self.seed_brier,
            "bss": self.bss,
            "ece": self.ece,
            "metadata": self.metadata,
        }


@dataclass
class EvaluationSummary:
    spec: ModelSpec
    per_year: List[YearMetrics]

    @property
    def mean_brier(self) -> float:
        return float(np.mean([row.brier for row in self.per_year])) if self.per_year else float("nan")

    @property
    def mean_seed_brier(self) -> float:
        return float(np.mean([row.seed_brier for row in self.per_year])) if self.per_year else float("nan")

    @property
    def mean_bss(self) -> float:
        if not self.per_year:
            return float("nan")
        return 1.0 - self.mean_brier / max(self.mean_seed_brier, 1e-12)

    @property
    def mean_ece(self) -> float:
        return float(np.mean([row.ece for row in self.per_year])) if self.per_year else float("nan")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "spec": asdict(self.spec),
            "mean_brier": self.mean_brier,
            "mean_seed_brier": self.mean_seed_brier,
            "mean_bss": self.mean_bss,
            "mean_ece": self.mean_ece,
            "per_year": [row.to_dict() for row in self.per_year],
        }


def load_pergame(path: Path) -> dict[int, list[dict]]:
    with open(path) as f:
        raw = json.load(f)
    return {int(year): rows for year, rows in raw.items()}


def _parse_years_arg(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def _normalize_param(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 10)
    return value


def _spec_name(mode: str, params: Dict[str, Any]) -> str:
    if not params:
        return mode
    parts = [mode]
    for key in sorted(params):
        value = _normalize_param(params[key])
        parts.append(f"{key}={value}")
    return "__".join(parts)


def _expand_search_space(entry: Dict[str, Any]) -> List[ModelSpec]:
    modes = entry.get("modes")
    if modes is None:
        mode = entry.get("mode")
        if mode is None:
            raise ValueError("Search-space entry must define 'mode' or 'modes'")
        modes = [mode]

    param_space = entry.get("params", {})
    keys = sorted(param_space)
    value_lists = []
    for key in keys:
        raw = param_space[key]
        value_lists.append(raw if isinstance(raw, list) else [raw])

    specs = []
    for mode in modes:
        if not keys:
            specs.append(ModelSpec(name=_spec_name(mode, {}), mode=mode, params={}))
            continue
        for combo in itertools.product(*value_lists):
            params = {key: _normalize_param(value) for key, value in zip(keys, combo)}
            specs.append(ModelSpec(name=_spec_name(mode, params), mode=mode, params=params))
    return specs


def build_candidate_experiments(spec_data: Dict[str, Any]) -> List[ModelSpec]:
    experiments = []

    for entry in spec_data.get("candidate_experiments", []):
        mode = entry["mode"]
        params = {key: _normalize_param(value) for key, value in entry.get("params", {}).items()}
        name = entry.get("name") or _spec_name(mode, params)
        experiments.append(ModelSpec(name=name, mode=mode, params=params))

    for entry in spec_data.get("candidate_search_spaces", []):
        experiments.extend(_expand_search_space(entry))

    if not experiments:
        raise ValueError("No candidate experiments defined")

    deduped = {}
    for experiment in experiments:
        key = (experiment.mode, tuple(sorted(experiment.params.items())))
        deduped[key] = experiment
    return list(deduped.values())


def _coerce_float(params: Dict[str, Any], key: str, default: float) -> float:
    return float(params.get(key, default))


def _coerce_int_or_none(params: Dict[str, Any], key: str, default: Optional[int]) -> Optional[int]:
    value = params.get(key, default)
    return None if value is None else int(value)


def _validate_split(dev_years: list[int], shadow_years: list[int], final_years: list[int]) -> None:
    if not dev_years or not shadow_years or not final_years:
        raise ValueError("Development, shadow, and final splits must all be non-empty")

    split_names = {
        "dev_years": dev_years,
        "shadow_years": shadow_years,
        "final_years": final_years,
    }
    seen: dict[int, str] = {}
    for split_name, years in split_names.items():
        if years != sorted(years):
            raise ValueError(f"{split_name} must be sorted ascending")
        for year in years:
            prior_split = seen.get(year)
            if prior_split is not None:
                raise ValueError(f"Year {year} appears in both {prior_split} and {split_name}")
            seen[year] = split_name

    if max(dev_years) >= min(shadow_years):
        raise ValueError("Development years must end before shadow years begin")
    if max(shadow_years) >= min(final_years):
        raise ValueError("Shadow years must end before final years begin")


def _evaluate_torvik(test_rows: list[dict], params: Dict[str, Any]) -> YearMetrics:
    clip_lo = _coerce_float(params, "clip_lo", 0.01)
    clip_hi = _coerce_float(params, "clip_hi", 0.99)
    preds = np.clip(np.asarray([float(row["torvik"]) for row in test_rows], dtype=float), clip_lo, clip_hi)
    outcomes = np.asarray([float(row["outcome"]) for row in test_rows], dtype=float)
    seed = np.asarray([float(row["seed"]) for row in test_rows], dtype=float)

    brier = float(np.mean((preds - outcomes) ** 2))
    seed_brier = float(np.mean((seed - outcomes) ** 2))
    return YearMetrics(
        year=int(test_rows[0]["year"]),
        brier=brier,
        seed_brier=seed_brier,
        bss=1.0 - brier / max(seed_brier, 1e-12),
        ece=expected_calibration_error(preds, outcomes),
        metadata={"clip_lo": clip_lo, "clip_hi": clip_hi},
    )


def _evaluate_ensemble(
    train_year_records: dict[int, list[dict]],
    test_rows: list[dict],
    params: Dict[str, Any],
) -> YearMetrics:
    clip_lo = _coerce_float(params, "clip_lo", 0.01)
    clip_hi = _coerce_float(params, "clip_hi", 0.99)
    recent_start = _coerce_int_or_none(params, "recent_year_start", 2021)
    recent_weight = _coerce_float(params, "recent_year_weight", 2.0)
    recent_count = _coerce_int_or_none(params, "recent_year_count", None)
    recent_total_ratio = _coerce_float(params, "recent_total_ratio", 1.0)

    alpha, alpha_info = _select_weighted_alpha(
        train_year_records,
        clip_lo,
        clip_hi,
        recent_year_start=recent_start,
        recent_year_weight=recent_weight,
        recent_year_count=recent_count,
        recent_total_ratio=recent_total_ratio,
    )

    outcomes = np.asarray([float(row["outcome"]) for row in test_rows], dtype=float)
    seed = np.asarray([float(row["seed"]) for row in test_rows], dtype=float)
    preds = np.asarray(
        [
            np.clip(alpha * float(row["torvik"]) + (1.0 - alpha) * float(row["pipeline"]), clip_lo, clip_hi)
            for row in test_rows
        ],
        dtype=float,
    )
    brier = float(np.mean((preds - outcomes) ** 2))
    seed_brier = float(np.mean((seed - outcomes) ** 2))
    metadata = {
        "alpha": alpha,
        "alpha_training_info": {
            **alpha_info,
            "recent_year_count": recent_count,
            "recent_total_ratio": recent_total_ratio,
        },
    }
    return YearMetrics(
        year=int(test_rows[0]["year"]),
        brier=brier,
        seed_brier=seed_brier,
        bss=1.0 - brier / max(seed_brier, 1e-12),
        ece=expected_calibration_error(preds, outcomes),
        metadata=metadata,
    )


def _evaluate_torvik_corrected(
    train_year_records: dict[int, list[dict]],
    test_rows: list[dict],
    params: Dict[str, Any],
) -> YearMetrics:
    clip_lo = _coerce_float(params, "clip_lo", 0.01)
    clip_hi = _coerce_float(params, "clip_hi", 0.99)
    recent_start = _coerce_int_or_none(params, "recent_year_start", 2021)
    recent_weight = _coerce_float(params, "recent_year_weight", 2.0)
    recent_count = _coerce_int_or_none(params, "recent_year_count", None)
    recent_total_ratio = _coerce_float(params, "recent_total_ratio", 1.0)
    ridge = _coerce_float(params, "correction_ridge", 5.0)
    max_correction = _coerce_float(params, "max_correction", 0.10)

    model = fit_torvik_correction_from_year_records(
        train_year_records,
        config=TorvikCorrectionConfig(
            clip_lo=clip_lo,
            clip_hi=clip_hi,
            ridge=ridge,
            max_correction=max_correction,
        ),
        recent_year_start=recent_start,
        recent_year_weight=recent_weight,
        recent_year_count=recent_count,
        recent_total_ratio=recent_total_ratio,
    )

    from src.prediction.torvik_correction import _resolve_signal, round_to_num

    torvik = np.asarray([float(row["torvik"]) for row in test_rows], dtype=float)
    seed1 = np.asarray([int(row["seed1"]) for row in test_rows], dtype=float)
    seed2 = np.asarray([int(row["seed2"]) for row in test_rows], dtype=float)
    outcomes = np.asarray([float(row["outcome"]) for row in test_rows], dtype=float)
    seed = np.asarray([float(row["seed"]) for row in test_rows], dtype=float)
    market = np.asarray([_resolve_signal(row.get("odds"), float(row["torvik"])) for row in test_rows], dtype=float)
    elo = np.asarray([_resolve_signal(row.get("elo"), float(row["torvik"])) for row in test_rows], dtype=float)
    rounds = np.asarray([round_to_num(row.get("round")) for row in test_rows], dtype=float)
    preds = model.predict(torvik, seed1, seed2, market_probs=market, elo_probs=elo, round_nums=rounds)
    brier = float(np.mean((preds - outcomes) ** 2))
    seed_brier = float(np.mean((seed - outcomes) ** 2))

    metadata = {
        "coef": model.coef_.tolist() if model.coef_ is not None else None,
        **(model.training_info_ or {}),
        "correction_ridge": ridge,
        "max_correction": max_correction,
    }
    return YearMetrics(
        year=int(test_rows[0]["year"]),
        brier=brier,
        seed_brier=seed_brier,
        bss=1.0 - brier / max(seed_brier, 1e-12),
        ece=expected_calibration_error(preds, outcomes),
        metadata=metadata,
    )


def _evaluate_torvik_isotonic(
    train_year_records: dict[int, list[dict]],
    test_rows: list[dict],
    params: Dict[str, Any],
) -> YearMetrics:
    clip_lo = _coerce_float(params, "clip_lo", 0.01)
    clip_hi = _coerce_float(params, "clip_hi", 0.99)
    recent_start = _coerce_int_or_none(params, "recent_year_start", 2021)
    recent_weight = _coerce_float(params, "recent_year_weight", 2.0)
    recent_count = _coerce_int_or_none(params, "recent_year_count", None)
    recent_total_ratio = _coerce_float(params, "recent_total_ratio", 1.0)
    ridge = _coerce_float(params, "correction_ridge", 5.0)
    max_correction = _coerce_float(params, "max_correction", 0.10)

    model = fit_torvik_isotonic_from_year_records(
        train_year_records,
        config=TorvikCorrectionConfig(
            clip_lo=clip_lo,
            clip_hi=clip_hi,
            ridge=ridge,
            max_correction=max_correction,
        ),
        recent_year_start=recent_start,
        recent_year_weight=recent_weight,
        recent_year_count=recent_count,
        recent_total_ratio=recent_total_ratio,
    )

    from src.prediction.torvik_correction import _resolve_signal, round_to_num

    torvik = np.asarray([float(row["torvik"]) for row in test_rows], dtype=float)
    seed1 = np.asarray([int(row["seed1"]) for row in test_rows], dtype=float)
    seed2 = np.asarray([int(row["seed2"]) for row in test_rows], dtype=float)
    outcomes = np.asarray([float(row["outcome"]) for row in test_rows], dtype=float)
    seed = np.asarray([float(row["seed"]) for row in test_rows], dtype=float)
    market = np.asarray([_resolve_signal(row.get("odds"), float(row["torvik"])) for row in test_rows], dtype=float)
    elo = np.asarray([_resolve_signal(row.get("elo"), float(row["torvik"])) for row in test_rows], dtype=float)
    rounds = np.asarray([round_to_num(row.get("round")) for row in test_rows], dtype=float)
    preds = model.predict(torvik, seed1, seed2, market_probs=market, elo_probs=elo, round_nums=rounds)
    brier = float(np.mean((preds - outcomes) ** 2))
    seed_brier = float(np.mean((seed - outcomes) ** 2))

    metadata = {
        "coef": model.coef_.tolist() if model.coef_ is not None else None,
        **(model.training_info_ or {}),
        "correction_ridge": ridge,
        "max_correction": max_correction,
    }
    return YearMetrics(
        year=int(test_rows[0]["year"]),
        brier=brier,
        seed_brier=seed_brier,
        bss=1.0 - brier / max(seed_brier, 1e-12),
        ece=expected_calibration_error(preds, outcomes),
        metadata=metadata,
    )


def evaluate_spec(
    data: dict[int, list[dict]],
    spec: ModelSpec,
    eval_years: Iterable[int],
    fit_years: Iterable[int],
    min_train_years: int = 3,
) -> EvaluationSummary:
    per_year = []
    fit_years_set = set(fit_years)

    for year in eval_years:
        train_years = sorted(y for y in fit_years_set if y < year and data.get(y))
        if len(train_years) < min_train_years:
            continue
        train_year_records = {y: data[y] for y in train_years}
        test_rows = [dict(row, year=year) for row in data.get(year, [])]
        if not train_year_records or not test_rows:
            continue

        if spec.mode == "torvik":
            metrics = _evaluate_torvik(test_rows, spec.params)
        elif spec.mode == "ensemble":
            metrics = _evaluate_ensemble(train_year_records, test_rows, spec.params)
        elif spec.mode == "torvik_corrected":
            metrics = _evaluate_torvik_corrected(train_year_records, test_rows, spec.params)
        elif spec.mode == "torvik_isotonic":
            metrics = _evaluate_torvik_isotonic(train_year_records, test_rows, spec.params)
        else:
            raise ValueError(f"Unsupported model mode: {spec.mode}")

        per_year.append(metrics)

    return EvaluationSummary(spec=spec, per_year=per_year)


def _merge_summaries(spec: ModelSpec, *summaries: EvaluationSummary) -> EvaluationSummary:
    per_year = []
    for summary in summaries:
        if summary.spec != spec:
            raise ValueError("Cannot merge summaries from different model specs")
        per_year.extend(summary.per_year)
    return EvaluationSummary(spec=spec, per_year=sorted(per_year, key=lambda row: row.year))


def select_best_candidate(evaluations: List[EvaluationSummary]) -> EvaluationSummary:
    if not evaluations:
        raise ValueError("No candidate evaluations to select from")

    return min(
        evaluations,
        key=lambda summary: (
            summary.mean_brier,
            summary.mean_ece,
            summary.spec.mode,
            summary.spec.name,
        ),
    )


def _coefficient_sign_stability(rows: List[YearMetrics]) -> Dict[str, Any]:
    coef_rows = [row.metadata.get("coef") for row in rows if row.metadata.get("coef") is not None]
    if not coef_rows:
        return {"available": False}

    matrix = np.asarray(coef_rows, dtype=float)
    per_feature = []
    for idx in range(1, matrix.shape[1]):  # Skip intercept.
        signs = np.sign(matrix[:, idx])
        signs = signs[signs != 0]
        if len(signs) == 0:
            consistency = 1.0
            dominant_sign = 0
        else:
            positive = int((signs > 0).sum())
            negative = int((signs < 0).sum())
            dominant = max(positive, negative)
            consistency = dominant / len(signs)
            dominant_sign = 1 if positive >= negative else -1
        per_feature.append(
            {
                "feature_index": idx,
                "consistency": float(consistency),
                "dominant_sign": dominant_sign,
            }
        )

    return {
        "available": True,
        "min_consistency": float(min(item["consistency"] for item in per_feature)),
        "per_feature": per_feature,
    }


def evaluate_admission_gate(
    incumbent_holdout: EvaluationSummary,
    candidate_holdout: EvaluationSummary,
    incumbent_final: EvaluationSummary,
    candidate_final: EvaluationSummary,
    min_final_mean_improvement: float,
    max_final_year_regression: float,
    min_holdout_win_rate: float,
    max_calibration_degradation: float,
    min_coef_sign_stability: float,
) -> Dict[str, Any]:
    incumbent_final_mean = incumbent_final.mean_brier
    candidate_final_mean = candidate_final.mean_brier
    mean_improvement = incumbent_final_mean - candidate_final_mean

    final_year_deltas = []
    for inc_row, cand_row in zip(incumbent_final.per_year, candidate_final.per_year):
        final_year_deltas.append(
            {
                "year": cand_row.year,
                "delta": cand_row.brier - inc_row.brier,
                "candidate_bss": cand_row.bss,
            }
        )

    holdout_pairs = list(zip(incumbent_holdout.per_year, candidate_holdout.per_year))
    holdout_wins = sum(1 for inc_row, cand_row in holdout_pairs if cand_row.brier < inc_row.brier)
    holdout_win_rate = holdout_wins / max(len(holdout_pairs), 1)

    final_ece_degradation = candidate_final.mean_ece - incumbent_final.mean_ece
    incumbent_holdout_briers = [row.brier for row in incumbent_holdout.per_year]
    tolerance = compute_ablation_threshold(incumbent_holdout_briers) if incumbent_holdout_briers else 0.0
    coef_stability = _coefficient_sign_stability(candidate_holdout.per_year)

    checks = {
        "mean_improvement": {
            "passed": mean_improvement >= min_final_mean_improvement,
            "value": mean_improvement,
            "threshold": min_final_mean_improvement,
        },
        "per_year_regression": {
            "passed": all(item["delta"] <= max_final_year_regression for item in final_year_deltas),
            "threshold": max_final_year_regression,
            "per_year": final_year_deltas,
        },
        "holdout_win_rate": {
            "passed": holdout_win_rate >= min_holdout_win_rate,
            "value": holdout_win_rate,
            "threshold": min_holdout_win_rate,
            "wins": holdout_wins,
            "n_years": len(holdout_pairs),
        },
        "positive_final_bss": {
            "passed": all(item["candidate_bss"] > 0.0 for item in final_year_deltas),
            "per_year": [{"year": item["year"], "candidate_bss": item["candidate_bss"]} for item in final_year_deltas],
        },
        "calibration_degradation": {
            "passed": final_ece_degradation <= max_calibration_degradation,
            "value": final_ece_degradation,
            "threshold": max_calibration_degradation,
        },
        "coefficient_stability": {
            "passed": (not coef_stability.get("available"))
            or coef_stability["min_consistency"] >= min_coef_sign_stability,
            "threshold": min_coef_sign_stability,
            "details": coef_stability,
        },
        "regression_tolerance": {
            "passed": candidate_final_mean - incumbent_final_mean <= tolerance,
            "delta": candidate_final_mean - incumbent_final_mean,
            "tolerance": tolerance,
        },
    }

    return {
        "passed": all(item["passed"] for item in checks.values()),
        "mean_improvement": mean_improvement,
        "checks": checks,
    }


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _parse_json_arg(text: Optional[str]) -> Dict[str, Any]:
    if not text:
        return {}
    return json.loads(text)


def build_specs_from_args(args: argparse.Namespace) -> tuple[ModelSpec, List[ModelSpec]]:
    if args.experiment_spec:
        with open(args.experiment_spec) as f:
            spec_data = json.load(f)

        incumbent_cfg = spec_data.get("incumbent")
        if incumbent_cfg is None:
            raise ValueError("Experiment spec must define an 'incumbent'")

        incumbent = ModelSpec(
            name=incumbent_cfg.get("name") or _spec_name(incumbent_cfg["mode"], incumbent_cfg.get("params", {})),
            mode=incumbent_cfg["mode"],
            params={key: _normalize_param(value) for key, value in incumbent_cfg.get("params", {}).items()},
        )
        candidates = build_candidate_experiments(spec_data)
        return incumbent, candidates

    incumbent = ModelSpec(
        name=_spec_name(args.incumbent_mode, _parse_json_arg(args.incumbent_params_json)),
        mode=args.incumbent_mode,
        params=_parse_json_arg(args.incumbent_params_json),
    )
    candidate = ModelSpec(
        name=_spec_name(args.candidate_mode, _parse_json_arg(args.candidate_params_json)),
        mode=args.candidate_mode,
        params=_parse_json_arg(args.candidate_params_json),
    )
    return incumbent, [candidate]


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict Kaggle admission gate")
    parser.add_argument("--pergame", default=str(PERGAME_ARTIFACT), help="Per-game artifact path")
    parser.add_argument("--experiment-spec", default=None, help="JSON spec with incumbent and candidate experiments")
    _MODES = ["torvik", "ensemble", "torvik_corrected", "torvik_isotonic"]
    parser.add_argument("--incumbent-mode", default="ensemble", choices=_MODES)
    parser.add_argument("--candidate-mode", default="torvik_corrected", choices=_MODES)
    parser.add_argument("--incumbent-params-json", default='{"recent_year_start": 2021, "recent_year_weight": 2.0}')
    parser.add_argument("--candidate-params-json", default='{"recent_year_start": 2021, "recent_year_weight": 2.0}')
    parser.add_argument("--dev-years", default=",".join(str(y) for y in DEFAULT_DEV_YEARS))
    parser.add_argument("--shadow-years", default=",".join(str(y) for y in DEFAULT_SHADOW_YEARS))
    parser.add_argument("--final-years", default=",".join(str(y) for y in DEFAULT_FINAL_YEARS))
    parser.add_argument("--baseline-output", default="artifacts/kaggle_baseline_ensemble.json")
    parser.add_argument("--report-output", default="artifacts/kaggle_admission_report.json")
    parser.add_argument("--min-final-mean-improvement", type=float, default=0.002)
    parser.add_argument("--max-final-year-regression", type=float, default=0.003)
    parser.add_argument("--min-holdout-win-rate", type=float, default=0.75)
    parser.add_argument("--max-calibration-degradation", type=float, default=0.01)
    parser.add_argument("--min-coef-sign-stability", type=float, default=0.75)
    parser.add_argument("--no-fail", action="store_true", help="Always exit 0 even if admission fails")
    args = parser.parse_args()

    data = load_pergame(Path(args.pergame))
    incumbent_spec, candidate_specs = build_specs_from_args(args)

    dev_years = _parse_years_arg(args.dev_years)
    shadow_years = _parse_years_arg(args.shadow_years)
    final_years = _parse_years_arg(args.final_years)
    try:
        _validate_split(dev_years, shadow_years, final_years)
    except ValueError as exc:
        parser.error(str(exc))

    available_years = set(data)
    for year in dev_years + shadow_years + final_years:
        if year not in available_years:
            parser.error(f"Requested year {year} is not available in {args.pergame}")

    shadow_incumbent = evaluate_spec(data, incumbent_spec, shadow_years, fit_years=dev_years)
    final_incumbent = evaluate_spec(data, incumbent_spec, final_years, fit_years=dev_years + shadow_years)
    holdout_incumbent = _merge_summaries(incumbent_spec, shadow_incumbent, final_incumbent)

    candidate_shadow_evals = [evaluate_spec(data, spec, shadow_years, fit_years=dev_years) for spec in candidate_specs]
    selected_shadow = select_best_candidate(candidate_shadow_evals)
    selected_spec = selected_shadow.spec
    final_candidate = evaluate_spec(data, selected_spec, final_years, fit_years=dev_years + shadow_years)
    holdout_candidate = _merge_summaries(selected_spec, selected_shadow, final_candidate)

    admission = evaluate_admission_gate(
        holdout_incumbent,
        holdout_candidate,
        final_incumbent,
        final_candidate,
        min_final_mean_improvement=args.min_final_mean_improvement,
        max_final_year_regression=args.max_final_year_regression,
        min_holdout_win_rate=args.min_holdout_win_rate,
        max_calibration_degradation=args.max_calibration_degradation,
        min_coef_sign_stability=args.min_coef_sign_stability,
    )

    baseline_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "split": {
            "dev_years": dev_years,
            "shadow_years": shadow_years,
            "final_years": final_years,
        },
        "incumbent": holdout_incumbent.to_dict(),
    }
    report_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "split": {
            "dev_years": dev_years,
            "shadow_years": shadow_years,
            "final_years": final_years,
        },
        "incumbent": {
            "shadow": shadow_incumbent.to_dict(),
            "final": final_incumbent.to_dict(),
            "holdout": holdout_incumbent.to_dict(),
        },
        "candidate_search": {
            "n_experiments": len(candidate_shadow_evals),
            "shadow_ranked": sorted(
                [summary.to_dict() for summary in candidate_shadow_evals],
                key=lambda item: (item["mean_brier"], item["mean_ece"], item["spec"]["name"]),
            ),
            "selected_spec": asdict(selected_spec),
        },
        "candidate_selected": {
            "shadow": selected_shadow.to_dict(),
            "final": final_candidate.to_dict(),
            "holdout": holdout_candidate.to_dict(),
        },
        "admission_gate": admission,
    }

    _write_json(Path(args.baseline_output), baseline_payload)
    _write_json(Path(args.report_output), report_payload)

    print(f"Incumbent: {incumbent_spec.name}")
    print(f"Selected candidate: {selected_spec.name}")
    print(f"Shadow mean Brier: {selected_shadow.mean_brier:.4f}")
    print(f"Final incumbent mean Brier: {final_incumbent.mean_brier:.4f}")
    print(f"Final candidate mean Brier: {final_candidate.mean_brier:.4f}")
    print(f"Mean improvement: {admission['mean_improvement']:+.4f}")
    print(f"Admission: {'PASS' if admission['passed'] else 'FAIL'}")
    print(f"Baseline artifact: {args.baseline_output}")
    print(f"Admission report: {args.report_output}")

    if not admission["passed"] and not args.no_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
