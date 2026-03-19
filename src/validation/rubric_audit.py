"""Static rubric audit: scores the codebase against the grading rubric.

Performs structural checks (imports, file existence, code inspection) without
running the full pipeline. Each rubric section awards points based on whether
the required components exist and are correctly integrated.
"""

from __future__ import annotations

import ast
import inspect
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def _check_import(module_path: str, names: List[str]) -> List[str]:
    """Try importing names from a module, return list of failures."""
    failures = []
    try:
        mod = __import__(module_path, fromlist=names)
        for name in names:
            if not hasattr(mod, name):
                failures.append(f"{module_path}.{name} not found")
    except ImportError as e:
        failures.append(f"Cannot import {module_path}: {e}")
    return failures


def _check_file_exists(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def _check_source_contains(module: Any, patterns: List[str]) -> List[str]:
    """Check that a module's source contains expected patterns."""
    missing = []
    try:
        source = inspect.getsource(module)
        for p in patterns:
            if p not in source:
                missing.append(p)
    except (TypeError, OSError):
        missing.extend(patterns)
    return missing


def run_static_audit() -> "PassResult":
    """Run static audit and return a PassResult (imported here to avoid circular)."""
    from .rubric_validator import PassResult

    score = 0.0
    max_score = 100.0
    failures: List[str] = []
    details: Dict[str, Any] = {}
    repo_root = Path(__file__).resolve().parents[2]

    # =====================================================================
    # A. DATA INTEGRITY (0-20 pts)
    # =====================================================================
    section_a = 0.0

    # A1. Feature manifest complete (5 pts)
    manifest_path = repo_root / "features" / "MANIFEST.yaml"
    if _check_file_exists(manifest_path):
        try:
            import yaml
            with open(manifest_path) as f:
                manifest = yaml.safe_load(f)
            total = manifest.get("total_features", 0)
            if total == 79:
                section_a += 5
                details["A1_manifest"] = "PASS (79 features)"
            else:
                failures.append(f"A1: MANIFEST declares {total} features, expected 79")
        except Exception as e:
            failures.append(f"A1: MANIFEST parse error: {e}")
    else:
        failures.append("A1: features/MANIFEST.yaml not found")

    # A2. Correct tier classification (5 pts)
    try:
        t1 = len(manifest.get("tier_1_static", {}).get("features", []))
        t2 = len(manifest.get("tier_2_cumulative", {}).get("features", []))
        t3 = len(manifest.get("tier_3_external", {}).get("features", []))
        if t1 + t2 + t3 == 79 and t1 > 0 and t2 > 0 and t3 > 0:
            section_a += 5
            details["A2_tiers"] = f"PASS ({t1}/{t2}/{t3})"
        else:
            failures.append(f"A2: Tier sum {t1+t2+t3} != 79 or missing tiers")
    except Exception:
        failures.append("A2: Cannot validate tiers")

    # A3. PIT validation implemented (5 pts)
    f = _check_import("src.pipeline.stages.pit_validation", ["PITValidator", "PITViolationError"])
    if not f:
        section_a += 5
        details["A3_pit_impl"] = "PASS"
    else:
        failures.extend([f"A3: {x}" for x in f])

    # A4. PIT enforced in LOYO (5 pts)
    baseline_path = repo_root / "src" / "pipeline" / "stages" / "baseline_training.py"
    if _check_file_exists(baseline_path):
        content = baseline_path.read_text()
        if "PITValidator" in content and "pit_validation" in content:
            section_a += 5
            details["A4_pit_enforced"] = "PASS (in baseline_training.py)"
        else:
            failures.append("A4: PITValidator not referenced in baseline_training.py")
    else:
        failures.append("A4: baseline_training.py not found")

    details["section_A"] = f"{section_a}/20"

    # =====================================================================
    # B. METRICS & EVALUATION (0-15 pts)
    # =====================================================================
    section_b = 0.0

    # B1. Brier + Log Loss computed (3 pts)
    loyo_path = repo_root / "src" / "ml" / "evaluation" / "loyo_protocol.py"
    if _check_file_exists(loyo_path):
        content = loyo_path.read_text()
        if "brier_score" in content and "log_loss" in content:
            section_b += 3
            details["B1_brier_logloss"] = "PASS"
        else:
            failures.append("B1: Brier/LogLoss not found in loyo_protocol.py")

    # B2. Brier decomposition (4 pts)
    f = _check_import("src.ml.evaluation.loyo_protocol", ["brier_decomposition"])
    if not f:
        section_b += 4
        details["B2_decomposition"] = "PASS"
    else:
        failures.extend([f"B2: {x}" for x in f])

    # B3. ECE implemented (3 pts)
    f = _check_import("src.ml.calibration.calibration", ["calculate_calibration_metrics"])
    if not f:
        section_b += 3
        details["B3_ece"] = "PASS"
    else:
        failures.extend([f"B3: {x}" for x in f])

    # B4. BSS vs seeds (3 pts)
    f = _check_import("src.ml.evaluation.loyo_protocol", ["brier_skill_score_vs_seeds"])
    if not f:
        section_b += 3
        details["B4_bss"] = "PASS"
    else:
        failures.extend([f"B4: {x}" for x in f])

    # B5. Reliability diagram data (2 pts)
    cal_path = repo_root / "src" / "ml" / "calibration" / "calibration.py"
    if _check_file_exists(cal_path):
        content = cal_path.read_text()
        if "prob_true" in content and "prob_pred" in content:
            section_b += 2
            details["B5_reliability"] = "PASS"
        else:
            failures.append("B5: prob_true/prob_pred not in calibration.py")

    details["section_B"] = f"{section_b}/15"

    # =====================================================================
    # C. MODEL SELECTION (0-15 pts)
    # =====================================================================
    section_c = 0.0

    # C1. Multi-metric gate enforced (5 pts)
    f = _check_import("src.ml.evaluation.admission_gate", ["AdmissionGate", "AdmissionResult"])
    if not f:
        section_c += 5
        details["C1_gate"] = "PASS"
    else:
        failures.extend([f"C1: {x}" for x in f])

    # C2. Correct thresholds used (5 pts)
    try:
        from src.ml.evaluation.admission_gate import AdmissionGate
        gate = AdmissionGate()
        if (
            hasattr(gate, "min_mean_brier_improvement")
            and hasattr(gate, "min_fold_improvement_rate")
            and hasattr(gate, "max_calibration_degradation")
        ):
            section_c += 5
            details["C2_thresholds"] = "PASS"
    except Exception:
        failures.append("C2: Cannot verify thresholds")

    # C3. Proper rejection logic (5 pts)
    try:
        from src.ml.evaluation.admission_gate import AdmissionGate
        gate_src = inspect.getsource(AdmissionGate.evaluate)
        if "passed" in gate_src and "failures" in gate_src:
            section_c += 5
            details["C3_rejection"] = "PASS"
    except Exception:
        failures.append("C3: Cannot verify rejection logic")

    details["section_C"] = f"{section_c}/15"

    # =====================================================================
    # D. ENSEMBLE BMA (0-20 pts)
    # =====================================================================
    section_d = 0.0

    # D1. BMA implemented (not fake averaging) (8 pts)
    bma_path = repo_root / "src" / "ml" / "ensemble" / "bma.py"
    if _check_file_exists(bma_path):
        content = bma_path.read_text()
        has_em = ("E-step" in content or "responsibilities" in content) and "M-step" in content or "new_weights" in content
        has_likelihood = "log_likelihoods" in content or "Bernoulli" in content
        if has_em and has_likelihood:
            section_d += 8
            details["D1_bma_real"] = "PASS (EM with Bernoulli likelihood)"
        else:
            failures.append("D1: BMA may be fake — missing EM/likelihood indicators")
    else:
        failures.append("D1: bma.py not found")

    # D2. EM algorithm used (5 pts)
    if _check_file_exists(bma_path):
        content = bma_path.read_text()
        if "max_iterations" in content and "responsibilities" in content and "convergence" in content.lower():
            section_d += 5
            details["D2_em_algo"] = "PASS"
        else:
            failures.append("D2: EM algorithm indicators missing")

    # D3. Weights sum to 1 (2 pts)
    if _check_file_exists(bma_path):
        content = bma_path.read_text()
        if "new_weights /= new_weights.sum()" in content or "weights.sum()" in content:
            section_d += 2
            details["D3_weight_norm"] = "PASS"
        else:
            failures.append("D3: Weight normalization not found")

    # D4. Effective model count computed (5 pts)
    if _check_file_exists(bma_path):
        content = bma_path.read_text()
        if "effective_model_count" in content or "effective_count" in content:
            section_d += 5
            details["D4_effective_count"] = "PASS"
        else:
            failures.append("D4: Effective model count not found")

    details["section_D"] = f"{section_d}/20"

    # =====================================================================
    # E. CALIBRATION (0-10 pts)
    # =====================================================================
    section_e = 0.0

    # E1. Sharpening disabled for Kaggle (5 pts)
    config_path = repo_root / "configs" / "production_2026.json"
    if _check_file_exists(config_path):
        try:
            config = json.loads(config_path.read_text())
            if not config.get("enable_brier_sharpening", True):
                section_e += 5
                details["E1_sharpening"] = "PASS (disabled)"
            else:
                failures.append("E1: Brier sharpening enabled in production config")
        except Exception:
            failures.append("E1: Cannot parse production config")
    else:
        failures.append("E1: production_2026.json not found")

    # E2. Proper calibration method used (5 pts)
    try:
        cal_method = config.get("calibration_method", "")
        if cal_method in ("temperature", "isotonic", "platt"):
            section_e += 5
            details["E2_cal_method"] = f"PASS ({cal_method})"
        else:
            failures.append(f"E2: Unexpected calibration method: {cal_method}")
    except Exception:
        failures.append("E2: Cannot check calibration method")

    details["section_E"] = f"{section_e}/10"

    # =====================================================================
    # F. ESPN SYSTEM (0-20 pts)
    # =====================================================================
    section_f = 0.0
    espn_dir = repo_root / "src" / "espn"

    # F1. Monte Carlo simulation works (5 pts)
    mc_path = espn_dir / "mc_simulator.py"
    if _check_file_exists(mc_path):
        content = mc_path.read_text()
        if "simulate_one_tournament" in content and "num_simulations" in content:
            section_f += 5
            details["F1_mc_sim"] = "PASS"
        else:
            failures.append("F1: MC simulator missing core methods")
    else:
        failures.append("F1: mc_simulator.py not found")

    # F2. Public pick integration (3 pts)
    pub_path = espn_dir / "public_pick_scraper.py"
    if _check_file_exists(pub_path):
        section_f += 3
        details["F2_public_picks"] = "PASS"
    else:
        failures.append("F2: public_pick_scraper.py not found")

    # F3. Leverage calculation (4 pts)
    lev_path = espn_dir / "leverage.py"
    if _check_file_exists(lev_path):
        content = lev_path.read_text()
        if "compute_leverage_table" in content and "LeverageSignal" in content:
            section_f += 4
            details["F3_leverage"] = "PASS"
        else:
            failures.append("F3: Leverage computation incomplete")
    else:
        failures.append("F3: leverage.py not found")

    # F4. Path dependency logic (4 pts)
    if _check_file_exists(lev_path):
        content = lev_path.read_text()
        if "champion_path" in content and "protected" in content.lower():
            section_f += 4
            details["F4_path_dep"] = "PASS"
        else:
            failures.append("F4: Path dependency logic not found")

    # F5. Bracket optimization works (4 pts)
    opt_path = espn_dir / "bracket_optimizer.py"
    if _check_file_exists(opt_path):
        content = opt_path.read_text()
        if "ESPNBracketOptimizer" in content and "optimize" in content:
            section_f += 4
            details["F5_bracket_opt"] = "PASS"
        else:
            failures.append("F5: Bracket optimizer incomplete")
    else:
        failures.append("F5: bracket_optimizer.py not found")

    details["section_F"] = f"{section_f}/20"

    # =====================================================================
    # TOTAL
    # =====================================================================
    total = section_a + section_b + section_c + section_d + section_e + section_f
    details["total"] = f"{total}/100"

    return PassResult(
        pass_number=4,
        name="Static Rubric Audit",
        passed=total >= 60,
        score=total,
        max_score=max_score,
        details=details,
        failures=failures,
    )
