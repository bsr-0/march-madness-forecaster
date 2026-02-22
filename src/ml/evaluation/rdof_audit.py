"""Researcher Degrees of Freedom (RDoF) audit framework.

Addresses cumulative researcher degrees of freedom by:
1. Cataloging every hand-tuned constant with its derivation tier
2. Running holdout evaluation on designated years
3. Sensitivity analysis: grid search each Tier 3 constant via LOYO on dev years
4. Producing a structured audit report with honest OOS metrics
5. Pipeline freeze/verify for pre-registration discipline

HOLDOUT INTEGRITY LEVELS
=========================
This framework distinguishes three levels of holdout integrity, each with
different evidential strength.  All audit reports explicitly tag which level
applies.

Level 3 — RETROSPECTIVE DIAGNOSTIC (current 2005-2025 data):
    The pipeline was iteratively refined using data from all 2005-2025 seasons.
    "Holdout" years (default: 2024, 2025) were observed during development for
    debugging, feature engineering, and sanity checks.  Results at this level
    are useful diagnostics but OVERSTATE true out-of-sample performance because
    the pipeline was shaped by knowledge of these years' outcomes.  Analogous to
    in-sample model selection — the test set has been "seen" even if not directly
    fitted.

Level 2 — QUASI-PROSPECTIVE (future years WITH frozen pipeline):
    A pipeline frozen via ``freeze-pipeline`` BEFORE a tournament starts, then
    evaluated on that tournament's outcomes AFTER they're known.  This is
    stronger than Level 3 because no pipeline modifications can occur between
    prediction and evaluation, but the pipeline's architecture and feature
    set were still informed by all prior years' data, including any structural
    similarities to the evaluation year.

Level 1 — TRUE PROSPECTIVE (pre-registered, independently verified):
    Pipeline frozen, predictions deposited in a public/immutable store (e.g.
    git tag + hash) BEFORE the tournament, verified by a third party or
    cryptographic timestamp.  This is the gold standard; the 2026 tournament
    is the first candidate.

TUNING-EVALUATION CIRCULARITY DISCLOSURE
==========================================
Many Tier 3 constants were originally calibrated using LOYO cross-validation
on the same historical data that the sensitivity analyzer now re-evaluates.
The sensitivity analysis can confirm that the Brier surface is FLAT near the
current value (indicating low overfitting risk), but it cannot prove the values
weren't overfit to begin with — the optimization and validation share the same
data distribution.

The ``circularity_warning`` field in sensitivity results flags constants whose
derivation notes mention "LOYO" or "calibrated", indicating this circularity.

Usage:
    python -m src.main audit-rdof --historical-dir data/raw/historical \\
        --holdout-years 2024,2025 --sensitivity --output rdof_report.json

    python -m src.main freeze-pipeline --output pipeline_freeze.json
    python -m src.main verify-freeze --freeze-file pipeline_freeze.json
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .statistical_tests import paired_brier_test

logger = logging.getLogger(__name__)


# ───────────────────────────────────────────────────────────────────────
# 1. Constant Registry
# ───────────────────────────────────────────────────────────────────────

@dataclass
class PipelineConstant:
    """Metadata for one hand-tuned pipeline constant."""

    name: str
    tier: int  # 1=external, 2=structurally constrained, 3=freely tuned
    current_value: Any
    config_path: str  # e.g. "SOTAPipelineConfig.tournament_shrinkage"
    valid_range: Tuple[float, float]  # search bounds for sensitivity
    derivation: str  # brief provenance note
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "tier": self.tier,
            "current_value": self.current_value,
            "config_path": self.config_path,
            "valid_range": list(self.valid_range),
            "derivation": self.derivation,
            "notes": self.notes,
        }


# fmt: off
CONSTANT_REGISTRY: List[PipelineConstant] = [
    # ── Tier 1: Externally Derived ──────────────────────────────────
    PipelineConstant("four_factors_weights", 1, [0.40, 0.25, 0.20, 0.15],
        "ProprietaryMetrics._four_factors", (0.0, 1.0),
        "Oliver 2004, Kubatko 2007 — published D1 regression coefficients"),
    PipelineConstant("vif_threshold", 1, 10.0,
        "SOTAPipelineConfig.vif_threshold", (5.0, 20.0),
        "Belsley 1980 — standard multicollinearity cutoff"),
    PipelineConstant("hca_points", 1, 3.75,
        "ProprietaryMetrics.HCA_POINTS", (2.5, 5.0),
        "Meta-analysis consensus: college basketball HCA is 3.5-4.0 points"),
    PipelineConstant("elo_hca_multiplier", 1, 13.3,
        "ProprietaryMetrics (derived: HCA_POINTS * 13.3)", (10.0, 16.0),
        "FiveThirtyEight Elo methodology — published conversion factor"),
    PipelineConstant("seed_prior_slope", 1, 0.175,
        "SOTAPipelineConfig.seed_prior_slope", (0.10, 0.25),
        "Logistic fit on 40 years of public tournament data (1985-2024, N>2500)"),
    PipelineConstant("pre_calibration_clips", 1, [0.03, 0.97],
        "SOTAPipelineConfig.pre_calibration_clip_lo/hi", (0.01, 0.10),
        "Historical 1-vs-16 upset rate ~1.5%; 3% provides safety margin"),

    # ── Tier 2: Structurally Constrained ────────────────────────────
    PipelineConstant("training_year_decay", 2, 0.85,
        "SOTAPipelineConfig.training_year_decay", (0.70, 0.95),
        "Bounded (0,1), monotonically decreasing importance of older data"),
    PipelineConstant("training_year_min_weight", 2, 0.15,
        "SOTAPipelineConfig.training_year_min_weight", (0.05, 0.30),
        "Floor > 0 prevents discarding old data entirely"),
    PipelineConstant("recency_half_life", 2, 0.3,
        "SOTAPipelineConfig.recency_half_life", (0.1, 0.8),
        "Within-season temporal weighting; bounded, monotone effect"),
    PipelineConstant("recency_decay_floor", 2, 0.3,
        "SOTAPipelineConfig.recency_decay_floor", (0.1, 0.6),
        "Floor prevents discarding early-season games"),
    PipelineConstant("late_season_cutoff_days", 2, 45,
        "SOTAPipelineConfig.late_season_training_cutoff_days", (20, 90),
        "Bounded, monotone effect on sample size vs feature quality"),
    PipelineConstant("round_correlation_decay", 2, [1.0, 0.6, 0.3, 0.15, 0.0, 0.0],
        "monte_carlo._run_batch (hardcoded)", (0.0, 1.0),
        "Must be monotonically decreasing R64→Championship; tournament structure"),
    PipelineConstant("lgb_num_leaves", 2, 8,
        "LightGBMRanker default params", (4, 32),
        "Lower = more regularized; conservative for small N (~400 samples)"),
    PipelineConstant("lgb_min_child_samples", 2, 50,
        "LightGBMRanker default params", (10, 100),
        "Higher = more regularized; ~12% of training data"),
    PipelineConstant("xgb_max_depth", 2, 3,
        "XGBoostRanker default params", (2, 6),
        "Lower = more regularized; standard shallow-tree choice"),
    PipelineConstant("xgb_min_child_weight", 2, 10,
        "XGBoostRanker default params", (3, 30),
        "Higher = more regularized"),

    # ── Tier 3: Freely Tuned (MUST be cross-validated) ──────────────
    PipelineConstant("ensemble_lgb_weight", 3, 0.45,
        "ensemble weights in _train_baseline_model", (0.20, 0.70),
        "Calibrated from LOYO; 2 free DoF (3 weights sum to 1)",
        notes="Paired with xgb=0.35, logistic=0.20"),
    PipelineConstant("ensemble_xgb_weight", 3, 0.35,
        "ensemble weights in _train_baseline_model", (0.10, 0.60),
        "Calibrated from LOYO; coupled with lgb weight"),
    PipelineConstant("tournament_shrinkage", 3, 0.08,
        "SOTAPipelineConfig.tournament_shrinkage", (0.0, 0.25),
        "Calibrated from LOYO — comment says so explicitly"),
    PipelineConstant("seed_prior_weight", 3, 0.05,
        "SOTAPipelineConfig.seed_prior_weight", (0.0, 0.15),
        "Iteratively adjusted alongside tournament_shrinkage"),
    PipelineConstant("consistency_bonus_max", 3, 0.02,
        "SOTAPipelineConfig.consistency_bonus_max", (0.0, 0.06),
        "Iteratively adjusted; small effect"),
    PipelineConstant("consistency_normalizer", 3, 15.0,
        "SOTAPipelineConfig.consistency_normalizer", (5.0, 30.0),
        "Paired with consistency_bonus_max; normalizes variance range"),
    PipelineConstant("mc_noise_std", 3, 0.12,
        "SimulationConfig.noise_std", (0.02, 0.25),
        "Changed 0.04→0.035→0.02→0.12 across fix rounds"),
    PipelineConstant("mc_regional_correlation", 3, 0.10,
        "SimulationConfig.regional_correlation", (0.0, 0.30),
        "Reduced from 0.25 during OOS fix round"),

    # ── Additional Tier 1: Published/External ─────────────────────────
    PipelineConstant("log5_scale", 1, 0.1735,
        "ProprietaryMetrics._pythagorean_win_pct + rdof_audit baselines", (0.10, 0.25),
        "D1 fix: analytically derived k=ln(17/3)/10=0.1735 so AdjEM=+10 maps to "
        "exactly 0.850 win%. Prior value 0.145 gave 0.810, not 0.85 as documented."),
    PipelineConstant("elo_dampening_numerator", 1, 2.2,
        "ProprietaryMetrics._compute_elo (hardcoded)", (1.5, 3.0),
        "FiveThirtyEight published Elo methodology dampening constant"),
    PipelineConstant("elo_dampening_denom_scale", 1, 0.001,
        "ProprietaryMetrics._compute_elo (hardcoded)", (0.0005, 0.005),
        "FiveThirtyEight published Elo methodology elo_diff scaling"),

    # ── Additional Tier 2: Structurally Constrained ───────────────────
    PipelineConstant("elo_season_regression", 2, 0.25,
        "ProprietaryMetrics._compute_elo_ratings + sota._load_year_samples",
        (0.10, 0.50),
        "D2 fix: 25% regression toward mean (1500) at each season boundary. "
        "Matches FiveThirtyEight NFL Elo methodology (Silver, 2014). "
        "Reduces cold-start noise in early-season ratings."),
    PipelineConstant("elo_k_base", 2, 38.0,
        "ProprietaryMetrics._compute_elo K_BASE", (20.0, 60.0),
        "MOV-adjusted Elo K-factor; bounded, monotone effect on update magnitude"),
    PipelineConstant("wab_k", 2, 11.5,
        "sota.py _load_year_samples _WAB_K", (5.0, 20.0),
        "log5 scaling denominator for WAB; bounded, controls sensitivity to AdjEM gap"),
    PipelineConstant("bubble_em_prior", 2, 5.0,
        "ProprietaryMetrics.BUBBLE_EM_PRIOR", (2.0, 10.0),
        "Historical average AdjEM of ~45th-ranked team; Bayesian prior for WAB"),
    PipelineConstant("margin_cap", 2, 16.0,
        "ProprietaryMetrics.MARGIN_CAP", (10.0, 25.0),
        "Blowout cap; bounded, prevents extreme margins from distorting efficiency"),
    PipelineConstant("sos_damping", 2, 0.7,
        "ProprietaryMetrics._iterative_sos_adjust DAMPING", (0.4, 0.9),
        "SOS convergence blend factor; bounded (0,1), higher = faster convergence"),
    PipelineConstant("three_pt_prior_weight", 2, 100.0,
        "ProprietaryMetrics (hardcoded prior_weight=100)", (30.0, 300.0),
        "Bayesian 3PT shrinkage prior strength; ~3 games of 3PA volume"),
    PipelineConstant("three_pt_league_avg", 2, 0.345,
        "ProprietaryMetrics (hardcoded 0.345)", (0.330, 0.360),
        "D1 average 3PT%; slowly drifts year-to-year but structurally bounded"),
    PipelineConstant("pit_noise_base", 2, 0.05,
        "sota.py _train_baseline_model base_noise", (0.02, 0.10),
        "PIT residual noise scale; bounded, proportional to season_remaining"),
    PipelineConstant("pit_stability_scaling", 2, 0.7,
        "sota.py (feature_noise_weight = 1.0 - stability * 0.7)", (0.4, 0.9),
        "How much feature stability reduces PIT noise; bounded [0,1]"),
    PipelineConstant("pit_feature_noise_reduction", 2, 0.3,
        "sota.py (feature_noise_weight *= 0.3 for PIT features)", (0.1, 0.6),
        "Noise reduction multiplier for PIT-adjusted features; bounded (0,1)"),
    PipelineConstant("mean_regression_shrinkage", 2, 0.10,
        "sota.py (shrinkage = 0.10 toward league mean)", (0.0, 0.25),
        "Regression toward league mean; bounded, monotone effect"),
    PipelineConstant("pit_weight_cap", 2, 0.9,
        "sota.py (min(0.9, 0.9 * season_remaining))", (0.5, 1.0),
        "Maximum PIT blend weight; bounded, caps PIT override of end-of-season"),

    # ── Additional Tier 3: MC Simulation Constants ────────────────────
    PipelineConstant("mc_injury_probability", 3, 0.02,
        "SimulationConfig.injury_probability", (0.0, 0.10),
        "Per-game injury probability; iteratively adjusted"),
    PipelineConstant("mc_injury_severity_lo", 3, 0.05,
        "monte_carlo._run_batch rng.uniform(0.05, 0.25)", (0.01, 0.15),
        "Lower bound of injury severity logit shift; iteratively set"),
    PipelineConstant("mc_injury_severity_hi", 3, 0.25,
        "monte_carlo._run_batch rng.uniform(0.05, 0.25)", (0.10, 0.50),
        "Upper bound of injury severity logit shift; iteratively set"),
    PipelineConstant("mc_lognormal_sigma", 3, 0.15,
        "monte_carlo._run_batch sigma=0.15 for cross-region", (0.05, 0.30),
        "Cross-region (Final Four) noise lognormal sigma; iteratively set"),
    PipelineConstant("mc_region_noise_floor", 3, 0.2,
        "monte_carlo._run_batch max(0.2, ...)", (0.05, 0.5),
        "Floor for regional noise multiplier; prevents zero-variance regions"),

    # ── Previously Unregistered Constants ─────────────────────────────
    # The following constants were identified as hardcoded in the codebase
    # but not tracked in the registry.  Added to close the audit gap.

    # Tier 1: Published formulas
    PipelineConstant("ts_pct_fta_weight", 1, 0.44,
        "ProprietaryMetrics (0.44 * FTA in True Shooting denominator)", (0.40, 0.48),
        "Hollinger's True Shooting %: 2*(FGA + 0.44*FTA); published formula"),

    # Tier 2: Structurally constrained (bounded, monotone, or threshold)
    PipelineConstant("sos_iterations", 2, 15,
        "ProprietaryMetrics._iterative_sos_adjust N_ITERS", (5, 30),
        "Convergence iterations for additive SOS; bounded, monotone convergence"),
    PipelineConstant("luck_min_games", 2, 12,
        "ProprietaryMetrics MIN_GAMES_LUCK", (5, 20),
        "Minimum games for luck metric; data quality threshold"),
    PipelineConstant("luck_full_weight_games", 2, 32,
        "ProprietaryMetrics (shrinkage reaches 1.0 at 32 games)", (20, 50),
        "Games at which luck shrinkage = 100%; structural ramp"),
    PipelineConstant("momentum_window", 2, 8,
        "rdof_audit._compute_derived_stats last_n; sota.py _load_year_samples", (4, 15),
        "Rolling game window for momentum calculation; bounded"),
    PipelineConstant("four_factors_composite_scale", 2, 2.0,
        "ProprietaryMetrics (2.0 * composite to PPP scale)", (1.0, 4.0),
        "Converts Four Factors composite to PPP-like units; structural scaling"),
    PipelineConstant("mc_final_clip_lo", 2, 0.01,
        "monte_carlo._run_batch np.clip(noisy, 0.01, 0.99)", (0.005, 0.05),
        "Post-noise probability floor in MC simulation; prevents log(0)"),
    PipelineConstant("mc_final_clip_hi", 2, 0.99,
        "monte_carlo._run_batch np.clip(noisy, 0.01, 0.99)", (0.95, 0.999),
        "Post-noise probability ceiling in MC simulation; structural bound"),
    PipelineConstant("seed_interaction_scale", 2, 128.0,
        "sota.py seed_interaction = (seed1*seed2)/128 - 1", (64.0, 256.0),
        "Normalization denominator for seed interaction feature; structural"),
    PipelineConstant("gnn_target_scale", 2, 30.0,
        "sota.py adj_em / 30.0 for GNN targets", (15.0, 50.0),
        "AdjEM normalization for GNN training targets; scale parameter"),
    PipelineConstant("early_stopping_rounds", 2, 30,
        "sota.py LGB/XGB early stopping patience", (10, 50),
        "Early stopping rounds; structural regularization, bounded"),
    PipelineConstant("optuna_n_trials", 2, 15,
        "SOTAPipelineConfig.optuna_n_trials", (5, 50),
        "Max Optuna tuning trials; bounded, more trials = more selection bias"),
]
# fmt: on

TIER3_GAME_LEVEL_CONSTANTS = [
    "ensemble_lgb_weight",
    "ensemble_xgb_weight",
    "tournament_shrinkage",
    "seed_prior_weight",
    "consistency_bonus_max",
    "consistency_normalizer",
]

TIER3_MC_CONSTANTS = [
    "mc_noise_std",
    "mc_regional_correlation",
    "mc_injury_probability",
    "mc_injury_severity_lo",
    "mc_injury_severity_hi",
    "mc_lognormal_sigma",
    "mc_region_noise_floor",
]


def get_tier3_constants() -> List[PipelineConstant]:
    """Return all Tier 3 constants from the registry."""
    return [c for c in CONSTANT_REGISTRY if c.tier == 3]


def get_constants_by_tier(tier: int) -> List[PipelineConstant]:
    return [c for c in CONSTANT_REGISTRY if c.tier == tier]


# ───────────────────────────────────────────────────────────────────────
# 2. Config Hashing (audit trail)
# ───────────────────────────────────────────────────────────────────────

def config_hash(config) -> str:
    """SHA-256 hash of all pipeline config fields for audit trail.

    Ensures the pipeline is frozen before holdout evaluation.
    """
    # Serialize all dataclass fields to a canonical JSON string
    d = {}
    for f in config.__dataclass_fields__:
        val = getattr(config, f)
        # Convert non-serializable types
        if isinstance(val, (list, tuple)):
            val = list(val)
        elif val is None or isinstance(val, (int, float, str, bool)):
            pass
        else:
            val = str(val)
        d[f] = val
    canonical = json.dumps(d, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def freeze_pipeline(
    config,
    output_path: str = "pipeline_freeze.json",
) -> Dict[str, Any]:
    """Create a tamper-evident pipeline freeze artifact.

    Captures the full pipeline state so that post-tournament analysis can
    verify the pipeline was locked before seeing results.  Also creates a
    git tag ``pre-registered/{date}/{hash}`` for an immutable reference.

    Returns:
        Dict with freeze artifact contents.
    """
    import subprocess

    cfg_hash = config_hash(config)
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    date_str = time.strftime("%Y-%m-%d")

    # Get git commit SHA
    git_sha = "unknown"
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            git_sha = result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # Check for uncommitted changes
    git_dirty = True
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=5,
        )
        git_dirty = bool(result.stdout.strip())
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # Constant registry snapshot
    registry_snapshot = [c.to_dict() for c in CONSTANT_REGISTRY]

    # Serialize config fields
    config_fields: Dict[str, Any] = {}
    for f in config.__dataclass_fields__:
        val = getattr(config, f)
        if isinstance(val, (list, tuple)):
            val = list(val)
        elif val is None or isinstance(val, (int, float, str, bool)):
            pass
        else:
            val = str(val)
        config_fields[f] = val

    freeze_artifact: Dict[str, Any] = {
        "freeze_type": "pre-registration",
        "config_hash": cfg_hash,
        "git_commit": git_sha,
        "git_dirty": git_dirty,
        "timestamp": timestamp,
        "n_constants": len(CONSTANT_REGISTRY),
        "constant_registry": registry_snapshot,
        "config_fields": config_fields,
    }

    # Write lockfile
    with open(output_path, "w") as f:
        json.dump(freeze_artifact, f, indent=2, sort_keys=True)
    logger.info("Pipeline freeze written to %s", output_path)

    if git_dirty:
        logger.warning(
            "Working tree has uncommitted changes.  The git tag will "
            "reference the LAST COMMITTED state.  Commit first for a "
            "clean freeze."
        )

    # Create git tag
    tag_name = f"pre-registered/{date_str}/{cfg_hash}"
    try:
        subprocess.run(
            ["git", "tag", "-a", tag_name, "-m",
             f"Pre-registered pipeline freeze\n\n"
             f"Config hash: {cfg_hash}\n"
             f"Timestamp: {timestamp}\n"
             f"Constants: {len(CONSTANT_REGISTRY)} registered"],
            capture_output=True, text=True, timeout=10,
        )
        freeze_artifact["git_tag"] = tag_name
        logger.info("Git tag created: %s", tag_name)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        logger.warning("Could not create git tag (git not available?)")

    return freeze_artifact


def verify_freeze(
    config,
    freeze_path: str = "pipeline_freeze.json",
) -> Dict[str, Any]:
    """Verify current pipeline config against a freeze artifact.

    Returns:
        Dict with ``matches`` (bool) and ``mismatches`` (list of strings).
    """
    with open(freeze_path, "r") as f:
        freeze = json.load(f)

    current_hash = config_hash(config)
    frozen_hash = freeze.get("config_hash", "")

    mismatches: List[str] = []

    if current_hash != frozen_hash:
        mismatches.append(
            f"Config hash mismatch: current={current_hash}, frozen={frozen_hash}"
        )
        # Detail which fields changed
        frozen_fields = freeze.get("config_fields", {})
        for field_name in config.__dataclass_fields__:
            current_val = getattr(config, field_name)
            frozen_val = frozen_fields.get(field_name)
            if isinstance(current_val, (list, tuple)):
                current_val = list(current_val)
            if str(current_val) != str(frozen_val):
                mismatches.append(
                    f"  Field '{field_name}': "
                    f"frozen={frozen_val}, current={current_val}"
                )

    # Check constant registry — distinguish behavioral changes (value
    # changed) from documentary additions (new constant registered).
    # Only value changes are mismatches; new registry entries are warnings
    # since they don't change pipeline behavior.
    frozen_constants = {
        c["name"]: c for c in freeze.get("constant_registry", [])
    }
    warnings: List[str] = []
    for c in CONSTANT_REGISTRY:
        frozen_c = frozen_constants.get(c.name)
        if frozen_c is None:
            warnings.append(f"New constant registered since freeze: {c.name}")
        elif str(c.current_value) != str(frozen_c.get("current_value")):
            mismatches.append(
                f"Constant '{c.name}' changed: "
                f"frozen={frozen_c['current_value']}, current={c.current_value}"
            )

    return {
        "matches": len(mismatches) == 0,
        "current_hash": current_hash,
        "frozen_hash": frozen_hash,
        "frozen_timestamp": freeze.get("timestamp"),
        "frozen_git_commit": freeze.get("git_commit"),
        "mismatches": mismatches,
        "warnings": warnings,
    }


# ───────────────────────────────────────────────────────────────────────
# 2b. Effective Model Complexity Audit
# ───────────────────────────────────────────────────────────────────────

@dataclass
class ModelComplexityAudit:
    """Quantifies effective model complexity relative to training sample size.

    The target is: effective_params < 10% of N (training samples).
    This prevents the model from having more free parameters than can
    be reliably estimated from the available data.

    Effective parameters are estimated for each model component:
    - LightGBM: num_leaves * num_trees * (1 - dropout_estimate)
    - XGBoost: (2^max_depth - 1) * num_trees
    - Logistic Regression: n_features + 1 (intercept)
    - GNN embedding projection: 2 * embedding_dim + 1
    - Transformer embedding projection: 2 * embedding_dim + 1
    - Ensemble weights: n_models - 1 (constrained to sum to 1)
    - Tier 3 tuned constants: count from registry
    """

    n_training_samples: int = 0
    component_params: Dict[str, int] = field(default_factory=dict)
    total_effective_params: int = 0
    target_ratio: float = 0.10  # 10% of N
    actual_ratio: float = 0.0
    passed: bool = False
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_training_samples": self.n_training_samples,
            "component_params": self.component_params,
            "total_effective_params": self.total_effective_params,
            "target_ratio": self.target_ratio,
            "actual_ratio": round(self.actual_ratio, 4),
            "passed": self.passed,
            "warnings": self.warnings,
        }


def estimate_model_complexity(
    config=None,
    n_training_samples: int = 400,
    n_features: int = 22,
    gnn_embedding_dim: int = 16,
    transformer_embedding_dim: int = 48,
    target_ratio: float = 0.10,
) -> ModelComplexityAudit:
    """Estimate effective model complexity and check against target ratio.

    Uses conservative estimates of effective parameter counts for each
    model component.  Tree-based models use an effective parameter count
    based on the number of leaf nodes times the number of trees, discounted
    by regularization strength.

    Args:
        config: SOTAPipelineConfig (for hyperparameters). If None, uses defaults.
        n_training_samples: Number of training samples (game-pairs).
        n_features: Number of features after selection.
        gnn_embedding_dim: GNN output dimension.
        transformer_embedding_dim: Transformer d_model dimension.
        target_ratio: Maximum allowed ratio of effective params / N.

    Returns:
        ModelComplexityAudit with breakdown and pass/fail verdict.
    """
    audit = ModelComplexityAudit(
        n_training_samples=n_training_samples,
        target_ratio=target_ratio,
    )

    # LightGBM: effective params ≈ num_leaves * num_trees * regularization_discount
    # Default: 8 leaves, ~200 trees, heavy regularization → ~50% discount
    lgb_leaves = 8
    lgb_trees = 200
    lgb_discount = 0.5  # min_child_samples=50, L2 reg, etc.
    lgb_params = int(lgb_leaves * lgb_trees * lgb_discount)
    audit.component_params["lightgbm"] = lgb_params

    # XGBoost: effective params ≈ (2^max_depth - 1) * num_trees * discount
    xgb_depth = 3
    xgb_trees = 200
    xgb_discount = 0.5  # min_child_weight=10, L2 reg, etc.
    xgb_params = int((2 ** xgb_depth - 1) * xgb_trees * xgb_discount)
    audit.component_params["xgboost"] = xgb_params

    # Logistic Regression: n_features + 1 (with L2 regularization)
    logistic_params = n_features + 1
    audit.component_params["logistic_regression"] = logistic_params

    # GNN embedding projection: logistic on diff + interaction
    # 2 * embedding_dim features + 1 intercept
    gnn_proj_params = 2 * gnn_embedding_dim + 1
    audit.component_params["gnn_projection"] = gnn_proj_params

    # Transformer embedding projection: logistic on diff + interaction
    transformer_proj_params = 2 * transformer_embedding_dim + 1
    audit.component_params["transformer_projection"] = transformer_proj_params

    # Ensemble weights: 2 free parameters (3 weights sum to 1)
    audit.component_params["ensemble_weights"] = 2

    # Tier 3 freely-tuned constants
    n_tier3 = len(get_tier3_constants())
    audit.component_params["tier3_constants"] = n_tier3

    # Total
    audit.total_effective_params = sum(audit.component_params.values())
    audit.actual_ratio = audit.total_effective_params / max(n_training_samples, 1)
    audit.passed = audit.actual_ratio <= target_ratio

    # Warnings for specific components
    if transformer_proj_params > n_training_samples * 0.2:
        audit.warnings.append(
            f"Transformer projection has {transformer_proj_params} params "
            f"({transformer_proj_params / max(n_training_samples, 1):.1%} of N). "
            f"Consider reducing d_model or disabling if ablation fails p<0.05."
        )
    if gnn_proj_params > n_training_samples * 0.1:
        audit.warnings.append(
            f"GNN projection has {gnn_proj_params} params "
            f"({gnn_proj_params / max(n_training_samples, 1):.1%} of N)."
        )
    if not audit.passed:
        audit.warnings.append(
            f"COMPLEXITY VIOLATION: {audit.total_effective_params} effective params "
            f"/ {n_training_samples} samples = {audit.actual_ratio:.1%} "
            f"(target < {target_ratio:.0%}). "
            f"Reduce model complexity or increase training data."
        )

    return audit


# ───────────────────────────────────────────────────────────────────────
# 3. Metrics
# ───────────────────────────────────────────────────────────────────────

@dataclass
class YearMetrics:
    """Evaluation metrics for a single holdout year."""

    year: int
    n_games: int
    brier_score: float
    log_loss: float
    accuracy: float
    ece: float  # Expected Calibration Error

    # Baselines — two comparison tiers:
    # Tier 1: AdjEM-logistic (efficiency margin → logistic probability).
    #   Stronger than seed-based, the best no-ML baseline from historical data.
    seed_baseline_brier: float  # named for interface compat; actually AdjEM-logistic
    # Tier 2: Standalone Elo baseline (Elo rating difference → logistic probability).
    #   A weaker but fully independent baseline that uses only game-by-game
    #   win/loss outcomes with no efficiency data.  Provides a second comparison
    #   tier to ensure the model beats both simple baselines.
    elo_baseline_brier: float = 0.250
    uniform_brier: float = 0.250

    # Confidence interval (bootstrap)
    brier_ci: Tuple[float, float] = (0.0, 0.0)

    def brier_skill_score(self) -> float:
        """1 - (model / seed_baseline). Positive = better than seed."""
        if self.seed_baseline_brier < 1e-9:
            return 0.0
        return 1.0 - self.brier_score / self.seed_baseline_brier

    def elo_skill_score(self) -> float:
        """1 - (model / elo_baseline). Positive = better than Elo."""
        if self.elo_baseline_brier < 1e-9:
            return 0.0
        return 1.0 - self.brier_score / self.elo_baseline_brier

    def to_dict(self) -> Dict[str, Any]:
        return {
            "year": self.year,
            "n_games": self.n_games,
            "brier_score": round(self.brier_score, 5),
            "log_loss": round(self.log_loss, 5),
            "accuracy": round(self.accuracy, 4),
            "ece": round(self.ece, 4),
            "seed_baseline_brier": round(self.seed_baseline_brier, 5),
            "elo_baseline_brier": round(self.elo_baseline_brier, 5),
            "uniform_brier": self.uniform_brier,
            "brier_skill_score": round(self.brier_skill_score(), 4),
            "elo_skill_score": round(self.elo_skill_score(), 4),
            "brier_ci": [round(x, 5) for x in self.brier_ci],
        }


@dataclass
class HoldoutReport:
    """Results from the full holdout evaluation.

    The ``integrity_level`` field explicitly communicates the evidential
    strength of these results (see module docstring for definitions).
    """

    holdout_years: List[int]
    per_year: Dict[int, YearMetrics] = field(default_factory=dict)
    config_hash_value: str = ""
    timestamp: str = ""
    integrity_level: int = 3  # 1=true prospective, 2=quasi-prospective, 3=retrospective
    integrity_note: str = (
        "RETROSPECTIVE DIAGNOSTIC: These holdout years were observed during "
        "pipeline development. Results overstate true OOS performance."
    )

    @property
    def aggregate_brier(self) -> float:
        if not self.per_year:
            return 0.0
        briers = [m.brier_score for m in self.per_year.values()]
        weights = [m.n_games for m in self.per_year.values()]
        return float(np.average(briers, weights=weights))

    @property
    def aggregate_seed_brier(self) -> float:
        if not self.per_year:
            return 0.0
        briers = [m.seed_baseline_brier for m in self.per_year.values()]
        weights = [m.n_games for m in self.per_year.values()]
        return float(np.average(briers, weights=weights))

    @property
    def aggregate_elo_brier(self) -> float:
        if not self.per_year:
            return 0.0
        briers = [m.elo_baseline_brier for m in self.per_year.values()]
        weights = [m.n_games for m in self.per_year.values()]
        return float(np.average(briers, weights=weights))

    @property
    def total_games(self) -> int:
        return sum(m.n_games for m in self.per_year.values())

    def verdict(self) -> str:
        delta = self.aggregate_seed_brier - self.aggregate_brier
        if delta > 0.005:
            return "PASS"
        elif delta > 0:
            return "WARN"
        else:
            return "FAIL"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "holdout_years": self.holdout_years,
            "config_hash": self.config_hash_value,
            "timestamp": self.timestamp,
            "integrity_level": self.integrity_level,
            "integrity_note": self.integrity_note,
            "total_games": self.total_games,
            "aggregate_brier": round(self.aggregate_brier, 5),
            "aggregate_seed_brier": round(self.aggregate_seed_brier, 5),
            "aggregate_elo_brier": round(self.aggregate_elo_brier, 5),
            "aggregate_brier_skill_score": round(
                1.0 - self.aggregate_brier / max(self.aggregate_seed_brier, 1e-9), 4
            ),
            "aggregate_elo_skill_score": round(
                1.0 - self.aggregate_brier / max(self.aggregate_elo_brier, 1e-9), 4
            ),
            "verdict": self.verdict(),
            "per_year": {
                yr: m.to_dict() for yr, m in sorted(self.per_year.items())
            },
        }


@dataclass
class ConstantSensitivityResult:
    """Sensitivity analysis for one Tier 3 constant."""

    constant_name: str
    grid_values: List[float]
    loyo_brier_scores: List[float]  # mean LOYO Brier at each grid point
    current_value: float
    optimal_value: float
    optimal_brier: float
    current_brier: float
    brier_range: float  # max - min across grid
    is_flat: bool  # brier_range < 0.005
    circularity_warning: bool = False  # True if constant was originally tuned via LOYO

    @property
    def brier_gap(self) -> float:
        return self.current_brier - self.optimal_brier

    @property
    def rank_of_current(self) -> int:
        sorted_vals = sorted(range(len(self.loyo_brier_scores)),
                             key=lambda i: self.loyo_brier_scores[i])
        for rank, idx in enumerate(sorted_vals):
            if abs(self.grid_values[idx] - self.current_value) < 1e-9:
                return rank + 1
        return len(self.grid_values)

    @property
    def rdof_risk_category(self) -> str:
        """Categorize this constant's RDoF risk.

        - 'flat_plateau': Brier range < 0.005 -- constant has negligible
          effect on OOS performance.  Consumes ~0 effective DoF.
        - 'mild_slope': Range 0.005-0.015, current near optimal -- constant
          has modest effect but is well-chosen.  Consumes ~0.5 effective DoF.
        - 'sharp_peak': Range > 0.015 or current far from optimal -- constant
          materially affects performance AND the specific value matters.
          Consumes ~1.0 effective DoF (a true free parameter).
        """
        if self.is_flat:
            return "flat_plateau"
        if self.brier_range < 0.015 and self.brier_gap < 0.003:
            return "mild_slope"
        return "sharp_peak"

    @property
    def effective_dof(self) -> float:
        """Approximate effective degrees of freedom consumed by this constant.

        Flat plateaus consume ~0 DoF (any value works equally well).
        Mild slopes consume ~0.5 DoF (some sensitivity, but current is close).
        Sharp peaks consume ~1.0 DoF (a genuine free parameter).

        This is a heuristic, not a formal statistical quantity.
        """
        cat = self.rdof_risk_category
        if cat == "flat_plateau":
            return 0.0
        elif cat == "mild_slope":
            return 0.5
        return 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "constant_name": self.constant_name,
            "grid_values": [round(v, 4) for v in self.grid_values],
            "loyo_brier_scores": [round(b, 5) for b in self.loyo_brier_scores],
            "current_value": round(self.current_value, 4),
            "optimal_value": round(self.optimal_value, 4),
            "optimal_brier": round(self.optimal_brier, 5),
            "current_brier": round(self.current_brier, 5),
            "brier_gap": round(self.brier_gap, 5),
            "brier_range": round(self.brier_range, 5),
            "is_flat": self.is_flat,
            "rank_of_current": self.rank_of_current,
            "rdof_risk_category": self.rdof_risk_category,
            "effective_dof": self.effective_dof,
            "circularity_warning": self.circularity_warning,
        }


# ───────────────────────────────────────────────────────────────────────
# 4. Metric computation helpers
# ───────────────────────────────────────────────────────────────────────

def _compute_brier(probs: np.ndarray, outcomes: np.ndarray) -> float:
    return float(np.mean((probs - outcomes) ** 2))


def _compute_log_loss(probs: np.ndarray, outcomes: np.ndarray) -> float:
    p = np.clip(probs, 1e-7, 1 - 1e-7)
    return float(-np.mean(outcomes * np.log(p) + (1 - outcomes) * np.log(1 - p)))


def _compute_accuracy(probs: np.ndarray, outcomes: np.ndarray) -> float:
    preds = (probs >= 0.5).astype(float)
    return float(np.mean(preds == outcomes))


def _compute_ece(probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error with equal-width bins."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
        if i == n_bins - 1:
            mask = mask | (probs == bin_edges[i + 1])
        count = mask.sum()
        if count == 0:
            continue
        avg_pred = probs[mask].mean()
        avg_outcome = outcomes[mask].mean()
        ece += (count / len(probs)) * abs(avg_pred - avg_outcome)
    return float(ece)


def _seed_baseline_prob(seed1: int, seed2: int, slope: float = 0.175) -> float:
    """Seed-only baseline: sigmoid(slope * (seed2 - seed1))."""
    diff = seed2 - seed1
    return 1.0 / (1.0 + math.exp(-slope * diff))


def _bootstrap_brier_ci(
    probs: np.ndarray,
    outcomes: np.ndarray,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
) -> Tuple[float, float]:
    """Bootstrap confidence interval for Brier score."""
    rng = np.random.RandomState(42)
    n = len(probs)
    briers = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        briers.append(float(np.mean((probs[idx] - outcomes[idx]) ** 2)))
    alpha = (1 - ci_level) / 2
    lo = float(np.percentile(briers, 100 * alpha))
    hi = float(np.percentile(briers, 100 * (1 - alpha)))
    return (lo, hi)


# ───────────────────────────────────────────────────────────────────────
# 5. Holdout Evaluator
# ───────────────────────────────────────────────────────────────────────

class HoldoutEvaluator:
    """Runs the pipeline on designated holdout tournament years.

    CAVEAT: These years were observed during pipeline development.  This is a
    retrospective holdout (useful diagnostic), not a prospective pre-registered
    holdout.  The pipeline's feature set, architecture, and hyperparameter
    bounds were iteratively refined with knowledge of all 2005-2025 data.
    See module docstring for full disclosure.

    For each holdout year:
    1. Load training data from all other years (via _load_year_samples)
    2. Train LGB + XGB + Logistic ensemble
    3. Load holdout year's tournament games
    4. Predict each tournament game, apply tournament adaptation
    5. Score against actual outcomes
    """

    def __init__(
        self,
        historical_dir: str = "data/raw/historical",
        all_years: Optional[List[int]] = None,
    ):
        self.historical_dir = Path(historical_dir)
        if all_years is None:
            # Auto-detect available years
            self.all_years = sorted(self._detect_years())
        else:
            self.all_years = sorted(all_years)

    def _detect_years(self) -> List[int]:
        """Find years with both games and metrics files."""
        years = []
        for p in self.historical_dir.glob("historical_games_*.json"):
            try:
                year = int(p.stem.split("_")[-1])
                metrics_path = self.historical_dir / f"team_metrics_{year}.json"
                if metrics_path.exists() and year != 2020:  # Exclude COVID year
                    years.append(year)
            except ValueError:
                continue
        return years

    def _load_year_data(self, year: int):
        """Load games and metrics for a single year."""
        games_path = self.historical_dir / f"historical_games_{year}.json"
        metrics_path = self.historical_dir / f"team_metrics_{year}.json"

        with open(games_path, "r") as f:
            games_payload = json.load(f)
        with open(metrics_path, "r") as f:
            metrics_payload = json.load(f)

        return games_payload, metrics_payload

    def _build_team_metrics(self, metrics_payload: dict) -> Dict[str, Dict[str, float]]:
        """Parse team metrics from payload."""
        team_metrics: Dict[str, Dict[str, float]] = {}
        teams_list = metrics_payload.get("teams", [])

        if isinstance(teams_list, list) and teams_list:
            off_vals = [float(tm.get("off_rtg", 0)) for tm in teams_list
                        if isinstance(tm, dict)]
            if off_vals and all(abs(v) < 1e-6 for v in off_vals):
                return {}
            unique_off = set(round(v, 4) for v in off_vals)
            if len(unique_off) <= 1:
                return {}

        if isinstance(teams_list, list):
            for tm in teams_list:
                tid = str(tm.get("team_id") or tm.get("name", "")).lower().strip()
                tid = tid.replace(" ", "_").replace("'", "").replace(".", "")
                if not tid:
                    continue
                off = float(tm.get("off_rtg", 0))
                drt = float(tm.get("def_rtg", 0))
                if off < 1e-6 or drt < 1e-6:
                    continue
                team_metrics[tid] = {
                    "off_rtg": off,
                    "def_rtg": drt,
                    "pace": float(tm.get("pace", 68.0)),
                    "srs": float(tm.get("srs", 0.0)),
                    "sos": float(tm.get("sos", 0.0)),
                    "wins": float(tm.get("wins", 15)),
                    "losses": float(tm.get("losses", 15)),
                }

        return team_metrics

    def _infer_dates_and_split(
        self, games: list, year: int
    ) -> Tuple[list, list]:
        """Infer dates if needed, return (regular_season, tournament) games.

        Returns lists of (t1, t2, s1, s2, adj_em1, adj_em2) tuples.
        """
        # Chronological date inference for single-date payloads
        raw_dates = [str(g.get("date", g.get("game_date", ""))) for g in games]
        unique_dates = set(d for d in raw_dates if d)
        if len(unique_dates) <= 1 and len(games) > 50:
            id_ordered = sorted(
                range(len(games)),
                key=lambda i: int(games[i].get("game_id", "0"))
                if str(games[i].get("game_id", "0")).isdigit() else 0,
            )
            season_start = date(year - 1, 11, 1)
            season_end = date(year, 4, 10)
            total_days = (season_end - season_start).days
            for rank, orig_idx in enumerate(id_ordered):
                frac = rank / max(len(id_ordered) - 1, 1)
                inferred = season_start + timedelta(days=int(frac * total_days))
                games[orig_idx]["date"] = inferred.isoformat()

        return games

    def _is_tournament_game(self, date_str: str, year: int) -> bool:
        """Detect NCAA Tournament games (mid-March through mid-April)."""
        try:
            parts = date_str.split("-")
            game_year = int(parts[0])
            month = int(parts[1])
            day = int(parts[2])
            game_day = date(game_year, month, day)
            # Tournament runs ~March 14 to April 15
            start = date(year, 3, 14)
            end = date(year, 4, 15)
            return start <= game_day <= end
        except (ValueError, IndexError):
            return False

    def _build_feature_vector(
        self,
        t1_metrics: Dict[str, float],
        t2_metrics: Dict[str, float],
        t1_derived: Dict[str, float],
        t2_derived: Dict[str, float],
        feature_dim: int = 77,
    ) -> np.ndarray:
        """Build differential feature vector from team metrics.

        Uses the same feature positions as _load_year_samples in sota.py.
        """
        vec = np.zeros(feature_dim, dtype=np.float64)

        # Differential features
        vec[0] = t1_metrics["off_rtg"] - t2_metrics["off_rtg"]
        vec[1] = t1_metrics["def_rtg"] - t2_metrics["def_rtg"]
        vec[2] = t1_metrics["pace"] - t2_metrics["pace"]
        vec[26] = t1_metrics["sos"] - t2_metrics["sos"]
        vec[47] = (
            t1_metrics["wins"] / max(t1_metrics["wins"] + t1_metrics["losses"], 1)
            - t2_metrics["wins"] / max(t2_metrics["wins"] + t2_metrics["losses"], 1)
        )
        vec[30] = t1_derived.get("luck", 0) - t2_derived.get("luck", 0)
        vec[31] = t1_derived.get("wab", 0) - t2_derived.get("wab", 0)
        vec[32] = t1_derived.get("momentum", 0) - t2_derived.get("momentum", 0)
        vec[33] = t1_derived.get("margin_std", 0) - t2_derived.get("margin_std", 0)
        vec[35] = t1_derived.get("elo", 1500) - t2_derived.get("elo", 1500)

        # Absolute features
        vec[66] = (t1_metrics["off_rtg"] + t2_metrics["off_rtg"]) / 2.0
        vec[67] = (t1_metrics["def_rtg"] + t2_metrics["def_rtg"]) / 2.0
        vec[68] = (t1_metrics["sos"] + t2_metrics["sos"]) / 2.0
        vec[69] = (t1_derived.get("elo", 1500) + t2_derived.get("elo", 1500)) / 2.0
        wp1 = t1_metrics["wins"] / max(t1_metrics["wins"] + t1_metrics["losses"], 1)
        wp2 = t2_metrics["wins"] / max(t2_metrics["wins"] + t2_metrics["losses"], 1)
        vec[70] = (wp1 + wp2) / 2.0

        return vec

    def _compute_derived_stats(
        self,
        games: list,
        team_metrics: Dict[str, Dict[str, float]],
        year: int,
    ) -> Dict[str, Dict[str, float]]:
        """Compute Elo, luck, WAB, momentum from game-by-game results.

        Mirrors the logic in sota.py _load_year_samples.
        """
        from scipy import stats as scipy_stats

        _K_BASE = 38.0
        _BUBBLE_EM = 5.0
        _WAB_K = 11.5

        # Build prefix resolver
        metric_keys = sorted(team_metrics.keys(), key=len, reverse=True)
        _prefix_cache: Dict[str, str] = {}

        def _resolve(game_id: str) -> Optional[str]:
            if game_id in team_metrics:
                return game_id
            if game_id in _prefix_cache:
                return _prefix_cache[game_id]
            for mk in metric_keys:
                if game_id.startswith(mk + "_") or game_id.startswith(mk):
                    _prefix_cache[game_id] = mk
                    return mk
            return None

        elo: Dict[str, float] = {t: 1500.0 for t in team_metrics}
        margins_by_team: Dict[str, list] = {t: [] for t in team_metrics}
        results_by_team: Dict[str, list] = {t: [] for t in team_metrics}

        # Sort games chronologically
        sorted_games = sorted(games, key=lambda g: g.get("date", g.get("game_date", "")))

        for game in sorted_games:
            raw_t1 = str(game.get("team1_id") or game.get("team1") or "").lower().strip()
            raw_t2 = str(game.get("team2_id") or game.get("team2") or "").lower().strip()
            raw_t1 = raw_t1.replace(" ", "_").replace("'", "").replace(".", "")
            raw_t2 = raw_t2.replace(" ", "_").replace("'", "").replace(".", "")
            s1 = int(game.get("team1_score", 0))
            s2 = int(game.get("team2_score", 0))

            t1 = _resolve(raw_t1) if raw_t1 else None
            t2 = _resolve(raw_t2) if raw_t2 else None
            if not t1 or not t2 or s1 == 0 or s2 == 0:
                continue
            if t1 not in team_metrics or t2 not in team_metrics:
                continue

            margin = s1 - s2
            t1_won = margin > 0

            # Elo update
            e1 = 1.0 / (1.0 + 10.0 ** (-(elo.get(t1, 1500) - elo.get(t2, 1500)) / 400.0))
            s1_elo = 1.0 if t1_won else (0.0 if margin < 0 else 0.5)
            mov_mult = math.log1p(abs(margin))
            elo_diff = abs(elo.get(t1, 1500) - elo.get(t2, 1500))
            elo_dampening = 2.2 / (elo_diff * 0.001 + 2.2)
            k = _K_BASE * mov_mult * elo_dampening
            delta = k * (s1_elo - e1)
            elo[t1] = elo.get(t1, 1500) + delta
            elo[t2] = elo.get(t2, 1500) - delta

            margins_by_team.setdefault(t1, []).append(margin)
            margins_by_team.setdefault(t2, []).append(-margin)

            # WAB accumulators
            em2 = team_metrics[t2]["off_rtg"] - team_metrics[t2]["def_rtg"]
            em1 = team_metrics[t1]["off_rtg"] - team_metrics[t1]["def_rtg"]

            def _log5(a_em, b_em):
                diff = float(max(-40.0, min(40.0, a_em - b_em)))
                return 1.0 / (1.0 + 10.0 ** (-diff / _WAB_K))

            results_by_team.setdefault(t1, []).append((_log5(_BUBBLE_EM, em2), t1_won))
            results_by_team.setdefault(t2, []).append((_log5(_BUBBLE_EM, em1), not t1_won))

        # Compute final derived stats
        team_derived: Dict[str, Dict[str, float]] = {}
        for t in team_metrics:
            margins = margins_by_team.get(t, [])
            res = results_by_team.get(t, [])
            n_games = len(margins)

            luck = 0.0
            if n_games >= 12:
                mean_m = float(np.mean(margins))
                std_m = float(np.std(margins, ddof=1)) if n_games > 1 else 1.0
                if std_m > 0.1:
                    z = mean_m / std_m
                    expected_wp = float(scipy_stats.norm.cdf(z))
                    actual_wp = sum(1 for m in margins if m > 0) / n_games
                    raw_luck = actual_wp - expected_wp
                    shrinkage = min(1.0, (n_games - 12) / 20.0)
                    luck = raw_luck * shrinkage

            wab = sum(
                (1.0 - bwp) if won else (0.0 - bwp)
                for bwp, won in res
            )

            momentum = 0.0
            if n_games >= 4:
                last_n = min(8, n_games)
                last_margins = margins[-last_n:]
                momentum = sum(1.0 for m in last_margins if m > 0) / last_n - 0.5

            margin_std = float(np.std(margins, ddof=1)) if n_games > 1 else 0.0

            team_derived[t] = {
                "elo": elo.get(t, 1500.0),
                "luck": luck,
                "wab": wab,
                "momentum": momentum,
                "margin_std": margin_std,
            }

        return team_derived

    def _extract_tournament_games(
        self,
        games: list,
        team_metrics: Dict[str, Dict[str, float]],
        year: int,
    ) -> list:
        """Extract tournament games with resolved team IDs."""
        metric_keys = sorted(team_metrics.keys(), key=len, reverse=True)
        _cache: Dict[str, str] = {}

        def _resolve(gid: str) -> Optional[str]:
            if gid in team_metrics:
                return gid
            if gid in _cache:
                return _cache[gid]
            for mk in metric_keys:
                if gid.startswith(mk + "_") or gid.startswith(mk):
                    _cache[gid] = mk
                    return mk
            return None

        tourney = []
        for game in games:
            date_str = str(game.get("date", game.get("game_date", "")))
            if not self._is_tournament_game(date_str, year):
                continue

            raw_t1 = str(game.get("team1_id", "")).lower().strip()
            raw_t2 = str(game.get("team2_id", "")).lower().strip()
            raw_t1 = raw_t1.replace(" ", "_").replace("'", "").replace(".", "")
            raw_t2 = raw_t2.replace(" ", "_").replace("'", "").replace(".", "")

            t1 = _resolve(raw_t1)
            t2 = _resolve(raw_t2)

            s1 = int(game.get("team1_score", 0))
            s2 = int(game.get("team2_score", 0))

            if t1 and t2 and s1 > 0 and s2 > 0:
                tourney.append({
                    "team1_id": t1,
                    "team2_id": t2,
                    "team1_score": s1,
                    "team2_score": s2,
                    "outcome": 1 if s1 > s2 else 0,
                })

        return tourney

    def _train_for_year(
        self,
        holdout_year: int,
        config,
        feature_dim: int = 77,
    ) -> dict:
        """Train models for one holdout year; return cached artifacts.

        Returns a dict with keys: lgb_model, xgb_model, logistic, scaler,
        lgb_trained, xgb_trained, holdout_team_metrics, holdout_derived,
        tournament_games, per_game_raw_preds (per-model probs before ensemble).

        Separating training from post-hoc parameter application lets the
        sensitivity analyzer train ONCE per holdout year, then sweep
        post-training constants (shrinkage, weights, etc.) cheaply.
        """
        training_years = [y for y in self.all_years
                          if y != holdout_year and y != 2020]
        if not training_years:
            raise ValueError(f"No training years available for holdout {holdout_year}")

        logger.info("Holdout %d: training on %d years: %s",
                     holdout_year, len(training_years), training_years)

        # ── 1. Collect training data ─────────────────────────────────
        all_train_X, all_train_y, all_train_w = [], [], []

        for train_year in training_years:
            try:
                games_payload, metrics_payload = self._load_year_data(train_year)
            except FileNotFoundError:
                logger.warning("Missing data for year %d, skipping", train_year)
                continue

            team_metrics = self._build_team_metrics(metrics_payload)
            if not team_metrics:
                continue

            games = games_payload.get("games", [])
            games = self._infer_dates_and_split(games, train_year)
            team_derived = self._compute_derived_stats(games, team_metrics, train_year)

            metric_keys = sorted(team_metrics.keys(), key=len, reverse=True)
            _cache: Dict[str, str] = {}

            def _resolve(gid, mk_list=metric_keys, c=_cache, tm=team_metrics):
                if gid in tm:
                    return gid
                if gid in c:
                    return c[gid]
                for mk in mk_list:
                    if gid.startswith(mk + "_") or gid.startswith(mk):
                        c[gid] = mk
                        return mk
                return None

            years_ago = max(1, holdout_year - train_year)
            year_weight = max(
                config.training_year_min_weight,
                config.training_year_decay ** (years_ago - 1),
            )

            for game in games:
                date_str = str(game.get("date", game.get("game_date", "")))
                if self._is_tournament_game(date_str, train_year):
                    continue

                raw_t1 = str(game.get("team1_id", "")).lower().strip()
                raw_t2 = str(game.get("team2_id", "")).lower().strip()
                raw_t1 = raw_t1.replace(" ", "_").replace("'", "").replace(".", "")
                raw_t2 = raw_t2.replace(" ", "_").replace("'", "").replace(".", "")
                t1 = _resolve(raw_t1)
                t2 = _resolve(raw_t2)
                s1 = int(game.get("team1_score", 0))
                s2 = int(game.get("team2_score", 0))

                if not t1 or not t2 or s1 == 0 or s2 == 0:
                    continue
                if t1 not in team_metrics or t2 not in team_metrics:
                    continue

                d1 = team_derived.get(t1, {})
                d2 = team_derived.get(t2, {})
                vec = self._build_feature_vector(
                    team_metrics[t1], team_metrics[t2], d1, d2, feature_dim
                )
                all_train_X.append(vec)
                all_train_y.append(1 if s1 > s2 else 0)
                all_train_w.append(year_weight)

        if not all_train_X:
            raise ValueError(f"No training data for holdout {holdout_year}")

        train_X = np.array(all_train_X, dtype=np.float64)
        train_y = np.array(all_train_y, dtype=np.float64)
        train_w = np.array(all_train_w, dtype=np.float64)

        logger.info("Holdout %d: %d training samples", holdout_year, len(train_X))

        # ── 2. Train ensemble ────────────────────────────────────────
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression

        scaler = StandardScaler()
        train_X_scaled = scaler.fit_transform(train_X)

        lgb_model, xgb_model = None, None
        lgb_trained = xgb_trained = False

        try:
            from ..ensemble.cfa import LightGBMRanker
            lgb_model = LightGBMRanker()
            lgb_model.train(train_X, train_y, num_rounds=200,
                            early_stopping_rounds=None, sample_weight=train_w)
            lgb_trained = True
        except Exception as e:
            logger.warning("LightGBM training failed: %s", e)

        try:
            from ..ensemble.cfa import XGBoostRanker
            xgb_model = XGBoostRanker()
            xgb_model.train(train_X, train_y, num_rounds=200,
                            early_stopping_rounds=None, sample_weight=train_w)
            xgb_trained = True
        except Exception as e:
            logger.warning("XGBoost training failed: %s", e)

        logistic = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
        logistic.fit(train_X_scaled, train_y, sample_weight=train_w)

        # ── 3. Load holdout tournament games and get per-model preds ─
        try:
            ho_games_payload, ho_metrics_payload = self._load_year_data(holdout_year)
        except FileNotFoundError:
            raise ValueError(f"Missing data for holdout year {holdout_year}")

        ho_team_metrics = self._build_team_metrics(ho_metrics_payload)
        if not ho_team_metrics:
            raise ValueError(f"No valid team metrics for holdout year {holdout_year}")

        ho_games = ho_games_payload.get("games", [])
        ho_games = self._infer_dates_and_split(ho_games, holdout_year)
        ho_derived = self._compute_derived_stats(ho_games, ho_team_metrics, holdout_year)

        tournament_games = self._extract_tournament_games(
            ho_games, ho_team_metrics, holdout_year
        )
        if not tournament_games:
            raise ValueError(f"No tournament games for holdout year {holdout_year}")

        logger.info("Holdout %d: %d tournament games", holdout_year, len(tournament_games))

        # Get per-model raw predictions for each tournament game
        per_game: list = []  # list of dicts with lgb_p, xgb_p, log_p, em_diff, margin_std_diff, outcome
        for tg in tournament_games:
            t1, t2 = tg["team1_id"], tg["team2_id"]
            if t1 not in ho_team_metrics or t2 not in ho_team_metrics:
                continue

            d1 = ho_derived.get(t1, {})
            d2 = ho_derived.get(t2, {})
            vec = self._build_feature_vector(
                ho_team_metrics[t1], ho_team_metrics[t2], d1, d2, feature_dim
            )
            vec_scaled = scaler.transform(vec.reshape(1, -1))

            lgb_p = float(lgb_model.predict(vec.reshape(1, -1))[0]) if lgb_trained else 0.5
            xgb_p = float(xgb_model.predict(vec.reshape(1, -1))[0]) if xgb_trained else 0.5
            log_p = float(logistic.predict_proba(vec_scaled)[0, 1])

            # AdjEM difference (for efficiency-based baseline & seed proxy)
            em1 = ho_team_metrics[t1]["off_rtg"] - ho_team_metrics[t1]["def_rtg"]
            em2 = ho_team_metrics[t2]["off_rtg"] - ho_team_metrics[t2]["def_rtg"]

            # Margin std difference (for consistency bonus)
            mstd1 = d1.get("margin_std", 0.0)
            mstd2 = d2.get("margin_std", 0.0)

            # Elo difference (for standalone Elo baseline — second comparison tier)
            elo1 = d1.get("elo", 1500.0)
            elo2 = d2.get("elo", 1500.0)

            per_game.append({
                "lgb_p": lgb_p,
                "xgb_p": xgb_p,
                "log_p": log_p,
                "em_diff": em1 - em2,
                "elo_diff": elo1 - elo2,
                "margin_std_diff": mstd2 - mstd1,  # positive = team1 more consistent
                "outcome": tg["outcome"],
            })

        # ── 4. Calibration: fit temperature scaling on training preds ─
        # Use out-of-bag training predictions for calibration fitting
        # (mirrors production pipeline's approach).
        train_preds_lgb = lgb_model.predict(train_X) if lgb_trained else np.full(len(train_X), 0.5)
        train_preds_xgb = xgb_model.predict(train_X) if xgb_trained else np.full(len(train_X), 0.5)
        train_preds_log = logistic.predict_proba(train_X_scaled)[:, 1]

        return {
            "per_game": per_game,
            "train_ensemble_preds": (train_preds_lgb, train_preds_xgb, train_preds_log),
            "train_y": train_y,
            "holdout_year": holdout_year,
        }

    def _apply_posthoc_and_score(
        self,
        cached: dict,
        config,
    ) -> YearMetrics:
        """Apply post-training parameters to cached predictions and score.

        Post-training parameters: ensemble weights, tournament shrinkage,
        seed prior weight, consistency bonus, calibration temperature.
        These can be swept without retraining.
        """
        from ..calibration.calibration import TemperatureScaling

        per_game = cached["per_game"]
        train_lgb, train_xgb, train_log = cached["train_ensemble_preds"]
        train_y = cached["train_y"]
        holdout_year = cached["holdout_year"]

        w_lgb = config.ensemble_lgb_weight
        w_xgb = config.ensemble_xgb_weight
        w_log = 1.0 - w_lgb - w_xgb

        # Fit calibration on training ensemble predictions
        train_ensemble = np.clip(
            w_lgb * train_lgb + w_xgb * train_xgb + w_log * train_log,
            0.03, 0.97,
        )
        calibrator = TemperatureScaling()
        try:
            calibrator.fit(train_ensemble, train_y)
        except Exception:
            pass  # calibrator stays at T=1.0 (identity)

        probs, outcomes, baseline_probs, elo_baseline_probs = [], [], [], []

        for g in per_game:
            raw = w_lgb * g["lgb_p"] + w_xgb * g["xgb_p"] + w_log * g["log_p"]
            raw = float(np.clip(raw, 0.03, 0.97))

            # Calibration
            if calibrator.fitted:
                raw = float(calibrator.calibrate(np.array([raw]))[0])

            # Tournament adaptation (full version using available data)
            adapted = self._tournament_adapt(
                raw, g["em_diff"], g["margin_std_diff"], config
            )

            probs.append(adapted)
            outcomes.append(g["outcome"])

            # AdjEM-quality baseline: efficiency margin → logistic probability
            # This is stronger than a pure seed baseline but is the best
            # no-ML baseline derivable from the historical data available.
            # Scale factor 0.1735 = ln(17/3)/10 (D1 fix: AdjEM=+10 → P=0.850)
            baseline_prob = 1.0 / (1.0 + math.exp(-0.1735 * g["em_diff"]))
            baseline_probs.append(baseline_prob)

            # Tier 2 baseline: Standalone Elo (Elo rating diff → logistic)
            # Uses only game-by-game win/loss outcomes with no efficiency data.
            # Standard Elo conversion: P(win) = 1 / (1 + 10^(-diff/400))
            elo_diff = g.get("elo_diff", 0.0)
            elo_prob = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))
            elo_baseline_probs.append(elo_prob)

        if not probs:
            raise ValueError(f"No valid predictions for holdout year {holdout_year}")

        probs_arr = np.array(probs)
        outcomes_arr = np.array(outcomes)
        baseline_arr = np.array(baseline_probs)
        elo_arr = np.array(elo_baseline_probs)

        return YearMetrics(
            year=holdout_year,
            n_games=len(probs),
            brier_score=_compute_brier(probs_arr, outcomes_arr),
            log_loss=_compute_log_loss(probs_arr, outcomes_arr),
            accuracy=_compute_accuracy(probs_arr, outcomes_arr),
            ece=_compute_ece(probs_arr, outcomes_arr),
            seed_baseline_brier=_compute_brier(baseline_arr, outcomes_arr),
            elo_baseline_brier=_compute_brier(elo_arr, outcomes_arr),
            brier_ci=_bootstrap_brier_ci(probs_arr, outcomes_arr),
        )

    def evaluate_holdout_year(
        self,
        holdout_year: int,
        config,
        feature_dim: int = 77,
    ) -> YearMetrics:
        """Full holdout evaluation: train + score with all post-hoc params."""
        cached = self._train_for_year(holdout_year, config, feature_dim)
        return self._apply_posthoc_and_score(cached, config)

    @staticmethod
    def _tournament_adapt(
        prob: float,
        em_diff: float,
        margin_std_diff: float,
        config,
    ) -> float:
        """Apply tournament domain adaptation matching sota.py logic.

        Uses AdjEM difference as a seed proxy (em_diff → approximate seed
        quality gap) and margin std difference for consistency bonus.
        Historical JSON files lack actual tournament seeds, but AdjEM is
        more informative than seed anyway — this tests the same adaptation
        LOGIC with the best available data.
        """
        # 1. Shrinkage toward 0.5
        shrinkage = config.tournament_shrinkage
        adapted = shrinkage * 0.5 + (1.0 - shrinkage) * prob

        # 2. Seed prior: use AdjEM → logistic as seed-quality proxy
        # em_diff > 0 means team1 is stronger (analogous to lower seed)
        slope = config.seed_prior_slope
        seed_prior = 1.0 / (1.0 + math.exp(-slope * em_diff / 2.5))
        # /2.5 rescales AdjEM to be roughly on the same scale as seed_diff:
        # a 10-point AdjEM gap ≈ a 4-seed gap (e.g. 1 vs 5 seed)
        w = config.seed_prior_weight
        adapted = (1.0 - w) * adapted + w * seed_prior

        # 3. Consistency bonus
        bonus_max = config.consistency_bonus_max
        normalizer = config.consistency_normalizer
        consistency_edge = bonus_max * float(np.clip(
            margin_std_diff / normalizer, -1.0, 1.0
        ))
        adapted += consistency_edge

        return float(np.clip(adapted, config.pre_calibration_clip_lo,
                             config.pre_calibration_clip_hi))

    def run_holdout_protocol(
        self,
        holdout_years: List[int],
        config,
    ) -> HoldoutReport:
        """Run frozen holdout evaluation across multiple years."""
        cfg_hash = config_hash(config)
        report = HoldoutReport(
            holdout_years=holdout_years,
            config_hash_value=cfg_hash,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        )

        for year in holdout_years:
            try:
                metrics = self.evaluate_holdout_year(year, config)
                report.per_year[year] = metrics
                logger.info(
                    "Holdout %d: Brier=%.4f, Seed=%.4f, Acc=%.3f",
                    year, metrics.brier_score, metrics.seed_baseline_brier,
                    metrics.accuracy,
                )
            except Exception as e:
                logger.error("Holdout %d failed: %s", year, e)

        return report


# ───────────────────────────────────────────────────────────────────────
# 6. Sensitivity Analyzer
# ───────────────────────────────────────────────────────────────────────

class SensitivityAnalyzer:
    """Grid search each Tier 3 constant via LOYO on dev years."""

    def __init__(
        self,
        historical_dir: str = "data/raw/historical",
        dev_years: Optional[List[int]] = None,
        holdout_years: Optional[List[int]] = None,
        n_grid_points: int = 11,
    ):
        self.evaluator = HoldoutEvaluator(historical_dir)
        self.holdout_years = holdout_years or []
        if dev_years is not None:
            self.dev_years = dev_years
        else:
            self.dev_years = [
                y for y in self.evaluator.all_years
                if y not in self.holdout_years and y != 2020
            ]
        self.n_grid_points = n_grid_points

    def _make_grid(self, constant: PipelineConstant) -> List[float]:
        """Generate grid values for a constant, always including current value."""
        lo, hi = constant.valid_range
        grid = list(np.linspace(lo, hi, self.n_grid_points))

        # Ensure current value is in the grid
        curr = float(constant.current_value) if not isinstance(
            constant.current_value, (list, tuple)) else None
        if curr is not None:
            # Find closest grid point and replace or insert
            closest_idx = min(range(len(grid)), key=lambda i: abs(grid[i] - curr))
            if abs(grid[closest_idx] - curr) > 1e-9:
                grid[closest_idx] = curr
            grid.sort()

        return grid

    def _apply_constant_to_config(
        self,
        config,
        constant: PipelineConstant,
        value: float,
    ):
        """Create a modified config with one constant changed."""
        cfg = copy.deepcopy(config)

        if constant.name == "tournament_shrinkage":
            cfg.tournament_shrinkage = value
        elif constant.name == "seed_prior_weight":
            cfg.seed_prior_weight = value
        elif constant.name == "consistency_bonus_max":
            cfg.consistency_bonus_max = value
        elif constant.name == "consistency_normalizer":
            cfg.consistency_normalizer = value
        elif constant.name == "ensemble_lgb_weight":
            cfg.ensemble_lgb_weight = value
            # Keep xgb weight, adjust logistic
        elif constant.name == "ensemble_xgb_weight":
            cfg.ensemble_xgb_weight = value
        elif constant.name == "mc_noise_std":
            pass  # MC constant — not evaluated at game level
        elif constant.name == "mc_regional_correlation":
            pass  # MC constant — not evaluated at game level

        return cfg

    def analyze_constant(
        self,
        constant: PipelineConstant,
        base_config,
        cached_by_year: Optional[Dict[int, dict]] = None,
    ) -> ConstantSensitivityResult:
        """Grid search one constant using LOYO on dev years.

        If cached_by_year is provided, reuses trained models (for post-hoc
        constants like shrinkage/weights that don't affect training).
        Otherwise trains fresh for each dev year.
        """
        if constant.name in TIER3_MC_CONSTANTS:
            logger.info("Skipping MC constant %s (bracket-level, not game-level)",
                        constant.name)
            curr = float(constant.current_value)
            return ConstantSensitivityResult(
                constant_name=constant.name,
                grid_values=[curr],
                loyo_brier_scores=[0.0],
                current_value=curr,
                optimal_value=curr,
                optimal_brier=0.0,
                current_brier=0.0,
                brier_range=0.0,
                is_flat=True,
            )

        is_posthoc = constant.name in {
            "tournament_shrinkage", "seed_prior_weight",
            "consistency_bonus_max", "consistency_normalizer",
            "ensemble_lgb_weight", "ensemble_xgb_weight",
        }

        grid = self._make_grid(constant)
        logger.info("Sensitivity: %s — %d grid points x %d dev years (posthoc=%s)",
                     constant.name, len(grid), len(self.dev_years), is_posthoc)

        # Train once per dev year if posthoc (reuse across grid points)
        if is_posthoc and cached_by_year is None:
            cached_by_year = {}
            for dev_year in self.dev_years:
                try:
                    cached_by_year[dev_year] = self.evaluator._train_for_year(
                        dev_year, base_config
                    )
                except Exception as e:
                    logger.debug("Training for dev year %d failed: %s", dev_year, e)

        grid_briers = []

        for val in grid:
            cfg = self._apply_constant_to_config(base_config, constant, val)
            year_briers = []

            for dev_year in self.dev_years:
                try:
                    if is_posthoc and cached_by_year and dev_year in cached_by_year:
                        # Reuse cached training, just re-apply post-hoc params
                        metrics = self.evaluator._apply_posthoc_and_score(
                            cached_by_year[dev_year], cfg
                        )
                    else:
                        # Full retrain (for training-phase constants)
                        metrics = self.evaluator.evaluate_holdout_year(dev_year, cfg)
                    year_briers.append(metrics.brier_score)
                except Exception as e:
                    logger.debug("Dev year %d failed for %s=%.3f: %s",
                                 dev_year, constant.name, val, e)
                    continue

            mean_brier = float(np.mean(year_briers)) if year_briers else 1.0
            grid_briers.append(mean_brier)
            logger.info("  %s=%.4f -> mean Brier=%.5f (%d years)",
                        constant.name, val, mean_brier, len(year_briers))

        curr = float(constant.current_value)
        best_idx = int(np.argmin(grid_briers))
        optimal_val = grid[best_idx]
        optimal_brier = grid_briers[best_idx]

        curr_idx = min(range(len(grid)), key=lambda i: abs(grid[i] - curr))
        current_brier = grid_briers[curr_idx]

        brier_range = max(grid_briers) - min(grid_briers)

        # Detect tuning-evaluation circularity: if this constant was originally
        # tuned via LOYO and we're now re-evaluating it via LOYO, the sensitivity
        # surface may be biased toward confirming the current value.
        derivation_lower = constant.derivation.lower()
        has_circularity = any(kw in derivation_lower for kw in
                              ["loyo", "calibrated", "iteratively"])

        return ConstantSensitivityResult(
            constant_name=constant.name,
            grid_values=grid,
            loyo_brier_scores=grid_briers,
            current_value=curr,
            optimal_value=optimal_val,
            optimal_brier=optimal_brier,
            current_brier=current_brier,
            brier_range=brier_range,
            is_flat=brier_range < 0.005,
            circularity_warning=has_circularity,
        )

    def analyze_mc_constant(
        self,
        constant: PipelineConstant,
        base_config,
        n_mc_trials: int = 200,
    ) -> ConstantSensitivityResult:
        """Evaluate an MC constant by simulating logit-noise on tournament games.

        For each grid point, for each dev year: loads historical tournament
        games, computes baseline matchup probabilities from AdjEM, applies
        MC-style logit noise at the constant's value, and measures how often
        the noisy prediction still matches the actual outcome.

        This is computationally cheaper than full bracket simulation but
        captures how MC noise parameters affect game-level accuracy.

        WARNING: The ``loyo_brier_scores`` field in the result contains a
        negated accuracy metric (lower = better, for consistency with Brier
        convention), NOT actual Brier scores.
        """
        grid = self._make_grid(constant)
        logger.info("MC Sensitivity: %s -- %d grid points, %d trials/game",
                     constant.name, len(grid), n_mc_trials)

        # Map constant name to which noise parameter it controls
        noise_param = constant.name  # e.g. "mc_noise_std"

        grid_scores: List[float] = []

        for val in grid:
            year_scores: List[float] = []

            for dev_year in self.dev_years:
                try:
                    score = self._eval_mc_for_year(
                        dev_year, noise_param, val,
                        n_mc_trials, base_config,
                    )
                    year_scores.append(score)
                except Exception as e:
                    logger.debug("MC eval year %d failed: %s", dev_year, e)
                    continue

            mean_score = float(np.mean(year_scores)) if year_scores else 0.0
            grid_scores.append(mean_score)
            logger.info("  %s=%.4f -> mean score=%.5f (%d years)",
                        constant.name, val, mean_score, len(year_scores))

        curr = float(constant.current_value)
        best_idx = int(np.argmin(grid_scores))
        optimal_val = grid[best_idx]
        optimal_score = grid_scores[best_idx]

        curr_idx = min(range(len(grid)), key=lambda i: abs(grid[i] - curr))
        current_score = grid_scores[curr_idx]

        score_range = max(grid_scores) - min(grid_scores) if grid_scores else 0.0

        derivation_lower = constant.derivation.lower()
        has_circularity = any(kw in derivation_lower for kw in
                              ["loyo", "calibrated", "iteratively"])

        return ConstantSensitivityResult(
            constant_name=constant.name,
            grid_values=grid,
            loyo_brier_scores=grid_scores,
            current_value=curr,
            optimal_value=optimal_val,
            optimal_brier=optimal_score,
            current_brier=current_score,
            brier_range=score_range,
            is_flat=score_range < 0.01,  # Wider threshold for MC metrics
            circularity_warning=has_circularity,
        )

    def _eval_mc_for_year(
        self,
        year: int,
        noise_param: str,
        value: float,
        n_trials: int,
        base_config,
    ) -> float:
        """Run MC noise evaluation for one year's tournament games.

        Returns negated mean accuracy (lower = better for consistency with
        Brier convention where lower = better).
        """
        games_path = os.path.join(self.evaluator.historical_dir,
                                  f"historical_games_{year}.json")
        metrics_path = os.path.join(self.evaluator.historical_dir,
                                    f"team_metrics_{year}.json")

        if not os.path.exists(games_path) or not os.path.exists(metrics_path):
            raise ValueError(f"Missing data for year {year}")

        with open(games_path) as f:
            games_payload = json.load(f)
        with open(metrics_path) as f:
            metrics_payload = json.load(f)

        # Build team metrics lookup
        team_metrics = {}
        for entry in metrics_payload.get("teams", []):
            tid = entry.get("team_id", "")
            off_rtg = entry.get("adj_o", entry.get("off_rtg", 0.0))
            def_rtg = entry.get("adj_d", entry.get("def_rtg", 0.0))
            if off_rtg and def_rtg:
                team_metrics[tid] = {"off_rtg": off_rtg, "def_rtg": def_rtg}

        if not team_metrics:
            raise ValueError(f"No metrics for year {year}")

        # Extract tournament games (March 14 - April 15)
        tourney_games = []
        for game in games_payload.get("games", []):
            game_date = game.get("date", "")
            if not game_date:
                continue
            try:
                month = int(game_date.split("-")[1])
                day = int(game_date.split("-")[2])
            except (IndexError, ValueError):
                continue
            if not ((month == 3 and day >= 14) or (month == 4 and day <= 15)):
                continue

            t1 = game.get("team1_id", "")
            t2 = game.get("team2_id", "")
            if t1 not in team_metrics or t2 not in team_metrics:
                continue

            outcome = 1 if game.get("team1_score", 0) > game.get("team2_score", 0) else 0
            tourney_games.append((t1, t2, outcome))

        if len(tourney_games) < 10:
            raise ValueError(f"Too few tournament games for year {year}")

        # Determine the noise_std to use for this evaluation
        base_noise = 0.12  # Default mc_noise_std
        if noise_param == "mc_noise_std":
            noise_std = value
        elif noise_param == "mc_regional_correlation":
            # Regional correlation scales the noise variance.
            # Model as: effective_noise = base * (1 + correlation)
            noise_std = base_noise * (1.0 + value)
        elif noise_param == "mc_injury_probability":
            # Injury adds extra variance proportional to probability
            noise_std = base_noise + value * 0.5
        elif noise_param in ("mc_injury_severity_lo", "mc_injury_severity_hi"):
            noise_std = base_noise + value * 0.2
        elif noise_param == "mc_lognormal_sigma":
            noise_std = base_noise * math.exp(value * 0.5)
        elif noise_param == "mc_region_noise_floor":
            noise_std = base_noise * max(value, 0.1)
        else:
            noise_std = base_noise

        rng = np.random.RandomState(42 + year)
        correct_total = 0.0
        n_games = len(tourney_games)

        for t1, t2, outcome in tourney_games:
            em1 = team_metrics[t1]["off_rtg"] - team_metrics[t1]["def_rtg"]
            em2 = team_metrics[t2]["off_rtg"] - team_metrics[t2]["def_rtg"]
            base_prob = 1.0 / (1.0 + math.exp(-0.1735 *(em1 - em2)))

            # Simulate n_trials noisy predictions
            safe_p = max(0.001, min(0.999, base_prob))
            logit_p = math.log(safe_p / (1.0 - safe_p))
            noisy_logits = logit_p + rng.normal(0, noise_std, size=n_trials)
            noisy_probs = 1.0 / (1.0 + np.exp(-noisy_logits))
            predicted_winners = (noisy_probs > 0.5).astype(int)
            correct_total += np.mean(predicted_winners == outcome)

        # Negate so lower = better (consistency with Brier)
        return -(correct_total / n_games)

    def analyze_all_tier3(
        self,
        base_config,
        include_mc: bool = False,
        mc_trials: int = 200,
    ) -> Dict[str, ConstantSensitivityResult]:
        """Run sensitivity for all Tier 3 constants.

        Trains once per dev year and reuses across all posthoc constants,
        reducing total trainings from O(constants * grid * years) to
        O(years) + O(constants * grid) cheap scoring passes.

        Args:
            base_config: SOTAPipelineConfig to use as baseline.
            include_mc: If True, also evaluate MC constants (slower).
            mc_trials: Noise trials per game for MC evaluation.
        """
        # Pre-train for all dev years (shared across posthoc constants)
        logger.info("Pre-training models for %d dev years...", len(self.dev_years))
        cached_by_year: Dict[int, dict] = {}
        for dev_year in self.dev_years:
            try:
                cached_by_year[dev_year] = self.evaluator._train_for_year(
                    dev_year, base_config
                )
            except Exception as e:
                logger.warning("Pre-training for dev year %d failed: %s", dev_year, e)

        results = {}
        for constant in get_tier3_constants():
            if constant.name in TIER3_MC_CONSTANTS:
                if include_mc:
                    try:
                        result = self.analyze_mc_constant(
                            constant, base_config, n_mc_trials=mc_trials,
                        )
                        results[constant.name] = result
                    except Exception as e:
                        logger.error("MC sensitivity for %s failed: %s",
                                     constant.name, e)
                else:
                    logger.info("Skipping MC constant: %s (use --include-mc)",
                                constant.name)
                continue
            try:
                result = self.analyze_constant(
                    constant, base_config, cached_by_year=cached_by_year
                )
                results[constant.name] = result
            except Exception as e:
                logger.error("Sensitivity analysis failed for %s: %s",
                             constant.name, e)
        return results


# ───────────────────────────────────────────────────────────────────────
# 6b. Monte Carlo Parameter Backtester
# ───────────────────────────────────────────────────────────────────────

@dataclass
class MCBacktestResult:
    """Result of backtesting one MC parameter combination against historical data."""

    noise_std: float
    regional_correlation: float
    years_tested: List[int]
    per_year_upset_calibration: Dict[int, float]  # year -> abs(predicted - actual upset rate)
    mean_upset_calibration_error: float
    per_year_seed_accuracy: Dict[int, float]  # year -> fraction correct by higher seed
    mean_seed_accuracy: float
    per_year_log_prob: Dict[int, float]  # year -> mean log probability of actual outcomes
    mean_log_prob: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "noise_std": round(self.noise_std, 4),
            "regional_correlation": round(self.regional_correlation, 4),
            "years_tested": self.years_tested,
            "mean_upset_calibration_error": round(self.mean_upset_calibration_error, 4),
            "mean_seed_accuracy": round(self.mean_seed_accuracy, 4),
            "mean_log_prob": round(self.mean_log_prob, 4),
            "per_year_upset_calibration": {
                str(y): round(v, 4) for y, v in self.per_year_upset_calibration.items()
            },
            "per_year_log_prob": {
                str(y): round(v, 4) for y, v in self.per_year_log_prob.items()
            },
        }


class MCParameterBacktester:
    """Backtest Monte Carlo parameters against historical bracket outcomes.

    For each historical year, simulates brackets using the given noise_std
    and regional_correlation values with AdjEM-logistic as the base predictor,
    then scores the simulated outcome probabilities against actual results.

    This evaluates whether the MC noise parameters produce realistic
    upset rates and outcome distributions, independent of the ML model.
    """

    def __init__(
        self,
        historical_dir: str = "data/raw/historical",
        years: Optional[List[int]] = None,
        n_simulations: int = 5000,
    ):
        self.evaluator = HoldoutEvaluator(historical_dir)
        self.years = years or [
            y for y in self.evaluator.all_years
            if 2017 <= y <= 2024 and y != 2020
        ]
        self.n_simulations = n_simulations

    def backtest_params(
        self,
        noise_std: float,
        regional_correlation: float,
    ) -> MCBacktestResult:
        """Backtest one parameter combination across all historical years.

        For each year:
        1. Load tournament games and team AdjEM values
        2. Build a simple AdjEM-logistic base predictor
        3. Run MC simulations with the given noise parameters
        4. Compute upset calibration: compare predicted vs actual upset rates
        5. Compute log probability of actual outcomes under the MC distribution

        This tests whether the noise parameters produce REALISTIC bracket
        outcome distributions, not whether our model is good.
        """
        per_year_upset_cal: Dict[int, float] = {}
        per_year_seed_acc: Dict[int, float] = {}
        per_year_log_prob: Dict[int, float] = {}

        for year in self.years:
            try:
                result = self._backtest_year(
                    year, noise_std, regional_correlation
                )
                if result is not None:
                    per_year_upset_cal[year] = result["upset_cal"]
                    per_year_seed_acc[year] = result["seed_acc"]
                    per_year_log_prob[year] = result["log_prob"]
            except Exception as e:
                logger.debug("MC backtest for year %d failed: %s", year, e)

        mean_upset_cal = (
            float(np.mean(list(per_year_upset_cal.values())))
            if per_year_upset_cal else 1.0
        )
        mean_seed_acc = (
            float(np.mean(list(per_year_seed_acc.values())))
            if per_year_seed_acc else 0.0
        )
        mean_log_prob = (
            float(np.mean(list(per_year_log_prob.values())))
            if per_year_log_prob else -10.0
        )

        return MCBacktestResult(
            noise_std=noise_std,
            regional_correlation=regional_correlation,
            years_tested=sorted(per_year_upset_cal.keys()),
            per_year_upset_calibration=per_year_upset_cal,
            mean_upset_calibration_error=mean_upset_cal,
            per_year_seed_accuracy=per_year_seed_acc,
            mean_seed_accuracy=mean_seed_acc,
            per_year_log_prob=per_year_log_prob,
            mean_log_prob=mean_log_prob,
        )

    def backtest_grid(
        self,
        noise_std_values: Optional[List[float]] = None,
        regional_corr_values: Optional[List[float]] = None,
    ) -> List[MCBacktestResult]:
        """Grid search over MC parameter combinations.

        Default grids bracket the current values (0.12, 0.10) with
        alternatives to verify they are near-optimal.
        """
        if noise_std_values is None:
            noise_std_values = [0.04, 0.08, 0.12, 0.16, 0.20]
        if regional_corr_values is None:
            regional_corr_values = [0.0, 0.05, 0.10, 0.15, 0.25]

        results = []
        for ns in noise_std_values:
            for rc in regional_corr_values:
                logger.info("MC backtest: noise_std=%.2f, regional_correlation=%.2f",
                            ns, rc)
                result = self.backtest_params(ns, rc)
                results.append(result)
                logger.info(
                    "  upset_cal=%.4f, seed_acc=%.3f, log_prob=%.3f (%d years)",
                    result.mean_upset_calibration_error,
                    result.mean_seed_accuracy,
                    result.mean_log_prob,
                    len(result.years_tested),
                )

        return results

    def _backtest_year(
        self,
        year: int,
        noise_std: float,
        regional_correlation: float,
    ) -> Optional[Dict[str, float]]:
        """Backtest MC parameters for a single historical year.

        Uses AdjEM-logistic probabilities as the base predictor, then
        measures how well the MC noise model predicts actual upset rates.
        """
        try:
            games_payload, metrics_payload = self.evaluator._load_year_data(year)
        except FileNotFoundError:
            return None

        team_metrics = self.evaluator._build_team_metrics(metrics_payload)
        if not team_metrics:
            return None

        games = games_payload.get("games", [])
        games = self.evaluator._infer_dates_and_split(games, year)
        tournament_games = self.evaluator._extract_tournament_games(
            games, team_metrics, year
        )

        if len(tournament_games) < 10:
            return None

        # Build AdjEM-logistic base predictor
        def _em_prob(t1_id: str, t2_id: str) -> float:
            m1 = team_metrics.get(t1_id)
            m2 = team_metrics.get(t2_id)
            if m1 is None or m2 is None:
                return 0.5
            em1 = m1["off_rtg"] - m1["def_rtg"]
            em2 = m2["off_rtg"] - m2["def_rtg"]
            em_diff = em1 - em2
            return 1.0 / (1.0 + math.exp(-0.145 * em_diff))

        # Score tournament games: for each game, compute base probability
        # then apply logit-space noise to get MC-adjusted probability.
        # Compare simulated upset rates against actual.
        rng = np.random.default_rng(year)
        actual_upsets = 0
        total_games = 0
        simulated_upset_probs = []
        game_log_probs = []

        for tg in tournament_games:
            t1, t2 = tg["team1_id"], tg["team2_id"]
            base_p = _em_prob(t1, t2)
            outcome = tg["outcome"]  # 1 if team1 won

            # Simulate N trials with noise
            safe_p = np.clip(base_p, 0.001, 0.999)
            logit = np.log(safe_p / (1.0 - safe_p))

            noise_samples = rng.normal(0, noise_std, self.n_simulations)
            noisy_logits = logit + noise_samples
            noisy_probs = 1.0 / (1.0 + np.exp(-noisy_logits))
            noisy_probs = np.clip(noisy_probs, 0.01, 0.99)

            # Mean probability across simulations
            mean_p = float(np.mean(noisy_probs))
            # Log probability of actual outcome
            if outcome == 1:
                game_log_probs.append(math.log(max(mean_p, 1e-7)))
            else:
                game_log_probs.append(math.log(max(1.0 - mean_p, 1e-7)))

            # Track upsets: team1 is "favorite" if base_p > 0.5
            is_favorite_t1 = base_p > 0.5
            actual_upset = (is_favorite_t1 and outcome == 0) or (
                not is_favorite_t1 and outcome == 1
            )
            if actual_upset:
                actual_upsets += 1

            # Simulated upset probability
            if is_favorite_t1:
                simulated_upset_probs.append(1.0 - mean_p)
            else:
                simulated_upset_probs.append(mean_p)
            total_games += 1

        if total_games == 0:
            return None

        actual_upset_rate = actual_upsets / total_games
        predicted_upset_rate = float(np.mean(simulated_upset_probs))
        upset_cal = abs(predicted_upset_rate - actual_upset_rate)

        # Seed accuracy: fraction where the higher-AdjEM team won
        seed_correct = sum(
            1 for tg in tournament_games
            if (
                _em_prob(tg["team1_id"], tg["team2_id"]) > 0.5
                and tg["outcome"] == 1
            ) or (
                _em_prob(tg["team1_id"], tg["team2_id"]) < 0.5
                and tg["outcome"] == 0
            )
        )
        seed_acc = seed_correct / max(len(tournament_games), 1)

        return {
            "upset_cal": upset_cal,
            "seed_acc": seed_acc,
            "log_prob": float(np.mean(game_log_probs)),
        }


# ───────────────────────────────────────────────────────────────────────
# 7. Audit Report
# ───────────────────────────────────────────────────────────────────────

class RDOFAuditReport:
    """Formats and outputs the full RDoF audit report."""

    def __init__(
        self,
        holdout_report: Optional[HoldoutReport] = None,
        sensitivity_results: Optional[Dict[str, ConstantSensitivityResult]] = None,
        complexity_audit: Optional[ModelComplexityAudit] = None,
    ):
        self.holdout_report = holdout_report
        self.sensitivity_results = sensitivity_results or {}
        self.complexity_audit = complexity_audit

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "metadata": {
                "report_type": "researcher_degrees_of_freedom_audit",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            "constant_inventory": {
                "tier1_externally_derived": [
                    c.to_dict() for c in get_constants_by_tier(1)
                ],
                "tier2_structurally_constrained": [
                    c.to_dict() for c in get_constants_by_tier(2)
                ],
                "tier3_freely_tuned": [
                    c.to_dict() for c in get_constants_by_tier(3)
                ],
                "tier3_count": len(get_tier3_constants()),
            },
        }

        if self.holdout_report:
            result["holdout_evaluation"] = self.holdout_report.to_dict()

        if self.sensitivity_results:
            result["sensitivity_analysis"] = {
                name: r.to_dict()
                for name, r in self.sensitivity_results.items()
            }

        if self.complexity_audit:
            result["model_complexity_audit"] = self.complexity_audit.to_dict()

        result["recommendations"] = self._generate_recommendations()
        return result

    def _generate_recommendations(self) -> List[str]:
        recs = []

        # Integrity level caveat (always first)
        if self.holdout_report:
            level = self.holdout_report.integrity_level
            level_names = {1: "TRUE PROSPECTIVE", 2: "QUASI-PROSPECTIVE",
                           3: "RETROSPECTIVE DIAGNOSTIC"}
            recs.append(
                f"INTEGRITY: Level {level} ({level_names.get(level, 'UNKNOWN')}). "
                f"{self.holdout_report.integrity_note}"
            )

        # Holdout recommendations
        if self.holdout_report:
            v = self.holdout_report.verdict()
            if v == "PASS":
                recs.append(
                    f"HOLDOUT: Pipeline beats AdjEM-logistic baseline by "
                    f"{(self.holdout_report.aggregate_seed_brier - self.holdout_report.aggregate_brier):.4f} "
                    f"Brier on {self.holdout_report.total_games} held-out tournament games."
                )
            elif v == "WARN":
                recs.append(
                    "HOLDOUT: Pipeline marginally better than AdjEM-logistic baseline (<0.005 Brier). "
                    "Improvement may not be statistically significant."
                )
            else:
                recs.append(
                    "HOLDOUT: Pipeline does NOT beat AdjEM-logistic baseline on held-out years. "
                    "Pipeline complexity may not be justified."
                )

        # Circularity summary
        if self.sensitivity_results:
            circular = [name for name, sr in self.sensitivity_results.items()
                        if sr.circularity_warning]
            if circular:
                recs.append(
                    f"CIRCULARITY: {len(circular)}/{len(self.sensitivity_results)} "
                    f"constants were originally tuned via LOYO and are now being "
                    f"re-evaluated via LOYO: {', '.join(circular)}. "
                    f"Sensitivity confirms flatness but cannot prove non-overfitting."
                )

        # Sensitivity recommendations
        for name, sr in self.sensitivity_results.items():
            circ_tag = " [CIRCULAR]" if sr.circularity_warning else ""
            if sr.is_flat:
                recs.append(
                    f"SENSITIVITY: {name} is INSENSITIVE (Brier range={sr.brier_range:.4f}).{circ_tag} "
                    f"Current value {sr.current_value} is defensible."
                )
            elif sr.brier_gap > 0.003:
                recs.append(
                    f"SENSITIVITY: {name} may be OVERFIT. Current={sr.current_value}, "
                    f"optimal={sr.optimal_value} (Brier gap={sr.brier_gap:.4f}).{circ_tag}"
                )
            else:
                recs.append(
                    f"SENSITIVITY: {name} is near-optimal. "
                    f"Current={sr.current_value}, optimal={sr.optimal_value}, "
                    f"gap={sr.brier_gap:.4f}.{circ_tag}"
                )

        # DoF ratio warning
        n_tier3 = len(get_tier3_constants())
        n_games = self.holdout_report.total_games if self.holdout_report else 0
        if n_games > 0:
            ratio = n_tier3 / n_games
            recs.append(
                f"DoF RATIO: {n_tier3} freely-tuned constants / "
                f"{n_games} holdout games = {ratio:.3f} "
                f"(target < 0.01)"
            )

        # Complexity audit warnings
        if self.complexity_audit:
            if self.complexity_audit.passed:
                recs.append(
                    f"COMPLEXITY: PASS — {self.complexity_audit.total_effective_params} "
                    f"effective params / {self.complexity_audit.n_training_samples} "
                    f"samples = {self.complexity_audit.actual_ratio:.1%} "
                    f"(target < {self.complexity_audit.target_ratio:.0%})"
                )
            for w in self.complexity_audit.warnings:
                recs.append(f"COMPLEXITY: {w}")

        # Effective DoF summary (from sensitivity analysis)
        if self.sensitivity_results:
            flat_count = sum(1 for sr in self.sensitivity_results.values()
                             if sr.rdof_risk_category == "flat_plateau")
            mild_count = sum(1 for sr in self.sensitivity_results.values()
                             if sr.rdof_risk_category == "mild_slope")
            sharp_count = sum(1 for sr in self.sensitivity_results.values()
                              if sr.rdof_risk_category == "sharp_peak")
            total_eff_dof = sum(sr.effective_dof
                                for sr in self.sensitivity_results.values())
            recs.append(
                f"EFFECTIVE DoF: {total_eff_dof:.1f} of "
                f"{len(self.sensitivity_results)} analyzed constants "
                f"({flat_count} flat, {mild_count} mild, {sharp_count} sharp). "
                f"Only sharp-peaked constants consume meaningful researcher freedom."
            )

        return recs

    def to_text(self) -> str:
        """Human-readable audit report."""
        lines = [
            "=" * 65,
            "RESEARCHER DEGREES OF FREEDOM AUDIT REPORT",
            f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        ]
        if self.holdout_report:
            lines.append(f"Config hash: {self.holdout_report.config_hash_value}")
            level_names = {1: "TRUE PROSPECTIVE", 2: "QUASI-PROSPECTIVE",
                           3: "RETROSPECTIVE DIAGNOSTIC"}
            lines.append(f"Integrity: Level {self.holdout_report.integrity_level} "
                         f"({level_names.get(self.holdout_report.integrity_level, '?')})")
        lines.append("=" * 65)

        # Section 1: Inventory
        lines.append("")
        lines.append("1. CONSTANT INVENTORY")
        lines.append("-" * 65)
        for tier in [1, 2, 3]:
            constants = get_constants_by_tier(tier)
            tier_names = {1: "Externally Derived", 2: "Structurally Constrained",
                          3: "Freely Tuned (REQUIRES VALIDATION)"}
            lines.append(f"  Tier {tier} ({tier_names[tier]}): {len(constants)} constants")
            for c in constants:
                val_str = (f"{c.current_value}" if not isinstance(c.current_value, list)
                           else str(c.current_value))
                lines.append(f"    - {c.name} = {val_str}")
                lines.append(f"      [{c.derivation}]")

        n_tier3 = len(get_tier3_constants())
        lines.append(f"\n  TOTAL Researcher DoF: {n_tier3} (Tier 3 count)")

        # Section 2: Holdout
        if self.holdout_report:
            lines.append("")
            lines.append(f"2. HOLDOUT EVALUATION (Years: {self.holdout_report.holdout_years})")
            lines.append("-" * 65)
            lines.append(f"  {'Year':<6} {'N':<5} {'Brier':<8} {'LogLoss':<9} {'Acc':<7} {'ECE':<6}")

            for yr, m in sorted(self.holdout_report.per_year.items()):
                lines.append(
                    f"  {m.year:<6} {m.n_games:<5} {m.brier_score:<8.4f} "
                    f"{m.log_loss:<9.4f} {m.accuracy:<7.3f} {m.ece:<6.3f}"
                )

            lines.append(f"  {'Pool':<6} {self.holdout_report.total_games:<5} "
                          f"{self.holdout_report.aggregate_brier:<8.4f}")
            lines.append("")
            lines.append(f"  AdjEM-logistic baseline Brier: {self.holdout_report.aggregate_seed_brier:.4f}")
            lines.append(f"    (Tier 1: no-ML logistic on efficiency margin)")
            lines.append(f"  Elo baseline Brier:            {self.holdout_report.aggregate_elo_brier:.4f}")
            lines.append(f"    (Tier 2: standalone Elo from game-by-game outcomes)")
            lines.append(f"  Uniform Brier:                 0.2500")
            lines.append(f"  Verdict: {self.holdout_report.verdict()}")

        # Section 3: Sensitivity
        if self.sensitivity_results:
            lines.append("")
            lines.append("3. SENSITIVITY ANALYSIS (Tier 3 Constants)")
            lines.append("-" * 65)
            lines.append(
                f"  {'Constant':<28} {'Current':<10} {'Optimal':<10} "
                f"{'Gap':<8} {'Risk':<14} {'EffDoF':<6} {'Circ':<4}"
            )

            for name, sr in sorted(self.sensitivity_results.items()):
                circ = "Y" if sr.circularity_warning else ""
                lines.append(
                    f"  {name:<28} {sr.current_value:<10.4f} "
                    f"{sr.optimal_value:<10.4f} {sr.brier_gap:<8.4f} "
                    f"{sr.rdof_risk_category:<14} {sr.effective_dof:<6.1f} {circ:<4}"
                )

            total_eff = sum(sr.effective_dof
                            for sr in self.sensitivity_results.values())
            lines.append(
                f"\n  TOTAL EFFECTIVE DoF: {total_eff:.1f} / "
                f"{len(self.sensitivity_results)} constants"
            )

            # Mini ASCII plots for non-flat constants
            for name, sr in sorted(self.sensitivity_results.items()):
                if not sr.is_flat and len(sr.grid_values) > 1:
                    lines.append(f"\n  {name}:")
                    min_b = min(sr.loyo_brier_scores)
                    max_b = max(sr.loyo_brier_scores)
                    range_b = max_b - min_b if max_b > min_b else 1.0
                    for val, brier in zip(sr.grid_values, sr.loyo_brier_scores):
                        bar_len = int(20 * (1.0 - (brier - min_b) / range_b))
                        bar = "#" * max(1, bar_len)
                        marker = ""
                        if abs(val - sr.optimal_value) < 1e-6:
                            marker = " <-- optimal"
                        elif abs(val - sr.current_value) < 1e-6:
                            marker = " <-- current"
                        lines.append(
                            f"    {val:7.3f} |{bar:<20s} {brier:.5f}{marker}"
                        )

        # Section 4: Model Complexity Audit
        if self.complexity_audit:
            lines.append("")
            lines.append("4. MODEL COMPLEXITY AUDIT")
            lines.append("-" * 65)
            audit = self.complexity_audit
            lines.append(f"  Training samples (N): {audit.n_training_samples}")
            lines.append(f"  Target: effective params < {audit.target_ratio:.0%} of N "
                         f"= {int(audit.n_training_samples * audit.target_ratio)}")
            lines.append("")
            lines.append(f"  {'Component':<30} {'Eff. Params':>12}")
            for comp, params in sorted(audit.component_params.items()):
                lines.append(f"  {comp:<30} {params:>12,}")
            lines.append(f"  {'─' * 42}")
            lines.append(f"  {'TOTAL':<30} {audit.total_effective_params:>12,}")
            lines.append(f"  Ratio: {audit.actual_ratio:.1%} "
                         f"{'✓ PASS' if audit.passed else '✗ FAIL'}")
            if audit.warnings:
                lines.append("")
                for w in audit.warnings:
                    lines.append(f"  ⚠ {w}")

        # Section 5: Recommendations
        recs = self._generate_recommendations()
        if recs:
            lines.append("")
            lines.append("5. RECOMMENDATIONS")
            lines.append("-" * 65)
            for rec in recs:
                lines.append(f"  - {rec}")

        # Disclosure
        lines.append("")
        lines.append("=" * 65)
        lines.append("DISCLOSURE: All constants were iteratively refined on 2005-2025")
        lines.append("data.  Holdout years (above) were observed during development.")
        lines.append("These results are retrospective diagnostics, NOT pre-registered")
        lines.append("predictions.  The first truly held-out evaluation will be 2026")
        lines.append("(if pipeline is frozen via `freeze-pipeline` before results).")
        lines.append("=" * 65)

        return "\n".join(lines)

    def to_file(self, path: str) -> None:
        """Write JSON report to file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info("RDoF audit report written to %s", path)


# ───────────────────────────────────────────────────────────────────────
# 8. Top-level runner
# ───────────────────────────────────────────────────────────────────────

# ───────────────────────────────────────────────────────────────────────
# 7b. Holdout Contamination Tracker
# ───────────────────────────────────────────────────────────────────────

_HOLDOUT_LOCKFILE = "holdout_evaluation.lock.json"


def _lockfile_path(historical_dir: str) -> str:
    """Path for the holdout contamination lockfile."""
    return os.path.join(historical_dir, _HOLDOUT_LOCKFILE)


def record_holdout_evaluation(
    historical_dir: str,
    holdout_years: List[int],
    config_hash_value: str,
    report_summary: Dict[str, Any],
) -> None:
    """Record that holdout years have been evaluated with a specific config.

    Creates a lockfile that subsequent pipeline runs can check to detect
    whether any Tier 3 constants have been modified after holdout evaluation.
    Once a holdout year has been evaluated, any changes to the pipeline
    config that produced those results constitute researcher degrees of
    freedom contamination — the developer has implicitly used holdout
    performance to guide parameter choices.

    The lockfile is intentionally stored inside the historical data directory
    (not in /tmp or .git) so it persists across sessions and is committed
    alongside the data it protects.
    """
    lock_path = _lockfile_path(historical_dir)
    lock_data = {
        "holdout_years": sorted(holdout_years),
        "config_hash": config_hash_value,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "aggregate_brier": report_summary.get("aggregate_brier"),
        "verdict": report_summary.get("verdict"),
        "warning": (
            "Do NOT modify Tier 3 pipeline constants after this evaluation. "
            "Any changes informed by these results constitute overfitting to "
            "the holdout set. To re-evaluate with new parameters, you must "
            "designate a FRESH holdout year that was excluded from all "
            "prior decision-making."
        ),
    }
    with open(lock_path, "w") as f:
        json.dump(lock_data, f, indent=2)
    logger.info("Holdout lockfile written to %s", lock_path)


def check_holdout_contamination(
    historical_dir: str,
    current_config,
) -> Optional[Dict[str, Any]]:
    """Check whether current config differs from the config used for holdout evaluation.

    Returns None if no lockfile exists (holdout never evaluated) or if the
    config hash matches.  Returns a warning dict if the config has changed
    since holdout evaluation — indicating potential contamination.
    """
    lock_path = _lockfile_path(historical_dir)
    if not os.path.exists(lock_path):
        return None

    try:
        with open(lock_path, "r") as f:
            lock_data = json.load(f)
    except (json.JSONDecodeError, IOError):
        return None

    locked_hash = lock_data.get("config_hash", "")
    current_hash = config_hash(current_config)

    if locked_hash == current_hash:
        return None

    return {
        "contamination_detected": True,
        "locked_config_hash": locked_hash,
        "current_config_hash": current_hash,
        "holdout_years": lock_data.get("holdout_years", []),
        "locked_timestamp": lock_data.get("timestamp", "unknown"),
        "message": (
            f"Pipeline config has changed since holdout evaluation on "
            f"{lock_data.get('timestamp', 'unknown')} "
            f"(hash {locked_hash[:8]}... → {current_hash[:8]}...). "
            f"Holdout years {lock_data.get('holdout_years', [])} may be "
            f"contaminated by post-hoc parameter tuning. Designate fresh "
            f"holdout years or revert config to the locked version."
        ),
    }


# ───────────────────────────────────────────────────────────────────────
# 7c. Sensitivity-Based Auto-Adoption
# ───────────────────────────────────────────────────────────────────────

def adopt_sensitivity_optima(
    sensitivity_results: Dict[str, ConstantSensitivityResult],
    base_config,
    min_brier_improvement: float = 0.002,
    require_not_flat: bool = True,
) -> Tuple[Any, Dict[str, Any]]:
    """Update config with sensitivity-optimal Tier 3 values.

    Only adopts a new value if:
    1. The constant is NOT flat (Brier range >= 0.005) — unless require_not_flat=False
    2. The optimal value improves Brier by at least min_brier_improvement
       over the current value (default 0.002)
    3. The optimal value differs from the current value

    This prevents adopting noise-level changes that don't represent genuine
    improvement.  The min_brier_improvement threshold is set at 0.002 because
    with ~60 tournament games per year, the standard error of the mean Brier
    score is approximately sqrt(0.25^2 / 60) ≈ 0.032 — so 0.002 represents
    roughly 6% of one SE, deliberately conservative to avoid chasing noise.

    Args:
        sensitivity_results: Output from SensitivityAnalyzer.analyze_all_tier3()
        base_config: SOTAPipelineConfig to modify
        min_brier_improvement: Minimum Brier improvement to adopt a change
        require_not_flat: Only adopt non-flat constants

    Returns:
        (updated_config, adoption_log) where adoption_log documents each
        constant's decision (adopted/skipped and reason).
    """
    cfg = copy.deepcopy(base_config)
    log: Dict[str, Any] = {}

    for name, sr in sensitivity_results.items():
        entry: Dict[str, Any] = {
            "current": sr.current_value,
            "optimal": sr.optimal_value,
            "brier_gap": round(sr.brier_gap, 5),
            "is_flat": sr.is_flat,
        }

        if require_not_flat and sr.is_flat:
            entry["action"] = "skipped"
            entry["reason"] = "constant is insensitive (flat)"
            log[name] = entry
            continue

        if sr.brier_gap < min_brier_improvement:
            entry["action"] = "skipped"
            entry["reason"] = (
                f"improvement {sr.brier_gap:.5f} below threshold "
                f"{min_brier_improvement:.5f}"
            )
            log[name] = entry
            continue

        if abs(sr.optimal_value - sr.current_value) < 1e-9:
            entry["action"] = "skipped"
            entry["reason"] = "already at optimal value"
            log[name] = entry
            continue

        # Apply the optimal value
        if name == "tournament_shrinkage":
            cfg.tournament_shrinkage = sr.optimal_value
        elif name == "seed_prior_weight":
            cfg.seed_prior_weight = sr.optimal_value
        elif name == "consistency_bonus_max":
            cfg.consistency_bonus_max = sr.optimal_value
        elif name == "consistency_normalizer":
            cfg.consistency_normalizer = sr.optimal_value
        elif name == "ensemble_lgb_weight":
            cfg.ensemble_lgb_weight = sr.optimal_value
        elif name == "ensemble_xgb_weight":
            cfg.ensemble_xgb_weight = sr.optimal_value
        else:
            entry["action"] = "skipped"
            entry["reason"] = f"no config mapping for {name}"
            log[name] = entry
            continue

        entry["action"] = "adopted"
        entry["new_value"] = sr.optimal_value
        log[name] = entry
        logger.info(
            "Auto-adopt: %s = %.4f -> %.4f (Brier improvement: %.5f)",
            name, sr.current_value, sr.optimal_value, sr.brier_gap,
        )

    return cfg, log


def run_rdof_audit(
    historical_dir: str = "data/raw/historical",
    holdout_years: Optional[List[int]] = None,
    run_holdout: bool = True,
    run_sensitivity: bool = False,
    sensitivity_grid: int = 11,
    output_path: Optional[str] = None,
    config=None,
    include_mc: bool = False,
    mc_trials: int = 200,
) -> Dict[str, Any]:
    """Run the full RDoF audit.

    Args:
        historical_dir: Path to historical game/metric JSON files
        holdout_years: Years to hold out (default: [2024, 2025])
        run_holdout: Whether to run holdout evaluation
        run_sensitivity: Whether to run sensitivity analysis
        sensitivity_grid: Grid points per constant for sensitivity
        output_path: Path to write JSON report (auto-generated if None)
        config: Pipeline config (default: SOTAPipelineConfig())
        include_mc: Include Monte Carlo constants in sensitivity analysis
        mc_trials: Noise trials per game for MC sensitivity (default: 200)

    Returns:
        Audit report dict
    """
    if holdout_years is None:
        holdout_years = [2024, 2025]

    if config is None:
        from ...pipeline.sota import SOTAPipelineConfig
        config = SOTAPipelineConfig()

    # ── Contamination check: warn if config changed since last holdout ──
    contamination = check_holdout_contamination(historical_dir, config)
    if contamination:
        logger.warning(
            "HOLDOUT CONTAMINATION: %s", contamination["message"]
        )

    holdout_report = None
    sensitivity_results = None

    if run_holdout:
        logger.info("Running holdout evaluation for years: %s", holdout_years)
        evaluator = HoldoutEvaluator(historical_dir)
        holdout_report = evaluator.run_holdout_protocol(holdout_years, config)

        # Record the evaluation to detect future contamination
        if holdout_report and holdout_report.per_year:
            record_holdout_evaluation(
                historical_dir,
                holdout_years,
                config_hash(config),
                holdout_report.to_dict(),
            )

    if run_sensitivity:
        logger.info("Running sensitivity analysis...")
        analyzer = SensitivityAnalyzer(
            historical_dir=historical_dir,
            holdout_years=holdout_years,
            n_grid_points=sensitivity_grid,
        )
        sensitivity_results = analyzer.analyze_all_tier3(
            config, include_mc=include_mc, mc_trials=mc_trials,
        )

    # ── Model complexity audit ──
    complexity_audit = estimate_model_complexity()

    report = RDOFAuditReport(
        holdout_report=holdout_report,
        sensitivity_results=sensitivity_results,
        complexity_audit=complexity_audit,
    )

    # Print text report
    print(report.to_text())

    # Include contamination warning in output
    report_dict = report.to_dict()
    if contamination:
        report_dict["holdout_contamination_warning"] = contamination

    # Include auto-adoption recommendations if sensitivity was run
    if sensitivity_results:
        _, adoption_log = adopt_sensitivity_optima(
            sensitivity_results, config,
        )
        report_dict["sensitivity_auto_adoption"] = adoption_log

    # Always write output — auto-generate path if not provided
    if output_path is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        output_dir = Path(historical_dir).parent
        output_path = str(output_dir / f"rdof_audit_{ts}.json")
    report.to_file(output_path)

    return report_dict


def run_prospective_evaluation(
    freeze_path: str,
    evaluation_year: int,
    historical_dir: str = "data/raw/historical",
    output_path: Optional[str] = None,
    config=None,
) -> Dict[str, Any]:
    """Run a Level 2 (quasi-prospective) evaluation against a freeze artifact.

    This function enforces the freeze-then-evaluate discipline:
    1. Verifies the current config matches the freeze artifact
    2. Runs holdout evaluation on the evaluation year
    3. Tags the result with integrity_level=2 and freeze provenance

    The evaluation year must have historical data available (i.e., the
    tournament is over and results have been ingested).

    Args:
        freeze_path: Path to the pipeline freeze artifact JSON.
        evaluation_year: The tournament year to evaluate (e.g. 2026).
        historical_dir: Directory with historical game/metric JSONs.
        output_path: Path to write the evaluation report.
        config: Pipeline config (must match the freeze).

    Returns:
        Evaluation report dict.

    Raises:
        ValueError: If config doesn't match freeze or data is missing.
    """
    if config is None:
        from ...pipeline.sota import SOTAPipelineConfig
        config = SOTAPipelineConfig()

    # Step 1: Verify freeze
    verification = verify_freeze(config, freeze_path)
    if not verification["matches"]:
        raise ValueError(
            f"Pipeline config does not match freeze artifact at {freeze_path}. "
            f"Mismatches: {verification['mismatches']}. "
            f"A prospective evaluation requires an unmodified pipeline."
        )

    frozen_timestamp = verification.get("frozen_timestamp", "unknown")
    logger.info("Freeze verified: config matches artifact from %s", frozen_timestamp)

    # Step 2: Run holdout evaluation on the new year
    evaluator = HoldoutEvaluator(historical_dir)
    holdout_report = evaluator.run_holdout_protocol([evaluation_year], config)

    # Step 3: Tag with quasi-prospective integrity
    holdout_report.integrity_level = 2
    holdout_report.integrity_note = (
        f"QUASI-PROSPECTIVE: Pipeline frozen at {frozen_timestamp} "
        f"(verified against {freeze_path}). Evaluation year {evaluation_year} "
        f"was not available when the pipeline was frozen. "
        f"Pipeline architecture was informed by prior years' data."
    )

    report = RDOFAuditReport(holdout_report=holdout_report)

    print(report.to_text())

    if output_path is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        output_dir = Path(historical_dir).parent
        output_path = str(output_dir / f"prospective_eval_{evaluation_year}_{ts}.json")

    # Write result with freeze provenance
    result = report.to_dict()
    result["provenance"] = {
        "freeze_path": freeze_path,
        "frozen_timestamp": frozen_timestamp,
        "frozen_config_hash": verification["frozen_hash"],
        "evaluation_year": evaluation_year,
        "evaluation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("Prospective evaluation written to %s", output_path)

    return result
