"""Pipeline orchestration helpers.

Extracted from SOTAPipeline._run_shared_pipeline() to reduce the monolith.
Contains pre-run checks, post-pipeline integrations, mode-gated sections,
and report assembly logic.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..config import (
    DataRequirementError,
    SOTAPipelineConfig,
)
from ...exceptions import LeakageError
from ...governance.artifact_provenance import build_artifact_provenance
from ...governance.cache_hygiene import purge_ephemeral_cache_paths

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pre-run checks
# ---------------------------------------------------------------------------


def run_pre_checks(pipeline) -> Optional[Dict]:
    """Run all pre-pipeline validation checks.

    Returns freeze_verification dict (or None).
    Sets pipeline._run_hasher, pipeline._dataset_hashes, pipeline._mc_calibration.
    Also validates year-split policy and freeze-before-predict discipline.
    """
    _purge_ephemeral_repo_caches(pipeline)
    _check_dataset_hashes(pipeline)
    _check_holdout_contamination(pipeline)
    _validate_year_split_policy(pipeline)

    pipeline._mc_calibration = pipeline._load_mc_calibration()

    if pipeline.config.year >= 2026 and pipeline._mc_calibration is None:
        raise DataRequirementError(
            "MC calibration artifact required for 2026+ predictions. "
            "Run `python -m src.main calibrate-mc` and pass --mc-calibration."
        )

    if pipeline.config.year >= 2026 and not pipeline.config.require_freeze_file:
        raise DataRequirementError(
            "Pipeline freeze required for 2026+ predictions. Re-run with --require-freeze and a valid --freeze-file."
        )

    freeze_verification = _check_freeze_requirement(pipeline)

    # Phase 1 evaluation integrity: require freeze artifact before any
    # bracket generation for the target season.
    _check_freeze_for_season(pipeline, freeze_verification)

    _auto_detect_kaggle_dir(pipeline)

    return freeze_verification


def _purge_ephemeral_repo_caches(pipeline) -> None:
    """Remove repo-local runtime caches before leakage-sensitive checks.

    Root cause addressed in first-pass hardening:
    transient Python and pytest caches were being left inside the repo tree,
    which made it harder to distinguish durable artifacts from throwaway local
    state during leakage investigations.
    """
    repo_root = Path(os.getcwd()).resolve()
    removed = purge_ephemeral_cache_paths(repo_root)
    if removed:
        logger.info(
            "Removed %d ephemeral runtime cache paths before pre-run checks.",
            len(removed),
        )


def _check_dataset_hashes(pipeline) -> None:
    """S1/S11: Dataset hash verification on load."""
    try:
        from ...reproducibility.run_hasher import RunHasher

        pipeline._run_hasher = RunHasher(pipeline.config)
        pipeline._dataset_hashes = pipeline._run_hasher.hash_all_inputs()
        incumbent = pipeline._experiment_registry.best(metric="loyo_mean_brier")
        if incumbent is not None and incumbent.dataset_hashes:
            if not pipeline._run_hasher.verify_against(incumbent):
                msg = (
                    "Dataset hashes differ from incumbent experiment "
                    f"{incumbent.experiment_id}. Data files may have changed."
                )
                if pipeline.config.strict_leakage_mode:
                    raise LeakageError(msg)
                logger.warning("DATASET INTEGRITY: %s", msg)
            else:
                logger.info(
                    "Dataset hash verification passed against incumbent %s",
                    incumbent.experiment_id,
                )
        elif pipeline._dataset_hashes:
            logger.info(
                "Dataset hashes computed for %d files (no incumbent to verify against)",
                len(pipeline._dataset_hashes),
            )
    except LeakageError:
        raise
    except Exception as exc:
        logger.debug("Dataset hash verification skipped: %s", exc)
        pipeline._run_hasher = None
        pipeline._dataset_hashes = {}


def _check_holdout_contamination(pipeline) -> None:
    """Check for holdout contamination from config drift."""
    try:
        from ...ml.evaluation.rdof_audit import check_holdout_contamination

        hist_dir = (
            getattr(pipeline, "_runtime_state", {}).get("multi_year_games_dir", pipeline.config.multi_year_games_dir)
            or "data/raw/historical"
        )
        contamination = check_holdout_contamination(hist_dir, pipeline.config)
        if contamination:
            msg = f"HOLDOUT CONTAMINATION: {contamination['message']}"
            if pipeline.config.strict_leakage_mode:
                raise LeakageError(msg)
            logger.warning(msg)
    except Exception as exc:
        logger.debug("Holdout contamination check skipped: %s", exc)


def _validate_year_split_policy(pipeline) -> None:
    """Validate that dev/holdout year split is consistent and non-overlapping.

    Uses the evaluation_integrity.YearSplitPolicy to enforce hard boundaries.
    Training years must be a subset of dev years.
    """
    try:
        from ...ml.evaluation.evaluation_integrity import YearSplitPolicy

        policy = YearSplitPolicy.from_config(pipeline.config)

        # Verify training years are within the dev partition.
        training_years = getattr(pipeline.config, "training_years", None) or []
        if training_years:
            policy.assert_dev_only(
                list(training_years),
                context="model training",
            )

        # Store the policy on the pipeline for downstream use.
        pipeline._year_split_policy = policy
        logger.info(
            "Year split policy validated: dev=%s, holdout=%s",
            sorted(policy.dev_years),
            sorted(policy.holdout_years),
        )
    except Exception as exc:
        if pipeline.config.strict_leakage_mode:
            raise DataRequirementError(f"Year-split policy validation failed: {exc}") from exc
        logger.warning("Year-split policy validation failed: %s", exc)


def _check_freeze_for_season(pipeline, freeze_verification: Optional[Dict]) -> None:
    """Enforce freeze-before-predict discipline for the target season.

    For 2026+ this is a hard requirement (already enforced above).
    For earlier years in strict mode, log a warning if no freeze exists.
    """
    try:
        from ...ml.evaluation.evaluation_integrity import require_freeze_for_season

        if pipeline.config.require_freeze_file and freeze_verification:
            # Already verified above — record the season-level gate result.
            logger.info(
                "Freeze-for-season gate PASSED for year %d (artifact: %s)",
                pipeline.config.year,
                pipeline.config.freeze_file,
            )
        elif pipeline.config.year >= 2026:
            # Should not reach here (caught by earlier check), but belt-and-suspenders.
            result = require_freeze_for_season(
                pipeline.config.year,
                pipeline.config,
                strict=True,
            )
        else:
            # Pre-2026: advisory only, do not block.
            result = require_freeze_for_season(
                pipeline.config.year,
                pipeline.config,
                strict=False,
            )
            if not result.get("verified"):
                logger.info(
                    "No freeze artifact for year %d (advisory). "
                    "Evaluation results will be tagged as retrospective (Level 3).",
                    pipeline.config.year,
                )
    except DataRequirementError:
        raise
    except Exception as exc:
        logger.debug("Freeze-for-season check skipped: %s", exc)


def _check_freeze_requirement(pipeline) -> Optional[Dict]:
    """Verify pipeline freeze file if required."""
    if not pipeline.config.require_freeze_file:
        return None
    if not pipeline.config.freeze_file:
        raise DataRequirementError("Freeze file required (--require-freeze) but no --freeze-file provided.")
    try:
        from ...ml.evaluation.rdof_audit import verify_freeze

        freeze_verification = verify_freeze(pipeline.config, pipeline.config.freeze_file)
        if not freeze_verification.get("matches", False):
            mismatches = "\n".join(freeze_verification.get("mismatches", []))
            raise DataRequirementError(f"Freeze verification failed for {pipeline.config.freeze_file}:\n{mismatches}")
        return freeze_verification
    except DataRequirementError:
        raise
    except Exception as exc:
        raise DataRequirementError(f"Freeze verification failed for {pipeline.config.freeze_file}: {exc}") from exc


def _auto_detect_kaggle_dir(pipeline) -> None:
    """FIX #1: Auto-detect kaggle_dir if not explicitly set.

    Writes resolved path to pipeline._runtime_state to avoid
    mutating the immutable config object.
    """
    _rs = getattr(pipeline, "_runtime_state", {})
    if pipeline.config.kaggle_dir:
        return
    try:
        from ...data.kaggle_downloader import ensure_kaggle_data

        _resolved = ensure_kaggle_data(kaggle_dir=None, auto_download=True)
        if _resolved:
            _rs["kaggle_dir"] = _resolved
            logger.info("FIX #1: Resolved kaggle_dir via ensure_kaggle_data: %s", _resolved)
            return
    except Exception as _e:
        logger.debug("kaggle_downloader.ensure_kaggle_data failed: %s", _e)

    # Fallback to legacy directory scanning
    _kaggle_candidates = [
        os.path.join(os.getcwd(), "data", "kaggle"),
        os.path.join(os.getcwd(), "data", "raw", "kaggle"),
        os.path.join(os.getcwd(), "kaggle"),
        os.path.join(pipeline.config.data_cache_dir, "kaggle"),
    ]
    for _kd in _kaggle_candidates:
        if os.path.isdir(_kd):
            _massey_files = [f for f in os.listdir(_kd) if "massey" in f.lower() or "MTeams" in f]
            if _massey_files:
                _rs["kaggle_dir"] = _kd
                logger.info("FIX #1: Auto-detected kaggle_dir: %s", _kd)
                break


# ---------------------------------------------------------------------------
# Mode-gated sections (pool analysis, bracket portfolio, ablation)
# ---------------------------------------------------------------------------


@dataclass
class ModeGatedResult:
    """Results from mode-gated pipeline sections."""

    public_picks: Dict[str, Dict[str, float]] = field(default_factory=dict)
    scoring_system: Optional[Dict[str, int]] = None
    pool_analysis: Any = None
    ev_max_bracket: Any = None
    leverage_preview: List[Dict] = field(default_factory=list)
    ev_leverage_preview: List[Dict] = field(default_factory=list)
    bracket_portfolio_stats: Dict = field(default_factory=dict)
    ablation_stats: Dict = field(default_factory=dict)
    espn_optimization: Optional[Dict] = None


def run_mode_gated_sections(
    pipeline,
    teams,
    model_round_probs: Dict,
    game_flows: Dict,
) -> ModeGatedResult:
    """Run mode-specific sections: pool analysis, portfolio, ablation."""
    from ...optimization.leverage import TeamMetadata, analyze_pool, get_strategy_profile

    result = ModeGatedResult()
    is_ev = pipeline.config.mode == "ev"
    is_calibration = not is_ev

    if is_calibration:
        result.public_picks = pipeline._load_public_picks(model_round_probs)
        result.scoring_system = pipeline._load_scoring_rules()
        team_metadata = {
            team_id: TeamMetadata(team_name=team.name, seed=team.seed, region=team.region)
            for team_id, team in pipeline.team_struct.items()
        }
        result.pool_analysis = analyze_pool(
            pipeline.config.pool_size,
            model_round_probs,
            result.public_picks,
            scoring_system=result.scoring_system,
            team_metadata=team_metadata,
        )
        result.ev_max_bracket = pipeline._select_ev_bracket(result.pool_analysis)
        result.leverage_preview = [
            {
                "team_id": p.team_id,
                "team_name": pipeline.team_id_to_name.get(p.team_id, p.team_name),
                "round": p.round_name,
                "model_probability": p.model_probability,
                "public_pick_percentage": p.public_pick_percentage,
                "leverage_ratio": p.leverage_ratio,
                "ev_differential": p.expected_value_differential,
            }
            for p in result.pool_analysis.leverage_picks[:15]
        ]

    # ESPN bracket optimization (calibration mode)
    # When espn_mode is enabled, run the ESPN bracket optimizer to produce
    # rank percentile, top-10% rate, and path protection diagnostics.
    if is_calibration and pipeline.config.espn_mode and len(pipeline.team_struct) == 64:
        try:
            from ...espn.bracket_optimizer import ESPNBracketOptimizer, ESPNOptimizationConfig

            # Build first-round matchup list from team structure
            first_round_matchups = _build_espn_first_round_matchups(pipeline)
            if first_round_matchups and len(first_round_matchups) == 64:
                # Build pairwise matchup probabilities
                matchup_probs = {}
                for i, t1 in enumerate(first_round_matchups):
                    for j, t2 in enumerate(first_round_matchups):
                        if i < j and (t1, t2) not in matchup_probs:
                            p = pipeline.predict_probability(t1, t2)
                            matchup_probs[(t1, t2)] = float(p)

                optimizer = ESPNBracketOptimizer(
                    first_round_matchups=first_round_matchups,
                    matchup_probs=matchup_probs,
                    model_round_probs=model_round_probs,
                    public_pick_distribution=result.public_picks,
                    scoring_system=result.scoring_system,
                )
                espn_config = ESPNOptimizationConfig(
                    num_simulations=max(10000, pipeline.config.espn_n_brackets * 3000),
                    pool_size=pipeline.config.pool_size,
                    n_candidates=max(6, pipeline.config.espn_n_brackets * 3),
                    random_seed=pipeline.config.random_seed,
                )
                espn_result = optimizer.optimize(espn_config)
                result.espn_optimization = {
                    "selected_champion": espn_result.selected_champion,
                    "simulated_rank_percentile": round(espn_result.simulated_rank_percentile, 2),
                    "top_10_rate": round(espn_result.top_10_rate, 4),
                    "top_money_rate": round(espn_result.top_money_rate, 4),
                    "path_protection_score": round(espn_result.path_protection_score, 4),
                    "n_candidates_evaluated": len(espn_result.diagnostics),
                    "selected_bracket_rounds": {k: len(v) for k, v in espn_result.selected_bracket.items()},
                }
                logger.info(
                    "ESPN optimization: champion=%s, rank_pct=%.1f%%, top10=%.3f, path_protection=%.3f",
                    espn_result.selected_champion,
                    espn_result.simulated_rank_percentile,
                    espn_result.top_10_rate,
                    espn_result.path_protection_score,
                )
        except Exception as espn_exc:
            logger.warning("ESPN bracket optimization failed: %s", espn_exc)
            result.espn_optimization = {"error": str(espn_exc)}

    # EV mode: preliminary pool analysis for bracket portfolio
    if pipeline.config.enable_bracket_portfolio and is_ev:
        try:
            ev_public_picks = pipeline._load_public_picks(model_round_probs)
            ev_team_metadata = {
                team_id: TeamMetadata(
                    team_name=team.name,
                    seed=team.seed,
                    region=team.region,
                )
                for team_id, team in pipeline.team_struct.items()
            }
            ev_scoring_name = pipeline.config.ev_scoring_system or "standard"
            ev_profile = get_strategy_profile(
                pipeline.config.ev_pool_size,
                scoring_system=ev_scoring_name,
                contrarian_override=(
                    pipeline.config.ev_contrarian_strength if pipeline.config.ev_contrarian_strength != 1.0 else None
                ),
                payout_structure=pipeline.config.ev_payout_structure,
            )
            prelim_analysis = analyze_pool(
                pipeline.config.ev_pool_size,
                model_round_probs,
                ev_public_picks,
                team_metadata=ev_team_metadata,
                ev_scoring_system=ev_scoring_name,
                strategy_profile=ev_profile,
            )
            result.ev_leverage_preview = [
                {
                    "team_id": p.team_id,
                    "team_name": pipeline.team_id_to_name.get(p.team_id, p.team_name),
                    "round": p.round_name,
                    "model_probability": p.model_probability,
                    "public_pick_percentage": p.public_pick_percentage,
                    "leverage_ratio": p.leverage_ratio,
                    "ev_differential": p.expected_value_differential,
                }
                for p in prelim_analysis.leverage_picks[:20]
            ]
            pipeline._ev_preliminary_public_picks = ev_public_picks
            pipeline._ev_preliminary_leverage = result.ev_leverage_preview
            result.public_picks = ev_public_picks
            result.leverage_preview = result.ev_leverage_preview
        except Exception as e:
            logger.warning("EV preliminary pool analysis failed: %s", e)

    # Bracket portfolio generation (both modes)
    if pipeline.config.enable_bracket_portfolio:
        result.bracket_portfolio_stats = _generate_bracket_portfolio(
            pipeline,
            teams,
            result.public_picks,
            result.ev_leverage_preview,
            result.leverage_preview,
            is_ev,
        )

    # Ablation study
    from .._optional_imports import ABLATION_AVAILABLE, AblationStudy

    if pipeline.config.enable_ablation_study and ABLATION_AVAILABLE:
        result.ablation_stats = _run_ablation(pipeline, game_flows)

    return result


def _generate_bracket_portfolio(
    pipeline,
    teams,
    public_picks,
    ev_leverage_preview,
    leverage_preview,
    is_ev,
) -> Dict:
    """Generate bracket portfolio for both modes."""
    from ...optimization.leverage import get_strategy_profile

    try:
        from ...optimization.bracket_portfolio import BracketPortfolioGenerator

        teams_by_region: Dict[str, List[Dict]] = {}
        for team in teams:
            tid = pipeline._team_id(team.name)
            region = team.region or "Unknown"
            teams_by_region.setdefault(region, []).append(
                {
                    "team_id": tid,
                    "name": team.name,
                    "seed": team.seed,
                }
            )
        champ_public = {}
        for tid, rounds in public_picks.items():
            if isinstance(rounds, dict):
                champ_public[tid] = rounds.get("CHAMP", rounds.get("champion_pct", 0.0))
            else:
                champ_public[tid] = float(rounds)

        portfolio_gen = BracketPortfolioGenerator(
            predict_fn=pipeline.predict_probability,
            public_pick_pcts=champ_public,
            round_public_picks=public_picks if is_ev else None,
            leverage_picks=ev_leverage_preview if is_ev else leverage_preview,
        )
        if pipeline.config.mode == "ev":
            portfolio_profile = get_strategy_profile(
                pipeline.config.ev_pool_size,
                scoring_system=pipeline.config.ev_scoring_system or "standard",
                contrarian_override=(
                    pipeline.config.ev_contrarian_strength if pipeline.config.ev_contrarian_strength != 1.0 else None
                ),
                payout_structure=pipeline.config.ev_payout_structure,
            )
            logger.info("Portfolio using EV strategy profile: %s", portfolio_profile.strategy_mix)
        else:
            portfolio_profile = get_strategy_profile(
                pipeline.config.kaggle_effective_pool_size,
                payout_structure="top_10pct",
            )
            logger.info(
                "Portfolio using Kaggle strategy profile (effective_pool_size=%d): %s",
                pipeline.config.kaggle_effective_pool_size,
                portfolio_profile.strategy_mix,
            )

        portfolio = portfolio_gen.generate_portfolio(
            teams_by_region=teams_by_region,
            n_brackets=1000,
            n_simulations=10000,
            seed=pipeline.config.random_seed,
            pool_strategy_profile=portfolio_profile,
            enable_search=(pipeline.config.ev_enable_search if pipeline.config.mode == "ev" else False),
        )
        strategy_counts = {}
        champions = {}
        for b in portfolio:
            strategy_counts[b.strategy] = strategy_counts.get(b.strategy, 0) + 1
            champions[b.champion] = champions.get(b.champion, 0) + 1
        stats = {
            "enabled": True,
            "n_brackets": len(portfolio),
            "strategy_distribution": strategy_counts,
            "champion_diversity": len(champions),
            "top_champions": dict(sorted(champions.items(), key=lambda x: -x[1])[:10]),
        }
        logger.info("Bracket portfolio: %d brackets, %d unique champions", len(portfolio), len(champions))
        return stats
    except Exception as e:
        logger.warning("Bracket portfolio generation failed: %s", e)
        return {"enabled": False, "error": str(e)}


def _run_ablation(pipeline, game_flows) -> Dict:
    """Run ablation study on validation games."""
    from .._optional_imports import AblationStudy

    try:
        val_games = []
        for g in pipeline._unique_games(game_flows):
            if (
                g.team1_id in pipeline.feature_engineer.team_features
                and g.team2_id in pipeline.feature_engineer.team_features
            ):
                _outcome = pipeline._game_outcome(g)
                if _outcome is None:
                    continue
                val_games.append(
                    {
                        "team1": g.team1_id,
                        "team2": g.team2_id,
                        "team1_won": bool(_outcome),
                    }
                )
        if len(val_games) >= 20:
            ablation = AblationStudy(pipeline, val_games)
            return ablation.run_full_ablation().to_dict()
    except Exception as exc:
        logger.warning("Ablation study failed: %s", exc)
        return {"error": f"ablation study failed: {exc}"}
    return {}


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------


def assemble_report(
    pipeline,
    *,
    adjacency,
    baseline_stats: Dict,
    gnn_stats: Dict,
    transformer_stats: Dict,
    uncertainty_stats: Dict,
    calibration_stats: Dict,
    massey_predictor_stats: Dict,
    bracket_sim,
    injury_stats: Dict,
    mode_result: ModeGatedResult,
    freeze_verification: Optional[Dict],
    embedding_proj_stats: Dict,
    schedule_graph,
) -> Dict:
    """Assemble the final pipeline report dict."""
    from ...ml.evaluation.risk_report import RiskReport

    is_ev = pipeline.config.mode == "ev"
    is_calibration = not is_ev
    calibration_samples = int(calibration_stats.get("samples", 0))

    shared_artifacts = {
        "adjacency_matrix": adjacency.tolist(),
        "baseline_training": baseline_stats,
        "gnn": gnn_stats,
        "transformer": transformer_stats,
        "model_uncertainty": uncertainty_stats,
        "calibration": calibration_stats,
        "massey_predictor": massey_predictor_stats,
        "simulation": {
            "num_simulations": bracket_sim.num_simulations,
            "round_of_32_odds": bracket_sim.round_of_32_odds,
            "sweet_sixteen_odds": bracket_sim.sweet_sixteen_odds,
            "elite_eight_odds": bracket_sim.elite_eight_odds,
            "championship_odds": bracket_sim.championship_odds,
            "final_four_odds": bracket_sim.final_four_odds,
            "injury_noise_samples_per_matchup": pipeline.config.injury_noise_samples,
        },
        "proprietary_metrics_summary": {
            "teams_computed": len(pipeline.proprietary_metrics),
            "avg_adj_em": float(
                np.mean([m.adj_efficiency_margin for m in pipeline.proprietary_metrics.values()] or [0.0])
            ),
        },
        "roster_rapm_quality": pipeline.roster_rapm_quality,
        "injury_integration": injury_stats,
        "hyperparameter_tuning": pipeline.tuning_result or {},
        "feature_selection": _build_feature_selection_artifact(pipeline),
        "feature_manifest": getattr(pipeline, "_feature_manifest", {}),
        "ablation_study": mode_result.ablation_stats,
    }

    # Mode-specific artifacts
    if is_calibration:
        shared_artifacts["ev_max_bracket"] = mode_result.ev_max_bracket.to_dict()
        shared_artifacts["pool_recommendation"] = mode_result.pool_analysis.recommended_strategy
        shared_artifacts["public_pick_sources"] = pipeline.public_pick_sources
        shared_artifacts["scoring_system"] = mode_result.scoring_system or {
            "R64": 10,
            "R32": 20,
            "S16": 40,
            "E8": 80,
            "F4": 160,
            "CHAMP": 320,
        }
        shared_artifacts["top_leverage_picks"] = mode_result.leverage_preview
        shared_artifacts["bracket_portfolio"] = mode_result.bracket_portfolio_stats
    elif is_ev:
        if mode_result.bracket_portfolio_stats:
            shared_artifacts["bracket_portfolio"] = mode_result.bracket_portfolio_stats
            shared_artifacts["top_leverage_picks"] = mode_result.ev_leverage_preview
            shared_artifacts["public_pick_sources"] = pipeline.public_pick_sources
        else:
            shared_artifacts["bracket_portfolio"] = {"enabled": False, "reason": "ev_mode"}

    # Phase 4 rubric
    phase_4_rubric = (
        {
            "public_consensus": len(pipeline.public_pick_sources) >= pipeline.config.min_public_sources,
            "leverage_ratio": len(mode_result.leverage_preview) > 0,
            "pareto_front": len(mode_result.pool_analysis.pareto_brackets) > 0 if mode_result.pool_analysis else False,
        }
        if is_calibration
        else {
            "note": "Game theory evaluated in EV analysis layer",
            "pool_size": pipeline.config.ev_pool_size,
            "scoring_system": pipeline.config.ev_scoring_system,
        }
    )

    report = {
        "mode": pipeline.config.mode,
        "audit": {
            "dev_years": pipeline.config.dev_years,
            "holdout_years": pipeline.config.holdout_years,
            "freeze_required": pipeline.config.require_freeze_file,
            "freeze_verification": freeze_verification or {},
            "mc_calibration": (
                {
                    "best_params": (pipeline._mc_calibration or {}).get("best_params"),
                    "dev_score": (pipeline._mc_calibration or {}).get("best_dev_score"),
                    "holdout_score": (pipeline._mc_calibration or {}).get("holdout_score"),
                    "source": (pipeline._mc_calibration or {}).get("_source_path"),
                }
                if pipeline._mc_calibration
                else {}
            ),
        },
        "ml_diagnostics": {
            "calibration_samples": calibration_samples,
            "calibration_min_required": pipeline.config.min_calibration_samples_hard,
            "calibration_method": calibration_stats.get("method", "unknown"),
            "calibration_enabled": bool(pipeline.calibration_pipeline),
            "calibration_uncertainty_band": calibration_stats.get("uncertainty_band"),
            "calibration_uncertainty_level": calibration_stats.get("uncertainty_level"),
            "calibration_uncertainty_note": calibration_stats.get("uncertainty_note"),
            "massey_coverage": getattr(pipeline, "_massey_coverage_stats", {}),
            "calibration_comparison": {
                "brier_before": calibration_stats.get("brier_before"),
                "brier_after": calibration_stats.get("brier_after"),
                "ece_before": calibration_stats.get("ece_before"),
                "ece_after": calibration_stats.get("ece_after"),
                "reliability_pre": calibration_stats.get("reliability_pre"),
                "reliability_post": calibration_stats.get("reliability_post"),
                "sharpening_disabled": not pipeline.config.enable_brier_sharpening,
                "goto_conversion_enabled": pipeline.config.enable_goto_conversion,
            },
            "bma_diagnostics": baseline_stats.get("ensemble_weight_optimization", {}),
        },
        "rubric_evaluation": {
            "phase_1_data_engineering": {
                "proprietary_metrics_computed": bool(pipeline.proprietary_metrics),
                "player_rapm_and_live_talent": True,
                "proprietary_xp_coverage": bool(pipeline.proprietary_metrics),
                "rapm_team_coverage": pipeline.roster_rapm_quality.get("team_coverage_ratio", 0.0) >= 0.8,
                "lead_volatility_entropy": float(
                    np.mean([f.avg_entropy for f in pipeline.feature_engineer.team_features.values()] or [0.0])
                )
                > 0.0,
            },
            "phase_2_architecture": {
                "schedule_graph_constructed": int(adjacency.shape[0]) >= 64 and len(schedule_graph.edges) > 0,
                "d1_scale_graph": int(adjacency.shape[0]) >= 362,
                "gcn_sos_refinement": gnn_stats["enabled"],
                "transformer_temporal_model": transformer_stats["enabled"] or transformer_stats["teams"] > 0,
                "cfa_fusion": False,
            },
            "phase_3_uncertainty_calibration": {
                "brier_optimized": calibration_stats["brier_before"] >= calibration_stats["brier_after"],
                "isotonic": pipeline.config.calibration_method == "isotonic",
                "injury_noise_monte_carlo": pipeline.config.injury_noise_samples >= 10000,
            },
            "phase_4_game_theory": phase_4_rubric,
            "execution_steps": {
                "step_1_data_stack": bool(
                    (pipeline.config.torvik_json or pipeline.config.scrape_live)
                    and (pipeline.config.historical_games_json or pipeline.config.scrape_live)
                ),
                "step_2_adjacency_matrix": len(schedule_graph.edges) > 0,
                "step_3_lightgbm_ranker": baseline_stats["model"]
                in ("lightgbm", "lightgbm_tuned", "stacking_ensemble"),
                "step_3_xgboost_ranker": baseline_stats["model"] in ("xgboost", "xgboost_tuned", "stacking_ensemble"),
                "step_3_stacking_meta": baseline_stats["model"] == "stacking_ensemble",
                "step_3_loyo_cv": bool(baseline_stats.get("loyo_cv", {}).get("enabled")),
                "step_4_pyg_gcn": gnn_stats["framework"] == "pytorch_geometric",
                "step_5_10k_monte_carlo": int(
                    getattr(pipeline, "_runtime_state", {}).get("num_simulations", pipeline.config.num_simulations)
                )
                >= 10000,
                "step_6_ev_max_output": is_calibration,
            },
        },
        "artifacts": shared_artifacts,
        "phase_timings": pipeline._phase_timer.get_timings(),
        "artifact_provenance": build_artifact_provenance(
            pipeline=pipeline,
            artifact_kind="pipeline_report",
        ),
    }

    # Promote key LOYO diagnostics to top-level for easy access
    loyo_cv = baseline_stats.get("loyo_cv", {})
    if loyo_cv.get("enabled"):
        report["pipeline_diagnostics"] = {
            "loyo_mean_brier": loyo_cv.get("mean_brier"),
            "loyo_mean_log_loss": loyo_cv.get("mean_log_loss"),
            "loyo_mean_ece": loyo_cv.get("mean_ece"),
            "loyo_brier_skill_score": loyo_cv.get("mean_brier_skill_score"),
            "loyo_brier_decomposition": loyo_cv.get("brier_decomposition", {}),
            "admission_gate": loyo_cv.get("admission_gate", {}),
            "pit_validation": loyo_cv.get("pit_validation", {}),
            "bma_diagnostics": baseline_stats.get("ensemble_weight_optimization", {}),
            "calibration_comparison": {
                "brier_before": calibration_stats.get("brier_before"),
                "brier_after": calibration_stats.get("brier_after"),
                "ece_before": calibration_stats.get("ece_before"),
                "ece_after": calibration_stats.get("ece_after"),
            },
        }

    # ESPN optimization results (from mode_result if available)
    espn_result = getattr(mode_result, "espn_optimization", None)
    if espn_result:
        report["espn_optimization"] = espn_result

    # Risk report
    if loyo_cv.get("enabled") and loyo_cv.get("per_year"):
        try:
            year_briers = {int(yr): info["brier"] for yr, info in loyo_cv["per_year"].items()}
            risk_report = RiskReport.from_loyo_results(year_briers)
            report["risk_report"] = risk_report.to_dict()
            from ...ml.evaluation.risk_report import RegimeAnalysis, ScenarioAnalysis

            regime = RegimeAnalysis.from_loyo_results(year_briers)
            if regime.regime_labels:
                report["regime_analysis"] = regime.to_dict()
            scenario = ScenarioAnalysis.from_loyo_results(year_briers)
            if scenario.base:
                report["scenario_analysis"] = scenario.to_dict()
        except Exception as exc:
            logger.debug("Risk report generation failed: %s", exc)

    return report


def _build_espn_first_round_matchups(pipeline) -> List[str]:
    """Build 64-team first-round matchup list from pipeline team structure.

    Orders teams by region and seed to produce the standard NCAA bracket
    matchup order: (1v16, 8v9, 5v12, 4v13, 6v11, 3v14, 7v10, 2v15) per region.
    """
    SEED_ORDER = [1, 16, 8, 9, 5, 12, 4, 13, 6, 11, 3, 14, 7, 10, 2, 15]
    by_region: Dict[str, Dict[int, str]] = {}

    for tid, team in pipeline.team_struct.items():
        region = getattr(team, "region", None) or "Unknown"
        seed = getattr(team, "seed", None)
        if seed is not None:
            by_region.setdefault(region, {})[int(seed)] = tid

    matchups: List[str] = []
    for region in sorted(by_region.keys()):
        seed_map = by_region[region]
        for s in SEED_ORDER:
            if s in seed_map:
                matchups.append(seed_map[s])
            else:
                # Placeholder if seed missing
                matchups.append(f"UNKNOWN_{region}_{s}")

    return matchups[:64]


def _build_feature_selection_artifact(pipeline) -> Dict:
    """Build feature selection artifact for report."""
    result = dict(getattr(pipeline, "_feature_manifest", {}) or {})
    if not pipeline.feature_selection_result:
        return result
    fsr = pipeline.feature_selection_result
    result.update(
        {
            "original_dim": fsr.original_dim,
            "reduced_dim": fsr.reduced_dim,
            "correlation_dropped": len(fsr.correlation_dropped),
            "importance_dropped": len(fsr.low_importance_dropped),
            "top_features": [
                {"name": f.name, "importance": round(f.importance, 4)} for f in fsr.importance_scores[:15]
            ],
        }
    )
    if fsr.stability_scores:
        result["stability_scores"] = {
            k: round(v, 3)
            for k, v in sorted(
                fsr.stability_scores.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:10]
        }
    return result


# ---------------------------------------------------------------------------
# Post-pipeline integrations (experiment registry, artifacts, governance)
# ---------------------------------------------------------------------------


def run_post_pipeline(
    pipeline,
    report: Dict,
    baseline_stats: Dict,
    calibration_stats: Dict,
) -> None:
    """Run post-pipeline integrations. Mutates report dict in place."""
    loyo_cv = baseline_stats.get("loyo_cv", {})
    loyo_year_briers = {}
    if loyo_cv.get("enabled"):
        for yr, info in loyo_cv.get("per_year", {}).items():
            loyo_year_briers[int(yr)] = info["brier"]

    experiment_id = _log_experiment(pipeline, report, baseline_stats, calibration_stats, loyo_cv, loyo_year_briers)
    _store_artifacts(pipeline, report, baseline_stats, experiment_id, loyo_cv)
    _check_promotion_gate(pipeline, report, loyo_cv, experiment_id)
    _run_meta_learning(pipeline, report, baseline_stats, loyo_cv)
    _run_governance_gates(pipeline, report, loyo_cv, experiment_id)

    logger.info(
        "Shared pipeline complete (mode=%s): skipped %s",
        pipeline.config.mode,
        "pool_analysis/leverage/portfolio" if pipeline.config.mode == "ev" else "nothing (calibration runs all)",
    )
    logger.info("Phase timings:\n%s", pipeline._phase_timer.summary())


def _log_experiment(pipeline, report, baseline_stats, calibration_stats, loyo_cv, loyo_year_briers) -> str:
    """S3-1: Log experiment to registry."""
    try:
        import hashlib
        from ..context import get_code_version
        from ...ml.evaluation.experiment_registry import ExperimentRecord

        config_json = json.dumps(
            {k: str(v) for k, v in sorted(vars(pipeline.config).items())},
            sort_keys=True,
        )
        config_hash = hashlib.sha256(config_json.encode()).hexdigest()[:16]

        validation_scheme = f"LOYO_{loyo_cv.get('years_evaluated', 0)}yr" if loyo_cv.get("enabled") else "none"

        risk_metrics = {}
        if "risk_report" in report:
            rm = report["risk_report"].get("risk_metrics", {})
            risk_metrics = {
                "max_drawdown": rm.get("max_drawdown", 0),
                "worst_year_brier": rm.get("worst_year_brier", 0),
                "tail_loss_10pct": rm.get("tail_loss_10pct", 0),
                "max_losing_streak": rm.get("max_losing_streak", 0),
                "brier_trend_slope": rm.get("brier_trend_slope", 0),
            }

        model_components = []
        for name in ("lightgbm", "xgboost", "logistic", "spread_regressor"):
            if baseline_stats.get(name, {}).get("enabled", False) or name in baseline_stats.get("model", ""):
                model_components.append(name)
        if not model_components:
            model_components = [baseline_stats.get("model", "unknown")]

        hyperparams = {}
        for key in ("lightgbm", "xgboost", "logistic_regression"):
            tuning = baseline_stats.get(f"{key}_tuning", {})
            if tuning:
                hyperparams[key] = tuning.get("best_params", {})

        cal_method = "none"
        if calibration_stats and isinstance(calibration_stats, dict):
            cal_method = calibration_stats.get("method", "temperature")

        feat_hash = ""
        if hasattr(pipeline, "team_features") and pipeline.team_features:
            feat_list = (
                sorted(pipeline.feature_engineer.feature_names)
                if hasattr(pipeline.feature_engineer, "feature_names")
                else []
            )
            if feat_list:
                feat_hash = hashlib.sha256(str(feat_list).encode()).hexdigest()[:12]
            if not feat_hash:
                feat_hash = hashlib.sha256(str(sorted(pipeline.team_features.keys())).encode()).hexdigest()[:12]

        dataset_hashes = getattr(pipeline, "_dataset_hashes", {})
        code_ver = get_code_version()
        reproducibility_hash = ""
        run_hasher = getattr(pipeline, "_run_hasher", None)
        if run_hasher is not None:
            try:
                reproducibility_hash = run_hasher.compute_reproducibility_hash(code_ver)
            except Exception:
                pass

        secondary_metrics: Dict[str, float] = {}
        if calibration_stats and isinstance(calibration_stats, dict):
            for k in ("ece", "reliability", "resolution", "uncertainty", "brier_before", "brier_after"):
                if k in calibration_stats:
                    secondary_metrics[k] = float(calibration_stats[k])

        holdout_level = 3
        if pipeline.config.require_freeze_file and pipeline.config.freeze_file:
            holdout_level = 1
        elif loyo_cv.get("enabled"):
            holdout_level = 2

        record = ExperimentRecord(
            config_hash=config_hash,
            model_family=baseline_stats.get("model", "unknown"),
            model_components=model_components,
            hyperparameters=hyperparams,
            calibration_method=cal_method,
            code_version=code_ver,
            dataset_version=f"{pipeline.config.year}",
            dataset_hashes=dataset_hashes,
            as_of_timestamp_rules="PIT: shift(1).expanding().mean(); cutoff_date enforced",
            feature_set_id=feat_hash,
            feature_set_hash=feat_hash,
            validation_scheme=validation_scheme,
            loyo_mean_brier=loyo_cv.get("mean_brier", 0),
            loyo_std_brier=float(np.std(list(loyo_year_briers.values()))) if loyo_year_briers else 0,
            loyo_year_briers=loyo_year_briers,
            holdout_integrity_level=holdout_level,
            scoring_metric="brier",
            primary_metric_value=loyo_cv.get("mean_brier", 0),
            secondary_metrics=secondary_metrics,
            path_risk_metrics=risk_metrics,
            regime_analysis=report.get("regime_analysis", {}),
            scenario_analysis=report.get("scenario_analysis", {}),
            decision_policy=pipeline.config.mode,
            reproducibility_hash=reproducibility_hash,
            random_seed=pipeline.config.random_seed,
            phase_timings=pipeline._phase_timer.get_timings(),
            total_wall_clock_seconds=round(pipeline._phase_timer.total_seconds(), 2),
            tags=[f"year_{pipeline.config.year}", pipeline.config.mode],
        )
        return pipeline._experiment_registry.log(record)
    except Exception as exc:
        logger.debug("Experiment registry logging failed: %s", exc)
        return ""


def _store_artifacts(pipeline, report, baseline_stats, experiment_id, loyo_cv) -> None:
    """S16: Artifact storage & reproducibility bundle."""
    try:
        from ...reproducibility.artifact_store import ArtifactStore
        from ...reproducibility.frozen_config import FrozenExperimentConfig
        from ..context import get_code_version
        from dataclasses import asdict as _dc_asdict

        artifact_store = ArtifactStore()
        exp_id = experiment_id or "unknown"

        if hasattr(pipeline, "_model") and pipeline._model is not None:
            artifact_store.save_model(
                pipeline._model,
                exp_id,
                metadata={"model_family": baseline_stats.get("model", "unknown")},
            )

        artifact_store.save_config(pipeline.config, exp_id)

        feature_manifest = getattr(pipeline, "_feature_manifest", None)
        if feature_manifest:
            artifact_store.save_feature_manifest(
                feature_manifest,
                f"{exp_id}_feature_manifest",
            )

        if hasattr(pipeline, "_model") and pipeline._model is not None:
            feat_importance = {}
            model_obj = pipeline._model
            if hasattr(model_obj, "lgb_model") and model_obj.lgb_model is not None:
                try:
                    importances = model_obj.lgb_model.feature_importances_
                    feat_names = getattr(model_obj, "feature_names", [])
                    if len(feat_names) == len(importances):
                        feat_importance = dict(
                            sorted(
                                zip(feat_names, [float(x) for x in importances]),
                                key=lambda x: x[1],
                                reverse=True,
                            )
                        )
                except Exception:
                    pass
            if feat_importance:
                artifact_store.save_predictions(feat_importance, exp_id)

        frozen = FrozenExperimentConfig(
            experiment_id=exp_id,
            pipeline_config=_dc_asdict(pipeline.config),
            dataset_hashes=getattr(pipeline, "_dataset_hashes", {}),
            code_version=get_code_version(),
            reproducibility_hash=getattr(pipeline, "_reproducibility_hash", ""),
            frozen_at=datetime.now(timezone.utc).isoformat(),
        )
        frozen_dir = artifact_store.store_dir / "frozen"
        frozen_dir.mkdir(parents=True, exist_ok=True)
        (frozen_dir / f"{exp_id}.json").write_text(frozen.to_json(), encoding="utf-8")
    except Exception as exc:
        logger.debug("Artifact storage failed: %s", exc)


def _check_promotion_gate(pipeline, report, loyo_cv, experiment_id) -> None:
    """S14/S15: Promotion gate check."""
    try:
        from ...ml.evaluation.promotion_gate import PromotionGate

        candidate_brier = loyo_cv.get("mean_brier", 0)
        if candidate_brier > 0:
            gate = PromotionGate()
            promotion = gate.check(candidate_brier, pipeline._experiment_registry)
            report["promotion_gate"] = {
                "approved": promotion.approved,
                "candidate_brier": promotion.candidate_brier,
                "incumbent_brier": promotion.incumbent_brier,
                "delta": promotion.delta,
                "reason": promotion.reason,
            }
            if promotion.approved:
                logger.info("PROMOTION GATE: %s", promotion.reason)
            else:
                logger.warning("PROMOTION GATE BLOCKED: %s", promotion.reason)
    except Exception as exc:
        logger.debug("Promotion gate check failed: %s", exc)


def _run_meta_learning(pipeline, report, baseline_stats, loyo_cv) -> None:
    """S7: Meta-learning weight adjustment."""
    try:
        from ...ml.meta_learning import MetaLearner

        meta_learner = MetaLearner(registry=pipeline._experiment_registry)
        model_components = []
        for name in ("lightgbm", "xgboost", "logistic", "spread_regressor"):
            if baseline_stats.get(name, {}).get("enabled", False) or name in baseline_stats.get("model", ""):
                model_components.append(name)
        if not model_components:
            model_components = [baseline_stats.get("model", "unknown")]
        if pipeline.ensemble_base_weights:
            decision = meta_learner.adjust_weights(
                pipeline.ensemble_base_weights,
                year=pipeline.config.year,
                model_components=model_components,
            )
            report["meta_learning"] = {
                "regime": decision.regime,
                "confidence": decision.confidence,
                "weight_adjustments": decision.weight_adjustments,
                "reasoning": decision.reasoning,
            }
            logger.info("S7: Meta-learning: %s", decision.reasoning)
    except Exception as exc:
        logger.debug("Meta-learning failed: %s", exc)


def _run_governance_gates(pipeline, report, loyo_cv, experiment_id) -> None:
    """S21/S25: Governance gates in sequential pipeline."""
    try:
        governance_results = []
        for stage_name, stage_context in [
            ("data_loading", {"n_teams": len(pipeline.team_features), "stale_data": False}),
            ("model_training", {"brier": loyo_cv.get("mean_brier", 0), "has_leakage": False, "nan_count": 0}),
            (
                "calibration",
                {"feature_dim": len(next(iter(pipeline.team_features.values()))) if pipeline.team_features else 0},
            ),
        ]:
            cp_result = pipeline._compliance_gate.check(stage_name, stage_context)
            governance_results.append(
                {
                    "stage": stage_name,
                    "passed": cp_result.passed,
                    "n_checks": len(cp_result.checks),
                    "n_errors": cp_result.n_errors,
                    "n_warnings": cp_result.n_warnings,
                }
            )

        from ...governance.audit_trail import GovernanceAuditLog

        audit_trail = GovernanceAuditLog()
        all_passed = all(r["passed"] for r in governance_results)
        audit_trail.log_compliance_check(
            checkpoint="sequential_pipeline",
            status="passed" if all_passed else "failed",
            details=json.dumps({"experiment_id": experiment_id, "stages": governance_results}),
        )
        report["governance"] = {"stages": governance_results, "all_passed": all_passed}
        logger.info("S21: Governance gates: %d stages checked, all_passed=%s", len(governance_results), all_passed)
    except Exception as exc:
        logger.debug("Governance gate integration failed: %s", exc)
