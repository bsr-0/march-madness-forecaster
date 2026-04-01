"""Pipeline orchestration for TournamentPipeline.

_PipelineRunner owns the imperative execution logic:
  - pre-run validation
  - data loading (thin delegation to stages/data_loader)
  - feature engineering
  - model training (delegation to stages/baseline_training)
  - calibration, simulation, export
  - chronological train/val boundary computation

The class is internal (_-prefix).  TournamentPipeline delegates to it via
``self._runner``.  All state is read/written through ``self._p.*`` so there
is a single source of truth.
"""

from __future__ import annotations

import logging
import time as _time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from .tournament_pipeline import TournamentPipeline
    from ..data.models.game_flow import GameFlow
    from ..models.team import Team
    from ..data.models.player import Roster

logger = logging.getLogger(__name__)


class _PipelineRunner:
    """Orchestrates a TournamentPipeline run end-to-end."""

    def __init__(self, pipeline: "TournamentPipeline") -> None:
        self._p = pipeline

    # ------------------------------------------------------------------
    # Public entry points (called by TournamentPipeline)
    # ------------------------------------------------------------------

    def run(self) -> Dict:
        """Dispatch to calibration or EV mode and wrap with monitoring."""
        p = self._p
        self._pre_run_validation()

        _run_start = _time.perf_counter()
        _run_error = None
        try:
            if p.config.mode == "ev":
                result = self._run_ev_mode()
            else:
                result = self._run_calibration_mode()
        except Exception as exc:
            _run_error = exc
            self._log_run_to_history(
                status="error",
                duration=_time.perf_counter() - _run_start,
                error_message=str(exc),
            )
            raise

        brier = result.get("loyo_mean_brier") if isinstance(result, dict) else None
        self._log_run_to_history(
            status="success",
            duration=_time.perf_counter() - _run_start,
            brier_score=brier,
        )

        if brier is not None and hasattr(p, "_run_history"):
            regression_msg = p._run_history.check_regression(brier, mode=p.config.mode)
            if regression_msg:
                logger.warning("REGRESSION: %s", regression_msg)

        if hasattr(p, "_resource_tracker"):
            logger.info(p._resource_tracker.summary())
            p._resource_tracker.check_budget()

        if hasattr(p, "_cost_tracker") and brier is not None:
            try:
                usage = p._resource_tracker.to_dict()
                phase_costs = p._cost_tracker.compute_phase_costs(usage.get("phases", {}))
                p._cost_tracker.add_run(
                    brier_score=brier,
                    wall_seconds=usage.get("total_wall_seconds", 0),
                    cpu_seconds=usage.get("total_cpu_seconds", 0),
                    peak_memory_mb=usage.get("peak_memory_mb", 0),
                    phase_costs=phase_costs,
                )
                logger.info(p._cost_tracker.summary())
            except Exception as exc:
                logger.debug("Failed to log cost-performance: %s", exc)

        return result

    def train_for_predictions(self) -> None:
        """Train data → features → baseline → calibration; skip simulation/export."""
        from .stages import orchestration as _orch
        p = self._p
        total_stages = 4
        stage_idx = 0

        def _progress(msg: str) -> None:
            pct = int((stage_idx / total_stages) * 100)
            print(f"[train_for_predictions] {pct:>3}% ({stage_idx}/{total_stages}) {msg}", flush=True)

        t0 = _time.time()
        _progress("pre-run checks")
        _orch.run_pre_checks(p)

        stage_idx = 1
        _progress("data loading: start")
        t_stage = _time.time()
        teams = self._load_teams()
        torvik_map, proprietary_map = self._load_team_stat_sources(teams)
        rosters = self._build_rosters(teams)
        self._apply_injury_reports(rosters)
        game_flows = self._build_or_load_game_flows(teams)
        p._external_composites = self._load_external_ratings(teams)
        p._massey_multi = self._load_massey_multi_system()
        _progress(f"data loading: done ({_time.time() - t_stage:.1f}s, teams={len(teams)})")

        stage_idx = 2
        _progress("feature engineering: start")
        t_stage = _time.time()
        total_teams = max(len(teams), 1)
        feature_report_step = max(1, len(teams) // 10)
        for team in teams:
            team_id = p._team_id(team.name)
            p.team_struct[team_id] = team
            p.team_id_to_name[team_id] = team.name
            p.team_name_to_id[team.name] = team_id

            features = p.feature_engineer.extract_team_features(
                team_id=team_id, team_name=team.name,
                seed=team.seed, region=team.region,
                proprietary_metrics=proprietary_map.get(team_id, {}),
                torvik_data=torvik_map.get(team_id, {}),
                roster=rosters.get(team_id),
                games=game_flows.get(team_id, []),
            )
            comp = p._external_composites.get(team_id)
            if comp is not None:
                features.external_rating_composite = comp.composite_rating
                features.external_rating_spread = comp.rating_spread

            massey_feat = p._massey_multi.get(team_id)
            if massey_feat is not None:
                for sys_code, rating in massey_feat.system_ratings.items():
                    attr = f"massey_{sys_code.lower()}"
                    if hasattr(features, attr):
                        setattr(features, attr, rating)
                features.massey_rank_mean = massey_feat.rank_mean
                features.massey_rank_std = massey_feat.rank_std

            p.team_features[team_id] = features.to_vector(include_embeddings=False)
            idx = len(p.team_features)
            if idx % feature_report_step == 0 or idx == len(teams):
                print(f"[train_for_predictions] team-features {int((idx / total_teams) * 100):>3}% ({idx}/{len(teams)})", flush=True)

        self._compute_train_val_boundary(game_flows)
        self._construct_schedule_graph(teams)
        _progress(f"feature engineering: done ({_time.time() - t_stage:.1f}s, team_features={len(p.team_features)})")

        stage_idx = 3
        _progress("model training: start")
        t_stage = _time.time()
        self._train_baseline_model(game_flows)
        _progress(f"model training: done ({_time.time() - t_stage:.1f}s)")

        stage_idx = 4
        _progress("calibration: start")
        t_stage = _time.time()
        try:
            self._fit_calibration(game_flows)
        except Exception as exc:
            logger.warning("Calibration failed, using uncalibrated model: %s", exc)
        try:
            self._fit_massey_predictor(game_flows)
        except Exception as exc:
            logger.debug("Massey predictor fitting skipped: %s", exc)
        _progress(f"calibration: done ({_time.time() - t_stage:.1f}s)")

        feat_dim = next(iter(p.team_features.values())).shape[0] if p.team_features else 0
        logger.info("Pipeline trained for predictions (%d teams, %d features)", len(p.team_features), feat_dim)
        print(f"[train_for_predictions] 100% complete in {_time.time() - t0:.1f}s (teams={len(p.team_features)})", flush=True)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _pre_run_validation(self) -> Dict:
        """Pre-run checklist: data freshness, required files, config sanity."""
        from ..exceptions import PreRunValidationError
        from ..monitoring.pipeline_monitor import PipelineMonitor
        import os
        p = self._p
        cfg = p.config

        checks: list = []
        critical_failures: list = []

        # Data freshness
        if cfg.enforce_feed_freshness:
            monitor = PipelineMonitor()
            freshness = monitor.check_data_freshness(cfg.data_cache_dir)
            stale = [c for c in freshness if c.status == "stale"]
            missing = [c for c in freshness if c.status == "missing"]
            if missing:
                critical_failures.append(
                    f"Missing data sources ({len(missing)}): " + ", ".join(c.source for c in missing)
                )
                checks.append({"check": "data_freshness", "status": "CRITICAL", "stale_count": len(stale), "missing_count": len(missing), "stale_details": []})
            elif stale:
                logger.warning("Stale data sources (%d): %s", len(stale), [(c.source, f"{c.staleness_hours:.0f}h") for c in stale])
                checks.append({"check": "data_freshness", "status": "warn", "stale_count": len(stale), "missing_count": 0, "stale_details": [{"source": c.source, "hours": c.staleness_hours, "sla": c.sla_hours} for c in stale]})
            else:
                checks.append({"check": "data_freshness", "status": "pass", "stale_count": 0, "missing_count": 0, "stale_details": []})
        else:
            checks.append({"check": "data_freshness", "status": "skipped", "stale_count": 0, "missing_count": 0, "stale_details": []})

        # Required files
        for label, path in [("teams_json", cfg.teams_json), ("historical_games_json", cfg.historical_games_json)]:
            if path:
                if not os.path.exists(path):
                    critical_failures.append(f"Required input file missing: {label}={path}")
                    checks.append({"check": f"file_{label}", "status": "CRITICAL", "path": path})
                else:
                    checks.append({"check": f"file_{label}", "status": "pass", "path": path})

        # Config sanity
        if cfg.num_simulations < 1000:
            checks.append({"check": "mc_simulations", "status": "warn", "message": f"num_simulations={cfg.num_simulations} is low"})
        if cfg.holdout_years and cfg.dev_years:
            overlap = set(cfg.holdout_years) & set(cfg.dev_years)
            if overlap:
                critical_failures.append(f"Holdout/dev year overlap: {overlap}")
                checks.append({"check": "holdout_dev_overlap", "status": "CRITICAL", "overlap": list(overlap)})
        if cfg.random_seed == 0:
            checks.append({"check": "random_seed", "status": "warn", "message": "random_seed=0"})

        report = {"checks": checks, "critical_failures": critical_failures, "passed": len(critical_failures) == 0}
        if critical_failures:
            logger.error("Pre-run validation FAILED: %s", critical_failures)
            if cfg.strict_leakage_mode:
                raise PreRunValidationError(
                    f"Pre-run validation failed ({len(critical_failures)} issues): " + "; ".join(critical_failures)
                )
        else:
            logger.info("Pre-run validation passed (%d checks)", len(checks))
        return report

    # ------------------------------------------------------------------
    # Mode dispatchers
    # ------------------------------------------------------------------

    def _run_calibration_mode(self) -> Dict:
        return self._run_shared_pipeline()

    def _run_ev_mode(self) -> Dict:
        from .stages import ev_analysis as _ev
        report = self._run_shared_pipeline()
        ev_report = _ev._build_ev_analysis(self._p, report)
        report["ev_analysis"] = ev_report.to_dict()
        report["mode"] = "ev"
        self._p._last_ev_report = report
        return report

    # ------------------------------------------------------------------
    # Shared pipeline (the main orchestration body)
    # ------------------------------------------------------------------

    def _run_shared_pipeline(self) -> Dict:
        """Core pipeline: data → features → training → simulation → export."""
        from .stages import orchestration as _orch
        p = self._p

        p._audit_log.log_action("pipeline_start", "system", {
            "year": p.config.year, "mode": p.config.mode, "num_simulations": p.config.num_simulations,
        })

        freeze_verification = _orch.run_pre_checks(p)

        teams, game_flows, rosters, injury_stats = self._load_data()
        self._engineer_features(teams, game_flows)
        adjacency, gnn_stats, baseline_stats, transformer_stats, uncertainty_stats, embedding_proj_stats = self._train_models(teams, game_flows)
        calibration_stats, massey_predictor_stats = self._calibrate(game_flows)
        bracket_sim, schedule_graph = self._simulate(teams, rosters, game_flows)

        # Market cross-reference and blend
        from .stages import simulation as _sim
        market_consensus = _sim.load_betting_markets(p)
        if market_consensus is not None:
            if getattr(p.config, "enable_vegas_cross_reference", True):
                try:
                    from ..governance.market_validation import validate_model_vs_market
                    model_champ = dict(bracket_sim.championship_odds) if hasattr(bracket_sim, "championship_odds") else {}
                    if model_champ:
                        validation = validate_model_vs_market(
                            model_probs=model_champ,
                            market_probs=market_consensus.team_probabilities,
                            adjust_vig=True,
                        )
                        if validation is not None:
                            logger.info("Vegas cross-reference: RMSD=%.4f, Spearman=%.4f, interpretation=%s (%d teams)",
                                        validation.rmsd, validation.spearman_rank_corr or 0.0, validation.interpretation, validation.n_common_teams)
                            if validation.interpretation in ("significant_divergence", "major_disagreement"):
                                logger.warning("Vegas cross-reference flagged %s (RMSD=%.4f)", validation.interpretation, validation.rmsd)
                except Exception as exc:
                    logger.debug("Vegas cross-reference skipped: %s", exc)
            if p.config.enable_market_blend:
                _sim.apply_market_blend(p, bracket_sim, market_consensus)

        model_round_probs = _sim.to_round_probabilities(p, bracket_sim)
        mode_result = _orch.run_mode_gated_sections(p, teams, model_round_probs, game_flows)

        report = _orch.assemble_report(
            p, adjacency=adjacency, baseline_stats=baseline_stats,
            gnn_stats=gnn_stats, transformer_stats=transformer_stats,
            uncertainty_stats=uncertainty_stats, calibration_stats=calibration_stats,
            massey_predictor_stats=massey_predictor_stats, bracket_sim=bracket_sim,
            injury_stats=injury_stats, mode_result=mode_result,
            freeze_verification=freeze_verification,
            embedding_proj_stats=embedding_proj_stats, schedule_graph=schedule_graph,
        )
        _orch.run_post_pipeline(p, report, baseline_stats, calibration_stats)
        return report

    def _load_data(self):
        """Stage 1: load teams, stats, rosters, game flows, external ratings."""
        p = self._p
        with p._resource_tracker.phase("data_loading"):
            teams = self._load_teams()
            torvik_map, proprietary_map = self._load_team_stat_sources(teams)
            rosters = self._build_rosters(teams)
            injury_stats = self._apply_injury_reports(rosters)
            game_flows = self._build_or_load_game_flows(teams)
            p._external_composites = self._load_external_ratings(teams)
            p._massey_multi = self._load_massey_multi_system()
            p._torvik_map = torvik_map
            p._proprietary_map = proprietary_map
        if hasattr(p, "_compliance_runner"):
            p._compliance_runner.run_stage_checks("post_data_load", ctx=p)
            if p._compliance_runner.has_blocking_failure("post_data_load"):
                logger.error("Blocking compliance failure after data loading")
        return teams, game_flows, rosters, injury_stats

    def _engineer_features(self, teams, game_flows) -> None:
        """Stage 2: build per-team feature vectors."""
        from ..data.features.feature_engineering import validate_population_stats
        p = self._p
        torvik_map = getattr(p, "_torvik_map", {})
        proprietary_map = getattr(p, "_proprietary_map", {})
        with p._resource_tracker.phase("feature_engineering"):
            for team in teams:
                team_id = p._team_id(team.name)
                p.team_struct[team_id] = team
                p.team_id_to_name[team_id] = team.name
                p.team_name_to_id[team.name] = team_id

                features = p.feature_engineer.extract_team_features(
                    team_id=team_id, team_name=team.name,
                    seed=team.seed, region=team.region,
                    proprietary_metrics=proprietary_map.get(team_id, {}),
                    torvik_data=torvik_map.get(team_id, {}),
                    roster=getattr(p, "_rosters", {}).get(team_id),
                    games=game_flows.get(team_id, []),
                )
                comp = p._external_composites.get(team_id)
                if comp is not None:
                    features.external_rating_composite = comp.composite_rating
                    features.external_rating_spread = comp.rating_spread

                massey_feat = p._massey_multi.get(team_id)
                if massey_feat is not None:
                    for sys_code, rating in massey_feat.system_ratings.items():
                        attr = f"massey_{sys_code.lower()}"
                        if hasattr(features, attr):
                            setattr(features, attr, rating)
                    features.massey_rank_mean = massey_feat.rank_mean
                    features.massey_rank_std = massey_feat.rank_std

                p.team_features[team_id] = features.to_vector(include_embeddings=False)

            p._massey_coverage_stats = self._verify_massey_coverage(teams, p._external_composites)
            pop_warnings = validate_population_stats(p.feature_engineer.team_features)
            if pop_warnings:
                logger.warning("FIX#9: %d features diverged from population stats.", len(pop_warnings))
            self._compute_train_val_boundary(game_flows)

    def _train_models(self, teams, game_flows):
        """Stage 3: GNN (optional), baseline, transformer (optional), calibration."""
        from .stages import baseline_training as _bt
        from .stages import simulation as _sim
        p = self._p

        schedule_graph = self._construct_schedule_graph(teams)
        adjacency = schedule_graph.get_adjacency_matrix(weighted=True)

        with p._resource_tracker.phase("model_training"):
            if p.config.enable_gnn:
                gnn_stats = self._run_gnn(schedule_graph)
            else:
                gnn_stats = {"enabled": False, "reason": "disabled_by_config", "framework": "none"}

            baseline_stats = self._train_baseline_model(game_flows)

            if p.config.enable_transformer:
                transformer_stats = self._run_transformer(game_flows)
            else:
                transformer_stats = {"enabled": False, "teams": 0, "reason": "disabled_by_config"}

            if p._sos_refinement_pending is not None:
                mh, pr = p._sos_refinement_pending
                self._apply_sos_refinement(mh, pr)
                p._sos_refinement_pending = None
            self._apply_win_quality_metrics(schedule_graph)

            embedding_proj_stats = {}
            if p.config.enable_embedding_projections:
                embedding_proj_stats = self._train_embedding_projections(game_flows)
            uncertainty_stats = _sim.estimate_model_confidence_intervals(p, game_flows)

            p.feature_engineer.attach_gnn_embeddings(p.gnn_embeddings)
            p.feature_engineer.attach_transformer_embeddings(p.transformer_embeddings)

        p._audit_log.log_action("training_complete", "system", {
            "gnn_enabled": p.config.enable_gnn, "transformer_enabled": p.config.enable_transformer,
        })
        if hasattr(p, "_compliance_runner"):
            p._compliance_runner.run_stage_checks("post_training", ctx=p)
            if p._compliance_runner.has_blocking_failure("post_training"):
                logger.error("Blocking compliance failure after model training")

        if p.config.enable_budget_degradation and hasattr(p, "_resource_tracker"):
            utilization = p._resource_tracker.budget_utilization()
            if utilization > 0.8:
                logger.warning("BUDGET DEGRADATION: %.0f%% budget consumed", utilization * 100)
                p._runtime_state["num_simulations"] = max(1000, int(p._runtime_state["num_simulations"]) // 2)
                p._runtime_state["enable_gnn"] = False
                p._runtime_state["enable_transformer"] = False

        return adjacency, gnn_stats, baseline_stats, transformer_stats, uncertainty_stats, embedding_proj_stats

    def _calibrate(self, game_flows):
        """Stage 4: fit calibration pipeline and Massey predictor."""
        p = self._p
        with p._resource_tracker.phase("calibration"):
            calibration_stats = self._fit_calibration(game_flows)
            massey_predictor_stats = self._fit_massey_predictor(game_flows)
        return calibration_stats, massey_predictor_stats

    def _simulate(self, teams, rosters, game_flows):
        """Stage 5: Monte Carlo bracket simulation."""
        from .stages import simulation as _sim
        p = self._p
        schedule_graph = self._construct_schedule_graph(teams)
        with p._resource_tracker.phase("simulation"):
            bracket_sim = _sim.run_monte_carlo(p, teams, rosters)
        p._audit_log.log_action("simulation_complete", "system", {"num_simulations": p.config.num_simulations})
        return bracket_sim, schedule_graph

    # ------------------------------------------------------------------
    # Run history
    # ------------------------------------------------------------------

    def _log_run_to_history(
        self,
        status: str,
        duration: float,
        brier_score: Optional[float] = None,
        error_message: Optional[str] = None,
    ) -> None:
        p = self._p
        if not hasattr(p, "_run_history"):
            return
        try:
            from ..monitoring.run_history import RunRecord
            resource_usage = p._resource_tracker.to_dict() if hasattr(p, "_resource_tracker") else {}
            record = RunRecord(
                mode=p.config.mode,
                year=p.config.year,
                status=status,
                duration_seconds=round(duration, 1),
                resource_usage=resource_usage,
                brier_score=brier_score,
                error_message=error_message,
            )
            p._run_history.log_run(record)
        except Exception as exc:
            logger.debug("Failed to log run to history: %s", exc)

    # ------------------------------------------------------------------
    # Chronological boundary helpers
    # ------------------------------------------------------------------

    def _filter_years(self, years: List[int], include_holdout: bool = False) -> List[int]:
        p = self._p
        cfg = p.config
        if not years:
            return []
        year_set = sorted({y for y in years if y != 2020})
        if cfg.dev_years:
            dev_set = set(cfg.dev_years)
            if include_holdout and cfg.holdout_years:
                dev_set = dev_set | set(cfg.holdout_years)
            year_set = [y for y in year_set if y in dev_set]
        if cfg.holdout_years and not include_holdout:
            holdout_set = set(cfg.holdout_years)
            year_set = [y for y in year_set if y not in holdout_set]
        return year_set

    def _load_mc_calibration(self) -> Optional[Dict]:
        import os
        import json as _json
        p = self._p
        path = p.config.mc_calibration_json
        if path is None:
            default_path = os.path.join(os.getcwd(), "data", "raw", "mc_calibration.json")
            if os.path.isfile(default_path):
                path = default_path
        if not path or not os.path.isfile(path):
            return None
        try:
            with open(path) as f:
                payload = _json.load(f)
        except Exception as exc:
            logger.debug("Failed to load MC calibration from %s: %s", path, exc)
            return None
        best = payload.get("best_params", {})
        if isinstance(best, dict):
            if "noise_std" in best:
                p._runtime_state["mc_noise_std"] = float(best["noise_std"])
            if "regional_correlation" in best:
                p._runtime_state["mc_regional_correlation"] = float(best["regional_correlation"])
        payload["_source_path"] = path
        return payload

    def _compute_train_val_boundary(self, game_flows: Dict[str, List]) -> None:
        """Establish train/val chronological boundary (80/20 split)."""
        p = self._p
        all_games = sorted(
            [
                g for g in self._unique_games(game_flows)
                if not self._is_tournament_game(getattr(g, "game_date", f"{p.config.year}-01-01"))
                and g.team1_id in p.feature_engineer.team_features
                and g.team2_id in p.feature_engineer.team_features
            ],
            key=lambda g: (self._game_sort_key(getattr(g, "game_date", f"{p.config.year}-01-01")), g.game_id),
        )
        n = len(all_games)
        if n < 25:
            p._validation_sort_key_boundary = None
            return
        valid_count = max(5, int(0.2 * n))
        train_count = n - valid_count
        if train_count < 10:
            p._validation_sort_key_boundary = None
            return
        p._validation_sort_key_boundary = self._game_sort_key(
            getattr(all_games[train_count], "game_date", f"{p.config.year}-01-01")
        )

    def _get_validation_era_games(self, game_flows: Dict[str, List]) -> List:
        p = self._p
        all_games = sorted(
            [
                g for g in self._unique_games(game_flows)
                if not self._is_tournament_game(getattr(g, "game_date", f"{p.config.year}-01-01"))
                and g.team1_id in p.feature_engineer.team_features
                and g.team2_id in p.feature_engineer.team_features
            ],
            key=lambda g: (self._game_sort_key(getattr(g, "game_date", f"{p.config.year}-01-01")), g.game_id),
        )
        if p._validation_sort_key_boundary is not None:
            return [
                g for g in all_games
                if self._game_sort_key(getattr(g, "game_date", f"{p.config.year}-01-01"))
                >= p._validation_sort_key_boundary
            ]
        n = len(all_games)
        start = max(0, n - max(10, int(0.2 * n)))
        return all_games[start:]

    def _get_validation_era_games_slice(self, game_flows: Dict[str, List], slice_index: int, n_slices: int = 3) -> List:
        all_val = self._get_validation_era_games(game_flows)
        n = len(all_val)
        if n < n_slices * 10:
            if n_slices == 3:
                mid = n // 2
                if slice_index == 0:
                    return all_val[:mid]
                elif slice_index == 2:
                    return all_val[mid:]
                return []
            return all_val if slice_index == 0 else []
        slice_size = n // n_slices
        start = slice_index * slice_size
        end = n if slice_index == n_slices - 1 else start + slice_size
        return all_val[start:end]

    # ------------------------------------------------------------------
    # Data loading delegation (thin wrappers → stages/data_loader)
    # ------------------------------------------------------------------

    def _load_teams(self) -> List:
        from .stages import data_loader as _dl
        return _dl.load_teams(self._p.config, self._p.bracket_pipeline)

    def _load_team_stat_sources(self, teams: List) -> Tuple:
        from .stages import data_loader as _dl
        result = _dl.load_team_stat_sources(self._p.config, teams, self._p.proprietary_engine)
        p = self._p
        p._prior_year_elo = result.prior_year_elo
        p._current_year_game_records = result.current_year_game_records
        p._current_year_conference_map = result.current_year_conference_map
        p.proprietary_metrics = result.proprietary_metrics
        p._torvik_name_to_id = result.torvik_name_to_id
        p._cbbpy_map = result.cbbpy_map
        p._mascot_cache = result.mascot_cache
        p._resolve_to_canonical = result.resolve_to_canonical
        return result.torvik_map, result.proprietary_map

    def _build_rosters(self, teams: List) -> Dict:
        from .stages import data_loader as _dl
        result = _dl.build_rosters(self._p.config, teams)
        self._p.roster_rapm_quality = result.roster_rapm_quality
        return result.rosters

    def _apply_injury_reports(self, rosters: Dict) -> Dict:
        from .stages import data_loader as _dl
        p = self._p
        stats, positional_impacts = _dl.apply_injury_reports(
            p.config, rosters, p.positional_depth_chart, p.injury_severity_model,
        )
        p.positional_impacts.update(positional_impacts)
        return stats

    def _build_or_load_game_flows(self, teams: List) -> Dict:
        from .stages import data_loader as _dl
        p = self._p
        result = _dl.build_or_load_game_flows(
            p.config, teams,
            resolve_to_canonical=getattr(p, "_resolve_to_canonical", None),
            mascot_cache=getattr(p, "_mascot_cache", None),
        )
        p.all_game_flows = result.all_game_flows
        return result.team_to_games

    def _load_external_ratings(self, teams: List) -> Dict:
        from .stages import data_loader as _dl
        return _dl.load_external_ratings(self._p.config, teams)

    def _load_massey_multi_system(self) -> Dict:
        from .stages import data_loader as _dl
        return _dl.load_massey_multi_system(self._p.config)

    def _verify_massey_coverage(self, teams: List, composites: Dict) -> Dict:
        from .stages import data_loader as _dl
        return _dl.verify_massey_coverage(teams, composites, self._p.feature_engineer.team_features)

    # ------------------------------------------------------------------
    # Training delegation (thin wrappers → stages/baseline_training)
    # ------------------------------------------------------------------

    def _construct_schedule_graph(self, teams: List):
        from .stages import baseline_training as _bt
        return _bt._construct_schedule_graph(self._p, teams)

    def _train_baseline_model(self, game_flows: Dict) -> Dict:
        from .stages import baseline_training as _bt
        return _bt._train_baseline_model(self._p, game_flows)

    def _run_gnn(self, graph) -> Dict:
        from .stages import baseline_training as _bt
        return _bt._run_gnn(self._p, graph)

    def _apply_sos_refinement(self, multi_hop: Dict, pagerank: Dict) -> None:
        from .stages import baseline_training as _bt
        return _bt._apply_sos_refinement(self._p, multi_hop, pagerank)

    def _apply_win_quality_metrics(self, graph) -> None:
        from .stages import baseline_training as _bt
        return _bt._apply_win_quality_metrics(self._p, graph)

    def _run_transformer(self, game_flows: Dict) -> Dict:
        from .stages import baseline_training as _bt
        return _bt._run_transformer(self._p, game_flows)

    def _train_embedding_projections(self, game_flows: Dict) -> Dict:
        from .stages import baseline_training as _bt
        return _bt._train_embedding_projections(self._p, game_flows)

    def _optimize_ensemble_weights_loyo(self, baseline_model: Any, feature_selector: Any = None) -> Dict:
        from .stages import baseline_training as _bt
        return _bt._optimize_ensemble_weights_loyo(self._p, baseline_model, feature_selector)

    # ------------------------------------------------------------------
    # Calibration delegation
    # ------------------------------------------------------------------

    def _fit_calibration(self, game_flows: Dict) -> Dict:
        from .stages import calibration as _cal
        return _cal._fit_calibration(self._p, game_flows)

    def _fit_massey_predictor(self, game_flows: Dict) -> Dict:
        from .stages import calibration as _cal
        return _cal._fit_massey_predictor(self._p, game_flows)

    def _fit_tournament_sigma(self, spread_model: Any, tuning_stats: Dict) -> None:
        from .stages import calibration as _cal
        return _cal._fit_tournament_sigma(self._p, spread_model, tuning_stats)

    # ------------------------------------------------------------------
    # Simulation delegation
    # ------------------------------------------------------------------

    def _run_monte_carlo(self, teams: List, rosters: Dict):
        from .stages import simulation as _sim
        return _sim.run_monte_carlo(self._p, teams, rosters)

    def _unique_games(self, game_flows: Dict) -> List:
        from .stages import simulation as _sim
        return _sim.unique_games(self._p, game_flows)

    def _is_tournament_game(self, date_str: str) -> bool:
        from .stages.game_utils import detect_tournament_game as _gu_detect_tournament_game
        return _gu_detect_tournament_game(date_str, fallback_year=self._p.config.year)

    def _game_sort_key(self, date_str: str) -> int:
        from .stages.game_utils import compute_game_sort_key as _gu_compute_game_sort_key
        return _gu_compute_game_sort_key(date_str, fallback_year=self._p.config.year)
