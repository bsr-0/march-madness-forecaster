"""Baseline model training — embeddings module."""


import logging

from ....data.models.game_flow import GameFlow
from ....ml.gnn.schedule_graph import ScheduleEdge, ScheduleGraph, compute_multi_hop_sos
from ....ml.transformer.game_sequence import GameEmbedding, SeasonSequence
from ....models.team import Team

# Optional imports — accessed via pipeline._optional_imports pattern
try:
    from ..._optional_imports import (
        BAYESIAN_BT_AVAILABLE,
        OPTUNA_AVAILABLE,
        SCALER_AVAILABLE,
        SKLEARN_AVAILABLE,
        SPREAD_MODEL_AVAILABLE,
        TOURNAMENT_SIGMA_AVAILABLE,
        BayesianBradleyTerry,
        BrierLightGBMTuner,
        EnsembleWeightOptimizer,
        LeaveOneYearOutCV,
        LightGBMTuner,
        LogisticRegression,
        LogisticTuner,
        SpreadRegressor,
        StandardScaler,
        TemporalCrossValidator,
        XGBoostTuner,
    )
except ImportError:
    pass

# BMA ensemble (Protocol v2, Section 3.2)
try:
    from ....ml.ensemble.bma import BayesianModelAveraging, BMAResult
    BMA_AVAILABLE = True
except ImportError:
    BMA_AVAILABLE = False

# Brier-objective LightGBM (Protocol Section 3.3, Phase 4)
try:
    from ....ml.ensemble.brier_objective import BrierLightGBMRanker
    BRIER_LGB_AVAILABLE = True
except ImportError:
    BRIER_LGB_AVAILABLE = False

# Calibration-first pipeline (Phase 4 research)
try:
    from ....ml.ensemble.calibration_first import CalibrationFirstPipeline
    CALIBRATION_FIRST_AVAILABLE = True
except ImportError:
    CALIBRATION_FIRST_AVAILABLE = False

logger = logging.getLogger(__name__)


from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _construct_schedule_graph(pipeline, teams: List[Team]) -> ScheduleGraph:
    team_ids = {pipeline._team_id(t.name) for t in teams}
    for flow in pipeline.all_game_flows:
        team_ids.add(flow.team1_id)
        team_ids.add(flow.team2_id)
    team_ids = sorted(team_ids)
    graph = ScheduleGraph(team_ids, temporal_decay=pipeline.config.gnn_temporal_decay)

    if pipeline.team_features:
        default_dim = len(next(iter(pipeline.team_features.values())))
    else:
        default_dim = 16
    default_features = np.zeros(default_dim, dtype=float)
    for team_id in team_ids:
        graph.set_team_features(team_id, pipeline.team_features.get(team_id, default_features))

    # Filter out tournament games AND validation-era games to prevent
    # leakage — the GNN graph should only contain regular-season results
    # from the training era.  Validation-era edges would let the GNN
    # learn from outcomes it is later evaluated on (Issue 2).
    boundary = pipeline._validation_sort_key_boundary
    pre_tournament_games = [
        g for g in pipeline.all_game_flows
        if not pipeline._is_tournament_game(getattr(g, "game_date", f"{pipeline.config.year}-01-01"))
        and (boundary is None
             or pipeline._game_sort_key(getattr(g, "game_date", f"{pipeline.config.year}-01-01")) < boundary)
    ]

    seen_games = set()
    for game in pre_tournament_games:
        if game.game_id in seen_games:
            continue
        seen_games.add(game.game_id)

        margin = game.lead_history[-1] if game.lead_history else 0

        # Compute xp_margin from proprietary metrics when possession-level xP is unavailable
        xp_margin = float(game.get_xp_margin())
        if abs(xp_margin) < 1e-6 and pipeline.proprietary_metrics:
            pm1 = pipeline.proprietary_metrics.get(game.team1_id)
            pm2 = pipeline.proprietary_metrics.get(game.team2_id)
            if pm1 is not None and pm2 is not None:
                xp_margin = float(
                    (pm1.offensive_xp_per_possession - pm2.defensive_xp_per_possession)
                    - (pm2.offensive_xp_per_possession - pm1.defensive_xp_per_possession)
                ) * 70.0  # scale to per-game margin (approx 70 possessions)

        graph.add_game(
            ScheduleEdge(
                game_id=game.game_id,
                team1_id=game.team1_id,
                team2_id=game.team2_id,
                actual_margin=float(margin),
                xp_margin=xp_margin,
                location_weight=float(getattr(game, "location_weight", 0.5)),
                game_date=str(getattr(game, "game_date", "2026-02-01")),
            )
        )

    return graph




def _run_gnn(pipeline, graph: ScheduleGraph) -> Dict:
    multi_hop = compute_multi_hop_sos(graph, hops=3)
    pagerank = graph.compute_pagerank_sos()
    training_era_teams = set()
    for edge in graph.edges:
        training_era_teams.add(edge.team1_id)
        training_era_teams.add(edge.team2_id)

    # GNN disabled — use fallback embedding from graph statistics.
    pipeline.gnn_embeddings = {}
    for team_id in graph.team_ids:
        pipeline.gnn_embeddings[team_id] = np.array([
            multi_hop.get(team_id, 0.0),
            pagerank.get(team_id, 0.0),
        ])

    # FIX M5: Defer SOS refinement (same as PyG path above).
    pipeline._sos_refinement_pending = (multi_hop, pagerank)

    # Fix 12: Validation-based confidence for fallback path.
    val_teams = [t for t in graph.team_ids if t not in training_era_teams]
    if val_teams and pipeline.feature_engineer.team_features:
        mh_preds = np.array([multi_hop.get(t, 0.0) for t in val_teams])
        actual_ems = np.array([
            getattr(pipeline.feature_engineer.team_features.get(t), "adj_efficiency_margin", 0.0) / 30.0
            for t in val_teams
        ])
        fallback_mse = float(np.mean((mh_preds - actual_ems) ** 2))
        pipeline.model_confidence["gnn"] = float(np.clip(1.0 / (1.0 + fallback_mse) * 0.7, 0.1, 0.4))
    else:
        pipeline.model_confidence["gnn"] = 0.35

    return {
        "enabled": False,
        "framework": "statistical_fallback",
        "nodes": graph.n_teams,
        "edges": len(graph.edges),
    }




def _apply_sos_refinement(pipeline, multi_hop: Dict[str, float], pagerank: Dict[str, float]) -> None:
    if not pipeline.feature_engineer.team_features:
        return
    pr_values = np.array(list(pagerank.values()) or [0.0], dtype=float)
    pr_mean = float(np.mean(pr_values))

    for team_id, feats in pipeline.feature_engineer.team_features.items():
        mh = float(multi_hop.get(team_id, 0.0))
        pr = float(pagerank.get(team_id, pr_mean))
        refined_sos = 0.5 * feats.sos_adj_em + 3.0 * mh + 12.0 * (pr - pr_mean)
        feats.sos_adj_em = float(refined_sos)

        # Expose PageRank and multi-hop as standalone features so the
        # ensemble can learn their weights independently rather than
        # relying on the hardcoded blend above.  The blend still
        # refines sos_adj_em for backward compatibility, but the raw
        # graph signals are now available as independent dimensions.
        feats.pagerank_sos = float(pr - pr_mean)
        feats.multi_hop_sos = float(mh)

        pipeline.team_features[team_id] = feats.to_vector(include_embeddings=False)




def _apply_win_quality_metrics(pipeline, graph: ScheduleGraph) -> None:
    """Compute and attach graph-theoretic win quality metrics to team features.

    These features capture *who you beat* and *how convincingly*, which
    traditional win-loss records miss entirely.  A 25-5 team with zero
    top-50 wins is fundamentally different from a 22-8 team with five
    top-25 wins — but both look similar in record-based features.

    The schedule graph has already been built from training-era games
    only (leakage-safe), so these metrics are valid for both training
    and inference.
    """
    if not pipeline.feature_engineer.team_features or not graph.edges:
        return

    win_quality = graph.compute_win_quality_metrics()

    for team_id, feats in pipeline.feature_engineer.team_features.items():
        metrics = win_quality.get(team_id, {})
        feats.best_win_percentile = float(metrics.get("best_win_percentile", 0.5))
        feats.paper_tiger_score = float(metrics.get("paper_tiger_score", 0.0))
        feats.dominance_ratio = float(metrics.get("dominance_ratio", 1.0))
        pipeline.team_features[team_id] = feats.to_vector(include_embeddings=False)

    n_enriched = sum(
        1 for tid in pipeline.feature_engineer.team_features
        if tid in win_quality
    )
    logger.info(
        "Win quality metrics: enriched %d/%d teams (best_win_pctile, paper_tiger, dominance)",
        n_enriched, len(pipeline.feature_engineer.team_features),
    )




def _run_transformer(pipeline, game_flows: Dict[str, List[GameFlow]]) -> Dict:
    sequences: Dict[str, SeasonSequence] = {}

    for team_id, games in game_flows.items():
        embeddings: List[GameEmbedding] = []
        # Filter out tournament games AND validation-era games to prevent
        # leakage — the transformer should only learn from training-era
        # regular-season sequences (Issue 3).
        boundary = pipeline._validation_sort_key_boundary
        pre_tournament = [
            g for g in games
            if not pipeline._is_tournament_game(getattr(g, "game_date", f"{pipeline.config.year}-01-01"))
            and (boundary is None
                 or pipeline._game_sort_key(getattr(g, "game_date", f"{pipeline.config.year}-01-01")) < boundary)
        ]
        ordered_games = sorted(
            pre_tournament,
            key=lambda g: (pipeline._game_sort_key(getattr(g, "game_date", f"{pipeline.config.year}-01-01")), g.game_id),
        )

        for idx, game in enumerate(ordered_games):
            is_team1 = game.team1_id == team_id
            opp_id = game.team2_id if is_team1 else game.team1_id
            margin = game.lead_history[-1] if game.lead_history else 0
            if not is_team1:
                margin *= -1

            team_poss = [p for p in game.possessions if p.team_id == team_id]
            opp_poss = [p for p in game.possessions if p.team_id == opp_id]

            off = 100.0 * (sum(p.actual_points for p in team_poss) / max(len(team_poss), 1))
            deff = 100.0 * (sum(p.actual_points for p in opp_poss) / max(len(opp_poss), 1))
            tempo = float(len(team_poss) + len(opp_poss)) / 2

            embeddings.append(
                GameEmbedding(
                    game_id=game.game_id,
                    team_id=team_id,
                    opponent_id=opp_id,
                    game_date=str(getattr(game, "game_date", f"{pipeline.config.year}-01-01")),
                    game_number=idx + 1,
                    offensive_efficiency=float(off),
                    defensive_efficiency=float(deff),
                    tempo=float(np.clip(tempo, 58, 82)),
                    margin=float(margin),
                    win=margin > 0,
                    is_conference_game=True,
                    is_neutral_site=True,
                    opponent_rank=120,
                )
            )

        if len(embeddings) >= 6:
            sequences[team_id] = SeasonSequence(team_id=team_id, games=embeddings)

    # Transformer disabled — use fallback from trend statistics.
    pipeline.transformer_embeddings = {}
    breakout_count = 0
    for team_id, seq in sequences.items():
        matrix = seq.to_matrix()
        trend = np.mean(np.diff(matrix[:, 0])) if len(matrix) > 1 else 0.0
        volatility = float(np.std(matrix[:, 3]))
        recent = float(np.mean(matrix[-5:, 0]))
        pipeline.transformer_embeddings[team_id] = np.array([trend, volatility, recent])
        if len(matrix) >= 10:
            early = float(np.mean(matrix[:5, 0]))
            late = float(np.mean(matrix[-5:, 0]))
            if late - early > 0.05:
                breakout_count += 1

    pipeline.model_confidence["transformer"] = 0.35
    return {
        "enabled": False,
        "framework": "trend_fallback",
        "teams": len(sequences),
        "breakout_windows_detected": breakout_count,
    }




def _train_embedding_projections(
    pipeline,
    game_flows: Dict[str, List[GameFlow]],
) -> Dict[str, float]:
    """Train logistic models that map embedding pairs to win probability.

    Uses slice 0 of the 3-way validation split.  Slices 1 and 2 are
    reserved for ensemble weight optimization and calibration
    respectively, preventing any data overlap (Issue 5).
    """
    stats: Dict[str, float] = {}
    if not SKLEARN_AVAILABLE:
        return stats

    train_games = pipeline._get_validation_era_games_slice(game_flows, slice_index=0, n_slices=3)
    if len(train_games) < 10:
        return stats

    for emb_name, embeddings in [
        ("gnn", pipeline.gnn_embeddings),
        ("transformer", pipeline.transformer_embeddings),
    ]:
        if not embeddings:
            continue

        X_rows, y_rows = [], []
        for g in train_games:
            v1 = embeddings.get(g.team1_id)
            v2 = embeddings.get(g.team2_id)
            if v1 is None or v2 is None:
                continue
            _outcome = pipeline._game_outcome(g)
            if _outcome is None:
                continue
            diff = v1 - v2
            interaction = v1 * v2
            X_rows.append(np.concatenate([diff, interaction]))
            y_rows.append(_outcome)
            # Symmetric sample
            X_rows.append(np.concatenate([v2 - v1, v2 * v1]))
            y_rows.append(1 - _outcome)

        if len(y_rows) < 20:
            continue

        X = np.array(X_rows)
        y = np.array(y_rows)

        lr = LogisticRegression(
            max_iter=500, C=1.0, solver="lbfgs", random_state=pipeline.config.random_seed
        )
        lr.fit(X, y)

        if emb_name == "gnn":
            pipeline._gnn_embedding_model = lr
        else:
            pipeline._transformer_embedding_model = lr

        stats[f"{emb_name}_projection_samples"] = len(y_rows)

    return stats

