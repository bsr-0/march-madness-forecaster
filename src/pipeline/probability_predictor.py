"""Win-probability inference for TournamentPipeline.

All predict_probability* variants, their shared raw-fusion core, and the
experimental adaptation layers live here.  No data loading, no training.

The class is internal (``_`` prefix) — callers should use the public
``TournamentPipeline.predict_probability*`` methods.
"""

from __future__ import annotations

import math
import logging
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from .tournament_pipeline import TournamentPipeline

logger = logging.getLogger(__name__)

from .config import BT_BLEND_WEIGHT as _BT_BLEND_WEIGHT


class _ProbabilityPredictor:
    """Inference engine that wraps a trained TournamentPipeline.

    Instantiated by TournamentPipeline and given a back-reference so it can
    read all fitted artifacts without owning or copying them.
    """

    def __init__(self, pipeline: "TournamentPipeline") -> None:
        self._p = pipeline

    # ------------------------------------------------------------------
    # Public routing
    # ------------------------------------------------------------------

    def predict_probability(self, team1_id: str, team2_id: str) -> float:
        """Route to production, pool, or experimental path based on config."""
        cfg = self._p.config
        if cfg.probability_profile == "experimental":
            return self.predict_probability_experimental(team1_id, team2_id)
        if cfg.probability_profile == "pool":
            return self.predict_probability_pool(team1_id, team2_id)
        return self.predict_probability_production(team1_id, team2_id)

    # ------------------------------------------------------------------
    # Production path
    # ------------------------------------------------------------------

    def predict_probability_production(self, team1_id: str, team2_id: str) -> float:
        """Production path: raw → symmetrize → calibrate → shrink → clip.

        No seed overrides, no sharpening, no goto_conversion.  This is the
        path that must be stable across re-runs for governance hash checks.
        """
        from .probability_pipeline import (
            apply_calibration,
            apply_final_clip,
            apply_tournament_shrinkage,
        )
        p = self._p
        cfg = p.config

        raw_forward = self._raw_fusion_probability(team1_id, team2_id)
        raw_reverse = self._raw_fusion_probability(team2_id, team1_id)
        raw = (raw_forward + (1.0 - raw_reverse)) / 2.0

        prob = apply_calibration(
            raw, p.calibration_pipeline,
            cfg.pre_calibration_clip_lo, cfg.pre_calibration_clip_hi,
        )

        if cfg.enable_tournament_adaptation:
            prob = apply_tournament_shrinkage(prob, cfg.tournament_shrinkage)

        return apply_final_clip(prob, cfg.pre_calibration_clip_lo, cfg.pre_calibration_clip_hi)

    # ------------------------------------------------------------------
    # Pool path
    # ------------------------------------------------------------------

    def predict_probability_pool(self, team1_id: str, team2_id: str) -> float:
        """Pool path: raw → symmetrize → calibrate → clip.  No shrinkage.

        Tournament shrinkage flattens the signals that pool contrarian
        strategy depends on, so it is intentionally omitted here.
        """
        from .probability_pipeline import apply_calibration, apply_final_clip
        p = self._p
        cfg = p.config

        raw_forward = self._raw_fusion_probability(team1_id, team2_id)
        raw_reverse = self._raw_fusion_probability(team2_id, team1_id)
        raw = (raw_forward + (1.0 - raw_reverse)) / 2.0

        prob = apply_calibration(
            raw, p.calibration_pipeline,
            cfg.pre_calibration_clip_lo, cfg.pre_calibration_clip_hi,
        )
        return apply_final_clip(prob, cfg.pre_calibration_clip_lo, cfg.pre_calibration_clip_hi)

    # ------------------------------------------------------------------
    # Experimental path
    # ------------------------------------------------------------------

    def predict_probability_experimental(self, team1_id: str, team2_id: str) -> float:
        """Experimental path — all optional layers active.

        Includes: round-weighted calibrator, BrierPostProcessor (seed overrides,
        goto_conversion, sharpening), FLB correction, seed prior, consistency
        bonus.  Only used when probability_profile == "experimental".
        """
        p = self._p
        cfg = p.config

        raw_forward = self._raw_fusion_probability(team1_id, team2_id)
        raw_reverse = self._raw_fusion_probability(team2_id, team1_id)
        raw = (raw_forward + (1.0 - raw_reverse)) / 2.0

        # Calibration: round-weighted preferred, standard fallback
        rw_cal = getattr(p, "_round_weighted_calibrator", None)
        if rw_cal is not None:
            logit = np.log(max(raw, 1e-8) / max(1.0 - raw, 1e-8))
            raw = float(np.clip(
                1.0 / (1.0 + np.exp(-logit / max(rw_cal.temperature, 0.01))),
                cfg.pre_calibration_clip_lo, cfg.pre_calibration_clip_hi,
            ))
        elif p.calibration_pipeline:
            calibrated = float(p.calibration_pipeline.calibrate(np.array([raw]))[0])
            raw = float(np.clip(calibrated, cfg.pre_calibration_clip_lo, cfg.pre_calibration_clip_hi))

        if cfg.enable_tournament_adaptation:
            raw = self._tournament_adapt_experimental(raw, team1_id, team2_id)

        _will_use_pp = (
            cfg.enable_seed_overrides
            and getattr(p, "_brier_post_processor", None) is not None
        )

        if not _will_use_pp:
            flb = getattr(p, "_flb_correction", None)
            if flb is not None and flb.fitted:
                raw = flb.correct_single(raw)

        if _will_use_pp:
            t1 = p.feature_engineer.team_features.get(team1_id)
            t2 = p.feature_engineer.team_features.get(team2_id)
            raw = p._brier_post_processor.process(
                raw,
                seed1=t1.seed if t1 else 0,
                seed2=t2.seed if t2 else 0,
                is_womens=False,
            )

        return raw

    # ------------------------------------------------------------------
    # Core ensemble blend
    # ------------------------------------------------------------------

    def _raw_fusion_probability(self, team1_id: str, team2_id: str) -> float:
        """Compute pre-calibration ensemble probability for team1 beating team2.

        Pipeline: feature matchup → baseline model → optional BT blend →
        optional Massey blend → pre-calibration clip.
        """
        p = self._p
        cfg = p.config

        matchup = p.feature_engineer.create_matchup_features(
            team1_id, team2_id, proprietary_engine=p.proprietary_engine
        )
        feat_vec = matchup.to_vector()

        if p.feature_selector is not None and p.feature_selector.is_fitted:
            feat_vec = p.feature_selector.transform(feat_vec.reshape(1, -1))[0]

        baseline_prob = p.baseline_model.predict_proba(feat_vec)

        # Bayesian Bradley-Terry blend (15% max)
        if p.bayesian_bt_model is not None:
            bt_prob, bt_unc = p.bayesian_bt_model.predict_probability(team1_id, team2_id)
            bt_weight = _BT_BLEND_WEIGHT * max(0.0, 1.0 - bt_unc)
            baseline_prob = (1.0 - bt_weight) * baseline_prob + bt_weight * bt_prob

        # Massey composite blend
        if (
            cfg.massey_blend_weight > 0
            and hasattr(p, "_external_composites")
            and p._external_composites
        ):
            c1 = p._external_composites.get(team1_id)
            c2 = p._external_composites.get(team2_id)
            if c1 is not None and c2 is not None:
                diff = c1.composite_rating - c2.composite_rating
                sigma = cfg.massey_sigma
                w = cfg.massey_blend_weight
                massey_pred = getattr(p, "_massey_predictor", None)
                if massey_pred is not None and massey_pred.fitted:
                    sigma = massey_pred.sigma
                    w = massey_pred.blend_weight
                massey_prob = 1.0 / (1.0 + math.exp(-diff / max(sigma, 0.01)))
                baseline_prob = (1.0 - w) * baseline_prob + w * massey_prob

        return float(np.clip(baseline_prob, cfg.pre_calibration_clip_lo, cfg.pre_calibration_clip_hi))

    # ------------------------------------------------------------------
    # Experimental adaptation helpers
    # ------------------------------------------------------------------

    def _tournament_adapt_experimental(
        self, prob: float, team1_id: str, team2_id: str
    ) -> float:
        """Shrinkage + seed prior + consistency bonus (experimental only)."""
        p = self._p
        cfg = p.config

        t1 = p.feature_engineer.team_features.get(team1_id)
        t2 = p.feature_engineer.team_features.get(team2_id)

        adapted = cfg.tournament_shrinkage * 0.5 + (1.0 - cfg.tournament_shrinkage) * prob

        if cfg.seed_prior_weight > 0 and t1 is not None and t2 is not None:
            seed1 = t1.seed if t1 is not None else 0
            seed2 = t2.seed if t2 is not None else 0
            if seed1 > 0 and seed2 > 0:
                seed_diff = seed2 - seed1
                seed_prior = 1.0 / (1.0 + math.exp(-cfg.seed_prior_slope * seed_diff))
                adapted = (1.0 - cfg.seed_prior_weight) * adapted + cfg.seed_prior_weight * seed_prior

        if t1 is not None and t2 is not None:
            consistency_edge = cfg.consistency_bonus_max * np.clip(
                (t2.pace_adjusted_variance - t1.pace_adjusted_variance) / cfg.consistency_normalizer,
                -1.0, 1.0,
            )
            adapted += consistency_edge

        return float(np.clip(adapted, cfg.pre_calibration_clip_lo, cfg.pre_calibration_clip_hi))

    def _embedding_probability(
        self,
        v1,
        v2,
        model_type: str = "gnn",
    ) -> float:
        """Convert embedding pair → win probability via learned projection."""
        from .stages.inference import embedding_probability as _inf_embedding_probability
        p = self._p
        proj = (
            p._gnn_embedding_model
            if model_type == "gnn"
            else p._transformer_embedding_model
        )
        return _inf_embedding_probability(v1, v2, projection_model=proj, model_type=model_type)
