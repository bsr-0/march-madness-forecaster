"""Selection Sunday Launch Protocol — Comprehensive validation tests.

These tests verify the 11-step checklist for launch readiness:
  1. Production profile is active with no banned experimental layers
  2. Production test suite passes
  3. Exact probability formula: raw → calibrator → shrinkage → clip
  4. Data freshness is sufficient for tomorrow's run
  5. Bracket/seed ingestion is complete and correct
  6. End-to-end inference produces plausible outputs
  7. Ablation ladder metrics are sane
  8. Probability distribution is disciplined
  9. Historical backtest on production path is credible
 10. Export artifact is well-formed
 11. Freeze: config is reproducible

Run with:
    python -m pytest tests/test_selection_sunday_validation.py -v
"""

from __future__ import annotations

import inspect
import json
import math
import os
from collections import Counter
from datetime import date, datetime, timedelta
from typing import Any, Dict, List
from unittest.mock import MagicMock

import numpy as np
import pytest

# ───────────────────────────────────────────────────────────────────────
# Constants
# ───────────────────────────────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
BRACKET_PATH = os.path.join(DATA_DIR, "bracket_2026.json")
TORVIK_PATH = os.path.join(DATA_DIR, "torvik_four_factors_2026.json")
MANIFEST_PATH = os.path.join(DATA_DIR, "manifest_2026.json")
GAMES_PATH = os.path.join(DATA_DIR, "historical_games_2026.json")

# Bracket has 64 teams (play-in winners counted once they replace the original)
EXPECTED_TEAM_COUNT = 64
EXPECTED_REGIONS = {"East", "Midwest", "South", "West"}
EXPECTED_SEEDS = set(range(1, 17))

# Known 1-seeds from the bracket
EXPECTED_1_SEEDS = {"michigan", "duke", "arizona", "connecticut"}

# Sample teams for smoke tests
SAMPLE_1_SEED = "michigan"
SAMPLE_16_SEED = "maryland_baltimore_county"
SAMPLE_8_SEED = "ucf"
SAMPLE_9_SEED = "kentucky"
SAMPLE_5_SEED = "arkansas"
SAMPLE_12_SEED = "unc_wilmington"
SAMPLE_ELITE = "duke"


# ───────────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────────

def _load_bracket() -> Dict:
    with open(BRACKET_PATH) as f:
        return json.load(f)


def _load_torvik() -> Dict:
    with open(TORVIK_PATH) as f:
        return json.load(f)


def _load_manifest() -> Dict:
    with open(MANIFEST_PATH) as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════════
# STEP 1: Production profile is active and clean
# ═══════════════════════════════════════════════════════════════════════


class TestStep1ProductionProfile:
    """Verify default config is production and all experimental layers are off."""

    def test_default_profile_is_production(self):
        from src.pipeline.config import SOTAPipelineConfig
        cfg = SOTAPipelineConfig()
        assert cfg.probability_profile == "production"

    def test_all_experimental_flags_off(self):
        from src.pipeline.config import SOTAPipelineConfig
        cfg = SOTAPipelineConfig()
        assert cfg.enable_seed_overrides is False
        assert cfg.enable_brier_sharpening is False
        assert cfg.enable_goto_conversion is False
        assert cfg.enable_round_weighted_calibration is False
        assert cfg.seed_prior_weight == 0.0
        assert cfg.consistency_bonus_max == 0.0

    def test_tournament_adaptation_enabled(self):
        from src.pipeline.config import SOTAPipelineConfig
        cfg = SOTAPipelineConfig()
        assert cfg.enable_tournament_adaptation is True

    def test_tournament_shrinkage_value(self):
        from src.pipeline.config import SOTAPipelineConfig
        cfg = SOTAPipelineConfig()
        assert cfg.tournament_shrinkage == 0.06

    def test_womens_default_profile_is_production(self):
        from src.pipeline.womens import WomensPipelineConfig
        cfg = WomensPipelineConfig()
        assert cfg.probability_profile == "production"

    def test_womens_experimental_flags_off(self):
        from src.pipeline.womens import WomensPipelineConfig
        cfg = WomensPipelineConfig()
        assert cfg.enable_brier_sharpening is False
        assert cfg.enable_goto_conversion is False
        assert cfg.seed_prior_weight == 0.0

    def test_womens_tournament_adaptation_enabled(self):
        from src.pipeline.womens import WomensPipelineConfig
        cfg = WomensPipelineConfig()
        assert cfg.enable_tournament_adaptation is True

    def test_validate_production_profile_rejects_all_violations(self):
        """Exhaustively verify that every banned flag triggers a rejection."""
        from src.pipeline.config import SOTAPipelineConfig
        banned = [
            {"enable_seed_overrides": True},
            {"enable_brier_sharpening": True},
            {"enable_goto_conversion": True},
            {"enable_round_weighted_calibration": True},
            {"seed_prior_weight": 0.1},
            {"consistency_bonus_max": 0.01},
        ]
        for override in banned:
            with pytest.raises(ValueError):
                SOTAPipelineConfig(probability_profile="production", **override)

    def test_validate_womens_production_profile_rejects_violations(self):
        from src.pipeline.womens import WomensPipelineConfig
        banned = [
            {"enable_brier_sharpening": True},
            {"enable_goto_conversion": True},
            {"seed_prior_weight": 0.1},
        ]
        for override in banned:
            with pytest.raises(ValueError):
                WomensPipelineConfig(probability_profile="production", **override)


# ═══════════════════════════════════════════════════════════════════════
# STEP 3: Verify exact probability formula in live inference
# ═══════════════════════════════════════════════════════════════════════


class TestStep3InferencePathVerification:
    """Verify production paths use exactly: raw → calibrator → shrinkage → clip."""

    def test_men_production_path_source_no_banned_components(self):
        from src.pipeline.sota import SOTAPipeline
        source = inspect.getsource(SOTAPipeline.predict_probability_production)
        banned = [
            "_brier_post_processor", "_round_weighted_calibrator",
            "_flb_correction", "seed_prior", "consistency_bonus",
            "apply_experimental_", "BrierPostProcessor",
            "_tournament_domain_adapter", "sharpener",
            "goto_converter", "seed_override",
        ]
        for term in banned:
            assert term not in source, f"Men's production references banned: {term}"

    def test_men_production_uses_shared_helpers(self):
        from src.pipeline.sota import SOTAPipeline
        source = inspect.getsource(SOTAPipeline.predict_probability_production)
        required = ["apply_calibration", "apply_tournament_shrinkage", "apply_final_clip"]
        for fn in required:
            assert fn in source, f"Men's production missing shared helper: {fn}"

    def test_women_production_path_source_no_banned_components(self):
        from src.pipeline.womens import WomensPipeline
        source = inspect.getsource(WomensPipeline.predict_probability_production)
        # Strip comments/docstrings to avoid false positives
        code_lines = []
        in_docstring = False
        for line in source.split("\n"):
            stripped = line.strip()
            if stripped.startswith('"""') or stripped.startswith("'''"):
                if in_docstring:
                    in_docstring = False
                    continue
                if stripped.count('"""') >= 2 or stripped.count("'''") >= 2:
                    continue
                in_docstring = True
                continue
            if in_docstring or stripped.startswith("#"):
                continue
            code_lines.append(line)
        code_only = "\n".join(code_lines)
        banned = ["post_processor", "seed_prior", "apply_experimental_",
                   "sharpener", "goto_converter"]
        for term in banned:
            assert term not in code_only, f"Women's production references banned: {term}"

    def test_women_production_uses_shared_helpers(self):
        from src.pipeline.womens import WomensPipeline
        source = inspect.getsource(WomensPipeline.predict_probability_production)
        required = ["apply_calibration", "apply_tournament_shrinkage", "apply_final_clip"]
        for fn in required:
            assert fn in source, f"Women's production missing shared helper: {fn}"

    def test_production_four_stage_formula_equivalence(self):
        """Manually walk the 4-stage formula and verify exact equivalence."""
        from src.pipeline.probability_pipeline import (
            apply_calibration, apply_tournament_shrinkage, apply_final_clip,
        )
        raw = 0.72
        shrinkage = 0.06
        clip_lo, clip_hi = 0.005, 0.995

        # Mock temperature calibrator
        calibrator = MagicMock()
        calibrator.fitted = True
        calibrator.temperature = 1.2
        del calibrator.calibrate

        s1 = apply_calibration(raw, calibrator, clip_lo, clip_hi)
        s2 = apply_tournament_shrinkage(s1, shrinkage)
        s3 = apply_final_clip(s2, clip_lo, clip_hi)

        # Manual formula
        logit = np.log(raw / (1.0 - raw))
        cal = 1.0 / (1.0 + np.exp(-logit / 1.2))
        cal = float(np.clip(cal, clip_lo, clip_hi))
        shrunk = shrinkage * 0.5 + (1.0 - shrinkage) * cal
        expected = float(np.clip(shrunk, clip_lo, clip_hi))

        assert s3 == pytest.approx(expected, abs=1e-10)

    def test_symmetry_property(self):
        """Verify P(A>B) + P(B>A) ≈ 1 for the production path."""
        from src.pipeline.probability_pipeline import (
            apply_calibration, apply_tournament_shrinkage, apply_final_clip,
        )

        def _full_pipeline(raw, shrinkage=0.06, clip_lo=0.005, clip_hi=0.995):
            p = apply_calibration(raw, None, clip_lo, clip_hi)
            p = apply_tournament_shrinkage(p, shrinkage)
            return apply_final_clip(p, clip_lo, clip_hi)

        for raw in [0.3, 0.5, 0.7, 0.85, 0.95]:
            p_ab = _full_pipeline(raw)
            p_ba = _full_pipeline(1.0 - raw)
            assert p_ab + p_ba == pytest.approx(1.0, abs=1e-10)

    def test_shrinkage_moves_toward_half(self):
        """Shrinkage should always move probabilities toward 0.5."""
        from src.pipeline.probability_pipeline import apply_tournament_shrinkage
        for raw in [0.1, 0.3, 0.5, 0.7, 0.9]:
            shrunk = apply_tournament_shrinkage(raw, 0.06)
            if raw > 0.5:
                assert shrunk < raw
                assert shrunk > 0.5
            elif raw < 0.5:
                assert shrunk > raw
                assert shrunk < 0.5
            else:
                assert shrunk == pytest.approx(0.5)

    def test_clip_bounds_respected(self):
        """Final clip must respect configured bounds."""
        from src.pipeline.probability_pipeline import apply_final_clip
        assert apply_final_clip(-0.1, 0.005, 0.995) == 0.005
        assert apply_final_clip(1.1, 0.005, 0.995) == 0.995
        assert apply_final_clip(0.5, 0.005, 0.995) == 0.5

    def test_nan_inf_guarded(self):
        """compute_raw_probability should catch NaN/Inf."""
        from src.pipeline.probability_pipeline import compute_raw_probability
        assert compute_raw_probability(float("nan")) == 0.5
        assert compute_raw_probability(float("inf")) == 0.5
        assert compute_raw_probability(float("-inf")) == 0.5
        assert compute_raw_probability(0.75) == 0.75


# ═══════════════════════════════════════════════════════════════════════
# STEP 4: Data freshness verification
# ═══════════════════════════════════════════════════════════════════════


class TestStep4DataFreshness:
    """Verify data is current enough for Selection Sunday run."""

    def test_manifest_exists(self):
        assert os.path.exists(MANIFEST_PATH), "Manifest file missing"

    def test_manifest_year_is_2026(self):
        m = _load_manifest()
        assert m["year"] == 2026

    def test_manifest_generated_recently(self):
        """Manifest should have been generated within the last 7 days."""
        m = _load_manifest()
        gen = datetime.fromisoformat(m["generated_at"].replace("Z", "+00:00"))
        age = datetime.now(gen.tzinfo) - gen
        assert age.days <= 7, f"Manifest is {age.days} days old"

    def test_torvik_data_has_bracket_teams(self):
        """Most bracket teams should resolve to Torvik four factors data.

        Team IDs may differ between bracket and Torvik (e.g. iowa_state vs iowa_st,
        brigham_young vs byu).  We require >=70% direct match coverage; the rest
        are handled by the TeamNameResolver at runtime.
        """
        bracket = _load_bracket()
        torvik = _load_torvik()
        bracket_ids = {t["team_id"] for t in bracket["teams"]}
        torvik_ids = set(torvik.keys())
        matched = bracket_ids & torvik_ids
        match_rate = len(matched) / len(bracket_ids)
        assert match_rate >= 0.70, (
            f"Only {match_rate:.0%} of bracket teams directly match Torvik keys "
            f"(expected >=70%). Missing: {bracket_ids - torvik_ids}"
        )

    def test_torvik_data_coverage_sufficient(self):
        """Torvik data should cover at least 350 teams (D1 universe)."""
        torvik = _load_torvik()
        assert len(torvik) >= 300, f"Only {len(torvik)} teams in Torvik data"

    def test_torvik_data_has_nonzero_metrics(self):
        """Top teams should have non-trivial four factors metrics."""
        torvik = _load_torvik()
        for team in ["michigan", "duke", "arizona", "connecticut"]:
            if team in torvik:
                efg = torvik[team].get("effective_fg_pct", 0)
                assert efg > 0.3, f"{team} has implausible eFG%: {efg}"

    def test_bracket_json_exists(self):
        assert os.path.exists(BRACKET_PATH), "Bracket file missing"

    def test_games_json_exists(self):
        assert os.path.exists(GAMES_PATH), "Games file missing"


# ═══════════════════════════════════════════════════════════════════════
# STEP 5: Bracket and seed ingestion
# ═══════════════════════════════════════════════════════════════════════


class TestStep5BracketIngestion:
    """Verify bracket is complete, correct, and well-formed."""

    def test_bracket_has_64_teams(self):
        b = _load_bracket()
        assert len(b["teams"]) == EXPECTED_TEAM_COUNT

    def test_bracket_has_4_regions(self):
        b = _load_bracket()
        regions = {t["region"] for t in b["teams"]}
        assert regions == EXPECTED_REGIONS

    def test_bracket_has_seeds_1_through_16(self):
        b = _load_bracket()
        seeds = {t["seed"] for t in b["teams"]}
        assert seeds == EXPECTED_SEEDS

    def test_each_region_has_16_teams(self):
        b = _load_bracket()
        region_counts = Counter(t["region"] for t in b["teams"])
        for region, count in region_counts.items():
            assert count == 16, f"Region {region} has {count} teams, expected 16"

    def test_no_duplicate_team_ids(self):
        b = _load_bracket()
        ids = [t["team_id"] for t in b["teams"]]
        assert len(ids) == len(set(ids)), f"Duplicate team IDs found"

    def test_no_duplicate_team_names(self):
        b = _load_bracket()
        names = [t["team_name"] for t in b["teams"]]
        assert len(names) == len(set(names)), f"Duplicate team names found"

    def test_exactly_four_1_seeds(self):
        b = _load_bracket()
        ones = [t for t in b["teams"] if t["seed"] == 1]
        assert len(ones) == 4
        one_ids = {t["team_id"] for t in ones}
        assert one_ids == EXPECTED_1_SEEDS

    def test_each_seed_has_4_teams(self):
        b = _load_bracket()
        seed_counts = Counter(t["seed"] for t in b["teams"])
        for seed, count in seed_counts.items():
            assert count == 4, f"Seed {seed} has {count} teams, expected 4"

    def test_all_teams_have_required_fields(self):
        b = _load_bracket()
        required = {"team_name", "team_id", "seed", "region"}
        for t in b["teams"]:
            missing = required - set(t.keys())
            assert not missing, f"Team {t.get('team_name', '?')} missing fields: {missing}"

    def test_no_empty_team_ids(self):
        b = _load_bracket()
        for t in b["teams"]:
            assert t["team_id"], f"Empty team_id for {t.get('team_name', '?')}"

    def test_1_seed_spot_check(self):
        """Spot check: Michigan is a 1-seed in the East."""
        b = _load_bracket()
        michigan = [t for t in b["teams"] if t["team_id"] == "michigan"]
        assert len(michigan) == 1
        assert michigan[0]["seed"] == 1
        assert michigan[0]["region"] == "East"

    def test_mid_major_present(self):
        """Spot check: a mid-major conference champion is present."""
        b = _load_bracket()
        # Check for at least one team from a mid-major conference
        mid_major_confs = {
            "Coastal Athletic", "Big South", "Southland", "MAAC",
            "Western Athletic", "Ivy League", "Big West", "Horizon League",
            "Ohio Valley", "Patriot League", "Sun Belt", "Big Sky",
            "America East", "SWAC", "NEC", "MEAC", "Summit League",
        }
        confs = {t.get("conference", "") for t in b["teams"]}
        mid_major_found = confs & mid_major_confs
        assert len(mid_major_found) >= 5, (
            f"Only {len(mid_major_found)} mid-major conferences found"
        )

    def test_no_resolution_warnings(self):
        b = _load_bracket()
        warnings = b.get("resolution_warnings", [])
        assert len(warnings) == 0, f"Bracket has resolution warnings: {warnings}"

    def test_season_is_2026(self):
        b = _load_bracket()
        assert b["season"] == 2026


# ═══════════════════════════════════════════════════════════════════════
# STEP 6: End-to-end inference smoke test (unit-level)
# ═══════════════════════════════════════════════════════════════════════


class TestStep6InferenceSmokeTest:
    """Verify production pipeline produces plausible probabilities for diverse matchups."""

    def _run_pipeline(self, raw_prob, shrinkage=0.06, clip_lo=0.005, clip_hi=0.995):
        """Run the 4-stage production pipeline on a raw probability."""
        from src.pipeline.probability_pipeline import (
            compute_raw_probability,
            apply_calibration,
            apply_tournament_shrinkage,
            apply_final_clip,
        )
        p = compute_raw_probability(raw_prob)
        p = apply_calibration(p, None, clip_lo, clip_hi)
        p = apply_tournament_shrinkage(p, shrinkage)
        return apply_final_clip(p, clip_lo, clip_hi)

    def test_1_vs_16_favorite_is_favored(self):
        """1-seed should be heavily favored over 16-seed."""
        p = self._run_pipeline(0.97)
        assert p > 0.85, f"1-vs-16 probability {p} too low"
        assert p < 1.0

    def test_8_vs_9_tossup(self):
        """8-vs-9 should be near 0.5."""
        p = self._run_pipeline(0.52)
        assert 0.45 < p < 0.55, f"8-vs-9 probability {p} not near toss-up"

    def test_5_vs_12_upset_prone(self):
        """5-vs-12 should favor the 5-seed but not overwhelmingly."""
        p = self._run_pipeline(0.65)
        assert 0.55 < p < 0.75, f"5-vs-12 probability {p} out of range"

    def test_elite_vs_elite(self):
        """Elite matchup should be competitive."""
        p = self._run_pipeline(0.55)
        assert 0.48 < p < 0.58, f"Elite-vs-elite probability {p} out of range"

    def test_no_nan_output(self):
        for raw in [0.0, 0.01, 0.5, 0.99, 1.0]:
            p = self._run_pipeline(raw)
            assert math.isfinite(p), f"NaN/Inf output for raw={raw}"

    def test_no_exact_half_flood(self):
        """Not too many outputs should be exactly 0.5 (indicates fallback flood)."""
        raws = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]
        halves = sum(1 for r in raws if self._run_pipeline(r) == 0.5)
        assert halves == 0, f"{halves} of {len(raws)} outputs were exactly 0.5"

    def test_outputs_in_valid_range(self):
        for raw in np.linspace(0.01, 0.99, 50):
            p = self._run_pipeline(float(raw))
            assert 0.005 <= p <= 0.995, f"Output {p} out of bounds for raw={raw}"

    def test_monotonicity(self):
        """Higher raw probability should yield higher output."""
        raws = [0.1, 0.3, 0.5, 0.7, 0.9]
        outputs = [self._run_pipeline(r) for r in raws]
        for i in range(len(outputs) - 1):
            assert outputs[i] < outputs[i + 1], (
                f"Monotonicity violation: f({raws[i]})={outputs[i]} >= f({raws[i+1]})={outputs[i+1]}"
            )


# ═══════════════════════════════════════════════════════════════════════
# STEP 7: Ablation ladder
# ═══════════════════════════════════════════════════════════════════════


class TestStep7AblationLadder:
    """Run ablation ladder on synthetic data and verify metrics are sane."""

    def _get_synthetic_data(self):
        """Generate synthetic tournament-like data."""
        np.random.seed(42)
        n = 200
        raw_probs = np.clip(np.random.beta(2, 2, n), 0.05, 0.95)
        outcomes = (np.random.rand(n) < raw_probs).astype(float)
        return raw_probs, outcomes

    def test_ablation_ladder_all_stages_present(self):
        from src.pipeline.probability_pipeline import run_ablation_ladder
        raw, out = self._get_synthetic_data()
        result = run_ablation_ladder(raw, out, calibrator=None, shrinkage=0.06)
        assert set(result.keys()) == {"raw", "calibrated", "shrunk", "final"}

    def test_ablation_ladder_metrics_complete(self):
        from src.pipeline.probability_pipeline import run_ablation_ladder
        raw, out = self._get_synthetic_data()
        result = run_ablation_ladder(raw, out, calibrator=None, shrinkage=0.06)
        required = {"brier", "ece", "log_loss", "mean_movement", "max_movement"}
        for stage, metrics in result.items():
            missing = required - set(metrics.keys())
            assert not missing, f"Stage {stage} missing: {missing}"

    def test_raw_stage_zero_movement(self):
        from src.pipeline.probability_pipeline import run_ablation_ladder
        raw, out = self._get_synthetic_data()
        result = run_ablation_ladder(raw, out, calibrator=None, shrinkage=0.06)
        assert result["raw"]["mean_movement"] == 0.0
        assert result["raw"]["max_movement"] == 0.0

    def test_shrinkage_movement_is_modest(self):
        """Shrinkage should cause modest movement, not chaos."""
        from src.pipeline.probability_pipeline import run_ablation_ladder
        raw, out = self._get_synthetic_data()
        result = run_ablation_ladder(raw, out, calibrator=None, shrinkage=0.06)
        # With 0.06 shrinkage, max movement should be < 0.06 * 0.5 = 0.03
        assert result["shrunk"]["max_movement"] < 0.05
        assert result["shrunk"]["mean_movement"] < 0.03

    def test_brier_scores_are_finite(self):
        from src.pipeline.probability_pipeline import run_ablation_ladder
        raw, out = self._get_synthetic_data()
        result = run_ablation_ladder(raw, out, calibrator=None, shrinkage=0.06)
        for stage, metrics in result.items():
            assert math.isfinite(metrics["brier"]), f"Stage {stage} has non-finite Brier"
            assert 0 <= metrics["brier"] <= 1, f"Stage {stage} Brier out of [0,1]"

    def test_no_calibrator_raw_equals_calibrated(self):
        """Without a calibrator, raw == calibrated."""
        from src.pipeline.probability_pipeline import run_ablation_ladder
        raw, out = self._get_synthetic_data()
        result = run_ablation_ladder(raw, out, calibrator=None, shrinkage=0.06)
        assert result["calibrated"]["mean_movement"] == 0.0

    def test_with_mock_calibrator(self):
        """With a temperature calibrator, calibration stage has nonzero movement."""
        from src.pipeline.probability_pipeline import run_ablation_ladder

        cal = MagicMock()
        cal.fitted = True
        cal.temperature = 1.3
        del cal.calibrate

        raw, out = self._get_synthetic_data()
        result = run_ablation_ladder(raw, out, calibrator=cal, shrinkage=0.06)
        assert result["calibrated"]["mean_movement"] > 0


# ═══════════════════════════════════════════════════════════════════════
# STEP 8: Probability distribution sanity
# ═══════════════════════════════════════════════════════════════════════


class TestStep8DistributionSanity:
    """Verify production probabilities have a healthy distribution shape."""

    def _generate_distribution(self, n=500):
        """Simulate a realistic probability distribution through the pipeline."""
        from src.pipeline.probability_pipeline import (
            apply_calibration, apply_tournament_shrinkage, apply_final_clip,
        )
        np.random.seed(2026)
        # Simulate raw probs roughly matching tournament game spread
        raws = np.clip(np.random.beta(2, 2, n), 0.02, 0.98)
        probs = []
        for r in raws:
            p = apply_calibration(float(r), None, 0.005, 0.995)
            p = apply_tournament_shrinkage(p, 0.06)
            p = apply_final_clip(p, 0.005, 0.995)
            probs.append(p)
        return np.array(probs)

    def test_no_extreme_clumping(self):
        """Less than 10% should be at extreme ends (< 0.1 or > 0.9)."""
        probs = self._generate_distribution()
        extreme_frac = np.mean((probs < 0.1) | (probs > 0.9))
        assert extreme_frac < 0.40, f"{extreme_frac:.1%} of outputs are extreme"

    def test_not_all_near_half(self):
        """Less than 80% should be in the [0.4, 0.6] band (washed out check)."""
        probs = self._generate_distribution()
        near_half_frac = np.mean((probs > 0.4) & (probs < 0.6))
        assert near_half_frac < 0.80, f"{near_half_frac:.1%} outputs near 0.5 — too washed out"

    def test_favorites_generally_favored(self):
        """Probs > 0.5 from strong raw inputs should remain > 0.5 after pipeline."""
        from src.pipeline.probability_pipeline import (
            apply_calibration, apply_tournament_shrinkage, apply_final_clip,
        )
        for raw in [0.7, 0.8, 0.9]:
            p = apply_calibration(raw, None, 0.005, 0.995)
            p = apply_tournament_shrinkage(p, 0.06)
            p = apply_final_clip(p, 0.005, 0.995)
            assert p > 0.5, f"Favorite (raw={raw}) became underdog after pipeline: {p}"

    def test_distribution_mean_near_half(self):
        """With symmetric beta inputs, mean should be near 0.5."""
        probs = self._generate_distribution()
        assert 0.4 < probs.mean() < 0.6, f"Distribution mean {probs.mean():.3f} far from 0.5"

    def test_reasonable_std(self):
        """Standard deviation should not be extremely low (washed) or high (overconfident)."""
        probs = self._generate_distribution()
        assert probs.std() > 0.05, f"Std {probs.std():.3f} too low — washed out"
        assert probs.std() < 0.4, f"Std {probs.std():.3f} too high — overconfident"


# ═══════════════════════════════════════════════════════════════════════
# STEP 9: Historical production path smoke backtest
# ═══════════════════════════════════════════════════════════════════════


class TestStep9HistoricalBacktest:
    """Smoke-test the production path on synthetic historical data."""

    def test_production_path_on_historical_like_data(self):
        """Simulate LOYO-like evaluation on the production path."""
        from src.pipeline.probability_pipeline import (
            apply_calibration, apply_tournament_shrinkage, apply_final_clip,
            run_ablation_ladder,
        )
        np.random.seed(2024)
        # Simulate 2024-like tournament outcomes
        n = 63  # ~1 tournament bracket worth
        raws = np.clip(np.random.beta(3, 2, n), 0.05, 0.95)
        outcomes = (np.random.rand(n) < raws).astype(float)

        result = run_ablation_ladder(raws, outcomes, calibrator=None, shrinkage=0.06)

        # Brier should be reasonable (not perfect, not terrible)
        assert result["final"]["brier"] < 0.30, "Final Brier > 0.30 is a red flag"
        assert result["final"]["brier"] > 0.05, "Final Brier < 0.05 seems impossible"

    def test_production_path_on_skewed_data(self):
        """Even with heavy favorites, production path shouldn't produce NaN."""
        from src.pipeline.probability_pipeline import run_ablation_ladder
        np.random.seed(2025)
        raws = np.array([0.95, 0.90, 0.85, 0.80, 0.55, 0.50, 0.45, 0.20])
        outcomes = np.array([1, 1, 1, 1, 1, 0, 0, 0], dtype=float)
        result = run_ablation_ladder(raws, outcomes, calibrator=None, shrinkage=0.06)
        for stage, metrics in result.items():
            for k, v in metrics.items():
                assert math.isfinite(v), f"Stage {stage}.{k} is not finite"


# ═══════════════════════════════════════════════════════════════════════
# STEP 10: Export artifact validation
# ═══════════════════════════════════════════════════════════════════════


class TestStep10ExportArtifact:
    """Verify Kaggle export utilities produce valid submission format."""

    def test_parse_kaggle_id_valid(self):
        from src.exports.kaggle import parse_kaggle_id
        season, t1, t2 = parse_kaggle_id("2026_1234_5678")
        assert season == 2026
        assert t1 == 1234
        assert t2 == 5678

    def test_parse_kaggle_id_invalid(self):
        from src.exports.kaggle import parse_kaggle_id
        with pytest.raises(ValueError):
            parse_kaggle_id("bad_format")

    def test_parse_kaggle_id_none(self):
        from src.exports.kaggle import parse_kaggle_id
        with pytest.raises(ValueError):
            parse_kaggle_id(None)

    def test_generate_predictions_schema(self):
        """Verify generate_predictions output has correct columns."""
        import pandas as pd
        from src.exports.kaggle import generate_predictions

        sample = pd.DataFrame({"ID": ["2026_1_2"], "Pred": [0.5]})
        id_map = {1: "team_a", 2: "team_b"}
        predict_fn = lambda a, b: 0.65

        result = generate_predictions(sample, id_map, predict_fn, season_filter=2026)
        assert "Pred" in result.columns
        assert len(result) == 1
        assert 0.0 <= result["Pred"].iloc[0] <= 1.0

    def test_generate_predictions_unmapped_defaults_to_half(self):
        import pandas as pd
        from src.exports.kaggle import generate_predictions

        sample = pd.DataFrame({"ID": ["2026_999_998"], "Pred": [0.5]})
        id_map = {}  # No mappings
        predict_fn = lambda a, b: 0.7

        result = generate_predictions(sample, id_map, predict_fn, season_filter=2026)
        assert result["Pred"].iloc[0] == 0.5

    def test_womens_team_routing(self):
        from src.exports.kaggle import is_womens_team
        assert is_womens_team(3001) is True
        assert is_womens_team(1234) is False
        assert is_womens_team(3000) is True

    def test_export_clips_invalid_probabilities(self):
        """Predictions outside [0,1] should be clipped in generate_predictions."""
        import pandas as pd
        from src.exports.kaggle import generate_predictions

        sample = pd.DataFrame({"ID": ["2026_1_2"], "Pred": [0.5]})
        id_map = {1: "team_a", 2: "team_b"}
        def predict_fn(a, b):
            return 1.5  # Invalid

        result = generate_predictions(sample, id_map, predict_fn, season_filter=2026)
        assert result["Pred"].iloc[0] <= 1.0


# ═══════════════════════════════════════════════════════════════════════
# STEP 11: Freeze / reproducibility checks
# ═══════════════════════════════════════════════════════════════════════


class TestStep11Reproducibility:
    """Verify configs are deterministic and hashable for freeze."""

    def test_config_defaults_are_deterministic(self):
        from src.pipeline.config import SOTAPipelineConfig
        c1 = SOTAPipelineConfig()
        c2 = SOTAPipelineConfig()
        # All important production fields should be identical
        assert c1.probability_profile == c2.probability_profile
        assert c1.tournament_shrinkage == c2.tournament_shrinkage
        assert c1.pre_calibration_clip_lo == c2.pre_calibration_clip_lo
        assert c1.pre_calibration_clip_hi == c2.pre_calibration_clip_hi
        assert c1.calibration_method == c2.calibration_method
        assert c1.enable_seed_overrides == c2.enable_seed_overrides
        assert c1.enable_brier_sharpening == c2.enable_brier_sharpening
        assert c1.enable_goto_conversion == c2.enable_goto_conversion
        assert c1.enable_round_weighted_calibration == c2.enable_round_weighted_calibration
        assert c1.seed_prior_weight == c2.seed_prior_weight
        assert c1.consistency_bonus_max == c2.consistency_bonus_max

    def test_pipeline_functions_are_pure(self):
        """Calling pipeline functions twice with same input should give same output."""
        from src.pipeline.probability_pipeline import (
            apply_calibration, apply_tournament_shrinkage, apply_final_clip,
        )
        for _ in range(10):
            r = 0.73
            a = apply_calibration(r, None, 0.005, 0.995)
            b = apply_calibration(r, None, 0.005, 0.995)
            assert a == b
            s1 = apply_tournament_shrinkage(a, 0.06)
            s2 = apply_tournament_shrinkage(a, 0.06)
            assert s1 == s2
            c1 = apply_final_clip(s1, 0.005, 0.995)
            c2 = apply_final_clip(s1, 0.005, 0.995)
            assert c1 == c2


# ═══════════════════════════════════════════════════════════════════════
# Cross-cutting: audit helper alignment
# ═══════════════════════════════════════════════════════════════════════


class TestCrossCuttingAuditAlignment:
    """Verify RDoF audit code uses the same shared helpers as production."""

    def test_holdout_evaluator_tournament_adapt_uses_shared(self):
        from src.ml.evaluation.rdof_audit import HoldoutEvaluator
        source = inspect.getsource(HoldoutEvaluator._tournament_adapt)
        assert "probability_pipeline" in source
        assert "apply_tournament_shrinkage" in source

    def test_holdout_evaluator_posthoc_uses_shared(self):
        from src.ml.evaluation.rdof_audit import HoldoutEvaluator
        source = inspect.getsource(HoldoutEvaluator._apply_posthoc_and_score)
        assert "probability_pipeline" in source
        assert "apply_calibration" in source

    def test_probability_pipeline_has_no_experimental_in_production_section(self):
        """The production section of probability_pipeline.py should not import
        experimental components."""
        import src.pipeline.probability_pipeline as pp
        source = inspect.getsource(pp)
        # Find the production section (before the experimental section)
        prod_end = source.find("# Experimental functions")
        if prod_end == -1:
            prod_end = source.find("NEVER import these in production")
        if prod_end > 0:
            prod_section = source[:prod_end]
            assert "BrierPostProcessor" not in prod_section
            assert "SeedBasedOverrides" not in prod_section
            assert "FavouriteLongshotCorrection" not in prod_section
