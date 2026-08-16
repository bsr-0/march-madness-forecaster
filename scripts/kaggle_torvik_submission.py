#!/usr/bin/env python3
"""Generate a Kaggle submission CSV using Torvik barthag + Log5.

Bypasses the ML pipeline entirely. Torvik log5 achieves BSS +0.049 vs
seeds across 18 years — better than the pipeline model (BSS -0.25).

Usage:
    python scripts/kaggle_torvik_submission.py --year 2026
    python scripts/kaggle_torvik_submission.py --year 2026 --sample-submission data/kaggle/SampleSubmission.csv
    python scripts/kaggle_torvik_submission.py --year 2026 --backtest  # score against actuals

The --backtest flag computes Brier score against actual tournament results
for validation without needing a Kaggle sample submission file.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from functools import lru_cache
from typing import Callable

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.team_name_resolver import TeamNameResolver
from src.exports.kaggle import (
    build_team_id_map_with_spellings,
    generate_predictions,
    is_womens_team,
    load_kaggle_spellings,
    load_kaggle_teams,
    load_kaggle_womens_teams,
)
from src.ml.calibration.post_processing import PostProcessingPipeline
from src.prediction.elo_probabilities import load_elo_barthag
from src.prediction.market_probabilities import load_market_ratings
from src.prediction.torvik_correction import TorvikCorrectionConfig, fit_torvik_correction_from_year_records
from src.prediction.torvik_kaggle import EnsembleKagglePredictor, TarvikKagglePredictor

ARTIFACTS_DIR = REPO_ROOT / "artifacts"
TORVIK_CORRECTED_RECENT5_CONSERVATIVE = "torvik_corrected_recent5_conservative"


def _load_context_subkey(year: int, sub_key: str, fallback_filename: str):
    """Return sub-key data for `year`, preferring the consolidated
    `tournament_context_{year}.json`, falling back to the old per-type
    file if the consolidated file doesn't exist yet or lacks that
    sub-key. Returns None if neither source has the data. Mirrors
    scripts._common._load_context_subkey; duplicated here because this
    module builds its paths directly rather than via scripts._common.HIST_DIR."""
    hist_dir = REPO_ROOT / "data/raw/historical"
    ctx_path = hist_dir / f"tournament_context_{year}.json"
    if ctx_path.exists():
        with open(ctx_path) as f:
            ctx = json.load(f)
        if sub_key in ctx:
            return ctx[sub_key]
    old_path = hist_dir / fallback_filename
    if not old_path.exists():
        return None
    with open(old_path) as f:
        return json.load(f)


def load_tournament_seeds(year: int) -> dict[str, int]:
    """Load tournament seeds from tournament_results or seeds JSON."""
    # Try tournament results first (has actual games with seeds)
    data = _load_context_subkey(year, "results", f"tournament_results_{year}.json")
    if data is not None:
        games = data.get("games", data) if isinstance(data, dict) else data
        seeds = {}
        for g in games:
            seeds[g["team1_id"]] = g.get("team1_seed", 8)
            seeds[g["team2_id"]] = g.get("team2_seed", 8)
        return seeds

    # Try seeds JSON
    seeds_data = _load_context_subkey(year, "seeds", f"tournament_seeds_{year}.json")
    if seeds_data is not None:
        return seeds_data

    return {}


def load_kaggle_id_mapping() -> dict[int, str]:
    """Load Kaggle TeamID -> canonical_id mapping."""
    teams_path = REPO_ROOT / "data/kaggle/MTeams.csv"
    spellings_path = REPO_ROOT / "data/kaggle/MTeamSpellings.csv"
    if not teams_path.exists():
        print(f"ERROR: {teams_path} not found")
        sys.exit(1)
    if not spellings_path.exists():
        print(f"ERROR: {spellings_path} not found")
        sys.exit(1)

    team_map = load_kaggle_teams(str(teams_path))
    spellings = load_kaggle_spellings(str(spellings_path))
    return build_team_id_map_with_spellings(team_map, TeamNameResolver(), spellings)


def _maybe_postprocess(
    pred: float,
    seed1: int,
    seed2: int,
    pp: PostProcessingPipeline | None,
) -> float:
    """Apply post-processing if configured."""
    if pp is None:
        return pred
    return pp.process(pred, seed1, seed2)


def _resolve_mens_kaggle_mode_config(
    mode: str,
    ensemble_recent_year_start: int | None,
    ensemble_recent_year_weight: float,
    ensemble_recent_year_count: int | None,
    ensemble_recent_total_ratio: float,
    torvik_correction_ridge: float,
    torvik_correction_max_correction: float,
) -> dict[str, object]:
    """Resolve CLI mode + fit knobs into one config bundle."""
    config = {
        "requested_mode": mode,
        "base_mode": mode,
        "recent_year_start": ensemble_recent_year_start,
        "recent_year_weight": ensemble_recent_year_weight,
        "recent_year_count": ensemble_recent_year_count,
        "recent_total_ratio": ensemble_recent_total_ratio,
        "correction_ridge": torvik_correction_ridge,
        "max_correction": torvik_correction_max_correction,
    }

    if mode == TORVIK_CORRECTED_RECENT5_CONSERVATIVE:
        # Admitted 2026-05-10: 8-feature model (market + elo + round).
        # ridge=20.0 selected by admission gate over {5,10,20,40}; beats
        # ensemble incumbent by +0.0071 Brier on final holdout, 3/4 years.
        config.update(
            {
                "base_mode": "torvik_corrected",
                "recent_year_start": None,
                "recent_year_weight": 1.0,
                "recent_year_count": 5,
                "recent_total_ratio": 1.0,
                "correction_ridge": 20.0,
                "max_correction": 0.06,
            }
        )

    return config


def _build_submission_predict_fn(
    year: int,
    mode: str,
    clip_lo: float,
    clip_hi: float,
    pp: PostProcessingPipeline | None = None,
    ensemble_recent_year_start: int | None = 2021,
    ensemble_recent_year_weight: float = 2.0,
    ensemble_recent_year_count: int | None = None,
    ensemble_recent_total_ratio: float = 1.0,
    torvik_correction_ridge: float = 5.0,
    torvik_correction_max_correction: float = 0.10,
) -> tuple[Callable[[str, str], float], dict[str, object]]:
    """Build the prediction function used by Kaggle submission export."""
    mode_config = _resolve_mens_kaggle_mode_config(
        mode,
        ensemble_recent_year_start,
        ensemble_recent_year_weight,
        ensemble_recent_year_count,
        ensemble_recent_total_ratio,
        torvik_correction_ridge,
        torvik_correction_max_correction,
    )
    base_mode = str(mode_config["base_mode"])
    seeds = load_tournament_seeds(year)
    torvik = TarvikKagglePredictor.from_year(year, seeds=seeds, clip_lo=clip_lo, clip_hi=clip_hi)

    if base_mode == "ensemble":
        pipe_lookup = _load_pipeline_lookup(year)
        if pipe_lookup is None:
            raise RuntimeError(
                f"No per-game pipeline predictions available for {year}; "
                "ensemble submission mode requires artifacts/backtest_result_temperature.json"
            )

        def pipe_predict(t1, t2, _lookup=pipe_lookup):
            key1, key2 = f"{t1}_{t2}", f"{t2}_{t1}"
            if key1 in _lookup:
                return _lookup[key1]
            if key2 in _lookup:
                return 1.0 - _lookup[key2]
            return 0.5

        ensemble = EnsembleKagglePredictor.from_walk_forward(
            torvik,
            pipe_predict,
            year,
            clip_lo=clip_lo,
            clip_hi=clip_hi,
            recent_year_start=mode_config["recent_year_start"],
            recent_year_weight=float(mode_config["recent_year_weight"]),
            recent_year_count=mode_config["recent_year_count"],
            recent_total_ratio=float(mode_config["recent_total_ratio"]),
        )

        def predict_fn(t1: str, t2: str, _ensemble=ensemble) -> float:
            return _maybe_postprocess(_ensemble.predict(t1, t2), seeds.get(t1, 8), seeds.get(t2, 8), pp)

        return predict_fn, {
            "mode": str(mode_config["requested_mode"]),
            "base_mode": "ensemble",
            "alpha": round(float(ensemble.alpha), 4),
            "alpha_training_info": dict(ensemble.alpha_training_info),
            "torvik_stats": torvik.stats(),
        }

    if base_mode == "torvik_corrected":
        correction = _build_torvik_correction_model(
            year,
            clip_lo=clip_lo,
            clip_hi=clip_hi,
            recent_year_start=mode_config["recent_year_start"],
            recent_year_weight=float(mode_config["recent_year_weight"]),
            recent_year_count=mode_config["recent_year_count"],
            recent_total_ratio=float(mode_config["recent_total_ratio"]),
            correction_ridge=float(mode_config["correction_ridge"]),
            max_correction=float(mode_config["max_correction"]),
        )

        # Load market and elo ratings once per year (Log5 matchup prob at inference).
        # Both return None when data is unavailable — correction degrades gracefully.
        _market_ratings = load_market_ratings(year, seeds) or {}
        _elo_ratings = load_elo_barthag(year, seeds) or {}

        def _log5_matchup(ratings: dict, t1: str, t2: str) -> float | None:
            b1 = ratings.get(t1)
            b2 = ratings.get(t2)
            if not b1 or not b2:
                return None
            num = b1 * (1.0 - b2)
            denom = num + b2 * (1.0 - b1)
            return num / denom if denom > 1e-9 else None

        def predict_fn(t1: str, t2: str, _torvik=torvik, _correction=correction) -> float:
            base_prob = _torvik.predict(t1, t2)
            s1, s2 = seeds.get(t1, 8), seeds.get(t2, 8)
            # round_num=0.0: Kaggle predicts all matchups without round context
            return _maybe_postprocess(
                _correction.predict_one(
                    base_prob,
                    s1,
                    s2,
                    market_prob=_log5_matchup(_market_ratings, t1, t2),
                    elo_prob=_log5_matchup(_elo_ratings, t1, t2),
                    round_num=0.0,
                ),
                s1,
                s2,
                pp,
            )

        info = {
            "mode": str(mode_config["requested_mode"]),
            "base_mode": "torvik_corrected",
            "torvik_stats": torvik.stats(),
        }
        if correction.coef_ is not None:
            info["correction_coefficients"] = [round(float(v), 6) for v in correction.coef_]
        info["correction_training_info"] = {
            "recent_year_start": mode_config["recent_year_start"],
            "recent_year_weight": float(mode_config["recent_year_weight"]),
            "recent_year_count": mode_config["recent_year_count"],
            "recent_total_ratio": float(mode_config["recent_total_ratio"]),
            **dict(getattr(correction, "training_info_", {}) or {}),
        }
        info["correction_training_info"]["correction_ridge"] = float(mode_config["correction_ridge"])
        info["correction_training_info"]["max_correction"] = float(mode_config["max_correction"])
        return predict_fn, info

    def predict_fn(t1: str, t2: str, _torvik=torvik) -> float:
        return _maybe_postprocess(_torvik.predict(t1, t2), seeds.get(t1, 8), seeds.get(t2, 8), pp)

    return predict_fn, {
        "mode": str(mode_config["requested_mode"]),
        "base_mode": "torvik",
        "torvik_stats": torvik.stats(),
    }


def _load_torvik_correction_training_data() -> dict[int, list[dict]]:
    """Load per-game walk-forward artifact keyed by season."""
    path = REPO_ROOT / "artifacts" / "loyo_pergame_predictions.json"
    if not path.exists():
        raise RuntimeError("torvik correction mode requires artifacts/loyo_pergame_predictions.json")
    with open(path) as f:
        data = json.load(f)
    return {int(year): records for year, records in data.items()}


def _build_torvik_correction_model(
    year: int,
    clip_lo: float,
    clip_hi: float,
    recent_year_start: int | None,
    recent_year_weight: float,
    recent_year_count: int | None = None,
    recent_total_ratio: float = 1.0,
    correction_ridge: float = 5.0,
    max_correction: float = 0.10,
):
    """Fit the torvik correction layer from prior-year walk-forward data."""
    records_by_year = _load_torvik_correction_training_data()
    train_year_records = {season: rows for season, rows in records_by_year.items() if season < year and rows}
    if not train_year_records:
        raise RuntimeError(f"No torvik correction training data available for seasons < {year}")

    config = TorvikCorrectionConfig(
        clip_lo=clip_lo,
        clip_hi=clip_hi,
        ridge=correction_ridge,
        max_correction=max_correction,
    )
    return fit_torvik_correction_from_year_records(
        train_year_records,
        config=config,
        recent_year_start=recent_year_start,
        recent_year_weight=recent_year_weight,
        recent_year_count=recent_year_count,
        recent_total_ratio=recent_total_ratio,
    )


def _load_kaggle_seed_numbers(csv_path: Path, year: int) -> dict[int, int]:
    """Load Kaggle tournament seeds as TeamID -> numeric seed for one season."""
    if not csv_path.exists():
        return {}

    import csv

    seeds: dict[int, int] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                if int(row.get("Season", "0")) != year:
                    continue
                kaggle_id = int(row.get("TeamID", "0"))
            except (TypeError, ValueError):
                continue
            seed_str = (row.get("Seed") or "").strip()
            seed_num = int("".join(ch for ch in seed_str[1:] if ch.isdigit()) or "8")
            seeds[kaggle_id] = seed_num
    return seeds


def _extract_womens_team_ids(sample_df: pd.DataFrame) -> set[int]:
    """Collect women's Kaggle TeamIDs present in a sample submission frame."""
    women_team_ids: set[int] = set()
    for raw_id in sample_df["ID"].astype(str):
        parts = raw_id.split("_")
        if len(parts) != 3:
            continue
        for tid_str in parts[1:]:
            try:
                tid = int(tid_str)
            except ValueError:
                continue
            if is_womens_team(tid):
                women_team_ids.add(tid)
    return women_team_ids


def _build_womens_submission_support(
    sample_df: pd.DataFrame,
    year: int,
    womens_model_weight: float | None = None,
    womens_seed_weight: float | None = None,
    womens_weight_step: float = 0.1,
    refresh_womens_weight_cache: bool = False,
) -> tuple[dict[int, str] | None, Callable[[str, str], float] | None, dict[str, float] | None]:
    """Build women's TeamID mapping and predictor for combined Kaggle samples."""
    women_team_ids = _extract_womens_team_ids(sample_df)
    if not women_team_ids:
        return None, None, None

    women_teams_path = REPO_ROOT / "data/kaggle/WTeams.csv"
    women_spellings_path = REPO_ROOT / "data/kaggle/WTeamSpellings.csv"
    women_seeds_path = REPO_ROOT / "data/kaggle/WNCAATourneySeeds.csv"
    if not women_teams_path.exists() or not women_spellings_path.exists():
        raise RuntimeError(
            "Women's rows were detected in the sample submission, but Kaggle women's team files are missing. "
            "Expected data/kaggle/WTeams.csv and data/kaggle/WTeamSpellings.csv."
        )

    from src.pipeline.womens import WomensPipeline, WomensPipelineConfig

    women_team_map = load_kaggle_womens_teams(str(women_teams_path))
    women_spellings = load_kaggle_spellings(str(women_spellings_path))
    women_canonical_map = build_team_id_map_with_spellings(women_team_map, TeamNameResolver(), women_spellings)

    womens_feature_logger = logging.getLogger("src.data.features.womens_feature_engineering")
    previous_womens_level = womens_feature_logger.level
    womens_feature_logger.setLevel(logging.ERROR)
    try:
        if womens_model_weight is None or womens_seed_weight is None:
            womens_model_weight, womens_seed_weight, train_brier = _get_cached_womens_weights_for_year(
                year,
                womens_weight_step,
                candidate_weights=_candidate_womens_weights(womens_weight_step),
                force_refresh=refresh_womens_weight_cache,
            )
        else:
            train_brier = float("nan")

        pipeline = WomensPipeline(
            WomensPipelineConfig(
                year=year,
                cache_dir=str(REPO_ROOT / "data/raw"),
                model_weight=womens_model_weight,
                seed_weight=womens_seed_weight,
            )
        )
        pipeline.run()
    finally:
        womens_feature_logger.setLevel(previous_womens_level)

    women_seed_lookup = _load_kaggle_seed_numbers(women_seeds_path, year)
    women_id_map: dict[int, str] = {}
    seed_backfill = {}
    for team_id in women_team_ids:
        raw_key = str(team_id)
        canonical = women_canonical_map.get(team_id, raw_key)

        if raw_key in pipeline.feature_engineer.team_features or raw_key in pipeline.team_stats:
            women_id_map[team_id] = raw_key
            continue
        if canonical in pipeline.feature_engineer.team_features or canonical in pipeline.team_stats:
            women_id_map[team_id] = canonical
            continue

        women_id_map[team_id] = raw_key
        seed_backfill[raw_key] = women_seed_lookup.get(team_id, 8)

    if seed_backfill:
        pipeline.set_team_seeds(seed_backfill)

    return (
        women_id_map,
        pipeline.predict_probability,
        {
            "model_weight": float(womens_model_weight),
            "seed_weight": float(womens_seed_weight),
            "train_brier": float(train_brier),
        },
    )


def _load_womens_seed_numbers(year: int) -> dict[int, int]:
    return _load_kaggle_seed_numbers(REPO_ROOT / "data/kaggle/WNCAATourneySeeds.csv", year)


@lru_cache(maxsize=None)
def _load_womens_tournament_games(year: int) -> list[dict[str, object]]:
    """Load women's tournament games from Kaggle detailed results."""
    csv_path = REPO_ROOT / "data/kaggle/WNCAATourneyDetailedResults.csv"
    if not csv_path.exists():
        return []

    import csv

    seed_lookup = _load_womens_seed_numbers(year)
    games: list[dict[str, object]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                season = int(row.get("Season", "0"))
            except (TypeError, ValueError):
                continue
            if season != year:
                continue

            winner = int(row["WTeamID"])
            loser = int(row["LTeamID"])
            team1, team2 = sorted((winner, loser))
            team1_won = team1 == winner
            games.append(
                {
                    "team1_id": str(team1),
                    "team2_id": str(team2),
                    "team1_seed": seed_lookup.get(team1, 8),
                    "team2_seed": seed_lookup.get(team2, 8),
                    "team1_won": team1_won,
                    "gender": "women",
                }
            )
    return games


def _evaluate_games(
    games: list[dict[str, object]],
    predict_fn: Callable[[str, str], float],
    pp: PostProcessingPipeline | None = None,
) -> dict[str, float]:
    """Compute Brier, seed baseline Brier, and accuracy over a list of games."""
    from scripts.eval_brier_postprocessing import seed_implied_prob

    if not games:
        return {"brier": float("nan"), "seed_brier": float("nan"), "bss": float("nan"), "accuracy": float("nan")}

    errors = []
    seed_errors = []
    correct = 0

    for g in games:
        t1 = str(g["team1_id"])
        t2 = str(g["team2_id"])
        s1 = int(g.get("team1_seed", 8))
        s2 = int(g.get("team2_seed", 8))
        outcome = 1.0 if g["team1_won"] else 0.0
        pred = float(predict_fn(t1, t2))
        pred = _maybe_postprocess(pred, s1, s2, pp)
        errors.append((pred - outcome) ** 2)
        seed_errors.append((seed_implied_prob(s1, s2) - outcome) ** 2)
        if (pred >= 0.5) == bool(g["team1_won"]):
            correct += 1

    brier = float(np.mean(errors))
    seed_brier = float(np.mean(seed_errors))
    return {
        "brier": brier,
        "seed_brier": seed_brier,
        "bss": 1.0 - brier / seed_brier if seed_brier > 0 else float("nan"),
        "accuracy": correct / len(games),
        "games": float(len(games)),
    }


def _build_womens_predict_fn(
    year: int,
    model_weight: float = 0.55,
    seed_weight: float = 0.45,
) -> Callable[[str, str], float]:
    from src.pipeline.womens import WomensPipeline, WomensPipelineConfig

    pipeline = WomensPipeline(
        WomensPipelineConfig(
            year=year,
            cache_dir=str(REPO_ROOT / "data/raw"),
            model_weight=model_weight,
            seed_weight=seed_weight,
        )
    )
    pipeline.run()
    return pipeline.predict_probability


@lru_cache(maxsize=None)
def _cached_womens_year_brier(year: int, model_weight: float, seed_weight: float) -> float:
    predict_fn = _build_womens_predict_fn(year, model_weight=model_weight, seed_weight=seed_weight)
    metrics = _evaluate_games(_load_womens_tournament_games(year), predict_fn)
    return float(metrics["brier"])


def _select_womens_weights_for_year(
    test_year: int,
    candidate_weights: list[tuple[float, float]],
    min_train_year: int = 2010,
) -> tuple[float, float, float]:
    """Choose women's blend weights using prior years only."""
    train_years = [y for y in range(min_train_year, test_year) if y != 2020 and _load_womens_tournament_games(y)]
    if not train_years:
        return 0.55, 0.45, float("nan")

    best_pair = (0.55, 0.45)
    best_brier = float("inf")
    for model_weight, seed_weight in candidate_weights:
        train_briers = [_cached_womens_year_brier(y, model_weight, seed_weight) for y in train_years]
        mean_brier = float(np.mean(train_briers))
        if mean_brier < best_brier:
            best_brier = mean_brier
            best_pair = (model_weight, seed_weight)
    return best_pair[0], best_pair[1], best_brier


def _candidate_womens_weights(step: float) -> list[tuple[float, float]]:
    weights = []
    current = 0.0
    while current <= 1.000001:
        model_weight = round(current, 4)
        weights.append((model_weight, round(1.0 - model_weight, 4)))
        current += step
    return weights


def _womens_weight_cache_path(step: float) -> Path:
    step_tag = f"{step:.3f}".rstrip("0").rstrip(".").replace(".", "p")
    return ARTIFACTS_DIR / f"womens_walk_forward_weights_step_{step_tag}.json"


def _load_womens_weight_cache(step: float) -> dict[str, dict[str, float]]:
    path = _womens_weight_cache_path(step)
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("weights_by_year", {})


def _save_womens_weight_cache(step: float, weights_by_year: dict[str, dict[str, float]]) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    path = _womens_weight_cache_path(step)
    payload = {
        "step": step,
        "weights_by_year": weights_by_year,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def build_womens_weight_cache(
    years: list[int],
    step: float,
    refresh: bool = False,
) -> dict[str, dict[str, float]]:
    """Build or extend the cached walk-forward women's weights artifact."""
    cache = {} if refresh else _load_womens_weight_cache(step)
    candidate_weights = _candidate_womens_weights(step)

    womens_feature_logger = logging.getLogger("src.data.features.womens_feature_engineering")
    previous_womens_level = womens_feature_logger.level
    womens_feature_logger.setLevel(logging.ERROR)
    try:
        for year in years:
            year_key = str(year)
            if not refresh and year_key in cache:
                continue
            model_weight, seed_weight, train_brier = _select_womens_weights_for_year(year, candidate_weights)
            cache[year_key] = {
                "model_weight": float(model_weight),
                "seed_weight": float(seed_weight),
                "train_brier": float(train_brier),
            }
    finally:
        womens_feature_logger.setLevel(previous_womens_level)

    _save_womens_weight_cache(step, cache)
    return cache


def _get_cached_womens_weights_for_year(
    year: int,
    step: float,
    candidate_weights: list[tuple[float, float]] | None = None,
    force_refresh: bool = False,
) -> tuple[float, float, float]:
    cache = _load_womens_weight_cache(step)
    year_key = str(year)
    if not force_refresh and year_key in cache:
        info = cache[year_key]
        return float(info["model_weight"]), float(info["seed_weight"]), float(info.get("train_brier", float("nan")))

    if force_refresh:
        cache = build_womens_weight_cache([year], step, refresh=True)
    else:
        if candidate_weights is None:
            candidate_weights = _candidate_womens_weights(step)
        model_weight, seed_weight, train_brier = _select_womens_weights_for_year(year, candidate_weights)
        cache[year_key] = {
            "model_weight": float(model_weight),
            "seed_weight": float(seed_weight),
            "train_brier": float(train_brier),
        }
        _save_womens_weight_cache(step, cache)
    info = cache[year_key]
    return float(info["model_weight"]), float(info["seed_weight"]), float(info.get("train_brier", float("nan")))


def run_backtest(year: int, clip_lo: float, clip_hi: float, pp: PostProcessingPipeline | None = None) -> None:
    """Score torvik predictions against actual tournament results."""
    data = _load_context_subkey(year, "results", f"tournament_results_{year}.json")
    if data is None:
        print(f"ERROR: No tournament results for {year}")
        sys.exit(1)

    games = data.get("games", data) if isinstance(data, dict) else data
    # Exclude First Four
    games = [g for g in games if g.get("round_name", "") not in ("FF", "First Four")]

    seeds = load_tournament_seeds(year)
    predictor = TarvikKagglePredictor.from_year(year, seeds=seeds, clip_lo=clip_lo, clip_hi=clip_hi)

    print(f"\n{'=' * 70}")
    print(f"TORVIK LOG5 BACKTEST — {year} ({len(games)} games)")
    print(f"{'=' * 70}")
    print(f"  Predictor stats: {predictor.stats()}")

    ROUND_WEIGHTS = {
        "R64": 1.0,
        "R32": 2.0,
        "S16": 4.0,
        "E8": 8.0,
        "F4": 16.0,
        "NCG": 32.0,
    }

    errors, weighted_se, weight_sum, correct = [], 0.0, 0.0, 0
    per_round: dict[str, list[float]] = {}

    # Seed baseline
    seed_errors = []
    from scripts.eval_brier_postprocessing import seed_implied_prob

    for g in games:
        t1, t2 = g["team1_id"], g["team2_id"]
        outcome = 1.0 if g["team1_won"] else 0.0
        rnd = g.get("round_name", "")

        s1, s2 = g.get("team1_seed", 8), g.get("team2_seed", 8)
        pred = _maybe_postprocess(predictor.predict(t1, t2), s1, s2, pp)
        se = (pred - outcome) ** 2
        errors.append(se)
        rw = ROUND_WEIGHTS.get(rnd, 1.0)
        weighted_se += rw * se
        weight_sum += rw
        if (pred >= 0.5) == g["team1_won"]:
            correct += 1
        per_round.setdefault(rnd, []).append(se)
        seed_p = seed_implied_prob(s1, s2)
        seed_errors.append((seed_p - outcome) ** 2)

    brier = float(np.mean(errors))
    rw_brier = weighted_se / weight_sum if weight_sum > 0 else brier
    seed_brier = float(np.mean(seed_errors))
    bss = 1.0 - brier / seed_brier

    print(f"\n  Brier Score:          {brier:.4f}")
    print(f"  Round-Weighted Brier: {rw_brier:.4f}")
    print(f"  Accuracy:             {correct}/{len(games)} ({correct / len(games):.1%})")
    print(f"  Seed Baseline Brier:  {seed_brier:.4f}")
    print(f"  BSS vs Seeds:         {bss:+.4f}")
    print(f"\n  Per-round Brier:")
    for rnd in ["R64", "R32", "S16", "E8", "F4", "NCG"]:
        if rnd in per_round:
            print(f"    {rnd:4s}: {np.mean(per_round[rnd]):.4f} ({len(per_round[rnd])} games)")


def run_submission(
    year: int,
    sample_path: str,
    output: str,
    mode: str,
    clip_lo: float,
    clip_hi: float,
    pp: PostProcessingPipeline | None = None,
    ensemble_recent_year_start: int | None = 2021,
    ensemble_recent_year_weight: float = 2.0,
    ensemble_recent_year_count: int | None = None,
    ensemble_recent_total_ratio: float = 1.0,
    torvik_correction_ridge: float = 5.0,
    torvik_correction_max_correction: float = 0.10,
    womens_model_weight: float | None = None,
    womens_seed_weight: float | None = None,
    womens_weight_step: float = 0.1,
    refresh_womens_weight_cache: bool = False,
) -> None:
    """Generate a Kaggle submission CSV."""
    id_map = load_kaggle_id_mapping()
    predict_fn, predictor_info = _build_submission_predict_fn(
        year,
        mode,
        clip_lo,
        clip_hi,
        pp,
        ensemble_recent_year_start=ensemble_recent_year_start,
        ensemble_recent_year_weight=ensemble_recent_year_weight,
        ensemble_recent_year_count=ensemble_recent_year_count,
        ensemble_recent_total_ratio=ensemble_recent_total_ratio,
        torvik_correction_ridge=torvik_correction_ridge,
        torvik_correction_max_correction=torvik_correction_max_correction,
    )

    sample_df = pd.read_csv(sample_path)
    print(f"Sample submission: {len(sample_df)} rows")
    womens_id_map, womens_predict_fn, womens_info = _build_womens_submission_support(
        sample_df,
        year,
        womens_model_weight=womens_model_weight,
        womens_seed_weight=womens_seed_weight,
        womens_weight_step=womens_weight_step,
        refresh_womens_weight_cache=refresh_womens_weight_cache,
    )
    out_df = generate_predictions(
        sample_df,
        id_map=id_map,
        predict_fn=predict_fn,
        season_filter=year,
        womens_id_map=womens_id_map,
        womens_predict_fn=womens_predict_fn,
    )
    out_df.to_csv(output, index=False)

    stats = out_df.attrs.get("kaggle_export_stats", {})
    print(f"\nSubmission written to {output}")
    print(f"  Mode: {predictor_info['mode']}")
    print(f"  Mapped: {stats.get('mapped_rows', 0)}, Unmapped: {stats.get('unmapped_rows', 0)}")
    if stats.get("womens_rows", 0):
        print(f"  Women's mapped: {stats.get('womens_mapped', 0)}/{stats.get('womens_rows', 0)}")
        if womens_info is not None:
            print(f"  Women's weights: model={womens_info['model_weight']:.2f}, seed={womens_info['seed_weight']:.2f}")
    if "alpha" in predictor_info:
        print(f"  Ensemble alpha: {predictor_info['alpha']:.2f}")
        alpha_info = predictor_info.get("alpha_training_info", {})
        if alpha_info:
            print(
                "  Alpha fit: "
                f"recent_start={alpha_info.get('recent_year_start')}, "
                f"recent_weight={alpha_info.get('recent_year_weight')}"
            )
    correction_info = predictor_info.get("correction_training_info", {})
    if correction_info:
        print(
            "  Correction fit: "
            f"recent_start={correction_info.get('recent_year_start')}, "
            f"recent_weight={correction_info.get('recent_year_weight')}, "
            f"recent_count={correction_info.get('recent_year_count')}, "
            f"recent_ratio={correction_info.get('recent_total_ratio')}, "
            f"ridge={correction_info.get('correction_ridge')}, "
            f"max_corr={correction_info.get('max_correction')}"
        )
    print(f"  Predictor: {predictor_info['torvik_stats']}")


def run_multi_year_backtest(
    years: list[int], clip_lo: float, clip_hi: float, pp: PostProcessingPipeline | None = None
) -> None:
    """Run backtest across multiple years and print summary."""
    from scripts.eval_brier_postprocessing import seed_implied_prob

    ROUND_WEIGHTS = {
        "R64": 1.0,
        "R32": 2.0,
        "S16": 4.0,
        "E8": 8.0,
        "F4": 16.0,
        "NCG": 32.0,
    }

    print(f"\n{'=' * 80}")
    print(f"TORVIK LOG5 MULTI-YEAR BACKTEST ({len(years)} years, clip=[{clip_lo}, {clip_hi}])")
    print(f"{'=' * 80}")
    print(f"\n  {'Year':<6} {'Torvik':>8} {'Seed':>8} {'BSS':>8} {'Acc':>6} {'Games':>6}")
    print(f"  {'-' * 48}")

    all_brier, all_seed_brier = [], []

    for year in years:
        data = _load_context_subkey(year, "results", f"tournament_results_{year}.json")
        if data is None:
            continue

        games = data.get("games", data) if isinstance(data, dict) else data
        games = [g for g in games if g.get("round_name", "") not in ("FF", "First Four")]

        seeds = load_tournament_seeds(year)
        predictor = TarvikKagglePredictor.from_year(year, seeds=seeds, clip_lo=clip_lo, clip_hi=clip_hi)

        errors, seed_errors, correct = [], [], 0
        for g in games:
            t1, t2 = g["team1_id"], g["team2_id"]
            outcome = 1.0 if g["team1_won"] else 0.0

            s1, s2 = g.get("team1_seed", 8), g.get("team2_seed", 8)
            pred = _maybe_postprocess(predictor.predict(t1, t2), s1, s2, pp)
            errors.append((pred - outcome) ** 2)
            if (pred >= 0.5) == g["team1_won"]:
                correct += 1

            seed_errors.append((seed_implied_prob(s1, s2) - outcome) ** 2)

        brier = float(np.mean(errors))
        seed_brier = float(np.mean(seed_errors))
        bss = 1.0 - brier / seed_brier
        acc = correct / len(games)

        all_brier.append(brier)
        all_seed_brier.append(seed_brier)

        print(f"  {year:<6} {brier:>8.4f} {seed_brier:>8.4f} {bss:>+8.4f} {acc:>5.1%} {len(games):>6}")

    if all_brier:
        mean_b = np.mean(all_brier)
        mean_s = np.mean(all_seed_brier)
        mean_bss = 1.0 - mean_b / mean_s
        years_better = sum(1 for b, s in zip(all_brier, all_seed_brier) if b < s)
        print(f"  {'-' * 48}")
        print(f"  {'Mean':<6} {mean_b:>8.4f} {mean_s:>8.4f} {mean_bss:>+8.4f}")
        print(f"\n  BSS > 0 in {years_better}/{len(all_brier)} years")


def _load_pipeline_lookup(year: int) -> dict[str, float] | None:
    """Load per-game pipeline predictions for a year from backtest artifact."""
    bt_path = REPO_ROOT / "artifacts" / "backtest_result_temperature.json"
    if not bt_path.exists():
        return None
    with open(bt_path) as f:
        data = json.load(f)
    games = data.get("per_year_games", {}).get(str(year), [])
    if not games:
        return None
    lookup = {}
    for g in games:
        t1, t2 = g["team1"], g["team2"]
        lookup[f"{t1}_{t2}"] = g["pipeline"]
    return lookup


def run_ensemble_backtest(
    years: list[int],
    clip_lo: float,
    clip_hi: float,
    pp: PostProcessingPipeline | None = None,
    ensemble_recent_year_start: int | None = 2021,
    ensemble_recent_year_weight: float = 2.0,
    ensemble_recent_year_count: int | None = None,
    ensemble_recent_total_ratio: float = 1.0,
) -> None:
    """Run ensemble (torvik+pipeline) backtest with walk-forward alpha."""
    from scripts.eval_brier_postprocessing import seed_implied_prob

    print(f"\n{'=' * 85}")
    print(f"ENSEMBLE (TORVIK+PIPELINE) WALK-FORWARD BACKTEST ({len(years)} years)")
    print(f"{'=' * 85}")
    print(f"\n  {'Year':<6} {'Alpha':>6} {'Ensemble':>10} {'Torvik':>10} {'Pipeline':>10} {'Seed':>10}")
    print(f"  {'-' * 58}")

    all_ens, all_torv, all_pipe, all_seed = [], [], [], []

    for year in years:
        data = _load_context_subkey(year, "results", f"tournament_results_{year}.json")
        if data is None:
            continue
        games = data.get("games", data) if isinstance(data, dict) else data
        games = [g for g in games if g.get("round_name", "") not in ("FF", "First Four")]

        seeds = load_tournament_seeds(year)
        torvik = TarvikKagglePredictor.from_year(year, seeds=seeds, clip_lo=clip_lo, clip_hi=clip_hi)

        pipe_lookup = _load_pipeline_lookup(year)
        if pipe_lookup is None:
            continue

        def pipe_predict(t1, t2, _lookup=pipe_lookup):
            key1, key2 = f"{t1}_{t2}", f"{t2}_{t1}"
            if key1 in _lookup:
                return _lookup[key1]
            if key2 in _lookup:
                return 1.0 - _lookup[key2]
            return 0.5

        ensemble = EnsembleKagglePredictor.from_walk_forward(
            torvik,
            pipe_predict,
            year,
            clip_lo=clip_lo,
            clip_hi=clip_hi,
            recent_year_start=ensemble_recent_year_start,
            recent_year_weight=ensemble_recent_year_weight,
            recent_year_count=ensemble_recent_year_count,
            recent_total_ratio=ensemble_recent_total_ratio,
        )

        ens_errors, torv_errors, pipe_errors, seed_errors = [], [], [], []
        for g in games:
            t1, t2 = g["team1_id"], g["team2_id"]
            s1, s2 = g.get("team1_seed", 8), g.get("team2_seed", 8)
            outcome = 1.0 if g["team1_won"] else 0.0

            e_pred = _maybe_postprocess(ensemble.predict(t1, t2), s1, s2, pp)
            t_pred = torvik.predict(t1, t2)
            p_pred = pipe_predict(t1, t2)

            ens_errors.append((e_pred - outcome) ** 2)
            torv_errors.append((t_pred - outcome) ** 2)
            pipe_errors.append((p_pred - outcome) ** 2)
            seed_errors.append((seed_implied_prob(s1, s2) - outcome) ** 2)

        ens_b = float(np.mean(ens_errors))
        torv_b = float(np.mean(torv_errors))
        pipe_b = float(np.mean(pipe_errors))
        seed_b = float(np.mean(seed_errors))

        all_ens.append(ens_b)
        all_torv.append(torv_b)
        all_pipe.append(pipe_b)
        all_seed.append(seed_b)

        print(f"  {year:<6} {ensemble.alpha:>6.2f} {ens_b:>10.4f} {torv_b:>10.4f} {pipe_b:>10.4f} {seed_b:>10.4f}")

    if all_ens:
        print(f"  {'-' * 58}")
        me, mt, mp, ms = np.mean(all_ens), np.mean(all_torv), np.mean(all_pipe), np.mean(all_seed)
        print(f"  {'Mean':<6} {'':>6} {me:>10.4f} {mt:>10.4f} {mp:>10.4f} {ms:>10.4f}")
        print(f"\n  BSS vs seed:")
        print(f"    Ensemble: {1.0 - me / ms:+.4f}")
        print(f"    Torvik:   {1.0 - mt / ms:+.4f}")
        print(f"    Pipeline: {1.0 - mp / ms:+.4f}")
        ens_best = sum(1 for e, t, p in zip(all_ens, all_torv, all_pipe) if e <= t and e <= p)
        print(f"\n  Ensemble best-or-tied in {ens_best}/{len(all_ens)} years")


def run_torvik_correction_backtest(
    years: list[int],
    clip_lo: float,
    clip_hi: float,
    pp: PostProcessingPipeline | None = None,
    ensemble_recent_year_start: int | None = 2021,
    ensemble_recent_year_weight: float = 2.0,
    ensemble_recent_year_count: int | None = None,
    ensemble_recent_total_ratio: float = 1.0,
    torvik_correction_ridge: float = 5.0,
    torvik_correction_max_correction: float = 0.10,
) -> None:
    """Run torvik + correction backtest against torvik and the current blend."""
    from scripts.eval_brier_postprocessing import seed_implied_prob

    print(f"\n{'=' * 98}")
    print(f"TORVIK CORRECTION WALK-FORWARD BACKTEST ({len(years)} years)")
    print(f"{'=' * 98}")
    print(f"\n  {'Year':<6} {'Corrected':>10} {'Blend':>10} {'Torvik':>10} {'Seed':>10} {'BSS-C':>8} {'BSS-B':>8}")
    print(f"  {'-' * 70}")

    all_corrected, all_blend, all_torvik, all_seed = [], [], [], []

    for year in years:
        data = _load_context_subkey(year, "results", f"tournament_results_{year}.json")
        if data is None:
            continue
        games = data.get("games", data) if isinstance(data, dict) else data
        games = [g for g in games if g.get("round_name", "") not in ("FF", "First Four")]

        seeds = load_tournament_seeds(year)
        torvik = TarvikKagglePredictor.from_year(year, seeds=seeds, clip_lo=clip_lo, clip_hi=clip_hi)
        try:
            correction = _build_torvik_correction_model(
                year,
                clip_lo=clip_lo,
                clip_hi=clip_hi,
                recent_year_start=ensemble_recent_year_start,
                recent_year_weight=ensemble_recent_year_weight,
                recent_year_count=ensemble_recent_year_count,
                recent_total_ratio=ensemble_recent_total_ratio,
                correction_ridge=torvik_correction_ridge,
                max_correction=torvik_correction_max_correction,
            )
        except RuntimeError:
            continue

        pipe_lookup = _load_pipeline_lookup(year)
        if pipe_lookup is None:
            continue

        def pipe_predict(t1, t2, _lookup=pipe_lookup):
            key1, key2 = f"{t1}_{t2}", f"{t2}_{t1}"
            if key1 in _lookup:
                return _lookup[key1]
            if key2 in _lookup:
                return 1.0 - _lookup[key2]
            return 0.5

        ensemble = EnsembleKagglePredictor.from_walk_forward(
            torvik,
            pipe_predict,
            year,
            clip_lo=clip_lo,
            clip_hi=clip_hi,
            recent_year_start=ensemble_recent_year_start,
            recent_year_weight=ensemble_recent_year_weight,
            recent_year_count=ensemble_recent_year_count,
            recent_total_ratio=ensemble_recent_total_ratio,
        )

        corrected_errors, blend_errors, torvik_errors, seed_errors = [], [], [], []
        for g in games:
            t1, t2 = g["team1_id"], g["team2_id"]
            s1, s2 = g.get("team1_seed", 8), g.get("team2_seed", 8)
            outcome = 1.0 if g["team1_won"] else 0.0

            t_pred = torvik.predict(t1, t2)
            c_pred = _maybe_postprocess(correction.predict_one(t_pred, s1, s2), s1, s2, pp)
            b_pred = _maybe_postprocess(ensemble.predict(t1, t2), s1, s2, pp)

            corrected_errors.append((c_pred - outcome) ** 2)
            blend_errors.append((b_pred - outcome) ** 2)
            torvik_errors.append((t_pred - outcome) ** 2)
            seed_errors.append((seed_implied_prob(s1, s2) - outcome) ** 2)

        corrected_b = float(np.mean(corrected_errors))
        blend_b = float(np.mean(blend_errors))
        torvik_b = float(np.mean(torvik_errors))
        seed_b = float(np.mean(seed_errors))

        all_corrected.append(corrected_b)
        all_blend.append(blend_b)
        all_torvik.append(torvik_b)
        all_seed.append(seed_b)

        print(
            f"  {year:<6} {corrected_b:>10.4f} {blend_b:>10.4f} {torvik_b:>10.4f} {seed_b:>10.4f}"
            f" {1.0 - corrected_b / seed_b:>+8.4f} {1.0 - blend_b / seed_b:>+8.4f}"
        )

    if all_corrected:
        print(f"  {'-' * 70}")
        mc, mb, mt, ms = np.mean(all_corrected), np.mean(all_blend), np.mean(all_torvik), np.mean(all_seed)
        print(f"  {'Mean':<6} {mc:>10.4f} {mb:>10.4f} {mt:>10.4f} {ms:>10.4f}")
        print(f"\n  BSS vs seed:")
        print(f"    Corrected: {1.0 - mc / ms:+.4f}")
        print(f"    Blend:     {1.0 - mb / ms:+.4f}")
        print(f"    Torvik:    {1.0 - mt / ms:+.4f}")
        correction_best = sum(1 for c, b, t in zip(all_corrected, all_blend, all_torvik) if c <= b and c <= t)
        print(f"\n  Corrected best-or-tied in {correction_best}/{len(all_corrected)} years")


def run_combined_backtest(
    years: list[int],
    mode: str,
    clip_lo: float,
    clip_hi: float,
    pp: PostProcessingPipeline | None = None,
    ensemble_recent_year_start: int | None = 2021,
    ensemble_recent_year_weight: float = 2.0,
    ensemble_recent_year_count: int | None = None,
    ensemble_recent_total_ratio: float = 1.0,
    torvik_correction_ridge: float = 5.0,
    torvik_correction_max_correction: float = 0.10,
    tune_womens_weights: bool = False,
    womens_model_weight: float = 0.55,
    womens_seed_weight: float = 0.45,
    womens_weight_step: float = 0.1,
    refresh_womens_weight_cache: bool = False,
) -> None:
    """Run a combined men+women Kaggle-style Brier backtest."""
    if tune_womens_weights and refresh_womens_weight_cache:
        build_womens_weight_cache(years, womens_weight_step, refresh=True)

    womens_feature_logger = logging.getLogger("src.data.features.womens_feature_engineering")
    previous_womens_level = womens_feature_logger.level
    womens_feature_logger.setLevel(logging.ERROR)

    print(f"\n{'=' * 98}")
    print(f"COMBINED MEN+WOMEN KAGGLE BACKTEST ({len(years)} years, mode={mode})")
    print(f"{'=' * 98}")
    print(
        f"\n  {'Year':<6} {'Men':>8} {'Women':>8} {'Combined':>9} {'Seed':>8} {'BSS':>8} {'W-Model':>8} {'W-Seed':>8}"
    )
    print(f"  {'-' * 75}")

    candidate_weights = _candidate_womens_weights(womens_weight_step)
    combined_briers = []
    combined_seed_briers = []

    try:
        for year in years:
            data = _load_context_subkey(year, "results", f"tournament_results_{year}.json")
            if data is None:
                continue

            mens_games = data.get("games", data) if isinstance(data, dict) else data
            mens_games = [g for g in mens_games if g.get("round_name", "") not in ("FF", "First Four")]

            womens_games = _load_womens_tournament_games(year)
            if not womens_games:
                continue

            mens_predict_fn, _info = _build_submission_predict_fn(
                year,
                mode,
                clip_lo,
                clip_hi,
                pp=None,
                ensemble_recent_year_start=ensemble_recent_year_start,
                ensemble_recent_year_weight=ensemble_recent_year_weight,
                ensemble_recent_year_count=ensemble_recent_year_count,
                ensemble_recent_total_ratio=ensemble_recent_total_ratio,
                torvik_correction_ridge=torvik_correction_ridge,
                torvik_correction_max_correction=torvik_correction_max_correction,
            )
            mens_metrics = _evaluate_games(mens_games, mens_predict_fn, pp)

            if tune_womens_weights:
                best_model_weight, best_seed_weight, _train_brier = _get_cached_womens_weights_for_year(
                    year,
                    womens_weight_step,
                    candidate_weights=candidate_weights,
                    force_refresh=False,
                )
            else:
                best_model_weight, best_seed_weight = womens_model_weight, womens_seed_weight

            womens_predict_fn = _build_womens_predict_fn(
                year,
                model_weight=best_model_weight,
                seed_weight=best_seed_weight,
            )
            womens_metrics = _evaluate_games(womens_games, womens_predict_fn, pp=None)

            mens_n = int(mens_metrics["games"])
            womens_n = int(womens_metrics["games"])
            total_n = mens_n + womens_n
            combined_brier = (mens_metrics["brier"] * mens_n + womens_metrics["brier"] * womens_n) / total_n
            combined_seed = (mens_metrics["seed_brier"] * mens_n + womens_metrics["seed_brier"] * womens_n) / total_n
            combined_bss = 1.0 - combined_brier / combined_seed if combined_seed > 0 else float("nan")

            combined_briers.append(combined_brier)
            combined_seed_briers.append(combined_seed)

            print(
                f"  {year:<6} {mens_metrics['brier']:>8.4f} {womens_metrics['brier']:>8.4f} "
                f"{combined_brier:>9.4f} {combined_seed:>8.4f} {combined_bss:>+8.4f} "
                f"{best_model_weight:>8.2f} {best_seed_weight:>8.2f}"
            )

        if combined_briers:
            mean_brier = float(np.mean(combined_briers))
            mean_seed = float(np.mean(combined_seed_briers))
            mean_bss = 1.0 - mean_brier / mean_seed if mean_seed > 0 else float("nan")
            print(f"  {'-' * 75}")
            print(f"  {'Mean':<6} {'':>8} {'':>8} {mean_brier:>9.4f} {mean_seed:>8.4f} {mean_bss:>+8.4f}")
    finally:
        womens_feature_logger.setLevel(previous_womens_level)


def main():
    parser = argparse.ArgumentParser(description="Torvik-based Kaggle submission")
    parser.add_argument("--year", type=int, required=True, help="Tournament year")
    parser.add_argument("--sample-submission", default=None, help="Kaggle SampleSubmission CSV")
    parser.add_argument("--output", "-o", default="kaggle_torvik_submission.csv", help="Output CSV")
    parser.add_argument("--backtest", action="store_true", help="Score against actual results")
    parser.add_argument("--multi-year-backtest", action="store_true", help="Run backtest across all available years")
    parser.add_argument(
        "--combined-backtest",
        action="store_true",
        help="Run combined men+women Kaggle-style backtest on overlapping historical years",
    )
    parser.add_argument(
        "--mode",
        choices=["torvik", "ensemble", "torvik_corrected", TORVIK_CORRECTED_RECENT5_CONSERVATIVE],
        default="torvik",
        help="Prediction mode: torvik, ensemble, torvik_corrected, or the admitted recent-5 conservative preset",
    )
    parser.add_argument("--clip-lo", type=float, default=0.01, help="Lower probability clip")
    parser.add_argument("--clip-hi", type=float, default=0.99, help="Upper probability clip")
    parser.add_argument(
        "--ensemble-recency-start-year",
        type=int,
        default=2021,
        help="Upweight ensemble alpha fit for seasons from this year onward (default: 2021)",
    )
    parser.add_argument(
        "--ensemble-recency-weight",
        type=float,
        default=2.0,
        help="Relative weight for recent years in ensemble alpha fit (default: 2.0)",
    )
    parser.add_argument(
        "--ensemble-recency-count",
        type=int,
        default=None,
        help="Use the most recent N observed seasons as a weighted recent block; overrides start-year semantics",
    )
    parser.add_argument(
        "--ensemble-recency-total-ratio",
        type=float,
        default=1.0,
        help="When using --ensemble-recency-count, set recent-block total weight as this multiple of older-history total weight",
    )
    parser.add_argument(
        "--torvik-correction-ridge",
        type=float,
        default=5.0,
        help="Ridge regularization for torvik correction mode (default: 5.0)",
    )
    parser.add_argument(
        "--torvik-correction-max-correction",
        type=float,
        default=0.10,
        help="Max absolute bounded adjustment for torvik correction mode (default: 0.10)",
    )
    parser.add_argument("--flb-threshold", type=float, default=None, help="FLB correction threshold (e.g. 0.85)")
    parser.add_argument("--flb-regression", type=float, default=0.20, help="FLB regression strength (default 0.20)")
    parser.add_argument(
        "--womens-submission-model-weight",
        type=float,
        default=None,
        help="Override women's model weight for submission export; default is walk-forward selected",
    )
    parser.add_argument(
        "--womens-submission-seed-weight",
        type=float,
        default=None,
        help="Override women's seed weight for submission export; default is walk-forward selected",
    )
    parser.add_argument(
        "--refresh-womens-weight-cache",
        action="store_true",
        help="Recompute cached walk-forward women's weights for the requested grid step",
    )
    parser.add_argument(
        "--tune-womens-weights",
        action="store_true",
        help="Walk-forward tune women's model/seed blend on prior years during combined backtest",
    )
    parser.add_argument("--womens-model-weight", type=float, default=0.55, help="Women's model blend weight")
    parser.add_argument("--womens-seed-weight", type=float, default=0.45, help="Women's seed blend weight")
    parser.add_argument(
        "--womens-weight-step", type=float, default=0.1, help="Grid step for women's walk-forward tuning"
    )
    args = parser.parse_args()

    # Build post-processor if FLB enabled
    pp = None
    if args.flb_threshold is not None:
        pp = PostProcessingPipeline(
            flb_threshold=args.flb_threshold,
            flb_regression=args.flb_regression,
            clip_lo=args.clip_lo,
            clip_hi=args.clip_hi,
        )
    if args.ensemble_recency_weight <= 0:
        parser.error("--ensemble-recency-weight must be positive")
    if args.ensemble_recency_count is not None and args.ensemble_recency_count <= 0:
        parser.error("--ensemble-recency-count must be positive")
    if args.ensemble_recency_total_ratio <= 0:
        parser.error("--ensemble-recency-total-ratio must be positive")
    if args.torvik_correction_ridge <= 0:
        parser.error("--torvik-correction-ridge must be positive")
    if args.torvik_correction_max_correction <= 0:
        parser.error("--torvik-correction-max-correction must be positive")

    resolved_mode_config = _resolve_mens_kaggle_mode_config(
        args.mode,
        args.ensemble_recency_start_year,
        args.ensemble_recency_weight,
        args.ensemble_recency_count,
        args.ensemble_recency_total_ratio,
        args.torvik_correction_ridge,
        args.torvik_correction_max_correction,
    )

    if args.combined_backtest:
        years = list(range(2010, 2026))
        years = [y for y in years if y != 2020]
        run_combined_backtest(
            years,
            args.mode,
            args.clip_lo,
            args.clip_hi,
            pp,
            ensemble_recent_year_start=args.ensemble_recency_start_year,
            ensemble_recent_year_weight=args.ensemble_recency_weight,
            ensemble_recent_year_count=args.ensemble_recency_count,
            ensemble_recent_total_ratio=args.ensemble_recency_total_ratio,
            torvik_correction_ridge=args.torvik_correction_ridge,
            torvik_correction_max_correction=args.torvik_correction_max_correction,
            tune_womens_weights=args.tune_womens_weights,
            womens_model_weight=args.womens_model_weight,
            womens_seed_weight=args.womens_seed_weight,
            womens_weight_step=args.womens_weight_step,
            refresh_womens_weight_cache=args.refresh_womens_weight_cache,
        )
    elif args.multi_year_backtest:
        years = list(range(2008, 2027))
        years = [y for y in years if y != 2020]
        if args.mode == "ensemble":
            run_ensemble_backtest(
                years,
                args.clip_lo,
                args.clip_hi,
                pp,
                ensemble_recent_year_start=resolved_mode_config["recent_year_start"],
                ensemble_recent_year_weight=float(resolved_mode_config["recent_year_weight"]),
                ensemble_recent_year_count=resolved_mode_config["recent_year_count"],
                ensemble_recent_total_ratio=float(resolved_mode_config["recent_total_ratio"]),
            )
        elif args.mode in ("torvik_corrected", TORVIK_CORRECTED_RECENT5_CONSERVATIVE):
            run_torvik_correction_backtest(
                years,
                args.clip_lo,
                args.clip_hi,
                pp,
                ensemble_recent_year_start=resolved_mode_config["recent_year_start"],
                ensemble_recent_year_weight=float(resolved_mode_config["recent_year_weight"]),
                ensemble_recent_year_count=resolved_mode_config["recent_year_count"],
                ensemble_recent_total_ratio=float(resolved_mode_config["recent_total_ratio"]),
                torvik_correction_ridge=float(resolved_mode_config["correction_ridge"]),
                torvik_correction_max_correction=float(resolved_mode_config["max_correction"]),
            )
        else:
            run_multi_year_backtest(years, args.clip_lo, args.clip_hi, pp)
    elif args.backtest:
        run_backtest(args.year, args.clip_lo, args.clip_hi, pp)
    elif args.sample_submission:
        run_submission(
            args.year,
            args.sample_submission,
            args.output,
            args.mode,
            args.clip_lo,
            args.clip_hi,
            pp,
            ensemble_recent_year_start=args.ensemble_recency_start_year,
            ensemble_recent_year_weight=args.ensemble_recency_weight,
            ensemble_recent_year_count=args.ensemble_recency_count,
            ensemble_recent_total_ratio=args.ensemble_recency_total_ratio,
            torvik_correction_ridge=args.torvik_correction_ridge,
            torvik_correction_max_correction=args.torvik_correction_max_correction,
            womens_model_weight=args.womens_submission_model_weight,
            womens_seed_weight=args.womens_submission_seed_weight,
            womens_weight_step=args.womens_weight_step,
            refresh_womens_weight_cache=args.refresh_womens_weight_cache,
        )
    else:
        print("Specify --backtest, --multi-year-backtest, --combined-backtest, or --sample-submission")
        sys.exit(1)


if __name__ == "__main__":
    main()
