"""Monte Carlo parameter calibration against historical first-round upset rates."""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .monte_carlo import MonteCarloEngine, SimulationConfig, TournamentTeam, TournamentBracket
from .monte_carlo import validate_upset_rates
from ..data.normalize import normalize_team_id, strip_ncaa_suffix


FIRST_ROUND_MATCHUPS = {
    (1, 16),
    (2, 15),
    (3, 14),
    (4, 13),
    (5, 12),
    (6, 11),
    (7, 10),
    (8, 9),
}


def _is_tournament_game(date_str: str, year: int) -> bool:
    try:
        parts = date_str.split("-")
        game_year = int(parts[0])
        month = int(parts[1])
        day = int(parts[2])
        from datetime import date as _dtdate
        game_day = _dtdate(game_year, month, day)
        start = _dtdate(year, 3, 14)
        end = _dtdate(year, 4, 15)
        return start <= game_day <= end
    except (ValueError, IndexError):
        return False


def _resolve_seed_id(raw_id: str, seed_ids_sorted: List[str]) -> Optional[str]:
    if not raw_id:
        return None
    norm = strip_ncaa_suffix(normalize_team_id(raw_id))
    if norm in seed_ids_sorted:
        return norm
    for sid in seed_ids_sorted:
        if norm.startswith(sid):
            return sid
    return None


def _load_seeds(path: str) -> Dict[str, Dict[str, object]]:
    with open(path, "r") as f:
        payload = json.load(f)
    teams = payload.get("teams", payload if isinstance(payload, list) else [])
    seed_info: Dict[str, Dict[str, object]] = {}
    for entry in teams:
        if not isinstance(entry, dict):
            continue
        raw_id = entry.get("team_id") or entry.get("school_slug") or entry.get("team_name") or ""
        team_id = strip_ncaa_suffix(normalize_team_id(raw_id))
        if not team_id:
            continue
        seed = int(entry.get("seed", 0) or 0)
        region = entry.get("region") or ""
        if seed <= 0 or not region:
            continue
        seed_info[team_id] = {"seed": seed, "region": region}
    return seed_info


def _load_team_strengths(
    metrics_path: str,
    seed_info: Dict[str, Dict[str, object]],
) -> Dict[str, float]:
    strengths: Dict[str, List[float]] = {}
    seed_ids_sorted = sorted(seed_info.keys(), key=len, reverse=True)
    with open(metrics_path, "r") as f:
        payload = json.load(f)
    teams = payload.get("teams", [])
    if not isinstance(teams, list):
        return {}
    for tm in teams:
        if not isinstance(tm, dict):
            continue
        raw_id = tm.get("team_id") or tm.get("name") or ""
        seed_id = _resolve_seed_id(str(raw_id), seed_ids_sorted)
        if seed_id is None or seed_id not in seed_info:
            continue
        off = float(tm.get("off_rtg", 0.0))
        drt = float(tm.get("def_rtg", 0.0))
        if off <= 1e-6 or drt <= 1e-6:
            continue
        strengths.setdefault(seed_id, []).append(off - drt)
    return {k: float(np.mean(v)) for k, v in strengths.items()}


def _extract_first_round_games(
    games_payload: Dict,
    seed_info: Dict[str, Dict[str, object]],
    year: int,
) -> Tuple[Dict[Tuple[int, int], Tuple[int, int]], List[str]]:
    seed_ids_sorted = sorted(seed_info.keys(), key=len, reverse=True)
    games = games_payload.get("games", [])
    match_counts: Dict[Tuple[int, int], Tuple[int, int]] = {}
    r64_teams: set = set()
    for g in games:
        if not isinstance(g, dict):
            continue
        date_str = str(g.get("date") or g.get("game_date") or "")
        if not _is_tournament_game(date_str, year):
            continue
        t1 = _resolve_seed_id(str(g.get("team1_id", "")), seed_ids_sorted)
        t2 = _resolve_seed_id(str(g.get("team2_id", "")), seed_ids_sorted)
        if t1 is None or t2 is None:
            continue
        s1 = int(seed_info.get(t1, {}).get("seed", 0))
        s2 = int(seed_info.get(t2, {}).get("seed", 0))
        if s1 <= 0 or s2 <= 0:
            continue
        pair = (min(s1, s2), max(s1, s2))
        if pair not in FIRST_ROUND_MATCHUPS:
            continue
        score1 = int(g.get("team1_score", 0) or 0)
        score2 = int(g.get("team2_score", 0) or 0)
        if score1 <= 0 or score2 <= 0:
            continue
        high_seed = min(s1, s2)
        low_seed = max(s1, s2)
        low_team_won = (s1 == low_seed and score1 > score2) or (s2 == low_seed and score2 > score1)
        wins, total = match_counts.get(pair, (0, 0))
        match_counts[pair] = (wins + (1 if low_team_won else 0), total + 1)
        r64_teams.update([t1, t2])
    return match_counts, sorted(r64_teams)


def _build_bracket_teams(
    seed_info: Dict[str, Dict[str, object]],
    r64_team_ids: List[str],
) -> Dict[str, List[TournamentTeam]]:
    teams_by_region: Dict[str, List[TournamentTeam]] = {}
    for team_id in r64_team_ids:
        info = seed_info.get(team_id)
        if not info:
            continue
        seed = int(info.get("seed", 0) or 0)
        region = str(info.get("region", ""))
        if seed <= 0 or not region:
            continue
        teams_by_region.setdefault(region, []).append(
            TournamentTeam(team_id=team_id, seed=seed, region=region, strength=0.5)
        )
    # Ensure each region has seeds 1-16
    filtered = {}
    for region, teams in teams_by_region.items():
        by_seed = {}
        for t in teams:
            by_seed[t.seed] = t
        if set(by_seed.keys()) == set(range(1, 17)):
            filtered[region] = [by_seed[s] for s in sorted(by_seed)]
    return filtered


def _predict_fn_factory(
    seed_info: Dict[str, Dict[str, object]],
    strengths: Dict[str, float],
    em_slope: float,
    seed_slope: float,
):
    def _seed_prob(s1: int, s2: int) -> float:
        diff = s2 - s1
        return 1.0 / (1.0 + math.exp(-seed_slope * diff))

    def _predict(team1_id: str, team2_id: str) -> float:
        s1 = int(seed_info.get(team1_id, {}).get("seed", 0))
        s2 = int(seed_info.get(team2_id, {}).get("seed", 0))
        e1 = strengths.get(team1_id)
        e2 = strengths.get(team2_id)
        if e1 is None or e2 is None:
            if s1 > 0 and s2 > 0:
                return _seed_prob(s1, s2)
            return 0.5
        diff = e1 - e2
        return 1.0 / (1.0 + math.exp(-em_slope * diff))

    return _predict


@dataclass
class MCCalibrationResult:
    noise_std: float
    regional_correlation: float
    dev_score: float
    holdout_score: Optional[float]
    per_year_scores: Dict[int, float]

    def to_dict(self) -> Dict[str, object]:
        return {
            "noise_std": round(self.noise_std, 4),
            "regional_correlation": round(self.regional_correlation, 4),
            "dev_score": round(self.dev_score, 5),
            "holdout_score": round(self.holdout_score, 5) if self.holdout_score is not None else None,
            "per_year_scores": {str(k): round(v, 5) for k, v in self.per_year_scores.items()},
        }


def _score_year(
    year: int,
    historical_dir: str,
    noise_std: float,
    regional_correlation: float,
    num_simulations: int,
    em_slope: float,
    seed_slope: float,
    random_seed: int,
    parallel_workers: Optional[int],
) -> Optional[Tuple[float, Dict[Tuple[int, int], float]]]:
    seeds_path = os.path.join(historical_dir, f"tournament_seeds_{year}.json")
    metrics_path = os.path.join(historical_dir, f"team_metrics_{year}.json")
    games_path = os.path.join(historical_dir, f"historical_games_{year}.json")
    if not (os.path.isfile(seeds_path) and os.path.isfile(metrics_path) and os.path.isfile(games_path)):
        return None

    seed_info = _load_seeds(seeds_path)
    if not seed_info:
        return None

    with open(games_path, "r") as f:
        games_payload = json.load(f)

    actual_counts, r64_team_ids = _extract_first_round_games(games_payload, seed_info, year)
    if not actual_counts or not r64_team_ids:
        return None

    actual_rates: Dict[Tuple[int, int], float] = {}
    for pair, (wins, total) in actual_counts.items():
        if total > 0:
            actual_rates[pair] = wins / total

    teams_by_region = _build_bracket_teams(seed_info, r64_team_ids)
    if not teams_by_region:
        return None

    strengths = _load_team_strengths(metrics_path, seed_info)
    predict_fn = _predict_fn_factory(seed_info, strengths, em_slope, seed_slope)

    cfg = SimulationConfig(
        num_simulations=num_simulations,
        noise_std=noise_std,
        injury_probability=0.0,
        random_seed=random_seed,
        batch_size=500,
        regional_correlation=regional_correlation,
        parallel_workers=parallel_workers,
    )
    bracket = TournamentBracket.create_standard_bracket(teams_by_region)
    engine = MonteCarloEngine(predict_fn, config=cfg)
    sim_results = engine.simulate_tournament(bracket, show_progress=False)
    sim_validation = validate_upset_rates(sim_results, teams_by_region)

    sim_rates = {}
    for k, entry in sim_validation.get("per_matchup", {}).items():
        try:
            hi, lo = k.split("v")
            pair = (int(hi), int(lo))
        except Exception:
            continue
        sim_rates[pair] = float(entry.get("simulated", 0.0))

    total_weight = 0.0
    total_error = 0.0
    for pair, actual in actual_rates.items():
        sim = sim_rates.get(pair)
        if sim is None:
            continue
        wins, total = actual_counts.get(pair, (0, 0))
        weight = float(total)
        total_weight += weight
        total_error += weight * abs(sim - actual)
    if total_weight <= 0:
        return None
    score = total_error / total_weight
    return score, actual_rates


def calibrate_mc_parameters(
    historical_dir: str,
    dev_years: List[int],
    holdout_years: Optional[List[int]] = None,
    noise_grid: Optional[List[float]] = None,
    corr_grid: Optional[List[float]] = None,
    num_simulations: int = 5000,
    em_slope: float = 0.1735,
    seed_slope: float = 0.175,
    random_seed: int = 42,
    parallel_workers: Optional[int] = None,
) -> Dict[str, object]:
    """Grid search MC parameters to minimize seed-upset MAE on dev years."""
    dev_years = [y for y in dev_years if y != 2020]
    holdout_years = [y for y in (holdout_years or []) if y != 2020]
    noise_grid = noise_grid or [0.06, 0.08, 0.10, 0.12, 0.14, 0.16]
    corr_grid = corr_grid or [0.0, 0.05, 0.10, 0.15]

    results: List[MCCalibrationResult] = []
    best: Optional[MCCalibrationResult] = None

    for ns in noise_grid:
        for rc in corr_grid:
            per_year_scores: Dict[int, float] = {}
            for yr in dev_years:
                scored = _score_year(
                    yr, historical_dir, ns, rc, num_simulations,
                    em_slope, seed_slope, random_seed, parallel_workers,
                )
                if scored is None:
                    continue
                per_year_scores[yr] = scored[0]
            if not per_year_scores:
                continue
            dev_score = float(np.mean(list(per_year_scores.values())))

            holdout_scores = []
            for yr in holdout_years:
                scored = _score_year(
                    yr, historical_dir, ns, rc, num_simulations,
                    em_slope, seed_slope, random_seed, parallel_workers,
                )
                if scored is None:
                    continue
                holdout_scores.append(scored[0])
            holdout_score = float(np.mean(holdout_scores)) if holdout_scores else None

            result = MCCalibrationResult(
                noise_std=ns,
                regional_correlation=rc,
                dev_score=dev_score,
                holdout_score=holdout_score,
                per_year_scores=per_year_scores,
            )
            results.append(result)
            if best is None or dev_score < best.dev_score:
                best = result

    if best is None:
        raise ValueError("No valid calibration results (missing data?)")

    return {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "historical_dir": historical_dir,
            "dev_years": dev_years,
            "holdout_years": holdout_years,
            "num_simulations": num_simulations,
            "objective": "seed_upset_mae_weighted",
            "em_slope": em_slope,
            "seed_slope": seed_slope,
        },
        "best_params": {
            "noise_std": best.noise_std,
            "regional_correlation": best.regional_correlation,
        },
        "best_dev_score": best.dev_score,
        "holdout_score": best.holdout_score,
        "grid_results": [r.to_dict() for r in results],
    }
