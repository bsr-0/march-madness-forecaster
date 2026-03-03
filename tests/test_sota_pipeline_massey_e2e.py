"""End-to-end test: SOTAPipeline.run() with Massey Ordinals integration.

This test verifies the full pipeline execution path with synthetic data
that includes Massey Ordinals.  It closes the gap where individual Massey
components were tested in isolation but no test called pipeline.run()
end-to-end and verified that Massey data flowed through to the final
report artifacts.

The test creates:
  - 68 synthetic tournament teams (4 regions x 17 seeds)
  - Torvik stats with varying metrics (not placeholder)
  - Historical games covering all 68 teams
  - Public picks for all teams (3 sources)
  - Rosters with player data
  - Kaggle MMasseyOrdinals.csv with 7 rating systems
  - Pre-populated external ratings cache

Then runs SOTAPipeline.run() and asserts:
  - Pipeline completes without error
  - massey_coverage stats are present in ml_diagnostics
  - Coverage percentage is > 50% (not seed-fallback)
  - Feature vectors have non-zero external_rating_composite
  - Report contains expected rubric artifacts
"""

import csv
import json
import math
import os
from pathlib import Path
from typing import Dict, List

import pytest


# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------

N_TEAMS = 64
REGIONS = ["East", "West", "South", "Midwest"]
YEAR = 2025


def _team_name(i: int) -> str:
    return f"Team{i}"


def _team_id(i: int) -> str:
    return f"team{i}"


def _write_csv(path: Path, headers: list, rows: list):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


def _make_teams_json(path: Path) -> str:
    """Create a teams JSON with 68 teams across 4 regions."""
    teams = []
    for i in range(N_TEAMS):
        region = REGIONS[i % 4]
        seed = (i // 4) + 1
        teams.append({
            "name": _team_name(i),
            "seed": min(seed, 16),
            "region": region,
        })
    fp = path / "teams.json"
    with open(fp, "w") as f:
        json.dump({"teams": teams}, f)
    return str(fp)


def _make_torvik_json(path: Path) -> str:
    """Create torvik data with varying stats (passes placeholder check)."""
    teams = []
    for i in range(N_TEAMS):
        rank = i + 1
        barthag = round(0.95 - (i * 0.008), 4)
        adj_o = round(120.0 - i * 0.4, 1)
        adj_d = round(90.0 + i * 0.3, 1)
        teams.append({
            "team_id": _team_id(i),
            "name": _team_name(i),
            "team_name": _team_name(i),
            "conference": ["ACC", "SEC", "B12", "B10"][i % 4],
            "t_rank": rank,
            "barthag": barthag,
            "adj_offensive_efficiency": adj_o,
            "adj_defensive_efficiency": adj_d,
            "adj_tempo": round(65.0 + (i % 10) * 0.5, 1),
            "effective_fg_pct": round(0.55 - i * 0.001, 3),
            "turnover_rate": round(0.17 + i * 0.001, 3),
            "offensive_reb_rate": round(0.32 - i * 0.001, 3),
            "free_throw_rate": round(0.35 - i * 0.0005, 4),
            "opp_effective_fg_pct": round(0.42 + i * 0.001, 3),
            "opp_turnover_rate": round(0.20 - i * 0.001, 3),
            "defensive_reb_rate": round(0.76 - i * 0.001, 3),
            "opp_free_throw_rate": round(0.30 + i * 0.0005, 4),
            "two_pt_pct": round(0.52 - i * 0.001, 3),
            "three_pt_pct": round(0.37 - i * 0.001, 3),
            "three_pt_rate": round(0.40 + i * 0.001, 3),
            "ft_pct": round(0.76 - i * 0.001, 3),
            "block_pct": round(0.10 - i * 0.0005, 4),
            "steal_pct": round(0.09 - i * 0.0004, 4),
            "wab": round(5.0 - i * 0.15, 2),
            "wins": max(30 - i, 5),
            "losses": min(2 + i, 25),
        })
    fp = path / "torvik.json"
    with open(fp, "w") as f:
        json.dump({"teams": teams}, f)
    return str(fp)


def _make_historical_games_json(path: Path) -> str:
    """Create historical games within the 2024-25 season window.

    Generates games so every team appears as team1_id in at least a few
    games (required by the proprietary metrics engine for Elo computation).
    """
    games = []
    game_id = 1000
    seen = set()
    for i in range(N_TEAMS):
        # Each team plays 3 games as team1 against the next 3 teams
        for offset in [1, 2, 3]:
            j = (i + offset) % N_TEAMS
            pair = (min(i, j), max(i, j))
            if pair in seen:
                continue
            seen.add(pair)
            score1 = 70 + (i % 15) + (game_id % 10)
            score2 = 68 + (j % 15) + ((game_id + 3) % 10)
            month = 11 + ((game_id - 1000) % 4)  # Nov-Feb
            day = 1 + ((game_id - 1000) % 28)
            if month > 12:
                month_str = f"2025-{month - 12:02d}"
            else:
                month_str = f"2024-{month:02d}"
            games.append({
                "game_id": f"g{game_id}",
                "date": f"{month_str}-{day:02d}",
                "season": YEAR,
                "team1_id": _team_id(i),
                "team1_name": _team_name(i),
                "team2_id": _team_id(j),
                "team2_name": _team_name(j),
                "team1_score": score1,
                "team2_score": score2,
                "lead_history": [0, score1 - score2],
            })
            game_id += 1
    fp = path / "historical_games.json"
    with open(fp, "w") as f:
        json.dump({"season": YEAR, "games": games}, f)
    return str(fp)


def _make_public_picks_json(path: Path) -> str:
    """Create public picks with 3 sources for all teams."""
    sources: Dict[str, Dict] = {}
    for source_name in ["espn", "yahoo", "cbs"]:
        teams_data: Dict[str, Dict] = {}
        for i in range(N_TEAMS):
            tid = _team_id(i)
            seed = min((i // 4) + 1, 16)
            # Better seeds get higher pick percentages
            base = max(99 - seed * 5, 5)
            teams_data[tid] = {
                "team_name": _team_name(i),
                "seed": seed,
                "region": REGIONS[i % 4],
                "round_of_64_pct": float(min(base + 10, 99)),
                "round_of_32_pct": float(max(base, 10)),
                "sweet_16_pct": float(max(base - 15, 5)),
                "elite_8_pct": float(max(base - 30, 3)),
                "final_four_pct": float(max(base - 50, 2)),
                "champion_pct": float(max(base - 70, 1)),
            }
        sources[source_name] = {"teams": teams_data}
    fp = path / "public_picks.json"
    with open(fp, "w") as f:
        json.dump(sources, f)
    return str(fp)


def _make_roster_json(path: Path) -> str:
    """Create roster data with 8 players per team."""
    teams = []
    positions = ["PG", "SG", "SF", "PF", "C", "SG", "SF", "PF"]
    for i in range(N_TEAMS):
        players = []
        for p in range(8):
            players.append({
                "player_id": f"{_team_id(i)}_p{p}",
                "name": f"Player{p} of {_team_name(i)}",
                "position": positions[p],
                "minutes_per_game": round(32.0 - p * 2.5, 1),
                "games_played": 30,
                "games_started": max(30 - p * 5, 0),
                "rapm_offensive": round(3.0 - p * 0.5, 1),
                "rapm_defensive": round(2.0 - p * 0.3, 1),
                "warp": round(0.6 - p * 0.08, 2),
                "box_plus_minus": round(8.0 - p * 1.5, 1),
                "usage_rate": round(22.0 - p * 2.0, 1),
                "injury_status": "healthy",
                "is_transfer": False,
                "eligibility_year": (p % 4) + 1,
            })
        teams.append({
            "team_id": _team_id(i),
            "team_name": _team_name(i),
            "players": players,
        })
    fp = path / "rosters.json"
    with open(fp, "w") as f:
        json.dump({"teams": teams}, f)
    return str(fp)


def _make_kaggle_dir(tmp_path: Path) -> Path:
    """Create a Kaggle directory with Massey Ordinals for N_TEAMS teams."""
    kaggle_dir = tmp_path / "kaggle"
    kaggle_dir.mkdir()

    # MTeams.csv — team IDs must match what pipeline normalizes to
    team_rows = [[str(1100 + i), _team_name(i)] for i in range(N_TEAMS)]
    _write_csv(kaggle_dir / "MTeams.csv", ["TeamID", "TeamName"], team_rows)

    # MMasseyOrdinals.csv with multiple rating systems
    ordinal_rows = []
    systems = ["POM", "SAG", "MOR", "DOL", "COL", "WOL", "RTH"]
    for system in systems:
        for i in range(N_TEAMS):
            ordinal_rows.append([
                str(YEAR), "128", system, str(1100 + i), str(i + 1),
            ])
        # Earlier day (should be ignored in favor of day 128)
        for i in range(N_TEAMS):
            ordinal_rows.append([
                str(YEAR), "100", system, str(1100 + i), str(N_TEAMS - i),
            ])
    _write_csv(
        kaggle_dir / "MMasseyOrdinals.csv",
        ["Season", "RankingDayNum", "SystemName", "TeamID", "OrdinalRank"],
        ordinal_rows,
    )

    # MNCAATourneySeeds.csv
    seed_rows = []
    region_codes = ["W", "X", "Y", "Z"]
    for i in range(N_TEAMS):
        region = region_codes[i % 4]
        seed_num = (i // 4) + 1
        seed_rows.append([str(YEAR), f"{region}{seed_num:02d}", str(1100 + i)])
    _write_csv(
        kaggle_dir / "MNCAATourneySeeds.csv",
        ["Season", "Seed", "TeamID"],
        seed_rows,
    )

    return kaggle_dir


def _populate_massey_cache(kaggle_dir: Path, cache_dir: Path) -> int:
    """Pre-populate external ratings cache from the synthetic Kaggle data."""
    from src.data.scrapers.external_ratings import ExternalRatingsLoader

    cache_dir.mkdir(exist_ok=True)
    loader = ExternalRatingsLoader(cache_dir=str(cache_dir))
    return loader.populate_from_massey_ordinals(str(kaggle_dir), YEAR)


# ---------------------------------------------------------------------------
# End-to-end test
# ---------------------------------------------------------------------------


def test_sota_pipeline_run_e2e_with_massey(tmp_path):
    """Full SOTAPipeline.run() end-to-end test with Massey Ordinals.

    This test verifies that Massey Ordinals data flows through the entire
    pipeline — from Kaggle CSV → external ratings cache → feature vectors
    → predictions → report artifacts.

    Uses fully synthetic data so it runs in CI without real data files.
    """
    from src.pipeline.sota import SOTAPipeline, SOTAPipelineConfig

    # --- Create synthetic data ---
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    teams_json = _make_teams_json(data_dir)
    torvik_json = _make_torvik_json(data_dir)
    historical_games_json = _make_historical_games_json(data_dir)
    public_picks_json = _make_public_picks_json(data_dir)
    roster_json = _make_roster_json(data_dir)

    # --- Create Kaggle dir and populate Massey cache ---
    kaggle_dir = _make_kaggle_dir(tmp_path)
    n_cached = _populate_massey_cache(kaggle_dir, cache_dir)
    assert n_cached >= 2, f"Massey cache population failed: only {n_cached} systems"

    # --- Configure pipeline for fast execution ---
    config = SOTAPipelineConfig(
        year=YEAR,
        num_simulations=50,
        pool_size=20,
        calibration_method="temperature",
        enforce_feed_freshness=False,
        # Disable expensive ML operations
        enable_hyperparameter_tuning=False,
        enable_loyo_cv=False,
        enable_stacking=False,
        enable_feature_selection=False,
        enable_gnn=False,
        enable_transformer=False,
        enable_embedding_projections=False,
        enable_ablation_study=False,
        enable_bracket_portfolio=False,
        enable_womens_pipeline=False,
        enable_dual_submission=False,
        enable_multi_year_training=False,
        enable_round_weighted_training=False,
        injury_noise_samples=50,
        min_calibration_samples_hard=10,
        # Data paths
        teams_json=teams_json,
        torvik_json=torvik_json,
        historical_games_json=historical_games_json,
        public_picks_json=public_picks_json,
        roster_json=roster_json,
        # Massey / external ratings
        kaggle_dir=str(kaggle_dir),
        external_ratings_dir=str(cache_dir),
        data_cache_dir=str(cache_dir),
        enable_external_ratings=True,
        # Use multi_year_games_dir=None to prevent scanning for historical dirs
        multi_year_games_dir=None,
    )

    pipeline = SOTAPipeline(config)
    report = pipeline.run()

    # --- Verify pipeline completed with standard artifacts ---
    assert "rubric_evaluation" in report
    assert "artifacts" in report
    assert "ml_diagnostics" in report

    # --- Verify Massey coverage is reported ---
    massey_coverage = report["ml_diagnostics"].get("massey_coverage", {})
    assert massey_coverage, (
        "massey_coverage missing from ml_diagnostics — "
        "_verify_massey_coverage was not called during run()"
    )

    # --- Verify Massey data actually flowed through (not seed fallback) ---
    coverage_pct = massey_coverage.get("coverage_pct", 0)
    assert coverage_pct > 0.5, (
        f"Massey coverage is only {coverage_pct:.0%} — expected > 50%. "
        f"Massey data did not flow through the pipeline. "
        f"Stats: {massey_coverage}"
    )

    # --- Verify feature vectors were populated with external ratings ---
    vec_coverage = massey_coverage.get("feature_vector_coverage_pct", 0)
    assert vec_coverage > 0.5, (
        f"Feature vector coverage is only {vec_coverage:.0%} — "
        f"external_rating_composite not populated in feature vectors. "
        f"Stats: {massey_coverage}"
    )

    # --- Verify Massey composites are present (not empty seed-fallback) ---
    # When only massey_composite is loaded via SYSTEM_WEIGHTS, n_systems=1
    # per team (which is still Massey-based, not seed fallback).  Verify
    # by checking composite mean and std show meaningful signal.
    n_with_composite = massey_coverage.get("n_with_composite", 0)
    assert n_with_composite > N_TEAMS * 0.5, (
        f"Only {n_with_composite}/{N_TEAMS} teams have composites. "
        f"Massey Ordinals were not loaded."
    )
    # Composites should have meaningful variance (not all 0.5 from seed fallback)
    composite_std = massey_coverage.get("composite_std", 0)
    assert composite_std > 0.01, (
        f"Composite std is {composite_std:.4f} — values are near-constant. "
        f"This suggests seed-fallback rather than real Massey data."
    )

    # --- Verify core pipeline artifacts exist ---
    artifacts = report["artifacts"]
    assert "simulation" in artifacts
    assert artifacts["simulation"]["num_simulations"] == 50
    assert "baseline_training" in artifacts
    assert "ev_max_bracket" in artifacts
    ev = artifacts["ev_max_bracket"]
    assert "champion" in ev
    assert len(ev.get("picks", [])) >= 63
