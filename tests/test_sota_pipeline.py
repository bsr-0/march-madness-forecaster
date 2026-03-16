"""Integration tests for the SOTA pipeline with real historical data."""

import csv
import inspect
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import pytest

from src.models.team import Team
from src.pipeline.sota import DataRequirementError, SOTAPipeline, SOTAPipelineConfig
import src.main as main_mod

_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
_GENERATED = os.path.join(_DATA_DIR, "generated_strict_2025")

REAL_DATA_PATHS = {
    "teams": os.path.join(_GENERATED, "teams_2025.json"),
    "torvik": os.path.join(_GENERATED, "torvik_2025.json"),
    "rosters": os.path.join(_GENERATED, "rosters_2025.json"),
    "public_picks": os.path.join(_GENERATED, "public_picks_2025.json"),
    "historical_games": os.path.join(_DATA_DIR, "historical", "historical_games_2025.json"),
}

_REAL_DATA_AVAILABLE = all(os.path.exists(p) for p in REAL_DATA_PATHS.values())


def _has_real_torvik_data() -> bool:
    """Return False if the torvik file contains only placeholder (all-identical) stats."""
    torvik_path = REAL_DATA_PATHS.get("torvik", "")
    if not os.path.exists(torvik_path):
        return False
    try:
        with open(torvik_path) as f:
            data = json.load(f)
        teams = data.get("teams", [])
        if not teams:
            return False
        # Check if all teams have identical barthag — a signature of placeholder data.
        barthag_values = {round(t.get("barthag", 0.5), 6) for t in teams}
        return len(barthag_values) > 1
    except Exception:
        return False


_HAS_REAL_TORVIK = _has_real_torvik_data()


@pytest.mark.skipif(
    not _REAL_DATA_AVAILABLE or not _HAS_REAL_TORVIK,
    reason="Real data files not present or torvik data contains placeholder stats",
)
def test_sota_pipeline_produces_rubric_artifacts():
    config = SOTAPipelineConfig(
        year=2025,
        num_simulations=120,
        pool_size=64,
        calibration_method="isotonic",
        enforce_feed_freshness=False,
        # Disable expensive ML operations for test speed.
        enable_hyperparameter_tuning=False,
        enable_loyo_cv=False,
        enable_stacking=False,
        enable_feature_selection=False,
        injury_noise_samples=100,
        teams_json=REAL_DATA_PATHS["teams"],
        torvik_json=REAL_DATA_PATHS["torvik"],
        roster_json=REAL_DATA_PATHS["rosters"],
        public_picks_json=REAL_DATA_PATHS["public_picks"],
        historical_games_json=REAL_DATA_PATHS["historical_games"],
    )
    pipeline = SOTAPipeline(config)

    report = pipeline.run()

    assert "rubric_evaluation" in report
    assert "artifacts" in report

    adjacency = report["artifacts"]["adjacency_matrix"]
    assert len(adjacency) >= 64  # All D1 teams with game data
    assert len(adjacency[0]) == len(adjacency)  # Square matrix

    sim = report["artifacts"]["simulation"]
    assert sim["num_simulations"] == 120

    ev_bracket = report["artifacts"]["ev_max_bracket"]
    assert "champion" in ev_bracket
    assert len(ev_bracket["final_four"]) <= 4
    assert len(ev_bracket["picks"]) >= 63

    baseline = report["artifacts"]["baseline_training"]
    assert baseline["model"] in {"lightgbm", "logistic_regression", "none", "stacking_ensemble"}
    assert "model_uncertainty" in report["artifacts"]
    assert sorted(report["artifacts"]["public_pick_sources"]) == ["cbs", "espn", "yahoo"]
    for pick in report["artifacts"]["top_leverage_picks"][:10]:
        assert 0.0 <= pick["public_pick_percentage"] <= 1.0


@pytest.mark.skipif(
    not _REAL_DATA_AVAILABLE or not _HAS_REAL_TORVIK,
    reason="Real data files not present or torvik data contains placeholder stats",
)
def test_sota_pipeline_output_file(tmp_path):
    output_path = tmp_path / "sota_report.json"

    config = SOTAPipelineConfig(
        year=2025,
        num_simulations=80,
        pool_size=20,
        enforce_feed_freshness=False,
        enable_hyperparameter_tuning=False,
        enable_loyo_cv=False,
        enable_stacking=False,
        enable_feature_selection=False,
        injury_noise_samples=100,
        teams_json=REAL_DATA_PATHS["teams"],
        torvik_json=REAL_DATA_PATHS["torvik"],
        roster_json=REAL_DATA_PATHS["rosters"],
        public_picks_json=REAL_DATA_PATHS["public_picks"],
        historical_games_json=REAL_DATA_PATHS["historical_games"],
    )
    pipeline = SOTAPipeline(config)
    report = pipeline.run()

    with open(output_path, "w") as f:
        json.dump(report, f)

    with open(output_path, "r") as f:
        restored = json.load(f)

    assert restored["artifacts"]["simulation"]["num_simulations"] == 80


def test_public_pick_loader_supports_explicit_multi_source_payload(tmp_path):
    payload = {
        "timestamp": "2026-03-17T12:00:00Z",
        "espn": {
            "teams": {
                "duke": {
                    "team_name": "Duke",
                    "seed": 1,
                    "region": "East",
                    "round_of_64_pct": 98.0,
                    "round_of_32_pct": 90.0,
                    "sweet_16_pct": 70.0,
                    "elite_8_pct": 50.0,
                    "final_four_pct": 30.0,
                    "champion_pct": 20.0,
                }
            }
        },
        "yahoo": {
            "teams": {
                "duke": {
                    "team_name": "Duke",
                    "seed": 1,
                    "region": "East",
                    "round_of_64_pct": 97.0,
                    "round_of_32_pct": 88.0,
                    "sweet_16_pct": 68.0,
                    "elite_8_pct": 48.0,
                    "final_four_pct": 28.0,
                    "champion_pct": 10.0,
                }
            }
        },
        "cbs": {
            "teams": {
                "duke": {
                    "team_name": "Duke",
                    "seed": 1,
                    "region": "East",
                    "round_of_64_pct": 96.0,
                    "round_of_32_pct": 87.0,
                    "sweet_16_pct": 66.0,
                    "elite_8_pct": 46.0,
                    "final_four_pct": 26.0,
                    "champion_pct": 5.0,
                }
            }
        },
    }
    picks_path = tmp_path / "picks.json"
    with open(picks_path, "w") as f:
        json.dump(payload, f)

    pipeline = SOTAPipeline(SOTAPipelineConfig(public_picks_json=str(picks_path)))
    pipeline.team_struct["duke"] = Team(name="Duke", seed=1, region="East")

    public = pipeline._load_public_picks({"duke": {"CHAMP": 0.1}})
    assert sorted(pipeline.public_pick_sources) == ["cbs", "espn", "yahoo"]
    assert abs(public["duke"]["CHAMP"] - 0.14) < 1e-9


def test_sota_pipeline_requires_real_data():
    config = SOTAPipelineConfig(num_simulations=20, pool_size=10)
    pipeline = SOTAPipeline(config)

    try:
        pipeline.run()
        assert False, "Expected DataRequirementError"
    except DataRequirementError:
        assert True


def test_sota_pipeline_rejects_stale_public_feed(tmp_path):
    payload = {
        "timestamp": "2020-01-01T00:00:00Z",
        "teams": {
            "duke": {
                "team_name": "Duke",
                "seed": 1,
                "region": "East",
                "round_of_64_pct": 98.0,
                "round_of_32_pct": 90.0,
                "sweet_16_pct": 70.0,
                "elite_8_pct": 50.0,
                "final_four_pct": 30.0,
                "champion_pct": 20.0,
            }
        },
        "sources": ["espn", "yahoo"],
    }
    picks_path = tmp_path / "stale_picks.json"
    with open(picks_path, "w") as f:
        json.dump(payload, f)

    pipeline = SOTAPipeline(
        SOTAPipelineConfig(
            public_picks_json=str(picks_path),
            max_feed_age_hours=1,
            min_public_sources=2,
        )
    )
    pipeline.team_struct["duke"] = Team(name="Duke", seed=1, region="East")

    try:
        pipeline._load_public_picks({"duke": {"CHAMP": 0.1}})
        assert False, "Expected stale feed rejection"
    except DataRequirementError:
        assert True


def test_rapm_enrichment_from_stints_backfills_missing_player_rapm():
    pipeline = SOTAPipeline(
        SOTAPipelineConfig(
            enforce_feed_freshness=False,
            min_rapm_players_per_team=3,
        )
    )
    players = [
        pipeline._player_from_dict(
            "duke",
            {
                "player_id": f"duke_p{i}",
                "name": f"P{i}",
                "position": "PG",
                "minutes_per_game": 30 - i,
                "games_played": 30,
                "usage_rate": 20,
            },
        )
        for i in range(5)
    ]
    team_block = {
        "stints": [
            {"players": ["duke_p0", "duke_p1", "duke_p2"], "plus_minus": 4, "possessions": 10},
            {"players": ["duke_p1", "duke_p2", "duke_p3"], "plus_minus": -2, "possessions": 8},
            {"players": ["duke_p0", "duke_p3", "duke_p4"], "plus_minus": 3, "possessions": 9},
            {"players": ["duke_p2", "duke_p3", "duke_p4"], "plus_minus": -1, "possessions": 7},
        ]
    }

    pipeline._enrich_roster_rapm(players, team_block)
    non_zero = sum(1 for p in players if abs(p.rapm_total) > 1e-8)
    assert non_zero >= 3


# ---------------------------------------------------------------------------
# Massey Ordinals end-to-end test (merged from test_sota_pipeline_massey_e2e)
# ---------------------------------------------------------------------------

_MASSEY_N_TEAMS = 64
_MASSEY_REGIONS = ["East", "West", "South", "Midwest"]
_MASSEY_YEAR = 2025


def _massey_team_name(i: int) -> str:
    return f"Team{i}"


def _massey_team_id(i: int) -> str:
    return f"team{i}"


def _massey_write_csv(path: Path, headers: list, rows: list):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


def _massey_make_teams_json(path: Path) -> str:
    teams = []
    for i in range(_MASSEY_N_TEAMS):
        region = _MASSEY_REGIONS[i % 4]
        seed = (i // 4) + 1
        teams.append({
            "name": _massey_team_name(i),
            "seed": min(seed, 16),
            "region": region,
        })
    fp = path / "teams.json"
    with open(fp, "w") as f:
        json.dump({"teams": teams}, f)
    return str(fp)


def _massey_make_torvik_json(path: Path) -> str:
    teams = []
    for i in range(_MASSEY_N_TEAMS):
        rank = i + 1
        barthag = round(0.95 - (i * 0.008), 4)
        adj_o = round(120.0 - i * 0.4, 1)
        adj_d = round(90.0 + i * 0.3, 1)
        teams.append({
            "team_id": _massey_team_id(i),
            "name": _massey_team_name(i),
            "team_name": _massey_team_name(i),
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


def _massey_make_historical_games_json(path: Path) -> str:
    games = []
    game_id = 1000
    seen = set()
    for i in range(_MASSEY_N_TEAMS):
        for offset in [1, 2, 3]:
            j = (i + offset) % _MASSEY_N_TEAMS
            pair = (min(i, j), max(i, j))
            if pair in seen:
                continue
            seen.add(pair)
            score1 = 70 + (i % 15) + (game_id % 10)
            score2 = 68 + (j % 15) + ((game_id + 3) % 10)
            month = 11 + ((game_id - 1000) % 4)
            day = 1 + ((game_id - 1000) % 28)
            if month > 12:
                month_str = f"2025-{month - 12:02d}"
            else:
                month_str = f"2024-{month:02d}"
            games.append({
                "game_id": f"g{game_id}",
                "date": f"{month_str}-{day:02d}",
                "season": _MASSEY_YEAR,
                "team1_id": _massey_team_id(i),
                "team1_name": _massey_team_name(i),
                "team2_id": _massey_team_id(j),
                "team2_name": _massey_team_name(j),
                "team1_score": score1,
                "team2_score": score2,
                "lead_history": [0, score1 - score2],
            })
            game_id += 1
    fp = path / "historical_games.json"
    with open(fp, "w") as f:
        json.dump({"season": _MASSEY_YEAR, "games": games}, f)
    return str(fp)


def _massey_make_public_picks_json(path: Path) -> str:
    sources: Dict[str, Dict] = {}
    for source_name in ["espn", "yahoo", "cbs"]:
        teams_data: Dict[str, Dict] = {}
        for i in range(_MASSEY_N_TEAMS):
            tid = _massey_team_id(i)
            seed = min((i // 4) + 1, 16)
            base = max(99 - seed * 5, 5)
            teams_data[tid] = {
                "team_name": _massey_team_name(i),
                "seed": seed,
                "region": _MASSEY_REGIONS[i % 4],
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


def _massey_make_roster_json(path: Path) -> str:
    teams = []
    positions = ["PG", "SG", "SF", "PF", "C", "SG", "SF", "PF"]
    for i in range(_MASSEY_N_TEAMS):
        players = []
        for p in range(8):
            players.append({
                "player_id": f"{_massey_team_id(i)}_p{p}",
                "name": f"Player{p} of {_massey_team_name(i)}",
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
            "team_id": _massey_team_id(i),
            "team_name": _massey_team_name(i),
            "players": players,
        })
    fp = path / "rosters.json"
    with open(fp, "w") as f:
        json.dump({"teams": teams}, f)
    return str(fp)


def _massey_make_kaggle_dir(tmp_path: Path) -> Path:
    kaggle_dir = tmp_path / "kaggle"
    kaggle_dir.mkdir()

    team_rows = [[str(1100 + i), _massey_team_name(i)] for i in range(_MASSEY_N_TEAMS)]
    _massey_write_csv(kaggle_dir / "MTeams.csv", ["TeamID", "TeamName"], team_rows)

    ordinal_rows = []
    systems = ["POM", "SAG", "MOR", "DOL", "COL", "WOL", "RTH"]
    for system in systems:
        for i in range(_MASSEY_N_TEAMS):
            ordinal_rows.append([
                str(_MASSEY_YEAR), "128", system, str(1100 + i), str(i + 1),
            ])
        for i in range(_MASSEY_N_TEAMS):
            ordinal_rows.append([
                str(_MASSEY_YEAR), "100", system, str(1100 + i), str(_MASSEY_N_TEAMS - i),
            ])
    _massey_write_csv(
        kaggle_dir / "MMasseyOrdinals.csv",
        ["Season", "RankingDayNum", "SystemName", "TeamID", "OrdinalRank"],
        ordinal_rows,
    )

    seed_rows = []
    region_codes = ["W", "X", "Y", "Z"]
    for i in range(_MASSEY_N_TEAMS):
        region = region_codes[i % 4]
        seed_num = (i // 4) + 1
        seed_rows.append([str(_MASSEY_YEAR), f"{region}{seed_num:02d}", str(1100 + i)])
    _massey_write_csv(
        kaggle_dir / "MNCAATourneySeeds.csv",
        ["Season", "Seed", "TeamID"],
        seed_rows,
    )

    return kaggle_dir


def _massey_populate_cache(kaggle_dir: Path, cache_dir: Path) -> int:
    from src.data.scrapers.external_ratings import ExternalRatingsLoader

    cache_dir.mkdir(exist_ok=True)
    loader = ExternalRatingsLoader(cache_dir=str(cache_dir))
    return loader.populate_from_massey_ordinals(str(kaggle_dir), _MASSEY_YEAR)


def test_sota_pipeline_run_e2e_with_massey(tmp_path):
    """Full SOTAPipeline.run() end-to-end test with Massey Ordinals."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    teams_json = _massey_make_teams_json(data_dir)
    torvik_json = _massey_make_torvik_json(data_dir)
    historical_games_json = _massey_make_historical_games_json(data_dir)
    public_picks_json = _massey_make_public_picks_json(data_dir)
    roster_json = _massey_make_roster_json(data_dir)

    kaggle_dir = _massey_make_kaggle_dir(tmp_path)
    n_cached = _massey_populate_cache(kaggle_dir, cache_dir)
    assert n_cached >= 2, f"Massey cache population failed: only {n_cached} systems"

    config = SOTAPipelineConfig(
        year=_MASSEY_YEAR,
        num_simulations=50,
        pool_size=20,
        calibration_method="temperature",
        enforce_feed_freshness=False,
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
        teams_json=teams_json,
        torvik_json=torvik_json,
        historical_games_json=historical_games_json,
        public_picks_json=public_picks_json,
        roster_json=roster_json,
        kaggle_dir=str(kaggle_dir),
        external_ratings_dir=str(cache_dir),
        data_cache_dir=str(cache_dir),
        enable_external_ratings=True,
        multi_year_games_dir=None,
    )

    pipeline = SOTAPipeline(config)
    report = pipeline.run()

    assert "rubric_evaluation" in report
    assert "artifacts" in report
    assert "ml_diagnostics" in report

    massey_coverage = report["ml_diagnostics"].get("massey_coverage", {})
    assert massey_coverage, "massey_coverage missing from ml_diagnostics"

    coverage_pct = massey_coverage.get("coverage_pct", 0)
    assert coverage_pct > 0.5, f"Massey coverage is only {coverage_pct:.0%}"

    vec_coverage = massey_coverage.get("feature_vector_coverage_pct", 0)
    assert vec_coverage > 0.5, f"Feature vector coverage is only {vec_coverage:.0%}"

    n_with_composite = massey_coverage.get("n_with_composite", 0)
    assert n_with_composite > _MASSEY_N_TEAMS * 0.5

    composite_std = massey_coverage.get("composite_std", 0)
    assert composite_std > 0.01, f"Composite std is {composite_std:.4f} — near-constant"

    artifacts = report["artifacts"]
    assert "simulation" in artifacts
    assert artifacts["simulation"]["num_simulations"] == 50
    assert "baseline_training" in artifacts
    assert "ev_max_bracket" in artifacts
    ev = artifacts["ev_max_bracket"]
    assert "champion" in ev
    assert len(ev.get("picks", [])) >= 63


# ---------------------------------------------------------------------------
# Manifest → SOTA wiring tests (merged from test_manifest_sota)
# ---------------------------------------------------------------------------


def test_run_sota_from_manifest_resolves_paths_and_runs(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    teams_file = data_dir / "teams_2026.json"
    rosters_file = data_dir / "rosters_2026.json"
    with open(teams_file, "w") as f:
        json.dump({"teams": []}, f)
    with open(rosters_file, "w") as f:
        json.dump({"teams": []}, f)

    manifest_file = tmp_path / "manifest_2026.json"
    with open(manifest_file, "w") as f:
        json.dump(
            {
                "year": 2026,
                "artifacts": {
                    "teams_json": "data/teams_2026.json",
                    "rosters_json": "data/rosters_2026.json",
                },
            },
            f,
        )

    captured = {}

    def fake_run(config, output_path):
        captured["config"] = config
        captured["output"] = output_path
        return {
            "artifacts": {
                "pool_recommendation": "balanced",
                "simulation": {"num_simulations": config.num_simulations},
            }
        }

    monkeypatch.setattr(main_mod, "run_sota_pipeline_to_file", fake_run)
    monkeypatch.setattr(main_mod, "_guard_production_2026", lambda config: None)

    args = SimpleNamespace(
        manifest=str(manifest_file),
        output=str(tmp_path / "report.json"),
        year=None,
        simulations=123,
        pool_size=12,
        injury_noise_samples=10000,
        seed=7,
        calibration="isotonic",
        input=None,
        torvik=None,
        historical_games=None,
        sports_reference=None,
        public_picks=None,
        rosters=None,
        transfer_portal=None,
        scoring_rules=None,
        preseason_ap=None,
        coach_tournament=None,
        conf_champions=None,
        betting_odds=None,
        scrape_live=False,
        cache_dir="data/raw/cache",
        allow_stale_feeds=False,
        max_feed_age_hours=168,
        min_public_sources=2,
        min_rapm_players_per_team=5,
    )

    code = main_mod.run_sota_from_manifest(args)

    assert code == 0
    assert captured["output"] == str(tmp_path / "report.json")
    assert captured["config"].teams_json == str(teams_file.resolve())
    assert captured["config"].roster_json == str(rosters_file.resolve())
    assert captured["config"].num_simulations == 123


def test_run_sota_from_manifest_allows_overrides(tmp_path, monkeypatch):
    manifest_file = tmp_path / "manifest_2026.json"
    with open(manifest_file, "w") as f:
        json.dump({"year": 2026, "artifacts": {}}, f)

    override_teams = tmp_path / "override_teams.json"
    with open(override_teams, "w") as f:
        json.dump({"teams": []}, f)

    override_rosters = tmp_path / "override_rosters.json"
    with open(override_rosters, "w") as f:
        json.dump({"teams": []}, f)

    captured = {}

    def fake_run(config, output_path):
        captured["config"] = config
        return {
            "artifacts": {
                "pool_recommendation": "balanced",
                "simulation": {"num_simulations": config.num_simulations},
            }
        }

    monkeypatch.setattr(main_mod, "run_sota_pipeline_to_file", fake_run)

    args = SimpleNamespace(
        manifest=str(manifest_file),
        output=str(tmp_path / "report.json"),
        year=2025,
        simulations=11,
        pool_size=9,
        injury_noise_samples=10000,
        seed=3,
        calibration="platt",
        input=str(override_teams),
        torvik=None,
        historical_games=None,
        sports_reference=None,
        public_picks=None,
        rosters=str(override_rosters),
        transfer_portal=None,
        scoring_rules=None,
        preseason_ap=None,
        coach_tournament=None,
        conf_champions=None,
        betting_odds=None,
        scrape_live=True,
        cache_dir="cache-dir",
        allow_stale_feeds=True,
        max_feed_age_hours=720,
        min_public_sources=1,
        min_rapm_players_per_team=3,
    )

    code = main_mod.run_sota_from_manifest(args)
    assert code == 0
    assert captured["config"].year == 2025
    assert captured["config"].teams_json == str(override_teams.resolve())
    assert captured["config"].roster_json == str(override_rosters.resolve())
    assert captured["config"].scrape_live is True


def test_fixed_feature_set_citation_markers():
    """C2: Key citation markers must appear in FIXED_FEATURE_SET docs."""
    from src.pipeline import config as pipeline_config

    src = inspect.getsource(pipeline_config)
    for marker in ["[KP]", "[OL]", "[KUB]", "[KAG]", "[VAR]"]:
        assert marker in src, (
            f"Missing citation marker {marker} in config.py FIXED_FEATURE_SET docs."
        )
