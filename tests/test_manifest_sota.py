"""Tests for manifest -> SOTA command wiring."""

import json
from types import SimpleNamespace

import src.main as main_mod


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
    """C2: Key citation markers must appear in sota.py source (RDoF methodology)."""
    import inspect
    from src.pipeline import sota

    src = inspect.getsource(sota)
    for marker in ["[KP]", "[OL]", "[KUB]", "[KAG]", "[VAR]"]:
        assert marker in src, (
            f"Missing citation marker {marker} in sota.py FIXED_FEATURE_SET docs. "
            "Each feature must have a published empirical source."
        )
