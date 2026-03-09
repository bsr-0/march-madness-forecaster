"""Tests for src.reproducibility.run_hasher."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import pytest

from src.reproducibility.run_hasher import RunHasher


# ---------------------------------------------------------------------------
# Lightweight stand-in for SOTAPipelineConfig so tests don't pull in the
# full pipeline dependency graph.
# ---------------------------------------------------------------------------

@dataclass
class _StubConfig:
    """Minimal stub that mirrors SOTAPipelineConfig's data-file fields."""

    year: int = 2026
    num_simulations: int = 50000
    random_seed: int = 2026
    teams_json: Optional[str] = None
    torvik_json: Optional[str] = None
    historical_games_json: Optional[str] = None
    sports_reference_json: Optional[str] = None
    public_picks_json: Optional[str] = None
    scoring_rules_json: Optional[str] = None
    roster_json: Optional[str] = None
    transfer_portal_json: Optional[str] = None
    preseason_ap_json: Optional[str] = None
    coach_tournament_json: Optional[str] = None
    conf_champions_json: Optional[str] = None
    injury_report_json: Optional[str] = None
    venue_locations_json: Optional[str] = None
    team_locations_json: Optional[str] = None
    bracket_json: Optional[str] = None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestHashConfigDeterministic:
    """hash_config must return the same digest for the same config."""

    def test_same_config_same_hash(self) -> None:
        cfg = _StubConfig()
        h1 = RunHasher(cfg).hash_config()
        h2 = RunHasher(cfg).hash_config()
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex digest


class TestHashConfigChangesOnModification:
    """Changing any field must change the config hash."""

    def test_change_year(self) -> None:
        cfg_a = _StubConfig(year=2025)
        cfg_b = _StubConfig(year=2026)
        assert RunHasher(cfg_a).hash_config() != RunHasher(cfg_b).hash_config()

    def test_change_seed(self) -> None:
        cfg_a = _StubConfig(random_seed=1)
        cfg_b = _StubConfig(random_seed=2)
        assert RunHasher(cfg_a).hash_config() != RunHasher(cfg_b).hash_config()


class TestHashAllInputs:
    """hash_all_inputs should compute SHA-256 for referenced data files."""

    def test_hashes_temp_files(self, tmp_path: Path) -> None:
        teams = tmp_path / "teams.json"
        torvik = tmp_path / "torvik.json"
        teams.write_text('{"a": 1}')
        torvik.write_text('{"b": 2}')

        cfg = _StubConfig(teams_json=str(teams), torvik_json=str(torvik))
        hashes = RunHasher(cfg).hash_all_inputs()

        assert str(teams) in hashes
        assert str(torvik) in hashes
        # Verify the hashes are correct.
        expected_teams = hashlib.sha256(b'{"a": 1}').hexdigest()
        assert hashes[str(teams)] == expected_teams

    def test_skips_none_paths(self) -> None:
        cfg = _StubConfig()  # all paths None
        hashes = RunHasher(cfg).hash_all_inputs()
        assert hashes == {}

    def test_skips_missing_files(self, tmp_path: Path) -> None:
        cfg = _StubConfig(teams_json=str(tmp_path / "nonexistent.json"))
        hashes = RunHasher(cfg).hash_all_inputs()
        assert hashes == {}


class TestComputeReproducibilityHash:
    """compute_reproducibility_hash combines code + data + config."""

    def test_includes_code_version(self, tmp_path: Path) -> None:
        teams = tmp_path / "teams.json"
        teams.write_text("{}")
        cfg = _StubConfig(teams_json=str(teams))

        h1 = RunHasher(cfg).compute_reproducibility_hash("v1.0")
        h2 = RunHasher(cfg).compute_reproducibility_hash("v2.0")
        assert h1 != h2

    def test_deterministic(self, tmp_path: Path) -> None:
        teams = tmp_path / "teams.json"
        teams.write_text("{}")
        cfg = _StubConfig(teams_json=str(teams))

        h1 = RunHasher(cfg).compute_reproducibility_hash("v1.0")
        h2 = RunHasher(cfg).compute_reproducibility_hash("v1.0")
        assert h1 == h2

    def test_changes_on_data_change(self, tmp_path: Path) -> None:
        teams = tmp_path / "teams.json"
        teams.write_text('{"version": 1}')
        cfg = _StubConfig(teams_json=str(teams))

        h1 = RunHasher(cfg).compute_reproducibility_hash("v1.0")

        teams.write_text('{"version": 2}')
        h2 = RunHasher(cfg).compute_reproducibility_hash("v1.0")
        assert h1 != h2


class TestVerifyAgainst:
    """verify_against compares current hashes to a stored record."""

    def test_matching_hashes(self, tmp_path: Path) -> None:
        teams = tmp_path / "teams.json"
        teams.write_text('{"x": 1}')
        cfg = _StubConfig(teams_json=str(teams))

        current_hashes = RunHasher(cfg).hash_all_inputs()

        @dataclass
        class _StubRecord:
            dataset_hashes: Dict[str, str] = field(default_factory=dict)

        record = _StubRecord(dataset_hashes=current_hashes)
        assert RunHasher(cfg).verify_against(record) is True

    def test_mismatched_hashes(self, tmp_path: Path) -> None:
        teams = tmp_path / "teams.json"
        teams.write_text('{"x": 1}')
        cfg = _StubConfig(teams_json=str(teams))

        @dataclass
        class _StubRecord:
            dataset_hashes: Dict[str, str] = field(default_factory=dict)

        record = _StubRecord(dataset_hashes={str(teams): "0000dead"})
        assert RunHasher(cfg).verify_against(record) is False

    def test_empty_record(self) -> None:
        cfg = _StubConfig()

        @dataclass
        class _StubRecord:
            dataset_hashes: Dict[str, str] = field(default_factory=dict)

        record = _StubRecord()
        assert RunHasher(cfg).verify_against(record) is False
