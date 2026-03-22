"""Tests for First Four play-in team resolution.

Verifies that the bracket data loading and simulation stages correctly
collapse 68-team brackets (with First Four play-in pairs) down to 64
teams (16 per region) before Monte Carlo simulation.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from typing import Dict, List

import pytest


# ---------------------------------------------------------------------------
# DataLoader.load_teams_from_json resolution
# ---------------------------------------------------------------------------

class TestDataLoaderFirstFourResolution:
    """Verify DataLoader collapses First Four pairs at load time."""

    def test_load_teams_resolves_first_four_to_64(self):
        """Loading teams_2026.json with 68 teams should return 64."""
        from src.data.loader import DataLoader

        teams_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "raw", "teams_2026.json",
        )
        if not os.path.exists(teams_path):
            pytest.skip("teams_2026.json not found")

        teams = DataLoader.load_teams_from_json(teams_path)
        assert len(teams) == 64, f"Expected 64 teams, got {len(teams)}"

    def test_load_teams_16_per_region(self):
        """Each region should have exactly 16 teams after resolution."""
        from src.data.loader import DataLoader

        teams_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "raw", "teams_2026.json",
        )
        if not os.path.exists(teams_path):
            pytest.skip("teams_2026.json not found")

        teams = DataLoader.load_teams_from_json(teams_path)
        by_region: Dict[str, List] = {}
        for t in teams:
            by_region.setdefault(t.region, []).append(t)

        for region in ["East", "West", "South", "Midwest"]:
            assert region in by_region, f"Missing region {region}"
            assert len(by_region[region]) == 16, (
                f"Region {region} has {len(by_region[region])} teams, expected 16"
            )

    def test_load_teams_seeds_1_through_16(self):
        """Each region should have exactly seeds 1-16 after resolution."""
        from src.data.loader import DataLoader

        teams_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "raw", "teams_2026.json",
        )
        if not os.path.exists(teams_path):
            pytest.skip("teams_2026.json not found")

        teams = DataLoader.load_teams_from_json(teams_path)
        by_region: Dict[str, List] = {}
        for t in teams:
            by_region.setdefault(t.region, []).append(t)

        for region, region_teams in by_region.items():
            seeds = {t.seed for t in region_teams}
            assert seeds == set(range(1, 17)), (
                f"Region {region} seeds {seeds} != {{1..16}}"
            )

    def test_load_teams_keeps_higher_rated_play_in(self):
        """First Four resolution should keep the higher-rated team."""
        from src.data.loader import DataLoader

        bracket = {
            "teams": [
                {"name": "Team A", "seed": 1, "region": "East", "rating": 1800},
                {"name": "Weak", "seed": 11, "region": "East", "rating": 1500,
                 "first_four": True},
                {"name": "Strong", "seed": 11, "region": "East", "rating": 1700,
                 "first_four": True},
            ] + [
                {"name": f"Filler{s}", "seed": s, "region": "East"}
                for s in range(2, 17) if s != 11
            ],
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(bracket, f)
            tmp_path = f.name

        try:
            teams = DataLoader.load_teams_from_json(tmp_path)
            seed_11 = [t for t in teams if t.seed == 11 and t.region == "East"]
            assert len(seed_11) == 1, f"Expected 1 seed-11 team, got {len(seed_11)}"
            assert seed_11[0].name == "Strong"
        finally:
            os.unlink(tmp_path)

    def test_load_teams_no_first_four_unchanged(self):
        """A 64-team file without first_four flags should load as-is."""
        from src.data.loader import DataLoader

        bracket = {
            "teams": [
                {"name": f"Team_{r}_{s}", "seed": s, "region": r}
                for r in ["East", "West", "South", "Midwest"]
                for s in range(1, 17)
            ],
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(bracket, f)
            tmp_path = f.name

        try:
            teams = DataLoader.load_teams_from_json(tmp_path)
            assert len(teams) == 64
        finally:
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# _resolve_play_in_teams (simulation stage safety net)
# ---------------------------------------------------------------------------

class TestResolvePlayInTeams:
    """Verify simulation-stage First Four resolution as safety net."""

    def test_resolve_collapses_duplicate_seeds(self):
        """Duplicate seeds in same region should be reduced to one."""
        from src.pipeline.stages.simulation import _resolve_play_in_teams
        from src.simulation.monte_carlo import TournamentTeam

        teams_by_region = {
            "West": [
                TournamentTeam(team_id=f"team_{s}", seed=s, region="West", strength=float(17 - s))
                for s in range(1, 17)
            ] + [
                TournamentTeam(team_id="play_in_extra", seed=11, region="West", strength=5.0),
            ],
            "East": [
                TournamentTeam(team_id=f"east_{s}", seed=s, region="East", strength=float(17 - s))
                for s in range(1, 17)
            ],
        }

        assert len(teams_by_region["West"]) == 17

        resolved = _resolve_play_in_teams(teams_by_region)

        assert len(resolved["West"]) == 16
        assert len(resolved["East"]) == 16

    def test_resolve_keeps_stronger_team(self):
        """Resolution should keep the team with higher strength."""
        from src.pipeline.stages.simulation import _resolve_play_in_teams
        from src.simulation.monte_carlo import TournamentTeam

        weak = TournamentTeam(team_id="weak", seed=11, region="West", strength=5.0)
        strong = TournamentTeam(team_id="strong", seed=11, region="West", strength=15.0)

        teams_by_region = {
            "West": [
                TournamentTeam(team_id=f"team_{s}", seed=s, region="West", strength=float(17 - s))
                for s in range(1, 17) if s != 11
            ] + [weak, strong],
        }

        resolved = _resolve_play_in_teams(teams_by_region)
        seed_11 = [t for t in resolved["West"] if t.seed == 11]
        assert len(seed_11) == 1
        assert seed_11[0].team_id == "strong"

    def test_resolve_64_teams_unchanged(self):
        """A 64-team bracket with no duplicates should pass through unchanged."""
        from src.pipeline.stages.simulation import _resolve_play_in_teams
        from src.simulation.monte_carlo import TournamentTeam

        teams_by_region = {
            r: [
                TournamentTeam(team_id=f"{r}_{s}", seed=s, region=r, strength=float(17 - s))
                for s in range(1, 17)
            ]
            for r in ["East", "West", "South", "Midwest"]
        }

        resolved = _resolve_play_in_teams(teams_by_region)

        for r in teams_by_region:
            assert len(resolved[r]) == 16

    def test_resolve_full_68_team_bracket(self):
        """Simulate full 68-team bracket with 4 First Four pairs."""
        from src.pipeline.stages.simulation import _resolve_play_in_teams
        from src.simulation.monte_carlo import TournamentTeam

        teams_by_region = {}
        # 4 regions, each with seeds 1-16
        for r in ["East", "West", "South", "Midwest"]:
            region_teams = [
                TournamentTeam(team_id=f"{r}_{s}", seed=s, region=r, strength=float(17 - s))
                for s in range(1, 17)
            ]
            teams_by_region[r] = region_teams

        # Add First Four extra teams (one per region to make 72, then we'll trim)
        # In practice: West seed 11, South seed 16, Midwest seed 11, Midwest seed 16
        teams_by_region["West"].append(
            TournamentTeam(team_id="west_ff", seed=11, region="West", strength=3.0)
        )
        teams_by_region["South"].append(
            TournamentTeam(team_id="south_ff", seed=16, region="South", strength=0.1)
        )
        teams_by_region["Midwest"].append(
            TournamentTeam(team_id="mw_ff_11", seed=11, region="Midwest", strength=4.0)
        )
        teams_by_region["Midwest"].append(
            TournamentTeam(team_id="mw_ff_16", seed=16, region="Midwest", strength=0.2)
        )

        total_before = sum(len(v) for v in teams_by_region.values())
        assert total_before == 68

        resolved = _resolve_play_in_teams(teams_by_region)
        total_after = sum(len(v) for v in resolved.values())
        assert total_after == 64

        for r in resolved:
            assert len(resolved[r]) == 16
            seeds = {t.seed for t in resolved[r]}
            assert seeds == set(range(1, 17))
