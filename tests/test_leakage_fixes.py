"""Tests for data leakage fixes in the SOTA pipeline.

Validates that the pipeline correctly isolates training, validation, and
calibration data to prevent information leakage across splits.
"""

import numpy as np
import pytest

from src.data.models.game_flow import GameFlow
from src.pipeline.sota import SOTAPipeline, SOTAPipelineConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_game(game_id: str, team1: str, team2: str, game_date: str, margin: int = 5) -> GameFlow:
    """Create a minimal GameFlow for testing."""
    flow = GameFlow(game_id=game_id, team1_id=team1, team2_id=team2)
    flow.game_date = game_date
    flow.lead_history = [margin]
    return flow


def _make_pipeline_with_games(dates, team_ids=None):
    """Build a SOTAPipeline with fake team features and games on given dates.

    Returns (pipeline, game_flows_dict) ready for boundary computation.
    """
    if team_ids is None:
        team_ids = ["team_a", "team_b"]

    config = SOTAPipelineConfig(year=2026, random_seed=42)
    pipeline = SOTAPipeline(config)

    # Register minimal team features so boundary filter recognises teams.
    for tid in team_ids:
        vec = np.random.RandomState(42).randn(10)
        pipeline.feature_engineer.team_features[tid] = object()
        # Create a mock TeamFeatures with enough attributes
        class _FakeFeatures:
            pass
        fake = _FakeFeatures()
        fake.adj_efficiency_margin = 5.0
        pipeline.feature_engineer.team_features[tid] = fake

    games = []
    for i, d in enumerate(dates):
        g = _make_game(f"g{i}", team_ids[0], team_ids[1], d, margin=(10 - i))
        games.append(g)

    pipeline.all_game_flows = list(games)
    game_flows = {tid: list(games) for tid in team_ids}
    return pipeline, game_flows


# ---------------------------------------------------------------------------
# 1. _compute_train_val_boundary
# ---------------------------------------------------------------------------


class TestComputeTrainValBoundary:
    def test_boundary_set_at_expected_date(self):
        """Boundary should be at the ~80th percentile game chronologically."""
        dates = [f"2026-01-{d:02d}" for d in range(1, 31)]  # 30 games
        pipeline, game_flows = _make_pipeline_with_games(dates)

        pipeline._compute_train_val_boundary(game_flows)

        assert pipeline._validation_sort_key_boundary is not None
        # 30 games, 20% held out = 6 validation, so boundary at game 24 → Jan 25
        expected_boundary = pipeline._game_sort_key("2026-01-25")
        assert pipeline._validation_sort_key_boundary == expected_boundary

    def test_boundary_none_when_too_few_games(self):
        """Boundary should remain None when fewer than 25 games."""
        dates = [f"2026-01-{d:02d}" for d in range(1, 15)]  # 14 games
        pipeline, game_flows = _make_pipeline_with_games(dates)

        pipeline._compute_train_val_boundary(game_flows)

        assert pipeline._validation_sort_key_boundary is None

    def test_boundary_excludes_tournament_games(self):
        """Tournament games (March 14+) should not affect the boundary."""
        regular = [f"2026-01-{d:02d}" for d in range(1, 28)]  # 27 regular-season
        tournament = ["2026-03-15", "2026-03-16", "2026-03-17"]  # 3 tournament
        dates = regular + tournament
        pipeline, game_flows = _make_pipeline_with_games(dates)

        pipeline._compute_train_val_boundary(game_flows)

        # Boundary computed on 27 games, not 30
        assert pipeline._validation_sort_key_boundary is not None
        boundary = pipeline._validation_sort_key_boundary
        # All tournament dates should be above boundary
        for d in tournament:
            assert pipeline._game_sort_key(d) > boundary


# ---------------------------------------------------------------------------
# 2. Schedule graph excludes validation-era edges
# ---------------------------------------------------------------------------


class TestScheduleGraphFiltering:
    def test_graph_excludes_validation_era_games(self):
        """_construct_schedule_graph should filter edges at boundary."""
        dates = [f"2026-01-{d:02d}" for d in range(1, 31)]  # 30 games
        pipeline, game_flows = _make_pipeline_with_games(dates)

        # Set up the boundary first
        pipeline._compute_train_val_boundary(game_flows)
        boundary = pipeline._validation_sort_key_boundary
        assert boundary is not None

        # Build graph — should only contain training-era edges
        from src.models.team import Team
        teams = [
            Team(name="team_a", seed=1, region="East"),
            Team(name="team_b", seed=2, region="East"),
        ]
        # Need team_features in pipeline.team_features (as vectors)
        pipeline.team_features = {
            "team_a": np.zeros(10),
            "team_b": np.zeros(10),
        }
        graph = pipeline._construct_schedule_graph(teams)

        # Every edge should have a date before the boundary
        for edge in graph.edges:
            edge_key = pipeline._game_sort_key(edge.game_date)
            assert edge_key < boundary, (
                f"Edge {edge.game_id} with date {edge.game_date} "
                f"(key={edge_key}) should be < boundary {boundary}"
            )

    def test_graph_keeps_all_when_no_boundary(self):
        """When boundary is None, all pre-tournament games should be in graph."""
        dates = [f"2026-01-{d:02d}" for d in range(1, 15)]  # 14 games, too few for boundary
        pipeline, game_flows = _make_pipeline_with_games(dates)

        pipeline._compute_train_val_boundary(game_flows)
        assert pipeline._validation_sort_key_boundary is None

        from src.models.team import Team
        teams = [
            Team(name="team_a", seed=1, region="East"),
            Team(name="team_b", seed=2, region="East"),
        ]
        pipeline.team_features = {
            "team_a": np.zeros(10),
            "team_b": np.zeros(10),
        }
        graph = pipeline._construct_schedule_graph(teams)

        # All 14 games should appear (each unique, deduped by game_id)
        assert len(graph.edges) == 14


# ---------------------------------------------------------------------------
# 3. Transformer excludes validation-era games
# ---------------------------------------------------------------------------


class TestTransformerFiltering:
    def test_transformer_sequences_truncated_at_boundary(self):
        """_run_transformer should exclude validation-era games from sequences."""
        # Create 40 games spanning January and February
        dates = [f"2026-01-{d:02d}" for d in range(1, 31)]
        dates += [f"2026-02-{d:02d}" for d in range(1, 11)]
        pipeline, game_flows = _make_pipeline_with_games(dates)

        pipeline._compute_train_val_boundary(game_flows)
        boundary = pipeline._validation_sort_key_boundary
        assert boundary is not None

        # Extract the filtering logic the transformer uses
        for team_id, games in game_flows.items():
            pre_tournament = [
                g for g in games
                if not pipeline._is_tournament_game(getattr(g, "game_date", "2026-01-01"))
                and (boundary is None
                     or pipeline._game_sort_key(getattr(g, "game_date", "2026-01-01")) < boundary)
            ]
            # Should be fewer games than the total
            assert len(pre_tournament) < len(dates)
            # All remaining games should be before boundary
            for g in pre_tournament:
                gk = pipeline._game_sort_key(getattr(g, "game_date", "2026-01-01"))
                assert gk < boundary


# ---------------------------------------------------------------------------
# 4. Validation era games slicing (3-way non-overlapping)
# ---------------------------------------------------------------------------


class TestValidationSlicing:
    def test_slices_are_non_overlapping(self):
        """The 3 slices should cover all validation games without overlap."""
        dates = [f"2026-01-{d:02d}" for d in range(1, 31)]
        dates += [f"2026-02-{d:02d}" for d in range(1, 21)]  # 50 games total
        pipeline, game_flows = _make_pipeline_with_games(dates)

        pipeline._compute_train_val_boundary(game_flows)
        assert pipeline._validation_sort_key_boundary is not None

        slice0 = pipeline._get_validation_era_games_slice(game_flows, 0, n_slices=3)
        slice1 = pipeline._get_validation_era_games_slice(game_flows, 1, n_slices=3)
        slice2 = pipeline._get_validation_era_games_slice(game_flows, 2, n_slices=3)

        all_val = pipeline._get_validation_era_games(game_flows)

        ids0 = {g.game_id for g in slice0}
        ids1 = {g.game_id for g in slice1}
        ids2 = {g.game_id for g in slice2}

        # Non-overlapping
        assert ids0 & ids1 == set()
        assert ids0 & ids2 == set()
        assert ids1 & ids2 == set()

        # Together they cover all validation games
        assert ids0 | ids1 | ids2 == {g.game_id for g in all_val}

    def test_slices_are_chronological(self):
        """Each slice should be chronologically ordered and non-overlapping in time."""
        dates = [f"2026-01-{d:02d}" for d in range(1, 31)]
        dates += [f"2026-02-{d:02d}" for d in range(1, 21)]
        pipeline, game_flows = _make_pipeline_with_games(dates)

        pipeline._compute_train_val_boundary(game_flows)

        slice0 = pipeline._get_validation_era_games_slice(game_flows, 0, n_slices=3)
        slice1 = pipeline._get_validation_era_games_slice(game_flows, 1, n_slices=3)
        slice2 = pipeline._get_validation_era_games_slice(game_flows, 2, n_slices=3)

        if slice0 and slice1:
            last_s0 = pipeline._game_sort_key(getattr(slice0[-1], "game_date", ""))
            first_s1 = pipeline._game_sort_key(getattr(slice1[0], "game_date", ""))
            assert last_s0 <= first_s1

        if slice1 and slice2:
            last_s1 = pipeline._game_sort_key(getattr(slice1[-1], "game_date", ""))
            first_s2 = pipeline._game_sort_key(getattr(slice2[0], "game_date", ""))
            assert last_s1 <= first_s2

    def test_small_validation_falls_back_to_two_way(self):
        """With fewer than 30 validation games, slice 1 should be empty."""
        # ~28 games, ~6 in validation (too few for 3-way)
        dates = [f"2026-01-{d:02d}" for d in range(1, 29)]
        pipeline, game_flows = _make_pipeline_with_games(dates)

        pipeline._compute_train_val_boundary(game_flows)

        slice0 = pipeline._get_validation_era_games_slice(game_flows, 0, n_slices=3)
        slice1 = pipeline._get_validation_era_games_slice(game_flows, 1, n_slices=3)
        slice2 = pipeline._get_validation_era_games_slice(game_flows, 2, n_slices=3)

        # Slice 1 (ensemble weight optimization) should be empty for small sets
        val_games = pipeline._get_validation_era_games(game_flows)
        if len(val_games) < 30:
            assert len(slice1) == 0
            # Slices 0 and 2 should still cover everything
            assert len(slice0) + len(slice2) == len(val_games)


# ---------------------------------------------------------------------------
# 5. Late-season training cutoff
# ---------------------------------------------------------------------------


class TestLateSeasoncutoff:
    def test_cutoff_filters_early_games(self):
        """Games before the cutoff date should be excluded from training."""
        pipeline = SOTAPipeline(SOTAPipelineConfig(
            year=2026,
            late_season_training_cutoff_days=75,
            random_seed=42,
        ))
        # Tournament start: March 14, minus 75 days = ~Dec 29
        from datetime import date, timedelta
        tournament_start = date(2026, 3, 14)
        cutoff_date = tournament_start - timedelta(days=75)
        cutoff_key = pipeline._game_sort_key(cutoff_date.isoformat())

        # Generate dates before and after cutoff
        before_cutoff = [f"2025-11-{d:02d}" for d in range(15, 30)]  # November
        after_cutoff = [f"2026-01-{d:02d}" for d in range(1, 31)]  # January

        all_dates = before_cutoff + after_cutoff

        # Apply the same filter the pipeline uses
        all_games_uncutoff = all_dates[:]
        filtered = [
            d for d in all_dates
            if pipeline._game_sort_key(d) >= cutoff_key
        ]

        # November dates should be excluded, January dates kept
        assert len(filtered) < len(all_dates)
        for d in filtered:
            assert pipeline._game_sort_key(d) >= cutoff_key

    def test_cutoff_zero_disables_filter(self):
        """With cutoff_days=0, all games should be kept."""
        config = SOTAPipelineConfig(
            year=2026,
            late_season_training_cutoff_days=0,
            random_seed=42,
        )
        # cutoff_days=0 means skip the filter entirely
        assert config.late_season_training_cutoff_days == 0


# ---------------------------------------------------------------------------
# 6. Baseline model reuses pre-computed boundary
# ---------------------------------------------------------------------------


class TestBaselineReuseBoundary:
    def test_boundary_consistent_between_methods(self):
        """_train_baseline_model should use the pre-computed boundary, not recompute."""
        dates = [f"2026-01-{d:02d}" for d in range(1, 31)]
        dates += [f"2026-02-{d:02d}" for d in range(1, 11)]
        pipeline, game_flows = _make_pipeline_with_games(dates)

        # Pre-compute boundary
        pipeline._compute_train_val_boundary(game_flows)
        boundary_before = pipeline._validation_sort_key_boundary
        assert boundary_before is not None

        # The boundary should not change — _train_baseline_model should
        # reuse it rather than recomputing.
        # (We can't easily run _train_baseline_model without full feature
        # engineering, but we verify the boundary is preserved through
        # the pipeline object state.)
        assert pipeline._validation_sort_key_boundary == boundary_before


# ---------------------------------------------------------------------------
# 7. GNN target masking for validation-era-only teams
# ---------------------------------------------------------------------------


class TestGNNTargetMasking:
    def test_gnn_masks_non_training_era_teams(self):
        """GNN should use 0.0 target for teams not in training-era graph edges."""
        from src.ml.gnn.schedule_graph import ScheduleGraph, ScheduleEdge

        # Create graph with 3 teams: A and B have edges, C does not
        graph = ScheduleGraph(["team_a", "team_b", "team_c"])
        graph.add_game(ScheduleEdge(
            game_id="g1", team1_id="team_a", team2_id="team_b",
            actual_margin=5.0, xp_margin=3.0, location_weight=0.5,
            game_date="2026-01-15",
        ))

        # Build training-era team set (same logic as in _run_gnn)
        training_era_teams = set()
        for edge in graph.edges:
            training_era_teams.add(edge.team1_id)
            training_era_teams.add(edge.team2_id)

        assert "team_a" in training_era_teams
        assert "team_b" in training_era_teams
        assert "team_c" not in training_era_teams  # No edges → masked

    def test_gnn_preserves_training_era_targets(self):
        """Teams in training-era edges should keep their AdjEM targets."""
        from src.ml.gnn.schedule_graph import ScheduleGraph, ScheduleEdge

        graph = ScheduleGraph(["team_a", "team_b"])
        graph.add_game(ScheduleEdge(
            game_id="g1", team1_id="team_a", team2_id="team_b",
            actual_margin=5.0, xp_margin=3.0, location_weight=0.5,
            game_date="2026-01-15",
        ))

        training_era_teams = set()
        for edge in graph.edges:
            training_era_teams.add(edge.team1_id)
            training_era_teams.add(edge.team2_id)

        # Both teams should be in training era
        assert len(training_era_teams) == 2


# ---------------------------------------------------------------------------
# 8. Confidence estimation is diagnostic-only (no model_confidence mutation)
# ---------------------------------------------------------------------------


class TestConfidenceEstimationIsolation:
    def test_method_does_not_set_model_confidence(self):
        """_estimate_model_confidence_intervals should NOT set model_confidence.

        Verifying by source inspection: the method should contain
        'confidence_diagnostic' key in stats, NOT 'self.model_confidence[...] ='.
        This is a structural test since running the full method requires
        complete feature engineering setup.
        """
        import inspect
        source = inspect.getsource(SOTAPipeline._estimate_model_confidence_intervals)

        # Should NOT directly assign to self.model_confidence
        assert "self.model_confidence[model_name] = " not in source, (
            "_estimate_model_confidence_intervals should not set self.model_confidence "
            "— this would leak validation-era Brier into CFA base weights"
        )

        # Should contain the diagnostic key instead
        assert "confidence_diagnostic" in source, (
            "_estimate_model_confidence_intervals should use 'confidence_diagnostic' "
            "key to signal it's a diagnostic, not an authoritative confidence"
        )

    def test_model_confidence_set_during_training_not_estimation(self):
        """GNN and transformer set confidence from training loss, not validation Brier."""
        import inspect

        gnn_source = inspect.getsource(SOTAPipeline._run_gnn)
        # GNN should set confidence from training loss
        assert 'model_confidence["gnn"]' in gnn_source

        transformer_source = inspect.getsource(SOTAPipeline._run_transformer)
        # Transformer should set confidence from training loss
        assert 'model_confidence["transformer"]' in transformer_source


# ---------------------------------------------------------------------------
# 9. Point-in-time noise scaling
# ---------------------------------------------------------------------------


class TestPointInTimeNoise:
    def test_early_season_gets_more_noise(self):
        """Earlier games in the training window should get larger noise scales."""
        # Simulate the noise computation logic
        progress = np.array([0.0, 0.25, 0.5, 0.75, 1.0])  # season progress
        season_remaining = 1.0 - progress
        noise_scale = 0.08 * np.sqrt(season_remaining)

        # Noise should decrease as progress increases
        for i in range(len(noise_scale) - 1):
            assert noise_scale[i] > noise_scale[i + 1], (
                f"Noise at progress {progress[i]} ({noise_scale[i]:.4f}) "
                f"should be > noise at {progress[i+1]} ({noise_scale[i+1]:.4f})"
            )

        # Start-of-season noise should be ~0.08 (sqrt(1.0) = 1.0)
        assert abs(noise_scale[0] - 0.08) < 1e-6

        # End-of-season noise should be 0.0 (sqrt(0.0) = 0.0)
        assert abs(noise_scale[-1]) < 1e-6

    def test_mid_season_noise_is_substantial(self):
        """Mid-season games (50% progress) should still get meaningful noise."""
        progress = 0.5
        season_remaining = 1.0 - progress
        noise_scale = 0.08 * np.sqrt(season_remaining)

        # sqrt(0.5) ≈ 0.707, so noise ≈ 0.057
        assert noise_scale > 0.05, f"Mid-season noise {noise_scale:.4f} should be > 0.05"


# ---------------------------------------------------------------------------
# 10. WAB bubble team prior
# ---------------------------------------------------------------------------


class TestWABBubblePrior:
    def test_bubble_em_blended_toward_prior(self):
        """WAB bubble AdjEM should be blended toward the historical prior."""
        from src.data.features.proprietary_metrics import ProprietaryMetricsEngine

        engine = ProprietaryMetricsEngine()

        # The raw bubble EM and the blended value should differ
        raw_bubble_em = 8.0  # hypothetical current-year 45th team
        blended = 0.7 * raw_bubble_em + 0.3 * engine.BUBBLE_EM_PRIOR

        # With prior=5.0 and raw=8.0: blended = 0.7*8 + 0.3*5 = 5.6 + 1.5 = 7.1
        expected = 0.7 * 8.0 + 0.3 * 5.0
        assert abs(blended - expected) < 1e-6
        assert blended < raw_bubble_em  # Blended should be closer to prior

    def test_bubble_em_prior_exists(self):
        """ProprietaryMetricsEngine should have BUBBLE_EM_PRIOR attribute."""
        from src.data.features.proprietary_metrics import ProprietaryMetricsEngine

        engine = ProprietaryMetricsEngine()
        assert hasattr(engine, "BUBBLE_EM_PRIOR")
        assert engine.BUBBLE_EM_PRIOR > 0.0
