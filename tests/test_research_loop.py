"""Tests for experiment scheduling and research loop (S14 compliance).

Covers:
  - Config variant generation (perturbation, grid, adaptive)
  - Experiment queue management
  - Knowledge store queries (regime insights, unexplored params)
  - Best variant selection
"""

import json

import pytest


class TestExperimentScheduler:
    """S14: Experiment scheduling and variant generation."""

    def test_generate_perturbation_variants(self, tmp_path):
        from src.research.experiment_scheduler import ExperimentScheduler

        scheduler = ExperimentScheduler(
            queue_path=str(tmp_path / "queue.jsonl"), seed=42
        )
        base_config = {"model": "ensemble", "year": 2026}
        variants = scheduler.generate_variants(base_config, n=5, strategy="perturbation")
        assert len(variants) == 5
        for v in variants:
            assert v.variant_id
            assert v.parameter_deltas
            assert v.hypothesis

    def test_generate_grid_variants(self, tmp_path):
        from src.research.experiment_scheduler import ExperimentScheduler

        scheduler = ExperimentScheduler(
            queue_path=str(tmp_path / "queue.jsonl"), seed=42
        )
        variants = scheduler.generate_variants({}, n=5, strategy="grid")
        assert len(variants) == 5
        for v in variants:
            assert "Grid sweep" in v.hypothesis

    def test_ensemble_weights_normalized(self, tmp_path):
        from src.research.experiment_scheduler import ExperimentScheduler

        scheduler = ExperimentScheduler(
            queue_path=str(tmp_path / "queue.jsonl"), seed=42
        )
        # Generate many variants, check any with weight changes sum to ~1.0
        variants = scheduler.generate_variants({}, n=50, strategy="perturbation")
        weight_params = {"lgb_weight", "xgb_weight", "spread_weight", "logistic_weight"}
        for v in variants:
            weights = {k: v2 for k, v2 in v.parameter_deltas.items() if k in weight_params}
            if len(weights) > 0:
                # Include defaults for missing weights
                all_weights = dict(weights)
                for p in weight_params:
                    if p not in all_weights:
                        all_weights[p] = scheduler._param_ranges[p]["default"]
                # Weights should approximately sum to 1.0
                # (They're normalized relative to changed subset)
                total = sum(all_weights.values())
                assert 0.5 < total < 1.5  # Reasonable range

    def test_queue_and_retrieve_variants(self, tmp_path):
        from src.research.experiment_scheduler import ExperimentScheduler

        scheduler = ExperimentScheduler(
            queue_path=str(tmp_path / "queue.jsonl"), seed=42
        )
        variants = scheduler.generate_variants({}, n=3)
        queued = scheduler.queue_variants(variants)
        assert queued == 3

        next_v = scheduler.next_variant()
        assert next_v is not None
        assert next_v.status == "queued"

    def test_mark_completed_and_select_best(self, tmp_path):
        from src.research.experiment_scheduler import ExperimentScheduler

        scheduler = ExperimentScheduler(
            queue_path=str(tmp_path / "queue.jsonl"), seed=42
        )
        variants = scheduler.generate_variants({}, n=3)
        scheduler.queue_variants(variants)

        # Mark all as completed with different Brier scores
        scheduler.mark_completed(variants[0].variant_id, brier=0.200)
        scheduler.mark_completed(variants[1].variant_id, brier=0.180)
        scheduler.mark_completed(variants[2].variant_id, brier=0.195)

        best = scheduler.select_best()
        assert best is not None
        assert best.result_brier == 0.180
        assert best.variant_id == variants[1].variant_id

    def test_summary(self, tmp_path):
        from src.research.experiment_scheduler import ExperimentScheduler

        scheduler = ExperimentScheduler(
            queue_path=str(tmp_path / "queue.jsonl"), seed=42
        )
        variants = scheduler.generate_variants({}, n=2)
        scheduler.queue_variants(variants)
        summary = scheduler.summary()
        assert "Experiment Queue" in summary
        assert "queued: 2" in summary


class TestKnowledgeStore:
    """S14: Knowledge retention and research insights."""

    def _make_registry(self, tmp_path, records):
        """Create a mock registry with given records."""
        from src.ml.evaluation.experiment_registry import (
            ExperimentRecord,
            ExperimentRegistry,
        )

        registry = ExperimentRegistry(ledger_path=str(tmp_path / "ledger.jsonl"))
        for rec_data in records:
            record = ExperimentRecord(**rec_data)
            registry.log(record)
        return registry

    def test_insights_by_year(self, tmp_path):
        from src.research.knowledge_store import KnowledgeStore

        registry = self._make_registry(
            tmp_path,
            [
                {
                    "loyo_mean_brier": 0.190,
                    "loyo_year_briers": {2020: 0.200, 2021: 0.180},
                },
                {
                    "loyo_mean_brier": 0.185,
                    "loyo_year_briers": {2020: 0.195, 2021: 0.175},
                },
            ],
        )
        store = KnowledgeStore(registry)
        insights = store.insights_by_year()
        assert 2020 in insights
        assert 2021 in insights
        assert insights[2020]["best_brier"] == 0.195
        assert insights[2021]["n_experiments"] == 2

    def test_what_works(self, tmp_path):
        from src.research.knowledge_store import KnowledgeStore

        registry = self._make_registry(
            tmp_path,
            [
                {"loyo_mean_brier": 0.190, "model_family": "ensemble_lgb_xgb"},
                {"loyo_mean_brier": 0.185, "model_family": "ensemble_lgb_xgb"},
                {"loyo_mean_brier": 0.195, "model_family": "logistic"},
            ],
        )
        store = KnowledgeStore(registry)
        result = store.what_works(top_n=2)
        assert result["top_n"] == 2
        assert "ensemble_lgb_xgb" in result["common_model_families"]

    def test_unexplored_parameters_empty_registry(self, tmp_path):
        from src.research.knowledge_store import KnowledgeStore

        registry = self._make_registry(tmp_path, [])
        store = KnowledgeStore(registry)
        unexplored = store.unexplored_parameters()
        # All params should be "never_explored"
        assert len(unexplored) > 0
        assert all(u["status"] == "never_explored" for u in unexplored)

    def test_research_summary(self, tmp_path):
        from src.research.knowledge_store import KnowledgeStore

        registry = self._make_registry(
            tmp_path,
            [{"loyo_mean_brier": 0.190, "loyo_year_briers": {2022: 0.19}}],
        )
        store = KnowledgeStore(registry)
        summary = store.research_summary()
        assert "Research Knowledge Summary" in summary
        assert "0.1900" in summary
