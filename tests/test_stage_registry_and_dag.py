"""Tests for pipeline stage registry (S2) and DAG orchestration (S19).

Covers:
  - Stage registration and dependency ordering
  - Circular dependency detection
  - Stage enable/disable
  - DAG task execution with idempotency caching
  - Cache invalidation
"""

import json

import pytest


class TestStageRegistry:
    """S2: Dynamic stage registry for composable pipeline."""

    def _make_stage(self, name, output=None):
        """Create a simple test stage."""

        class SimpleStage:
            def __init__(self, name, output):
                self._name = name
                self._output = output

            @property
            def name(self):
                return self._name

            def run(self, context=None, **kwargs):
                return self._output or f"{self._name}_result"

        return SimpleStage(name, output)

    def test_register_and_list_stages(self):
        from src.pipeline.stage_registry import StageRegistry

        registry = StageRegistry()
        registry.register("load", self._make_stage("load"))
        registry.register("train", self._make_stage("train"), depends_on=["load"])

        assert "load" in registry.stages
        assert "train" in registry.stages

    def test_execution_order_respects_dependencies(self):
        from src.pipeline.stage_registry import StageRegistry

        registry = StageRegistry()
        registry.register("c", self._make_stage("c"), depends_on=["a", "b"])
        registry.register("a", self._make_stage("a"))
        registry.register("b", self._make_stage("b"), depends_on=["a"])

        order = registry.get_execution_order()
        assert order.index("a") < order.index("b")
        assert order.index("b") < order.index("c")

    def test_circular_dependency_detected(self):
        from src.pipeline.stage_registry import StageRegistry

        registry = StageRegistry()
        registry.register("a", self._make_stage("a"), depends_on=["b"])
        registry.register("b", self._make_stage("b"), depends_on=["a"])

        with pytest.raises(ValueError, match="Circular"):
            registry.get_execution_order()

    def test_execute_all_stages(self):
        from src.pipeline.stage_registry import StageRegistry

        registry = StageRegistry()
        registry.register("load", self._make_stage("load", output="data"))
        registry.register(
            "train", self._make_stage("train", output="model"), depends_on=["load"]
        )

        results = registry.execute_all()
        assert results["load"].success
        assert results["train"].success
        assert results["load"].output == "data"

    def test_disable_stage_skips_execution(self):
        from src.pipeline.stage_registry import StageRegistry

        registry = StageRegistry()
        registry.register("load", self._make_stage("load"))
        registry.register("optional", self._make_stage("optional"))
        registry.disable("optional")

        order = registry.get_execution_order()
        assert "optional" not in order

    def test_validate_missing_dependency(self):
        from src.pipeline.stage_registry import StageRegistry

        registry = StageRegistry()
        registry.register("train", self._make_stage("train"), depends_on=["load"])

        errors = registry.validate_dependencies()
        assert len(errors) > 0
        assert "load" in errors[0]

    def test_execution_stops_on_failure(self):
        from src.pipeline.stage_registry import StageRegistry

        class FailingStage:
            @property
            def name(self):
                return "failing"

            def run(self, context=None, **kwargs):
                raise RuntimeError("Stage failed!")

        registry = StageRegistry()
        registry.register("load", self._make_stage("load"))
        registry.register("fail", FailingStage(), depends_on=["load"])
        registry.register("train", self._make_stage("train"), depends_on=["fail"])

        results = registry.execute_all()
        assert results["load"].success
        assert not results["fail"].success
        assert "train" not in results  # Never reached

    def test_summary(self):
        from src.pipeline.stage_registry import StageRegistry

        registry = StageRegistry()
        registry.register("load", self._make_stage("load"))
        registry.register("train", self._make_stage("train"), depends_on=["load"])

        summary = registry.summary()
        assert "Pipeline Stage Registry" in summary
        assert "load" in summary


class TestDagExecutor:
    """S19: DAG orchestration with idempotency."""

    def _make_task(self, name, depends=None, output="result"):
        """Create a simple test task."""
        from src.data.ingestion.dag import DagTask

        class SimpleTask(DagTask):
            def __init__(self):
                self.name = name
                self.depends_on = depends or []
                self._output = output
                self.executed = False

            def run(self, context, **upstream):
                self.executed = True
                return self._output

            def output_key(self, context):
                return f"{self.name}_{context.get('season', 'default')}"

        return SimpleTask()

    def test_execute_simple_dag(self, tmp_path):
        from src.data.ingestion.dag import DagExecutor

        dag = DagExecutor(cache_dir=str(tmp_path / "cache"), skip_cached=False)
        fetch = self._make_task("fetch", output="games_data")
        validate = self._make_task("validate", depends=["fetch"], output="valid")

        dag.add_task(fetch)
        dag.add_task(validate)

        results = dag.execute({"season": 2026})
        assert results["fetch"].success
        assert results["validate"].success
        assert fetch.executed
        assert validate.executed

    def test_execution_order(self, tmp_path):
        from src.data.ingestion.dag import DagExecutor

        dag = DagExecutor(cache_dir=str(tmp_path / "cache"), skip_cached=False)
        dag.add_task(self._make_task("c", depends=["a", "b"]))
        dag.add_task(self._make_task("a"))
        dag.add_task(self._make_task("b", depends=["a"]))

        order = dag.execution_order()
        assert order.index("a") < order.index("b")
        assert order.index("b") < order.index("c")

    def test_idempotency_skips_cached(self, tmp_path):
        from src.data.ingestion.dag import DagExecutor

        dag = DagExecutor(cache_dir=str(tmp_path / "cache"), skip_cached=True)
        task = self._make_task("fetch", output="data")
        dag.add_task(task)

        # First run — executes
        results1 = dag.execute({"season": 2026})
        assert results1["fetch"].success
        assert not results1["fetch"].cached
        assert task.executed

        # Reset and re-run — should be cached
        task.executed = False
        results2 = dag.execute({"season": 2026})
        assert results2["fetch"].cached
        assert not task.executed  # Not re-executed

    def test_force_ignores_cache(self, tmp_path):
        from src.data.ingestion.dag import DagExecutor

        dag = DagExecutor(cache_dir=str(tmp_path / "cache"), skip_cached=True)
        task = self._make_task("fetch", output="data")
        dag.add_task(task)

        dag.execute({"season": 2026})
        task.executed = False

        # Force re-execution
        results = dag.execute({"season": 2026}, force=True)
        assert not results["fetch"].cached
        assert task.executed

    def test_invalidate_cache(self, tmp_path):
        from src.data.ingestion.dag import DagExecutor

        dag = DagExecutor(cache_dir=str(tmp_path / "cache"), skip_cached=True)
        task = self._make_task("fetch", output="data")
        dag.add_task(task)

        dag.execute({"season": 2026})
        assert dag.invalidate("fetch", {"season": 2026})

        # Should re-execute after invalidation
        task.executed = False
        results = dag.execute({"season": 2026})
        assert not results["fetch"].cached
        assert task.executed

    def test_cache_marker_rejected_when_context_changes(self, tmp_path):
        from src.data.ingestion.dag import DagExecutor, DagTask

        class ConstantKeyTask(DagTask):
            name = "fetch"
            depends_on = []

            def __init__(self):
                self.executions = 0

            def run(self, context, **upstream):
                self.executions += 1
                return {"season": context["season"]}

            def output_key(self, context):
                return "shared_key"

        dag = DagExecutor(cache_dir=str(tmp_path / "cache"), skip_cached=True)
        task = ConstantKeyTask()
        dag.add_task(task)

        first = dag.execute({"season": 2026})
        assert not first["fetch"].cached
        assert task.executions == 1

        second = dag.execute({"season": 2027})
        assert not second["fetch"].cached
        assert task.executions == 2

    def test_legacy_marker_without_context_fingerprint_is_invalidated(self, tmp_path):
        import json

        from src.data.ingestion.dag import DagExecutor

        dag = DagExecutor(cache_dir=str(tmp_path / "cache"), skip_cached=True)
        task = self._make_task("fetch", output="data")
        dag.add_task(task)

        marker = tmp_path / "cache" / "fetch_2026.marker"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps({"task": "fetch", "key": "fetch_2026"}))

        results = dag.execute({"season": 2026})
        assert not results["fetch"].cached
        assert task.executed
        assert marker.exists()

    def test_circular_dependency_detected(self, tmp_path):
        from src.data.ingestion.dag import DagExecutor

        dag = DagExecutor(cache_dir=str(tmp_path / "cache"))
        dag.add_task(self._make_task("a", depends=["b"]))
        dag.add_task(self._make_task("b", depends=["a"]))

        with pytest.raises(ValueError, match="Circular"):
            dag.execution_order()

    def test_missing_dependency_detected(self, tmp_path):
        from src.data.ingestion.dag import DagExecutor

        dag = DagExecutor(cache_dir=str(tmp_path / "cache"))
        dag.add_task(self._make_task("validate", depends=["fetch"]))

        with pytest.raises(ValueError, match="unknown task"):
            dag.execution_order()

    def test_summary(self, tmp_path):
        from src.data.ingestion.dag import DagExecutor

        dag = DagExecutor(cache_dir=str(tmp_path / "cache"))
        dag.add_task(self._make_task("fetch"))
        dag.add_task(self._make_task("validate", depends=["fetch"]))

        summary = dag.summary()
        assert "Data Ingestion DAG" in summary
        assert "fetch" in summary
