"""Lightweight DAG orchestrator for data ingestion pipelines.

Models ingestion tasks as a directed acyclic graph with idempotency,
content-hash-based caching, and dependency-aware execution.

Implements Agent Directive V7 S19 (data engineering — DAG orchestration).

Usage:
    from src.data.ingestion.dag import DagExecutor, DagTask

    class FetchGamesTask(DagTask):
        name = "fetch_games"
        depends_on = []
        def run(self, context):
            return load_games(context["season"])
        def output_key(self, context):
            return f"games_{context['season']}"

    dag = DagExecutor()
    dag.add_task(FetchGamesTask())
    dag.add_task(ValidateGamesTask())
    results = dag.execute({"season": 2026})
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from ...governance.cache_hygiene import CACHE_PROTOCOL_VERSION, fingerprint_context

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = "data/cache/dag"


@dataclass
class TaskResult:
    """Result of executing a single DAG task."""

    task_name: str
    success: bool = True
    output: Any = None
    cached: bool = False
    wall_seconds: float = 0.0
    error: Optional[str] = None
    output_hash: Optional[str] = None


class DagTask(ABC):
    """Abstract base class for DAG tasks.

    Subclass this and implement ``run()`` and ``output_key()``.
    Tasks with the same output_key and unchanged inputs are skipped
    via idempotency caching.
    """

    name: str = ""
    depends_on: List[str] = []

    @abstractmethod
    def run(self, context: Dict[str, Any], **upstream: Any) -> Any:
        """Execute the task.

        Args:
            context: Shared execution context (e.g., season, config).
            **upstream: Outputs from dependency tasks.

        Returns:
            Task output (will be passed to downstream tasks).
        """
        ...

    def output_key(self, context: Dict[str, Any]) -> str:
        """Return a cache key for this task's output.

        Override to include context-dependent info (e.g., season).
        Default: task name.
        """
        return self.name

    def should_skip(self, context: Dict[str, Any], cache_dir: Path) -> bool:
        """Check if this task can be skipped (idempotency check).

        Override for custom skip logic. Default checks the cache marker.
        """
        marker = cache_dir / f"{self.output_key(context)}.marker"
        return marker.exists()


class DagExecutor:
    """Execute DAG tasks in dependency order with idempotency caching.

    Tasks are topologically sorted by their dependencies, then executed
    sequentially. Completed tasks write a marker file to enable skipping
    on subsequent runs.
    """

    def __init__(
        self,
        cache_dir: str = DEFAULT_CACHE_DIR,
        skip_cached: bool = True,
    ) -> None:
        self._tasks: Dict[str, DagTask] = {}
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._skip_cached = skip_cached

    def add_task(self, task: DagTask) -> None:
        """Register a task in the DAG."""
        if not task.name:
            raise ValueError("Task must have a non-empty name")
        self._tasks[task.name] = task

    def remove_task(self, name: str) -> None:
        """Remove a task from the DAG."""
        self._tasks.pop(name, None)

    def execution_order(self) -> List[str]:
        """Compute topological execution order.

        Returns:
            Task names in dependency order.

        Raises:
            ValueError: On circular dependencies or missing dependencies.
        """
        # Validate dependencies exist
        for name, task in self._tasks.items():
            for dep in task.depends_on:
                if dep not in self._tasks:
                    raise ValueError(f"Task '{name}' depends on unknown task '{dep}'")

        # Kahn's algorithm
        in_degree: Dict[str, int] = {name: 0 for name in self._tasks}
        for name, task in self._tasks.items():
            for dep in task.depends_on:
                in_degree[name] += 1

        queue = sorted([n for n, d in in_degree.items() if d == 0])
        order: List[str] = []

        while queue:
            node = queue.pop(0)
            order.append(node)
            for name, task in self._tasks.items():
                if node in task.depends_on:
                    in_degree[name] -= 1
                    if in_degree[name] == 0:
                        queue.append(name)
            queue.sort()

        if len(order) != len(self._tasks):
            raise ValueError("Circular dependency detected in DAG")

        return order

    def execute(
        self,
        context: Dict[str, Any],
        force: bool = False,
    ) -> Dict[str, TaskResult]:
        """Execute all tasks in dependency order.

        Args:
            context: Shared execution context.
            force: If True, ignore cache and re-execute all tasks.

        Returns:
            Dict of task_name → TaskResult.
        """
        order = self.execution_order()
        results: Dict[str, TaskResult] = {}
        outputs: Dict[str, Any] = {}

        for name in order:
            task = self._tasks[name]

            # Idempotency check
            if self._skip_cached and not force and task.should_skip(context, self._cache_dir):
                cached_output = self._load_cached(task, context)
                if cached_output is not None:
                    results[name] = TaskResult(
                        task_name=name,
                        success=True,
                        output=cached_output,
                        cached=True,
                    )
                    outputs[name] = cached_output
                    logger.info("Skipped (cached): %s", name)
                    continue

            # Collect upstream outputs
            upstream = {dep: outputs.get(dep) for dep in task.depends_on if dep in outputs}

            # Execute
            start = time.perf_counter()
            try:
                output = task.run(context, **upstream)
                elapsed = time.perf_counter() - start

                # Write cache marker
                self._write_marker(task, context, output)
                outputs[name] = output

                results[name] = TaskResult(
                    task_name=name,
                    success=True,
                    output=output,
                    wall_seconds=elapsed,
                )
                logger.info("Completed: %s (%.1fs)", name, elapsed)

            except Exception as exc:
                elapsed = time.perf_counter() - start
                results[name] = TaskResult(
                    task_name=name,
                    success=False,
                    error=str(exc),
                    wall_seconds=elapsed,
                )
                logger.error("Failed: %s — %s", name, exc)
                break  # Stop on failure

        return results

    def invalidate(self, task_name: str, context: Dict[str, Any]) -> bool:
        """Invalidate a task's cache, forcing re-execution on next run.

        Also invalidates all downstream dependents.
        """
        task = self._tasks.get(task_name)
        if task is None:
            return False

        marker = self._cache_dir / f"{task.output_key(context)}.marker"
        removed = False
        if marker.exists():
            marker.unlink()
            removed = True

        # Cascade to dependents
        for name, t in self._tasks.items():
            if task_name in t.depends_on:
                self.invalidate(name, context)

        return removed

    def summary(self, context: Optional[Dict[str, Any]] = None) -> str:
        """Human-readable DAG summary."""
        lines = [
            "Data Ingestion DAG",
            "=" * 50,
            f"Tasks: {len(self._tasks)}",
            "",
        ]

        try:
            order = self.execution_order()
        except ValueError as e:
            return f"DAG Error: {e}"

        for name in order:
            task = self._tasks[name]
            deps = ", ".join(task.depends_on) if task.depends_on else "(root)"
            cached = ""
            if context:
                cached = " [cached]" if task.should_skip(context, self._cache_dir) else ""
            lines.append(f"  {name:<30s} deps={deps}{cached}")

        return "\n".join(lines)

    def _write_marker(
        self,
        task: DagTask,
        context: Dict[str, Any],
        output: Any,
    ) -> None:
        """Write an idempotency marker for a completed task."""
        key = task.output_key(context)
        marker = self._cache_dir / f"{key}.marker"
        context_fingerprint = fingerprint_context(context)

        # Hash the output for change detection
        output_str = json.dumps(output, sort_keys=True, default=str)
        output_hash = hashlib.sha256(output_str.encode()).hexdigest()[:16]

        marker_data = {
            "protocol_version": CACHE_PROTOCOL_VERSION,
            "task": task.name,
            "key": key,
            "context_fingerprint": context_fingerprint,
            "output_hash": output_hash,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        with open(marker, "w") as f:
            json.dump(marker_data, f)

    def _load_cached(self, task: DagTask, context: Dict[str, Any]) -> Optional[Any]:
        """Load cached output for a task (marker data only)."""
        key = task.output_key(context)
        marker = self._cache_dir / f"{key}.marker"
        if marker.exists():
            try:
                with open(marker) as f:
                    payload = json.load(f)
                if not self._marker_matches(payload, task, context, key):
                    marker.unlink(missing_ok=True)
                    return None
                return payload
            except (json.JSONDecodeError, OSError):
                return None
        return None

    def _marker_matches(
        self,
        payload: Dict[str, Any],
        task: DagTask,
        context: Dict[str, Any],
        key: str,
    ) -> bool:
        """Validate that a cache marker belongs to this exact execution context."""
        if not isinstance(payload, dict):
            return False
        if payload.get("protocol_version") != CACHE_PROTOCOL_VERSION:
            return False
        if payload.get("task") != task.name:
            return False
        if payload.get("key") != key:
            return False
        expected_context = fingerprint_context(context)
        return payload.get("context_fingerprint") == expected_context
