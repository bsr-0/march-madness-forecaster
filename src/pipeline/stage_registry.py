"""Dynamic stage registry for composable pipeline execution.

Enables runtime composition of pipeline stages via registration,
dependency ordering, and configurable execution. Replaces hard-coded
stage calls with a registry-driven approach.

Implements Agent Directive V7 S2 (modular, agent-like architecture).

Usage:
    from src.pipeline.stage_registry import StageRegistry

    registry = StageRegistry()
    registry.register("data_loading", DataLoadingStage(), depends_on=[])
    registry.register("feature_eng", FeatureEngStage(), depends_on=["data_loading"])
    results = registry.execute_all(context)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class StageEntry:
    """Registry entry for a pipeline stage."""

    name: str
    stage: Any  # PipelineStage instance
    depends_on: List[str] = field(default_factory=list)
    enabled: bool = True
    timeout_seconds: Optional[float] = None
    description: str = ""


@dataclass
class StageResult:
    """Result from executing a single stage."""

    name: str
    output: Any = None
    success: bool = True
    error: Optional[str] = None
    wall_seconds: float = 0.0


class StageRegistry:
    """Dynamic registry for composable pipeline stage execution.

    Stages are registered with dependencies, then executed in
    topological order. Supports enabling/disabling stages, dependency
    validation, and result propagation.
    """

    def __init__(self) -> None:
        self._stages: Dict[str, StageEntry] = {}
        self._execution_order: Optional[List[str]] = None

    def register(
        self,
        name: str,
        stage: Any,
        depends_on: Optional[List[str]] = None,
        enabled: bool = True,
        timeout_seconds: Optional[float] = None,
        description: str = "",
    ) -> None:
        """Register a pipeline stage.

        Args:
            name: Unique stage identifier.
            stage: PipelineStage instance (must have a run() method).
            depends_on: List of stage names this stage depends on.
            enabled: Whether the stage is active.
            timeout_seconds: Optional execution timeout.
            description: Human-readable description.
        """
        if name in self._stages:
            logger.warning("Replacing existing stage '%s'", name)

        self._stages[name] = StageEntry(
            name=name,
            stage=stage,
            depends_on=depends_on or [],
            enabled=enabled,
            timeout_seconds=timeout_seconds,
            description=description,
        )
        self._execution_order = None  # Invalidate cached order

    def unregister(self, name: str) -> None:
        """Remove a stage from the registry."""
        self._stages.pop(name, None)
        self._execution_order = None

    def enable(self, name: str) -> None:
        """Enable a stage."""
        if name in self._stages:
            self._stages[name].enabled = True

    def disable(self, name: str) -> None:
        """Disable a stage."""
        if name in self._stages:
            self._stages[name].enabled = False

    def get_execution_order(self) -> List[str]:
        """Compute topological execution order respecting dependencies.

        Returns:
            List of stage names in execution order.

        Raises:
            ValueError: If circular dependencies are detected.
        """
        if self._execution_order is not None:
            return self._execution_order

        enabled = {
            name: entry
            for name, entry in self._stages.items()
            if entry.enabled
        }

        # Kahn's algorithm for topological sort
        in_degree: Dict[str, int] = {name: 0 for name in enabled}
        for name, entry in enabled.items():
            for dep in entry.depends_on:
                if dep in enabled:
                    in_degree[name] += 1

        queue = [name for name, deg in in_degree.items() if deg == 0]
        order: List[str] = []

        while queue:
            # Sort for deterministic ordering among peers
            queue.sort()
            node = queue.pop(0)
            order.append(node)

            for name, entry in enabled.items():
                if node in entry.depends_on:
                    in_degree[name] -= 1
                    if in_degree[name] == 0:
                        queue.append(name)

        if len(order) != len(enabled):
            missing = set(enabled) - set(order)
            raise ValueError(
                f"Circular dependency detected among stages: {missing}"
            )

        self._execution_order = order
        return order

    def validate_dependencies(self) -> List[str]:
        """Validate that all dependencies are satisfiable.

        Returns:
            List of error messages (empty if valid).
        """
        errors: List[str] = []
        enabled_names = {
            name for name, entry in self._stages.items() if entry.enabled
        }

        for name, entry in self._stages.items():
            if not entry.enabled:
                continue
            for dep in entry.depends_on:
                if dep not in enabled_names:
                    errors.append(
                        f"Stage '{name}' depends on '{dep}' which is not registered/enabled"
                    )

        # Check for circular dependencies
        try:
            self.get_execution_order()
        except ValueError as e:
            errors.append(str(e))

        return errors

    def execute_all(
        self,
        context: Any = None,
        on_stage_complete: Optional[Callable[[StageResult], None]] = None,
    ) -> Dict[str, StageResult]:
        """Execute all enabled stages in dependency order.

        Args:
            context: Shared pipeline context passed to each stage.
            on_stage_complete: Optional callback after each stage completes.

        Returns:
            Dict mapping stage name to its StageResult.
        """
        errors = self.validate_dependencies()
        if errors:
            raise ValueError(f"Dependency validation failed: {errors}")

        order = self.get_execution_order()
        results: Dict[str, StageResult] = {}
        upstream_outputs: Dict[str, Any] = {}

        for name in order:
            entry = self._stages[name]
            logger.info("Executing stage: %s", name)

            start = time.perf_counter()
            try:
                # Collect upstream outputs
                deps = {
                    dep: upstream_outputs.get(dep)
                    for dep in entry.depends_on
                    if dep in upstream_outputs
                }

                # Execute stage
                if hasattr(entry.stage, "run"):
                    output = entry.stage.run(context, **deps)
                else:
                    output = entry.stage(context, **deps)

                elapsed = time.perf_counter() - start
                result = StageResult(
                    name=name, output=output, success=True, wall_seconds=elapsed
                )
                upstream_outputs[name] = output

            except Exception as exc:
                elapsed = time.perf_counter() - start
                result = StageResult(
                    name=name,
                    success=False,
                    error=str(exc),
                    wall_seconds=elapsed,
                )
                logger.error("Stage '%s' failed: %s", name, exc)

            results[name] = result
            if on_stage_complete:
                on_stage_complete(result)

            if not result.success:
                logger.error(
                    "Stopping pipeline: stage '%s' failed after %.1fs",
                    name,
                    elapsed,
                )
                break

        return results

    @property
    def stages(self) -> Dict[str, StageEntry]:
        """Return registered stages."""
        return dict(self._stages)

    def summary(self) -> str:
        """Human-readable summary of registered stages."""
        lines = [
            "Pipeline Stage Registry",
            "=" * 60,
            f"  {'Stage':<25s} {'Enabled':<10s} {'Dependencies'}",
            "-" * 60,
        ]

        try:
            order = self.get_execution_order()
        except ValueError:
            order = list(self._stages.keys())

        for name in order:
            entry = self._stages[name]
            enabled = "Yes" if entry.enabled else "No"
            deps = ", ".join(entry.depends_on) if entry.depends_on else "(none)"
            lines.append(f"  {name:<25s} {enabled:<10s} {deps}")

        disabled = [
            n for n, e in self._stages.items() if not e.enabled
        ]
        if disabled:
            lines.append(f"\nDisabled: {', '.join(disabled)}")

        return "\n".join(lines)
