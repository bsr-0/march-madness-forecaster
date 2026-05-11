"""Pipeline exports.

Keep package imports light so submodules like ``src.pipeline.womens`` do not
eagerly pull in the full tournament pipeline and its optional heavy
dependencies.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["TournamentPipeline", "ForecastConfig", "run_pipeline_to_file"]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(".tournament_pipeline", __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
