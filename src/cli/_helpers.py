"""Shared CLI helpers used by multiple command modules.

The ForecastConfig builders, manifest resolvers and production guards that
used to live here served the ML pipeline, removed on 2026-09-11 (audit H10).
"""

import datetime


def _default_year() -> int:
    """Current calendar year — used as the default --year across CLI commands."""
    return datetime.date.today().year
