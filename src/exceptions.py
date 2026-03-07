"""Shared exception types for the march-madness-forecaster pipeline."""


class LeakageError(RuntimeError):
    """Raised when temporal or data leakage is detected in the pipeline.

    In strict leakage mode, any leakage check failure raises this error
    instead of logging a warning, halting the pipeline immediately.
    """

    pass


class DataFreshnessError(RuntimeError):
    """Raised when required data sources are stale or missing."""

    pass


class PreRunValidationError(RuntimeError):
    """Raised when pre-run validation checks fail."""

    pass
