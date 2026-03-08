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


class ComputeBudgetExceeded(RuntimeError):
    """Raised when pipeline execution exceeds compute budget limits.

    Only raised in strict mode; otherwise budget violations are logged
    as warnings.
    """

    pass


class DataRequirementError(RuntimeError):
    """Raised when a required data artifact is missing or invalid."""

    pass
