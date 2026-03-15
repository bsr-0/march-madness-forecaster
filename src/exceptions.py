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


class IntegrityError(RuntimeError):
    """Raised when model calibration or mathematical integrity checks fail.

    In production mode, a Brier Score exceeding the rejection threshold
    triggers this error, halting the pipeline.  In experimental mode,
    the same condition is logged as a warning instead.
    """

    pass


class GovernanceApprovalRequired(RuntimeError):
    """Raised when a pipeline action requires human approval.

    The request_id attribute can be used to approve or deny the request
    via the governance CLI commands.
    """

    def __init__(self, message: str, request_id: str = ""):
        super().__init__(message)
        self.request_id = request_id
