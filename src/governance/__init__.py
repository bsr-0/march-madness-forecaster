"""Production governance: compliance gate, audit trail, frozen-config validation."""

from .compliance import ComplianceGate, ComplianceCheck, CheckpointResult
from .audit_trail import GovernanceAuditLog
from .gate import GovernanceGate
from .production_validator import ProductionValidationError, validate_production_2026

# Backward-compat aliases
ComplianceRunner = ComplianceGate
ComplianceCheckpoint = ComplianceCheck

__all__ = [
    "ComplianceGate",
    "ComplianceCheck",
    "CheckpointResult",
    "ComplianceRunner",
    "ComplianceCheckpoint",
    "GovernanceAuditLog",
    "GovernanceGate",
    "ProductionValidationError",
    "validate_production_2026",
]
