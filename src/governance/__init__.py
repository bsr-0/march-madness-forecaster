"""Human-in-the-loop governance: approval gates, authority matrix, compliance."""

from .approval_gate import ApprovalGate, ApprovalRequest
from .authority_matrix import AuthorityMatrix, AuthorityRule
from .compliance import ComplianceGate, ComplianceCheck, CheckpointResult
from .audit_trail import GovernanceAuditLog
from .gate import GovernanceGate

# Backward-compat aliases
ComplianceRunner = ComplianceGate
ComplianceCheckpoint = ComplianceCheck

__all__ = [
    "ApprovalGate", "ApprovalRequest",
    "AuthorityMatrix", "AuthorityRule",
    "ComplianceGate", "ComplianceCheck", "CheckpointResult",
    "ComplianceRunner", "ComplianceCheckpoint",
    "GovernanceAuditLog",
    "GovernanceGate",
]
