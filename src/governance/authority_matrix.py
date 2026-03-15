"""Decision authority matrix for pipeline actions."""

import fnmatch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AuthorityRule:
    action_pattern: str  # glob pattern matching action types
    risk_level: str  # "low", "medium", "high", "critical"
    required_approver: str  # "auto", "operator", "lead"
    auto_approve_conditions: Optional[Dict[str, Any]] = None


DEFAULT_AUTHORITY_MATRIX: List[AuthorityRule] = [
    AuthorityRule("submit_predictions", "critical", "operator"),
    AuthorityRule("promote_model", "high", "operator",
                  auto_approve_conditions={"brier_improvement": 0.002, "ablation_pass": True, "admission_gate_pass": True}),
    AuthorityRule("modify_config.*", "medium", "auto",
                  auto_approve_conditions={"within_sensitivity_range": True}),
    AuthorityRule("override_threshold.*", "high", "operator"),
    AuthorityRule("retrain_model", "medium", "auto",
                  auto_approve_conditions={"loyo_regression": False}),
]


class AuthorityMatrix:
    def __init__(self, rules: Optional[List[AuthorityRule]] = None):
        self.rules = rules or list(DEFAULT_AUTHORITY_MATRIX)

    def get_rule(self, action: str) -> Optional[AuthorityRule]:
        """Find the first matching rule for an action."""
        for rule in self.rules:
            if fnmatch.fnmatch(action, rule.action_pattern):
                return rule
        return None

    def requires_approval(self, action: str, risk_level: str) -> bool:
        """Check if an action requires human approval."""
        rule = self.get_rule(action)
        if rule is None:
            # Default: high/critical require approval
            return risk_level in ("high", "critical")
        return rule.required_approver != "auto"

    def can_auto_approve(self, action: str, evidence: Dict[str, Any]) -> bool:
        """Check if evidence satisfies auto-approval conditions."""
        rule = self.get_rule(action)
        if rule is None:
            return False
        if rule.required_approver == "auto" and rule.auto_approve_conditions:
            return all(
                evidence.get(k) == v if not isinstance(v, (int, float))
                else evidence.get(k, float('inf')) <= v if isinstance(v, float) else evidence.get(k) == v
                for k, v in rule.auto_approve_conditions.items()
            )
        if rule.auto_approve_conditions is None and rule.required_approver == "auto":
            return True
        return False
