"""Approval gates for high-stakes pipeline changes."""

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ApprovalRequest:
    request_id: str = ""
    action: str = ""  # "promote_model", "modify_config", "override_threshold", "submit_predictions"
    requester: str = ""
    description: str = ""
    risk_level: str = "medium"  # "low", "medium", "high", "critical"
    evidence: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""
    status: str = "pending"  # "pending", "approved", "rejected", "expired"
    reviewer: Optional[str] = None
    review_timestamp: Optional[str] = None
    review_notes: Optional[str] = None


class ApprovalGate:
    def __init__(self, ledger_path: str = "data/governance_ledger.jsonl"):
        self.ledger_path = Path(ledger_path)

    def request_approval(self, request: ApprovalRequest) -> str:
        """Submit approval request. Auto-generates request_id and timestamp. Returns request_id."""
        if not request.request_id:
            request.request_id = str(uuid.uuid4())[:8]
        if not request.timestamp:
            request.timestamp = datetime.now(timezone.utc).isoformat()
        request.status = "pending"
        self._append(request)
        return request.request_id

    def approve(self, request_id: str, reviewer: str, notes: str = "") -> bool:
        """Approve a pending request. Returns True if found and approved."""
        return self._update_status(request_id, "approved", reviewer, notes)

    def reject(self, request_id: str, reviewer: str, notes: str = "") -> bool:
        """Reject a pending request."""
        return self._update_status(request_id, "rejected", reviewer, notes)

    def check_approval(self, request_id: str) -> Optional[ApprovalRequest]:
        """Look up a request by ID."""
        for req in self._load_all():
            if req.request_id == request_id:
                return req
        return None

    def pending_requests(self) -> List[ApprovalRequest]:
        """Return all pending requests."""
        return [r for r in self._load_all() if r.status == "pending"]

    def _update_status(self, request_id: str, new_status: str, reviewer: str, notes: str) -> bool:
        """Update status by rewriting the latest entry for this request_id."""
        all_requests = self._load_all()
        found = False
        for req in all_requests:
            if req.request_id == request_id and req.status == "pending":
                req.status = new_status
                req.reviewer = reviewer
                req.review_timestamp = datetime.now(timezone.utc).isoformat()
                req.review_notes = notes
                found = True
        if found:
            # Append the updated record (append-only pattern -- old entries preserved)
            updated = [r for r in all_requests if r.request_id == request_id]
            if updated:
                self._append(updated[-1])
        return found

    def _append(self, request: ApprovalRequest) -> None:
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.ledger_path, "a") as f:
            f.write(json.dumps(asdict(request)) + "\n")

    def _load_all(self) -> List[ApprovalRequest]:
        if not self.ledger_path.exists():
            return []
        requests = []
        seen: Dict[str, int] = {}  # request_id -> last index
        with open(self.ledger_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    req = ApprovalRequest(**{k: v for k, v in data.items()
                                           if k in ApprovalRequest.__dataclass_fields__})
                    rid = req.request_id
                    if rid in seen:
                        requests[seen[rid]] = req  # replace with latest
                    else:
                        seen[rid] = len(requests)
                        requests.append(req)
                except (json.JSONDecodeError, TypeError):
                    continue
        return requests
