"""Base agent protocol and message types for multi-agent architecture."""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Protocol, runtime_checkable


class AgentStatus(Enum):
    """Possible states for an agent."""

    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class AgentMessage:
    """Message passed between agents via the MessageBus."""

    sender: str
    recipient: str  # or "*" for broadcast
    msg_type: str  # "request", "result", "error", "status"
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])


@runtime_checkable
class Agent(Protocol):
    """Protocol that all agents must satisfy."""

    @property
    def name(self) -> str:
        """Unique agent identifier."""
        ...

    def capabilities(self) -> List[str]:
        """Return list of actions this agent can perform."""
        ...

    def handle(self, message: AgentMessage, ctx: Any) -> AgentMessage:
        """Process an incoming message and return a response."""
        ...

    def status(self) -> AgentStatus:
        """Return current agent status."""
        ...
