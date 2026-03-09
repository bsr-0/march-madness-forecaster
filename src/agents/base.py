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


class AgentHealthMonitor:
    """Tracks agent health via last-seen timestamps and status checks."""

    def __init__(self, timeout_seconds: float = 300.0) -> None:
        self._last_seen: Dict[str, str] = {}
        self._timeout_seconds = timeout_seconds

    def record_heartbeat(self, agent_name: str) -> None:
        """Record that an agent was last seen now."""
        self._last_seen[agent_name] = datetime.now(timezone.utc).isoformat()

    def check_health(self, agents: Dict[str, "Agent"]) -> Dict[str, str]:
        """Check health of all agents. Returns agent_name -> status string."""
        health: Dict[str, str] = {}
        now = datetime.now(timezone.utc)
        for name, agent in agents.items():
            agent_status = agent.status().value
            last_seen = self._last_seen.get(name)
            if last_seen:
                try:
                    last_dt = datetime.fromisoformat(last_seen)
                    elapsed = (now - last_dt).total_seconds()
                    if elapsed > self._timeout_seconds:
                        agent_status = f"{agent_status} (stale: {elapsed:.0f}s)"
                except (ValueError, TypeError):
                    pass
            health[name] = agent_status
        return health
