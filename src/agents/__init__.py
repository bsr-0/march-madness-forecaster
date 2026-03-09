"""Multi-agent architecture for pipeline orchestration.

Defines the agent protocol, message bus, and base class that enable
coordinated execution of pipeline stages by specialized agents.

Implements Agent Directive V7 S2 (multi-agent system architecture).

Agent roles:
    - DATA_SCOUT: Data loading, freshness checks, schema validation
    - FEATURE_ENGINEER: Feature engineering, ablation, selection
    - MODELER: Model training, calibration, simulation
    - AUDITOR: Robustness checks, leakage detection, veto authority
    - ORCHESTRATOR: Coordinates agent execution, conflict resolution

Usage:
    from src.agents import AgentRole, MessageBus, BaseAgent, AgentResult

    class MyAgent(BaseAgent):
        role = AgentRole.DATA_SCOUT
        name = "DataScout"

        def run(self, ctx, bus, **kwargs):
            # Do work, publish findings
            self.publish(bus, MessageType.DATA_READY, {"quality": "good"})
            return AgentResult(agent_role=self.role, success=True, output=data)
"""

from __future__ import annotations

import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class AgentRole(str, Enum):
    """Named agent roles in the multi-agent system (Directive V7 S2)."""

    DATA_SCOUT = "data_scout"
    FEATURE_ENGINEER = "feature_engineer"
    MODELER = "modeler"
    AUDITOR = "auditor"
    ORCHESTRATOR = "orchestrator"


class MessageType(str, Enum):
    """Types of inter-agent messages."""

    DATA_READY = "data_ready"
    FEATURES_READY = "features_ready"
    MODEL_READY = "model_ready"
    AUDIT_RESULT = "audit_result"
    VETO = "veto"
    CONFLICT = "conflict"
    DISSENT = "dissent"
    STATUS = "status"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class AgentMessage:
    """A message passed between agents via the MessageBus.

    Attributes:
        sender: Role of the agent that sent the message.
        msg_type: Type of message (determines routing/handling).
        payload: Arbitrary data carried by the message.
        recipient: Target agent role, or None for broadcast.
        message_id: Unique identifier.
        timestamp: ISO-8601 creation time.
    """

    sender: AgentRole
    msg_type: MessageType
    payload: Dict[str, Any] = field(default_factory=dict)
    recipient: Optional[AgentRole] = None
    message_id: str = ""
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.message_id:
            self.message_id = str(uuid.uuid4())[:8]
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["sender"] = self.sender.value
        d["msg_type"] = self.msg_type.value
        if self.recipient:
            d["recipient"] = self.recipient.value
        return d


@dataclass
class AgentResult:
    """Result returned by an agent after execution.

    Attributes:
        agent_role: Which agent produced this result.
        success: Whether the agent completed successfully.
        output: Typed output (LoadedData, TrainedModels, etc.).
        findings: Human-readable findings from the agent's work.
        messages_sent: Count of messages published during run.
        wall_seconds: Elapsed wall-clock time.
    """

    agent_role: AgentRole
    success: bool = True
    output: Any = None
    findings: List[str] = field(default_factory=list)
    messages_sent: int = 0
    wall_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Message Bus
# ---------------------------------------------------------------------------


class MessageBus:
    """In-process pub/sub message bus for inter-agent communication.

    Provides a simple, synchronous message passing mechanism. All messages
    are retained for audit trail purposes.
    """

    def __init__(self) -> None:
        self._messages: List[AgentMessage] = []

    def publish(self, message: AgentMessage) -> None:
        """Publish a message to the bus."""
        self._messages.append(message)
        logger.debug(
            "Agent %s published %s (id=%s)",
            message.sender.value,
            message.msg_type.value,
            message.message_id,
        )

    def get_messages(
        self,
        topic: Optional[MessageType] = None,
        recipient: Optional[AgentRole] = None,
        sender: Optional[AgentRole] = None,
    ) -> List[AgentMessage]:
        """Get messages filtered by topic, recipient, and/or sender.

        Messages with recipient=None (broadcasts) are included when
        filtering by recipient.
        """
        result = list(self._messages)
        if topic is not None:
            result = [m for m in result if m.msg_type == topic]
        if recipient is not None:
            result = [
                m for m in result
                if m.recipient is None or m.recipient == recipient
            ]
        if sender is not None:
            result = [m for m in result if m.sender == sender]
        return result

    def has_veto(self) -> bool:
        """Check if any VETO message has been published."""
        return any(m.msg_type == MessageType.VETO for m in self._messages)

    def all_messages(self) -> List[AgentMessage]:
        """Return all messages (full audit trail)."""
        return list(self._messages)

    def clear(self) -> None:
        """Clear all messages (primarily for testing)."""
        self._messages.clear()

    def __len__(self) -> int:
        return len(self._messages)


# ---------------------------------------------------------------------------
# Base Agent
# ---------------------------------------------------------------------------


class BaseAgent(ABC):
    """Abstract base class for all pipeline agents.

    Each agent has a named role, executes pipeline work via ``run()``,
    and communicates findings through the MessageBus.
    """

    @property
    @abstractmethod
    def role(self) -> AgentRole:
        """The agent's role in the multi-agent system."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable agent name."""
        ...

    @abstractmethod
    def run(self, ctx: Any, bus: MessageBus, **kwargs: Any) -> AgentResult:
        """Execute the agent's work.

        Args:
            ctx: PipelineContext with shared config/resources.
            bus: MessageBus for inter-agent communication.
            **kwargs: Upstream data from prior agents.

        Returns:
            AgentResult with typed output and findings.
        """
        ...

    def publish(
        self,
        bus: MessageBus,
        msg_type: MessageType,
        payload: Dict[str, Any],
        recipient: Optional[AgentRole] = None,
    ) -> None:
        """Convenience: create and publish a message."""
        msg = AgentMessage(
            sender=self.role,
            msg_type=msg_type,
            payload=payload,
            recipient=recipient,
        )
        bus.publish(msg)

    def log_finding(self, bus: MessageBus, finding: str) -> None:
        """Publish a STATUS message with a human-readable finding."""
        self.publish(bus, MessageType.STATUS, {"finding": finding})
