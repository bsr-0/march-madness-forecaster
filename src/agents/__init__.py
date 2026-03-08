"""Multi-agent architecture for pipeline orchestration."""

from .base import Agent, AgentMessage, AgentStatus
from .registry import AgentRegistry, MessageBus
from .orchestrator import ResearchOrchestrator

__all__ = [
    "Agent",
    "AgentMessage",
    "AgentStatus",
    "AgentRegistry",
    "MessageBus",
    "ResearchOrchestrator",
]
