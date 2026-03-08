"""Agent registry and message bus for inter-agent communication."""

from typing import Any, Dict, List

from .base import Agent, AgentMessage


class AgentRegistry:
    """Registry that tracks all available agents by name."""

    def __init__(self) -> None:
        self._agents: Dict[str, Agent] = {}

    def register(self, agent: Agent) -> None:
        """Register an agent. Overwrites if name already exists."""
        self._agents[agent.name] = agent

    def get(self, name: str) -> Agent:
        """Retrieve agent by name. Raises KeyError if not found."""
        if name not in self._agents:
            raise KeyError(f"Agent not found: {name}")
        return self._agents[name]

    def list_agents(self) -> List[str]:
        """Return sorted list of registered agent names."""
        return sorted(self._agents.keys())

    def find_by_capability(self, capability: str) -> List[Agent]:
        """Return all agents that advertise a given capability."""
        return [
            agent
            for agent in self._agents.values()
            if capability in agent.capabilities()
        ]


class MessageBus:
    """Routes messages between agents and records history."""

    def __init__(self, registry: AgentRegistry) -> None:
        self._registry = registry
        self._history: List[AgentMessage] = []

    def send(self, message: AgentMessage, ctx: Any = None) -> AgentMessage:
        """Send message to a named agent and return its response."""
        self._history.append(message)
        agent = self._registry.get(message.recipient)
        response = agent.handle(message, ctx)
        self._history.append(response)
        return response

    def broadcast(
        self, message: AgentMessage, ctx: Any = None
    ) -> List[AgentMessage]:
        """Send message to all registered agents and collect responses."""
        responses: List[AgentMessage] = []
        for agent_name in self._registry.list_agents():
            directed = AgentMessage(
                sender=message.sender,
                recipient=agent_name,
                msg_type=message.msg_type,
                payload=message.payload,
                correlation_id=message.correlation_id,
            )
            self._history.append(directed)
            agent = self._registry.get(agent_name)
            response = agent.handle(directed, ctx)
            self._history.append(response)
            responses.append(response)
        return responses

    def history(self) -> List[AgentMessage]:
        """Return full message history."""
        return list(self._history)
