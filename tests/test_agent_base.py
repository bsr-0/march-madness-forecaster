"""Tests for agent base types: AgentMessage, AgentStatus, Agent protocol."""

from src.agents.base import Agent, AgentMessage, AgentStatus


class TestAgentMessageCreation:
    """Test AgentMessage field population."""

    def test_agent_message_creation(self):
        msg = AgentMessage(
            sender="test_sender",
            recipient="test_recipient",
            msg_type="request",
            payload={"action": "load_data", "key": "value"},
        )
        assert msg.sender == "test_sender"
        assert msg.recipient == "test_recipient"
        assert msg.msg_type == "request"
        assert msg.payload == {"action": "load_data", "key": "value"}

    def test_agent_message_defaults(self):
        msg = AgentMessage(
            sender="a",
            recipient="b",
            msg_type="status",
        )
        # timestamp should be auto-populated as an ISO string
        assert msg.timestamp is not None
        assert "T" in msg.timestamp  # ISO format contains T separator
        # correlation_id should be auto-populated as a short uuid
        assert msg.correlation_id is not None
        assert len(msg.correlation_id) == 8
        # payload defaults to empty dict
        assert msg.payload == {}

    def test_agent_message_custom_correlation_id(self):
        msg = AgentMessage(
            sender="a",
            recipient="b",
            msg_type="result",
            correlation_id="custom01",
        )
        assert msg.correlation_id == "custom01"


class TestAgentStatusEnum:
    """Test AgentStatus enum values."""

    def test_agent_status_enum(self):
        assert AgentStatus.IDLE.value == "idle"
        assert AgentStatus.RUNNING.value == "running"
        assert AgentStatus.COMPLETED.value == "completed"
        assert AgentStatus.FAILED.value == "failed"

    def test_agent_status_members(self):
        members = [s.name for s in AgentStatus]
        assert set(members) == {"IDLE", "RUNNING", "COMPLETED", "FAILED"}


class TestAgentProtocol:
    """Test that the Agent protocol is checkable at runtime."""

    def test_agent_protocol_check(self):
        """A class implementing all required methods satisfies Agent."""

        class MockAgent:
            @property
            def name(self) -> str:
                return "mock"

            def capabilities(self):
                return ["test"]

            def handle(self, message, ctx):
                return message

            def status(self):
                return AgentStatus.IDLE

        agent = MockAgent()
        assert isinstance(agent, Agent)

    def test_non_agent_fails_check(self):
        """A class missing methods does not satisfy Agent."""

        class NotAnAgent:
            pass

        assert not isinstance(NotAnAgent(), Agent)
