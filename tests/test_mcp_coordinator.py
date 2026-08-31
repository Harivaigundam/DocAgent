"""Tests for MCP Coordinator and AgentProxy."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from mcp_client import MCPCoordinator, AgentProxy
from mcp_client.coordinator import AgentStatus


class TestAgentProxy:
    """Tests for AgentProxy class."""

    def test_proxy_initialization(self):
        """Test proxy initialization with config."""
        config = {
            "transport": "stdio",
            "command": "python",
            "args": ["-m", "agents.invoice_agent"]
        }
        proxy = AgentProxy("invoice", config)

        assert proxy.agent_name == "invoice"
        assert proxy.config == config
        assert proxy.connected is False
        assert proxy._client is None

    def test_proxy_repr(self):
        """Test proxy string representation."""
        config = {"transport": "stdio", "command": "python", "args": []}
        proxy = AgentProxy("test_agent", config)

        repr_str = repr(proxy)
        assert "test_agent" in repr_str
        assert "connected=False" in repr_str

    @pytest.mark.asyncio
    async def test_proxy_connect_success(self):
        """Test successful proxy connection."""
        config = {
            "transport": "stdio",
            "command": "python",
            "args": ["-m", "agents.invoice_agent"]
        }
        proxy = AgentProxy("invoice", config)

        with patch("mcp_client.agent_proxy.MultiServerMCPClient") as mock_client:
            mock_client.return_value.__aenter__ = AsyncMock()
            mock_client.return_value.__aexit__ = AsyncMock()

            await proxy.connect()

            assert proxy.connected is True
            mock_client.assert_called_once_with({"invoice": config})

    @pytest.mark.asyncio
    async def test_proxy_connect_failure(self):
        """Test proxy connection failure."""
        config = {"transport": "stdio", "command": "invalid_command"}
        proxy = AgentProxy("invoice", config)

        with patch("mcp_client.agent_proxy.MultiServerMCPClient") as mock_client:
            mock_client.side_effect = Exception("Connection failed")

            with pytest.raises(RuntimeError) as exc_info:
                await proxy.connect()

            assert "Failed to connect to agent 'invoice'" in str(exc_info.value)
            assert proxy.connected is False

    @pytest.mark.asyncio
    async def test_proxy_close(self):
        """Test proxy connection closure."""
        config = {"transport": "stdio", "command": "python", "args": []}
        proxy = AgentProxy("invoice", config)

        with patch("mcp_client.agent_proxy.MultiServerMCPClient") as mock_client:
            mock_client.return_value.__aenter__ = AsyncMock()
            mock_client.return_value.__aexit__ = AsyncMock()

            await proxy.connect()
            assert proxy.connected is True

            await proxy.close()
            assert proxy.connected is False
            assert proxy._client is None

    @pytest.mark.asyncio
    async def test_proxy_get_tools_not_connected(self):
        """Test getting tools when not connected."""
        config = {"transport": "stdio", "command": "python", "args": []}
        proxy = AgentProxy("invoice", config)

        with pytest.raises(RuntimeError) as exc_info:
            await proxy.get_tools()

        assert "Not connected" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_proxy_invoke_tool_not_found(self):
        """Test invoking non-existent tool."""
        config = {"transport": "stdio", "command": "python", "args": []}
        proxy = AgentProxy("invoice", config)

        with patch("mcp_client.agent_proxy.MultiServerMCPClient") as mock_client:
            mock_client.return_value.__aenter__ = AsyncMock()
            mock_client.return_value.__aexit__ = AsyncMock()
            mock_client.return_value.get_tools = AsyncMock(return_value=[])

            await proxy.connect()

            with pytest.raises(ValueError) as exc_info:
                await proxy.invoke_tool("nonexistent_tool", {})

            assert "not found" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_proxy_context_manager(self):
        """Test proxy as async context manager."""
        config = {"transport": "stdio", "command": "python", "args": []}

        with patch("mcp_client.agent_proxy.MultiServerMCPClient") as mock_client:
            mock_client.return_value.__aenter__ = AsyncMock()
            mock_client.return_value.__aexit__ = AsyncMock()

            async with AgentProxy("invoice", config) as proxy:
                assert proxy.connected is True

            assert proxy.connected is False


class TestMCPCoordinator:
    """Tests for MCPCoordinator class."""

    def setup_method(self):
        """Reset singleton before each test."""
        MCPCoordinator.reset()

    def test_coordinator_singleton(self):
        """Test coordinator is a singleton."""
        coordinator1 = MCPCoordinator()
        coordinator2 = MCPCoordinator()

        assert coordinator1 is coordinator2

    def test_register_agent(self):
        """Test agent registration."""
        coordinator = MCPCoordinator()
        config = {"transport": "stdio", "command": "python", "args": []}

        coordinator.register_agent("invoice", config)

        assert "invoice" in coordinator.agent_names
        assert coordinator._agents["invoice"].config == config

    def test_register_agent_duplicate(self):
        """Test duplicate agent registration raises error."""
        coordinator = MCPCoordinator()
        config = {"transport": "stdio", "command": "python", "args": []}

        coordinator.register_agent("invoice", config)

        with pytest.raises(ValueError) as exc_info:
            coordinator.register_agent("invoice", config)

        assert "already registered" in str(exc_info.value)

    def test_unregister_agent(self):
        """Test agent unregistration."""
        coordinator = MCPCoordinator()
        config = {"transport": "stdio", "command": "python", "args": []}

        coordinator.register_agent("invoice", config)
        assert "invoice" in coordinator.agent_names

        coordinator.unregister_agent("invoice")
        assert "invoice" not in coordinator.agent_names

    def test_unregister_agent_not_found(self):
        """Test unregistering non-existent agent raises error."""
        coordinator = MCPCoordinator()

        with pytest.raises(ValueError) as exc_info:
            coordinator.unregister_agent("nonexistent")

        assert "not registered" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_connect_agent_success(self):
        """Test successful agent connection."""
        coordinator = MCPCoordinator()
        config = {"transport": "stdio", "command": "python", "args": []}

        coordinator.register_agent("invoice", config)

        with patch("mcp_client.agent_proxy.MultiServerMCPClient") as mock_client:
            mock_client.return_value.__aenter__ = AsyncMock()
            mock_client.return_value.__aexit__ = AsyncMock()
            mock_client.return_value.get_tools = AsyncMock(return_value=[])

            await coordinator.connect_agent("invoice")

            agent_status = coordinator.get_agent_status("invoice")
            assert agent_status["status"] == AgentStatus.CONNECTED.value
            assert agent_status["connected"] is True

    @pytest.mark.asyncio
    async def test_connect_agent_failure(self):
        """Test agent connection failure."""
        coordinator = MCPCoordinator()
        config = {"transport": "stdio", "command": "invalid"}

        coordinator.register_agent("invoice", config)

        with patch("mcp_client.agent_proxy.MultiServerMCPClient") as mock_client:
            mock_client.side_effect = Exception("Connection failed")

            with pytest.raises(RuntimeError) as exc_info:
                await coordinator.connect_agent("invoice")

            assert "Failed to connect" in str(exc_info.value)

            agent_status = coordinator.get_agent_status("invoice")
            assert agent_status["status"] == AgentStatus.ERROR.value

    @pytest.mark.asyncio
    async def test_connect_all_agents(self):
        """Test connecting to all agents."""
        coordinator = MCPCoordinator()
        config1 = {"transport": "stdio", "command": "python", "args": ["agent1"]}
        config2 = {"transport": "stdio", "command": "python", "args": ["agent2"]}

        coordinator.register_agent("agent1", config1)
        coordinator.register_agent("agent2", config2)

        with patch("mcp_client.agent_proxy.MultiServerMCPClient") as mock_client:
            mock_client.return_value.__aenter__ = AsyncMock()
            mock_client.return_value.__aexit__ = AsyncMock()
            mock_client.return_value.get_tools = AsyncMock(return_value=[])

            await coordinator.connect_all()

            assert coordinator.connected is True
            assert len(coordinator.agent_names) == 2

    @pytest.mark.asyncio
    async def test_connect_all_no_agents(self):
        """Test connecting when no agents registered."""
        coordinator = MCPCoordinator()

        with pytest.raises(RuntimeError) as exc_info:
            await coordinator.connect_all()

        assert "No agents registered" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_proxy(self):
        """Test getting agent proxy."""
        coordinator = MCPCoordinator()
        config = {"transport": "stdio", "command": "python", "args": []}

        coordinator.register_agent("invoice", config)

        with patch("mcp_client.agent_proxy.MultiServerMCPClient") as mock_client:
            mock_client.return_value.__aenter__ = AsyncMock()
            mock_client.return_value.__aexit__ = AsyncMock()
            mock_client.return_value.get_tools = AsyncMock(return_value=[])

            await coordinator.connect_agent("invoice")
            proxy = coordinator.get_proxy("invoice")

            assert proxy is not None
            assert proxy.agent_name == "invoice"

    def test_get_proxy_not_connected(self):
        """Test getting proxy when not connected."""
        coordinator = MCPCoordinator()
        config = {"transport": "stdio", "command": "python", "args": []}

        coordinator.register_agent("invoice", config)

        with pytest.raises(ValueError) as exc_info:
            coordinator.get_proxy("invoice")

        assert "not connected" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_all_tools(self):
        """Test getting all tools from all agents."""
        coordinator = MCPCoordinator()
        config = {"transport": "stdio", "command": "python", "args": []}

        coordinator.register_agent("invoice", config)

        with patch("mcp_client.agent_proxy.MultiServerMCPClient") as mock_client:
            mock_client.return_value.__aenter__ = AsyncMock()
            mock_client.return_value.__aexit__ = AsyncMock()
            mock_client.return_value.get_tools = AsyncMock(return_value=[])

            await coordinator.connect_all()
            tools = await coordinator.get_all_tools()

            assert isinstance(tools, list)

    def test_get_agent_status(self):
        """Test getting agent status."""
        coordinator = MCPCoordinator()
        config = {"transport": "stdio", "command": "python", "args": []}

        coordinator.register_agent("invoice", config)

        status = coordinator.get_agent_status("invoice")

        assert status["name"] == "invoice"
        assert status["status"] == AgentStatus.DISCONNECTED.value
        assert status["connected"] is False

    def test_get_all_statuses(self):
        """Test getting all agent statuses."""
        coordinator = MCPCoordinator()
        config1 = {"transport": "stdio", "command": "python", "args": []}
        config2 = {"transport": "stdio", "command": "python", "args": []}

        coordinator.register_agent("agent1", config1)
        coordinator.register_agent("agent2", config2)

        statuses = coordinator.get_all_statuses()

        assert "agent1" in statuses
        assert "agent2" in statuses
        assert len(statuses) == 2

    @pytest.mark.asyncio
    async def test_disconnect_all(self):
        """Test disconnecting from all agents."""
        coordinator = MCPCoordinator()
        config = {"transport": "stdio", "command": "python", "args": []}

        coordinator.register_agent("invoice", config)

        with patch("mcp_client.agent_proxy.MultiServerMCPClient") as mock_client:
            mock_client.return_value.__aenter__ = AsyncMock()
            mock_client.return_value.__aexit__ = AsyncMock()
            mock_client.return_value.get_tools = AsyncMock(return_value=[])

            await coordinator.connect_all()
            assert coordinator.connected is True

            await coordinator.disconnect_all()
            assert coordinator.connected is False

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test coordinator as async context manager."""
        config = {"transport": "stdio", "command": "python", "args": []}

        with patch("mcp_client.agent_proxy.MultiServerMCPClient") as mock_client:
            mock_client.return_value.__aenter__ = AsyncMock()
            mock_client.return_value.__aexit__ = AsyncMock()
            mock_client.return_value.get_tools = AsyncMock(return_value=[])

            coordinator = MCPCoordinator()
            coordinator.register_agent("invoice", config)

            async with coordinator:
                assert coordinator.connected is True

            assert coordinator.connected is False

    def test_reset_singleton(self):
        """Test resetting singleton instance."""
        coordinator1 = MCPCoordinator()
        MCPCoordinator.reset()
        coordinator2 = MCPCoordinator()

        assert coordinator1 is not coordinator2