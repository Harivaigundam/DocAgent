"""Unit tests for MCPClientManager (MCP coordinator)."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from mcp_client.client import MCPClientManager


class TestMCPClientManagerInit:
    def test_initial_state(self):
        mgr = MCPClientManager()
        assert mgr.connected is False
        assert mgr._agent_configs == {}
        assert mgr._client is None
        assert mgr._tools_cache is None

    def test_connected_property(self):
        mgr = MCPClientManager()
        assert mgr.connected is False
        mgr._connected = True
        assert mgr.connected is True


class TestAgentRegistration:
    def test_register_single_agent(self):
        mgr = MCPClientManager()
        cfg = {"transport": "stdio", "command": "python", "args": ["-m", "server"]}
        mgr.register_agent("invoice_agent", cfg)
        assert "invoice_agent" in mgr._agent_configs
        assert mgr._agent_configs["invoice_agent"] == cfg

    def test_register_multiple_agents(self):
        mgr = MCPClientManager()
        mgr.register_agent("agent_a", {"transport": "stdio", "command": "a"})
        mgr.register_agent("agent_b", {"transport": "sse", "command": "b"})
        mgr.register_agent("agent_c", {"transport": "stdio", "command": "c"})
        assert len(mgr._agent_configs) == 3

    def test_register_overwrites_existing(self):
        mgr = MCPClientManager()
        mgr.register_agent("x", {"old_config": True})
        mgr.register_agent("x", {"new_config": True})
        assert mgr._agent_configs["x"] == {"new_config": True}

    def test_register_preserves_env(self):
        mgr = MCPClientManager()
        cfg = {
            "transport": "stdio",
            "command": "node",
            "args": ["server.js"],
            "env": {"API_KEY": "secret", "DB_URL": "localhost"},
        }
        mgr.register_agent("env_agent", cfg)
        assert mgr._agent_configs["env_agent"]["env"]["API_KEY"] == "secret"
        assert mgr._agent_configs["env_agent"]["env"]["DB_URL"] == "localhost"

    def test_register_all_doc_agents(self):
        mgr = MCPClientManager()
        agent_configs = {
            "invoice": {"transport": "stdio", "command": "python", "args": ["-m", "agents.invoice_agent"]},
            "receipt": {"transport": "stdio", "command": "python", "args": ["-m", "agents.receipt_agent"]},
            "contract": {"transport": "stdio", "command": "python", "args": ["-m", "agents.contract_agent"]},
            "report": {"transport": "stdio", "command": "python", "args": ["-m", "agents.report_agent"]},
        }
        for name, cfg in agent_configs.items():
            mgr.register_agent(name, cfg)
        assert len(mgr._agent_configs) == 4


class TestConnectionLifecycle:
    @pytest.mark.asyncio
    async def test_connect_without_agents_raises(self):
        mgr = MCPClientManager()
        with pytest.raises(RuntimeError, match="No agent configs registered"):
            await mgr.connect()

    @pytest.mark.asyncio
    @patch("mcp_client.client.MultiServerMCPClient")
    async def test_connect_sets_connected_true(self, MockMCPClient):
        mgr = MCPClientManager()
        mgr.register_agent("test", {"transport": "stdio", "command": "test"})
        await mgr.connect()
        assert mgr.connected is True
        MockMCPClient.assert_called_once_with(mgr._agent_configs)

    @pytest.mark.asyncio
    async def test_close_resets_state(self):
        mgr = MCPClientManager()
        mgr._connected = True
        mgr._tools_cache = ["dummy"]
        mgr._client = MagicMock()
        mgr._client.__aexit__ = AsyncMock()
        await mgr.close()
        assert mgr.connected is False
        assert mgr._client is None
        assert mgr._tools_cache is None

    @pytest.mark.asyncio
    async def test_close_when_no_client(self):
        mgr = MCPClientManager()
        await mgr.close()
        assert mgr.connected is False


class TestToolDiscovery:
    @pytest.mark.asyncio
    async def test_get_tools_not_connected_raises(self):
        mgr = MCPClientManager()
        with pytest.raises(RuntimeError, match="Not connected"):
            await mgr.get_tools()

    @pytest.mark.asyncio
    async def test_get_tools_returns_cached(self):
        mgr = MCPClientManager()
        mgr._connected = True
        mock_tool = MagicMock()
        mock_tool.name = "extract_invoice"
        mgr._client = MagicMock()
        mgr._client.get_tools = AsyncMock(return_value=[mock_tool])

        tools = await mgr.get_tools()
        assert len(tools) == 1
        assert mgr._tools_cache is not None

        # Second call should use cache
        tools2 = await mgr.get_tools()
        assert tools2 == tools

    @pytest.mark.asyncio
    async def test_get_tools_filters_by_agent_name(self):
        mgr = MCPClientManager()
        mgr._connected = True
        tool1 = MagicMock()
        tool1.name = "extract_invoice"
        tool2 = MagicMock()
        tool2.name = "extract_receipt"
        tool3 = MagicMock()
        tool3.name = "validate_extraction"
        mgr._client = MagicMock()
        mgr._client.get_tools = AsyncMock(return_value=[tool1, tool2, tool3])

        tools = await mgr.get_tools(agent_name="invoice")
        assert len(tools) == 1
        assert tools[0].name == "extract_invoice"

    @pytest.mark.asyncio
    async def test_get_tools_case_insensitive_filter(self):
        mgr = MCPClientManager()
        mgr._connected = True
        tool = MagicMock()
        tool.name = "InvoiceExtract"
        mgr._client = MagicMock()
        mgr._client.get_tools = AsyncMock(return_value=[tool])

        tools = await mgr.get_tools(agent_name="invoice")
        assert len(tools) == 1

    @pytest.mark.asyncio
    async def test_get_tools_no_match_returns_empty(self):
        mgr = MCPClientManager()
        mgr._connected = True
        tool = MagicMock()
        tool.name = "extract_invoice"
        mgr._client = MagicMock()
        mgr._client.get_tools = AsyncMock(return_value=[tool])

        tools = await mgr.get_tools(agent_name="nonexistent")
        assert tools == []


class TestToolInvocation:
    @pytest.mark.asyncio
    async def test_invoke_tool_not_connected_raises(self):
        mgr = MCPClientManager()
        with pytest.raises(RuntimeError, match="Not connected"):
            await mgr.invoke_tool("any_tool", {})

    @pytest.mark.asyncio
    async def test_invoke_tool_not_found_raises(self):
        mgr = MCPClientManager()
        mgr._connected = True
        tool = MagicMock()
        tool.name = "extract_invoice"
        mgr._client = MagicMock()
        mgr._client.get_tools = AsyncMock(return_value=[tool])

        with pytest.raises(ValueError, match="Tool 'nonexistent' not found"):
            await mgr.invoke_tool("nonexistent", {})

    @pytest.mark.asyncio
    async def test_invoke_tool_success(self):
        mgr = MCPClientManager()
        mgr._connected = True
        mock_tool = MagicMock()
        mock_tool.name = "extract_invoice"
        mock_tool.ainvoke = AsyncMock(return_value={"result": "success"})
        mgr._client = MagicMock()
        mgr._client.get_tools = AsyncMock(return_value=[mock_tool])

        result = await mgr.invoke_tool("extract_invoice", {"content": "test"})
        assert result == {"result": "success"}
        mock_tool.ainvoke.assert_called_once_with({"content": "test"})

    @pytest.mark.asyncio
    async def test_invoke_tool_error_propagates(self):
        mgr = MCPClientManager()
        mgr._connected = True
        mock_tool = MagicMock()
        mock_tool.name = "failing_tool"
        mock_tool.ainvoke = AsyncMock(side_effect=RuntimeError("LLM timeout"))
        mgr._client = MagicMock()
        mgr._client.get_tools = AsyncMock(return_value=[mock_tool])

        with pytest.raises(RuntimeError, match="LLM timeout"):
            await mgr.invoke_tool("failing_tool", {})


class TestConnectionIntegration:
    """Test realistic multi-agent connection scenarios."""

    @pytest.mark.asyncio
    @patch("mcp_client.client.MultiServerMCPClient")
    async def test_connect_then_get_tools(self, MockMCPClient):
        mock_client = MagicMock()
        mock_client.get_tools = AsyncMock(return_value=[
            MagicMock(name="extract"),
            MagicMock(name="validate"),
        ])
        MockMCPClient.return_value = mock_client

        mgr = MCPClientManager()
        mgr.register_agent("agent1", {"transport": "stdio", "command": "test"})
        await mgr.connect()

        tools = await mgr.get_tools()
        assert len(tools) == 2

    @pytest.mark.asyncio
    async def test_connect_register_connect_cycle(self):
        mgr = MCPClientManager()
        mgr.register_agent("agent1", {"transport": "stdio", "command": "a"})
        assert len(mgr._agent_configs) == 1

        # Can register more agents before connecting
        mgr.register_agent("agent2", {"transport": "sse", "command": "b"})
        assert len(mgr._agent_configs) == 2

    @pytest.mark.asyncio
    async def test_close_allows_reconnect(self):
        mgr = MCPClientManager()
        mgr.register_agent("agent1", {"transport": "stdio", "command": "test"})

        # First connection
        mgr._connected = True
        mgr._client = MagicMock()
        mgr._client.__aexit__ = AsyncMock()
        await mgr.close()
        assert mgr.connected is False
        assert len(mgr._agent_configs) == 1
