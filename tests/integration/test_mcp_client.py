"""Tests for the MCP client manager."""

from __future__ import annotations

import pytest

from mcp_client.client import MCPClientManager


class TestMCPClientManagerInit:
    """Verify MCPClientManager initial state."""

    def test_default_state(self) -> None:
        mgr = MCPClientManager()
        assert mgr.connected is False
        assert mgr._agent_configs == {}
        assert mgr._client is None
        assert mgr._tools_cache is None


class TestRegisterAgent:
    """Verify register_agent stores configs correctly."""

    def test_stores_single_config(self) -> None:
        mgr = MCPClientManager()
        cfg = {"transport": "stdio", "command": "python", "args": ["-m", "server"]}
        mgr.register_agent("my_agent", cfg)
        assert "my_agent" in mgr._agent_configs
        assert mgr._agent_configs["my_agent"] == cfg

    def test_stores_multiple_configs(self) -> None:
        mgr = MCPClientManager()
        mgr.register_agent("agent_a", {"transport": "stdio", "command": "a"})
        mgr.register_agent("agent_b", {"transport": "sse", "command": "b"})
        assert len(mgr._agent_configs) == 2
        assert mgr._agent_configs["agent_a"]["command"] == "a"
        assert mgr._agent_configs["agent_b"]["transport"] == "sse"

    def test_overwrite_existing(self) -> None:
        mgr = MCPClientManager()
        mgr.register_agent("x", {"old": True})
        mgr.register_agent("x", {"new": True})
        assert mgr._agent_configs["x"] == {"new": True}

    def test_config_includes_env(self) -> None:
        mgr = MCPClientManager()
        cfg = {
            "transport": "stdio",
            "command": "node",
            "args": ["server.js"],
            "env": {"API_KEY": "secret"},
        }
        mgr.register_agent("env_agent", cfg)
        assert mgr._agent_configs["env_agent"]["env"]["API_KEY"] == "secret"


class TestConnectWithoutConfig:
    """Verify connect raises when no configs registered."""

    @pytest.mark.asyncio
    async def test_raises_runtime_error(self) -> None:
        mgr = MCPClientManager()
        with pytest.raises(RuntimeError, match="No agent configs registered"):
            await mgr.connect()


class TestGetToolsNotConnected:
    """Verify get_tools raises when not connected."""

    @pytest.mark.asyncio
    async def test_raises_runtime_error(self) -> None:
        mgr = MCPClientManager()
        with pytest.raises(RuntimeError, match="Not connected"):
            await mgr.get_tools()


class TestInvokeToolNotConnected:
    """Verify invoke_tool raises when not connected."""

    @pytest.mark.asyncio
    async def test_raises_runtime_error(self) -> None:
        mgr = MCPClientManager()
        with pytest.raises(RuntimeError, match="Not connected"):
            await mgr.invoke_tool("any_tool", {})


class TestClose:
    """Verify close resets state."""

    @pytest.mark.asyncio
    async def test_close_resets_state(self) -> None:
        mgr = MCPClientManager()
        mgr._connected = True
        mgr._tools_cache = ["dummy"]
        await mgr.close()
        assert mgr.connected is False
        assert mgr._client is None
        assert mgr._tools_cache is None
