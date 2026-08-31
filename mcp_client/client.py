from __future__ import annotations

import logging
from typing import Any

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)


class MCPClientManager:
    """Manages connections to MCP servers and provides tool access for agents."""

    def __init__(self) -> None:
        self._agent_configs: dict[str, dict[str, Any]] = {}
        self._client: MultiServerMCPClient | None = None
        self._connected = False
        self._tools_cache: list[BaseTool] | None = None

    @property
    def connected(self) -> bool:
        return self._connected

    def register_agent(self, name: str, config: dict[str, Any]) -> None:
        """Register an MCP server configuration for an agent.

        Args:
            name: Agent/server identifier used for tool filtering.
            config: Server config with keys: transport, command, args, env.
        """
        self._agent_configs[name] = config
        logger.info("Registered MCP server config for agent '%s'", name)

    async def connect(self) -> None:
        """Initialize the MultiServerMCPClient with all registered configs."""
        if not self._agent_configs:
            raise RuntimeError(
                "No agent configs registered. Call register_agent() first."
            )
        self._client = MultiServerMCPClient(self._agent_configs)
        self._connected = True
        self._tools_cache = None
        logger.info(
            "MCPClientManager connected with %d server(s)", len(self._agent_configs)
        )

    async def get_tools(self, agent_name: str | None = None) -> list[BaseTool]:
        """Retrieve tools, optionally filtered by agent name.

        Args:
            agent_name: If provided, only return tools whose name contains this string.

        Returns:
            List of LangChain BaseTool instances.
        """
        if not self._connected or self._client is None:
            raise RuntimeError("Not connected. Call connect() first.")

        if self._tools_cache is None:
            self._tools_cache = await self._client.get_tools()

        if agent_name:
            return [
                t for t in self._tools_cache if agent_name.lower() in t.name.lower()
            ]
        return list(self._tools_cache)

    async def invoke_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Invoke a specific tool by name with the given arguments.

        Args:
            tool_name: Exact name of the tool to invoke.
            arguments: Arguments to pass to the tool.

        Returns:
            Result from the tool invocation.

        Raises:
            ValueError: If the tool is not found.
            RuntimeError: If not connected.
        """
        tools = await self.get_tools()
        tool = next((t for t in tools if t.name == tool_name), None)
        if not tool:
            raise ValueError(
                f"Tool '{tool_name}' not found. "
                f"Available: {[t.name for t in tools]}"
            )
        return await tool.ainvoke(arguments)

    async def close(self) -> None:
        """Close the MCP client connection."""
        if self._client is not None:
            await self._client.__aexit__(None, None, None)
        self._connected = False
        self._tools_cache = None
        self._client = None
        logger.info("MCPClientManager closed")
