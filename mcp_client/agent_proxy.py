"""Agent proxy for wrapping MCP servers and invoking tools.

Provides a proxy class that wraps an agent's MCP server,
handles connection lifecycle, and provides convenient methods
to invoke tools on the agent.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)


class AgentProxy:
    """Proxy for interacting with an agent's MCP server.

    Wraps an agent's MCP server and provides methods to invoke tools,
    handle connection lifecycle, and manage agent communication.

    Usage:
        proxy = AgentProxy("invoice", {
            "transport": "stdio",
            "command": "python",
            "args": ["-m", "agents.invoice_agent"]
        })
        await proxy.connect()

        # Get available tools
        tools = await proxy.get_tools()

        # Invoke specific tool
        result = await proxy.invoke_tool("extract", {"document_content": "..."})

        # Invoke extract tool (convenience method)
        result = await proxy.invoke_extract("document content here")

        # Get agent status
        status = await proxy.get_status()

        # Close connection
        await proxy.close()
    """

    def __init__(self, agent_name: str, config: Dict[str, Any]) -> None:
        """Initialize the agent proxy.

        Args:
            agent_name: Name/identifier of the agent.
            config: MCP server configuration with keys:
                - transport: Transport type (e.g., "stdio", "sse")
                - command: Command to run (for stdio transport)
                - args: Command arguments (for stdio transport)
                - env: Environment variables (optional)
                - url: Server URL (for sse transport)
        """
        self.agent_name = agent_name
        self.config = config
        self._client: Optional[MultiServerMCPClient] = None
        self._connected = False
        self._tools_cache: Optional[List[BaseTool]] = None

    @property
    def connected(self) -> bool:
        """Check if proxy is connected to the agent."""
        return self._connected

    async def connect(self) -> None:
        """Connect to the agent's MCP server.

        Raises:
            RuntimeError: If connection fails.
        """
        try:
            # Create MultiServerMCPClient with this agent's config
            self._client = MultiServerMCPClient({self.agent_name: self.config})
            self._connected = True
            self._tools_cache = None  # Invalidate cache

            logger.info("Connected to agent '%s'", self.agent_name)

        except Exception as e:
            self._connected = False
            logger.error("Failed to connect to agent '%s': %s", self.agent_name, e)
            raise RuntimeError(
                f"Failed to connect to agent '{self.agent_name}': {e}"
            ) from e

    async def get_tools(self) -> List[BaseTool]:
        """Get all tools from this agent.

        Returns:
            List of LangChain BaseTool instances.

        Raises:
            RuntimeError: If not connected.
        """
        if not self._connected or self._client is None:
            raise RuntimeError(
                f"Not connected to agent '{self.agent_name}'. Call connect() first."
            )

        if self._tools_cache is None:
            self._tools_cache = await self._client.get_tools()

        return list(self._tools_cache)

    async def get_tool_names(self) -> List[str]:
        """Get names of all available tools.

        Returns:
            List of tool names.
        """
        tools = await self.get_tools()
        return [tool.name for tool in tools]

    async def has_tool(self, tool_name: str) -> bool:
        """Check if a specific tool is available.

        Args:
            tool_name: Name of the tool to check.

        Returns:
            True if tool exists, False otherwise.
        """
        tool_names = await self.get_tool_names()
        return tool_name in tool_names

    async def invoke_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> Any:
        """Invoke a specific tool by name.

        Args:
            tool_name: Exact name of the tool to invoke.
            arguments: Arguments to pass to the tool.

        Returns:
            Result from the tool invocation.

        Raises:
            ValueError: If tool not found.
            RuntimeError: If not connected.
        """
        tools = await self.get_tools()
        tool = next((t for t in tools if t.name == tool_name), None)

        if not tool:
            available_tools = [t.name for t in tools]
            raise ValueError(
                f"Tool '{tool_name}' not found on agent '{self.agent_name}'. "
                f"Available: {available_tools}"
            )

        return await tool.ainvoke(arguments)

    async def invoke_extract(
        self,
        document_content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Convenience method to invoke the extract tool.

        Args:
            document_content: Raw text content of the document.
            metadata: Optional dict with document metadata.

        Returns:
            Dict with extraction results (fields, confidence, model_used, cost, requires_review).

        Raises:
            ValueError: If extract tool not found.
            RuntimeError: If not connected.
        """
        arguments = {
            "document_content": document_content,
            "metadata": metadata or {}
        }

        return await self.invoke_tool("extract", arguments)

    async def invoke_custom(
        self,
        tool_name: str,
        **kwargs: Any
    ) -> Any:
        """Invoke a tool with keyword arguments.

        Args:
            tool_name: Name of the tool to invoke.
            **kwargs: Keyword arguments to pass to the tool.

        Returns:
            Result from the tool invocation.
        """
        return await self.invoke_tool(tool_name, kwargs)

    async def get_status(self) -> Dict[str, Any]:
        """Get status information about this agent.

        Returns:
            Dict with agent status information.
        """
        tools = await self.get_tools() if self._connected else []

        return {
            "agent_name": self.agent_name,
            "connected": self._connected,
            "tool_count": len(tools),
            "tools": [tool.name for tool in tools],
            "config": {
                "transport": self.config.get("transport"),
                "command": self.config.get("command"),
            }
        }

    async def close(self) -> None:
        """Close the connection to the agent's MCP server."""
        if self._client is not None:
            try:
                await self._client.__aexit__(None, None, None)
            except Exception as e:
                logger.warning(
                    "Error closing connection to agent '%s': %s",
                    self.agent_name,
                    e
                )

        self._connected = False
        self._tools_cache = None
        self._client = None
        logger.info("Closed connection to agent '%s'", self.agent_name)

    async def __aenter__(self) -> "AgentProxy":
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"AgentProxy(agent_name='{self.agent_name}', "
            f"connected={self._connected}, "
            f"config={self.config})"
        )