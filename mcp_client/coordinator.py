"""Central MCP coordinator for inter-agent communication.

Manages connections to all agent MCP servers, provides a unified tool registry,
enables agents to call tools on other agents, and tracks agent status and results.
"""

from __future__ import annotations

import logging
import asyncio
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.tools import BaseTool

from mcp_client.agent_proxy import AgentProxy

logger = logging.getLogger(__name__)


class AgentStatus(Enum):
    """Status of an agent in the coordinator."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


@dataclass
class AgentInfo:
    """Information about a registered agent."""
    name: str
    config: dict[str, Any]
    status: AgentStatus = AgentStatus.DISCONNECTED
    proxy: Optional[AgentProxy] = None
    tools: List[BaseTool] = field(default_factory=list)
    last_result: Optional[dict[str, Any]] = None
    last_error: Optional[str] = None
    connected_at: Optional[datetime] = None


class MCPCoordinator:
    """Central coordinator for MCP-based inter-agent communication.

    Singleton that manages connections to all agent MCP servers,
    provides a unified tool registry, and enables agents to call tools on each other.

    Usage:
        coordinator = MCPCoordinator()
        coordinator.register_agent("invoice", {"transport": "stdio", "command": "python", "args": ["-m", "agents.invoice_agent"]})
        coordinator.register_agent("receipt", {"transport": "stdio", "command": "python", "args": ["-m", "agents.receipt_agent"]})
        await coordinator.connect_all()

        # Create proxy for an agent
        invoice_proxy = coordinator.get_proxy("invoice")

        # Call tool on agent
        result = await invoice_proxy.invoke_extract("document content here")

        # Get all available tools
        all_tools = await coordinator.get_all_tools()

        # Get tools for specific agent
        invoice_tools = await coordinator.get_tools_for_agent("invoice")

        # Check agent status
        status = coordinator.get_agent_status("invoice")
    """

    _instance: Optional["MCPCoordinator"] = None
    _lock = asyncio.Lock()

    def __new__(cls) -> "MCPCoordinator":
        """Ensure singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """Initialize the coordinator."""
        if self._initialized:
            return

        self._agents: Dict[str, AgentInfo] = {}
        self._client: Optional[MultiServerMCPClient] = None
        self._connected = False
        self._tools_cache: Optional[List[BaseTool]] = None
        self._initialized = True
        logger.info("MCPCoordinator initialized")

    @property
    def connected(self) -> bool:
        """Check if coordinator is connected to any agents."""
        return self._connected

    @property
    def agent_names(self) -> List[str]:
        """Get list of registered agent names."""
        return list(self._agents.keys())

    def register_agent(self, name: str, config: dict[str, Any]) -> None:
        """Register an agent's MCP server configuration.

        Args:
            name: Agent identifier (e.g., "invoice", "receipt").
            config: Server config with keys: transport, command, args, env.

        Raises:
            ValueError: If agent name already registered.
        """
        if name in self._agents:
            raise ValueError(f"Agent '{name}' already registered")

        self._agents[name] = AgentInfo(
            name=name,
            config=config,
            status=AgentStatus.DISCONNECTED
        )
        self._tools_cache = None  # Invalidate cache
        logger.info("Registered agent '%s' with config: %s", name, config)

    def unregister_agent(self, name: str) -> None:
        """Unregister an agent.

        Args:
            name: Agent identifier to remove.

        Raises:
            ValueError: If agent not registered.
        """
        if name not in self._agents:
            raise ValueError(f"Agent '{name}' not registered")

        # Close proxy if connected
        agent_info = self._agents[name]
        if agent_info.proxy:
            asyncio.create_task(agent_info.proxy.close())

        del self._agents[name]
        self._tools_cache = None
        logger.info("Unregistered agent '%s'", name)

    async def connect_agent(self, name: str) -> None:
        """Connect to a specific agent's MCP server.

        Args:
            name: Agent identifier to connect to.

        Raises:
            ValueError: If agent not registered.
            RuntimeError: If connection fails.
        """
        if name not in self._agents:
            raise ValueError(f"Agent '{name}' not registered")

        agent_info = self._agents[name]
        agent_info.status = AgentStatus.CONNECTING

        try:
            # Create proxy for this agent
            proxy = AgentProxy(name, agent_info.config)
            await proxy.connect()

            agent_info.proxy = proxy
            agent_info.status = AgentStatus.CONNECTED
            agent_info.connected_at = datetime.now()
            agent_info.tools = await proxy.get_tools()
            agent_info.last_error = None
            self._tools_cache = None  # Invalidate cache

            logger.info("Connected to agent '%s' with %d tools", name, len(agent_info.tools))

        except Exception as e:
            agent_info.status = AgentStatus.ERROR
            agent_info.last_error = str(e)
            logger.error("Failed to connect to agent '%s': %s", name, e)
            raise RuntimeError(f"Failed to connect to agent '{name}': {e}") from e

    async def connect_all(self) -> None:
        """Connect to all registered agents.

        Attempts to connect to each agent, logging errors but continuing
        with other agents if one fails.
        """
        if not self._agents:
            raise RuntimeError("No agents registered. Call register_agent() first.")

        tasks = [self.connect_agent(name) for name in self._agents]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        connected_count = sum(
            1 for name, result in zip(self._agents.keys(), results)
            if result is None
        )

        self._connected = connected_count > 0

        logger.info(
            "Connected to %d/%d agents",
            connected_count,
            len(self._agents)
        )

        if connected_count == 0:
            raise RuntimeError("Failed to connect to any agents")

    async def disconnect_agent(self, name: str) -> None:
        """Disconnect from a specific agent.

        Args:
            name: Agent identifier to disconnect from.
        """
        if name not in self._agents:
            return

        agent_info = self._agents[name]
        if agent_info.proxy:
            await agent_info.proxy.close()
            agent_info.proxy = None

        agent_info.status = AgentStatus.DISCONNECTED
        agent_info.tools = []
        self._tools_cache = None

        logger.info("Disconnected from agent '%s'", name)

    async def disconnect_all(self) -> None:
        """Disconnect from all agents."""
        tasks = [self.disconnect_agent(name) for name in list(self._agents.keys())]
        await asyncio.gather(*tasks)
        self._connected = False
        logger.info("Disconnected from all agents")

    def get_proxy(self, name: str) -> AgentProxy:
        """Get a proxy for a specific agent.

        Args:
            name: Agent identifier.

        Returns:
            AgentProxy instance for the agent.

        Raises:
            ValueError: If agent not registered or not connected.
        """
        if name not in self._agents:
            raise ValueError(f"Agent '{name}' not registered")

        agent_info = self._agents[name]
        if agent_info.proxy is None:
            raise ValueError(f"Agent '{name}' not connected")

        return agent_info.proxy

    async def get_all_tools(self) -> List[BaseTool]:
        """Get all tools from all connected agents.

        Returns:
            List of all available tools.
        """
        if not self._connected:
            raise RuntimeError("Not connected. Call connect_all() first.")

        if self._tools_cache is not None:
            return self._tools_cache

        all_tools = []
        for agent_info in self._agents.values():
            if agent_info.proxy and agent_info.status == AgentStatus.CONNECTED:
                tools = await agent_info.proxy.get_tools()
                all_tools.extend(tools)

        self._tools_cache = all_tools
        return all_tools

    async def get_tools_for_agent(self, agent_name: str) -> List[BaseTool]:
        """Get tools for a specific agent.

        Args:
            agent_name: Agent identifier.

        Returns:
            List of tools for the specified agent.
        """
        if agent_name not in self._agents:
            raise ValueError(f"Agent '{agent_name}' not registered")

        agent_info = self._agents[agent_name]
        if agent_info.proxy is None:
            return []

        return await agent_info.proxy.get_tools()

    async def invoke_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Invoke a tool by name across all agents.

        Args:
            tool_name: Exact name of the tool to invoke.
            arguments: Arguments to pass to the tool.

        Returns:
            Result from the tool invocation.

        Raises:
            ValueError: If tool not found.
            RuntimeError: If not connected.
        """
        tools = await self.get_all_tools()
        tool = next((t for t in tools if t.name == tool_name), None)

        if not tool:
            available_tools = [t.name for t in tools]
            raise ValueError(
                f"Tool '{tool_name}' not found. Available: {available_tools}"
            )

        return await tool.ainvoke(arguments)

    async def invoke_agent_tool(
        self,
        agent_name: str,
        tool_name: str,
        arguments: dict[str, Any]
    ) -> Any:
        """Invoke a tool on a specific agent.

        Args:
            agent_name: Agent identifier.
            tool_name: Exact name of the tool to invoke.
            arguments: Arguments to pass to the tool.

        Returns:
            Result from the tool invocation.

        Raises:
            ValueError: If agent or tool not found.
        """
        proxy = self.get_proxy(agent_name)
        return await proxy.invoke_tool(tool_name, arguments)

    def get_agent_status(self, name: str) -> Dict[str, Any]:
        """Get status information for an agent.

        Args:
            name: Agent identifier.

        Returns:
            Dict with agent status information.
        """
        if name not in self._agents:
            raise ValueError(f"Agent '{name}' not registered")

        agent_info = self._agents[name]
        return {
            "name": agent_info.name,
            "status": agent_info.status.value,
            "connected": agent_info.status == AgentStatus.CONNECTED,
            "tool_count": len(agent_info.tools),
            "connected_at": agent_info.connected_at.isoformat() if agent_info.connected_at else None,
            "last_error": agent_info.last_error,
            "last_result": agent_info.last_result,
        }

    def get_all_statuses(self) -> Dict[str, Dict[str, Any]]:
        """Get status for all registered agents.

        Returns:
            Dict mapping agent names to their status information.
        """
        return {
            name: self.get_agent_status(name)
            for name in self._agents
        }

    async def call_agent(
        self,
        agent_name: str,
        document_content: str,
        metadata: Optional[dict[str, Any]] = None
    ) -> dict[str, Any]:
        """Convenience method to call an agent's extract tool.

        Args:
            agent_name: Agent identifier.
            document_content: Raw document text.
            metadata: Optional metadata dict.

        Returns:
            Extraction result from the agent.
        """
        proxy = self.get_proxy(agent_name)
        result = await proxy.invoke_extract(document_content, metadata)

        # Update agent info with last result
        if agent_name in self._agents:
            self._agents[agent_name].last_result = result

        return result

    async def __aenter__(self) -> "MCPCoordinator":
        """Async context manager entry."""
        await self.connect_all()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.disconnect_all()

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance (useful for testing)."""
        if cls._instance is not None:
            # Try to disconnect if there's a running event loop
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(cls._instance.disconnect_all())
            except RuntimeError:
                # No running event loop, just mark as disconnected
                cls._instance._connected = False
                for agent_info in cls._instance._agents.values():
                    agent_info.proxy = None
                    agent_info.status = AgentStatus.DISCONNECTED
        cls._instance = None