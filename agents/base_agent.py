from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)


@dataclass
class AgentConnection:
    """Tracks a connection to another agent's MCP server."""

    name: str
    config: dict[str, Any]
    client: Any = None
    tools: list[dict[str, Any]] = field(default_factory=list)
    connected: bool = False
    _exit_stack: AsyncExitStack | None = field(default=None, repr=False)


class BaseMCPAgent(ABC):
    """Abstract base class for all MCP-based document extraction agents.

    Each agent exposes an MCP tool named ``extract`` that processes document
    content and returns structured fields. Subclasses must implement
    ``_get_extraction_prompt`` to supply the LLM prompt (added in later phases).

    This base class also provides MCP client capabilities for inter-agent
    communication, allowing agents to call tools on other connected agents.
    """

    # Class-level registry shared across all agent instances
    _connected_agents: dict[str, AgentConnection] = {}

    def __init__(self, name: str, schema_class: type) -> None:
        self.name = name
        self.schema_class = schema_class
        self.mcp = FastMCP(name)
        self._register_tools()
        self._register_client_tools()
        logger.info("Initialized agent '%s'", name)

    @abstractmethod
    def _get_extraction_prompt(self, content: str) -> str:
        """Return the prompt that instructs the LLM to extract fields.

        Args:
            content: Raw document text to extract from.

        Returns:
            Formatted prompt string.
        """
        ...

    def _register_tools(self) -> None:
        """Register the ``extract`` tool on the MCP server."""

        @self.mcp.tool()
        async def extract(document_content: str, metadata: dict | None = None) -> dict:
            """Extract structured fields from a document.

            Args:
                document_content: Raw text content of the document.
                metadata: Optional dict with document metadata (source, type, etc).

            Returns:
                Dict with keys: fields, confidence, model_used, cost, requires_review.
            """
            return await self._extract(document_content, metadata or {})

    def _register_client_tools(self) -> None:
        """Register inter-agent MCP tools on the server."""

        @self.mcp.tool()
        async def call_other_agent(
            agent_name: str,
            tool_name: str,
            arguments: dict[str, Any] | None = None,
        ) -> dict:
            """Call a tool on another connected agent.

            Args:
                agent_name: Name of the target agent.
                tool_name: Name of the tool to invoke.
                arguments: Optional arguments to pass to the tool.

            Returns:
                Result dict from the target agent's tool, or error dict.
            """
            return await self.call_agent_tool(agent_name, tool_name, arguments or {})

        @self.mcp.tool()
        async def list_connected_agents() -> dict:
            """List all connected agents and their available tools.

            Returns:
                Dict mapping agent names to their tool lists and connection status.
            """
            agents = self.get_available_agents()
            return {"agents": agents}

        @self.mcp.tool()
        async def share_result(
            result: dict[str, Any],
            source_agent: str | None = None,
            target_agents: list[str] | None = None,
        ) -> dict:
            """Share an extraction result with other connected agents.

            Args:
                result: The extraction result to share.
                source_agent: Name of the agent sharing the result (defaults to self).
                target_agents: Optional list of agent names to share with.
                    If None, shares with all connected agents.

            Returns:
                Dict with delivery status per target agent.
            """
            source = source_agent or self.name
            targets = target_agents or [
                name for name in BaseMCPAgent._connected_agents if name != source
            ]

            delivery_status: dict[str, str] = {}
            for target in targets:
                try:
                    conn = BaseMCPAgent._connected_agents.get(target)
                    if conn and conn.connected:
                        await self.call_agent_tool(target, "receive_shared_result", {
                            "result": result,
                            "source_agent": source,
                        })
                        delivery_status[target] = "delivered"
                    else:
                        delivery_status[target] = "not_connected"
                except Exception as e:
                    logger.warning("Failed to share result to '%s': %s", target, e)
                    delivery_status[target] = f"error: {e}"

            return {"source": source, "delivery_status": delivery_status}

    # ------------------------------------------------------------------
    # MCP client methods
    # ------------------------------------------------------------------

    async def connect_to_agent(self, agent_name: str, config: dict[str, Any]) -> dict[str, Any]:
        """Connect to another agent's MCP server.

        Args:
            agent_name: Unique name for the remote agent.
            config: Connection config, e.g.::

                {
                    "transport": "stdio",
                    "command": "python",
                    "args": ["-m", "agents.some_agent"],
                }
                # or SSE:
                {
                    "transport": "sse",
                    "url": "http://localhost:8001/sse",
                }

        Returns:
            Dict with connection status and discovered tools.
        """
        if agent_name in BaseMCPAgent._connected_agents:
            existing = BaseMCPAgent._connected_agents[agent_name]
            if existing.connected:
                logger.info("Already connected to agent '%s'", agent_name)
                return {"status": "already_connected", "tools": [t["name"] for t in existing.tools]}

        try:
            from langchain_mcp_adapters.client import MultiServerMCPClient

            transport = config.get("transport", "stdio")
            client_kwargs: dict[str, Any] = {}

            if transport == "stdio":
                client_kwargs[agent_name] = {
                    "transport": "stdio",
                    "command": config.get("command", "python"),
                    "args": config.get("args", []),
                }
            elif transport == "sse":
                client_kwargs[agent_name] = {
                    "transport": "sse",
                    "url": config.get("url", ""),
                }
            else:
                return {"status": "error", "message": f"Unknown transport: {transport}"}

            exit_stack = AsyncExitStack()
            client = MultiServerMCPClient(client_kwargs)
            await exit_stack.__aenter__()

            session = await client.get_session(agent_name)
            tools_response = await session.list_tools()
            tools = [
                {"name": t.name, "description": t.description, "input_schema": t.inputSchema}
                for t in tools_response.tools
            ]

            conn = AgentConnection(
                name=agent_name,
                config=config,
                client=client,
                tools=tools,
                connected=True,
                _exit_stack=exit_stack,
            )
            BaseMCPAgent._connected_agents[agent_name] = conn

            logger.info(
                "Connected to agent '%s' (%d tools available)",
                agent_name,
                len(tools),
            )
            return {"status": "connected", "tools": [t["name"] for t in tools]}

        except ImportError:
            msg = "langchain_mcp_adapters is not installed. Install with: pip install langchain-mcp-adapters"
            logger.error(msg)
            return {"status": "error", "message": msg}
        except Exception as e:
            logger.error("Failed to connect to agent '%s': %s", agent_name, e)
            return {"status": "error", "message": str(e)}

    async def disconnect_from_agent(self, agent_name: str) -> dict[str, str]:
        """Disconnect from a previously connected agent.

        Args:
            agent_name: Name of the agent to disconnect from.

        Returns:
            Disconnection status.
        """
        conn = BaseMCPAgent._connected_agents.pop(agent_name, None)
        if conn is None:
            return {"status": "not_found"}

        if conn._exit_stack:
            try:
                await conn._exit_stack.aclose()
            except Exception as e:
                logger.warning("Error closing connection to '%s': %s", agent_name, e)

        logger.info("Disconnected from agent '%s'", agent_name)
        return {"status": "disconnected"}

    async def call_agent_tool(
        self,
        agent_name: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Call a tool on a connected agent.

        Args:
            agent_name: Name of the target agent.
            tool_name: Name of the tool to invoke.
            arguments: Arguments to pass to the tool.

        Returns:
            Tool execution result, or error dict.
        """
        conn = BaseMCPAgent._connected_agents.get(agent_name)
        if conn is None or not conn.connected:
            return {"error": f"Agent '{agent_name}' is not connected"}

        if conn.client is None:
            return {"error": f"Agent '{agent_name}' has no client"}

        try:
            session = await conn.client.get_session(agent_name)
            result = await session.call_tool(tool_name, arguments=arguments)
            # Extract content from MCP response
            if hasattr(result, "content"):
                content_parts = []
                for block in result.content:
                    if hasattr(block, "text"):
                        content_parts.append(block.text)
                raw = "\n".join(content_parts)
                try:
                    return json.loads(raw)
                except json.JSONDecodeError:
                    return {"result": raw}
            return {"result": str(result)}
        except Exception as e:
            logger.error("Tool call '%s' on agent '%s' failed: %s", tool_name, agent_name, e)
            return {"error": str(e)}

    def get_available_agents(self) -> dict[str, dict[str, Any]]:
        """List all connected agents and their tools.

        Returns:
            Dict mapping agent names to connection info and tool lists.
        """
        result: dict[str, dict[str, Any]] = {}
        for name, conn in BaseMCPAgent._connected_agents.items():
            result[name] = {
                "connected": conn.connected,
                "tools": [t["name"] for t in conn.tools],
                "transport": conn.config.get("transport", "unknown"),
            }
        return result

    def aggregate_results(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        """Combine results from multiple agent calls into a single response.

        Picks the result with the highest confidence when multiple extraction
        results are present. For non-extraction results, merges them by source
        agent.

        Args:
            results: List of result dicts from different agents.

        Returns:
            Aggregated result dict.
        """
        if not results:
            return {
                "fields": {},
                "confidence": 0.0,
                "model_used": "none",
                "cost": 0.0,
                "requires_review": True,
                "source_agents": [],
            }

        if len(results) == 1:
            merged = dict(results[0])
            merged.setdefault("source_agents", [merged.get("agent_name", "unknown")])
            return merged

        # Partition extraction results (with 'confidence' key) from others
        extraction_results = [r for r in results if "confidence" in r]
        other_results = [r for r in results if "confidence" not in r]

        aggregated: dict[str, Any] = {}

        if extraction_results:
            best = max(extraction_results, key=lambda r: r.get("confidence", 0.0))
            aggregated.update({
                "fields": best.get("fields", {}),
                "confidence": best.get("confidence", 0.0),
                "model_used": best.get("model_used", "unknown"),
                "cost": best.get("cost", 0.0),
                "requires_review": best.get("requires_review", True),
                "source_agents": [
                    r.get("agent_name", "unknown") for r in extraction_results
                ],
            })

            # Merge fields from other extraction results where best had empty values
            for r in extraction_results:
                if r is best:
                    continue
                r_fields = r.get("fields", {})
                for k, v in r_fields.items():
                    if k in aggregated["fields"] and aggregated["fields"][k] in (None, "", {}, []):
                        aggregated["fields"][k] = v

        if other_results:
            aggregated.setdefault("additional_results", []).extend(other_results)
            aggregated.setdefault("source_agents", []).extend(
                [r.get("agent_name", "unknown") for r in other_results]
            )

        # Deduplicate source_agents
        aggregated["source_agents"] = list(dict.fromkeys(aggregated.get("source_agents", [])))

        return aggregated

    async def disconnect_all(self) -> None:
        """Disconnect from all connected agents."""
        names = list(BaseMCPAgent._connected_agents.keys())
        for name in names:
            await self.disconnect_from_agent(name)
        logger.info("Disconnected from all agents")

    async def _extract(self, content: str, metadata: dict[str, Any]) -> dict[str, Any]:
        """Run extraction using the LLM cascade.

        Tries models in cascade order, validates extraction against schema,
        returns the best result.
        """
        from model_router.llm_client import call_llm, parse_json_response
        from model_router.confidence import compute_confidence
        from config import get_settings

        settings = get_settings()
        prompt = self._get_extraction_prompt(content)

        last_error = None
        best_result = None
        best_confidence = 0.0

        for tier in settings.model_cascade:
            # Skip prompt_guard tier (it's for content safety, not extraction)
            if tier.name == "prompt_guard":
                continue

            try:
                logger.info("Trying %s (%s) for agent '%s'", tier.name, tier.model, self.name)

                response_text = await call_llm(
                    model=tier.model,
                    prompt=prompt,
                    provider=tier.provider,
                )

                extracted = parse_json_response(response_text)

                # Compute confidence from raw extraction (includes _confidence keys)
                confidence = compute_confidence(extracted)

                # Validate against schema if available
                try:
                    validated = self.schema_class.model_validate(extracted)
                    fields = validated.model_dump()
                except Exception as e:
                    logger.warning("Schema validation failed for %s: %s", tier.name, e)
                    fields = extracted

                result = {
                    "fields": fields,
                    "confidence": confidence,
                    "model_used": f"{tier.provider}:{tier.model}",
                    "cost": tier.cost_per_doc,
                    "requires_review": confidence < settings.confidence_threshold,
                    "agent_name": self.name,
                }

                # Return immediately if confidence is high enough
                if confidence >= tier.confidence_threshold:
                    logger.info("Extraction succeeded with %s (confidence: %.2f)", tier.name, confidence)
                    return result

                # Track best result for fallback
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_result = result

            except Exception as e:
                logger.warning("Model %s failed: %s", tier.name, e)
                last_error = e
                continue

        # Return best result found, or create error result
        if best_result is not None:
            logger.info("Using best available result from cascade (confidence: %.2f)", best_confidence)
            return best_result

        logger.error("All models failed. Last error: %s", last_error)
        return {
            "fields": {},
            "confidence": 0.0,
            "model_used": "none",
            "cost": 0.0,
            "requires_review": True,
            "agent_name": self.name,
        }

    def run(self) -> None:
        """Start the MCP server (blocking)."""
        logger.info("Starting MCP server for agent '%s'", self.name)
        self.mcp.run()
