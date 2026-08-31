from __future__ import annotations

from mcp.server.fastmcp import FastMCP


def create_mcp_server(name: str) -> FastMCP:
    """Create and return a FastMCP server instance.

    Args:
        name: Human-readable server name.

    Returns:
        Configured FastMCP instance ready for tool registration.
    """
    return FastMCP(name)
