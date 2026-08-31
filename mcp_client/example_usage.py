"""Example usage of the MCP Coordinator for inter-agent communication.

This script demonstrates how to use the MCPCoordinator and AgentProxy
classes to enable communication between different document processing agents.
"""

import asyncio
import logging
from mcp_client import MCPCoordinator, AgentProxy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    """Demonstrate MCP coordinator usage."""
    print("=== MCP Coordinator Example ===\n")

    # Example 1: Using the coordinator as a singleton
    print("1. Creating MCPCoordinator singleton...")
    coordinator = MCPCoordinator()

    # Register agents with their MCP server configurations
    print("2. Registering agents...")
    coordinator.register_agent(
        "invoice",
        {
            "transport": "stdio",
            "command": "python",
            "args": ["-m", "agents.invoice_agent"],
        }
    )

    coordinator.register_agent(
        "receipt",
        {
            "transport": "stdio",
            "command": "python",
            "args": ["-m", "agents.receipt_agent"],
        }
    )

    coordinator.register_agent(
        "contract",
        {
            "transport": "stdio",
            "command": "python",
            "args": ["-m", "agents.contract_agent"],
        }
    )

    print(f"   Registered agents: {coordinator.agent_names}")

    # Connect to all agents
    print("\n3. Connecting to all agents...")
    try:
        await coordinator.connect_all()
        print("   Connected successfully!")
    except RuntimeError as e:
        print(f"   Connection failed: {e}")
        print("   (This is expected if agents are not running)")
        return

    # Get all available tools
    print("\n4. Getting all available tools...")
    try:
        all_tools = await coordinator.get_all_tools()
        print(f"   Found {len(all_tools)} tools:")
        for tool in all_tools:
            print(f"   - {tool.name}")
    except Exception as e:
        print(f"   Error getting tools: {e}")

    # Get tools for specific agent
    print("\n5. Getting tools for invoice agent...")
    try:
        invoice_tools = await coordinator.get_tools_for_agent("invoice")
        print(f"   Invoice agent has {len(invoice_tools)} tools:")
        for tool in invoice_tools:
            print(f"   - {tool.name}")
    except Exception as e:
        print(f"   Error: {e}")

    # Get agent status
    print("\n6. Checking agent statuses...")
    statuses = coordinator.get_all_statuses()
    for name, status in statuses.items():
        print(f"   {name}: {status['status']} ({status['tool_count']} tools)")

    # Example 2: Using AgentProxy directly
    print("\n7. Using AgentProxy directly...")
    try:
        async with AgentProxy("invoice", {
            "transport": "stdio",
            "command": "python",
            "args": ["-m", "agents.invoice_agent"],
        }) as proxy:
            print(f"   Connected to proxy: {proxy}")

            # Get tool names
            tool_names = await proxy.get_tool_names()
            print(f"   Available tools: {tool_names}")

            # Check if extract tool exists
            has_extract = await proxy.has_tool("extract")
            print(f"   Has extract tool: {has_extract}")

            # Invoke extract tool (if agents were running)
            # result = await proxy.invoke_extract("Sample invoice content...")
            # print(f"   Extraction result: {result}")

    except Exception as e:
        print(f"   Proxy example failed: {e}")
        print("   (This is expected if agents are not running)")

    # Example 3: Calling an agent through the coordinator
    print("\n8. Calling agent through coordinator...")
    try:
        # Sample document content
        sample_invoice = """
        INVOICE #12345
        Date: 2024-01-15
        From: ABC Corporation
        To: XYZ Company
        Amount: $1,234.56
        Due: 2024-02-15
        """

        result = await coordinator.call_agent(
            "invoice",
            sample_invoice,
            {"source": "email", "type": "invoice"}
        )
        print(f"   Extraction result: {result}")

    except Exception as e:
        print(f"   Agent call failed: {e}")
        print("   (This is expected if agents are not running)")

    # Disconnect from all agents
    print("\n9. Disconnecting from all agents...")
    await coordinator.disconnect_all()
    print("   Disconnected successfully!")

    print("\n=== Example Complete ===")


async def example_with_context_manager():
    """Example using async context manager for automatic cleanup."""
    print("\n=== Context Manager Example ===\n")

    try:
        async with MCPCoordinator() as coordinator:
            # Register agents
            coordinator.register_agent(
                "invoice",
                {
                    "transport": "stdio",
                    "command": "python",
                    "args": ["-m", "agents.invoice_agent"],
                }
            )

            # Use coordinator...
            print("Connected to coordinator!")
            print(f"Agent names: {coordinator.agent_names}")

            # Work with agents...

        # Automatically disconnected when exiting context
        print("Disconnected automatically!")

    except Exception as e:
        print(f"Example failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
    # asyncio.run(example_with_context_manager())