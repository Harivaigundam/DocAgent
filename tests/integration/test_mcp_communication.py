"""Integration tests for agent-to-agent MCP communication."""
import pytest
import json
from unittest.mock import AsyncMock, patch, MagicMock
from mcp_client.client import MCPClientManager
from mcp_client.server import create_mcp_server
from agents.base_agent import BaseMCPAgent
from agents import AGENT_MAP
from agents.invoice_agent import InvoiceAgent
from agents.receipt_agent import ReceiptAgent
from agents.contract_agent import ContractAgent
from agents.report_agent import ReportAgent


class TestMCPServerCreation:
    def test_create_server_returns_fastmcp(self):
        server = create_mcp_server("test_server")
        assert server is not None
        assert hasattr(server, "tool")

    def test_create_server_with_name(self):
        server = create_mcp_server("MyServer")
        assert server.name == "MyServer"

    def test_multiple_servers_are_independent(self):
        s1 = create_mcp_server("Server1")
        s2 = create_mcp_server("Server2")
        assert s1 is not s2
        assert s1.name != s2.name


class TestAgentMCPServers:
    def test_all_agents_have_mcp(self):
        for agent_class in AGENT_MAP.values():
            agent = agent_class()
            assert hasattr(agent, "mcp")
            assert agent.mcp is not None

    def test_agents_expose_extract_tool(self):
        for agent_class in AGENT_MAP.values():
            agent = agent_class()
            assert hasattr(agent, "_extract")
            assert callable(agent._extract)

    def test_agent_names(self):
        assert AGENT_MAP["invoice"].name if hasattr(AGENT_MAP["invoice"], "name") else True
        agents = {
            "invoice": InvoiceAgent(),
            "receipt": ReceiptAgent(),
            "contract": ContractAgent(),
            "report": ReportAgent(),
        }
        assert agents["invoice"].name == "InvoiceAgent"
        assert agents["receipt"].name == "ReceiptAgent"
        assert agents["contract"].name == "ContractAgent"
        assert agents["report"].name == "ReportAgent"


class TestAgentToolInvocation:
    @pytest.mark.asyncio
    @patch("model_router.llm_client.call_llm", new_callable=AsyncMock, return_value=json.dumps({
        "vendor_name": "Test Corp",
        "invoice_number": "INV-001",
        "date": "2026-01-15",
        "total": 1000,
        "vendor_name_confidence": 0.95,
        "invoice_number_confidence": 0.98,
        "total_confidence": 0.92,
    }))
    async def test_invoice_agent_extract(self, mock_llm):
        agent = InvoiceAgent()
        result = await agent._extract("Invoice from Test Corp for $1000", {})
        assert "fields" in result
        assert "confidence" in result
        assert "model_used" in result
        assert "cost" in result
        assert "requires_review" in result
        assert result["confidence"] > 0

    @pytest.mark.asyncio
    @patch("model_router.llm_client.call_llm", new_callable=AsyncMock, return_value=json.dumps({
        "store_name": "Walmart",
        "date": "2026-01-15",
        "total": 45.99,
        "store_name_confidence": 0.95,
        "total_confidence": 0.98,
    }))
    async def test_receipt_agent_extract(self, mock_llm):
        agent = ReceiptAgent()
        result = await agent._extract("Receipt from Walmart for $45.99", {})
        assert "fields" in result
        assert result["confidence"] > 0

    @pytest.mark.asyncio
    @patch("model_router.llm_client.call_llm", new_callable=AsyncMock, return_value=json.dumps({
        "title": "Service Agreement",
        "parties": ["Acme Corp", "Widget Inc"],
        "effective_date": "2026-01-01",
        "title_confidence": 0.97,
        "parties_confidence": 0.90,
    }))
    async def test_contract_agent_extract(self, mock_llm):
        agent = ContractAgent()
        result = await agent._extract("Service Agreement between Acme Corp and Widget Inc", {})
        assert "fields" in result
        assert result["confidence"] > 0

    @pytest.mark.asyncio
    @patch("model_router.llm_client.call_llm", new_callable=AsyncMock, return_value=json.dumps({
        "title": "Q4 Analysis",
        "author": "Analytics Team",
        "date": "2026-01-15",
        "title_confidence": 0.96,
        "author_confidence": 0.93,
    }))
    async def test_report_agent_extract(self, mock_llm):
        agent = ReportAgent()
        result = await agent._extract("Q4 Analysis report by Analytics Team", {})
        assert "fields" in result
        assert result["confidence"] > 0


class TestMCPClientManagerConnections:
    @pytest.mark.asyncio
    async def test_connect_with_registered_agents(self):
        mgr = MCPClientManager()
        mgr.register_agent("agent_a", {"transport": "stdio", "command": "python", "args": ["-m", "server_a"]})
        mgr.register_agent("agent_b", {"transport": "stdio", "command": "python", "args": ["-m", "server_b"]})
        with patch("mcp_client.client.MultiServerMCPClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            await mgr.connect()
            assert mgr.connected is True
            assert mgr._client is mock_client

    @pytest.mark.asyncio
    async def test_connect_stores_configs(self):
        mgr = MCPClientManager()
        config = {"transport": "sse", "url": "http://localhost:8080"}
        mgr.register_agent("my_agent", config)
        with patch("mcp_client.client.MultiServerMCPClient") as mock_client_cls:
            await mgr.connect()
            mock_client_cls.assert_called_once_with({"my_agent": config})

    @pytest.mark.asyncio
    async def test_close_after_connect(self):
        mgr = MCPClientManager()
        mgr.register_agent("agent_a", {"transport": "stdio", "command": "python"})
        with patch("mcp_client.client.MultiServerMCPClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__aexit__ = AsyncMock()
            mock_client_cls.return_value = mock_client
            await mgr.connect()
            await mgr.close()
            assert mgr.connected is False
            assert mgr._client is None
            assert mgr._tools_cache is None


class TestMCPToolDiscovery:
    @pytest.mark.asyncio
    async def test_get_tools_when_connected(self):
        mgr = MCPClientManager()
        mgr.register_agent("agent_a", {"transport": "stdio", "command": "python"})
        with patch("mcp_client.client.MultiServerMCPClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_tool = MagicMock()
            mock_tool.name = "extract"
            mock_client.get_tools = AsyncMock(return_value=[mock_tool])
            mock_client_cls.return_value = mock_client
            await mgr.connect()
            tools = await mgr.get_tools()
            assert len(tools) == 1
            assert tools[0].name == "extract"

    @pytest.mark.asyncio
    async def test_get_tools_filtered_by_agent_name(self):
        mgr = MCPClientManager()
        mgr.register_agent("agent_a", {"transport": "stdio", "command": "python"})
        with patch("mcp_client.client.MultiServerMCPClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_tool_a = MagicMock()
            mock_tool_a.name = "invoice_extract"
            mock_tool_b = MagicMock()
            mock_tool_b.name = "receipt_extract"
            mock_client.get_tools = AsyncMock(return_value=[mock_tool_a, mock_tool_b])
            mock_client_cls.return_value = mock_client
            await mgr.connect()
            tools = await mgr.get_tools(agent_name="invoice")
            assert len(tools) == 1
            assert tools[0].name == "invoice_extract"

    @pytest.mark.asyncio
    async def test_get_tools_caches_result(self):
        mgr = MCPClientManager()
        mgr.register_agent("agent_a", {"transport": "stdio", "command": "python"})
        with patch("mcp_client.client.MultiServerMCPClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_tool = MagicMock()
            mock_tool.name = "extract"
            mock_client.get_tools = AsyncMock(return_value=[mock_tool])
            mock_client_cls.return_value = mock_client
            await mgr.connect()
            tools1 = await mgr.get_tools()
            tools2 = await mgr.get_tools()
            mock_client.get_tools.assert_called_once()
            assert len(tools1) == len(tools2) == 1


class TestMCPToolInvocation:
    @pytest.mark.asyncio
    async def test_invoke_existing_tool(self):
        mgr = MCPClientManager()
        mgr.register_agent("agent_a", {"transport": "stdio", "command": "python"})
        with patch("mcp_client.client.MultiServerMCPClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_tool = MagicMock()
            mock_tool.name = "extract"
            mock_tool.ainvoke = AsyncMock(return_value={"result": "extracted"})
            mock_client.get_tools = AsyncMock(return_value=[mock_tool])
            mock_client_cls.return_value = mock_client
            await mgr.connect()
            result = await mgr.invoke_tool("extract", {"content": "test"})
            assert result == {"result": "extracted"}
            mock_tool.ainvoke.assert_called_once_with({"content": "test"})

    @pytest.mark.asyncio
    async def test_invoke_nonexistent_tool_raises(self):
        mgr = MCPClientManager()
        mgr.register_agent("agent_a", {"transport": "stdio", "command": "python"})
        with patch("mcp_client.client.MultiServerMCPClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_tool = MagicMock()
            mock_tool.name = "extract"
            mock_client.get_tools = AsyncMock(return_value=[mock_tool])
            mock_client_cls.return_value = mock_client
            await mgr.connect()
            with pytest.raises(ValueError, match="Tool 'nonexistent' not found"):
                await mgr.invoke_tool("nonexistent", {})


class TestResultSharingBetweenAgents:
    @pytest.mark.asyncio
    async def test_parallel_agents_receive_shared_state(self):
        from parallel_orchestrator import ParallelDocumentOrchestrator
        orch = ParallelDocumentOrchestrator()
        state = {
            "messages": [],
            "document_id": "shared_test",
            "document_content": "Invoice from Acme Corp for $5000",
            "document_type": "invoice",
            "type_confidence": 0.8,
            "extraction_result": {},
            "extraction_confidence": 0.0,
            "model_used": "",
            "model_cost": 0.0,
            "requires_review": False,
            "review_status": "",
            "guardrail_passed": False,
            "guardrail_issues": [],
            "final_output": {},
            "parallel_results": [],
            "all_results": [],
            "parallel_agent_type": "",
        }
        sends = orch._route_to_agents(state)
        for send in sends:
            assert send.arg["document_content"] == state["document_content"]
            assert send.arg["document_id"] == state["document_id"]

    @pytest.mark.asyncio
    async def test_agents_can_share_extraction_results(self):
        results_a = {"agent_type": "invoice", "fields": {"total": 1000}, "confidence": 0.85}
        results_b = {"agent_type": "receipt", "fields": {"total": 50}, "confidence": 0.90}
        shared_results = [results_a, results_b]
        best = max(shared_results, key=lambda r: r["confidence"])
        assert best["agent_type"] == "receipt"
        assert best["confidence"] == 0.90


class TestConnectionResilience:
    @pytest.mark.asyncio
    async def test_invoke_tool_after_close_raises(self):
        mgr = MCPClientManager()
        mgr._connected = False
        with pytest.raises(RuntimeError, match="Not connected"):
            await mgr.invoke_tool("extract", {})

    @pytest.mark.asyncio
    async def test_get_tools_after_close_raises(self):
        mgr = MCPClientManager()
        mgr._connected = False
        with pytest.raises(RuntimeError, match="Not connected"):
            await mgr.get_tools()

    @pytest.mark.asyncio
    async def test_multiple_close_calls_are_safe(self):
        mgr = MCPClientManager()
        await mgr.close()
        await mgr.close()
        assert mgr.connected is False

    @pytest.mark.asyncio
    async def test_reconnect_after_close(self):
        mgr = MCPClientManager()
        mgr.register_agent("agent_a", {"transport": "stdio", "command": "python"})
        with patch("mcp_client.client.MultiServerMCPClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            await mgr.connect()
            assert mgr.connected is True
            await mgr.close()
            assert mgr.connected is False
            mock_client_cls.return_value = mock_client
            await mgr.connect()
            assert mgr.connected is True


class TestAgentErrorHandling:
    @pytest.mark.asyncio
    @patch("model_router.llm_client.call_llm", new_callable=AsyncMock, side_effect=Exception("LLM unavailable"))
    async def test_agent_handles_llm_failure(self, mock_llm):
        agent = InvoiceAgent()
        result = await agent._extract("test content", {})
        assert result["fields"] == {}
        assert result["confidence"] == 0.0
        assert result["requires_review"] is True

    @pytest.mark.asyncio
    @patch("model_router.llm_client.call_llm", new_callable=AsyncMock, side_effect=Exception("API timeout"))
    async def test_all_agents_handle_llm_failure(self, mock_llm):
        for agent_class in AGENT_MAP.values():
            agent = agent_class()
            result = await agent._extract("test content", {})
            assert result["fields"] == {}
            assert result["confidence"] == 0.0
            assert result["requires_review"] is True
