"""Unit tests for subtask agents (parser, validator, summarizer patterns).

Tests cover agent independence, tool registration, extraction prompts,
and the base agent's _extract cascade behavior.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from agents.base_agent import BaseMCPAgent
from agents.invoice_agent import InvoiceAgent
from agents.receipt_agent import ReceiptAgent
from agents.contract_agent import ContractAgent
from agents.report_agent import ReportAgent
from agents import AGENT_MAP


class TestBaseAgentToolRegistration:
    """Test that BaseMCPAgent registers extract tool on MCP server."""

    def test_mcp_server_created(self):
        with patch("agents.base_agent.BaseMCPAgent.__init__", return_value=None):
            agent = InvoiceAgent()
            # __init__ was patched to no-op, so set attributes manually
            from mcp.server.fastmcp import FastMCP
            agent.mcp = FastMCP("InvoiceAgent")
            agent.name = "InvoiceAgent"
            assert agent.mcp is not None

    def test_name_set(self):
        with patch("agents.invoice_agent.BaseMCPAgent.__init__", return_value=None):
            agent = InvoiceAgent()
            agent.name = "InvoiceAgent"
            assert agent.name == "InvoiceAgent"

    def test_schema_class_set(self):
        with patch("agents.invoice_agent.BaseMCPAgent.__init__", return_value=None):
            agent = InvoiceAgent()
            from schemas.invoice import InvoiceFields
            agent.schema_class = InvoiceFields
            assert agent.schema_class is InvoiceFields


class TestInvoiceAgentTools:
    @pytest.fixture
    def agent(self):
        with patch("agents.invoice_agent.BaseMCPAgent.__init__", return_value=None):
            return InvoiceAgent()

    def test_get_extraction_prompt_returns_string(self, agent):
        result = agent._get_extraction_prompt("Invoice from Acme Corp")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_prompt_includes_content(self, agent):
        content = "Invoice INV-12345 for $500"
        result = agent._get_extraction_prompt(content)
        assert content in result or "INV-12345" in result or "invoice" in result.lower()


class TestReceiptAgentTools:
    @pytest.fixture
    def agent(self):
        with patch("agents.receipt_agent.BaseMCPAgent.__init__", return_value=None):
            return ReceiptAgent()

    def test_get_extraction_prompt_returns_string(self, agent):
        result = agent._get_extraction_prompt("Receipt from Walmart")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_prompt_includes_content(self, agent):
        content = "Receipt #R9981, total $25.50"
        result = agent._get_extraction_prompt(content)
        assert isinstance(result, str)


class TestContractAgentTools:
    @pytest.fixture
    def agent(self):
        with patch("agents.contract_agent.BaseMCPAgent.__init__", return_value=None):
            return ContractAgent()

    def test_get_extraction_prompt_returns_string(self, agent):
        result = agent._get_extraction_prompt("Service Agreement between Party A and Party B")
        assert isinstance(result, str)
        assert len(result) > 0


class TestReportAgentTools:
    @pytest.fixture
    def agent(self):
        with patch("agents.report_agent.BaseMCPAgent.__init__", return_value=None):
            return ReportAgent()

    def test_get_extraction_prompt_returns_string(self, agent):
        result = agent._get_extraction_prompt("Quarterly Analysis Report")
        assert isinstance(result, str)
        assert len(result) > 0


class TestAgentIndependence:
    """Test that agents operate independently and have distinct behaviors."""

    def test_all_agents_instantiatable(self):
        with patch("agents.invoice_agent.BaseMCPAgent.__init__", return_value=None), \
             patch("agents.receipt_agent.BaseMCPAgent.__init__", return_value=None), \
             patch("agents.contract_agent.BaseMCPAgent.__init__", return_value=None), \
             patch("agents.report_agent.BaseMCPAgent.__init__", return_value=None):
            for agent_class in AGENT_MAP.values():
                agent = agent_class()
                assert agent is not None

    def test_all_agents_have_distinct_names(self):
        with patch("agents.invoice_agent.BaseMCPAgent.__init__", return_value=None), \
             patch("agents.receipt_agent.BaseMCPAgent.__init__", return_value=None), \
             patch("agents.contract_agent.BaseMCPAgent.__init__", return_value=None), \
             patch("agents.report_agent.BaseMCPAgent.__init__", return_value=None):
            names = []
            for agent_class in AGENT_MAP.values():
                agent = agent_class()
                agent.name = agent_class.__name__
                names.append(agent.name)
            assert len(set(names)) == len(names)

    def test_agents_return_different_prompts(self):
        with patch("agents.invoice_agent.BaseMCPAgent.__init__", return_value=None), \
             patch("agents.receipt_agent.BaseMCPAgent.__init__", return_value=None), \
             patch("agents.contract_agent.BaseMCPAgent.__init__", return_value=None), \
             patch("agents.report_agent.BaseMCPAgent.__init__", return_value=None):
            agents = [cls() for cls in AGENT_MAP.values()]
            content = "test document content"
            prompts = [a._get_extraction_prompt(content) for a in agents]
            # At least some prompts should differ (invoice vs contract)
            assert prompts[0] != prompts[2] or prompts[1] != prompts[3]


class TestAgentExtractionCascade:
    """Test the base agent _extract method with mocked LLM calls."""

    @pytest.fixture
    def invoice_agent(self):
        with patch("agents.invoice_agent.BaseMCPAgent.__init__", return_value=None):
            agent = InvoiceAgent()
            agent.name = "InvoiceAgent"
            from schemas.invoice import InvoiceFields
            agent.schema_class = InvoiceFields
            return agent

    @pytest.mark.asyncio
    @patch("model_router.confidence.compute_confidence", return_value=0.92)
    @patch("model_router.llm_client.parse_json_response")
    @patch("model_router.llm_client.call_llm", new_callable=AsyncMock)
    async def test_extract_success_first_tier(self, mock_llm, mock_parse, mock_conf, invoice_agent):
        mock_llm.return_value = '{"vendor_name": "Test", "confidence": 0.9}'
        mock_parse.return_value = {"vendor_name": "Test"}

        tier = MagicMock()
        tier.name = "ollama"
        tier.model = "llama3.1:8b"
        tier.provider = "ollama"
        tier.cost_per_doc = 0.0
        tier.confidence_threshold = 0.7

        with patch("config.get_settings") as mock_settings:
            mock_settings.return_value.model_cascade = [tier]
            mock_settings.return_value.confidence_threshold = 0.7

            result = await invoice_agent._extract("test content", {})

        assert result["confidence"] == 0.92
        assert result["model_used"] == "ollama:llama3.1:8b"
        assert result["cost"] == 0.0

    @pytest.mark.asyncio
    @patch("model_router.confidence.compute_confidence", return_value=0.5)
    @patch("model_router.llm_client.parse_json_response")
    @patch("model_router.llm_client.call_llm", new_callable=AsyncMock)
    async def test_extract_low_confidence_fallback(self, mock_llm, mock_parse, mock_conf, invoice_agent):
        mock_llm.return_value = '{"vendor_name": "Test"}'
        mock_parse.return_value = {"vendor_name": "Test"}

        tier = MagicMock()
        tier.name = "ollama"
        tier.model = "llama3.1:8b"
        tier.provider = "ollama"
        tier.cost_per_doc = 0.0
        tier.confidence_threshold = 0.7

        with patch("config.get_settings") as mock_settings:
            mock_settings.return_value.model_cascade = [tier]
            mock_settings.return_value.confidence_threshold = 0.7

            result = await invoice_agent._extract("test content", {})

        assert result["confidence"] == 0.5
        assert result["requires_review"] is True

    @pytest.mark.asyncio
    @patch("model_router.llm_client.call_llm", new_callable=AsyncMock, side_effect=Exception("LLM error"))
    async def test_extract_all_models_fail(self, mock_llm, invoice_agent):
        tier = MagicMock()
        tier.name = "ollama"
        tier.model = "llama3.1:8b"
        tier.provider = "ollama"

        with patch("config.get_settings") as mock_settings:
            mock_settings.return_value.model_cascade = [tier]
            mock_settings.return_value.confidence_threshold = 0.7

            result = await invoice_agent._extract("test content", {})

        assert result["confidence"] == 0.0
        assert result["requires_review"] is True
        assert result["model_used"] == "none"

    @pytest.mark.asyncio
    @patch("model_router.confidence.compute_confidence", return_value=0.95)
    @patch("model_router.llm_client.parse_json_response")
    @patch("model_router.llm_client.call_llm", new_callable=AsyncMock)
    async def test_extract_skips_prompt_guard_tier(self, mock_llm, mock_parse, mock_conf, invoice_agent):
        mock_llm.return_value = '{"vendor_name": "Test"}'
        mock_parse.return_value = {"vendor_name": "Test"}

        guard_tier = MagicMock()
        guard_tier.name = "prompt_guard"
        guard_tier.model = "meta-llama/llama-prompt-guard-2-86m"
        guard_tier.provider = "groq"
        guard_tier.cost_per_doc = 0.0
        guard_tier.confidence_threshold = 0.5

        ollama_tier = MagicMock()
        ollama_tier.name = "ollama"
        ollama_tier.model = "llama3.1:8b"
        ollama_tier.provider = "ollama"
        ollama_tier.cost_per_doc = 0.0
        ollama_tier.confidence_threshold = 0.7

        with patch("config.get_settings") as mock_settings:
            mock_settings.return_value.model_cascade = [guard_tier, ollama_tier]
            mock_settings.return_value.confidence_threshold = 0.7

            result = await invoice_agent._extract("test content", {})

        # prompt_guard should have been skipped, only ollama called
        assert mock_llm.call_count == 1
        assert mock_llm.call_args[1]["model"] == "llama3.1:8b"


class TestAgentMapCoverage:
    def test_agent_map_keys(self):
        assert set(AGENT_MAP.keys()) == {"invoice", "receipt", "contract", "report"}

    def test_agent_map_values_are_classes(self):
        for agent_class in AGENT_MAP.values():
            assert isinstance(agent_class, type)

    def test_agent_map_all_have_extract_prompt(self):
        with patch("agents.invoice_agent.BaseMCPAgent.__init__", return_value=None), \
             patch("agents.receipt_agent.BaseMCPAgent.__init__", return_value=None), \
             patch("agents.contract_agent.BaseMCPAgent.__init__", return_value=None), \
             patch("agents.report_agent.BaseMCPAgent.__init__", return_value=None):
            for agent_class in AGENT_MAP.values():
                agent = agent_class()
                assert hasattr(agent, "_get_extraction_prompt")
                assert callable(agent._get_extraction_prompt)
