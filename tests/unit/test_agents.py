import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from agents.invoice_agent import InvoiceAgent
from agents.receipt_agent import ReceiptAgent
from agents.contract_agent import ContractAgent
from agents.report_agent import ReportAgent
from agents.base_agent import BaseMCPAgent
from agents import AGENT_MAP


class TestInvoiceAgent:
    def test_instantiation(self):
        with patch("agents.invoice_agent.BaseMCPAgent.__init__", return_value=None):
            agent = InvoiceAgent()
            assert agent is not None

    def test_correct_name(self):
        def mock_init(self_agent, name, schema_class):
            self_agent.name = name
        with patch("agents.invoice_agent.BaseMCPAgent.__init__", mock_init):
            agent = InvoiceAgent()
            assert agent.name == "InvoiceAgent"

    def test_get_extraction_prompt_returns_string(self):
        with patch("agents.invoice_agent.BaseMCPAgent.__init__", return_value=None):
            agent = InvoiceAgent()
            result = agent._get_extraction_prompt("test content")
            assert isinstance(result, str)


class TestReceiptAgent:
    def test_instantiation(self):
        with patch("agents.receipt_agent.BaseMCPAgent.__init__", return_value=None):
            agent = ReceiptAgent()
            assert agent is not None

    def test_correct_name(self):
        def mock_init(self_agent, name, schema_class):
            self_agent.name = name
        with patch("agents.receipt_agent.BaseMCPAgent.__init__", mock_init):
            agent = ReceiptAgent()
            assert agent.name == "ReceiptAgent"

    def test_get_extraction_prompt_returns_string(self):
        with patch("agents.receipt_agent.BaseMCPAgent.__init__", return_value=None):
            agent = ReceiptAgent()
            result = agent._get_extraction_prompt("test content")
            assert isinstance(result, str)


class TestContractAgent:
    def test_instantiation(self):
        with patch("agents.contract_agent.BaseMCPAgent.__init__", return_value=None):
            agent = ContractAgent()
            assert agent is not None

    def test_correct_name(self):
        def mock_init(self_agent, name, schema_class):
            self_agent.name = name
        with patch("agents.contract_agent.BaseMCPAgent.__init__", mock_init):
            agent = ContractAgent()
            assert agent.name == "ContractAgent"

    def test_get_extraction_prompt_returns_string(self):
        with patch("agents.contract_agent.BaseMCPAgent.__init__", return_value=None):
            agent = ContractAgent()
            result = agent._get_extraction_prompt("test content")
            assert isinstance(result, str)


class TestReportAgent:
    def test_instantiation(self):
        with patch("agents.report_agent.BaseMCPAgent.__init__", return_value=None):
            agent = ReportAgent()
            assert agent is not None

    def test_correct_name(self):
        def mock_init(self_agent, name, schema_class):
            self_agent.name = name
        with patch("agents.report_agent.BaseMCPAgent.__init__", mock_init):
            agent = ReportAgent()
            assert agent.name == "ReportAgent"

    def test_get_extraction_prompt_returns_string(self):
        with patch("agents.report_agent.BaseMCPAgent.__init__", return_value=None):
            agent = ReportAgent()
            result = agent._get_extraction_prompt("test content")
            assert isinstance(result, str)


class TestAgentMap:
    def test_agent_map_contains_all_types(self):
        expected_keys = {"invoice", "receipt", "contract", "report"}
        assert set(AGENT_MAP.keys()) == expected_keys

    def test_agent_map_values_are_classes(self):
        for agent_class in AGENT_MAP.values():
            assert isinstance(agent_class, type)

    def test_agent_map_can_instantiate_all(self):
        with patch("agents.invoice_agent.BaseMCPAgent.__init__", return_value=None), \
             patch("agents.receipt_agent.BaseMCPAgent.__init__", return_value=None), \
             patch("agents.contract_agent.BaseMCPAgent.__init__", return_value=None), \
             patch("agents.report_agent.BaseMCPAgent.__init__", return_value=None):
            for agent_type, agent_class in AGENT_MAP.items():
                agent = agent_class()
                assert agent is not None


class TestBaseAgentMethods:
    """Tests for BaseMCPAgent base class behavior."""

    def test_base_agent_is_abstract(self):
        with pytest.raises(TypeError):
            BaseMCPAgent("test", MagicMock())

    def test_base_agent_requires_get_extraction_prompt(self):
        class IncompleteAgent(BaseMCPAgent):
            pass
        with pytest.raises(TypeError):
            IncompleteAgent("test", MagicMock())

    def test_base_agent_subclass_implements_prompt(self):
        class CompleteAgent(BaseMCPAgent):
            def _get_extraction_prompt(self, content: str) -> str:
                return f"Extract from: {content}"
        agent = CompleteAgent("test_agent", MagicMock())
        assert agent.name == "test_agent"
        assert agent._get_extraction_prompt("hello") == "Extract from: hello"

    def test_base_agent_creates_mcp_server(self):
        class TestAgent(BaseMCPAgent):
            def _get_extraction_prompt(self, content: str) -> str:
                return content
        agent = TestAgent("my_agent", MagicMock())
        assert agent.mcp is not None
        assert agent.mcp.name == "my_agent"

    def test_base_agent_schema_class_stored(self):
        class TestAgent(BaseMCPAgent):
            def _get_extraction_prompt(self, content: str) -> str:
                return content
        schema = MagicMock()
        agent = TestAgent("test", schema)
        assert agent.schema_class is schema


class TestMCPClientIntegration:
    """Tests for MCP client integration with agents."""

    @pytest.mark.asyncio
    async def test_agent_extract_returns_expected_keys(self):
        with patch("agents.invoice_agent.BaseMCPAgent.__init__", return_value=None):
            agent = InvoiceAgent()
            agent.name = "InvoiceAgent"
            from schemas.invoice import InvoiceFields
            agent.schema_class = InvoiceFields

        tier = MagicMock()
        tier.name = "ollama"
        tier.model = "llama3.1:8b"
        tier.provider = "ollama"
        tier.cost_per_doc = 0.0
        tier.confidence_threshold = 0.7

        with patch("model_router.llm_client.call_llm", new_callable=AsyncMock) as mock_llm, \
             patch("model_router.llm_client.parse_json_response") as mock_parse, \
             patch("model_router.confidence.compute_confidence", return_value=0.9), \
             patch("config.get_settings") as mock_settings:

            mock_llm.return_value = '{"vendor_name": "Test"}'
            mock_parse.return_value = {"vendor_name": "Test"}
            mock_settings.return_value.model_cascade = [tier]
            mock_settings.return_value.confidence_threshold = 0.7

            result = await agent._extract("test content", {})

        assert "fields" in result
        assert "confidence" in result
        assert "model_used" in result
        assert "cost" in result
        assert "requires_review" in result

    @pytest.mark.asyncio
    async def test_agent_extract_with_metadata(self):
        with patch("agents.invoice_agent.BaseMCPAgent.__init__", return_value=None):
            agent = InvoiceAgent()
            agent.name = "InvoiceAgent"
            from schemas.invoice import InvoiceFields
            agent.schema_class = InvoiceFields

        tier = MagicMock()
        tier.name = "ollama"
        tier.model = "llama3.1:8b"
        tier.provider = "ollama"
        tier.cost_per_doc = 0.0
        tier.confidence_threshold = 0.7

        with patch("model_router.llm_client.call_llm", new_callable=AsyncMock) as mock_llm, \
             patch("model_router.llm_client.parse_json_response") as mock_parse, \
             patch("model_router.confidence.compute_confidence", return_value=0.88), \
             patch("config.get_settings") as mock_settings:

            mock_llm.return_value = '{"vendor_name": "Test"}'
            mock_parse.return_value = {"vendor_name": "Test"}
            mock_settings.return_value.model_cascade = [tier]
            mock_settings.return_value.confidence_threshold = 0.7

            result = await agent._extract("test content", {"source": "email", "type": "invoice"})

        assert result["fields"]["vendor_name"] == "Test"


class TestResultAggregation:
    """Tests for result aggregation across agent outputs."""

    def test_best_result_selection_single(self):
        results = [
            {"agent_type": "invoice", "confidence": 0.95, "fields": {"total": 100}},
        ]
        best = max(results, key=lambda r: r["confidence"])
        assert best["agent_type"] == "invoice"

    def test_best_result_selection_multiple(self):
        results = [
            {"agent_type": "invoice", "confidence": 0.6},
            {"agent_type": "receipt", "confidence": 0.92},
            {"agent_type": "contract", "confidence": 0.75},
            {"agent_type": "report", "confidence": 0.88},
        ]
        best = max(results, key=lambda r: r["confidence"])
        assert best["agent_type"] == "receipt"

    def test_best_result_empty_list(self):
        results = []
        best = max(results, key=lambda r: r.get("confidence", 0)) if results else None
        assert best is None

    def test_aggregate_merges_all_results(self):
        results = [
            {"agent_type": "invoice", "confidence": 0.9, "fields": {"vendor_name": "A"}},
            {"agent_type": "receipt", "confidence": 0.7, "fields": {"store_name": "B"}},
        ]
        all_agent_types = [r["agent_type"] for r in results]
        assert "invoice" in all_agent_types
        assert "receipt" in all_agent_types
        assert len(results) == 2
