"""Integration tests for the DocumentOrchestrator pipeline."""
import pytest
import json
import time
from unittest.mock import AsyncMock, patch, MagicMock
from orchestrator import DocumentOrchestrator
from parallel_orchestrator import ParallelDocumentOrchestrator

MOCK_EXTRACTION_RESPONSE = json.dumps({
    "vendor_name": "Test Corp",
    "invoice_number": "INV-001",
    "date": "2026-01-15",
    "total": 1000,
    "line_items": [],
    "vendor_name_confidence": 0.95,
    "invoice_number_confidence": 0.98,
    "total_confidence": 0.92,
})


class TestOrchestratorInit:
    def test_creates_router(self):
        orch = DocumentOrchestrator()
        assert orch.router is not None

    def test_creates_cost_tracker(self):
        orch = DocumentOrchestrator()
        assert orch.cost_tracker is not None

    def test_creates_compiled_graph(self):
        orch = DocumentOrchestrator()
        assert orch.graph is not None


class TestTypeDetection:
    @pytest.fixture
    def orch(self):
        return DocumentOrchestrator()

    @pytest.mark.asyncio
    async def test_detects_invoice(self, orch):
        result = await orch._detect_type_node({
            "document_content": "Invoice from Acme Corp, amount due $5000, payment due net 30",
            "messages": [], "document_id": "", "document_type": "",
            "type_confidence": 0.0, "extraction_result": {},
            "extraction_confidence": 0.0, "model_used": "", "model_cost": 0.0,
            "requires_review": False, "review_status": "", "guardrail_passed": False,
            "guardrail_issues": [], "final_output": {},
        })
        assert result["document_type"] == "invoice"
        assert result["type_confidence"] > 0

    @pytest.mark.asyncio
    async def test_detects_receipt(self, orch):
        result = await orch._detect_type_node({
            "document_content": "Receipt from Walmart, purchased items, total paid $50",
            "messages": [], "document_id": "", "document_type": "",
            "type_confidence": 0.0, "extraction_result": {},
            "extraction_confidence": 0.0, "model_used": "", "model_cost": 0.0,
            "requires_review": False, "review_status": "", "guardrail_passed": False,
            "guardrail_issues": [], "final_output": {},
        })
        assert result["document_type"] == "receipt"

    @pytest.mark.asyncio
    async def test_detects_contract(self, orch):
        result = await orch._detect_type_node({
            "document_content": "Service agreement contract between Party A and Party B, terms and conditions",
            "messages": [], "document_id": "", "document_type": "",
            "type_confidence": 0.0, "extraction_result": {},
            "extraction_confidence": 0.0, "model_used": "", "model_cost": 0.0,
            "requires_review": False, "review_status": "", "guardrail_passed": False,
            "guardrail_issues": [], "final_output": {},
        })
        assert result["document_type"] == "contract"

    @pytest.mark.asyncio
    async def test_detects_report(self, orch):
        result = await orch._detect_type_node({
            "document_content": "Quarterly analysis report with metrics and findings summary",
            "messages": [], "document_id": "", "document_type": "",
            "type_confidence": 0.0, "extraction_result": {},
            "extraction_confidence": 0.0, "model_used": "", "model_cost": 0.0,
            "requires_review": False, "review_status": "", "guardrail_passed": False,
            "guardrail_issues": [], "final_output": {},
        })
        assert result["document_type"] == "report"


class TestExtractionNode:
    @pytest.fixture
    def orch(self):
        return DocumentOrchestrator()

    @pytest.mark.asyncio
    @patch("model_router.llm_client.call_llm", new_callable=AsyncMock, return_value=MOCK_EXTRACTION_RESPONSE)
    async def test_extracts_with_known_type(self, mock_llm, orch):
        state = {
            "document_content": "test content",
            "document_type": "invoice",
            "messages": [], "document_id": "", "type_confidence": 0.9,
            "extraction_result": {}, "extraction_confidence": 0.0,
            "model_used": "", "model_cost": 0.0, "requires_review": False,
            "review_status": "", "guardrail_passed": False,
            "guardrail_issues": [], "final_output": {},
        }
        result = await orch._extract_node(state)
        assert "extraction_result" in result
        assert "extraction_confidence" in result
        assert "model_used" in result

    @pytest.mark.asyncio
    async def test_extracts_with_unknown_type(self, orch):
        state = {
            "document_content": "test content",
            "document_type": "unknown",
            "messages": [], "document_id": "", "type_confidence": 0.5,
            "extraction_result": {}, "extraction_confidence": 0.0,
            "model_used": "", "model_cost": 0.0, "requires_review": False,
            "review_status": "", "guardrail_passed": False,
            "guardrail_issues": [], "final_output": {},
        }
        result = await orch._extract_node(state)
        assert result["extraction_result"] == {}
        assert result["extraction_confidence"] == 0.0


class TestGuardrailsNode:
    @pytest.fixture
    def orch(self):
        return DocumentOrchestrator()

    @pytest.mark.asyncio
    async def test_guardrails_with_empty_extraction(self, orch):
        state = {
            "extraction_result": {},
            "document_type": "invoice",
            "document_content": "test",
            "messages": [], "document_id": "", "type_confidence": 0.0,
            "extraction_confidence": 0.0, "model_used": "", "model_cost": 0.0,
            "requires_review": False, "review_status": "", "guardrail_passed": False,
            "guardrail_issues": [], "final_output": {},
        }
        result = await orch._guardrails_node(state)
        assert "guardrail_passed" in result
        assert "guardrail_issues" in result


class TestFullPipeline:
    @pytest.fixture
    def orch(self):
        return DocumentOrchestrator()

    @pytest.mark.asyncio
    @patch("model_router.llm_client.call_llm", new_callable=AsyncMock, return_value=MOCK_EXTRACTION_RESPONSE)
    async def test_run_returns_expected_keys(self, mock_llm, orch):
        content = json.dumps({
            "vendor_name": "Test Corp",
            "invoice_number": "INV-001",
            "date": "2026-01-15",
            "total": 1000,
        })
        result = await orch.run("test_doc_1", content)

        assert result["document_id"] == "test_doc_1"
        assert "document_type" in result
        assert "extraction" in result
        assert "confidence" in result
        assert "model_used" in result
        assert "cost" in result
        assert "guardrail_passed" in result
        assert "requires_review" in result

    @pytest.mark.asyncio
    @patch("model_router.llm_client.call_llm", new_callable=AsyncMock, return_value=MOCK_EXTRACTION_RESPONSE)
    async def test_run_detects_type(self, mock_llm, orch):
        content = "Invoice from Acme Corp for web development services"
        result = await orch.run("doc_inv", content)
        assert result["document_type"] in ["invoice", "receipt", "contract", "report"]

    @pytest.mark.asyncio
    @patch("model_router.llm_client.call_llm", new_callable=AsyncMock, return_value=MOCK_EXTRACTION_RESPONSE)
    async def test_run_records_cost(self, mock_llm, orch):
        initial_cost = orch.cost_tracker.total_cost
        content = "Receipt from store, total $50"
        await orch.run("doc_cost", content)
        assert orch.cost_tracker.docs_processed >= 1


class TestSequentialVsParallelComparison:
    """Compare sequential and parallel orchestrators on the same input."""

    @pytest.fixture
    def seq_orch(self):
        return DocumentOrchestrator()

    @pytest.fixture
    def par_orch(self):
        return ParallelDocumentOrchestrator()

    @pytest.mark.asyncio
    @patch("agents.guardrail_agent.check_content_safety", new_callable=AsyncMock, return_value=MagicMock(is_safe=True, label="safe", error=None))
    @patch("model_router.llm_client.call_llm", new_callable=AsyncMock, return_value=MOCK_EXTRACTION_RESPONSE)
    async def test_same_document_type_detected(self, mock_llm, mock_safety, seq_orch, par_orch):
        content = "Invoice from Acme Corp, amount due $5000, payment due net 30"
        seq_result = await seq_orch.run("seq_doc", content)
        par_result = await par_orch.run("par_doc", content)
        assert seq_result["document_type"] == par_result["document_type"]

    @pytest.mark.asyncio
    @patch("agents.guardrail_agent.check_content_safety", new_callable=AsyncMock, return_value=MagicMock(is_safe=True, label="safe", error=None))
    @patch("model_router.llm_client.call_llm", new_callable=AsyncMock, return_value=MOCK_EXTRACTION_RESPONSE)
    async def test_both_return_extraction(self, mock_llm, mock_safety, seq_orch, par_orch):
        content = "Invoice from Acme Corp, amount due $5000"
        seq_result = await seq_orch.run("seq_doc2", content)
        par_result = await par_orch.run("par_doc2", content)
        assert seq_result["extraction"] == par_result["extraction"]

    @pytest.mark.asyncio
    @patch("agents.guardrail_agent.check_content_safety", new_callable=AsyncMock, return_value=MagicMock(is_safe=True, label="safe", error=None))
    @patch("model_router.llm_client.call_llm", new_callable=AsyncMock, return_value=MOCK_EXTRACTION_RESPONSE)
    async def test_both_have_same_output_structure(self, mock_llm, mock_safety, seq_orch, par_orch):
        content = "Invoice from Acme Corp, amount due $5000"
        seq_result = await seq_orch.run("seq_doc3", content)
        par_result = await par_orch.run("par_doc3", content)
        for key in ["document_id", "document_type", "extraction", "confidence", "model_used", "cost", "guardrail_passed", "requires_review"]:
            assert key in seq_result, f"Sequential missing key: {key}"
            assert key in par_result, f"Parallel missing key: {key}"

    @pytest.mark.asyncio
    @patch("agents.guardrail_agent.check_content_safety", new_callable=AsyncMock, return_value=MagicMock(is_safe=True, label="safe", error=None))
    @patch("model_router.llm_client.call_llm", new_callable=AsyncMock, return_value=MOCK_EXTRACTION_RESPONSE)
    async def test_parallel_has_all_agent_results(self, mock_llm, mock_safety, seq_orch, par_orch):
        content = "Invoice from Acme Corp, amount due $5000"
        seq_result = await seq_orch.run("seq_doc4", content)
        par_result = await par_orch.run("par_doc4", content)
        assert "all_agent_results" not in seq_result
        assert "all_agent_results" in par_result
        assert len(par_result["all_agent_results"]) == 4

    @pytest.mark.asyncio
    @patch("agents.guardrail_agent.check_content_safety", new_callable=AsyncMock, return_value=MagicMock(is_safe=True, label="safe", error=None))
    @patch("model_router.llm_client.call_llm", new_callable=AsyncMock, return_value=MOCK_EXTRACTION_RESPONSE)
    async def test_parallel_extracts_all_agent_types(self, mock_llm, mock_safety, par_orch):
        content = "Invoice from Acme Corp, amount due $5000"
        par_result = await par_orch.run("par_doc5", content)
        agent_types = {r["agent_type"] for r in par_result["all_agent_results"]}
        assert agent_types == {"invoice", "receipt", "contract", "report"}

    @pytest.mark.asyncio
    @patch("agents.guardrail_agent.check_content_safety", new_callable=AsyncMock, return_value=MagicMock(is_safe=True, label="safe", error=None))
    @patch("model_router.llm_client.call_llm", new_callable=AsyncMock, return_value=MOCK_EXTRACTION_RESPONSE)
    async def test_parallel_selects_best_confidence(self, mock_llm, mock_safety, par_orch):
        content = "Invoice from Acme Corp, amount due $5000"
        par_result = await par_orch.run("par_doc6", content)
        if par_result["all_agent_results"]:
            max_conf = max(r["confidence"] for r in par_result["all_agent_results"])
            assert par_result["confidence"] == max_conf


class TestBackwardCompatibility:
    """Ensure the sequential orchestrator is unaffected by parallel additions."""

    @pytest.fixture
    def orch(self):
        return DocumentOrchestrator()

    @pytest.mark.asyncio
    @patch("model_router.llm_client.call_llm", new_callable=AsyncMock, return_value=MOCK_EXTRACTION_RESPONSE)
    async def test_sequential_still_works(self, mock_llm, orch):
        content = "Invoice from Acme Corp, amount due $5000"
        result = await orch.run("compat_doc", content)
        assert result["document_id"] == "compat_doc"
        assert result["document_type"] in ["invoice", "receipt", "contract", "report"]
        assert "extraction" in result

    @pytest.mark.asyncio
    @patch("model_router.llm_client.call_llm", new_callable=AsyncMock, return_value=MOCK_EXTRACTION_RESPONSE)
    async def test_sequential_output_keys_unchanged(self, mock_llm, orch):
        content = "Invoice from Acme Corp, amount due $5000"
        result = await orch.run("compat_doc2", content)
        expected_keys = {"document_id", "document_type", "extraction", "confidence", "model_used", "cost", "guardrail_passed", "requires_review"}
        assert set(result.keys()) == expected_keys

    @pytest.mark.asyncio
    @patch("model_router.llm_client.call_llm", new_callable=AsyncMock, return_value=MOCK_EXTRACTION_RESPONSE)
    async def test_sequential_cost_tracking_unchanged(self, mock_llm, orch):
        initial_cost = orch.cost_tracker.total_cost
        initial_docs = orch.cost_tracker.docs_processed
        content = "Invoice from Acme Corp, amount due $5000"
        await orch.run("compat_doc3", content)
        assert orch.cost_tracker.total_cost >= initial_cost
        assert orch.cost_tracker.docs_processed == initial_docs + 1

    @pytest.mark.asyncio
    @patch("model_router.llm_client.call_llm", new_callable=AsyncMock, return_value=MOCK_EXTRACTION_RESPONSE)
    async def test_sequential_guardrails_still_run(self, mock_llm, orch):
        content = "Invoice from Acme Corp, amount due $5000"
        result = await orch.run("compat_doc4", content)
        assert "guardrail_passed" in result
        assert isinstance(result["guardrail_passed"], bool)

    @pytest.mark.asyncio
    async def test_sequential_graph_structure_unchanged(self, orch):
        nodes = list(orch.graph.get_graph().nodes)
        assert "detect_type" in nodes
        assert "extract" in nodes
        assert "guardrails" in nodes
        assert "finalize" in nodes
        assert "parallel_extract" not in nodes
        assert "select_best" not in nodes
        assert "aggregate_results" not in nodes
