"""Integration tests for the parallel multi-agent execution system."""
import pytest
import json
from unittest.mock import AsyncMock, patch, MagicMock
from parallel_orchestrator import ParallelDocumentOrchestrator, ALL_AGENT_TYPES
from state import DocumentState
from agents import AGENT_MAP
from model_router.cost_tracker import CostTracker


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

MOCK_RECEIPT_RESPONSE = json.dumps({
    "store_name": "Walmart",
    "date": "2026-01-15",
    "total": 45.99,
    "line_items": [],
    "store_name_confidence": 0.95,
    "total_confidence": 0.98,
})

MOCK_CONTRACT_RESPONSE = json.dumps({
    "title": "Service Agreement",
    "parties": ["Acme Corp", "Widget Inc"],
    "effective_date": "2026-01-01",
    "title_confidence": 0.97,
    "parties_confidence": 0.90,
})

MOCK_REPORT_RESPONSE = json.dumps({
    "title": "Q4 Analysis",
    "author": "Analytics Team",
    "date": "2026-01-15",
    "title_confidence": 0.96,
    "author_confidence": 0.93,
})

INVOICE_STATE = {
    "messages": [],
    "document_id": "test_parallel_001",
    "document_content": "Invoice from Acme Corp, amount due $5000, payment due net 30",
    "document_type": "",
    "type_confidence": 0.0,
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


class TestParallelOrchestratorInit:
    def test_creates_router(self):
        orch = ParallelDocumentOrchestrator()
        assert orch.router is not None

    def test_creates_cost_tracker(self):
        orch = ParallelDocumentOrchestrator()
        assert orch.cost_tracker is not None

    def test_creates_compiled_graph(self):
        orch = ParallelDocumentOrchestrator()
        assert orch.graph is not None

    def test_all_agent_types_defined(self):
        assert ALL_AGENT_TYPES == ["invoice", "receipt", "contract", "report"]

    def test_all_agent_types_have_agents(self):
        for agent_type in ALL_AGENT_TYPES:
            assert agent_type in AGENT_MAP, f"No agent for type: {agent_type}"


class TestParallelTypeDetection:
    @pytest.fixture
    def orch(self):
        return ParallelDocumentOrchestrator()

    @pytest.mark.asyncio
    async def test_detects_invoice(self, orch):
        state = {**INVOICE_STATE, "document_content": "Invoice from Acme Corp, amount due $5000, payment due net 30"}
        result = await orch._detect_type_node(state)
        assert result["document_type"] == "invoice"
        assert result["type_confidence"] > 0

    @pytest.mark.asyncio
    async def test_detects_receipt(self, orch):
        state = {**INVOICE_STATE, "document_content": "Receipt from Walmart, purchased items, total paid $50"}
        result = await orch._detect_type_node(state)
        assert result["document_type"] == "receipt"

    @pytest.mark.asyncio
    async def test_detects_contract(self, orch):
        state = {**INVOICE_STATE, "document_content": "Service agreement contract between Party A and Party B, terms and conditions"}
        result = await orch._detect_type_node(state)
        assert result["document_type"] == "contract"

    @pytest.mark.asyncio
    async def test_detects_report(self, orch):
        state = {**INVOICE_STATE, "document_content": "Quarterly analysis report with metrics and findings summary"}
        result = await orch._detect_type_node(state)
        assert result["document_type"] == "report"


class TestParallelRouting:
    @pytest.fixture
    def orch(self):
        return ParallelDocumentOrchestrator()

    def test_routes_to_all_agents(self, orch):
        state = {**INVOICE_STATE, "document_type": "invoice"}
        sends = orch._route_to_agents(state)
        assert len(sends) == len(ALL_AGENT_TYPES)
        sent_types = [s.arg["document_type"] for s in sends]
        assert set(sent_types) == set(ALL_AGENT_TYPES)

    def test_each_send_contains_agent_type(self, orch):
        state = {**INVOICE_STATE, "document_type": "invoice"}
        sends = orch._route_to_agents(state)
        for send in sends:
            assert "parallel_agent_type" in send.arg
            assert send.arg["parallel_agent_type"] in ALL_AGENT_TYPES

    def test_each_send_preserves_document_content(self, orch):
        state = {**INVOICE_STATE, "document_type": "invoice"}
        sends = orch._route_to_agents(state)
        for send in sends:
            assert send.arg["document_content"] == state["document_content"]


class TestAgentExtractionNode:
    @pytest.fixture
    def orch(self):
        return ParallelDocumentOrchestrator()

    @pytest.mark.asyncio
    @patch("model_router.llm_client.call_llm", new_callable=AsyncMock, return_value=MOCK_EXTRACTION_RESPONSE)
    async def test_extracts_with_known_type(self, mock_llm, orch):
        state = {
            **INVOICE_STATE,
            "document_type": "invoice",
            "parallel_agent_type": "invoice",
        }
        result = await orch._agent_extract_node(state)
        assert "parallel_results" in result
        assert len(result["parallel_results"]) == 1
        agent_result = result["parallel_results"][0]
        assert agent_result["agent_type"] == "invoice"
        assert agent_result["confidence"] > 0
        assert agent_result["model_used"] != ""

    @pytest.mark.asyncio
    async def test_extracts_with_unknown_type(self, orch):
        state = {
            **INVOICE_STATE,
            "document_type": "unknown",
            "parallel_agent_type": "unknown",
        }
        result = await orch._agent_extract_node(state)
        assert "parallel_results" in result
        assert len(result["parallel_results"]) == 1
        agent_result = result["parallel_results"][0]
        assert agent_result["confidence"] == 0.0
        assert "No agent for type" in agent_result["error"]

    @pytest.mark.asyncio
    @patch("model_router.llm_client.call_llm", new_callable=AsyncMock, return_value=MOCK_RECEIPT_RESPONSE)
    async def test_extracts_receipt_type(self, mock_llm, orch):
        state = {
            **INVOICE_STATE,
            "document_type": "receipt",
            "parallel_agent_type": "receipt",
        }
        result = await orch._agent_extract_node(state)
        agent_result = result["parallel_results"][0]
        assert agent_result["agent_type"] == "receipt"


class TestSelectBestNode:
    @pytest.fixture
    def orch(self):
        return ParallelDocumentOrchestrator()

    @pytest.mark.asyncio
    async def test_selects_highest_confidence(self, orch):
        state = {
            **INVOICE_STATE,
            "parallel_results": [
                {"agent_type": "invoice", "fields": {"total": 1000}, "confidence": 0.6, "model_used": "m1", "cost": 0.01, "requires_review": True},
                {"agent_type": "receipt", "fields": {"total": 50}, "confidence": 0.9, "model_used": "m2", "cost": 0.02, "requires_review": False},
                {"agent_type": "contract", "fields": {}, "confidence": 0.3, "model_used": "m3", "cost": 0.005, "requires_review": True},
            ],
        }
        result = await orch._select_best_node(state)
        assert result["document_type"] == "receipt"
        assert result["extraction_confidence"] == 0.9
        assert result["extraction_result"] == {"total": 50}
        assert result["model_used"] == "m2"
        assert result["model_cost"] == 0.02

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_results(self, orch):
        state = {**INVOICE_STATE, "parallel_results": []}
        result = await orch._select_best_node(state)
        assert result["extraction_result"] == {}
        assert result["extraction_confidence"] == 0.0
        assert result["requires_review"] is True

    @pytest.mark.asyncio
    async def test_single_result(self, orch):
        state = {
            **INVOICE_STATE,
            "parallel_results": [
                {"agent_type": "invoice", "fields": {"total": 1000}, "confidence": 0.85, "model_used": "m1", "cost": 0.01, "requires_review": False},
            ],
        }
        result = await orch._select_best_node(state)
        assert result["document_type"] == "invoice"
        assert result["extraction_confidence"] == 0.85

    @pytest.mark.asyncio
    async def test_all_results_preserved(self, orch):
        results = [
            {"agent_type": "invoice", "fields": {}, "confidence": 0.5, "model_used": "m1", "cost": 0.01, "requires_review": True},
            {"agent_type": "receipt", "fields": {}, "confidence": 0.8, "model_used": "m2", "cost": 0.02, "requires_review": False},
        ]
        state = {**INVOICE_STATE, "parallel_results": results}
        result = await orch._select_best_node(state)
        assert len(result["all_results"]) == 2


class TestGuardrailsNode:
    @pytest.fixture
    def orch(self):
        return ParallelDocumentOrchestrator()

    @pytest.mark.asyncio
    async def test_guardrails_with_valid_extraction(self, orch):
        state = {
            **INVOICE_STATE,
            "extraction_result": {"vendor_name": "Acme", "invoice_number": "INV-001", "date": "2026-01-15", "total": 1000},
            "document_type": "invoice",
            "document_content": "Invoice from Acme Corp",
        }
        result = await orch._guardrails_node(state)
        assert "guardrail_passed" in result
        assert "guardrail_issues" in result

    @pytest.mark.asyncio
    async def test_guardrails_with_empty_extraction(self, orch):
        state = {
            **INVOICE_STATE,
            "extraction_result": {},
            "document_type": "invoice",
            "document_content": "test",
        }
        result = await orch._guardrails_node(state)
        assert result["guardrail_passed"] is False
        assert len(result["guardrail_issues"]) > 0


class TestFinalizeNode:
    @pytest.fixture
    def orch(self):
        return ParallelDocumentOrchestrator()

    @pytest.mark.asyncio
    async def test_finalize_records_cost(self, orch):
        state = {
            **INVOICE_STATE,
            "model_used": "openai:gpt-4o",
            "model_cost": 0.01,
            "extraction_result": {"vendor_name": "Acme"},
        }
        initial_cost = orch.cost_tracker.total_cost
        result = await orch._finalize_node(state)
        assert orch.cost_tracker.total_cost == initial_cost + 0.01
        assert result["final_output"] == {"vendor_name": "Acme"}

    @pytest.mark.asyncio
    async def test_finalize_increments_docs_processed(self, orch):
        initial_count = orch.cost_tracker.docs_processed
        state = {
            **INVOICE_STATE,
            "model_used": "openai:gpt-4o",
            "model_cost": 0.01,
            "extraction_result": {},
        }
        await orch._finalize_node(state)
        assert orch.cost_tracker.docs_processed == initial_count + 1


class TestFullParallelPipeline:
    @pytest.fixture
    def orch(self):
        return ParallelDocumentOrchestrator()

    @pytest.mark.asyncio
    @patch("agents.guardrail_agent.check_content_safety", new_callable=AsyncMock, return_value=MagicMock(is_safe=True, label="safe", error=None))
    @patch("model_router.llm_client.call_llm", new_callable=AsyncMock, return_value=MOCK_EXTRACTION_RESPONSE)
    async def test_run_returns_expected_keys(self, mock_llm, mock_safety, orch):
        content = "Invoice from Acme Corp, amount due $5000, payment due net 30"
        result = await orch.run("test_parallel_001", content)
        assert result["document_id"] == "test_parallel_001"
        assert "document_type" in result
        assert "extraction" in result
        assert "confidence" in result
        assert "model_used" in result
        assert "cost" in result
        assert "guardrail_passed" in result
        assert "requires_review" in result
        assert "all_agent_results" in result

    @pytest.mark.asyncio
    @patch("agents.guardrail_agent.check_content_safety", new_callable=AsyncMock, return_value=MagicMock(is_safe=True, label="safe", error=None))
    @patch("model_router.llm_client.call_llm", new_callable=AsyncMock, return_value=MOCK_EXTRACTION_RESPONSE)
    async def test_run_invokes_all_agents(self, mock_llm, mock_safety, orch):
        content = "Invoice from Acme Corp for web development services"
        result = await orch.run("doc_inv", content)
        assert len(result["all_agent_results"]) == len(ALL_AGENT_TYPES)
        agent_types = {r["agent_type"] for r in result["all_agent_results"]}
        assert agent_types == set(ALL_AGENT_TYPES)

    @pytest.mark.asyncio
    @patch("agents.guardrail_agent.check_content_safety", new_callable=AsyncMock, return_value=MagicMock(is_safe=True, label="safe", error=None))
    @patch("model_router.llm_client.call_llm", new_callable=AsyncMock, return_value=MOCK_EXTRACTION_RESPONSE)
    async def test_run_detects_type(self, mock_llm, mock_safety, orch):
        content = "Invoice from Acme Corp for web development services"
        result = await orch.run("doc_inv", content)
        assert result["document_type"] in ALL_AGENT_TYPES

    @pytest.mark.asyncio
    @patch("agents.guardrail_agent.check_content_safety", new_callable=AsyncMock, return_value=MagicMock(is_safe=True, label="safe", error=None))
    @patch("model_router.llm_client.call_llm", new_callable=AsyncMock, return_value=MOCK_EXTRACTION_RESPONSE)
    async def test_run_records_cost(self, mock_llm, mock_safety, orch):
        initial_cost = orch.cost_tracker.total_cost
        content = "Receipt from store, total $50"
        await orch.run("doc_cost", content)
        assert orch.cost_tracker.total_cost >= initial_cost

    @pytest.mark.asyncio
    @patch("agents.guardrail_agent.check_content_safety", new_callable=AsyncMock, return_value=MagicMock(is_safe=True, label="safe", error=None))
    @patch("model_router.llm_client.call_llm", new_callable=AsyncMock, return_value=MOCK_EXTRACTION_RESPONSE)
    async def test_run_selects_best_by_confidence(self, mock_llm, mock_safety, orch):
        content = "Invoice from Acme Corp, amount due $5000, payment due net 30"
        result = await orch.run("doc_best", content)
        if result["all_agent_results"]:
            max_conf = max(r["confidence"] for r in result["all_agent_results"])
            assert result["confidence"] == max_conf


class TestCostTrackingParallel:
    @pytest.fixture
    def orch(self):
        return ParallelDocumentOrchestrator()

    @pytest.mark.asyncio
    @patch("agents.guardrail_agent.check_content_safety", new_callable=AsyncMock, return_value=MagicMock(is_safe=True, label="safe", error=None))
    @patch("model_router.llm_client.call_llm", new_callable=AsyncMock, return_value=MOCK_EXTRACTION_RESPONSE)
    async def test_cost_tracker_accumulates(self, mock_llm, mock_safety, orch):
        await orch.run("doc1", "Invoice from Acme Corp")
        await orch.run("doc2", "Receipt from Walmart")
        assert orch.cost_tracker.docs_processed >= 2

    @pytest.mark.asyncio
    async def test_cost_tracker_summary(self, orch):
        orch.cost_tracker.record("model_a", 0.01)
        orch.cost_tracker.record("model_b", 0.02)
        summary = orch.cost_tracker.summary()
        assert summary["total_cost"] == 0.03
        assert summary["docs_processed"] == 2
        assert "model_a" in summary["model_usage"]
        assert "model_b" in summary["model_usage"]


class TestWebSocketUpdatesDuringParallel:
    """Test that WebSocket updates would be sent during parallel execution."""

    @pytest.mark.asyncio
    async def test_parallel_execution_provides_enough_data_for_ws_updates(self):
        orch = ParallelDocumentOrchestrator()
        state = {
            **INVOICE_STATE,
            "parallel_results": [
                {"agent_type": "invoice", "fields": {"total": 1000}, "confidence": 0.85, "model_used": "m1", "cost": 0.01, "requires_review": False},
                {"agent_type": "receipt", "fields": {"total": 50}, "confidence": 0.9, "model_used": "m2", "cost": 0.02, "requires_review": False},
            ],
        }
        result = await orch._select_best_node(state)
        ws_payload = {
            "event": "parallel_complete",
            "document_type": result["document_type"],
            "confidence": result["extraction_confidence"],
            "agents_completed": len(state["parallel_results"]),
            "all_results": state["parallel_results"],
        }
        assert ws_payload["event"] == "parallel_complete"
        assert ws_payload["agents_completed"] == 2
        assert ws_payload["confidence"] == 0.9
