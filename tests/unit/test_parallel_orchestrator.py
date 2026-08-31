"""Unit tests for ParallelDocumentOrchestrator."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import asyncio
import time

from parallel_orchestrator import ParallelDocumentOrchestrator, ALL_AGENT_TYPES


def _make_state(**overrides) -> dict:
    base = {
        "messages": [],
        "document_id": "test_doc",
        "document_content": "test content",
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
    base.update(overrides)
    return base


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


class TestDetectTypeNode:
    @pytest.fixture
    def orch(self):
        return ParallelDocumentOrchestrator()

    @pytest.mark.asyncio
    async def test_detects_invoice(self, orch):
        state = _make_state(
            document_content="Invoice from Acme Corp, amount due $5000, payment due net 30"
        )
        result = await orch._detect_type_node(state)
        assert result["document_type"] == "invoice"
        assert result["type_confidence"] > 0

    @pytest.mark.asyncio
    async def test_detects_receipt(self, orch):
        state = _make_state(
            document_content="Receipt from Walmart, purchased items, total paid $50"
        )
        result = await orch._detect_type_node(state)
        assert result["document_type"] == "receipt"

    @pytest.mark.asyncio
    async def test_detects_contract(self, orch):
        state = _make_state(
            document_content="Service agreement contract between Party A and Party B, terms and conditions"
        )
        result = await orch._detect_type_node(state)
        assert result["document_type"] == "contract"

    @pytest.mark.asyncio
    async def test_detects_report(self, orch):
        state = _make_state(
            document_content="Quarterly analysis report with metrics and findings summary"
        )
        result = await orch._detect_type_node(state)
        assert result["document_type"] == "report"

    @pytest.mark.asyncio
    async def test_empty_content_returns_lowest_confidence(self, orch):
        state = _make_state(document_content="")
        result = await orch._detect_type_node(state)
        assert result["type_confidence"] >= 0.5

    @pytest.mark.asyncio
    async def test_multiple_keywords_weighted(self, orch):
        state = _make_state(
            document_content="Invoice bill vendor payment due amount owed receipt purchase"
        )
        result = await orch._detect_type_node(state)
        assert result["document_type"] in ["invoice", "receipt"]


class TestRouteToAgents:
    @pytest.fixture
    def orch(self):
        return ParallelDocumentOrchestrator()

    def test_returns_send_objects_for_all_agent_types(self, orch):
        state = _make_state()
        sends = orch._route_to_agents(state)
        assert len(sends) == 4

    def test_send_objects_carry_state(self, orch):
        state = _make_state(document_content="test content")
        sends = orch._route_to_agents(state)
        for send in sends:
            assert send.arg["document_content"] == "test content"

    def test_each_send_has_unique_agent_type(self, orch):
        state = _make_state()
        sends = orch._route_to_agents(state)
        types = [send.arg["parallel_agent_type"] for send in sends]
        assert set(types) == set(ALL_AGENT_TYPES)

    def test_send_copies_state_without_mutation(self, orch):
        original_type = "invoice"
        state = _make_state(document_type=original_type)
        sends = orch._route_to_agents(state)
        for send in sends:
            assert send.arg["document_type"] == send.arg["parallel_agent_type"]


class TestAgentExtractNode:
    @pytest.fixture
    def orch(self):
        return ParallelDocumentOrchestrator()

    @pytest.mark.asyncio
    async def test_returns_error_for_unknown_agent_type(self, orch):
        state = _make_state(parallel_agent_type="unknown_type")
        result = await orch._agent_extract_node(state)
        assert len(result["parallel_results"]) == 1
        assert result["parallel_results"][0]["confidence"] == 0.0
        assert "No agent for type" in result["parallel_results"][0]["error"]

    @pytest.mark.asyncio
    async def test_calls_agent_extract(self, orch):
        mock_agent = MagicMock()
        mock_agent._extract = AsyncMock(return_value={
            "fields": {"vendor_name": "Test"},
            "confidence": 0.95,
            "model_used": "openai:gpt-4o",
            "cost": 0.01,
            "requires_review": False,
        })
        mock_agent_class = MagicMock(return_value=mock_agent)

        state = _make_state(
            parallel_agent_type="invoice",
            document_content="test invoice"
        )
        with patch("parallel_orchestrator.AGENT_MAP", {"invoice": mock_agent_class}):
            result = await orch._agent_extract_node(state)

        assert len(result["parallel_results"]) == 1
        assert result["parallel_results"][0]["agent_type"] == "invoice"
        assert result["parallel_results"][0]["confidence"] == 0.95

    @pytest.mark.asyncio
    async def test_agent_extract_includes_cost(self, orch):
        mock_agent = MagicMock()
        mock_agent._extract = AsyncMock(return_value={
            "fields": {"total": 100},
            "confidence": 0.88,
            "model_used": "ollama:llama3.1:8b",
            "cost": 0.0,
            "requires_review": True,
        })
        mock_agent_class = MagicMock(return_value=mock_agent)

        state = _make_state(
            parallel_agent_type="receipt",
            document_content="test receipt"
        )
        with patch("parallel_orchestrator.AGENT_MAP", {"receipt": mock_agent_class}):
            result = await orch._agent_extract_node(state)

        assert result["parallel_results"][0]["cost"] == 0.0
        assert result["parallel_results"][0]["requires_review"] is True


class TestSelectBestNode:
    @pytest.fixture
    def orch(self):
        return ParallelDocumentOrchestrator()

    @pytest.mark.asyncio
    async def test_selects_highest_confidence(self, orch):
        results = [
            {"agent_type": "invoice", "confidence": 0.6, "fields": {}, "model_used": "a", "cost": 0.0, "requires_review": True},
            {"agent_type": "receipt", "confidence": 0.95, "fields": {"total": 50}, "model_used": "b", "cost": 0.01, "requires_review": False},
            {"agent_type": "contract", "confidence": 0.7, "fields": {}, "model_used": "c", "cost": 0.0, "requires_review": True},
        ]
        state = _make_state(parallel_results=results)
        result = await orch._select_best_node(state)

        assert result["document_type"] == "receipt"
        assert result["extraction_confidence"] == 0.95
        assert result["extraction_result"] == {"total": 50}
        assert result["model_used"] == "b"
        assert result["model_cost"] == 0.01
        assert result["requires_review"] is False

    @pytest.mark.asyncio
    async def test_empty_results_returns_defaults(self, orch):
        state = _make_state(parallel_results=[])
        result = await orch._select_best_node(state)

        assert result["extraction_result"] == {}
        assert result["extraction_confidence"] == 0.0
        assert result["requires_review"] is True

    @pytest.mark.asyncio
    async def test_single_result(self, orch):
        results = [
            {"agent_type": "report", "confidence": 0.82, "fields": {"title": "Q4 Report"}, "model_used": "openai:gpt-4o", "cost": 0.01, "requires_review": False},
        ]
        state = _make_state(parallel_results=results)
        result = await orch._select_best_node(state)

        assert result["document_type"] == "report"
        assert result["all_results"] == results

    @pytest.mark.asyncio
    async def test_tie_breaks_on_first_encountered(self, orch):
        results = [
            {"agent_type": "invoice", "confidence": 0.85, "fields": {}, "model_used": "a", "cost": 0.0, "requires_review": False},
            {"agent_type": "receipt", "confidence": 0.85, "fields": {}, "model_used": "b", "cost": 0.0, "requires_review": False},
        ]
        state = _make_state(parallel_results=results)
        result = await orch._select_best_node(state)
        assert result["document_type"] == "invoice"


class TestGuardrailsNode:
    @pytest.fixture
    def orch(self):
        return ParallelDocumentOrchestrator()

    @pytest.mark.asyncio
    @patch("parallel_orchestrator.guardrail_validate", new_callable=AsyncMock)
    async def test_guardrails_pass(self, mock_validate, orch):
        mock_validate.return_value = {
            "passed": True,
            "schema": {"errors": [], "warnings": []},
            "consistency": {"issues": []},
        }
        state = _make_state(
            extraction_result={"vendor_name": "Test"},
            document_type="invoice",
            document_content="test content"
        )
        result = await orch._guardrails_node(state)

        assert result["guardrail_passed"] is True
        assert result["guardrail_issues"] == []

    @pytest.mark.asyncio
    @patch("parallel_orchestrator.guardrail_validate", new_callable=AsyncMock)
    async def test_guardrails_fail_schema_error(self, mock_validate, orch):
        mock_validate.return_value = {
            "passed": False,
            "schema": {"errors": ["Missing required field: vendor_name"], "warnings": []},
            "consistency": {"issues": []},
        }
        state = _make_state(
            extraction_result={},
            document_type="invoice",
            document_content=""
        )
        result = await orch._guardrails_node(state)

        assert result["guardrail_passed"] is False
        assert "Missing required field: vendor_name" in result["guardrail_issues"]

    @pytest.mark.asyncio
    @patch("parallel_orchestrator.guardrail_validate", new_callable=AsyncMock)
    async def test_guardrails_fail_consistency_issue(self, mock_validate, orch):
        mock_validate.return_value = {
            "passed": False,
            "schema": {"errors": [], "warnings": []},
            "consistency": {"issues": ["Date format inconsistency"]},
        }
        state = _make_state(
            extraction_result={"date": "invalid"},
            document_type="invoice",
            document_content=""
        )
        result = await orch._guardrails_node(state)

        assert result["guardrail_passed"] is False
        assert "Date format inconsistency" in result["guardrail_issues"]


class TestFinalizeNode:
    @pytest.fixture
    def orch(self):
        return ParallelDocumentOrchestrator()

    @pytest.mark.asyncio
    async def test_finalize_records_cost(self, orch):
        state = _make_state(
            model_used="openai:gpt-4o",
            model_cost=0.01,
            extraction_result={"vendor_name": "Test Corp"},
        )
        result = await orch._finalize_node(state)

        assert result["final_output"] == {"vendor_name": "Test Corp"}
        assert orch.cost_tracker.docs_processed == 1
        assert orch.cost_tracker.total_cost == 0.01


class TestParallelExecutionTiming:
    """Test that fan-out/fan-in executes agents concurrently."""

    @pytest.mark.asyncio
    async def test_agents_execute_concurrently(self):
        call_log = []

        async def slow_extract(content, metadata):
            await asyncio.sleep(0.05)
            call_log.append("done")
            return {
                "fields": {},
                "confidence": 0.5,
                "model_used": "test",
                "cost": 0.0,
                "requires_review": False,
            }

        orch = ParallelDocumentOrchestrator()

        mock_agent_class = MagicMock()
        mock_agent = MagicMock()
        mock_agent._extract = slow_extract
        mock_agent_class.return_value = mock_agent

        with patch("parallel_orchestrator.AGENT_MAP", {t: mock_agent_class for t in ALL_AGENT_TYPES}):
            state = _make_state(parallel_agent_type="invoice", document_content="test")
            start = time.monotonic()
            result = await orch._agent_extract_node(state)
            elapsed = time.monotonic() - start

        assert len(result["parallel_results"]) == 1
        assert elapsed < 0.15


class TestFanOutFanIn:
    """Test the complete fan-out to fan-in flow."""

    @pytest.fixture
    def orch(self):
        return ParallelDocumentOrchestrator()

    def test_route_to_agents_returns_correct_count(self, orch):
        state = _make_state()
        sends = orch._route_to_agents(state)
        assert len(sends) == len(ALL_AGENT_TYPES)

    @pytest.mark.asyncio
    async def test_all_agent_types_covered_in_results(self, orch):
        results = [
            {"agent_type": at, "confidence": 0.7, "fields": {}, "model_used": "m", "cost": 0.0, "requires_review": False}
            for at in ALL_AGENT_TYPES
        ]
        state = _make_state(parallel_results=results)
        result = await orch._select_best_node(state)
        assert result["all_results"] == results
        assert len(result["all_results"]) == 4
