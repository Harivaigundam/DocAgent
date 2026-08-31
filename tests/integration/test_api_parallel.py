"""Integration tests for the parallel extraction API endpoint."""
import pytest
import json
from unittest.mock import AsyncMock, patch, MagicMock
from httpx import AsyncClient, ASGITransport
from app import app


MOCK_LLM_RESPONSE = json.dumps({
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


@pytest.fixture
def transport():
    return ASGITransport(app=app)


class TestParallelExtractEndpoint:
    @pytest.mark.asyncio
    @patch("agents.guardrail_agent.check_content_safety", new_callable=AsyncMock, return_value=MagicMock(is_safe=True, label="safe", error=None))
    @patch("model_router.llm_client.call_llm", new_callable=AsyncMock, return_value=MOCK_LLM_RESPONSE)
    async def test_parallel_extract_returns_200(self, mock_llm, mock_safety, transport):
        content = json.dumps({
            "vendor_name": "Test Corp",
            "invoice_number": "INV-001",
            "date": "2026-01-15",
            "total": 1000,
        })
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/extract/parallel",
                files={"file": ("test.json", content.encode(), "application/json")},
            )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    @patch("agents.guardrail_agent.check_content_safety", new_callable=AsyncMock, return_value=MagicMock(is_safe=True, label="safe", error=None))
    @patch("model_router.llm_client.call_llm", new_callable=AsyncMock, return_value=MOCK_LLM_RESPONSE)
    async def test_parallel_extract_returns_expected_keys(self, mock_llm, mock_safety, transport):
        content = json.dumps({
            "vendor_name": "Test Corp",
            "invoice_number": "INV-001",
            "date": "2026-01-15",
            "total": 1000,
        })
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/extract/parallel",
                files={"file": ("test.json", content.encode(), "application/json")},
            )
        data = resp.json()
        assert "document_id" in data
        assert "document_type" in data
        assert "extraction" in data
        assert "confidence" in data
        assert "model_used" in data
        assert "cost" in data
        assert "guardrail_passed" in data
        assert "requires_review" in data
        assert "all_agent_results" in data

    @pytest.mark.asyncio
    @patch("agents.guardrail_agent.check_content_safety", new_callable=AsyncMock, return_value=MagicMock(is_safe=True, label="safe", error=None))
    @patch("model_router.llm_client.call_llm", new_callable=AsyncMock, return_value=MOCK_LLM_RESPONSE)
    async def test_parallel_extract_includes_all_agent_results(self, mock_llm, mock_safety, transport):
        content = "Invoice from Acme Corp, amount due $5000, payment due net 30"
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/extract/parallel",
                files={"file": ("invoice.txt", content.encode(), "text/plain")},
            )
        data = resp.json()
        assert len(data["all_agent_results"]) == 4
        agent_types = {r["agent_type"] for r in data["all_agent_results"]}
        assert agent_types == {"invoice", "receipt", "contract", "report"}

    @pytest.mark.asyncio
    async def test_parallel_extract_with_empty_file(self, transport):
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/extract/parallel",
                files={"file": ("empty.txt", b"", "text/plain")},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert "document_id" in data


class TestParallelFileUpload:
    @pytest.mark.asyncio
    @patch("agents.guardrail_agent.check_content_safety", new_callable=AsyncMock, return_value=MagicMock(is_safe=True, label="safe", error=None))
    @patch("model_router.llm_client.call_llm", new_callable=AsyncMock, return_value=MOCK_LLM_RESPONSE)
    async def test_upload_text_file_parallel(self, mock_llm, mock_safety, transport):
        content = "Invoice from Acme Corp, total $5000".encode("utf-8")
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/extract/parallel",
                files={"file": ("document.txt", content, "text/plain")},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert "document_type" in data
        assert "all_agent_results" in data

    @pytest.mark.asyncio
    @patch("agents.guardrail_agent.check_content_safety", new_callable=AsyncMock, return_value=MagicMock(is_safe=True, label="safe", error=None))
    @patch("model_router.llm_client.call_llm", new_callable=AsyncMock, return_value=MOCK_LLM_RESPONSE)
    async def test_upload_json_file_parallel(self, mock_llm, mock_safety, transport):
        data_dict = {
            "vendor_name": "Test Corp",
            "invoice_number": "INV-001",
            "date": "2026-01-15",
            "total": 1000,
        }
        content = json.dumps(data_dict).encode("utf-8")
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/extract/parallel",
                files={"file": ("invoice.json", content, "application/json")},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["document_type"] in ["invoice", "receipt", "contract", "report"]

    @pytest.mark.asyncio
    @patch("agents.guardrail_agent.check_content_safety", new_callable=AsyncMock, return_value=MagicMock(is_safe=True, label="safe", error=None))
    @patch("model_router.llm_client.call_llm", new_callable=AsyncMock, return_value=MOCK_RECEIPT_RESPONSE)
    async def test_upload_receipt_parallel(self, mock_llm, mock_safety, transport):
        content = "Receipt from Walmart, total $45.99".encode("utf-8")
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/extract/parallel",
                files={"file": ("receipt.txt", content, "text/plain")},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["document_type"] in ["invoice", "receipt", "contract", "report"]


class TestParallelErrorHandling:
    @pytest.mark.asyncio
    async def test_parallel_extract_with_invalid_content_type(self, transport):
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/extract/parallel",
                files={"file": ("test.pdf", b"%PDF-1.4 fake", "application/pdf")},
            )
        assert resp.status_code in (200, 400)
        data = resp.json()
        if resp.status_code == 200:
            assert "document_id" in data
        else:
            assert "error" in data

    @pytest.mark.asyncio
    @patch("agents.guardrail_agent.check_content_safety", new_callable=AsyncMock, return_value=MagicMock(is_safe=True, label="safe", error=None))
    @patch("model_router.llm_client.call_llm", new_callable=AsyncMock, side_effect=Exception("LLM service unavailable"))
    async def test_parallel_extract_handles_llm_failure(self, mock_llm, mock_safety, transport):
        content = "Invoice from Acme Corp"
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/extract/parallel",
                files={"file": ("invoice.txt", content.encode(), "text/plain")},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["confidence"] == 0.0
        assert data["requires_review"] is True


class TestParallelPerformanceComparison:
    @pytest.mark.asyncio
    @patch("agents.guardrail_agent.check_content_safety", new_callable=AsyncMock, return_value=MagicMock(is_safe=True, label="safe", error=None))
    @patch("model_router.llm_client.call_llm", new_callable=AsyncMock, return_value=MOCK_LLM_RESPONSE)
    async def test_parallel_endpoint_returns_all_agent_types(self, mock_llm, mock_safety, transport):
        content = "Invoice from Acme Corp, amount due $5000"
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/extract/parallel",
                files={"file": ("doc.txt", content.encode(), "text/plain")},
            )
        data = resp.json()
        agent_types = {r["agent_type"] for r in data["all_agent_results"]}
        assert len(agent_types) == 4

    @pytest.mark.asyncio
    @patch("agents.guardrail_agent.check_content_safety", new_callable=AsyncMock, return_value=MagicMock(is_safe=True, label="safe", error=None))
    @patch("model_router.llm_client.call_llm", new_callable=AsyncMock, return_value=MOCK_LLM_RESPONSE)
    async def test_parallel_vs_sequential_same_document_type(self, mock_llm, mock_safety, transport):
        content = "Invoice from Acme Corp, amount due $5000"
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            parallel_resp = await client.post(
                "/api/extract/parallel",
                files={"file": ("doc.txt", content.encode(), "text/plain")},
            )
            sequential_resp = await client.post(
                "/api/extract",
                files={"file": ("doc.txt", content.encode(), "text/plain")},
            )
        p_data = parallel_resp.json()
        s_data = sequential_resp.json()
        assert p_data["document_type"] == s_data["document_type"]
        assert "all_agent_results" in p_data
        assert "all_agent_results" not in s_data


class TestParallelWithDifferentDocTypes:
    @pytest.mark.asyncio
    @patch("agents.guardrail_agent.check_content_safety", new_callable=AsyncMock, return_value=MagicMock(is_safe=True, label="safe", error=None))
    @patch("model_router.llm_client.call_llm", new_callable=AsyncMock, return_value=MOCK_LLM_RESPONSE)
    async def test_parallel_invoice(self, mock_llm, mock_safety, transport):
        content = "Invoice from Acme Corp, amount due $5000, payment due net 30"
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/extract/parallel",
                files={"file": ("invoice.txt", content.encode(), "text/plain")},
            )
        assert resp.status_code == 200
        assert resp.json()["document_type"] in ["invoice", "receipt", "contract", "report"]

    @pytest.mark.asyncio
    @patch("agents.guardrail_agent.check_content_safety", new_callable=AsyncMock, return_value=MagicMock(is_safe=True, label="safe", error=None))
    @patch("model_router.llm_client.call_llm", new_callable=AsyncMock, return_value=MOCK_LLM_RESPONSE)
    async def test_parallel_receipt(self, mock_llm, mock_safety, transport):
        content = "Receipt from Walmart, purchased items, total paid $50"
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/extract/parallel",
                files={"file": ("receipt.txt", content.encode(), "text/plain")},
            )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    @patch("agents.guardrail_agent.check_content_safety", new_callable=AsyncMock, return_value=MagicMock(is_safe=True, label="safe", error=None))
    @patch("model_router.llm_client.call_llm", new_callable=AsyncMock, return_value=MOCK_LLM_RESPONSE)
    async def test_parallel_contract(self, mock_llm, mock_safety, transport):
        content = "Service agreement contract between Party A and Party B, terms and conditions"
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/extract/parallel",
                files={"file": ("contract.txt", content.encode(), "text/plain")},
            )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    @patch("agents.guardrail_agent.check_content_safety", new_callable=AsyncMock, return_value=MagicMock(is_safe=True, label="safe", error=None))
    @patch("model_router.llm_client.call_llm", new_callable=AsyncMock, return_value=MOCK_LLM_RESPONSE)
    async def test_parallel_report(self, mock_llm, mock_safety, transport):
        content = "Quarterly analysis report with metrics and findings summary"
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/extract/parallel",
                files={"file": ("report.txt", content.encode(), "text/plain")},
            )
        assert resp.status_code == 200
