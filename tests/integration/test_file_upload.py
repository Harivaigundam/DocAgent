"""Integration tests for file upload with different file types."""
import pytest
import json
from unittest.mock import AsyncMock, patch
from httpx import AsyncClient, ASGITransport
from app import app

MOCK_LLM_RESPONSE = json.dumps({
    "vendor_name": "Acme Corp",
    "invoice_number": "INV-001",
    "date": "2026-01-15",
    "total": 5000,
    "line_items": [],
    "vendor_name_confidence": 0.95,
    "invoice_number_confidence": 0.98,
    "total_confidence": 0.92,
})

MOCK_RECEIPT_RESPONSE = json.dumps({
    "store_name": "Walmart",
    "date": "2026-01-15",
    "total": 50.0,
    "line_items": [],
    "store_name_confidence": 0.95,
    "total_confidence": 0.98,
})


@pytest.fixture
def transport():
    return ASGITransport(app=app)


@pytest.mark.asyncio
@patch("model_router.llm_client.call_llm", new_callable=AsyncMock, return_value=MOCK_LLM_RESPONSE)
async def test_upload_text_file(mock_llm, transport):
    content = "Invoice from Acme Corp, total $5000".encode("utf-8")
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/extract",
            files={"file": ("document.txt", content, "text/plain")},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert "document_type" in data


@pytest.mark.asyncio
@patch("model_router.llm_client.call_llm", new_callable=AsyncMock, return_value=MOCK_LLM_RESPONSE)
async def test_upload_json_file(mock_llm, transport):
    data_dict = {
        "vendor_name": "Test Corp",
        "invoice_number": "INV-001",
        "date": "2026-01-15",
        "total": 1000,
    }
    content = json.dumps(data_dict).encode("utf-8")
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/extract",
            files={"file": ("invoice.json", content, "application/json")},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["document_type"] == "invoice"


@pytest.mark.asyncio
@patch("model_router.llm_client.call_llm", new_callable=AsyncMock, return_value=MOCK_RECEIPT_RESPONSE)
async def test_upload_with_type_endpoint(mock_llm, transport):
    content = "Receipt from Walmart, total $50".encode("utf-8")
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/extract/receipt",
            files={"file": ("receipt.txt", content, "text/plain")},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["document_type"] == "receipt"
