"""Integration tests for FastAPI endpoints."""
import pytest
import json
from unittest.mock import AsyncMock, patch
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


@pytest.mark.asyncio
async def test_health_endpoint(transport):
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"


@pytest.mark.asyncio
async def test_models_endpoint(transport):
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/models")
    assert resp.status_code == 200
    data = resp.json()
    assert "models" in data
    assert len(data["models"]) >= 4


@pytest.mark.asyncio
async def test_costs_endpoint(transport):
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/costs")
    assert resp.status_code == 200
    data = resp.json()
    assert "total_cost" in data
    assert "docs_processed" in data


@pytest.mark.asyncio
async def test_review_queue_endpoint(transport):
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/review/queue")
    assert resp.status_code == 200
    data = resp.json()
    assert "items" in data
    assert "count" in data


@pytest.mark.asyncio
@patch("model_router.llm_client.call_llm", new_callable=AsyncMock, return_value=MOCK_LLM_RESPONSE)
async def test_extract_endpoint(mock_llm, transport):
    content = json.dumps({
        "vendor_name": "Test Corp",
        "invoice_number": "INV-001",
        "date": "2026-01-15",
        "total": 1000,
    })
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/extract",
            files={"file": ("test.json", content.encode(), "application/json")},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert "document_id" in data
    assert "document_type" in data
    assert "extraction" in data


@pytest.mark.asyncio
@patch("model_router.llm_client.call_llm", new_callable=AsyncMock, return_value=MOCK_RECEIPT_RESPONSE)
async def test_extract_with_type_endpoint(mock_llm, transport):
    content = json.dumps({
        "store_name": "Walmart",
        "date": "2026-01-15",
        "total": 45.99,
    })
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/extract/receipt",
            files={"file": ("receipt.json", content.encode(), "application/json")},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["document_type"] == "receipt"


@pytest.mark.asyncio
async def test_extract_invalid_type(transport):
    content = json.dumps({"test": "data"})
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/extract/invalid_type",
            files={"file": ("test.json", content.encode(), "application/json")},
        )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_submit_review(transport):
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/review/test_doc_123",
            json={"approved": True, "corrections": {}},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "reviewed"


@pytest.mark.asyncio
async def test_root_serves_html(transport):
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/")
    assert resp.status_code == 200
    assert "Documentor" in resp.text
