"""Tests for real LLM extraction flow."""
import pytest
from unittest.mock import patch, MagicMock, AsyncMock


class TestLLMClient:
    """Tests for model_router/llm_client.py"""

    @pytest.mark.asyncio
    async def test_call_llm_unknown_provider_raises(self):
        from model_router.llm_client import call_llm
        with pytest.raises(ValueError, match="Unknown provider"):
            await call_llm("model", "prompt", "unknown_provider")

    def test_parse_json_response_direct(self):
        from model_router.llm_client import parse_json_response
        result = parse_json_response('{"key": "value"}')
        assert result == {"key": "value"}

    def test_parse_json_response_code_block(self):
        from model_router.llm_client import parse_json_response
        text = '```json\n{"key": "value"}\n```'
        result = parse_json_response(text)
        assert result == {"key": "value"}

    def test_parse_json_response_with_surrounding_text(self):
        from model_router.llm_client import parse_json_response
        text = 'Here is the extraction:\n{"key": "value"}\nDone.'
        result = parse_json_response(text)
        assert result == {"key": "value"}

    def test_parse_json_response_no_json_raises(self):
        from model_router.llm_client import parse_json_response
        with pytest.raises(ValueError, match="No valid JSON"):
            parse_json_response("no json here")

    @pytest.mark.asyncio
    async def test_call_groq_with_mock(self):
        from model_router.llm_client import _call_groq
        mock_client = MagicMock()
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock(message=MagicMock(content="test response"))]
        mock_client.chat.completions.create.return_value = mock_completion

        with patch("groq.Groq", return_value=mock_client):
            result = await _call_groq("model", "prompt", "test_key", 0.0, 100)
            assert result == "test response"

    @pytest.mark.asyncio
    async def test_call_ollama_with_mock(self):
        from model_router.llm_client import _call_ollama
        mock_response = MagicMock()
        mock_response.json.return_value = {"message": {"content": "test response"}}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await _call_ollama("model", "prompt", "http://localhost:11434", 0.0, 100)
            assert result == "test response"

    @pytest.mark.asyncio
    async def test_call_openai_with_mock(self):
        from model_router.llm_client import _call_openai
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock(message=MagicMock(content="test response"))]

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_completion)

        with patch("openai.AsyncOpenAI", return_value=mock_client):
            result = await _call_openai("model", "prompt", "test_key", 0.0, 100)
            assert result == "test response"


class TestBaseAgentExtraction:
    """Tests for real extraction in base_agent.py"""

    @pytest.mark.asyncio
    async def test_extract_returns_structured_fields(self):
        from agents.invoice_agent import InvoiceAgent

        with patch("model_router.llm_client.call_llm", new_callable=AsyncMock, return_value='{"vendor_name": "Acme", "invoice_number": "INV-001", "total": 100.0, "date": "2025-01-15", "due_date": "2025-02-15", "line_items": []}'):
            agent = InvoiceAgent()
            result = await agent._extract("Invoice from Acme for $100", {})

            assert "fields" in result
            assert "confidence" in result
            assert "model_used" in result
            assert result["confidence"] >= 0.0

    @pytest.mark.asyncio
    async def test_extract_handles_llm_failure(self):
        from agents.invoice_agent import InvoiceAgent

        with patch("model_router.llm_client.call_llm", new_callable=AsyncMock, side_effect=Exception("API Error")):
            agent = InvoiceAgent()
            result = await agent._extract("test content", {})

            assert result["confidence"] == 0.0
            assert result["requires_review"] is True

    @pytest.mark.asyncio
    async def test_extract_uses_cascade(self):
        from agents.invoice_agent import InvoiceAgent

        call_count = 0

        async def mock_call_llm(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise Exception("API Error")
            return '{"vendor_name": "Test", "invoice_number": "INV-001", "total": 50.0, "date": "2025-01-15", "due_date": "2025-02-15", "line_items": [], "vendor_name_confidence": 0.9, "invoice_number_confidence": 0.95, "total_confidence": 0.92}'

        with patch("model_router.llm_client.call_llm", side_effect=mock_call_llm):
            agent = InvoiceAgent()
            result = await agent._extract("Invoice content", {})

            assert result["fields"] is not None
            # ollama fails, groq fails, openai_mini succeeds and returns with confidence
            assert call_count == 3
