"""Tests for model_router.llm_client."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from model_router.llm_client import (
    _call_groq,
    _call_ollama,
    _call_openai,
    call_llm,
    parse_json_response,
)


# ---------------------------------------------------------------------------
# parse_json_response tests
# ---------------------------------------------------------------------------


class TestParseJsonResponse:
    def test_direct_json(self):
        result = parse_json_response('{"key": "value", "num": 42}')
        assert result == {"key": "value", "num": 42}

    def test_direct_json_with_whitespace(self):
        result = parse_json_response('  \n  {"key": "value"}  \n  ')
        assert result == {"key": "value"}

    def test_markdown_code_block(self):
        text = 'Here is the result:\n```json\n{"status": "ok"}\n```\nDone.'
        result = parse_json_response(text)
        assert result == {"status": "ok"}

    def test_markdown_code_block_without_language(self):
        text = 'Some text\n```\n{"status": "ok"}\n```'
        result = parse_json_response(text)
        assert result == {"status": "ok"}

    def test_extra_text_around_json(self):
        text = 'The extracted data is: {"name": "test"} as shown above.'
        result = parse_json_response(text)
        assert result == {"name": "test"}

    def test_nested_json(self):
        inner = {"a": {"b": "c"}}
        text = f'Prefix: {json.dumps(inner)} suffix'
        result = parse_json_response(text)
        assert result == inner

    def test_raises_on_invalid_input(self):
        with pytest.raises(ValueError, match="No valid JSON found"):
            parse_json_response("no json here at all")

    def test_raises_on_garbage(self):
        with pytest.raises(ValueError, match="No valid JSON found"):
            parse_json_response("just a random string with no braces")

    def test_brace_fallback_finds_outermost_json(self):
        text = 'prefix {"a": {"b": 1}} suffix'
        result = parse_json_response(text)
        assert result == {"a": {"b": 1}}


# ---------------------------------------------------------------------------
# call_llm routing tests
# ---------------------------------------------------------------------------


class TestCallLlmRouting:
    @pytest.mark.asyncio
    async def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown provider: superai"):
            await call_llm(model="m", prompt="p", provider="superai")

    @pytest.mark.asyncio
    async def test_routes_to_groq(self):
        with patch("model_router.llm_client._call_groq", new_callable=AsyncMock) as mock:
            mock.return_value = "groq response"
            result = await call_llm(model="llama3-70b", prompt="hi", provider="groq")
            assert result == "groq response"
            mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_routes_to_ollama(self):
        with patch("model_router.llm_client._call_ollama", new_callable=AsyncMock) as mock:
            mock.return_value = "ollama response"
            result = await call_llm(model="llama3.1:8b", prompt="hi", provider="ollama")
            assert result == "ollama response"
            mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_routes_to_openai(self):
        with patch("model_router.llm_client._call_openai", new_callable=AsyncMock) as mock:
            mock.return_value = "openai response"
            result = await call_llm(model="gpt-4o", prompt="hi", provider="openai")
            assert result == "openai response"
            mock.assert_called_once()


# ---------------------------------------------------------------------------
# _call_groq tests
# ---------------------------------------------------------------------------


class TestCallGroq:
    @pytest.mark.asyncio
    async def test_uses_provided_key(self):
        mock_message = MagicMock()
        mock_message.content = "hello from groq"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]

        with patch("groq.Groq") as MockGroq:
            MockGroq.return_value.chat.completions.create.return_value = mock_completion
            result = await _call_groq("model", "prompt", "test-key", 0.0, 100)
            assert result == "hello from groq"
            MockGroq.assert_called_once_with(api_key="test-key")

    @pytest.mark.asyncio
    async def test_uses_settings_key_when_none(self):
        mock_message = MagicMock()
        mock_message.content = "groq from settings"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]

        mock_settings = MagicMock()
        mock_settings.groq_api_key.get_secret_value.return_value = "sk-from-settings"

        with patch("groq.Groq") as MockGroq, \
             patch.dict("sys.modules", {
                 "config": MagicMock(get_settings=lambda: mock_settings)
             }):
            MockGroq.return_value.chat.completions.create.return_value = mock_completion
            result = await _call_groq("model", "prompt", None, 0.0, 100)
            assert result == "groq from settings"
            MockGroq.assert_called_once_with(api_key="sk-from-settings")

    @pytest.mark.asyncio
    async def test_raises_when_no_key(self):
        mock_settings = MagicMock()
        mock_settings.groq_api_key.get_secret_value.return_value = None

        with patch.dict("sys.modules", {
            "config": MagicMock(get_settings=lambda: mock_settings)
        }):
            with pytest.raises(ValueError, match="No Groq API key configured"):
                await _call_groq("model", "prompt", None, 0.0, 100)


# ---------------------------------------------------------------------------
# _call_ollama tests
# ---------------------------------------------------------------------------


class TestCallOllama:
    @pytest.mark.asyncio
    async def test_calls_localhost_by_default(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {"message": {"content": "ollama says hi"}}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response

        with patch("httpx.AsyncClient") as MockClient:
            MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await _call_ollama("llama3.1:8b", "hello", None, 0.0, 512)
            assert result == "ollama says hi"

            call_args = mock_client.post.call_args
            assert "localhost:11434" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_uses_custom_base_url(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {"message": {"content": "remote ollama"}}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response

        with patch("httpx.AsyncClient") as MockClient:
            MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await _call_ollama("llama3.1:8b", "hello", "http://remote:9999", 0.5, 256)
            assert result == "remote ollama"

            call_args = mock_client.post.call_args
            assert call_args[0][0] == "http://remote:9999/api/chat"

            payload = call_args[1]["json"]
            assert payload["model"] == "llama3.1:8b"
            assert payload["options"]["temperature"] == 0.5
            assert payload["options"]["num_predict"] == 256


# ---------------------------------------------------------------------------
# _call_openai tests
# ---------------------------------------------------------------------------


class TestCallOpenai:
    @pytest.mark.asyncio
    async def test_uses_provided_key(self):
        mock_message = MagicMock()
        mock_message.content = "openai reply"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]

        with patch("openai.AsyncOpenAI") as MockOA:
            mock_instance = AsyncMock()
            mock_instance.chat.completions.create = AsyncMock(return_value=mock_completion)
            MockOA.return_value = mock_instance

            result = await _call_openai("gpt-4o", "hi", "sk-test", 0.0, 100)
            assert result == "openai reply"
            MockOA.assert_called_once_with(api_key="sk-test")

    @pytest.mark.asyncio
    async def test_raises_when_no_key(self):
        mock_settings = MagicMock()
        mock_settings.openai_api_key.get_secret_value.return_value = None

        with patch.dict("sys.modules", {
            "config": MagicMock(get_settings=lambda: mock_settings)
        }):
            with pytest.raises(ValueError, match="No OpenAI API key configured"):
                await _call_openai("gpt-4o", "hi", None, 0.0, 100)
