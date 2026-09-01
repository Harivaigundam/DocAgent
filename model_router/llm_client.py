"""Unified LLM client that routes to Groq, Ollama, or OpenAI."""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)


async def call_llm(
    model: str,
    prompt: str,
    provider: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 4096,
) -> str:
    """Call an LLM and return the raw response text.

    Routes to the appropriate provider based on the provider string.
    """
    if provider == "groq":
        return await _call_groq(model, prompt, api_key, temperature, max_tokens)
    elif provider == "ollama":
        return await _call_ollama(model, prompt, base_url, temperature, max_tokens)
    elif provider == "openai":
        return await _call_openai(model, prompt, api_key, base_url, temperature, max_tokens)
    else:
        raise ValueError(f"Unknown provider: {provider}")


async def _call_groq(
    model: str, prompt: str, api_key: Optional[str], temperature: float, max_tokens: int
) -> str:
    """Call Groq API."""
    from groq import Groq

    key = api_key
    if not key:
        from config import get_settings
        key = get_settings().groq_api_key.get_secret_value()

    if not key:
        raise ValueError("No Groq API key configured")

    client = Groq(api_key=key)
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return completion.choices[0].message.content or ""


async def _call_ollama(
    model: str, prompt: str, base_url: Optional[str], temperature: float, max_tokens: int
) -> str:
    """Call Ollama API."""
    import httpx

    url = (base_url or "http://localhost:11434") + "/api/chat"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data.get("message", {}).get("content", "")


async def _call_openai(
    model: str, prompt: str, api_key: Optional[str], base_url: Optional[str], temperature: float, max_tokens: int
) -> str:
    """Call OpenAI API."""
    from openai import AsyncOpenAI

    key = api_key
    url = base_url
    if not key or not url:
        from config import get_settings
        settings = get_settings()
        key = key or settings.openai_api_key.get_secret_value()
        url = url or settings.openai_base_url

    if not key:
        raise ValueError("No OpenAI API key configured")

    client_kwargs: dict[str, Any] = {"api_key": key}
    if url:
        client_kwargs["base_url"] = url

    client = AsyncOpenAI(**client_kwargs)
    completion = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return completion.choices[0].message.content or ""


def parse_json_response(text: str) -> dict:
    """Extract a JSON object from LLM response text.

    Handles cases where the LLM wraps JSON in markdown code blocks
    or adds extra text around the JSON.
    """
    text = text.strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from code blocks
    json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try finding first { ... } block
    depth = 0
    start = -1
    for i, c in enumerate(text):
        if c == "{":
            if depth == 0:
                start = i
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    start = -1

    raise ValueError(f"No valid JSON found in response: {text[:200]}")
