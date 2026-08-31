"""Application configuration using Pydantic Settings.

All config is loaded from environment variables or .env file.
Model cascade is configurable via env vars, not hardcoded.
"""
from __future__ import annotations

import json
from functools import lru_cache
from typing import Optional

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelTierConfig:
    """Represents a single model tier in the cascade."""
    def __init__(self, name: str, model: str, cost_per_doc: float, confidence_threshold: float, provider: str):
        self.name = name
        self.model = model
        self.cost_per_doc = cost_per_doc
        self.confidence_threshold = confidence_threshold
        self.provider = provider


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="",
        case_sensitive=False,
        extra="ignore",
    )

    # API Keys
    groq_api_key: SecretStr = Field(default=SecretStr(""), description="Groq API key")
    openai_api_key: SecretStr = Field(default=SecretStr(""), description="OpenAI API key")
    ollama_base_url: str = Field(default="http://localhost:11434", description="Ollama base URL")

    # Application Settings
    hitl_enabled: bool = Field(default=True, description="Enable human-in-the-loop")
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Minimum confidence threshold")
    max_cost_per_doc: float = Field(default=0.05, ge=0.0, description="Maximum cost per document")
    log_level: str = Field(default="INFO", description="Logging level")

    # Prompt Guard
    prompt_guard_enabled: bool = Field(default=True, description="Enable llama-prompt-guard-2-86m")
    prompt_guard_model: str = Field(default="meta-llama/llama-prompt-guard-2-86m", description="Prompt guard model ID")
    prompt_guard_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Prompt guard malicious threshold")

    # Model Cascade (JSON string or individual env vars)
    # Example: MODEL_CASCADE='[{"name":"ollama","model":"llama3.1:8b","cost_per_doc":0.0,"confidence_threshold":0.7,"provider":"ollama"}]'
    model_cascade_json: Optional[str] = Field(default=None, alias="MODEL_CASCADE", description="Model cascade as JSON array")

    @property
    def model_cascade(self) -> list[ModelTierConfig]:
        """Parse model cascade from env var or use defaults."""
        if self.model_cascade_json:
            try:
                data = json.loads(self.model_cascade_json)
                return [
                    ModelTierConfig(
                        name=t["name"],
                        model=t["model"],
                        cost_per_doc=t.get("cost_per_doc", 0.0),
                        confidence_threshold=t.get("confidence_threshold", 0.7),
                        provider=t.get("provider", "unknown"),
                    )
                    for t in data
                ]
            except (json.JSONDecodeError, KeyError):
                pass

        # Default cascade
        return [
            ModelTierConfig("ollama", "llama3.1:8b", 0.0, 0.70, "ollama"),
            ModelTierConfig("groq", "llama3-70b-8192", 0.0, 0.75, "groq"),
            ModelTierConfig("prompt_guard", "meta-llama/llama-prompt-guard-2-86m", 0.00004, 0.5, "groq"),
            ModelTierConfig("openai_mini", "gpt-4o-mini", 0.001, 0.80, "openai"),
            ModelTierConfig("openai", "gpt-4o", 0.01, 0.85, "openai"),
        ]


@lru_cache
def get_settings() -> Settings:
    """Cached singleton for settings."""
    return Settings()
