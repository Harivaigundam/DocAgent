"""Tests for application configuration."""
import json
import os
import pytest
from unittest.mock import patch

from config import Settings, get_settings


class TestSettings:
    def test_default_settings_load(self):
        with patch.dict(os.environ, {}, clear=False):
            s = Settings(_env_file=None)
            assert s.confidence_threshold == 0.7
            assert s.max_cost_per_doc == 0.05
            assert s.hitl_enabled is True
            assert s.prompt_guard_enabled is True
            assert s.prompt_guard_model == "meta-llama/llama-prompt-guard-2-86m"
            assert s.prompt_guard_threshold == 0.5
            assert s.log_level == "INFO"
            assert s.ollama_base_url == "http://localhost:11434"

    def test_model_cascade_from_json(self):
        cascade_json = json.dumps([
            {"name": "test", "model": "test-model", "cost_per_doc": 0.001, "confidence_threshold": 0.8, "provider": "test"}
        ])
        with patch.dict(os.environ, {"MODEL_CASCADE": cascade_json}, clear=False):
            s = Settings(_env_file=None)
            cascade = s.model_cascade
            assert len(cascade) == 1
            assert cascade[0].name == "test"
            assert cascade[0].model == "test-model"
            assert cascade[0].cost_per_doc == 0.001
            assert cascade[0].confidence_threshold == 0.8
            assert cascade[0].provider == "test"

    def test_model_cascade_default_when_no_env_var(self):
        with patch.dict(os.environ, {}, clear=False):
            s = Settings(_env_file=None)
            cascade = s.model_cascade
            assert len(cascade) == 5
            assert cascade[0].name == "ollama"
            assert cascade[0].model == "llama3.1:8b"
            assert cascade[0].cost_per_doc == 0.0
            assert cascade[1].name == "groq"
            assert cascade[2].name == "prompt_guard"
            assert cascade[3].name == "openai_mini"
            assert cascade[4].name == "openai"

    def test_secret_fields_masked_in_repr(self):
        from pydantic import SecretStr
        with patch.dict(os.environ, {"GROQ_API_KEY": "test_key_12345"}, clear=False):
            s = Settings(_env_file=None)
            repr_str = repr(s)
            assert "test_key_12345" not in repr_str
            assert "**********" in repr_str

    def test_confidence_threshold_bounds(self):
        with pytest.raises(Exception):
            Settings(_env_file=None, confidence_threshold=1.5)

    def test_prompt_guard_defaults(self):
        with patch.dict(os.environ, {}, clear=False):
            s = Settings(_env_file=None)
            assert s.prompt_guard_threshold == 0.5
            assert s.prompt_guard_enabled is True


class TestGetSettings:
    def test_get_settings_returns_cached_instance(self):
        get_settings.cache_clear()
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2
