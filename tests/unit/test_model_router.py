import asyncio
import random

import pytest

from model_router.router import ModelChoice, ModelRouter, ModelTier


def _run(coro):
    return asyncio.run(coro)


class TestCascadeOrder:
    def test_cascade_starts_with_ollama(self):
        router = ModelRouter()
        assert router.CASCADE[0].name == "ollama"
        assert router.CASCADE[0].model == "llama3.1:8b"
        assert router.CASCADE[0].cost_per_doc == 0.0

    def test_cascade_escalates_to_groq(self):
        router = ModelRouter()
        assert router.CASCADE[1].name == "groq"
        assert router.CASCADE[1].cost_per_doc == 0.0

    def test_cascade_has_prompt_guard(self):
        router = ModelRouter()
        assert router.CASCADE[2].name == "prompt_guard"
        assert router.CASCADE[2].model == "meta-llama/llama-prompt-guard-2-86m"

    def test_cascade_escalates_to_openai_mini(self):
        router = ModelRouter()
        assert router.CASCADE[3].name == "openai_mini"
        assert router.CASCADE[3].cost_per_doc == 0.001

    def test_cascade_ends_with_openai(self):
        router = ModelRouter()
        assert router.CASCADE[-1].name == "openai"
        assert router.CASCADE[-1].cost_per_doc == 0.01


class TestRouting:
    def test_returns_first_tier_when_confidence_high_enough(self):
        random.seed(42)
        router = ModelRouter()
        choice = _run(router.route("some document content"))
        assert isinstance(choice, ModelChoice)
        assert choice.tier in [t.name for t in router.CASCADE]

    def test_escalates_when_confidence_below_threshold(self):
        random.seed(0)
        router = ModelRouter()
        choice = _run(router.route("content"))
        assert choice.tier == "ollama"

    def test_fallback_when_all_below_threshold(self):
        random.seed(1)
        tiers = [
            ModelTier("t1", "m1", 0.0, 0.99, "p1"),
            ModelTier("t2", "m2", 0.01, 0.99, "p2"),
        ]
        router = ModelRouter(cascade=tiers)
        choice = _run(router.route("content"))
        assert 0.0 <= choice.confidence <= 1.0
        assert choice.tier in ["t1", "t2"]

    def test_fallback_to_openai_when_empty_cascade(self):
        router = ModelRouter(cascade=[])
        choice = _run(router.route("content"))
        assert choice.model == "gpt-4o"
        assert choice.cost == 0.01

    def test_route_passes_task_complexity(self):
        random.seed(42)
        router = ModelRouter()
        choice = _run(router.route("content", task_complexity="complex"))
        assert isinstance(choice, ModelChoice)


class TestModelChoice:
    def test_model_choice_fields(self):
        choice = ModelChoice(model="gpt-4o", confidence=0.9, cost=0.01, tier="openai")
        assert choice.model == "gpt-4o"
        assert choice.confidence == 0.9
        assert choice.cost == 0.01
        assert choice.tier == "openai"


class TestModelTier:
    def test_model_tier_fields(self):
        tier = ModelTier("ollama", "llama3.1:8b", 0.0, 0.70, "ollama")
        assert tier.name == "ollama"
        assert tier.model == "llama3.1:8b"
        assert tier.cost_per_doc == 0.0
        assert tier.confidence_threshold == 0.70
        assert tier.provider == "ollama"
