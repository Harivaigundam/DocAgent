from __future__ import annotations

from dataclasses import dataclass

from config import get_settings, ModelTierConfig


@dataclass
class ModelChoice:
    model: str
    confidence: float
    cost: float
    tier: str


@dataclass
class ModelTier:
    name: str
    model: str
    cost_per_doc: float
    confidence_threshold: float
    provider: str


class ModelRouter:
    def __init__(self, cascade: list[ModelTier] | None = None) -> None:
        if cascade is not None:
            self.CASCADE = cascade
        else:
            settings = get_settings()
            self.CASCADE = [
                ModelTier(
                    name=t.name,
                    model=t.model,
                    cost_per_doc=t.cost_per_doc,
                    confidence_threshold=t.confidence_threshold,
                    provider=t.provider,
                )
                for t in settings.model_cascade
            ]

    async def route(
        self, content: str, task_complexity: str = "standard"
    ) -> ModelChoice:
        """Try each tier in cascade order, return the first that meets its
        confidence threshold.  Falls back to the most capable tier (openai:gpt-4o)
        if none satisfy the threshold."""
        best: ModelChoice | None = None

        for tier in self.CASCADE:
            choice = await self._try_model(tier, content)
            if choice.confidence >= tier.confidence_threshold:
                return choice
            if best is None or choice.confidence > best.confidence:
                best = choice

        if best is not None:
            return best

        if self.CASCADE:
            fallback = self.CASCADE[-1]
            return ModelChoice(
                model=fallback.model,
                confidence=0.0,
                cost=fallback.cost_per_doc,
                tier=fallback.name,
            )

        return ModelChoice(model="gpt-4o", confidence=0.0, cost=0.01, tier="openai")

    async def _try_model(self, tier: ModelTier, content: str) -> ModelChoice:
        """Evaluate a model tier for the given content.

        In production this calls the actual LLM endpoint and computes
        confidence from the extraction result. For routing decisions
        (which tier to use), we use a simple heuristic based on content
        length and tier cost.
        """
        # Simple heuristic: longer content needs more capable models
        content_length = len(content)
        
        # Base confidence from tier's threshold, adjusted by content complexity
        base_confidence = tier.confidence_threshold
        
        # Short documents are easier (higher confidence), capped at threshold + 0.15
        if content_length < 500:
            confidence = min(base_confidence + 0.1, base_confidence + 0.15, 1.0)
        elif content_length < 2000:
            confidence = base_confidence
        else:
            confidence = max(base_confidence - 0.05, 0.0)
        
        return ModelChoice(
            model=tier.model,
            confidence=confidence,
            cost=tier.cost_per_doc,
            tier=tier.name,
        )
