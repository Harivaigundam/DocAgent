from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CostTracker:
    """Accumulates per-model cost information across processed documents."""

    total_cost: float = 0.0
    docs_processed: int = 0
    model_usage: dict = field(default_factory=dict)

    def record(self, model: str, cost: float) -> None:
        """Record the cost incurred by a single model for one document."""
        self.total_cost += cost
        self.docs_processed += 1
        if model not in self.model_usage:
            self.model_usage[model] = {"count": 0, "total_cost": 0.0}
        self.model_usage[model]["count"] += 1
        self.model_usage[model]["total_cost"] += cost

    def summary(self) -> dict:
        """Return a snapshot of accumulated statistics."""
        avg = (
            self.total_cost / self.docs_processed if self.docs_processed > 0 else 0.0
        )
        return {
            "total_cost": self.total_cost,
            "docs_processed": self.docs_processed,
            "model_usage": dict(self.model_usage),
            "avg_cost_per_doc": avg,
        }
