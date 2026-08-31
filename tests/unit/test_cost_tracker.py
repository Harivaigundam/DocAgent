import pytest

from model_router.cost_tracker import CostTracker


class TestCostTracker:
    def test_initial_state(self):
        tracker = CostTracker()
        summary = tracker.summary()
        assert summary["total_cost"] == 0.0
        assert summary["docs_processed"] == 0
        assert summary["model_usage"] == {}
        assert summary["avg_cost_per_doc"] == 0.0

    def test_record_single_model(self):
        tracker = CostTracker()
        tracker.record("gpt-4o", 0.01)
        summary = tracker.summary()
        assert summary["total_cost"] == 0.01
        assert summary["docs_processed"] == 1
        assert summary["model_usage"]["gpt-4o"]["count"] == 1
        assert summary["model_usage"]["gpt-4o"]["total_cost"] == 0.01
        assert summary["avg_cost_per_doc"] == 0.01

    def test_record_multiple_models(self):
        tracker = CostTracker()
        tracker.record("gpt-4o", 0.01)
        tracker.record("gpt-4o-mini", 0.001)
        tracker.record("gpt-4o", 0.01)
        summary = tracker.summary()
        assert summary["total_cost"] == pytest.approx(0.021)
        assert summary["docs_processed"] == 3
        assert summary["model_usage"]["gpt-4o"]["count"] == 2
        assert summary["model_usage"]["gpt-4o"]["total_cost"] == 0.02
        assert summary["model_usage"]["gpt-4o-mini"]["count"] == 1
        assert summary["model_usage"]["gpt-4o-mini"]["total_cost"] == 0.001
        assert summary["avg_cost_per_doc"] == pytest.approx(0.007)

    def test_record_zero_cost(self):
        tracker = CostTracker()
        tracker.record("ollama", 0.0)
        summary = tracker.summary()
        assert summary["total_cost"] == 0.0
        assert summary["docs_processed"] == 1
        assert summary["model_usage"]["ollama"]["count"] == 1
        assert summary["avg_cost_per_doc"] == 0.0

    def test_summary_returns_copy(self):
        tracker = CostTracker()
        tracker.record("gpt-4o", 0.01)
        s1 = tracker.summary()
        s2 = tracker.summary()
        s1["total_cost"] = 999.0
        assert s2["total_cost"] == 0.01

    def test_avg_cost_zero_docs(self):
        tracker = CostTracker()
        assert tracker.summary()["avg_cost_per_doc"] == 0.0
