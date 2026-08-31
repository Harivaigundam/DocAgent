import logging
from langgraph.graph import StateGraph, END, START
from langgraph.types import Send
from state import DocumentState
from agents import AGENT_MAP
from model_router.router import ModelRouter
from model_router.cost_tracker import CostTracker
from agents.guardrail_agent import validate_extraction as guardrail_validate

logger = logging.getLogger(__name__)

ALL_AGENT_TYPES = ["invoice", "receipt", "contract", "report"]


class ParallelDocumentOrchestrator:
    def __init__(self):
        self.router = ModelRouter()
        self.cost_tracker = CostTracker()
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        graph = StateGraph(DocumentState)

        graph.add_node("detect_type", self._detect_type_node)
        graph.add_node("parallel_extract", self._parallel_extract_node)
        graph.add_node("agent_extract", self._agent_extract_node)
        graph.add_node("aggregate_results", self._aggregate_results_node)
        graph.add_node("select_best", self._select_best_node)
        graph.add_node("guardrails", self._guardrails_node)
        graph.add_node("finalize", self._finalize_node)

        graph.add_edge(START, "detect_type")
        graph.add_conditional_edges("detect_type", self._route_to_agents, ["agent_extract"])
        graph.add_edge("agent_extract", "aggregate_results")
        graph.add_edge("aggregate_results", "select_best")
        graph.add_edge("select_best", "guardrails")
        graph.add_edge("guardrails", "finalize")
        graph.add_edge("finalize", END)

        return graph.compile()

    async def _detect_type_node(self, state: DocumentState) -> dict:
        keywords = {
            "invoice": ["invoice", "bill", "vendor", "payment due", "amount owed"],
            "receipt": ["receipt", "purchase", "transaction", "paid", "store"],
            "contract": ["agreement", "contract", "party", "terms", "signature"],
            "report": ["report", "analysis", "summary", "metrics", "findings"],
        }
        content_lower = state["document_content"].lower()
        scores = {dt: sum(1 for w in words if w in content_lower) for dt, words in keywords.items()}
        best = max(scores, key=scores.get)
        total = sum(scores.values()) or 1
        return {"document_type": best, "type_confidence": max(scores[best] / total, 0.5)}

    def _route_to_agents(self, state: DocumentState) -> list[Send]:
        return [
            Send("agent_extract", {
                **state,
                "document_type": agent_type,
                "parallel_agent_type": agent_type,
            })
            for agent_type in ALL_AGENT_TYPES
        ]

    async def _parallel_extract_node(self, state: DocumentState) -> dict:
        return {}

    async def _agent_extract_node(self, state: DocumentState) -> dict:
        agent_type = state.get("parallel_agent_type", state.get("document_type", ""))
        agent_class = AGENT_MAP.get(agent_type)
        if not agent_class:
            return {
                "parallel_results": [{
                    "agent_type": agent_type,
                    "fields": {},
                    "confidence": 0.0,
                    "model_used": "",
                    "cost": 0.0,
                    "requires_review": False,
                    "error": f"No agent for type: {agent_type}",
                }]
            }
        agent = agent_class()
        result = await agent._extract(state["document_content"], {})
        return {
            "parallel_results": [{
                "agent_type": agent_type,
                "fields": result["fields"],
                "confidence": result["confidence"],
                "model_used": result["model_used"],
                "cost": result["cost"],
                "requires_review": result["requires_review"],
            }]
        }

    def _aggregate_results_node(self, state: DocumentState) -> dict:
        return {}

    async def _select_best_node(self, state: DocumentState) -> dict:
        results = state.get("parallel_results", [])
        if not results:
            return {
                "extraction_result": {},
                "extraction_confidence": 0.0,
                "model_used": "",
                "model_cost": 0.0,
                "requires_review": True,
            }

        best = max(results, key=lambda r: r.get("confidence", 0))
        return {
            "document_type": best["agent_type"],
            "extraction_result": best["fields"],
            "extraction_confidence": best["confidence"],
            "model_used": best["model_used"],
            "model_cost": best["cost"],
            "requires_review": best["requires_review"],
            "all_results": results,
        }

    async def _guardrails_node(self, state: DocumentState) -> dict:
        result = await guardrail_validate(
            state["extraction_result"], state["document_type"], state["document_content"]
        )
        schema_errors = result.get("schema", {}).get("errors", [])
        consistency_issues = result.get("consistency", {}).get("issues", [])
        return {
            "guardrail_passed": result["passed"],
            "guardrail_issues": schema_errors + consistency_issues,
        }

    async def _finalize_node(self, state: DocumentState) -> dict:
        self.cost_tracker.record(state["model_used"], state["model_cost"])
        return {"final_output": state["extraction_result"]}

    async def run(self, document_id: str, content: str) -> dict:
        initial_state = {
            "messages": [],
            "document_id": document_id,
            "document_content": content,
            "document_type": "",
            "type_confidence": 0.0,
            "extraction_result": {},
            "extraction_confidence": 0.0,
            "model_used": "",
            "model_cost": 0.0,
            "requires_review": False,
            "review_status": "",
            "guardrail_passed": False,
            "guardrail_issues": [],
            "final_output": {},
            "parallel_results": [],
            "all_results": [],
            "parallel_agent_type": "",
        }
        result = await self.graph.ainvoke(initial_state)
        return {
            "document_id": document_id,
            "document_type": result["document_type"],
            "extraction": result["final_output"],
            "confidence": result["extraction_confidence"],
            "model_used": result["model_used"],
            "cost": result["model_cost"],
            "guardrail_passed": result["guardrail_passed"],
            "requires_review": result["requires_review"],
            "all_agent_results": result.get("all_results", []),
        }
