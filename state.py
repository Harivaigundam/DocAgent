from typing import TypedDict, Annotated, Any
from langchain_core.messages import AnyMessage
import operator


class DocumentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    document_id: str
    document_content: str
    document_type: str
    type_confidence: float
    extraction_result: dict
    extraction_confidence: float
    model_used: str
    model_cost: float
    requires_review: bool
    review_status: str
    guardrail_passed: bool
    guardrail_issues: list[str]
    final_output: dict
    parallel_results: Annotated[list[dict], operator.add]
    all_results: Annotated[list[dict], operator.add]
    parallel_agent_type: str
