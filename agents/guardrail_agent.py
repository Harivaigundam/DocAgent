"""Guardrail agent for document validation and protection."""
from mcp.server.fastmcp import FastMCP

from guardrails.schema_validator import validate_schema
from guardrails.pii_detector import detect_pii, redact_pii
from guardrails.consistency_checker import check_consistency
from guardrails.prompt_guard import check_content_safety

mcp = FastMCP("GuardrailAgent")

REQUIRED_FIELDS = {
    "invoice": ["vendor_name", "invoice_number", "date", "total"],
    "receipt": ["store_name", "date", "total"],
    "contract": ["title", "parties", "effective_date"],
    "report": ["title", "author", "date"],
}


@mcp.tool()
async def validate_extraction(extraction: dict, doc_type: str, content: str = "") -> dict:
    """
    Run full guardrail validation on document extraction.
    
    Performs:
    - Schema validation for required fields
    - PII detection on provided content
    - Consistency checking based on document type
    
    Args:
        extraction: Extracted document data
        doc_type: Type of document (invoice, receipt, contract, report)
        content: Optional raw text content for PII detection
    
    Returns:
        Dict with passed bool, schema validation, pii detection, and consistency results
    """
    required_fields = REQUIRED_FIELDS.get(doc_type, [])
    
    schema_result = validate_schema(extraction, required_fields)
    
    pii_matches = []
    redacted_content = content
    if content:
        pii_matches = detect_pii(content)
        redacted_content = redact_pii(content)
    
    consistency_issues = check_consistency(extraction, doc_type)
    
    safety_result = await check_content_safety(content) if content else None
    
    passed = (
        schema_result.is_valid
        and len(pii_matches) == 0
        and len(consistency_issues) == 0
        and (safety_result is None or safety_result.is_safe)
    )
    
    return {
        "passed": passed,
        "schema": {
            "is_valid": schema_result.is_valid,
            "errors": schema_result.errors,
            "warnings": schema_result.warnings,
        },
        "pii": {
            "detected": len(pii_matches) > 0,
            "matches": [
                {"type": m.type, "value": m.value, "start": m.start, "end": m.end}
                for m in pii_matches
            ],
            "redacted_content": redacted_content,
        },
        "consistency": {
            "issues": consistency_issues,
        },
        "content_safety": {
            "is_safe": safety_result.is_safe if safety_result else True,
            "label": safety_result.label if safety_result else "unknown",
            "error": safety_result.error if safety_result else None,
        },
    }


if __name__ == "__main__":
    mcp.run()
