from schemas.receipt import ReceiptFields

_SCHEMA_DESC = ReceiptFields.model_json_schema()

def extract_receipt_prompt(document_content: str) -> str:
    """Return a prompt that asks the LLM to extract receipt fields as JSON."""
    truncated = document_content[:5000]
    return (
        "You are a receipt data extraction expert. Extract all relevant fields "
        "from the following receipt document.\n\n"
        "Return ONLY a JSON object matching this schema:\n"
        f"{_SCHEMA_DESC}\n\n"
        "Additional rules:\n"
        "- Every field in the schema must be present in the JSON output.\n"
        "- For each extracted field, add a sibling key named `<field>_confidence` "
        "with a float between 0.0 and 1.0 indicating your confidence.\n"
        "- Use null for optional fields you cannot determine.\n"
        "- The total must be a positive number.\n\n"
        "Document content:\n"
        f"---\n{truncated}\n---\n"
        "JSON:"
    )
