def detect_type_prompt(document_content: str) -> str:
    """Return a prompt that asks the LLM to classify the document type."""
    truncated = document_content[:5000]
    return (
        "You are a document classification expert. Analyze the following document "
        "content and determine its type.\n\n"
        "Respond with ONLY a JSON object in this exact format:\n"
        '{"type": "<invoice|receipt|contract|report>", '
        '"confidence": <float between 0.0 and 1.0>, '
        '"reasoning": "<brief explanation>"}\n\n'
        "Document content:\n"
        f"---\n{truncated}\n---\n"
        "JSON:"
    )
