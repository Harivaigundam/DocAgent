import re
from dataclasses import dataclass


PII_PATTERNS = {
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    "credit_card": r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",
    "email": r"\b[\w.+-]+@[\w-]+\.[\w.]+\b",
    "phone": r"\b(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
    "account_number": r"\b\d{8,17}\b",
}


@dataclass
class PIIMatch:
    """A detected PII occurrence in text."""
    type: str
    value: str
    start: int
    end: int


def detect_pii(text: str) -> list[PIIMatch]:
    """
    Detect all PII occurrences in the given text.
    
    Args:
        text: Text to scan for PII
    
    Returns:
        List of PIIMatch objects with type, value, and position info
    """
    matches = []
    
    for pii_type, pattern in PII_PATTERNS.items():
        for match in re.finditer(pattern, text):
            matches.append(PIIMatch(
                type=pii_type,
                value=match.group(),
                start=match.start(),
                end=match.end()
            ))
    
    matches.sort(key=lambda m: m.start)
    
    return matches


def redact_pii(text: str) -> str:
    """
    Replace all PII in text with [REDACTED_TYPE] placeholders.
    
    Args:
        text: Text containing PII to redact
    
    Returns:
        Text with PII replaced by placeholders
    """
    matches = detect_pii(text)
    
    if not matches:
        return text
    
    result = []
    last_end = 0
    
    for match in matches:
        result.append(text[last_end:match.start])
        result.append(f"[REDACTED_{match.type.upper()}]")
        last_end = match.end
    
    result.append(text[last_end:])
    
    return "".join(result)
