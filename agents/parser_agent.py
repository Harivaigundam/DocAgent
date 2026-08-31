"""Parser agent for document text extraction and normalization."""
from __future__ import annotations

import re
from datetime import datetime
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("ParserAgent")


@mcp.tool()
async def parse_document(content: str, doc_type: str = "generic") -> dict:
    """Parse raw document content into normalized text with structure hints.

    Args:
        content: Raw text content to parse.
        doc_type: Document type hint (invoice, receipt, contract, report, generic).

    Returns:
        Dict with normalized_text, line_count, char_count, structure hints, and doc_type.
    """
    if not content or not content.strip():
        return {
            "normalized_text": "",
            "line_count": 0,
            "char_count": 0,
            "structure": [],
            "doc_type": doc_type,
            "error": "Empty content provided",
        }

    normalized = _normalize_whitespace(content)
    lines = [line for line in normalized.split("\n") if line.strip()]

    structure = _detect_structure(lines, doc_type)

    return {
        "normalized_text": normalized,
        "line_count": len(lines),
        "char_count": len(normalized),
        "structure": structure,
        "doc_type": doc_type,
    }


@mcp.tool()
async def extract_metadata(content: str) -> dict:
    """Extract metadata (dates, monetary amounts, addresses, identifiers) from text.

    Args:
        content: Text to extract metadata from.

    Returns:
        Dict with dates, amounts, addresses, identifiers, and emails found.
    """
    if not content:
        return {"dates": [], "amounts": [], "addresses": [], "identifiers": [], "emails": []}

    return {
        "dates": _extract_dates(content),
        "amounts": _extract_amounts(content),
        "addresses": _extract_addresses(content),
        "identifiers": _extract_identifiers(content),
        "emails": _extract_emails(content),
    }


@mcp.tool()
async def normalize_text(content: str) -> str:
    """Normalize text format: fix whitespace, normalize unicode, standardize line endings.

    Args:
        content: Text to normalize.

    Returns:
        Normalized text string.
    """
    if not content:
        return ""

    result = content
    result = result.replace("\r\n", "\n").replace("\r", "\n")
    result = _collapse_whitespace(result)
    result = result.strip()
    return result


# --- Internal helpers ---

def _normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    normalized = []
    for line in lines:
        collapsed = re.sub(r"[ \t]+", " ", line).strip()
        normalized.append(collapsed)
    return "\n".join(normalized)


def _collapse_whitespace(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def _detect_structure(lines: list[str], doc_type: str) -> list[dict]:
    structure = []
    for i, line in enumerate(lines):
        entry = {"line_number": i + 1, "text": line, "type": "body"}
        if re.match(r"^[\d]{1,2}[/\-][\d]{1,2}[/\-][\d]{2,4}", line):
            entry["type"] = "date_line"
        elif re.match(r"^[A-Z\s]{5,}$", line):
            entry["type"] = "heading"
        elif re.match(r"^(total|subtotal|tax|amount|balance|due)[:\s]", line, re.IGNORECASE):
            entry["type"] = "financial_line"
        elif re.match(r"^(invoice|receipt|contract|agreement|report)[:\s#]", line, re.IGNORECASE):
            entry["type"] = "document_header"
        structure.append(entry)
    return structure


def _extract_dates(text: str) -> list[dict]:
    patterns = [
        (r"\b(\d{1,2}/\d{1,2}/\d{2,4})\b", "%m/%d/%Y"),
        (r"\b(\d{1,2}-\d{1,2}-\d{2,4})\b", "%m-%d-%Y"),
        (r"\b(\w+ \d{1,2},?\s*\d{4})\b", None),
        (r"\b(\d{4}-\d{2}-\d{2})\b", "%Y-%m-%d"),
    ]
    found = []
    seen = set()
    for pattern, fmt in patterns:
        for match in re.finditer(pattern, text):
            raw = match.group(1)
            if raw in seen:
                continue
            seen.add(raw)
            parsed = None
            if fmt:
                try:
                    parsed = datetime.strptime(raw, fmt).date().isoformat()
                except ValueError:
                    pass
            found.append({"raw": raw, "parsed": parsed, "start": match.start(), "end": match.end()})
    return found


def _extract_amounts(text: str) -> list[dict]:
    pattern = r"\$[\d,]+\.?\d*"
    found = []
    for match in re.finditer(pattern, text):
        raw = match.group()
        try:
            value = float(raw.replace("$", "").replace(",", ""))
        except ValueError:
            value = None
        found.append({"raw": raw, "value": value, "start": match.start(), "end": match.end()})
    return found


def _extract_addresses(text: str) -> list[dict]:
    pattern = r"\d{1,5}\s+[\w\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Court|Ct|Way|Place|Pl)\b[^,\n]*,\s*[\w\s]+,\s*[A-Z]{2}\s+\d{5}"
    found = []
    for match in re.finditer(pattern, text, re.IGNORECASE):
        found.append({"raw": match.group().strip(), "start": match.start(), "end": match.end()})
    return found


def _extract_identifiers(text: str) -> list[dict]:
    patterns = {
        "invoice_number": r"(?:invoice|inv)[\s#:]*([A-Z0-9\-]{3,20})",
        "po_number": r"(?:purchase\s*order|po)[\s#:]*([A-Z0-9\-]{3,20})",
        "contract_id": r"(?:contract|agreement)[\s#:]*([A-Z0-9\-]{3,20})",
        "tax_id": r"(?:tax\s*id|ein|tin)[\s#:]*([\d\-]{5,15})",
    }
    found = []
    for id_type, pattern in patterns.items():
        for match in re.finditer(pattern, text, re.IGNORECASE):
            found.append({"type": id_type, "value": match.group(1), "start": match.start(), "end": match.end()})
    return found


def _extract_emails(text: str) -> list[str]:
    return list(set(re.findall(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b", text)))


if __name__ == "__main__":
    mcp.run()
