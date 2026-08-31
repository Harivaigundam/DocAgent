"""Summarizer agent for document summaries, insights, and confidence explanations."""
from __future__ import annotations

import re
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("SummarizerAgent")


@mcp.tool()
async def summarize_document(
    content: str, fields: dict | None = None, doc_type: str = "generic"
) -> dict:
    """Create a structured summary of document content and extracted fields.

    Args:
        content: Raw or normalized document text.
        fields: Optional extracted fields dict to incorporate into the summary.
        doc_type: Document type for summary customization.

    Returns:
        Dict with summary, key_points, doc_type, word_count, and sections.
    """
    if not content or not content.strip():
        return {
            "summary": "Empty document provided.",
            "key_points": [],
            "doc_type": doc_type,
            "word_count": 0,
            "sections": [],
        }

    word_count = len(content.split())
    sentences = _split_sentences(content)
    sections = _identify_sections(content, doc_type)
    key_points = _extract_key_points(content, fields, doc_type)
    summary = _build_summary(sentences, key_points, doc_type, fields)

    return {
        "summary": summary,
        "key_points": key_points,
        "doc_type": doc_type,
        "word_count": word_count,
        "sections": sections,
    }


@mcp.tool()
async def extract_insights(content: str, fields: dict | None = None) -> dict:
    """Extract actionable insights and notable patterns from document content.

    Args:
        content: Document text to analyze.
        fields: Optional extracted fields for context.

    Returns:
        Dict with insights list, categories, and actionable_items.
    """
    if not content:
        return {"insights": [], "categories": {}, "actionable_items": []}

    insights = []
    categories = {
        "financial": [],
        "temporal": [],
        "entity": [],
        "risk": [],
        "compliance": [],
    }

    financial = _extract_financial_insights(content, fields)
    categories["financial"] = financial
    insights.extend(financial)

    temporal = _extract_temporal_insights(content, fields)
    categories["temporal"] = temporal
    insights.extend(temporal)

    entity = _extract_entity_insights(content, fields)
    categories["entity"] = entity
    insights.extend(entity)

    risk = _extract_risk_insights(content, fields)
    categories["risk"] = risk
    insights.extend(risk)

    actionable = _identify_actionable_items(content, fields)

    return {
        "insights": insights,
        "categories": categories,
        "actionable_items": actionable,
    }


@mcp.tool()
async def explain_confidence(
    fields: dict, confidence: float, model_used: str = "unknown"
) -> dict:
    """Generate a human-readable explanation of extraction confidence.

    Args:
        fields: The extracted fields dict.
        confidence: Overall confidence score (0.0-1.0).
        model_used: Model identifier used for extraction.

    Returns:
        Dict with explanation, factor_breakdown, recommendations, and risk_assessment.
    """
    factors = _analyze_confidence_factors(fields, confidence)
    explanation = _build_confidence_explanation(confidence, factors, model_used)
    recommendations = _generate_recommendations(confidence, factors)
    risk_assessment = _assess_risk(confidence, factors)

    return {
        "explanation": explanation,
        "factor_breakdown": factors,
        "recommendations": recommendations,
        "risk_assessment": risk_assessment,
    }


# --- Summary helpers ---

def _split_sentences(text: str) -> list[str]:
    raw = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in raw if s.strip()]


def _identify_sections(content: str, doc_type: str) -> list[dict]:
    sections = []
    lines = content.split("\n")
    current_section = None

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if _is_heading(stripped):
            if current_section:
                current_section["line_end"] = len(sections)
            current_section = {"title": stripped, "type": "heading", "line_start": len(sections)}
            sections.append(current_section)
        elif re.match(r"^(total|subtotal|tax|amount|due)[:\s]", stripped, re.IGNORECASE):
            sections.append({"title": stripped[:60], "type": "financial_line"})
        elif re.match(r"^(total|balance|amount\s*due)[:\s]", stripped, re.IGNORECASE):
            sections.append({"title": stripped[:60], "type": "total_line"})

    return sections


def _is_heading(line: str) -> bool:
    if re.match(r"^[A-Z\s]{5,}$", line):
        return True
    if re.match(r"^#{1,3}\s+", line):
        return True
    if len(line) < 80 and line.strip().endswith(":") and not re.search(r"\d", line):
        return True
    return False


def _extract_key_points(
    content: str, fields: dict | None, doc_type: str
) -> list[str]:
    points = []

    if fields:
        if fields.get("total") is not None:
            points.append(f"Total amount: ${fields['total']}")
        if fields.get("date"):
            points.append(f"Date: {fields['date']}")
        if fields.get("vendor_name") or fields.get("store_name"):
            name = fields.get("vendor_name") or fields.get("store_name")
            points.append(f"Entity: {name}")
        if fields.get("invoice_number"):
            points.append(f"Invoice number: {fields['invoice_number']}")
        if fields.get("title"):
            points.append(f"Title: {fields['title']}")

    amount_pattern = r"\$[\d,]+\.?\d*"
    amounts = re.findall(amount_pattern, content)
    if amounts and not fields:
        points.append(f"Contains {len(amounts)} monetary amount(s): {', '.join(amounts[:3])}")

    date_pattern = r"\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b"
    dates = re.findall(date_pattern, content)
    if dates and not fields:
        points.append(f"Contains {len(dates)} date(s): {', '.join(dates[:3])}")

    return points[:10]


def _build_summary(
    sentences: list[str], key_points: list[str], doc_type: str, fields: dict | None
) -> str:
    if not sentences:
        return "No content to summarize."

    summary_parts = []

    if fields:
        entity = fields.get("vendor_name") or fields.get("store_name") or fields.get("title", "")
        if entity:
            summary_parts.append(f"Document: {entity}")

    important = [s for s in sentences if _is_important_sentence(s)]
    if important:
        summary_parts.append("Key: " + " ".join(important[:3]))
    else:
        summary_parts.append(" ".join(sentences[:2]))

    if key_points:
        summary_parts.append("Highlights: " + "; ".join(key_points[:5]))

    return " | ".join(summary_parts)


def _is_important_sentence(sentence: str) -> bool:
    indicators = ["total", "due", "must", "required", "deadline", "payment", "terms", "agreement"]
    lower = sentence.lower()
    return any(word in lower for word in indicators)


# --- Insight helpers ---

def _extract_financial_insights(content: str, fields: dict | None) -> list[dict]:
    insights = []
    total = fields.get("total") if fields else None
    if isinstance(total, (int, float)):
        if total > 10000:
            insights.append({
                "category": "financial",
                "text": f"Large transaction amount: ${total:,.2f}",
                "severity": "medium",
            })
        if total == 0:
            insights.append({
                "category": "financial",
                "text": "Zero-dollar transaction detected",
                "severity": "low",
            })

    tax = fields.get("tax") if fields else None
    subtotal = fields.get("subtotal") if fields else None
    if isinstance(tax, (int, float)) and isinstance(subtotal, (int, float)) and subtotal > 0:
        rate = tax / subtotal
        if rate > 0.2:
            insights.append({
                "category": "financial",
                "text": f"Tax rate appears high: {rate:.1%}",
                "severity": "medium",
            })

    return insights


def _extract_temporal_insights(content: str, fields: dict | None) -> list[dict]:
    insights = []
    date_val = fields.get("date") if fields else None
    if date_val:
        insights.append({
            "category": "temporal",
            "text": f"Document date: {date_val}",
            "severity": "info",
        })

    urgency_words = ["urgent", "asap", "immediately", "due now", "past due", "overdue"]
    lower = content.lower()
    for word in urgency_words:
        if word in lower:
            insights.append({
                "category": "temporal",
                "text": f"Urgency indicator found: '{word}'",
                "severity": "medium",
            })
            break

    return insights


def _extract_entity_insights(content: str, fields: dict | None) -> list[dict]:
    insights = []
    entity = None
    if fields:
        entity = fields.get("vendor_name") or fields.get("store_name") or fields.get("author")
    if entity:
        insights.append({
            "category": "entity",
            "text": f"Primary entity identified: {entity}",
            "severity": "info",
        })

    parties = fields.get("parties") if fields else None
    if isinstance(parties, list) and len(parties) > 0:
        insights.append({
            "category": "entity",
            "text": f"Document involves {len(parties)} parties",
            "severity": "info",
        })

    return insights


def _extract_risk_insights(content: str, fields: dict | None) -> list[dict]:
    insights = []
    lower = content.lower()

    risk_words = ["penalty", "late fee", "default", "breach", "terminate", "lawsuit", "litigation"]
    for word in risk_words:
        if word in lower:
            insights.append({
                "category": "risk",
                "text": f"Risk keyword detected: '{word}'",
                "severity": "high",
            })

    return insights


def _identify_actionable_items(content: str, fields: dict | None) -> list[str]:
    items = []
    lower = content.lower()

    action_patterns = [
        r"(?:please|kindly)\s+(.+?)(?:\.|$)",
        r"(?:must|should|need to)\s+(.+?)(?:\.|$)",
        r"(?:due\s+by|deadline[:\s]+)\s+(.+?)(?:\.|$)",
        r"(?:pay|payment)\s+(?:of\s+)?(.+?)(?:\.|$)",
    ]

    for pattern in action_patterns:
        matches = re.findall(pattern, lower)
        for match in matches:
            cleaned = match.strip()[:100]
            if cleaned and cleaned not in items:
                items.append(cleaned)

    return items[:10]


# --- Confidence explanation helpers ---

def _analyze_confidence_factors(fields: dict, confidence: float) -> list[dict]:
    factors = []

    total_fields = len(fields)
    null_fields = sum(1 for v in fields.values() if v is None)
    completeness = 1.0 - (null_fields / total_fields) if total_fields > 0 else 0.0
    factors.append({
        "factor": "completeness",
        "score": completeness,
        "detail": f"{total_fields - null_fields}/{total_fields} fields populated",
    })

    numeric_fields = sum(1 for v in fields.values() if isinstance(v, (int, float)))
    type_consistency = numeric_fields / total_fields if total_fields > 0 else 0.0
    factors.append({
        "factor": "type_consistency",
        "score": type_consistency,
        "detail": f"{numeric_fields}/{total_fields} fields have expected numeric types",
    })

    confidence_fields = 0
    for v in fields.values():
        if isinstance(v, dict) and "confidence" in v:
            confidence_fields += 1
    has_confidence = confidence_fields > 0
    factors.append({
        "factor": "model_confidence_provided",
        "score": 1.0 if has_confidence else 0.5,
        "detail": f"{'Per-field' if has_confidence else 'No'} confidence scores available",
    })

    return factors


def _build_confidence_explanation(
    confidence: float, factors: list[dict], model_used: str
) -> str:
    level = "high" if confidence >= 0.8 else "medium" if confidence >= 0.5 else "low"
    parts = [
        f"Extraction confidence: {confidence:.1%} ({level})",
        f"Model: {model_used}",
    ]

    for f in factors:
        score = f["score"]
        label = f["factor"].replace("_", " ")
        parts.append(f"- {label}: {score:.1%} — {f['detail']}")

    return "\n".join(parts)


def _generate_recommendations(confidence: float, factors: list[dict]) -> list[str]:
    recs = []
    if confidence < 0.5:
        recs.append("Confidence is critically low. Manual review strongly recommended.")
    elif confidence < 0.7:
        recs.append("Confidence is moderate. Review extracted fields before using.")

    completeness = next((f for f in factors if f["factor"] == "completeness"), None)
    if completeness and completeness["score"] < 0.7:
        recs.append("Many fields are missing. Consider re-extraction or manual input.")

    if confidence >= 0.8:
        recs.append("Confidence is high. Extraction may be used with minimal review.")

    return recs


def _assess_risk(confidence: float, factors: list[dict]) -> dict:
    risk_level = "low"
    if confidence < 0.3:
        risk_level = "critical"
    elif confidence < 0.5:
        risk_level = "high"
    elif confidence < 0.7:
        risk_level = "medium"

    return {
        "level": risk_level,
        "can_auto_process": confidence >= 0.8,
        "requires_human_review": confidence < 0.7,
        "recommendation": (
            "Auto-process" if confidence >= 0.8
            else "Human review required" if confidence >= 0.5
            else "Manual extraction recommended"
        ),
    }


if __name__ == "__main__":
    mcp.run()
