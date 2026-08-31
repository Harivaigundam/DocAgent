"""Validator agent for field validation, consistency checks, and anomaly detection."""
from __future__ import annotations

import re
from datetime import datetime
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("ValidatorAgent")

REQUIRED_FIELDS = {
    "invoice": ["vendor_name", "invoice_number", "date", "total"],
    "receipt": ["store_name", "date", "total"],
    "contract": ["title", "parties", "effective_date"],
    "report": ["title", "author", "date"],
}


@mcp.tool()
async def validate_fields(
    data: dict, doc_type: str = "generic", required_fields: list[str] | None = None
) -> dict:
    """Validate extracted fields against a schema of required fields.

    Args:
        data: Extracted document fields to validate.
        doc_type: Document type for built-in required field rules.
        required_fields: Override required fields list (takes precedence over doc_type defaults).

    Returns:
        Dict with is_valid, errors, warnings, and missing_fields.
    """
    fields_to_check = required_fields or REQUIRED_FIELDS.get(doc_type, [])
    errors = []
    warnings = []
    missing = []

    for field_name in fields_to_check:
        if field_name not in data:
            errors.append(f"Missing required field: {field_name}")
            missing.append(field_name)
        elif data[field_name] is None:
            errors.append(f"Required field is null: {field_name}")
            missing.append(field_name)
        elif isinstance(data[field_name], str) and not data[field_name].strip():
            errors.append(f"Required field is empty: {field_name}")
            missing.append(field_name)

    if "total" in data and data["total"] is not None:
        if not isinstance(data["total"], (int, float)):
            errors.append("Field 'total' must be numeric")
        elif data["total"] < 0:
            errors.append("Field 'total' must be non-negative")

    if "date" in data and data["date"] is not None:
        date_str = str(data["date"])
        if not _is_valid_date(date_str):
            warnings.append(f"Field 'date' may be invalid: {date_str}")

    for key, value in data.items():
        if value is not None and key not in fields_to_check:
            warnings.append(f"Unexpected field present: {key}")

    return {
        "is_valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "missing_fields": missing,
        "checked_fields": fields_to_check,
    }


@mcp.tool()
async def check_consistency(data: dict, doc_type: str = "generic") -> dict:
    """Check logical consistency of extracted document data.

    Args:
        data: Extracted document fields.
        doc_type: Document type for consistency rule selection.

    Returns:
        Dict with is_consistent, issues, and checked_rules.
    """
    issues = []
    checked_rules = []

    if doc_type == "invoice":
        issues.extend(_check_invoice_consistency(data))
        checked_rules = ["subtotal_tax_total", "positive_line_items", "date_format"]
    elif doc_type == "receipt":
        issues.extend(_check_receipt_consistency(data))
        checked_rules = ["items_total_match", "positive_items"]
    elif doc_type == "contract":
        issues.extend(_check_contract_consistency(data))
        checked_rules = ["date_ordering", "parties_present"]
    elif doc_type == "report":
        issues.extend(_check_report_consistency(data))
        checked_rules = ["date_format", "author_present"]
    else:
        issues.extend(_check_generic_consistency(data))
        checked_rules = ["null_values", "type_consistency"]

    return {
        "is_consistent": len(issues) == 0,
        "issues": issues,
        "checked_rules": checked_rules,
    }


@mcp.tool()
async def detect_anomalies(data: dict, doc_type: str = "generic") -> dict:
    """Detect anomalies and unusual patterns in extracted data.

    Args:
        data: Extracted document fields.
        doc_type: Document type for anomaly heuristic selection.

    Returns:
        Dict with anomalies list, risk_level (low/medium/high), and summary.
    """
    anomalies = []

    anomalies.extend(_check_value_range_anomalies(data))
    anomalies.extend(_check_format_anomalies(data))
    anomalies.extend(_check_completeness_anomalies(data))

    risk_level = "low"
    high_count = sum(1 for a in anomalies if a.get("severity") == "high")
    medium_count = sum(1 for a in anomalies if a.get("severity") == "medium")
    if high_count > 0:
        risk_level = "high"
    elif medium_count > 0:
        risk_level = "medium"

    return {
        "anomalies": anomalies,
        "risk_level": risk_level,
        "summary": f"Found {len(anomalies)} anomalies ({risk_level} risk)",
    }


# --- Consistency checkers ---

def _check_invoice_consistency(data: dict) -> list[str]:
    issues = []
    subtotal = data.get("subtotal")
    tax = data.get("tax")
    total = data.get("total")

    if all(v is not None for v in [subtotal, tax, total]):
        if all(isinstance(v, (int, float)) for v in [subtotal, tax, total]):
            expected = subtotal + tax
            if abs(total - expected) > 0.01:
                issues.append(f"Total {total} != subtotal {subtotal} + tax {tax} (expected {expected:.2f})")
        else:
            issues.append("Subtotal, tax, and total must all be numeric")

    items = data.get("items", [])
    if items and isinstance(items, list):
        for i, item in enumerate(items):
            if isinstance(item, dict):
                qty = item.get("quantity", 1)
                price = item.get("price") or item.get("unit_price")
                if qty is not None and price is not None:
                    if isinstance(qty, (int, float)) and isinstance(price, (int, float)):
                        if qty < 0 or price < 0:
                            issues.append(f"Item {i} has negative quantity or price")

    return issues


def _check_receipt_consistency(data: dict) -> list[str]:
    issues = []
    items = data.get("items", [])
    total = data.get("total")

    if items and total is not None and isinstance(total, (int, float)):
        items_total = 0.0
        for item in items:
            if isinstance(item, dict):
                price = item.get("price") or item.get("amount")
                quantity = item.get("quantity", 1)
                if isinstance(price, (int, float)) and isinstance(quantity, (int, float)):
                    items_total += price * quantity
        if abs(total - items_total) > 0.05:
            issues.append(f"Total {total} != sum of items {items_total:.2f}")

    return issues


def _check_contract_consistency(data: dict) -> list[str]:
    issues = []
    start = data.get("effective_date")
    end = data.get("expiration_date") or data.get("end_date")

    if start and end:
        try:
            d_start = _parse_date(str(start))
            d_end = _parse_date(str(end))
            if d_start and d_end and d_start >= d_end:
                issues.append(f"Effective date {start} is not before end date {end}")
        except Exception:
            pass

    parties = data.get("parties")
    if parties and isinstance(parties, list) and len(parties) < 2:
        issues.append("Contract should have at least 2 parties")

    return issues


def _check_report_consistency(data: dict) -> list[str]:
    issues = []
    date_val = data.get("date")
    if date_val and not _is_valid_date(str(date_val)):
        issues.append(f"Report date may be invalid: {date_val}")
    return issues


def _check_generic_consistency(data: dict) -> list[str]:
    issues = []
    for key, value in data.items():
        if isinstance(value, str) and len(value) > 5000:
            issues.append(f"Field '{key}' is unusually long ({len(value)} chars)")
    return issues


# --- Anomaly detectors ---

def _check_value_range_anomalies(data: dict) -> list[dict]:
    anomalies = []
    total = data.get("total")
    if isinstance(total, (int, float)):
        if total > 1_000_000:
            anomalies.append({
                "field": "total",
                "value": total,
                "severity": "high",
                "reason": f"Unusually large total: {total}",
            })
        elif total == 0:
            anomalies.append({
                "field": "total",
                "value": total,
                "severity": "medium",
                "reason": "Total is zero",
            })
    return anomalies


def _check_format_anomalies(data: dict) -> list[dict]:
    anomalies = []
    for key, value in data.items():
        if isinstance(value, str):
            if re.search(r"<script|javascript:|on\w+\s*=", value, re.IGNORECASE):
                anomalies.append({
                    "field": key,
                    "value": value[:100],
                    "severity": "high",
                    "reason": "Potential injection pattern detected",
                })
            if len(value) > 10000:
                anomalies.append({
                    "field": key,
                    "value": f"{len(value)} chars",
                    "severity": "medium",
                    "reason": f"Field '{key}' exceeds 10000 characters",
                })
    return anomalies


def _check_completeness_anomalies(data: dict) -> list[dict]:
    anomalies = []
    total_fields = len(data)
    null_fields = sum(1 for v in data.values() if v is None)
    if total_fields > 0 and null_fields / total_fields > 0.5:
        anomalies.append({
            "field": "_completeness",
            "value": f"{null_fields}/{total_fields} null",
            "severity": "medium",
            "reason": f"More than half of fields are null ({null_fields}/{total_fields})",
        })
    return anomalies


# --- Helpers ---

def _is_valid_date(date_str: str) -> bool:
    return _parse_date(date_str) is not None


def _parse_date(date_str: str) -> datetime | None:
    formats = ["%Y-%m-%d", "%m/%d/%Y", "%m-%d-%Y", "%d/%m/%Y", "%B %d, %Y", "%b %d, %Y"]
    for fmt in formats:
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except ValueError:
            continue
    return None


if __name__ == "__main__":
    mcp.run()
