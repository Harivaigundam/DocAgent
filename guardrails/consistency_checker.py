def check_consistency(extraction: dict, doc_type: str) -> list[str]:
    """
    Check logical consistency of extracted document data.
    
    Args:
        extraction: Extracted document data
        doc_type: Type of document (invoice, receipt, etc.)
    
    Returns:
        List of consistency issues found (empty if consistent)
    """
    issues = []
    
    if doc_type == "invoice":
        subtotal = extraction.get("subtotal")
        tax = extraction.get("tax")
        total = extraction.get("total")
        
        if all(v is not None for v in [subtotal, tax, total]):
            if isinstance(subtotal, (int, float)) and isinstance(tax, (int, float)) and isinstance(total, (int, float)):
                expected_total = subtotal + tax
                if abs(total - expected_total) > 0.01:
                    issues.append(
                        f"Invoice total {total} does not equal subtotal {subtotal} + tax {tax} "
                        f"(expected {expected_total:.2f})"
                    )
            else:
                issues.append("Invoice subtotal, tax, and total must be numeric")
    
    elif doc_type == "receipt":
        items = extraction.get("items", [])
        total = extraction.get("total")
        
        if items and total is not None:
            if isinstance(total, (int, float)):
                items_total = 0.0
                items_valid = True
                
                for item in items:
                    if isinstance(item, dict):
                        price = item.get("price") or item.get("amount")
                        quantity = item.get("quantity", 1)
                        
                        if price is not None and isinstance(price, (int, float)):
                            if quantity is not None and isinstance(quantity, (int, float)):
                                items_total += price * quantity
                            else:
                                items_valid = False
                                issues.append(f"Item quantity must be numeric: {item}")
                                break
                        else:
                            items_valid = False
                            issues.append(f"Item price must be numeric: {item}")
                            break
                
                if items_valid and abs(total - items_total) > 0.05:
                    issues.append(
                        f"Receipt total {total} does not match sum of items {items_total:.2f}"
                    )
            else:
                issues.append("Receipt total must be numeric")
    
    return issues
