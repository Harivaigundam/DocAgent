"""Tests for schema validation and consistency checking."""
from guardrails.schema_validator import validate_schema, ValidationResult
from guardrails.consistency_checker import check_consistency


class TestValidateSchema:
    def test_valid_data_passes(self):
        data = {
            "vendor_name": "Acme Corp",
            "invoice_number": "INV-001",
            "date": "2024-01-15",
            "total": 100.00,
        }
        required_fields = ["vendor_name", "invoice_number", "date", "total"]
        result = validate_schema(data, required_fields)
        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_missing_field_fails(self):
        data = {
            "vendor_name": "Acme Corp",
            "invoice_number": "INV-001",
        }
        required_fields = ["vendor_name", "invoice_number", "date", "total"]
        result = validate_schema(data, required_fields)
        assert result.is_valid is False
        assert len(result.errors) == 2
        assert any("date" in e for e in result.errors)
        assert any("total" in e for e in result.errors)

    def test_none_field_fails(self):
        data = {
            "vendor_name": "Acme Corp",
            "invoice_number": "INV-001",
            "date": None,
            "total": 100.00,
        }
        required_fields = ["vendor_name", "invoice_number", "date", "total"]
        result = validate_schema(data, required_fields)
        assert result.is_valid is False
        assert len(result.errors) == 1
        assert "None" in result.errors[0]

    def test_negative_total_fails(self):
        data = {
            "vendor_name": "Acme Corp",
            "invoice_number": "INV-001",
            "date": "2024-01-15",
            "total": -50.00,
        }
        required_fields = ["vendor_name", "invoice_number", "date", "total"]
        result = validate_schema(data, required_fields)
        assert result.is_valid is False
        assert len(result.errors) == 1
        assert "non-negative" in result.errors[0]

    def test_string_total_fails(self):
        data = {
            "vendor_name": "Acme Corp",
            "invoice_number": "INV-001",
            "date": "2024-01-15",
            "total": "one hundred",
        }
        required_fields = ["vendor_name", "invoice_number", "date", "total"]
        result = validate_schema(data, required_fields)
        assert result.is_valid is False
        assert len(result.errors) == 1
        assert "number" in result.errors[0]

    def test_empty_required_fields(self):
        data = {"key": "value"}
        result = validate_schema(data, [])
        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_result_has_warnings_list(self):
        result = ValidationResult(is_valid=True)
        assert result.warnings == []


class TestCheckConsistency:
    def test_consistent_invoice_passes(self):
        extraction = {
            "subtotal": 100.00,
            "tax": 8.00,
            "total": 108.00,
        }
        issues = check_consistency(extraction, "invoice")
        assert len(issues) == 0

    def test_inconsistent_invoice_fails(self):
        extraction = {
            "subtotal": 100.00,
            "tax": 8.00,
            "total": 150.00,
        }
        issues = check_consistency(extraction, "invoice")
        assert len(issues) == 1
        assert "total" in issues[0].lower()

    def test_invoice_missing_fields(self):
        extraction = {"subtotal": 100.00}
        issues = check_consistency(extraction, "invoice")
        assert len(issues) == 0

    def test_invoice_non_numeric_total(self):
        extraction = {
            "subtotal": 100.00,
            "tax": 8.00,
            "total": "unknown",
        }
        issues = check_consistency(extraction, "invoice")
        assert len(issues) == 1
        assert "numeric" in issues[0]

    def test_consistent_receipt_passes(self):
        extraction = {
            "items": [
                {"name": "Item 1", "price": 10.00, "quantity": 2},
                {"name": "Item 2", "price": 5.00, "quantity": 1},
            ],
            "total": 25.00,
        }
        issues = check_consistency(extraction, "receipt")
        assert len(issues) == 0

    def test_inconsistent_receipt_fails(self):
        extraction = {
            "items": [
                {"name": "Item 1", "price": 10.00, "quantity": 2},
            ],
            "total": 50.00,
        }
        issues = check_consistency(extraction, "receipt")
        assert len(issues) == 1
        assert "sum of items" in issues[0]

    def test_receipt_no_items(self):
        extraction = {"total": 25.00}
        issues = check_consistency(extraction, "receipt")
        assert len(issues) == 0

    def test_receipt_with_amount_field(self):
        extraction = {
            "items": [
                {"name": "Item 1", "amount": 15.00},
            ],
            "total": 15.00,
        }
        issues = check_consistency(extraction, "receipt")
        assert len(issues) == 0

    def test_unknown_doc_type(self):
        extraction = {"data": "value"}
        issues = check_consistency(extraction, "unknown")
        assert len(issues) == 0
