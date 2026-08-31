import pytest
from pydantic import ValidationError

from schemas.invoice import InvoiceFields, LineItem
from schemas.receipt import ReceiptFields, ReceiptItem
from schemas.contract import ContractFields, ContractClause
from schemas.report import ReportFields, ReportSection
from schemas import SCHEMA_MAP


class TestInvoiceFields:
    def test_valid_invoice(self):
        invoice = InvoiceFields(
            vendor_name="Acme Corp",
            invoice_number="INV-001",
            date="2025-01-15",
            total=150.00,
            subtotal=130.00,
            tax=20.00,
            line_items=[
                LineItem(description="Widget", quantity=2, unit_price=50.0, amount=100.0),
                LineItem(description="Service fee", amount=30.0),
            ],
            payment_terms="Net 30",
            due_date="2025-02-14",
        )
        assert invoice.vendor_name == "Acme Corp"
        assert invoice.total == 150.00
        assert len(invoice.line_items) == 2
        assert invoice.line_items[0].quantity == 2.0

    def test_missing_required_field(self):
        with pytest.raises(ValidationError):
            InvoiceFields(
                vendor_name="Acme Corp",
                # missing invoice_number
                date="2025-01-15",
                total=100.0,
            )

    def test_total_must_be_positive(self):
        with pytest.raises(ValidationError):
            InvoiceFields(
                vendor_name="Acme Corp",
                invoice_number="INV-002",
                date="2025-01-15",
                total=-50.0,
            )

    def test_total_zero_rejected(self):
        with pytest.raises(ValidationError):
            InvoiceFields(
                vendor_name="Acme Corp",
                invoice_number="INV-003",
                date="2025-01-15",
                total=0.0,
            )

    def test_line_item_default_quantity(self):
        item = LineItem(description="Thing", amount=10.0)
        assert item.quantity == 1.0


class TestReceiptFields:
    def test_valid_receipt(self):
        receipt = ReceiptFields(
            store_name="Grocery Mart",
            date="2025-03-10",
            total=42.50,
            subtotal=38.00,
            tax=4.50,
            items=[
                ReceiptItem(name="Milk", price=3.50, quantity=2),
                ReceiptItem(name="Bread", price=2.99),
            ],
            payment_method="Credit Card",
        )
        assert receipt.store_name == "Grocery Mart"
        assert len(receipt.items) == 2
        assert receipt.payment_method == "Credit Card"

    def test_missing_required_field(self):
        with pytest.raises(ValidationError):
            ReceiptFields(
                # missing store_name
                date="2025-03-10",
                total=10.0,
            )

    def test_total_must_be_positive(self):
        with pytest.raises(ValidationError):
            ReceiptFields(
                store_name="Store",
                date="2025-03-10",
                total=-5.0,
            )


class TestContractFields:
    def test_valid_contract(self):
        contract = ContractFields(
            title="Service Agreement",
            parties=["Acme Corp", "Beta LLC"],
            effective_date="2025-01-01",
            expiration_date="2025-12-31",
            key_terms=["Confidentiality", "Non-compete"],
            clauses=[
                ContractClause(title="Term", content="This agreement lasts one year."),
                ContractClause(title="Termination", content="Either party may terminate with 30 days notice."),
            ],
            total_value=50000.0,
        )
        assert contract.title == "Service Agreement"
        assert len(contract.parties) == 2
        assert len(contract.clauses) == 2

    def test_parties_minimum_length(self):
        with pytest.raises(ValidationError):
            ContractFields(
                title="Agreement",
                parties=["Only One Party"],
                effective_date="2025-01-01",
            )

    def test_missing_required_field(self):
        with pytest.raises(ValidationError):
            ContractFields(
                # missing title
                parties=["A", "B"],
                effective_date="2025-01-01",
            )


class TestReportFields:
    def test_valid_report(self):
        report = ReportFields(
            title="Q4 2025 Analysis",
            author="Jane Smith",
            date="2025-12-31",
            sections=[
                ReportSection(heading="Executive Summary", content="Growth continued."),
                ReportSection(heading="Financials", content="Revenue up 12%."),
            ],
            metrics={"revenue": 1_200_000, "growth_pct": 12.0},
            summary="Strong quarter overall.",
            conclusions=["Expand into APAC", "Hire 5 engineers"],
        )
        assert report.title == "Q4 2025 Analysis"
        assert len(report.sections) == 2
        assert report.metrics["revenue"] == 1_200_000

    def test_missing_required_field(self):
        with pytest.raises(ValidationError):
            ReportFields(
                # missing title and author
                date="2025-12-31",
            )


class TestSchemaMap:
    def test_all_types_registered(self):
        assert "invoice" in SCHEMA_MAP
        assert "receipt" in SCHEMA_MAP
        assert "contract" in SCHEMA_MAP
        assert "report" in SCHEMA_MAP

    def test_schema_map_values_are_correct_classes(self):
        assert SCHEMA_MAP["invoice"] is InvoiceFields
        assert SCHEMA_MAP["receipt"] is ReceiptFields
        assert SCHEMA_MAP["contract"] is ContractFields
        assert SCHEMA_MAP["report"] is ReportFields
