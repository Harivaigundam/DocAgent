from schemas.invoice import InvoiceFields
from schemas.receipt import ReceiptFields
from schemas.contract import ContractFields
from schemas.report import ReportFields

SCHEMA_MAP: dict[str, type] = {
    "invoice": InvoiceFields,
    "receipt": ReceiptFields,
    "contract": ContractFields,
    "report": ReportFields,
}

__all__ = [
    "InvoiceFields",
    "ReceiptFields",
    "ContractFields",
    "ReportFields",
    "SCHEMA_MAP",
]
