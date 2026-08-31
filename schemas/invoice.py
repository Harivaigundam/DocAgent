from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class LineItem(BaseModel):
    description: str
    quantity: Optional[float] = Field(default=1.0, ge=0)
    unit_price: Optional[float] = None
    amount: float


class InvoiceFields(BaseModel):
    vendor_name: str
    invoice_number: str
    date: str
    total: float = Field(gt=0)
    subtotal: Optional[float] = None
    tax: Optional[float] = None
    line_items: list[LineItem] = Field(default_factory=list)
    payment_terms: Optional[str] = None
    due_date: Optional[str] = None
