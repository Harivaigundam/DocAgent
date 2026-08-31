from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class ReceiptItem(BaseModel):
    name: str
    price: float
    quantity: Optional[float] = Field(default=1.0, ge=0)


class ReceiptFields(BaseModel):
    store_name: str
    date: str
    total: float = Field(gt=0)
    subtotal: Optional[float] = None
    tax: Optional[float] = None
    items: list[ReceiptItem] = Field(default_factory=list)
    payment_method: Optional[str] = None
