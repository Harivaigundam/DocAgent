from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class ContractClause(BaseModel):
    title: str
    content: str


class ContractFields(BaseModel):
    title: str
    parties: list[str] = Field(min_length=2)
    effective_date: str
    expiration_date: Optional[str] = None
    key_terms: list[str] = Field(default_factory=list)
    clauses: list[ContractClause] = Field(default_factory=list)
    total_value: Optional[float] = None
