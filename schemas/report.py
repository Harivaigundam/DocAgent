from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class ReportSection(BaseModel):
    heading: str
    content: str


class ReportFields(BaseModel):
    title: str
    author: str
    date: str
    sections: list[ReportSection] = Field(default_factory=list)
    metrics: dict = Field(default_factory=dict)
    summary: Optional[str] = None
    conclusions: list[str] = Field(default_factory=list)
