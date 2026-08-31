import asyncio
from datetime import datetime
from dataclasses import dataclass, field


@dataclass
class ReviewItem:
    document_id: str
    extraction: dict
    confidence: float
    timestamp: datetime
    status: str = "pending_review"
    human_feedback: dict = field(default_factory=dict)


class BatchReviewQueue:
    def __init__(self):
        self._queue: dict[str, ReviewItem] = {}

    async def add(self, document_id: str, extraction: dict, confidence: float):
        self._queue[document_id] = ReviewItem(
            document_id=document_id,
            extraction=extraction,
            confidence=confidence,
            timestamp=datetime.now(),
        )

    async def get_pending(self) -> list[dict]:
        return [
            {
                "document_id": item.document_id,
                "extraction": item.extraction,
                "confidence": item.confidence,
                "timestamp": item.timestamp.isoformat(),
                "status": item.status,
            }
            for item in self._queue.values()
            if item.status == "pending_review"
        ]

    async def approve(self, document_id: str, feedback: dict):
        if document_id in self._queue:
            self._queue[document_id].status = "approved"
            self._queue[document_id].human_feedback = feedback

    async def reject(self, document_id: str, feedback: dict):
        if document_id in self._queue:
            self._queue[document_id].status = "rejected"
            self._queue[document_id].human_feedback = feedback
