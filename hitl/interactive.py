from fastapi import WebSocket
import logging

logger = logging.getLogger(__name__)


class InteractiveReview:
    def __init__(self):
        self._sessions: dict[str, WebSocket] = {}

    async def start_session(self, session_id: str, websocket: WebSocket):
        await websocket.accept()
        self._sessions[session_id] = websocket

    async def request_review(self, session_id: str, data: dict):
        ws = self._sessions.get(session_id)
        if ws:
            await ws.send_json({
                "type": "review_request",
                "document_id": data["document_id"],
                "extraction": data["extraction"],
                "confidence": data["confidence"],
                "questions": data.get("questions", []),
            })

    async def receive_response(self, session_id: str) -> dict:
        ws = self._sessions.get(session_id)
        if ws:
            return await ws.receive_json()
        return {}

    async def close_session(self, session_id: str):
        ws = self._sessions.pop(session_id, None)
        if ws:
            await ws.close()
