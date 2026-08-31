import uuid
import logging
import uvicorn
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse

from orchestrator import DocumentOrchestrator
from parallel_orchestrator import ParallelDocumentOrchestrator
from hitl.batch_queue import BatchReviewQueue
from file_parser import extract_text_from_bytes

logger = logging.getLogger(__name__)

orchestrator = DocumentOrchestrator()
parallel_orchestrator = ParallelDocumentOrchestrator()
batch_queue = BatchReviewQueue()
active_sessions: dict[str, WebSocket] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(
    title="Documentor API",
    description="Multi-agent document extraction system",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    with open("static/index.html", "r") as f:
        return HTMLResponse(content=f.read())


@app.post("/api/extract")
async def extract_auto(file: UploadFile = File(...)):
    content = await file.read()
    try:
        text = extract_text_from_bytes(content, file.filename or "unknown.txt")
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": f"File parsing failed: {str(e)}"})
    doc_id = str(uuid.uuid4())

    try:
        result = await orchestrator.run(doc_id, text)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error("Extraction failed: %s", e)
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/api/extract/parallel")
async def extract_parallel(file: UploadFile = File(...)):
    content = await file.read()
    try:
        text = extract_text_from_bytes(content, file.filename or "unknown.txt")
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": f"File parsing failed: {str(e)}"})
    doc_id = str(uuid.uuid4())

    try:
        result = await parallel_orchestrator.run(doc_id, text)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error("Parallel extraction failed: %s", e)
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/api/extract/{doc_type}")
async def extract_with_type(doc_type: str, file: UploadFile = File(...)):
    content = await file.read()
    try:
        text = extract_text_from_bytes(content, file.filename or "unknown.txt")
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": f"File parsing failed: {str(e)}"})
    doc_id = str(uuid.uuid4())

    try:
        initial_state = {
            "messages": [],
            "document_id": doc_id,
            "document_content": text,
            "document_type": doc_type,
            "type_confidence": 1.0,
            "extraction_result": {},
            "extraction_confidence": 0.0,
            "model_used": "",
            "model_cost": 0.0,
            "requires_review": False,
            "review_status": "",
            "guardrail_passed": False,
            "guardrail_issues": [],
            "final_output": {},
        }

        agent_class = None
        from agents import AGENT_MAP
        agent_class = AGENT_MAP.get(doc_type)

        if not agent_class:
            return JSONResponse(status_code=400, content={"error": f"Unknown document type: {doc_type}"})

        agent = agent_class()
        result = await agent._extract(text, {})

        orchestrator.cost_tracker.record(result["model_used"], result["cost"])

        return JSONResponse(content={
            "document_id": doc_id,
            "document_type": doc_type,
            "extraction": result["fields"],
            "confidence": result["confidence"],
            "model_used": result["model_used"],
            "cost": result["cost"],
            "guardrail_passed": True,
            "requires_review": result["requires_review"],
        })
    except Exception as e:
        logger.error("Extraction failed: %s", e)
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/api/review/queue")
async def get_review_queue():
    items = await batch_queue.get_pending()
    return JSONResponse(content={"items": items, "count": len(items)})


@app.post("/api/review/{doc_id}")
async def submit_review(doc_id: str, feedback: dict):
    if feedback.get("approved"):
        await batch_queue.approve(doc_id, feedback)
    else:
        await batch_queue.reject(doc_id, feedback)
    return JSONResponse(content={"status": "reviewed", "doc_id": doc_id})


@app.websocket("/ws/review/{session_id}")
async def websocket_review(websocket: WebSocket, session_id: str):
    await websocket.accept()
    active_sessions[session_id] = websocket
    try:
        while True:
            data = await websocket.receive_json()
            await websocket.send_json({"status": "received", "data": data})
    except WebSocketDisconnect:
        active_sessions.pop(session_id, None)


@app.get("/api/costs")
async def get_costs():
    return JSONResponse(content=orchestrator.cost_tracker.summary())


@app.get("/api/models")
async def get_models():
    models = []
    for tier in orchestrator.router.CASCADE:
        models.append({
            "name": tier.name,
            "model": tier.model,
            "cost_per_doc": tier.cost_per_doc,
            "confidence_threshold": tier.confidence_threshold,
            "provider": tier.provider,
        })
    return JSONResponse(content={"models": models})


@app.get("/api/health")
async def health_check():
    return JSONResponse(content={"status": "healthy", "version": "1.0.0"})

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8001,
        reload=True
    )