# Documentor Multi-Agent System — Design Spec

**Date:** 2026-08-26
**Status:** Approved
**Purpose:** Demo POC for client presentation — multi-agent document processing with cost-aware model routing, guardrails, and HITL.

---

## 1. Overview

Documentor is a multi-agent document processing system that automatically detects document types (invoices, receipts, contracts, reports), extracts structured fields using specialized agents, and validates outputs through guardrails. A cost-aware model router cascades from free (Ollama) to paid (OpenAI) models based on confidence scores, minimizing cost while maximizing extraction quality.

### Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Architecture | Agent-as-MCP-Server | Independent agents, testable, extensible |
| LLM Providers | Ollama → Groq → OpenAI | Cost cascade from free to paid |
| Orchestration | LangGraph StateGraph | Proven pattern (TripAgent reference) |
| HITL | Hybrid (interactive + batch) | Critical docs get interactive review; others batched |
| Agent Selection | Auto-detection by content analysis | No user configuration needed |
| Testing | pytest + pytest-asyncio | Standard Python testing ecosystem |
| Tech Stack | Python + LangGraph + FastAPI | Matches TripAgent reference |

---

## 2. System Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        DOCUMENTOR MULTI-AGENT SYSTEM                     │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌────────────┐    ┌─────────────────────────────────────────────────┐   │
│  │  FastAPI   │◄──►│              LangGraph Orchestrator              │   │
│  │  Gateway   │    │  ┌──────────┐  ┌──────────┐  ┌──────────────┐  │   │
│  │ REST + WS  │    │  │   Doc    │  │  Agent   │  │   Model      │  │   │
│  └────────────┘    │  │  Type    │→ │ Selector │→ │   Router     │  │   │
│                    │  │ Detector │  │          │  │ (cost-aware) │  │   │
│                    │  └──────────┘  └──────────┘  └──────┬───────┘  │   │
│                    └──────────────────────────────────────┼──────────┘   │
│                                                          │              │
│         ┌────────────────────────────────────────────────┼──────┐       │
│         │              MCP Protocol Layer                │      │       │
│         │  ┌──────────┬──────────┬──────────┬──────────┐│      │       │
│         │  │ Invoice  │ Receipt  │ Contract │  Report  ││      │       │
│         │  │ Agent    │ Agent    │ Agent    │  Agent   ││      │       │
│         │  │ (MCP Srv)│ (MCP Srv)│ (MCP Srv)│ (MCP Srv)││      │       │
│         │  └────┬─────┴────┬─────┴────┬─────┴────┬─────┘│      │       │
│         │       │          │          │          │       │      │       │
│         │  ┌────▼──────────▼──────────▼──────────▼─────┐│      │       │
│         │  │         Guardrail Agent (MCP Srv)         ││      │       │
│         │  │  - Confidence validation                  ││      │       │
│         │  │  - Schema validation                     ││      │       │
│         │  │  - PII detection                         ││      │       │
│         │  └──────────────────────────────────────────┘│      │       │
│         └──────────────────────────────────────────────┘      │       │
│                                                               │       │
│  ┌─────────────────────────────────────────────────────────────┘       │
│  │  Model Router (Cost Cascade)                                       │
│  │  ┌─────────┐    ┌─────────┐    ┌─────────┐                        │
│  │  │ Ollama  │───►│  Groq   │───►│ OpenAI  │                        │
│  │  │ (free)  │    │ (free/  │    │ (paid)  │                        │
│  │  │ local   │    │  rate-  │    │ gpt-4o  │                        │
│  │  │         │    │ limited)│    │         │                        │
│  │  └─────────┘    └─────────┘    └─────────┘                        │
│  │  confidence < 0.7 → escalate to next model                        │
│  └───────────────────────────────────────────────────────────────────  │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │                    Human-in-the-Loop Layer                         │  │
│  │  Interactive: WebSocket approval at key decision points            │  │
│  │  Batch: Low-confidence results queued for review                   │  │
│  └────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────┘
```

### Core Flow

1. Document uploaded via FastAPI (`POST /api/extract`)
2. Orchestrator detects document type (confidence-scored)
3. Model Router selects cheapest model that meets confidence threshold
4. Document routed to specialized agent (Invoice/Receipt/Contract/Report)
5. Agent processes document, returns structured extraction
6. Guardrail Agent validates output (schema, confidence, PII)
7. If confidence < threshold → escalate model or route to HITL
8. Final result returned or queued for batch human review

---

## 3. Agent Definitions

### Agent Inventory

| # | Agent | Role | MCP Tools Exposed | LLM Required |
|---|-------|------|-------------------|--------------|
| 1 | **Orchestrator** | Routes documents, manages flow | `detect_type`, `select_agent`, `route_document` | Yes (model router) |
| 2 | **Invoice Agent** | Extract invoice fields | `extract_invoice`, `validate_invoice_fields` | Yes (model router) |
| 3 | **Receipt Agent** | Extract receipt fields | `extract_receipt`, `validate_receipt_fields` | Yes (model router) |
| 4 | **Contract Agent** | Extract contract fields | `extract_contract`, `summarize_clauses` | Yes (model router) |
| 5 | **Report Agent** | Extract report fields | `extract_report`, `summarize_report` | Yes (model router) |
| 6 | **Guardrail Agent** | Validates all outputs | `validate_schema`, `check_confidence`, `detect_pii` | No (rule-based) |
| 7 | **Model Router** | Cost-aware model selection | `route_to_model`, `get_model_cost`, `check_confidence_threshold` | No (logic-based) |

### MCP Server Pattern

Each extraction agent follows this pattern:

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("AgentName")

@mcp.tool()
async def extract_<type>(document_content: str, metadata: dict) -> dict:
    """
    Extract structured fields from a <type> document.
    
    Returns:
        {
            "fields": {...},
            "confidence": 0.85,
            "model_used": "ollama",
            "cost": 0.0,
            "requires_review": False
        }
    """
    model = await route_to_model(document_content)
    result = await extract_with_model(model, document_content, SCHEMA)
    validated = await validate_output(result, SCHEMA)
    return validated

if __name__ == "__main__":
    mcp.run()  # Runs as stdio subprocess
```

### MCP Client Connection

```python
from langchain_mcp_adapters.client import MultiServerMCPClient

client = MultiServerMCPClient({
    "invoice_agent": {
        "transport": "stdio",
        "command": "python",
        "args": ["agents/invoice_agent.py"],
        "env": {"AGENT_ROLE": "invoice"}
    },
    # ... same pattern for all agents
})
```

---

## 4. Model Router & Cost Cascade

### Model Cascade Tiers

| Tier | Model | Cost/Doc | Speed | Confidence Threshold |
|------|-------|----------|-------|---------------------|
| **Tier 0** | Ollama (llama3.1:8b) | $0.00 | Fast (local) | 0.70 |
| **Tier 1** | Groq (llama3-70b) | $0.00 | Very fast | 0.75 |
| **Tier 2** | OpenAI (gpt-4o-mini) | ~$0.001 | Fast | 0.80 |
| **Tier 3** | OpenAI (gpt-4o) | ~$0.01 | Moderate | 0.85 |

### Routing Logic

```python
class ModelRouter:
    async def route(self, content: str, task_complexity: str = "standard") -> ModelChoice:
        for tier in self.CASCADE:
            result = await self._try_model(tier, content)
            if result.confidence >= tier["confidence_threshold"]:
                return ModelChoice(
                    model=tier["name"],
                    confidence=result.confidence,
                    cost=tier["cost_per_doc"],
                    tier=tier["name"]
                )
        return ModelChoice(model="openai:gpt-4o", confidence=0.0, cost=0.01, tier="fallback")
```

### Confidence Scoring

- **Field-level confidence**: Each extracted field gets a confidence score
- **Overall document confidence**: Weighted average of field confidences
- **Schema completeness**: Bonus for filling all required fields
- **Cross-field consistency**: Penalty for contradictory values

### Cost Tracking

```python
@dataclass
class CostTracker:
    total_cost: float = 0.0
    docs_processed: int = 0
    model_usage: dict = field(default_factory=dict)
    
    def record(self, model: str, cost: float):
        self.total_cost += cost
        self.docs_processed += 1
        self.model_usage[model] = self.model_usage.get(model, 0) + 1
```

---

## 5. Guardrails

### Validation Layers

| Layer | Check | Action on Fail |
|-------|-------|----------------|
| **Schema Validation** | All required fields present, correct types | Reject → re-extract with different model |
| **Confidence Threshold** | Overall confidence ≥ 0.7 | If < 0.7 → HITL batch queue |
| **PII Detection** | Scan for SSN, credit cards, phones, emails | Flag for redaction |
| **Cross-field Consistency** | Dates logical, totals match line items | Flag → human review |
| **Output Sanitization** | Remove chain-of-thought tags, normalize | Clean before returning |

### PII Detection Patterns

```python
PII_PATTERNS = {
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    "credit_card": r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",
    "email": r"\b[\w.+-]+@[\w-]+\.[\w.]+\b",
    "phone": r"\b(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
    "account_number": r"\b\d{8,17}\b",
}
```

---

## 6. Human-in-the-Loop (HITL)

### Interactive HITL (WebSocket)

Triggered when:
- Document type detection confidence < 0.6
- First extraction attempt fails
- User explicitly requests review
- Critical document (high-value invoice, legal contract)

```python
@app.websocket("/ws/review/{session_id}")
async def hitl_review(websocket: WebSocket, session_id: str):
    await websocket.accept()
    await websocket.send_json({
        "type": "review_request",
        "document": doc_data,
        "extraction": extraction_result,
        "confidence": confidence_score,
        "questions": ["Is this an invoice?", "Is the total correct?"]
    })
    response = await websocket.receive_json()
    # Incorporate human feedback
```

### Batch HITL (Queue)

Triggered when:
- Confidence between 0.4 - 0.7
- Non-critical documents
- High volume processing

### HITL Decision Matrix

| Confidence | Document Type | Action |
|-----------|---------------|--------|
| ≥ 0.8 | Any | Auto-approve |
| 0.6 - 0.8 | Invoice/Contract | Interactive review |
| 0.6 - 0.8 | Receipt/Report | Batch queue |
| < 0.6 | Any | Interactive + re-extract |
| PII detected | Any | Redaction review |
| Extraction failed | N/A | Retry → HITL |

---

## 7. Project Structure

```
C:\LLM ENGINEER\Doc_agent\
├── .env                          # API keys
├── .env.example                  # Template
├── .gitignore
├── pyproject.toml
├── requirements.txt
├── README.md
│
├── app.py                        # FastAPI entry point
├── orchestrator.py               # LangGraph orchestrator
├── state.py                      # Shared state TypedDict
│
├── agents/                       # Each agent = MCP server
│   ├── __init__.py
│   ├── base_agent.py             # Base MCP agent class
│   ├── invoice_agent.py
│   ├── receipt_agent.py
│   ├── contract_agent.py
│   ├── report_agent.py
│   └── guardrail_agent.py
│
├── model_router/                 # Cost-aware model routing
│   ├── __init__.py
│   ├── router.py
│   ├── cost_tracker.py
│   └── confidence.py
│
├── mcp/                          # MCP infrastructure
│   ├── __init__.py
│   ├── client.py
│   └── server.py
│
├── guardrails/                   # Guardrail logic
│   ├── __init__.py
│   ├── schema_validator.py
│   ├── pii_detector.py
│   └── consistency_checker.py
│
├── hitl/                         # Human-in-the-loop
│   ├── __init__.py
│   ├── interactive.py
│   └── batch_queue.py
│
├── prompts/                      # LLM prompt templates
│   ├── __init__.py
│   ├── type_detection.py
│   ├── invoice_extraction.py
│   ├── receipt_extraction.py
│   ├── contract_extraction.py
│   └── report_extraction.py
│
├── schemas/                      # Output Pydantic models
│   ├── __init__.py
│   ├── invoice.py
│   ├── receipt.py
│   ├── contract.py
│   └── report.py
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── unit/
│   │   ├── __init__.py
│   │   ├── test_model_router.py
│   │   ├── test_guardrails.py
│   │   ├── test_confidence.py
│   │   ├── test_pii_detector.py
│   │   ├── test_schemas.py
│   │   └── test_cost_tracker.py
│   ├── integration/
│   │   ├── __init__.py
│   │   ├── test_mcp_client.py
│   │   ├── test_orchestrator.py
│   │   ├── test_agent_pipeline.py
│   │   └── test_hitl.py
│   └── fixtures/
│       ├── sample_invoice.json
│       ├── sample_receipt.json
│       ├── sample_contract.json
│       └── sample_report.json
│
├── docs/superpowers/specs/       # Design specs
│
├── venv/                         # Python virtual environment
│
└── static/                       # Frontend
    ├── index.html
    ├── style.css
    └── script.js
```

---

## 8. REST API

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/extract` | Upload document → auto-detect → extract |
| `POST` | `/api/extract/{doc_type}` | Upload with specified type → extract |
| `GET` | `/api/status/{doc_id}` | Check extraction status |
| `GET` | `/api/review/queue` | Get batch review queue |
| `POST` | `/api/review/{doc_id}` | Submit human review |
| `WS` | `/ws/review/{session_id}` | Interactive review WebSocket |
| `GET` | `/api/costs` | Cost tracking report |
| `GET` | `/api/health` | System health |
| `GET` | `/api/models` | Available models & status |

---

## 9. Testing Strategy

### Unit Tests (~20+ tests)

- Model router cascade logic (all tier transitions)
- Confidence scoring computation
- PII detection accuracy (all patterns)
- Schema validation (valid + invalid inputs)
- Cost tracking calculations
- Each agent's extraction logic in isolation

### Integration Tests (~10+ tests)

- MCP client ↔ agent server communication
- Full orchestrator pipeline (upload → detect → extract → validate → return)
- HITL interactive flow (WebSocket mock)
- HITL batch queue flow
- Model fallback cascade (mock LLM failures)
- End-to-end with synthetic documents

### Synthetic Test Documents

1. **Invoice** - Standard invoice with vendor, line items, tax, total
2. **Receipt** - Store receipt with items, payment method, date
3. **Contract** - Service agreement with parties, terms, signatures
4. **Report** - Business report with sections, metrics, conclusions

---

## 10. Dependencies

### Python Packages

```
# Core
langgraph>=0.2.0
langchain>=0.3.0
langchain-core>=0.3.0
langchain-mcp-adapters>=0.1.0
fastapi>=0.115.0
uvicorn[standard]>=0.32.0
websockets>=13.0

# MCP
mcp>=1.28.1

# LLM Providers
langchain-groq>=0.2.0
langchain-openai>=0.3.0
ollama>=0.4.0

# Validation & Guardrails
pydantic>=2.10.0

# Utilities
python-dotenv>=1.0.0
certifi>=2024.0.0
aiohttp>=3.11.0

# Testing
pytest>=8.3.0
pytest-asyncio>=0.24.0
pytest-mock>=3.14.0
pytest-cov>=6.0.0
aioresponses>=0.7.0
httpx>=0.28.0
```

---

## 11. Environment Variables

```bash
# .env.example
GROQ_API_KEY=groq_xxxxx
OPENAI_API_KEY=sk-xxxxx
OLLAMA_BASE_URL=http://localhost:11434

# Optional
HITL_ENABLED=true
CONFIDENCE_THRESHOLD=0.7
MAX_COST_PER_DOC=0.05
LOG_LEVEL=INFO
```

---

## 12. Subagent Distribution (5+ subagents)

For parallel implementation, work is split across subagents:

| # | Subagent | Task | Dependencies |
|---|----------|------|--------------|
| 1 | **Scaffold Agent** | Project structure, .env, pyproject.toml, virtual env | None |
| 2 | **Schema Agent** | Pydantic models, prompt templates, schemas | None |
| 3 | **MCP Agent** | MCP client, server, base agent class | Scaffold |
| 4 | **Router Agent** | Model router, cost tracker, confidence scoring | None |
| 5 | **Guardrail Agent** | Schema validator, PII detector, consistency checker | None |
| 6 | **HITL Agent** | Interactive WebSocket, batch queue | None |
| 7 | **Orchestrator Agent** | LangGraph state graph, agent wiring | Schemas, MCP |
| 8 | **Integration Agent** | FastAPI app, tests, frontend | All above |

---

## 13. Success Criteria

- [ ] All 7 agents implemented as MCP servers
- [ ] Model router cascades Ollama → Groq → OpenAI
- [ ] Cost tracker reports per-doc and total costs
- [ ] Guardrails validate schema, confidence, PII
- [ ] HITL interactive (WebSocket) and batch (queue) work
- [ ] 4 synthetic test documents pass extraction
- [ ] 20+ unit tests pass
- [ ] 10+ integration tests pass
- [ ] FastAPI serves all endpoints
- [ ] Frontend allows document upload and result viewing
- [ ] Virtual environment created and all deps installed
- [ ] Demo-ready for client presentation
