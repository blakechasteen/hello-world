# HoloLoom Agentic API Server

FastAPI server exposing HoloLoom's agentic intelligence to external clients (VS Code Squad extension, web apps, etc).

## Quick Start

```bash
# Install dependencies
pip install fastapi uvicorn

# Development mode (with auto-reload)
PYTHONPATH=. uvicorn HoloLoom.server.agentic_api:app --reload --port 8000

# Production mode
PYTHONPATH=. uvicorn HoloLoom.server.agentic_api:app --host 0.0.0.0 --port 8000 --workers 4
```

## Endpoints

### Health Check
```bash
GET http://localhost:8000/health

Response:
{
  "status": "ok",
  "service": "HoloLoom Agentic API",
  "version": "1.0.0"
}
```

### Query (Main Endpoint)
```bash
POST http://localhost:8000/query
Content-Type: application/json

{
  "text": "Explain this TypeScript code",
  "context": {
    "languageId": "typescript",
    "fileName": "example.ts",
    "selection": "function foo() { return 42; }"
  },
  "mode": "verify",
  "max_steps": 5
}

Response:
{
  "response": "This is a simple function that returns 42...",
  "confidence": 0.92,
  "reasoning_mode": "verify",
  "steps_taken": [
    {
      "type": "initial_answer",
      "query": "Explain this TypeScript code",
      "confidence": 0.85,
      "tool": "answer"
    },
    {
      "type": "verification",
      "query": "What are weaknesses in this answer?",
      "confidence": 0.90
    }
  ],
  "total_queries": 4,
  "total_duration_ms": 587.3,
  "verification": {
    "verified": true,
    "contradictions": [],
    "supporting_evidence": ["...", "..."]
  }
}
```

### Statistics
```bash
GET http://localhost:8000/stats

Response:
{
  "orchestrator_ready": true,
  "memory_shards": 150,
  "audit_trail_entries": 342
}
```

### Audit Trail
```bash
GET http://localhost:8000/audit-trail?limit=10

Response:
{
  "total": 342,
  "entries": [...]
}
```

## Reasoning Modes

| Mode | Description | Latency | Use Case |
|------|-------------|---------|----------|
| `direct` | Single-pass answer | ~150ms | Simple factual queries |
| `verify` | Answer + verification | ~600ms | Claims needing verification |
| `research` | Multi-query exploration | ~900ms | Open-ended research |
| `plan_execute` | Goal decomposition | ~750ms | Multi-step tasks |

## VS Code Integration

This server is designed to work with the Squad VS Code extension.

**squad/src/HoloLoomBridge.ts**:
```typescript
const bridge = new HoloLoomBridge('http://localhost:8000');

const result = await bridge.query(
  "Explain this code",
  codeContext,
  'verify',
  5
);

console.log(result.response);
console.log(result.verification.verified);
```

## Configuration

Edit startup() in `agentic_api.py`:

```python
@app.on_event("startup")
async def startup():
    # Load config
    state.config = Config.fast()  # or Config.fused() for higher quality
    state.config.enable_agentic_reasoning = True

    # Load your data
    state.shards = load_from_database()  # Your data source
```

## Development

```bash
# Run with auto-reload
uvicorn HoloLoom.server.agentic_api:app --reload

# Test health
curl http://localhost:8000/health

# Test query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"text": "What is Thompson Sampling?", "mode": "direct"}'
```

## Architecture

```
VS Code Extension (TypeScript)
    ↓ HTTP
FastAPI Server (Python)
    ↓
AgenticOrchestrator
    ├─ FullLearningEngine
    ├─ AuditTrail
    └─ ReasoningModes (DIRECT/VERIFY/RESEARCH/PLAN_EXECUTE)
```

## See Also

- [AGENTIC_VSCODE_INTEGRATION.md](../../AGENTIC_VSCODE_INTEGRATION.md) - Full integration guide
- [HoloLoom/agentic/](../agentic/) - Agentic reasoning implementation
- [squad/src/HoloLoomBridge.ts](../../squad/src/HoloLoomBridge.ts) - TypeScript client
