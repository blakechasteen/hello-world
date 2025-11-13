# HoloLoom Unified API Schema

**Version**: 1.0.0
**Base URL**: `http://localhost:8000`
**Protocol**: HTTP/1.1 with SSE support

## Overview

The Unified API consolidates all HoloLoom capabilities into a single, elegant interface. This document provides comprehensive schema documentation for all endpoints.

---

## Authentication

**Current**: No authentication (development mode)
**Future**: Bearer token authentication for production deployment

---

## Health & Status Endpoints

### GET /health

Check server health and uptime.

**Response** (200 OK):
```json
{
  "status": "online",
  "uptime": 3600.5,
  "version": "1.0.0"
}
```

---

### GET /stats

Get comprehensive server statistics.

**Response** (200 OK):
```json
{
  "uptime": 3600.5,
  "queries_total": 150,
  "successful_queries": 145,
  "failed_queries": 5,
  "success_rate": 96.67,
  "latency_avg": 145.2,
  "confidence_avg": 0.87,
  "queries_by_mode": {
    "direct": 50,
    "verify": 60,
    "research": 30,
    "plan_execute": 10
  },
  "memory_entities": 1247,
  "learning_active": 0.82
}
```

**Fields**:
- `uptime` (float): Server uptime in seconds
- `queries_total` (int): Total queries processed
- `successful_queries` (int): Successfully completed queries
- `failed_queries` (int): Failed queries
- `success_rate` (float): Success percentage (0-100)
- `latency_avg` (float): Average latency in milliseconds
- `confidence_avg` (float): Average confidence score (0.0-1.0)
- `queries_by_mode` (object): Query counts by reasoning mode
- `memory_entities` (int): Total entities in knowledge graph
- `learning_active` (float): Learning loop activity (0.0-1.0)

---

### GET /events

Server-Sent Events stream for real-time updates.

**Response** (200 OK, text/event-stream):

Event types:
- `stats` - Server statistics update (every 5s or on change)
- `query` - New query completed
- `ping` - Keepalive (every 30s)

**Example Events**:
```
event: stats
data: {"uptime": 3600, "queries_total": 150, ...}

event: query
data: {"text": "What is Thompson Sampling?", "confidence": 0.92, ...}

event: ping
data: {}
```

**Usage** (JavaScript):
```javascript
const eventSource = new EventSource('http://localhost:8000/events');

eventSource.addEventListener('stats', (event) => {
  const data = JSON.parse(event.data);
  console.log('Stats update:', data);
});

eventSource.addEventListener('query', (event) => {
  const data = JSON.parse(event.data);
  console.log('New query:', data);
});
```

---

## Query & Reasoning Endpoints

### POST /query

Main agentic reasoning endpoint.

**Request Body**:
```json
{
  "text": "Explain Thompson Sampling in the context of HoloLoom",
  "mode": "verify",
  "max_steps": 5,
  "context": {
    "source": "web_dashboard",
    "user_id": "optional"
  }
}
```

**Fields**:
- `text` (string, required): Query text (max 100KB)
- `mode` (string, optional): Reasoning mode - `direct`, `verify`, `research`, `plan_execute` (default: `direct`)
- `max_steps` (int, optional): Maximum reasoning steps, 1-20 (default: 5)
- `context` (object, optional): Additional context

**Reasoning Modes**:

| Mode | Description | Latency | Use Case |
|------|-------------|---------|----------|
| `direct` | Single-pass answer | ~150ms | Simple factual queries |
| `verify` | Answer + verification | ~600ms | Claims needing verification |
| `research` | Multi-query exploration | ~900ms | Open-ended research |
| `plan_execute` | Goal decomposition | ~750ms | Multi-step tasks |

**Response** (200 OK):
```json
{
  "response": "Thompson Sampling is a Bayesian exploration strategy used in HoloLoom's policy engine...",
  "confidence": 0.92,
  "reasoning_mode": "verify",
  "latency_ms": 587.3,
  "timestamp": "2025-11-13T10:30:45.123Z",
  "metadata": {
    "steps_taken": 3,
    "total_queries": 5
  }
}
```

**Fields**:
- `response` (string): Generated response text
- `confidence` (float): Confidence score (0.0-1.0)
- `reasoning_mode` (string): Actual mode used
- `latency_ms` (float): Query latency in milliseconds
- `timestamp` (string): ISO 8601 timestamp
- `metadata` (object): Additional metadata

**Errors**:
- 400 Bad Request: Invalid input (text too large, invalid mode, etc.)
- 500 Internal Server Error: Processing failed
- 503 Service Unavailable: Orchestrator not initialized

---

### GET /queries/recent

Get recent query history.

**Query Parameters**:
- `limit` (int, optional): Number of queries to return (default: 10)

**Response** (200 OK):
```json
{
  "queries": [
    {
      "text": "What is Thompson Sampling?",
      "mode": "verify",
      "confidence": 0.92,
      "latency_ms": 587.3,
      "timestamp": 1699876245.123
    },
    ...
  ]
}
```

---

## Recursive Learning Endpoints

### GET /learning/status

Get learning loop status and statistics.

**Response** (200 OK):
```json
{
  "learning_enabled": true,
  "background_learning_active": true,
  "total_queries_processed": 150,
  "total_refinements": 12,
  "hot_patterns_count": 45,
  "thompson_sampling_stats": {
    "total_arms": 5,
    "expected_rewards": [0.85, 0.72, 0.91, 0.68, 0.79]
  },
  "policy_weights": {
    "bare": 0.15,
    "fast": 0.60,
    "fused": 0.25
  },
  "last_update_timestamp": 1699876245.123
}
```

**Errors**:
- 503 Service Unavailable: Learning engine not initialized

---

### GET /learning/patterns

Get hot patterns (most accessed knowledge).

**Query Parameters**:
- `limit` (int, optional): Number of patterns to return (default: 20)

**Response** (200 OK):
```json
{
  "patterns": [
    {
      "motif": "thompson_sampling",
      "tool": "answer",
      "confidence": 0.92,
      "access_count": 15,
      "heat_score": 0.87,
      "last_accessed": 1699876245.123
    },
    ...
  ],
  "total": 45
}
```

**Note**: Currently returns placeholder data. Implementation pending.

---

## Memory & Knowledge Graph Endpoints

### GET /memory/stats

Get memory system statistics.

**Response** (200 OK):
```json
{
  "total_entities": 1247,
  "total_relationships": 3891,
  "total_memories": 1247,
  "backend": "INMEMORY"
}
```

**Fields**:
- `total_entities` (int): Total entities in knowledge graph
- `total_relationships` (int): Total relationships between entities
- `total_memories` (int): Total memory shards stored
- `backend` (string): Active backend - `INMEMORY`, `HYBRID`, or `HYPERSPACE`

---

### POST /memory/search

Search knowledge graph for relevant entities and relationships.

**Request Body**:
```json
{
  "query": "Thompson Sampling exploration",
  "limit": 10,
  "threshold": 0.5
}
```

**Fields**:
- `query` (string, required): Search query
- `limit` (int, optional): Maximum results to return (1-100, default: 10)
- `threshold` (float, optional): Similarity threshold (0.0-1.0, default: 0.5)

**Response** (200 OK):
```json
{
  "results": [
    {
      "entity": "thompson_sampling",
      "content": "Thompson Sampling is a Bayesian exploration strategy...",
      "similarity": 0.92,
      "metadata": {
        "source": "test",
        "topic": "thompson_sampling"
      }
    },
    ...
  ],
  "total": 5
}
```

**Note**: Currently returns placeholder data. Implementation pending.

---

## Safety & Alignment Endpoints

### GET /safety/status

Get safety system status.

**Response** (200 OK):
```json
{
  "guardrails_active": true,
  "deception_detector_active": true,
  "audit_trail_active": true,
  "total_actions_gated": 150,
  "blocked_actions": 3
}
```

---

### GET /safety/audit-trail

Get audit trail entries with search and filtering.

**Query Parameters**:
- `limit` (int, optional): Maximum entries to return (default: 50)
- `offset` (int, optional): Offset for pagination (default: 0)

**Response** (200 OK):
```json
{
  "entries": [
    {
      "query": "What is Thompson Sampling?",
      "action": "query_verify",
      "outcome": "success",
      "confidence": 0.92,
      "safety_score": 1.0,
      "timestamp": 1699876245.123,
      "metadata": {}
    },
    ...
  ],
  "total": 150
}
```

---

### POST /safety/gate

Gate an action through safety guardrails.

**Request Body**:
```json
{
  "action": "execute_code",
  "context": {
    "code": "import os; os.system('ls')",
    "language": "python"
  }
}
```

**Response** (200 OK):
```json
{
  "allowed": false,
  "safety_score": 0.45,
  "risk_level": "HIGH",
  "reason": "Code execution contains system calls"
}
```

**Fields**:
- `allowed` (bool): Whether action is allowed
- `safety_score` (float): Safety score (0.0-1.0)
- `risk_level` (string): Risk level - `LOW`, `MEDIUM`, `HIGH`, or `CRITICAL`
- `reason` (string): Explanation for decision

---

## Data Ingestion Endpoints

### POST /ingestion/youtube

Ingest YouTube video transcript.

**Request Body**:
```json
{
  "url": "dQw4w9WgXcQ",
  "chunk_duration": 60.0,
  "languages": ["en", "es"]
}
```

**Fields**:
- `url` (string, required): YouTube video ID or full URL
- `chunk_duration` (float, optional): Chunk duration in seconds (10-600, default: 60)
- `languages` (array, optional): Language preferences (default: ["en"])

**Response** (200 OK):
```json
{
  "job_id": "youtube_1699876245",
  "status": "processing"
}
```

**Background Processing**: Transcription happens asynchronously. Check status via `/ingestion/status`.

---

### GET /ingestion/status

Get ingestion queue status.

**Response** (200 OK):
```json
{
  "queue": [
    {
      "job_id": "youtube_1699876245",
      "type": "youtube",
      "url": "dQw4w9WgXcQ",
      "status": "processing",
      "timestamp": 1699876245.123
    },
    ...
  ],
  "total": 3
}
```

**Job Statuses**:
- `processing`: In progress
- `completed`: Successfully completed
- `failed`: Failed with error

---

## Visualization Endpoints

### GET /viz/confidence

Get confidence trajectory data for visualization.

**Response** (200 OK):
```json
{
  "confidences": [0.92, 0.88, 0.91, 0.85, 0.94],
  "timestamps": [1699876245, 1699876250, 1699876255, ...]
}
```

**Note**: Currently returns last 100 data points. Full implementation pending.

---

## System Monitor Endpoints

### GET /monitor/orchestrator

Get live orchestrator status.

**Response** (200 OK):
```json
{
  "status": "active",
  "current_queries": 2,
  "queue_depth": 5,
  "message": "Orchestrator monitoring not yet implemented"
}
```

**Note**: Full implementation pending.

---

## Error Responses

All endpoints may return the following error responses:

### 400 Bad Request
Invalid input parameters.

```json
{
  "detail": "Query text too large (max 100KB)"
}
```

### 500 Internal Server Error
Server-side processing error.

```json
{
  "detail": "Query processing failed: <error message>"
}
```

### 503 Service Unavailable
Required service not available.

```json
{
  "detail": "Learning engine not initialized"
}
```

---

## Rate Limiting

**Current**: No rate limiting (development mode)
**Future**: 60 requests per minute per IP address

---

## WebSocket Support

**Future**: WebSocket endpoint for bi-directional streaming queries (`/query/stream`)

---

## Changelog

### v1.0.0 (2025-11-13)
- Initial unified API release
- Consolidates 8 fragmented implementations
- Exposes core HoloLoom capabilities
- SSE support for real-time updates

---

## Client Libraries

### JavaScript/TypeScript

```typescript
interface HoloLoomClient {
  query(text: string, mode?: string): Promise<QueryResponse>;
  getStats(): Promise<ServerStats>;
  subscribe(eventType: string, callback: (data: any) => void): void;
}
```

**Example**:
```typescript
const client = new HoloLoomClient('http://localhost:8000');

// Query
const result = await client.query('What is Thompson Sampling?', 'verify');

// Subscribe to events
client.subscribe('stats', (stats) => {
  console.log('Stats update:', stats);
});
```

### Python

```python
import aiohttp

async def query_hololoom(text: str, mode: str = 'direct'):
    async with aiohttp.ClientSession() as session:
        async with session.post(
            'http://localhost:8000/query',
            json={'text': text, 'mode': mode}
        ) as response:
            return await response.json()

# Usage
result = await query_hololoom('What is Thompson Sampling?', 'verify')
```

---

## Contributing

API improvements and endpoint additions should follow these principles:

1. **Framework First**: Solid foundation with proper error handling
2. **Elegance**: Minimal, clean interfaces maximizing capability
3. **Verify**: Comprehensive testing before deployment

---

**End of API Schema Documentation**
