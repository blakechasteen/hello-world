# API Reference

Complete REST and WebSocket API documentation for the Workflow Builder.

## Overview

The Workflow Builder exposes two API interfaces:
- **REST API**: Standard HTTP endpoints for workflow management
- **WebSocket API**: Real-time communication for execution and collaboration

**Base URLs**:
- REST: `http://localhost:8001/api`
- WebSocket: `ws://localhost:8001/ws`

## Authentication

### API Key Authentication

```http
GET /api/workflows
Authorization: Bearer your-api-key
```

### Session Authentication

For browser clients:
```http
POST /api/auth/login
Content-Type: application/json

{
  "username": "user@example.com",
  "password": "password"
}
```

Response:
```json
{
  "session_token": "eyJhbGc...",
  "expires_at": "2025-12-16T10:30:00Z"
}
```

## REST API

### Workflows

#### List Workflows

```http
GET /api/workflows
```

**Query Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `page` | int | Page number (default: 1) |
| `limit` | int | Items per page (default: 20) |
| `search` | string | Search term |
| `category` | string | Filter by category |

**Response**:
```json
{
  "workflows": [
    {
      "id": "wf-abc123",
      "name": "Research Pipeline",
      "description": "Multi-query research with verification",
      "created_at": "2025-12-15T10:30:00Z",
      "updated_at": "2025-12-15T14:45:00Z",
      "node_count": 5,
      "tags": ["research", "rag"]
    }
  ],
  "total": 42,
  "page": 1,
  "pages": 3
}
```

#### Get Workflow

```http
GET /api/workflows/{workflow_id}
```

**Response**:
```json
{
  "id": "wf-abc123",
  "version": "1.0",
  "name": "Research Pipeline",
  "description": "Multi-query research with verification",
  "nodes": [...],
  "connections": [...],
  "metadata": {...}
}
```

#### Create Workflow

```http
POST /api/workflows
Content-Type: application/json

{
  "name": "New Workflow",
  "description": "My workflow description",
  "nodes": [],
  "connections": []
}
```

**Response**:
```json
{
  "id": "wf-xyz789",
  "created_at": "2025-12-15T10:30:00Z"
}
```

#### Update Workflow

```http
PUT /api/workflows/{workflow_id}
Content-Type: application/json

{
  "name": "Updated Name",
  "nodes": [...],
  "connections": [...]
}
```

#### Delete Workflow

```http
DELETE /api/workflows/{workflow_id}
```

**Response**:
```json
{
  "deleted": true
}
```

### Execution

#### Execute Workflow

```http
POST /api/workflow/execute
Content-Type: application/json

{
  "workflow_id": "wf-abc123",
  "input_data": {
    "query": "What is Thompson Sampling?"
  },
  "options": {
    "timeout": 300,
    "stream": false,
    "debug": false
  }
}
```

**Response** (sync):
```json
{
  "job_id": "job-123",
  "status": "completed",
  "result": {
    "output": {
      "response": "Thompson Sampling is...",
      "confidence": 0.92
    },
    "execution_time_ms": 1250,
    "nodes_executed": 5
  }
}
```

#### Execute Workflow (Async)

```http
POST /api/workflow/execute-async
Content-Type: application/json

{
  "workflow_id": "wf-abc123",
  "input_data": {...}
}
```

**Response**:
```json
{
  "job_id": "job-456",
  "status": "pending",
  "status_url": "/api/jobs/job-456"
}
```

#### Get Job Status

```http
GET /api/jobs/{job_id}
```

**Response**:
```json
{
  "job_id": "job-456",
  "status": "running",
  "progress": {
    "current_node": "query-1",
    "nodes_completed": 2,
    "nodes_total": 5,
    "percent": 40
  },
  "started_at": "2025-12-15T10:30:00Z"
}
```

#### Cancel Job

```http
POST /api/jobs/{job_id}/cancel
```

**Response**:
```json
{
  "job_id": "job-456",
  "status": "cancelled"
}
```

### Templates

#### List Templates

```http
GET /api/templates
```

**Query Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `category` | string | Filter by category |
| `search` | string | Search term |
| `sort` | string | Sort by: downloads, rating, recent |

**Response**:
```json
{
  "templates": [
    {
      "id": "tpl-basic-rag",
      "name": "Basic RAG Pipeline",
      "description": "Simple retrieve-and-generate",
      "category": "RAG",
      "node_count": 3,
      "downloads": 1234,
      "rating": 4.8
    }
  ]
}
```

#### Get Template

```http
GET /api/templates/{template_id}
```

#### Create Template

```http
POST /api/templates
Content-Type: application/json

{
  "name": "My Template",
  "description": "Template description",
  "category": "custom",
  "nodes": [...],
  "connections": [...]
}
```

### Export/Import

#### Export Workflow

```http
POST /api/workflow/export
Content-Type: application/json

{
  "workflow_id": "wf-abc123",
  "format": "python",
  "options": {
    "includeDocstrings": true,
    "includeMain": true
  }
}
```

**Response**:
```json
{
  "format": "python",
  "content": "# Workflow: ...\nimport asyncio...",
  "filename": "research_pipeline.py"
}
```

#### Import Workflow

```http
POST /api/workflow/import
Content-Type: application/json

{
  "format": "json",
  "content": "{...workflow JSON...}",
  "options": {
    "merge": false
  }
}
```

### Collaboration

#### Create Session

```http
POST /api/collaboration/create
Content-Type: application/json

{
  "workflow_id": "wf-abc123"
}
```

**Response**:
```json
{
  "session_id": "collab-xyz",
  "join_url": "https://app.example.com/collab/xyz",
  "expires_at": "2025-12-16T10:30:00Z"
}
```

#### Join Session

```http
POST /api/collaboration/{session_id}/join
Content-Type: application/json

{
  "display_name": "Alice"
}
```

#### Leave Session

```http
POST /api/collaboration/{session_id}/leave
```

## WebSocket API

### Connection

```javascript
const ws = new WebSocket('ws://localhost:8001/ws');

ws.onopen = () => {
  // Authenticate
  ws.send(JSON.stringify({
    type: 'auth',
    token: 'your-api-key'
  }));
};
```

### Message Types

#### Subscribe to Execution

```javascript
// Subscribe
ws.send(JSON.stringify({
  type: 'subscribe',
  channel: 'execution',
  job_id: 'job-123'
}));

// Receive events
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  switch (data.type) {
    case 'node_start':
      console.log(`Starting: ${data.node_id}`);
      break;
    case 'node_complete':
      console.log(`Complete: ${data.node_id}`, data.output);
      break;
    case 'workflow_complete':
      console.log('Done!', data.result);
      break;
    case 'error':
      console.error('Error:', data.error);
      break;
  }
};
```

#### Collaboration Events

```javascript
// Join collaboration session
ws.send(JSON.stringify({
  type: 'join_session',
  session_id: 'collab-xyz',
  display_name: 'Alice'
}));

// Send cursor position
ws.send(JSON.stringify({
  type: 'cursor_move',
  x: 150,
  y: 200
}));

// Request node lock
ws.send(JSON.stringify({
  type: 'node_lock',
  node_id: 'query-1',
  action: 'acquire'
}));

// Broadcast workflow operation
ws.send(JSON.stringify({
  type: 'workflow_operation',
  op: 'update_node',
  node_id: 'query-1',
  data: { label: 'New Label' }
}));
```

### Event Reference

#### Execution Events

| Event | Description | Payload |
|-------|-------------|---------|
| `node_start` | Node execution started | `{node_id, timestamp}` |
| `node_progress` | Node progress update | `{node_id, progress, message}` |
| `node_complete` | Node execution finished | `{node_id, output, duration_ms}` |
| `node_error` | Node execution failed | `{node_id, error, stack}` |
| `workflow_complete` | Workflow finished | `{result, total_duration_ms}` |
| `workflow_error` | Workflow failed | `{error, failed_node}` |

#### Collaboration Events

| Event | Description | Payload |
|-------|-------------|---------|
| `participant_joined` | User joined session | `{user_id, name, color}` |
| `participant_left` | User left session | `{user_id}` |
| `cursor_update` | Remote cursor moved | `{user_id, x, y}` |
| `node_locked` | Node locked by user | `{node_id, locked_by}` |
| `node_unlocked` | Node lock released | `{node_id}` |
| `remote_operation` | Workflow change | `{op, data, user_id}` |

## Error Handling

### Error Response Format

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid workflow configuration",
    "details": [
      {
        "field": "nodes[0].config.timeout",
        "message": "Must be a positive number"
      }
    ]
  }
}
```

### Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `AUTHENTICATION_REQUIRED` | 401 | Missing or invalid credentials |
| `PERMISSION_DENIED` | 403 | Insufficient permissions |
| `NOT_FOUND` | 404 | Resource not found |
| `VALIDATION_ERROR` | 400 | Invalid request data |
| `WORKFLOW_INVALID` | 400 | Workflow validation failed |
| `EXECUTION_FAILED` | 500 | Workflow execution error |
| `TIMEOUT` | 504 | Execution timeout |
| `RATE_LIMITED` | 429 | Too many requests |

### Rate Limiting

Rate limit headers:
```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1702641600
```

When rate limited:
```json
{
  "error": {
    "code": "RATE_LIMITED",
    "message": "Rate limit exceeded",
    "retry_after": 60
  }
}
```

## SDK Examples

### Python

```python
from hololoom import WorkflowClient

client = WorkflowClient(
    base_url="http://localhost:8001",
    api_key="your-api-key"
)

# Execute workflow
result = await client.execute(
    workflow_id="wf-abc123",
    input_data={"query": "What is Thompson Sampling?"}
)

print(result.output)

# Stream execution
async for event in client.execute_stream(workflow_id, input_data):
    if event.type == "node_complete":
        print(f"Node {event.node_id} complete")
```

### JavaScript/TypeScript

```typescript
import { WorkflowClient } from '@hololoom/workflow-sdk';

const client = new WorkflowClient({
  baseUrl: 'http://localhost:8001',
  apiKey: 'your-api-key'
});

// Execute workflow
const result = await client.execute('wf-abc123', {
  query: 'What is Thompson Sampling?'
});

console.log(result.output);

// Real-time execution
const stream = client.executeStream('wf-abc123', inputData);

stream.on('node_complete', (event) => {
  console.log(`Node ${event.nodeId} complete`);
});

stream.on('workflow_complete', (result) => {
  console.log('Done!', result);
});
```

### cURL

```bash
# Execute workflow
curl -X POST http://localhost:8001/api/workflow/execute \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "workflow_id": "wf-abc123",
    "input_data": {"query": "What is Thompson Sampling?"}
  }'

# Export as Python
curl -X POST http://localhost:8001/api/workflow/export \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "workflow_id": "wf-abc123",
    "format": "python"
  }'
```

## Webhook Integration

### Configure Webhooks

```http
POST /api/webhooks
Content-Type: application/json

{
  "url": "https://your-server.com/webhook",
  "events": ["workflow_complete", "workflow_error"],
  "secret": "your-webhook-secret"
}
```

### Webhook Payload

```json
{
  "event": "workflow_complete",
  "timestamp": "2025-12-15T10:30:00Z",
  "data": {
    "workflow_id": "wf-abc123",
    "job_id": "job-456",
    "result": {...}
  }
}
```

### Signature Verification

```python
import hmac
import hashlib

def verify_webhook(payload, signature, secret):
    expected = hmac.new(
        secret.encode(),
        payload.encode(),
        hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(f"sha256={expected}", signature)
```

## OpenAPI Specification

Full OpenAPI 3.0 specification available at:
```
GET /api/openapi.json
GET /api/openapi.yaml
```

Interactive documentation:
```
GET /api/docs      # Swagger UI
GET /api/redoc     # ReDoc
```

---

← [Performance Optimization](performance.md) | [Tutorials: RAG Pipeline](../tutorials/rag-pipeline.md) →
