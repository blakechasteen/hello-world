# Promptly API Complete Reference

**Version:** 1.0.0
**Base URL:** `http://localhost:8000` (development) | `https://api.promptly.dev` (production)
**API Prefix:** `/api/v1`

---

## Table of Contents

1. [Introduction](#introduction)
2. [Authentication](#authentication)
3. [Rate Limiting](#rate-limiting)
4. [Error Handling](#error-handling)
5. [API Endpoints](#api-endpoints)
   - [System](#system-endpoints)
   - [Authentication](#authentication-endpoints)
   - [Prompts](#prompts-endpoints)
   - [Branches](#branches-endpoints)
   - [History](#history-endpoints)
   - [Evaluations](#evaluations-endpoints)
   - [Chains](#chains-endpoints)
   - [Plugins](#plugins-endpoints)
   - [WebSocket](#websocket-endpoints)
6. [SDK Examples](#sdk-examples)
7. [Request/Response Examples](#request-response-examples)

---

## Introduction

The Promptly REST API provides programmatic access to all Promptly features. It follows RESTful principles and returns JSON responses.

### Key Features
- ✅ **RESTful design** - Standard HTTP methods
- ✅ **JSON format** - All requests and responses
- ✅ **OpenAPI/Swagger** - Interactive documentation at `/docs`
- ✅ **Authentication** - API key based
- ✅ **Rate limiting** - Protection against abuse
- ✅ **CORS support** - Cross-origin requests
- ✅ **WebSocket support** - Real-time updates
- ✅ **Versioned API** - Stable `/api/v1` prefix

### Base URLs

| Environment | Base URL |
|-------------|----------|
| Development | `http://localhost:8000` |
| Staging | `https://staging.promptly.dev` |
| Production | `https://api.promptly.dev` |

---

## Authentication

### API Key Authentication

All API requests require authentication using an API key sent in the `X-API-Key` header.

**Request Header:**
```
X-API-Key: your-api-key-here
```

**Example:**
```bash
curl -X GET "http://localhost:8000/api/v1/prompts" \
  -H "X-API-Key: dev-test-key-1234567890"
```

**Python SDK:**
```python
from promptly.sdk import PromptlyClient

client = PromptlyClient(
    base_url="http://localhost:8000",
    api_key="dev-test-key-1234567890"
)
```

### Getting an API Key

**Development Mode:**
```bash
# Start API server - prints default dev key
uvicorn promptly.api.main:app

# Output: 🔑 Development API Key: dev-test-key-1234567890
```

**Production:**
```bash
# Create API key via CLI
promptly api-key create --name "Production Key" --scopes read,write

# Or via API
curl -X POST "http://localhost:8000/api/v1/auth/keys" \
  -H "X-API-Key: admin-key" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Production Key",
    "scopes": ["read", "write"],
    "expires_in_days": 90
  }'
```

### JWT Authentication (Optional)

**Login:**
```bash
curl -X POST "http://localhost:8000/api/v1/auth/login" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "alice",
    "password": "secure-password"
  }'

# Response:
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

**Use Token:**
```bash
curl -X GET "http://localhost:8000/api/v1/prompts" \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
```

---

## Rate Limiting

### Limits

| Plan | Requests/Minute | Burst |
|------|-----------------|-------|
| Development | Unlimited | Unlimited |
| Free Tier | 60 | 10 |
| Pro | 600 | 100 |
| Enterprise | Custom | Custom |

### Rate Limit Headers

Every response includes rate limit information:

```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 59
X-RateLimit-Reset: 1640995200
```

### Rate Limit Exceeded

**Status Code:** `429 Too Many Requests`

**Response:**
```json
{
  "message": "Rate limit exceeded",
  "detail": "Too many requests. Try again in 60 seconds.",
  "error_code": "RATE_LIMIT_EXCEEDED",
  "retry_after": 60
}
```

**Retry Header:**
```
Retry-After: 60
```

---

## Error Handling

### Standard Error Response

```json
{
  "message": "Error message",
  "detail": "Detailed error information",
  "error_code": "ERROR_CODE",
  "timestamp": "2025-01-15T10:30:00Z",
  "request_id": "abc-123-def"
}
```

### HTTP Status Codes

| Code | Description | When Used |
|------|-------------|-----------|
| `200` | OK | Successful request |
| `201` | Created | Resource created |
| `204` | No Content | Successful deletion |
| `400` | Bad Request | Invalid request data |
| `401` | Unauthorized | Missing/invalid authentication |
| `403` | Forbidden | Insufficient permissions |
| `404` | Not Found | Resource doesn't exist |
| `409` | Conflict | Resource conflict (e.g., duplicate) |
| `422` | Unprocessable Entity | Validation error |
| `429` | Too Many Requests | Rate limit exceeded |
| `500` | Internal Server Error | Server error |
| `503` | Service Unavailable | Server maintenance |

### Error Codes

| Error Code | Description |
|------------|-------------|
| `VALIDATION_ERROR` | Request validation failed |
| `AUTHENTICATION_ERROR` | Authentication failed |
| `AUTHORIZATION_ERROR` | Insufficient permissions |
| `NOT_FOUND` | Resource not found |
| `ALREADY_EXISTS` | Resource already exists |
| `RATE_LIMIT_EXCEEDED` | Too many requests |
| `INTERNAL_ERROR` | Server error |

---

## API Endpoints

### System Endpoints

#### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-01-15T10:30:00Z",
  "version": "1.0.0",
  "uptime_seconds": 3600.5
}
```

**Example:**
```bash
curl http://localhost:8000/health
```

```python
response = client.health()
```

---

#### GET /

API information.

**Response:**
```json
{
  "name": "Promptly API",
  "version": "1.0.0",
  "docs": "/docs",
  "health": "/health"
}
```

---

### Authentication Endpoints

#### POST /api/v1/auth/login

Authenticate and get JWT token.

**Request:**
```json
{
  "username": "alice",
  "password": "secure-password"
}
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

---

#### POST /api/v1/auth/keys

Create new API key (requires admin).

**Request:**
```json
{
  "name": "Production Key",
  "scopes": ["read", "write"],
  "expires_in_days": 90
}
```

**Response:**
```json
{
  "key_id": "key-123",
  "api_key": "pk_live_abc123def456",
  "name": "Production Key",
  "scopes": ["read", "write"],
  "expires_at": "2025-04-15T10:30:00Z",
  "created_at": "2025-01-15T10:30:00Z"
}
```

---

#### GET /api/v1/auth/keys

List API keys.

**Response:**
```json
{
  "keys": [
    {
      "key_id": "key-123",
      "name": "Production Key",
      "scopes": ["read", "write"],
      "created_at": "2025-01-15T10:30:00Z",
      "last_used": "2025-01-15T12:00:00Z"
    }
  ]
}
```

---

#### DELETE /api/v1/auth/keys/{key_id}

Revoke API key.

**Response:** `204 No Content`

---

### Prompts Endpoints

#### POST /api/v1/prompts

Create or update a prompt.

**Request:**
```json
{
  "name": "summarizer",
  "content": "Summarize the following text:\n{text}",
  "metadata": {
    "author": "alice",
    "tags": ["summarization", "production"],
    "version": "1.0"
  }
}
```

**Response:**
```json
{
  "name": "summarizer",
  "content": "Summarize the following text:\n{text}",
  "branch": "main",
  "version": 1,
  "commit_hash": "abc123def456",
  "created_at": "2025-01-15T10:30:00Z",
  "metadata": {
    "author": "alice",
    "tags": ["summarization", "production"],
    "version": "1.0"
  }
}
```

**Example:**
```bash
curl -X POST "http://localhost:8000/api/v1/prompts" \
  -H "X-API-Key: dev-key" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "summarizer",
    "content": "Summarize: {text}",
    "metadata": {"author": "alice"}
  }'
```

```python
response = client.create_prompt(
    name="summarizer",
    content="Summarize: {text}",
    metadata={"author": "alice"}
)
```

---

#### GET /api/v1/prompts

List all prompts.

**Query Parameters:**
- `branch` (string, optional) - Filter by branch (default: current branch)
- `limit` (integer, optional) - Max results (default: 100)
- `offset` (integer, optional) - Pagination offset (default: 0)
- `tags` (string[], optional) - Filter by tags

**Response:**
```json
{
  "prompts": [
    {
      "name": "summarizer",
      "version": 2,
      "commit_hash": "abc123def456",
      "created_at": "2025-01-15T10:30:00Z",
      "metadata": {}
    }
  ],
  "total": 1,
  "limit": 100,
  "offset": 0
}
```

**Example:**
```bash
curl "http://localhost:8000/api/v1/prompts?branch=main&limit=10" \
  -H "X-API-Key: dev-key"
```

```python
prompts = client.list_prompts(branch="main")
```

---

#### GET /api/v1/prompts/{name}

Get a specific prompt.

**Query Parameters:**
- `version` (integer, optional) - Specific version
- `commit_hash` (string, optional) - Specific commit
- `branch` (string, optional) - Branch (default: current)

**Response:**
```json
{
  "name": "summarizer",
  "content": "Summarize: {text}",
  "branch": "main",
  "version": 2,
  "commit_hash": "abc123def456",
  "created_at": "2025-01-15T10:30:00Z",
  "metadata": {
    "author": "alice",
    "tags": ["summarization"]
  }
}
```

**Example:**
```bash
# Get latest version
curl "http://localhost:8000/api/v1/prompts/summarizer" \
  -H "X-API-Key: dev-key"

# Get specific version
curl "http://localhost:8000/api/v1/prompts/summarizer?version=1" \
  -H "X-API-Key: dev-key"
```

```python
# Latest version
prompt = client.get_prompt("summarizer")

# Specific version
prompt = client.get_prompt("summarizer", version=1)
```

---

#### POST /api/v1/prompts/search

Search prompts.

**Request:**
```json
{
  "query": "summarization",
  "tags": ["production"],
  "branch": "main",
  "limit": 20
}
```

**Response:**
```json
{
  "results": [
    {
      "name": "summarizer",
      "content": "Summarize: {text}",
      "version": 2,
      "score": 0.95,
      "metadata": {}
    }
  ],
  "total": 1,
  "query": "summarization"
}
```

**Example:**
```python
results = client.search_prompts(
    query="summarization",
    tags=["production"],
    limit=20
)
```

---

#### GET /api/v1/prompts/{name}/diff

Get diff between two versions.

**Query Parameters:**
- `version_old` (integer, required) - Old version
- `version_new` (integer, required) - New version
- `level` (string, optional) - `char`, `word`, `line`, `semantic` (default: `line`)
- `format` (string, optional) - `unified`, `json` (default: `json`)

**Response:**
```json
{
  "prompt_name": "summarizer",
  "version_old": 1,
  "version_new": 2,
  "diff_level": "line",
  "additions": 1,
  "deletions": 1,
  "changes": 0,
  "similarity": 0.85,
  "chunks": [
    {
      "type": "delete",
      "old_text": "Summarize: {text}",
      "old_start": 0,
      "old_end": 1
    },
    {
      "type": "insert",
      "new_text": "Provide a summary of: {text}",
      "new_start": 0,
      "new_end": 1
    }
  ]
}
```

**Example:**
```python
diff = client.get_prompt_diff("summarizer", version_old=1, version_new=2)
```

---

#### DELETE /api/v1/prompts/{name}

Delete a prompt.

**Query Parameters:**
- `version` (integer, optional) - Delete specific version (default: all versions)
- `branch` (string, optional) - Branch (default: current)

**Response:** `204 No Content`

**Example:**
```bash
curl -X DELETE "http://localhost:8000/api/v1/prompts/summarizer" \
  -H "X-API-Key: dev-key"
```

---

### Branches Endpoints

#### POST /api/v1/branches

Create a new branch.

**Request:**
```json
{
  "name": "experiment",
  "from_branch": "main"
}
```

**Response:**
```json
{
  "name": "experiment",
  "head_commit": "abc123def456",
  "created_at": "2025-01-15T10:30:00Z",
  "from_branch": "main"
}
```

**Example:**
```python
response = client.create_branch("experiment", from_branch="main")
```

---

#### GET /api/v1/branches

List all branches.

**Response:**
```json
{
  "current_branch": "main",
  "branches": [
    {
      "name": "main",
      "head_commit": "abc123def456",
      "created_at": "2025-01-10T10:00:00Z",
      "prompt_count": 5
    },
    {
      "name": "experiment",
      "head_commit": "def456abc123",
      "created_at": "2025-01-15T10:30:00Z",
      "prompt_count": 5
    }
  ]
}
```

**Example:**
```python
branches = client.list_branches()
```

---

#### GET /api/v1/branches/{name}

Get branch details.

**Response:**
```json
{
  "name": "experiment",
  "head_commit": "def456abc123",
  "created_at": "2025-01-15T10:30:00Z",
  "prompt_count": 5,
  "latest_prompts": [
    {
      "name": "summarizer",
      "version": 2,
      "commit_hash": "abc123"
    }
  ]
}
```

---

#### POST /api/v1/branches/checkout

Checkout a branch.

**Request:**
```json
{
  "branch_name": "experiment"
}
```

**Response:**
```json
{
  "previous_branch": "main",
  "current_branch": "experiment"
}
```

**Example:**
```python
response = client.checkout_branch("experiment")
```

---

#### DELETE /api/v1/branches/{name}

Delete a branch.

**Query Parameters:**
- `force` (boolean, optional) - Force delete even with changes (default: false)

**Response:** `204 No Content`

**Example:**
```bash
curl -X DELETE "http://localhost:8000/api/v1/branches/experiment?force=true" \
  -H "X-API-Key: dev-key"
```

---

### History Endpoints

#### GET /api/v1/history/log

Get commit history.

**Query Parameters:**
- `name` (string, optional) - Filter by prompt name
- `branch` (string, optional) - Branch (default: current)
- `limit` (integer, optional) - Max results (default: 10)
- `since` (string, optional) - Since timestamp (ISO 8601)
- `until` (string, optional) - Until timestamp (ISO 8601)

**Response:**
```json
{
  "commits": [
    {
      "commit_hash": "abc123def456",
      "prompt_name": "summarizer",
      "version": 2,
      "branch": "main",
      "created_at": "2025-01-15T10:30:00Z",
      "metadata": {}
    }
  ],
  "total": 1
}
```

**Example:**
```python
log = client.get_log(name="summarizer", limit=10)
```

---

#### GET /api/v1/history/blame/{name}

Get blame information (who changed what).

**Query Parameters:**
- `version` (integer, optional) - Specific version (default: latest)

**Response:**
```json
{
  "prompt_name": "summarizer",
  "version": 2,
  "lines": [
    {
      "line_number": 1,
      "content": "Summarize: {text}",
      "commit_hash": "abc123",
      "author": "alice",
      "created_at": "2025-01-15T10:30:00Z"
    }
  ]
}
```

---

### Evaluations Endpoints

#### POST /api/v1/evaluations

Run evaluation on a prompt.

**Request:**
```json
{
  "prompt_name": "summarizer",
  "test_cases": [
    {
      "inputs": {"text": "Long article..."},
      "expected": "Brief summary...",
      "metadata": {"test_id": "test1"}
    }
  ],
  "evaluator": "semantic",
  "model_endpoint": "https://api.openai.com/v1/completions"
}
```

**Response:**
```json
{
  "evaluation_id": "eval-123",
  "prompt_name": "summarizer",
  "status": "completed",
  "results": [
    {
      "test_case_id": 0,
      "score": 0.85,
      "actual": "Summary output...",
      "expected": "Brief summary...",
      "metrics": {
        "semantic_similarity": 0.85,
        "length_match": 0.9
      }
    }
  ],
  "summary": {
    "total_cases": 1,
    "average_score": 0.85,
    "pass_rate": 1.0
  },
  "created_at": "2025-01-15T10:30:00Z",
  "completed_at": "2025-01-15T10:30:05Z"
}
```

**Example:**
```python
result = client.run_evaluation(
    prompt_name="summarizer",
    test_cases=[
        {
            'inputs': {'text': 'Article...'},
            'expected': 'Summary...'
        }
    ],
    evaluator="semantic"
)
```

---

#### GET /api/v1/evaluations/{evaluation_id}

Get evaluation results.

**Response:**
```json
{
  "evaluation_id": "eval-123",
  "prompt_name": "summarizer",
  "status": "completed",
  "results": [...],
  "summary": {...}
}
```

---

#### GET /api/v1/evaluations

List evaluations.

**Query Parameters:**
- `prompt_name` (string, optional) - Filter by prompt
- `limit` (integer, optional) - Max results (default: 20)
- `status` (string, optional) - Filter by status (`pending`, `running`, `completed`, `failed`)

**Response:**
```json
{
  "evaluations": [
    {
      "evaluation_id": "eval-123",
      "prompt_name": "summarizer",
      "status": "completed",
      "average_score": 0.85,
      "created_at": "2025-01-15T10:30:00Z"
    }
  ],
  "total": 1
}
```

**Example:**
```python
evals = client.list_evaluations(prompt_name="summarizer", limit=20)
```

---

#### POST /api/v1/evaluations/compare

Compare multiple evaluations.

**Request:**
```json
{
  "evaluation_ids": ["eval-123", "eval-456"]
}
```

**Response:**
```json
{
  "comparison": {
    "eval-123": {
      "average_score": 0.85,
      "pass_rate": 1.0
    },
    "eval-456": {
      "average_score": 0.90,
      "pass_rate": 1.0
    }
  },
  "winner": "eval-456",
  "improvement": 0.05
}
```

---

### Chains Endpoints

#### POST /api/v1/chains

Create a new chain.

**Request:**
```json
{
  "name": "entity_pipeline",
  "steps": ["extract", "categorize", "summarize"],
  "description": "Extract entities, categorize, then summarize"
}
```

**Response:**
```json
{
  "name": "entity_pipeline",
  "steps": ["extract", "categorize", "summarize"],
  "description": "Extract entities, categorize, then summarize",
  "created_at": "2025-01-15T10:30:00Z"
}
```

**Example:**
```python
chain = client.create_chain(
    name="entity_pipeline",
    steps=["extract", "categorize", "summarize"],
    description="Entity processing pipeline"
)
```

---

#### GET /api/v1/chains

List all chains.

**Response:**
```json
{
  "chains": [
    {
      "name": "entity_pipeline",
      "steps": ["extract", "categorize", "summarize"],
      "step_count": 3,
      "created_at": "2025-01-15T10:30:00Z"
    }
  ],
  "total": 1
}
```

---

#### GET /api/v1/chains/{name}

Get chain details.

**Response:**
```json
{
  "name": "entity_pipeline",
  "steps": ["extract", "categorize", "summarize"],
  "description": "Entity processing pipeline",
  "created_at": "2025-01-15T10:30:00Z",
  "execution_count": 42,
  "average_duration_ms": 1500
}
```

---

#### POST /api/v1/chains/execute

Execute a chain.

**Request:**
```json
{
  "chain_name": "entity_pipeline",
  "initial_input": {
    "text": "Apple Inc. announced new products."
  },
  "model_endpoint": "https://api.openai.com/v1/completions",
  "stream": false
}
```

**Response:**
```json
{
  "execution_id": "exec-123",
  "chain_name": "entity_pipeline",
  "status": "completed",
  "results": [
    {
      "step": "extract",
      "output": "Apple Inc., products",
      "duration_ms": 500
    },
    {
      "step": "categorize",
      "output": "Company: Apple Inc., Product: products",
      "duration_ms": 400
    },
    {
      "step": "summarize",
      "output": "Tech company product announcement",
      "duration_ms": 600
    }
  ],
  "total_duration_ms": 1500,
  "created_at": "2025-01-15T10:30:00Z",
  "completed_at": "2025-01-15T10:30:01.5Z"
}
```

**Example:**
```python
result = client.execute_chain(
    chain_name="entity_pipeline",
    initial_input={'text': 'Article...'}
)
```

---

#### GET /api/v1/chains/executions/{execution_id}

Get chain execution status.

**Response:**
```json
{
  "execution_id": "exec-123",
  "chain_name": "entity_pipeline",
  "status": "running",
  "current_step": "categorize",
  "progress": 0.66,
  "results": [...]
}
```

---

#### DELETE /api/v1/chains/{name}

Delete a chain.

**Response:** `204 No Content`

---

### Plugins Endpoints

#### GET /api/v1/plugins

List all available plugins.

**Response:**
```json
{
  "evaluators": [
    {
      "name": "keyword",
      "description": "Keyword matching evaluator"
    },
    {
      "name": "semantic",
      "description": "Semantic similarity evaluator"
    }
  ],
  "storage_backends": [
    {
      "name": "sqlite",
      "description": "SQLite storage backend"
    },
    {
      "name": "postgresql",
      "description": "PostgreSQL storage backend"
    }
  ],
  "processors": [
    {
      "name": "conditional",
      "description": "Conditional execution processor"
    },
    {
      "name": "parallel",
      "description": "Parallel execution processor"
    }
  ]
}
```

**Example:**
```python
plugins = client.list_plugins()
```

---

#### GET /api/v1/plugins/{plugin_type}/{plugin_name}

Get plugin details.

**Path Parameters:**
- `plugin_type` - `evaluators`, `storage`, or `processors`
- `plugin_name` - Plugin name

**Response:**
```json
{
  "name": "semantic",
  "type": "evaluator",
  "description": "Semantic similarity evaluator using embeddings",
  "version": "1.0.0",
  "parameters": [
    {
      "name": "model",
      "type": "string",
      "default": "all-MiniLM-L6-v2",
      "description": "Sentence transformer model"
    },
    {
      "name": "threshold",
      "type": "float",
      "default": 0.7,
      "description": "Similarity threshold"
    }
  ],
  "examples": [...]
}
```

---

### WebSocket Endpoints

#### WS /ws/updates

Real-time updates via WebSocket.

**Connection:**
```javascript
// JavaScript
const ws = new WebSocket('ws://localhost:8000/ws/updates?api_key=dev-key');

ws.onmessage = (event) => {
  const message = JSON.parse(event.data);
  console.log('Update:', message);
};
```

**Python:**
```python
from promptly.sdk import AsyncPromptlyClient

async with AsyncPromptlyClient("ws://localhost:8000", api_key="dev-key") as client:
    async for message in client.subscribe_updates():
        print(f"Update: {message}")
```

**Message Format:**
```json
{
  "type": "prompt_updated",
  "prompt_name": "summarizer",
  "version": 3,
  "branch": "main",
  "timestamp": "2025-01-15T10:30:00Z"
}
```

**Event Types:**
- `prompt_updated` - Prompt created/updated
- `prompt_deleted` - Prompt deleted
- `branch_created` - New branch created
- `branch_deleted` - Branch deleted
- `evaluation_complete` - Evaluation finished
- `chain_complete` - Chain execution finished

---

## SDK Examples

### Python Synchronous SDK

```python
from promptly.sdk import PromptlyClient

# Initialize client
client = PromptlyClient(
    base_url="http://localhost:8000",
    api_key="dev-test-key",
    timeout=30,
    max_retries=3
)

# Create prompt
prompt = client.create_prompt(
    name="summarizer",
    content="Summarize: {text}",
    metadata={"author": "alice"}
)

# Get prompt
prompt = client.get_prompt("summarizer")

# List prompts
prompts = client.list_prompts(branch="main")

# Search prompts
results = client.search_prompts(query="summarization", tags=["prod"])

# Create branch
client.create_branch("experiment", from_branch="main")

# Checkout branch
client.checkout_branch("experiment")

# Run evaluation
eval_result = client.run_evaluation(
    prompt_name="summarizer",
    test_cases=[...],
    evaluator="semantic"
)

# Execute chain
chain_result = client.execute_chain(
    chain_name="pipeline",
    initial_input={'text': 'Input...'}
)

# Close client
client.close()

# Or use context manager
with PromptlyClient("http://localhost:8000", api_key="key") as client:
    prompts = client.list_prompts()
```

### Python Asynchronous SDK

```python
from promptly.sdk import AsyncPromptlyClient
import asyncio

async def main():
    async with AsyncPromptlyClient(
        base_url="http://localhost:8000",
        api_key="dev-key"
    ) as client:
        # Concurrent operations
        prompts, branches = await asyncio.gather(
            client.list_prompts(),
            client.list_branches()
        )

        # Create prompt
        prompt = await client.create_prompt(
            name="summarizer",
            content="Summarize: {text}"
        )

        # Get prompt
        prompt = await client.get_prompt("summarizer")

        # Subscribe to updates
        async for message in client.subscribe_updates():
            if message['type'] == 'prompt_updated':
                print(f"Prompt updated: {message['prompt_name']}")
            if message['type'] == 'evaluation_complete':
                print(f"Evaluation done: {message['score']}")

asyncio.run(main())
```

### curl Examples

```bash
# Create prompt
curl -X POST "http://localhost:8000/api/v1/prompts" \
  -H "X-API-Key: dev-key" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "summarizer",
    "content": "Summarize: {text}"
  }'

# Get prompt
curl "http://localhost:8000/api/v1/prompts/summarizer" \
  -H "X-API-Key: dev-key"

# List prompts
curl "http://localhost:8000/api/v1/prompts?branch=main&limit=10" \
  -H "X-API-Key: dev-key"

# Create branch
curl -X POST "http://localhost:8000/api/v1/branches" \
  -H "X-API-Key: dev-key" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "experiment",
    "from_branch": "main"
  }'

# Run evaluation
curl -X POST "http://localhost:8000/api/v1/evaluations" \
  -H "X-API-Key: dev-key" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt_name": "summarizer",
    "test_cases": [
      {"inputs": {"text": "Test"}, "expected": "Summary"}
    ],
    "evaluator": "keyword"
  }'

# Execute chain
curl -X POST "http://localhost:8000/api/v1/chains/execute" \
  -H "X-API-Key: dev-key" \
  -H "Content-Type: application/json" \
  -d '{
    "chain_name": "pipeline",
    "initial_input": {"text": "Input text"}
  }'
```

---

## Request/Response Examples

### Complete Workflow Example

```python
from promptly.sdk import PromptlyClient

client = PromptlyClient("http://localhost:8000", api_key="dev-key")

# 1. Create prompts
client.create_prompt("extract", "Extract entities: {text}")
client.create_prompt("classify", "Classify: {entities}")

# 2. Create experimental branch
client.create_branch("experiment", from_branch="main")
client.checkout_branch("experiment")

# 3. Modify on experimental branch
client.create_prompt("extract", "Extract named entities and relationships: {text}")

# 4. Evaluate experimental version
eval_result = client.run_evaluation(
    prompt_name="extract",
    test_cases=[
        {
            'inputs': {'text': 'Apple Inc. announced...'},
            'expected': 'Apple Inc. (organization)'
        }
    ],
    evaluator="keyword"
)

# 5. If good, merge back to main
if eval_result['summary']['average_score'] > 0.8:
    # Get experimental content
    exp_prompt = client.get_prompt("extract")

    # Switch to main
    client.checkout_branch("main")

    # Apply changes
    client.create_prompt("extract", exp_prompt['content'])

    print("Merged to main!")
else:
    print("Evaluation failed, staying on experiment branch")
```

---

## Additional Resources

- **OpenAPI Documentation**: `http://localhost:8000/docs`
- **ReDoc Documentation**: `http://localhost:8000/redoc`
- **Python SDK Source**: `/Promptly/promptly/sdk/`
- **API Source**: `/Promptly/promptly/api/`

**For issues or questions**, see:
- GitHub Issues
- API Changelog
- Production Handbook

---

**API Version:** 1.0.0
**Last Updated:** January 2025
**Status:** Production Ready
