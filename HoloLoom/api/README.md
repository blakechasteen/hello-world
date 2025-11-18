# HoloLoom Production API

**Version**: 1.0.0
**Status**: Production Ready
**Date**: 2025-11-18 (Week 8B: Docker Deployment & API Endpoints)

Production-ready REST API for HoloLoom's advanced memory systems from Weeks 5-7.

## Table of Contents

1. [Quick Start](#quick-start)
2. [API Overview](#api-overview)
3. [Authentication](#authentication)
4. [Endpoints Reference](#endpoints-reference)
5. [Deployment](#deployment)
6. [Monitoring](#monitoring)
7. [Development](#development)
8. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Prerequisites

- Docker 20.10+ and Docker Compose 2.0+
- 4GB RAM minimum (8GB recommended)
- 10GB disk space

### 1. Deploy with Docker Compose

```bash
# Clone repository
cd /path/to/hololoom

# Deploy all services
./scripts/deploy.sh

# Access API
curl http://localhost:8000/health
```

### 2. Generate API Key

```bash
# Set your API key (production)
export HOLOLOOM_API_KEY=$(openssl rand -hex 32)
echo "API Key: $HOLOLOOM_API_KEY"
```

### 3. Test API

```bash
# Health check
curl http://localhost:8000/health

# Store memory (requires authentication)
curl -X POST http://localhost:8000/api/v1/experience \
  -H "X-API-Key: $HOLOLOOM_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Thompson Sampling balances exploration and exploitation",
    "scope": "SESSION"
  }'

# Recall memories
curl -X POST http://localhost:8000/api/v1/recall \
  -H "X-API-Key: $HOLOLOOM_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is Thompson Sampling?",
    "max_results": 10
  }'
```

---

## API Overview

### Architecture

```
┌─────────────────────────────────────────┐
│         Client Applications              │
│   (Web, Mobile, CLI, VS Code, etc)      │
└────────────────┬────────────────────────┘
                 │ HTTP/REST
                 │
┌────────────────▼────────────────────────┐
│       HoloLoom API Server (FastAPI)     │
│                                          │
│  ┌────────────────────────────────────┐ │
│  │  Middleware                        │ │
│  │  - Authentication (API Keys)       │ │
│  │  - Rate Limiting (60 req/min)      │ │
│  │  - CORS                            │ │
│  │  - Request Logging                 │ │
│  └────────────────────────────────────┘ │
│                                          │
│  ┌────────────────────────────────────┐ │
│  │  Memory Systems (Week 5-7)         │ │
│  │  - Consolidation                   │ │
│  │  - Semantic Transition             │ │
│  │  - Temporal Evolution              │ │
│  │  - Curiosity Engine                │ │
│  │  - Graph Reasoning                 │ │
│  └────────────────────────────────────┘ │
└────────────────┬────────────────────────┘
                 │
    ┌────────────┼────────────┐
    │            │            │
┌───▼───┐   ┌───▼───┐   ┌───▼───┐
│ Neo4j │   │Qdrant │   │ Redis │
│ Graph │   │Vector │   │Cache  │
└───────┘   └───────┘   └───────┘
```

### Services

| Service | Port | Purpose |
|---------|------|---------|
| **HoloLoom API** | 8000 | Main API server |
| **Neo4j** | 7474, 7687 | Graph database |
| **Qdrant** | 6333, 6334 | Vector database |
| **Redis** | 6379 | Caching & rate limiting |
| **Prometheus** | 9090 | Metrics collection |
| **Grafana** | 3000 | Visualization |

### Performance

| Metric | Value |
|--------|-------|
| **Average latency** | <200ms |
| **Max throughput** | 60 req/min (configurable) |
| **Consolidation** | <5s per cycle |
| **Pattern detection** | <500ms for 100 memories |
| **Temporal query** | <100ms |
| **Multi-hop query** | <200ms (3 hops) |

---

## Authentication

### API Key Authentication

All endpoints (except `/health` and `/metrics`) require authentication via API key.

**Include API key in request header**:
```
X-API-Key: your-api-key-here
```

### Generate API Key

```bash
# Random 256-bit key
openssl rand -hex 32

# Or use environment variable
export HOLOLOOM_API_KEY="your-key-here"
```

### Configure API Key

**Option 1: Environment variable**
```bash
export HOLOLOOM_API_KEY="your-key"
docker-compose up -d
```

**Option 2: .env file**
```bash
# Create .env.production
echo "HOLOLOOM_API_KEY=your-key" > .env.production

# Deploy with environment
./scripts/deploy.sh production
```

**Option 3: Docker Compose**
```yaml
services:
  hololoom-api:
    environment:
      - HOLOLOOM_API_KEY=your-key
```

### Disable Authentication (Development Only)

**WARNING: Do not use in production!**

```bash
# Unset API key
unset HOLOLOOM_API_KEY

# API will accept all requests
```

---

## Endpoints Reference

### Health & Stats

#### GET /health

Health check endpoint (no authentication required).

```bash
curl http://localhost:8000/health
```

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2025-11-18T10:00:00Z",
  "systems": {
    "consolidation": "ok",
    "semantic_transition": "ok",
    "temporal_evolution": "ok",
    "curiosity": "ok",
    "graph_reasoning": "ok"
  },
  "version": "1.0.0"
}
```

#### GET /api/v1/stats

Get complete system statistics.

```bash
curl -H "X-API-Key: $HOLOLOOM_API_KEY" \
  http://localhost:8000/api/v1/stats
```

**Response**:
```json
{
  "consolidation": {
    "total_consolidations": 42,
    "total_episodes_processed": 1500,
    "total_facts_created": 250,
    "average_consolidation_time_ms": 4500.0,
    "last_consolidation": "2025-11-18T09:30:00Z"
  },
  "semantic": {
    "total_patterns_detected": 85,
    "total_concepts_created": 32,
    "average_pattern_frequency": 3.5,
    "episodic_to_semantic_ratio": 0.21,
    "last_transition": "2025-11-18T09:45:00Z"
  },
  "curiosity": {
    "total_suggestions_generated": 120,
    "suggestions_by_type": {
      "gap": 45,
      "contradiction": 12,
      "related_concept": 38,
      "trending": 15,
      "deep_dive": 10
    },
    "average_importance": 0.72,
    "suggestions_followed": 35,
    "follow_rate": 0.29
  }
}
```

---

### Memory Operations

#### POST /api/v1/experience

Store new episodic memory.

```bash
curl -X POST http://localhost:8000/api/v1/experience \
  -H "X-API-Key: $HOLOLOOM_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Thompson Sampling balances exploration and exploitation",
    "scope": "SESSION",
    "metadata": {"source": "user_query"}
  }'
```

**Request Body**:
```typescript
{
  content: string;        // Memory content (required)
  scope: string;          // SESSION, AGENT, USER, GLOBAL (default: SESSION)
  metadata?: object;      // Additional metadata (optional)
}
```

**Response**:
```json
{
  "status": "success",
  "memory_id": "mem_abc123",
  "scope": "SESSION",
  "timestamp": "2025-11-18T10:00:00Z"
}
```

#### POST /api/v1/recall

Retrieve relevant memories.

```bash
curl -X POST http://localhost:8000/api/v1/recall \
  -H "X-API-Key: $HOLOLOOM_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is Thompson Sampling?",
    "max_results": 10,
    "include_metadata": false
  }'
```

**Request Body**:
```typescript
{
  query: string;          // Query text (required)
  max_results: number;    // Max results to return (default: 10, max: 100)
  scope?: string;         // Limit to specific scope (optional)
  include_metadata: bool; // Include full metadata (default: false)
}
```

**Response**:
```json
{
  "query": "What is Thompson Sampling?",
  "memories": [
    {
      "memory_id": "mem_abc123",
      "content": "Thompson Sampling balances exploration and exploitation",
      "scope": "SESSION",
      "relevance_score": 0.95,
      "timestamp": "2025-11-18T10:00:00Z",
      "metadata": null
    }
  ],
  "count": 1,
  "total_available": 1,
  "latency_ms": 125.5
}
```

---

### Consolidation

#### POST /api/v1/consolidation/trigger

Manually trigger consolidation (background job).

```bash
curl -X POST http://localhost:8000/api/v1/consolidation/trigger \
  -H "X-API-Key: $HOLOLOOM_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "strategy": "fact_extraction",
    "max_episodes": 100,
    "prune_episodes": false
  }'
```

**Request Body**:
```typescript
{
  strategy?: string;      // fact_extraction, entity_extraction, summarization, deduplication
  max_episodes: number;   // Max episodes to process (default: 100)
  prune_episodes: bool;   // Delete consolidated episodes (default: false)
}
```

**Response**:
```json
{
  "status": "queued",
  "job_id": "job_xyz789",
  "estimated_duration_ms": 5000.0
}
```

---

### Semantic Transition

#### POST /api/v1/semantic/detect-patterns

Detect episodic patterns.

```bash
curl -X POST http://localhost:8000/api/v1/semantic/detect-patterns \
  -H "X-API-Key: $HOLOLOOM_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "threshold": 3,
    "similarity_threshold": 0.75,
    "window_days": 7,
    "max_patterns": 50
  }'
```

**Response**:
```json
{
  "patterns": [
    {
      "pattern_id": "pattern_abc123",
      "pattern_type": "query_cluster",
      "frequency": 5,
      "similarity_score": 0.82,
      "representative_query": "What is Thompson Sampling?",
      "common_entities": ["thompson_sampling", "exploration"],
      "common_motifs": ["question", "concept_query"],
      "episodic_memory_ids": ["mem_1", "mem_2", "mem_3"],
      "first_seen": "2025-11-11T10:00:00Z",
      "last_seen": "2025-11-18T10:00:00Z"
    }
  ],
  "count": 1,
  "detection_time_ms": 450.0
}
```

#### POST /api/v1/semantic/promote

Promote pattern to semantic concept (background job).

```bash
curl -X POST http://localhost:8000/api/v1/semantic/promote \
  -H "X-API-Key: $HOLOLOOM_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "pattern_id": "pattern_abc123",
    "concept_name": "thompson_sampling_basics"
  }'
```

**Response**:
```json
{
  "status": "queued",
  "pattern_id": "pattern_abc123",
  "job_id": "job_xyz789"
}
```

---

### Temporal Evolution

#### POST /api/v1/temporal/query

Query understanding at specific time (point-in-time snapshot).

```bash
curl -X POST http://localhost:8000/api/v1/temporal/query \
  -H "X-API-Key: $HOLOLOOM_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "concept": "thompson_sampling",
    "timestamp": "2025-11-01T12:00:00Z"
  }'
```

**Response**:
```json
{
  "concept": "thompson_sampling",
  "timestamp": "2025-11-01T12:00:00Z",
  "snapshot": {
    "concept": "thompson_sampling",
    "state": "learning",
    "memory_count": 5,
    "confidence": 0.65,
    "timestamp": "2025-11-01T12:00:00Z",
    "notable_memories": ["mem_1", "mem_2", "mem_3"],
    "metadata": {}
  },
  "found": true
}
```

#### GET /api/v1/temporal/history/{concept}

Get complete evolution history for a concept.

```bash
curl -H "X-API-Key: $HOLOLOOM_API_KEY" \
  http://localhost:8000/api/v1/temporal/history/thompson_sampling
```

**Response**:
```json
{
  "concept": "thompson_sampling",
  "first_learned": "2025-11-01T10:00:00Z",
  "current_state": "familiar",
  "state_transitions": [
    {
      "timestamp": "2025-11-01T10:00:00Z",
      "from_state": "unknown",
      "to_state": "introduced",
      "trigger": "first_exposure"
    }
  ],
  "memory_timeline": [
    {
      "timestamp": "2025-11-01T10:00:00Z",
      "memory_id": "mem_1",
      "confidence": 0.75
    }
  ],
  "milestones": [
    {
      "type": "first_learned",
      "timestamp": "2025-11-01T10:00:00Z",
      "description": "First exposure to thompson_sampling"
    }
  ],
  "total_memories": 12
}
```

#### POST /api/v1/temporal/summary

Get evolution summary for specified period.

```bash
curl -X POST http://localhost:8000/api/v1/temporal/summary \
  -H "X-API-Key: $HOLOLOOM_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"days": 30}'
```

**Response**:
```json
{
  "period_days": 30,
  "total_concepts": 45,
  "concepts_by_state": {
    "introduced": 12,
    "learning": 18,
    "familiar": 10,
    "mastery": 5
  },
  "new_concepts": 15,
  "mastered_concepts": 3,
  "forgotten_concepts": 0,
  "top_learning_areas": ["machine_learning", "reinforcement_learning", "statistics"],
  "learning_velocity": 0.5
}
```

---

### Curiosity Engine

#### POST /api/v1/curiosity/suggest

Get proactive exploration suggestions.

```bash
curl -X POST http://localhost:8000/api/v1/curiosity/suggest \
  -H "X-API-Key: $HOLOLOOM_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "limit": 5,
    "importance_threshold": 0.5,
    "include_serendipity": true
  }'
```

**Response**:
```json
{
  "suggestions": [
    {
      "type": "gap",
      "concept": "multi_armed_bandits",
      "reason": "You've explored Thompson Sampling but not multi-armed bandits",
      "importance": 0.85,
      "suggested_query": "What are multi-armed bandits?",
      "expected_benefit": "Understanding the broader context of exploration strategies",
      "metadata": {},
      "created_at": "2025-11-18T10:00:00Z"
    }
  ],
  "count": 5,
  "generation_time_ms": 125.0
}
```

---

### Graph Reasoning

#### POST /api/v1/graph/multi-hop

Multi-hop graph traversal query.

```bash
curl -X POST http://localhost:8000/api/v1/graph/multi-hop \
  -H "X-API-Key: $HOLOLOOM_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What papers cite Transformers?",
    "max_hops": 2,
    "max_results": 10,
    "edge_types": ["CITES", "RELATED"]
  }'
```

**Response**:
```json
{
  "query": "What papers cite Transformers?",
  "max_hops": 2,
  "results": [
    {
      "memory_id": "mem_abc123",
      "content": "BERT is based on Transformers...",
      "relevance_score": 0.92,
      "hop_distance": 1,
      "reasoning_path": {
        "start_entity": "transformers",
        "end_entity": "bert",
        "path": ["transformers", "bert"],
        "edge_types": ["CITES"],
        "total_weight": 1.0,
        "hop_count": 1
      }
    }
  ],
  "total_paths_explored": 15,
  "query_entities": ["papers", "transformers"],
  "latency_ms": 185.5
}
```

---

## Deployment

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run server directly
PYTHONPATH=. python -m uvicorn HoloLoom.api.server:app --reload --port 8000

# Access API
curl http://localhost:8000/health
```

### Docker Deployment

```bash
# Build image
docker build -t hololoom-api:latest .

# Run container
docker run -d \
  -p 8000:8000 \
  -e HOLOLOOM_API_KEY="your-key" \
  --name hololoom-api \
  hololoom-api:latest

# Check logs
docker logs -f hololoom-api
```

### Docker Compose Deployment (Recommended)

```bash
# Deploy all services
./scripts/deploy.sh

# Or manually
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f hololoom-api

# Stop services
docker-compose down
```

### Production Deployment

1. **Set secure API key**:
   ```bash
   export HOLOLOOM_API_KEY=$(openssl rand -hex 32)
   ```

2. **Configure environment**:
   ```bash
   # Create .env.production
   cat > .env.production <<EOF
   HOLOLOOM_API_KEY=your-secure-key
   CORS_ORIGINS=https://yourdomain.com
   RATE_LIMIT_RPM=100
   RATE_LIMIT_BURST=20
   MEMORY_BACKEND=HYBRID
   EOF
   ```

3. **Deploy**:
   ```bash
   ./scripts/deploy.sh production
   ```

4. **Configure reverse proxy** (Nginx example):
   ```nginx
   server {
       listen 80;
       server_name api.yourdomain.com;

       location / {
           proxy_pass http://localhost:8000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
       }
   }
   ```

5. **Enable SSL** (Let's Encrypt):
   ```bash
   certbot --nginx -d api.yourdomain.com
   ```

---

## Monitoring

### Prometheus Metrics

Access Prometheus at http://localhost:9090

**Key metrics**:
- `hololoom_requests_total` - Total requests by endpoint
- `hololoom_request_duration_seconds` - Request latency
- `hololoom_memories_total` - Total memories by scope
- `hololoom_consolidations_total` - Total consolidations
- `hololoom_semantic_concepts_total` - Semantic concepts
- `hololoom_system_health` - Component health

### Grafana Dashboards

Access Grafana at http://localhost:3000 (admin/admin)

**Pre-configured dashboards**:
1. API Performance (latency, throughput, errors)
2. Memory Systems (consolidations, semantic transitions)
3. System Health (component status, resource usage)

### Health Checks

```bash
# Continuous monitoring
./scripts/health_check.sh --watch

# One-time check
./scripts/health_check.sh
```

### Logs

```bash
# API server logs
docker-compose logs -f hololoom-api

# All service logs
docker-compose logs -f

# Application logs (mounted volume)
tail -f logs/hololoom.log
```

---

## Development

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio pytest-cov

# Run API tests
pytest HoloLoom/api/tests/ -v

# Run with coverage
pytest HoloLoom/api/tests/ --cov=HoloLoom.api --cov-report=html

# View coverage
open htmlcov/index.html
```

### API Documentation

**Swagger UI** (interactive): http://localhost:8000/docs
**ReDoc** (readable): http://localhost:8000/redoc

### Code Quality

```bash
# Format code
black HoloLoom/api/

# Type checking
mypy HoloLoom/api/

# Linting
ruff HoloLoom/api/
```

---

## Troubleshooting

### Common Issues

**1. API not responding**

```bash
# Check container status
docker-compose ps

# Check logs
docker-compose logs hololoom-api

# Restart container
docker-compose restart hololoom-api
```

**2. Authentication errors**

```bash
# Verify API key is set
echo $HOLOLOOM_API_KEY

# Check API key in request
curl -v -H "X-API-Key: $HOLOLOOM_API_KEY" http://localhost:8000/api/v1/stats
```

**3. Rate limiting errors**

```bash
# Increase rate limit (docker-compose.yml)
environment:
  - RATE_LIMIT_RPM=120
  - RATE_LIMIT_BURST=30

# Restart service
docker-compose restart hololoom-api
```

**4. Database connection errors**

```bash
# Check database health
docker-compose ps neo4j qdrant redis

# Restart databases
docker-compose restart neo4j qdrant redis

# Check logs
docker-compose logs neo4j qdrant redis
```

### Performance Tuning

**1. Increase worker count**:
```dockerfile
# Dockerfile
CMD ["uvicorn", "HoloLoom.api.server:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "4"]  # 4 workers
```

**2. Enable production memory backend**:
```bash
# .env.production
MEMORY_BACKEND=HYBRID  # Neo4j + Qdrant
```

**3. Tune database resources**:
```yaml
# docker-compose.yml
neo4j:
  environment:
    - NEO4J_dbms_memory_heap_max__size=4G
    - NEO4J_dbms_memory_pagecache_size=2G
```

---

## Support

**Documentation**: See `HOLOLOOM_MASTER_SCOPE_AND_SEQUENCE.md`
**Issues**: File issues on GitHub
**Community**: Join Discord/Slack channel

---

## License

See LICENSE file in repository root.

---

## Changelog

### v1.0.0 (2025-11-18)

**Week 8B: Docker Deployment & API Endpoints**

- ✅ Complete REST API for Weeks 5-7 memory systems
- ✅ Docker Compose deployment
- ✅ API key authentication
- ✅ Rate limiting (60 req/min)
- ✅ Prometheus metrics
- ✅ Grafana dashboards
- ✅ Health monitoring
- ✅ Deployment scripts
- ✅ Production-ready configuration

**Memory Systems**:
- Consolidation API (Week 5)
- Semantic transition API (Week 6)
- Temporal evolution API (Week 7)
- Curiosity engine API
- Graph reasoning API

**Infrastructure**:
- FastAPI server
- Neo4j + Qdrant + Redis
- Prometheus + Grafana
- Automatic health checks
- Background job processing

---

**Author**: HoloLoom Team
**Date**: 2025-11-18 (Week 8B)
