# HoloLoom API - Quick Start Guide

**5-Minute Setup** | Production-Ready REST API

## 1. Deploy (30 seconds)

```bash
# Generate API key
export HOLOLOOM_API_KEY=$(openssl rand -hex 32)
echo "API Key: $HOLOLOOM_API_KEY"

# Deploy all services
./scripts/deploy.sh
```

## 2. Test (30 seconds)

```bash
# Health check
curl http://localhost:8000/health

# Store memory
curl -X POST http://localhost:8000/api/v1/experience \
  -H "X-API-Key: $HOLOLOOM_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"content": "Thompson Sampling balances exploration", "scope": "SESSION"}'

# Recall memory
curl -X POST http://localhost:8000/api/v1/recall \
  -H "X-API-Key: $HOLOLOOM_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query": "Thompson Sampling", "max_results": 10}'
```

## 3. Explore (1 minute)

**Interactive API Docs**: http://localhost:8000/docs
**Prometheus Metrics**: http://localhost:9090
**Grafana Dashboards**: http://localhost:3000

## 4. Monitor (1 minute)

```bash
# Watch health in real-time
./scripts/health_check.sh --watch

# View logs
docker-compose logs -f hololoom-api
```

## 5. Example Script (2 minutes)

```bash
# Set your API key
export HOLOLOOM_API_KEY="your-key"

# Run comprehensive example
python HoloLoom/api/example_usage.py
```

## Services

| Service | URL | Credentials |
|---------|-----|-------------|
| **API** | http://localhost:8000 | API Key |
| **API Docs** | http://localhost:8000/docs | - |
| **Prometheus** | http://localhost:9090 | - |
| **Grafana** | http://localhost:3000 | admin/admin |
| **Neo4j** | http://localhost:7474 | neo4j/hololoom123 |

## Common Commands

```bash
# Deploy
./scripts/deploy.sh

# Health check
./scripts/health_check.sh

# Backup
./scripts/backup.sh

# Stop services
docker-compose down

# View logs
docker-compose logs -f

# Restart service
docker-compose restart hololoom-api
```

## Endpoints (12 total)

**Memory**:
- POST /api/v1/experience - Store memory
- POST /api/v1/recall - Retrieve memories

**Consolidation (Week 5)**:
- POST /api/v1/consolidation/trigger - Background consolidation

**Semantic (Week 6)**:
- POST /api/v1/semantic/detect-patterns - Find patterns
- POST /api/v1/semantic/promote - Create concept

**Temporal (Week 7)**:
- POST /api/v1/temporal/query - Point-in-time snapshot
- GET /api/v1/temporal/history/{concept} - Evolution history
- POST /api/v1/temporal/summary - Learning summary

**Curiosity**:
- POST /api/v1/curiosity/suggest - Exploration suggestions

**Graph**:
- POST /api/v1/graph/multi-hop - Multi-hop reasoning

**Stats**:
- GET /health - Health check
- GET /api/v1/stats - System statistics

## Documentation

**Complete Guide**: `HoloLoom/api/README.md` (851 lines)
**Implementation**: `WEEK_8B_DEPLOYMENT_SUMMARY.md` (400+ lines)
**Example Code**: `HoloLoom/api/example_usage.py` (334 lines)

## Troubleshooting

**API not responding?**
```bash
docker-compose ps
docker-compose logs hololoom-api
docker-compose restart hololoom-api
```

**Authentication error?**
```bash
echo $HOLOLOOM_API_KEY
curl -v -H "X-API-Key: $HOLOLOOM_API_KEY" http://localhost:8000/health
```

**Database connection error?**
```bash
docker-compose ps neo4j qdrant redis
docker-compose restart neo4j qdrant redis
```

---

**Total Setup Time**: ~5 minutes
**Ready for**: Development, Testing, Production
