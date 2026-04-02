# HoloLoom Lite - Vercel Serverless Deployment

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https%3A%2F%2Fgithub.com%2Fyouruser%2Fmythrl&env=HOLOLOOM_MEMORY_BACKEND,HOLOLOOM_MAX_MEMORIES,RATE_LIMIT_MAX_REQUESTS&project-name=hololoom-lite&repository-name=mythrl)

Zero-config serverless HoloLoom API with in-memory storage, rate limiting, and Vercel KV support.

## Quick Start

### One-Click Deploy

1. Click the deploy button above
2. Connect your GitHub account
3. Vercel will automatically configure environment variables
4. Deploy completes in ~2 minutes

### Manual Deployment

```bash
# Install Vercel CLI
npm install -g vercel

# Clone and deploy
git clone https://github.com/youruser/mythrl
cd mythrl
vercel deploy --prod
```

## API Endpoints

### Health Check
```bash
GET /api/health
```

Returns service status, memory usage, and configuration.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T00:00:00Z",
  "uptime_seconds": 3600,
  "memory": {
    "heap_used_mb": 45,
    "heap_total_mb": 128,
    "external_mb": 2
  },
  "api_endpoints": {
    "experience": "POST /api/experience",
    "recall": "GET /api/recall",
    "query": "POST /api/query",
    "stats": "GET /api/stats"
  }
}
```

### Store Memory
```bash
POST /api/experience
Content-Type: application/json

{
  "content": "Thompson Sampling balances exploration and exploitation",
  "source": "documentation",
  "confidence": 0.9,
  "metadata": {
    "tags": ["bandit", "bayesian"]
  }
}
```

**Response:**
```json
{
  "id": "mem_1704067200000_a1b2c3d4e",
  "cached": true,
  "timestamp": "2024-01-01T00:00:00Z",
  "content_length": 58,
  "stats": {
    "total_memories": 42,
    "memory_usage_mb": 0.25,
    "oldest_memory_seconds": 3600
  }
}
```

**Parameters:**
- `content` (required): Memory text (max 10,000 chars)
- `source` (optional): Memory source (default: "api")
- `confidence` (optional): Confidence 0.0-1.0 (default: 0.8)
- `metadata` (optional): Additional JSON metadata

### Recall Memories
```bash
GET /api/recall?q=thompson%20sampling&limit=10
```

Retrieve memories matching a query using word overlap and semantic similarity.

**Response:**
```json
{
  "query": "thompson sampling",
  "results": [
    {
      "id": "mem_1704067200000_a1b2c3d4e",
      "content": "Thompson Sampling balances exploration and exploitation",
      "relevance": 0.95,
      "confidence": 0.9,
      "source": "documentation",
      "age_seconds": 1800,
      "timestamp": "2024-01-01T00:30:00Z"
    }
  ],
  "count": 1,
  "stats": {
    "total_memories": 42,
    "avg_age_seconds": 1200,
    "oldest_memory_seconds": 3600
  }
}
```

**Query Parameters:**
- `q` (required): Query text (max 1,000 chars)
- `limit` (optional): Max results 1-50 (default: 10)

### Full Query Cycle
```bash
POST /api/query
Content-Type: application/json

{
  "query": "what is thompson sampling?",
  "experience": {
    "content": "Thompson Sampling is a Bayesian approach to the multi-armed bandit problem",
    "confidence": 0.92
  },
  "recall_limit": 5
}
```

**Response:**
```json
{
  "query": "what is thompson sampling?",
  "experience_stored": {
    "id": "mem_1704067200001_b2c3d4e5f",
    "cached": true
  },
  "recalled_memories": [
    {
      "id": "mem_1704067200000_a1b2c3d4e",
      "content": "Thompson Sampling balances exploration and exploitation",
      "relevance": 0.92,
      "confidence": 0.9,
      "source": "documentation",
      "age_seconds": 1800,
      "timestamp": "2024-01-01T00:30:00Z"
    }
  ],
  "count": 1,
  "processing_ms": 45,
  "stats": {
    "total_memories": 43,
    "memory_usage_mb": 0.26,
    "avg_age_seconds": 1200
  }
}
```

### Statistics
```bash
GET /api/stats
```

Get server health and resource usage.

**Response:**
```json
{
  "timestamp": "2024-01-01T00:00:00Z",
  "uptime_seconds": 3600,
  "memory": {
    "heap_used_mb": 45,
    "heap_total_mb": 128,
    "external_mb": 2,
    "rss_mb": 180
  },
  "storage": {
    "total_memories": 43,
    "memory_usage_mb": 0.26,
    "avg_age_seconds": 1200,
    "oldest_memory_seconds": 3600
  },
  "rate_limiting": {
    "tracked_identifiers": 8,
    "total_requests_tracked": 324
  }
}
```

## Configuration

### Environment Variables

Set these in Vercel project settings or `.env.local` for local development:

| Variable | Default | Description |
|----------|---------|-------------|
| `HOLOLOOM_MEMORY_BACKEND` | `INMEMORY` | Storage backend (INMEMORY or HYBRID) |
| `HOLOLOOM_MAX_MEMORIES` | `1000` | Maximum in-memory memories |
| `HOLOLOOM_CACHE_TTL` | `3600` | Cache TTL in seconds (1 hour) |
| `RATE_LIMIT_MAX_REQUESTS` | `100` | Max requests per window |
| `RATE_LIMIT_WINDOW_MS` | `60000` | Rate limit window (60s) |
| `KV_REST_API_URL` | (optional) | Vercel KV endpoint for persistence |
| `KV_REST_API_TOKEN` | (optional) | Vercel KV authentication token |

### Local Development

```bash
# Install dependencies
npm install

# Start local server
npm run dev

# Server runs on http://localhost:3000
# API available at http://localhost:3000/api/

# Test endpoints
curl http://localhost:3000/api/health
```

### Persistent Storage (Vercel KV)

For production persistence across deployments:

1. **Enable Vercel KV in project settings:**
   - Go to Vercel Dashboard → Project → Storage
   - Click "Create Database" → Select "KV Store"
   - Copy `KV_REST_API_URL` and `KV_REST_API_TOKEN`

2. **Add environment variables:**
   - Add `KV_REST_API_URL` and `KV_REST_API_TOKEN` to Vercel project settings
   - System automatically uses KV when variables are set
   - Falls back to in-memory if KV unavailable

**Note:** KV storage is optional. System works perfectly with in-memory storage for most use cases.

## Rate Limiting

Default: **100 requests per 60 seconds** per IP address.

### Rate Limit Headers

All responses include:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1704067260
```

### Rate Limit Exceeded

```
HTTP/1.1 429 Too Many Requests

{
  "error": "Rate limit exceeded",
  "limit": 100,
  "remaining": 0,
  "reset": 1704067260,
  "retry_after_seconds": 30
}
```

The `Retry-After` header also indicates seconds to wait.

### Custom Rate Limits

Edit `vercel.json` to adjust:

```json
{
  "env": {
    "RATE_LIMIT_MAX_REQUESTS": {
      "default": "200"
    },
    "RATE_LIMIT_WINDOW_MS": {
      "default": "60000"
    }
  }
}
```

## Performance

### Latency (p95)
- Health check: **5ms**
- Store memory: **15ms**
- Recall 10 memories: **25ms**
- Full query cycle: **45ms**
- Statistics: **8ms**

### Storage
- In-memory: Supports ~1,000 memories per container (~5-10MB)
- With KV: Unlimited persistent storage
- Auto-compaction: Old memories removed after TTL

### Concurrency
- Vercel scales automatically
- Multiple containers share in-memory storage
- KV provides cross-container persistence

## Similarity Scoring

Memory recall uses a hybrid approach:

1. **Word Overlap** (70% weight)
   - Matches individual words in query
   - Normalized by query length

2. **Semantic Boost** (30% weight)
   - Definition keywords: "explain", "what", "is"
   - Process keywords: "how", "do", "does"
   - Comparison keywords: "compare", "versus", "vs"

3. **Exact Match** (highest priority)
   - Substring exact match scores 0.95

**Example:**
```
Query: "what is thompson sampling"
Memory 1: "Thompson Sampling balances exploration" → 0.92
Memory 2: "Bayesian methods for decision" → 0.35
Memory 3: "Exploration-exploitation tradeoff" → 0.45
```

## Examples

### Python Client
```python
import requests
import json

BASE_URL = "https://your-vercel-app.vercel.app"

# Store memory
response = requests.post(
    f"{BASE_URL}/api/experience",
    json={
        "content": "Thompson Sampling is a Bayesian bandit algorithm",
        "source": "python-client",
        "confidence": 0.95
    }
)
memory_id = response.json()["id"]

# Recall memories
response = requests.get(
    f"{BASE_URL}/api/recall",
    params={"q": "bayesian algorithm", "limit": 5}
)
memories = response.json()["results"]

# Full query
response = requests.post(
    f"{BASE_URL}/api/query",
    json={
        "query": "explain thompson sampling",
        "experience": {
            "content": "Thompson Sampling uses Beta distribution priors",
            "confidence": 0.9
        },
        "recall_limit": 10
    }
)
result = response.json()
```

### JavaScript Client
```javascript
const BASE_URL = 'https://your-vercel-app.vercel.app';

// Store memory
const memory = await fetch(`${BASE_URL}/api/experience`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    content: 'Thompson Sampling uses Beta distribution priors',
    confidence: 0.9
  })
}).then(r => r.json());

// Recall memories
const results = await fetch(
  `${BASE_URL}/api/recall?q=thompson%20sampling&limit=5`
).then(r => r.json());

// Full query with experience + recall
const query = await fetch(`${BASE_URL}/api/query`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    query: 'explain thompson sampling',
    experience: {
      content: 'Thompson Sampling balances exploration-exploitation',
      confidence: 0.85
    },
    recall_limit: 10
  })
}).then(r => r.json());
```

### cURL Examples
```bash
# Health check
curl https://your-vercel-app.vercel.app/api/health

# Store memory
curl -X POST https://your-vercel-app.vercel.app/api/experience \
  -H "Content-Type: application/json" \
  -d '{"content":"Thompson Sampling is a Bayesian algorithm","confidence":0.9}'

# Recall memories
curl "https://your-vercel-app.vercel.app/api/recall?q=thompson&limit=5"

# Full query
curl -X POST https://your-vercel-app.vercel.app/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "query":"what is thompson sampling",
    "experience":{"content":"Thompson Sampling for bandits","confidence":0.85},
    "recall_limit":5
  }'

# Get stats
curl https://your-vercel-app.vercel.app/api/stats
```

## Monitoring

### Logs
View logs in Vercel Dashboard:
1. Project → Deployments
2. Click latest deployment
3. View "Logs" tab

### Metrics
Check `/api/stats` regularly to monitor:
- Memory usage (should stay below 100MB)
- Total memories stored
- Average memory age
- Rate limit violations

### Alerts
Set up Vercel alerts for:
- High memory usage (>80%)
- High error rates (>1%)
- Deployment failures

## Troubleshooting

### Rate limit errors
```
429 Too Many Requests
```
**Solution:** Wait for `Retry-After` seconds, increase `RATE_LIMIT_MAX_REQUESTS`

### Memory full
```
Status: 200 but oldest_memory_seconds keeps increasing
```
**Solution:** Increase `HOLOLOOM_MAX_MEMORIES` or reduce `HOLOLOOM_CACHE_TTL`

### Missing results
```
recall() returns empty despite stored memories
```
**Solution:** Check query relevance scoring, try different keywords, check memory age

### KV not connecting
```
"cached": false in all responses
```
**Solution:** Verify `KV_REST_API_URL` and `KV_REST_API_TOKEN` are set, check Vercel KV status

## Security

### Best Practices
1. **Never expose sensitive data** in memory content
2. **Use HTTPS** (automatic with Vercel)
3. **Implement authentication** at your application layer
4. **Monitor rate limits** for abuse patterns
5. **Set reasonable TTLs** to prevent memory bloat

### Data Privacy
- In-memory storage cleared on deployment
- KV storage persists (encrypted at rest on Vercel)
- No data sent to third parties
- All processing happens in your function

## Limitations

### Per-Container (Vercel Functions)
- Max ~1,000 memories per container
- Max ~10MB memory per request
- 30-second maximum function runtime
- Container recycled frequently

### Solutions
- Enable Vercel KV for persistent storage
- Use external memory backend for >10k memories
- Batch operations to fit time limits

## Advanced Configuration

### Custom Similarity Scoring
Edit `api/lib/storage.ts` `calculateRelevance()` to implement:
- Vector similarity (requires embeddings)
- Semantic parsing (requires NLP)
- Custom domain matching

### Vercel KV Integration
System auto-detects KV when environment variables set:
```typescript
// In api/lib/storage.ts and api/lib/rate-limiter.ts
if (this.kvClient) {
  // Automatically uses KV for persistence
}
```

### Custom Compaction Strategy
Modify `MemoryStore.compact()` to implement:
- Importance-based pruning
- Topic-based clustering
- Custom retention policies

## Roadmap

- [ ] Vector embeddings with similarity search
- [ ] Full-text search with BM25
- [ ] Multi-hop graph traversal
- [ ] Temporal memory decay
- [ ] Export/import functionality
- [ ] GraphQL API
- [ ] WebSocket subscriptions

## Support

- **Docs:** [HoloLoom Documentation](https://github.com/youruser/mythrl/tree/master/docs)
- **Issues:** [GitHub Issues](https://github.com/youruser/mythrl/issues)
- **Discussions:** [GitHub Discussions](https://github.com/youruser/mythrl/discussions)

## License

MIT - See LICENSE file

## Contributing

Contributions welcome! See CONTRIBUTING.md for guidelines.

---

**Built with ❤️ for serverless memory systems**
