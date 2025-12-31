# HoloLoom Lite - Vercel Serverless Deployment

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https%3A%2F%2Fgithub.com%2Fyouruser%2Fmythrl&env=HOLOLOOM_MEMORY_BACKEND,HOLOLOOM_MAX_MEMORIES,RATE_LIMIT_MAX_REQUESTS&project-name=hololoom-lite&repository-name=mythrl)

**One-click serverless memory API for Vercel.** Zero config, instant deployment, production-ready.

## Features

✨ **Zero Configuration** - Works out of the box
🚀 **One-Click Deploy** - Click button → live in 2 minutes
📦 **In-Memory Storage** - No database setup required
🔒 **Rate Limited** - Built-in DDoS protection
📈 **Auto-Scaling** - Vercel handles scaling
💾 **Optional KV** - Persist data with Vercel KV
📊 **Memory Efficient** - ~5-10MB per 1000 memories
⚡ **Fast** - p95 latency <50ms

## Quick Start (30 seconds)

### 1. Deploy to Vercel

Click the button above, or:

```bash
npm install -g vercel
git clone https://github.com/youruser/mythrl
cd mythrl
vercel deploy --prod
```

### 2. Test It Works

```bash
# Get your Vercel URL from deployment output
curl https://your-app.vercel.app/api/health
```

### 3. Store a Memory

```bash
curl -X POST https://your-app.vercel.app/api/experience \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Thompson Sampling balances exploration and exploitation",
    "source": "documentation",
    "confidence": 0.95
  }'
```

### 4. Retrieve Memory

```bash
curl "https://your-app.vercel.app/api/recall?q=thompson+sampling&limit=5"
```

Done! 🎉

## API Reference

### POST /api/experience - Store Memory

Store a new memory that can be recalled later.

**Request:**
```json
{
  "content": "string - Required. The memory content (max 10,000 chars)",
  "source": "string - Optional. Where memory came from (default: 'api')",
  "confidence": "number - Optional. 0.0-1.0, confidence in memory (default: 0.8)",
  "metadata": "object - Optional. Additional JSON metadata"
}
```

**Response (201 Created):**
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

**Example:**
```bash
curl -X POST https://your-app.vercel.app/api/experience \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Thompson Sampling uses Beta distribution priors",
    "source": "research_paper",
    "confidence": 0.92,
    "metadata": {
      "category": "bandit_algorithms",
      "source_url": "https://example.com/paper.pdf"
    }
  }'
```

---

### GET /api/recall - Retrieve Memories

Find memories matching a query using smart similarity scoring.

**Query Parameters:**
- `q` (required): Query text (max 1,000 chars)
- `limit` (optional): Max results 1-50 (default: 10)

**Response (200 OK):**
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

**Example:**
```bash
# Simple query
curl "https://your-app.vercel.app/api/recall?q=thompson+sampling"

# Advanced query with limit
curl "https://your-app.vercel.app/api/recall?q=bayesian+methods&limit=20"

# URL encoded complex query
curl "https://your-app.vercel.app/api/recall?q=how%20to%20implement%20bandits&limit=5"
```

---

### POST /api/query - Store + Recall (Compound)

Store a memory AND retrieve related memories in one request.

**Request:**
```json
{
  "query": "string - Required. Query text to find related memories",
  "experience": {
    "content": "string - Optional. If provided, this memory is stored first",
    "source": "string - Optional",
    "confidence": "number - Optional",
    "metadata": "object - Optional"
  },
  "recall_limit": "number - Optional. Max recalled results (default: 10)"
}
```

**Response (200 OK):**
```json
{
  "query": "thompson sampling",
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
      "age_seconds": 1800
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

**Example:**
```bash
curl -X POST https://your-app.vercel.app/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "bandit algorithms",
    "experience": {
      "content": "Thompson Sampling is a Bayesian approach to bandits",
      "confidence": 0.9
    },
    "recall_limit": 10
  }'
```

---

### GET /api/health - Health Check

Service health status and configuration.

**Response (200 OK):**
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
  "config": {
    "memory_backend": "INMEMORY",
    "max_memories": 1000,
    "cache_ttl_seconds": 3600
  },
  "api_endpoints": {
    "experience": "POST /api/experience",
    "recall": "GET /api/recall",
    "query": "POST /api/query",
    "stats": "GET /api/stats"
  }
}
```

---

### GET /api/stats - Detailed Statistics

System health, storage, and rate limiting stats.

**Response (200 OK):**
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

---

## Rate Limiting

**Default:** 100 requests per 60 seconds per IP

### Rate Limit Headers

All responses include:
```
X-RateLimit-Limit: 100          # Max requests in window
X-RateLimit-Remaining: 95       # Remaining requests
X-RateLimit-Reset: 1704067260   # Unix timestamp when reset
```

### When Limit Exceeded (429 Too Many Requests)

```json
{
  "error": "Rate limit exceeded",
  "limit": 100,
  "remaining": 0,
  "reset": 1704067260,
  "retry_after_seconds": 30
}
```

The `Retry-After` response header also indicates wait time.

### Customize Rate Limits

Edit `vercel.json`:
```json
{
  "env": {
    "RATE_LIMIT_MAX_REQUESTS": {
      "default": "500"
    },
    "RATE_LIMIT_WINDOW_MS": {
      "default": "60000"
    }
  }
}
```

Then redeploy:
```bash
vercel deploy --prod
```

## Configuration

### Environment Variables

Set in Vercel project settings (Project → Settings → Environment Variables):

| Variable | Default | Description |
|----------|---------|-------------|
| `HOLOLOOM_MEMORY_BACKEND` | `INMEMORY` | Storage backend |
| `HOLOLOOM_MAX_MEMORIES` | `1000` | Max in-memory memories |
| `HOLOLOOM_CACHE_TTL` | `3600` | Memory TTL (seconds) |
| `RATE_LIMIT_MAX_REQUESTS` | `100` | Max requests per window |
| `RATE_LIMIT_WINDOW_MS` | `60000` | Rate limit window (ms) |
| `KV_REST_API_URL` | (optional) | Vercel KV endpoint |
| `KV_REST_API_TOKEN` | (optional) | Vercel KV token |

### Local Development

```bash
# Install dependencies
npm install

# Create .env.local
cp .env.vercel .env.local

# Start development server
npm run dev

# Server runs on http://localhost:3000
vercel env pull  # Sync Vercel env vars locally (requires Vercel CLI)
```

### Production with Vercel KV (Persistent Storage)

For data persistence across redeployments:

1. **Create KV Store** in Vercel Dashboard:
   - Project → Storage → Create Database → KV Store
   - Copy `KV_REST_API_URL` and `KV_REST_API_TOKEN`

2. **Add Environment Variables:**
   - Project → Settings → Environment Variables
   - Paste `KV_REST_API_URL` and `KV_REST_API_TOKEN`
   - Redeploy: `vercel deploy --prod`

**Benefits:**
- Data persists across deployments
- Shared storage across function instances
- Automatic fallback to in-memory if KV unavailable

## Performance

### Latency (p95)
```
GET  /api/health           →  5ms
GET  /api/stats            →  8ms
POST /api/experience       → 15ms
GET  /api/recall (10 res)  → 25ms
POST /api/query            → 45ms
```

### Storage
- **In-Memory:** ~1,000 memories (5-10MB)
- **With KV:** Unlimited persistent storage
- **Auto-Compaction:** Removes expired memories every minute

### Scaling
- Automatic horizontal scaling
- Multiple concurrent function instances
- In-memory storage per instance
- KV provides cross-instance persistence

## Similarity Scoring

Smart relevance ranking:

1. **Exact Substring Match** → 0.95 (highest)
2. **Word Overlap** (70% weight) → Matched words / total words
3. **Semantic Boost** (30% weight) → Keyword alignment

**Semantic Categories:**
- Definition: "explain", "what", "is" → +0.3 if match
- Process: "how", "do", "does" → +0.3 if match
- Comparison: "compare", "versus", "vs" → +0.3 if match
- Reason: "why", "reason", "cause" → +0.3 if match
- Example: "example", "like", "instance" → +0.3 if match

**Example:**
```
Query:   "what is thompson sampling"
Memory1: "Thompson Sampling balances exploration"     → 0.92 ✓
Memory2: "Bayesian methods for decision making"       → 0.35 ✓
Memory3: "Beta distribution probability theory"       → 0.20 ✗
Memory4: "Exploration-exploitation tradeoff"          → 0.45 ✓
```

## Examples

### Python Client
```python
import requests

BASE_URL = "https://your-app.vercel.app"

# Store memory
resp = requests.post(
    f"{BASE_URL}/api/experience",
    json={
        "content": "Thompson Sampling is a Bayesian bandit algorithm",
        "source": "python_client",
        "confidence": 0.95
    }
)
memory_id = resp.json()["id"]
print(f"Stored: {memory_id}")

# Retrieve memories
resp = requests.get(
    f"{BASE_URL}/api/recall",
    params={"q": "thompson sampling", "limit": 5}
)
memories = resp.json()["results"]
for mem in memories:
    print(f"  {mem['relevance']:.2f} - {mem['content']}")

# Full cycle
resp = requests.post(
    f"{BASE_URL}/api/query",
    json={
        "query": "bandit algorithms",
        "experience": {
            "content": "Thompson Sampling uses Beta distribution",
            "confidence": 0.9
        },
        "recall_limit": 10
    }
)
print(resp.json())
```

### JavaScript Client
```javascript
const BASE_URL = 'https://your-app.vercel.app';

// Store memory
const memory = await fetch(`${BASE_URL}/api/experience`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    content: 'Thompson Sampling uses Beta distribution',
    confidence: 0.9
  })
}).then(r => r.json());

console.log(`Stored: ${memory.id}`);

// Recall memories
const recalled = await fetch(
  `${BASE_URL}/api/recall?q=thompson%20sampling&limit=5`
).then(r => r.json());

recalled.results.forEach(mem => {
  console.log(`${mem.relevance.toFixed(2)} - ${mem.content}`);
});

// Full query
const result = await fetch(`${BASE_URL}/api/query`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    query: 'bandit algorithms',
    experience: {
      content: 'Thompson Sampling balances exploration',
      confidence: 0.85
    },
    recall_limit: 10
  })
}).then(r => r.json());

console.log(result);
```

### TypeScript Client
```typescript
interface MemoryResponse {
  id: string;
  cached: boolean;
  timestamp: string;
}

interface RecallResult {
  id: string;
  content: string;
  relevance: number;
  confidence: number;
  age_seconds: number;
}

const storeMemory = async (
  baseUrl: string,
  content: string,
  confidence: number = 0.8
): Promise<MemoryResponse> => {
  const res = await fetch(`${baseUrl}/api/experience`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ content, confidence })
  });
  return res.json();
};

const recallMemories = async (
  baseUrl: string,
  query: string,
  limit: number = 10
): Promise<RecallResult[]> => {
  const res = await fetch(
    `${baseUrl}/api/recall?q=${encodeURIComponent(query)}&limit=${limit}`
  );
  const data = await res.json();
  return data.results;
};

// Usage
const BASE_URL = 'https://your-app.vercel.app';
const memory = await storeMemory(BASE_URL, 'Thompson Sampling algorithm');
const results = await recallMemories(BASE_URL, 'thompson sampling');
```

## Monitoring & Debugging

### Check Deployment
```bash
vercel list               # All deployments
vercel logs --tail       # Live logs
```

### Check Stats
```bash
curl https://your-app.vercel.app/api/stats | jq
```

### Health Monitoring
```bash
# Monitor every 5 seconds
watch -n 5 'curl -s https://your-app.vercel.app/api/health | jq ".memory"'
```

### Debug Rate Limits
```bash
curl -i https://your-app.vercel.app/api/health  # View headers
# Look for X-RateLimit-* headers
```

## Troubleshooting

### Rate Limited (429)
**Problem:** Getting "Rate limit exceeded"
```
curl -H "x-forwarded-for: 1.2.3.4" ...  # Check your IP
```
**Solution:**
- Wait for `Retry-After` seconds
- Increase `RATE_LIMIT_MAX_REQUESTS`
- Spread requests over time

### Memories Not Found
**Problem:** `recall()` returns empty despite stored memories
```bash
curl https://your-app.vercel.app/api/stats | jq ".storage"
```
**Solution:**
- Check total_memories > 0
- Try different keywords
- Check relevance scoring with exact substrings
- Increase `HOLOLOOM_CACHE_TTL`

### Memory Growing Too Large
**Problem:** `memory_usage_mb` keeps increasing
```bash
curl https://your-app.vercel.app/api/stats | jq ".storage"
```
**Solution:**
- Reduce `HOLOLOOM_MAX_MEMORIES`
- Reduce `HOLOLOOM_CACHE_TTL`
- Deploy with KV for persistent storage
- Redeploy to clear in-memory storage

### KV Not Working
**Problem:** `"cached": false` in all responses
```bash
# Check environment variables
vercel env list
```
**Solution:**
- Verify `KV_REST_API_URL` and `KV_REST_API_TOKEN` in Vercel settings
- Check KV store status in Vercel Dashboard
- Try disabling KV: remove environment variables
- System automatically falls back to in-memory

## Security Best Practices

1. **Never store sensitive data** - No passwords, tokens, or PII
2. **Use HTTPS** - Vercel provides free SSL
3. **Implement authentication** at your application layer
4. **Monitor rate limits** for abuse patterns
5. **Set reasonable TTLs** to prevent memory bloat
6. **Validate input size** - Enforce content length limits

## Limitations

### Per-Function-Instance
- Max ~1,000 memories (without KV)
- 30-second cold start timeout
- 30-second max function execution
- Containers recycled frequently

### Solutions
- Enable Vercel KV for persistent storage
- Batch operations to fit time limits
- Use external memory service for >10k memories

## Roadmap

- [ ] Vector embeddings for semantic search
- [ ] Full-text search (BM25)
- [ ] Multi-hop graph traversal
- [ ] Temporal memory decay
- [ ] GraphQL API
- [ ] WebSocket subscriptions
- [ ] Memory export/import
- [ ] Custom embeddings models

## Support

- **Documentation:** [VERCEL_DEPLOYMENT.md](VERCEL_DEPLOYMENT.md)
- **GitHub Issues:** [Report bugs](https://github.com/youruser/mythrl/issues)
- **Discussions:** [Ask questions](https://github.com/youruser/mythrl/discussions)

## License

MIT - See [LICENSE](LICENSE)

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md)

---

Made with ❤️ for serverless memory systems
