# Dashboard Quick Start Guide 🚀

**Get the Promptly Performance Dashboard running in 2 minutes!**

---

## Prerequisites

```bash
pip install flask flask-cors flask-socketio
```

---

## Step 1: Generate Sample Data (Optional)

If you don't have existing metrics data:

```bash
cd promptly_skills/analytics
python test_metrics_system.py
```

This creates `test_metrics.db` with 100 sample queries.

---

## Step 2: Start the API Server

```bash
cd promptly_skills/analytics
python dashboard_api.py
```

You should see:

```
============================================================
  Promptly Performance Dashboard API
  http://localhost:5001
============================================================

Endpoints:
  GET  /api/health           - Health check
  GET  /api/stats            - Aggregated statistics
  GET  /api/trends           - Time-series trends
  GET  /api/top_strategies   - Top performing strategies
  GET  /api/cache_stats      - Cache performance
  WebSocket /socket.io       - Real-time updates

Dashboard API initialized with database: test_metrics.db
Started background metrics broadcast (every 2s)
 * Running on http://0.0.0.0:5001
```

---

## Step 3: Open the Dashboard

**Option A**: Open directly in browser

```bash
# Linux/Mac
open dashboard/index.html

# Windows
start dashboard/index.html
```

**Option B**: Serve via HTTP (better for WebSocket)

```bash
# Python 3
cd promptly_skills/dashboard
python -m http.server 8000

# Then open: http://localhost:8000
```

---

## What You'll See 📊

### Dashboard Features

**Summary Metrics** (real-time):
- Total queries
- Average confidence
- Average latency
- Cache hit rate

**Latency Percentiles**:
- P50 (median)
- P95
- P99

**Cache Performance**:
- Total hits/misses
- Speedup factor

**Trends** (6-hour window):
- Latency trend chart
- Confidence trend chart

**Top Strategies**:
- Performance comparison
- Usage statistics
- Visual bars

**Real-time Updates**:
- WebSocket connection (updates every 2s)
- Connection status indicator
- Last update timestamp

---

## Testing the API

### Health Check

```bash
curl http://localhost:5001/api/health
```

**Response**:
```json
{
  "status": "healthy",
  "timestamp": 1234567890.123,
  "version": "1.0.0"
}
```

### Get Statistics

```bash
curl "http://localhost:5001/api/stats?period=24h"
```

**Response**:
```json
{
  "period": "24h",
  "total_queries": 100,
  "avg_latency_ms": 145.2,
  "avg_confidence": 0.918,
  "cache_hit_rate": 0.30,
  "strategy_distribution": {
    "deep": 23,
    "optimize": 18,
    ...
  }
}
```

### Get Trends

```bash
curl "http://localhost:5001/api/trends?metric=latency_ms&period=24h&interval=60"
```

**Response**:
```json
{
  "metric": "latency_ms",
  "period": "24h",
  "data": [
    {"timestamp": 1234567800, "value": 150.5},
    {"timestamp": 1234571400, "value": 142.3},
    ...
  ]
}
```

### Get Top Strategies

```bash
curl "http://localhost:5001/api/top_strategies?period=24h&limit=5"
```

**Response**:
```json
{
  "strategies": [
    {
      "strategy": "optimize",
      "performance": {
        "avg_confidence": 0.940,
        "avg_latency_ms": 198.5,
        "total_uses": 18,
        "success_rate": 1.0
      }
    },
    ...
  ]
}
```

---

## API Reference

### Endpoints

| Endpoint | Method | Parameters | Description |
|----------|--------|------------|-------------|
| `/api/health` | GET | - | Health check |
| `/api/stats` | GET | `period` (1h/24h/7d/30d) | Aggregated statistics |
| `/api/trends` | GET | `metric`, `period`, `interval` | Time-series trends |
| `/api/top_strategies` | GET | `period`, `limit`, `metric` | Top performing strategies |
| `/api/cache_stats` | GET | `period` | Cache performance |

### WebSocket Events

| Event | Direction | Payload | Description |
|-------|-----------|---------|-------------|
| `connect` | Client → Server | - | Initial connection |
| `subscribe_metrics` | Client → Server | - | Subscribe to updates |
| `metrics_update` | Server → Client | Metrics object | Real-time metrics (every 2s) |

---

## Customization

### Change Update Interval

Edit `dashboard/index.html`:

```javascript
const UPDATE_INTERVAL = 10000; // Change to 5000 for 5 seconds
```

### Change API URL

Edit `dashboard/index.html`:

```javascript
const API_URL = 'http://localhost:5001'; // Change to your server
```

### Modify Charts

Edit chart configurations in `dashboard/index.html`:

```javascript
function initializeCharts() {
    // Customize chart options here
    latencyChart = new Chart(latencyCtx, {
        type: 'line',  // Change to 'bar', 'scatter', etc.
        options: {
            // Your custom options
        }
    });
}
```

---

## Troubleshooting

### "Connection refused" error

**Problem**: API server not running
**Solution**: Start the API server first: `python dashboard_api.py`

### "No metrics found"

**Problem**: No data in database
**Solution**: Generate sample data: `python test_metrics_system.py`

### WebSocket not connecting

**Problem**: CORS or port issues
**Solution**:
1. Serve dashboard via HTTP: `python -m http.server 8000`
2. Open `http://localhost:8000` (not `file://`)

### Charts not showing

**Problem**: No trend data or wrong time range
**Solution**: Generate more sample data or adjust time range in API calls

---

## Production Deployment

### Option 1: Docker (Recommended)

Create `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY analytics/ ./analytics/
COPY dashboard/ ./dashboard/

EXPOSE 5001

CMD ["python", "analytics/dashboard_api.py"]
```

Build and run:

```bash
docker build -t promptly-dashboard .
docker run -p 5001:5001 promptly-dashboard
```

### Option 2: Gunicorn

```bash
pip install gunicorn

gunicorn -w 4 -b 0.0.0.0:5001 \
  --worker-class eventlet \
  analytics.dashboard_api:app
```

### Option 3: Nginx Reverse Proxy

```nginx
server {
    listen 80;
    server_name dashboard.promptly.ai;

    location / {
        root /var/www/dashboard;
        try_files $uri $uri/ /index.html;
    }

    location /api/ {
        proxy_pass http://localhost:5001/api/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }

    location /socket.io/ {
        proxy_pass http://localhost:5001/socket.io/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

---

## Integration with Promptly

### Hook into Orchestrator

Add metrics collection to your weaving orchestrator:

```python
from analytics import get_metrics_collector

collector = get_metrics_collector()
await collector.start()

# In your orchestrator
async def weave(query: str, strategy: str):
    start_time = time.time()

    # Execute strategy
    result = await strategy.enhance(query)

    # Record metrics
    latency_ms = (time.time() - start_time) * 1000
    await collector.record_query(
        query=query,
        strategy=strategy.name,
        latency_ms=latency_ms,
        confidence=result.confidence,
        cache_hit=False  # or your cache logic
    )

    return result
```

### Background Daemon

Run collector as background service:

```python
import asyncio
from analytics import MetricsCollector, TimeSeriesDB

async def start_metrics_service():
    db = TimeSeriesDB('promptly_metrics.db')
    await db.initialize()

    collector = MetricsCollector(db=db)
    await collector.start()

    # Keep running
    try:
        await asyncio.Event().wait()
    finally:
        await collector.stop()

if __name__ == '__main__':
    asyncio.run(start_metrics_service())
```

---

## Next Steps

1. ✅ **Day 2 Complete**: You have a working dashboard!
2. **Week 1**: Polish dashboard (add more charts, filters, date ranges)
3. **Week 3-4**: Add A/B testing UI
4. **Week 5-6**: Add strategy composer
5. **Week 7-8**: Add advanced learning visualizations

---

## Support

**Issues?** Check:
1. Is Python 3.8+ installed?
2. Are all dependencies installed?
3. Is the API server running?
4. Is the database created?
5. Are ports 5001 and 8000 available?

**Still stuck?**
- Check server logs in terminal
- Open browser console (F12) for errors
- Test API endpoints with `curl`

---

## Summary

**What you built today** ✨:
- ✅ Complete REST API (5 endpoints)
- ✅ WebSocket real-time updates
- ✅ Beautiful HTML dashboard
- ✅ Interactive charts (Chart.js)
- ✅ Performance metrics visualization
- ✅ Production-ready code

**Time to completion**: ~2 minutes (after setup)

**Lines of code**: ~800 (dashboard + API)

**It just works!** 🚀

---

**Enjoy your real-time performance dashboard!** 📊✨
