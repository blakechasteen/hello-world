# Distributed Tracing Quick Start

Get up and running with OpenTelemetry tracing in 5 minutes.

---

## Step 1: Install Dependencies

```bash
pip install -r HoloLoom/tracing/requirements.txt
```

**Or manually**:
```bash
pip install \
  opentelemetry-api \
  opentelemetry-sdk \
  opentelemetry-exporter-jaeger \
  opentelemetry-instrumentation-fastapi
```

---

## Step 2: Start Jaeger

```bash
# Using docker-compose
docker-compose -f docker-compose.tracing.yml up jaeger -d

# Or standalone
docker run -d --name jaeger \
  -p 16686:16686 \
  -p 6831:6831/udp \
  jaegertracing/all-in-one:latest
```

**Verify**: Open http://localhost:16686 (Jaeger UI)

---

## Step 3: Enable Tracing

### Option A: Environment Variables

```bash
export TRACING_ENABLED=true
export TRACING_EXPORTER=jaeger
export TRACING_SERVICE_NAME=hololoom-dashboard
export TRACING_SAMPLE_RATE=1.0
```

### Option B: Python Configuration

```python
from HoloLoom.tracing import TracingConfig, init_tracing, instrument_app

app = FastAPI()

config = TracingConfig(
    service_name="hololoom-dashboard",
    exporter="jaeger",
    sample_rate=1.0
)
tracer = init_tracing(config)
instrument_app(app)
```

---

## Step 4: Add to Your Application

### FastAPI Server

```python
from fastapi import FastAPI
from HoloLoom.tracing import init_tracing, instrument_app, instrument_hololoom

app = FastAPI()

# Initialize
tracer = init_tracing(TracingConfig(exporter="jaeger"))

# Auto-instrument
instrument_app(app)
instrument_hololoom()

@app.get("/query")
async def query(text: str):
    # Automatically traced!
    result = await process_query(text)
    return result
```

### Custom Functions

```python
from HoloLoom.tracing import traced, create_span

@traced(attributes={"component": "memory"})
async def recall_memories(query: str):
    # Automatically traced
    return memories

# Or manual spans
async def complex_operation():
    with create_span("phase_1"):
        await phase_1()

    with create_span("phase_2"):
        await phase_2()
```

---

## Step 5: Run and View

```bash
# Run your application
uvicorn app:app --reload

# Make requests
curl http://localhost:8000/query?text=test

# View traces
# Open http://localhost:16686
# Search for service: hololoom-dashboard
```

---

## Common Use Cases

### 1. Basic Span

```python
from HoloLoom.tracing import create_span

with create_span("my_operation", {"key": "value"}):
    result = do_work()
```

### 2. Error Tracking

```python
from HoloLoom.tracing import record_exception

try:
    risky_operation()
except Exception as e:
    record_exception(e)
    raise
```

### 3. Events

```python
from HoloLoom.tracing import add_span_event

add_span_event("cache_hit", {"key": "query_123"})
```

### 4. Database Queries

```python
from HoloLoom.tracing import trace_database_query

with trace_database_query("SELECT", "SELECT * FROM users", "postgres"):
    cursor.execute(query)
```

### 5. WebSocket

```python
from HoloLoom.tracing.instrumentation import WebSocketTracing

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    async with WebSocketTracing(websocket) as ws_tracer:
        with ws_tracer.trace_message("receive"):
            data = await websocket.receive_json()
```

---

## Testing

```bash
# Run test suite
python HoloLoom/tracing/test_tracing.py

# Run integration example
TRACING_ENABLED=true python HoloLoom/tracing/integration_example.py
```

---

## Troubleshooting

### No traces in Jaeger?

**Check**:
1. Is Jaeger running? `docker ps | grep jaeger`
2. Is tracing enabled? `echo $TRACING_ENABLED`
3. Try console exporter: `TRACING_EXPORTER=console`

### Performance issues?

**Reduce sampling**:
```bash
export TRACING_SAMPLE_RATE=0.1  # 10% sampling
```

### Import errors?

**Install dependencies**:
```bash
pip install -r HoloLoom/tracing/requirements.txt
```

---

## Next Steps

- Read full documentation: [README.md](README.md)
- Configure for production: See "Production Deployment" in README
- Analyze performance: Use `PerformanceAnalyzer`
- Set up alerts: Monitor dropped spans

---

## Resources

- **Jaeger UI**: http://localhost:16686
- **OpenTelemetry Docs**: https://opentelemetry.io/docs/
- **Jaeger Docs**: https://www.jaegertracing.io/docs/

---

**Questions?** See [README.md](README.md) for complete documentation.
