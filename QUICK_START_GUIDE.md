# HoloLoom Quick Start Guide

**Get started with HoloLoom in 5-15 minutes**
**Version**: v1.1 (November 2025)

This guide helps you quickly set up and start using HoloLoom, from basic installation to production deployment.

---

## 🎯 Choose Your Path

| Path | Time | Audience | Goal |
|------|------|----------|------|
| **[Quick Demo](#quick-demo-2-minutes)** | 2 min | First-time users | See it work |
| **[Basic Setup](#basic-setup-5-minutes)** | 5 min | Developers | Local development |
| **[Production Setup](#production-setup-15-minutes)** | 15 min | DevOps/SRE | Deploy to production |
| **[Full Tutorial](#full-tutorial-30-minutes)** | 30 min | All users | Deep dive |

---

## 🚀 Quick Demo (2 minutes)

**Try HoloLoom without installation** using our online demo or Docker image.

### Option 1: Docker (Recommended)

```bash
# Pull and run official Docker image
docker run -it blakechasteen/hololoom:latest

# Inside container, run demo
python demos/demo_simple.py
```

### Option 2: Google Colab

Click here: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/blakechasteen/mythRL/blob/master/notebooks/Quick_Start.ipynb)

### Option 3: Repl.it

Click here: [Try on Repl.it](https://replit.com/@blakechasteen/HoloLoom-Demo)

---

## ⚡ Basic Setup (5 minutes)

**For local development and experimentation**

### 1. Prerequisites

```bash
# Check Python version (3.10+ required)
python --version

# Should output: Python 3.10.x or higher
```

### 2. Installation

```bash
# Clone repository
git clone https://github.com/blakechasteen/mythRL.git
cd mythRL

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install core dependencies
pip install --upgrade pip
pip install torch numpy networkx sentence-transformers
```

**Minimal installation** (~500MB, 2-3 minutes download):
```bash
pip install torch numpy networkx
```

**Full installation** (~2GB, 5-10 minutes download):
```bash
pip install -r requirements.txt
```

### 3. First Query

```python
from HoloLoom.config import Config
from HoloLoom.weaving_orchestrator import WeavingOrchestrator
from HoloLoom.Documentation.types import Query, MemoryShard

# Create configuration (FAST mode recommended for development)
config = Config.fast()

# Create sample memory
shards = [
    MemoryShard(
        content="Thompson Sampling is a Bayesian approach to balancing exploration and exploitation.",
        source="docs/thompson_sampling.md",
        metadata={"topic": "algorithms"}
    )
]

# Create orchestrator
async def main():
    async with WeavingOrchestrator(cfg=config, shards=shards) as orchestrator:
        # Process query
        spacetime = await orchestrator.weave(Query(text="What is Thompson Sampling?"))

        # Print response
        print(spacetime.response)
        print(f"Confidence: {spacetime.confidence:.2f}")

# Run
import asyncio
asyncio.run(main())
```

**Output**:
```
Thompson Sampling is a Bayesian approach to balancing exploration and exploitation.
It maintains probability distributions over possible actions and samples from them
to decide which action to take.

Confidence: 0.92
```

**That's it!** You now have a working HoloLoom system.

---

## 🏭 Production Setup (15 minutes)

**For staging/production deployment with fault tolerance and monitoring**

### 1. Install Production Dependencies

```bash
# Core dependencies
pip install torch numpy networkx sentence-transformers

# Production dependencies
pip install fastapi uvicorn psutil prometheus-client

# Optional: Docker backends
docker-compose up -d  # Starts Neo4j + Qdrant
```

### 2. Production Configuration

Create `production_config.py`:

```python
from HoloLoom.config import Config, MemoryBackend
from HoloLoom.weaving_orchestrator import WeavingOrchestrator
from HoloLoom.context import ProductionConfig
from HoloLoom.routing import create_smart_router

# Load production config
prod_config = ProductionConfig.production()

# Create HoloLoom config
config = Config.fused()
config.memory_backend = MemoryBackend.HYBRID  # Neo4j + Qdrant
config.enable_semantic_cache = True

# Create smart router
router = create_smart_router(
    enable_fast_paths=True,
    enable_learning=True,
    enable_validation=True
)

# Create orchestrator with production hardening
async def create_production_orchestrator():
    orchestrator = WeavingOrchestrator(
        cfg=config,
        memory=None,  # Will be created dynamically
        enable_production_hardening=True,
        production_config=prod_config,
        rate_limit_qps=100.0,
        rate_limit_concurrent=50,
        enable_circuit_breakers=True,
        circuit_breaker_threshold=5
    )
    return orchestrator
```

### 3. Create FastAPI Service

Create `main.py`:

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from production_config import create_production_orchestrator, router
from HoloLoom.Documentation.types import Query
import asyncio

app = FastAPI(title="HoloLoom API v1.1")

# Global orchestrator (initialized on startup)
orchestrator = None

class QueryRequest(BaseModel):
    text: str
    enable_fast_path: bool = True

class QueryResponse(BaseModel):
    response: str
    confidence: float
    latency_ms: float
    fast_path: bool

@app.on_event("startup")
async def startup():
    global orchestrator
    orchestrator = await create_production_orchestrator()

@app.on_event("shutdown")
async def shutdown():
    global orchestrator
    if orchestrator:
        await orchestrator.close()

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """Process a query with smart routing"""
    import time
    start = time.time()

    # Classify and route
    if request.enable_fast_path:
        classification = router.classify(request.text)

        if classification.complexity.value in ["trivial", "simple"]:
            # Fast path
            response = await router.handle(request.text, classification.complexity)
            latency = (time.time() - start) * 1000

            return QueryResponse(
                response=response,
                confidence=classification.confidence,
                latency_ms=latency,
                fast_path=True
            )

    # Full orchestrator path
    spacetime = await orchestrator.weave(Query(text=request.text))
    latency = (time.time() - start) * 1000

    return QueryResponse(
        response=spacetime.response,
        confidence=spacetime.confidence,
        latency_ms=latency,
        fast_path=False
    )

@app.get("/health")
async def health():
    """Health check for load balancers"""
    from HoloLoom.context import create_health_checker

    health_checker = create_health_checker()
    status = await health_checker.check()

    return {
        "status": status.overall.value,
        "timestamp": status.timestamp,
        "components": {
            "performance": status.performance.value,
            "resources": status.resources.value,
            "circuit_breakers": status.circuit_breakers.value
        }
    }

@app.get("/metrics")
async def metrics():
    """Prometheus metrics"""
    return router.get_metrics()
```

### 4. Run Production Server

```bash
# Development mode (auto-reload)
uvicorn main:app --reload --port 8000

# Production mode (4 workers)
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4

# With Gunicorn (recommended)
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### 5. Test Production Deployment

```bash
# Health check
curl http://localhost:8000/health

# Query endpoint
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"text": "What is Thompson Sampling?"}'

# Metrics
curl http://localhost:8000/metrics
```

### 6. Set Up Monitoring

**Prometheus** (`prometheus.yml`):
```yaml
scrape_configs:
  - job_name: 'hololoom'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
    scrape_interval: 15s
```

**Grafana Dashboard**: Import `HoloLoom/grafana/overview-dashboard.json`

**Alerts** (`alerts.yml`):
```yaml
groups:
  - name: hololoom
    rules:
      - alert: HighLatency
        expr: hololoom_query_latency_ms{quantile="0.95"} > 200
        for: 5m
        annotations:
          summary: "High P95 latency"

      - alert: CircuitBreakerTripped
        expr: increase(hololoom_circuit_breaker_trips_total[1h]) > 5
        annotations:
          summary: "Circuit breaker tripped multiple times"
```

**Production deployment complete!** ✅

---

## 📚 Full Tutorial (30 minutes)

### Step 1: Understanding the Architecture

HoloLoom uses a "weaving" metaphor:

```
Query → [Loom Command] → [Yarn Graph] → [Warp Space] → [Convergence] → Response
         Pattern Card      Discrete      Continuous    Decision        Provenance
```

**Key components**:
- **Yarn Graph**: Discrete symbolic memory (NetworkX MultiDiGraph)
- **Warp Space**: Continuous neural manifold (embeddings)
- **Convergence Engine**: Decision collapse (Thompson Sampling)
- **Spacetime**: Structured output with complete provenance

### Step 2: Configuration Modes

HoloLoom has 3 execution modes optimized for different use cases:

```python
from HoloLoom.config import Config

# BARE: Minimal (fastest, least accurate)
config_bare = Config.bare()
# - Regex motifs only
# - Single scale (96D embeddings)
# - Simple policy
# - ~50ms per query

# FAST: Balanced (recommended for development)
config_fast = Config.fast()
# - Hybrid motifs (regex + basic NLP)
# - 2 scales (96D, 192D)
# - Neural policy with Thompson Sampling
# - ~150ms per query

# FUSED: Full power (best quality, production)
config_fused = Config.fused()
# - Full NLP motifs (spaCy)
# - 3 scales (96D, 192D, 384D)
# - Multi-head attention policy
# - Multi-scale retrieval
# - ~300ms per query
```

### Step 3: Memory Systems

HoloLoom supports 3 memory backends:

```python
from HoloLoom.config import MemoryBackend

# 1. INMEMORY: In-memory NetworkX (development)
config.memory_backend = MemoryBackend.INMEMORY
# - No persistence
# - Fast (~50ms queries)
# - Always available

# 2. HYBRID: Neo4j + Qdrant (production, recommended)
config.memory_backend = MemoryBackend.HYBRID
# - Persistent storage
# - Graph + vector search
# - Auto-fallback to INMEMORY if unavailable
# - ~150ms queries

# 3. HYPERSPACE: Advanced features (research)
config.memory_backend = MemoryBackend.HYPERSPACE
# - Gated multipass
# - Experimental
```

### Step 4: Creating Memory

```python
from HoloLoom.Documentation.types import MemoryShard

# Manual memory creation
shards = [
    MemoryShard(
        content="Thompson Sampling balances exploration and exploitation.",
        source="algorithm_docs.md",
        metadata={"category": "algorithms", "difficulty": "advanced"}
    ),
    MemoryShard(
        content="It uses Bayesian probability distributions over actions.",
        source="algorithm_docs.md",
        metadata={"category": "algorithms", "difficulty": "advanced"}
    )
]

# Or ingest from files
from HoloLoom.spinningWheel import AudioSpinner, YouTubeSpinner

# From audio
audio_shards = await AudioSpinner().spin({
    'file_path': 'recording.mp3',
    'enable_enrichment': True
})

# From YouTube
yt_shards = await YouTubeSpinner().spin({
    'url': 'VIDEO_ID',
    'chunk_duration': 60.0
})
```

### Step 5: Querying with Provenance

```python
async with WeavingOrchestrator(cfg=config, shards=shards) as orchestrator:
    spacetime = await orchestrator.weave(Query(text="Explain Thompson Sampling"))

    # Response
    print(spacetime.response)

    # Provenance
    print(f"Tool used: {spacetime.trace.tool_used}")
    print(f"Sources: {len(spacetime.trace.sources)} memory shards")
    print(f"Features extracted: {spacetime.trace.features.motifs}")

    # Confidence
    print(f"Confidence: {spacetime.confidence:.2%}")

    # Metadata
    print(f"Duration: {spacetime.trace.duration_ms:.1f}ms")
    print(f"Complexity: {spacetime.trace.complexity.value}")
```

### Step 6: Learning from Feedback

```python
from HoloLoom.reflection import ReflectionBuffer

async with ReflectionBuffer(capacity=1000) as buffer:
    # Store interaction with feedback
    await buffer.store(spacetime, feedback={
        "helpful": True,
        "accurate": True,
        "complete": False  # Response could be more complete
    })

    # System learns from feedback
    # - Adjusts Thompson Sampling priors
    # - Updates policy adapter weights
    # - Strengthens hot patterns
```

### Step 7: Production Features

```python
# Enable all production features
orchestrator = WeavingOrchestrator(
    cfg=Config.fused(),
    shards=shards,

    # Smart routing
    enable_smart_routing=True,

    # Production hardening
    enable_production_hardening=True,
    rate_limit_qps=100.0,
    rate_limit_concurrent=50,
    enable_circuit_breakers=True,

    # Monitoring
    enable_monitoring=True,
    metrics_export="prometheus",

    # Caching
    enable_semantic_cache=True,
    query_cache_size=10000
)
```

---

## 🎓 Learning Resources

### Beginner (5-15 minutes)

- **[README.md](README.md)** - Project overview
- **This guide** - Quick start
- **[VISUAL_QUICK_START.md](VISUAL_QUICK_START.md)** - Visual learning path

### Intermediate (15-30 minutes)

- **[CLAUDE.md](CLAUDE.md)** - Developer reference (4,000+ lines)
- **[ARCHITECTURE_VISUAL_MAP.md](ARCHITECTURE_VISUAL_MAP.md)** - Architecture diagrams
- **[demos/](demos/)** - 50+ demo scripts

### Advanced (1-2 hours)

- **[HOLOLOOM_MASTER_SCOPE_AND_SEQUENCE.md](HOLOLOOM_MASTER_SCOPE_AND_SEQUENCE.md)** - Complete architectural map (25,000+ lines)
- **[PHASE_3_DOCUMENTATION.md](PHASE_3_DOCUMENTATION.md)** - Adaptive learning deep dive
- **[RECURSIVE_LEARNING_COMPLETE.md](RECURSIVE_LEARNING_COMPLETE.md)** - 7 learning systems

### Production Deployment

- **[MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)** - Upgrade to v1.1
- **[PERFORMANCE_BENCHMARKS.md](PERFORMANCE_BENCHMARKS.md)** - Performance analysis
- **[HoloLoom/context/PRODUCTION_QUICK_START.md](HoloLoom/context/PRODUCTION_QUICK_START.md)** - Production setup
- **[HoloLoom/context/TROUBLESHOOTING_GUIDE.md](HoloLoom/context/TROUBLESHOOTING_GUIDE.md)** - Common issues

---

## 🔧 Common Tasks

### Task 1: Change Execution Mode

```python
# From FAST to FUSED
config = Config.fused()  # Was: Config.fast()
```

### Task 2: Add New Memory

```python
# Create new shard
new_shard = MemoryShard(
    content="New knowledge to remember",
    source="user_input",
    metadata={"timestamp": "2025-11-20"}
)

# Add to orchestrator
shards.append(new_shard)
```

### Task 3: Enable Production Features

```python
# Before: Basic setup
orchestrator = WeavingOrchestrator(cfg=config, shards=shards)

# After: Production setup
orchestrator = WeavingOrchestrator(
    cfg=config,
    shards=shards,
    enable_production_hardening=True,
    rate_limit_qps=100.0
)
```

### Task 4: Export Metrics

```python
# Get Prometheus metrics
from HoloLoom.routing import get_telemetry

telemetry = get_telemetry()
print(f"Classification accuracy: {telemetry.accuracy:.1%}")
print(f"Avg latency: {telemetry.avg_classification_ms:.1f}ms")
```

### Task 5: Debug Slow Queries

```python
# Enable detailed tracing
spacetime = await orchestrator.weave(Query(text="..."))

# Inspect timing
print(f"Total: {spacetime.trace.duration_ms:.1f}ms")
print(f"Retrieval: {spacetime.trace.retrieval_ms:.1f}ms")
print(f"Features: {spacetime.trace.feature_extraction_ms:.1f}ms")
print(f"Policy: {spacetime.trace.policy_forward_ms:.1f}ms")
```

---

## ⚙️ Configuration Reference

### Core Settings

```python
config = Config.fused()

# Execution mode
config.execution_mode = "fused"  # bare, fast, fused

# Memory backend
config.memory_backend = MemoryBackend.HYBRID  # inmemory, hybrid, hyperspace

# Embedding scales
config.embedding_scales = [96, 192, 384]  # Matryoshka dimensions

# Caching
config.enable_semantic_cache = True
config.query_cache_size = 10000
config.query_cache_ttl = 3600  # 1 hour

# Complexity detection
config.enable_complexity_auto_detect = True
```

### Production Settings

```python
# Rate limiting
config.rate_limit_qps = 100.0
config.rate_limit_concurrent = 50

# Circuit breakers
config.enable_circuit_breakers = True
config.circuit_breaker_threshold = 5
config.circuit_breaker_timeout = 60.0

# Monitoring
config.enable_monitoring = True
config.metrics_export = "prometheus"  # prometheus, json, none
config.metrics_port = 9090
```

### Routing Settings

```python
# Smart routing
config.enable_smart_routing = True
config.enable_fast_paths = True

# Adaptive learning
config.enable_adaptive_learning = True
config.pattern_quality_threshold = 0.95
config.pattern_support_threshold = 10

# Validation
config.enable_continuous_validation = True
config.validation_interval = 3600.0  # 1 hour
config.regression_threshold = 0.02  # 2% accuracy drop
```

---

## 🐛 Troubleshooting

### Issue 1: Import Error

**Error**: `ModuleNotFoundError: No module named 'HoloLoom'`

**Solution**:
```bash
# Add repository root to PYTHONPATH
export PYTHONPATH=.
python your_script.py

# Or run from repository root
cd mythRL
python your_script.py
```

### Issue 2: Slow Performance

**Symptom**: Queries taking >1 second

**Solution**:
```python
# Use FAST mode instead of FUSED
config = Config.fast()  # Was: Config.fused()

# Enable smart routing
enable_smart_routing = True

# Enable query caching
config.enable_semantic_cache = True
```

### Issue 3: Out of Memory

**Symptom**: `MemoryError` or system freezing

**Solution**:
```python
# Reduce embedding scales
config.embedding_scales = [96]  # Was: [96, 192, 384]

# Reduce cache size
config.query_cache_size = 1000  # Was: 10000

# Use INMEMORY backend
config.memory_backend = MemoryBackend.INMEMORY
```

### Issue 4: Docker Backend Not Found

**Symptom**: "Neo4j connection failed, falling back to INMEMORY"

**Solution**:
```bash
# Start Docker backends
docker-compose up -d

# Verify running
docker ps

# Check logs
docker-compose logs neo4j
docker-compose logs qdrant
```

### More Help

- **[HoloLoom/context/TROUBLESHOOTING_GUIDE.md](HoloLoom/context/TROUBLESHOOTING_GUIDE.md)** - Comprehensive troubleshooting
- **[GitHub Issues](https://github.com/blakechasteen/mythRL/issues)** - Report bugs
- **[Discussions](https://github.com/blakechasteen/mythRL/discussions)** - Ask questions

---

## ✅ Next Steps

**After completing this guide, you can**:
- ✅ Run HoloLoom locally
- ✅ Process queries and get responses
- ✅ Understand configuration modes
- ✅ Deploy to production with fault tolerance
- ✅ Monitor performance with Prometheus

**Continue learning**:
1. **Try more demos**: Browse `demos/` directory for 50+ examples
2. **Read architecture docs**: [CLAUDE.md](CLAUDE.md) for deep dive
3. **Join the community**: [GitHub Discussions](https://github.com/blakechasteen/mythRL/discussions)

**Happy weaving!** 🧶✨
