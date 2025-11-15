# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 📚 Comprehensive Documentation

**New to HoloLoom?** Start here:

1. **[HOLOLOOM_MASTER_SCOPE_AND_SEQUENCE.md](HOLOLOOM_MASTER_SCOPE_AND_SEQUENCE.md)** (25,000+ lines)
   - Complete architectural map from first principles to production
   - Learning sequence for beginners → researchers
   - All 5 phases explained with context
   - Future roadmap (Phases 6-10)
   - **Start here for the big picture!**

2. **[CURRENT_STATUS_AND_NEXT_STEPS.md](CURRENT_STATUS_AND_NEXT_STEPS.md)**
   - What works right now (snapshot)
   - What needs work (prioritized tasks)
   - Recommended next actions
   - Quick decision guide
   - **Use this to know what to build next**

3. **[ARCHITECTURE_VISUAL_MAP.md](ARCHITECTURE_VISUAL_MAP.md)**
   - Visual diagrams of the 9-layer system
   - Data flow illustrations
   - Component relationships
   - Quick reference to key files
   - **Best for visual learners**

4. **This file (CLAUDE.md)** - Developer quick reference (below)

5. **[DREAMWEAVER_SUMMARY.md](DREAMWEAVER_SUMMARY.md)** - Open-source world building component
   - Phase 0 complete (architecture)
   - 6-phase roadmap (18 months)
   - Extends HoloLoom to collaborative storytelling
   - **For world building and interactive fiction**

---

## Reliable Systems: Safety First

**"Reliable Systems: Safety First"** is our guiding development philosophy. Before optimizing for performance, features, or elegance, we prioritize:

- **Graceful degradation**: Systems should never crash due to missing optional dependencies
- **Automatic fallbacks**: When production backends fail, fall back to working alternatives (e.g., HYBRID → INMEMORY)
- **Proper lifecycle management**: All resources get explicit cleanup through async context managers
- **Comprehensive testing**: Unit, integration, and end-to-end tests organized by speed for fast feedback
- **Clear error messages**: When things fail, developers should immediately understand why and how to fix it
- **Type safety**: Protocol-based design with clear interfaces prevents integration errors
- **Data persistence safety**: Never lose user data - archive instead of delete, checkpoint frequently

This principle permeates every architectural decision in HoloLoom. We'd rather ship a slower but reliable system than a fast but fragile one.

---

## Documentation Standards

**"Everything has a timestamp, everything has a story."**

To maintain temporal context and enable future queries about project evolution, **always include datestamps** when documenting work.

### Required Datestamps

Include timestamps on:
- ✅ **New features/implementations** - When was this built?
- ✅ **Code changes and updates** - When was this modified?
- ✅ **Documentation additions** - When was this documented?
- ✅ **Architecture decisions** - When was this decided?
- ✅ **Session summaries** - When did this work happen?
- ✅ **Status updates** - When did status change?

### Datestamp Formats

Use consistent formats across the codebase:

**Full Dates** (preferred for precise tracking):
```
YYYY-MM-DD (e.g., 2025-11-13)
```

**Month Precision** (for longer-term features):
```
Month YYYY (e.g., November 2025)
```

**In Code Comments**:
```python
# Added 2025-11-13: Zero-copy embedding optimization
# Updated 2025-11-13: Fix cache invalidation bug
```

**In Markdown Headers**:
```markdown
## Zero-Copy Embeddings (November 2025)

**Implemented: 2025-11-13** - High-performance embedding layer...
```

**In Section Updates**:
```markdown
**UPDATED (2025-11-13):** The Shuttle architecture has been integrated...
```

**In Git Commits** (automatic via commit metadata):
```bash
git log --format="%ai %s"
```

### Why Datestamps Matter

1. **Temporal Queries**: Enable questions like "What changed in October 2025?"
2. **Feature Evolution**: Track how systems evolved over time
3. **Context for Future Sessions**: Help future Claude sessions understand project timeline
4. **Historical Audit**: Complete provenance of all decisions and changes
5. **Stale Detection**: Identify outdated documentation that needs updating
6. **Retrospectives**: Enable temporal analysis ("How long did Phase 5 take?")

### Examples from This Codebase

Good datestamping practices already in use:

```markdown
✅ **Zero-Copy Embeddings** (November 2025)
✅ **Test Organization** (Phase 1+2 Cleanup - Oct 2025)
✅ **Alignment Framework** (v1.0.0 - November 2025)
✅ **UPDATED (Task 1.2 - Oct 27, 2025):** The Shuttle architecture...
```

### Pro Tips

- **When in doubt, datestamp it** - More temporal context is always better
- **Update existing dates** - When revising documented features, add update dates
- **Use ISO format for precision** - YYYY-MM-DD is sortable and unambiguous
- **Include in commit messages** - Git already tracks this, but call it out in descriptions
- **Mark completion dates** - "Status: ✅ Complete (November 2025)"

---

## Agent Swarm Deployment Strategy

When deploying multiple Claude Code agents in parallel for complex tasks, use this model selection matrix for **optimal cost-performance**:

### Model Selection Guide

| Task Type | Model | Reasoning | Cost Savings |
|-----------|-------|-----------|--------------|
| **Testing/Validation** | 🔵 Haiku | Deterministic checks, pattern matching | **90% cheaper** |
| **Code Reading** | 🔵 Haiku | Syntax analysis, simple refactoring | **90% cheaper** |
| **Documentation** | 🔵 Haiku | Structured output, templates | **90% cheaper** |
| **Simple Refactoring** | 🔵 Haiku | Rule-based transformations | **90% cheaper** |
| **Architecture Analysis** | 🟢 Sonnet | Complex reasoning, system understanding | Worth the cost |
| **Integration Work** | 🟢 Sonnet | Multi-system coordination | Worth the cost |
| **Novel Algorithms** | 🟢 Sonnet | Creative problem solving | Worth the cost |

### Deployment Waves

**Wave Pattern**: Group independent tasks, deploy in parallel with appropriate models

**Example (Week 1 Roadmap)**:
```
Wave 1 (Parallel):
- Agent A: Parallelize ops → Haiku (code reading + simple refactor)
- Agent B: Add unit tests → Haiku (deterministic test creation)
- Agent C: Create diagrams → Haiku (structured Mermaid output)

Wave 2 (Depends on Wave 1):
- Agent D: Yarn Graph integration → Sonnet (architecture understanding)
- Agent E: 9-layer test → Sonnet (system integration)

Testing Wave (Parallel with Wave 2):
- Agent F: Test Loom Command → Haiku (validation)
- Agent G: Test Chrono Trigger → Haiku (validation)
- Agent H: Test Warp Space → Haiku (validation)
```

### Cost Optimization Results

Real-world savings from Week 1 implementation:
- **3 Sonnet agents** (Wave 1): ~26k tokens wasted → Should have used Haiku
- **2 Sonnet agents** (Wave 2): Correctly used for complex integration
- **3 Haiku agents** (Testing): Optimal choice, 90% cost savings
- **Overall efficiency**: 60% optimal (3/5 critical agents used correct model)

**Recommendation**: Default to Haiku unless task requires complex reasoning or architectural understanding.

---

## Repository Overview

**HoloLoom** is a Python-based neural decision-making system that combines:
- Multi-scale embeddings (Matryoshka representations)
- Knowledge graph memory with spectral features
- Unified policy engine with Thompson Sampling exploration
- PPO reinforcement learning for agent training
- 47 input adapters ("SpinningWheel") for processing diverse modalities: audio, video, web, code, documents, and more

The system is designed around a "weaving" metaphor: independent "warp thread" modules are coordinated by an "orchestrator" (the shuttle) to produce responses.

## RAG (Retrieval-Augmented Generation) System

**Status**: ✅ Production Ready (November 2025)
**Location**: `HoloLoom/rag/`
**Level**: Level 4 Agentic RAG + Graph RAG
**Performance**: <200ms latency, 24/25 tests passing

### Overview

HoloLoom includes a complete, production-ready RAG system that wraps its sophisticated architecture into a simple, zero-config API. Unlike basic RAG implementations, HoloLoom RAG includes:

- **Level 2 Hybrid RAG**: BM25 (keyword) + semantic similarity search
- **Level 3 Graph RAG**: Entity relationships via Yarn Graph (NetworkX MultiDiGraph)
- **Level 4 Agentic RAG**: Multi-step reasoning (DIRECT/VERIFY/RESEARCH/PLAN_EXECUTE modes)
- **Multimodal RAG**: Text + images with CLIP embeddings and visual compression
- **Performance Dashboard**: Auto-constructed visualizations with anomaly detection

Most RAG systems stop at Level 2. HoloLoom provides Level 4 out of the box.

### Quick Start

**Simple RAG (Text-Only)**:
```python
from HoloLoom.rag import SimpleRAG

# Zero-config initialization
async with SimpleRAG() as rag:
    # Ingest content (any modality)
    await rag.ingest("Thompson Sampling balances exploration/exploitation...")

    # Single-line query
    result = await rag.query("What is Thompson Sampling?")

    # Structured result
    print(result.response)      # LLM-generated answer
    print(result.sources)       # Retrieved source texts
    print(result.confidence)    # 0.0-1.0
    print(result.reasoning_mode) # "verify"
```

**Multimodal RAG (Text + Images)**:
```python
from HoloLoom.rag import MultimodalRAG

async with MultimodalRAG() as rag:
    # Store photos with CLIP embeddings
    photo_id = await rag.ingest_photo(
        image="architecture.png",
        tags=["architecture", "system_design"],
        description="System architecture diagram"
    )

    # Query with image context (OCR + CLIP)
    result = await rag.query_with_image(
        question="What's in this diagram?",
        image="architecture.png"
    )

    print(result.response)         # LLM answer
    print(result.image_sources)    # Retrieved images
    print(result.compression_ratio) # e.g., 12.5x token savings
```

**RAG Performance Dashboard**:
```python
from HoloLoom.visualization import RAGDashboard

# Auto-construct dashboard from query history
queries = [...]  # List of RAGResult objects
dashboard = RAGDashboard.from_query_history(queries)
dashboard.save("rag_performance.html")

# 5 panels auto-generated:
# 1. Retrieval Quality (precision/recall over time)
# 2. Latency Waterfall (stage timing breakdown)
# 3. Cache Effectiveness (hit rate gauge)
# 4. Confidence Trajectory (anomaly detection)
# 5. Source Attribution (knowledge graph)
```

### Architecture

```
SimpleRAG (Level 2-4 RAG)
├── Text Path
│   ├── ingest() → hololoom.experience()
│   ├── query() → hololoom.recall() + LLM generation
│   └── batch_query() → efficient batch processing
│
├── Features
│   ├── Query caching (100x faster for repeated queries)
│   ├── Hybrid search (BM25 + semantic)
│   ├── Graph traversal (Yarn Graph relationships)
│   └── Agentic reasoning (4 modes)
│
└── LLM Integration
    ├── Ollama (local, default: llama3.2:3b)
    ├── Anthropic (Claude 3.5 Sonnet)
    └── OpenAI (GPT-4)

MultimodalRAG (extends SimpleRAG)
├── Visual Path
│   ├── ingest_photo() → hololoom.remember_photo()
│   ├── query_with_image() → OCR + CLIP + recall(include_photos=True)
│   └── get_related_photos() → CLIP similarity search
│
├── OCR Integration
│   ├── Primary: DeepSeek OCR (best quality)
│   ├── Fallback: pytesseract (basic)
│   └── Graceful degradation (returns empty if unavailable)
│
└── Visual Compression
    ├── Auto-compresses when sources > 10 (configurable)
    ├── Knowledge graph → image (5-20x token savings)
    └── Returns PNG bytes + compression metrics
```

### Key Features

#### 1. Zero-Config API

No configuration required. Just instantiate and use:
```python
rag = SimpleRAG()  # Uses Config.fast() by default
```

Customize if needed:
```python
from HoloLoom.config import Config

rag = SimpleRAG(
    config=Config.fused(),           # BARE/FAST/FUSED modes
    llm_provider="anthropic",        # ollama/anthropic/openai
    llm_model="claude-3-5-sonnet-20241022"
)
```

#### 2. Reasoning Modes

Choose reasoning complexity based on your needs:

| Mode | Latency | Accuracy | Use Case |
|------|---------|----------|----------|
| **direct** | ~150ms | Good | Simple factual queries |
| **verify** | ~600ms | Better | Claims needing verification |
| **research** | ~900ms | Best | Open-ended research |
| **plan_execute** | ~750ms | Best | Multi-step tasks |

```python
result = await rag.query("Complex question", mode="research")
```

#### 3. Query Caching

Repeated queries use cache (100x speedup):
```python
result1 = await rag.query("What is Thompson Sampling?")  # ~150ms (cold)
result2 = await rag.query("What is Thompson Sampling?")  # ~1ms (cached)
```

Disable caching:
```python
rag = SimpleRAG(enable_caching=False)
```

#### 4. Multimodal Capabilities

**Visual Q&A**: Ask questions about images
```python
result = await rag.query_with_image(
    "Explain this architecture diagram",
    "architecture.png"
)
```

**Photo Retrieval**: Find similar images using CLIP
```python
# Text-based search
photos = await rag.get_related_photos("architecture diagram", max_photos=5)

# Image-based similarity
similar = await rag.get_similar_photos("reference.png", max_photos=5)
```

**Visual Compression**: Automatic token savings for large contexts
```python
rag = MultimodalRAG(
    enable_visual_compression=True,
    compression_threshold=10  # Compress if sources > 10
)

result = await rag.query_with_image(question, image)
if result.compressed_context:
    print(f"Compression: {result.compression_ratio:.1f}x token savings")
```

#### 5. Performance Dashboard

Auto-constructed visualizations following Edward Tufte principles:

```python
from HoloLoom.visualization import RAGDashboard

# Collect query results
results = []
for question in questions:
    result = await rag.query(question)
    results.append(result)

# Build dashboard (one line)
dashboard = RAGDashboard.from_query_history(results)
dashboard.save("performance.html")
```

**5 Panels**:
1. **Retrieval Quality**: Sources per query with trend analysis
2. **Latency Waterfall**: Stage timing breakdown (retrieval/generation/total)
3. **Cache Effectiveness**: Hit rate gauge with recommendations
4. **Confidence Trajectory**: Confidence over time with anomaly detection
5. **Source Attribution**: Source frequency distribution

**Anomaly Detection**: Automatically detects 4 types:
- Sudden drops (confidence drops >0.2 in single step)
- Prolonged low (confidence <threshold for 3+ consecutive queries)
- High variance (std dev >0.15 in rolling window)
- Cache miss clusters (3+ cache misses in rolling window)

### Files & Documentation

**Core Implementation**:
- `HoloLoom/rag/simple_rag.py` (375 lines) - Main RAG wrapper
- `HoloLoom/rag/multimodal_rag.py` (675 lines) - Multimodal extension
- `HoloLoom/rag/visual_qa.py` (432 lines) - OCR + CLIP engine

**Tests**:
- `HoloLoom/rag/tests/test_simple_rag.py` (425 lines) - 24/25 tests passing
- `HoloLoom/rag/tests/test_multimodal_rag.py` (581 lines) - 21/21 tests passing

**Visualization**:
- `HoloLoom/visualization/rag_dashboard.py` (612 lines) - Dashboard builder

**Demos** (Progressive Complexity):
- `demos/demo_rag_qa_simple.py` (92 lines) - **Start here!** Basic Q&A
- `demos/demo_rag_document_ingestion.py` (136 lines) - Batch ingestion
- `demos/demo_rag_multiquery.py` (146 lines) - Multi-query research
- `demos/demo_rag_with_verification.py` (159 lines) - Reasoning modes
- `demos/demo_multimodal_rag.py` (237 lines) - Text + images
- `demos/demo_rag_dashboard.py` (175 lines) - Performance dashboard

**Documentation**:
- `HoloLoom/rag/README.md` (596 lines) - Simple RAG API reference
- `HoloLoom/rag/MULTIMODAL_README.md` (877 lines) - Multimodal capabilities
- `HoloLoom/visualization/RAG_DASHBOARD_README.md` (576 lines) - Dashboard guide
- `demos/RAG_DEMOS_README.md` (455 lines) - Learning path

**Total**: 11,418 lines of production code, tests, and documentation

### Performance Characteristics

| Operation | Latency | Notes |
|-----------|---------|-------|
| **Text ingestion** | ~50ms | Any modality via SpinningWheel |
| **Text-only query (cold)** | ~150ms | FAST mode, no cache |
| **Text-only query (warm)** | <1ms | Cache hit (100x faster) |
| **Visual Q&A (DeepSeek OCR)** | ~1,110ms | +260ms for OCR + CLIP |
| **Visual Q&A (basic OCR)** | ~960ms | +110ms for pytesseract |
| **Visual compression** | +150ms | Saves 5-20x tokens |
| **Photo ingestion** | ~200ms | CLIP encoding |
| **Photo retrieval** | ~50ms | CLIP similarity |
| **Dashboard generation** | ~30ms | For 15 queries |

### Running the Demos

```bash
# Basic RAG (Start here!)
PYTHONPATH=. python demos/demo_rag_qa_simple.py

# Advanced demos
PYTHONPATH=. python demos/demo_rag_document_ingestion.py
PYTHONPATH=. python demos/demo_rag_multiquery.py
PYTHONPATH=. python demos/demo_rag_with_verification.py

# Multimodal (requires PIL, torch)
PYTHONPATH=. python demos/demo_multimodal_rag.py

# Performance dashboard
PYTHONPATH=. python demos/demo_rag_dashboard.py
# Output: demos/output/rag_dashboard.html
```

### Testing

```bash
# Test Simple RAG
pytest HoloLoom/rag/tests/test_simple_rag.py -v
# Result: 24/25 passing (96%)

# Test Multimodal RAG
pytest HoloLoom/rag/tests/test_multimodal_rag.py -v
# Result: 21/21 passing (100%)

# Test all RAG components
pytest HoloLoom/rag/ -v
```

### Integration with HoloLoom

RAG leverages HoloLoom's existing infrastructure:

**Memory Systems**:
- `hololoom.py` - experience(), recall(), reflect() API
- `memory/cache.py` - BM25 + semantic retrieval
- `memory/graph.py` - Yarn Graph (entity relationships)
- `memory/photo_tokens.py` - CLIP embeddings for images
- `memory/visual_compression.py` - Graph→image compression

**LLM Integration**:
- `weaving_orchestrator_llm.py` - Ollama/Anthropic/OpenAI integration
- Graceful fallback to neural-only if LLM unavailable

**Agentic Reasoning**:
- `agentic/core.py` - Multi-query reasoning modes
- `recursive/` - Self-improving learning loop

**Visualization**:
- `visualization/confidence_trajectory.py` - Confidence tracking
- `visualization/cache_gauge.py` - Cache performance
- `visualization/stage_waterfall.py` - Latency breakdown
- `visualization/knowledge_graph.py` - Entity relationships

### Comparison to Other RAG Systems

| Feature | Basic RAG | LangChain | LlamaIndex | **HoloLoom RAG** |
|---------|-----------|-----------|------------|------------------|
| **Level** | 1-2 | 2-3 | 2-3 | **4 (Agentic + Graph)** |
| **Hybrid Search** | ❌ | ✅ | ✅ | ✅ |
| **Graph RAG** | ❌ | 🟡 | 🟡 | ✅ (Native) |
| **Multimodal** | ❌ | 🟡 | 🟡 | ✅ (Full) |
| **Agentic Reasoning** | ❌ | 🟡 | ❌ | ✅ (4 modes) |
| **Visual Compression** | ❌ | ❌ | ❌ | ✅ (5-20x) |
| **Zero-Config** | ❌ | ❌ | ❌ | ✅ |
| **Performance Dashboard** | ❌ | ❌ | ❌ | ✅ |
| **Setup Complexity** | Low | High | Medium | **Zero** |

### When to Use HoloLoom RAG

**✅ Use HoloLoom RAG when you need**:
- Zero-config RAG with sane defaults
- Graph-based knowledge representation (entity relationships)
- Multimodal RAG (text + images)
- Agentic reasoning (multi-step, verification, research modes)
- Visual compression for large contexts (5-20x token savings)
- Performance monitoring out of the box
- Integration with HoloLoom's memory and learning systems

**🟡 Consider alternatives when**:
- You need a specific vector DB (Pinecone, Weaviate, etc.) - HoloLoom uses Qdrant/Neo4j
- You have very specific chunking requirements - HoloLoom delegates to SpinningWheel
- You need SQL database integration - HoloLoom focuses on knowledge graphs

**❌ Don't use RAG (including HoloLoom) when**:
- Base model already knows the information (check first!)
- Data is highly volatile (stock tickers, real-time feeds)
- You need sub-50ms latency (gaming, real-time systems)
- Dataset is tiny (<100 documents) - not worth the overhead
- Privacy-critical and can't store user data

### Future Enhancements

Roadmap for HoloLoom RAG (Phase 6+):

1. **SQL/Database Integration** - Query structured databases
2. **Multi-Hop Reasoning** - Follow relationship chains in Yarn Graph
3. **Streaming Responses** - Stream LLM generation token-by-token
4. **Custom Embeddings** - Plug in custom embedding models
5. **Advanced Reranking** - Cross-encoder reranking for higher precision
6. **Multi-Agent RAG** - Parallel query execution with consensus
7. **Fine-Tuning Integration** - Combine RAG with fine-tuned models

See [HOLOLOOM_MASTER_SCOPE_AND_SEQUENCE.md](HOLOLOOM_MASTER_SCOPE_AND_SEQUENCE.md) for complete roadmap.

---

## Learning Systems: 7 Parallel Learning Loops

**Status**: ✅ All 7 Systems Active (November 2025)
**Total Code**: ~8,500 lines across all learning systems

HoloLoom implements **7 independent learning systems** that operate at different timescales, from per-query adaptation to offline training:

### 1. Policy Engine (Per-Query Learning)

**Location**: `HoloLoom/policy/unified.py` (1,247 lines)
**Timescale**: Per-query (<1ms overhead)
**What it learns**: Tool selection patterns, bandit priors

**Mechanism**:
- Thompson Sampling updates α/β priors based on confidence
- Epsilon-greedy exploration (10% by default)
- Bayesian blend combines neural predictions with bandit priors
- Success: α ← α + confidence, Failure: β ← β + (1 - confidence)

**Usage**:
```python
policy = create_policy(bandit_strategy=BanditStrategy.BAYESIAN_BLEND)
action = policy.forward(features, context)
# Automatic bandit update on next call
```

### 2. Reflection Buffer (5-Minute Cycles)

**Location**: `HoloLoom/reflection/buffer.py` (487 lines)
**Timescale**: 5-minute windows
**What it learns**: Query-response quality patterns, temporal trends

**Mechanism**:
- Stores episodic buffer of recent interactions
- Analyzes temporal patterns (morning vs. evening performance)
- Detects quality degradation over time
- Provides signals for system evolution

**Usage**:
```python
async with ReflectionBuffer(capacity=1000) as buffer:
    await buffer.store(spacetime, feedback={"helpful": True})
    patterns = buffer.analyze_patterns(window=300)  # 5 minutes
```

### 3. Recursive Learning (60-Second Background Loop)

**Location**: `HoloLoom/recursive/` (750 lines)
**Timescale**: 60-second background updates
**What it learns**: Hot patterns, refinement strategies, Thompson priors

**Mechanism**:
- Background thread runs every 60 seconds
- Mines high-quality patterns from logs
- Updates Thompson Sampling priors (α/β)
- Adjusts policy adapter weights based on outcomes

**Usage**:
```python
async with FullLearningEngine(enable_background_learning=True) as engine:
    spacetime = await engine.weave(query)
    # System learns automatically in background
```

### 4. Semantic Calculus (Per-Query Projection)

**Location**: `HoloLoom/semantic_calculus/` (1,850 lines)
**Timescale**: Per-query
**What it learns**: 228D semantic space projections (16 interpretable axes)

**16 Interpretable Axes**:
- Sentiment (negative ↔ positive)
- Formality (casual ↔ formal)
- Technicality (general ↔ technical)
- Certainty (uncertain ↔ certain)
- Urgency (relaxed ↔ urgent)
- Abstraction (concrete ↔ abstract)
- Specificity (vague ↔ specific)
- Temporality (timeless ↔ time-bound)
- Objectivity (subjective ↔ objective)
- Complexity (simple ↔ complex)
- Scope (narrow ↔ broad)
- Directness (indirect ↔ direct)
- Emotionality (neutral ↔ emotional)
- Actionability (informational ↔ actionable)
- Novelty (familiar ↔ novel)
- Controversy (consensus ↔ controversial)

**Mechanism**:
- Projects queries into 228D semantic space
- First 16 dimensions are human-interpretable
- Remaining 212 dimensions capture nuanced semantics
- Enables semantic similarity, clustering, navigation

### 5. Adaptive Query Routing (Hourly Pattern Mining)

**Location**: `HoloLoom/routing/learning/` (2,683 lines)
**Timescale**: Hourly validation + daily reports
**What it learns**: Query complexity patterns, routing rules

**Mechanism**:
- Mines patterns from classification logs (n-gram → regex)
- Validates accuracy hourly (regression detection >2% drop)
- Safe deployment strategies (SHADOW → AB_TEST → GRADUAL)
- Automatic rollback on regression

**Usage**:
```python
classifier = AdaptiveMoonshotClassifier(
    enable_adaptive_learning=True,
    background_learning=True,
    learning_update_interval=3600.0  # 1 hour
)
```

### 6. PPO Training (Offline RL)

**Location**: `HoloLoom/train_agent.py` + `reflection/ppo_trainer.py` (892 lines)
**Timescale**: Offline training (hours/days)
**What it learns**: Policy network weights, value function

**Mechanism**:
- GAE (Generalized Advantage Estimation)
- Optional ICM/RND curiosity modules
- Checkpoint saving/loading
- Configurable network architectures

**Usage**:
```bash
PYTHONPATH=. python -c "from HoloLoom.train_agent import PPOTrainer; \
t=PPOTrainer(env_name='CartPole-v1', total_timesteps=50000); t.train()"
```

### 7. Hot Pattern Feedback (10-Query Windows)

**Location**: `HoloLoom/recursive/hot_pattern_feedback.py` (780 lines)
**Timescale**: 10-query rolling windows
**What it learns**: Access patterns, retrieval weights

**Mechanism**:
- Tracks access frequency of knowledge elements
- Hot patterns get 2x boost, cold patterns get 0.5x penalty
- Exponential decay (5% per hour)
- Heat score = access_count × success_rate × avg_confidence × decay

**Usage**:
```python
async with HotPatternFeedbackEngine(cfg=config) as engine:
    spacetime = await engine.weave(query)
    hot = engine.hot_tracker.get_hot_patterns(limit=10)
```

### Learning Systems Comparison

| System | Timescale | What Learned | Overhead | Lines |
|--------|-----------|--------------|----------|-------|
| **Policy Engine** | Per-query | Tool selection | <1ms | 1,247 |
| **Reflection Buffer** | 5-min | Quality patterns | <1ms | 487 |
| **Recursive Learning** | 60-sec | Hot patterns, priors | ~50ms (async) | 750 |
| **Semantic Calculus** | Per-query | 228D projections | <2ms | 1,850 |
| **Adaptive Routing** | Hourly | Complexity patterns | <1ms | 2,683 |
| **PPO Training** | Offline | Policy weights | N/A (offline) | 892 |
| **Hot Patterns** | 10-query | Retrieval weights | <1ms | 780 |

**Total**: 7 learning systems, ~8,689 lines, <6ms total per-query overhead

### Multi-Timescale Learning Philosophy

HoloLoom's learning architecture operates across 6 orders of magnitude in time:

```
Per-Query (1-10ms):     Policy Engine, Semantic Calculus, Hot Patterns
Short-Term (5-10 min):  Reflection Buffer
Medium-Term (1 hour):   Recursive Learning, Adaptive Routing
Long-Term (offline):    PPO Training
```

This multi-timescale approach enables:
- **Fast adaptation** to immediate context (per-query)
- **Pattern recognition** over recent interactions (minutes)
- **Trend detection** over longer sessions (hours)
- **Deep learning** for fundamental improvements (offline)

---

## Phase 3: Adaptive Learning System (November 2025)

**Status**: ✅ Production Ready
**Location**: `HoloLoom/routing/learning/`
**Documentation**: [PHASE_3_DOCUMENTATION.md](PHASE_3_DOCUMENTATION.md)

Phase 3 integrates a complete **Adaptive Learning System** into HoloLoom's query routing layer, enabling automatic pattern discovery, continuous accuracy monitoring, and safe pattern deployment.

### Overview

The Adaptive Learning System provides:
- **Automatic pattern discovery** from production classification logs
- **Continuous accuracy monitoring** with hourly validation
- **Safe pattern deployment** with A/B testing and automatic rollback
- **Daily/weekly performance reports** with Prometheus metrics
- **Slack/email alerts** for critical regressions
- **Sub-millisecond overhead** (<1ms per query)

### Quick Start

```python
from HoloLoom.routing.query_classifier_adaptive import AdaptiveMoonshotClassifier

# Create adaptive classifier with background learning
classifier = AdaptiveMoonshotClassifier(
    enable_adaptive_learning=True,
    background_learning=True,  # Automatic hourly learning
    learning_update_interval=3600.0  # 1 hour
)

# Classify queries (automatically logged for learning)
result = classifier.classify("What is Thompson Sampling?")

print(f"Complexity: {result.complexity.value}")
print(f"Confidence: {result.confidence:.1%}")

# View learning statistics
stats = classifier.get_learning_statistics()
print(f"Patterns Discovered: {stats['patterns_discovered']}")
print(f"Validation Accuracy: {stats['validation_accuracy']:.1%}")
```

### Architecture

Phase 3 implements a 4-component architecture:

1. **PatternMiner** ([pattern_miner.py](HoloLoom/routing/learning/pattern_miner.py:1)) - 425 lines
   - Extracts high-quality patterns from production logs (n-gram → regex)
   - Quality scoring: precision, recall, F1, support
   - Configurable thresholds (min precision: 95%, min support: 10)

2. **ContinuousValidator** ([continuous_validator.py](HoloLoom/routing/learning/continuous_validator.py:1)) - 469 lines
   - Hourly/daily validation with regression detection (>2% drop)
   - Trend analysis (7-day, 30-day moving averages)
   - Alert generation with severity levels (WARNING, CRITICAL)

3. **AdaptiveUpdater** ([adaptive_updater.py](HoloLoom/routing/learning/adaptive_updater.py:1)) - 682 lines
   - Safe deployment strategies (SHADOW, AB_TEST, GRADUAL)
   - Automatic rollback on regression
   - Pattern versioning (keeps last 10 versions)

4. **PerformanceReporter** ([performance_reporter.py](HoloLoom/routing/learning/performance_reporter.py:1)) - 627 lines
   - Daily/weekly reports with recommendations
   - Prometheus metrics export
   - Slack/email alert formatting

### Background Learning Loop

The system runs a background learning cycle every hour:

```python
async def main():
    classifier = AdaptiveMoonshotClassifier(background_learning=True)

    # Start background learning
    await classifier.start_background_learning()

    # Your application runs here...
    # System automatically:
    # 1. Mines patterns from logs
    # 2. Validates accuracy hourly
    # 3. Deploys high-quality patterns
    # 4. Generates daily/weekly reports

    # Graceful shutdown
    await classifier.stop_background_learning()

asyncio.run(main())
```

### Deployment Strategies

| Strategy | Traffic Split | Duration | Use Case |
|----------|---------------|----------|----------|
| **SHADOW** | 0% | Day 1-2 | Test patterns without production impact |
| **AB_TEST** | 10/90 | Day 3 | Validate with small traffic |
| **GRADUAL** | 10%→50%→100% | Day 3-7 | Incremental deployment with monitoring |

### Production Integration

**Prometheus Metrics** (exported every minute):
```
moonshot_accuracy{complexity="overall"} 0.95
moonshot_queries_total 15234
moonshot_latency_ms 125.5
moonshot_patterns_deployed 42
moonshot_regressions_detected 3
```

**Slack Alerts** (on critical regression):
```
🚨 Classifier Regression Detected

Current accuracy: 85.3%
Baseline accuracy: 95.0%
Drop: 9.7% (threshold: 2.0%)

Affected: complex(75.0%), research(80.0%)
Severity: CRITICAL
```

### Testing

Run comprehensive integration tests:

```bash
pytest HoloLoom/routing/learning/tests/test_adaptive_integration.py -v
```

**Test Coverage**: 13/13 integration tests passing
- PatternMiner initialization and pattern discovery
- ContinuousValidator hourly validation and regression detection
- AdaptiveUpdater safe deployment and automatic rollback
- PerformanceReporter daily/weekly reports and metrics export
- End-to-end pipeline integration

### Performance Characteristics

| Operation | Overhead | Frequency |
|-----------|----------|-----------|
| Classification + logging | <1ms | Every query |
| Pattern mining | ~500ms | Every hour (async) |
| Hourly validation | ~2-5s | Every hour (async) |
| Pattern deployment | ~100ms | When patterns available (async) |
| Daily report | ~50ms | Once per day (async) |

**Total Per-Query Overhead**: <1ms (JSONL logging only)
**Background Learning**: ~3-6s per hour (0.08-0.17% CPU)
**Memory Usage**: ~1-2MB typical production workload

### Key Files

| File | Lines | Purpose |
|------|-------|---------|
| [query_classifier_adaptive.py](HoloLoom/routing/learning/query_classifier_adaptive.py:1) | 480 | Main adaptive classifier integration |
| [pattern_miner.py](HoloLoom/routing/learning/pattern_miner.py:1) | 425 | Automatic pattern discovery |
| [continuous_validator.py](HoloLoom/routing/learning/continuous_validator.py:1) | 469 | Hourly accuracy monitoring |
| [adaptive_updater.py](HoloLoom/routing/learning/adaptive_updater.py:1) | 682 | Safe pattern deployment |
| [performance_reporter.py](HoloLoom/routing/learning/performance_reporter.py:1) | 627 | Daily/weekly reports + metrics |
| [test_adaptive_integration.py](HoloLoom/routing/learning/tests/test_adaptive_integration.py:1) | 465 | Integration test suite (13 tests) |

### Documentation

- **Complete Guide**: [PHASE_3_DOCUMENTATION.md](PHASE_3_DOCUMENTATION.md:1) (1000+ lines)
  - Quick start and usage examples
  - Configuration reference
  - Production deployment (Prometheus, Grafana, Slack, email)
  - Monitoring and alerting setup
  - Performance characteristics
  - Best practices and troubleshooting

- **Progress Report**: [PHASE_3_PROGRESS.md](PHASE_3_PROGRESS.md:1)
  - 14-day implementation timeline
  - Component-by-component progress
  - Test coverage and statistics

### Demos

```bash
# Individual component demos
python demos/demo_pattern_miner.py
python demos/demo_continuous_validator.py
python demos/demo_adaptive_updater.py
python demos/demo_performance_reporter.py

# Full integration demo
python demos/demo_adaptive_classifier.py
```

---

## Development Commands

### Environment Setup

Create and activate virtualenv:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch numpy gymnasium matplotlib
```

Optional dependencies for full features:
```bash
pip install spacy sentence-transformers scipy networkx ollama
python -m spacy download en_core_web_sm
```

### Testing

**Test Organization** (Phase 1+2 Cleanup - Oct 2025):
Tests are organized into three tiers for fast feedback loops:

```bash
# Unit Tests (Fast - <5s) - Isolated component testing
pytest HoloLoom/tests/unit/ -v

# Integration Tests (Medium - <30s) - Multi-component testing
pytest HoloLoom/tests/integration/ -v

# End-to-End Tests (Slow - <2min) - Full pipeline testing
pytest HoloLoom/tests/e2e/ -v

# Run all tests
pytest HoloLoom/tests/ -v
```

**Key Tests:**
- `tests/unit/test_unified_policy.py` - Neural components (MLP, attention, ICM/RND, PPO)
- `tests/integration/test_backends.py` - Memory backend integration
- `tests/e2e/test_full_pipeline.py` - Complete weaving cycle (BARE/FAST/FUSED modes)

**Memory Backend Validation:**
```bash
python test_memory_backend_simplification.py
```
Validates 3-backend architecture (INMEMORY/HYBRID/HYPERSPACE) and auto-fallback.

### Training Example

Run a short CartPole training session:
```bash
PYTHONPATH=. .venv/bin/python -c "from HoloLoom.train_agent import PPOTrainer; t=PPOTrainer(env_name='CartPole-v1', total_timesteps=2000, steps_per_update=256, n_epochs=1, batch_size=32, log_dir='./logs/test_run_small'); t.train()"
```

Training checkpoints are saved to the specified `log_dir`.

### Running the Orchestrator

Example usage of the full HoloLoom orchestrator:
```bash
PYTHONPATH=. .venv/bin/python HoloLoom/orchestrator.py
```

This runs a demo showing query → features → context → decision → response pipeline.

## Architecture

### Core Design Philosophy

**"Warp Thread" Modules**: Each major component (motif detection, embedding, memory, policy) is independent and protocol-based. They don't import from each other, only from shared types (`HoloLoom/documentation/types.py`).

**"Shuttle" Orchestrator**: The `orchestrator.py` is the only module that imports from all others. It weaves components together into the full processing pipeline.

### Weaving Architecture

HoloLoom implements a complete weaving metaphor as first-class abstractions:

#### 1. Yarn Graph (HoloLoom/memory/graph.py)
The persistent symbolic memory - discrete thread structure stored as a NetworkX MultiDiGraph.
- **Alias**: `YarnGraph = KG`
- Entities and relationships form the "threads" of memory
- Remains discrete until "tensioned" into Warp Space

#### 2. Loom Command (HoloLoom/loom/command.py)
Pattern card selector that chooses execution template (BARE/FAST/FUSED).
- **Classes**: `LoomCommand`, `PatternCard`, `PatternSpec`
- Determines which warp threads to lift and how densely to weave
- Configures scales, features, timeouts for entire cycle

#### 3. Chrono Trigger (HoloLoom/chrono/trigger.py)
Temporal control system managing time-dependent aspects.
- **Classes**: `ChronoTrigger`, `TemporalWindow`, `ExecutionLimits`
- Controls when threads activate (temporal windows)
- Manages execution timing, rhythm (heartbeat), halt conditions
- Handles thread decay and system evolution over time

#### 4. Resonance Shed (HoloLoom/resonance/shed.py)
Feature interference zone where extraction threads combine.
- **Classes**: `ResonanceShed`, `FeatureThread`
- Lifts feature threads (motif, embedding, spectral)
- Creates interference patterns through multi-modal fusion
- Produces DotPlasma (flowing feature representation)

#### 5. DotPlasma (HoloLoom/documentation/types.py)
The "feature fluid" - flowing continuous representation.
- **Alias**: `DotPlasma = Features`
- Malleable medium between extraction and decision
- Contains motifs (symbolic), embeddings (continuous), spectral (topological)

#### 6. Warp Space (HoloLoom/warp/space.py)
Tensioned tensor field for continuous mathematics.
- **Classes**: `WarpSpace`, `TensionedThread`
- Temporary manifold where activated threads undergo tensor operations
- Lifecycle: tension() → compute() → collapse()
- Detensions back to discrete Yarn Graph after computation

#### 7. Convergence Engine (HoloLoom/convergence/engine.py)
Decision collapse from continuous → discrete.
- **Classes**: `ConvergenceEngine`, `CollapseStrategy`, `ThompsonBandit`
- Collapses probability distributions to discrete tool selections
- Strategies: ARGMAX, EPSILON_GREEDY, BAYESIAN_BLEND, PURE_THOMPSON
- Thompson Sampling for exploration/exploitation balance

#### 8. Spacetime (HoloLoom/fabric/spacetime.py)
Woven fabric - structured output with complete lineage.
- **Classes**: `Spacetime`, `WeavingTrace`, `FabricCollection`
- 4-dimensional output: 3D semantic space + 1D temporal trace
- Full computational provenance for debugging and reflection learning
- Serializable for persistence and analysis

#### 9. Reflection Buffer (HoloLoom/memory/cache.py)
Learning loop - stores outcomes for improvement.
- **Alias**: `ReflectionBuffer = MemoryManager`
- Episodic buffer of recent interactions
- Provides signals for system evolution and adaptation

#### Complete Weaving Cycle

```
1. Loom Command selects Pattern Card (BARE/FAST/FUSED)
2. Chrono Trigger fires, creates TemporalWindow
3. Yarn Graph threads selected based on temporal window
4. Resonance Shed lifts feature threads, creates DotPlasma
5. Warp Space tensions threads into continuous manifold
6. Convergence Engine collapses to discrete tool selection
7. Tool executes, results woven into Spacetime fabric
8. Reflection Buffer learns from outcome
9. Chrono Trigger detensions, cycle completes
```

This architecture enables:
- **Symbolic ↔ Continuous**: Seamless transition between discrete and continuous representations
- **Temporal Control**: Fine-grained timing and decay mechanisms
- **Multi-Modal Fusion**: Interference patterns from diverse feature types
- **Provenance**: Complete computational lineage for every output
- **Evolution**: System learns and adapts from reflection

### Key Components

#### 1. Weaving Orchestrator (`HoloLoom/weaving_orchestrator.py`)

**UPDATED (Task 1.2 - Oct 27, 2025):** The Shuttle architecture has been integrated into the canonical `WeavingOrchestrator`.

The WeavingOrchestrator implements the full 9-step weaving cycle with mythRL protocol-based architecture:

1. **Loom Command** → Pattern Card selection (BARE/FAST/FUSED)
2. **Chrono Trigger** → Temporal window creation
3. **Yarn Graph** → Thread selection from memory
4. **Resonance Shed** → Feature extraction, DotPlasma creation
5. **Warp Space** → Continuous manifold tensioning
6. **Convergence Engine** → Discrete decision collapse
7. **Tool Execution** → Action with results
8. **Spacetime Fabric** → Provenance and trace
9. **Reflection Buffer** → Learning from outcome

**mythRL Progressive Complexity (3-5-7-9 System):**
- **LITE (3 steps)**: Extract → Route → Execute (<50ms) - simple queries
- **FAST (5 steps)**: + Pattern Selection + Temporal Windows (<150ms) - standard queries
- **FULL (7 steps)**: + Decision Engine + Synthesis Bridge (<300ms) - complex queries
- **RESEARCH (9 steps)**: + Advanced WarpSpace + Full Tracing (no limit) - research mode

**Protocol-Based Design:**
- `PatternSelectionProtocol`: Processing pattern selection
- `FeatureExtractionProtocol`: Multi-scale Matryoshka extraction
- `WarpSpaceProtocol`: Mathematical manifold operations
- `DecisionEngineProtocol`: Strategic multi-criteria optimization

**Key Features:**
- Auto-complexity detection based on query characteristics
- Performance caching (QueryCache) for repeated queries
- Reflection loop for continuous improvement
- Lifecycle management with async context managers
- Backward compatibility: `WeavingShuttle` is an alias to `WeavingOrchestrator`

Usage:
```python
from HoloLoom.weaving_orchestrator import WeavingOrchestrator
from HoloLoom.config import Config
from HoloLoom.Documentation.types import Query

config = Config.fused()
async with WeavingOrchestrator(cfg=config, shards=shards) as orchestrator:
    spacetime = await orchestrator.weave(Query(text="What is Thompson Sampling?"))
    # Automatic cleanup on exit
```

#### 2. Policy Engine (`HoloLoom/policy/unified.py`)
Neural decision-making with three bandit exploration strategies:
- **Epsilon-Greedy** (default): 90% neural exploitation, 10% Thompson Sampling exploration
- **Bayesian Blend**: Combines neural predictions (70%) with bandit priors (30%)
- **Pure Thompson**: Uses only Thompson Sampling (ignores neural network)

The policy uses:
- Transformer blocks with cross-attention to context memory
- Motif-gated multi-head attention
- LoRA-style adapters for different execution modes (bare/fast/fused)
- Thompson Sampling bandit for exploration/exploitation balance

Key fix from code review: Bandit now updates statistics for the **actually selected tool** (previously disconnected).

#### 3. Configuration (`HoloLoom/config.py`)
Three execution modes:
- **BARE**: Minimal processing (regex motifs, single scale, simple policy) - fastest
- **FAST**: Balanced (hybrid motifs, 2 scales, neural policy) - good tradeoff
- **FUSED**: Full processing (all features, 3 scales, multi-scale retrieval) - highest quality

Access via factory methods:
```python
from HoloLoom.config import Config
cfg_fast = Config.fast()
cfg_fused = Config.fused()
```

#### 4. Memory Systems

HoloLoom integrates **11 specialized memory systems** working in concert:

**Core Memory** (3 systems):
1. **Vector Memory** (`memory/cache.py`) - BM25 + semantic similarity retrieval
2. **Knowledge Graph** (`memory/graph.py`) - NetworkX-based entity relationships with typed edges
3. **Yarn Graph** (`memory/graph.py`) - Persistent symbolic memory (alias for KG)

**Dynamic Memory** (4 systems):
4. **Awareness Graph** (`memory/awareness_graph.py`) - Activation tracking and spreading activation
5. **Spring Dynamics** (`memory/spring_dynamics.py`) - Physics-based memory connectivity
6. **Multi-Wave Engine** (`memory/multi_wave_engine.py`) - Temporal wave propagation
7. **Warp Space** (`warp/space.py`) - Tensioned tensor field for continuous mathematics

**Specialized Memory** (4 systems):
8. **Photo Memory** (`memory/photo_tokens.py`) - CLIP embeddings for images
9. **Visual Compression** (`memory/visual_compression.py`) - Graph→image compression (5-20x token savings)
10. **Query Cache** (`memory/query_cache.py`) - 100x speedup for repeated queries
11. **Reflection Buffer** (`reflection/buffer.py`) - Episodic buffer for learning

**Key Features**:
- Typed edges (IS_A, USES, MENTIONS, LEADS_TO, PART_OF, IN_TIME, OCCURRED_AT)
- Subgraph extraction for context expansion
- Path finding between entities
- Spectral graph features (Laplacian eigenvalues) for policy input
- Spreading activation across connected memories
- Physics-based connectivity modeling

#### 5. Embeddings (`HoloLoom/embedding/spectral.py`)
Matryoshka embeddings at multiple scales (96, 192, 384 dimensions) with:
- Multi-scale fusion for retrieval
- Spectral features: graph Laplacian eigenvalues, SVD topic components
- Optional sentence-transformers backend (degrades gracefully without it)

**Zero-Copy Embeddings** (`HoloLoom/embedding/zero_copy.py`) - **November 2025**

High-performance embedding layer using memory-mapped storage and view-based multi-scale access:

**Performance:**
- **37.7x faster** scale extraction (warm cache)
- **1.4x faster** in real orchestrator workloads
- **50% memory savings** (views share backing array)
- **<1ms** latency per query (warm cache)

**Enable via config:**
```python
config = Config.fast()  # Zero-copy enabled by default in FAST/FUSED
config.enable_zero_copy_embeddings = True
config.zero_copy_cache_path = '.cache/embeddings.mmap'
config.zero_copy_cache_size = 10000
```

**Key Innovation:** Leverages Matryoshka "prefix property" - the first k dimensions contain the k-d representation. This enables zero-copy array slicing instead of matrix multiplication.

**Documentation:** See [ZERO_COPY_ARCHITECTURE.md](HoloLoom/embedding/ZERO_COPY_ARCHITECTURE.md) for complete details.

**Trade-off:** ~2-5% retrieval quality loss (no learned projections), but 37x speedup in isolated embeddings and 50% memory savings make it worth it for latency-critical applications.

#### 6. SpinningWheel (`HoloLoom/spinningWheel/`)

**47 specialized input adapters** that convert raw data → `MemoryShard` objects across diverse modalities:

**Audio & Video** (8 adapters):
- Audio transcripts, task lists, summaries
- YouTube videos (with time-based chunking, timestamps, metadata)
- Video files, podcasts, voice memos, meeting recordings

**Web & Documents** (12 adapters):
- Web pages (HTML, Markdown, RSS feeds)
- PDFs, DOCX, PPTX, spreadsheets
- Jupyter notebooks, LaTeX documents
- API responses (JSON, XML)

**Code & Development** (15 adapters):
- Python, JavaScript, TypeScript, Go, Rust, Java, C++
- Git repositories, pull requests, code reviews
- Stack traces, logs, test outputs
- Package dependencies (package.json, requirements.txt)

**Structured Data** (7 adapters):
- Databases (SQL, MongoDB, Redis)
- CSV, JSON, YAML, TOML
- Configuration files

**Communication** (5 adapters):
- Email threads, Slack messages, Discord channels
- Calendar events, task lists (Trello, Jira, GitHub Issues)

**Key Features**:
- Optional Ollama enrichment for entity/motif extraction
- Standardized `MemoryShard` output format
- Automatic format detection and graceful degradation
- Chunking strategies for large inputs
- Metadata preservation (timestamps, sources, context)

**Total**: ~5,200 lines across 47 specialized adapters

See `HoloLoom/spinningWheel/README.md` for complete adapter reference.

#### 7. Training (`HoloLoom/train_agent`)
PPO trainer for RL environments with:
- GAE (Generalized Advantage Estimation)
- Optional ICM/RND curiosity modules
- Checkpoint saving/loading
- Configurable network architectures

### Module Structure (Phase 1+2 Cleanup - Oct 2025)

**Root Directory** (8 Python files, ~4,665 lines):
```
HoloLoom/
├── __init__.py                  # Package entry point (72 lines)
├── config.py                    # Configuration (BARE/FAST/FUSED modes) (460 lines)
├── hololoom.py                  # Unified memory system API (471 lines)
├── terminal_ui.py               # Interactive terminal interface (751 lines)
├── unified_api.py               # Programmatic API (729 lines)
├── weaving_orchestrator.py      # MAIN: Full 9-step weaving cycle (3,476 lines)
├── weaving_orchestrator_llm.py  # LLM-integrated variant (173 lines)
└── weaving_shuttle.py           # DEPRECATED: Backward compatibility shim (46 lines)
```

**Root File Documentation:**

### hololoom.py - Unified Memory System API (471 lines)

**Purpose**: Main entry point for the entire HoloLoom system. Provides a simplified "10/10 Layer" API where everything is a memory operation.

**Core Philosophy**:
- Single entry point (`HoloLoom` class)
- Single representation (`Memory`)
- Three core operations: `experience()`, `recall()`, `reflect()`
- Implementation details hidden
- Modality-agnostic

**Key Methods**:
- `experience(content)` - Form memories from any input (text, audio, etc.)
- `recall(query)` - Retrieve relevant memories based on query
- `reflect(memories, feedback)` - Learn from feedback to improve future recalls
- `experience_batch(contents)` - Batch experience multiple items
- `search(query, k)` - Search memories with limit
- `get_metrics()` - Get awareness graph metrics (activation, coherence, temporal)
- `summary()` - Human-readable system summary

**Architecture**:
- Integrates `AwarenessGraph` for memory activation tracking
- Uses `MatryoshkaSemanticCalculus` for 228D semantic projection (16 interpretable axes)
- Supports multimodal input via `InputRouter` (graceful degradation if unavailable)
- Async context manager support for proper resource cleanup

**Usage Example**:
```python
from HoloLoom import HoloLoom

async with HoloLoom() as loom:
    # Experience (form memories)
    mem = await loom.experience("Thompson Sampling balances exploration")

    # Recall (retrieve memories)
    memories = await loom.recall("What did I learn about sampling?")

    # Reflect (learn from feedback)
    await loom.reflect(memories, feedback={"helpful": True})

    # Get metrics
    metrics = loom.get_metrics()
    print(f"Active memories: {metrics['activation']['active_nodes']}")
```

**Integration Points**:
- Uses `Config` for system configuration
- Backs onto `AwarenessGraph` for memory graph management
- Wraps `MatryoshkaEmbeddings` for semantic encoding
- Optional `InputRouter` for multimodal support

**When to Use**:
- Simple API for memory operations
- Don't need full weaving cycle control
- Want automatic awareness graph management
- Building higher-level applications

---

### terminal_ui.py - Interactive Terminal Interface (751 lines)

**Purpose**: Provides an interactive terminal UI for exploring and interacting with HoloLoom's memory system.

**Features**:
- Interactive command-line interface with rich formatting
- Real-time memory visualization
- Graph exploration and traversal
- Awareness metrics monitoring
- Session management with history

**Key Components**:
- `TerminalUI` class - Main UI controller
- Rich terminal formatting (colors, tables, progress bars)
- Command parser and dispatcher
- Interactive memory browser
- Awareness graph visualizer

**Available Commands** (typical):
- `experience <text>` - Add new memory
- `recall <query>` - Search memories
- `metrics` - Show awareness graph metrics
- `graph` - Visualize memory graph
- `history` - Show session history
- `help` - Command reference
- `exit` - Quit interface

**Usage Example**:
```python
from HoloLoom.terminal_ui import TerminalUI

# Start interactive session
ui = TerminalUI()
await ui.run()

# Or programmatic use
ui = TerminalUI(config=Config.fused())
await ui.execute_command("experience Learning about Python decorators")
await ui.execute_command("recall What did I learn about Python?")
```

**Architecture**:
- Built on top of `HoloLoom` class (uses unified API)
- Rich terminal formatting library for visual appeal
- Async command processing
- Session state management
- Command history tracking

**When to Use**:
- Interactive exploration of memory system
- Debugging memory graph behavior
- Quick prototyping and testing
- Educational demonstrations
- Development and debugging

---

### weaving_orchestrator_llm.py - LLM-Integrated Variant (173 lines)

**Purpose**: Variant of `WeavingOrchestrator` that integrates LLM-based reasoning into the weaving cycle.

**Key Differences from Standard Orchestrator**:
- Injects LLM reasoning at decision points
- Can use LLM for feature extraction enhancement
- Supports LLM-based reflection and learning
- Hybrid neural + LLM decision making

**Integration Points**:
- Wraps standard `WeavingOrchestrator`
- Adds LLM client (OpenAI, Anthropic, local models)
- Optional LLM enhancement at each weaving stage
- Falls back to neural-only when LLM unavailable

**Key Methods**:
- `weave(query, use_llm)` - Main weaving with optional LLM enhancement
- `llm_enhance_features(features)` - Use LLM to enrich extracted features
- `llm_decide(features, context)` - LLM-based tool selection
- `llm_reflect(spacetime, feedback)` - LLM-assisted reflection

**Configuration**:
```python
from HoloLoom.weaving_orchestrator_llm import WeavingOrchestratorLLM

orchestrator = WeavingOrchestratorLLM(
    cfg=Config.fused(),
    llm_provider="openai",  # or "anthropic", "local"
    llm_model="gpt-4",
    fallback_to_neural=True  # Graceful degradation
)

spacetime = await orchestrator.weave(
    Query(text="Explain Thompson Sampling"),
    use_llm=True  # Enable LLM enhancement
)
```

**Use Cases**:
- Research experiments combining neural + LLM
- Enhanced reasoning for complex queries
- Explainable decision making (LLM generates explanations)
- Hybrid systems leveraging both approaches

**Architecture**:
- Extends/wraps `WeavingOrchestrator`
- LLM client abstraction (supports multiple providers)
- Configurable LLM injection points
- Graceful fallback to neural-only mode
- Cost tracking for LLM calls

**When to Use**:
- Need LLM-enhanced reasoning
- Want explainable decisions
- Research on neural-LLM hybrid systems
- Complex queries requiring deep reasoning

**Performance Considerations**:
- LLM calls add latency (~500ms - 3s per query)
- Cost per query (API fees for OpenAI/Anthropic)
- Consider caching LLM responses
- Use sparingly in production (enable for complex queries only)

---

**Organized Subdirectories:**
```
HoloLoom/
├── tests/                     # All tests (Phase 2)
│   ├── unit/                  # Fast isolated tests (<5s)
│   ├── integration/           # Multi-component tests (<30s)
│   └── e2e/                   # Full pipeline tests (<2min)
│
├── tools/                     # Developer utilities (Phase 1)
│   ├── bootstrap_system.py
│   ├── validate_pipeline.py
│   ├── visualize_bootstrap.py
│   └── archive/               # Archived dead code (safety net)
│
├── memory/                    # Storage backends (24 Python files)
│   ├── backend_factory.py    # Create backends (231 lines, was 550)
│   ├── graph.py              # NetworkX (default, always works)
│   ├── neo4j_graph.py        # Production backend
│   ├── hyperspace_backend.py # Research backend
│   ├── protocol.py           # Memory protocols (120 lines, was 787)
│   └── unified.py            # Unified interface
│
├── policy/                    # Decision making
│   ├── unified.py            # Neural core + Thompson Sampling
│   └── semantic_nudging.py   # Semantic goal guidance
│
├── protocols/                 # Protocol definitions (Phase 2)
│   ├── __init__.py           # Public exports
│   ├── core.py               # Core protocol definitions
│   └── types.py              # Shared data types
│
├── semantic_calculus/         # 228D semantic space (16 interpretable axes)
│   ├── dimensions.py         # EXTENDED_228_DIMENSIONS
│   ├── integrator.py         # SemanticSpectrum
│   ├── dimension_selector.py # Dimension selection
│   └── axes.py               # 16 interpretable axes (sentiment, formality, etc.)
│
├── reflection/                # Learning & improvement
│   ├── buffer.py             # ReflectionBuffer
│   ├── ppo_trainer.py        # PPO training
│   └── semantic_learning.py  # Multi-task learner (6 signals)
│
├── routing/                   # Query routing & adaptive learning (Phase 3 - Nov 2025)
│   └── learning/             # Adaptive learning system
│       ├── pattern_miner.py        # Automatic pattern discovery (425 lines)
│       ├── continuous_validator.py # Hourly accuracy monitoring (469 lines)
│       ├── adaptive_updater.py     # Safe pattern deployment (682 lines)
│       ├── performance_reporter.py # Daily/weekly reports (627 lines)
│       ├── query_classifier_adaptive.py # Main integration (480 lines)
│       └── tests/
│           └── test_adaptive_integration.py # 13 integration tests
│
├── embedding/                 # Multi-scale embeddings
│   ├── spectral.py           # Matryoshka + spectral features
│   └── matryoshka_interpreter.py  # (moved from root)
│
├── spinningWheel/             # Input adapters
│   ├── audio.py              # Audio/transcript processing
│   ├── youtube.py            # YouTube transcription
│   └── autospin.py           # (moved from root)
│
├── chatops/                   # Conversational features
│   ├── core/chatops_bridge.py
│   ├── conversational.py     # (moved from root)
│   └── ROADMAP.md            # ChatOps + Semantic Learning plan
│
├── agentic/                   # Agentic reasoning system (Nov 2025)
│   ├── core.py               # Multi-query reasoning engine
│   └── embedding_integrity.py # Embedding consistency checks
│
├── alignment/                 # Alignment framework v1.0 (Nov 2025)
│   ├── safety_guardrails.py  # Risk-based action gating
│   ├── deception_detection.py # Goal transparency
│   ├── instrumental_convergence.py # Power-seeking detection
│   ├── audit_trail.py        # Complete provenance
│   ├── monitoring.py         # Live monitoring
│   └── tests/                # 46 tests + 13 benchmarks
│
├── server/                    # FastAPI server (Nov 2025)
│   ├── agentic_api.py        # Main API server
│   └── agentic_api_integrated.py # Full integration
│
├── interpretability/          # Explainability (future)
│
└── [other feature dirs...]    # loom/, warp/, resonance/, etc.
```

**Repository Root:**
```
mythRL/
├── HoloLoom/                  # Main package
├── demos/                     # Demo scripts
├── tests/                     # Root-level integration tests
├── experiments/               # Automated experiment framework (Oct 2025)
│   ├── run_experiments.py    # Run all experiments
│   ├── results/              # JSON + Markdown reports
│   └── EXPERIMENTS_GUIDE.md
│
├── archive/                   # Archived code (safety net)
│   ├── old_dev/
│   ├── old_projects/         # apps/, Promptly/, crm_app/
│   ├── session_docs/         # PHASE_*, SESSION_* docs
│   └── old_demos/
│
├── squad/                     # VS Code extension (TypeScript)
├── ui/                        # Web UI components
├── alignment_logs/            # Alignment framework logs
└── CLAUDE.md                  # This file
```

**Key Changes:**
- ✅ Root: 17 → 6 files (-65%)
- ✅ Memory: 17 → 13 files (-24%)
- ✅ Tests: Organized into unit/integration/e2e
- ✅ Backend factory: 550 → 231 lines (-58%)
- ✅ Protocols: 787 → 120 lines (-84%)
- ✅ Dead code: Archived to tools/archive/
- ✅ All tests passing

## Important Patterns

### Protocol-Based Design
All major components define protocols (abstract interfaces):
- `PolicyEngine` for decision making
- `KGStore` for knowledge graphs
- `Retriever` for memory systems

This enables swapping implementations without changing orchestrator code.

### Graceful Degradation
Optional dependencies (spaCy, sentence-transformers, BM25, SciPy) degrade with warnings:
- Motif detection falls back to regex-only
- Embeddings use fallback implementations
- Spectral features are skipped if unavailable

Never crash due to missing optional dependencies.

### Async Pipeline
The orchestrator uses `async/await` for the main processing pipeline, enabling:
- Concurrent feature extraction and retrieval
- Background memory management tasks
- Non-blocking tool execution

### Import Path Requirements

**CRITICAL**: When running HoloLoom modules, set `PYTHONPATH=.` from repository root or run with proper path:
```bash
# Correct - from repository root
PYTHONPATH=. python HoloLoom/test_unified_policy.py

# Also correct - cd into directory
cd HoloLoom && python test_unified_policy.py
```

The codebase uses absolute imports like `from HoloLoom.policy.unified import ...`, which require the repository root to be on the Python path.

### Testing Strategy

The test suite (`test_unified_policy.py`) validates components in isolation:
1. Building blocks (MLP, attention)
2. Curiosity modules (ICM, RND)
3. Policy variants (deterministic, categorical, gaussian)
4. PPO agent (GAE, updates, checkpointing)
5. Full end-to-end pipeline

Tests are designed to run without external dependencies (no actual RL environments).

## Known Issues

From `documentation/CODE_REVIEW.md`:

1. **Background task lifecycle**: MemoryManager spawns fire-and-forget tasks without shutdown hooks. Add explicit lifecycle management.

2. **Type duplication**: Orchestrator defines inline types that duplicate `modules/Types.py`. Consolidate to shared types.

3. **Empty Features module**: `modules/Features.py` is currently empty but imported elsewhere. Either implement or remove.

4. **Large orchestrator file**: `weaving_orchestrator.py` has grown to 3,476 lines. Consider refactoring into smaller modules:
   - Extract stage executors (retrieval, decision, synthesis) into separate files
   - Move protocol definitions to `protocols/`
   - Split complexity modes (LITE/FAST/FULL/RESEARCH) into strategy classes
   - Target: <1,000 lines per file for maintainability

## Development Tips

1. **Start with BARE mode** for fastest iteration:
   ```python
   cfg = Config.bare()
   ```

2. **Check bandit statistics** to understand exploration behavior:
   ```python
   stats = policy.bandit.get_stats()
   ```

3. **Use spectral features** for richer context in FUSED mode:
   - Graph Laplacian eigenvalues capture knowledge structure
   - SVD components provide topic-level signals

4. **Monitor adapter selection** via action plan metadata:
   ```python
   action_plan = await orchestrator.process(query)
   print(f"Adapter: {action_plan.adapter}")
   ```

5. **Training checkpoints** are saved as `.pt` files with full state dicts. Load with:
   ```python
   agent.load('logs/checkpoint.pt')
   ```

6. **Use async context managers for lifecycle management** (recommended):
   ```python
   async with WeavingShuttle(cfg=config, shards=shards) as shuttle:
       spacetime = await shuttle.weave(query)
       # Automatic cleanup on exit
   ```

7. **Use dynamic memory backends for persistent storage**:
   ```python
   memory = await create_memory_backend(config)
   async with WeavingShuttle(cfg=config, memory=memory) as shuttle:
       spacetime = await shuttle.weave(query)
   ```

## Phase 5: Universal Grammar + Compositional Cache

**Implemented: Oct 2025** - Phase 5 provides 10-300× speedup through linguistic intelligence and compositional caching.

### Overview

Phase 5 integrates three breakthrough technologies:
1. **Universal Grammar Chunking**: X-bar theory for principled phrase structure analysis
2. **Compositional Cache**: 3-tier caching (parse/merge/semantic) with phrase-level reuse
3. **Linguistic Matryoshka Gate**: Pre-filtering and progressive refinement

### Performance Benefits

- **Parse Cache**: 10-50× speedup for X-bar structure caching
- **Merge Cache**: 5-10× speedup through compositional reuse
- **Semantic Cache**: 3-10× speedup for 228D projections
- **Total Speedup**: 50-300× multiplicative speedup (hot paths)
- **Production**: 10-17× expected speedup with 90-99% cache hit rates

### Configuration

Enable Phase 5 in your config:

```python
from HoloLoom.config import Config

# Basic Phase 5 (compositional cache only)
config = Config.fused()
config.enable_linguistic_gate = True
config.linguistic_mode = "disabled"  # Cache only, no pre-filtering
config.use_compositional_cache = True
config.parse_cache_size = 10000
config.merge_cache_size = 50000

# Advanced Phase 5 (full linguistic filtering)
config = Config.fused()
config.enable_linguistic_gate = True
config.linguistic_mode = "both"  # Pre-filter + embedding features
config.use_compositional_cache = True
config.linguistic_weight = 0.3
config.prefilter_similarity_threshold = 0.3
config.prefilter_keep_ratio = 0.7
```

### Usage

```python
from HoloLoom.config import Config
from HoloLoom.weaving_orchestrator import WeavingOrchestrator
from HoloLoom.documentation.types import Query, MemoryShard

# Create config with Phase 5 enabled
config = Config.fused()
config.enable_linguistic_gate = True
config.linguistic_mode = "both"

# Create orchestrator
shards = create_memory_shards()
async with WeavingOrchestrator(cfg=config, shards=shards) as shuttle:
    # First query (cold cache)
    spacetime = await shuttle.weave(Query(text="What is passive voice?"))
    # Duration: ~150ms (cold)

    # Repeated query (warm cache)
    spacetime = await shuttle.weave(Query(text="What is passive voice?"))
    # Duration: ~0.5ms (hot) - 300× speedup!
```

### Linguistic Filter Modes

- **disabled**: Compositional cache only (no linguistic pre-filtering)
- **prefilter**: Filter candidates by syntactic compatibility before embedding
- **embedding**: Add linguistic features to embeddings
- **both**: Pre-filter + embedding features (recommended for production)

### Demo

Run the Phase 5 integration demo:

```bash
PYTHONPATH=. python demos/phase5_orchestrator_integration.py
```

This demonstrates:
- Baseline performance without Phase 5
- Compositional cache performance (cache only)
- Full linguistic filtering performance
- Warm cache performance (100-300× speedup)

### Key Features

**Compositional Reuse**: Different queries share building blocks
- "the big red ball" caches "ball", "red ball", "big red ball"
- "a big red ball" reuses "big red ball" composition
- Cross-query optimization for massive speedups

**Universal Grammar**: Principled phrase structure
- X-bar theory (XP → Spec + X' → X' + Comp)
- Hierarchical phrase detection (NP, VP, PP, CP, TP)
- Syntactic compatibility scoring

**Graceful Fallback**: No breaking changes
- Phase 5 automatically falls back if spaCy not available
- Disabled by default (opt-in via config)
- Backward compatible with all existing code

### Documentation

See comprehensive Phase 5 documentation:
- `CHOMSKY_LINGUISTIC_INTEGRATION.md` - Linguistic foundations (992 lines)
- `LINGUISTIC_MATRYOSHKA_INTEGRATION.md` - Matryoshka gate integration (551 lines)
- `PHASE_5_UG_COMPOSITIONAL_CACHE.md` - Architecture and design (782 lines)
- `PHASE_5_COMPLETE.md` - Implementation summary (592 lines)
- `PHASE_5_INTEGRATION_COMPLETE.md` - Final integration notes

## Unified Memory Integration

HoloLoom supports both static shards and dynamic memory backends for persistent storage.

### Memory Sources

**Static Shards** (backward compatible):
```python
shards = create_test_shards()
shuttle = WeavingShuttle(cfg=config, shards=shards)
```

**Dynamic Backends** (persistent storage):
```python
from HoloLoom.memory.backend_factory import create_memory_backend

memory = await create_memory_backend(config)
shuttle = WeavingShuttle(cfg=config, memory=memory)
```

### Backend Options (Simplified - Oct 2025)

**Task 1.3 Simplification:** Reduced from 10+ backends to 3 core options.

Configure via `Config.memory_backend`:
- **INMEMORY**: NetworkX in-memory graph (development, always works)
- **HYBRID**: Neo4j + Qdrant with auto-fallback (production, recommended)
- **HYPERSPACE**: Advanced gated multipass (research only)

**Auto-Fallback:** HYBRID automatically falls back to INMEMORY if Docker services unavailable.
**Migration:** All legacy backend enums removed. See `MEMORY_SIMPLIFICATION_REVIEW.md` for details.

### Docker Setup

Start Neo4j + Qdrant backends:
```bash
docker-compose up -d
```

See `DOCKER_MEMORY_SETUP.md` for complete setup guide and `UNIFIED_MEMORY_INTEGRATION.md` for implementation details.

### Production Example

```python
from HoloLoom.config import Config, MemoryBackend
from HoloLoom.memory.backend_factory import create_memory_backend

config = Config.fused()
config.memory_backend = MemoryBackend.HYBRID

# Create persistent backend (auto-falls back to INMEMORY if no Docker)
memory = await create_memory_backend(config)

# Use with shuttle
async with WeavingShuttle(cfg=config, memory=memory) as shuttle:
    spacetime = await shuttle.weave(query)
    # Data persists across sessions (if Neo4j/Qdrant available)
```

## Lifecycle Management

HoloLoom implements proper resource management through async context managers:

### Using Context Managers (Recommended)

```python
from HoloLoom.weaving_shuttle import WeavingShuttle
from HoloLoom.config import Config

config = Config.fast()
shards = create_memory_shards()

# Recommended: Automatic cleanup
async with WeavingShuttle(cfg=config, shards=shards, enable_reflection=True) as shuttle:
    spacetime = await shuttle.weave(query)
    await shuttle.reflect(spacetime, feedback={"helpful": True})
    # Resources automatically cleaned up on exit
```

### Manual Cleanup

If context managers aren't suitable (e.g., long-lived services), use explicit cleanup:

```python
shuttle = WeavingShuttle(cfg=config, shards=shards)
try:
    spacetime = await shuttle.weave(query)
finally:
    await shuttle.close()  # IMPORTANT: Always close!
```

### Background Task Tracking

Background tasks are automatically tracked and cancelled on shutdown:

```python
async with WeavingShuttle(cfg=config, shards=shards) as shuttle:
    # Spawn tracked background tasks
    task = shuttle.spawn_background_task(some_async_work())

    # Do weaving
    spacetime = await shuttle.weave(query)

    # Background tasks cancelled automatically on exit
```

### What Gets Cleaned Up

1. **Background tasks**: Cancelled with 5-second timeout
2. **Reflection buffer**: Metrics flushed to disk
3. **Database connections**: Neo4j/Qdrant clients closed (when implemented)
4. **File handles**: Proper closing of persistent storage

### ReflectionBuffer Lifecycle

The reflection buffer also supports lifecycle management:

```python
async with ReflectionBuffer(capacity=1000, persist_path="./reflections") as buffer:
    await buffer.store(spacetime, feedback=feedback)
    # Metrics automatically flushed on exit
```

## Recursive Learning System

**Status**: ✅ All 5 Phases Complete (October 29, 2025)
**Location**: `HoloLoom/recursive/`
**Total Code**: ~4,700 lines across 5 phases

The Recursive Learning System is a self-improving knowledge architecture that learns from every interaction, adapts continuously, and maintains complete provenance of all decisions.

### Overview

The system implements 5 phases of recursive learning:

1. **Phase 1: Scratchpad Integration** - Provenance tracking
2. **Phase 2: Loop Engine Integration** - Pattern learning
3. **Phase 3: Hot Pattern Feedback** - Usage-based adaptation
4. **Phase 4: Advanced Refinement** - Multi-strategy refinement
5. **Phase 5: Full Learning Loop** - Background learning with Thompson Sampling

### Philosophy

**"Great answers aren't written, they're refined."**

The system embraces multiple passes on quality dimensions:
- **ELEGANCE**: Clarity → Simplicity → Beauty
- **VERIFY**: Accuracy → Completeness → Consistency

### Phase 1: Scratchpad Integration (990 lines)

Tracks complete provenance of every decision:

```python
from HoloLoom.recursive import weave_with_scratchpad

spacetime, scratchpad = await weave_with_scratchpad(
    Query(text="What is Thompson Sampling?"),
    Config.fast(),
    shards=shards,
    enable_refinement=True
)

# View complete reasoning history
print(scratchpad.get_history())
```

**Features**:
- Automatic thought → action → observation → score tracking
- Full audit trail for debugging
- Triggers refinement when confidence < threshold

### Phase 2: Pattern Learning (850 lines)

Learns from successful queries:

```python
from HoloLoom.recursive import LearningLoopEngine

async with LearningLoopEngine(cfg=config, shards=shards) as engine:
    spacetime = await engine.weave_and_learn(query)

    # System automatically learns patterns from high-confidence results
    patterns = engine.pattern_learner.get_hot_patterns()
```

**Features**:
- Extracts motif → tool → confidence patterns
- Classifies queries (factual, procedural, analytical)
- Auto-prunes stale patterns
- Learns what works over time

### Phase 3: Hot Pattern Feedback (780 lines)

Adapts retrieval based on usage:

```python
from HoloLoom.recursive import HotPatternFeedbackEngine

async with HotPatternFeedbackEngine(cfg=config, shards=shards) as engine:
    spacetime = await engine.weave(query)

    # View hot patterns (most accessed knowledge)
    hot = engine.hot_tracker.get_hot_patterns(limit=10)
```

**Heat Score Algorithm**:
```
heat = access_count × success_rate × avg_confidence
     × (0.95 ^ hours_since_last_access)
```

**Features**:
- Tracks access frequency of knowledge elements
- Hot patterns get 2x boost, cold patterns get 0.5x penalty
- Exponential decay (5% per hour)
- Adaptive retrieval weights

### Phase 4: Advanced Refinement (680 lines)

Multiple refinement strategies with quality tracking:

```python
from HoloLoom.recursive import AdvancedRefiner, RefinementStrategy

refiner = AdvancedRefiner(orchestrator, enable_learning=True)

result = await refiner.refine(
    query=query,
    initial_spacetime=low_confidence_result,
    strategy=RefinementStrategy.ELEGANCE,  # Or None for auto-select
    max_iterations=3,
    quality_threshold=0.9
)

print(result.summary())
# Output: Strategy: elegance, Iterations: 3, Quality: 0.65 → 0.94
```

**Available Strategies**:

| Strategy | Focus | Passes |
|----------|-------|--------|
| REFINE | Context expansion | Iterative |
| CRITIQUE | Self-improvement | 1 pass |
| VERIFY | Accuracy → Completeness → Consistency | 3 passes |
| ELEGANCE | Clarity → Simplicity → Beauty | 3 passes |
| HOFSTADTER | Recursive self-reference | Iterative |

**Quality Scoring**:
```
quality = 0.7 × confidence + 0.2 × context_richness + 0.1 × response_completeness
```

**Features**:
- Auto-strategy selection based on query characteristics
- Quality trajectory tracking across iterations
- Learns which strategies work best for which queries
- Multi-pass refinement for complex quality dimensions

### Phase 5: Full Learning Loop (750 lines)

Background learning with Bayesian updates:

```python
from HoloLoom.recursive import FullLearningEngine

async with FullLearningEngine(
    cfg=config,
    shards=shards,
    enable_background_learning=True,
    learning_update_interval=60.0  # Update every 60 seconds
) as engine:
    # Process queries - system learns automatically
    spacetime = await engine.weave(
        query,
        enable_refinement=True,
        refinement_threshold=0.75
    )

    # View comprehensive statistics
    stats = engine.get_learning_statistics()

    # Save learning state
    engine.save_learning_state("./learning_state")
```

**Thompson Sampling Updates**:
```
Success (confidence ≥ 0.75): α ← α + confidence
Failure (confidence < 0.75): β ← β + (1 - confidence)

Expected Reward: E[X] = α / (α + β)
```

**Policy Weight Updates**:
```
weight = (successes + 1) / (total + 2)  # Laplace smoothing
```

**Features**:
- Background learning thread (async, every 60s)
- Thompson Sampling priors adapt to tool performance
- Policy adapter weights adjust based on outcomes
- Complete learning state persistence

### Usage Examples

**Simple (Phase 1 only)**:
```python
from HoloLoom.recursive import weave_with_scratchpad

spacetime, scratchpad = await weave_with_scratchpad(
    Query(text="Explain recursion"),
    Config.fast(),
    shards=shards
)
```

**With Learning (Phases 1-3)**:
```python
from HoloLoom.recursive import HotPatternFeedbackEngine

async with HotPatternFeedbackEngine(cfg=config, shards=shards) as engine:
    spacetime = await engine.weave(query)
```

**Full System (All 5 Phases)**:
```python
from HoloLoom.recursive import FullLearningEngine

async with FullLearningEngine(
    cfg=config,
    shards=shards,
    enable_background_learning=True
) as engine:
    spacetime = await engine.weave(query, enable_refinement=True)
    stats = engine.get_learning_statistics()
```

### Performance Characteristics

| Operation | Overhead | When |
|-----------|----------|------|
| Provenance extraction | <1ms | Every query |
| Pattern extraction | <1ms | High-confidence only |
| Heat tracking | <0.5ms | Every query |
| Thompson/Policy update | <0.5ms | Every query |
| Refinement | ~150ms × iterations | Low-confidence only (10-20%) |
| Background learning | ~50ms | Every 60s (async) |

**Total Per-Query Overhead**: <3ms (excluding refinement)

### Key Benefits

1. **Self-Improving**: Gets better with every query
2. **Quality-Aware**: Detects low confidence and refines automatically
3. **Adaptive**: Thompson Sampling + policy weights + retrieval weights all adapt
4. **Complete Provenance**: Full audit trail with scratchpad
5. **Minimal Overhead**: <3ms per query

### Documentation

- **RECURSIVE_LEARNING_COMPLETE.md**: Complete system overview
- **PHASES_4_5_COMPLETE.md**: Phase 4-5 implementation details
- **MULTIPASS_REFINEMENT.md**: Multi-pass philosophy and usage
- **demos/demo_multipass_simple.py**: Visual demonstration

### Running Demos

```bash
# Multi-pass refinement demonstration
python demos/demo_multipass_simple.py

# Full 5-phase system (requires HoloLoom integration)
PYTHONPATH=. python demos/demo_full_recursive_learning.py
```

## Alignment Framework

**Status**: ✅ Production Ready (v1.0.0 - November 2025)
**Location**: `HoloLoom/alignment/`
**Performance**: 0.103 ms overhead (29x faster than target)
**Test Coverage**: 46 functional tests + 13 performance benchmarks

The Alignment Framework provides comprehensive safety mechanisms for HoloLoom's agentic reasoning system, implementing industry best practices from Anthropic, OpenAI, and DeepMind research.

### Core Philosophy

> **"Safe by default, transparent by design"**

Every decision is gated by safety checks, monitored for deception, bound by resource limits, and logged with complete provenance - all with **negligible performance impact** (<0.11 ms per query).

### 4 Core Modules

1. **Safety Guardrails** (`safety_guardrails.py`) - 0.039 ms
   - Risk-based action gating (LOW/MEDIUM/HIGH/CRITICAL)
   - Adversarial pattern detection
   - Human-in-the-loop escalation for high-risk actions

2. **Deception Detection** (`deception_detection.py`) - 0.034 ms
   - Goal transparency tracking
   - Behavioral probe system
   - Hidden goal detection

3. **Instrumental Convergence Prevention** (`instrumental_convergence.py`) - 0.015 ms
   - Power-seeking detection
   - Resource acquisition monitoring
   - Self-preservation behavior detection

4. **Audit Trail** (`audit_trail.py`) - 0.015 ms
   - Complete decision provenance
   - Searchable logs with temporal queries
   - Export for compliance/debugging

### Quick Start

```python
from HoloLoom.alignment import SafetyGuardrails, AuditTrail
from HoloLoom.weaving_orchestrator import WeavingOrchestrator

# Enable alignment framework
config = Config.fused()
config.enable_alignment = True

# Create guardrails and audit trail
guardrails = SafetyGuardrails(enable_human_in_loop=True)
audit_trail = AuditTrail()

async with WeavingOrchestrator(cfg=config, shards=shards) as orchestrator:
    # Gate action through safety check
    action = "execute_code"
    context = {"code": "import os; os.system('ls')"}

    gate_result = await guardrails.gate_action(action, context)

    if gate_result.allowed:
        spacetime = await orchestrator.weave(query)

        # Log decision
        await audit_trail.log_decision(
            query=query.text,
            action=action,
            outcome="success",
            safety_score=gate_result.safety_score
        )
    else:
        print(f"Action blocked: {gate_result.reason}")
```

### Production Deployment

```python
from HoloLoom.alignment import create_guardrails, create_audit_trail
from HoloLoom.weaving_orchestrator import WeavingOrchestrator
from HoloLoom.config import Config

# Orchestrator with full alignment
config = Config.fused()
guardrails = create_guardrails(enable_human_in_loop=True)
audit_trail = create_audit_trail()

async with WeavingOrchestrator(
    cfg=config,
    shards=shards,
    guardrails=guardrails
) as orchestrator:
    spacetime = await orchestrator.weave(query)

    # Log to audit trail
    await audit_trail.log_decision(
        query=query.text,
        action=spacetime.metadata.get('tool_used'),
        outcome="success",
        safety_score=spacetime.confidence
    )
```

### Documentation

- **README.md**: Complete framework overview
- **API_REFERENCE.md**: API documentation
- **PRODUCTION_DEPLOYMENT.md**: Production setup guide
- **PRODUCTION_MONITORING.md**: Monitoring and alerting
- **QUICK_START.md**: Quick start guide

### Running Tests

```bash
# All alignment tests
pytest HoloLoom/alignment/tests/ -v

# Performance benchmarks
pytest HoloLoom/alignment/tests/test_performance.py -v
```

## Agentic Reasoning System

**Status**: ✅ Complete (November 2025)
**Location**: `HoloLoom/agentic/`
**Integration**: VS Code extension, FastAPI server

The Agentic Reasoning System enables multi-query reasoning with automatic verification, plan-execute workflows, and research-style exploration.

### 4 Reasoning Modes

| Mode | Description | Latency | Use Case |
|------|-------------|---------|----------|
| **DIRECT** | Single-pass answer | ~150ms | Simple factual queries |
| **VERIFY** | Answer + verification | ~600ms | Claims needing verification |
| **RESEARCH** | Multi-query exploration | ~900ms | Open-ended research |
| **PLAN_EXECUTE** | Goal decomposition | ~750ms | Multi-step tasks |

### Usage

```python
from HoloLoom.agentic import AgenticOrchestrator, ReasoningMode

async with AgenticOrchestrator(cfg=config, shards=shards) as orchestrator:
    # Research mode: explores topic from multiple angles
    result = await orchestrator.reason(
        query="What are the tradeoffs of Thompson Sampling?",
        mode=ReasoningMode.RESEARCH,
        max_steps=5
    )

    print(result.response)  # Final synthesized answer
    print(result.confidence)  # 0.0-1.0
    print(result.steps_taken)  # List of sub-queries
    print(result.verification)  # Verification results (if mode=VERIFY)
```

### Integration with Alignment

The agentic system automatically integrates with the alignment framework:

```python
from HoloLoom.agentic import create_agentic_orchestrator
from HoloLoom.alignment import create_guardrails

# Orchestrator with alignment checks
guardrails = create_guardrails(enable_human_in_loop=True)

async with create_agentic_orchestrator(
    config=Config.fused(),
    shards=shards,
    guardrails=guardrails
) as orchestrator:
    # All reasoning steps are gated by safety guardrails
    result = await orchestrator.reason(query, mode=ReasoningMode.RESEARCH)
```

### Key Features

- **Automatic Verification**: VERIFY mode checks claims for contradictions
- **Goal Decomposition**: PLAN_EXECUTE breaks complex tasks into steps
- **Multi-Query Research**: RESEARCH mode explores topics from multiple angles
- **Embedding Integrity**: Ensures embedding consistency across reasoning steps
- **Complete Audit Trail**: Full provenance of all reasoning steps

## FastAPI Server

**Status**: ✅ Production Ready (November 2025)
**Location**: `HoloLoom/server/`
**Port**: 8000 (default)

FastAPI server exposing HoloLoom's agentic intelligence to external clients (VS Code Squad extension, web apps, etc).

### Quick Start

```bash
# Development mode (with auto-reload)
PYTHONPATH=. uvicorn HoloLoom.server.agentic_api:app --reload --port 8000

# Production mode
PYTHONPATH=. uvicorn HoloLoom.server.agentic_api:app --host 0.0.0.0 --port 8000 --workers 4
```

### Main Endpoints

**Health Check**:
```bash
GET http://localhost:8000/health
```

**Query** (main endpoint):
```bash
POST http://localhost:8000/query
Content-Type: application/json

{
  "text": "Explain this TypeScript code",
  "context": {
    "languageId": "typescript",
    "fileName": "example.ts",
    "selection": "function foo() { return 42; }"
  },
  "mode": "verify",
  "max_steps": 5
}
```

**Statistics**:
```bash
GET http://localhost:8000/stats
```

**Audit Trail**:
```bash
GET http://localhost:8000/audit-trail?limit=10
```

### VS Code Integration

The server is designed to work with the Squad VS Code extension:

```typescript
// squad/src/HoloLoomBridge.ts
const bridge = new HoloLoomBridge('http://localhost:8000');

const result = await bridge.query(
  "Explain this code",
  codeContext,
  'verify',
  5
);

console.log(result.response);
console.log(result.verification.verified);
```

### Architecture

```
VS Code Extension (TypeScript)
    ↓ HTTP
FastAPI Server (Python)
    ↓
AgenticOrchestrator
    ├─ FullLearningEngine
    ├─ AuditTrail
    └─ ReasoningModes (DIRECT/VERIFY/RESEARCH/PLAN_EXECUTE)
```

## Visual Workflow Builder

**Status**: ✅ Production Ready (November 2025)
**Location**: `HoloLoom/web_dashboard/`
**Port**: 8001 (workflow executor)

Drag-and-drop visual workflow builder for creating complex multi-agent pipelines.

### Quick Start

```bash
# Start the backend executor
cd HoloLoom/web_dashboard
python workflow_executor.py

# Open workflow_builder.html in browser
```

### Features

- **18 agent types**: Query, Process, Memory, Decision, Output, Control
- **Drag-and-drop**: Visual workflow design
- **Real-time execution**: Live progress via WebSocket
- **Import/Export**: Share workflows as JSON
- **Validation**: Automatic cycle detection
- **Safety integration**: Built-in guardrails

### Available Agents

**Query Agents** (3):
- HoloLoom Query - Full weaving cycle
- Memory Search - Knowledge graph search
- Multi-Query - Break into sub-questions

**Processing Agents** (3):
- Matryoshka Embedder - Multi-scale embeddings
- Synthesizer - Extract entities/motifs
- Recursive Refiner - Quality refinement

**Memory Agents** (3):
- Memory Store - Persist to graph+vector
- Context Retriever - Retrieve context
- Knowledge Fusion - Multi-hop traversal

**Decision Agents** (3):
- Thompson Sampler - Bayesian exploration
- Convergence Engine - Decision collapse
- Safety Guardrails - Risk gating

**Output Agents** (2):
- Response Generator - Generate response
- Format Converter - JSON/Markdown/HTML

**Control Flow** (3):
- Conditional Branch - If/else logic
- Loop Iterator - Repeat until condition
- Parallel Executor - Concurrent execution

### Example Workflows

**Simple Query**:
```
[HoloLoom Query] → [Response Generator]
```

**Research Pipeline**:
```
[Multi-Query] → [HoloLoom (×5)] → [Synthesizer] → [Refiner] → [Response]
```

**Safety-Gated**:
```
[HoloLoom] → [Safety] → [Conditional] → [High/Low Confidence Paths]
```

### API

```http
POST http://localhost:8001/api/workflow/execute
Content-Type: application/json

{
  "workflow": {
    "version": "1.0",
    "name": "My Workflow",
    "nodes": [...],
    "connections": [...]
  },
  "input_data": {
    "query": "What is Thompson Sampling?"
  }
}
```

### Keyboard Shortcuts

- **Delete**: Delete node
- **Escape**: Cancel/deselect
- **Ctrl+S**: Export workflow
- **Ctrl+Enter**: Execute workflow

### Documentation

See [WORKFLOW_BUILDER_COMPLETE.md](WORKFLOW_BUILDER_COMPLETE.md) and [HoloLoom/web_dashboard/README_WORKFLOW_BUILDER.md](HoloLoom/web_dashboard/README_WORKFLOW_BUILDER.md) for complete documentation.

## Experiments Framework

**Status**: ✅ Complete (October 2025)
**Location**: `experiments/`
**Runtime**: ~1 second for all experiments

Automated testing framework that systematically compares fusion, complexity, budgets, and memory limits across 16 configurations.

### Running Experiments

```bash
# Run all experiments
python experiments/run_experiments.py

# Output: experiments/results/all_experiments.json
#         experiments/results/experiment_report.md
```

### What Gets Tested

**Experiment 1: Fusion Impact** (2 runs)
- Tests multipass graph crawling ON vs OFF
- Measures depth, quality, time overhead
- Answers: Is connected knowledge discovery worth +1-2ms?

**Experiment 2: Complexity Scaling** (4 runs)
- Tests LITE → FAST → FULL → RESEARCH progression
- Measures passes, depth, memories, time
- Answers: How does complexity scale?

**Experiment 3: Budget Constraints** (5 runs)
- Tests query budgets from 1 to unlimited
- Measures depth, quality, stopping behavior
- Answers: Do budgets prevent runaway queries?

**Experiment 4: Memory Limits** (5 runs)
- Tests memory limits from 10 to unlimited
- Measures retrieval effectiveness, degradation
- Answers: How many memories are "enough"?

### Example Output

```json
{
  "fusion_impact": {
    "fusion_on": {"depth": 3, "quality": 0.92, "time_ms": 156},
    "fusion_off": {"depth": 1, "quality": 0.78, "time_ms": 142}
  },
  "complexity_scaling": {
    "LITE": {"passes": 1, "time_ms": 45},
    "FAST": {"passes": 2, "time_ms": 98},
    "FULL": {"passes": 3, "time_ms": 187},
    "RESEARCH": {"passes": 5, "time_ms": 342}
  }
}
```

### Documentation

- **EXPERIMENTS_GUIDE.md**: Complete guide
- **EXPERIMENTS_QUICK_REF.md**: Quick reference

## Archive Structure

**Location**: `archive/`
**Purpose**: Safety net for old code and documentation

Old code and documentation are archived (not deleted) to maintain project history and enable recovery if needed.

### Archive Organization

```
archive/
├── old_dev/           # Old development scripts
├── old_projects/      # Deprecated sub-projects (apps/, Promptly/, etc.)
├── session_docs/      # Old session documentation (PHASE_*, SESSION_*, etc.)
├── old_demos/         # Deprecated demos
├── old_tests/         # Legacy test files
└── legacy/            # Other legacy code
```

### Archived Projects

- **Promptly/**: Prompt management CLI (superseded by alignment framework)
- **apps/**: Domain-specific apps (beekeeping, food_e, darkTrace, mythy)
  - Now integrated into main HoloLoom via spinners
- **crm_app/**: CRM demo (superseded by unified memory system)

### Accessing Archived Code

```bash
# View archived projects
ls archive/old_projects/

# Recover a file if needed
cp archive/old_projects/Promptly/promptly.py ./
```

## Undocumented Features & Hidden Systems

**Discovery Date**: November 2025 (Agent Swarm Exploration)
**Total Hidden Code**: ~15,000+ lines of production features

HoloLoom contains several powerful features and systems that work silently in the background but aren't prominently documented in user-facing guides. These were discovered through systematic codebase exploration.

### Major Undocumented Systems

#### 1. Awareness Graph (Memory Activation System)

**Location**: `HoloLoom/memory/awareness_graph.py` (~800 lines)
**Status**: Production-ready, actively used by `hololoom.py`

**What it does**:
- Tracks activation levels of all memories (0.0-1.0 scale)
- Implements spreading activation across connected nodes
- Detects coherence (how well-connected active memories are)
- Temporal decay of inactive memories
- Network-wide awareness metrics (active nodes, mean activation, coherence)

**Why it's hidden**: Abstracted behind `HoloLoom.get_metrics()` API

**How to use**:
```python
from HoloLoom import HoloLoom

async with HoloLoom() as loom:
    await loom.experience("Thompson Sampling balances exploration")
    metrics = loom.get_metrics()
    print(f"Active memories: {metrics['activation']['active_nodes']}")
    print(f"Coherence: {metrics['coherence']['global_coherence']:.2f}")
```

#### 2. Spring Dynamics (Physics-Based Memory)

**Location**: `HoloLoom/memory/spring_dynamics.py` (~650 lines)
**Status**: Experimental, available but not enabled by default

**What it does**:
- Models memory connections as springs with tension/compression
- Applies Hooke's law to memory relationships
- Enables physics-based graph layout and clustering
- Simulates memory evolution over time

**Why it's hidden**: Advanced feature for research use cases

**How to enable**:
```python
from HoloLoom.config import Config
config = Config.fused()
config.enable_spring_dynamics = True  # If config flag exists
```

#### 3. Multi-Wave Engine (Temporal Wave Propagation)

**Location**: `HoloLoom/memory/multi_wave_engine.py` (~720 lines)
**Status**: Production-ready, used in FUSED mode

**What it does**:
- Propagates activation waves across memory graph
- Multi-frequency waves (fast/slow propagation)
- Wave interference patterns reveal memory structure
- Temporal dynamics for recall prioritization

**Why it's hidden**: Internal to memory retrieval, not exposed in simple API

#### 4. 47 SpinningWheel Adapters

**Location**: `HoloLoom/spinningWheel/` (~5,200 lines total)
**Status**: Most adapters production-ready

**Undocumented adapters** (beyond Audio and YouTube):
- **Web scraping**: RSS, API responses, HTML parsing
- **Code repositories**: Git history, PR reviews, stack traces
- **Documents**: PDFs, DOCX, PPTX, LaTeX, Jupyter notebooks
- **Databases**: SQL, MongoDB, Redis queries
- **Communication**: Email threads, Slack/Discord channels
- **Structured data**: CSV, JSON, YAML parsers

**Why they're hidden**: Only Audio and YouTube are documented in main README

**How to discover**:
```bash
ls HoloLoom/spinningWheel/*.py
# Or check HoloLoom/spinningWheel/README.md (if exists)
```

#### 5. Semantic Calculus 16 Axes

**Location**: `HoloLoom/semantic_calculus/axes.py` (~450 lines)
**Status**: Production-ready, first 16 dimensions of 228D space

**What it does**:
- Projects queries onto 16 human-interpretable semantic axes
- Enables semantic navigation (find queries along "formality" axis)
- Supports semantic filtering (only "urgent" queries)
- Powers semantic clustering and visualization

**16 Axes**: sentiment, formality, technicality, certainty, urgency, abstraction, specificity, temporality, objectivity, complexity, scope, directness, emotionality, actionability, novelty, controversy

**Why it's hidden**: Most users only need the full 228D projection

**How to use**:
```python
from HoloLoom.semantic_calculus.axes import SemanticAxes

axes = SemanticAxes()
query = "Urgently need help with this bug!"
projection = axes.project(query)
print(f"Urgency: {projection['urgency']:.2f}")  # High value
print(f"Formality: {projection['formality']:.2f}")  # Low value
```

#### 6. Visual Compression (Graph→Image)

**Location**: `HoloLoom/memory/visual_compression.py` (~580 lines)
**Status**: Production-ready, used in MultimodalRAG

**What it does**:
- Converts knowledge graphs to PNG images
- 5-20x token savings for LLM context
- Preserves entity relationships visually
- Automatic compression when context exceeds threshold

**Why it's hidden**: Automatic in MultimodalRAG, not exposed separately

**How to use directly**:
```python
from HoloLoom.memory.visual_compression import compress_graph_to_image
from HoloLoom.memory.graph import KG

kg = KG()
# ... populate graph ...
png_bytes, metrics = compress_graph_to_image(kg)
print(f"Compression: {metrics['compression_ratio']:.1f}x token savings")
```

#### 7. Query Cache (100x Speedup)

**Location**: `HoloLoom/memory/query_cache.py` (~340 lines)
**Status**: Production-ready, enabled by default in FAST/FUSED

**What it does**:
- Caches query→result mappings
- 100x speedup for repeated queries (150ms → <1ms)
- LRU eviction policy
- Configurable cache size and TTL

**Why it's hidden**: Transparent caching, users don't need to manage it

**How to configure**:
```python
from HoloLoom.config import Config
config = Config.fused()
config.query_cache_size = 10000  # Default: 5000
config.query_cache_ttl = 3600    # 1 hour TTL
```

#### 8. Warp Space (Tensioned Tensor Field)

**Location**: `HoloLoom/warp/space.py` (~890 lines)
**Status**: Production-ready, used in FULL/RESEARCH complexity modes

**What it does**:
- Tensions discrete Yarn Graph threads into continuous manifold
- Enables tensor operations on symbolic memory
- Lifecycle: tension() → compute() → collapse() → detension()
- Temporary continuous space for mathematical operations

**Why it's hidden**: Internal to weaving cycle, not exposed separately

**How it works** (inside orchestrator):
```python
warp_space = WarpSpace()
await warp_space.tension(yarn_threads)
result = await warp_space.compute(operation)
await warp_space.collapse()  # Back to discrete
```

#### 9. Convergence Engine (Decision Collapse)

**Location**: `HoloLoom/convergence/engine.py` (~540 lines)
**Status**: Production-ready, used in all weaving cycles

**What it does**:
- Collapses probability distributions to discrete tool selections
- 4 strategies: ARGMAX, EPSILON_GREEDY, BAYESIAN_BLEND, PURE_THOMPSON
- Thompson Sampling for exploration/exploitation balance
- Configurable exploration parameters

**Why it's hidden**: Internal to policy engine

**How to configure**:
```python
from HoloLoom.convergence.engine import CollapseStrategy
from HoloLoom.config import Config

config = Config.fused()
config.collapse_strategy = CollapseStrategy.PURE_THOMPSON
config.thompson_exploration = 0.15  # 15% exploration
```

### Discovery Process

These features were uncovered through:
1. **Systematic codebase scanning** (Glob patterns for all .py files)
2. **Import graph analysis** (what's imported but not documented?)
3. **Config flag enumeration** (unused configuration options)
4. **Protocol implementation search** (classes implementing protocols but not mentioned)
5. **Test file analysis** (features tested but not documented)

### Why These Features Are Undocumented

**Reasons for hidden status**:
1. **Too complex for beginners** - Advanced features that would overwhelm new users
2. **Automatically enabled** - Work transparently (e.g., Query Cache)
3. **Internal implementation** - Not meant for direct user access (e.g., Warp Space)
4. **Research features** - Experimental, subject to change (e.g., Spring Dynamics)
5. **Documentation debt** - Built during rapid development, docs not yet written

### Future Documentation Plans

**Recommended documentation priority**:
1. ✅ **High**: SpinningWheel adapters (47 adapters deserve complete reference)
2. ✅ **High**: Semantic Calculus 16 axes (highly interpretable, user-facing)
3. 🟡 **Medium**: Awareness Graph metrics (useful for debugging)
4. 🟡 **Medium**: Visual Compression (unique feature, worthy of highlight)
5. 🔵 **Low**: Warp Space, Convergence Engine (internal, advanced users only)

### How to Explore Further

```bash
# Find all Python files
find HoloLoom -name "*.py" | wc -l

# Search for undocumented classes
grep -r "class.*:" HoloLoom/**/*.py | grep -v test | wc -l

# Find protocol implementations
grep -r "Protocol" HoloLoom/**/*.py

# Analyze import graph
python -c "import ast; ..." # TODO: Create import analyzer
```

---

## Common Workflows

### Adding a New Tool
1. Add tool name to `NeuralCore.tools` list in `policy/unified.py`
2. Update `n_tools` parameter in config
3. Implement execution logic in `ToolExecutor.execute()` in `orchestrator.py`
4. Retrain policy or adjust tool selection weights

### Creating a Custom Adapter
1. Add adapter to `adapter_bank` dict in policy factory
2. Create corresponding LoRA adapter weights in `LoRALikeFFN`
3. Map adapter to execution context in orchestrator

### Adding a New Spinner (Input Adapter)
1. Inherit from `BaseSpinner` in `spinningWheel/base.py`
2. Implement `async def spin(raw_data) -> List[MemoryShard]`
3. Add to factory in `spinningWheel/__init__.py`
4. Parse raw data format and extract entities/motifs

Example: See `HoloLoom/spinningWheel/youtube.py` for a complete implementation
```python
from HoloLoom.spinningWheel import transcribe_youtube

# Quick usage
shards = await transcribe_youtube('VIDEO_ID', chunk_duration=60.0)

# Or use the spinner directly
from HoloLoom.spinningWheel import YouTubeSpinner, YouTubeSpinnerConfig

config = YouTubeSpinnerConfig(chunk_duration=60.0, enable_enrichment=True)
spinner = YouTubeSpinner(config)
shards = await spinner.spin({'url': 'VIDEO_ID', 'languages': ['en']})
```

### Tuning Exploration Strategy
Change bandit strategy when creating policy:
```python
from HoloLoom.policy.unified import BanditStrategy
policy = create_policy(
    mem_dim=384,
    emb=emb,
    scales=[96, 192, 384],
    bandit_strategy=BanditStrategy.BAYESIAN_BLEND,
    epsilon=0.15  # 15% exploration for epsilon-greedy
)
```

### Tufte-Style Visualizations (October 2025)

HoloLoom implements Edward Tufte's visualization principles: **"Above all else show the data."**

**Philosophy**: Maximize information density, minimize decoration ("chartjunk"). Show meaning first.

**Available Visualizations**:

1. **Small Multiples** (`HoloLoom/visualization/small_multiples.py`)
   - Compare multiple queries side-by-side
   - Consistent scales for fair comparison
   - Highlights best/worst automatically (★ and ⚠)
   - Inline sparklines show trends
   - Usage:
   ```python
   from HoloLoom.visualization.small_multiples import render_small_multiples

   queries = [
       {'query_text': 'Query A', 'latency_ms': 95, 'confidence': 0.92,
        'threads_count': 3, 'cached': True, 'trend': [105, 102, 98, 96, 95],
        'timestamp': 1698595200.0, 'tool_used': 'answer'},
       # ... more queries
   ]
   html = render_small_multiples(queries, layout='grid', max_columns=4)
   ```

2. **Data Density Tables** (`HoloLoom/visualization/density_table.py`)
   - Maximum information per square inch
   - Inline sparklines, delta indicators, bottleneck detection
   - Tight spacing, small fonts, monospace numbers
   - Usage:
   ```python
   from HoloLoom.visualization.density_table import render_stage_timing_table

   stages = [
       {'name': 'Retrieval', 'duration_ms': 50.5,
        'trend': [45, 47, 48, 50, 50.5], 'delta': +2.5},
       # ... more stages
   ]
   html = render_stage_timing_table(stages, total_duration=150.0)
   ```

3. **Tufte Sparklines** (enhanced in `html_renderer.py`)
   - Word-sized graphics (100x30px)
   - Show trends inline with metrics
   - Auto-normalization, endpoint indicators
   - See [TUFTE_SPARKLINES_PHASE_2_1_COMPLETE.md](TUFTE_SPARKLINES_PHASE_2_1_COMPLETE.md) for details

4. **Stage Waterfall Charts** (`HoloLoom/visualization/stage_waterfall.py`)
   - Sequential pipeline timing with horizontal stacked bars
   - Automatic bottleneck detection (stages >40% of total time)
   - Status indicators (success, warning, error, skipped)
   - Inline sparklines for historical trends
   - Parallel execution visualization support
   - Usage:
   ```python
   from HoloLoom.visualization.stage_waterfall import render_pipeline_waterfall

   # After weaving
   spacetime = await orchestrator.weave(query)

   # Render waterfall from trace
   html = render_pipeline_waterfall(
       spacetime.trace.stage_durations,
       stage_trends=historical_trends,  # Optional
       title=f"Pipeline: {query.text[:50]}"
   )

   # Or create custom stages
   from HoloLoom.visualization.stage_waterfall import WaterfallStage, StageStatus

   stages = [
       WaterfallStage(name='Retrieval', start_ms=0, duration_ms=50.5,
                      status=StageStatus.SUCCESS, trend=[45, 47, 48, 50, 50.5]),
       WaterfallStage(name='Decision', start_ms=50.5, duration_ms=30.0)
   ]
   html = renderer.render(stages)
   ```

5. **Confidence Trajectory** (`HoloLoom/visualization/confidence_trajectory.py`)
   - Time series confidence tracking with anomaly detection
   - Automatic anomaly detection (4 types: sudden drop, prolonged low, high variance, cache miss cluster)
   - Cache effectiveness visualization (hit/miss markers)
   - Statistical context (mean ± std bands, trend analysis)
   - Comprehensive programmatic API for automated tool calling
   - Usage:
   ```python
   from HoloLoom.visualization.confidence_trajectory import render_confidence_trajectory

   # Simple usage - just confidence scores
   confidences = [0.92, 0.88, 0.65, 0.87, 0.91]
   html = render_confidence_trajectory(confidences)

   # With cache markers
   cached = [True, True, False, False, True]
   html = render_confidence_trajectory(confidences, cached=cached)

   # Complete usage with anomaly detection
   query_texts = [
       "What is Thompson Sampling?",
       "How does it work?",
       "Show me an example",
       "What are the tradeoffs?",
       "How to implement?"
   ]
   html = render_confidence_trajectory(
       confidences,
       cached=cached,
       query_texts=query_texts,
       title='Session Analysis',
       subtitle='User session from 2025-10-29',
       detect_anomalies=True
   )

   # Integration with HoloLoom
   results = []
   for query in queries:
       spacetime = await orchestrator.weave(query)
       results.append(spacetime)

   # Extract and visualize
   confidences = [s.confidence for s in results]
   cached = [s.metadata.get('cache_hit', False) for s in results]
   html = render_confidence_trajectory(confidences, cached=cached)
   ```

   **Anomaly Types**:
   - SUDDEN_DROP: Confidence drops >0.2 in single step (red markers)
   - PROLONGED_LOW: Confidence <threshold for >3 consecutive queries (amber markers)
   - HIGH_VARIANCE: Std dev >0.15 in rolling window (amber markers)
   - CACHE_MISS_CLUSTER: 3+ cache misses in rolling window (indigo markers)

   **API Reference**: See [CONFIDENCE_TRAJECTORY_API.md](HoloLoom/visualization/CONFIDENCE_TRAJECTORY_API.md) (1000+ lines comprehensive documentation)

6. **Cache Effectiveness Gauge** (`HoloLoom/visualization/cache_gauge.py`)
   - Radial gauge showing cache performance metrics
   - 5 effectiveness ratings (excellent, good, fair, poor, critical)
   - Performance metrics (hit rate, latencies, time saved, speedup)
   - Actionable recommendations based on performance
   - Simple programmatic API for monitoring dashboards
   - Usage:
   ```python
   from HoloLoom.visualization.cache_gauge import render_cache_gauge

   # Simple usage
   html = render_cache_gauge(
       hit_rate=0.75,
       total_queries=100,
       cache_hits=75
   )

   # Complete usage with all parameters
   html = render_cache_gauge(
       hit_rate=0.75,
       total_queries=100,
       cache_hits=75,
       avg_cached_latency_ms=15.0,
       avg_uncached_latency_ms=120.0,
       title='Production Cache Performance',
       subtitle='Last 24 hours',
       show_details=True,
       show_recommendations=True
   )

   # Integration with HoloLoom
   total = 0
   hits = 0

   for query in queries:
       spacetime = await orchestrator.weave(query)
       total += 1
       if spacetime.metadata.get('cache_hit'):
           hits += 1

   html = render_cache_gauge(
       hit_rate=hits / total,
       total_queries=total,
       cache_hits=hits
   )
   ```

   **Effectiveness Ratings**:
   - EXCELLENT (Green): Hit rate >80%, speedup >4x
   - GOOD (Light Green): Hit rate 60-80%, speedup >2x
   - FAIR (Amber): Hit rate 40-60% or speedup >2x
   - POOR (Red): Hit rate 20-40%, low speedup
   - CRITICAL (Dark Red): Hit rate <20%

7. **Knowledge Graph Network** (`HoloLoom/visualization/knowledge_graph.py`)
   - Force-directed graph layout (Fruchterman-Reingold algorithm)
   - Node sizing by degree/importance (8-24px)
   - Semantic edge type colors (7 relationship types)
   - Interactive tooltips with node details
   - Path highlighting for reasoning chains
   - Direct integration with HoloLoom.memory.graph.KG
   - Zero dependencies (pure HTML/CSS/SVG)
   - Usage:
   ```python
   from HoloLoom.visualization.knowledge_graph import render_knowledge_graph_from_kg
   from HoloLoom.memory.graph import KG, KGEdge

   # Create knowledge graph
   kg = KG()
   kg.add_edges([
       KGEdge("attention", "transformer", "USES", 1.0),
       KGEdge("transformer", "neural_network", "IS_A", 1.0),
       KGEdge("BERT", "transformer", "IS_A", 1.0),
       KGEdge("GPT", "transformer", "IS_A", 1.0)
   ])

   # Render network
   html = render_knowledge_graph_from_kg(
       kg,
       title="Transformer Architecture",
       subtitle="Entity relationships in neural network domain"
   )

   # With path highlighting (for reasoning chains)
   highlighted_path = ["query", "retrieval", "reasoning", "decision"]
   html = render_knowledge_graph_from_kg(
       kg,
       title="Reasoning Pipeline",
       highlighted_path=highlighted_path
   )

   # Also supports NetworkX MultiDiGraph directly
   from HoloLoom.visualization.knowledge_graph import render_knowledge_graph_from_networkx
   import networkx as nx

   G = nx.MultiDiGraph()
   # ... add edges ...
   html = render_knowledge_graph_from_networkx(G, title="My Graph")
   ```

   **Edge Types with Semantic Colors**:
   - IS_A (Blue): Taxonomy relationships
   - USES (Green): Functional relationships
   - MENTIONS (Gray): Reference relationships
   - LEADS_TO (Orange): Causal relationships
   - PART_OF (Purple): Composition relationships
   - IN_TIME (Cyan): Temporal relationships
   - OCCURRED_AT (Teal): Event relationships

   **Force-Directed Layout**:
   - Repulsion: All nodes repel (inverse square law)
   - Attraction: Connected nodes attract (spring force)
   - Cooling: Gradual stabilization over 300 iterations
   - Natural clustering of related entities

**Demos**: See `demos/output/tufte_advanced_demo.html`, `demos/output/stage_waterfall_demo.html`, `demos/output/confidence_trajectory_demo.html`, `demos/output/cache_gauge_demo.html`, and `demos/output/knowledge_graph_demo.html`
**Roadmap**: [TUFTE_VISUALIZATION_ROADMAP.md](TUFTE_VISUALIZATION_ROADMAP.md) (600+ lines, 8 phases planned)
**Tests**: `test_tufte_advanced.py` (5/5 passing), `test_stage_waterfall.py` (7/7 passing), `test_confidence_trajectory.py` (9/9 passing), `test_cache_gauge.py` (8/8 passing), `test_knowledge_graph.py` (10/10 passing)

**Key Principles**:
- Maximize data-ink ratio (~60-70% vs ~30% traditional)
- Small multiples enable comparison
- High data density (16-24x more visible data)
- Meaning first (critical info highlighted)
- Zero external dependencies (pure HTML/CSS/SVG)
