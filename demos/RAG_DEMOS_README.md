# SimpleRAG Demos

This directory contains 4 self-contained demo scripts showcasing different RAG (Retrieval-Augmented Generation) use cases with HoloLoom's `SimpleRAG` API.

## Quick Start

Run any demo with:
```bash
PYTHONPATH=. python demos/demo_rag_qa_simple.py
PYTHONPATH=. python demos/demo_rag_document_ingestion.py
PYTHONPATH=. python demos/demo_rag_multiquery.py
PYTHONPATH=. python demos/demo_rag_with_verification.py
```

All demos require no configuration - they use sensible defaults and work immediately.

## Demo Overview

### 1. Simple Q&A (Beginner)
**File**: `demo_rag_qa_simple.py`
**Lines**: ~70
**Purpose**: Learn the absolute simplest RAG workflow

This is the best starting point if you're new to SimpleRAG. It demonstrates:
- Zero-config initialization
- Single document ingestion
- Basic query and result display
- Understanding RAGResult structure (response, sources, confidence)

**Concepts introduced**:
```python
async with SimpleRAG() as rag:
    await rag.ingest(content)
    result = await rag.query(question)
    print(result.response)
    print(f"Confidence: {result.confidence:.2f}")
```

**What you'll learn**:
- How to initialize SimpleRAG
- Basic ingest/query pattern
- Result structure and fields
- Simple output formatting

**Time**: ~30 seconds
**Next**: Move to demo 2

---

### 2. Document Ingestion (Intermediate)
**File**: `demo_rag_document_ingestion.py`
**Lines**: ~100
**Purpose**: Handle multiple documents with progress tracking

This demo extends demo 1 to show realistic document management:
- Ingest multiple documents (4-5 items)
- Track progress during ingestion
- Get system metrics (memory count, cache stats)
- Query across entire knowledge base
- Understand source attribution

**Concepts introduced**:
```python
documents = [
    {"title": "Doc 1", "content": "..."},
    {"title": "Doc 2", "content": "..."},
]

for doc in documents:
    await rag.ingest(doc['content'])

metrics = rag.get_metrics()
print(f"Total memories: {metrics['n_memories']}")

result = await rag.query(question, max_sources=3)
print(f"Retrieved {len(result.sources)} sources")
```

**What you'll learn**:
- Batch ingestion patterns
- Progress tracking
- System metrics and monitoring
- Source attribution (which documents helped answer)
- Understanding max_sources parameter

**Time**: ~45 seconds
**Prerequisites**: Completed demo 1
**Next**: Move to demo 3

---

### 3. Multi-Query Research (Advanced)
**File**: `demo_rag_multiquery.py`
**Lines**: ~120
**Purpose**: Research workflow with batch query processing

This demo shows efficient research patterns with `batch_query()`:
- Build comprehensive knowledge base (~10 items)
- Define research questions (4-5 related questions)
- Process all questions efficiently with `batch_query()`
- Compare results across queries
- Aggregate statistics

**Concepts introduced**:
```python
knowledge = [
    "Thompson Sampling is...",
    "Multi-armed bandits...",
    # ... more items
]

for item in knowledge:
    await rag.ingest(item)

questions = [
    "What is Thompson Sampling?",
    "What are advantages?",
    "What are applications?",
]

results = await rag.batch_query(questions)

for result in results:
    print(f"Confidence: {result.confidence:.2f}")

# Aggregate statistics
avg_confidence = sum(r.confidence for r in results) / len(results)
total_sources = sum(len(r.sources) for r in results)
```

**What you'll learn**:
- Building comprehensive knowledge bases
- Batch query efficiency
- Aggregating results from multiple queries
- Calculating statistics across results
- Understanding batch vs. sequential processing

**Time**: ~1 minute
**Prerequisites**: Completed demos 1-2
**Next**: Move to demo 4

---

### 4. Verification Modes (Expert)
**File**: `demo_rag_with_verification.py`
**Lines**: ~140
**Purpose**: Understand reasoning modes and confidence levels

This demo compares different reasoning strategies:
- Three modes: "direct", "verify", "research"
- Compare latency, confidence, and sources
- Show when to use each mode
- Understand mode selection tradeoffs

**Concepts introduced**:
```python
modes = ["direct", "verify", "research"]

for mode in modes:
    result = await rag.query(question, mode=mode)
    print(f"Mode: {mode}")
    print(f"  Latency: {result.metadata.get('latency_ms')}ms")
    print(f"  Confidence: {result.confidence:.2f}")
    print(f"  Sources: {len(result.sources)}")
```

**Mode Selection Guide**:

| Mode | Latency | Confidence | Sources | Best For |
|------|---------|-----------|---------|----------|
| **direct** | ~150ms | Low (50-70%) | Few | Fast factual queries |
| **verify** | ~600ms | Medium (70-85%) | Moderate | Balanced (default) |
| **research** | ~900ms | High (85-95%) | Many | Open-ended questions |

**What you'll learn**:
- Different reasoning modes and tradeoffs
- Confidence vs. latency tradeoff
- Mode selection based on use case
- When verification adds value
- Production recommendations

**Time**: ~1-2 minutes
**Prerequisites**: Completed demos 1-3
**Next**: Build your own RAG application

---

## Learning Path

```
Demo 1: Simple Q&A
    ↓ Learn basics
Demo 2: Document Ingestion
    ↓ Learn batch operations
Demo 3: Multi-Query Research
    ↓ Learn efficiency
Demo 4: Verification Modes
    ↓ Learn tradeoffs
Build Your Own RAG App
```

**Estimated time**: ~5-10 minutes for all 4 demos

## Common Patterns

### Pattern 1: Simple Ingestion
```python
async with SimpleRAG() as rag:
    await rag.ingest("content")
    result = await rag.query("question")
```

### Pattern 2: Batch Ingestion
```python
async with SimpleRAG() as rag:
    for doc in documents:
        await rag.ingest(doc)
    result = await rag.query("question")
```

### Pattern 3: Batch Queries
```python
async with SimpleRAG() as rag:
    # Ingest first
    await rag.ingest(content)

    # Query multiple questions at once
    results = await rag.batch_query(questions)
```

### Pattern 4: Mode Comparison
```python
async with SimpleRAG() as rag:
    await rag.ingest(content)

    # Try different modes
    direct = await rag.query(q, mode="direct")
    verify = await rag.query(q, mode="verify")
    research = await rag.query(q, mode="research")
```

## API Quick Reference

### Initialization
```python
# Default (zero-config)
async with SimpleRAG() as rag:
    ...

# With custom config
from hololoom.config import Config
config = Config.fast()
async with SimpleRAG(config=config) as rag:
    ...
```

### Ingestion
```python
# Single item
await rag.ingest("text content")
await rag.ingest({"structured": "data"})
await rag.ingest("path/to/file.pdf")

# Batch (loop pattern)
for item in items:
    await rag.ingest(item)
```

### Querying
```python
# Single query
result = await rag.query("question")

# With options
result = await rag.query(
    "question",
    mode="verify",      # "direct", "verify", "research"
    max_sources=5       # limit retrieved sources
)

# Batch queries
results = await rag.batch_query([
    "question 1",
    "question 2",
    "question 3",
])
```

### Results
```python
result = await rag.query("What is X?")

# Fields
result.response           # LLM-generated answer
result.sources           # List of retrieved documents
result.confidence        # 0.0-1.0 score
result.reasoning_mode    # "direct", "verify", "research"
result.metadata          # Dict with latency, cache_hit, etc.
```

### Monitoring
```python
# Get system metrics
metrics = rag.get_metrics()
print(f"Memories: {metrics['n_memories']}")
print(f"Cache size: {metrics['cache_size']}")
print(f"Cache hit rate: {metrics['cache_hit_rate']:.1%}")

# Get summary
print(rag.summary())

# Clear cache if needed
rag.clear_cache()
```

## Real-World Examples

### Use Case 1: Documentation Q&A
```python
async with SimpleRAG() as rag:
    # Ingest API docs
    for doc in api_documentation:
        await rag.ingest(doc)

    # Answer user questions
    result = await rag.query(user_question, mode="verify")
    print(result.response)
    print(f"Sources: {len(result.sources)}")
```

### Use Case 2: Knowledge Base Search
```python
async with SimpleRAG() as rag:
    # Load knowledge base
    for article in knowledge_articles:
        await rag.ingest(article)

    # Research workflow
    results = await rag.batch_query([
        "What is X?",
        "How does X work?",
        "Examples of X?",
    ])

    # Aggregate
    for i, result in enumerate(results):
        print(f"Question {i+1}: Confidence {result.confidence:.2f}")
```

### Use Case 3: Contract Analysis
```python
async with SimpleRAG() as rag:
    # Ingest contract
    contract = open("contract.txt").read()
    await rag.ingest(contract)

    # Detailed verification mode
    result = await rag.query(
        "What are payment terms?",
        mode="verify"
    )

    print(result.response)
    print(f"Verified: {result.confidence > 0.8}")
```

## Troubleshooting

### Error: "SimpleRAG not initialized"
**Cause**: Using RAG outside of `async with` block

**Fix**:
```python
# Wrong
rag = SimpleRAG()
await rag.ingest(content)  # Error!

# Correct
async with SimpleRAG() as rag:
    await rag.ingest(content)
```

### Error: "No results found"
**Cause**: Knowledge base is empty or queries don't match content

**Fix**:
```python
# Make sure to ingest first
async with SimpleRAG() as rag:
    await rag.ingest("Thompson Sampling is...")
    result = await rag.query("What is Thompson Sampling?")
    # Now should have results
```

### Low confidence scores
**Cause**: Using "direct" mode or insufficient sources

**Fix**:
```python
# Use verify mode instead
result = await rag.query(question, mode="verify")

# Or research for even higher confidence
result = await rag.query(question, mode="research")
```

## Next Steps

1. **Run all 4 demos**: Follow the learning path above
2. **Read SimpleRAG API docs**: See `hololoom/rag/simple_rag.py`
3. **Build your own**: Create a RAG app for your domain
4. **Advanced features**: Explore caching, metrics, custom configs
5. **Integration**: Use SimpleRAG in larger applications

## Documentation Links

- **SimpleRAG API**: `hololoom/rag/simple_rag.py`
- **HoloLoom Memory**: `hololoom/hololoom.py`
- **Configuration**: `hololoom/config.py`
- **Complete Examples**: `demos/demo_simple_rag.py` (full-featured demo)

## Performance Characteristics

| Operation | Typical Time |
|-----------|--------------|
| Initialize RAG | ~500ms |
| Ingest document | ~100-200ms per doc |
| Query (direct) | ~150ms |
| Query (verify) | ~600ms |
| Batch query 4 items | ~800ms total |
| Cache hit | ~10ms |

Times vary based on document size, system load, and LLM backend availability.

## Tips for Best Results

1. **Use clear, specific questions**: "What is Thompson Sampling?" works better than "tell me stuff"
2. **Ingest relevant content first**: RAG quality depends on knowledge base quality
3. **Use batch_query for research**: More efficient than sequential queries
4. **Choose mode strategically**: Direct for speed, verify for balance, research for depth
5. **Monitor metrics**: Check memory count and cache hit rate
6. **Cache warm-up**: First query in session is slower; subsequent queries use cache

## Contributing

To add a new demo:
1. Create `demo_rag_[name].py`
2. Keep it under 100 lines
3. Follow the pattern in existing demos
4. Add to this README
5. Ensure it runs standalone: `PYTHONPATH=. python demos/demo_rag_[name].py`

## Questions?

See the docstrings in each demo file - they contain detailed explanations of every step.
