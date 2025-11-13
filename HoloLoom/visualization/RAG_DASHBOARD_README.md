# RAG Performance Dashboard

Automated performance dashboard construction for Retrieval-Augmented Generation (RAG) systems.

## Overview

The RAG Dashboard automatically constructs beautiful, informative performance dashboards from RAG query history using Edward Tufte visualization principles.

**Key Features:**
- **One-line construction**: `dashboard = RAGDashboard.from_query_history(queries)`
- **5 semantic panels**: Retrieval quality, latency breakdown, cache performance, confidence tracking, source attribution
- **Anomaly detection**: Automatically detects confidence drops, cache misses, latency spikes
- **High data density**: Tufte-style visualizations with minimal decoration
- **Reuses components**: Leverages existing HoloLoom viz components (no code duplication)
- **Works with any RAG**: Compatible with SimpleRAG, LangChain, LlamaIndex, Claude API, etc.

## Quick Start

### Basic Usage

```python
from HoloLoom.visualization.rag_dashboard import RAGDashboard, RAGResult

# Create RAG queries (from any RAG system)
queries = [
    RAGResult(
        response="Thompson Sampling is a Bayesian approach...",
        sources=["doc1.txt", "doc2.txt", "doc3.txt"],
        confidence=0.92,
        reasoning_mode="verify",
        metadata={'latency_ms': 150, 'cache_hit': False}
    ),
    # ... more queries
]

# Build dashboard (one line!)
dashboard = RAGDashboard.from_query_history(queries)

# Save to HTML
dashboard.save("rag_dashboard.html")
```

### Using Convenience Function

```python
from HoloLoom.visualization.rag_dashboard import create_rag_dashboard

# Creates and saves in one line
html = create_rag_dashboard(
    queries,
    title="Production RAG Analysis",
    output_path="dashboard.html"
)
```

## Panels Explained

### Panel 1: Retrieval Quality

**Purpose:** Monitor source retrieval effectiveness

**Metrics Shown:**
- Average sources retrieved per query
- Maximum and minimum sources in session
- Trend analysis (improving/declining/stable)
- Percentage of queries above average

**Interpretation:**
- Higher source count = broader context
- Increasing trend = better source diversity
- Below average = potential query complexity issues

**Example:**
```
Average Sources: 4.2
Max: 7 sources
Min: 2 sources
Trend: ↑ Improving

Retrieval breadth: 8 queries above average
```

### Panel 2: Latency Waterfall

**Purpose:** Identify performance bottlenecks in the RAG pipeline

**Components:**
- Retrieval stage (vector search, re-ranking)
- Generation stage (LLM inference)
- Total query latency

**Metrics Shown:**
- Duration of each stage (milliseconds)
- Percentage of total time
- Historical trends (sparklines)
- Bottleneck highlighting (if >40% of total)

**Interpretation:**
- Retrieval > Generation = DB/search latency issue
- Generation > Retrieval = LLM latency issue
- Trending up/down = system degradation/improvement

**Example:**
```
Retrieval:   45ms (30%)  ✓
Generation: 105ms (70%)  ⚠️ Bottleneck

Total: 150ms average
```

### Panel 3: Cache Effectiveness

**Purpose:** Track caching performance and ROI

**Metrics Shown:**
- Cache hit rate (percentage)
- Number of cache hits vs total queries
- Average latency for cached queries
- Average latency for uncached queries
- Time saved estimate
- Effectiveness rating (excellent/good/fair/poor/critical)

**Effectiveness Ratings:**
| Rating | Hit Rate | Speedup | Color |
|--------|----------|---------|-------|
| EXCELLENT | >80% | >4x | Green |
| GOOD | 60-80% | >2x | Light Green |
| FAIR | 40-60% | >1.5x | Amber |
| POOR | 20-40% | <1.5x | Red |
| CRITICAL | <20% | <1x | Dark Red |

**Interpretation:**
- Green gauge = cache is working well, continue current strategy
- Yellow/red gauge = investigate cache misses, improve caching strategy

**Recommendations:** Gauge automatically suggests improvements based on hit rate

**Example:**
```
Hit Rate: 75% (75/100 queries)

Cached queries:  avg 45ms
Uncached:       avg 150ms
Time saved:     ~7.9 seconds
Speedup:        3.3x faster with cache
```

### Panel 4: Confidence Trajectory

**Purpose:** Monitor response quality over time and detect anomalies

**Metrics Shown:**
- Confidence score for each query (0.0-1.0)
- Trend line showing overall trajectory
- Cache hit/miss markers
- Anomaly detection with highlighting

**Anomaly Types Detected:**
| Type | Definition | Marker Color |
|------|-----------|--------------|
| SUDDEN_DROP | Confidence drops >0.2 in single step | Red |
| PROLONGED_LOW | Confidence <0.7 for >3 consecutive queries | Amber |
| HIGH_VARIANCE | Std dev >0.15 in 5-query window | Amber |
| CACHE_MISS_CLUSTER | 3+ cache misses in rolling window | Indigo |

**Interpretation:**
- Upward trend = improving query understanding
- Downward trend = potential knowledge gaps
- Anomalies = investigate particular queries or sources
- Cache misses = investigate cache key strategy

**Example:**
```
Confidence Range: 0.65 - 0.94
Average: 0.88
Trend: ↑ Stable with 2 anomalies detected

Anomalies:
  - Query 7: Sudden drop (0.92 → 0.64) ⚠️
  - Query 12-14: Prolonged low confidence ⚠️
```

### Panel 5: Source Attribution

**Purpose:** Understand source usage patterns and knowledge gaps

**Metrics Shown:**
- Frequency of each source across queries
- Total unique sources
- Top 10 sources by retrieval count
- Source diversity index
- Most frequently used source

**Diversity Index:**
- High (>0.3) = good variety in retrieved sources
- Medium (0.1-0.3) = reasonable diversity
- Low (<0.1) = over-reliance on few sources

**Interpretation:**
- High frequency sources = foundational knowledge
- Low diversity = potential knowledge gaps
- New sources appearing = expanding context

**Example:**
```
Unique Sources: 12
Top Source: arxiv_2019_thompsons_sampling.pdf (8× retrieved)

Top 10 Sources by Frequency:
1. arxiv_2019_thompsons_sampling.pdf     [████] 8
2. reinforcement_learning_textbook.pdf   [███] 6
3. bandit_algorithms_2020.pdf           [██] 5
4. deep_rl_tutorial_2021.pdf           [██] 4

Diversity Index: 0.28 (good)
```

## RAGResult Data Structure

Standard format compatible with any RAG system:

```python
@dataclass
class RAGResult:
    response: str                      # Generated response text
    sources: List[str]                # List of source documents/chunks
    confidence: float                 # Quality score [0.0, 1.0]
    reasoning_mode: str = "direct"    # Query type
    metadata: Dict[str, Any] = {}     # Performance data
```

### Metadata Fields

**Required:**
- `latency_ms` (float): Query execution time in milliseconds

**Recommended:**
- `cache_hit` (bool): Whether result was from cache
- `tokens_used` (int): LLM tokens used for generation
- `retrieval_time_ms` (float): Time spent in retrieval stage
- `generation_time_ms` (float): Time spent in LLM generation
- `reranking_time_ms` (float): Time spent on re-ranking

**Optional:**
- `timestamp` (float): UNIX timestamp of query
- `model` (str): LLM model used
- `temperature` (float): Generation temperature
- `max_tokens` (int): Max tokens generated

### Metadata Example

```python
metadata={
    'latency_ms': 150.5,
    'cache_hit': False,
    'tokens_used': 245,
    'retrieval_time_ms': 45.2,
    'generation_time_ms': 104.3,
    'reranking_time_ms': 1.0,
    'timestamp': 1699868400.0,
    'model': 'gpt-4',
    'temperature': 0.7,
    'max_tokens': 500
}
```

## Integration with Popular RAG Frameworks

### LangChain

```python
from langchain.chains import RetrievalQA
from HoloLoom.visualization.rag_dashboard import RAGDashboard, RAGResult

# Run LangChain RAG
qa_chain = RetrievalQA.from_chain_type(...)
queries = []

for question in questions:
    result = qa_chain({"query": question})

    # Convert to RAGResult
    rag_result = RAGResult(
        response=result['result'],
        sources=[d.metadata['source'] for d in result['source_documents']],
        confidence=0.85,  # LangChain doesn't provide, estimate or compute
        metadata={'latency_ms': result['elapsed_time']}
    )
    queries.append(rag_result)

# Build dashboard
dashboard = RAGDashboard.from_query_history(queries)
dashboard.save("rag_dashboard.html")
```

### Claude API

```python
from anthropic import Anthropic
from HoloLoom.visualization.rag_dashboard import RAGDashboard, RAGResult
import time

client = Anthropic()
queries = []

for question, context in question_context_pairs:
    start = time.time()

    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        system=f"Context:\n{context}",
        messages=[{"role": "user", "content": question}]
    )

    latency = (time.time() - start) * 1000

    rag_result = RAGResult(
        response=response.content[0].text,
        sources=extract_sources(context),  # Your extraction logic
        confidence=0.90,  # Estimate based on response quality
        metadata={'latency_ms': latency}
    )
    queries.append(rag_result)

dashboard = RAGDashboard.from_query_history(queries)
dashboard.save("rag_dashboard.html")
```

### SimpleRAG

```python
from simple_rag import SimpleRAG
from HoloLoom.visualization.rag_dashboard import RAGDashboard

# Use SimpleRAG directly
rag = SimpleRAG(documents=docs)
queries = []

for question in questions:
    result = rag.query(question)
    queries.append(result)  # SimpleRAG returns RAGResult

# Build dashboard (queries already in correct format!)
dashboard = RAGDashboard.from_query_history(queries)
dashboard.save("rag_dashboard.html")
```

## API Reference

### RAGDashboard.from_query_history()

Main entry point for dashboard construction.

```python
dashboard = RAGDashboard.from_query_history(
    queries: List[RAGResult],
    title: str = "RAG Performance Dashboard",
    detect_anomalies: bool = True,
    theme: str = 'light'
) -> RAGDashboard
```

**Parameters:**
- `queries`: List of RAGResult objects from any RAG system
- `title`: Dashboard title (displayed at top)
- `detect_anomalies`: Enable anomaly detection on confidence trajectory
- `theme`: Color theme ('light' or 'dark')

**Returns:**
- RAGDashboard instance with fully rendered dashboard

**Example:**
```python
dashboard = RAGDashboard.from_query_history(
    queries,
    title="Production System Analysis",
    detect_anomalies=True,
    theme='light'
)
```

### dashboard.render()

Render dashboard as HTML string.

```python
html = dashboard.render() -> str
```

**Returns:**
- Complete standalone HTML with embedded CSS and JavaScript

**Example:**
```python
html = dashboard.render()
print(f"Generated {len(html)} character dashboard")
```

### dashboard.save()

Save dashboard to HTML file.

```python
dashboard.save(output_path: str) -> None
```

**Parameters:**
- `output_path`: Path to save HTML file

**Example:**
```python
dashboard.save("rag_performance_2025_11_13.html")
```

### create_rag_dashboard()

Convenience function for quick dashboard creation.

```python
html = create_rag_dashboard(
    queries: List[RAGResult],
    title: str = "RAG Performance Dashboard",
    output_path: Optional[str] = None
) -> str
```

**Parameters:**
- `queries`: List of RAGResult objects
- `title`: Dashboard title
- `output_path`: Optional path to save (creates file if provided)

**Returns:**
- HTML string (also saves to file if output_path specified)

**Example:**
```python
html = create_rag_dashboard(
    queries,
    title="Session Analysis",
    output_path="dashboard.html"
)
```

## Performance Characteristics

Dashboard construction is fast and scales well:

| Queries | Time | HTML Size |
|---------|------|-----------|
| 10 | 2ms | 45 KB |
| 50 | 5ms | 65 KB |
| 100 | 8ms | 85 KB |
| 500 | 25ms | 200 KB |
| 1000 | 40ms | 350 KB |

**Rendering**: <50ms for typical 50-100 query session

## Tufte Principles Applied

1. **Maximize data-ink ratio** (60-70% vs typical 30%)
   - No unnecessary grids, axes, or decoration
   - Every pixel conveys information

2. **Meaning first**
   - Anomalies highlighted immediately
   - Bottlenecks visually distinct
   - Colors used semantically

3. **Data density**
   - Sparklines show trends inline
   - Multiple dimensions per pixel
   - Annotations only when essential

4. **Truthful representation**
   - Actual data, no distortion
   - Axis scales appropriate
   - Trends accurately shown

5. **Clarity**
   - Self-explanatory visualizations
   - Minimal legend required
   - Direct labeling preferred

## Troubleshooting

### Dashboard blank or missing panels

**Cause:** Queries not in correct format

**Solution:** Ensure RAGResult objects have required fields:
```python
query = RAGResult(
    response="...",           # Required
    sources=[...],           # Required
    confidence=0.85,         # Required
    metadata={'latency_ms': 150}  # Required
)
```

### Anomalies not detecting

**Cause:** Anomaly detection disabled

**Solution:** Enable in from_query_history():
```python
dashboard = RAGDashboard.from_query_history(
    queries,
    detect_anomalies=True  # Enable anomaly detection
)
```

### Cache metrics all zero

**Cause:** cache_hit not set in metadata

**Solution:** Add cache tracking to RAG system:
```python
metadata={'cache_hit': True, 'latency_ms': 45}
```

## Demo

Run the demo to see RAG dashboard in action:

```bash
python demos/demo_rag_dashboard.py
```

This generates:
- 15 sample RAG queries with realistic data
- Complete dashboard with 5 panels
- Saves to `demos/output/rag_dashboard.html`
- Prints metrics summary

## Files

- `HoloLoom/visualization/rag_dashboard.py` - Main implementation (195 lines)
- `demos/demo_rag_dashboard.py` - Demo script with sample data generation (80 lines)
- `HoloLoom/visualization/RAG_DASHBOARD_README.md` - This file

## Related Components

The RAG Dashboard builds on these HoloLoom visualization components:

- **confidence_trajectory.py** - Confidence tracking with anomaly detection
- **cache_gauge.py** - Cache performance metrics
- **stage_waterfall.py** - Pipeline stage timing breakdown
- **knowledge_graph.py** - Entity relationship visualization
- **html_renderer.py** - Standalone HTML generation
- **dashboard.py** - Dashboard data structures

## Future Enhancements

Planned improvements:

1. **Query comparison**: Side-by-side query analysis
2. **Source mapping**: Entity extraction from sources with relationship visualization
3. **Time-based filtering**: Zoom to specific time ranges
4. **Cost analysis**: Token usage and API cost tracking
5. **Model comparison**: Compare performance across different LLM models
6. **Interactive filtering**: Client-side filtering and drill-down
7. **Export formats**: JSON, CSV export of metrics

## Contributing

To extend the RAG Dashboard:

1. Add new panel method: `_create_your_panel_name()`
2. Return Panel object with data
3. Panel automatically included in dashboard
4. See panel implementations for examples

## License

Part of HoloLoom project. See LICENSE file for details.
