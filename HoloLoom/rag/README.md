# SimpleRAG - Zero-Config RAG for HoloLoom

**Retrieval-Augmented Generation** made simple. A clean wrapper around HoloLoom's memory system + LLM integration.

## Philosophy

RAG doesn't need to be complicated. SimpleRAG hides all complexity:
- **Ingest**: Any modality (text, PDF, image, etc.)
- **Query**: Single-line interface with structured results
- **Results**: Response + sources + confidence + metadata

Three lines to get started:
```python
async with SimpleRAG() as rag:
    await rag.ingest("Thompson Sampling balances exploration/exploitation")
    result = await rag.query("What is Thompson Sampling?")
    print(result.response)
```

## Features

- **Zero-config initialization** - Works out of the box with `SimpleRAG()`
- **Multimodal ingestion** - Text, PDF, image, audio, structured data
- **LLM integration** - Ollama, Anthropic, OpenAI support
- **Query caching** - Repeated queries use cache (100x faster)
- **Graceful degradation** - Works without LLM (fallback to neural-only)
- **Structured results** - Response, sources, confidence, metadata
- **Batch processing** - Query multiple questions efficiently
- **System metrics** - Monitor memory count, cache hit rate, etc.

## Quick Start

### Installation

SimpleRAG is included in HoloLoom. Just import:

```python
from HoloLoom.rag import SimpleRAG
```

### Basic Usage

```python
import asyncio
from HoloLoom.rag import SimpleRAG

async def main():
    async with SimpleRAG() as rag:
        # Ingest content
        await rag.ingest("Thompson Sampling balances exploration/exploitation")
        await rag.ingest("It's used in multi-armed bandit problems")

        # Query
        result = await rag.query("What is Thompson Sampling?")

        # Use result
        print(result.response)              # LLM-generated answer
        print(f"Sources: {len(result.sources)}")  # Number of retrieved sources
        print(f"Confidence: {result.confidence:.1%}")

asyncio.run(main())
```

### Multimodal Ingestion

SimpleRAG delegates to HoloLoom's multimodal input router:

```python
async with SimpleRAG() as rag:
    # Text
    await rag.ingest("Thompson Sampling uses Bayesian statistics")

    # Structured data
    await rag.ingest({
        "algorithm": "Thompson Sampling",
        "type": "Bayesian bandit",
        "year": 1933
    })

    # File paths
    await rag.ingest("document.pdf")
    await rag.ingest("diagram.png")

    # Any other format (with graceful fallback)
    await rag.ingest(some_custom_object)
```

## API Reference

### SimpleRAG Class

Main RAG interface.

#### Initialization

```python
rag = SimpleRAG(
    config: Optional[Config] = None,          # System config (defaults: fast)
    llm_provider: str = "ollama",             # "ollama", "anthropic", "openai"
    llm_model: Optional[str] = None,          # Specific model (uses defaults)
    enable_caching: bool = True               # Enable query result caching
)
```

**Parameters:**
- `config`: HoloLoom configuration (BARE/FAST/FUSED modes)
  - Default: `Config.fast()` - good balance
  - Options: `Config.bare()` (speed), `Config.fused()` (quality)
- `llm_provider`: Which LLM to use
  - `"ollama"` - Local Ollama instance (default, free)
  - `"anthropic"` - Anthropic Claude (requires API key)
  - `"openai"` - OpenAI GPT (requires API key)
- `llm_model`: Optional model override (uses provider defaults if None)
- `enable_caching`: Cache query results for repeated queries

#### Context Manager

Use as async context manager (required):

```python
async with SimpleRAG() as rag:
    # Use rag here
    await rag.ingest(...)
    result = await rag.query(...)
    # Automatic cleanup on exit
```

#### Methods

##### `ingest(content: Any) -> None`

Add content to the knowledge base. Supports any modality through HoloLoom's input router.

```python
await rag.ingest("Thompson Sampling is a Bayesian algorithm")
await rag.ingest("document.pdf")
await rag.ingest({"field": "value"})
```

**Parameters:**
- `content`: Text string, file path, bytes, or structured data

**Raises:**
- `RuntimeError`: If not used in async context manager

##### `query(question: str, mode: str = "verify", max_sources: int = 5, use_cache: bool = True) -> RAGResult`

Query the knowledge base with retrieval + LLM generation.

```python
result = await rag.query(
    "What is Thompson Sampling?",
    mode="verify",           # Reasoning mode
    max_sources=5,          # Max sources to retrieve
    use_cache=True          # Use cache if available
)

print(result.response)              # LLM answer
print(result.sources)               # Retrieved sources
print(result.confidence)             # 0.0-1.0
print(result.reasoning_mode)        # "verify"
print(result.metadata)              # {"cache_hit": False, ...}
```

**Parameters:**
- `question`: Query text
- `mode`: Reasoning mode
  - `"direct"` - Single-pass answer (default)
  - `"verify"` - Answer with verification
  - `"research"` - Multi-query exploration
  - `"plan_execute"` - Goal decomposition
- `max_sources`: Maximum number of retrieved sources (1-10, default: 5)
- `use_cache`: Use cached results for identical queries

**Returns:**
- `RAGResult` with:
  - `response: str` - LLM-generated answer
  - `sources: List[str]` - Retrieved source texts
  - `confidence: float` - 0.0-1.0 confidence score
  - `reasoning_mode: str` - Mode used
  - `metadata: Dict` - Additional info (cache_hit, latency_ms, etc.)

##### `batch_query(questions: List[str], mode: str = "verify", max_sources: int = 5) -> List[RAGResult]`

Query multiple questions efficiently.

```python
questions = [
    "What is Thompson Sampling?",
    "How does it work?",
    "What are the tradeoffs?"
]

results = await rag.batch_query(questions, mode="verify")

for result in results:
    print(f"{result.confidence:.1%} {result.response[:50]}")
```

**Parameters:**
- `questions`: List of query strings
- `mode`: Reasoning mode (applied to all queries)
- `max_sources`: Max sources per query

**Returns:**
- List of `RAGResult` objects (same length as `questions`)

##### `get_metrics() -> Dict[str, Any]`

Get system metrics for monitoring.

```python
metrics = rag.get_metrics()

print(f"Memories: {metrics['n_memories']}")
print(f"Cache size: {metrics['cache_size']}")
print(f"Cache hit rate: {metrics['cache_hit_rate']:.1%}")
print(f"LLM available: {metrics['llm_available']}")
```

**Returns:**
- Dictionary with metrics:
  - `n_memories`: Total memories stored
  - `n_connections`: Total connections
  - `cache_size`: Cached queries
  - `cache_hit_rate`: Ratio of cache hits
  - `llm_provider`: LLM provider name
  - `llm_available`: Whether LLM is available

##### `clear_cache() -> None`

Clear the query cache.

```python
rag.clear_cache()
```

##### `summary() -> str`

Get human-readable system summary.

```python
print(rag.summary())
```

Output:
```
SimpleRAG System
===============
Memories: 42
Cache size: 5
Cache hit rate: 20.0%
LLM provider: ollama
LLM available: True
```

### RAGResult Class

Structured result from a query.

```python
@dataclass
class RAGResult:
    response: str                          # LLM-generated answer
    sources: List[str]                     # Retrieved sources
    confidence: float                      # 0.0-1.0
    reasoning_mode: str = "direct"        # Mode used
    metadata: Dict[str, Any] = {}         # Additional info
```

**Attributes:**
- `response`: The LLM-generated answer
- `sources`: List of source texts that were retrieved
- `confidence`: Confidence score (0.0-1.0)
- `reasoning_mode`: Reasoning mode used for this query
- `metadata`: Additional information:
  - `cache_hit: bool` - Whether result came from cache
  - `n_sources: int` - Number of sources retrieved
  - `llm_provider: str` - Which LLM was used
  - `latency_ms: float` - Query latency

**Methods:**
- `__str__()` - Pretty-print the result

```python
result = await rag.query("What is Thompson Sampling?")
print(result)  # Formatted output
```

## Configuration

### Execution Modes

```python
from HoloLoom.config import Config
from HoloLoom.rag import SimpleRAG

# Speed-optimized (fastest)
rag = SimpleRAG(config=Config.bare())

# Balanced (default)
rag = SimpleRAG(config=Config.fast())

# Quality-optimized (slowest)
rag = SimpleRAG(config=Config.fused())
```

### LLM Providers

#### Ollama (Local, Free)

```python
rag = SimpleRAG(llm_provider="ollama")

# Make sure Ollama is running:
# ollama serve
```

Requires Ollama installed. Download at https://ollama.ai

#### Anthropic (Claude)

```python
import os
os.environ["ANTHROPIC_API_KEY"] = "sk-..."

rag = SimpleRAG(llm_provider="anthropic")
```

Requires API key. Get at https://console.anthropic.com

#### OpenAI (GPT)

```python
import os
os.environ["OPENAI_API_KEY"] = "sk-..."

rag = SimpleRAG(llm_provider="openai")
```

Requires API key. Get at https://platform.openai.com

## Examples

### Example 1: Document Q&A

```python
async def document_qa():
    async with SimpleRAG() as rag:
        # Ingest document
        with open("manual.pdf", "r") as f:
            await rag.ingest(f.read())

        # Q&A session
        while True:
            question = input("Q: ")
            result = await rag.query(question)
            print(f"A: {result.response}\n")
```

### Example 2: Knowledge Base Search

```python
async def knowledge_search():
    async with SimpleRAG() as rag:
        # Load knowledge base
        knowledge = [
            "Thompson Sampling is Bayesian",
            "It balances exploration/exploitation",
            "Used in multi-armed bandits"
        ]

        for fact in knowledge:
            await rag.ingest(fact)

        # Search
        result = await rag.query("How does Thompson Sampling work?")

        print(f"Answer: {result.response}")
        print(f"Sources: {result.sources}")
        print(f"Confidence: {result.confidence:.1%}")
```

### Example 3: Batch Processing

```python
async def batch_analysis():
    async with SimpleRAG() as rag:
        # Ingest dataset
        dataset = ["fact1", "fact2", "fact3", ...]
        for fact in dataset:
            await rag.ingest(fact)

        # Batch analyze
        questions = ["Q1", "Q2", "Q3", ...]
        results = await rag.batch_query(questions)

        # Process results
        for question, result in zip(questions, results):
            print(f"{question}: {result.response}")
```

### Example 4: Monitoring and Metrics

```python
async def monitor_rag():
    async with SimpleRAG() as rag:
        await rag.ingest("Thompson Sampling uses Bayesian methods")

        for i in range(100):
            result = await rag.query("Tell me about Thompson Sampling")

            if i % 10 == 0:
                metrics = rag.get_metrics()
                print(f"Step {i}: Cache hit rate = {metrics['cache_hit_rate']:.1%}")
```

## Comparison: Direct Orchestrator vs SimpleRAG

### Using WeavingOrchestrator Directly

```python
from HoloLoom.weaving_orchestrator import WeavingOrchestrator
from HoloLoom.documentation.types import Query, MemoryShard

async def direct_usage():
    config = Config.fast()

    shards = [
        MemoryShard(id="1", text="Thompson Sampling content", entities=[]),
        MemoryShard(id="2", text="More about Thompson Sampling", entities=[])
    ]

    async with WeavingOrchestrator(cfg=config, shards=shards) as orchestrator:
        spacetime = await orchestrator.weave(Query(text="What is Thompson Sampling?"))
        print(spacetime.response)
```

### Using SimpleRAG

```python
from HoloLoom.rag import SimpleRAG

async def rag_usage():
    async with SimpleRAG() as rag:
        await rag.ingest("Thompson Sampling content")
        await rag.ingest("More about Thompson Sampling")

        result = await rag.query("What is Thompson Sampling?")
        print(result.response)
```

**SimpleRAG advantages:**
- ✓ No need to create MemoryShard objects
- ✓ No Config import needed
- ✓ Simpler query API (takes string, returns RAGResult)
- ✓ Built-in caching
- ✓ Better error messages
- ✓ 50% less code

## Graceful Degradation

SimpleRAG degrades gracefully when components are unavailable:

### Without LLM

If no LLM is available (Ollama not running, no API key, etc.):
- Still works! Falls back to neural-only synthesis from sources
- Response is created from retrieved sources
- Confidence is lower but system remains functional

```python
async with SimpleRAG() as rag:
    await rag.ingest("Thompson Sampling is a Bayesian algorithm")

    result = await rag.query("What is Thompson Sampling?")
    # Works even if Ollama/API unavailable
    print(result.response)  # Synthesized from sources
```

### Without Multimodal Support

If multimodal dependencies aren't installed:
- Ingest works for text only
- File paths fall back to reading as text
- Graceful error message if binary file used

## Performance

### Caching

Repeated queries are extremely fast:

```python
async with SimpleRAG() as rag:
    await rag.ingest("Thompson Sampling uses Bayesian statistics")

    # First query: ~150ms
    result1 = await rag.query("What is Thompson Sampling?")

    # Second query (identical): ~1ms (from cache)
    result2 = await rag.query("What is Thompson Sampling?")
```

Cache hit rate depends on:
- Query similarity (exact matches hit cache)
- Cache size (default: unlimited)
- Query parameters (mode, max_sources affect key)

### Optimization Tips

1. **Use Config.bare()** for speed-critical applications
   ```python
   rag = SimpleRAG(config=Config.bare())
   ```

2. **Enable caching** (default: True) for repeated queries
   ```python
   rag = SimpleRAG(enable_caching=True)
   ```

3. **Batch queries** instead of sequential
   ```python
   results = await rag.batch_query(questions)  # Better than loop
   ```

4. **Use "direct" mode** instead of "verify" for simple queries
   ```python
   result = await rag.query(question, mode="direct")
   ```

## Testing

Run tests:

```bash
# Unit tests (with mocks)
pytest HoloLoom/rag/tests/test_simple_rag.py::TestSimpleRAGInit -v

# Integration tests (real components)
pytest HoloLoom/rag/tests/test_simple_rag.py::test_full_ingest_and_query -v

# All tests
pytest HoloLoom/rag/tests/ -v
```

## Troubleshooting

### "SimpleRAG not initialized" error

Make sure you use async context manager:

```python
# ❌ Wrong
rag = SimpleRAG()
await rag.query("test")  # RuntimeError!

# ✓ Right
async with SimpleRAG() as rag:
    await rag.query("test")  # Works
```

### LLM not available

If you see "LLM orchestrator unavailable", either:
1. Start Ollama: `ollama serve`
2. Set API key: `export ANTHROPIC_API_KEY=...`
3. Or just use without LLM (fallback to neural-only)

### Slow first query

First query is slower because of cold cache and LLM initialization. Subsequent queries are much faster:
- First query: ~150ms
- Repeated queries: ~1ms (from cache)
- No LLM: ~30ms

### Out of memory

If ingesting large datasets:
1. Use `Config.bare()` for smaller memory footprint
2. Clear cache periodically: `rag.clear_cache()`
3. Query in batches instead of all at once

## Contributing

SimpleRAG is designed to be simple and elegant. When contributing:
- Keep API surface minimal
- Delegate to existing components
- Add tests for new functionality
- Maintain graceful degradation

## Multi-Agent RAG

For advanced use cases requiring consensus from multiple agents with diverse strategies, use `MultiAgentRAG`:

### Basic Usage

```python
from HoloLoom.rag import MultiAgentRAG

async def multi_agent_example():
    async with MultiAgentRAG(
        n_agents=5,
        consensus_method="confidence_weighted",
        agent_timeout=30.0
    ) as rag:
        # Ingest
        await rag.ingest("Thompson Sampling uses Bayesian statistics")

        # Query with multi-agent consensus
        result = await rag.query_multiagent(
            "What is Thompson Sampling?",
            explain_disagreement=True
        )

        # View results
        print(f"Consensus: {result.response}")
        print(f"Agreement: {result.agreement_score:.2f}")
        print(f"Confidence: {result.confidence:.2f}")

        # View individual agent responses
        for agent_resp in result.agent_responses:
            print(f"  {agent_resp.agent_id}: {agent_resp.confidence:.2f}")
```

### Consensus Methods

- **majority_vote**: Most common answer (simple, fast)
- **confidence_weighted**: Weight by agent confidence (default, precision-focused)
- **llm_judge**: Use LLM to select best or synthesize (highest quality, slowest)
- **ensemble**: Combine all responses into synthesized answer

### Agent Diversity

Agents automatically vary along multiple dimensions:
- **Retrieval parameters**: k (3, 5, 10), reranking (on/off)
- **Reasoning modes**: direct, verify, research, plan_execute
- **Embedding models**: Matryoshka, HuggingFace (if available)
- **Multi-hop**: max_hops (if available)
- **SQL**: Some agents enable SQL (if available)

### Features

- **Parallel execution**: All agents run concurrently (latency = max, not sum)
- **Timeout handling**: Slow agents killed after timeout
- **Partial failures**: Consensus works even if some agents fail
- **Agreement scoring**: Detect when agents disagree (0.0-1.0)
- **Disagreement detection**: Explain why agents disagree
- **Source deduplication**: Merge sources across agents
- **Performance tracking**: Latency per agent, consensus overhead

### Example: Consensus Comparison

```python
async def compare_consensus():
    for method in ["majority_vote", "confidence_weighted", "ensemble"]:
        async with MultiAgentRAG(
            n_agents=5,
            consensus_method=method
        ) as rag:
            await rag.ingest("Thompson Sampling is a Bayesian strategy")

            result = await rag.query_multiagent("What is Thompson Sampling?")

            print(f"{method}:")
            print(f"  Agreement: {result.agreement_score:.2f}")
            print(f"  Confidence: {result.confidence:.2f}")
            print(f"  Time: {result.consensus_metadata['consensus_time_ms']:.1f}ms")
```

### Performance

- **5 agents in parallel**: ~max(agent_latency), not 5× sum
- **Typical speedup**: 3-5× vs sequential execution
- **Agreement scoring**: O(N²) pairwise comparison (negligible for N≤10)
- **Overhead**: <10ms for consensus computation

### Demo

Run the multi-agent demo:

```bash
python demos/demo_rag_multiagent.py
```

This demonstrates:
- Simple query with 5 agents
- All agent responses side-by-side
- Consensus process visualization
- Agreement scores and disagreement detection
- Performance comparison (1 agent vs 5 agents)
- Different consensus methods comparison

### When to Use Multi-Agent

Use multi-agent RAG when:
- Query is controversial or ambiguous
- Need high confidence (multiple perspectives)
- Risk of bias from single strategy
- Want to detect uncertainty/disagreement
- Need robustness (fault tolerance)

Use single-agent RAG when:
- Query is simple/factual
- Latency is critical (<50ms)
- Cost is a concern (LLM API fees)
- Confidence is already high

## See Also

- **[WeavingOrchestrator](../weaving_orchestrator.py)** - Underlying LLM orchestrator
- **[HoloLoom](../hololoom.py)** - Memory system
- **[Config](../config.py)** - Configuration options
- **[multiagent_rag.py](multiagent_rag.py)** - Multi-agent implementation
