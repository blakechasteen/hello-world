# HoloLoom RAG Moonshot Phase: Architecture Plan

**Version**: 1.0
**Date**: November 13, 2025
**Status**: Architecture Planning Complete

---

## Executive Summary

This document defines the architecture for 6 advanced RAG features that will transform HoloLoom RAG from Level 4 to a complete enterprise-grade system. All features maintain backward compatibility, zero-config philosophy, and graceful degradation.

### The 6 Moonshot Features

1. **Streaming Responses** - Token-by-token LLM generation (real-time UX)
2. **Custom Embeddings** - Plugin architecture for any embedding model
3. **Advanced Reranking** - Cross-encoder precision boost (10-20% accuracy)
4. **SQL Integration** - Hybrid knowledge graph + structured database queries
5. **Multi-Hop Reasoning** - Follow relationship chains in Yarn Graph (up to N hops)
6. **Multi-Agent RAG** - Parallel execution with consensus (5+ agents)

### Timeline & Deployment

| Wave | Agents | Model | Effort | Dependencies |
|------|--------|-------|--------|--------------|
| **Wave 3** | E (Streaming), F (Custom Embeddings), G (Reranking) | Haiku | 2-3 days | None - run in parallel |
| **Wave 4** | H (SQL), I (Multi-Hop) | Sonnet | 1 week | May benefit from Wave 3 |
| **Wave 5** | J (Multi-Agent) | Sonnet | 1 week | Orchestrates E-I |
| **Post-Wave** | L (Verification), M (Elegance) | Haiku | 2-3 days | After all features |

**Total Timeline**: 2-3 weeks (1-2 weeks with parallel execution)
**Total Cost**: ~$10.25

---

## Table of Contents

1. [Current Architecture Analysis](#current-architecture-analysis)
2. [Integration Strategy](#integration-strategy)
3. [Feature 1: Streaming Responses](#feature-1-streaming-responses)
4. [Feature 2: Custom Embeddings](#feature-2-custom-embeddings)
5. [Feature 3: Advanced Reranking](#feature-3-advanced-reranking)
6. [Feature 4: SQL Integration](#feature-4-sql-integration)
7. [Feature 5: Multi-Hop Reasoning](#feature-5-multi-hop-reasoning)
8. [Feature 6: Multi-Agent RAG](#feature-6-multi-agent-rag)
9. [Unified API Design](#unified-api-design)
10. [Dependency Graph](#dependency-graph)
11. [Conflict Resolution](#conflict-resolution)
12. [Testing Strategy](#testing-strategy)
13. [Implementation Specifications](#implementation-specifications)
14. [Risk Assessment](#risk-assessment)

---

## Current Architecture Analysis

### SimpleRAG (Base Class)

```python
class SimpleRAG:
    def __init__(config, llm_provider, llm_model, enable_caching):
        self.config = config or Config.fast()
        self.llm_provider = llm_provider
        self.enable_caching = enable_caching
        self.query_cache = {}  # LRU cache for repeated queries
        self.loom = None  # HoloLoom instance (initialized in __aenter__)
        self.orchestrator = None  # WeavingOrchestrator (LLM integration)

    async def ingest(content: Any) -> None:
        # Delegates to hololoom.experience()

    async def query(question: str, mode: str, max_sources: int) -> RAGResult:
        # 1. Check cache
        # 2. Recall memories (hololoom.recall())
        # 3. Generate answer (orchestrator.weave())
        # 4. Return RAGResult

    async def batch_query(questions: List[str]) -> List[RAGResult]:
        # Batch processing

    def get_metrics() -> Dict:
        # System stats
```

### MultimodalRAG (Extends SimpleRAG)

```python
class MultimodalRAG(SimpleRAG):
    def __init__(..., enable_visual_compression, compression_threshold):
        super().__init__(...)
        self.visual_qa_engine = None  # OCR + CLIP

    async def ingest_photo(image, tags, description):
        # Store photo with CLIP embedding

    async def query_with_image(question, image, mode):
        # OCR + CLIP + recall + LLM generation

    async def get_related_photos(query):
        # CLIP text-image similarity

    async def get_similar_photos(image):
        # CLIP image-image similarity
```

### Extension Points Identified

1. **Streaming**: Add `query_stream()` method to SimpleRAG
2. **Custom Embeddings**: Protocol-based plugin in `__init__()`
3. **Reranking**: Insert between `recall()` and `orchestrator.weave()`
4. **SQL**: Hybrid routing in `query()` method
5. **Multi-Hop**: Extend Yarn Graph traversal in `recall()`
6. **Multi-Agent**: New class `MultiAgentRAG` that orchestrates multiple SimpleRAG instances

---

## Integration Strategy

### Design Philosophy

1. **Backward Compatibility**: All existing code continues working
2. **Optional Features**: Each feature is opt-in via parameters
3. **Graceful Degradation**: Features degrade if dependencies unavailable
4. **Protocol-Based**: Use protocols for extensibility
5. **Performance-Conscious**: Don't sacrifice speed for features

### Integration Approaches

| Feature | Approach | Rationale |
|---------|----------|-----------|
| **Streaming** | Add method to SimpleRAG | Simple, non-invasive |
| **Custom Embeddings** | Plugin protocol in __init__() | Composable, testable |
| **Reranking** | Middleware in query pipeline | Transparent, optional |
| **SQL** | Adapter pattern + routing | Separation of concerns |
| **Multi-Hop** | Extend recall() logic | Natural fit with Yarn Graph |
| **Multi-Agent** | New orchestrator class | Complex, deserves own class |

---

## Feature 1: Streaming Responses

### Goal

Enable token-by-token LLM generation for real-time user experience.

### API Design

```python
class SimpleRAG:
    async def query_stream(
        self,
        question: str,
        mode: str = "direct",  # Only direct mode supports streaming
        max_sources: int = 5
    ) -> AsyncGenerator[StreamToken, None]:
        """
        Stream response token-by-token.

        Yields:
            StreamToken(text, metadata) for each token
        """
```

### StreamToken Structure

```python
@dataclass
class StreamToken:
    text: str  # Single token text
    index: int  # Token index in response
    cumulative_text: str  # All tokens so far
    metadata: Dict[str, Any]  # latency_ms, tokens_per_sec, etc.
    is_final: bool = False  # True for last token
```

### Integration Point

**File**: `HoloLoom/rag/streaming.py` (new file)
**Hook**: Call from `SimpleRAG.query_stream()`

### Implementation Approach

1. Check if LLM provider supports streaming (Ollama ✓, Anthropic ✓, OpenAI ✓)
2. Recall memories (same as regular query)
3. Stream LLM generation via `orchestrator.weave_stream()`
4. Yield tokens with metadata

### Key Decisions

**Q: Does streaming work with caching?**
A: **No**. Skip cache for streaming queries (can't partially stream cached results).

**Q: Does streaming work with all reasoning modes?**
A: **No**. Only `mode="direct"` supports streaming. Verify/research/plan_execute require multiple LLM calls.

**Q: Do we cache streamed responses?**
A: **Yes**. After streaming completes, cache the full response.

**Q: Fallback if streaming unavailable?**
A: Fall back to regular `query()`, return all tokens at once.

### Dependencies

- `weaving_orchestrator_llm.py` - Add `weave_stream()` method
- Ollama/Anthropic/OpenAI APIs support streaming

### Files to Create

- `HoloLoom/rag/streaming.py` (~200 lines) - StreamToken, streaming logic
- `HoloLoom/rag/tests/test_streaming.py` (~150 lines) - Unit tests
- `demos/demo_streaming_rag.py` (~80 lines) - Demo

**Total**: ~430 lines

---

## Feature 2: Custom Embeddings

### Goal

Enable users to plug in any embedding model (HuggingFace, OpenAI, Cohere, custom).

### API Design

```python
from HoloLoom.rag.embedding_plugins import EmbeddingProvider

class SimpleRAG:
    def __init__(
        self,
        ...,
        embedding_provider: Optional[EmbeddingProvider] = None
    ):
        self.embedding_provider = embedding_provider or DefaultMatryoshkaEmbedding()
```

### EmbeddingProvider Protocol

```python
from typing import Protocol, List
import numpy as np

class EmbeddingProvider(Protocol):
    """Protocol for custom embedding models."""

    @property
    def dimension(self) -> int:
        """Embedding dimension."""
        ...

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts to embeddings.

        Args:
            texts: List of strings to encode

        Returns:
            np.ndarray of shape (len(texts), dimension)
        """
        ...

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode query (may differ from document encoding).

        Returns:
            np.ndarray of shape (dimension,)
        """
        ...
```

### Built-in Providers

```python
class MatryoshkaEmbedding(EmbeddingProvider):
    """Default Matryoshka embeddings (96/192/384 dims)."""
    dimension = 384

class HuggingFaceEmbedding(EmbeddingProvider):
    """Any HuggingFace model."""
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()

class OpenAIEmbedding(EmbeddingProvider):
    """OpenAI text-embedding-3-small/large."""
    dimension = 1536  # text-embedding-3-small

class CohereEmbedding(EmbeddingProvider):
    """Cohere embeddings."""
    dimension = 1024  # embed-english-v3.0
```

### Integration Point

**File**: `HoloLoom/rag/embedding_plugins.py` (new file)
**Hook**: Pass to `hololoom.__init__(embedding_provider=...)`

### Implementation Approach

1. Define `EmbeddingProvider` protocol
2. Create built-in providers (Matryoshka, HuggingFace, OpenAI, Cohere)
3. Update `HoloLoom.__init__()` to accept custom provider
4. Graceful fallback to Matryoshka if custom provider fails

### Key Decisions

**Q: How do we handle different embedding dimensions?**
A: **Store dimension in provider**. Vector DB (Qdrant) handles variable dimensions.

**Q: Does this affect zero-copy embeddings?**
A: **Yes**. Zero-copy only works with Matryoshka (96/192/384). Disable for custom embeddings.

**Q: How do we validate custom embeddings work?**
A: **Test encode()** on sample texts during initialization. Warn if errors.

**Q: Plugin discovery mechanism?**
A: **Manual for now**. User instantiates provider and passes to `SimpleRAG(embedding_provider=...)`.

### Dependencies

- `sentence-transformers` (optional, for HuggingFace)
- `openai` (optional, for OpenAI)
- `cohere` (optional, for Cohere)

### Files to Create

- `HoloLoom/rag/embedding_plugins.py` (~300 lines) - Protocol + providers
- `HoloLoom/rag/tests/test_embedding_plugins.py` (~200 lines) - Tests
- `demos/demo_custom_embeddings.py` (~100 lines) - Demo

**Total**: ~600 lines

---

## Feature 3: Advanced Reranking

### Goal

Rerank top-k retrieved results using cross-encoder for higher precision (10-20% accuracy boost).

### API Design

```python
class SimpleRAG:
    def __init__(
        self,
        ...,
        enable_reranking: bool = False,
        reranker: str = "cross-encoder",  # or "colbert", "custom"
        rerank_top_k: int = 20
    ):
        self.enable_reranking = enable_reranking
        self.reranker = create_reranker(reranker) if enable_reranking else None
        self.rerank_top_k = rerank_top_k
```

### Reranker Protocol

```python
class Reranker(Protocol):
    """Protocol for reranking models."""

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int
    ) -> List[Tuple[int, float]]:
        """
        Rerank documents by relevance to query.

        Returns:
            List of (index, score) tuples, sorted by score (descending)
        """
        ...
```

### Built-in Rerankers

```python
class CrossEncoderReranker(Reranker):
    """Cross-encoder reranking (ms-marco-MiniLM-L-6-v2)."""
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name)

class ColBERTReranker(Reranker):
    """ColBERT late interaction reranking."""
    ...

class CustomReranker(Reranker):
    """User-provided reranker."""
    ...
```

### Integration Point

**Location**: Insert between `recall()` and `orchestrator.weave()` in `query()` method

```python
async def query(self, question, mode, max_sources):
    # 1. Recall memories
    memories = await self.loom.recall(question, k=self.rerank_top_k if self.enable_reranking else max_sources)

    # 2. Rerank (NEW)
    if self.enable_reranking and len(memories) > max_sources:
        reranked_indices = self.reranker.rerank(question, [m.text for m in memories], max_sources)
        memories = [memories[i] for i, _ in reranked_indices]

    # 3. Generate answer
    spacetime = await self.orchestrator.weave(Query(text=question))
    ...
```

### Implementation Approach

1. Retrieve top-k results (k > max_sources, e.g., k=20 for max_sources=5)
2. Rerank using cross-encoder
3. Take top max_sources after reranking
4. Continue with LLM generation

### Key Decisions

**Q: Where in the pipeline does reranking happen?**
A: **After recall, before LLM**. Rerank retrieval results to improve context quality.

**Q: Does reranking work with multimodal results?**
A: **Text only for now**. Images are not reranked (CLIP similarity already good).

**Q: How do we balance latency vs precision?**
A: **Configurable top_k**. Default: retrieve 20, rerank to top 5. User can adjust.

**Q: Fallback if reranker unavailable?**
A: **Use original retrieval order**. Graceful degradation (warn user).

### Dependencies

- `sentence-transformers` (for cross-encoder)

### Files to Create

- `HoloLoom/rag/reranking.py` (~350 lines) - Reranker protocol + implementations
- `HoloLoom/rag/tests/test_reranking.py` (~200 lines) - Tests
- `demos/demo_reranking_rag.py` (~120 lines) - Precision comparison demo

**Total**: ~670 lines

---

## Feature 4: SQL Integration

### Goal

Enable querying structured databases alongside knowledge graph, with automatic text-to-SQL translation.

### API Design

```python
class SimpleRAG:
    def __init__(
        self,
        ...,
        enable_sql: bool = False,
        sql_connection_string: Optional[str] = None,
        sql_schema: Optional[Dict] = None
    ):
        self.enable_sql = enable_sql
        self.sql_adapter = SQLAdapter(sql_connection_string, sql_schema) if enable_sql else None

    async def query(self, question: str, ...):
        # Hybrid routing: detect SQL vs semantic
        if self.enable_sql and self.sql_adapter.is_sql_query(question):
            return await self._query_sql(question)
        else:
            return await self._query_semantic(question)  # Original path
```

### SQL Adapter

```python
class SQLAdapter:
    """Adapter for SQL database queries."""

    def __init__(self, connection_string: str, schema: Dict):
        self.connection = create_engine(connection_string)
        self.schema = schema  # Table definitions
        self.text_to_sql_translator = TextToSQLTranslator(schema)

    def is_sql_query(self, question: str) -> bool:
        """Detect if question requires SQL."""
        # Keywords: SELECT, WHERE, COUNT, SUM, AVG, GROUP BY, etc.
        # Or use LLM classifier

    async def execute_query(self, question: str) -> SQLResult:
        """Translate text to SQL and execute."""
        sql_query = self.text_to_sql_translator.translate(question)
        result = self.connection.execute(sql_query)
        return SQLResult(rows=result.fetchall(), columns=result.keys())
```

### Hybrid Result

```python
@dataclass
class HybridRAGResult(RAGResult):
    """Extended result with SQL data."""
    sql_data: Optional[pd.DataFrame] = None
    sql_query: Optional[str] = None
    query_type: str = "semantic"  # or "sql" or "hybrid"
```

### Integration Point

**File**: `HoloLoom/rag/sql_adapter.py` (new file)
**Hook**: Routing in `SimpleRAG.query()`

### Implementation Approach

1. **Detection**: Use keywords or LLM classifier to detect SQL queries
2. **Translation**: Text-to-SQL using LLM (GPT-4 or Claude)
3. **Execution**: Run SQL query on database
4. **Hybrid**: For ambiguous queries, run both SQL + semantic, merge results
5. **LLM Synthesis**: Use LLM to explain SQL results in natural language

### Key Decisions

**Q: How do we detect SQL queries?**
A: **Two-phase**: (1) Keyword detection (fast), (2) LLM classifier (accurate). Use keyword first, LLM if uncertain.

**Q: Where does SQL fit in the architecture?**
A: **Routing layer** in `query()` method. Separate path for SQL vs semantic.

**Q: How do we combine SQL results with semantic results?**
A: **Hybrid mode**: Run both, LLM synthesizes combined answer. Use when query touches both structured + unstructured data.

**Q: What databases do we support?**
A: **SQLAlchemy-compatible**: SQLite, PostgreSQL, MySQL. Start with SQLite (simplest).

### Dependencies

- `sqlalchemy` (database abstraction)
- LLM for text-to-SQL (uses existing orchestrator)

### Files to Create

- `HoloLoom/rag/sql_adapter.py` (~550 lines) - SQL adapter + text-to-SQL
- `HoloLoom/rag/tests/test_sql_adapter.py` (~250 lines) - Tests
- `demos/demo_sql_rag.py` (~150 lines) - Demo with sample DB

**Total**: ~950 lines

---

## Feature 5: Multi-Hop Reasoning

### Goal

Follow relationship chains in Yarn Graph (multi-hop traversal) to answer complex questions requiring inference.

### API Design

```python
class SimpleRAG:
    async def query(
        self,
        question: str,
        mode: str = "verify",
        max_sources: int = 5,
        hops: int = 1  # NEW: Number of graph hops
    ) -> RAGResult:
        """
        Query with multi-hop reasoning.

        Args:
            hops: Number of relationship hops to traverse (1-5)
                  1 = direct neighbors only (default)
                  2 = neighbors of neighbors
                  3+ = deeper reasoning
        """
```

### Multi-Hop Result

```python
@dataclass
class MultiHopRAGResult(RAGResult):
    """Extended result with reasoning path."""
    reasoning_path: List[Tuple[str, str, str]]  # [(entity, relation, entity), ...]
    hop_count: int
    path_confidence: float  # Confidence in the reasoning path
```

### Integration Point

**File**: `HoloLoom/rag/multi_hop.py` (new file)
**Hook**: Extend `recall()` logic in `SimpleRAG`

### Implementation Approach

1. **Initial Retrieval**: Get direct matches (hop 0)
2. **Graph Traversal**: For each match, traverse Yarn Graph relationships
3. **Relevance Scoring**: Score each hop by semantic similarity to query
4. **Pruning**: Beam search to avoid combinatorial explosion
5. **Path Ranking**: Rank complete paths by cumulative relevance
6. **LLM Synthesis**: Use LLM to synthesize answer from multi-hop evidence

### Multi-Hop Algorithm

```python
def multi_hop_traversal(query: str, hops: int, beam_width: int = 5):
    # Start with direct matches
    current_nodes = yarn_graph.query(query, k=10)
    paths = [[node] for node in current_nodes]

    for hop in range(1, hops + 1):
        next_paths = []
        for path in paths:
            # Expand last node in path
            neighbors = yarn_graph.get_neighbors(path[-1])
            for neighbor in neighbors:
                # Score relevance
                score = compute_relevance(query, path + [neighbor])
                next_paths.append((path + [neighbor], score))

        # Beam search: keep top-k paths
        next_paths.sort(key=lambda x: x[1], reverse=True)
        paths = [p for p, _ in next_paths[:beam_width]]

    return paths
```

### Key Decisions

**Q: How do we score relationship relevance?**
A: **Semantic similarity** of (source entity + relation + target entity) to query, weighted by edge strength.

**Q: How do we avoid combinatorial explosion?**
A: **Beam search** with configurable beam_width (default: 5 paths).

**Q: Does this extend SimpleRAG or MultimodalRAG?**
A: **SimpleRAG**. Multimodal inherits via extension.

**Q: How do we visualize the reasoning path?**
A: Return `reasoning_path` in result. Dashboard can render as graph.

### Dependencies

- `HoloLoom/memory/graph.py` (Yarn Graph)
- Traversal algorithms (BFS, beam search)

### Files to Create

- `HoloLoom/rag/multi_hop.py` (~500 lines) - Multi-hop traversal + reasoning
- `HoloLoom/rag/tests/test_multi_hop.py` (~250 lines) - Tests
- `demos/demo_multi_hop_rag.py` (~180 lines) - Demo with reasoning paths

**Total**: ~930 lines

---

## Feature 6: Multi-Agent RAG

### Goal

Spawn multiple agents with different strategies, execute in parallel, reach consensus on answer.

### API Design

```python
class MultiAgentRAG:
    """
    Multi-agent RAG orchestrator.

    Spawns N agents with diverse strategies, runs queries in parallel,
    merges results using consensus mechanism.
    """

    def __init__(
        self,
        config: Config,
        n_agents: int = 5,
        consensus_method: str = "confidence_weighted",  # or "majority_vote", "llm_judge"
        diversity_strategies: Optional[List[str]] = None
    ):
        self.n_agents = n_agents
        self.consensus_method = consensus_method
        self.agents = self._create_diverse_agents(diversity_strategies)

    async def query(
        self,
        question: str,
        explain_disagreement: bool = False
    ) -> MultiAgentRAGResult:
        """
        Query with multiple agents in parallel, reach consensus.

        Args:
            explain_disagreement: Include explanation of agent disagreements
        """
```

### Agent Diversity Strategies

```python
def _create_diverse_agents(self, strategies: List[str]) -> List[SimpleRAG]:
    """
    Create agents with diverse configurations.

    Diversity dimensions:
    - Embeddings: Different models (Matryoshka, OpenAI, Cohere)
    - Reasoning modes: Different modes (direct, verify, research)
    - Prompts: Different system prompts
    - Retrieval: Different top-k, reranking settings
    - LLM: Different models (GPT-4, Claude, local)
    """
    agents = []
    for i in range(self.n_agents):
        agent_config = self._get_diversity_config(i, strategies)
        agents.append(SimpleRAG(**agent_config))
    return agents
```

### Consensus Mechanisms

```python
class ConsensusMechanism:
    """Base class for consensus algorithms."""

    def reach_consensus(
        self,
        results: List[RAGResult]
    ) -> Tuple[str, float, Dict]:
        """
        Merge results from multiple agents.

        Returns:
            (final_answer, confidence, metadata)
        """
        ...

class MajorityVoteConsensus(ConsensusMechanism):
    """Simple majority vote on answers."""
    ...

class ConfidenceWeightedConsensus(ConsensusMechanism):
    """Weight by agent confidence scores."""
    ...

class LLMJudgeConsensus(ConsensusMechanism):
    """Use LLM to judge and synthesize answers."""
    ...
```

### Multi-Agent Result

```python
@dataclass
class MultiAgentRAGResult(RAGResult):
    """Result from multi-agent consensus."""
    agent_results: List[RAGResult]  # Individual agent results
    consensus_method: str
    agreement_score: float  # 0.0-1.0, how much agents agree
    disagreements: Optional[List[str]] = None  # Explanation of disagreements
```

### Integration Point

**File**: `HoloLoom/rag/multi_agent.py` (new file)
**Usage**: New class, doesn't extend SimpleRAG

### Implementation Approach

1. **Agent Creation**: Spawn N agents with diverse configs
2. **Parallel Execution**: Run all agents in parallel (asyncio.gather)
3. **Result Collection**: Collect all agent results
4. **Consensus**: Apply consensus mechanism to merge results
5. **Disagreement Analysis**: Identify and explain disagreements
6. **Final Answer**: Return consensus result

### Key Decisions

**Q: How do agents differ?**
A: **Multiple dimensions**: Embeddings, reasoning modes, prompts, retrieval settings, LLMs.

**Q: What consensus methods do we support?**
A: **Three**: (1) Majority vote, (2) Confidence weighted, (3) LLM judge (best quality, slowest).

**Q: How do we handle conflicts?**
A: **Explain**: Include `disagreements` in result. User decides how to interpret.

**Q: Does this replace SimpleRAG or extend it?**
A: **New class**: `MultiAgentRAG`. Orchestrates multiple `SimpleRAG` instances.

### Dependencies

- `asyncio` (parallel execution)
- Multiple SimpleRAG instances

### Files to Create

- `HoloLoom/rag/multi_agent.py` (~700 lines) - Multi-agent orchestrator + consensus
- `HoloLoom/rag/tests/test_multi_agent.py` (~300 lines) - Tests
- `demos/demo_multi_agent_rag.py` (~200 lines) - Demo with consensus comparison

**Total**: ~1,200 lines

---

## Unified API Design

### The Complete API

```python
from HoloLoom.rag import (
    SimpleRAG,           # Level 2-4 RAG (current)
    MultimodalRAG,       # Text + images (current)
    AdvancedRAG,         # All Moonshot features (NEW)
    MultiAgentRAG        # Multi-agent consensus (NEW)
)

# Option 1: SimpleRAG (existing, unchanged)
async with SimpleRAG() as rag:
    result = await rag.query("question")

# Option 2: AdvancedRAG (all Moonshot features)
async with AdvancedRAG(
    # Existing features
    config=Config.fused(),
    llm_provider="anthropic",
    enable_caching=True,

    # NEW: Streaming
    enable_streaming=False,  # Use query_stream() instead

    # NEW: Custom embeddings
    embedding_provider=HuggingFaceEmbedding("all-MiniLM-L6-v2"),

    # NEW: Reranking
    enable_reranking=True,
    reranker="cross-encoder",
    rerank_top_k=20,

    # NEW: SQL
    enable_sql=True,
    sql_connection_string="sqlite:///my_database.db",
    sql_schema={"users": ["id", "name", "age"], ...},

    # NEW: Multi-hop
    default_hops=1  # Can override per query
) as rag:
    # Streaming
    async for token in rag.query_stream("question"):
        print(token.text, end='')

    # Multi-hop
    result = await rag.query("Complex question", hops=3)

    # SQL
    result = await rag.query("SELECT * FROM users WHERE age > 30")

    # All features work together!

# Option 3: MultiAgentRAG (consensus)
async with MultiAgentRAG(
    n_agents=5,
    consensus_method="llm_judge",
    diversity_strategies=["embeddings", "reasoning", "prompts"]
) as rag:
    result = await rag.query("Controversial question", explain_disagreement=True)

    print(f"Agreement: {result.agreement_score:.2f}")
    if result.disagreements:
        print(f"Disagreements: {result.disagreements}")
```

### Inheritance Hierarchy

```
SimpleRAG (base)
├── MultimodalRAG (extends SimpleRAG)
│   └── AdvancedRAG (extends MultimodalRAG, adds Moonshot features)
│
└── MultiAgentRAG (orchestrates multiple SimpleRAG/AdvancedRAG instances)
```

---

## Dependency Graph

### Wave 3: Parallel Execution (No Dependencies)

```
E: Streaming ────────────┐
                         │
F: Custom Embeddings ────┼──→ All can run in parallel
                         │
G: Reranking ────────────┘
```

**Timeline**: 2-3 days (parallel)
**Model**: Haiku (all simple implementations)

### Wave 4: Complex Features (May Have Dependencies)

```
H: SQL Integration ──→ May benefit from reranking (G)
                       But not required

I: Multi-Hop ───────→ May benefit from reranking (G)
                       But not required
```

**Timeline**: 1 week (can run in parallel, but each takes longer)
**Model**: Sonnet (complex architecture)

### Wave 5: Meta-Orchestration (Depends on All Above)

```
J: Multi-Agent ──→ Orchestrates E, F, G, H, I
                   (Requires all features to exist)
```

**Timeline**: 1 week
**Model**: Sonnet (orchestration complexity)

---

## Conflict Resolution

### Identified Conflicts

#### 1. Custom Embeddings + Zero-Copy Embeddings

**Conflict**: Zero-copy only works with Matryoshka (96/192/384 dims). Custom embeddings may have different dimensions.

**Solution**: Disable zero-copy when custom embeddings enabled. Add warning in docs.

```python
if embedding_provider and not isinstance(embedding_provider, MatryoshkaEmbedding):
    logger.warning("Zero-copy embeddings disabled with custom embedding provider")
    config.enable_zero_copy_embeddings = False
```

#### 2. Streaming + Caching

**Conflict**: Can't stream partial cached results (cache stores full response).

**Solution**: Skip cache lookup for streaming queries. Cache full response after streaming completes.

```python
async def query_stream(self, question, ...):
    # Don't check cache (can't stream cached results)
    # ... stream generation ...
    # Cache full response after completion
    if self.enable_caching:
        self.query_cache[cache_key] = cumulative_response
```

#### 3. Multi-Agent + Latency

**Conflict**: Spawning 5+ agents in parallel may be slow (5× latency).

**Solution**:
- Run agents in parallel (asyncio.gather) - actual latency = max(agent_latency), not sum
- Timeout per agent (default: 5s)
- Early termination if consensus reached before all agents finish

```python
async def query(self, question, timeout=5.0):
    tasks = [agent.query(question) for agent in self.agents]
    results = await asyncio.gather(*tasks, timeout=timeout, return_exceptions=True)
    # Process results, reach consensus
```

#### 4. SQL + Semantic Queries

**Conflict**: Some queries are ambiguous (could be SQL or semantic).

**Solution**: **Hybrid mode** - run both, LLM synthesizes combined answer.

```python
if is_ambiguous(question):
    # Run both paths in parallel
    sql_result, semantic_result = await asyncio.gather(
        self._query_sql(question),
        self._query_semantic(question)
    )
    # LLM synthesizes combined answer
    return self._synthesize_hybrid(sql_result, semantic_result)
```

---

## Testing Strategy

### Per-Feature Testing

| Feature | Unit Tests | Integration Tests | Performance Benchmarks |
|---------|------------|-------------------|------------------------|
| **Streaming** | Mock LLM streaming | End-to-end stream | Tokens per second |
| **Custom Embeddings** | Protocol compliance | Retrieval accuracy | Encoding speed |
| **Reranking** | Reranker algorithms | Precision improvement | Reranking latency |
| **SQL** | Text-to-SQL translation | Hybrid queries | Query execution time |
| **Multi-Hop** | Graph traversal | Reasoning accuracy | Hop latency |
| **Multi-Agent** | Consensus algorithms | Agreement analysis | Parallel speedup |

### Backward Compatibility Tests

**Critical**: All existing tests must pass.

```bash
# Run all existing tests
pytest HoloLoom/rag/tests/test_simple_rag.py -v  # Must pass
pytest HoloLoom/rag/tests/test_multimodal_rag.py -v  # Must pass

# Run new Moonshot tests
pytest HoloLoom/rag/tests/test_streaming.py -v
pytest HoloLoom/rag/tests/test_embedding_plugins.py -v
pytest HoloLoom/rag/tests/test_reranking.py -v
pytest HoloLoom/rag/tests/test_sql_adapter.py -v
pytest HoloLoom/rag/tests/test_multi_hop.py -v
pytest HoloLoom/rag/tests/test_multi_agent.py -v
```

### Integration Test Plan

**Test Scenario**: All features working together

```python
async def test_moonshot_integration():
    """Test all Moonshot features integrated."""
    async with AdvancedRAG(
        enable_reranking=True,
        enable_sql=True,
        embedding_provider=HuggingFaceEmbedding("all-MiniLM-L6-v2")
    ) as rag:
        # Ingest data
        await rag.ingest("Thompson Sampling uses Bayesian statistics")

        # Test streaming
        tokens = []
        async for token in rag.query_stream("What is Thompson Sampling?"):
            tokens.append(token.text)
        assert len(tokens) > 10

        # Test multi-hop
        result = await rag.query("Who influenced Thompson?", hops=2)
        assert result.hop_count == 2

        # Test SQL
        result = await rag.query("SELECT * FROM papers WHERE author='Thompson'")
        assert result.sql_data is not None
```

---

## Implementation Specifications

### Agent E: Streaming Responses

**Model**: Haiku
**Effort**: 1-2 days
**Files**: 3 files, ~430 lines

**Implementation Checklist**:
- [ ] Create `HoloLoom/rag/streaming.py`
- [ ] Define `StreamToken` dataclass
- [ ] Add `query_stream()` method to SimpleRAG
- [ ] Implement streaming for Ollama
- [ ] Implement streaming for Anthropic
- [ ] Implement streaming for OpenAI
- [ ] Handle cache skip for streaming
- [ ] Cache full response after streaming
- [ ] Unit tests (15 tests)
- [ ] Integration test (end-to-end stream)
- [ ] Demo script

**Key Algorithm**: Async generator pattern

```python
async def query_stream(self, question: str, mode: str = "direct"):
    # Skip cache (can't stream cached)
    memories = await self.loom.recall(question)

    # Stream from LLM
    cumulative_text = ""
    async for chunk in self.orchestrator.weave_stream(Query(text=question)):
        token = StreamToken(
            text=chunk.text,
            index=chunk.index,
            cumulative_text=cumulative_text + chunk.text,
            metadata={"latency_ms": chunk.latency}
        )
        cumulative_text += chunk.text
        yield token

    # Cache full response
    if self.enable_caching:
        self.query_cache[question] = cumulative_text
```

---

### Agent F: Custom Embeddings

**Model**: Haiku
**Effort**: 1-2 days
**Files**: 3 files, ~600 lines

**Implementation Checklist**:
- [ ] Create `HoloLoom/rag/embedding_plugins.py`
- [ ] Define `EmbeddingProvider` protocol
- [ ] Implement `MatryoshkaEmbedding` (default)
- [ ] Implement `HuggingFaceEmbedding`
- [ ] Implement `OpenAIEmbedding`
- [ ] Implement `CohereEmbedding`
- [ ] Update `SimpleRAG.__init__()` to accept custom provider
- [ ] Disable zero-copy for custom embeddings
- [ ] Unit tests (20 tests, one per provider)
- [ ] Integration test (end-to-end retrieval with custom)
- [ ] Demo script

**Key Algorithm**: Protocol-based plugin architecture

```python
class EmbeddingProvider(Protocol):
    dimension: int
    def encode(self, texts: List[str]) -> np.ndarray: ...

class SimpleRAG:
    def __init__(self, ..., embedding_provider: Optional[EmbeddingProvider] = None):
        self.embedding_provider = embedding_provider or MatryoshkaEmbedding()

        # Disable zero-copy for custom embeddings
        if not isinstance(self.embedding_provider, MatryoshkaEmbedding):
            self.config.enable_zero_copy_embeddings = False
```

---

### Agent G: Advanced Reranking

**Model**: Haiku
**Effort**: 2 days
**Files**: 3 files, ~670 lines

**Implementation Checklist**:
- [ ] Create `HoloLoom/rag/reranking.py`
- [ ] Define `Reranker` protocol
- [ ] Implement `CrossEncoderReranker`
- [ ] Implement `ColBERTReranker` (optional)
- [ ] Add `enable_reranking` to `SimpleRAG.__init__()`
- [ ] Insert reranking between recall and LLM
- [ ] Handle case where retrieved < rerank_top_k
- [ ] Unit tests (15 tests)
- [ ] Integration test (precision measurement)
- [ ] Demo script (show precision improvement)

**Key Algorithm**: Cross-encoder scoring

```python
async def query(self, question, mode, max_sources):
    # Retrieve more results than needed
    k = self.rerank_top_k if self.enable_reranking else max_sources
    memories = await self.loom.recall(question, k=k)

    # Rerank if enabled
    if self.enable_reranking and len(memories) > max_sources:
        scores = self.reranker.rerank(
            question,
            [m.text for m in memories],
            max_sources
        )
        memories = [memories[i] for i, _ in scores[:max_sources]]

    # Continue with LLM generation
    ...
```

---

### Agent H: SQL Integration

**Model**: Sonnet
**Effort**: 3-4 days
**Files**: 3 files, ~950 lines

**Implementation Checklist**:
- [ ] Create `HoloLoom/rag/sql_adapter.py`
- [ ] Define `SQLAdapter` class
- [ ] Implement SQL query detection (keywords + LLM)
- [ ] Implement text-to-SQL translation (LLM-based)
- [ ] Implement SQL execution (SQLAlchemy)
- [ ] Implement hybrid mode (SQL + semantic)
- [ ] Add `enable_sql` to `SimpleRAG.__init__()`
- [ ] Add routing in `query()` method
- [ ] Unit tests (25 tests)
- [ ] Integration test (hybrid query)
- [ ] Demo script with sample SQLite DB

**Key Algorithm**: Text-to-SQL with LLM

```python
class TextToSQLTranslator:
    def translate(self, question: str, schema: Dict) -> str:
        prompt = f"""
        Given this database schema:
        {json.dumps(schema, indent=2)}

        Translate this question to SQL:
        {question}

        Return ONLY the SQL query, no explanation.
        """

        sql_query = await self.llm.generate(prompt)
        return sql_query.strip()
```

---

### Agent I: Multi-Hop Reasoning

**Model**: Sonnet
**Effort**: 3-4 days
**Files**: 3 files, ~930 lines

**Implementation Checklist**:
- [ ] Create `HoloLoom/rag/multi_hop.py`
- [ ] Implement beam search graph traversal
- [ ] Implement relevance scoring per hop
- [ ] Implement path ranking
- [ ] Add `hops` parameter to `query()` method
- [ ] Define `MultiHopRAGResult` dataclass
- [ ] Unit tests (20 tests)
- [ ] Integration test (3-hop reasoning)
- [ ] Demo script with reasoning path visualization

**Key Algorithm**: Beam search traversal

```python
def multi_hop_search(query: str, hops: int, beam_width: int = 5):
    current_paths = [[node] for node in get_initial_nodes(query)]

    for hop in range(hops):
        next_paths = []
        for path in current_paths:
            neighbors = yarn_graph.get_neighbors(path[-1])
            for neighbor in neighbors:
                extended_path = path + [neighbor]
                score = compute_path_score(query, extended_path)
                next_paths.append((extended_path, score))

        # Keep top beam_width paths
        next_paths.sort(key=lambda x: x[1], reverse=True)
        current_paths = [p for p, _ in next_paths[:beam_width]]

    return current_paths
```

---

### Agent J: Multi-Agent RAG

**Model**: Sonnet
**Effort**: 4-5 days
**Files**: 3 files, ~1,200 lines

**Implementation Checklist**:
- [ ] Create `HoloLoom/rag/multi_agent.py`
- [ ] Define `MultiAgentRAG` class
- [ ] Implement agent diversity strategies
- [ ] Implement parallel execution (asyncio.gather)
- [ ] Implement `MajorityVoteConsensus`
- [ ] Implement `ConfidenceWeightedConsensus`
- [ ] Implement `LLMJudgeConsensus`
- [ ] Implement disagreement analysis
- [ ] Define `MultiAgentRAGResult` dataclass
- [ ] Unit tests (25 tests)
- [ ] Integration test (5-agent consensus)
- [ ] Demo script (compare consensus methods)

**Key Algorithm**: Confidence-weighted consensus

```python
async def query(self, question: str):
    # Run all agents in parallel
    tasks = [agent.query(question) for agent in self.agents]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Reach consensus
    if self.consensus_method == "confidence_weighted":
        # Weight by confidence
        total_weight = sum(r.confidence for r in results)
        # Use LLM to synthesize weighted answer
        final_answer = await self._synthesize_weighted(results)

    return MultiAgentRAGResult(
        response=final_answer,
        agent_results=results,
        agreement_score=compute_agreement(results)
    )
```

---

## Risk Assessment

### High-Risk Areas

1. **Backward Compatibility Breaking**: New features inadvertently break existing code
   - **Mitigation**: Extensive backward compatibility tests, all features opt-in

2. **Performance Degradation**: New features slow down existing queries
   - **Mitigation**: All features disabled by default, performance benchmarks

3. **Dependency Explosion**: Too many optional dependencies
   - **Mitigation**: Graceful degradation, clear documentation of what requires what

4. **API Complexity**: Too many parameters, confusing for users
   - **Mitigation**: Sane defaults, progressive disclosure (SimpleRAG → AdvancedRAG)

5. **LLM Cost Explosion**: Multi-agent + text-to-SQL uses many LLM calls
   - **Mitigation**: Caching, user warnings about costs, configurable limits

### Medium-Risk Areas

1. **SQL Injection**: Text-to-SQL could generate malicious queries
   - **Mitigation**: Query validation, parameterized queries, read-only mode

2. **Multi-Hop Explosion**: Combinatorial explosion with high hop count
   - **Mitigation**: Beam search, max hop limit (5), timeout

3. **Streaming Reliability**: Stream interruptions, partial results
   - **Mitigation**: Error handling, cache full response on completion

4. **Custom Embedding Compatibility**: Users provide incompatible embeddings
   - **Mitigation**: Validation on initialization, clear error messages

---

## Timeline & Cost Estimate

### Wave-by-Wave Breakdown

| Wave | Agents | Model | Days | Cost | Dependencies |
|------|--------|-------|------|------|--------------|
| **Wave 3** | E, F, G | Haiku | 2-3 | $0.25 | None (parallel) |
| **Wave 4** | H, I | Sonnet | 7 | $6.00 | Optional: G (reranking) |
| **Wave 5** | J | Sonnet | 5 | $4.00 | Requires E-I |
| **Post-Wave** | L, M | Haiku | 2-3 | - | After all features |

**Total Timeline**: 16-18 days sequential, 10-12 days with parallel execution
**Total Cost**: ~$10.25

### Gantt Chart (Parallel Execution)

```
Week 1:
Day 1-3:  [E: Streaming] [F: Custom Embeddings] [G: Reranking] (parallel)
Day 4-7:  [H: SQL Integration] (sequential)

Week 2:
Day 1-4:  [I: Multi-Hop Reasoning] (sequential, can overlap with H)
Day 5-7:  [J: Multi-Agent RAG] (sequential)

Week 3:
Day 1-2:  [L: Verification & Testing]
Day 3-4:  [M: Elegance Pass]
Day 5:    Final integration & documentation
```

**Actual Timeline with Parallel**: ~2-3 weeks

---

## Conclusion

This architecture plan provides a comprehensive roadmap for implementing 6 advanced RAG features while maintaining HoloLoom's core principles:

✅ **Backward Compatible**: All existing code continues working
✅ **Zero-Config**: Defaults "just work", features are opt-in
✅ **Graceful Degradation**: Features degrade if dependencies unavailable
✅ **Protocol-Based**: Extensible, testable architecture
✅ **Performance-Conscious**: No slowdown for users who don't enable features

**Next Steps**:
1. Review and approve this architecture plan
2. Launch Wave 3 (Agents E, F, G in parallel)
3. Launch Wave 4 (Agents H, I)
4. Launch Wave 5 (Agent J)
5. Verification & elegance pass (Agents L, M)
6. Integration testing and documentation

**Expected Outcome**: HoloLoom RAG will be the most advanced open-source RAG system available, with enterprise-grade features and research-level capabilities.

---

**Document Version**: 1.0
**Last Updated**: November 13, 2025
**Status**: Ready for Implementation
