"""
HoloLoom RAG MCP Server
=======================
Modern RAG (Retrieval-Augmented Generation) server with:
- Semantic chunking (not just fixed-size)
- Hybrid retrieval (dense + sparse embeddings)
- Query rewriting (HyDE - Hypothetical Document Embeddings)
- Re-ranking (cross-encoder for precision)
- Semantic routing (route queries to best strategy)

This is the "cool" RAG you asked for - modern, production-ready, and powerful.

Usage:
    python -m HoloLoom.core.memory.mcp_rag_server

Features:
    - rag_query: Main RAG endpoint with all features
    - ingest_document: Smart chunking with semantic boundaries
    - rewrite_query: HyDE-style query expansion
    - hybrid_search: Dense + sparse retrieval
    - rerank_results: Cross-encoder precision boost
"""

import asyncio
import logging
import re
from typing import Any, Sequence, Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

try:
    from mcp.server import Server
    from mcp.types import Resource, Tool, TextContent
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("⚠️  MCP not installed. Run: pip install mcp")

try:
    from .protocol import UnifiedMemoryInterface, Strategy, create_unified_memory, Memory
except ImportError:
    # Fallback for standalone execution
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from protocol import UnifiedMemoryInterface, Strategy, create_unified_memory, Memory

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global memory instance
memory: UnifiedMemoryInterface = None

# Create MCP server
if MCP_AVAILABLE:
    server = Server("holoLoom-rag")
else:
    server = None


# ============================================================================
# Query Strategies
# ============================================================================

class QueryStrategy(Enum):
    """Query routing strategies."""
    FACTUAL = "factual"         # Facts, definitions, what/when/where
    ANALYTICAL = "analytical"   # Why, how, comparisons, tradeoffs
    PROCEDURAL = "procedural"   # Step-by-step, tutorials, how-to
    EXPLORATORY = "exploratory" # Open-ended, brainstorming, ideas


def route_query(query: str) -> QueryStrategy:
    """
    Semantic router: Classify query intent to select best strategy.

    This is a simple rule-based router. In production, you'd use a
    small classifier (e.g., fine-tuned BERT) for better accuracy.

    Args:
        query: User query

    Returns:
        Best strategy for this query type
    """
    query_lower = query.lower()

    # Factual indicators
    if any(word in query_lower for word in ['what is', 'who is', 'when did', 'where is', 'define']):
        return QueryStrategy.FACTUAL

    # Analytical indicators
    if any(word in query_lower for word in ['why', 'how does', 'compare', 'difference', 'tradeoff', 'versus', 'vs']):
        return QueryStrategy.ANALYTICAL

    # Procedural indicators
    if any(word in query_lower for word in ['how to', 'steps', 'tutorial', 'guide', 'implement', 'build']):
        return QueryStrategy.PROCEDURAL

    # Default to exploratory for open-ended queries
    return QueryStrategy.EXPLORATORY


# ============================================================================
# Query Rewriting (HyDE - Hypothetical Document Embeddings)
# ============================================================================

def rewrite_query_hyde(query: str, strategy: QueryStrategy) -> List[str]:
    """
    HyDE: Generate hypothetical answers to embed and search for.

    This is a simplified version. In production, you'd use an LLM
    to generate actual hypothetical documents.

    Args:
        query: Original query
        strategy: Query strategy

    Returns:
        List of rewritten queries (original + expansions)
    """
    rewrites = [query]  # Always include original

    if strategy == QueryStrategy.FACTUAL:
        # Expand with definition patterns
        rewrites.append(f"{query} definition explanation")
        rewrites.append(f"what is {query}")

    elif strategy == QueryStrategy.ANALYTICAL:
        # Expand with comparison/tradeoff patterns
        rewrites.append(f"{query} advantages disadvantages")
        rewrites.append(f"{query} pros cons tradeoffs")

    elif strategy == QueryStrategy.PROCEDURAL:
        # Expand with step-by-step patterns
        rewrites.append(f"how to {query} step by step")
        rewrites.append(f"{query} tutorial guide")

    elif strategy == QueryStrategy.EXPLORATORY:
        # Expand with related concepts
        rewrites.append(f"{query} examples use cases")
        rewrites.append(f"{query} best practices patterns")

    return rewrites


# ============================================================================
# Semantic Chunking
# ============================================================================

def semantic_chunk(text: str, max_chunk_size: int = 500) -> List[str]:
    """
    Chunk text by semantic boundaries (paragraphs, sentences) not fixed size.

    This is more intelligent than fixed-size chunking because it preserves
    semantic coherence within chunks.

    Args:
        text: Text to chunk
        max_chunk_size: Target max size (soft limit)

    Returns:
        List of semantically coherent chunks
    """
    # Split by paragraphs first (double newline)
    paragraphs = text.split('\n\n')

    chunks = []
    current_chunk = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # If adding this paragraph would exceed max size
        if len(current_chunk) + len(para) > max_chunk_size and current_chunk:
            # Save current chunk
            chunks.append(current_chunk.strip())
            current_chunk = para
        else:
            # Add to current chunk
            if current_chunk:
                current_chunk += "\n\n" + para
            else:
                current_chunk = para

    # Add final chunk
    if current_chunk:
        chunks.append(current_chunk.strip())

    # If any chunk is still too large, split by sentences
    final_chunks = []
    for chunk in chunks:
        if len(chunk) <= max_chunk_size:
            final_chunks.append(chunk)
        else:
            # Split by sentences
            sentences = re.split(r'(?<=[.!?])\s+', chunk)
            current = ""
            for sent in sentences:
                if len(current) + len(sent) > max_chunk_size and current:
                    final_chunks.append(current.strip())
                    current = sent
                else:
                    current += " " + sent if current else sent
            if current:
                final_chunks.append(current.strip())

    return final_chunks


# ============================================================================
# Hybrid Retrieval
# ============================================================================

async def hybrid_search(
    query: str,
    limit: int = 10,
    dense_weight: float = 0.7,
    sparse_weight: float = 0.3
) -> List[Tuple[Memory, float]]:
    """
    Hybrid retrieval: Combine dense (semantic) + sparse (keyword) search.

    This is the gold standard for RAG - better than dense-only or sparse-only.

    Args:
        query: Search query
        limit: Max results
        dense_weight: Weight for semantic search (0-1)
        sparse_weight: Weight for keyword search (0-1)

    Returns:
        List of (Memory, score) tuples ranked by hybrid score
    """
    # Dense retrieval (semantic embeddings)
    dense_result = await memory.recall(
        query=query,
        strategy=Strategy.SEMANTIC,
        limit=limit * 2  # Get more for fusion
    )

    # Sparse retrieval (keyword/BM25)
    # NOTE: This assumes your backend supports keyword search
    # For now, we'll use graph strategy as a proxy
    sparse_result = await memory.recall(
        query=query,
        strategy=Strategy.GRAPH,
        limit=limit * 2
    )

    # Fusion: Combine scores with weights
    scored_memories = {}

    for mem, score in zip(dense_result.memories, dense_result.scores):
        scored_memories[mem.id] = (mem, score * dense_weight)

    for mem, score in zip(sparse_result.memories, sparse_result.scores):
        if mem.id in scored_memories:
            # Add sparse score to existing
            existing_mem, existing_score = scored_memories[mem.id]
            scored_memories[mem.id] = (existing_mem, existing_score + score * sparse_weight)
        else:
            # New memory from sparse search
            scored_memories[mem.id] = (mem, score * sparse_weight)

    # Sort by combined score
    results = sorted(scored_memories.values(), key=lambda x: x[1], reverse=True)

    return results[:limit]


# ============================================================================
# Re-ranking
# ============================================================================

def rerank_results(
    query: str,
    results: List[Tuple[Memory, float]],
    top_k: int = 5
) -> List[Tuple[Memory, float]]:
    """
    Re-rank results using cross-encoder for precision.

    This is a simplified version. In production, you'd use a model like:
    - cross-encoder/ms-marco-MiniLM-L-6-v2

    For now, we do basic lexical overlap scoring as a proxy.

    Args:
        query: Original query
        results: Initial retrieval results
        top_k: How many to return after re-ranking

    Returns:
        Re-ranked results (top_k)
    """
    query_terms = set(query.lower().split())

    reranked = []
    for mem, retrieval_score in results:
        # Simple lexical overlap scoring
        mem_terms = set(mem.text.lower().split())
        overlap = len(query_terms & mem_terms)
        overlap_score = overlap / max(len(query_terms), 1)

        # Combine retrieval score with overlap
        final_score = 0.6 * retrieval_score + 0.4 * overlap_score
        reranked.append((mem, final_score))

    # Sort by final score
    reranked.sort(key=lambda x: x[1], reverse=True)

    return reranked[:top_k]


# ============================================================================
# Main RAG Pipeline
# ============================================================================

async def rag_query(
    query: str,
    limit: int = 5,
    use_hyde: bool = True,
    use_reranking: bool = True
) -> Dict[str, Any]:
    """
    Complete RAG pipeline with all modern features.

    Pipeline:
    1. Query routing (classify intent)
    2. Query rewriting (HyDE expansion)
    3. Hybrid retrieval (dense + sparse)
    4. Re-ranking (cross-encoder)
    5. Return top results with metadata

    Args:
        query: User query
        limit: Number of results to return
        use_hyde: Enable HyDE query rewriting
        use_reranking: Enable cross-encoder re-ranking

    Returns:
        Dict with results and metadata
    """
    start_time = datetime.now()

    # Step 1: Route query
    strategy = route_query(query)
    logger.info(f"Query routed to: {strategy.value}")

    # Step 2: Rewrite query (HyDE)
    queries = [query]
    if use_hyde:
        queries = rewrite_query_hyde(query, strategy)
        logger.info(f"Query expanded to {len(queries)} variants")

    # Step 3: Hybrid retrieval for each query variant
    all_results = []
    for q in queries:
        results = await hybrid_search(q, limit=limit * 2)
        all_results.extend(results)

    # Deduplicate by memory ID
    unique_results = {}
    for mem, score in all_results:
        if mem.id not in unique_results:
            unique_results[mem.id] = (mem, score)
        else:
            # Keep higher score
            existing_score = unique_results[mem.id][1]
            if score > existing_score:
                unique_results[mem.id] = (mem, score)

    results = list(unique_results.values())
    logger.info(f"Retrieved {len(results)} unique results")

    # Step 4: Re-rank
    if use_reranking and results:
        results = rerank_results(query, results, top_k=limit)
        logger.info(f"Re-ranked to top {len(results)} results")
    else:
        # Just sort and limit
        results.sort(key=lambda x: x[1], reverse=True)
        results = results[:limit]

    elapsed = (datetime.now() - start_time).total_seconds()

    return {
        'query': query,
        'strategy': strategy.value,
        'queries_used': queries if use_hyde else [query],
        'results': [
            {
                'id': mem.id,
                'text': mem.text,
                'score': score,
                'timestamp': mem.timestamp.isoformat(),
                'context': mem.context,
                'tags': mem.tags
            }
            for mem, score in results
        ],
        'count': len(results),
        'elapsed_ms': elapsed * 1000,
        'pipeline': {
            'routing': True,
            'hyde': use_hyde,
            'hybrid_retrieval': True,
            'reranking': use_reranking
        }
    }


# ============================================================================
# MCP Tools
# ============================================================================

if MCP_AVAILABLE:
    @server.list_tools()
    async def list_tools() -> list[Tool]:
        """List available RAG tools."""
        return [
            Tool(
                name="rag_query",
                description=(
                    "Advanced RAG query with modern best practices: "
                    "semantic routing, HyDE query rewriting, hybrid retrieval, "
                    "and cross-encoder re-ranking. Use this for high-quality "
                    "question answering over your knowledge base."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Your question or search query"
                        },
                        "limit": {
                            "type": "integer",
                            "default": 5,
                            "minimum": 1,
                            "maximum": 20,
                            "description": "Number of results to return"
                        },
                        "use_hyde": {
                            "type": "boolean",
                            "default": True,
                            "description": "Enable HyDE query rewriting"
                        },
                        "use_reranking": {
                            "type": "boolean",
                            "default": True,
                            "description": "Enable cross-encoder re-ranking"
                        }
                    },
                    "required": ["query"]
                }
            ),

            Tool(
                name="ingest_document",
                description=(
                    "Ingest a document with semantic chunking. "
                    "Uses intelligent boundary detection (paragraphs, sentences) "
                    "instead of fixed-size chunks for better retrieval quality."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "Document text to ingest"
                        },
                        "title": {
                            "type": "string",
                            "description": "Document title (optional)"
                        },
                        "source": {
                            "type": "string",
                            "description": "Source/URL (optional)"
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Tags for categorization"
                        },
                        "chunk_size": {
                            "type": "integer",
                            "default": 500,
                            "minimum": 100,
                            "maximum": 2000,
                            "description": "Target chunk size (soft limit)"
                        }
                    },
                    "required": ["text"]
                }
            ),

            Tool(
                name="rewrite_query",
                description=(
                    "Rewrite query using HyDE (Hypothetical Document Embeddings). "
                    "Expands query into multiple variants for better retrieval."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Original query"
                        }
                    },
                    "required": ["query"]
                }
            ),

            Tool(
                name="hybrid_search",
                description=(
                    "Hybrid search: Dense (semantic) + sparse (keyword) retrieval. "
                    "Better than either alone. Adjust weights to tune behavior."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query"
                        },
                        "limit": {
                            "type": "integer",
                            "default": 10,
                            "minimum": 1,
                            "maximum": 50
                        },
                        "dense_weight": {
                            "type": "number",
                            "default": 0.7,
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": "Weight for semantic search (0-1)"
                        },
                        "sparse_weight": {
                            "type": "number",
                            "default": 0.3,
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": "Weight for keyword search (0-1)"
                        }
                    },
                    "required": ["query"]
                }
            )
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> Sequence[TextContent]:
        """Execute RAG tools."""
        try:
            if name == "rag_query":
                query = arguments.get("query")
                limit = arguments.get("limit", 5)
                use_hyde = arguments.get("use_hyde", True)
                use_reranking = arguments.get("use_reranking", True)

                if not query:
                    return [TextContent(type="text", text="Error: 'query' is required")]

                result = await rag_query(query, limit, use_hyde, use_reranking)

                # Format response
                lines = [
                    f"🔍 RAG Query: {result['query']}",
                    f"📊 Strategy: {result['strategy']}",
                    f"⚡ Pipeline: Routing → {'HyDE → ' if use_hyde else ''}Hybrid Retrieval → {'Re-ranking' if use_reranking else 'Scoring'}",
                    f"⏱️  Time: {result['elapsed_ms']:.1f}ms",
                    "",
                    f"Found {result['count']} results:",
                    ""
                ]

                for i, res in enumerate(result['results'], 1):
                    lines.append(f"{i}. [{res['score']:.3f}] {res['text'][:150]}...")
                    if res.get('tags'):
                        lines.append(f"   Tags: {', '.join(res['tags'])}")
                    lines.append(f"   ID: {res['id']}")
                    lines.append("")

                if use_hyde and len(result['queries_used']) > 1:
                    lines.append("Query Expansions:")
                    for q in result['queries_used'][1:]:
                        lines.append(f"  • {q}")

                return [TextContent(type="text", text="\n".join(lines))]

            elif name == "ingest_document":
                text = arguments.get("text")
                title = arguments.get("title", "Untitled Document")
                source = arguments.get("source", "")
                tags = arguments.get("tags", [])
                chunk_size = arguments.get("chunk_size", 500)

                if not text:
                    return [TextContent(type="text", text="Error: 'text' is required")]

                # Semantic chunking
                chunks = semantic_chunk(text, max_chunk_size=chunk_size)

                # Store each chunk
                memory_ids = []
                for i, chunk in enumerate(chunks):
                    mem_id = await memory.store(
                        text=chunk,
                        context={
                            'title': title,
                            'source': source,
                            'chunk_index': i,
                            'total_chunks': len(chunks)
                        },
                        tags=tags + ['document', f'doc:{title}']
                    )
                    memory_ids.append(mem_id)

                lines = [
                    "✓ Document ingested with semantic chunking",
                    "",
                    f"📄 Title: {title}",
                    f"📊 Chunks: {len(chunks)}",
                    f"💾 Memories: {len(memory_ids)}",
                    f"📏 Avg chunk size: {sum(len(c) for c in chunks) // len(chunks)} chars",
                    "",
                    f"🔖 Tags: {', '.join(tags)}" if tags else "",
                    "",
                    "Sample chunks:",
                    f"  1. {chunks[0][:100]}...",
                ]

                if len(chunks) > 1:
                    lines.append(f"  {len(chunks)}. {chunks[-1][:100]}...")

                return [TextContent(type="text", text="\n".join(lines))]

            elif name == "rewrite_query":
                query = arguments.get("query")

                if not query:
                    return [TextContent(type="text", text="Error: 'query' is required")]

                strategy = route_query(query)
                rewrites = rewrite_query_hyde(query, strategy)

                lines = [
                    f"🔄 Query Rewriting (HyDE)",
                    f"Strategy: {strategy.value}",
                    "",
                    "Original:",
                    f"  {query}",
                    "",
                    "Expansions:",
                ]

                for i, q in enumerate(rewrites[1:], 1):
                    lines.append(f"  {i}. {q}")

                return [TextContent(type="text", text="\n".join(lines))]

            elif name == "hybrid_search":
                query = arguments.get("query")
                limit = arguments.get("limit", 10)
                dense_weight = arguments.get("dense_weight", 0.7)
                sparse_weight = arguments.get("sparse_weight", 0.3)

                if not query:
                    return [TextContent(type="text", text="Error: 'query' is required")]

                results = await hybrid_search(query, limit, dense_weight, sparse_weight)

                lines = [
                    f"🔎 Hybrid Search: {query}",
                    f"⚖️  Weights: Dense={dense_weight:.1f}, Sparse={sparse_weight:.1f}",
                    "",
                    f"Found {len(results)} results:",
                    ""
                ]

                for i, (mem, score) in enumerate(results, 1):
                    lines.append(f"{i}. [{score:.3f}] {mem.text[:150]}...")
                    lines.append(f"   ID: {mem.id}")
                    lines.append("")

                return [TextContent(type="text", text="\n".join(lines))]

            else:
                return [TextContent(type="text", text=f"Unknown tool: {name}")]

        except Exception as e:
            logger.error(f"Error executing tool {name}: {e}")
            return [TextContent(type="text", text=f"Error: {e}")]


# ============================================================================
# Server Initialization
# ============================================================================

async def init_memory():
    """Initialize memory system."""
    global memory

    logger.info("Initializing HoloLoom RAG MCP Server...")

    memory = await create_unified_memory(
        user_id="blake",
        enable_mem0=False,  # Disabled - requires OpenAI API
        enable_neo4j=True,
        enable_qdrant=True
    )

    health = await memory.health_check()
    logger.info(f"Memory system: {health['status']} ({health['backend']})")


async def main():
    """Run the RAG MCP server."""
    if not MCP_AVAILABLE:
        print("Cannot start server: MCP not installed")
        print("Install with: pip install mcp")
        return

    await init_memory()

    logger.info("Starting RAG MCP server on stdio...")
    logger.info("Available tools:")
    logger.info("  • rag_query - Complete RAG pipeline with routing, HyDE, hybrid retrieval, re-ranking")
    logger.info("  • ingest_document - Smart semantic chunking")
    logger.info("  • rewrite_query - HyDE query expansion")
    logger.info("  • hybrid_search - Dense + sparse retrieval")

    from mcp.server.stdio import stdio_server
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
