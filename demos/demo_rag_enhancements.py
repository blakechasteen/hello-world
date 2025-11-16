"""
Comprehensive RAG Enhancements Demo
====================================

Demonstrates all three major RAG enhancements:
1. SQL Database Integration - Query structured databases alongside vector/graph
2. Multi-hop Reasoning - Follow relationship chains through knowledge graph
3. Streaming Responses - Token-by-token LLM generation for real-time UX

Author: Agent 1 (Claude Code)
Date: November 16, 2025
"""

import asyncio
import logging
import sqlite3
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def demo_sql_integration():
    """
    Demo: SQL Database Integration

    Shows how to query structured databases alongside semantic search.
    """
    print("\n" + "=" * 80)
    print("DEMO 1: SQL Database Integration")
    print("=" * 80 + "\n")

    # Create sample database
    db_path = Path("demos/output/sample_users.db")
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Create schema and populate
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            name TEXT,
            age INTEGER,
            interests TEXT
        )
    """)

    # Sample data
    users = [
        (1, "Alice", 28, "machine learning, Python"),
        (2, "Bob", 35, "web development, JavaScript"),
        (3, "Carol", 42, "data science, statistics"),
        (4, "David", 31, "machine learning, deep learning"),
        (5, "Eve", 26, "natural language processing, AI"),
    ]

    cursor.executemany("INSERT OR REPLACE INTO users VALUES (?, ?, ?, ?)", users)
    conn.commit()
    conn.close()

    print(f"✓ Created sample database: {db_path}")
    print(f"  Tables: users")
    print(f"  Records: {len(users)}\n")

    # Initialize RAG with SQL integration
    from HoloLoom.rag import SimpleRAG, SQLRAGMixin
    from HoloLoom.config import Config

    class SQLEnabledRAG(SimpleRAG, SQLRAGMixin):
        """RAG with SQL integration."""

        def __init__(self, *args, **kwargs):
            # Extract SQL-specific kwargs
            db_connection = kwargs.pop('db_connection', None)
            enable_hybrid_routing = kwargs.pop('enable_hybrid_routing', True)

            # Initialize both mixins
            SimpleRAG.__init__(self, *args, **kwargs)
            SQLRAGMixin.__init__(
                self,
                db_connection=db_connection,
                enable_hybrid_routing=enable_hybrid_routing
            )

        async def _query_semantic_only(self, question: str, max_sources: int):
            """Override to use parent's query method."""
            result = await self.query(question, mode="verify", max_sources=max_sources)

            from HoloLoom.rag.sql_integration import SQLRAGResult
            return SQLRAGResult(
                response=result.response,
                sources=result.sources,
                confidence=result.confidence,
                query_type="semantic",
                metadata=result.metadata
            )

    async with SQLEnabledRAG(
        config=Config.fast(),
        db_connection=f"sqlite:///{db_path}",
        enable_hybrid_routing=True
    ) as rag:
        # Connect SQL components
        await rag.connect_sql(llm_provider=rag.orchestrator if hasattr(rag, 'orchestrator') else None)

        # Ingest semantic knowledge
        await rag.ingest("Machine learning is a subset of artificial intelligence.")
        await rag.ingest("Python is a popular programming language for data science.")
        await rag.ingest("Deep learning uses neural networks with multiple layers.")

        print("✓ Ingested semantic knowledge\n")

        # Query 1: SQL-only (factual lookup)
        print("Query 1: How many users are interested in machine learning?")
        print("  (Expected: SQL-only path)")

        result = await rag.query_with_sql(
            "How many users are interested in machine learning?",
            mode="auto"
        )

        print(f"  Response: {result.response[:150]}...")
        print(f"  Query Type: {result.query_type}")
        print(f"  Confidence: {result.confidence:.2f}")
        if result.sql_query:
            print(f"  SQL Query: {result.sql_query}")
        print()

        # Query 2: Hybrid (SQL + semantic)
        print("Query 2: Show me users interested in AI and explain what it is")
        print("  (Expected: Hybrid path - SQL + semantic)")

        result = await rag.query_with_sql(
            "Show me users interested in AI and explain what it is",
            mode="hybrid"
        )

        print(f"  Response: {result.response[:200]}...")
        print(f"  Query Type: {result.query_type}")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  Semantic sources: {len(result.sources)}")
        print()

        # Show stats
        stats = rag.get_sql_stats()
        print(f"SQL Stats:")
        if 'adapter' in stats:
            print(f"  Total queries: {stats['adapter']['total_queries']}")
            print(f"  Successful: {stats['adapter']['successful_queries']}")
            print(f"  Avg latency: {stats['adapter']['avg_latency_ms']:.1f}ms")
        print()


async def demo_multihop_reasoning():
    """
    Demo: Multi-hop Graph Reasoning

    Shows how to follow relationship chains through knowledge graph.
    """
    print("\n" + "=" * 80)
    print("DEMO 2: Multi-hop Reasoning")
    print("=" * 80 + "\n")

    from HoloLoom.rag import SimpleRAG, MultiHopRAGMixin
    from HoloLoom.config import Config
    from HoloLoom.memory.graph import KG, KGEdge

    class MultiHopRAG(SimpleRAG, MultiHopRAGMixin):
        """RAG with multi-hop reasoning."""

        def __init__(self, *args, **kwargs):
            # Extract multi-hop kwargs
            max_hops = kwargs.pop('max_hops', 3)
            beam_width = kwargs.pop('beam_width', 5)

            # Initialize both mixins
            SimpleRAG.__init__(self, *args, **kwargs)
            MultiHopRAGMixin.__init__(
                self,
                max_hops=max_hops,
                beam_width=beam_width
            )

    async with MultiHopRAG(
        config=Config.fast(),
        max_hops=3,
        beam_width=5
    ) as rag:
        # Build knowledge graph
        kg = KG()

        # Add domain knowledge (transformer architecture)
        edges = [
            KGEdge("attention", "transformer", "USES", 1.0),
            KGEdge("transformer", "neural_network", "IS_A", 1.0),
            KGEdge("BERT", "transformer", "IS_A", 1.0),
            KGEdge("GPT", "transformer", "IS_A", 1.0),
            KGEdge("multi-head_attention", "attention", "IS_A", 1.0),
            KGEdge("self-attention", "attention", "IS_A", 1.0),
            KGEdge("encoder", "transformer", "PART_OF", 0.8),
            KGEdge("decoder", "transformer", "PART_OF", 0.8),
            KGEdge("BERT", "encoder", "USES", 0.9),
            KGEdge("GPT", "decoder", "USES", 0.9),
        ]

        kg.add_edges(edges)
        print(f"✓ Built knowledge graph:")
        print(f"  Nodes: {kg.G.number_of_nodes()}")
        print(f"  Edges: {kg.G.number_of_edges()}\n")

        # Inject KG into RAG (hack for demo)
        if hasattr(rag.loom, 'awareness_graph'):
            rag.loom.awareness_graph.kg = kg

        # Ingest semantic content
        await rag.ingest("Transformers revolutionized natural language processing.")
        await rag.ingest("BERT is a bidirectional transformer model.")
        await rag.ingest("Attention mechanisms allow models to focus on relevant parts.")

        print("✓ Ingested semantic knowledge\n")

        # Multi-hop query
        print("Query: How does attention relate to BERT?")
        print("  (Expected path: attention → transformer → BERT)")

        result = await rag.query_multihop(
            "How does attention relate to BERT?",
            max_hops=3,
            return_paths=True
        )

        print(f"\n  Response: {result.response[:200]}...")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  Paths explored: {result.paths_explored}")
        print(f"  Hops used: {result.hop_count}\n")

        # Show best path
        if result.best_path:
            print("  Best reasoning path:")
            print(f"    Entities: {' → '.join(result.best_path.entities)}")
            print(f"    Relationships: {', '.join(result.best_path.relationships)}")
            print(f"    Confidence: {result.best_path.confidence:.3f}")
            print(f"    Explanation: {result.best_path.explanation}")

        # Show all paths
        if result.reasoning_paths:
            print(f"\n  All paths (top 3):")
            for i, path in enumerate(result.reasoning_paths[:3], 1):
                print(f"    {i}. {' → '.join(path.entities)} (confidence: {path.confidence:.3f})")

        print()


async def demo_streaming_responses():
    """
    Demo: Streaming LLM Responses

    Shows token-by-token generation for real-time UX.
    """
    print("\n" + "=" * 80)
    print("DEMO 3: Streaming Responses")
    print("=" * 80 + "\n")

    from HoloLoom.rag import SimpleRAG
    from HoloLoom.config import Config

    async with SimpleRAG(config=Config.fast()) as rag:
        # Ingest knowledge
        await rag.ingest("Thompson Sampling is a Bayesian approach to the multi-armed bandit problem.")
        await rag.ingest("It balances exploration and exploitation by sampling from posterior distributions.")
        await rag.ingest("Thompson Sampling is optimal in terms of expected regret.")

        print("✓ Ingested knowledge about Thompson Sampling\n")

        # Streaming query
        print("Query: What is Thompson Sampling?")
        print("  (Streaming response token-by-token)")
        print("\n  Response: ", end='', flush=True)

        token_count = 0
        final_token = None

        async for token in rag.query_stream("What is Thompson Sampling?", mode="direct"):
            print(token.text, end='', flush=True)
            token_count += 1
            if token.is_final:
                final_token = token

        print("\n")

        if final_token:
            print(f"  Tokens: {token_count}")
            print(f"  Total text length: {len(final_token.cumulative_text)} chars")
            if 'latency_ms' in final_token.metadata:
                print(f"  Latency: {final_token.metadata['latency_ms']:.1f}ms")
            if 'tokens_per_sec' in final_token.metadata:
                print(f"  Speed: {final_token.metadata['tokens_per_sec']:.1f} tokens/sec")

        print()


async def demo_integrated_workflow():
    """
    Demo: All Three Features Together

    Shows how SQL, multi-hop, and streaming can work together.
    """
    print("\n" + "=" * 80)
    print("DEMO 4: Integrated Workflow (All Features)")
    print("=" * 80 + "\n")

    print("This demo would combine:")
    print("  1. SQL query to retrieve structured user data")
    print("  2. Multi-hop reasoning to connect concepts in knowledge graph")
    print("  3. Streaming the final synthesized response")
    print()
    print("In production, you would:")
    print("  - Create a custom RAG class that inherits from all three mixins")
    print("  - Use SQL to get factual data")
    print("  - Use multi-hop to find reasoning chains")
    print("  - Stream the LLM's synthesized answer in real-time")
    print()
    print("Example query:")
    print('  "Find users interested in deep learning and explain how it relates to neural networks"')
    print()
    print("Processing flow:")
    print("  SQL     → Retrieve users: Alice, David")
    print("  Graph   → Find path: deep_learning → neural_networks → AI")
    print("  Stream  → 'Based on the data, Alice and David are interested...'")
    print()


async def main():
    """Run all demos."""
    print("\n" + "=" * 80)
    print("HoloLoom RAG Enhancements - Comprehensive Demo")
    print("=" * 80)
    print("\nThis demo showcases three major RAG enhancements:")
    print("  1. SQL Database Integration")
    print("  2. Multi-hop Graph Reasoning")
    print("  3. Streaming LLM Responses")
    print()

    try:
        # Run demos sequentially
        await demo_sql_integration()
        await demo_multihop_reasoning()
        await demo_streaming_responses()
        await demo_integrated_workflow()

        print("\n" + "=" * 80)
        print("✓ All demos completed successfully!")
        print("=" * 80 + "\n")

    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        print(f"\n✗ Demo failed: {e}")
        print("\nNote: Some features require additional dependencies:")
        print("  - SQL integration: pip install sqlalchemy pandas")
        print("  - Multi-hop reasoning: pip install networkx")
        print("  - Streaming: Requires LLM provider (Ollama/Anthropic/OpenAI)")


if __name__ == "__main__":
    asyncio.run(main())
