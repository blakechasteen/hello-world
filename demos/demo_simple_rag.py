"""
Simple RAG Demo

Demonstrates zero-config RAG with HoloLoom.

Features:
- Zero-config initialization
- Simple ingestion (any modality)
- Clean query interface
- Structured results (response, sources, confidence)
"""

import asyncio
import sys
from pathlib import Path

# Add repo to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from hololoom.rag import SimpleRAG, RAGResult


def print_header(text: str):
    """Print formatted header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)
    sys.stdout.flush()


def print_result(result: RAGResult, query: str):
    """Pretty-print a RAG result."""
    print(f"\nQuery: {query}")
    print("-" * 70)
    print(f"\nResponse:")
    print(f"   {result.response}")

    print(f"\nSources ({len(result.sources)}):")
    if result.sources:
        for i, source in enumerate(result.sources, 1):
            # Truncate long sources
            source_text = source[:70] + "..." if len(source) > 70 else source
            print(f"   [{i}] {source_text}")
    else:
        print("   (none)")

    print(f"\nConfidence: {result.confidence:.1%}")
    print(f"Mode: {result.reasoning_mode}")
    print(f"Metadata: {result.metadata}")
    sys.stdout.flush()


async def demo_basic():
    """Basic RAG workflow: ingest -> query."""
    print_header("Basic RAG Workflow")

    async with SimpleRAG() as rag:
        print("\n[1] Ingesting content...")

        # Ingest some content
        contents = [
            "Thompson Sampling is a Bayesian approach to the multi-armed bandit problem.",
            "It uses posterior distributions to balance exploration and exploitation.",
            "Thompson Sampling maintains a prior distribution over each arm's reward.",
            "At each step, it samples from the posterior and plays the best arm.",
            "The method is theoretically optimal and practically efficient.",
        ]

        for i, content in enumerate(contents, 1):
            await rag.ingest(content)
            print(f"   [OK] Ingested {i}/{len(contents)}")

        print("\n[2] Querying knowledge base...")

        # Query the knowledge base
        questions = [
            "What is Thompson Sampling?",
            "How does Thompson Sampling work?",
            "Why is Thompson Sampling efficient?",
        ]

        for question in questions:
            print(f"\n   Query: {question}")
            result = await rag.query(question)
            print_result(result, question)

        print("\n[3] System metrics:")
        metrics = rag.get_metrics()
        print(f"   Memories: {metrics.get('n_memories', 0)}")
        print(f"   Cache size: {metrics.get('cache_size', 0)}")
        print(f"   Cache hit rate: {metrics.get('cache_hit_rate', 0):.1%}")


async def demo_batch():
    """Batch processing demonstration."""
    print_header("Batch Query Processing")

    async with SimpleRAG() as rag:
        print("\n[1] Ingesting knowledge base...")

        # Ingest content
        knowledge = {
            "Reinforcement Learning": [
                "Reinforcement learning trains agents through reward signals.",
                "The agent learns a policy that maximizes cumulative reward.",
                "Q-learning is a value-based RL algorithm.",
                "Policy gradient methods directly optimize the policy.",
            ],
        }

        for topic, facts in knowledge.items():
            for fact in facts:
                await rag.ingest(fact)

        print(f"   [OK] Ingested {sum(len(v) for v in knowledge.values())} facts")

        print("\n[2] Batch querying...")

        questions = [
            "What is reinforcement learning?",
            "What is Q-learning?",
            "What are policy gradients?",
        ]

        print(f"\n   Processing {len(questions)} queries...")
        results = await rag.batch_query(questions)

        print(f"\n   Results:")
        for question, result in zip(questions, results):
            confidence_bar = "#" * int(result.confidence * 10)
            print(f"   [{confidence_bar:<10}] {question}")


async def demo_multimodal():
    """Demonstrate multimodal ingestion (graceful fallback)."""
    print_header("Multimodal Content Ingestion")

    async with SimpleRAG() as rag:
        print("\nSimpleRAG supports any modality through HoloLoom:")
        print("  - Text strings")
        print("  - Structured data (dicts, JSON)")
        print("  - File paths (PDF, image, etc.)")
        print("  - Bytes")

        print("\nDemo: Ingesting mixed content...")

        try:
            # Text
            await rag.ingest("Thompson Sampling uses Bayesian statistics")
            print("  [OK] Text ingested")

            # Structured
            await rag.ingest({"algorithm": "Thompson Sampling", "type": "bandit"})
            print("  [OK] Structured data ingested")

            # Query
            result = await rag.query("What algorithms did we ingest?")
            print(f"\n  Query result: {result.response[:80]}...")

        except Exception as e:
            print(f"  [WARN] Multimodal error (expected if dependencies missing): {e}")


async def demo_performance():
    """Demonstrate caching and performance."""
    print_header("Caching and Performance")

    async with SimpleRAG(enable_caching=True) as rag:
        print("\n[1] Ingesting content...")

        await rag.ingest(
            "Policy gradient methods optimize the policy directly. "
            "They use gradient ascent on the expected return."
        )
        print("   [OK] Content ingested")

        print("\n[2] Performance: repeated queries (with caching)...")

        question = "What are policy gradients?"

        # First query (cold cache)
        print(f"\n   Query: {question}")
        result1 = await rag.query(question)
        print(f"   Response: {result1.response[:60]}...")
        print(f"   Cache hit: {result1.metadata.get('cache_hit', False)}")

        # Second query (warm cache)
        print(f"\n   Query (again): {question}")
        result2 = await rag.query(question)
        print(f"   Response: {result2.response[:60]}...")
        print(f"   Cache hit: {result2.metadata.get('cache_hit', False)}")

        # Same response
        assert result1.response == result2.response
        print("\n   [OK] Identical cached response")

        print("\n[3] Cache statistics:")
        metrics = rag.get_metrics()
        print(f"   Cache size: {metrics.get('cache_size', 0)}")
        print(f"   Cache hit rate: {metrics.get('cache_hit_rate', 0):.1%}")


async def main():
    """Run all demos."""
    print("\n" + "=" * 70)
    print("  SimpleRAG Demo - Zero-Config RAG for HoloLoom")
    print("=" * 70)

    try:
        # Run demos
        await demo_basic()
        await demo_batch()
        await demo_multimodal()
        await demo_performance()

        print_header("[OK] All Demos Complete")
        print("\nSimpleRAG features demonstrated:")
        print("  [OK] Zero-config initialization")
        print("  [OK] Multimodal ingestion (text, structured, files)")
        print("  [OK] Clean query API with structured results")
        print("  [OK] Batch processing")
        print("  [OK] Query caching for performance")
        print("  [OK] System metrics and monitoring")

    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
