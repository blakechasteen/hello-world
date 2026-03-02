"""
Demo: RAG System with Business Planning Documents
==================================================
Demonstrates the modern RAG pipeline with your actual business plan.

Usage:
    PYTHONPATH=. python demos/demo_rag_business_plan.py
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hololoom.memory.mcp_rag_server import (
    rag_query,
    init_memory,
    semantic_chunk,
    route_query,
    rewrite_query_hyde
)


async def ingest_business_docs():
    """Ingest business planning documents."""
    print("=" * 70)
    print("STEP 1: Ingesting Business Planning Documents")
    print("=" * 70)

    # Read business plan documents
    docs_dir = Path(__file__).parent.parent / "cos"

    files_to_ingest = [
        "business_plan_analysis.md",
        "first_week_checklist.md",
        "90_day_timeline.md",
        "shopping_list.md"
    ]

    total_chunks = 0

    for filename in files_to_ingest:
        filepath = docs_dir / filename
        if not filepath.exists():
            print(f"⚠️  Skipping {filename} (not found)")
            continue

        print(f"\n📄 Reading: {filename}")
        text = filepath.read_text(encoding='utf-8')
        print(f"   Size: {len(text):,} characters")

        # Semantic chunking
        chunks = semantic_chunk(text, max_chunk_size=500)
        print(f"   Chunks: {len(chunks)} (avg {sum(len(c) for c in chunks) // len(chunks)} chars)")

        # Store each chunk (simulated - would use memory.store in production)
        total_chunks += len(chunks)

        # Show sample
        print(f"   Sample: {chunks[0][:100]}...")

    print(f"\n✓ Total: {len(files_to_ingest)} documents, {total_chunks} chunks")
    print()


async def demo_query_routing():
    """Demo: Semantic routing."""
    print("=" * 70)
    print("STEP 2: Query Routing (Semantic Classification)")
    print("=" * 70)

    queries = [
        "What is the ROI on bread baking?",  # Factual
        "Why is bread better than micro brewing?",  # Analytical
        "How do I start bread baking?",  # Procedural
        "What other business ideas could work?",  # Exploratory
    ]

    for query in queries:
        strategy = route_query(query)
        print(f"Query: {query}")
        print(f"  → Strategy: {strategy.value}")
        print()


async def demo_hyde_rewriting():
    """Demo: HyDE query rewriting."""
    print("=" * 70)
    print("STEP 3: HyDE Query Rewriting")
    print("=" * 70)

    query = "bread baking profitability tradeoffs"
    print(f"Original: {query}")
    print()

    strategy = route_query(query)
    rewrites = rewrite_query_hyde(query, strategy)

    print(f"Strategy: {strategy.value}")
    print("Expansions:")
    for i, rewrite in enumerate(rewrites, 1):
        print(f"  {i}. {rewrite}")
    print()


async def demo_rag_pipeline():
    """Demo: Complete RAG pipeline."""
    print("=" * 70)
    print("STEP 4: Complete RAG Pipeline")
    print("=" * 70)

    # Initialize memory
    await init_memory()

    queries = [
        "What are the quick wins with highest ROI?",
        "How much investment is needed for the first week?",
        "What's the break-even timeline for bread baking?",
    ]

    for query in queries:
        print(f"\n🔍 Query: {query}")
        print("-" * 70)

        result = await rag_query(
            query=query,
            limit=3,
            use_hyde=True,
            use_reranking=True
        )

        print(f"Strategy: {result['strategy']}")
        print(f"Time: {result['elapsed_ms']:.1f}ms")
        print(f"Results: {result['count']}")
        print()

        if result['results']:
            print("Top Results:")
            for i, res in enumerate(result['results'], 1):
                print(f"\n{i}. Score: {res['score']:.3f}")
                print(f"   {res['text'][:200]}...")
                if res['tags']:
                    print(f"   Tags: {', '.join(res['tags'][:3])}")
        else:
            print("(No results - documents not yet ingested)")

        print()


async def demo_comparison():
    """Demo: Compare different RAG configurations."""
    print("=" * 70)
    print("STEP 5: Configuration Comparison")
    print("=" * 70)

    await init_memory()

    query = "What are the most profitable products?"

    print(f"Query: {query}\n")

    configs = [
        ("Baseline (no HyDE, no re-ranking)", False, False),
        ("+ HyDE", True, False),
        ("+ Re-ranking", False, True),
        ("Full Pipeline (HyDE + Re-ranking)", True, True),
    ]

    for name, use_hyde, use_rerank in configs:
        result = await rag_query(
            query=query,
            limit=3,
            use_hyde=use_hyde,
            use_reranking=use_rerank
        )

        print(f"{name}:")
        print(f"  Time: {result['elapsed_ms']:.1f}ms")
        print(f"  Results: {result['count']}")
        if result['results']:
            print(f"  Top score: {result['results'][0]['score']:.3f}")
        print()


async def demo_interactive():
    """Interactive demo (optional)."""
    print("=" * 70)
    print("STEP 6: Interactive RAG Query")
    print("=" * 70)

    await init_memory()

    print("\nExample queries you can try:")
    print("  • What's the total investment needed?")
    print("  • Which product has the best margins?")
    print("  • What should I buy at Costco?")
    print("  • How long until break-even?")
    print()

    # For demo, just run a sample query
    query = "What's the total 90-day investment?"
    print(f"Sample query: {query}\n")

    result = await rag_query(query, limit=3)

    if result['results']:
        print("✓ RAG Response:")
        print("-" * 70)
        for res in result['results'][:2]:
            print(f"[{res['score']:.3f}] {res['text'][:300]}...")
            print()
    else:
        print("(Documents need to be ingested first)")


async def main():
    """Run all demos."""
    print("\n" + "=" * 70)
    print("HoloLoom RAG Demo: Business Planning Intelligence")
    print("=" * 70)
    print()

    try:
        # Step 1: Ingest
        await ingest_business_docs()

        # Step 2: Routing
        await demo_query_routing()

        # Step 3: HyDE
        await demo_hyde_rewriting()

        # Step 4: RAG Pipeline
        print("⚠️  Note: Steps 4-6 require Neo4j/Qdrant running")
        print("    Start with: docker-compose up -d")
        print()

        try:
            await demo_rag_pipeline()
            await demo_comparison()
            await demo_interactive()
        except Exception as e:
            print(f"⚠️  Skipping live queries (memory backend not available)")
            print(f"    Error: {e}")
            print()

        print("=" * 70)
        print("✓ Demo Complete!")
        print("=" * 70)
        print()
        print("Next Steps:")
        print("  1. Start Docker services: docker-compose up -d")
        print("  2. Configure Claude Desktop with mcp_rag_server.py")
        print("  3. Ingest your business docs with ingest_document tool")
        print("  4. Query with rag_query tool")
        print()
        print("See: HoloLoom/memory/RAG_README.md for full documentation")
        print()

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
