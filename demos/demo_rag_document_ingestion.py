"""
Demo: Batch Document Ingestion
================================
Demonstrates ingesting multiple documents from different sources and then
querying across the entire knowledge base.

Shows:
    - Multiple document ingestion with progress tracking
    - Getting metrics (number of memories formed)
    - Querying across all ingested content
    - Source attribution (which documents were retrieved)

Usage:
    PYTHONPATH=. python demos/demo_rag_document_ingestion.py

Output:
    - Progress for each document
    - System metrics after ingestion
    - Query results with source attribution
"""

import asyncio
import sys
from pathlib import Path

# Add repo to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hololoom.rag import SimpleRAG


async def main():
    """Run document ingestion demo."""
    print("=" * 70)
    print("Demo: Batch Document Ingestion")
    print("=" * 70)
    print()

    try:
        async with SimpleRAG() as rag:
            # 1. Define documents
            print("1. Loading documents...")
            documents = [
                {
                    "title": "Thompson Sampling Introduction",
                    "content": (
                        "Thompson Sampling is a Bayesian approach to the "
                        "multi-armed bandit problem. It maintains a posterior "
                        "distribution over each arm's reward probability and "
                        "samples from these distributions to guide exploration."
                    )
                },
                {
                    "title": "Bayesian Methods",
                    "content": (
                        "Bayesian inference combines prior beliefs with observed "
                        "data to form posterior distributions. This framework is "
                        "fundamental to Thompson Sampling and other decision-making "
                        "algorithms. It naturally represents uncertainty."
                    )
                },
                {
                    "title": "Multi-Armed Bandits",
                    "content": (
                        "The multi-armed bandit problem models the exploration vs "
                        "exploitation tradeoff. An agent must decide whether to "
                        "exploit the best-known option or explore alternatives. "
                        "Thompson Sampling provides an elegant solution."
                    )
                },
                {
                    "title": "Exploration Strategies",
                    "content": (
                        "Exploration strategies include epsilon-greedy, UCB, and "
                        "Thompson Sampling. Thompson Sampling is particularly "
                        "effective because it learns from uncertainty and adapts "
                        "exploration naturally."
                    )
                },
            ]
            print(f"   ✓ Loaded {len(documents)} documents")

            # 2. Ingest documents
            print("\n2. Ingesting documents...")
            for i, doc in enumerate(documents, 1):
                await rag.ingest(doc['content'])
                print(f"   [{i}/{len(documents)}] {doc['title']}")

            print(f"\n   ✓ All {len(documents)} documents ingested")

            # 3. Get metrics
            print("\n3. System metrics after ingestion:")
            metrics = rag.get_metrics()
            print(f"   Total memories: {metrics.get('n_memories', 'unknown')}")
            print(f"   Cache size: {metrics.get('cache_size', 0)}")
            print(f"   LLM available: {metrics.get('llm_available', False)}")

            # 4. Query across documents
            print("\n4. Querying knowledge base...")
            questions = [
                "What is Thompson Sampling?",
                "How does Bayesian inference work?",
                "What is the multi-armed bandit problem?",
            ]

            for i, question in enumerate(questions, 1):
                print(f"\n   Query {i}: {question}")
                result = await rag.query(question, max_sources=3)

                # Display result
                print(f"   Response: {result.response[:80]}...")
                print(f"   Sources retrieved: {len(result.sources)}")
                if result.sources:
                    print(f"      Source 1: {result.sources[0][:60]}...")
                print(f"   Confidence: {result.confidence:.2f}")

            # 5. Summary
            print("\n5. Summary:")
            print(f"   ✓ Ingested {len(documents)} documents")
            print(f"   ✓ Formed {metrics.get('n_memories', 'unknown')} memories")
            print(f"   ✓ Queried {len(questions)} questions")
            print(f"   ✓ Retrieved sources across all documents")
            print()

        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
