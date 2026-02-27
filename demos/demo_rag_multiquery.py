"""
Demo: Multi-Query Research Workflow
=====================================
Demonstrates research-style querying with batch_query() API.

Shows:
    - Ingesting comprehensive knowledge base
    - Batch query processing (multiple questions at once)
    - Efficiency of batch processing
    - Comparing results across queries
    - Aggregating insights from multiple questions

Usage:
    PYTHONPATH=. python demos/demo_rag_multiquery.py

Output:
    - Knowledge base setup
    - Batch query results
    - Comparison across questions
    - Aggregated statistics (average confidence, etc.)
"""

import asyncio
import sys
from pathlib import Path

# Add repo to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hololoom.rag import SimpleRAG


async def main():
    """Run multi-query research demo."""
    print("=" * 70)
    print("Demo: Multi-Query Research Workflow")
    print("=" * 70)
    print()

    try:
        async with SimpleRAG() as rag:
            # 1. Build comprehensive knowledge base
            print("1. Building comprehensive knowledge base...")
            knowledge_base = [
                "Thompson Sampling uses Bayesian posterior distributions to "
                "balance exploration and exploitation in decision making.",

                "The multi-armed bandit problem asks: should I exploit the "
                "best option I know, or explore alternatives? Thompson Sampling "
                "answers this elegantly.",

                "Epsilon-greedy explores randomly with probability epsilon and "
                "exploits the best known option otherwise. It's simpler than "
                "Thompson Sampling but less theoretically principled.",

                "UCB (Upper Confidence Bound) uses optimism under uncertainty "
                "to guide exploration. It's different from Thompson Sampling "
                "but also effective.",

                "Thompson Sampling maintains posterior distributions over reward "
                "probabilities. By sampling from these posteriors and selecting "
                "the best sample, it explores naturally without explicit tuning.",

                "Applications of Thompson Sampling include: A/B testing in web "
                "applications, recommendation systems, clinical trials, and "
                "online advertising.",

                "The advantage of Thompson Sampling over epsilon-greedy is that "
                "it adapts exploration based on uncertainty. As you learn more, "
                "exploration decreases automatically.",

                "Thompson Sampling is computationally efficient because it only "
                "requires sampling from posterior distributions, not solving "
                "complex optimization problems.",

                "Common applications include personalized medicine, online "
                "recommendation systems, and adaptive resource allocation.",

                "Thompson Sampling has theoretical guarantees: it achieves "
                "logarithmic regret, which is optimal for the bandit problem.",
            ]

            for i, content in enumerate(knowledge_base, 1):
                await rag.ingest(content)

            print(f"   ✓ Ingested {len(knowledge_base)} knowledge items")

            # 2. Prepare research questions
            print("\n2. Preparing research questions...")
            questions = [
                "What is Thompson Sampling?",
                "How does it balance exploration and exploitation?",
                "What are the advantages over epsilon-greedy?",
                "What are common applications?",
            ]
            print(f"   ✓ Prepared {len(questions)} questions")

            # 3. Batch query processing
            print(f"\n3. Processing {len(questions)} questions in batch...")
            results = await rag.batch_query(questions)
            print(f"   ✓ Completed batch processing")

            # 4. Display individual results
            print("\n4. Individual Query Results:")
            print("-" * 70)
            for i, (question, result) in enumerate(zip(questions, results), 1):
                print(f"\n   Query {i}: {question}")
                print(f"   Response: {result.response[:75]}...")
                print(f"   Sources: {len(result.sources)}")
                print(f"   Confidence: {result.confidence:.2f}")

            # 5. Aggregate statistics
            print("\n5. Aggregate Statistics:")
            print("-" * 70)
            confidences = [r.confidence for r in results]
            avg_confidence = sum(confidences) / len(confidences)
            max_confidence = max(confidences)
            min_confidence = min(confidences)
            total_sources = sum(len(r.sources) for r in results)

            print(f"   Questions processed: {len(results)}")
            print(f"   Average confidence: {avg_confidence:.2f}")
            print(f"   Max confidence: {max_confidence:.2f}")
            print(f"   Min confidence: {min_confidence:.2f}")
            print(f"   Total sources retrieved: {total_sources}")
            print(f"   Avg sources per query: {total_sources/len(results):.1f}")

            # 6. Summary
            print("\n6. Summary:")
            print("   ✓ Multi-query research workflow complete!")
            print("   ✓ Demonstrates batch_query() for efficient processing")
            print("   ✓ Shows aggregation of results across multiple queries")
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
