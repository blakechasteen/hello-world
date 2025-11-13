"""
Demo: Simple Q&A Workflow
==========================
Demonstrates the absolute simplest RAG use case: ingest one piece of content,
ask one question, get an answer back.

This is the best starting point for learning SimpleRAG.

Usage:
    PYTHONPATH=. python demos/demo_rag_qa_simple.py

Output:
    - Shows zero-config initialization
    - Ingests one document
    - Queries the knowledge base
    - Displays: response, sources, confidence score
"""

import asyncio
import sys
from pathlib import Path

# Add repo to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from HoloLoom.rag import SimpleRAG


async def main():
    """Run simple Q&A demo."""
    print("=" * 70)
    print("Demo: Simple Q&A Workflow")
    print("=" * 70)
    print()

    try:
        # 1. Initialize (zero-config)
        print("1. Initializing SimpleRAG...")
        async with SimpleRAG() as rag:
            print("   ✓ Ready!")

            # 2. Ingest content
            print("\n2. Ingesting content about Thompson Sampling...")
            content = (
                "Thompson Sampling is a Bayesian approach to the multi-armed "
                "bandit problem that balances exploration and exploitation by "
                "sampling from posterior distributions. It's particularly "
                "effective because it naturally adjusts exploration based on "
                "uncertainty. At each step, Thompson Sampling samples a value "
                "from each arm's posterior distribution and plays the arm with "
                "the highest sample value. This simple algorithm is both "
                "theoretically optimal and practically efficient."
            )

            await rag.ingest(content)
            print("   ✓ Ingested 1 document")

            # 3. Query
            print("\n3. Querying knowledge base...")
            question = "What is Thompson Sampling?"
            print(f"   Question: {question}")

            result = await rag.query(question)

            # 4. Display results
            print("\n4. Results:")
            print(f"   Response: {result.response[:120]}...")
            print(f"   Sources retrieved: {len(result.sources)}")
            if result.sources:
                print(f"      Source 1: {result.sources[0][:70]}...")
            print(f"   Confidence: {result.confidence:.2f}")
            print(f"   Reasoning mode: {result.reasoning_mode}")

            # 5. Summary
            print("\n5. Summary:")
            print("   ✓ Simple Q&A workflow complete!")
            print("   ✓ Demonstrates zero-config initialization")
            print("   ✓ Shows single ingest + query + display pattern")
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
