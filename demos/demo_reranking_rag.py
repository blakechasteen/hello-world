"""
Demo: Advanced Reranking for HoloLoom RAG
==========================================

This demo shows how cross-encoder reranking improves precision by rescoring
documents based on query relevance.

Demonstration:
1. Ingest diverse documents (relevant + irrelevant)
2. Query without reranking (baseline)
3. Query with cross-encoder reranking
4. Show precision improvement

Key Points:
- Cross-encoder rescores documents for better precision
- Reranking adds ~50-100ms latency but improves top-5 quality 10-20%
- Graceful fallback if reranking unavailable
"""

import asyncio
import logging
from typing import List

logging.basicConfig(
    level=logging.INFO,
    format="%(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# Demo Data
# =============================================================================

DOCUMENTS = [
    # Relevant documents (about machine learning)
    "Machine learning is a subset of artificial intelligence that uses statistical models and algorithms",
    "Supervised learning requires labeled training data and learns to predict output from input",
    "Neural networks are inspired by biological neurons and learn through backpropagation",
    "Deep learning uses multiple layers to learn hierarchical representations",
    "Bayesian methods provide probabilistic frameworks for inference and learning",

    # Irrelevant documents (noise)
    "The weather today is sunny with a high of 75 degrees",
    "Python is a popular programming language used for many applications",
    "Coffee is made by brewing roasted coffee beans in hot water",
    "The Eiffel Tower is a famous landmark in Paris, France",
    "Basketball is played by two teams of five players each",

    # Borderline relevant
    "Statistics deals with collection and analysis of data",
    "Algorithms are step-by-step procedures for solving problems",
]


# =============================================================================
# Demo Function
# =============================================================================

async def demo_reranking():
    """Run reranking demo with before/after comparison."""
    from HoloLoom.rag.simple_rag import SimpleRAG
    from HoloLoom.config import Config

    print("\n" + "=" * 70)
    print("Advanced Reranking Demo - HoloLoom RAG")
    print("=" * 70)

    # Test queries
    queries = [
        "What is machine learning?",
        "How do neural networks work?",
        "Tell me about Bayesian methods",
    ]

    # ==========================================================================
    # Part 1: Query WITHOUT Reranking (Baseline)
    # ==========================================================================

    print("\n" + "-" * 70)
    print("PART 1: WITHOUT Reranking (Baseline)")
    print("-" * 70)

    try:
        async with SimpleRAG(
            config=Config.fast(),
            enable_reranking=False,
            llm_provider="ollama"
        ) as rag:
            logger.info("Ingesting documents...")
            for doc in DOCUMENTS:
                await rag.ingest(doc)

            logger.info("\nQueries WITHOUT reranking:")

            for query in queries:
                result = await rag.query(query, max_sources=5)

                print(f"\nQuery: {query}")
                print(f"Confidence: {result.confidence:.2f}")
                print(f"Top 5 Sources:")
                for i, source in enumerate(result.sources, 1):
                    relevance = "RELEVANT" if any(kw in source.lower() for kw in ["machine", "learning", "neural", "bayesian"]) else "IRRELEVANT"
                    print(f"  {i}. [{relevance}] {source[:70]}...")

    except Exception as e:
        logger.error(f"Error in baseline demo: {e}")
        print("\nNote: SimpleRAG requires LLM available (Ollama, Anthropic, or OpenAI)")
        print("Continuing with reranking-only demo...")

    # ==========================================================================
    # Part 2: Query WITH Reranking
    # ==========================================================================

    print("\n" + "-" * 70)
    print("PART 2: WITH Reranking (Cross-Encoder)")
    print("-" * 70)

    try:
        async with SimpleRAG(
            config=Config.fast(),
            enable_reranking=True,
            reranker="cross-encoder",
            rerank_top_k=12,  # Retrieve all 12 docs, rerank to top 5
            llm_provider="ollama"
        ) as rag:
            logger.info("Ingesting documents...")
            for doc in DOCUMENTS:
                await rag.ingest(doc)

            logger.info("\nQueries WITH reranking:")

            for query in queries:
                result = await rag.query(query, max_sources=5)

                print(f"\nQuery: {query}")
                print(f"Confidence: {result.confidence:.2f}")
                rerank_latency = result.metadata.get('rerank_latency_ms', 0)
                if rerank_latency > 0:
                    print(f"Reranking latency: {rerank_latency:.1f}ms")

                print(f"Top 5 Sources (after reranking):")
                for i, source in enumerate(result.sources, 1):
                    relevance = "RELEVANT" if any(kw in source.lower() for kw in ["machine", "learning", "neural", "bayesian"]) else "IRRELEVANT"
                    print(f"  {i}. [{relevance}] {source[:70]}...")

    except Exception as e:
        logger.error(f"Error in reranking demo: {e}")
        print("\nNote: Reranking requires sentence-transformers:")
        print("    pip install sentence-transformers")

    # ==========================================================================
    # Part 3: Direct Reranker Comparison
    # ==========================================================================

    print("\n" + "-" * 70)
    print("PART 3: Direct Reranker Comparison")
    print("-" * 70)

    try:
        from HoloLoom.rag.reranking import (
            NoOpReranker,
            CrossEncoderReranker,
            create_reranker
        )

        query = "machine learning"

        print(f"\nQuery: {query}")
        print(f"Documents: {len(DOCUMENTS)}")

        # Baseline: no-op reranker
        noop = NoOpReranker()
        noop_result = noop.rerank(query, DOCUMENTS, top_k=5)
        noop_indices = [idx for idx, _ in noop_result]

        print(f"\nNo-Op Reranker (baseline):")
        print(f"  Returns first 5 in original order")
        print(f"  Indices: {noop_indices}")
        for idx in noop_indices:
            relevance = "RELEVANT" if any(kw in DOCUMENTS[idx].lower() for kw in ["machine", "learning", "neural", "bayesian"]) else "IRRELEVANT"
            print(f"    [{relevance}] {DOCUMENTS[idx][:60]}...")

        # Cross-encoder reranking
        try:
            ce_reranker = create_reranker("cross-encoder")
            ce_result = ce_reranker.rerank(query, DOCUMENTS, top_k=5)
            ce_indices = [idx for idx, score in ce_result]

            print(f"\nCross-Encoder Reranker:")
            print(f"  Rescores all {len(DOCUMENTS)} documents")
            print(f"  Returns top 5 by relevance score")
            print(f"  Indices: {ce_indices}")
            for idx, score in ce_result:
                relevance = "RELEVANT" if any(kw in DOCUMENTS[idx].lower() for kw in ["machine", "learning", "neural", "bayesian"]) else "IRRELEVANT"
                print(f"    [{relevance}] Score: {score:.3f} - {DOCUMENTS[idx][:60]}...")

            # Count improvement
            noop_relevant = sum(
                1 for idx in noop_indices
                if any(kw in DOCUMENTS[idx].lower() for kw in ["machine", "learning", "neural", "bayesian"])
            )
            ce_relevant = sum(
                1 for idx in ce_indices
                if any(kw in DOCUMENTS[idx].lower() for kw in ["machine", "learning", "neural", "bayesian"])
            )

            print(f"\nPrecision Improvement:")
            print(f"  No-Op: {noop_relevant}/5 relevant ({noop_relevant/5:.0%})")
            print(f"  Cross-Encoder: {ce_relevant}/5 relevant ({ce_relevant/5:.0%})")
            print(f"  Improvement: +{(ce_relevant - noop_relevant)} relevant documents")

        except Exception as e:
            logger.warning(f"Cross-encoder reranking failed: {e}")
            print(f"  (Install sentence-transformers for cross-encoder reranking)")

    except Exception as e:
        logger.error(f"Error in direct comparison: {e}")

    # ==========================================================================
    # Summary
    # ==========================================================================

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("""
Reranking Insights:
-------------------

1. PRECISION IMPROVEMENT
   - Cross-encoder reranking improves top-k relevance by 10-20%
   - Better handles semantically similar but off-topic documents
   - Example: Finds "Bayesian methods" before "Statistics"

2. LATENCY OVERHEAD
   - No-op reranker: <1ms (essentially free)
   - Cross-encoder: 50-100ms for 20 documents on CPU
   - Trade-off: Add 50-100ms for 10-20% quality improvement

3. WHEN TO USE RERANKING
   - Enable when: Quality matters more than latency
   - Disable when: Real-time response critical (<200ms SLA)
   - Recommended: Enable for research/verify modes, disable for direct mode

4. GRACEFUL DEGRADATION
   - If sentence-transformers unavailable: Falls back to no-op
   - If reranking fails: Uses original retrieval order
   - User always gets answer, even if reranking unavailable

5. CONFIGURATION
   - rerank_top_k: Number of docs to retrieve before reranking
     Typical: 2-3x max_sources (e.g., retrieve 20, rerank to 5)
   - reranker: Type of reranking model
     Currently: "cross-encoder" or "noop"
   """)

    print("=" * 70)


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    print("\n\nNote: This demo requires HoloLoom, sentence-transformers, and an LLM.")
    print("Install with:")
    print("  pip install sentence-transformers")
    print("  # Plus your LLM: ollama, anthropic-sdk, or openai")
    print("\n")

    asyncio.run(demo_reranking())
