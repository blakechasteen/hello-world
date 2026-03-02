"""
Demo: Custom Embedding Providers for HoloLoom RAG

This demo shows how to use different embedding models with SimpleRAG:
- MatryoshkaEmbedding (default): Fast, multi-scale
- HuggingFaceEmbedding: Any Sentence Transformer model
- OpenAIEmbedding: OpenAI's text-embedding-3-small/large
- CohereEmbedding: Cohere's embed-english-v3.0

Demonstrates:
1. Protocol-based plugin architecture
2. Graceful fallback on errors
3. Zero-copy optimization detection
4. Comparison of retrieval quality across providers
"""

import asyncio
import logging
from typing import List, Tuple

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import RAG system
from hololoom.rag import SimpleRAG
from hololoom.rag.embedding_plugins import (
    MatryoshkaEmbedding,
    HuggingFaceEmbedding,
    OpenAIEmbedding,
    CohereEmbedding,
    create_embedding_provider,
)


async def demo_default_embeddings():
    """Demo 1: Default MatryoshkaEmbedding"""
    print("\n" + "=" * 70)
    print("DEMO 1: Default MatryoshkaEmbedding (384 dims)")
    print("=" * 70)

    async with SimpleRAG() as rag:
        # Ingest sample data
        docs = [
            "Thompson Sampling is a Bayesian approach to exploration-exploitation",
            "It maintains a posterior distribution over the reward function",
            "Each time step, it samples from the posterior and takes the optimistic action",
            "This naturally balances exploration and exploitation",
            "Thompson Sampling is optimal for many bandit problems",
        ]

        for doc in docs:
            await rag.ingest(doc)
            logger.info(f"Ingested: {doc[:50]}...")

        # Query
        question = "What is Thompson Sampling?"
        result = await rag.query(question)

        print(f"\nQuestion: {question}")
        print(f"Response: {result.response}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Sources retrieved: {len(result.sources)}")
        print(f"Embedding provider: {rag.embedding_provider.__class__.__name__}")
        print(f"Embedding dimension: {rag.embedding_provider.dimension}")


async def demo_huggingface_embeddings():
    """Demo 2: HuggingFace Sentence Transformer (if available)"""
    print("\n" + "=" * 70)
    print("DEMO 2: HuggingFace Sentence Transformer (MiniLM-L6-v2, 384 dims)")
    print("=" * 70)

    try:
        # Create custom embedding provider
        embedding = HuggingFaceEmbedding("all-MiniLM-L6-v2")
        print(f"✓ Created HuggingFace embedding provider")
        print(f"  Model: all-MiniLM-L6-v2")
        print(f"  Dimension: {embedding.dimension}")

        async with SimpleRAG(embedding_provider=embedding) as rag:
            # Ingest sample data
            docs = [
                "Thompson Sampling is a Bayesian approach to exploration-exploitation",
                "It maintains a posterior distribution over the reward function",
                "Each time step, it samples from the posterior and takes the optimistic action",
            ]

            for doc in docs:
                await rag.ingest(doc)

            # Query
            question = "What is Thompson Sampling?"
            result = await rag.query(question)

            print(f"\nQuestion: {question}")
            print(f"Response: {result.response[:100]}...")
            print(f"Confidence: {result.confidence:.2f}")
            print(f"Embedding provider: {rag.embedding_provider.__class__.__name__}")
            print(f"Embedding dimension: {rag.embedding_provider.dimension}")

    except ImportError as e:
        print(f"⚠  HuggingFace embedding unavailable: {e}")
        print("  Install with: pip install sentence-transformers")


async def demo_factory_pattern():
    """Demo 3: Factory pattern for provider creation"""
    print("\n" + "=" * 70)
    print("DEMO 3: Factory Pattern for Provider Creation")
    print("=" * 70)

    # Create providers using factory function
    providers_to_test = [
        ("matryoshka", {}),
        ("huggingface", {"model_name": "all-MiniLM-L6-v2"}),
    ]

    for provider_type, kwargs in providers_to_test:
        try:
            print(f"\nCreating {provider_type} provider...")
            provider = create_embedding_provider(provider_type, **kwargs)
            print(f"✓ Created: {type(provider).__name__}")
            print(f"  Dimension: {provider.dimension}")

            # Test encoding
            test_texts = ["hello", "world"]
            embeddings = provider.encode(test_texts)
            print(f"  Encoded {len(test_texts)} texts: shape {embeddings.shape}")

        except Exception as e:
            print(f"⚠  Failed to create {provider_type}: {e}")


async def demo_embedding_quality_comparison():
    """Demo 4: Compare embedding quality across providers"""
    print("\n" + "=" * 70)
    print("DEMO 4: Embedding Quality Comparison")
    print("=" * 70)

    providers = [
        ("MatryoshkaEmbedding (default)", MatryoshkaEmbedding()),
    ]

    # Try to add HuggingFace if available
    try:
        providers.append(("HuggingFace MiniLM", HuggingFaceEmbedding("all-MiniLM-L6-v2")))
    except ImportError:
        pass

    sample_queries = [
        "What is Thompson Sampling?",
        "How does exploration and exploitation work?",
        "Bayesian optimization techniques",
    ]

    for provider_name, provider in providers:
        print(f"\nTesting {provider_name}:")
        print(f"  Dimension: {provider.dimension}")

        try:
            # Encode sample queries
            query_embeddings = provider.encode(sample_queries)
            print(f"  Successfully encoded {len(sample_queries)} queries")
            print(f"  Embedding shape: {query_embeddings.shape}")

            # Show embedding statistics
            import numpy as np
            mean_norm = np.mean(np.linalg.norm(query_embeddings, axis=1))
            print(f"  Mean embedding norm: {mean_norm:.4f}")

        except Exception as e:
            print(f"  ⚠  Error: {e}")


async def demo_protocol_compliance():
    """Demo 5: Protocol compliance validation"""
    print("\n" + "=" * 70)
    print("DEMO 5: Protocol Compliance Validation")
    print("=" * 70)

    from hololoom.rag.embedding_plugins import validate_embedding_provider

    providers_to_validate = [
        ("MatryoshkaEmbedding", MatryoshkaEmbedding()),
    ]

    # Try to add HuggingFace if available
    try:
        providers_to_validate.append(
            ("HuggingFaceEmbedding", HuggingFaceEmbedding("all-MiniLM-L6-v2"))
        )
    except ImportError:
        pass

    for name, provider in providers_to_validate:
        print(f"\nValidating {name}...")
        is_valid = validate_embedding_provider(provider)
        if is_valid:
            print(f"✓ Protocol compliance: PASS")
            print(f"  Dimension: {provider.dimension}")
            print(f"  Has encode(): {hasattr(provider, 'encode')}")
            print(f"  Has encode_query(): {hasattr(provider, 'encode_query')}")
        else:
            print(f"✗ Protocol compliance: FAIL")


async def demo_graceful_degradation():
    """Demo 6: Graceful degradation on errors"""
    print("\n" + "=" * 70)
    print("DEMO 6: Graceful Degradation on Errors")
    print("=" * 70)

    class BrokenEmbedding:
        """A broken embedding provider that doesn't satisfy protocol"""
        @property
        def dimension(self) -> str:  # Wrong type!
            return "not_an_int"

    broken_provider = BrokenEmbedding()

    print("\nAttempting to use broken embedding provider...")
    print("Expected: Should fall back to MatryoshkaEmbedding")

    try:
        async with SimpleRAG(embedding_provider=broken_provider) as rag:
            # Should have fallen back to Matryoshka
            print(f"\n✓ Fallback successful!")
            print(f"  Using: {type(rag.embedding_provider).__name__}")
            print(f"  Dimension: {rag.embedding_provider.dimension}")

            # Query should still work
            await rag.ingest("Thompson Sampling test")
            result = await rag.query("What is Thompson Sampling?")
            print(f"  Query succeeded: confidence={result.confidence:.2f}")

    except Exception as e:
        print(f"✗ Unexpected error: {e}")


async def demo_metrics():
    """Demo 7: Embedding provider in metrics"""
    print("\n" + "=" * 70)
    print("DEMO 7: Embedding Provider in System Metrics")
    print("=" * 70)

    async with SimpleRAG() as rag:
        # Ingest sample data
        await rag.ingest("Thompson Sampling balances exploration")
        await rag.ingest("Posterior distributions model uncertainty")

        # Query
        await rag.query("What is Thompson Sampling?")

        # Get metrics
        metrics = rag.get_metrics()
        print(f"\nSystem Metrics:")
        print(f"  Embedding provider: {metrics.get('embedding_provider')}")
        print(f"  Embedding dimension: {metrics.get('embedding_dimension')}")
        print(f"  Cache size: {metrics.get('cache_size')}")
        print(f"  Cache hit rate: {metrics.get('cache_hit_rate'):.1%}")
        print(f"  Memories: {metrics.get('n_memories', 0)}")


async def main():
    """Run all demos"""
    print("\n" + "=" * 70)
    print("CUSTOM EMBEDDINGS PLUGIN DEMO")
    print("=" * 70)

    try:
        await demo_default_embeddings()
        await demo_huggingface_embeddings()
        await demo_factory_pattern()
        await demo_embedding_quality_comparison()
        await demo_protocol_compliance()
        await demo_graceful_degradation()
        await demo_metrics()

    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)

    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    # Run async main
    asyncio.run(main())
