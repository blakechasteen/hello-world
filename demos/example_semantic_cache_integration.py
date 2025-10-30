#!/usr/bin/env python3
"""
Example: Semantic Cache Integration
====================================
Shows how to integrate AdaptiveSemanticCache into existing code.
"""

from HoloLoom.semantic_calculus.dimensions import EXTENDED_244_DIMENSIONS, SemanticSpectrum
from HoloLoom.embedding.spectral import MatryoshkaEmbeddings
from HoloLoom.performance.semantic_cache import AdaptiveSemanticCache


# Example 1: Simple standalone usage
def example_standalone():
    """Simplest usage: standalone semantic analysis with caching."""
    print("=" * 60)
    print("Example 1: Standalone Usage")
    print("=" * 60)

    # Setup
    emb = MatryoshkaEmbeddings(sizes=[96, 192, 384])
    spectrum = SemanticSpectrum(dimensions=EXTENDED_244_DIMENSIONS)
    spectrum.learn_axes(lambda word: emb.encode([word])[0])

    # Create cache (auto-preloads hot tier)
    cache = AdaptiveSemanticCache(spectrum, emb, hot_size=100, warm_size=500)

    # Use it
    queries = [
        "hero's journey",
        "shadow integration",
        "hero's journey",  # Repeated - will hit cache
    ]

    for query in queries:
        scores = cache.get_scores(query)
        top_3 = sorted(scores.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        print(f"\n'{query}':")
        for dim, score in top_3:
            print(f"  {dim}: {score:.3f}")

    cache.print_stats()


# Example 2: Integration with WeavingOrchestrator pattern
class WeavingOrchestratorWithCache:
    """
    Example showing how to integrate semantic cache into WeavingOrchestrator.

    This is a simplified version - real orchestrator has more complexity.
    """

    def __init__(self, enable_semantic_cache=True):
        print("\n" + "=" * 60)
        print("Example 2: WeavingOrchestrator Integration")
        print("=" * 60)

        # Setup embeddings and spectrum
        self.emb = MatryoshkaEmbeddings(sizes=[96, 192, 384])
        self.spectrum = SemanticSpectrum(dimensions=EXTENDED_244_DIMENSIONS)
        self.spectrum.learn_axes(lambda word: self.emb.encode([word])[0])

        # Optional semantic cache
        if enable_semantic_cache:
            print("\n✓ Semantic cache enabled")
            self.semantic_cache = AdaptiveSemanticCache(
                self.spectrum,
                self.emb,
                hot_size=1000,  # Production size
                warm_size=5000
            )
        else:
            print("\n✗ Semantic cache disabled")
            self.semantic_cache = None

    def analyze_query(self, query_text: str):
        """
        Analyze query semantics with optional caching.

        This method shows the pattern: try cache first, fallback to computation.
        """
        if self.semantic_cache:
            # Fast path: Use cache
            semantic_scores = self.semantic_cache.get_scores(query_text)
        else:
            # Slow path: Compute directly
            vec = self.emb.encode([query_text])[0]
            semantic_scores = self.spectrum.project_vector(vec)

        return semantic_scores

    def weave(self, query_text: str):
        """Simplified weaving with semantic analysis."""
        print(f"\nProcessing query: '{query_text}'")

        # Get semantic analysis (cached if available)
        semantic_scores = self.analyze_query(query_text)

        # Use top dimensions to guide weaving
        top_dims = sorted(semantic_scores.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        print("Top semantic dimensions:")
        for dim, score in top_dims:
            print(f"  {dim}: {score:.3f}")

        return {"semantic_scores": semantic_scores, "top_dimensions": top_dims}


# Example 3: Batch processing
def example_batch_processing():
    """Shows batch optimization for multiple queries."""
    print("\n" + "=" * 60)
    print("Example 3: Batch Processing")
    print("=" * 60)

    # Setup
    emb = MatryoshkaEmbeddings(sizes=[96, 192, 384])
    spectrum = SemanticSpectrum(dimensions=EXTENDED_244_DIMENSIONS)
    spectrum.learn_axes(lambda word: emb.encode([word])[0])
    cache = AdaptiveSemanticCache(spectrum, emb)

    # Process batch of queries
    queries = [
        "hero",
        "journey",
        "shadow",
        "transformation",
        "novel phrase example"
    ]

    print(f"\nProcessing batch of {len(queries)} queries...")
    results = cache.get_batch_scores(queries)

    print("\nResults:")
    for query, scores in zip(queries, results):
        top_dim = max(scores.items(), key=lambda x: abs(x[1]))
        print(f"  '{query}': {top_dim[0]} = {top_dim[1]:.3f}")

    cache.print_stats()


def main():
    """Run all examples."""
    # Example 1: Standalone
    example_standalone()

    # Example 2: Integration pattern
    orchestrator_with_cache = WeavingOrchestratorWithCache(enable_semantic_cache=True)
    orchestrator_with_cache.weave("hero's journey into darkness")
    orchestrator_with_cache.weave("hero's journey into darkness")  # Cached!

    if orchestrator_with_cache.semantic_cache:
        orchestrator_with_cache.semantic_cache.print_stats()

    # Example 3: Batch processing
    example_batch_processing()

    print("\n" + "=" * 60)
    print("✅ All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()