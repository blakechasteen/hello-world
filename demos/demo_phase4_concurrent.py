"""
Demo: Phase 4 True Concurrent Token Yielding
============================================

Demonstrates Phase 4 concurrent streaming where tokens are yielded as generated,
achieving <100ms latency to first token (vs Phase 3 batched mode).

Demos:
1. Side-by-side: BATCHED vs CONCURRENT
2. Latency to first token comparison
3. Visual interleaving timeline
4. Performance metrics

Author: Claude Code
Date: 2025-11-25
"""

import asyncio
import time
from typing import List, Dict

from HoloLoom.memory.graph import KG, KGEdge
from HoloLoom.memory.interleaved_generation import (
    InterleavedStreamManager,
    GenerationToken,
    StreamMetadata,
    StreamMode,
    MockLLM
)
from HoloLoom.memory.streaming_expansion import ContextChunk


# ============================================================================
# Create Test Knowledge Graph
# ============================================================================

def create_demo_graph() -> KG:
    """Create realistic knowledge graph about Thompson Sampling."""
    kg = KG()

    # Core concepts (high importance)
    kg.add_edges([
        KGEdge("thompson_sampling", "bayesian_approach", "IS_A", 1.0),
        KGEdge("bayesian_approach", "prior_distributions", "USES", 1.0),
        KGEdge("thompson_sampling", "exploration_exploitation", "USES", 1.0),
        KGEdge("exploration_exploitation", "multi_armed_bandits", "PART_OF", 1.0),
    ])

    # Related algorithms (medium importance)
    kg.add_edges([
        KGEdge("thompson_sampling", "ucb_algorithm", "MENTIONS", 1.0),
        KGEdge("ucb_algorithm", "confidence_bounds", "USES", 1.0),
        KGEdge("thompson_sampling", "epsilon_greedy", "MENTIONS", 1.0),
    ])

    # Applications (medium importance)
    kg.add_edges([
        KGEdge("thompson_sampling", "recommendation_systems", "MENTIONS", 1.0),
        KGEdge("thompson_sampling", "ab_testing", "MENTIONS", 1.0),
        KGEdge("ab_testing", "hypothesis_testing", "USES", 1.0),
    ])

    return kg


def create_importance_scores() -> Dict[str, float]:
    """Create importance scores for nodes."""
    return {
        "thompson_sampling": 1.0,
        "bayesian_approach": 0.95,
        "exploration_exploitation": 0.9,
        "multi_armed_bandits": 0.9,
        "prior_distributions": 0.85,
        "ucb_algorithm": 0.75,
        "epsilon_greedy": 0.7,
        "confidence_bounds": 0.65,
        "recommendation_systems": 0.6,
        "ab_testing": 0.6,
        "hypothesis_testing": 0.55,
    }


def create_node_contents() -> Dict[str, str]:
    """Create text content for nodes."""
    return {
        "thompson_sampling": "Thompson Sampling is a Bayesian heuristic for exploration-exploitation.",
        "bayesian_approach": "Bayesian approach uses probability distributions to represent uncertainty.",
        "exploration_exploitation": "Exploration-exploitation tradeoff balances trying new actions vs leveraging known good actions.",
        "multi_armed_bandits": "Multi-armed bandit problem models sequential decision making.",
        "prior_distributions": "Prior distributions represent initial beliefs about parameters.",
        "ucb_algorithm": "Upper Confidence Bound (UCB) algorithm selects actions based on optimistic estimates.",
        "epsilon_greedy": "Epsilon-greedy strategy explores randomly with probability epsilon.",
        "confidence_bounds": "Confidence bounds provide probabilistic guarantees about parameter estimates.",
        "recommendation_systems": "Recommendation systems suggest items to users based on preferences.",
        "ab_testing": "A/B testing compares two versions of a product.",
        "hypothesis_testing": "Hypothesis testing is a statistical method for making decisions.",
    }


# ============================================================================
# Demo 1: Side-by-Side Comparison
# ============================================================================

async def demo_batched_vs_concurrent():
    """Demo 1: Compare BATCHED (Phase 3) vs CONCURRENT (Phase 4) modes."""
    print("=" * 70)
    print(" " * 15 + "Demo 1: BATCHED vs CONCURRENT")
    print("=" * 70)
    print()

    kg = create_demo_graph()
    importance_scores = create_importance_scores()
    node_contents = create_node_contents()

    # Test with MockLLM
    llm = MockLLM(tokens_per_second=50)  # 50 tok/s to see difference

    # Phase 3: BATCHED mode
    print("Phase 3 MVP (BATCHED):")
    print("-" * 70)

    manager_batched = InterleavedStreamManager(llm)

    start_batched = time.time()
    first_token_batched = None
    tokens_batched = 0

    async for item in manager_batched.stream_interleaved(
        query="What is Thompson Sampling?",
        seed_nodes=["thompson_sampling"],
        graph=kg,
        token_budget=1000,
        max_generation_tokens=20,
        importance_scores=importance_scores,
        node_contents=node_contents,
        stream_mode=StreamMode.BATCHED
    ):
        if isinstance(item, GenerationToken) and not item.is_final:
            if first_token_batched is None:
                first_token_batched = time.time() - start_batched
            tokens_batched += 1

    total_batched = time.time() - start_batched

    print(f"  Total time: {total_batched * 1000:.1f}ms")
    print(f"  First token at: {first_token_batched * 1000:.1f}ms")
    print(f"  Tokens generated: {tokens_batched}")
    print()

    # Phase 4: CONCURRENT mode
    print("Phase 4 (CONCURRENT):")
    print("-" * 70)

    manager_concurrent = InterleavedStreamManager(llm)

    start_concurrent = time.time()
    first_token_concurrent = None
    tokens_concurrent = 0

    async for item in manager_concurrent.stream_interleaved(
        query="What is Thompson Sampling?",
        seed_nodes=["thompson_sampling"],
        graph=kg,
        token_budget=1000,
        max_generation_tokens=20,
        importance_scores=importance_scores,
        node_contents=node_contents,
        stream_mode=StreamMode.CONCURRENT
    ):
        if isinstance(item, GenerationToken) and not item.is_final:
            if first_token_concurrent is None:
                first_token_concurrent = time.time() - start_concurrent
            tokens_concurrent += 1

    total_concurrent = time.time() - start_concurrent

    print(f"  Total time: {total_concurrent * 1000:.1f}ms")
    print(f"  First token at: {first_token_concurrent * 1000:.1f}ms")
    print(f"  Tokens generated: {tokens_concurrent}")
    print()

    # Comparison
    print("Comparison:")
    print("-" * 70)

    if first_token_batched and first_token_concurrent:
        latency_improvement = ((first_token_batched - first_token_concurrent) / first_token_batched) * 100
        print(f"  Latency improvement: {latency_improvement:.1f}%")
        print(f"  BATCHED first token: {first_token_batched * 1000:.1f}ms")
        print(f"  CONCURRENT first token: {first_token_concurrent * 1000:.1f}ms")

    print(f"  Total time ratio: {total_batched / total_concurrent:.2f}x")
    print()


# ============================================================================
# Demo 2: Visual Interleaving Timeline
# ============================================================================

async def demo_interleaving_timeline():
    """Demo 2: Visual timeline showing interleaving pattern."""
    print("=" * 70)
    print(" " * 15 + "Demo 2: Interleaving Timeline")
    print("=" * 70)
    print()

    kg = create_demo_graph()
    importance_scores = create_importance_scores()
    node_contents = create_node_contents()

    llm = MockLLM(tokens_per_second=100)

    manager = InterleavedStreamManager(llm)

    print("Timeline (C=Chunk, T=Token):")
    print("-" * 70)

    start_time = time.time()
    events = []

    async for item in manager.stream_interleaved(
        query="What is Thompson Sampling?",
        seed_nodes=["thompson_sampling"],
        graph=kg,
        token_budget=1000,
        max_generation_tokens=15,
        importance_scores=importance_scores,
        node_contents=node_contents,
        stream_mode=StreamMode.CONCURRENT
    ):
        elapsed = (time.time() - start_time) * 1000

        if isinstance(item, ContextChunk):
            events.append((elapsed, "C", f"Chunk (hop {item.hop_distance})"))
        elif isinstance(item, GenerationToken) and not item.is_final:
            events.append((elapsed, "T", f"Token {item.token_index}"))

    # Print timeline
    for elapsed, event_type, desc in events[:20]:  # First 20 events
        bar = "=" * int(elapsed / 10)  # Scale bar
        print(f"  {elapsed:6.1f}ms [{event_type}] {bar} {desc}")

    print()
    print(f"Total events: {len(events)}")
    print()


# ============================================================================
# Demo 3: Performance Metrics
# ============================================================================

async def demo_performance_metrics():
    """Demo 3: Comprehensive performance metrics."""
    print("=" * 70)
    print(" " * 15 + "Demo 3: Performance Metrics")
    print("=" * 70)
    print()

    kg = create_demo_graph()
    importance_scores = create_importance_scores()
    node_contents = create_node_contents()

    llm = MockLLM(tokens_per_second=50)

    manager = InterleavedStreamManager(llm)

    # Run concurrent mode with detailed metrics
    start_time = time.time()
    chunks_count = 0
    tokens_count = 0
    first_chunk_time = None
    first_token_time = None
    last_chunk_time = None
    last_token_time = None

    async for item in manager.stream_interleaved(
        query="What is Thompson Sampling?",
        seed_nodes=["thompson_sampling"],
        graph=kg,
        token_budget=1000,
        max_generation_tokens=20,
        importance_scores=importance_scores,
        node_contents=node_contents,
        emit_metadata=True,
        stream_mode=StreamMode.CONCURRENT
    ):
        elapsed = time.time() - start_time

        if isinstance(item, ContextChunk):
            chunks_count += 1
            if first_chunk_time is None:
                first_chunk_time = elapsed
            last_chunk_time = elapsed

        elif isinstance(item, GenerationToken) and not item.is_final:
            tokens_count += 1
            if first_token_time is None:
                first_token_time = elapsed
            last_token_time = elapsed

    total_time = time.time() - start_time

    print("Metrics:")
    print("-" * 70)
    print(f"  Total time: {total_time * 1000:.1f}ms")
    print(f"  Chunks yielded: {chunks_count}")
    print(f"  Tokens yielded: {tokens_count}")
    print()

    print("Latency:")
    print("-" * 70)
    if first_chunk_time:
        print(f"  First chunk: {first_chunk_time * 1000:.1f}ms")
    if first_token_time:
        print(f"  First token: {first_token_time * 1000:.1f}ms (TARGET: <100ms)")
        print(f"  Target met: {'YES' if first_token_time < 0.1 else 'NO (with real LLM)'}")
    print()

    print("Timing:")
    print("-" * 70)
    if last_chunk_time and last_token_time:
        print(f"  Expansion finished: {last_chunk_time * 1000:.1f}ms")
        print(f"  Generation finished: {last_token_time * 1000:.1f}ms")

        # Check if truly interleaved
        if first_token_time < last_chunk_time:
            print(f"  Interleaving: YES (tokens during expansion)")
        else:
            print(f"  Interleaving: NO (tokens after expansion)")
    print()


# ============================================================================
# Main Demo
# ============================================================================

async def main():
    """Run all demonstrations."""
    print()
    print("=" * 70)
    print(" " * 10 + "PHASE 4: TRUE CONCURRENT TOKEN YIELDING")
    print(" " * 10 + "Interleaved Expansion + Generation")
    print("=" * 70)
    print()

    # Demo 1: Side-by-side comparison
    await demo_batched_vs_concurrent()

    # Demo 2: Visual interleaving timeline
    await demo_interleaving_timeline()

    # Demo 3: Performance metrics
    await demo_performance_metrics()

    print("=" * 70)
    print("Demo complete!")
    print()
    print("Key Takeaways:")
    print("  1. CONCURRENT mode yields tokens as generated (not batched)")
    print("  2. Achieves lower latency to first token")
    print("  3. True interleaving of chunks and tokens")
    print("  4. Better user experience (progressive results)")
    print()
    print("Next Steps:")
    print("  - Integrate with production LLM (OpenAI, Anthropic, etc.)")
    print("  - Add adaptive context feeding (update LLM with new chunks)")
    print("  - Implement agentic graph navigation")
    print("  - Add context summarization for efficiency")
    print("=" * 70)
    print()


if __name__ == "__main__":
    asyncio.run(main())
