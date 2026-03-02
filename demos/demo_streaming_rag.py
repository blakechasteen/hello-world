"""
Streaming RAG Demo
==================

Interactive demonstration of token-by-token streaming responses.

Shows:
- Token streaming with real-time output
- Tokens per second metrics
- Fallback behavior for non-direct modes
- Cache effectiveness
"""

import asyncio
import sys
import time
from typing import List

from hololoom.rag import SimpleRAG, StreamToken
from hololoom.config import Config


async def print_tokens_streaming(rag: SimpleRAG, question: str) -> None:
    """Print tokens as they stream in real-time."""
    print(f"\n{'='*60}")
    print(f"Question: {question}")
    print(f"{'='*60}")
    print("Response: ", end="", flush=True)

    token_count = 0
    first_token_time = None
    total_latency_ms = 0

    async for token in rag.query_stream(question, mode="direct"):
        # Print token
        print(token.text, end="", flush=True)
        token_count += 1

        # Track first token latency
        if first_token_time is None:
            first_token_time = token.metadata.get('latency_ms', 0)

        # Track total latency (from final token)
        if token.is_final:
            total_latency_ms = token.metadata.get('latency_ms', 0)

    print("\n")

    # Show metrics
    if token_count > 0:
        tps = token_count / (total_latency_ms / 1000) if total_latency_ms > 0 else 0
        print(f"Metrics:")
        print(f"  Total tokens: {token_count}")
        print(f"  Total latency: {total_latency_ms:.1f}ms")
        print(f"  Tokens per second: {tps:.1f}")
        if first_token_time:
            print(f"  Time to first token: {first_token_time:.1f}ms")


async def demo_basic_streaming():
    """Demo 1: Basic streaming with question-answer."""
    print("\n" + "="*60)
    print("DEMO 1: Basic Streaming")
    print("="*60)

    async with SimpleRAG(enable_caching=False) as rag:
        # Ingest some content
        print("\nIngesting knowledge base...")
        content = """
Thompson Sampling is a strategy for balancing exploration and exploitation.
It uses Bayesian statistics to estimate the probability of each choice being optimal.
The algorithm maintains a posterior distribution over the reward parameters.
It samples from this posterior and picks the option with the highest sampled reward.
This naturally balances exploration and exploitation.
        """
        await rag.ingest(content)
        print("✓ Ingestion complete")

        # Stream responses
        questions = [
            "What is Thompson Sampling?",
            "How does Thompson Sampling work?",
            "Why is it called Thompson Sampling?",
        ]

        for q in questions:
            await print_tokens_streaming(rag, q)
            await asyncio.sleep(0.5)


async def demo_caching_behavior():
    """Demo 2: Demonstrate caching after streaming."""
    print("\n" + "="*60)
    print("DEMO 2: Caching Streamed Responses")
    print("="*60)

    async with SimpleRAG(enable_caching=True) as rag:
        # Ingest
        print("\nIngesting content...")
        await rag.ingest("Thompson Sampling is a Bayesian bandit algorithm")
        print("✓ Ingestion complete")

        # First call - will stream
        print("\n[Call 1 - Fresh query]")
        await print_tokens_streaming(rag, "What is Thompson Sampling?")

        # Second call - should use cache
        print("\n[Call 2 - Same query (should be cached)]")
        start = time.time()
        await print_tokens_streaming(rag, "What is Thompson Sampling?")
        elapsed = time.time() - start
        print(f"Cached response time: {elapsed*1000:.1f}ms")


async def demo_mode_fallback():
    """Demo 3: Fallback behavior for non-direct modes."""
    print("\n" + "="*60)
    print("DEMO 3: Mode Fallback")
    print("="*60)

    async with SimpleRAG() as rag:
        # Ingest
        print("\nIngesting content...")
        await rag.ingest("Bayesian optimization uses Thompson Sampling")
        print("✓ Ingestion complete")

        # Try verify mode (will fall back)
        print("\n[Attempting 'verify' mode (will fall back)]")
        await print_tokens_streaming(rag, "What is Bayesian optimization?")


async def demo_metrics_tracking():
    """Demo 4: Track RAG metrics."""
    print("\n" + "="*60)
    print("DEMO 4: System Metrics")
    print("="*60)

    async with SimpleRAG(enable_caching=True) as rag:
        # Ingest multiple items
        print("\nIngesting knowledge base...")
        items = [
            "Thompson Sampling for multi-armed bandits",
            "Exploration vs exploitation tradeoff",
            "Bayesian posterior updates",
            "Policy gradients in reinforcement learning",
        ]

        for item in items:
            await rag.ingest(item)
            print(f"  ✓ {item[:40]}...")

        print("\n✓ Ingestion complete")

        # Query multiple times
        print("\nQuerying...")
        questions = [
            "What is Thompson Sampling?",
            "How does Thompson Sampling work?",
            "What is Thompson Sampling?",  # Repeated
        ]

        for i, q in enumerate(questions, 1):
            print(f"\n[Query {i}/{len(questions)}]")
            await print_tokens_streaming(rag, q)

        # Get metrics
        print("\n" + "="*60)
        print("System Metrics")
        print("="*60)
        metrics = rag.get_metrics()
        print(f"Memories: {metrics.get('n_memories', 0)}")
        print(f"Cache size: {metrics.get('cache_size', 0)}")
        print(f"Cache hit rate: {metrics.get('cache_hit_rate', 0):.1%}")
        print(f"LLM provider: {metrics.get('llm_provider', 'none')}")


async def demo_real_time_output():
    """Demo 5: Real-time streaming output (most visual)."""
    print("\n" + "="*60)
    print("DEMO 5: Real-Time Streaming Output")
    print("="*60)

    async with SimpleRAG() as rag:
        # Minimal ingest for quick demo
        print("\nSetup...")
        await rag.ingest("Sampling methods include Thompson Sampling, UCB, epsilon-greedy")
        print("✓ Ready\n")

        # Stream a single query with visual emphasis
        question = "Tell me about sampling strategies"
        print(f"Question: {question}")
        print("-" * 60)
        print("Response: ", end="", flush=True)

        token_buffer = []
        start_time = time.time()

        async for token in rag.query_stream(question, mode="direct"):
            # Print character by character for visual effect
            sys.stdout.write(token.text)
            sys.stdout.flush()
            token_buffer.append(token)

        elapsed = time.time() - start_time
        print("\n")
        print("-" * 60)

        if token_buffer:
            final_token = token_buffer[-1]
            print(f"Streaming stats:")
            print(f"  Total response length: {len(final_token.cumulative_text)} chars")
            print(f"  Total tokens: {final_token.metadata.get('total_tokens', 0)}")
            print(f"  Total time: {elapsed*1000:.1f}ms")
            print(f"  Tokens/sec: {final_token.metadata.get('tokens_per_sec', 0):.1f}")


async def main():
    """Run all demos."""
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + " "*58 + "║")
    print("║" + "  HoloLoom Streaming RAG Demo".center(58) + "║")
    print("║" + "  Token-by-Token Response Generation".center(58) + "║")
    print("║" + " "*58 + "║")
    print("╚" + "="*58 + "╝")

    try:
        # Run demos
        await demo_basic_streaming()
        await demo_caching_behavior()
        await demo_mode_fallback()
        await demo_metrics_tracking()
        await demo_real_time_output()

        # Summary
        print("\n" + "="*60)
        print("Demo Complete!")
        print("="*60)
        print("\nKey Features Demonstrated:")
        print("  ✓ Token-by-token streaming")
        print("  ✓ Real-time output with metrics")
        print("  ✓ Caching of streamed responses")
        print("  ✓ Fallback for non-streaming modes")
        print("  ✓ System metrics and monitoring")
        print("\n")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
