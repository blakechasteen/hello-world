"""
Demo: Advanced Interleaved Generation Features
================================================

Demonstrates the 3 advanced features for interleaved expansion + generation:
1. Adaptive Context Feeding - Update LLM with new chunks during generation
2. Agentic Graph Navigation - LLM can request specific nodes
3. Context Summarization - Compress less relevant chunks

Author: Claude Code (Opus 4.5)
Date: 2025-11-29
"""

import asyncio
import time
from typing import Dict, List, Any

import networkx as nx

from HoloLoom.memory.interleaved_generation import (
    ContextChunk,
    GenerationToken,
    StreamMetadata
)
from HoloLoom.memory.interleaved_generation_advanced import (
    AdvancedInterleavedManager,
    MockAdaptiveLLM,
    stream_advanced_generation
)


def create_knowledge_graph() -> nx.MultiDiGraph:
    """Create a sample knowledge graph about machine learning concepts."""
    G = nx.MultiDiGraph()

    # Core ML concepts
    G.add_edges_from([
        ("machine_learning", "supervised_learning", {"type": "INCLUDES"}),
        ("machine_learning", "unsupervised_learning", {"type": "INCLUDES"}),
        ("machine_learning", "reinforcement_learning", {"type": "INCLUDES"}),

        ("supervised_learning", "classification", {"type": "INCLUDES"}),
        ("supervised_learning", "regression", {"type": "INCLUDES"}),

        ("reinforcement_learning", "thompson_sampling", {"type": "USES"}),
        ("reinforcement_learning", "exploration_exploitation", {"type": "ADDRESSES"}),
        ("reinforcement_learning", "bandits", {"type": "INCLUDES"}),

        ("thompson_sampling", "bayesian_methods", {"type": "USES"}),
        ("thompson_sampling", "exploration_exploitation", {"type": "SOLVES"}),
        ("thompson_sampling", "probability", {"type": "USES"}),

        ("bandits", "thompson_sampling", {"type": "INCLUDES"}),
        ("bandits", "ucb", {"type": "INCLUDES"}),
        ("bandits", "epsilon_greedy", {"type": "INCLUDES"}),

        ("bayesian_methods", "probability", {"type": "USES"}),
        ("bayesian_methods", "prior", {"type": "HAS"}),
        ("bayesian_methods", "posterior", {"type": "PRODUCES"}),

        # Add bayesian_approach (requested by MockAdaptiveLLM)
        ("thompson_sampling", "bayesian_approach", {"type": "IMPLEMENTS"}),
    ])

    return G


def create_node_contents() -> Dict[str, str]:
    """Create content for each node in the knowledge graph."""
    return {
        "machine_learning": "Machine learning is a subset of AI focused on systems that learn from data.",
        "supervised_learning": "Supervised learning trains models on labeled data to predict outcomes.",
        "unsupervised_learning": "Unsupervised learning finds patterns in unlabeled data.",
        "reinforcement_learning": "Reinforcement learning trains agents through rewards and penalties.",
        "classification": "Classification predicts discrete categories or classes.",
        "regression": "Regression predicts continuous numerical values.",
        "thompson_sampling": "Thompson Sampling is a Bayesian approach to the exploration-exploitation tradeoff.",
        "exploration_exploitation": "The exploration-exploitation tradeoff balances trying new options vs using known good ones.",
        "bandits": "Multi-armed bandits are a framework for sequential decision making under uncertainty.",
        "ucb": "Upper Confidence Bound (UCB) uses optimism to guide exploration.",
        "epsilon_greedy": "Epsilon-greedy explores randomly with probability epsilon, exploits otherwise.",
        "bayesian_methods": "Bayesian methods use probability theory to update beliefs based on evidence.",
        "probability": "Probability theory quantifies uncertainty and likelihood.",
        "prior": "Prior distributions represent initial beliefs before observing data.",
        "posterior": "Posterior distributions represent updated beliefs after observing data.",
        "bayesian_approach": "Bayesian approaches use probability theory to update beliefs based on evidence.",
    }


def create_importance_scores() -> Dict[str, float]:
    """Create importance scores for each node."""
    return {
        "machine_learning": 0.9,
        "supervised_learning": 0.7,
        "unsupervised_learning": 0.7,
        "reinforcement_learning": 0.95,
        "classification": 0.5,
        "regression": 0.5,
        "thompson_sampling": 1.0,  # Most relevant
        "exploration_exploitation": 0.9,
        "bandits": 0.8,
        "ucb": 0.6,
        "epsilon_greedy": 0.6,
        "bayesian_methods": 0.85,
        "probability": 0.7,
        "prior": 0.65,
        "posterior": 0.65,
        "bayesian_approach": 0.85,
    }


async def demo_adaptive_context_feeding():
    """Demo 1: Adaptive Context Feeding - Update LLM with new chunks during generation."""
    print("\n" + "=" * 70)
    print("DEMO 1: Adaptive Context Feeding")
    print("=" * 70)
    print("""
This demo shows how the LLM receives context updates during generation.
As new chunks are discovered, they're fed to the LLM to improve the response.
""")

    graph = create_knowledge_graph()
    node_contents = create_node_contents()
    importance_scores = create_importance_scores()

    llm = MockAdaptiveLLM(tokens_per_second=50)
    manager = AdvancedInterleavedManager(
        llm=llm,
        enable_adaptive_feeding=True,
        enable_agentic_navigation=False,
        enable_summarization=False
    )

    query = "What is Thompson Sampling?"
    print(f"Query: {query}")
    print("-" * 50)

    chunks_received = 0
    tokens_received = 0
    context_updates_seen = 0

    start_time = time.time()

    async for item in manager.stream_advanced(
        query=query,
        seed_nodes=["thompson_sampling"],
        graph=graph,
        token_budget=1500,
        max_generation_tokens=50,
        importance_scores=importance_scores,
        node_contents=node_contents,
        emit_metadata=True
    ):
        if isinstance(item, ContextChunk):
            chunks_received += 1
            print(f"[CHUNK {chunks_received}] {len(item.nodes)} nodes at hop {item.hop_distance}")

        elif isinstance(item, GenerationToken) and not item.is_final:
            tokens_received += 1
            # Check for context update markers
            if "[New context:" in item.cumulative_text:
                context_updates_seen += 1
            print(item.token, end="", flush=True)

        elif isinstance(item, StreamMetadata):
            if item.event_type == "generation_start":
                print("\n[GENERATION STARTED]")
            elif item.event_type == "stream_complete":
                print(f"\n[COMPLETE] {item.data.get('total_time_ms', 0):.0f}ms")

    elapsed = (time.time() - start_time) * 1000

    print(f"\n\nResults:")
    print(f"  Chunks received: {chunks_received}")
    print(f"  Tokens generated: {tokens_received}")
    print(f"  Context updates during generation: {context_updates_seen}")
    print(f"  Total time: {elapsed:.0f}ms")


async def demo_agentic_navigation():
    """Demo 2: Agentic Graph Navigation - LLM can request specific nodes."""
    print("\n" + "=" * 70)
    print("DEMO 2: Agentic Graph Navigation")
    print("=" * 70)
    print("""
This demo shows how the LLM can request specific nodes during generation.
When the LLM emits <request_node>name</request_node>, that node is fetched.
""")

    graph = create_knowledge_graph()
    node_contents = create_node_contents()
    importance_scores = create_importance_scores()

    # Use a response template that will trigger node requests
    llm = MockAdaptiveLLM(tokens_per_second=50)
    manager = AdvancedInterleavedManager(
        llm=llm,
        enable_adaptive_feeding=True,  # Required for agentic navigation
        enable_agentic_navigation=True,
        enable_summarization=False
    )

    query = "Explain the relationship between Thompson Sampling and Bayesian methods"
    print(f"Query: {query}")
    print("-" * 50)

    node_requests = []

    start_time = time.time()

    async for item in manager.stream_advanced(
        query=query,
        seed_nodes=["thompson_sampling"],
        graph=graph,
        token_budget=2000,
        max_generation_tokens=80,
        importance_scores=importance_scores,
        node_contents=node_contents,
        emit_metadata=True
    ):
        if isinstance(item, ContextChunk):
            print(f"[CHUNK] {len(item.nodes)} nodes: {item.nodes[:3]}...")

        elif isinstance(item, GenerationToken) and not item.is_final:
            # Highlight node request tokens
            if "<request_node>" in item.token or "</request_node>" in item.token:
                print(f"\n  >>> NODE REQUEST: {item.token}", end="")
            else:
                print(item.token, end="", flush=True)

        elif isinstance(item, StreamMetadata):
            if item.event_type == "node_request_fulfilled":
                node_requests.append(item.data)
                print(f"\n  <<< FULFILLED: {item.data.get('target')} ({item.data.get('nodes_fetched')} nodes)")
            elif item.event_type == "stream_complete":
                print(f"\n[COMPLETE]")

    elapsed = (time.time() - start_time) * 1000

    print(f"\nResults:")
    print(f"  Node requests fulfilled: {len(node_requests)}")
    for req in node_requests:
        print(f"    - {req.get('target')}: {req.get('nodes_fetched')} nodes fetched")
    print(f"  Total time: {elapsed:.0f}ms")


async def demo_context_summarization():
    """Demo 3: Context Summarization - Compress less relevant chunks."""
    print("\n" + "=" * 70)
    print("DEMO 3: Context Summarization")
    print("=" * 70)
    print("""
This demo shows how large contexts are automatically summarized.
When chunks exceed a threshold, less relevant ones are compressed.
""")

    graph = create_knowledge_graph()
    node_contents = create_node_contents()
    importance_scores = create_importance_scores()

    llm = MockAdaptiveLLM(tokens_per_second=50)
    manager = AdvancedInterleavedManager(
        llm=llm,
        enable_adaptive_feeding=False,
        enable_agentic_navigation=False,
        enable_summarization=True
    )

    # Use a broader query that will retrieve more chunks
    query = "How does machine learning relate to Thompson Sampling?"
    print(f"Query: {query}")
    print("-" * 50)

    summarization_events = []

    start_time = time.time()

    async for item in manager.stream_advanced(
        query=query,
        seed_nodes=["machine_learning"],  # Start from broad concept
        graph=graph,
        token_budget=3000,
        max_generation_tokens=40,
        chunk_size=200,  # Smaller chunks to trigger summarization
        importance_scores=importance_scores,
        node_contents=node_contents,
        emit_metadata=True
    ):
        if isinstance(item, ContextChunk):
            print(f"[CHUNK] {len(item.nodes)} nodes, {item.token_count} tokens")

        elif isinstance(item, GenerationToken) and not item.is_final:
            print(item.token, end="", flush=True)

        elif isinstance(item, StreamMetadata):
            if item.event_type == "summarization_applied":
                summarization_events.append(item.data)
                print(f"\n[SUMMARIZATION] {item.data}")
            elif item.event_type == "stream_complete":
                print(f"\n[COMPLETE]")

    elapsed = (time.time() - start_time) * 1000

    print(f"\nResults:")
    print(f"  Summarization events: {len(summarization_events)}")
    for event in summarization_events:
        orig = event.get('original_chunks', 0)
        kept = event.get('kept_chunks', 0)
        ratio = event.get('compression_ratio', 0)
        print(f"    - {orig} chunks -> {kept} kept + summary ({ratio:.1%} compression)")
    print(f"  Total time: {elapsed:.0f}ms")


async def demo_all_features():
    """Demo 4: All Features Together - Complete advanced generation."""
    print("\n" + "=" * 70)
    print("DEMO 4: All Features Together")
    print("=" * 70)
    print("""
This demo shows all 3 features working together for maximum capability.
""")

    graph = create_knowledge_graph()
    node_contents = create_node_contents()
    importance_scores = create_importance_scores()

    query = "Explain Thompson Sampling comprehensively"
    print(f"Query: {query}")
    print("-" * 50)

    # Use convenience function
    chunks = 0
    tokens = 0
    events = []

    start_time = time.time()

    async for item in stream_advanced_generation(
        query=query,
        seed_nodes=["thompson_sampling"],
        graph=graph,
        node_contents=node_contents,
        importance_scores=importance_scores,
        token_budget=2000,
        max_generation_tokens=60,
        enable_adaptive_feeding=True,
        enable_agentic_navigation=True,
        enable_summarization=True
    ):
        if isinstance(item, ContextChunk):
            chunks += 1
        elif isinstance(item, GenerationToken) and not item.is_final:
            tokens += 1
            print(item.token, end="", flush=True)
        elif isinstance(item, StreamMetadata):
            events.append(item.event_type)

    elapsed = (time.time() - start_time) * 1000

    print(f"\n\nResults (All Features Enabled):")
    print(f"  Chunks: {chunks}")
    print(f"  Tokens: {tokens}")
    print(f"  Events: {events}")
    print(f"  Total time: {elapsed:.0f}ms")


async def main():
    """Run all demos."""
    print("""
+======================================================================+
|     Advanced Interleaved Generation - Feature Demonstration          |
+======================================================================+
|  This demo showcases 3 advanced features for interleaved generation: |
|  1. Adaptive Context Feeding - Live context updates to LLM           |
|  2. Agentic Graph Navigation - LLM-driven node requests              |
|  3. Context Summarization - Smart context compression                |
+======================================================================+
""")

    await demo_adaptive_context_feeding()
    await demo_agentic_navigation()
    await demo_context_summarization()
    await demo_all_features()

    print("\n" + "=" * 70)
    print("All demos complete!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
