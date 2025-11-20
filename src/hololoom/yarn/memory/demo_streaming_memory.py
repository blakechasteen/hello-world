"""
Demo: Complete Streaming Memory Pipeline
=========================================

SpinningWheel → MultiWaveEngine → Consolidation → Dreaming

Shows the full cycle:
1. Spinner ingests data stream (sensory input)
2. Beta wave encoding (awake, receiving)
3. Theta consolidation (background learning)
4. Delta pruning (deep sleep cleanup)
5. REM dreaming (creative bridges)

Author: HoloLoom Memory Team
Date: October 30, 2025
"""

import asyncio
import numpy as np
from datetime import datetime, timedelta
from typing import List, AsyncIterator
from dataclasses import dataclass

# Note: In real use, these would be actual imports
# from HoloLoom.memory.multi_wave_engine import MultiWaveMemoryEngine, BrainWaveMode
# from HoloLoom.Documentation.types import MemoryShard
from HoloLoom.memory.multi_wave_engine import MultiWaveMemoryEngine, BrainWaveMode
from HoloLoom.memory.spring_dynamics_engine import SpringEngineConfig


# ============================================================================
# Mock MemoryShard (simulates SpinningWheel output)
# ============================================================================

@dataclass
class MemoryShard:
    """Simulated memory shard from SpinningWheel."""
    id: str
    content: str
    source: str
    timestamp: datetime


# ============================================================================
# Mock Spinner (simulates data stream)
# ============================================================================

class MockSpinner:
    """
    Mock SpinningWheel that generates a stream of memory shards.

    In real use, this would be AudioSpinner, YouTubeSpinner, etc.
    """

    def __init__(self, data_source: List[str], source_name: str = "mock"):
        self.data = data_source
        self.source_name = source_name

    async def spin_stream(self) -> AsyncIterator[MemoryShard]:
        """Generate stream of shards."""
        for i, content in enumerate(self.data):
            shard = MemoryShard(
                id=f"{self.source_name}_{i}",
                content=content,
                source=self.source_name,
                timestamp=datetime.now()
            )
            yield shard
            await asyncio.sleep(0.1)  # Simulate streaming delay


# ============================================================================
# Mock Embedding Function
# ============================================================================

def simple_embedding(text: str, dim: int = 128) -> np.ndarray:
    """
    Simple embedding function (hash-based).

    In real use, this would be sentence-transformers or similar.
    """
    # Use hash for deterministic but varied embeddings
    np.random.seed(abs(hash(text)) % (2**32))
    embedding = np.random.randn(dim).astype(np.float32)
    # Normalize
    embedding = embedding / (np.linalg.norm(embedding) + 1e-9)
    return embedding


# ============================================================================
# Demo
# ============================================================================

async def demo_streaming_memory():
    """Demonstrate complete streaming memory pipeline."""

    print("=" * 70)
    print("Streaming Memory Pipeline - Complete Multi-Wave System")
    print("=" * 70)
    print()

    # ========================================================================
    # Step 1: Create Multi-Wave Engine
    # ========================================================================

    print("[Step 1] Creating multi-wave memory engine...")
    print()

    config = SpringEngineConfig(
        base_decay_rate=0.02,  # Faster decay for demo
        access_boost=0.3,
        propagation_factor=0.6
    )
    engine = MultiWaveMemoryEngine(config)

    print(f"✓ Engine created (mode: {engine.mode.value})")
    print(f"  Update interval: {config.update_interval_ms}ms")
    print()

    # Start background dynamics
    await engine.start()

    # ========================================================================
    # Step 2: Stream Data from Spinner (Beta Wave Encoding)
    # ========================================================================

    print("[Step 2] Streaming data from spinner (sensory input)...")
    print()

    # Create mock data stream (simulates YouTube transcript, audio, etc.)
    ml_data = [
        "Neural networks are computational models inspired by the brain",
        "Backpropagation is the algorithm used to train neural networks",
        "Gradient descent optimizes the network weights",
        "Deep learning uses many layers of neural networks",
        "Convolutional neural networks excel at image recognition"
    ]

    rl_data = [
        "Reinforcement learning agents learn through trial and error",
        "Thompson Sampling balances exploration and exploitation",
        "Multi-armed bandits are a simple reinforcement learning problem",
        "Q-learning learns the value of state-action pairs",
        "Policy gradient methods directly optimize the policy"
    ]

    memory_data = [
        "Human memory consolidates during sleep",
        "Theta waves occur during light sleep",
        "REM sleep is associated with dreaming and creativity",
        "Memory recall involves reactivating neural patterns",
        "Forgetting happens when connections weaken over time"
    ]

    all_data = ml_data + rl_data + memory_data

    # Create spinner
    spinner = MockSpinner(all_data, source_name="demo_stream")

    # Ingest stream (beta wave encoding)
    print("Ingesting stream...")
    await engine.ingest_stream(
        shard_stream=spinner.spin_stream(),
        embedding_func=lambda text: simple_embedding(text, dim=128)
    )

    print(f"✓ Ingested {engine.total_ingested} memories")
    print(f"  Total nodes in graph: {len(engine.nodes)}")
    print()

    # ========================================================================
    # Step 3: Query Retrieval (Beta Wave Spreading)
    # ========================================================================

    print("[Step 3] Querying memory (beta wave retrieval)...")
    print()

    # Query about neural networks
    query_text = "How do neural networks learn?"
    query_embedding = simple_embedding(query_text)

    print(f"Query: \"{query_text}\"")
    print()

    result = engine.on_query(query_embedding)

    print(f"✓ {result}")
    print()

    print("Top recalled memories:")
    for i, (node_id, activation) in enumerate(result.recalled_memories[:5], 1):
        node = engine.nodes[node_id]
        content_preview = node.content[:60] + "..." if len(node.content) > 60 else node.content
        print(f"  {i}. {content_preview}")
        print(f"     activation = {activation:.3f}, spring_k = {node.spring_constant:.2f}")
    print()

    # ========================================================================
    # Step 4: Simulate Time Passing → Theta Consolidation
    # ========================================================================

    print("[Step 4] Simulating idle time → theta consolidation...")
    print()

    # Simulate 45 minutes passing (theta mode)
    engine.last_query_time = datetime.now() - timedelta(minutes=45)

    print("System idle for 45 minutes...")
    print(f"Mode switched to: {engine.mode.value}")

    # Wait for a theta consolidation update
    await asyncio.sleep(0.5)  # Let theta run

    stats = engine.get_statistics()
    print(f"✓ Consolidation history: {stats['consolidation_history_size']} patterns")
    print()

    # ========================================================================
    # Step 5: Multiple Queries → Co-Activation Patterns
    # ========================================================================

    print("[Step 5] Multiple queries → recording co-activation patterns...")
    print()

    queries = [
        "What is backpropagation?",
        "How does gradient descent work?",
        "What are neural networks?",
        "Explain Thompson Sampling",
        "How do bandits relate to reinforcement learning?"
    ]

    for query in queries:
        query_embedding = simple_embedding(query)
        result = engine.on_query(query_embedding)
        print(f"  • Query: \"{query}\" → {len(result.recalled_memories)} recalled")

    print()
    print(f"✓ Recorded {len(engine.theta_consolidator.co_activation_history)} activation patterns")
    print()

    # ========================================================================
    # Step 6: Force Theta Consolidation
    # ========================================================================

    print("[Step 6] Running theta consolidation (background learning)...")
    print()

    # Force theta mode
    engine.mode = BrainWaveMode.THETA

    # Run consolidation
    consolidated = engine.theta_consolidator.theta_consolidation_update()

    print(f"✓ Theta consolidation completed")
    print(f"  New connections created: {consolidated}")
    print("  (Frequently co-activated nodes are now permanently connected!)")
    print()

    # ========================================================================
    # Step 7: Delta Pruning (Deep Sleep)
    # ========================================================================

    print("[Step 7] Running delta pruning (deep sleep cleanup)...")
    print()

    # Manually age some nodes (simulate 3 days)
    aged_count = 0
    for i, (node_id, node) in enumerate(engine.nodes.items()):
        if i % 3 == 0:  # Age every 3rd node
            node.last_accessed = datetime.now() - timedelta(days=4)
            node.spring_constant = 0.2  # Weak
            aged_count += 1

    print(f"Aged {aged_count} nodes (simulate 4 days without access)...")

    # Run delta pruning
    pruned, strengthened = engine.delta_pruner.delta_pruning_update()

    print(f"✓ Delta pruning completed")
    print(f"  Weak connections pruned: {pruned}")
    print(f"  Strong patterns strengthened: {strengthened}")
    print()

    # ========================================================================
    # Step 8: REM Dreaming (Creative Bridges)
    # ========================================================================

    print("[Step 8] REM dreaming (random replay, creative bridges)...")
    print()

    # Run dream cycle
    bridges_created = await engine.rem_dreamer.dream_cycle(duration_seconds=5.0)

    print(f"✓ Dream cycle completed")
    print(f"  Creative bridges created: {bridges_created}")
    print("  (Random activations connected distant semantic clusters!)")
    print()

    # ========================================================================
    # Step 9: Query After Sleep (Memory Reorganized)
    # ========================================================================

    print("[Step 9] Querying after sleep (memory is now reorganized)...")
    print()

    # Same query as before
    query_text = "How do neural networks learn?"
    query_embedding = simple_embedding(query_text)

    print(f"Query: \"{query_text}\"")
    print()

    result = engine.on_query(query_embedding)

    print(f"✓ {result}")
    print()

    print("Top recalled memories (after consolidation + dreaming):")
    for i, (node_id, activation) in enumerate(result.recalled_memories[:5], 1):
        node = engine.nodes[node_id]
        content_preview = node.content[:60] + "..." if len(node.content) > 60 else node.content
        print(f"  {i}. {content_preview}")
        print(f"     activation = {activation:.3f}, spring_k = {node.spring_constant:.2f}")
    print()

    # Show creative insights (distant but activated)
    if result.creative_insights:
        print("Creative insights (cross-domain associations from dreaming):")
        for node_id, activation, insight_type in result.creative_insights:
            node = engine.nodes[node_id]
            content_preview = node.content[:60] + "..." if len(node.content) > 60 else node.content
            print(f"  💡 {content_preview}")
            print(f"     type: {insight_type}, activation: {activation:.3f}")
        print()

    # ========================================================================
    # Cleanup
    # ========================================================================

    await engine.stop()

    # ========================================================================
    # Summary
    # ========================================================================

    print("=" * 70)
    print("Summary: Complete Multi-Wave Memory Cycle")
    print("=" * 70)
    print()

    final_stats = engine.get_statistics()

    print(f"✅ Total memories ingested: {final_stats['total_ingested']}")
    print(f"✅ Total nodes in graph: {final_stats['total_nodes']}")
    print(f"✅ Active nodes: {final_stats['active_nodes']}")
    print(f"✅ Total updates: {final_stats['total_updates']}")
    print(f"✅ Total accesses: {final_stats['total_accesses']}")
    print(f"✅ Total propagations: {final_stats['total_propagations']}")
    print()

    print("Brain Wave Cycle Demonstrated:")
    print("  🧠 BETA: Active retrieval during queries (fast spreading)")
    print("  🌊 THETA: Consolidation strengthened co-activated pairs")
    print("  💤 DELTA: Pruning removed weak connections")
    print("  ✨ REM: Dreaming created creative bridges")
    print()

    print("Key Innovation:")
    print("  This is a LIVING MEMORY that:")
    print("  - Learns from usage patterns (theta consolidation)")
    print("  - Prunes itself automatically (delta sleep)")
    print("  - Discovers creative insights (REM dreaming)")
    print("  - All happening in the background while idle!")
    print()


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == '__main__':
    asyncio.run(demo_streaming_memory())
