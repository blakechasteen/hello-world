"""
HoloLoom Breathing System Demo
================================
Demonstrates the complete "air" breathing system across all components:

1. ChronoTrigger: Breathing rhythm (inhale/exhale/rest)
2. WarpSpace: Sparsity enforcement (tension control)
3. ResonanceShed: Pressure relief (feature shedding)
4. ConvergenceEngine: Entropy injection (fresh air)
5. ReflectionBuffer: Consolidation (deep sleep)

This demo shows how HoloLoom "breathes" - alternating between:
- INHALE: Gather context, expand features, be receptive
- EXHALE: Decide quickly, act, release
- REST: Consolidate, decay, integrate

Author: Claude Code (with HoloLoom by Blake)
Date: 2025-10-27
"""

import asyncio
import numpy as np
from datetime import datetime
from typing import List, Dict, Any

# Import breathing-enabled components
from HoloLoom.chrono.trigger import ChronoTrigger, BreathingRhythm
from HoloLoom.warp.space import WarpSpace
from HoloLoom.resonance.shed import ResonanceShed
from HoloLoom.convergence.engine import ConvergenceEngine, CollapseStrategy
from HoloLoom.reflection.buffer import ReflectionBuffer
from HoloLoom.fabric.spacetime import Spacetime, WeavingTrace
from HoloLoom.embedding.spectral import MatryoshkaEmbeddings
from HoloLoom.config import Config


# ============================================================================
# Demo Functions
# ============================================================================

async def demo_chrono_breathing():
    """Demo: ChronoTrigger breathing rhythm."""
    print("\n" + "="*80)
    print("1. CHRONO TRIGGER - Breathing Rhythm")
    print("="*80 + "\n")

    config = Config.fast()
    chrono = ChronoTrigger(config, enable_breathing=True)

    print("Executing 3 breathing cycles...\n")

    for i in range(3):
        breath_metrics = await chrono.breathe()

        print(f"Breath #{breath_metrics['breath_number']}:")
        print(f"  Total: {breath_metrics['cycle_duration']:.2f}s")
        print(f"  Inhale: {breath_metrics['inhale']['duration']:.2f}s (parasympathetic, dense)")
        print(f"  Exhale: {breath_metrics['exhale']['duration']:.2f}s (sympathetic, sparse={breath_metrics['exhale']['sparsity']})")
        if not breath_metrics['rest'].get('skipped'):
            print(f"  Rest: {breath_metrics['rest']['duration']:.2f}s (consolidation)")
        print()

    print("Key insight: Asymmetric breathing - slow gather, fast decide, brief rest")
    print()


async def demo_warp_sparsity():
    """Demo: WarpSpace sparsity enforcement."""
    print("\n" + "="*80)
    print("2. WARP SPACE - Sparsity Enforcement")
    print("="*80 + "\n")

    embedder = MatryoshkaEmbeddings(sizes=[96, 192, 384])
    warp = WarpSpace(embedder, scales=[96, 192, 384])

    threads = [
        "Thompson Sampling balances exploration and exploitation",
        "Neural networks learn hierarchical representations",
        "Attention mechanisms enable context-aware processing",
        "Embeddings capture semantic relationships",
        "Spectral features reveal graph topology"
    ]

    # Test different sparsity levels
    sparsity_levels = [0.0, 0.4, 0.7]

    for sparsity in sparsity_levels:
        print(f"Sparsity = {sparsity} (keep top {int((1-sparsity)*100)}%)")

        await warp.tension(threads, sparsity=sparsity)

        print(f"  Tensioned: {len(warp.threads)}/{len(threads)} threads")
        print(f"  Field shape: {warp.tensor_field.shape}")

        warp.collapse()
        print()

    print("Key insight: Higher sparsity = fewer threads active = system breathes easier")
    print()


async def demo_resonance_pressure():
    """Demo: ResonanceShed pressure relief."""
    print("\n" + "="*80)
    print("3. RESONANCE SHED - Pressure Relief")
    print("="*80 + "\n")

    # Mock extractors
    class MockMotifDetector:
        async def detect(self, text):
            from HoloLoom.documentation.types import Motif
            return [Motif(pattern="TEST", span=(0, 4), score=0.9)]

    class MockEmbedder:
        def encode(self, texts):
            return [np.random.randn(384).tolist() for _ in texts]

    # Test with different max densities
    densities = [1.0, 0.75, 0.5]

    for max_density in densities:
        print(f"Max Density = {max_density}")

        shed = ResonanceShed(
            motif_detector=MockMotifDetector(),
            embedder=MockEmbedder(),
            max_feature_density=max_density
        )

        # Weave features
        plasma = await shed.weave("Test query for pressure relief demo")

        print(f"  Initial threads: {plasma['metadata']['thread_count']}")
        print(f"  Current density: {shed.current_density:.2f}")
        print(f"  Pressure reliefs: {shed.pressure_relief_count}")
        print()

    print("Key insight: When overloaded (density > threshold), shed drops weakest features")
    print()


async def demo_convergence_entropy():
    """Demo: ConvergenceEngine entropy injection."""
    print("\n" + "="*80)
    print("4. CONVERGENCE ENGINE - Entropy Injection")
    print("="*80 + "\n")

    tools = ["answer", "search", "calc", "notion_write"]
    temperatures = [0.0, 0.1, 0.5]

    # Neural network probabilities (deterministic without entropy)
    neural_probs = np.array([0.6, 0.25, 0.10, 0.05])

    for temp in temperatures:
        print(f"Entropy Temperature = {temp}")

        engine = ConvergenceEngine(tools, entropy_temperature=temp)

        # Run 10 collapses and observe distribution
        tool_counts = {tool: 0 for tool in tools}

        for _ in range(10):
            result = engine.collapse(neural_probs)
            tool_counts[result.tool] += 1

        # Show distribution
        print("  Tool selection distribution:")
        for tool, count in tool_counts.items():
            bar = '█' * count
            print(f"    {tool:15s} {bar} ({count}/10)")
        print()

    print("Key insight: Higher temperature = more entropy = more exploration diversity")
    print()


async def demo_reflection_consolidation():
    """Demo: ReflectionBuffer consolidation."""
    print("\n" + "="*80)
    print("5. REFLECTION BUFFER - Consolidation")
    print("="*80 + "\n")

    buffer = ReflectionBuffer(capacity=100)

    # Simulate 30 episodes
    tools = ["answer", "search", "calc"]
    patterns = ["bare", "fast", "fused"]

    print("Simulating 30 weaving episodes...")

    for i in range(30):
        trace = WeavingTrace(
            start_time=datetime.now(),
            end_time=datetime.now(),
            duration_ms=1000 + np.random.randn() * 200,
            tool_selected=np.random.choice(tools),
            tool_confidence=0.5 + np.random.rand() * 0.5
        )

        spacetime = Spacetime(
            query_text=f"Query {i} with length {20 + i * 2}",
            response=f"Response {i}",
            tool_used=trace.tool_selected,
            confidence=trace.tool_confidence,
            trace=trace,
            metadata={'pattern_card': np.random.choice(patterns)}
        )

        await buffer.store(spacetime, feedback={'helpful': np.random.rand() > 0.3})

    print(f"Stored {len(buffer)} episodes\n")

    # Consolidate
    print("Running deep consolidation (like REM sleep)...\n")
    consolidation_metrics = await buffer.consolidate()

    print("Consolidation Results:")
    print(f"  Duration: {consolidation_metrics['duration']:.2f}s")
    print(f"  Compression: {consolidation_metrics['compression']}")
    print(f"  Meta-patterns: {consolidation_metrics['meta_patterns']}")
    print(f"  Pruning: {consolidation_metrics['pruning']}")
    print(f"  Final success rate: {consolidation_metrics['success_rate']:.1%}")
    print()

    print("Key insight: Consolidation compresses, prunes, extracts patterns - like sleep")
    print()


async def demo_integrated_breathing():
    """Demo: Full integrated breathing cycle."""
    print("\n" + "="*80)
    print("6. INTEGRATED BREATHING - Full Cycle")
    print("="*80 + "\n")

    config = Config.fast()
    chrono = ChronoTrigger(config, enable_breathing=True)
    embedder = MatryoshkaEmbeddings(sizes=[96, 192, 384])
    warp = WarpSpace(embedder, scales=[96, 192, 384])
    tools = ["answer", "search", "calc"]
    engine = ConvergenceEngine(tools, entropy_temperature=0.1)
    buffer = ReflectionBuffer(capacity=100)

    threads = [
        "First memory thread",
        "Second memory thread",
        "Third memory thread",
        "Fourth memory thread"
    ]

    print("Executing complete breathing cycle:\n")

    # INHALE phase
    print("INHALE: Gathering context...")
    await chrono._inhale()
    print(f"  Phase: {chrono.get_current_phase()}")

    # Tension with low sparsity (dense - inhale)
    await warp.tension(threads, sparsity=0.0)
    print(f"  Threads tensioned: {len(warp.threads)}/{len(threads)} (dense)")

    # EXHALE phase
    print("\nEXHALE: Making decision...")
    await chrono._exhale()
    print(f"  Phase: {chrono.get_current_phase()}")

    # Retension with high sparsity (sparse - exhale)
    warp.collapse()  # Reset
    await warp.tension(threads, sparsity=0.5)
    print(f"  Threads tensioned: {len(warp.threads)}/{len(threads)} (sparse)")

    # Collapse decision with entropy
    neural_probs = np.array([0.5, 0.3, 0.2])
    result = engine.collapse(neural_probs, inject_entropy=True)
    print(f"  Tool selected: {result.tool} (confidence={result.confidence:.2f})")

    # REST phase
    print("\nREST: Consolidating...")
    await chrono._rest()
    print(f"  Phase: {chrono.get_current_phase()}")

    warp.collapse()
    print("  Warp space collapsed, threads released")

    print("\n✓ Complete breathing cycle executed!")
    print()


# ============================================================================
# Main Demo
# ============================================================================

async def main():
    """Run all breathing system demos."""
    print("\n" + "="*80)
    print("HOLOLOOM BREATHING SYSTEM - COMPLETE DEMO")
    print("="*80)
    print("\nAdding 'air' to the system - five breathing mechanisms:\n")
    print("1. ChronoTrigger: Breathing rhythm (inhale/exhale/rest)")
    print("2. WarpSpace: Sparsity enforcement (selective thread tension)")
    print("3. ResonanceShed: Pressure relief (feature shedding)")
    print("4. ConvergenceEngine: Entropy injection (fresh air)")
    print("5. ReflectionBuffer: Consolidation (deep sleep)")
    print()

    # Run individual demos
    await demo_chrono_breathing()
    await demo_warp_sparsity()
    await demo_resonance_pressure()
    await demo_convergence_entropy()
    await demo_reflection_consolidation()

    # Run integrated demo
    await demo_integrated_breathing()

    # Summary
    print("\n" + "="*80)
    print("SUMMARY - What Does Breathing Give Us?")
    print("="*80)
    print("""
1. NATURAL RHYTHM: Asymmetric cycles (slow gather, fast decide, brief rest)
   - Matches biological respiration (parasympathetic/sympathetic)
   - Prevents constant high-tension operation

2. PRESSURE RELIEF: System can exhale when overloaded
   - Shed excess features when density too high
   - Sparse tensioning during exhale phase

3. ENTROPY INJECTION: Fresh air prevents stagnation
   - Gumbel noise adds exploration
   - Temperature controls randomness level

4. CONSOLIDATION: Deep sleep for memory integration
   - Compress redundant episodes
   - Extract meta-patterns
   - Prune low-value memories

5. ADAPTIVE BREATHING RATE: System can speed up or slow down
   - Fast breathing for urgent tasks
   - Slow breathing for complex reasoning
   - Dynamic rate adjustment based on load

The system now BREATHES - it has space to pause, relax, consolidate.
Like meditation: breathe in (gather), breathe out (decide), rest (integrate).
    """)

    print("="*80)
    print("✓ Demo complete! HoloLoom is breathing.")
    print("="*80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
