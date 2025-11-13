"""
Direct Statistical Mechanics Integration Test (No Semantic Cache)

Tests statistical mechanics integration directly without orchestrator overhead.
"""

import sys
import asyncio
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from HoloLoom.physics import (
    StatisticalMechanicsEngine,
    Microstate
)


async def test_direct_consolidation():
    """Test statistical mechanics consolidation directly."""
    print("=" * 70)
    print("Direct Statistical Mechanics Test")
    print("=" * 70)
    print()

    # Create test microstates
    microstates = [
        Microstate(
            id="ts1",
            state_vector=np.array([1.0, 0.8, 0.2]),
            energy=0.0,
            metadata={"topic": "thompson_sampling"}
        ),
        Microstate(
            id="ts2",
            state_vector=np.array([0.9, 0.9, 0.1]),
            energy=0.0,
            metadata={"topic": "thompson_sampling"}
        ),
        Microstate(
            id="gd1",
            state_vector=np.array([0.1, 0.2, 1.0]),
            energy=0.0,
            metadata={"topic": "gradient_descent"}
        ),
    ]

    print(f"Created {len(microstates)} test microstates")
    print()

    # Create engine
    engine = StatisticalMechanicsEngine(temperature=1.0)
    print("Statistical Mechanics Engine initialized")
    print(f"  Temperature: {engine.temperature:.2f}")
    print()

    # Consolidate
    print("Consolidating memories via Gibbs distribution...")
    macrostates = engine.consolidate_memories(microstates, num_clusters=2)

    print(f"Consolidated {len(microstates)} -> {len(macrostates)} clusters")
    print()

    for i, macro in enumerate(macrostates):
        print(f"[Cluster {i}] {macro.label}")
        print(f"  Size: {len(macro.microstates)} memories")
        print(f"  Average Energy: {macro.average_energy:.3f}")
        print(f"  Entropy: {macro.entropy:.3f}")
        print(f"  Free Energy: {macro.free_energy:.3f}")
        print(f"  Order Parameter: {macro.order_parameter:.3f}")
        print(f"  Members: {[s.id for s in macro.microstates]}")
        print()

    # Get statistics
    stats = engine.get_statistics()
    print("Engine Statistics:")
    print(f"  Temperature: {stats['temperature']:.2f}")
    print(f"  Entropy: {stats['entropy']:.3f}")
    print(f"  Free Energy: {stats['free_energy']:.3f}")
    print()

    print("=" * 70)
    print("[SUCCESS] Direct statistical mechanics test complete!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(test_direct_consolidation())
