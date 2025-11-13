"""Test physics module in complete isolation."""

import asyncio
import numpy as np
from HoloLoom.physics import (
    StatisticalMechanicsEngine,
    Microstate
)


async def main():
    print("Testing physics module...")

    # Create microstates
    microstates = [
        Microstate(
            id="s1",
            state_vector=np.array([1.0, 0.8, 0.2]),
            energy=0.0,
            metadata={}
        ),
        Microstate(
            id="s2",
            state_vector=np.array([0.9, 0.9, 0.1]),
            energy=0.0,
            metadata={}
        ),
        Microstate(
            id="s3",
            state_vector=np.array([0.1, 0.2, 1.0]),
            energy=0.0,
            metadata={}
        ),
    ]

    print(f"Created {len(microstates)} microstates")

    # Create engine
    engine = StatisticalMechanicsEngine(temperature=1.0)
    print("Engine created")

    # Consolidate
    print("Consolidating...")
    macrostates = engine.consolidate_memories(microstates, num_clusters=2)
    print(f"Result: {len(macrostates)} clusters")

    print("SUCCESS!")


if __name__ == "__main__":
    asyncio.run(main())
