"""Quick test of 10/10 API."""
import sys
sys.path.insert(0, '.')

import asyncio
from HoloLoom import HoloLoom


async def quick_test():
    """Quick functional test."""
    print("Testing 10/10 API...")
    print()

    # Initialize
    loom = HoloLoom()
    print("✓ Created HoloLoom")

    # Experience
    mem = await loom.experience("Thompson Sampling balances exploration")
    print(f"✓ Experienced: {mem.id[:8]}")

    # Recall
    memories = await loom.recall("Thompson Sampling")
    print(f"✓ Recalled: {len(memories)} memories")

    # Summary
    print()
    print(loom.summary())

    print("\n✅ 10/10 API is WORKING!")


if __name__ == "__main__":
    asyncio.run(quick_test())
