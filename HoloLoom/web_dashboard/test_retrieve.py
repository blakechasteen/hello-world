#!/usr/bin/env python3
"""
Test retrieving messages from persistent memory.

Run this AFTER server restart to verify persistence.
"""

import asyncio
import sys
sys.path.insert(0, 'c:/Users/blake/Documents/mythRL')

from HoloLoom.config import Config, MemoryBackend
from HoloLoom.memory.backend_factory import create_memory_backend


async def test_retrieval():
    """Retrieve messages from persistent memory"""

    print("\n" + "="*60)
    print("  Memory Persistence Test - Retrieval")
    print("="*60 + "\n")

    # Initialize memory backend (same as server)
    config = Config.fast()
    config.memory_backend = MemoryBackend.HYBRID

    print("[INIT] Connecting to memory backend...")
    memory = await create_memory_backend(config)
    print("[OK] Memory backend connected\n")

    # Search for our distinctive test message
    search_term = "PERSISTENCE_TEST_UNIQUE_12345"

    print(f"[SEARCH] Looking for: '{search_term}'")
    print("[SEARCH] Querying persistent memory...\n")

    try:
        # Use the protocol's retrieve() method
        from HoloLoom.memory.protocol import MemoryQuery

        # Create query object
        query = MemoryQuery(
            text=search_term,
            user_id="chat_user",
            k=10
        )

        print("[METHOD] Using protocol retrieve() method...")
        results = await memory.retrieve(query)
        print(f"[RESULTS] Found {len(results)} results\n")

        # Display results
        if results:
            print(f"\n{'='*60}")
            print(f"  PERSISTENCE VERIFIED!")
            print(f"{'='*60}\n")

            for i, result in enumerate(results, 1):
                print(f"Result {i}:")

                # RetrievalResult has: memory, score, explanation
                if hasattr(result, 'memory'):
                    mem = result.memory
                    print(f"  Score: {result.score:.3f}")
                    print(f"  Content: {mem.text[:200]}...")
                    if hasattr(mem, 'metadata') and mem.metadata:
                        print(f"  Thread ID: {mem.metadata.get('thread_id', 'N/A')}")
                        print(f"  Role: {mem.metadata.get('role', 'N/A')}")
                        print(f"  Topic: {mem.metadata.get('thread_topic', 'N/A')}")
                else:
                    print(f"  {str(result)[:200]}...")
                print()

            print("[SUCCESS] Messages survived server restart!")
            print("[SUCCESS] Persistent memory is WORKING! \n")
            return True

        else:
            print(f"\n{'='*60}")
            print(f"  NO RESULTS FOUND")
            print(f"{'='*60}\n")
            print("[WARNING] Could not find test message in memory")
            print("[INFO] This could mean:")
            print("  1. Archiving failed (check server logs)")
            print("  2. Search method doesn't match backend")
            print("  3. Background task didn't complete before shutdown")
            print()
            return False

    except Exception as e:
        print(f"\n[ERROR] Retrieval failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def check_memory_stats():
    """Check general memory statistics"""
    config = Config.fast()
    config.memory_backend = MemoryBackend.HYBRID

    print("\n" + "="*60)
    print("  Memory Backend Statistics")
    print("="*60 + "\n")

    try:
        memory = await create_memory_backend(config)

        # Try to get stats
        if hasattr(memory, 'get_stats'):
            stats = await memory.get_stats()
            print(f"Stats: {stats}\n")

        # Try to count entities/memories
        if hasattr(memory, 'count'):
            count = await memory.count()
            print(f"Total memories: {count}\n")

        print("[INFO] Memory backend is operational")

    except Exception as e:
        print(f"[ERROR] {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "stats":
        asyncio.run(check_memory_stats())
    else:
        success = asyncio.run(test_retrieval())
        sys.exit(0 if success else 1)
