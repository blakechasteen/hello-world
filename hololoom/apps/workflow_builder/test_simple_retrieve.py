#!/usr/bin/env python3
"""Simple retrieval test - just get all recent memories"""

import asyncio
import sys
sys.path.insert(0, 'c:/Users/blake/Documents/mythRL')

from hololoom.config import Config, MemoryBackend
from hololoom.memory.backend_factory import create_memory_backend
from hololoom.memory.protocol import MemoryQuery


async def test():
    config = Config.fast()
    config.memory_backend = MemoryBackend.HYBRID

    print("Connecting to memory...")
    memory = await create_memory_backend(config)
    print("Connected!\n")

    # Simple query - get recent
    query = MemoryQuery(text="semantic", user_id="chat_user", limit=20)

    print("Querying for 'semantic'...")
    result = await memory.recall(query, limit=20)

    print(f"Found {len(result.memories)} memories:\n")

    for i, mem in enumerate(result.memories, 1):
        print(f"{i}. {mem.text[:100]}...")
        if hasattr(mem, 'metadata') and mem.metadata:
            print(f"   Thread: {mem.metadata.get('thread_id', 'N/A')[:20]}...")
        print()


if __name__ == "__main__":
    asyncio.run(test())
