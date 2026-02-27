#!/usr/bin/env python3
"""Test archiving directly to see any errors"""

import asyncio
import sys
sys.path.insert(0, 'c:/Users/blake/Documents/mythRL')

from hololoom.config import Config, MemoryBackend
from hololoom.memory.backend_factory import create_memory_backend
from hololoom.memory.protocol import Memory
from datetime import datetime


async def test_archive():
    """Test archiving a message directly"""

    config = Config.fast()
    config.memory_backend = MemoryBackend.HYBRID

    print("Connecting to memory...")
    memory = await create_memory_backend(config)
    print("Connected!\n")

    # Create a test message
    test_message = Memory(
        id="test_message_123",
        text="TEST DIRECT ARCHIVE - This is a test chat message to verify archiving works",
        timestamp=datetime.now(),
        context={},
        metadata={
            'thread_id': "test_thread_456",
            'role': "user",
            'depth': 0,
            'thread_topic': "TESTING",
            'source': "chat_test",
            'user_id': "chat_user",
        }
    )

    print(f"Archiving message: {test_message.text[:50]}...")

    try:
        # Direct archive (no background task)
        result = await memory.store(test_message, user_id="chat_user")
        print(f"✓ Archived successfully! ID: {result}\n")

        # Try to retrieve it immediately
        from hololoom.memory.protocol import MemoryQuery

        query = MemoryQuery(text="TEST DIRECT ARCHIVE", user_id="chat_user", limit=5)
        print("Retrieving...")
        recall_result = await memory.recall(query, limit=5)

        print(f"Found {len(recall_result.memories)} results:")
        for mem in recall_result.memories:
            print(f"  - {mem.text[:60]}...")

        if any("TEST DIRECT ARCHIVE" in m.text for m in recall_result.memories):
            print("\n✓✓✓ ARCHIVING WORKS! ✓✓✓")
        else:
            print("\n⚠ Message archived but not found in search")

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_archive())
