"""
Simple test of INMEMORY backend to verify it works correctly.
"""
import asyncio
from HoloLoom.config import Config, MemoryBackend
from HoloLoom.memory.backend_factory import create_memory_backend
from HoloLoom.memory.protocol import Memory, MemoryQuery
from datetime import datetime

async def test_inmemory_basic():
    """Test basic INMEMORY backend operations."""
    print("\n" + "="*70)
    print("INMEMORY Backend - Basic Test")
    print("="*70)
    
    # Create config with INMEMORY backend
    config = Config.fast()
    config.memory_backend = MemoryBackend.INMEMORY
    
    print(f"\n[1/6] Backend: {config.memory_backend.value}")
    
    # Create backend
    print("[2/6] Creating INMEMORY backend...")
    try:
        backend = await create_memory_backend(config)
        print("      [OK] Backend created successfully")
    except Exception as e:
        print(f"      [FAIL] Failed to create backend: {e}")
        return False
    
    # Store a memory
    print("\n[3/6] Storing test memory...")
    try:
        memory = Memory(
            id="test_001",
            text="This is a test memory about machine learning",
            timestamp=datetime.now(),
            context={"episode": "test", "entities": ["ml", "learning"]},
            metadata={"source": "test", "user_id": "default"}
        )
        stored_id = await backend.store(memory, user_id="default")
        print(f"      [OK] Stored memory: {stored_id}")
    except Exception as e:
        print(f"      [FAIL] Failed to store: {e}")
        return False
    
    # Store multiple memories
    print("\n[4/6] Storing batch of memories...")
    try:
        memories = [
            Memory(
                id=f"test_{i:03d}",
                text=f"Test memory #{i} about neural networks",
                timestamp=datetime.now(),
                context={"episode": "test", "entities": ["nn"]},
                metadata={"source": "test", "user_id": "default"}
            )
            for i in range(2, 5)
        ]
        stored_ids = await backend.store_many(memories, user_id="default")
        print(f"      [OK] Stored {len(stored_ids)} memories: {stored_ids}")
    except Exception as e:
        print(f"      [FAIL] Failed to store batch: {e}")
        return False
    
    # Query memories
    print("\n[5/6] Querying stored memories...")
    try:
        query = MemoryQuery(
            text="machine learning",
            user_id="default",
            limit=5
        )
        result = await backend.recall(query, limit=5)
        print(f"      [OK] Retrieved {len(result.memories)} memories")
        for i, mem in enumerate(result.memories[:3]):
            print(f"        [{i+1}] {mem.text[:60]}...")
    except Exception as e:
        print(f"      [FAIL] Failed to query: {e}")
        return False
    
    # Health check
    print("\n[6/6] Health check...")
    try:
        health = await backend.health_check()
        print(f"      [OK] Backend health: {health}")
    except Exception as e:
        print(f"      [FAIL] Health check failed: {e}")
        return False
    
    print("\n" + "="*70)
    print("[SUCCESS] ALL TESTS PASSED")
    print("="*70)
    return True


async def test_inmemory_edge_cases():
    """Test edge cases."""
    print("\n" + "="*70)
    print("INMEMORY Backend - Edge Cases")
    print("="*70)
    
    config = Config.fast()
    config.memory_backend = MemoryBackend.INMEMORY
    backend = await create_memory_backend(config)
    
    # Test: Empty query
    print("\n[1/3] Testing empty query...")
    try:
        query = MemoryQuery(
            text="",
            user_id="default",
            limit=5
        )
        result = await backend.recall(query, limit=5)
        print(f"      [OK] Empty query handled: {len(result.memories)} results")
    except Exception as e:
        print(f"      [FAIL] Empty query failed: {e}")
        return False
    
    # Test: Non-existent user
    print("\n[2/3] Testing non-existent user...")
    try:
        query = MemoryQuery(
            text="test",
            user_id="nonexistent_user",
            limit=5
        )
        result = await backend.recall(query, limit=5)
        print(f"      [OK] Non-existent user handled: {len(result.memories)} results")
    except Exception as e:
        print(f"      [FAIL] Non-existent user failed: {e}")
        return False
    
    # Test: Multiple stores then query
    print("\n[3/3] Testing multiple operations...")
    try:
        # Store 10 memories
        for i in range(10):
            mem = Memory(
                id=f"edge_case_{i}",
                text=f"Memory about topic {i % 3}",
                timestamp=datetime.now(),
                context={},
                metadata={"source": "edge_case", "user_id": "test_user"}
            )
            await backend.store(mem, user_id="test_user")
        
        # Query them
        query = MemoryQuery(text="topic", user_id="test_user", limit=5)
        result = await backend.recall(query, limit=5)
        print(f"      [OK] Multiple operations handled: {len(result.memories)} results")
    except Exception as e:
        print(f"      [FAIL] Multiple operations failed: {e}")
        return False
    
    print("\n" + "="*70)
    print("[SUCCESS] ALL EDGE CASE TESTS PASSED")
    print("="*70)
    return True


async def main():
    """Run all tests."""
    success = True
    
    try:
        success = await test_inmemory_basic() and success
    except Exception as e:
        print(f"\n[ERROR] Basic test crashed: {e}")
        import traceback
        traceback.print_exc()
        success = False
    
    try:
        success = await test_inmemory_edge_cases() and success
    except Exception as e:
        print(f"\n[ERROR] Edge case test crashed: {e}")
        import traceback
        traceback.print_exc()
        success = False
    
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
