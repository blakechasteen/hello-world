"""
Simple test to verify the unified memory system works.
Run from repository root: python test_unified_memory.py
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import asyncio
from datetime import datetime
from pathlib import Path

# Import the modules directly without going through package __init__
root = Path(__file__).parent
protocol_path = root / 'HoloLoom' / 'memory' / 'protocol.py'
store_path = root / 'HoloLoom' / 'memory' / 'stores' / 'in_memory_store.py'

# Load modules directly
import importlib.util

# Load protocol module
spec = importlib.util.spec_from_file_location("protocol", protocol_path)
protocol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(protocol)

# Load store module  
spec = importlib.util.spec_from_file_location("in_memory_store", store_path)
store_module = importlib.util.module_from_spec(spec)
sys.modules['protocol'] = protocol  # Make protocol available for store
spec.loader.exec_module(store_module)

# Extract classes
Memory = protocol.Memory
MemoryQuery = protocol.MemoryQuery
RetrievalResult = protocol.RetrievalResult
Strategy = protocol.Strategy
UnifiedMemoryInterface = protocol.UnifiedMemoryInterface
InMemoryStore = store_module.InMemoryStore


async def test_basic():
    """Test basic store and recall."""
    print("\n" + "="*60)
    print("TEST: Basic Store & Recall")
    print("="*60)
    
    # Create store
    store = InMemoryStore()
    memory = UnifiedMemoryInterface(_store=store)
    
    # Store memories
    print("\n1. Storing memories...")
    mem1 = await memory.store(
        "Hive Jodi has 8 frames of brood, very active",
        context={'place': 'apiary', 'time': 'morning'},
        user_id="blake"
    )
    print(f"   ✓ Stored: {mem1}")
    
    mem2 = await memory.store(
        "Need to prep hives for winter - add insulation",
        context={'place': 'apiary', 'time': 'evening'},
        user_id="blake"
    )
    print(f"   ✓ Stored: {mem2}")
    
    mem3 = await memory.store(
        "Harvested 2 gallons of honey from Hive Matriarch",
        context={'place': 'apiary', 'time': 'afternoon'},
        user_id="blake"
    )
    print(f"   ✓ Stored: {mem3}")
    
    # Test semantic search
    print("\n2. Testing SEMANTIC strategy...")
    results = await memory.recall(
        "winter preparation",
        strategy=Strategy.SEMANTIC,
        user_id="blake"
    )
    
    print(f"   Found {len(results.memories)} memories:")
    for mem, score in zip(results.memories, results.scores):
        print(f"   [{score:.3f}] {mem.text[:60]}...")
    
    assert len(results.memories) > 0, "Should find memories"
    assert results.scores[0] > 0, "Should have scores"
    
    # Test temporal search
    print("\n3. Testing TEMPORAL strategy...")
    results = await memory.recall(
        "hive",
        strategy=Strategy.TEMPORAL,
        user_id="blake"
    )
    
    print(f"   Found {len(results.memories)} memories:")
    for mem, score in zip(results.memories, results.scores):
        print(f"   [{score:.3f}] {mem.text[:60]}...")
    
    # Test fused search
    print("\n4. Testing FUSED strategy...")
    results = await memory.recall(
        "honey bees",
        strategy=Strategy.FUSED,
        user_id="blake"
    )
    
    print(f"   Found {len(results.memories)} memories:")
    for mem, score in zip(results.memories, results.scores):
        print(f"   [{score:.3f}] {mem.text[:60]}...")
    
    # Health check
    print("\n5. Health check...")
    health = await memory.health_check()
    print(f"   Status: {health['store']['status']}")
    print(f"   Backend: {health['store']['backend']}")
    print(f"   Memory count: {health['store']['memory_count']}")
    
    assert health['store']['status'] == 'healthy', "Should be healthy"
    
    print("\n✅ All tests passed!\n")


async def test_strategies():
    """Test all strategies work."""
    print("\n" + "="*60)
    print("TEST: All Strategies")
    print("="*60)
    
    store = InMemoryStore()
    memory = UnifiedMemoryInterface(_store=store)
    
    # Store test data
    await memory.store("Test memory 1", context={}, user_id="test")
    await memory.store("Test memory 2", context={}, user_id="test")
    
    strategies = [
        Strategy.TEMPORAL,
        Strategy.SEMANTIC,
        Strategy.GRAPH,
        Strategy.PATTERN,
        Strategy.FUSED
    ]
    
    print("\nTesting strategies:")
    for strategy in strategies:
        results = await memory.recall("test", strategy=strategy, user_id="test")
        print(f"   {strategy.value:12s} → {len(results.memories)} memories")
        assert isinstance(results, RetrievalResult), f"Strategy {strategy} failed"
    
    print("\n✅ All strategies work!\n")


async def test_delete():
    """Test memory deletion."""
    print("\n" + "="*60)
    print("TEST: Delete Memory")
    print("="*60)
    
    store = InMemoryStore()
    memory = UnifiedMemoryInterface(_store=store)
    
    # Store
    mem_id = await memory.store("Temporary memory", context={}, user_id="test")
    print(f"\n1. Stored: {mem_id}")
    
    # Verify exists
    results = await memory.recall("temporary", user_id="test")
    print(f"2. Found {len(results.memories)} memories before delete")
    assert len(results.memories) == 1, "Should find memory"
    
    # Delete
    deleted = await store.delete(mem_id)
    print(f"3. Deleted: {deleted}")
    assert deleted, "Should delete successfully"
    
    # Verify gone
    results = await memory.recall("temporary", user_id="test")
    print(f"4. Found {len(results.memories)} memories after delete")
    assert len(results.memories) == 0, "Should not find memory"
    
    print("\n✅ Delete works!\n")


async def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("UNIFIED MEMORY SYSTEM - TEST SUITE")
    print("="*60)
    print("\nTesting protocol-based memory architecture...")
    
    try:
        await test_basic()
        await test_strategies()
        await test_delete()
        
        print("\n" + "="*60)
        print("🎉 ALL TESTS PASSED!")
        print("="*60)
        print("\nThe unified memory system is working correctly!")
        print("Next: Try running examples/unified_memory_demo.py")
        print()
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
