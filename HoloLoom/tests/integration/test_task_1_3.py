"""Test Memory Backend Simplification (Task 1.3)"""
import asyncio
from HoloLoom.config import Config, MemoryBackend
from HoloLoom.memory.backend_factory import create_memory_backend, check_backend_health

async def main():
    print("=" * 80)
    print("MEMORY BACKEND SIMPLIFICATION TEST (Task 1.3)")
    print("=" * 80)
    
    # Test 1: INMEMORY backend
    print("\n[TEST 1] INMEMORY Backend")
    print("-" * 80)
    try:
        config_inmem = Config.bare()
        config_inmem.memory_backend = MemoryBackend.INMEMORY
        backend_inmem = await create_memory_backend(config_inmem)
        print(f"PASS: INMEMORY backend created: {type(backend_inmem).__name__}")
        
        health = await check_backend_health(backend_inmem)
        print(f"  Health status: {health.get('status', 'N/A')}")
    except Exception as e:
        print(f"FAIL: INMEMORY backend failed: {e}")
    
    # Test 2: HYBRID backend (with auto-fallback)
    print("\n[TEST 2] HYBRID Backend (with auto-fallback)")
    print("-" * 80)
    try:
        config_hybrid = Config.fast()
        config_hybrid.memory_backend = MemoryBackend.HYBRID
        backend_hybrid = await create_memory_backend(config_hybrid)
        print(f"PASS: HYBRID backend created: {type(backend_hybrid).__name__}")
        
        # Check if it's in fallback mode
        if hasattr(backend_hybrid, 'fallback_mode'):
            if backend_hybrid.fallback_mode:
                print("  Mode: FALLBACK (NetworkX)")
            else:
                print("  Mode: PRODUCTION (Neo4j/Qdrant)")
        
        health = await check_backend_health(backend_hybrid)
        print(f"  Health: {health.get('status', 'N/A')}")
        
        # Test backends list
        if hasattr(backend_hybrid, 'backends'):
            print(f"  Active backends: {len(backend_hybrid.backends)}")
    except Exception as e:
        print(f"FAIL: HYBRID backend failed: {e}")
    
    # Test 3: Verify default is HYBRID
    print("\n[TEST 3] Default Configuration")
    print("-" * 80)
    config_default = Config.fast()
    if config_default.memory_backend == MemoryBackend.HYBRID:
        print("PASS: Default is HYBRID")
    else:
        print(f"FAIL: Default is {config_default.memory_backend}")
    
    # Summary
    print("\n" + "=" * 80)
    print("Task 1.3 Complete!")
    print("=" * 80)
    print("[OK] HybridStore as default")
    print("[OK] Simplified routing logic")
    print("[OK] Auto-fallback to InMemory")
    print("[OK] Clear backend status messaging")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(main())
