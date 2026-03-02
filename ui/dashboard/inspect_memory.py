#!/usr/bin/env python3
"""
🔥 Memory Store Inspector
=========================
Check the status of all HoloLoom memory backends.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from hololoom.unified_api import HoloLoom


async def inspect_memory_stores():
    """Inspect all memory store backends."""
    print("\n" + "="*70)
    print("🔥 HOLOLOOM MEMORY STORE INSPECTION")
    print("="*70 + "\n")
    
    print("📡 Initializing HoloLoom...")
    loom = await HoloLoom.create(pattern="fast")
    print("✅ HoloLoom initialized\n")
    
    # Check if memory attribute exists
    if not hasattr(loom, 'memory'):
        print("⚠️  No memory attribute found on HoloLoom instance")
        print("   This version may use a different memory interface\n")
        
        # Try to access through orchestrator
        if hasattr(loom, 'orchestrator'):
            print("🔍 Checking orchestrator memory...")
            orch = loom.orchestrator
            
            if hasattr(orch, 'memory'):
                print("✅ Found memory through orchestrator")
                memory = orch.memory
                
                # Try to get stats
                if hasattr(memory, 'get_stats'):
                    stats = await memory.get_stats()
                    print("\n📊 Memory Statistics:")
                    for key, value in stats.items():
                        print(f"   • {key}: {value}")
                else:
                    print("   ℹ️  No get_stats() method available")
                    
                # Show available memory methods
                print("\n🔧 Available Memory Methods:")
                methods = [m for m in dir(memory) if not m.startswith('_')]
                for method in methods[:10]:  # Show first 10
                    print(f"   • {method}")
                if len(methods) > 10:
                    print(f"   ... and {len(methods) - 10} more")
                    
            else:
                print("   ⚠️  No memory attribute on orchestrator either")
        
        return
    
    # If memory exists directly on loom
    memory = loom.memory
    print("✅ Memory backend found\n")
    
    # Get statistics
    print("📊 Fetching memory statistics...")
    try:
        stats = await memory.get_stats()
        print("\n🔥 Memory Store Statistics:")
        print("-" * 70)
        
        for key, value in stats.items():
            if isinstance(value, dict):
                print(f"\n{key}:")
                for subkey, subval in value.items():
                    print(f"  • {subkey}: {subval}")
            else:
                print(f"• {key}: {value}")
                
    except AttributeError:
        print("⚠️  get_stats() method not available")
        print("   Available memory methods:")
        methods = [m for m in dir(memory) if not m.startswith('_') and callable(getattr(memory, m))]
        for method in methods:
            print(f"   • {method}()")
    
    except Exception as e:
        print(f"❌ Error getting stats: {e}")
    
    # Try to query memory
    print("\n\n🔍 Testing Memory Query...")
    try:
        results = await loom.query("test memory query")
        print(f"✅ Query successful")
        print(f"   Response type: {type(results)}")
        if hasattr(results, 'content'):
            print(f"   Content: {str(results.content)[:100]}...")
    except Exception as e:
        print(f"❌ Query failed: {e}")
    
    print("\n" + "="*70)
    print("🔥 INSPECTION COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    asyncio.run(inspect_memory_stores())
