"""
Test MCP Server Initialization
===============================
Standalone test for the MCP server without package imports.
"""

import asyncio
import sys
import importlib.util
from pathlib import Path

async def test_mcp():
    print("=" * 60)
    print("TESTING MCP SERVER")
    print("=" * 60)
    
    # Load modules directly to avoid package __init__ issues
    base_path = Path(__file__).parent / "HoloLoom" / "memory"
    
    # Load protocol module
    spec = importlib.util.spec_from_file_location("protocol", base_path / "protocol.py")
    protocol = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(protocol)
    
    # Load in_memory_store
    spec = importlib.util.spec_from_file_location(
        "in_memory_store", 
        base_path / "stores" / "in_memory_store.py"
    )
    in_memory_store = importlib.util.module_from_spec(spec)
    sys.modules['protocol'] = protocol  # Make it available for imports
    spec.loader.exec_module(in_memory_store)
    
    # Load mcp_server
    spec = importlib.util.spec_from_file_location("mcp_server", base_path / "mcp_server.py")
    mcp_server = importlib.util.module_from_spec(spec)
    sys.modules['protocol'] = protocol
    spec.loader.exec_module(mcp_server)
    
    # Get the functions we need
    init_memory = mcp_server.init_memory
    server = mcp_server.server
    Strategy = protocol.Strategy
    
    print("\n1. Initializing memory system...")
    await init_memory(user_id="test", enable_mem0=False, enable_neo4j=False, enable_qdrant=False)
    memory = mcp_server.memory  # Get the global memory instance
    print("   ✓ Memory system initialized")
    
    print("\n2. Testing memory operations...")
    
    # Store some test memories
    mem1 = await memory.store("Hive Jodi needs winter prep", tags=["apiary", "urgent"])
    print(f"   ✓ Stored memory 1: {mem1}")
    
    mem2 = await memory.store("Queen bee spotted in hive 3", context={"place": "apiary"})
    print(f"   ✓ Stored memory 2: {mem2}")
    
    mem3 = await memory.store("Need to order more frames", tags=["supplies"])
    print(f"   ✓ Stored memory 3: {mem3}")
    
    print("\n3. Testing recall...")
    
    result = await memory.recall("hive", strategy=Strategy.FUSED, limit=5)
    print(f"   ✓ Found {len(result.memories)} memories")
    for i, (mem, score) in enumerate(zip(result.memories, result.scores), 1):
        print(f"     {i}. [{score:.3f}] {mem.text[:50]}")
    
    print("\n4. Testing health check...")
    health = await memory.health_check()
    print(f"   ✓ Status: {health['status']}")
    print(f"   ✓ Backend: {health['backend']}")
    print(f"   ✓ Memory count: {health['memory_count']}")
    
    print("\n5. Verifying MCP server components...")
    if server:
        print("   ✓ MCP Server object created")
        # Test tool listing
        try:
            list_tools = mcp_server.list_tools
            tools = await list_tools()
            print(f"   ✓ Available tools: {len(tools)}")
            for tool in tools:
                print(f"     - {tool.name}: {tool.description[:50]}...")
        except Exception as e:
            print(f"   ⚠ Could not list tools: {e}")
        
        # Test resource listing
        try:
            list_memories = mcp_server.list_memories
            resources = await list_memories()
            print(f"   ✓ Available resources: {len(resources)}")
            for res in resources[:3]:
                print(f"     - {res.uri}: {res.name}")
        except Exception as e:
            print(f"   ⚠ Could not list resources: {e}")
    else:
        print("   ⚠ MCP Server not available")
    
    print("\n" + "=" * 60)
    print("✨ MCP SERVER TEST COMPLETE")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Run the server: python -m HoloLoom.memory.mcp_server")
    print("2. Configure Claude Desktop with mcp_config.json")
    print("3. Ask Claude to search your memories!")

if __name__ == "__main__":
    asyncio.run(test_mcp())
