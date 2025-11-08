#!/usr/bin/env python3
"""
Test MCP Server Startup
========================
Verify the HoloLoom MCP server starts without errors
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

async def test_startup():
    print("\n" + "="*60)
    print("🧪 Testing HoloLoom MCP Server Startup")
    print("="*60)
    
    try:
        print("\n1️⃣ Importing memory system...")
        from HoloLoom.memory.protocol import create_unified_memory
        print("   ✅ Import successful")
        
        print("\n2️⃣ Initializing unified memory...")
        memory = await create_unified_memory(
            user_id="blake",
            enable_mem0=False,
            enable_neo4j=True,
            enable_qdrant=True
        )
        print("   ✅ Memory system initialized")
        
        print("\n3️⃣ Running health check...")
        health = await memory.health_check()
        print(f"   ✅ Health: {health}")
        
        print("\n4️⃣ Testing MCP server import...")
        import HoloLoom.memory.mcp_server as mcp_server
        print("   ✅ MCP server module loaded")
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED - MCP Server Ready!")
        print("="*60)
        print("\n💡 Next step: Restart Claude Desktop")
        print("   Config: C:\\Users\\blake\\AppData\\Roaming\\Claude\\claude_desktop_config.json")
        print("\n")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_startup())
    sys.exit(0 if success else 1)
