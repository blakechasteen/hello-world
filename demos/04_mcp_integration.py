#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HoloLoom MCP Integration Demo
==============================
Demonstrates how to use HoloLoom via Model Context Protocol (MCP).

This shows:
1. HoloLoom MCP server setup
2. Exposing memory operations to Claude Desktop
3. Conversational interface via MCP
4. Configuration and testing

MCP allows Claude Desktop, VS Code, and other MCP-compatible tools
to query and store memories in your HoloLoom system.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def show_mcp_config():
    """Show MCP server configuration."""

    print("=" * 80)
    print("HOLOLOOM MCP INTEGRATION DEMO")
    print("=" * 80)

    print("\n[OVERVIEW]")
    print("-" * 80)
    print("HoloLoom provides an MCP server that exposes:")
    print("  - Memory storage (store individual memories)")
    print("  - Memory search (semantic + BM25)")
    print("  - Batch operations (store multiple memories)")
    print("  - Conversational interface (chat with auto-memory)")
    print("  - Episode management (retrieve conversation history)")

    print("\n[SETUP]")
    print("-" * 80)
    print("1. Configure Claude Desktop to use HoloLoom MCP server:")
    print()
    print("   Location: %APPDATA%\\Claude\\claude_desktop_config.json")
    print("   (or ~/Library/Application Support/Claude/claude_desktop_config.json on Mac)")
    print()
    print("   Add this configuration:")
    print("""
   {
     "mcpServers": {
       "holoLoom-memory": {
         "command": "python",
         "args": [
           "-m",
           "HoloLoom.memory.mcp_server"
         ],
         "env": {
           "PYTHONPATH": "C:\\\\Users\\\\YOUR_USERNAME\\\\Documents\\\\mythRL"
         }
       }
     }
   }
   """)

    print("\n2. The server will start automatically when Claude Desktop launches")

    print("\n[AVAILABLE MCP TOOLS]")
    print("-" * 80)
    print("""
    1. store_memory
       - Store a single memory
       - Args: text, user_id, metadata (optional)

    2. search_memories
       - Search memories by query
       - Args: query, user_id, limit (default: 10)

    3. store_batch
       - Store multiple memories at once
       - Args: memories (list), user_id

    4. chat
       - Conversational interface with auto-memory
       - Args: message, user_id
       - Automatically scores importance and stores significant turns

    5. retrieve_episode
       - Get all memories from a specific episode/conversation
       - Args: episode_id, user_id

    6. get_conversation_stats
       - Get statistics about conversations
       - Args: user_id
    """)

    print("\n[EXAMPLE USAGE IN CLAUDE DESKTOP]")
    print("-" * 80)
    print("""
    You: "Store this in memory: HoloLoom uses a weaving metaphor"
    Claude: [Uses store_memory tool]

    You: "What do you remember about HoloLoom?"
    Claude: [Uses search_memories tool with query "HoloLoom"]

    You: "Let's chat about the architecture"
    Claude: [Uses chat tool, which auto-stores important exchanges]
    """)

    print("\n[TESTING THE SERVER]")
    print("-" * 80)
    print("To test the server standalone:")
    print()
    print("  python -m HoloLoom.memory.mcp_server")
    print()
    print("The server will log all operations to console.")

    print("\n[CONFIGURATION OPTIONS]")
    print("-" * 80)
    print("""
    You can configure the memory backend in mcp_server.py:

    - 'simple': In-memory (no persistence, fast)
    - 'neo4j': Graph database (requires Neo4j running)
    - 'qdrant': Vector database (requires Qdrant running)
    - 'neo4j+qdrant': Hybrid (both backends)

    Default: 'simple' (no external dependencies)

    To use Neo4j or Qdrant, see:
      - HoloLoom/memory/NEO4J_README.md
      - HoloLoom/memory/QUICKSTART.md
    """)

    print("\n[CONVERSATIONAL AUTO-SPIN]")
    print("-" * 80)
    print("""
    The chat() tool has intelligent importance scoring:

    HIGH importance (remembered):
    - Questions and answers
    - Facts and decisions
    - Domain-specific terms
    - Technical discussions

    LOW importance (forgotten):
    - Greetings ("hi", "hello")
    - Acknowledgments ("ok", "thanks")
    - Trivial exchanges

    This filters signal from noise, keeping your memory clean.
    """)

    print("\n[CURRENT CONFIGURATION]")
    print("-" * 80)

    # Check if MCP config file exists
    import os
    config_paths = [
        Path.home() / "AppData" / "Roaming" / "Claude" / "claude_desktop_config.json",  # Windows
        Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json",  # Mac
        Path(__file__).parent.parent / "mcp_server" / "claude_desktop_config.json"  # Local
    ]

    found_config = False
    for config_path in config_paths:
        if config_path.exists():
            print(f"✓ Found config: {config_path}")
            found_config = True

            try:
                import json
                with open(config_path) as f:
                    config = json.load(f)

                if "mcpServers" in config and "holoLoom-memory" in config["mcpServers"]:
                    print("✓ HoloLoom MCP server configured")
                    server_config = config["mcpServers"]["holoLoom-memory"]
                    print(f"  Command: {server_config.get('command', 'N/A')}")
                    print(f"  Args: {server_config.get('args', [])}")
                else:
                    print("⚠ HoloLoom MCP server NOT configured in this file")
            except Exception as e:
                print(f"⚠ Error reading config: {e}")

    if not found_config:
        print("⚠ No Claude Desktop config found")
        print("  Run setup above to create configuration")

    print("\n" + "=" * 80)
    print("MCP INTEGRATION INFO COMPLETE")
    print("=" * 80)
    print("Next steps:")
    print("  1. Configure Claude Desktop (see setup above)")
    print("  2. Restart Claude Desktop")
    print("  3. Try using memory tools in conversations")
    print("  4. Check HoloLoom/memory/QUICKSTART.md for details")
    print("=" * 80)


if __name__ == '__main__':
    show_mcp_config()
