#!/usr/bin/env python3
"""
MCP Memory Server for HoloLoom
================================
Exposes HoloLoom hybrid memory (Neo4j + Qdrant) to Claude Desktop via MCP protocol.

This bridges Claude Desktop MCP <-> HoloLoom Hybrid Memory Store.
"""

import asyncio
import sys
import io
from pathlib import Path
from typing import Any, Sequence
from datetime import datetime
import importlib.util

# Fix Windows encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, str(Path(__file__).parent))

# Load HoloLoom modules
def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

hybrid_module = load_module("hybrid", "HoloLoom/memory/stores/hybrid_neo4j_qdrant.py")
HybridNeo4jQdrant = hybrid_module.HybridNeo4jQdrant
Memory = hybrid_module.Memory
MemoryQuery = hybrid_module.MemoryQuery
Strategy = hybrid_module.Strategy

loom_module = load_module("loom_command", "HoloLoom/loom/command.py")
LoomCommand = loom_module.LoomCommand
PatternCard = loom_module.PatternCard

# MCP imports
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Resource, Tool, TextContent, ImageContent, EmbeddedResource
except ImportError:
    print("ERROR: MCP SDK not installed. Install with: pip install mcp", file=sys.stderr)
    sys.exit(1)


# ============================================================================
# HoloLoom Memory Server
# ============================================================================

class HoloLoomMemoryServer:
    """MCP server exposing HoloLoom hybrid memory."""

    def __init__(self):
        self.app = Server("hololoom-memory")
        self.memory_store = None
        self.loom_command = None

        # Register handlers
        self.app.list_tools()(self.list_tools)
        self.app.call_tool()(self.call_tool)
        self.app.list_resources()(self.list_resources)
        self.app.read_resource()(self.read_resource)

    async def initialize(self):
        """Initialize HoloLoom memory store."""
        print("Initializing HoloLoom hybrid memory...", file=sys.stderr)

        self.memory_store = HybridNeo4jQdrant(
            neo4j_uri="bolt://localhost:7687",
            neo4j_password="hololoom123",
            qdrant_url="http://localhost:6333"
        )

        self.loom_command = LoomCommand(
            default_pattern=PatternCard.FAST,
            auto_select=True
        )

        print("✓ HoloLoom memory server ready", file=sys.stderr)

    async def list_tools(self) -> list[Tool]:
        """List available memory tools."""
        return [
            Tool(
                name="store_memory",
                description="Store a memory in HoloLoom (Neo4j + Qdrant)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "The memory content to store"
                        },
                        "user_id": {
                            "type": "string",
                            "description": "User identifier (default: blake)",
                            "default": "blake"
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional tags for categorization"
                        }
                    },
                    "required": ["text"]
                }
            ),
            Tool(
                name="recall_memories",
                description="Recall memories from HoloLoom using pattern-based retrieval",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Query text to search for"
                        },
                        "user_id": {
                            "type": "string",
                            "description": "User identifier (default: blake)",
                            "default": "blake"
                        },
                        "pattern": {
                            "type": "string",
                            "enum": ["bare", "fast", "fused"],
                            "description": "Pattern card (bare=fast graph, fast=semantic, fused=hybrid)"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Max memories to return (default: 5)",
                            "default": 5
                        }
                    },
                    "required": ["query"]
                }
            ),
            Tool(
                name="health_check",
                description="Check HoloLoom memory store health and statistics",
                inputSchema={
                    "type": "object",
                    "properties": {}
                }
            )
        ]

    async def call_tool(self, name: str, arguments: dict) -> Sequence[TextContent | ImageContent | EmbeddedResource]:
        """Execute memory tool."""

        if name == "store_memory":
            return await self._store_memory(arguments)

        elif name == "recall_memories":
            return await self._recall_memories(arguments)

        elif name == "health_check":
            return await self._health_check()

        else:
            raise ValueError(f"Unknown tool: {name}")

    async def _store_memory(self, args: dict) -> Sequence[TextContent]:
        """Store memory."""
        text = args["text"]
        user_id = args.get("user_id", "blake")
        tags = args.get("tags", [])

        # Create memory
        memory = Memory(
            id=f"mcp_{datetime.now().timestamp()}",
            text=text,
            timestamp=datetime.now(),
            context={"user_id": user_id, "tags": tags, "source": "claude_desktop"},
            metadata={"char_count": len(text), "word_count": len(text.split())}
        )

        # Store
        mem_id = await self.memory_store.store(memory)

        result = f"✓ Memory stored successfully\n"
        result += f"  ID: {mem_id}\n"
        result += f"  Text: {text[:100]}...\n" if len(text) > 100 else f"  Text: {text}\n"
        result += f"  User: {user_id}\n"
        result += f"  Tags: {tags}\n"

        return [TextContent(type="text", text=result)]

    async def _recall_memories(self, args: dict) -> Sequence[TextContent]:
        """Recall memories."""
        query_text = args["query"]
        user_id = args.get("user_id", "blake")
        pattern_pref = args.get("pattern")
        limit = args.get("limit", 5)

        # Select pattern
        pattern = self.loom_command.select_pattern(
            query_text=query_text,
            user_preference=pattern_pref
        )

        # Determine strategy
        if pattern.card == PatternCard.BARE:
            strategy = Strategy.GRAPH
        elif pattern.card == PatternCard.FAST:
            strategy = Strategy.SEMANTIC
        else:
            strategy = Strategy.FUSED

        # Query
        query = MemoryQuery(text=query_text, user_id=user_id, limit=limit)
        result = await self.memory_store.retrieve(query, strategy)

        # Format response
        if not result.memories:
            response = f"No memories found for: {query_text}\n"
        else:
            response = f"✓ Found {len(result.memories)} memories\n"
            response += f"  Pattern: {pattern.card.value.upper()}\n"
            response += f"  Strategy: {strategy}\n\n"

            for i, (mem, score) in enumerate(zip(result.memories, result.scores), 1):
                response += f"{i}. [{score:.3f}] {mem.text[:200]}...\n\n" if len(mem.text) > 200 else f"{i}. [{score:.3f}] {mem.text}\n\n"

        return [TextContent(type="text", text=response)]

    async def _health_check(self) -> Sequence[TextContent]:
        """Health check."""
        health = await self.memory_store.health_check()

        response = f"✓ HoloLoom Memory Health\n\n"
        response += f"Status: {health['status']}\n"
        response += f"Neo4j memories: {health['neo4j']['memories']}\n"
        response += f"Qdrant memories: {health['qdrant']['memories']}\n"
        response += f"\nBackend: Neo4j (graph) + Qdrant (vectors)\n"
        response += f"Architecture: Hyperspace (symbolic + semantic)\n"

        return [TextContent(type="text", text=response)]

    async def list_resources(self) -> list[Resource]:
        """List available memory resources."""
        return [
            Resource(
                uri="memory://hololoom/statistics",
                name="HoloLoom Memory Statistics",
                mimeType="text/plain",
                description="Current statistics from HoloLoom hybrid memory"
            )
        ]

    async def read_resource(self, uri: str) -> str:
        """Read memory resource."""
        if uri == "memory://hololoom/statistics":
            health = await self.memory_store.health_check()
            return f"""HoloLoom Memory Statistics
==========================

Status: {health['status']}
Neo4j memories: {health['neo4j']['memories']}
Qdrant memories: {health['qdrant']['memories']}

Backend: Hybrid (Neo4j + Qdrant)
Architecture: Hyperspace (symbolic + semantic)
"""
        else:
            raise ValueError(f"Unknown resource: {uri}")

    async def run(self):
        """Run MCP server."""
        await self.initialize()

        async with stdio_server() as (read_stream, write_stream):
            await self.app.run(
                read_stream,
                write_stream,
                self.app.create_initialization_options()
            )


# ============================================================================
# Main
# ============================================================================

async def main():
    server = HoloLoomMemoryServer()
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())
