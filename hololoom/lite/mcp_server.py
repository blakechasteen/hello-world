"""
HoloLoom Lite MCP Server
========================

Exposes HoloLoom Lite's simple 6-method API via Model Context Protocol.
This is the simplest HoloLoom interface for AI agents.

Tools:
- loom_experience: Store a memory
- loom_recall: Search memories
- loom_reflect: Learn from feedback
- loom_reason: Multi-step reasoning
- loom_query: One-shot Q&A
- loom_related: Graph navigation

Usage:
    python -m HoloLoom.lite.mcp_server

Claude Desktop config:
    {
        "mcpServers": {
            "hololoom-lite": {
                "command": "python",
                "args": ["-m", "HoloLoom.lite.mcp_server"],
                "cwd": "C:/path/to/mythRL"
            }
        }
    }

Author: Claude Code
Date: December 2025
"""

import asyncio
import logging
from typing import Any, Dict, List, Sequence

try:
    from mcp.server import Server
    from mcp.types import Tool, TextContent
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("MCP not installed. Run: pip install mcp")

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global HoloLoom Lite instance
_loom = None


async def get_loom():
    """Get or create HoloLoom Lite instance."""
    global _loom
    if _loom is None:
        from HoloLoom.lite.core import HoloLoomLite
        _loom = HoloLoomLite()
        await _loom._initialize()
        logger.info("HoloLoom Lite initialized")
    return _loom


# Create MCP server
if MCP_AVAILABLE:
    server = Server("hololoom-lite")
else:
    server = None


# ============================================================================
# Tools - The 6 HoloLoom Lite methods as MCP tools
# ============================================================================

if MCP_AVAILABLE:
    @server.list_tools()
    async def list_tools() -> List[Tool]:
        """List the 6 HoloLoom Lite tools."""
        return [
            Tool(
                name="loom_experience",
                description=(
                    "Store a memory. The simplest way to add knowledge to HoloLoom. "
                    "Memories are automatically indexed for semantic search and graph traversal."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "The content to remember"
                        },
                        "context": {
                            "type": "object",
                            "description": "Optional context metadata (source, tags, etc.)"
                        }
                    },
                    "required": ["text"]
                }
            ),

            Tool(
                name="loom_recall",
                description=(
                    "Search memories by semantic similarity. "
                    "Returns memories most relevant to the query."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "What to search for"
                        },
                        "limit": {
                            "type": "integer",
                            "default": 5,
                            "minimum": 1,
                            "maximum": 50,
                            "description": "Maximum memories to return"
                        }
                    },
                    "required": ["query"]
                }
            ),

            Tool(
                name="loom_reflect",
                description=(
                    "Learn from feedback on memories. "
                    "Strengthens good memories and weakens poor ones."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "memory_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "IDs of memories to reflect on"
                        },
                        "feedback": {
                            "type": "object",
                            "description": "Feedback object (e.g., {helpful: true, rating: 0.9})"
                        }
                    },
                    "required": ["memory_ids", "feedback"]
                }
            ),

            Tool(
                name="loom_reason",
                description=(
                    "Multi-step reasoning over memories. "
                    "Modes: 'direct' (single pass), 'verify' (with verification), "
                    "'research' (multi-query exploration)."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The question to reason about"
                        },
                        "mode": {
                            "type": "string",
                            "enum": ["direct", "verify", "research"],
                            "default": "direct",
                            "description": "Reasoning mode"
                        }
                    },
                    "required": ["query"]
                }
            ),

            Tool(
                name="loom_query",
                description=(
                    "One-shot Q&A. The simplest way to ask HoloLoom a question. "
                    "Searches memories and returns a synthesized answer."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "Your question"
                        }
                    },
                    "required": ["text"]
                }
            ),

            Tool(
                name="loom_related",
                description=(
                    "Navigate the memory graph. Find memories connected to a given memory "
                    "via semantic or temporal edges."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "memory_id": {
                            "type": "string",
                            "description": "Starting memory ID"
                        },
                        "hops": {
                            "type": "integer",
                            "default": 1,
                            "minimum": 1,
                            "maximum": 5,
                            "description": "How many hops to traverse (1=direct neighbors)"
                        },
                        "direction": {
                            "type": "string",
                            "enum": ["out", "in", "both"],
                            "default": "both",
                            "description": "Edge direction to follow"
                        }
                    },
                    "required": ["memory_id"]
                }
            )
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: Dict[str, Any]) -> Sequence[TextContent]:
        """Execute a HoloLoom Lite tool."""
        try:
            loom = await get_loom()

            if name == "loom_experience":
                text = arguments.get("text")
                context = arguments.get("context", {})

                if not text:
                    return [TextContent(type="text", text="Error: 'text' is required")]

                memory = await loom.experience(text, context=context)

                return [TextContent(
                    type="text",
                    text=f"Memory stored\nID: {memory.id}\nText: {text[:100]}..."
                )]

            elif name == "loom_recall":
                query = arguments.get("query")
                limit = arguments.get("limit", 5)

                if not query:
                    return [TextContent(type="text", text="Error: 'query' is required")]

                memories = await loom.recall(query, limit=limit)

                if not memories:
                    return [TextContent(type="text", text=f"No memories found for: {query}")]

                lines = [f"Found {len(memories)} memories:\n"]
                for i, mem in enumerate(memories, 1):
                    lines.append(f"{i}. [{mem.id[:8]}] {mem.text[:100]}...")

                return [TextContent(type="text", text="\n".join(lines))]

            elif name == "loom_reflect":
                memory_ids = arguments.get("memory_ids", [])
                feedback = arguments.get("feedback", {})

                if not memory_ids:
                    return [TextContent(type="text", text="Error: 'memory_ids' is required")]

                # Convert IDs to Memory objects (simplified)
                memories = await loom.recall(" ".join(memory_ids), limit=len(memory_ids))

                await loom.reflect(memories, feedback)

                return [TextContent(
                    type="text",
                    text=f"Reflected on {len(memories)} memories with feedback: {feedback}"
                )]

            elif name == "loom_reason":
                query = arguments.get("query")
                mode = arguments.get("mode", "direct")

                if not query:
                    return [TextContent(type="text", text="Error: 'query' is required")]

                result = await loom.reason(query, mode=mode)

                lines = [
                    f"Mode: {mode}",
                    f"Confidence: {result.confidence:.2f}",
                    "",
                    "Response:",
                    result.response or "(no response)"
                ]

                return [TextContent(type="text", text="\n".join(lines))]

            elif name == "loom_query":
                text = arguments.get("text")

                if not text:
                    return [TextContent(type="text", text="Error: 'text' is required")]

                result = await loom.query(text)

                lines = [
                    f"Confidence: {result.confidence:.2f}",
                    "",
                    "Answer:",
                    result.response or "(no answer)"
                ]

                return [TextContent(type="text", text="\n".join(lines))]

            elif name == "loom_related":
                memory_id = arguments.get("memory_id")
                hops = arguments.get("hops", 1)
                direction = arguments.get("direction", "both")

                if not memory_id:
                    return [TextContent(type="text", text="Error: 'memory_id' is required")]

                related = await loom.related(memory_id, hops=hops, direction=direction)

                if not related:
                    return [TextContent(type="text", text=f"No related memories found for: {memory_id}")]

                lines = [f"Found {len(related)} related memories ({hops} hop(s), {direction}):\n"]
                for i, mem in enumerate(related, 1):
                    lines.append(f"{i}. [{mem.id[:8]}] {mem.text[:100]}...")

                return [TextContent(type="text", text="\n".join(lines))]

            else:
                return [TextContent(type="text", text=f"Unknown tool: {name}")]

        except Exception as e:
            logger.error(f"Error in {name}: {e}")
            return [TextContent(type="text", text=f"Error: {e}")]


# ============================================================================
# Server Entry Point
# ============================================================================

async def main():
    """Run the MCP server."""
    if not MCP_AVAILABLE:
        print("Cannot start: MCP not installed")
        print("Install with: pip install mcp")
        return

    logger.info("Starting HoloLoom Lite MCP Server...")
    logger.info("Tools: loom_experience, loom_recall, loom_reflect, loom_reason, loom_query, loom_related")

    from mcp.server.stdio import stdio_server
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
