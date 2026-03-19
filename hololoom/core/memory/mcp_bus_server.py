"""
MCP Server for Memory Bus — spec §10.4.

Exposes 5 MCP tools + 2 resources wrapping the MemoryBus.
Lives alongside existing mcp_server.py (does not replace it).

Tools:
    memory_query       — Query long-term memory (4-level cascade)
    memory_store       — Store/promote information to long-term memory
    memory_entities    — Browse the entity graph
    memory_resolve_entity — Resolve name/alias to entity IDs
    memory_explain     — Provenance chain for a memory item

Resources:
    memory://status    — System stats and health
    memory://pressure  — Current pressure signals and tier
"""

import json
import logging
from collections.abc import Sequence
from typing import Any

logger = logging.getLogger(__name__)

try:
    from mcp.server import Server
    from mcp.types import Resource, TextContent, Tool
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    logger.info("MCP SDK not installed — mcp_bus_server unavailable")

from .bus import MemoryBus, MemoryItem, MemoryQuery


def create_mcp_bus_server(
    bus: MemoryBus,
    name: str = "hololoom-memory-bus",
) -> Any:
    """Create and configure the MCP server wrapping a MemoryBus.

    Args:
        bus: An initialized MemoryBus instance.
        name: Server name for MCP registration.

    Returns:
        Configured MCP Server instance.
    """
    if not MCP_AVAILABLE:
        raise ImportError("MCP SDK required. Install with: pip install mcp")

    server = Server(name)

    # =========================================================================
    # Tool Definitions
    # =========================================================================

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name="memory_query",
                description=(
                    "Query long-term memory. Structured filters applied first, "
                    "semantic as fallback. Resolution cascade: exact → structured → graph → semantic."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "intent": {
                            "type": "string",
                            "description": "Human-readable description of what you're looking for",
                        },
                        "entity_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Exact entity IDs to look up (Level 1: exact match)",
                        },
                        "entity_type": {
                            "type": "string",
                            "description": "Filter by entity type (e.g. 'Hive', 'Location')",
                        },
                        "memory_type": {
                            "type": "string",
                            "enum": ["episodic", "entity", "factual", "procedural", "plan", "config"],
                            "description": "Filter by memory type",
                        },
                        "time_after": {
                            "type": "string",
                            "description": "ISO timestamp lower bound",
                        },
                        "time_before": {
                            "type": "string",
                            "description": "ISO timestamp upper bound",
                        },
                        "semantic_text": {
                            "type": "string",
                            "description": "Natural language similarity search (DEFERRED — returns 'not available')",
                        },
                        "max_results": {
                            "type": "integer",
                            "default": 5,
                            "description": "Maximum results to return",
                        },
                    },
                    "required": ["intent"],
                },
            ),
            Tool(
                name="memory_store",
                description="Promote information to long-term memory.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "The content to store",
                        },
                        "memory_type": {
                            "type": "string",
                            "enum": ["episodic", "entity", "factual", "procedural", "plan"],
                            "description": "Type of memory",
                        },
                        "entity_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Link to existing entity IDs",
                        },
                        "importance": {
                            "type": "number",
                            "default": 0.5,
                            "description": "Importance score 0.0–1.0",
                        },
                        "scope": {
                            "type": "string",
                            "enum": ["local", "shared"],
                            "default": "local",
                        },
                    },
                    "required": ["content", "memory_type"],
                },
            ),
            Tool(
                name="memory_entities",
                description="Browse the entity graph.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search entities by name",
                        },
                        "entity_type": {
                            "type": "string",
                            "description": "Filter by entity type",
                        },
                        "connected_to": {
                            "type": "string",
                            "description": "Entity ID to find neighbors of",
                        },
                        "max_hops": {
                            "type": "integer",
                            "default": 2,
                            "description": "Max traversal depth",
                        },
                    },
                },
            ),
            Tool(
                name="memory_resolve_entity",
                description=(
                    "Resolve a name or alias to entity IDs. "
                    "Returns candidates ranked by confidence. "
                    "Prevents 'wrong entity ID' cascading errors."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "name_or_alias": {
                            "type": "string",
                            "description": "Name or alias to resolve",
                        },
                        "loom_scope": {
                            "type": "string",
                            "description": "Restrict to a specific Loom's entities",
                        },
                        "max_candidates": {
                            "type": "integer",
                            "default": 3,
                        },
                    },
                    "required": ["name_or_alias"],
                },
            ),
            Tool(
                name="memory_explain",
                description=(
                    "Return the provenance chain for a memory item. "
                    "Shows what it derived from, which reasoning loop created it, "
                    "and whether it's confirmed or inferred."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "node_id": {
                            "type": "string",
                            "description": "ID of the memory item to explain",
                        },
                    },
                    "required": ["node_id"],
                },
            ),
        ]

    # =========================================================================
    # Tool Execution
    # =========================================================================

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> Sequence[TextContent]:
        try:
            if name == "memory_query":
                q = MemoryQuery(
                    intent=arguments["intent"],
                    entity_ids=arguments.get("entity_ids"),
                    entity_type=arguments.get("entity_type"),
                    memory_type=arguments.get("memory_type"),
                    time_after=arguments.get("time_after"),
                    time_before=arguments.get("time_before"),
                    semantic_text=arguments.get("semantic_text"),
                    max_results=arguments.get("max_results", 5),
                )
                result = await bus.query(q)
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "resolution_path": result.resolution_path,
                        "token_estimate": result.token_estimate,
                        "truncated": result.truncated,
                        "count": len(result.items),
                        "items": result.items,
                    }, indent=2, default=str),
                )]

            elif name == "memory_store":
                item = MemoryItem(
                    content=arguments["content"],
                    memory_type=arguments["memory_type"],
                    entity_ids=arguments.get("entity_ids"),
                    importance=arguments.get("importance", 0.5),
                    scope=arguments.get("scope", "local"),
                )
                node_id = await bus.store(item)
                return [TextContent(
                    type="text",
                    text=json.dumps({"stored": True, "node_id": node_id}),
                )]

            elif name == "memory_entities":
                entities = await bus.entities(
                    query=arguments.get("query"),
                    entity_type=arguments.get("entity_type"),
                    connected_to=arguments.get("connected_to"),
                    max_hops=arguments.get("max_hops", 2),
                )
                return [TextContent(
                    type="text",
                    text=json.dumps({"count": len(entities), "entities": entities}, default=str),
                )]

            elif name == "memory_resolve_entity":
                candidates = await bus.resolve_entity(
                    name_or_alias=arguments["name_or_alias"],
                    loom_scope=arguments.get("loom_scope"),
                    max_candidates=arguments.get("max_candidates", 3),
                )
                return [TextContent(
                    type="text",
                    text=json.dumps({"candidates": candidates}, default=str),
                )]

            elif name == "memory_explain":
                explanation = await bus.explain(arguments["node_id"])
                return [TextContent(
                    type="text",
                    text=json.dumps(explanation, indent=2, default=str),
                )]

            else:
                return [TextContent(type="text", text=f"Unknown tool: {name}")]

        except Exception as e:
            logger.error("Tool %s failed: %s", name, e)
            return [TextContent(type="text", text=json.dumps({"error": str(e)}))]

    # =========================================================================
    # Resources
    # =========================================================================

    @server.list_resources()
    async def list_resources() -> list[Resource]:
        return [
            Resource(
                uri="memory://status",
                name="Memory System Status",
                description="Current stats: query counts, resolution distribution, health",
                mimeType="application/json",
            ),
            Resource(
                uri="memory://pressure",
                name="Memory Pressure",
                description="Current pressure signals, tier, and token utilization",
                mimeType="application/json",
            ),
        ]

    @server.read_resource()
    async def read_resource(uri: str) -> str:
        if uri == "memory://status":
            return json.dumps(bus.status(), indent=2, default=str)
        elif uri == "memory://pressure":
            return json.dumps(bus.pressure(), indent=2, default=str)
        else:
            return json.dumps({"error": f"Unknown resource: {uri}"})

    return server
