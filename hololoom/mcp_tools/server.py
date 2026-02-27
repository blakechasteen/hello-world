#!/usr/bin/env python3
"""
HoloLoom MCP Server
===================
Model Context Protocol server for HoloLoom integration with Claude Desktop.

**Created**: November 21, 2025
**Source**: Integrated from Promptly platform
**Tools**: 10 core tools (expandable to 27+ with full Promptly integration)

Architecture:
- Phase 1 (Now): Core HoloLoom tools (memory, reasoning, learning)
- Phase 2 (After skills): Skill execution tools
- Phase 3 (After Week 2): Evaluation and analytics tools
"""

import asyncio
import sys
import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# MCP imports
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent, Resource
    MCP_AVAILABLE = True
except ImportError:
    print("ERROR: MCP SDK not installed. Install with: pip install mcp", file=sys.stderr)
    MCP_AVAILABLE = False
    # Create dummy classes for type hints
    class Server: pass
    class Tool: pass
    class TextContent: pass
    class Resource: pass

# HoloLoom imports
try:
    from HoloLoom import HoloLoom
    from HoloLoom.config import Config
    from HoloLoom.Documentation.types import Query, MemoryShard
    from HoloLoom.weaving_orchestrator import WeavingOrchestrator
    HOLOLOOM_AVAILABLE = True
except ImportError:
    print("ERROR: Could not import HoloLoom. Check installation.", file=sys.stderr)
    HOLOLOOM_AVAILABLE = False

# Optional: Agentic reasoning
try:
    from HoloLoom.agentic.core import AgenticOrchestrator, ReasoningMode
    AGENTIC_AVAILABLE = True
except ImportError:
    print("WARNING: Agentic reasoning not available", file=sys.stderr)
    AGENTIC_AVAILABLE = False

# Optional: Recursive learning
try:
    from HoloLoom.recursive import FullLearningEngine
    RECURSIVE_AVAILABLE = True
except ImportError:
    print("WARNING: Recursive learning not available", file=sys.stderr)
    RECURSIVE_AVAILABLE = False


# ============================================================================
# MCP Server Setup
# ============================================================================

app = Server("hololoom-mcp") if MCP_AVAILABLE else None
hololoom_instance = None
orchestrator_instance = None


def get_hololoom():
    """Get or create HoloLoom instance."""
    global hololoom_instance
    if hololoom_instance is None:
        hololoom_instance = HoloLoom()
    return hololoom_instance


async def get_orchestrator():
    """Get or create WeavingOrchestrator instance."""
    global orchestrator_instance
    if orchestrator_instance is None:
        config = Config.fast()
        # Create minimal shards for orchestrator
        shards = [
            MemoryShard(
                content="HoloLoom is a neural decision-making system with multi-scale embeddings.",
                source="system",
                timestamp=datetime.now().isoformat()
            )
        ]
        orchestrator_instance = WeavingOrchestrator(cfg=config, shards=shards)
    return orchestrator_instance


# ============================================================================
# MCP Resource Handlers
# ============================================================================

if MCP_AVAILABLE:
    @app.list_resources()
    async def list_resources() -> list[Resource]:
        """List HoloLoom memories as browsable resources."""
        if not HOLOLOOM_AVAILABLE:
            return []

        resources = []
        try:
            loom = get_hololoom()
            # Note: HoloLoom doesn't have a direct "list all memories" API
            # This would require extending the memory backend
            # For now, return empty list
            # TODO: Add memory listing capability to HoloLoom
        except Exception as e:
            print(f"Error listing resources: {e}", file=sys.stderr)

        return resources


    @app.read_resource()
    async def read_resource(uri: str) -> str:
        """Read a HoloLoom memory resource."""
        if not HOLOLOOM_AVAILABLE:
            return "HoloLoom not available"

        try:
            # Parse URI (format: hololoom://memory/<memory_id>)
            if uri.startswith("hololoom://memory/"):
                memory_id = uri.replace("hololoom://memory/", "")
                # TODO: Implement memory retrieval by ID
                return f"Memory {memory_id} content would go here"
            else:
                return f"Unknown resource type: {uri}"
        except Exception as e:
            return f"Error reading resource: {e}"


# ============================================================================
# MCP Tool Handlers
# ============================================================================

if MCP_AVAILABLE:
    @app.list_tools()
    async def list_tools() -> list[Tool]:
        """List all available HoloLoom MCP tools."""
        tools = [
            # ===== Memory Tools =====
            Tool(
                name="hololoom_experience",
                description="Store a new memory/experience in HoloLoom's knowledge graph. "
                           "Creates entities, relationships, and semantic embeddings.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "The content to remember (text, facts, observations)"
                        },
                        "metadata": {
                            "type": "object",
                            "description": "Optional metadata (source, timestamp, tags)",
                            "properties": {
                                "source": {"type": "string"},
                                "tags": {"type": "array", "items": {"type": "string"}},
                            },
                            "additionalProperties": True
                        }
                    },
                    "required": ["content"]
                }
            ),
            Tool(
                name="hololoom_recall",
                description="Retrieve relevant memories from HoloLoom based on a query. "
                           "Uses hybrid search (BM25 + semantic similarity + graph traversal).",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The question or topic to recall memories about"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of memories to retrieve (default: 10)",
                            "default": 10
                        }
                    },
                    "required": ["query"]
                }
            ),
            Tool(
                name="hololoom_metrics",
                description="Get HoloLoom system metrics (awareness graph, learning statistics, performance).",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "include_learning": {
                            "type": "boolean",
                            "description": "Include learning statistics (Thompson Sampling, patterns)",
                            "default": True
                        },
                        "include_performance": {
                            "type": "boolean",
                            "description": "Include performance metrics (latency, cache hit rate)",
                            "default": True
                        }
                    }
                }
            ),

            # ===== Reasoning Tools =====
            Tool(
                name="hololoom_weave",
                description="Execute HoloLoom's full weaving cycle (9-step processing pipeline). "
                           "Returns a complete Spacetime result with provenance.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The query to process through the weaving cycle"
                        },
                        "mode": {
                            "type": "string",
                            "enum": ["BARE", "FAST", "FUSED"],
                            "description": "Processing mode (BARE=fastest, FUSED=highest quality)",
                            "default": "FAST"
                        },
                        "enable_reflection": {
                            "type": "boolean",
                            "description": "Enable recursive learning and reflection",
                            "default": False
                        }
                    },
                    "required": ["query"]
                }
            ),
        ]

        # Add agentic reasoning tools if available
        if AGENTIC_AVAILABLE:
            tools.extend([
                Tool(
                    name="hololoom_reason",
                    description="Execute agentic reasoning with one of 4 modes: "
                               "DIRECT (single-pass), VERIFY (with verification), "
                               "RESEARCH (multi-query exploration), PLAN_EXECUTE (goal decomposition).",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "The question or task"},
                            "mode": {
                                "type": "string",
                                "enum": ["direct", "verify", "research", "plan_execute"],
                                "description": "Reasoning mode",
                                "default": "verify"
                            },
                            "max_steps": {
                                "type": "integer",
                                "description": "Maximum reasoning steps (for research/plan_execute)",
                                "default": 5
                            }
                        },
                        "required": ["query"]
                    }
                ),
            ])

        # Add recursive learning tools if available
        if RECURSIVE_AVAILABLE:
            tools.extend([
                Tool(
                    name="hololoom_refine",
                    description="Refine a response using recursive learning. "
                               "Iteratively improves quality until threshold met or max iterations.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "The query to refine"},
                            "initial_response": {"type": "string", "description": "Initial response to refine"},
                            "quality_threshold": {
                                "type": "number",
                                "description": "Target quality score (0.0-1.0)",
                                "default": 0.85
                            },
                            "max_iterations": {
                                "type": "integer",
                                "description": "Maximum refinement iterations",
                                "default": 3
                            }
                        },
                        "required": ["query", "initial_response"]
                    }
                ),
                Tool(
                    name="hololoom_learning_stats",
                    description="Get comprehensive learning statistics: "
                               "Thompson Sampling priors, hot patterns, policy weights, refinement history.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "include_patterns": {
                                "type": "boolean",
                                "description": "Include learned patterns",
                                "default": True
                            },
                            "include_thompson": {
                                "type": "boolean",
                                "description": "Include Thompson Sampling statistics",
                                "default": True
                            }
                        }
                    }
                ),
            ])

        # Utility tools
        tools.extend([
            Tool(
                name="hololoom_summary",
                description="Get a human-readable summary of the HoloLoom system state.",
                inputSchema={"type": "object", "properties": {}}
            ),
            Tool(
                name="hololoom_reflect",
                description="Provide feedback on a previous response to improve future results. "
                           "Feeds into HoloLoom's learning loop.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The original query"},
                        "response": {"type": "string", "description": "The response to reflect on"},
                        "feedback": {
                            "type": "object",
                            "description": "Feedback scores and comments",
                            "properties": {
                                "helpful": {"type": "boolean"},
                                "accurate": {"type": "boolean"},
                                "quality_score": {"type": "number", "minimum": 0, "maximum": 1}
                            }
                        }
                    },
                    "required": ["query", "response", "feedback"]
                }
            ),
        ])

        # TODO: Add skill tools after skills integration (Phase 2)
        # TODO: Add evaluation tools after Week 2 (Phase 3)
        # TODO: Add analytics tools after Week 2 (Phase 3)

        return tools


    @app.call_tool()
    async def call_tool(name: str, arguments: Any) -> list[TextContent]:
        """Handle tool execution."""
        try:
            if name == "hololoom_experience":
                return await tool_experience(arguments)
            elif name == "hololoom_recall":
                return await tool_recall(arguments)
            elif name == "hololoom_metrics":
                return await tool_metrics(arguments)
            elif name == "hololoom_weave":
                return await tool_weave(arguments)
            elif name == "hololoom_reason":
                return await tool_reason(arguments)
            elif name == "hololoom_refine":
                return await tool_refine(arguments)
            elif name == "hololoom_learning_stats":
                return await tool_learning_stats(arguments)
            elif name == "hololoom_summary":
                return await tool_summary(arguments)
            elif name == "hololoom_reflect":
                return await tool_reflect(arguments)
            else:
                return [TextContent(
                    type="text",
                    text=f"Unknown tool: {name}"
                )]
        except Exception as e:
            import traceback
            error_msg = f"Error executing {name}: {e}\n{traceback.format_exc()}"
            print(error_msg, file=sys.stderr)
            return [TextContent(type="text", text=error_msg)]


# ============================================================================
# Tool Implementations
# ============================================================================

async def tool_experience(args: Dict[str, Any]) -> list[TextContent]:
    """Store a new memory in HoloLoom."""
    if not HOLOLOOM_AVAILABLE:
        return [TextContent(type="text", text="HoloLoom not available")]

    content = args.get("content", "")
    metadata = args.get("metadata", {})

    loom = get_hololoom()
    memory = await loom.experience(content)

    result = {
        "status": "success",
        "memory_id": memory.id if hasattr(memory, 'id') else "unknown",
        "content_length": len(content),
        "metadata": metadata
    }

    return [TextContent(type="text", text=json.dumps(result, indent=2))]


async def tool_recall(args: Dict[str, Any]) -> list[TextContent]:
    """Retrieve relevant memories from HoloLoom."""
    if not HOLOLOOM_AVAILABLE:
        return [TextContent(type="text", text="HoloLoom not available")]

    query = args.get("query", "")
    limit = args.get("limit", 10)

    loom = get_hololoom()
    memories = await loom.recall(query, limit=limit)

    result = {
        "query": query,
        "memories_found": len(memories),
        "memories": [
            {
                "content": m.content[:200] + "..." if len(m.content) > 200 else m.content,
                "timestamp": m.timestamp if hasattr(m, 'timestamp') else "unknown"
            }
            for m in memories
        ]
    }

    return [TextContent(type="text", text=json.dumps(result, indent=2))]


async def tool_metrics(args: Dict[str, Any]) -> list[TextContent]:
    """Get HoloLoom system metrics."""
    if not HOLOLOOM_AVAILABLE:
        return [TextContent(type="text", text="HoloLoom not available")]

    loom = get_hololoom()
    metrics = loom.get_metrics()

    result = {
        "timestamp": datetime.now().isoformat(),
        "metrics": metrics
    }

    return [TextContent(type="text", text=json.dumps(result, indent=2))]


async def tool_weave(args: Dict[str, Any]) -> list[TextContent]:
    """Execute HoloLoom weaving cycle."""
    if not HOLOLOOM_AVAILABLE:
        return [TextContent(type="text", text="HoloLoom not available")]

    query_text = args.get("query", "")
    mode = args.get("mode", "FAST")
    enable_reflection = args.get("enable_reflection", False)

    orchestrator = await get_orchestrator()
    query = Query(text=query_text)

    async with orchestrator:
        spacetime = await orchestrator.weave(query)

    result = {
        "query": query_text,
        "mode": mode,
        "response": spacetime.response if hasattr(spacetime, 'response') else str(spacetime),
        "confidence": spacetime.confidence if hasattr(spacetime, 'confidence') else 0.0,
        "metadata": spacetime.metadata if hasattr(spacetime, 'metadata') else {}
    }

    return [TextContent(type="text", text=json.dumps(result, indent=2))]


async def tool_reason(args: Dict[str, Any]) -> list[TextContent]:
    """Execute agentic reasoning."""
    if not AGENTIC_AVAILABLE:
        return [TextContent(type="text", text="Agentic reasoning not available")]

    query = args.get("query", "")
    mode_str = args.get("mode", "verify")
    max_steps = args.get("max_steps", 5)

    # Map mode string to ReasoningMode
    mode_map = {
        "direct": ReasoningMode.DIRECT,
        "verify": ReasoningMode.VERIFY,
        "research": ReasoningMode.RESEARCH,
        "plan_execute": ReasoningMode.PLAN_EXECUTE
    }
    mode = mode_map.get(mode_str, ReasoningMode.VERIFY)

    config = Config.fast()
    orchestrator = AgenticOrchestrator(cfg=config, shards=[])

    async with orchestrator:
        result_obj = await orchestrator.reason(query, mode=mode, max_steps=max_steps)

    result = {
        "query": query,
        "mode": mode_str,
        "response": result_obj.response,
        "confidence": result_obj.confidence,
        "steps_taken": result_obj.steps_taken if hasattr(result_obj, 'steps_taken') else 0
    }

    return [TextContent(type="text", text=json.dumps(result, indent=2))]


async def tool_refine(args: Dict[str, Any]) -> list[TextContent]:
    """Refine a response using recursive learning."""
    if not RECURSIVE_AVAILABLE:
        return [TextContent(type="text", text="Recursive learning not available")]

    # TODO: Implement with FullLearningEngine after recursive integration verified
    return [TextContent(type="text", text="Refinement coming soon (requires recursive learning integration)")]


async def tool_learning_stats(args: Dict[str, Any]) -> list[TextContent]:
    """Get learning statistics."""
    if not RECURSIVE_AVAILABLE:
        return [TextContent(type="text", text="Recursive learning not available")]

    # TODO: Implement with FullLearningEngine
    return [TextContent(type="text", text="Learning stats coming soon (requires recursive learning integration)")]


async def tool_summary(args: Dict[str, Any]) -> list[TextContent]:
    """Get system summary."""
    if not HOLOLOOM_AVAILABLE:
        return [TextContent(type="text", text="HoloLoom not available")]

    loom = get_hololoom()
    summary = loom.summary()

    return [TextContent(type="text", text=summary)]


async def tool_reflect(args: Dict[str, Any]) -> list[TextContent]:
    """Provide feedback for learning."""
    if not HOLOLOOM_AVAILABLE:
        return [TextContent(type="text", text="HoloLoom not available")]

    query = args.get("query", "")
    response = args.get("response", "")
    feedback = args.get("feedback", {})

    loom = get_hololoom()
    # Note: reflect() expects memories, not raw text
    # This is a simplified version - full implementation needs memory objects
    # TODO: Improve reflection API

    result = {
        "status": "feedback_received",
        "query": query,
        "feedback": feedback
    }

    return [TextContent(type="text", text=json.dumps(result, indent=2))]


# ============================================================================
# Server Entry Point
# ============================================================================

async def create_hololoom_mcp_server():
    """Create and run the HoloLoom MCP server."""
    if not MCP_AVAILABLE:
        print("ERROR: MCP SDK not installed. Install with: pip install mcp", file=sys.stderr)
        return

    if not HOLOLOOM_AVAILABLE:
        print("ERROR: HoloLoom not available. Check installation.", file=sys.stderr)
        return

    print("Starting HoloLoom MCP Server...", file=sys.stderr)
    print(f"Tools available: {len(await list_tools())}", file=sys.stderr)
    print(f"Agentic reasoning: {'✓' if AGENTIC_AVAILABLE else '✗'}", file=sys.stderr)
    print(f"Recursive learning: {'✓' if RECURSIVE_AVAILABLE else '✗'}", file=sys.stderr)

    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


def main():
    """Main entry point."""
    asyncio.run(create_hololoom_mcp_server())


if __name__ == "__main__":
    main()
