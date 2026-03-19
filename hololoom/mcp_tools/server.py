#!/usr/bin/env python3
"""
HoloLoom MCP Server
===================
Model Context Protocol server for HoloLoom integration with Claude Desktop.

**Created**: November 21, 2025
**Source**: Integrated from Promptly platform
**Tools**: 9 core tools (expandable to 27+ with full Promptly integration)

Architecture:
- Phase 1 (Now): Core HoloLoom tools (memory, reasoning, learning)
- Phase 2 (After skills): Skill execution tools
- Phase 3 (After Week 2): Evaluation and analytics tools
"""

import asyncio
import json
import sys
from datetime import datetime
from typing import Any

from hololoom.mcp_tools.mcp_base import (
    MCP_AVAILABLE,
    Resource,
    create_server,
    error_content,
    json_content,
    stdio_server,
    text_content,
    tool,
)

# HoloLoom imports
try:
    from hololoom import HoloLoom
    from hololoom.config import Config
    from hololoom.Documentation.types import MemoryShard, Query
    from hololoom.weaving_orchestrator import WeavingOrchestrator
    HOLOLOOM_AVAILABLE = True
except ImportError:
    print("ERROR: Could not import HoloLoom. Check installation.", file=sys.stderr)
    HOLOLOOM_AVAILABLE = False

# Optional: Agentic reasoning
try:
    from hololoom.agentic.core import AgenticOrchestrator, ReasoningMode
    AGENTIC_AVAILABLE = True
except ImportError:
    print("WARNING: Agentic reasoning not available", file=sys.stderr)
    AGENTIC_AVAILABLE = False

# Optional: Recursive learning
try:
    from hololoom.recursive import FullLearningEngine
    RECURSIVE_AVAILABLE = True
except ImportError:
    print("WARNING: Recursive learning not available", file=sys.stderr)
    RECURSIVE_AVAILABLE = False


# ============================================================================
# MCP Server Setup
# ============================================================================

app = create_server("hololoom-mcp") if MCP_AVAILABLE else None
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
        # TODO: Add memory listing capability to HoloLoom
        return []

    @app.read_resource()
    async def read_resource(uri: str) -> str:
        """Read a HoloLoom memory resource."""
        if not HOLOLOOM_AVAILABLE:
            return "HoloLoom not available"
        if uri.startswith("hololoom://memory/"):
            memory_id = uri.replace("hololoom://memory/", "")
            return f"Memory {memory_id} content would go here"  # TODO
        return f"Unknown resource type: {uri}"


# ============================================================================
# Tool Definitions
# ============================================================================

# --- Memory Tools ---
_TOOL_EXPERIENCE = tool(
    "hololoom_experience",
    "Store a new memory/experience in HoloLoom's knowledge graph. "
    "Creates entities, relationships, and semantic embeddings.",
    {"content": {"type": "string", "description": "The content to remember"},
     "metadata": {"type": "object", "description": "Optional metadata (source, tags)",
                  "properties": {"source": {"type": "string"},
                                 "tags": {"type": "array", "items": {"type": "string"}}},
                  "additionalProperties": True}},
    required=["content"],
)

_TOOL_RECALL = tool(
    "hololoom_recall",
    "Retrieve relevant memories from HoloLoom. "
    "Uses hybrid search (BM25 + semantic similarity + graph traversal).",
    {"query": {"type": "string", "description": "The question or topic to recall"},
     "limit": {"type": "integer", "description": "Max memories to retrieve", "default": 10}},
    required=["query"],
)

_TOOL_METRICS = tool(
    "hololoom_metrics",
    "Get HoloLoom system metrics (awareness graph, learning statistics, performance).",
    {"include_learning": {"type": "boolean", "description": "Include learning stats", "default": True},
     "include_performance": {"type": "boolean", "description": "Include perf metrics", "default": True}},
)

# --- Reasoning Tools ---
_TOOL_WEAVE = tool(
    "hololoom_weave",
    "Execute HoloLoom's full weaving cycle (9-step processing pipeline). "
    "Returns a complete Spacetime result with provenance.",
    {"query": {"type": "string", "description": "The query to process"},
     "mode": {"type": "string", "enum": ["BARE", "FAST", "FUSED"],
              "description": "Processing mode (BARE=fastest, FUSED=highest quality)", "default": "FAST"},
     "enable_reflection": {"type": "boolean", "description": "Enable recursive learning", "default": False}},
    required=["query"],
)

_TOOL_REASON = tool(
    "hololoom_reason",
    "Execute agentic reasoning. Modes: DIRECT (single-pass), VERIFY (with verification), "
    "RESEARCH (multi-query exploration), PLAN_EXECUTE (goal decomposition).",
    {"query": {"type": "string", "description": "The question or task"},
     "mode": {"type": "string", "enum": ["direct", "verify", "research", "plan_execute"],
              "description": "Reasoning mode", "default": "verify"},
     "max_steps": {"type": "integer", "description": "Max reasoning steps", "default": 5}},
    required=["query"],
)

# --- Learning Tools ---
_TOOL_REFINE = tool(
    "hololoom_refine",
    "Refine a response using recursive learning. "
    "Iteratively improves quality until threshold met or max iterations.",
    {"query": {"type": "string", "description": "The query to refine"},
     "initial_response": {"type": "string", "description": "Initial response to refine"},
     "quality_threshold": {"type": "number", "description": "Target quality (0.0-1.0)", "default": 0.85},
     "max_iterations": {"type": "integer", "description": "Max refinement iterations", "default": 3}},
    required=["query", "initial_response"],
)

_TOOL_LEARNING_STATS = tool(
    "hololoom_learning_stats",
    "Get comprehensive learning statistics: Thompson Sampling priors, "
    "hot patterns, policy weights, refinement history.",
    {"include_patterns": {"type": "boolean", "description": "Include learned patterns", "default": True},
     "include_thompson": {"type": "boolean", "description": "Include Thompson Sampling stats", "default": True}},
)

# --- Utility Tools ---
_TOOL_SUMMARY = tool(
    "hololoom_summary",
    "Get a human-readable summary of the HoloLoom system state.",
    {},
)

_TOOL_REFLECT = tool(
    "hololoom_reflect",
    "Provide feedback on a previous response to improve future results. "
    "Feeds into HoloLoom's learning loop.",
    {"query": {"type": "string", "description": "The original query"},
     "response": {"type": "string", "description": "The response to reflect on"},
     "feedback": {"type": "object", "description": "Feedback scores and comments",
                  "properties": {"helpful": {"type": "boolean"},
                                 "accurate": {"type": "boolean"},
                                 "quality_score": {"type": "number", "minimum": 0, "maximum": 1}}}},
    required=["query", "response", "feedback"],
)


# ============================================================================
# Tool Registration & Dispatch
# ============================================================================

if MCP_AVAILABLE:
    @app.list_tools()
    async def list_tools() -> list:
        """List all available HoloLoom MCP tools."""
        tools = [_TOOL_EXPERIENCE, _TOOL_RECALL, _TOOL_METRICS, _TOOL_WEAVE]
        if AGENTIC_AVAILABLE:
            tools.append(_TOOL_REASON)
        if RECURSIVE_AVAILABLE:
            tools.extend([_TOOL_REFINE, _TOOL_LEARNING_STATS])
        tools.extend([_TOOL_SUMMARY, _TOOL_REFLECT])
        return tools

    @app.call_tool()
    async def call_tool(name: str, arguments: Any) -> list:
        """Handle tool execution."""
        dispatch = {
            "hololoom_experience": tool_experience,
            "hololoom_recall": tool_recall,
            "hololoom_metrics": tool_metrics,
            "hololoom_weave": tool_weave,
            "hololoom_reason": tool_reason,
            "hololoom_refine": tool_refine,
            "hololoom_learning_stats": tool_learning_stats,
            "hololoom_summary": tool_summary,
            "hololoom_reflect": tool_reflect,
        }
        handler = dispatch.get(name)
        if not handler:
            return error_content(f"Unknown tool: {name}")
        try:
            return await handler(arguments)
        except Exception as e:
            import traceback
            print(f"Error executing {name}: {e}\n{traceback.format_exc()}", file=sys.stderr)
            return error_content(f"Error executing {name}: {e}")


# ============================================================================
# Tool Implementations
# ============================================================================

async def tool_experience(args: dict[str, Any]) -> list:
    if not HOLOLOOM_AVAILABLE:
        return text_content("HoloLoom not available")
    content = args.get("content", "")
    metadata = args.get("metadata", {})
    loom = get_hololoom()
    memory = await loom.experience(content)
    return json_content({
        "status": "success",
        "memory_id": memory.id if hasattr(memory, 'id') else "unknown",
        "content_length": len(content),
        "metadata": metadata,
    })


async def tool_recall(args: dict[str, Any]) -> list:
    if not HOLOLOOM_AVAILABLE:
        return text_content("HoloLoom not available")
    query = args.get("query", "")
    limit = args.get("limit", 10)
    loom = get_hololoom()
    memories = await loom.recall(query, limit=limit)
    return json_content({
        "query": query,
        "memories_found": len(memories),
        "memories": [
            {"content": m.content[:200] + "..." if len(m.content) > 200 else m.content,
             "timestamp": m.timestamp if hasattr(m, 'timestamp') else "unknown"}
            for m in memories
        ],
    })


async def tool_metrics(args: dict[str, Any]) -> list:
    if not HOLOLOOM_AVAILABLE:
        return text_content("HoloLoom not available")
    loom = get_hololoom()
    return json_content({"timestamp": datetime.now().isoformat(), "metrics": loom.get_metrics()})


async def tool_weave(args: dict[str, Any]) -> list:
    if not HOLOLOOM_AVAILABLE:
        return text_content("HoloLoom not available")
    query_text = args.get("query", "")
    mode = args.get("mode", "FAST")
    orchestrator = await get_orchestrator()
    query = Query(text=query_text)
    async with orchestrator:
        spacetime = await orchestrator.weave(query)
    return json_content({
        "query": query_text,
        "mode": mode,
        "response": spacetime.response if hasattr(spacetime, 'response') else str(spacetime),
        "confidence": spacetime.confidence if hasattr(spacetime, 'confidence') else 0.0,
        "metadata": spacetime.metadata if hasattr(spacetime, 'metadata') else {},
    })


async def tool_reason(args: dict[str, Any]) -> list:
    if not AGENTIC_AVAILABLE:
        return text_content("Agentic reasoning not available")
    query = args.get("query", "")
    mode_str = args.get("mode", "verify")
    max_steps = args.get("max_steps", 5)
    mode_map = {"direct": ReasoningMode.DIRECT, "verify": ReasoningMode.VERIFY,
                "research": ReasoningMode.RESEARCH, "plan_execute": ReasoningMode.PLAN_EXECUTE}
    mode = mode_map.get(mode_str, ReasoningMode.VERIFY)
    config = Config.fast()
    orchestrator = AgenticOrchestrator(cfg=config, shards=[])
    async with orchestrator:
        result_obj = await orchestrator.reason(query, mode=mode, max_steps=max_steps)
    return json_content({
        "query": query, "mode": mode_str,
        "response": result_obj.response, "confidence": result_obj.confidence,
        "steps_taken": result_obj.steps_taken if hasattr(result_obj, 'steps_taken') else 0,
    })


async def tool_refine(args: dict[str, Any]) -> list:
    if not RECURSIVE_AVAILABLE:
        return text_content("Recursive learning not available")
    # TODO: Implement with FullLearningEngine after recursive integration verified
    return text_content("Refinement coming soon (requires recursive learning integration)")


async def tool_learning_stats(args: dict[str, Any]) -> list:
    if not RECURSIVE_AVAILABLE:
        return text_content("Recursive learning not available")
    # TODO: Implement with FullLearningEngine
    return text_content("Learning stats coming soon (requires recursive learning integration)")


async def tool_summary(args: dict[str, Any]) -> list:
    if not HOLOLOOM_AVAILABLE:
        return text_content("HoloLoom not available")
    loom = get_hololoom()
    return text_content(loom.summary())


async def tool_reflect(args: dict[str, Any]) -> list:
    if not HOLOLOOM_AVAILABLE:
        return text_content("HoloLoom not available")
    query = args.get("query", "")
    feedback = args.get("feedback", {})
    # TODO: Improve reflection API — needs memory objects, not raw text
    return json_content({"status": "feedback_received", "query": query, "feedback": feedback})


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
    print(f"Agentic reasoning: {'available' if AGENTIC_AVAILABLE else 'unavailable'}", file=sys.stderr)
    print(f"Recursive learning: {'available' if RECURSIVE_AVAILABLE else 'unavailable'}", file=sys.stderr)

    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


def main():
    """Main entry point."""
    asyncio.run(create_hololoom_mcp_server())


if __name__ == "__main__":
    main()
