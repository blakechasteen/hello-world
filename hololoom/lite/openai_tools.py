"""
HoloLoom Lite OpenAI Function Calling Tools
============================================

Defines HoloLoom Lite's 6-method API as OpenAI-compatible function tools.
These schemas work with any LLM that supports OpenAI's function calling format:
- OpenAI GPT-4/3.5
- Anthropic Claude (via tool_use)
- Local models via llama.cpp, Ollama, etc.
- LangChain, LlamaIndex, and other frameworks

Usage:
    from hololoom.lite.openai_tools import HOLOLOOM_LITE_TOOLS, execute_tool

    # Pass tools to OpenAI
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[...],
        tools=HOLOLOOM_LITE_TOOLS
    )

    # Execute tool call
    if response.choices[0].message.tool_calls:
        tool_call = response.choices[0].message.tool_calls[0]
        result = await execute_tool(loom, tool_call.function.name, tool_call.function.arguments)

Author: Claude Code
Date: December 2025
"""

import json
from typing import Any

# ============================================================================
# OpenAI Function Schemas for HoloLoom Lite's 6 Methods
# ============================================================================

HOLOLOOM_LITE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "loom_experience",
            "description": (
                "Store a memory in HoloLoom. The simplest way to add knowledge. "
                "Memories are automatically indexed for semantic search and graph traversal. "
                "Use this to remember facts, observations, or any text content."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The content to remember (any text)"
                    },
                    "context": {
                        "type": "object",
                        "description": "Optional context metadata (source, tags, timestamp, etc.)",
                        "additionalProperties": True
                    }
                },
                "required": ["text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "loom_recall",
            "description": (
                "Search memories by semantic similarity. Returns memories most relevant "
                "to your query, ranked by relevance. Use this to find information you've "
                "previously stored."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search for (natural language)"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum memories to return (default: 5, max: 50)",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 50
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "loom_reflect",
            "description": (
                "Learn from feedback on memories. Strengthens good memories and weakens "
                "poor ones. Use this after evaluating whether recalled memories were helpful."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "memory_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "IDs of memories to provide feedback on"
                    },
                    "feedback": {
                        "type": "object",
                        "description": "Feedback object (e.g., {helpful: true, rating: 0.9, reason: '...'})",
                        "properties": {
                            "helpful": {"type": "boolean"},
                            "rating": {"type": "number", "minimum": 0, "maximum": 1},
                            "reason": {"type": "string"}
                        }
                    }
                },
                "required": ["memory_ids", "feedback"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "loom_reason",
            "description": (
                "Multi-step reasoning over memories. Modes: "
                "'direct' (single pass, fast), "
                "'verify' (with verification, medium), "
                "'research' (multi-query exploration, thorough). "
                "Use for complex questions that need deeper analysis."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The question to reason about"
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["direct", "verify", "research"],
                        "description": "Reasoning mode (default: direct)",
                        "default": "direct"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "loom_query",
            "description": (
                "One-shot Q&A. The simplest way to ask HoloLoom a question. "
                "Searches memories and returns a synthesized answer. "
                "Use for quick factual queries."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Your question"
                    }
                },
                "required": ["text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "loom_related",
            "description": (
                "Navigate the memory graph. Find memories connected to a given memory "
                "via semantic or temporal edges. Use to explore related concepts and "
                "discover knowledge connections."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "memory_id": {
                        "type": "string",
                        "description": "Starting memory ID"
                    },
                    "hops": {
                        "type": "integer",
                        "description": "How many hops to traverse (1=direct neighbors, default: 1)",
                        "default": 1,
                        "minimum": 1,
                        "maximum": 5
                    },
                    "direction": {
                        "type": "string",
                        "enum": ["out", "in", "both"],
                        "description": "Edge direction to follow (default: both)",
                        "default": "both"
                    }
                },
                "required": ["memory_id"]
            }
        }
    }
]


# ============================================================================
# Tool Execution Helper
# ============================================================================

async def execute_tool(
    loom,
    tool_name: str,
    arguments: str | dict[str, Any]
) -> dict[str, Any]:
    """
    Execute a HoloLoom Lite tool call.

    Args:
        loom: HoloLoomLite instance
        tool_name: Name of the tool to execute
        arguments: Tool arguments (JSON string or dict)

    Returns:
        Tool result as a dictionary

    Example:
        result = await execute_tool(loom, "loom_recall", '{"query": "Thompson Sampling"}')
    """
    # Parse arguments if string
    if isinstance(arguments, str):
        arguments = json.loads(arguments)

    if tool_name == "loom_experience":
        text = arguments.get("text")
        context = arguments.get("context", {})

        if not text:
            return {"error": "'text' is required"}

        memory = await loom.experience(text, context=context)
        return {
            "success": True,
            "memory_id": memory.id,
            "text": text[:100] + "..." if len(text) > 100 else text
        }

    elif tool_name == "loom_recall":
        query = arguments.get("query")
        limit = arguments.get("limit", 5)

        if not query:
            return {"error": "'query' is required"}

        memories = await loom.recall(query, limit=limit)

        return {
            "success": True,
            "count": len(memories),
            "memories": [
                {
                    "id": mem.id,
                    "text": mem.text[:200] + "..." if len(mem.text) > 200 else mem.text,
                    "relevance": getattr(mem, 'relevance', None)
                }
                for mem in memories
            ]
        }

    elif tool_name == "loom_reflect":
        memory_ids = arguments.get("memory_ids", [])
        feedback = arguments.get("feedback", {})

        if not memory_ids:
            return {"error": "'memory_ids' is required"}

        # Recall memories by IDs (simplified - search for them)
        memories = await loom.recall(" ".join(memory_ids), limit=len(memory_ids))

        await loom.reflect(memories, feedback)

        return {
            "success": True,
            "reflected_count": len(memories),
            "feedback": feedback
        }

    elif tool_name == "loom_reason":
        query = arguments.get("query")
        mode = arguments.get("mode", "direct")

        if not query:
            return {"error": "'query' is required"}

        result = await loom.reason(query, mode=mode)

        return {
            "success": True,
            "mode": mode,
            "confidence": result.confidence,
            "response": result.response
        }

    elif tool_name == "loom_query":
        text = arguments.get("text")

        if not text:
            return {"error": "'text' is required"}

        result = await loom.query(text)

        return {
            "success": True,
            "confidence": result.confidence,
            "response": result.response
        }

    elif tool_name == "loom_related":
        memory_id = arguments.get("memory_id")
        hops = arguments.get("hops", 1)
        direction = arguments.get("direction", "both")

        if not memory_id:
            return {"error": "'memory_id' is required"}

        related = await loom.related(memory_id, hops=hops, direction=direction)

        return {
            "success": True,
            "count": len(related),
            "related_memories": [
                {
                    "id": mem.id,
                    "text": mem.text[:200] + "..." if len(mem.text) > 200 else mem.text
                }
                for mem in related
            ]
        }

    else:
        return {"error": f"Unknown tool: {tool_name}"}


# ============================================================================
# Convenience Functions
# ============================================================================

def get_tools_for_openai() -> list[dict[str, Any]]:
    """Get tools in OpenAI format."""
    return HOLOLOOM_LITE_TOOLS


def get_tools_for_anthropic() -> list[dict[str, Any]]:
    """
    Get tools in Anthropic Claude format.

    Anthropic uses a slightly different schema format.
    """
    return [
        {
            "name": tool["function"]["name"],
            "description": tool["function"]["description"],
            "input_schema": tool["function"]["parameters"]
        }
        for tool in HOLOLOOM_LITE_TOOLS
    ]


def get_tool_names() -> list[str]:
    """Get list of tool names."""
    return [tool["function"]["name"] for tool in HOLOLOOM_LITE_TOOLS]


def get_tool_descriptions() -> dict[str, str]:
    """Get tool name -> description mapping."""
    return {
        tool["function"]["name"]: tool["function"]["description"]
        for tool in HOLOLOOM_LITE_TOOLS
    }


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    import asyncio

    async def demo():
        """Demonstrate tool schemas."""
        print("HoloLoom Lite OpenAI Function Tools")
        print("=" * 50)
        print()

        print("Available Tools:")
        for name, desc in get_tool_descriptions().items():
            print(f"\n  {name}:")
            print(f"    {desc[:80]}...")

        print("\n" + "=" * 50)
        print("\nOpenAI Format (first tool):")
        print(json.dumps(HOLOLOOM_LITE_TOOLS[0], indent=2))

        print("\n" + "=" * 50)
        print("\nAnthropic Format (first tool):")
        print(json.dumps(get_tools_for_anthropic()[0], indent=2))

    asyncio.run(demo())
