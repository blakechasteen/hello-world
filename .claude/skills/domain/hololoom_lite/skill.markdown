# HoloLoom Lite Agent Skill

## Metadata
- **Name**: HoloLoom Lite
- **Version**: 1.0.0
- **Author**: Claude Code
- **Date**: December 2025
- **Category**: Memory & Knowledge Management

## Description

HoloLoom Lite provides a simple 6-method API for AI agents to store, search, and reason over memories. This is the simplest interface to HoloLoom's sophisticated memory system.

### What This Skill Does

1. **Store Memories** (`experience`) - Remember any text with optional context
2. **Search Memories** (`recall`) - Find relevant memories by semantic similarity
3. **Learn from Feedback** (`reflect`) - Strengthen good memories, weaken poor ones
4. **Multi-Step Reasoning** (`reason`) - Direct, verify, or research modes
5. **Quick Q&A** (`query`) - One-shot question answering
6. **Graph Navigation** (`related`) - Find connected memories

### When to Use This Skill

- When you need to remember information across conversations
- When you need to search for previously stored knowledge
- When you need to reason over accumulated knowledge
- When exploring related concepts in a knowledge graph

## Input Schema

```json
{
  "type": "object",
  "properties": {
    "action": {
      "type": "string",
      "enum": ["experience", "recall", "reflect", "reason", "query", "related"],
      "description": "The HoloLoom Lite method to call"
    },
    "params": {
      "type": "object",
      "description": "Parameters for the action (varies by action type)",
      "properties": {
        "text": {
          "type": "string",
          "description": "Text content (for experience, query)"
        },
        "query": {
          "type": "string",
          "description": "Search query (for recall, reason)"
        },
        "context": {
          "type": "object",
          "description": "Optional context metadata (for experience)"
        },
        "limit": {
          "type": "integer",
          "description": "Max results (for recall, default: 5)"
        },
        "mode": {
          "type": "string",
          "enum": ["direct", "verify", "research"],
          "description": "Reasoning mode (for reason, default: direct)"
        },
        "memory_ids": {
          "type": "array",
          "items": {"type": "string"},
          "description": "Memory IDs (for reflect, related)"
        },
        "memory_id": {
          "type": "string",
          "description": "Single memory ID (for related)"
        },
        "feedback": {
          "type": "object",
          "description": "Feedback object (for reflect)"
        },
        "hops": {
          "type": "integer",
          "description": "Graph hops (for related, default: 1)"
        },
        "direction": {
          "type": "string",
          "enum": ["out", "in", "both"],
          "description": "Edge direction (for related, default: both)"
        }
      }
    }
  },
  "required": ["action"]
}
```

## Output Schema

```json
{
  "type": "object",
  "properties": {
    "success": {
      "type": "boolean",
      "description": "Whether the action succeeded"
    },
    "action": {
      "type": "string",
      "description": "The action that was performed"
    },
    "result": {
      "type": "object",
      "description": "Action-specific result data"
    },
    "error": {
      "type": "string",
      "description": "Error message if success=false"
    }
  }
}
```

## Examples

### Example 1: Store a Memory

**Input:**
```json
{
  "action": "experience",
  "params": {
    "text": "Thompson Sampling is a Bayesian approach to the multi-armed bandit problem that balances exploration and exploitation.",
    "context": {
      "source": "research_notes",
      "tags": ["algorithms", "machine_learning"]
    }
  }
}
```

**Output:**
```json
{
  "success": true,
  "action": "experience",
  "result": {
    "memory_id": "mem_abc123",
    "stored_text": "Thompson Sampling is a Bayesian approach..."
  }
}
```

### Example 2: Search Memories

**Input:**
```json
{
  "action": "recall",
  "params": {
    "query": "What algorithms balance exploration?",
    "limit": 3
  }
}
```

**Output:**
```json
{
  "success": true,
  "action": "recall",
  "result": {
    "count": 3,
    "memories": [
      {
        "id": "mem_abc123",
        "text": "Thompson Sampling is a Bayesian approach...",
        "relevance": 0.92
      }
    ]
  }
}
```

### Example 3: Reason Over Memories

**Input:**
```json
{
  "action": "reason",
  "params": {
    "query": "What are the tradeoffs between Thompson Sampling and UCB?",
    "mode": "research"
  }
}
```

**Output:**
```json
{
  "success": true,
  "action": "reason",
  "result": {
    "mode": "research",
    "confidence": 0.87,
    "response": "Thompson Sampling and UCB both address exploration-exploitation..."
  }
}
```

### Example 4: Quick Q&A

**Input:**
```json
{
  "action": "query",
  "params": {
    "text": "What is Thompson Sampling?"
  }
}
```

**Output:**
```json
{
  "success": true,
  "action": "query",
  "result": {
    "confidence": 0.91,
    "response": "Thompson Sampling is a Bayesian approach..."
  }
}
```

### Example 5: Find Related Memories

**Input:**
```json
{
  "action": "related",
  "params": {
    "memory_id": "mem_abc123",
    "hops": 2,
    "direction": "both"
  }
}
```

**Output:**
```json
{
  "success": true,
  "action": "related",
  "result": {
    "count": 5,
    "related_memories": [
      {
        "id": "mem_def456",
        "text": "Multi-armed bandits are a classic problem..."
      }
    ]
  }
}
```

### Example 6: Provide Feedback

**Input:**
```json
{
  "action": "reflect",
  "params": {
    "memory_ids": ["mem_abc123", "mem_def456"],
    "feedback": {
      "helpful": true,
      "rating": 0.9,
      "reason": "These memories directly answered my question"
    }
  }
}
```

**Output:**
```json
{
  "success": true,
  "action": "reflect",
  "result": {
    "reflected_count": 2,
    "feedback_applied": true
  }
}
```

## Prompt Template

When using this skill, follow this pattern:

```
To use HoloLoom Lite memory system:

1. **Store knowledge**: Use "experience" action with text and optional context
2. **Search knowledge**: Use "recall" action with a natural language query
3. **Answer questions**: Use "query" action for quick Q&A
4. **Deep reasoning**: Use "reason" action with mode (direct/verify/research)
5. **Explore connections**: Use "related" action with a memory_id
6. **Improve results**: Use "reflect" action after evaluating memories

The system automatically:
- Creates semantic embeddings for similarity search
- Builds a knowledge graph of entity relationships
- Learns from feedback to improve future retrievals
- Persists memories across sessions (if persist=True)
```

## Implementation Notes

### Python Usage

```python
from HoloLoom.lite import HoloLoomLite

async with HoloLoomLite(persist=True) as loom:
    # Store
    mem = await loom.experience("Important fact here")

    # Search
    results = await loom.recall("relevant query", limit=5)

    # Q&A
    answer = await loom.query("What do I know about X?")

    # Reasoning
    analysis = await loom.reason("Complex question", mode="research")

    # Graph navigation
    related = await loom.related(mem.id, hops=2)

    # Feedback
    await loom.reflect(results, {"helpful": True})
```

### MCP Server

For Claude Desktop, add to claude_desktop_config.json:

```json
{
  "mcpServers": {
    "hololoom-lite": {
      "command": "python",
      "args": ["-m", "HoloLoom.lite.mcp_server"],
      "cwd": "C:/path/to/mythRL"
    }
  }
}
```

### OpenAI Function Calling

```python
from HoloLoom.lite.openai_tools import HOLOLOOM_LITE_TOOLS, execute_tool

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
```

## Related Skills

- **loom** - Full HoloLoom orchestrator skill
- **spinning-wheel** - Data ingestion skill
- **prompt** - Prompt management skill