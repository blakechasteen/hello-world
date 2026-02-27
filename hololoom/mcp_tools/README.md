# HoloLoom MCP Tools

**Status**: Production Ready (Phase 1 Complete - November 21, 2025)
**Integration**: From Promptly platform
**Tools**: 10 core tools (expandable to 27+ with full Promptly integration)

---

## Overview

Model Context Protocol (MCP) server that exposes HoloLoom's memory, reasoning, and learning capabilities to Claude Desktop and other MCP clients.

**Architecture**:
- **Phase 1** (Current): Core HoloLoom tools (memory, reasoning, learning)
- **Phase 2** (After skills integration): Skill execution tools
- **Phase 3** (After Week 2 evaluation): A/B testing, LLM-as-judge, analytics tools

---

## Installation

### 1. Install MCP SDK

```bash
pip install mcp
```

### 2. Configure Claude Desktop

Add to your Claude Desktop config file:

**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "hololoom": {
      "command": "python",
      "args": [
        "-m",
        "hololoom.mcp_tools.server"
      ],
      "env": {
        "PYTHONPATH": "/path/to/mythRL"
      }
    }
  }
}
```

### 3. Restart Claude Desktop

The HoloLoom MCP tools will now be available in Claude Desktop!

---

## Available Tools (Phase 1)

### Memory Tools

#### `hololoom_experience`
Store a new memory/experience in HoloLoom's knowledge graph.

**Parameters**:
- `content` (string, required): The content to remember
- `metadata` (object, optional): Source, timestamp, tags

**Example**:
```json
{
  "content": "Thompson Sampling balances exploration and exploitation using Bayesian priors",
  "metadata": {
    "source": "research_paper",
    "tags": ["machine_learning", "bandit_algorithms"]
  }
}
```

**Returns**: Memory ID and confirmation

---

#### `hololoom_recall`
Retrieve relevant memories from hololoom based on a query.

**Parameters**:
- `query` (string, required): The question or topic to recall
- `limit` (integer, optional): Maximum memories to retrieve (default: 10)

**Example**:
```json
{
  "query": "What did I learn about Thompson Sampling?",
  "limit": 5
}
```

**Returns**: List of relevant memories with content and timestamps

---

#### `hololoom_metrics`
Get HoloLoom system metrics (awareness graph, learning statistics, performance).

**Parameters**:
- `include_learning` (boolean, optional): Include learning stats (default: true)
- `include_performance` (boolean, optional): Include performance metrics (default: true)

**Returns**: Complete system metrics including activation levels, coherence, learning progress

---

### Reasoning Tools

#### `hololoom_weave`
Execute HoloLoom's full weaving cycle (9-step processing pipeline).

**Parameters**:
- `query` (string, required): The query to process
- `mode` (enum, optional): `BARE`, `FAST`, or `FUSED` (default: `FAST`)
- `enable_reflection` (boolean, optional): Enable recursive learning (default: false)

**Example**:
```json
{
  "query": "Explain the tradeoffs between Thompson Sampling and epsilon-greedy",
  "mode": "FUSED",
  "enable_reflection": true
}
```

**Returns**: Spacetime result with response, confidence, and provenance

---

#### `hololoom_reason` (if agentic available)
Execute agentic reasoning with one of 4 modes.

**Parameters**:
- `query` (string, required): The question or task
- `mode` (enum, optional): `direct`, `verify`, `research`, `plan_execute` (default: `verify`)
- `max_steps` (integer, optional): Maximum reasoning steps (default: 5)

**Modes**:
- **direct**: Single-pass answer (~150ms)
- **verify**: Answer + verification (~600ms)
- **research**: Multi-query exploration (~900ms)
- **plan_execute**: Goal decomposition (~750ms)

**Example**:
```json
{
  "query": "What are all the tradeoffs of Thompson Sampling?",
  "mode": "research",
  "max_steps": 7
}
```

**Returns**: Response, confidence, verification results, steps taken

---

### Learning Tools

#### `hololoom_refine` (if recursive available)
Refine a response using recursive learning.

**Parameters**:
- `query` (string, required): The query to refine
- `initial_response` (string, required): Initial response to improve
- `quality_threshold` (number, optional): Target quality 0.0-1.0 (default: 0.85)
- `max_iterations` (integer, optional): Maximum iterations (default: 3)

**Returns**: Refined response, quality progression, iterations taken

---

#### `hololoom_learning_stats` (if recursive available)
Get comprehensive learning statistics.

**Parameters**:
- `include_patterns` (boolean, optional): Include learned patterns (default: true)
- `include_thompson` (boolean, optional): Include Thompson Sampling stats (default: true)

**Returns**: Thompson Sampling priors, hot patterns, policy weights, refinement history

---

### Utility Tools

#### `hololoom_summary`
Get a human-readable summary of the HoloLoom system state.

**Parameters**: None

**Returns**: Complete system summary with active memories, learning progress, performance

---

#### `hololoom_reflect`
Provide feedback on a previous response to improve future results.

**Parameters**:
- `query` (string, required): The original query
- `response` (string, required): The response to reflect on
- `feedback` (object, required): Feedback scores
  - `helpful` (boolean)
  - `accurate` (boolean)
  - `quality_score` (number, 0.0-1.0)

**Example**:
```json
{
  "query": "Explain Thompson Sampling",
  "response": "Thompson Sampling is a Bayesian approach...",
  "feedback": {
    "helpful": true,
    "accurate": true,
    "quality_score": 0.92
  }
}
```

**Returns**: Feedback confirmation

---

## Coming Soon (Phase 2 & 3)

### Skill Tools (After Skills Integration)
- `hololoom_skill_execute`: Execute a skill template
- `hololoom_skill_list`: List available skills
- `hololoom_skill_create`: Create custom skill

### Evaluation Tools (After Week 2)
- `hololoom_ab_test`: Run A/B test on strategies
- `hololoom_llm_judge`: Evaluate with LLM-as-judge
- `hololoom_cost_estimate`: Estimate API costs

### Analytics Tools (After Week 2)
- `hololoom_analytics_summary`: Overall analytics
- `hololoom_analytics_query_stats`: Query performance stats
- `hololoom_analytics_recommendations`: AI recommendations

---

## Usage Examples

### Example 1: Store and Recall Knowledge

```python
# In Claude Desktop, use the MCP tools:

# 1. Store a fact
hololoom_experience({
  "content": "HoloLoom uses Thompson Sampling for exploration/exploitation balance"
})

# 2. Recall it later
hololoom_recall({
  "query": "How does HoloLoom handle exploration?",
  "limit": 3
})
```

### Example 2: Complex Reasoning

```python
# Research mode for comprehensive analysis
hololoom_reason({
  "query": "What are all the design decisions in HoloLoom's architecture?",
  "mode": "research",
  "max_steps": 10
})
```

### Example 3: Iterative Refinement

```python
# 1. Get initial answer
result = hololoom_weave({
  "query": "Explain the weaving cycle"
})

# 2. Refine for higher quality
hololoom_refine({
  "query": "Explain the weaving cycle",
  "initial_response": result.response,
  "quality_threshold": 0.95,
  "max_iterations": 5
})
```

---

## Architecture

```
Claude Desktop
    ↓ MCP Protocol
HoloLoom MCP Server
    ├─ Memory Tools
    │  ├─ experience() → hololoom.experience()
    │  ├─ recall() → hololoom.recall()
    │  └─ metrics() → hololoom.get_metrics()
    │
    ├─ Reasoning Tools
    │  ├─ weave() → WeavingOrchestrator.weave()
    │  └─ reason() → AgenticOrchestrator.reason()
    │
    └─ Learning Tools
       ├─ refine() → FullLearningEngine.refine()
       └─ learning_stats() → FullLearningEngine.get_statistics()
```

---

## Performance

| Tool | Latency | Notes |
|------|---------|-------|
| `experience` | ~50ms | Memory storage |
| `recall` | ~150ms | Hybrid search (BM25 + semantic + graph) |
| `weave` (FAST) | ~150ms | Standard weaving cycle |
| `weave` (FUSED) | ~300ms | Full quality processing |
| `reason` (direct) | ~150ms | Single-pass |
| `reason` (research) | ~900ms | Multi-query exploration |
| `refine` | ~450ms | 3 iterations average |

---

## Testing

Run the test suite:

```bash
# Unit tests
pytest hololoom/mcp_tools/tests/test_mcp_tools.py -v

# Integration test with Claude Desktop
python demos/demo_mcp_tools.py
```

---

## Troubleshooting

### "MCP SDK not installed"
```bash
pip install mcp
```

### "HoloLoom not available"
Ensure PYTHONPATH includes the mythRL directory:
```bash
export PYTHONPATH=/path/to/mythRL:$PYTHONPATH
```

### "Agentic reasoning not available"
This is optional. Install with:
```bash
# Agentic reasoning is part of HoloLoom
# Ensure hololoom/agentic/ directory exists
```

### Tools not showing in Claude Desktop
1. Check config file syntax (valid JSON)
2. Restart Claude Desktop
3. Check logs: `~/Library/Logs/Claude/` (macOS)

---

## API Reference

See [HoloLoom API Documentation](../../VISUAL_QUICK_START.md) for complete HoloLoom API reference.

---

## Development

### Adding New Tools

1. Add tool definition to `list_tools()` in `server.py`
2. Implement handler function (`async def tool_name(args) -> list[TextContent]`)
3. Add case to `call_tool()` dispatcher
4. Write tests in `tests/test_mcp_tools.py`
5. Update this README

### Testing Locally

```bash
# Run server directly
python -m hololoom.mcp_tools.server

# Test with MCP inspector
mcp inspect python -m hololoom.mcp_tools.server
```

---

## Contributing

Promptly integration is ongoing. Priority areas:

1. **Phase 2** (Week 1): Skill execution tools
2. **Phase 3** (Week 2): Evaluation and analytics tools
3. **Documentation**: More usage examples
4. **Testing**: Integration tests with real workflows

---

## License

Part of the HoloLoom project. See main repository LICENSE.

---

## Credits

**Original Source**: Promptly platform (archive/old_projects/Promptly/)
**Integration**: November 21, 2025
**Maintainer**: HoloLoom team
