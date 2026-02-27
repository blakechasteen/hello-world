# HoloLoom Promptly MCP Server Setup

**Status**: ✅ Complete (Phase 4 - November 2025)
**Integration**: Promptly (Phases 1-3) → Claude Desktop via MCP
**Tools Exposed**: 17 (4 core + 13 professional skills)

This guide shows how to connect Claude Desktop to HoloLoom's complete Promptly integration, giving Claude access to recursive reasoning, professional skills, and memory operations.

## What You Get

Claude Desktop gains access to:

### Core HoloLoom Tools (4)

1. **hololoom_experience** - Store memories in knowledge graph
2. **hololoom_recall** - Semantic search + graph traversal
3. **hololoom_weave** - Recursive reasoning with strategy selection
4. **hololoom_analytics_summary** - Performance metrics and recommendations

### Professional Skills (13)

5. **skill_code_reviewer** - Review code with CRITIQUE strategy
6. **skill_bug_detective** - Debug with DECOMPOSE strategy
7. **skill_test_generator** - Generate tests with EXPLORE strategy
8. **skill_api_designer** - Design REST APIs with REFINE
9. **skill_documentation_writer** - Write docs with REFINE
10. **skill_performance_profiler** - Analyze performance with DECOMPOSE
11. **skill_architecture_advisor** - System design with HOFSTADTER
12. **skill_migration_planner** - Plan migrations with DECOMPOSE
13. **skill_code_explainer** - Explain code with REFINE
14. **skill_naming_consultant** - Suggest better names with CRITIQUE
15. **skill_sql_optimizer** - Optimize SQL with REFINE
16. **skill_refactoring_expert** - Refactor code with CRITIQUE
17. **skill_security_auditor** - Security audit with VERIFY

## Setup Instructions

### Step 1: Install MCP Package

```bash
pip install mcp
```

### Step 2: Locate Claude Desktop Config

**Windows**:
```
%APPDATA%\Claude\claude_desktop_config.json
```

**Mac**:
```
~/Library/Application Support/Claude/claude_desktop_config.json
```

**Linux**:
```
~/.config/Claude/claude_desktop_config.json
```

### Step 3: Add HoloLoom MCP Server

Edit `claude_desktop_config.json` and add:

```json
{
  "mcpServers": {
    "hololoom-promptly": {
      "command": "python",
      "args": [
        "-m",
        "hololoom.mcp_server_promptly"
      ],
      "env": {
        "PYTHONPATH": "/path/to/mythRL"
      }
    }
  }
}
```

**Important**: Replace `/path/to/mythRL` with the actual path to your mythRL directory.

**Example** (Windows):
```json
"PYTHONPATH": "C:\\Users\\YourName\\Documents\\mythRL"
```

**Example** (Mac/Linux):
```json
"PYTHONPATH": "/home/user/mythRL"
```

### Step 4: Restart Claude Desktop

Close and reopen Claude Desktop completely. The MCP server will start automatically.

### Step 5: Verify Connection

In Claude Desktop, try:

```
Use the hololoom_experience tool to store this memory: "HoloLoom uses recursive reasoning"
```

If successful, you'll see the tool execution and result.

## Testing the Server Standalone

Before configuring Claude Desktop, test the server:

```bash
cd /path/to/mythRL
PYTHONPATH=. python -m hololoom.mcp_server_promptly
```

You should see:
```
[2025-11-16 12:00:00] INFO: Initializing HoloLoom Promptly MCP Server...
[2025-11-16 12:00:00] INFO: Configuration: fast mode
[2025-11-16 12:00:00] INFO: Loaded 13 professional skills
[2025-11-16 12:00:00] INFO: HoloLoom Promptly MCP Server ready
[2025-11-16 12:00:00] INFO: Serving 17 MCP tools
[2025-11-16 12:00:00] INFO: Server running on stdio
```

Press Ctrl+C to stop.

## Usage Examples in Claude Desktop

### Example 1: Code Review

**You**:
```
Use skill_code_reviewer to review this Python code:

def process_data(data):
    result = []
    for item in data:
        if item > 0:
            result.append(item * 2)
    return result
```

**Claude** (uses tool):
```json
{
  "code": "def process_data(data):\n    result = []\n    for item in data:\n        if item > 0:\n            result.append(item * 2)\n    return result",
  "language": "python"
}
```

**Result**:
```json
{
  "status": "success",
  "skill": "code-reviewer",
  "output": "Overall rating: 6/10\n\nCritical issues:\n- Missing type hints\n- No input validation\n...",
  "confidence": 0.92,
  "iterations": 2,
  "strategy_used": "critique"
}
```

### Example 2: Bug Detective

**You**:
```
Use skill_bug_detective to debug this code that crashes with NullPointerException:

function getUserName(user) {
  return user.profile.name;
}
```

**Claude** (uses tool):
```json
{
  "code": "function getUserName(user) {\n  return user.profile.name;\n}",
  "bug_description": "Crashes when user has no profile",
  "language": "javascript",
  "error_message": "TypeError: Cannot read property 'name' of undefined"
}
```

**Result**:
```json
{
  "status": "success",
  "output": "Root cause: Missing null check for user.profile\n\nFixed code:\nfunction getUserName(user) {\n  return user?.profile?.name || 'Unknown';\n}\n\nTest case: ...",
  "confidence": 0.94,
  "iterations": 3
}
```

### Example 3: Recursive Weaving

**You**:
```
Use hololoom_weave to analyze: "What are the tradeoffs of Thompson Sampling?"
Use the DECOMPOSE strategy.
```

**Claude** (uses tool):
```json
{
  "query": "What are the tradeoffs of Thompson Sampling?",
  "strategy": "decompose",
  "max_iterations": 5
}
```

**Result**:
```json
{
  "status": "success",
  "response": "Thompson Sampling balances exploration and exploitation by...",
  "confidence": 0.91,
  "iterations": 3,
  "strategy_used": "decompose",
  "reasoning_journal": "Iteration 1:\n  Thought: Breaking down Thompson Sampling into components...\n  ..."
}
```

### Example 4: Memory Operations

**You**:
```
Use hololoom_experience to remember: "Matryoshka embeddings enable multi-scale retrieval"
```

**Claude** (uses tool):
```json
{
  "content": "Matryoshka embeddings enable multi-scale retrieval"
}
```

**Result**:
```json
{
  "status": "success",
  "content": "Matryoshka embeddings enable multi-scale retrieval",
  "entities": ["Matryoshka embeddings", "multi-scale retrieval"],
  "timestamp": "2025-11-16T12:00:00"
}
```

**You** (later):
```
Use hololoom_recall to find what I know about embeddings
```

**Claude** (uses tool):
```json
{
  "query": "embeddings",
  "limit": 5
}
```

**Result**: Returns stored memories about embeddings.

### Example 5: Analytics

**You**:
```
Use hololoom_analytics_summary to see performance metrics
```

**Claude** (uses tool):
```json
{}
```

**Result**:
```json
{
  "total_queries": 42,
  "avg_quality_gain": 0.087,
  "avg_iterations": 2.3,
  "total_cost": 0.45,
  "strategies": {
    "critique": {
      "count": 15,
      "avg_quality_gain": 0.092,
      "success_rate": 93.3
    },
    ...
  },
  "recommendations": [
    "Best performing strategy: critique (avg gain: 9.2%)",
    ...
  ]
}
```

## Available Recursive Reasoning Strategies

When using `hololoom_weave` or skills, you can specify:

| Strategy | Best For | Example Skills |
|----------|----------|----------------|
| **refine** | Iterative improvement | sql-optimizer, api-designer |
| **critique** | Self-critique loop | code-reviewer, refactoring-expert |
| **decompose** | Break down complex problems | bug-detective, performance-profiler |
| **explore** | Multiple approaches | test-generator |
| **verify** | Verify claims rigorously | security-auditor |
| **hofstadter** | Meta-reasoning | architecture-advisor |
| **adaptive** | Auto-select best strategy | General queries |

## Tool Parameters

### hololoom_experience

```json
{
  "content": "string (required)",
  "context": "string (optional)"
}
```

### hololoom_recall

```json
{
  "query": "string (required)",
  "limit": "integer (default: 5)"
}
```

### hololoom_weave

```json
{
  "query": "string (required)",
  "strategy": "refine|critique|decompose|explore|verify|hofstadter|adaptive (default: adaptive)",
  "max_iterations": "integer (default: 3)",
  "quality_threshold": "number (default: 0.85)"
}
```

### skill_code_reviewer

```json
{
  "code": "string (required)",
  "language": "string (required)",
  "filename": "string (optional)",
  "focus_areas": "string (optional)"
}
```

### skill_bug_detective

```json
{
  "code": "string (required)",
  "bug_description": "string (required)",
  "language": "string (required)",
  "error_message": "string (optional)",
  "expected_behavior": "string (optional)",
  "actual_behavior": "string (optional)"
}
```

### skill_test_generator

```json
{
  "code": "string (required)",
  "language": "string (required)",
  "framework": "string (optional)",
  "happy_path": "boolean (default: true)",
  "edge_cases": "boolean (default: true)",
  "error_handling": "boolean (default: true)"
}
```

(See `hololoom/agentic/SKILL_AGENTS_README.md` for all 13 skills)

## Troubleshooting

### Issue: Server doesn't start

**Check**:
1. MCP package installed: `pip list | grep mcp`
2. PYTHONPATH is correct in config
3. Run standalone to see errors: `python -m hololoom.mcp_server_promptly`

### Issue: Tools not appearing in Claude

**Solutions**:
1. Restart Claude Desktop completely
2. Check config file syntax (valid JSON)
3. Check Claude Desktop logs:
   - Windows: `%APPDATA%\Claude\logs\`
   - Mac: `~/Library/Logs/Claude/`

### Issue: Tool execution fails

**Debug**:
1. Run server standalone and watch logs
2. Check error messages in tool response
3. Verify parameters match schema

### Issue: Skills not loading

**Check**:
```bash
PYTHONPATH=. python -c "from hololoom.agentic.skill_agents import SkillRegistry; import asyncio; registry = SkillRegistry(); asyncio.run(registry.load_all_skills()); print(f'Loaded {len(registry.skills)} skills')"
```

Should output: `Loaded 13 skills`

## Performance

| Operation | Latency | Notes |
|-----------|---------|-------|
| hololoom_experience | ~50ms | Store memory |
| hololoom_recall | ~100ms | Semantic search |
| hololoom_weave | ~300-600ms | Recursive reasoning (2-3 iterations) |
| skill_* (simple) | ~200-400ms | FAST config, 2 iterations |
| skill_* (complex) | ~500-900ms | FUSED config, 5 iterations |

**Network overhead**: MCP adds ~10-20ms per tool call (stdio communication)

## Security Considerations

1. **Local execution**: MCP server runs locally on your machine
2. **No data sent to cloud**: All processing happens locally
3. **File access**: Server has same permissions as Python process
4. **Memory storage**: Uses SQLite database in `.hololoom/` directory

## Advanced Configuration

### Change HoloLoom Config Mode

Edit `hololoom/mcp_server_promptly.py`:

```python
# Line ~51
config = Config.fast()  # Change to Config.fused() for higher quality
```

### Enable More Verbose Logging

Edit `hololoom/mcp_server_promptly.py`:

```python
# Line ~42
logging.basicConfig(level=logging.DEBUG)  # More detailed logs
```

### Custom Skill Directory

Edit `hololoom/mcp_server_promptly.py`:

```python
# Add to initialize_hololoom()
skill_registry = SkillRegistry(skills_dir="/path/to/custom/skills")
```

## Next Steps

1. **Try the skills**: Start with `skill_code_reviewer` on some code
2. **Explore strategies**: Test CRITIQUE vs DECOMPOSE vs EXPLORE
3. **Use analytics**: Track which skills/strategies work best
4. **Create custom skills**: See `hololoom/agentic/SKILL_AGENTS_README.md`

## See Also

- **Phase 1**: [PROMPTLY_HOLOLOOM_INTEGRATION.md](PROMPTLY_HOLOLOOM_INTEGRATION.md) - Recursive reasoning
- **Phase 2**: [hololoom/analytics/README.md](hololoom/analytics/README.md) - Analytics
- **Phase 3**: [hololoom/agentic/SKILL_AGENTS_README.md](hololoom/agentic/SKILL_AGENTS_README.md) - Professional skills
- **Phase 5**: [DASHBOARD.md](DASHBOARD.md) - Real-time visualization (coming soon)

---

**Version**: 1.0.0
**Created**: 2025-11-16
**Integration**: Promptly (Phases 1-3) → Claude Desktop
**MCP Server**: `hololoom/mcp_server_promptly.py` (750 lines)
