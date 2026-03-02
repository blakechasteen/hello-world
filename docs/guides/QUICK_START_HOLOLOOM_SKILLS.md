# HoloLoom Skills for Claude Desktop

Setup guide for the SpinningWheel and Loom skills in Claude Desktop.

## Skills

| Skill | Purpose | Trigger |
|-------|---------|---------|
| **spinning-wheel** | Data ingestion (web, docs, browser history) | "Add this to memory", "Ingest this URL" |
| **loom** | Memory retrieval and synthesis | "What did I learn about...", "How does X relate to Y" |

## Setup

### 1. Verify Skills Exist

```bash
ls ~/.claude/skills/
# Should show: spinning-wheel/ and loom/
```

### 2. Configure MCP Server

See [MCP_SERVER_SETUP.md](MCP_SERVER_SETUP.md) for the full config. The key entry in your Claude Desktop config:

```json
{
  "mcpServers": {
    "HoloLoom-memory": {
      "command": "python",
      "args": ["-m", "hololoom.mcp_tools.server"],
      "env": { "PYTHONPATH": "/path/to/mythRL" }
    }
  }
}
```

### 3. Start Backends (Optional)

```bash
cd config && docker-compose up -d  # Neo4j + Qdrant
```

### 4. Restart Claude Desktop

Skills and MCP config only load at startup.

## Test

```
You: "What can the spinning-wheel skill do?"
Expected: Claude explains ingestion capabilities

You: "Add this article to memory: https://docs.anthropic.com"
Expected: SpinningWheel ingests the page, stores chunks

You: "What did I just learn about Claude?"
Expected: Loom retrieves and synthesizes stored memories
```

## Example Workflows

### Research a Topic

```
1. "Ingest all Claude API docs"         -> SpinningWheel crawls ~30 pages
2. "What did I learn about agents?"     -> Loom retrieves with temporal strategy
3. "Synthesize best practices"          -> Loom fused strategy with synthesis
```

### Process Local Notes

```
1. "Process ~/notes/project.md"         -> SpinningWheel chunks and stores
2. "What are my notes about X?"         -> Loom semantic search
```

## Troubleshooting

| Issue | Fix |
|-------|-----|
| "Skill not found" | Check `~/.claude/skills/` exists, restart Claude Desktop |
| "MCP server failed" | Test manually: `python -m hololoom.mcp_tools.server` |
| "Memory unavailable" | Start backends: `cd config && docker-compose up -d` |

## Architecture

See [HOLOLOOM_CLAUDE_DESKTOP_ARCHITECTURE.md](HOLOLOOM_CLAUDE_DESKTOP_ARCHITECTURE.md) for the full two-skill architecture.
