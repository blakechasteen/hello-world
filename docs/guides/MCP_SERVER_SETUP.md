# MCP Server Setup

Connect Claude Desktop to HoloLoom via the Model Context Protocol (MCP).

## What You Get

Claude Desktop gains access to HoloLoom's memory system:

| Tool | Purpose |
|------|---------|
| `hololoom_experience` | Store memories in knowledge graph |
| `hololoom_recall` | Semantic search + graph traversal |
| `hololoom_weave` | Recursive reasoning with strategy selection |
| `hololoom_analytics_summary` | Performance metrics |

Plus 13 professional skills (code review, writing, research, etc.).

## Setup

### 1. Configure Claude Desktop

Add to your Claude Desktop config:

**Windows:** `%APPDATA%\Claude\claude_desktop_config.json`
**Mac:** `~/Library/Application Support/Claude/claude_desktop_config.json`
**Linux:** `~/.config/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "HoloLoom-memory": {
      "command": "python",
      "args": ["-m", "hololoom.mcp_tools.server"],
      "env": {
        "PYTHONPATH": "/path/to/mythRL"
      }
    }
  }
}
```

### 2. Start Backends (Optional)

For persistent memory:

```bash
cd config && docker-compose up -d
```

Without Docker, the system uses in-memory storage (data lost on restart).

### 3. Restart Claude Desktop

Claude Desktop only loads MCP config at startup. Quit and relaunch.

## Verify

In Claude Desktop:

```
You: "Check memory health"
Expected: Claude calls HoloLoom-memory:memory_health, shows backend status
```

```
You: "Remember that HoloLoom uses Matryoshka embeddings"
Expected: Claude calls hololoom_experience to store the memory
```

```
You: "What do you know about embeddings?"
Expected: Claude calls hololoom_recall with semantic search
```

## Transport

The MCP server uses **STDIO** (standard input/output). Claude Desktop launches it as a subprocess — no ports, no networking, no security concerns. This is the recommended approach for local MCP servers.

For remote access (multi-machine), switch to HTTP transport in the server config.

## Troubleshooting

| Problem | Fix |
|---------|-----|
| "Skill not found" | Restart Claude Desktop |
| "MCP server connection failed" | Check PYTHONPATH in config |
| "Memory backend unavailable" | `docker-compose up -d` in config/ |
| Tool call failed | Check Claude Desktop console logs |

Test the server manually:

```bash
python -m hololoom.mcp_tools.server
```

If it starts without errors, the server is working and Claude Desktop should be able to connect.
