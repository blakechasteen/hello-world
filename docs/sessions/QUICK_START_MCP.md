# Quick Start: HoloLoom Memory in Claude Desktop

## 3 Steps to Enable

### 1. Install MCP
```bash
pip install mcp
```

### 2. Edit Claude Desktop Config

**Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

Add this (update YOUR_PATH to your actual repository location):
```json
{
  "mcpServers": {
    "hololoom-memory": {
      "command": "python",
      "args": ["-m", "HoloLoom.memory.mcp_server"],
      "cwd": "YOUR_PATH\\mythRL",
      "env": {
        "PYTHONPATH": "YOUR_PATH\\mythRL",
        "MEMORY_USER_ID": "blake",
        "ENABLE_NEO4J": "false",
        "ENABLE_MEM0": "false",
        "ENABLE_QDRANT": "false"
      }
    }
  }
}
```

**Note**: This uses in-memory storage (no persistence). See [full guide](HoloLoom/Documentation/MCP_SETUP_GUIDE.md) for Neo4j/Mem0 setup.

**Optional - Enable Neo4j** (for persistent graph storage):
```json
"env": {
  "PYTHONPATH": "YOUR_PATH\\mythRL",
  "MEMORY_USER_ID": "blake",
  "ENABLE_NEO4J": "true",
  "NEO4J_URI": "bolt://localhost:7687",
  "NEO4J_USER": "neo4j",
  "NEO4J_PASSWORD": "your-password-here"
}
```

### 3. Restart Claude Desktop

Close and reopen Claude Desktop.

---

## Test It

Ask Claude:
```
Check my memory system health
```

You should see:
```
Memory System Health Check:
Status: healthy
Backend: neo4j
Total Memories: <count>
```

---

## What You Can Do

### Search Memories
```
Search my memories for "winter preparation"
```

### Store Memories
```
Remember: "Installed mouse guards on all hives today"
```

### Process Long Documents (NEW!)
```
Process this article into my memory system:

[paste long beekeeping article, meeting notes, blog post, etc.]
```

**The system will:**
- Smart chunk the text (by paragraph, sentence, or fixed size)
- Extract entities and motifs from each chunk
- Store all chunks with searchable metadata
- Return statistics about what was extracted

### Check Health
```
Check my memory system health
```

### Different Strategies
```
Show recent hive inspections (temporal)
Find memories about queen bees (semantic)
What's related to Hive Jodi? (graph)
```

---

## Troubleshooting

**"MCP server not found"**
- Check `cwd` path in config
- Verify `PYTHONPATH` is correct

**"Neo4j connection failed"**
- Verify Neo4j is running: http://localhost:7475
- Check credentials in config
- Or set `ENABLE_NEO4J: "false"` to use in-memory

**"Import errors"**
- Check `pip install mcp` completed
- Verify paths in config

---

## Full Documentation

See [HoloLoom/Documentation/MCP_SETUP_GUIDE.md](HoloLoom/Documentation/MCP_SETUP_GUIDE.md)

---

**Status**: Ready! 🚀