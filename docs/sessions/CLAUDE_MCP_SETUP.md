# Claude Desktop MCP Setup - COMPLETE ✅

## Configuration Status

✅ **Config file created:** `C:\Users\blake\AppData\Roaming\Claude\claude_desktop_config.json`  
✅ **MCP server registered:** `holoLoom-memory`  
✅ **Python environment configured:** Using your `.venv`

## Next Steps

### 1. Restart Claude Desktop

If Claude Desktop is currently running:
1. **Completely quit** Claude Desktop (not just close the window)
2. **Reopen** Claude Desktop
3. Look for the **🔌 icon** in the bottom right (indicates MCP connected)

### 2. Test Connection

Once you see the 🔌 icon, try these commands in Claude:

#### Basic Test
```
What tools do you have available?
```
You should see: `recall_memories`, `store_memory`, `memory_health`

#### Store a Memory
```
Store this memory: "Hive Jodi needs winter prep by November" with tags ["apiary", "urgent"]
```

#### Search Memories
```
Can you recall memories about "winter" using the fused strategy?
```

#### Check System Health
```
What's the memory system health status?
```

#### Browse Memories as Resources
```
Can you list recent memories?
```

## Architecture

```
┌─────────────────────────┐
│   Claude Desktop        │
│   (You're here!)        │
└───────────┬─────────────┘
            │ JSON-RPC (stdio)
            ▼
┌─────────────────────────┐
│   mcp_server.py         │
│   • recall_memories     │
│   • store_memory        │
│   • memory_health       │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  UnifiedMemoryInterface │
│  (Protocol-based API)   │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│   Memory Backends       │
│   • InMemory (active)   │
│   • Mem0 (available)    │
│   • Neo4j (available)   │
│   • Qdrant (available)  │
└─────────────────────────┘
```

## What You Can Do

### Search Strategies

Ask Claude to use different strategies:

**Temporal** (recent first):
```
Recall memories about "apiary" using temporal strategy
```

**Semantic** (meaning-based):
```
Recall memories about "beekeeping" using semantic strategy
```

**Graph** (relationship-based, requires Neo4j):
```
Recall memories about "Jodi" using graph strategy
```

**Fused** (best overall, combines all):
```
Recall memories about "winter" using fused strategy
```

### Store Rich Memories

**With context:**
```
Store: "Queen bee spotted in hive 3" 
with context: {"place": "apiary", "hive": 3, "actor": "Jodi"}
and tags: ["observation", "queen"]
```

**Simple store:**
```
Store this memory: "Need to order more frames for spring"
```

### Browse Like Files

Memories are exposed as resources with URIs like `memory://abc123...`

```
Show me recent memories
Read memory://[paste ID here]
```

## Troubleshooting

### 🔌 Icon Not Showing

1. **Check logs:** `%APPDATA%\Claude\logs\`
2. **Verify config:** Should be in `%APPDATA%\Claude\claude_desktop_config.json`
3. **Test manually:**
   ```powershell
   C:/Users/blake/Documents/mythRL/.venv/Scripts/python.exe -m HoloLoom.memory.mcp_server
   ```
   Should output: `Initializing HoloLoom Memory MCP Server...`

### "Module not found" Error

The MCP server will gracefully fall back to InMemoryStore if external backends aren't installed.

To add backends:
```powershell
# Mem0 (LLM extraction)
pip install mem0ai

# Neo4j (graph storage)
pip install neo4j

# Qdrant (vector search)
pip install qdrant-client sentence-transformers
```

### Config Not Loading

Make sure the JSON is valid:
```powershell
Get-Content "$env:APPDATA\Claude\claude_desktop_config.json" | ConvertFrom-Json
```

Should show the config without errors.

## Example Conversation

**You:** "Can you store this memory: 'Met with Jodi to discuss hive placement for spring expansion' with tags apiary and planning?"

**Claude:** ✓ Memory stored successfully
ID: abc123-def456-...
Text: Met with Jodi to discuss hive placement for spring expansion...

**You:** "Now recall memories about spring"

**Claude:** Found 1 memory using FUSED strategy:
1. [0.892] Met with Jodi to discuss hive placement for spring expansion
   Context: {}
   Tags: apiary, planning
   ID: abc123-def456-... | Time: 2025-10-24T00:45:00

## Advanced Usage

### Configure Backend Weights

Edit `HoloLoom/memory/mcp_server.py` → `init_memory()`:

```python
memory = await create_unified_memory(
    user_id="blake",
    enable_mem0=True,    # LLM extraction
    enable_neo4j=True,   # Graph storage  
    enable_qdrant=True   # Vector search
)
```

### Adjust Fusion Strategy

In `HoloLoom/memory/stores/hybrid_store.py`:

```python
HybridMemoryStore(
    backends=[...],
    fusion_method="weighted"  # or "max", "mean", "rrf"
)
```

### Add More Tools

In `mcp_server.py` → `list_tools()`, add:

```python
Tool(
    name="your_tool_name",
    description="What it does",
    inputSchema={...}
)
```

## Files Reference

- **Server:** `HoloLoom/memory/mcp_server.py`
- **Protocol:** `HoloLoom/memory/protocol.py`
- **Config:** `C:\Users\blake\AppData\Roaming\Claude\claude_desktop_config.json`
- **Docs:** 
  - `HoloLoom/memory/MCP_SETUP.md` - Detailed setup
  - `HoloLoom/memory/QUICKSTART.md` - Memory basics
  - `HoloLoom/memory/REFERENCE.md` - API reference

## Support

If you see errors:
1. Check Claude logs: `%APPDATA%\Claude\logs\mcp*.log`
2. Test server: `python -m HoloLoom.memory.mcp_server`
3. Verify Python path in config matches your venv

---

**Status:** Ready to use! Restart Claude Desktop and look for 🔌
