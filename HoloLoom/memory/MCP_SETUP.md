# HoloLoom Memory MCP Server Setup

## What is This?

The HoloLoom Memory MCP Server exposes your unified memory system (Mem0, Neo4j, Qdrant) to any MCP-compatible tool like:
- **Claude Desktop** - Ask Claude to search your memories
- **VS Code** - Access memories in your editor
- **Custom AI tools** - Any tool supporting Model Context Protocol

## Quick Start

### 1. Install MCP SDK

```bash
pip install mcp
```

### 2. Test the Server Standalone

```bash
# From the mythRL directory
python -m HoloLoom.memory.mcp_server
```

You should see:
```
Initializing HoloLoom Memory MCP Server...
✓ InMemory store available
Memory system initialized: healthy (in-memory)
Starting MCP server on stdio...
```

Press `Ctrl+C` to stop.

### 3. Configure Claude Desktop (Optional)

#### Windows Location:
```
%APPDATA%\Claude\claude_desktop_config.json
```

#### Add this configuration:

```json
{
  "mcpServers": {
    "holoLoom-memory": {
      "command": "python",
      "args": ["-m", "HoloLoom.memory.mcp_server"],
      "env": {
        "PYTHONPATH": "c:\\Users\\blake\\Documents\\mythRL"
      }
    }
  }
}
```

#### Restart Claude Desktop

You should see a 🔌 icon indicating the server is connected.

## Available Tools

### 1. `recall_memories`
Search your memories with various strategies.

**Example in Claude:**
```
Can you recall memories about "winter preparations" using the fused strategy?
```

**Parameters:**
- `query` (required): Search keywords or concepts
- `strategy` (optional): `temporal`, `semantic`, `graph`, `pattern`, or `fused` (default)
- `limit` (optional): Max results (1-100, default 10)
- `user_id` (optional): Filter by user

### 2. `store_memory`
Store a new memory.

**Example in Claude:**
```
Store this memory: "Hive Jodi needs winter prep by November" with tags ["apiary", "urgent"]
```

**Parameters:**
- `text` (required): Memory content
- `context` (optional): Metadata object (e.g., `{"place": "apiary", "time": "autumn"}`)
- `tags` (optional): Array of tags
- `user_id` (optional): User identifier

### 3. `memory_health`
Check system status and statistics.

**Example in Claude:**
```
What's the health status of the memory system?
```

## Available Resources

Memories are exposed as browsable resources with URIs like:
```
memory://abc123-def456-...
```

You can:
- **List** recent memories (last 100)
- **Read** specific memories by URI
- **Browse** like files in a filesystem

## Architecture

```
┌─────────────────────────┐
│  Claude Desktop / Tools │
│  (MCP Client)           │
└───────────┬─────────────┘
            │ JSON-RPC over stdio
            ▼
┌─────────────────────────┐
│  mcp_server.py          │
│  - list_resources()     │
│  - read_resource()      │
│  - list_tools()         │
│  - call_tool()          │
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
│  Memory Backends        │
│  • InMemory (default)   │
│  • Mem0 (LLM)          │
│  • Neo4j (graph)        │
│  • Qdrant (vectors)     │
└─────────────────────────┘
```

## Configuration

### Enable/Disable Backends

Edit `mcp_server.py` → `main()`:

```python
await init_memory(
    user_id="blake",
    enable_mem0=True,    # LLM extraction
    enable_neo4j=True,   # Graph storage
    enable_qdrant=True   # Vector search
)
```

### Set Default User

Change `user_id` in `init_memory()` call.

### Adjust Logging

```python
logging.basicConfig(level=logging.DEBUG)  # More verbose
logging.basicConfig(level=logging.WARNING)  # Less verbose
```

## Troubleshooting

### "MCP not installed"
```bash
pip install mcp
```

### "Module not found: HoloLoom"
Set `PYTHONPATH` to your workspace root:
```bash
# Windows PowerShell
$env:PYTHONPATH = "c:\Users\blake\Documents\mythRL"

# Windows CMD
set PYTHONPATH=c:\Users\blake\Documents\mythRL
```

### Claude Desktop not connecting
1. Check config file location: `%APPDATA%\Claude\claude_desktop_config.json`
2. Verify JSON is valid (use JSONLint)
3. Check logs: `%APPDATA%\Claude\logs\`
4. Restart Claude Desktop completely

### Server starts but no data
- Server uses InMemoryStore by default (no persistence)
- Store some test memories first
- To use persistent backends, install and configure Mem0/Neo4j/Qdrant

## Testing

### Test with Stdio

```bash
# Start server
python -m HoloLoom.memory.mcp_server

# In another terminal, send MCP request:
echo '{"jsonrpc":"2.0","method":"tools/list","id":1}' | python -m HoloLoom.memory.mcp_server
```

### Test Memory Operations

```python
# test_mcp_memory.py
import asyncio
from HoloLoom.memory.mcp_server import init_memory, memory

async def test():
    await init_memory(user_id="test")
    
    # Store
    mem_id = await memory.store("Test memory", tags=["test"])
    print(f"Stored: {mem_id}")
    
    # Recall
    result = await memory.recall("test", limit=5)
    print(f"Found: {len(result.memories)} memories")

asyncio.run(test())
```

## Next Steps

1. **Test locally** - Run server standalone first
2. **Add to Claude** - Configure Claude Desktop
3. **Try queries** - Ask Claude to search your memories
4. **Install backends** - Add Mem0/Neo4j/Qdrant for production use
5. **Customize** - Adjust strategies, weights, and filters

## Example Usage in Claude

Once configured, you can:

**Search memories:**
> "Can you recall memories about beekeeping using the fused strategy?"

**Store new memories:**
> "Store this memory: 'Met with Jodi to discuss hive placement' with context place='apiary' and tags ['meeting', 'planning']"

**Check system:**
> "What's the memory system health status?"

**Browse resources:**
> "Show me recent memories" (Claude will list resources)

**Read specific memory:**
> "Read memory://abc123..." (Claude will fetch full content)

## Advanced: Custom Strategies

You can add custom strategies by implementing the `MemoryStore` protocol:

```python
class CustomStore:
    async def recall(self, query: MemoryQuery) -> RetrievalResult:
        # Your custom search logic
        pass
```

Then add to hybrid store in `init_memory()`.

---

**Questions?** Check the main documentation:
- `memory/QUICKSTART.md` - Basic usage
- `memory/REFERENCE.md` - API reference
- `Documentation/HANDOFF_UNIFIED_MEMORY.md` - Full system overview
