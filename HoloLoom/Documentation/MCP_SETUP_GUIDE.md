# Using HoloLoom Memory in Claude Desktop
## MCP Integration Setup Guide

**Last Updated**: October 24, 2025

---

## What This Gives You

Once configured, you (Claude) can:
- 📖 **Read your memories** - Browse and search anything you've stored
- 💾 **Store new memories** - Save insights during conversations
- 📄 **Process documents** - **NEW!** Smart chunking with entity/motif extraction
- 🔍 **Search strategically** - Use temporal, semantic, graph, or fused strategies
- 🗺️ **Navigate** - Explore forward/backward through memory space
- 📊 **Check health** - See what backends are active

### Universal Memory System

**The system is domain-agnostic** - it works for ANY content:
- 🐝 Beekeeping logs and hive inspections
- 📝 Meeting notes and project updates
- 💻 Code snippets and architecture decisions
- 📚 Research papers and learning notes
- 💡 Ideas and brainstorming sessions
- 📊 Data analysis observations
- 🎯 Goal tracking and reflections

**The domain context comes from:**
1. **What you store** - The text content and metadata you provide
2. **How you tag it** - Use tags like `["beekeeping"]`, `["work"]`, `["research"]`
3. **What you search for** - "winter prep" vs "API design" retrieves different content

**One system, infinite contexts!**

---

## Quick Setup (3 Steps)

### Step 1: Install MCP Python Package

```bash
pip install mcp
```

### Step 2: Configure Claude Desktop

**Location**: Find your Claude Desktop config file:
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
- **Mac**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Linux**: `~/.config/Claude/claude_desktop_config.json`

**Edit the file** and add the HoloLoom memory server:

#### Basic Config (In-Memory, No Setup Required)

```json
{
  "mcpServers": {
    "hololoom-memory": {
      "command": "python",
      "args": ["-m", "HoloLoom.memory.mcp_server"],
      "cwd": "/path/to/your/mythRL",
      "env": {
        "PYTHONPATH": "/path/to/your/mythRL",
        "MEMORY_USER_ID": "your-username"
      }
    }
  }
}
```

**This basic config**:
- ✅ Works immediately, no setup
- ✅ Stores memories in RAM (non-persistent)
- ✅ Perfect for testing
- ❌ Loses data when Claude Desktop restarts

#### Production Config (With Neo4j Persistence)

```json
{
  "mcpServers": {
    "hololoom-memory": {
      "command": "python",
      "args": ["-m", "HoloLoom.memory.mcp_server"],
      "cwd": "/path/to/your/mythRL",
      "env": {
        "PYTHONPATH": "/path/to/your/mythRL",
        "MEMORY_USER_ID": "your-username",
        "ENABLE_NEO4J": "true",
        "NEO4J_URI": "bolt://localhost:7687",
        "NEO4J_USER": "neo4j",
        "NEO4J_PASSWORD": "your-neo4j-password"
      }
    }
  }
}
```

**This production config**:
- ✅ Persistent storage in Neo4j
- ✅ Graph relationships between memories
- ✅ Multiple retrieval strategies
- ⚠️ Requires Neo4j running

**Important**:
- Replace `/path/to/your/mythRL` with your actual repository path
  - Windows: `"c:\\Users\\yourname\\Documents\\mythRL"`
  - Mac/Linux: `"/home/yourname/mythRL"`
- Replace `your-username` with your actual username
- Replace `your-neo4j-password` with your Neo4j password
- Default Neo4j URI is `bolt://localhost:7687` (standard port)

### Step 3: Restart Claude Desktop

Close and reopen Claude Desktop to load the MCP server.

---

## Verification

Once restarted, you should see the HoloLoom memory server in the MCP panel.

**Test it** by asking Claude:
```
Can you list my available memory tools?
```

You should see:
- `recall_memories` - Search memories
- `store_memory` - Save new memories
- `process_text` - **NEW!** Process long documents with smart chunking & entity extraction
- `memory_health` - Check system status
- `chat` - Conversational interface with importance filtering
- `conversation_stats` - View conversation signal/noise ratio

---

## Usage Examples

### Example 1: Search Your Memories

**You ask:**
```
Search my memories for "project meeting notes" using the fused strategy
```

**Claude will:**
```python
# Use the recall_memories tool
{
  "query": "project meeting notes",
  "strategy": "fused",
  "limit": 10
}
```

**You get:**
```
Found 5 memories using FUSED strategy:

1. [0.92] Team standup - discussed API refactoring approach.
   Sarah proposed protocol-based design.
   ID: mem_abc123 | Time: 2025-10-24T14:30:00

2. [0.88] Meeting Notes: Q4 planning session
   - Launch date: November 15
   - Need to finalize architecture decisions
   ...
```

**Context-Aware**: The **same system** works for:
- 📝 Meeting notes → `"Search my memories for Q4 planning"`
- 🐝 Beekeeping logs → `"Search my memories for winter preparation"`
- 💻 Code snippets → `"Search my memories for authentication patterns"`
- 📚 Research notes → `"Search my memories for papers on transformers"`

The **domain** comes from **what you store**, not the configuration!

### Example 2: Store a New Memory

**You say:**
```
Remember: "Decided to use protocol-based architecture for the
memory system. Key insight: separate interface from implementation."
```

**Claude will:**
```python
# Use the store_memory tool
{
  "text": "Decided to use protocol-based architecture for the memory system. Key insight: separate interface from implementation.",
  "context": {
    "date": "2025-10-24",
    "category": "architecture_decision"
  }
}
```

**You get:**
```
✓ Memory stored successfully
ID: mem_xyz789
```

### Example 3: Different Search Strategies

**Temporal** (recent first):
```
Show me my most recent memories
```

**Semantic** (meaning-based):
```
Find memories about architectural patterns
```

**Graph** (relationship-based):
```
What's connected to "protocol-based design" in my memories?
```

### Example 4: Process Long Text Documents

**NEW!** The `process_text` tool integrates the SpinningWheel text spinner for intelligent text processing.

**You say:**
```
Process this article into my memory system:

[paste a long article about beekeeping, a blog post, meeting notes, etc.]
```

**Claude will:**
```python
# Use the process_text tool
{
  "text": "Your long article text...",
  "source": "article",
  "chunk_by": "paragraph",  # Smart chunking
  "tags": ["beekeeping", "article"],
  "user_id": "blake"
}
```

**What happens:**
1. **Smart Chunking**: Text is split into coherent chunks (by paragraph, sentence, or fixed size)
2. **Entity Extraction**: Identifies entities (people, places, concepts) in each chunk
3. **Motif Detection**: Finds patterns and themes across the text
4. **Batch Storage**: All chunks stored efficiently with extracted features
5. **Searchable**: Each chunk becomes independently searchable

**You get:**
```
✓ Text processed and stored via SpinningWheel

📄 Source: article
📊 Chunks Created: 7
💾 Memories Stored: 7

🔍 Extracted Features:
  • Entities: 23 total (12 unique)
  • Motifs: 15 total (8 unique)

  Sample Entities: Winter, Queen, Cluster, Honey, Colony
  Sample Motifs: temperature, feeding, ventilation

📝 Memory IDs:
  1. shard_001
  2. shard_002
  ... and 5 more
```

**Why use process_text instead of store_memory?**

| Feature | `store_memory` | `process_text` |
|---------|----------------|----------------|
| **Text length** | Short snippets | Long documents |
| **Chunking** | Manual | Automatic |
| **Entity extraction** | No | Yes |
| **Motif detection** | No | Yes |
| **Best for** | Quick notes | Articles, transcripts, documents |

**Chunking modes:**
- `"paragraph"` - Split by paragraphs (best for articles)
- `"sentence"` - Split by sentences (for detailed analysis)
- `"fixed"` - Fixed character size chunks (for uniform storage)

### Example 5: System Health Check

**You ask:**
```
Check my memory system health
```

**Claude will:**
```python
# Use the memory_health tool
{}
```

**You get:**
```
Memory System Health Check:
Status: healthy
Backend: neo4j
Total Memories: 143

Active Backends:
  • Neo4j: healthy
  • In-Memory: healthy
```

---

## Advanced Configuration

### Using Multiple Backends

If you want to use both Neo4j AND Mem0:

```json
{
  "mcpServers": {
    "hololoom-memory": {
      "command": "python",
      "args": ["-m", "HoloLoom.memory.mcp_server"],
      "cwd": "c:\\Users\\blake\\Documents\\mythRL",
      "env": {
        "PYTHONPATH": "c:\\Users\\blake\\Documents\\mythRL",
        "MEMORY_USER_ID": "blake",
        "ENABLE_NEO4J": "true",
        "ENABLE_MEM0": "true",
        "ENABLE_QDRANT": "false",
        "NEO4J_URI": "bolt://localhost:7688",
        "NEO4J_USER": "neo4j",
        "NEO4J_PASSWORD": "beekeeper123",
        "MEM0_API_KEY": "your-api-key-here"
      }
    }
  }
}
```

### Development Mode (In-Memory Only)

For testing without Neo4j:

```json
{
  "mcpServers": {
    "hololoom-memory": {
      "command": "python",
      "args": ["-m", "HoloLoom.memory.mcp_server"],
      "cwd": "c:\\Users\\blake\\Documents\\mythRL",
      "env": {
        "PYTHONPATH": "c:\\Users\\blake\\Documents\\mythRL",
        "MEMORY_USER_ID": "blake",
        "ENABLE_NEO4J": "false",
        "ENABLE_MEM0": "false",
        "ENABLE_QDRANT": "false"
      }
    }
  }
}
```

This will use `InMemoryStore` (no persistence, but works immediately).

---

## Troubleshooting

### "MCP server not found"

**Problem**: Claude Desktop can't find the MCP server.

**Solutions**:
1. Check `cwd` path is correct
2. Check `PYTHONPATH` points to repository root
3. Verify Python is in your PATH
4. Try absolute path to Python: `"command": "C:\\Python311\\python.exe"`

### "Import errors"

**Problem**: Python can't import HoloLoom modules.

**Solutions**:
1. Ensure `PYTHONPATH` includes repository root
2. Check that `HoloLoom/memory/mcp_server.py` exists
3. Verify dependencies: `pip install mcp`

### "Neo4j connection failed"

**Problem**: Can't connect to Neo4j.

**Solutions**:
1. Verify Neo4j is running: Check `http://localhost:7475`
2. Check credentials are correct
3. Verify port (default beekeeping is 7688, not 7687)
4. Set `ENABLE_NEO4J: "false"` to use in-memory instead

### "No memories found"

**Problem**: Searches return empty.

**Solutions**:
1. Check you have memories stored
2. Try broader search query
3. Check `memory_health` to see backend status
4. Verify `MEMORY_USER_ID` matches stored memories

---

## MCP Server Architecture

```
┌─────────────────────────────┐
│   Claude Desktop            │  ← You interact here
└──────────────┬──────────────┘
               │ MCP Protocol (JSON-RPC)
┌──────────────▼──────────────┐
│   mcp_server.py             │  ← Thin wrapper (~420 lines)
│   - list_resources()        │
│   - list_tools()            │
│   - call_tool()             │
└──────────────┬──────────────┘
               │ Python API
┌──────────────▼──────────────┐
│   UnifiedMemoryInterface    │  ← Protocol-based system
└──────────────┬──────────────┘
               │ MemoryStore Protocol
┌──────────────▼──────────────┐
│   Backend Stores            │
│   - Neo4j (graph)           │
│   - Mem0 (AI extraction)    │
│   - Qdrant (vectors)        │
│   - InMemory (fallback)     │
└─────────────────────────────┘
```

---

## Available Tools Reference

### `recall_memories`

Search and retrieve memories.

**Parameters**:
- `query` (required): Search text
- `strategy`: `"temporal"`, `"semantic"`, `"graph"`, `"pattern"`, or `"fused"` (default)
- `limit`: Max results (default: 10)
- `user_id`: Filter by user

**Example**:
```
Search my memories for "Hive Jodi" using semantic strategy, limit 5
```

### `store_memory`

Save a new memory.

**Parameters**:
- `text` (required): Memory content
- `context`: Dict with metadata (place, time, entities, etc.)
- `tags`: List of tags
- `user_id`: User identifier

**Example**:
```
Remember: "Queen spotted on frame 3, excellent laying pattern"
with tags ["inspection", "queen"]
```

### `memory_health`

Check system status.

**Parameters**: None

**Example**:
```
Check my memory system health
```

---

## Resources (Browse Memories)

In addition to tools, HoloLoom exposes memories as **resources** you can browse.

Each memory has a URI: `memory://<memory_id>`

**You can ask**:
```
Show me the resource memory://mem_abc123
```

**Claude will read** the full memory with all context and metadata.

---

## Tips for Using with Claude

### Be Specific with Strategies

- **"Show recent memories"** → Uses `TEMPORAL`
- **"Find similar memories about X"** → Uses `SEMANTIC`
- **"What's related to Y?"** → Uses `GRAPH`
- **"Best matches for Z"** → Uses `FUSED`

### Store Context-Rich Memories

The key to a useful memory system is **rich context**. Compare:

**❌ Vague**:
```
Remember: "Meeting went well"
```

**✅ Specific**:
```
Remember: "Team standup - decided to use protocol-based architecture
for memory system. Sarah pointed out this matches Go interfaces pattern.
Action: Review existing PolicyEngine protocol for consistency."
with tags: ["meeting", "architecture", "decision"]
```

### Use Tags for Organization

Tags help filter and organize across domains:

```
# Work project
Remember: "API rate limit increased to 1000 req/min"
with tags: ["work", "api", "infrastructure"]

# Beekeeping
Remember: "Hive Jodi - 15 frames of brood, very strong"
with tags: ["beekeeping", "inspection", "hive-jodi"]

# Research
Remember: "Paper: Attention Is All You Need - introduced Transformer architecture"
with tags: ["research", "ml", "paper", "transformers"]
```

### Ask for Summaries

```
Search my memories for "architecture decisions" and summarize the key patterns
```

Claude will search memories AND synthesize the results.

### Mix Domains Naturally

The **same session** can handle multiple contexts:

```
You: Search my work memories for API design patterns
Claude: [searches with tag filter "work"]

You: Now search my beekeeping notes for winter tasks
Claude: [searches different content]

You: Compare these - are there any common patterns?
Claude: [synthesizes across domains]
```

---

## What's Different from Regular Claude?

### Without MCP (Regular Claude)
- ❌ Can't access your local memories
- ❌ Can't store persistent information
- ❌ No memory across sessions
- ✅ General knowledge only

### With MCP (This Setup)
- ✅ **Access your beekeeping notes**
- ✅ **Store insights permanently**
- ✅ **Search multiple strategies**
- ✅ **Navigate memory space**
- ✅ **Graph relationships** (if using Neo4j)
- ✅ **Persistent across sessions**

---

## Example Session

**You**: "Check my memory system health"

**Claude**:
```
Memory System Health Check:
Status: healthy
Backend: neo4j
Total Memories: 143
```

**You**: "What are my most recent hive inspections?"

**Claude** (uses `recall_memories` with `strategy: "temporal"`):
```
Found 5 memories:

1. [0.95] Hive Jodi inspection - 15 frames of brood...
   Time: 2025-10-24T14:30:00

2. [0.92] Dennis's double stack showing good population...
   Time: 2025-10-23T15:00:00

...
```

**You**: "Focus on Hive 5. What's the situation?"

**Claude** (searches for "Hive 5"):
```
Found critical situation:

Hive 5 is critically weak - only 10 bees on inner cover.
Colony appears queenless. Must combine with stronger colony
ASAP or will not survive winter.

Action required: URGENT combination before first frost (Nov 2-3).
```

**You**: "Remember that I combined Hive 5 with Jodi Split today,
used newspaper method, both colonies calm."

**Claude** (uses `store_memory`):
```
✓ Memory stored successfully
ID: mem_new123

This resolves the critical action item for Hive 5!
```

---

## Next Steps

1. ✅ **Configure** Claude Desktop (edit config file)
2. ✅ **Restart** Claude Desktop
3. ✅ **Test** with `memory_health` tool
4. ✅ **Store** some beekeeping notes
5. ✅ **Search** your memories
6. 🎉 **Enjoy** persistent memory in Claude!

---

## Additional Resources

- **MCP Documentation**: https://modelcontextprotocol.io/
- **HoloLoom Architecture**: [ARCHITECTURE_PATTERNS.md](ARCHITECTURE_PATTERNS.md)
- **Memory Refactor**: [MEMORY_ARCHITECTURE_REFACTOR.md](MEMORY_ARCHITECTURE_REFACTOR.md)
- **MCP Server Code**: [../memory/mcp_server.py](../memory/mcp_server.py)

---

**Status**: Ready to use! 🚀
**Support**: Check logs if issues occur, verify paths and credentials