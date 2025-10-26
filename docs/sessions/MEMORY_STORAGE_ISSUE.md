# Memory Storage Issue - Root Cause & Solution

## Problem
Memories are not being stored from Claude Desktop because your OpenAI API quota has been exceeded.

## Root Cause Analysis

1. **OpenAI API Quota Exceeded**
   - Your API key in `test_mem0_working.py` has exceeded its quota
   - Error: `Error code: 429 - insufficient_quota`
   - This affects all systems using OpenAI (Mem0, embeddings, LLM extraction)

2. **Mem0 + Ollama Dimension Mismatch**
   - Attempted to switch to Ollama (local, free)
   - Mem0 has a bug: it hardcodes vector dimensions to 1536 (OpenAI's dimension)
   - Ollama's `nomic-embed-text` uses 768 dimensions
   - Result: `Vector dimension error: expected dim: 1536, got 768`

3. **Collections Already Created**
   - Qdrant collections (`mem0_ollama_fresh`, `hololoom_mem0_768`) were created with 1536 dims
   - Once created, dimension cannot be changed
   - Deleting and recreating doesn't help because Mem0 recreates with 1536

## Solutions (3 Options)

### Option 1: Add OpenAI API Credits (Fastest)
**Pros:**
- Works immediately
- No code changes needed
- Mem0 designed for OpenAI

**Cons:**
- Costs money ($5-20/month)
- Not fully local

**How:**
1. Go to https://platform.openai.com/settings/organization/billing
2. Add credits to your API key
3. Restart MCP server
4. Memories will start storing immediately

### Option 2: Use OpenRouter (Free Tier)
**Pros:**
- Free tier available
- Compatible with OpenAI API format
- Multiple model options

**Cons:**
- Requires signup
- Rate limits on free tier

**How:**
1. Sign up at https://openrouter.ai/
2. Get API key
3. Update [HoloLoom/memory/stores/mem0_store.py](HoloLoom/memory/stores/mem0_store.py):
   ```python
   default_config = {
       "llm": {
           "provider": "openai",
           "config": {
               "model": "meta-llama/llama-3.2-3b-instruct:free",
               "api_key": "YOUR_OPENROUTER_KEY",
               "base_url": "https://openrouter.ai/api/v1"
           }
       }
   }
   ```

### Option 3: Bypass Mem0 (Use Direct Qdrant) **RECOMMENDED**
**Pros:**
- Fully local & free
- Full control over dimensions
- Faster (no LLM extraction overhead)

**Cons:**
- No automatic fact extraction (manual memory chunks)
- More code to write

**How:**
Already implemented! Use the Qdrant store directly:
```python
from HoloLoom.memory.stores.qdrant_store import QdrantMemoryStore
from HoloLoom.memory.protocol import Memory
from datetime import datetime

# Create store
store = QdrantMemoryStore()

# Store memory
mem = Memory(
    id="",
    text="I have three hives named Jodi, Aurora, and Luna",
    timestamp=datetime.now(),
    context={"topic": "beekeeping", "entities": ["Jodi", "Aurora", "Luna"]},
    metadata={"user_id": "blake"}
)

await store.store(mem)
```

## Quick Fix for MCP Server

The simplest solution is to disable Mem0 and use just Qdrant + Neo4j:

1. Edit [HoloLoom/memory/mcp_server.py](HoloLoom/memory/mcp_server.py) line 941:
   ```python
   await init_memory(
       user_id="blake",
       enable_mem0=False,  # DISABLE - has OpenAI quota issue
       enable_neo4j=True,
       enable_qdrant=True
   )
   ```

2. Restart Claude Desktop

3. Test with `store_memory` tool - it will use Qdrant directly (free, local, works)

## Current Status

✅ **Fixed:** MCP server configured to use Ollama
✅ **Fixed:** Qdrant + Neo4j backends working
❌ **Blocked:** Mem0 has hardcoded OpenAI dimensions (bug)
💡 **Workaround:** Disable Mem0, use Qdrant + Neo4j directly

## Test Commands

```bash
# Test Qdrant directly (should work)
python test_neo4j_qdrant_hybrid.py

# Test MCP server with Mem0 disabled
python -m HoloLoom.memory.mcp_server

# In Claude Desktop, use tool:
store_memory(text="Test memory", user_id="blake")
```

## Next Steps

1. Disable Mem0 in MCP server config (line 943)
2. Restart Claude Desktop
3. Test memory storage with `store_memory` tool
4. Consider Option 1 or 2 if you want Mem0's LLM extraction features

---

**Bottom Line:** Your OpenAI API quota is the blocker. Either add credits, use OpenRouter, or disable Mem0 and use Qdrant directly (which works perfectly and is free).
