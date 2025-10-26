# Quick Start: HoloLoom Skills for Claude Desktop

**Get up and running in 5 minutes**

---

## What You're Getting

**Two Claude Desktop skills that make HoloLoom usable:**

1. **spinning-wheel** - Smart data ingestion (websites, docs, browser history)
2. **loom** - Intelligent memory retrieval and synthesis

---

## Setup Steps

### 1. Verify Skills Are Installed

```bash
# Check skills directory
ls ~/.claude/skills/

# Should show:
# spinning-wheel/
# loom/
```

✅ **Already done!** Skills were created in `~/.claude/skills/`

### 2. Update Claude Desktop Config

**Location:** This file needs to be copied to Claude Desktop's config location.

**Current location:**
```
c:\Users\blake\Documents\mythRL\mcp_server\claude_desktop_config.json
```

**Claude Desktop looks for config at:**
```
Windows: %APPDATA%\Claude\claude_desktop_config.json
Mac: ~/Library/Application Support/Claude/claude_desktop_config.json
Linux: ~/.config/Claude/claude_desktop_config.json
```

**Action needed:**

```bash
# Windows (PowerShell)
Copy-Item "c:\Users\blake\Documents\mythRL\mcp_server\claude_desktop_config.json" "$env:APPDATA\Claude\claude_desktop_config.json"

# OR manually copy the file to the correct location
```

**The config now includes:**
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

### 3. Ensure Services Are Running

```bash
# Check Neo4j
neo4j status

# If not running:
neo4j start

# Check Qdrant (if using Docker)
docker ps | grep qdrant

# If not running:
docker run -p 6333:6333 -d qdrant/qdrant
```

### 4. Restart Claude Desktop

**Important:** Claude Desktop only loads skills and MCP config at startup.

1. Quit Claude Desktop completely
2. Restart Claude Desktop
3. Wait for initialization (~5-10 seconds)

---

## Test Your Setup

### Test 1: Skills Loaded

**In Claude Desktop:**

```
You: "What can the spinning-wheel skill do?"

Expected: Claude explains ingestion capabilities
```

```
You: "What can the loom skill do?"

Expected: Claude explains memory retrieval capabilities
```

### Test 2: MCP Server Connected

```
You: "Check memory health"

Expected: Claude calls holoLoom-memory:memory_health
Shows Neo4j and Qdrant status
```

### Test 3: Ingest Data (SpinningWheel)

```
You: "Add this article to memory: https://docs.anthropic.com/en/docs/intro-to-claude"

Expected:
- SpinningWheel skill activates
- Scrapes webpage
- Calls holoLoom-memory:ingest_webpage
- Reports: "Stored X chunks from article"
```

### Test 4: Retrieve Data (Loom)

```
You: "What did I just learn about Claude?"

Expected:
- Loom skill activates
- Calls holoLoom-memory:recall_memories
- Returns memories with citations
```

### Test 5: Conversational Intelligence

```
You: "How do embeddings work in my memory system?"

Expected:
- Loom skill assesses complexity
- Selects appropriate strategy
- Retrieves relevant memories
- Synthesizes answer with provenance
```

---

## Understanding HTTP vs STDIO

**Your question: "How do I get an HTTP MCP endpoint? What is it?"**

### Answer: You Don't Need HTTP (STDIO Works!)

**What you have:**
- HoloLoom MCP server uses **STDIO** (Standard Input/Output)
- This is the **recommended** way for local MCP servers
- Claude Desktop launches it as a subprocess

**What STDIO means:**
```
Claude Desktop starts:
  python -m HoloLoom.memory.mcp_server

Claude Desktop communicates via:
  stdin/stdout (like piping commands)

Benefits:
  ✅ Simple setup (no ports/networking)
  ✅ Automatic process management
  ✅ No security concerns (local only)
  ✅ Recommended by Anthropic
```

**When you'd need HTTP:**
```
Remote server (access from multiple machines)
Cloud deployment
Web-based integrations

For local Claude Desktop use → STDIO is better
```

---

## Architecture at a Glance

```
You type query in Claude Desktop
        ↓
Claude loads appropriate skill (spinning-wheel or loom)
        ↓
Skill provides workflow guidance
        ↓
Claude calls MCP tools (holoLoom-memory:*)
        ↓
MCP server (STDIO) processes request
        ↓
Data stored in Neo4j + Qdrant
        ↓
Claude receives result
        ↓
Skill formats response
        ↓
You see answer with citations
```

---

## Example Workflows

### Workflow 1: Research a Topic

```
1. You: "Ingest all Claude API docs"
   → SpinningWheel: Recursive crawl (~30 pages)

2. You: "What did I learn about agents?"
   → Loom: Retrieve with temporal strategy

3. You: "How do agent skills work?"
   → Loom: Retrieve with semantic strategy

4. You: "Synthesize best practices for agents"
   → Loom: Fused strategy with synthesis
```

### Workflow 2: Process Your Notes

```
1. You: "Process ~/org/notes.org"
   → SpinningWheel: Org-mode chunking

2. You: "What are my notes about mirrorCore?"
   → Loom: Semantic search

3. You: "Show all my project ideas"
   → Loom: Pattern matching
```

### Workflow 3: Browser History Mining

```
1. You: "Ingest my Chrome history from this week about AI"
   → SpinningWheel: Filter + batch process

2. You: "What domains did I research most?"
   → Loom: Aggregate statistics

3. You: "Summarize my AI research"
   → Loom: Cross-domain synthesis
```

---

## Common Issues

### Issue: "Skill not found"

**Solution:**
- Check `~/.claude/skills/spinning-wheel/SKILL.md` exists
- Check `~/.claude/skills/loom/SKILL.md` exists
- Restart Claude Desktop

### Issue: "MCP server connection failed"

**Solution:**
```bash
# Test server manually
python -m HoloLoom.memory.mcp_server

# Check PYTHONPATH
echo $PYTHONPATH  # Should include c:\Users\blake\Documents\mythRL

# Verify imports work
python -c "from HoloLoom.memory import protocol"
```

### Issue: "Memory backend unavailable"

**Solution:**
```bash
# Check Neo4j
neo4j status
# If down: neo4j start

# Check Qdrant
docker ps | grep qdrant
# If down: docker run -p 6333:6333 -d qdrant/qdrant

# Then restart Claude Desktop
```

### Issue: "Tool call failed"

**Solution:**
- Check tool name is correct (e.g., `holoLoom-memory:recall_memories`)
- Check parameters are valid JSON
- Look for error in Claude Desktop console logs

---

## What's Next

### After Basic Testing

1. **Run evaluations** (test cases in `evaluations/`)
2. **Try complex queries** (synthesis, cross-domain)
3. **Build your knowledge graph** (ingest docs, notes, research)

### Feature Exploration

- Try different retrieval strategies (temporal, semantic, graph, fused)
- Use conversational interface (`chat` tool)
- Process long documents (`process_text` tool)
- Crawl documentation sites (recursive crawler)

### Advanced Usage

- Customize importance thresholds (conversational filtering)
- Adjust matryoshka scales (performance tuning)
- Build custom retrieval patterns
- Create domain-specific ingestion workflows

---

## Getting Help

**Documentation:**
- [Full Architecture](HOLOLOOM_CLAUDE_DESKTOP_ARCHITECTURE.md)
- [SpinningWheel README](~/.claude/skills/spinning-wheel/README.md)
- [Loom SKILL.md](~/.claude/skills/loom/SKILL.md)

**Workflows:**
- [Website Ingestion](~/.claude/skills/spinning-wheel/workflows/website.md)
- [Memory Recall](~/.claude/skills/loom/workflows/recall_workflow.md)

**Examples:**
- [SpinningWheel Examples](~/.claude/skills/spinning-wheel/examples/example_usage.md)

---

## Success Checklist

- [ ] Skills directory exists: `~/.claude/skills/`
- [ ] Config copied to Claude Desktop config location
- [ ] Neo4j running
- [ ] Qdrant running
- [ ] Claude Desktop restarted
- [ ] Skills respond to "What can you do?"
- [ ] MCP health check works
- [ ] Can ingest webpage
- [ ] Can retrieve memories
- [ ] Conversational filtering works

**All checked?** You're ready to build your knowledge graph! 🎉

---

## Quick Reference

### SpinningWheel Commands

```
"Add [URL] to memory"
"Process my [browser] history from [timeframe]"
"Ingest [file path]"
"Crawl [documentation site]"
"Process this text: [paste]"
```

### Loom Commands

```
"What did I learn about [topic]?"
"How does [X] relate to [Y]?"
"Synthesize insights about [topic]"
"What was I working on [timeframe]?"
"Show memories from [domain/source]"
```

### MCP Tools (Direct)

```
holoLoom-memory:recall_memories
holoLoom-memory:store_memory
holoLoom-memory:process_text
holoLoom-memory:ingest_webpage
holoLoom-memory:chat
holoLoom-memory:memory_health
holoLoom-memory:conversation_stats
```

---

**Ready? Restart Claude Desktop and try: "What can spinning-wheel do?"**
