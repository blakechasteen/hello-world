# HoloLoom Demos

Canonical examples demonstrating HoloLoom functionality.

## Quick Start

All demos run without external dependencies using the 'simple' memory backend.

```bash
# From repository root
PYTHONPATH=. python demos/01_quickstart.py
```

---

## Available Demos

### 01_quickstart.py
**Simplest usage: text in, questions answered**

- Load text into memory
- Query with natural language
- Get contextual responses
- No external dependencies

**Run:** `PYTHONPATH=. python demos/01_quickstart.py`

---

### 02_web_to_memory.py
**Complete web scraping → memory storage pipeline**

- Scrape webpages with WebsiteSpinner
- Generate MemoryShards
- Store in unified memory (Neo4j + Qdrant)
- Query and retrieve

**Features:**
- Image extraction (optional)
- Entity recognition
- Recursive crawling (bonus demo)
- Multiple memory backends

**Run:** `PYTHONPATH=. python demos/02_web_to_memory.py`

---

### 03_conversational.py
**Chat interface with automatic memory formation**

- Back-and-forth conversation
- Automatic importance scoring
- Signal vs noise filtering
- Memory formation from significant exchanges

**Features:**
- Pre-defined conversation
- Interactive mode
- Conversation statistics
- Memory analysis

**Run:** `PYTHONPATH=. python demos/03_conversational.py`

---

### 04_mcp_integration.py
**Model Context Protocol integration guide**

- MCP server setup instructions
- Claude Desktop configuration
- Available MCP tools
- Testing procedures

**Shows:**
- Memory storage via MCP
- Search via MCP
- Conversational interface via MCP
- Configuration examples

**Run:** `python demos/04_mcp_integration.py`

---

## Demo Progression

1. **Start with 01_quickstart.py** - Understand basic usage
2. **Try 02_web_to_memory.py** - Learn data ingestion
3. **Explore 03_conversational.py** - See auto-memory in action
4. **Setup 04_mcp_integration.py** - Connect to Claude Desktop

---

## Memory Backends

All demos default to 'simple' memory (in-memory, no persistence).

For production, configure:

- **Neo4j**: Graph database for entities/relationships
- **Qdrant**: Vector database for semantic search
- **Neo4j+Qdrant**: Hybrid (best quality)

See `HoloLoom/memory/QUICKSTART.md` for setup.

---

## Execution Modes

HoloLoom has three execution modes:

| Mode | Speed | Quality | Use Case |
|------|-------|---------|----------|
| BARE | Fastest | Basic | Quick iterations |
| FAST | Balanced | Good | Default, recommended |
| FUSED | Slowest | Highest | Production, high quality |

Change mode in demos:
```python
from HoloLoom.config import Config

# Use FUSED mode instead of FAST
config = Config.fused()
orchestrator = await AutoSpinOrchestrator.from_text(text, config=config)
```

---

## Next Steps

After running demos:

1. **Read architecture**: `CLAUDE.md` - System design
2. **Review sprint plan**: `INTEGRATION_SPRINT.md` - Roadmap
3. **Check subsystems**:
   - `HoloLoom/spinningWheel/` - Data ingestion
   - `HoloLoom/memory/` - Storage systems
   - `HoloLoom/policy/` - Decision making

4. **Advanced usage**: See `HoloLoom/examples/` for more

---

## Troubleshooting

**Import errors:**
```bash
# Always run from repository root
cd /path/to/mythRL
PYTHONPATH=. python demos/01_quickstart.py
```

**Module not found:**
```bash
# Check you're in the right directory
pwd  # Should end with /mythRL

# Check HoloLoom exists
ls HoloLoom/
```

**Memory backend errors:**
- Default 'simple' backend needs no setup
- For Neo4j/Qdrant, see `HoloLoom/memory/NEO4J_README.md`

---

## Contributing

Want to add a demo?

1. Follow naming: `NN_descriptive_name.py`
2. Include docstring with overview
3. Use clear step-by-step structure
4. Default to 'simple' memory backend
5. Add entry to this README

See `CLAUDE.md` for development guidelines.
