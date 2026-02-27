# HoloLoom Lite

**The simplest way to use HoloLoom - 5 methods, zero dependencies, safety by default.**

> **Philosophy**: Every user should have access to safe AI. HoloLoom Lite removes all barriers.

## Quick Start

```bash
pip install hololoom

# Start interactive REPL
python -m HoloLoom.lite repl
```

Or in Python:

```python
from HoloLoom.lite import HoloLoomLite

async with HoloLoomLite() as loom:
    # Learn something
    await loom.experience("Thompson Sampling balances exploration and exploitation")

    # Remember it
    memories = await loom.recall("What is Thompson Sampling?")

    # Ask questions
    result = await loom.query("Explain Thompson Sampling")
    print(result.response)
```

## 5 Core Methods

| Method | Purpose | Example |
|--------|---------|---------|
| `experience()` | Store memories | `await loom.experience("New knowledge")` |
| `recall()` | Retrieve memories | `await loom.recall("query", limit=5)` |
| `reflect()` | Learn from feedback | `await loom.reflect(memories, {"helpful": True})` |
| `reason()` | Multi-step reasoning | `await loom.reason("complex question", mode="verify")` |
| `query()` | Get answers | `await loom.query("simple question")` |

## Safety by Default

Every HoloLoom Lite instance includes safety guardrails:

```python
# Safety is ON by default
loom = HoloLoomLite()  # enable_safety=True

# Check safety before risky actions
result = await loom.check_safety("execute_code", {"code": "import os"})
if result.safe:
    # Proceed
else:
    print(f"Blocked: {result.reason}")
```

**Safety features**:
- Risk-based action gating (LOW/MEDIUM/HIGH/CRITICAL)
- Deception detection
- Instrumental convergence prevention
- Complete audit trail

## UI Modes

Run Lite in different interfaces:

```bash
# Interactive REPL (default)
python -m HoloLoom.lite repl

# Rich terminal UI
python -m HoloLoom.lite terminal

# Web interface (requires fastapi, uvicorn)
python -m HoloLoom.lite web

# Desktop app (requires tkinter)
python -m HoloLoom.lite desktop
```

## Claude Desktop Integration (MCP)

HoloLoom Lite includes an MCP server for Claude Desktop:

```bash
# Start MCP server
python -m HoloLoom.lite.mcp_server
```

Add to Claude Desktop config (`~/.claude/config.json`):

```json
{
  "mcpServers": {
    "hololoom": {
      "command": "python",
      "args": ["-m", "HoloLoom.lite.mcp_server"]
    }
  }
}
```

**Available tools in Claude Desktop**:
- `hololoom_experience` - Store memories
- `hololoom_recall` - Search memories
- `hololoom_query` - Ask questions
- `hololoom_reason` - Complex reasoning

## OpenAI/Anthropic Tool Integration

Use Lite as tools in your AI applications:

```python
from HoloLoom.lite import get_tools_for_openai, execute_tool

# Get tool definitions
tools = get_tools_for_openai()

# In your OpenAI function calling loop
if tool_call.function.name.startswith("hololoom_"):
    result = await execute_tool(
        tool_call.function.name,
        json.loads(tool_call.function.arguments)
    )
```

## Configuration

```python
from HoloLoom.lite import HoloLoomLite, LiteConfig

config = LiteConfig(
    # Memory settings
    persist=False,          # In-memory by default
    persist_path="./data",  # Where to persist (if enabled)

    # Safety settings
    enable_safety=True,     # Safety guardrails (default: True)

    # Performance settings
    lazy_load=True,         # Load components on first use
    cache_size=1000,        # Query cache size
)

loom = HoloLoomLite(config=config)
```

### Persistence

By default, Lite uses in-memory storage. Enable persistence for data that survives restarts:

```python
# Enable persistence (uses SQLite)
loom = HoloLoomLite(persist=True)

# Or specify path
loom = HoloLoomLite(persist=True, persist_path="./my_data")
```

For production, use Neo4j + Qdrant:

```bash
# Start Docker services
docker-compose -f docker-compose.lite.yml up -d

# Use with Lite
loom = HoloLoomLite(persist=True, backend="hybrid")
```

## Comparison: Lite vs Full HoloLoom

| Feature | Lite | Full |
|---------|------|------|
| **Methods** | 5 | 50+ |
| **Size** | ~75% smaller | Full system |
| **Dependencies** | Optional | Required |
| **Default Storage** | In-memory | Configurable |
| **Safety** | Built-in | Built-in |
| **Federation** | Not included | Included |
| **Weaving Cycle** | Simplified | Full 9-step |
| **Use Case** | Personal, embedded | Production, enterprise |

## Upgrade Path

When you need more power, upgrade to full HoloLoom:

```python
# Lite
from HoloLoom.lite import HoloLoomLite

async with HoloLoomLite(persist=True) as loom:
    await loom.experience("My knowledge")

# Full HoloLoom (uses same data!)
from HoloLoom import HoloLoom

async with HoloLoom() as loom:
    # Your memories are here
    memories = await loom.recall("My knowledge")

    # Plus full features
    spacetime = await loom.weave(Query(text="Complex reasoning"))
```

## Integration Patterns

### Pattern 1: Personal Assistant

```python
from HoloLoom.lite import HoloLoomLite

async with HoloLoomLite() as loom:
    # Learn from conversations
    await loom.experience(user_message)

    # Recall relevant context
    memories = await loom.recall(current_topic)

    # Generate response with context
    result = await loom.query(
        question,
        context=[m.content for m in memories]
    )
```

### Pattern 2: Web Service with Auth

```python
from fastapi import FastAPI, Depends
from HoloLoom.lite import HoloLoomLite
from HoloLoom.saas import create_saas_backend
from HoloLoom.saas.auth import validate_api_key

app = FastAPI()
loom = HoloLoomLite(persist=True)
backend = create_saas_backend()

@app.post("/api/query")
async def query(
    question: str,
    auth = Depends(validate_api_key)
):
    result = await loom.query(question)
    return {"response": result.response}
```

See [Integration Strategy](../../docs/INTEGRATION_STRATEGY.md) for more patterns.

### Pattern 3: Claude Desktop Memory

Use Lite as persistent memory for Claude Desktop:

1. Start MCP server: `python -m HoloLoom.lite.mcp_server`
2. Add to Claude Desktop config
3. Claude can now remember across conversations!

## API Reference

### HoloLoomLite

```python
class HoloLoomLite:
    async def experience(
        self,
        content: str,
        context: Optional[Dict] = None
    ) -> Memory:
        """Store a memory. Returns Memory object."""

    async def recall(
        self,
        query: str,
        limit: int = 5
    ) -> List[Memory]:
        """Retrieve relevant memories."""

    async def reflect(
        self,
        memories: List[Memory],
        feedback: Optional[Dict] = None
    ) -> None:
        """Learn from feedback on memories."""

    async def reason(
        self,
        query: str,
        mode: str = "direct"  # "direct", "verify", "research"
    ) -> LiteResult:
        """Multi-step reasoning."""

    async def query(
        self,
        question: str,
        mode: str = "direct"
    ) -> LiteResult:
        """Get an answer (alias for reason)."""

    async def check_safety(
        self,
        action: str,
        context: Optional[Dict] = None
    ) -> LiteResult:
        """Check if an action is safe."""
```

### LiteResult

```python
@dataclass
class LiteResult:
    response: str           # The answer
    confidence: float       # 0.0 to 1.0
    reasoning_mode: str     # "direct", "verify", etc.
    sources: List[Memory]   # Source memories
    safe: bool              # Safety check passed
    metadata: Dict          # Additional info
```

### Memory

```python
@dataclass
class Memory:
    id: str                 # Unique identifier
    content: str            # Memory content
    timestamp: datetime     # When created
    relevance: float        # Relevance to query
    context: Dict           # Additional context
```

## Files

| File | Purpose | Lines |
|------|---------|-------|
| `core.py` | Main HoloLoomLite class | 718 |
| `__init__.py` | Package exports + CLI | 89 |
| `mcp_server.py` | Claude Desktop integration | ~200 |
| `openai_tools.py` | OpenAI function calling | ~150 |
| `ui/repl.py` | Interactive REPL | ~200 |
| `ui/terminal.py` | Rich terminal UI | ~300 |
| `ui/web.py` | FastAPI web interface | ~250 |
| `ui/desktop.py` | Tkinter desktop app | ~200 |

## Why Lite?

**Mission: Make AI Safe**

HoloLoom Lite exists to maximize adoption of safe AI:

1. **Zero friction** - No Docker, no databases, just `pip install`
2. **Safety by default** - Every user gets guardrails
3. **Gateway drug** - Easy start → upgrade to full HoloLoom
4. **Agent integrations** - Claude Desktop, OpenAI, Anthropic tools
5. **Ecosystem enabler** - Build apps on safe foundations

See [SAFETY.md](../../docs/SAFETY.md) for HoloLoom's safety methodology.

## Related Documentation

- [Integration Strategy](../../docs/INTEGRATION_STRATEGY.md) - How Lite fits the ecosystem
- [SaaS Toolkit](../saas/README.md) - Add auth to your Lite apps
- [Self-Hosting Guide](../../docs/self-hosting/README.md) - Production deployment
- [Full HoloLoom](../README.md) - When you need more power
