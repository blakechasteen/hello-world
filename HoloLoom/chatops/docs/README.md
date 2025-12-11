# HoloLoom Matrix Bot

**AI-powered chatops for Matrix with full HoloLoom integration**

Complete Matrix bot that brings HoloLoom's weaving orchestrator, MCTS decision-making, and hybrid memory system into your Matrix chat rooms.

**Total Commands: 69** (December 2025)

---

## Quick Start

### Installation

```bash
# Install Matrix dependencies
pip install matrix-nio[e2e]

# Ensure HoloLoom is installed
pip install -e .
```

### Basic Usage

```bash
# Using environment variables
export MATRIX_HOMESERVER="https://matrix.org"
export MATRIX_USER="@mybot:matrix.org"
export MATRIX_PASSWORD="secret"

python HoloLoom/chatops/run_bot.py

# Or with command line arguments
python HoloLoom/chatops/run_bot.py \
  --homeserver https://matrix.org \
  --user @mybot:matrix.org \
  --password secret \
  --hololoom-mode fast
```

---

## Commands Reference

### Core Commands (8)

| Command | Description |
|---------|-------------|
| `!weave <query>` | Execute full weaving cycle with MCTS decision-making |
| `!trace [query_id]` | Show reasoning trace for a query |
| `!learn <feedback>` | Provide feedback for learning |
| `!stats` | Show HoloLoom system statistics |
| `!analyze <text>` | Analyze text with MCTS |
| `!memory add <text>` | Add knowledge to HoloLoom memory |
| `!memory search <query>` | Search memory semantically |
| `!memory stats` | Show memory statistics |

#### Example: !weave
```
!weave Explain Thompson Sampling in MCTS
```
**Response:**
- Tool decision with confidence
- Duration and context shards
- MCTS simulations used
- Result preview

---

### Testing Commands (6)

| Command | Description |
|---------|-------------|
| `!test run [path]` | Run tests (optional path filter) |
| `!test status` | Show test runner status |
| `!test coverage` | Show code coverage report |
| `!test benchmark` | Run performance benchmarks |
| `!test ci` | Run full CI pipeline |
| `!test help` | Show testing help |

#### Example: !test run
```
!test run HoloLoom/tests/unit/
```

---

### Code Commands (8)

| Command | Description |
|---------|-------------|
| `!code query <question>` | Ask questions about the codebase |
| `!code refactor <file>` | Suggest refactoring for a file |
| `!code explain <file:line>` | Explain code at specific location |
| `!code test <file>` | Generate tests for a file |
| `!code fix <file>` | Suggest fixes for issues |
| `!code context` | Show current code context |
| `!code status` | Show code analysis status |
| `!code help` | Show code command help |

#### Example: !code explain
```
!code explain HoloLoom/policy/unified.py:150
```

---

### RAG Commands (5)

| Command | Description |
|---------|-------------|
| `!rag query <question>` | Query with retrieval-augmented generation |
| `!rag ingest <source>` | Ingest document or URL into RAG |
| `!rag search <query>` | Search RAG knowledge base |
| `!rag stats` | Show RAG statistics |
| `!rag help` | Show RAG command help |

#### Example: !rag query
```
!rag query What are the tradeoffs of Thompson Sampling?
```

---

### Agentic Commands (5)

| Command | Description |
|---------|-------------|
| `!research <topic>` | Multi-query exploration mode (RESEARCH) |
| `!verify <claim>` | Cross-check verification mode (VERIFY) |
| `!plan <goal>` | Goal decomposition mode (PLAN_EXECUTE) |
| `!reason <query>` | Standard reasoning mode (DIRECT) |
| `!agentic help` | Show agentic reasoning help |

#### Example: !research
```
!research Compare all exploration-exploitation strategies
```

---

### Visualization Commands (7)

| Command | Description |
|---------|-------------|
| `!dashboard confidence` | Show confidence trajectory chart |
| `!dashboard cache` | Show cache effectiveness gauge |
| `!dashboard waterfall` | Show stage timing waterfall |
| `!dashboard knowledge` | Show knowledge graph network |
| `!dashboard rag` | Show RAG performance dashboard |
| `!dashboard help` | Show dashboard help |
| `!dashboard reset` | Reset dashboard metrics |

#### Example: !dashboard confidence
```
!dashboard confidence
```
Shows confidence over last N queries with anomaly detection.

---

### Memory Symphony Commands (5)

| Command | Description |
|---------|-------------|
| `!memory strategy [name]` | Get/set memory coordination strategy |
| `!memory metrics` | Show memory system metrics |
| `!memory systems` | List all 7 memory systems and status |
| `!memory history [limit]` | Show memory access history |
| `!memory help` | Show memory symphony help |

#### Example: !memory systems
```
!memory systems
```
Lists: Vector Memory, Knowledge Graph, Query Cache, Hot Patterns, Awareness Graph, Spring Dynamics, Multi-Wave Engine.

---

### Temporal Commands (4)

| Command | Description |
|---------|-------------|
| `!temporal travel <timestamp>` | View memory state at past time |
| `!temporal between <start> <end>` | Query memories in time range |
| `!temporal patterns` | Detect temporal patterns |
| `!temporal help` | Show temporal query help |

#### Example: !temporal travel
```
!temporal travel 2025-12-01T10:00:00
```

---

### Department Commands (5)

| Command | Description |
|---------|-------------|
| `!dept list` | List all departments |
| `!dept status [name]` | Show department status |
| `!dept process <request>` | Process through department |
| `!dept capabilities [name]` | Show department capabilities |
| `!dept help` | Show department help |

#### Example: !dept list
```
!dept list
```
Lists: Quality Assurance, Analytics, Context, Infrastructure, Memory.

---

### Feedback Commands (3)

| Command | Description |
|---------|-------------|
| `!feedback stats` | Show Thompson Sampling feedback statistics |
| `!feedback process <event>` | Process feedback event manually |
| `!feedback help` | Show feedback help |

#### Example: !feedback stats
```
!feedback stats
```
Shows reaction counts, success rates, confidence improvements.

---

### Ingestion Commands (8)

| Command | Description |
|---------|-------------|
| `!ingest <source>` | Auto-detect and ingest any source |
| `!ingest youtube <url>` | Ingest YouTube video transcript |
| `!ingest pdf <url/path>` | Ingest PDF document |
| `!ingest url <url>` | Ingest web page content |
| `!ingest git <repo>` | Ingest Git repository |
| `!ingest image <url/path>` | Ingest image with OCR/CLIP |
| `!ingest status` | Show ingestion job status |
| `!ingest help` | Show ingestion help |

#### Example: !ingest youtube
```
!ingest youtube https://youtube.com/watch?v=VIDEO_ID
```

---

### Conversation Commands (2)

| Command | Description |
|---------|-------------|
| `!continue [text]` | Continue previous query with optional context |
| `!context [--full]` | Show current conversation context |

#### Example: !continue
```
!weave What is Thompson Sampling?
...
!continue Tell me more about the Bayesian aspects
```
Continues the conversation with accumulated context.

---

### Cluster Commands (4)

| Command | Description |
|---------|-------------|
| `!cluster status` | Show cluster health overview |
| `!cluster nodes` | List all connected nodes |
| `!cluster balance` | Show load distribution |
| `!cluster help` | Show cluster help |

#### Example: !cluster status
```
!cluster status
```
Shows: Active nodes, average load, total tasks, uptime.

---

### Utility Commands (4)

| Command | Description |
|---------|-------------|
| `!help` | Show command help |
| `!ping` | Health check |
| `!version` | Show HoloLoom version |
| `!config` | Show current configuration |

---

## Command Categories Summary

| Category | Commands | Handler File |
|----------|----------|--------------|
| **Core** | 8 | `hololoom_handlers.py` |
| **Testing** | 6 | `test_handlers.py` |
| **Code** | 8 | `code_handlers.py` |
| **RAG** | 5 | `rag_handlers.py` |
| **Agentic** | 5 | `agentic_handlers.py` |
| **Visualization** | 7 | `visualization_handlers.py` |
| **Memory Symphony** | 5 | `memory_symphony_handlers.py` |
| **Temporal** | 4 | `temporal_handlers.py` |
| **Departments** | 5 | `department_handlers.py` |
| **Feedback** | 3 | `feedback_handler.py` |
| **Ingestion** | 8 | `ingestion_handlers.py` |
| **Conversation** | 2 | `conversation_handlers.py` |
| **Cluster** | 4 | `cluster_handlers.py` |
| **Utility** | 4 | `handler_registry.py` |
| **Total** | **69** | |

---

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| MATRIX_HOMESERVER | Matrix server URL | https://matrix.org |
| MATRIX_USER | Bot user ID | Required |
| MATRIX_PASSWORD | Bot password | Required (or use token) |
| MATRIX_ACCESS_TOKEN | Access token | Alternative to password |
| MATRIX_ROOMS | Comma-separated room IDs | None (accepts invites) |

### Command Line Arguments

**Connection:**
- --homeserver URL - Matrix homeserver URL
- --user USER_ID - Bot user ID
- --password PASS - Bot password
- --token TOKEN - Access token

**HoloLoom Configuration:**
- --hololoom-mode {bare,fast,fused} - Execution mode (default: fast)
- --mcts-sims N - MCTS simulations per decision (default: 50)

**System:**
- --log-level {DEBUG,INFO,WARNING,ERROR} - Logging level
- --store-path PATH - Matrix encryption keys path

---

## HoloLoom Modes

### BARE Mode
Fastest, minimal features (~50ms per query)

### FAST Mode (Default)
Balanced performance and quality (~150ms per query)

### FUSED Mode
Highest quality, comprehensive analysis (~300ms per query)

---

## Deployment

### Systemd Service (Linux)

Create /etc/systemd/system/hololoom-bot.service

### Docker

```bash
docker build -t hololoom-bot .
docker run -d --name hololoom-bot hololoom-bot
```

---

## Memory Backends

- **File-Only (Default)** - No external dependencies
- **Qdrant + File** - Vector database with fallback
- **Full Hybrid** - Qdrant + Neo4j + File

The bot automatically uses the best available backend with graceful degradation.

---

## Adding New Commands

To add a new command handler:

```python
from HoloLoom.chatops.handlers import chatops_handler, HandlerCategory

@chatops_handler(
    command="mycommand",
    description="My custom command",
    category=HandlerCategory.QUERY,
    usage="!mycommand <arg>"
)
async def handle_mycommand(args: str, room_id: str, sender: str) -> str:
    """Handle my custom command."""
    return f"Result: {args}"
```

Register in `__init__.py` with graceful degradation:
```python
try:
    from HoloLoom.chatops.handlers.my_handlers import (
        register_my_handlers,
        handle_mycommand
    )
    __all__.extend(["register_my_handlers", "handle_mycommand"])
except ImportError:
    pass
```

---

## License

MIT License - See main HoloLoom LICENSE

---

**Documentation Updated: December 2025**
**Total Commands: 69**
