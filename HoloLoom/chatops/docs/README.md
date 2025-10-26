# HoloLoom Matrix Bot

**AI-powered chatops for Matrix with full HoloLoom integration**

Complete Matrix bot that brings HoloLoom's weaving orchestrator, MCTS decision-making, and hybrid memory system into your Matrix chat rooms.

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

## Commands

### Core Commands

#### !weave <query>
Execute full weaving cycle with MCTS decision-making.

**Example:**
!weave Explain Thompson Sampling in MCTS

**Response:**
- Tool decision with confidence
- Duration and context shards
- MCTS simulations used
- Result preview

#### !memory add <text>
Add knowledge to HoloLoom memory.

#### !memory search <query>
Search memory semantically.

#### !memory stats
Show memory statistics.

#### !analyze <text>
Analyze text with MCTS.

#### !stats
Show HoloLoom system statistics.

#### !help
Show command help.

#### !ping
Health check.

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

## License

MIT License - See main HoloLoom LICENSE

---

**Matrix Bot Integration COMPLETE**
