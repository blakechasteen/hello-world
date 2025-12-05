# Portal: Distributed Compute Control Plane

Portal is a 4-component distributed system for personal compute orchestration:

```
┌─────────────────────────────────────────────────────────────────┐
│                         Matrix Room                              │
│                    (Human ↔ Shuttle Bot)                        │
└─────────────────────────────┬───────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│                        Shuttle Bot                               │
│        Parses !loom, !nodes, !job commands                      │
└─────────────────────────────┬───────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│                       Portal Server                              │
│              Node registry, Loom status                          │
└─────────────────────────────┬───────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
        ┌─────────┐     ┌─────────┐     ┌─────────┐
        │ Node 1  │     │ Node 2  │     │ Node 3  │
        │  WASM   │     │  WASM   │     │  WASM   │
        └─────────┘     └─────────┘     └─────────┘
```

## Quick Start

### 1. Install Dependencies

```bash
pip install fastapi uvicorn pydantic pyyaml httpx psutil
pip install wasmtime  # For WASM execution
pip install matrix-nio  # For Shuttle bot
```

### 2. Start Portal Server

```bash
# From repository root
PYTHONPATH=. python -m HoloLoom.portal.portal_server.main

# Or with custom config
PYTHONPATH=. python -m HoloLoom.portal.portal_server.main --config HoloLoom/portal/configs/portal.yaml
```

Portal will start on `http://localhost:8080`

### 3. Start Node Daemon

```bash
# On each compute node
PYTHONPATH=. python -m HoloLoom.portal.node_daemon.main

# With custom node ID
PYTHONPATH=. python -m HoloLoom.portal.node_daemon.main --node-id laptop-1 --port 9091
```

Node will register with Portal and start sending heartbeats.

### 4. Start Shuttle Bot (Optional)

First, edit `configs/shuttle.yaml` with your Matrix homeserver details:

```yaml
homeserver: "https://matrix.example.com"
user_id: "@shuttle:example.com"
password: "your-bot-password"
room_id: "!your-room-id:example.com"
```

Then start the bot:

```bash
PYTHONPATH=. python -m HoloLoom.portal.shuttle_bot.main --config HoloLoom/portal/configs/shuttle.yaml
```

## Matrix Commands

Once Shuttle is running, use these commands in your Matrix room:

| Command | Description |
|---------|-------------|
| `!help` | Show help |
| `!loom status` | Show Loom overview |
| `!nodes` | List all nodes |
| `!modules` | List available WASM modules |
| `!job run add {"a":2,"b":3}` | Run a WASM job |

## API Reference

### Portal Server (`:8080`)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/nodes/register` | POST | Register a node |
| `/nodes/{id}/heartbeat` | POST | Node heartbeat |
| `/nodes` | GET | List all nodes |
| `/nodes/{id}` | GET | Get specific node |
| `/loom/status` | GET | Loom overview |

All endpoints (except `/health`) require `X-Shared-Secret` header.

### Node Daemon (`:9090`)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/status` | GET | Node status with metrics |
| `/jobs` | POST | Submit a job |
| `/jobs/{id}` | GET | Get job result |
| `/modules` | GET | List WASM modules |

## WASM Modules

WASM modules are stored in `wasm_modules/` with a companion manifest:

```
wasm_modules/
├── add.wasm       # WASM binary
├── add.json       # Manifest
├── echo.wasm
└── echo.json
```

### Manifest Format

```json
{
  "id": "add-v1",
  "name": "Simple Adder",
  "version": "1.0.0",
  "entry_function": "run",
  "input_schema": {...},
  "output_schema": {...}
}
```

### Mock Mode

If `wasmtime` is not installed, the node runs in mock mode with built-in modules:
- `add` / `add-v1`: Adds `a + b`
- `echo` / `echo-v1`: Returns input unchanged
- `multiply` / `multiply-v1`: Multiplies `a * b`

## Configuration

### Environment Variables

**Portal Server:**
- `PORTAL_LOOM_ID` - Loom identifier
- `PORTAL_HOST` - Bind host
- `PORTAL_PORT` - Bind port
- `PORTAL_SHARED_SECRET` - Auth secret

**Node Daemon:**
- `NODE_ID` - Node identifier
- `NODE_HOST` - Bind host
- `NODE_PORT` - Bind port
- `NODE_PORTAL_URL` - Portal URL
- `NODE_SHARED_SECRET` - Auth secret

**Shuttle Bot:**
- `SHUTTLE_HOMESERVER` - Matrix homeserver
- `SHUTTLE_USER_ID` - Bot user ID
- `SHUTTLE_PASSWORD` - Bot password
- `SHUTTLE_ROOM_ID` - Room to join
- `SHUTTLE_PORTAL_URL` - Portal URL

## Development

### Running Tests

```bash
pytest HoloLoom/portal/tests/ -v
```

### Project Structure

```
portal/
├── shared/           # Shared types and utilities
│   ├── types.py      # Pydantic models
│   └── logging.py    # Structured logging
├── portal_server/    # Control plane
│   ├── main.py       # FastAPI app
│   ├── registry.py   # Node registry
│   └── config.py     # Configuration
├── node_daemon/      # WASM executor
│   ├── main.py       # FastAPI app
│   ├── wasm_runner.py
│   ├── module_registry.py
│   └── config.py
├── shuttle_bot/      # Matrix ChatOps
│   ├── main.py       # Bot entry
│   ├── commands.py   # Command handlers
│   ├── scheduler.py  # Node picker
│   └── config.py
├── configs/          # Example configs
├── wasm_modules/     # WASM modules
└── tests/            # Test suite
```

## Future Roadmap

See the original plan for future enhancements:
- PKI/mTLS authentication
- Content-addressed WASM storage
- Multi-loom federation
- Load balancing and job queues
- GPU compute support
