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
PYTHONPATH=. python -m hololoom.portal.portal_server.main

# Or with custom config
PYTHONPATH=. python -m hololoom.portal.portal_server.main --config hololoom/portal/configs/portal.yaml
```

Portal will start on `http://localhost:8080`

### 3. Start Node Daemon

```bash
# On each compute node
PYTHONPATH=. python -m hololoom.portal.node_daemon.main

# With custom node ID
PYTHONPATH=. python -m hololoom.portal.node_daemon.main --node-id laptop-1 --port 9091
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
PYTHONPATH=. python -m hololoom.portal.shuttle_bot.main --config hololoom/portal/configs/shuttle.yaml
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
pytest hololoom/portal/tests/ -v
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

## Docker Quick Start (Recommended)

The easiest way to run Portal is with Docker Compose:

### 1. Start All Services

```bash
cd hololoom/portal
docker-compose up -d
```

This starts:
- **Portal Server** - http://localhost:8080
- **Node Daemon 1** - http://localhost:9091
- **Node Daemon 2** - http://localhost:9092

### 2. Verify Health

```bash
curl http://localhost:8080/health
curl http://localhost:9091/health
curl http://localhost:9092/health
```

### 3. Run End-to-End Demo

```bash
pip install httpx
python demo_e2e.py
```

### 4. Submit a Test Job

```bash
curl -X POST http://localhost:9091/jobs \
  -H "Content-Type: application/json" \
  -H "X-Shared-Secret: portal-demo-secret-2025" \
  -d '{
    "job_id": "test-001",
    "module_id": "mock-add",
    "entry_function": "run",
    "input_json": {"a": 5, "b": 3},
    "timeout_seconds": 30
  }'
```

### 5. Stop Services

```bash
docker-compose down
```

### Build Images Manually

```bash
# From repository root
docker build --target portal-server -t portal-server:latest -f hololoom/portal/Dockerfile .
docker build --target node-daemon -t node-daemon:latest -f hololoom/portal/Dockerfile .
```

### View Logs

```bash
docker-compose logs -f              # All services
docker-compose logs -f portal-server # Portal only
```

## Production Hardening (Phase 1 Complete)

The following production features have been implemented:

| Feature | Status | File |
|---------|--------|------|
| Load Balancer | ✅ Complete | `shared/load_balancer.py` |
| Type System | ✅ Complete | `shared/types.py` |
| Prometheus Metrics | ✅ Complete | `shared/metrics.py` |
| PKI/mTLS | ✅ Complete | `shared/pki.py` |
| CAS Storage | ✅ Complete | `shared/cas_storage.py` |
| Integration Tests | ✅ Complete | `shared/tests/` |

**Total: ~3,100 lines of production code**

### Load Balancing Strategies

Configure via `PORTAL_LB_STRATEGY`:

- **`round_robin`**: Rotate through nodes sequentially
- **`least_loaded`**: Pick node with fewest current jobs (default)
- **`weighted`**: Score by capacity / current load
- **`capability`**: Match job requirements to node capabilities

### Content-Addressed Storage

WASM modules are stored by SHA256 hash for deduplication:

```python
from hololoom.portal.shared.cas_storage import CASStorage

storage = CASStorage(Path("./cas"))
content_hash = storage.store(wasm_bytes, "my-module", "v1.0.0")
# Identical content stored only once
```

## Future Roadmap

See the original plan for future enhancements:
- ~~PKI/mTLS authentication~~ ✅ Done
- ~~Content-addressed WASM storage~~ ✅ Done
- ~~Load balancing and job queues~~ ✅ Done
- Multi-loom federation
- GPU compute support
