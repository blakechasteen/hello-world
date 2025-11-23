# Zero-G: A Gravity-Free Operating Environment for AI-Native Applications

**Version:** 1.0 MVP
**Status:** 🚧 In Development
**License:** TBD

---

## Overview

**Zero-G** is a dual-layered architecture that enables **safe, reversible data onboarding** for AI-native applications. It solves the brutal truth of modern data work:

> Most valuable data cannot be moved, is too fragile to reformat, or too mission-critical to risk touching.

Zero-G provides:
- **Zero-Move Access**: Data stays in place, only metadata moves
- **Formalized Lifecycle**: NASA-style launch sequence prevents accidents
- **Complete Provenance**: Full audit trails for compliance
- **Multi-App Interoperability**: Apps communicate via shared semantics

---

## Core Metaphors

Zero-G uses **dual metaphors** for conceptual coherence:

### 1. Loom (Technical Substrate)
- **WarpSpace**: Semantic retrieval (vector search)
- **YarnGraph**: Knowledge graph (entities, relationships)
- **ResonanceShed**: Multimodal fusion
- **ConvergenceEngine**: Decision planning
- **Rift**: Tool/API invocation
- **SpacetimeFabric**: Provenance tracking
- **ReflectionBuffer**: Learning and adaptation
- **ThreadSpinner**: Memory management

### 2. Spaceflight (User Experience)
- **Preflight**: Safety checks, permissions
- **Countdown**: T-10 to T-0 sequence
- **Lift-Off**: Non-invasive metadata scanning
- **Boosters**: Schema discovery, indexing
- **Orbit**: Stable operation
- **EVA (Space Walk)**: Manual intervention
- **Mission Control**: Monitoring, safety

---

## Quick Start

### Prerequisites

- **Python 3.9+**
- **Node.js 18+** (for frontend, optional)
- **Git**

### Run the MVP Demo

```bash
# 1. Clone repository
git clone <repo-url>
cd zero-g

# 2. Run the complete mission demo
cd examples
python simple_mission.py
```

That's it! The MVP has **zero external dependencies**. Everything uses Python standard library only.

### Expected Output

The demo will:
1. ✅ Execute Preflight checks
2. ✅ Run Launch Countdown (T-10 to T-0)
3. ✅ Perform Lift-Off (metadata scanning)
4. ✅ Engage Boosters (schema discovery)
5. ✅ Achieve Orbit (stable operation)
6. ✅ Execute sample queries using Loom Core
7. ✅ Land gracefully

---

## Project Structure

```
zero-g/
├── backend/                    # Python backend
│   ├── loom_core/             # Reasoning engine
│   │   ├── protocols.py       # Loom Core interfaces
│   │   └── simple_loom.py     # MVP implementation
│   ├── launch_system/         # Lifecycle orchestrator
│   │   ├── protocols.py       # Launch System interfaces
│   │   └── simple_orchestrator.py  # MVP state machine
│   ├── g_series/              # Data access layer
│   │   ├── protocols.py       # G-Series interfaces
│   │   └── g1_contact/        # G1 connectors
│   │       └── simple_json_connector.py  # JSON file connector
│   ├── mission_control/       # Observability
│   ├── apps/                  # App Orbit Layer
│   └── requirements.txt       # Python dependencies (empty for MVP!)
│
├── frontend/                   # React/TypeScript UI (future)
│   ├── cabin_ui/              # Launch dashboard
│   └── mission_control_ui/    # Mission Control dashboard
│
├── docs/                       # Comprehensive documentation
│   ├── ARCHITECTURE_OVERVIEW.md  # System architecture
│   ├── MVP_SCOPE.md           # What's in the MVP
│   ├── LAUNCH_SYSTEM_SPEC.md  # Launch sequence details
│   └── api/                   # API reference
│
├── examples/                   # Usage examples
│   ├── simple_mission.py      # ⭐ Complete demo (START HERE!)
│   └── data/                  # Sample data (auto-generated)
│
└── README.md                   # This file
```

---

## Key Features

### 1. Zero-Move Data Access

Data **never leaves its original location**. Zero-G accesses:
- ✅ Metadata only (file size, modification time)
- ✅ Schema inference from samples
- ✅ No unnecessary reads
- ✅ No mutations, no copies

### 2. Formalized Lifecycle

Every operation goes through a **NASA-style launch sequence**:
- **Preflight**: Safety checks prevent accidents
- **Countdown**: Subsystems initialize in order
- **Lift-Off**: Gradual, non-invasive operations
- **Abort**: Emergency rollback at any point

### 3. Complete Provenance

Every decision is logged via **SpacetimeFabric**:
- What was queried
- What was retrieved
- What decision was made
- What tools were executed
- Complete causal chains for debugging

### 4. Loom-Based Reasoning

Powered by **Loom Core**, a sophisticated reasoning engine:
- **WarpSpace**: Semantic search across knowledge
- **YarnGraph**: Structured entity relationships
- **ConvergenceEngine**: Intelligent decision making
- **Rift**: Safe tool execution

---

## MVP vs. Production

### What's Included in MVP

- ✅ Complete Loom Core (simplified implementations)
- ✅ Full Launch System state machine
- ✅ G1 JSON connector (zero-move access)
- ✅ Complete example demonstrating all stages
- ✅ Comprehensive documentation
- ✅ **Zero external dependencies**

### What's Coming Post-MVP

- 🔜 Real embeddings (Matryoshka multi-scale)
- 🔜 Additional connectors (HTTP APIs, SQL, NoSQL)
- 🔜 Frontend UI (React dashboard)
- 🔜 App Orbit Layer (Promptly, Elle, Trough integration)
- 🔜 EVA visual UI (manual graph manipulation)
- 🔜 Production deployment (Kubernetes, monitoring)

See [MVP_SCOPE.md](docs/MVP_SCOPE.md) for complete details.

---

## Documentation

Comprehensive docs in `/docs`:

- **[ARCHITECTURE_OVERVIEW.md](docs/ARCHITECTURE_OVERVIEW.md)** - System architecture
- **[MVP_SCOPE.md](docs/MVP_SCOPE.md)** - What's in scope for MVP
- **[LAUNCH_SYSTEM_SPEC.md](docs/LAUNCH_SYSTEM_SPEC.md)** - Launch sequence details
- **[Whitepaper](whitepaper)** - Complete vision (read this first!)

---

## Example Usage

### Run a Complete Mission

```bash
cd examples
python simple_mission.py
```

### Use Loom Core Directly

```python
from loom_core.simple_loom import create_simple_loom_core

async def main():
    # Create and initialize Loom Core
    loom = await create_simple_loom_core()

    # Execute a query
    result = await loom.weave("What is Thompson Sampling?")

    print(result['result'])

    # Shutdown
    await loom.shutdown()

import asyncio
asyncio.run(main())
```

### Use G1 JSON Connector

```python
from g_series.g1_contact.simple_json_connector import create_json_connector
from g_series.protocols import DataSensitivity

async def main():
    # Create connector
    connector = await create_json_connector("./data")

    # Connect to a JSON file (zero-move)
    health = await connector.connect(
        source_id="users",
        file_path="users.json",
        sensitivity=DataSensitivity.CONFIDENTIAL
    )

    # Get metadata (no read!)
    metadata = await connector.get_metadata("users")
    print(metadata)

    # Discover schema (minimal read)
    schema = await connector.discover_schema("users", sample_size=100)
    print(schema)

asyncio.run(main())
```

---

## Development

### Running Tests

```bash
# Backend tests (future)
cd backend
pytest

# Frontend tests (future)
cd frontend/cabin_ui
npm test
```

### Code Quality

```bash
# Format code
black backend/

# Type checking
mypy backend/

# Linting
ruff check backend/
```

---

## Architecture Highlights

### Loom Core

The heart of Zero-G's intelligence:

```
Query → WarpSpace (search) → YarnGraph (traverse) →
ConvergenceEngine (decide) → Rift (execute) →
SpacetimeFabric (log) → Result
```

### Launch System

State machine managing the complete lifecycle:

```
GROUNDED → PREFLIGHT → COUNTDOWN → LIFTOFF → BOOST → ORBIT
    ↓                                              ↑
    └────────────── ABORT (rollback) ─────────────┘
```

### G-Series

Staged data onboarding:

- **G1**: Contact (zero-move access)
- **G2**: Lift-Off (schema discovery)
- **G3**: Orbital Ops (streaming)
- **G4**: Space Walk (manual intervention)
- **G5+**: Legacy Repair (B2B tooling)

---

## Contributing

Zero-G is currently in MVP development. Contributions welcome after v1.0 release.

---

## Roadmap

### Phase 1: MVP (Current)
- ✅ Loom Core (simplified)
- ✅ Launch System (complete)
- ✅ G1 JSON connector
- ✅ Example demo
- ✅ Documentation

### Phase 1.1 (Weeks 5-6)
- Real embeddings (HoloLoom Matryoshka)
- HTTP API connector
- SQL connector
- Persistent storage (basic)

### Phase 1.2 (Weeks 7-8)
- App docking protocol
- Promptly integration
- Event bus
- Frontend UI (basic)

### Phase 2.0 (Months 3-6)
- Full HoloLoom integration
- Spindles (LightSpindle, SolidSpindle)
- Production deployment
- Enterprise features

---

## License

TBD

---

## Contact

Questions? Issues? Feedback?

- **GitHub Issues**: [Create an issue](https://github.com/...) (coming soon)
- **Documentation**: See `/docs`
- **Whitepaper**: See `whitepaper` for complete vision

---

**Status**: 🚀 MVP in active development
**Last Updated**: 2025-11-22
**Next Milestone**: Complete frontend UI (Week 2)
