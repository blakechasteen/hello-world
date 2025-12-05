# Zero-G MVP Scope

**Version:** 1.0
**Target Completion**: 2025-12-15
**Status:** In Development

## Overview

This document defines the **Minimum Viable Product (MVP)** scope for Zero-G. The MVP demonstrates the complete vertical slice of functionality from data source connection through query execution, while leaving extension points for future capabilities.

---

## MVP Goals

1. **Demonstrate Core Concept**: Show that Zero-G's dual metaphor (Loom + Spaceflight) works
2. **Prove Zero-Move Access**: Access data without moving it
3. **Complete Lifecycle**: Execute a full mission from Preflight through Orbit
4. **Buildable and Runnable**: Anyone can clone, build, and run the MVP
5. **Clear Extension Points**: Easy to add more connectors, apps, and capabilities

---

## In Scope (MVP)

### 1. Loom Core (Minimal)

**Implementation**: `SimpleLoomCore` in `/backend/loom_core/simple_loom.py`

**Components Included**:

- ✅ **SimpleWarpSpace**
  - In-memory thread storage
  - Basic embedding (deterministic hash for MVP, not real embeddings)
  - Recency-based search (no actual vector similarity)
  - Thread indexing

- ✅ **SimpleYarnGraph**
  - In-memory NetworkX-style graph
  - Node and edge storage
  - Basic neighbor finding
  - Simple path finding (BFS)

- ✅ **SimpleResonanceShed**
  - Text-only fusion (no actual multimodal for MVP)
  - Pass-through temporal alignment

- ✅ **SimpleConvergenceEngine**
  - Rule-based decision making
  - Simple tool selection (keyword matching)
  - Single-step planning

- ✅ **SimpleRift**
  - Three default tools: answer, search, analyze
  - Canned responses for MVP
  - Tool registry pattern

- ✅ **SimpleSpacetimeFabric**
  - In-memory event logging
  - Trace retrieval
  - Causal chain construction

- ✅ **SimpleReflectionBuffer**
  - Experience storage
  - Batch sampling
  - Metrics tracking

- ✅ **SimpleThreadSpinner**
  - Hot/cold classification
  - Page in/out operations
  - Recency-based access tracking

**Not Included** (future):
- ❌ Real embedding models (use HoloLoom's Matryoshka later)
- ❌ Actual vector similarity search
- ❌ True multimodal fusion
- ❌ MCTS or Thompson Sampling (just rule-based)
- ❌ Persistent storage (all in-memory for MVP)

---

### 2. Launch System (Complete)

**Implementation**: `SimpleLaunchOrchestrator` in `/backend/launch_system/simple_orchestrator.py`

**Stages Included**:

- ✅ **GROUNDED**: Initial state
- ✅ **PREFLIGHT**: All 4 checks (suit fitting, training, physical exam, briefing)
- ✅ **COUNTDOWN**: Complete T-10 to T-0 sequence
- ✅ **LIFTOFF**: Metadata scanning, graph preview, warp alignment
- ✅ **BOOST**: Schema inference, indexing
- ✅ **ORBIT**: Stable operation, app docking
- ✅ **EVA**: Airlock cycling, heddle tensioning
- ✅ **LANDING**: Graceful shutdown
- ✅ **ABORT**: Emergency rollback

**Features**:

- ✅ State machine with proper transitions
- ✅ Safety checks at each stage
- ✅ Checkpoint creation for rollback
- ✅ Telemetry collection
- ✅ Logging with visual output

---

### 3. G-Series (G1 Only)

**Implementation**: `SimpleJSONConnector` in `/backend/g_series/g1_contact/simple_json_connector.py`

**G1 Capabilities**:

- ✅ **Zero-Move Access**
  - Connect to local JSON files
  - Metadata-only reading (file size, mod time)
  - No unnecessary data reads

- ✅ **Schema Discovery**
  - Infer JSON structure
  - Sample-based schema inference
  - Field type detection

- ✅ **Summaries**
  - Count summary (minimal read)
  - Schema summary (partial read)
  - Sample summary (limited read)

- ✅ **Health Checks**
  - Connection status
  - Error tracking
  - Latency monitoring

**Not Included** (future):
- ❌ G1.2-G1.4 (MCP, API layer, security) - stubs only
- ❌ G2-G5 stages - stubs only
- ❌ Other connectors (HTTP, DB, streaming) - future
- ❌ Encryption, region locks, audit logs - future

---

### 4. Frontend (Basic)

**Implementation**: React app in `/frontend/cabin_ui/`

**Screens Included**:

- ✅ **Launch Dashboard**
  - Current stage indicator
  - Countdown display (T-minus)
  - Safety status (green/yellow/red)
  - Mission parameters

- ✅ **Mission Control Dashboard**
  - Telemetry display
  - System health indicators
  - Error log
  - Abort button

**Features**:

- ✅ Voice-first design (text for MVP, voice in future)
- ✅ NASA-style countdown visualization
- ✅ Real-time WebSocket updates (future)
- ✅ Responsive layout

**Not Included** (future):
- ❌ Full EVA visualization (manual heddle UI)
- ❌ Voice narration (text only for MVP)
- ❌ 3D visualizations
- ❌ Advanced query interface

---

### 5. App Orbit Layer (Demo Apps)

**Implementation**: Demo apps in `/backend/apps/`

**Apps Included**:

- ✅ **Demo Chat App**
  - Simple Q&A interface
  - Uses Loom Core for reasoning
  - Demonstrates docking protocol

- ✅ **Demo Data Explorer**
  - Browse connected data sources
  - View schema previews
  - Simple query interface

**Not Included** (future):
- ❌ Promptly integration
- ❌ Elle integration
- ❌ Trough integration
- ❌ Full pub/sub event bus

---

### 6. Documentation (Complete)

**Files Included**:

- ✅ **ARCHITECTURE_OVERVIEW.md** - System architecture
- ✅ **MVP_SCOPE.md** - This document
- ✅ **LAUNCH_SYSTEM_SPEC.md** - Launch sequence details
- ✅ **API_REFERENCE.md** - API documentation
- ✅ **DEPLOYMENT_GUIDE.md** - How to run locally
- ✅ **EXAMPLES.md** - Usage examples

---

### 7. Examples and Demos

**Scripts Included**:

- ✅ **examples/simple_mission.py**
  - Complete mission from start to finish
  - Demonstrates all lifecycle stages
  - Commented for learning

- ✅ **examples/json_connector_demo.py**
  - G1 JSON connector usage
  - Zero-move access demonstration
  - Schema discovery

- ✅ **examples/loom_core_demo.py**
  - Loom Core reasoning demonstration
  - Query execution
  - Provenance tracking

---

## Out of Scope (Post-MVP)

### Loom Core Enhancements

- ❌ Real Matryoshka embeddings (96/192/384D)
- ❌ True Thompson Sampling in ConvergenceEngine
- ❌ MCTS planning
- ❌ Actual multimodal fusion (audio, video, sensors)
- ❌ Persistent storage (Neo4j, Qdrant)
- ❌ HoloLoom integration (full reasoning engine)

### Launch System Enhancements

- ❌ Advanced EVA UI (manual graph manipulation)
- ❌ Real-time telemetry dashboard
- ❌ Distributed deployment
- ❌ Kubernetes orchestration
- ❌ Production monitoring (Prometheus, Grafana)

### G-Series Expansion

- ❌ G1.2: MCP connector framework
- ❌ G1.3: REST/GraphQL/gRPC API layer
- ❌ G1.4: Full encryption and audit
- ❌ G2: Advanced schema discovery
- ❌ G3: Streaming and hot/cold paging
- ❌ G4: Complete EVA heddle system
- ❌ G5+: Legacy repair tools

### Additional Connectors

- ❌ HTTP/REST APIs
- ❌ SQL databases (Postgres, MySQL)
- ❌ NoSQL databases (MongoDB, Redis)
- ❌ Cloud storage (S3, Azure Blob, GCS)
- ❌ Streaming (Kafka, Websocket)
- ❌ Legacy systems (FTP, SFTP)

### App Orbit Layer

- ❌ Promptly integration
- ❌ Elle integration
- ❌ Trough integration
- ❌ Full event bus (pub/sub)
- ❌ Session management
- ❌ App marketplace

### Spindles (Future Systems)

- ❌ LightSpindle (image generation)
- ❌ SolidSpindle (3D generation)
- ❌ WorldSpindle (simulation)

---

## MVP Success Criteria

The MVP is considered successful if:

1. ✅ **Runnable**: Can be cloned and run on a local machine
2. ✅ **Complete Lifecycle**: Can execute Preflight → Countdown → Liftoff → Boost → Orbit
3. ✅ **Zero-Move Demo**: Can connect to a JSON file and extract metadata without reading the full file
4. ✅ **Query Execution**: Can accept a query and return a response using Loom Core
5. ✅ **State Machine**: Launch System correctly transitions through all stages
6. ✅ **Rollback**: Can abort and rollback to a previous checkpoint
7. ✅ **Documentation**: All major concepts are documented
8. ✅ **Extension Points**: Clear TODOs and hooks for future expansion

---

## MVP Timeline

| Week | Milestone | Deliverables |
|------|-----------|--------------|
| **Week 1** | Backend Scaffolding | Loom Core + Launch System + G1 JSON |
| **Week 2** | Frontend UI | Launch Dashboard + Mission Control |
| **Week 3** | Integration | End-to-end mission flow working |
| **Week 4** | Polish & Docs | Examples, documentation, bug fixes |

**Target Completion**: Week 4, 2025-12-15

---

## Post-MVP Roadmap

After MVP, prioritize based on user feedback:

**Phase 1.1** (Weeks 5-6):
- Additional connectors (HTTP API, SQL)
- Real embeddings (HoloLoom Matryoshka)
- Persistent storage (basic)

**Phase 1.2** (Weeks 7-8):
- App docking protocol (full)
- Promptly integration
- Event bus

**Phase 1.3** (Weeks 9-10):
- EVA UI (manual heddle tensioning)
- G2-G3 capabilities (schema discovery, streaming)
- Production deployment guide

**Phase 2.0** (Months 3-6):
- Full HoloLoom integration
- Spindles (LightSpindle, SolidSpindle)
- Enterprise features (multi-tenancy, SSO)

---

## Getting Started with MVP

To run the MVP:

```bash
# 1. Clone repository
git clone <repo-url>
cd zero-g

# 2. Install dependencies
cd backend
pip install -r requirements.txt

cd ../frontend/cabin_ui
npm install

# 3. Run example mission
cd ../../examples
python simple_mission.py

# 4. Run frontend (separate terminal)
cd ../frontend/cabin_ui
npm start
```

See [DEPLOYMENT_GUIDE.md](./guides/DEPLOYMENT_GUIDE.md) for detailed instructions.

---

**Document Status**: ✅ Complete
**Last Updated**: 2025-11-22
**Next Review**: After MVP completion
