# Zero-G MVP Mission Walkthrough

This document provides a detailed walkthrough of the complete MVP mission from Preflight through Landing.

---

## Mission Overview

**Objective**: Onboard local JSON files and demonstrate the complete Zero-G lifecycle

**Data Sources**:
- `users.json` (5 users with names, emails, roles)
- `products.json` (4 products with names, prices, categories)

**Duration**: ~30 seconds

**Stages**: Preflight → Countdown → Lift-Off → Boost → Orbit → Landing

---

## Step-by-Step Walkthrough

### Step 0: System Initialization

```python
# Initialize Loom Core
loom = await create_simple_loom_core()
```

**What happens**:
1. WarpSpace initializes (semantic retrieval)
2. YarnGraph initializes (knowledge graph)
3. ResonanceShed initializes (multimodal fusion)
4. ConvergenceEngine initializes (decision making)
5. Rift initializes (tool execution)
6. SpacetimeFabric initializes (provenance logging)
7. ReflectionBuffer initializes (learning)
8. ThreadSpinner initializes (memory paging)

**Output**:
```
WarpSpace: Initializing...
WarpSpace: Initialized successfully
YarnGraph: Initializing...
YarnGraph: Initialized successfully
...
SimpleLoomCore: All subsystems initialized successfully
```

**Duration**: ~800ms

---

### Step 1: Launch Orchestrator Creation

```python
orchestrator = await create_simple_launch_orchestrator(loom)
```

**What happens**:
1. Preflight system created
2. Countdown system created
3. Lift-off system created
4. Boost system created
5. Orbit system created
6. EVA system created
7. Mission Control created

**Current Stage**: `GROUNDED`

---

### Step 2: G1 Connector Setup

```python
connector = await create_json_connector("./data")
await connector.connect("users", "users.json", DataSensitivity.INTERNAL)
await connector.connect("products", "products.json", DataSensitivity.INTERNAL)
```

**What happens**:
1. JSON connector initialized with base path
2. Connect to `users.json` (zero-move - no read)
3. Connect to `products.json` (zero-move - no read)

**Output**:
```
G1 JSON Connector: Connecting to users.json
G1 JSON Connector: Connecting to products.json
users.json: healthy
products.json: healthy
```

**Zero-Move Principle**: Files are **not read** at this stage. Only connection verification.

---

### Step 3: Metadata Retrieval (Zero-Move)

```python
users_meta = await connector.get_metadata("users")
products_meta = await connector.get_metadata("products")
```

**What happens**:
1. File stats retrieved via OS (no read)
2. File size, modification time, permissions gathered
3. Metadata cached

**Output**:
```json
{
  "file_name": "users.json",
  "file_size_bytes": 412,
  "file_size_human": "412 B",
  "modified_time": "2025-11-22T21:30:00",
  "is_readable": true,
  "access_mode": "zero_move",
  "sensitivity": "internal"
}
```

**Zero-Move Principle**: File **content not read**. Only OS metadata accessed.

---

### Step 4: PREFLIGHT

```python
mission_params = {
    "astronaut_id": "demo_astronaut",
    "permissions": {"read": True, "write": False},
    "environment": {"network": "local", "storage": "disk"},
    "data_sources": ["users.json", "products.json"]
}

preflight_report = await orchestrator.execute_preflight(mission_params)
```

**What happens**:

#### 4.1 Suit Fitting
- Verify astronaut ID: `demo_astronaut`
- Check permissions: `read=True, write=False`
- Validate access controls

**Output**:
```
Preflight: Running suit fitting for demo_astronaut
✅ Suit fitting complete. All permissions verified.
```

#### 4.2 Astronaut Training
- Introduce key concepts: WarpSpace, YarnGraph, Zero-Move Access
- Confirm astronaut familiarity

**Output**:
```
Preflight: Running training for demo_astronaut
✅ Training complete. Astronaut familiar with: WarpSpace, YarnGraph, Zero-Move Access
```

#### 4.3 Physical Examination
- Check connection stability: `local network`
- Verify environment health: `disk storage`
- Ensure system readiness

**Output**:
```
Preflight: Running physical examination
✅ All systems nominal. Connection stable. Environment healthy.
```

#### 4.4 Mission Briefing
- Data sources to access: `users.json`, `products.json`
- Access mode: `Zero-move` (metadata only)
- Safety guarantees: `No writes, no copies, rollback-capable`

**Output**:
```
Preflight: Running mission briefing
✅ Mission briefing complete. Will access 2 data sources. Zero-move protocols in effect.
```

#### 4.5 Preflight Report
```
✅ Preflight Status: GREEN
Can proceed: True
```

**Current Stage**: `PREFLIGHT`

---

### Step 5: COUNTDOWN (T-10 to T-0)

```python
countdown_status = await orchestrator.execute_countdown()
```

**What happens**:

```
T-10: Final authorization request confirmed
T-9:  Atmospheric telemetry stable
T-8:  Zero-copy channels warming
T-7:  Thread anchors confirmed
T-6:  WarpSpace initialized
T-5:  YarnGraph scaffolding detected
T-4:  ResonanceShed fusion online
T-3:  Rift safety locks engaged
T-2:  ConvergenceEngine synchronized
T-1:  Mission Control releases hold
T-0:  🔥 IGNITION - Lift-Off!
```

**Duration**: ~3 seconds (300ms per event)

**Current Stage**: `COUNTDOWN`

---

### Step 6: LIFTOFF

```python
liftoff_check = await orchestrator.execute_liftoff()
```

**What happens**:

#### 6.1 Metadata Scanning
- Scan metadata from `users.json` and `products.json`
- No actual data read - only stats

**Output**:
```
Lift-Off: Scanning metadata from 2 sources
Metadata scanned: 2 sources
```

#### 6.2 Graph Preview
- Infer initial graph structure from metadata
- Discover potential nodes and relationships
- Estimate graph complexity

**Output**:
```
Lift-Off: Generating graph preview
Graph preview: 42 nodes discovered, 87 relationships inferred
Confidence: 0.85
```

#### 6.3 WarpSpace Alignment
- Prepare semantic retrieval layer
- No indexing yet - just alignment

**Output**:
```
Lift-Off: Aligning WarpSpace
✅ WarpSpace aligned successfully
```

**Current Stage**: `LIFTOFF`

---

### Step 7: Schema Discovery (Boost Preparation)

```python
users_schema = await connector.discover_schema("users", sample_size=100)
products_schema = await connector.discover_schema("products", sample_size=100)
```

**What happens**:
1. **First actual data read** (limited to sample)
2. Parse JSON structure
3. Infer field names and types
4. Detect nested structures

**Output**:
```json
{
  "source_id": "users",
  "elements": [
    {
      "element_id": "users_root",
      "element_type": "json_structure",
      "name": "root",
      "properties": {
        "type": "array",
        "item_schema": {
          "type": "object",
          "fields": {
            "id": "int",
            "name": "str",
            "email": "str",
            "role": "str"
          }
        },
        "consistent": true,
        "sample_size": 5
      }
    }
  ],
  "conflicts": [],
  "warnings": []
}
```

---

### Step 8: BOOST

```python
boost_check = await orchestrator.execute_boost()
```

**What happens**:

#### 8.1 Schema Inference
- Infer complete schema from both sources
- Analyze field types and relationships

**Output**:
```
Boost: Inferring schema from 2 sources
Schema inference complete
```

#### 8.2 Conflict Detection
- Check for schema conflicts across sources
- Detect type mismatches, naming inconsistencies

**Output**:
```
Boost: Detecting schema conflicts
Conflicts detected: 0
```

#### 8.3 Indexing Plan
- Plan hybrid indexing strategy (keyword + semantic)
- Estimate time and memory requirements

**Output**:
```
Boost: Planning indexing strategy
Indexing plan: hybrid
Estimated time: 120 seconds
Memory required: 512 MB
```

#### 8.4 Indexing Execution
- Execute the indexing plan
- Index all discovered records

**Output**:
```
Boost: Executing indexing
✅ Indexing complete. System ready for orbit.
```

**Current Stage**: `BOOST`

---

### Step 9: ORBIT

```python
orbit_status = await orchestrator.achieve_orbit()
```

**What happens**:
1. Transition to stable orbit
2. Enable streaming mode (if applicable)
3. Activate app docking
4. Stabilize yarn (knowledge graph)
5. Align warp (semantic layer)

**Output**:
```
Orbit: Achieving stable orbit...
🚀 ORBIT ACHIEVED - Zero-G is live!

Orbital Status:
  Altitude: G3_OPERATIONAL
  Apps docked: 0
  Streaming active: False
  Yarn stable: True
  Warp aligned: True
  Uptime: 0.0 seconds
```

**Checkpoint Created**: Rollback point established

**Current Stage**: `ORBIT`

---

### Step 10: Query Execution (In Orbit)

```python
result = await loom.weave("What is Thompson Sampling?")
```

**What happens**:

#### 10.1 Query Received
- SpacetimeFabric logs query event
- Timestamp: Start

#### 10.2 Input Fusion (ResonanceShed)
- Fuse multimodal inputs (text-only in MVP)
- Create unified thread

#### 10.3 Semantic Search (WarpSpace)
- Search for relevant threads
- Return top 5 matches

#### 10.4 Graph Traversal (YarnGraph)
- Traverse graph for context (if applicable)

#### 10.5 Decision Making (ConvergenceEngine)
- Analyze query: "What is Thompson Sampling?"
- Decide tool: `answer` (factual question)

#### 10.6 Tool Execution (Rift)
- Execute `answer` tool with parameters
- Generate response

#### 10.7 Provenance Logging (SpacetimeFabric)
- Log complete execution trace
- Calculate duration

#### 10.8 Learning (ReflectionBuffer)
- Store experience for future learning
- Update metrics

**Output**:
```json
{
  "query": "What is Thompson Sampling?",
  "action": "answer",
  "result": "Response to: What is Thompson Sampling?\n\nThis is a simulated response from the Loom Core MVP. In production, this would use full HoloLoom reasoning.",
  "duration_ms": 42.3,
  "threads_consulted": 5
}
```

---

### Step 11: Multiple Queries

The demo executes 3 queries to demonstrate:
1. Different tool selection (answer, search, analyze)
2. Provenance tracking
3. Performance monitoring

---

### Step 12: LANDING

```python
landing_check = await orchestrator.land()
```

**What happens**:
1. Save current state
2. Archive logs to SpacetimeFabric
3. Close data source connections
4. Graceful subsystem shutdown

**Output**:
```
=== LANDING SEQUENCE ===
✅ Landing complete - Mission successful
Landing Status: GREEN
```

**Current Stage**: `LANDING`

---

### Step 13: Cleanup

```python
await connector.disconnect("users")
await connector.disconnect("products")
await loom.shutdown()
```

**What happens**:
1. Disconnect from data sources
2. Clear metadata cache
3. Shutdown Loom subsystems

**Output**:
```
🧹 Cleaning up...
G1 JSON Connector: Disconnecting users
G1 JSON Connector: Disconnecting products
SimpleLoomCore: Shutting down...
✅ Cleanup complete
```

---

## Mission Summary

**Total Duration**: ~30 seconds

**Stages Completed**:
1. ✅ System Initialization (8 subsystems)
2. ✅ Data Source Connection (G1 zero-move)
3. ✅ Preflight (4 checks)
4. ✅ Countdown (T-10 to T-0)
5. ✅ Lift-Off (metadata scanning)
6. ✅ Boost (schema discovery, indexing)
7. ✅ Orbit (stable operation)
8. ✅ Query Execution (3 queries)
9. ✅ Landing (graceful shutdown)

**Zero-Move Verification**:
- ✅ Initial connection: No data read
- ✅ Metadata retrieval: OS stats only
- ✅ Schema discovery: Sample read only (limited)
- ✅ No writes performed
- ✅ All operations reversible

**Safety Features Demonstrated**:
- ✅ Preflight safety checks
- ✅ Checkpoint creation (rollback point)
- ✅ Complete provenance logging
- ✅ Graceful error handling
- ✅ Clean shutdown

**Loom Core Capabilities**:
- ✅ Semantic retrieval (WarpSpace)
- ✅ Knowledge graph (YarnGraph)
- ✅ Decision making (ConvergenceEngine)
- ✅ Tool execution (Rift)
- ✅ Provenance tracking (SpacetimeFabric)
- ✅ Learning (ReflectionBuffer)

---

## What's Next?

**Try These Demos**:
1. EVA Mission: `python demo_eva_mission.py` (manual graph tuning)
2. App Docking: `python demo_app_docking.py` (multi-app interop)
3. Streaming: `python demo_streaming.py` (G3 real-time)

**Explore the Code**:
- Loom Core: `backend/loom_core/protocols.py`
- Launch System: `backend/launch_system/protocols.py`
- G-Series: `backend/g_series/protocols.py`

**Read the Docs**:
- Architecture: `docs/ARCHITECTURE_OVERVIEW.md`
- Launch System: `docs/LAUNCH_SYSTEM_SPEC.md`
- G-Series: `docs/G_SERIES_OVERVIEW.md`

---

**Mission Complete. Zero-G is operational. 🚀**
