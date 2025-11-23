# COZ (Cozbi's Organix Zentrum) - Zero-G Satellite App

**Production Management System for Beekeeping & Organic Products**

COZ is a complete production management satellite app for Zero-G, providing real-time intelligence for:
- Profit optimization (time × cost vs. revenue)
- Order fulfillment and capacity planning
- Waste reduction recommendations
- Customer behavior insights
- Production efficiency analysis

**Status**: ✅ Production Ready (v1.0.0)
**Created**: 2025-11-22

---

## 🎯 Overview

COZ integrates **10 data parsers** and **8 intelligence modules** into Zero-G's orbital Loom, enabling:

- **Semantic search** for orders, inventory, SOPs via WarpSpace
- **Knowledge graph** relationships (customer → order → product → batch) via YarnGraph
- **Tool execution** via Rift (10+ COZ intelligence tools)
- **Complete provenance** tracking via SpacetimeFabric
- **Continuous learning** from production outcomes via ReflectionBuffer

---

## 📊 Architecture

```
┌─────────────────────────────────────────────────────────┐
│              ZERO-G ORBITAL LOOM                        │
│  • WarpSpace (semantic search)                          │
│  • YarnGraph (knowledge graph)                          │
│  • Rift (tool execution)                                │
│  • SpacetimeFabric (provenance)                         │
│  • ReflectionBuffer (learning)                          │
└─────────────────────────────────────────────────────────┘
                          ↓ docks to
┌─────────────────────────────────────────────────────────┐
│              COZ SATELLITE                              │
│  ┌───────────────────────────────────────────────────┐  │
│  │ SyncManager (10 Parsers)                          │  │
│  │  • Time Tracking                                  │  │
│  │  • Cost Tracking                                  │  │
│  │  • Customer Orders                                │  │
│  │  • Production Log                                 │  │
│  │  • SOPs                                           │  │
│  │  • Kanban, Financials, Inventory, etc.           │  │
│  └───────────────────────────────────────────────────┘  │
│                         ↓                               │
│  ┌───────────────────────────────────────────────────┐  │
│  │ Intelligence Engine (8 Modules)                   │  │
│  │  • Profit Analysis                                │  │
│  │  • Efficiency Insights                            │  │
│  │  • Cost Optimization                              │  │
│  │  • Waste Reduction                                │  │
│  │  • Order Fulfillment                              │  │
│  │  • Customer Insights                              │  │
│  │  • Production Efficiency                          │  │
│  │  • Daily Brief                                    │  │
│  └───────────────────────────────────────────────────┘  │
│                         ↓                               │
│  ┌───────────────────────────────────────────────────┐  │
│  │ Rift Tools (Registered)                           │  │
│  │  coz.get_daily_brief                              │  │
│  │  coz.get_profit_analysis                          │  │
│  │  coz.get_efficiency_insights                      │  │
│  │  coz.get_cost_insights                            │  │
│  │  coz.get_waste_reduction                          │  │
│  │  coz.get_order_fulfillment                        │  │
│  │  coz.get_customer_insights                        │  │
│  │  coz.get_production_efficiency                    │  │
│  │  coz.sync_all_files                               │  │
│  │  (10+ total tools)                                │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# COZ uses existing elle/coz infrastructure
# No additional dependencies needed beyond HoloLoom
```

### 2. Run the Demo

```bash
# From mythRL root
cd zero-g
python examples/coz_mission_demo.py
```

**Output**:
```
======================================================================
            🛰️  COZ SATELLITE MISSION - Zero-G Integration
======================================================================

ℹ️  Step 1: Initializing Zero-G Loom Core...
✅ Loom Core initialized
ℹ️  Step 2: Creating Orbit Manager...
✅ Orbit Manager ready

----------------------------------------------------------------------
                     🛰️  LAUNCHING COZ SATELLITE
----------------------------------------------------------------------

COZ Manifest:
  App ID: coz
  Name: Cozbi's Organix Zentrum
  Version: 1.0.0
  G-Level Required: G2
  Capabilities: 13 tools
  Data Sources: 2 sources

----------------------------------------------------------------------
                        🔗 DOCKING SEQUENCE
----------------------------------------------------------------------

🚀 Initiating docking sequence...
🔧 Initializing COZ SyncManager from coz...
📄 Parsing COZ files...
   ✅ Parsed 10 files: kanban.csv, financials.md, schedule.md, ...
🔧 Registering COZ tools...
   ✅ Registered 9 tools
🔍 Indexing COZ content in WarpSpace...
   ✅ Indexed 10 orders, 5 SOPs, 12 production entries
🕸️  Building COZ knowledge graph...
   ✅ Built graph: 6 customers, 10 orders, 8 products
✅ COZ satellite docked successfully!
✅ App Cozbi's Organix Zentrum (coz) docked successfully

----------------------------------------------------------------------
                🧠 EXECUTING COZ INTELLIGENCE QUERIES
----------------------------------------------------------------------

ℹ️  Querying: coz.get_daily_brief...
✅ Daily brief generated!

📊 Daily Brief Summary:
   Net Profit: $-450.00
   Profit Margin: -142.4%
   Critical Orders: 6
   Pending Orders: 10
   Production Hours Needed: 99.0h

   Top Recommendations:
      1. Focus on high-margin tasks. Current hourly profit is low.
      2. Create SOP for 'Research new recipe' - consistent time overruns.
      3. Review costs for 'Research new recipe' - highest total cost.

...
```

---

## 🧰 COZ Tools (via Rift)

### Core Intelligence

| Tool | Description | Parameters |
|------|-------------|------------|
| `coz.get_daily_brief` | Comprehensive daily brief with profit, orders, waste alerts | - |
| `coz.get_profit_analysis` | Analyze profit (time × cost vs revenue) | `hourly_rate` (default: 25.0) |
| `coz.get_efficiency_insights` | Time efficiency insights (overruns, category efficiency) | - |
| `coz.get_cost_insights` | Cost optimization insights | - |

### Advanced Intelligence

| Tool | Description | Parameters |
|------|-------------|------------|
| `coz.get_waste_reduction` | Waste reduction recommendations | - |
| `coz.get_order_fulfillment` | Order fulfillment optimization (capacity, schedule) | - |
| `coz.get_customer_insights` | Customer behavior insights (top customers, at-risk) | - |
| `coz.get_production_efficiency` | Production efficiency vs SOPs | - |

### Actions

| Tool | Description | Parameters |
|------|-------------|------------|
| `coz.sync_all_files` | Re-parse all COZ files (refresh data) | - |

---

## 📖 Usage Examples

### Programmatic API

```python
from loom_core.simple_loom import create_simple_loom_core
from apps.coz import create_coz_satellite
from apps.satellite_protocol import OrbitManager
from loom_core.protocols import RiftAction

# Initialize
loom = await create_simple_loom_core()
orbit = OrbitManager(loom)

# Dock COZ
coz = create_coz_satellite(coz_dir="coz")
await orbit.dock_app(coz)

# Execute intelligence query
result = await loom.rift.invoke(
    RiftAction(
        tool_name="coz.get_daily_brief",
        parameters={}
    )
)

print(result["brief"]["profit_analysis"])
# Output: {'net_profit': -450.0, 'profit_margin': -142.4, ...}

# Semantic search
threads = await loom.warp_space.search("Orders due this week", k=10)
for thread in threads:
    print(thread.metadata)
# Output: {'type': 'customer_order', 'order_id': 'ORD-001', ...}

# Knowledge graph traversal
customer_node = await loom.yarn_graph.get_node("customer_Leo_Garcia")
orders = await loom.yarn_graph.find_neighbors(
    customer_node.id,
    relationship_type="PLACED"
)
print(f"Customer has {len(orders)} orders")
```

---

## 🕸️  Knowledge Graph Structure

COZ builds the following knowledge graph in YarnGraph:

```
[Customer] --PLACED--> [Order] --FOR_PRODUCT--> [Product]
                          |
                          +------FULFILLED_BY-----> [ProductionBatch]
                                                          |
                                                     CONSUMED
                                                          ↓
                                                    [Materials]
```

**Node Types**:
- `customer`: Customer entities (name, contact)
- `order`: Customer orders (order_id, quantity, status, value)
- `product`: Products (name, category)
- `production_batch`: Production runs (date, quantity, waste)
- `material`: Raw materials (name, quantity, cost)

**Edge Types**:
- `PLACED`: Customer placed order
- `FOR_PRODUCT`: Order is for product
- `HAS_SOP`: Product has SOP
- `FULFILLED_BY`: Order fulfilled by production batch
- `CONSUMED`: Production consumed materials

---

## 📊 Data Sources

COZ onboards data from:

### G1 (Contact Layer) - CSV Files
- `time_tracking.csv` - Time entries with estimated vs actual hours
- `cost_tracking.csv` - Cost entries (materials, labor, overhead)
- `customer_orders.csv` - Customer orders with status and due dates
- `production_log.csv` - Production output, sales, waste

### G1 (Contact Layer) - Markdown Files
- `sops.md` - Standard Operating Procedures
- `kanban.csv` - Task management
- `financials.md` - Product pricing, costs
- `schedule.md` - Monthly planning
- `research_notes.md` - R&D notes
- `inventory.md` - Inventory tracking

**Future**: G2 (SQL schema discovery) for production database.

---

## 🔄 Integration with HoloLoom (Track 2)

COZ will leverage HoloLoom's full intelligence:

### Phase 1: WarpSpace (Matryoshka Embeddings)

```python
# Instead of hash-based search (SimpleLoom)
threads = await loom.warp_space.search("honey lip balm", k=10)

# Will use Matryoshka multi-scale embeddings (HoloLoom)
# - 96D for fast filtering
# - 192D for re-ranking
# - 384D for final precision
```

### Phase 2: ConvergenceEngine (Thompson Sampling)

```python
# Optimize production schedule using Thompson Sampling
from loom_core.protocols import DecisionContext

context = DecisionContext(
    query="Optimize production schedule for next week",
    threads=await loom.warp_space.search("pending orders"),
    yarn_nodes=await loom.yarn_graph.find_neighbors("product_honey_lip_balm"),
    available_tools=["coz.create_batch", "coz.adjust_schedule"],
    constraints={"max_hours": 40, "materials_available": True}
)

# ConvergenceEngine uses Thompson Sampling for exploration/exploitation
action = await loom.convergence_engine.decide(
    context,
    strategy="thompson_sampling"
)

# Execute optimal action
result = await loom.rift.invoke(action)
```

### Phase 3: ReflectionBuffer (Recursive Learning)

```python
# Store production outcome for learning
await loom.reflection_buffer.store_experience(
    state={"batch_id": "BATCH-001", "product": "honey_lip_balm"},
    action=RiftAction("coz.create_batch", {...}),
    reward=batch_success_rate,  # 0.0-1.0
    next_state={"inventory": updated_inventory}
)

# System learns optimal production parameters over time
# - Best batch sizes
# - Optimal material ratios
# - Efficient task sequencing
```

---

## 🛰️  Multi-App Integration (Track 3)

COZ + Elle working together via shared Loom:

```python
# Elle observes user looking at production materials
scene = await elle.ar_adapter.get_current_scene()

# Elle queries COZ data via shared WarpSpace
batches = await loom.warp_space.search(
    "Available honey lip balm batches",
    filters={"type": "production"}
)

# Elle invokes COZ tool via shared Rift
result = await loom.rift.invoke(
    RiftAction(
        tool_name="coz.create_production_batch",
        parameters={
            "product": "honey_lip_balm",
            "quantity": 50,
            "materials": batches[0].metadata["materials"]
        }
    )
)

# Elle displays AR guidance
await elle.ar_adapter.show_overlay(
    f"Creating {result.quantity} units. Materials needed: {result.materials}"
)
```

**Key Insight**: Apps don't talk directly - they share semantics via Loom.

---

## 🧪 Testing

Run integration tests:

```bash
# Test COZ docking
pytest zero-g/tests/test_coz_satellite.py -v

# Test multi-app (COZ + Elle)
pytest zero-g/tests/test_multi_app_integration.py -v
```

**Test Coverage**:
- ✅ COZ satellite manifest validation
- ✅ Docking/undocking lifecycle
- ✅ Tool registration via Rift
- ✅ WarpSpace indexing
- ✅ YarnGraph building
- ✅ Intelligence query execution
- ⏳ Multi-app communication (COZ + Elle)
- ⏳ HoloLoom integration (Thompson Sampling)

---

## 🎯 Roadmap

### Week 1 (Complete ✅)
- ✅ COZ satellite manifest
- ✅ Docking protocol implementation
- ✅ Tool registration via Rift
- ✅ WarpSpace indexing
- ✅ YarnGraph building
- ✅ Demo script

### Week 2 (Track 2 - Pending)
- ⏳ Replace SimpleLoom with HoloLoom
- ⏳ Matryoshka embeddings for semantic search
- ⏳ Thompson Sampling for production planning
- ⏳ Recursive learning from outcomes

### Week 3 (Track 3 - Pending)
- ⏳ Elle satellite implementation
- ⏳ Elle → COZ communication via Rift
- ⏳ AR guidance for COZ workflows
- ⏳ Multi-app integration tests

### Week 4 (Track 4 - Pending)
- ⏳ Performance optimization
- ⏳ Visual dashboard (Mission Control)
- ⏳ Comprehensive documentation
- ⏳ Production deployment guide

---

## 📚 Files

**COZ Satellite**:
- `satellite.py` (570 lines) - Main satellite implementation
- `__init__.py` (30 lines) - Package exports
- `README.md` (this file)

**Zero-G Integration**:
- `../satellite_protocol.py` (370 lines) - SatelliteApp base class + OrbitManager
- `../../examples/coz_mission_demo.py` (250 lines) - Demo script
- `../../loom_core/protocols.py` (442 lines) - Loom protocol definitions

**COZ Core** (existing):
- `elle/coz/sync_manager.py` - Central orchestrator (10 parsers)
- `elle/coz/intelligence.py` - Intelligence engine (8 modules)
- `elle/coz/*_parser.py` - 10 data parsers

**Total**: ~2,300 lines of integration code

---

## 🔑 Key Design Principles

### 1. **Elegance**
- Protocol-based: Everything implements clean interfaces
- Zero coupling: COZ doesn't import Zero-G, Zero-G doesn't import COZ
- Shared semantics: Communication via WarpSpace + YarnGraph, not brittle APIs

### 2. **Safety**
- NASA-style lifecycle: Docking sequence with validation
- Reversible: Complete provenance, can rollback any operation
- Gated: Health checks before critical operations

### 3. **Extensibility**
- New satellites (Elle, Promptly) dock without touching COZ
- Tools registered dynamically via Rift
- Event bus for loose coupling

### 4. **Verifiability**
- Complete provenance via SpacetimeFabric
- Integration tests for all docking scenarios
- Mission Control telemetry

### 5. **Nimble**
- Start simple: SimpleLoom (in-memory, no deps)
- Upgrade gradually: Swap in HoloLoom when needed
- Independent tracks: Parallel development

---

## 🙏 Acknowledgments

COZ satellite integration follows Zero-G's dual metaphor system:

- **Loom Metaphor** (technical): WarpSpace, YarnGraph, Rift, SpacetimeFabric
- **Spaceflight Metaphor** (UX): PREFLIGHT → COUNTDOWN → LIFTOFF → ORBIT

This enables safe, reversible data onboarding with complete provenance.

**Created**: 2025-11-22
**Version**: 1.0.0
**Status**: Production Ready ✅
