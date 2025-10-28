# HoloLoom Farm Management Suite

**AI-powered farm management apps built on HoloLoom's neural decision-making infrastructure**

---

## Overview

This is a **modular farm management suite** inspired by farmOS, built on HoloLoom's semantic memory and neural decision engine. Unlike traditional farm software, these apps:

- **Understand natural language** - "Which hives need inspection?" instead of SQL queries
- **Learn from patterns** - Recognizes seasonal cycles, correlates weather with yields
- **Multimodal ingestion** - Voice notes, photos, receipts via domain-specific spinners
- **Semantic connections** - Links related activities across different farm domains
- **Decision support** - Suggests actions based on historical patterns

---

## Architecture

### Inspired by farmOS

| farmOS Concept | HoloLoom Component | Implementation |
|----------------|-------------------|----------------|
| **Assets** | Neo4j Entities | Hives, animals, fields, equipment |
| **Logs** | MemoryShards | Inspections, treatments, harvests |
| **Taxonomy** | Motifs & Tags | Pattern recognition markers |
| **Areas** | Graph relationships | LOCATED_IN edges |
| **Query** | WeavingOrchestrator | Natural language interface |

### Core Components

```
apps/
├── farm_core/              # Shared framework
│   ├── models.py           # Asset & Log base classes
│   └── tracker.py          # FarmTracker with HoloLoom integration
│
├── beekeeping/             # Apiary management
│   ├── models.py           # Hive, Queen, Inspection logs
│   ├── spinners/           # Voice note processing
│   └── tracker.py          # BeekeepingTracker
│
└── food-e/                 # Food tracking
    └── spinners/           # Grocery receipt OCR
```

---

## Data Flow

```
Voice Note -> Domain Spinner -> MemoryShard -> HoloLoom Storage
                    |                |              |
            Automatic extraction  Standardized  Neo4j + Qdrant
            - Hive ID              format:        |
            - Queen status         - ID, text   Graph + Vectors
            - Population           - Entities
            - Health issues        - Motifs    Natural Language
            - Actions taken        - Metadata       Query
                                                     |
                                              WeavingOrchestrator
                                                     |
                                              "Which hives are
                                               strong enough
                                               to split?"
```

---

## Quick Start

### 1. Run the Demo

```bash
cd apps
python demo_farm_simple.py
```

This demonstrates:
- ✓ Voice note processing (beekeeping inspection)
- ✓ Structured data extraction
- ✓ Asset and log models
- ✓ MemoryShard generation

### 2. Use Spinners Directly

```python
from apps.beekeeping.spinners.bee_inspection import process_bee_inspection

shards = await process_bee_inspection(
    transcript="Saw queen laying, 8 frames of bees, calm temperament",
    hive_id="A",
    inspector="Blake"
)

# Returns structured MemoryShards:
# - Hive ID: A
# - Queen status: present_laying
# - Population: strong
# - Temperament: calm
```

### 3. Work with Models

```python
from apps.beekeeping.models import Hive, InspectionLog
from apps.farm_core.models import AssetType, LogType

# Create a hive
hive = Hive.create(
    name="Hive Alpha",
    location="Backyard",
    queen_status=QueenStatus.PRESENT_LAYING,
    population=PopulationStrength.STRONG
)

# Convert to memory shard for HoloLoom storage
shard = hive.to_shard()
```

---

## Apps

### Beekeeping (Apiary Management)

**Features:**
- Voice note inspections with automatic extraction
- Hive inventory and status tracking
- Treatment schedules and tracking
- Harvest records
- Inspection reminders

**Assets:**
- `Hive` - Individual colonies
- `Queen` - Queen bee lineage tracking
- `Apiary` - Location groupings

**Logs:**
- `InspectionLog` - Hive observations
- `TreatmentLog` - Varroa/disease treatments
- `FeedingLog` - Sugar syrup, pollen patties
- `HarvestLog` - Honey, wax, propolis
- `SplitLog` - Colony divisions

**Spinners:**
- `BeeInspectionAudioSpinner` - Processes voice notes
  - Extracts: hive ID, queen status, population, health issues, actions

**Example Queries** (future):
```
"What's the queen status of hive Alpha?"
"Show all inspections with mite problems"
"Which hives are strong enough to split?"
"When was hive B last treated?"
```

### Food-e (Food & Nutrition Tracking)

**Features:**
- Grocery receipt OCR
- Food inventory tracking
- Spending analysis
- Nutrition tracking (future)

**Spinners:**
- `GroceryReceiptSpinner` - Processes receipt images
  - Extracts: store, date, items, categories, totals

**Example Queries** (future):
```
"How much did I spend on groceries this month?"
"What produce did I buy last week?"
"Show all purchases from Whole Foods"
"Alert me when milk prices spike"
```

---

## Creating New Apps

### 1. Define Domain Models

```python
# apps/livestock/models.py
from apps.farm_core.models import Asset, Log

@dataclass
class Animal(Asset):
    species: str = None
    breed: str = None
    tag_id: str = None

    def __post_init__(self):
        if self.asset_type != AssetType.ANIMAL:
            object.__setattr__(self, 'asset_type', AssetType.ANIMAL)
```

### 2. Create Domain Spinners

```python
# apps/livestock/spinners/health_observation.py
from HoloLoom.spinning_wheel.audio import AudioSpinner

class HealthObservationSpinner(AudioSpinner):
    async def spin(self, raw_data):
        # Parse audio transcript
        # Extract: animal ID, symptoms, treatment
        # Return structured MemoryShards
```

### 3. Build Tracker App

```python
# apps/livestock/tracker.py
from apps.farm_core.tracker import FarmTracker

class LivestockTracker(FarmTracker):
    async def add_animal(self, animal: Animal):
        await self.add_asset(animal)

    async def record_medical(self, log: MedicalLog):
        await self.add_log(log)
```

---

## Integration with HoloLoom

### Weaving Cycle for Farm Queries

```
1. User Query: "Which hives need inspection?"
2. Loom Command: Select FAST pattern
3. Chrono Trigger: Temporal window (last 30 days)
4. Resonance Shed: Extract features (motifs, embeddings, spectral)
5. Warp Space: Semantic similarity search in Qdrant
6. Convergence Engine: Rank hives by priority
7. Spacetime Output: Natural language + structured data
8. Reflection Buffer: Learn from feedback
```

### Memory Backend Configuration

```python
from apps.farm_core.tracker import FarmTracker, FarmTrackerConfig
from HoloLoom.config import MemoryBackend

config = FarmTrackerConfig(
    app_name="My Farm",
    memory_backend=MemoryBackend.NEO4J_QDRANT  # Persistent storage
)

tracker = FarmTracker(config)
await tracker.initialize()
```

---

## Advantages Over Traditional Software

| Feature | Traditional | HoloLoom Apps |
|---------|------------|---------------|
| **Data Entry** | Forms, dropdowns | Voice notes, photos |
| **Search** | SQL filters | Natural language |
| **Patterns** | Manual reports | Automatic recognition |
| **Integration** | Siloed domains | Unified semantic memory |
| **Insights** | Retrospective | Proactive recommendations |
| **Learning** | Static | Adapts from feedback (PPO) |

---

## Development Status

### ✓ Completed

- [x] farm_core framework (models, tracker)
- [x] Beekeeping models (Hive, InspectionLog, etc.)
- [x] BeeInspectionAudioSpinner (voice note processing)
- [x] GroceryReceiptSpinner (receipt OCR)
- [x] Demo script showing full pipeline
- [x] farmOS architecture research

### 🚧 In Progress

- [ ] FarmTracker integration with WeavingOrchestrator
- [ ] Natural language query() implementation
- [ ] Full BeekeepingTracker with reminders

### 📋 Planned

- [ ] LivestockTracker (animal husbandry)
- [ ] CropTracker (field management)
- [ ] Sensor integration (temperature, weight scales)
- [ ] Photo analysis (plant disease, body condition)
- [ ] CLI/API interface
- [ ] Grafana dashboards

---

## Technical Requirements

### Dependencies

```bash
# Core HoloLoom
torch, numpy, networkx, scipy

# Memory backends
neo4j, qdrant-client

# Spinners
pytesseract, pillow  # Image OCR
ollama               # Vision models, enrichment

# Apps
python-dateutil      # Date parsing
pydantic            # Data validation
```

### Infrastructure

```yaml
# docker-compose.yml
services:
  neo4j:
    image: neo4j:latest
    ports: ["7687:7687", "7474:7474"]

  qdrant:
    image: qdrant/qdrant:latest
    ports: ["6333:6333"]
```

---

## Example Use Cases

### Beekeeping

**Voice Inspection:**
```
"Inspecting hive A. Queen laying well, 8 frames of bees,
calm temperament. Added a super. No mite signs."
```

**Extracted:**
- Hive: A
- Queen: Present & laying
- Population: Strong (8 frames)
- Temperament: Calm
- Actions: Added super
- Health: No issues

### Food Tracking

**Receipt Image:**
```
WHOLE FOODS
BANANAS $3.99
MILK $5.49
EGGS $4.99
TOTAL $14.47
```

**Extracted:**
- Store: Whole Foods
- Items: 3 (produce: 1, dairy: 2)
- Total: $14.47
- Categories for budget analysis

---

## Contributing

This farm management suite is part of the mythRL/HoloLoom ecosystem. To add new apps:

1. Create `apps/{app_name}/` directory
2. Define domain models extending `Asset` and `Log`
3. Create spinners for multimodal data ingestion
4. Implement tracker extending `FarmTracker`
5. Add demos and tests

---

## License

Part of the mythRL project. See main repository LICENSE.

---

## Credits

- **Architecture**: Inspired by farmOS open-source platform
- **Framework**: Built on HoloLoom neural memory system
- **Concept**: Multimodal AI-powered farm management

---

**Status**: Demo ready, framework complete, apps in development
**Next**: Integrate WeavingOrchestrator for natural language queries
