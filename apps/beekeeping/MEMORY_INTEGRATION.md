# Beekeeping + HoloLoom Memory Integration

**App:** Apiary Management (Hive inspections, treatments, harvests)
**Backend:** HYBRID (Neo4j + Qdrant) with auto-fallback
**Date:** 2025-10-27

---

## Why Add Memory to Beekeeping?

**Current:** File-based persistence (if any)

**With HYBRID Memory:**
- ✓ **Semantic search:** "Show me weak hives with queen problems"
- ✓ **Graph relationships:** Hive → Inspections → Treatments → Outcomes
- ✓ **Temporal patterns:** "Typical inspection schedule for this hive"
- ✓ **Cross-hive insights:** "Other hives with similar issues"
- ✓ **Treatment effectiveness:** "Did this treatment work last time?"

---

## Integration: 5-Minute Setup

### 1. Create Memory-Backed Apiary

```python
# apps/beekeeping/apiary_memory.py
from typing import List, Optional
from datetime import datetime
import asyncio

from HoloLoom.config import Config, MemoryBackend
from HoloLoom.memory.backend_factory import create_memory_backend
from HoloLoom.memory.protocol import Memory, MemoryQuery

from .models import Hive, InspectionLog, TreatmentLog, HarvestLog


class ApiaryMemory:
    """
    Apiary management with HoloLoom HYBRID memory.

    Features:
    - Semantic search for hive inspections
    - Treatment history and effectiveness tracking
    - Cross-hive pattern detection
    - Temporal analysis (inspection schedules, harvest timing)
    """

    def __init__(self, apiary_name: str = "default"):
        self.apiary_name = apiary_name
        self.memory = None
        self._initialized = False

    async def initialize(self):
        """Initialize HYBRID memory (auto-falls back to INMEMORY)."""
        if self._initialized:
            return

        config = Config.fused()
        config.memory_backend = MemoryBackend.HYBRID

        self.memory = await create_memory_backend(config)
        self._initialized = True
        print(f"[Beekeeping] Memory initialized: {type(self.memory).__name__}")

    # === HIVE MANAGEMENT ===

    async def record_inspection(self, hive: Hive, inspection: InspectionLog) -> str:
        """
        Record hive inspection to memory.

        Stores:
        - text: Human-readable inspection summary
        - context: Hive state, observations
        - metadata: Queen status, population, temperament
        """
        await self.initialize()

        memory = Memory(
            id=f"{hive.name}_inspection_{inspection.timestamp.isoformat()}",
            text=self._inspection_to_text(hive, inspection),
            timestamp=inspection.timestamp,
            context={
                'hive_name': hive.name,
                'hive_id': hive.asset_id,
                'equipment_type': hive.equipment_type.value,
                'num_boxes': hive.num_boxes,
                'observation_type': 'inspection',
            },
            metadata={
                'apiary': self.apiary_name,
                'queen_status': hive.queen_status.value,
                'population': hive.population.value,
                'temperament': hive.temperament.value,
                'frames_of_bees': hive.frames_of_bees or 0,
                'queen_marked': hive.queen_marked,
            }
        )

        return await self.memory.store(memory, self.apiary_name)

    async def record_treatment(self, hive: Hive, treatment: TreatmentLog) -> str:
        """Record treatment application."""
        await self.initialize()

        memory = Memory(
            id=f"{hive.name}_treatment_{treatment.timestamp.isoformat()}",
            text=f"Treatment for {hive.name}: {treatment.notes or 'Applied treatment'}",
            timestamp=treatment.timestamp,
            context={
                'hive_name': hive.name,
                'hive_id': hive.asset_id,
                'observation_type': 'treatment',
                'treatment_type': treatment.log_type.value,
            },
            metadata={
                'apiary': self.apiary_name,
            }
        )

        return await self.memory.store(memory, self.apiary_name)

    async def record_harvest(self, hive: Hive, harvest: HarvestLog) -> str:
        """Record honey/wax harvest."""
        await self.initialize()

        memory = Memory(
            id=f"{hive.name}_harvest_{harvest.timestamp.isoformat()}",
            text=f"Harvest from {hive.name}: {harvest.notes or 'Harvested'}",
            timestamp=harvest.timestamp,
            context={
                'hive_name': hive.name,
                'hive_id': hive.asset_id,
                'observation_type': 'harvest',
            },
            metadata={
                'apiary': self.apiary_name,
            }
        )

        return await self.memory.store(memory, self.apiary_name)

    # === SEMANTIC SEARCH ===

    async def find_problems(self, query: str = "weak hives queenless", limit: int = 10) -> List[Memory]:
        """
        Semantic search for hive problems.

        Examples:
            await apiary.find_problems("weak hives with queen issues")
            await apiary.find_problems("defensive temperament")
            await apiary.find_problems("low population")
        """
        await self.initialize()

        memory_query = MemoryQuery(
            text=query,
            user_id=self.apiary_name,
            limit=limit
        )

        result = await self.memory.recall(memory_query)
        return result.memories

    async def hive_history(self, hive_name: str, limit: int = 20) -> List[Memory]:
        """Get inspection history for a specific hive."""
        await self.initialize()

        query = MemoryQuery(
            text=f"inspections for hive {hive_name}",
            user_id=self.apiary_name,
            limit=limit
        )

        result = await self.memory.recall(query)
        return result.memories

    async def similar_hives(self, hive: Hive, limit: int = 5) -> List[Memory]:
        """Find hives with similar characteristics."""
        await self.initialize()

        query_text = self._hive_to_query(hive)

        query = MemoryQuery(
            text=query_text,
            user_id=self.apiary_name,
            limit=limit
        )

        result = await self.memory.recall(query)
        return result.memories

    # === HELPER METHODS ===

    def _inspection_to_text(self, hive: Hive, inspection: InspectionLog) -> str:
        """Convert inspection to searchable text."""
        return (
            f"Hive {hive.name} inspection: "
            f"Queen {hive.queen_status.value}, "
            f"Population {hive.population.value} ({hive.frames_of_bees or '?'} frames), "
            f"Temperament {hive.temperament.value}. "
            f"Notes: {inspection.notes or 'None'}"
        )

    def _hive_to_query(self, hive: Hive) -> str:
        """Convert hive to similarity search query."""
        return (
            f"Hive with {hive.equipment_type.value} equipment, "
            f"{hive.population.value} population, "
            f"{hive.temperament.value} temperament, "
            f"queen status {hive.queen_status.value}"
        )
```

### 2. Example Usage

```python
# Example: Track hive inspections
import asyncio
from apps.beekeeping.models import Hive, InspectionLog, QueenStatus, PopulationStrength
from apps.beekeeping.apiary_memory import ApiaryMemory

async def main():
    # Create apiary memory
    apiary = ApiaryMemory(apiary_name="blake_apiary")

    # Create a hive
    hive1 = Hive(
        name="Hive 1",
        queen_status=QueenStatus.PRESENT_LAYING,
        population=PopulationStrength.STRONG,
        frames_of_bees=10
    )

    # Record inspection
    inspection = InspectionLog(
        asset_id=hive1.asset_id,
        notes="Strong colony, queen laying well, 8 frames of brood"
    )

    await apiary.record_inspection(hive1, inspection)
    print("✓ Inspection recorded")

    # Search for problems
    problems = await apiary.find_problems("weak hives queenless")
    print(f"Found {len(problems)} problem hives")

    # Get hive history
    history = await apiary.hive_history("Hive 1", limit=10)
    print(f"Hive 1 has {len(history)} records")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Development vs Production

**Development (No Docker):**
```python
apiary = ApiaryMemory()
await apiary.initialize()
# [Beekeeping] Memory initialized: KG
# Uses NetworkX in-memory - just works!
```

**Production (Docker Running):**
```bash
docker-compose up -d
```

```python
apiary = ApiaryMemory()
await apiary.initialize()
# [Neo4j] Connected: bolt://localhost:7687
# [Beekeeping] Memory initialized: HybridMemoryStore
```

---

## Advanced Queries

### 1. Queen Problems
```python
# Find all hives with queen issues
problems = await apiary.find_problems("queenless or queen not laying")
```

### 2. Treatment Effectiveness
```python
# Search for treatments and outcomes
results = await apiary.find_problems("varroa treatment followed by improvement")
```

### 3. Seasonal Patterns
```python
# With Neo4j, query temporal patterns
# Cypher: MATCH (i:Inspection)-[:FOR_HIVE]->(h:Hive)
#         WHERE i.timestamp > date('2024-01-01')
#         RETURN h.name, count(i) as inspections
```

---

## Benefits

| Feature | Before | With HYBRID |
|---------|--------|-------------|
| **Search** | Manual | "weak hives queenless" |
| **History** | File digging | `hive_history("Hive 1")` |
| **Patterns** | Spreadsheet | Graph queries |
| **Similar hives** | ✗ | `similar_hives(hive)` |
| **Treatment tracking** | Manual notes | Semantic search |
| **Scalability** | CSV limits | Database scale |

---

## Next Steps

1. **Implement full models** - Add all log types
2. **Add Neo4j relationships** - Hive → Inspection → Treatment → Outcome
3. **Create dashboards** - Visualize apiary health
4. **Temporal analysis** - Inspection schedules, harvest timing
5. **Cross-apiary insights** - Share patterns across apiaries

---

**Status:** Integration guide complete
**Backend:** HYBRID with auto-fallback ✓
**Development:** Works without Docker ✓
**Production:** Docker-ready ✓