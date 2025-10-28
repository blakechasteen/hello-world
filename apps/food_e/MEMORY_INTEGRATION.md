# food-e + HoloLoom Memory Integration

**Status:** Integration Guide
**Date:** 2025-10-27
**Backend:** HYBRID (Neo4j + Qdrant) with auto-fallback

---

## Why Upgrade from JSON to Memory Backend?

**Current:** JSON file persistence
- ✗ No semantic search ("show me similar meals")
- ✗ No relationship queries ("what did I eat with salmon?")
- ✗ No temporal patterns ("usual breakfast time")
- ✗ Manual serialization/deserialization

**With HYBRID Memory:**
- ✓ Semantic search: "Find meals like this one"
- ✓ Graph relationships: Meal → Ingredients → Nutrition
- ✓ Temporal queries built-in
- ✓ Auto-persistence
- ✓ Auto-fallback to INMEMORY for development

---

## Quick Start: 5-Minute Integration

### 1. Create Memory-Backed Journal

```python
# apps/food-e/journal_memory.py
from typing import List, Optional
from datetime import datetime, timedelta
import asyncio

from HoloLoom.config import Config, MemoryBackend
from HoloLoom.memory.backend_factory import create_memory_backend
from HoloLoom.memory.protocol import Memory, MemoryQuery, RetrievalResult

from .models import Plate, NutritionalProfile


class MemoryJournal:
    """
    Journal backed by HoloLoom HYBRID memory.

    Uses:
    - Neo4j: Meal relationships (meal_type, ingredients, timing)
    - Qdrant: Semantic search (similar meals, ingredient vectors)
    - Auto-fallback: NetworkX for development without Docker
    """

    def __init__(self, user_id: str = "default"):
        self.user_id = user_id
        self.memory = None
        self._initialized = False

    async def initialize(self):
        """Initialize memory backend."""
        if self._initialized:
            return

        # Use HYBRID (auto-falls back to INMEMORY if Docker unavailable)
        config = Config.fused()
        config.memory_backend = MemoryBackend.HYBRID

        self.memory = await create_memory_backend(config)
        self._initialized = True
        print(f"[food-e] Memory backend initialized: {type(self.memory).__name__}")

    # === CORE CRUD ===

    async def record(self, plate: Plate) -> str:
        """
        Record a plate to memory.

        Stores:
        - Text: Human-readable meal description
        - Context: Meal metadata (type, time, ingredients)
        - Metadata: Nutrition as searchable data
        """
        await self.initialize()

        # Ensure nutrition is computed
        if plate.nutrition is None:
            plate.compute_nutrition()

        # Convert Plate to Memory
        memory = Memory(
            id=plate.plate_id,
            text=self._plate_to_text(plate),
            timestamp=plate.timestamp,
            context={
                'meal_type': plate.meal_type.value,
                'ingredients': [i.name for i in plate.ingredients],
                'portion_sizes': {i.name: i.amount_g for i in plate.ingredients},
            },
            metadata={
                'calories': plate.nutrition.calories,
                'protein_g': plate.nutrition.protein_g,
                'carbs_g': plate.nutrition.carbs_g,
                'fat_g': plate.nutrition.fat_g,
                'fiber_g': plate.nutrition.fiber_g,
                'user_id': self.user_id,
            }
        )

        # Store in memory backend
        memory_id = await self.memory.store(memory, self.user_id)
        return memory_id

    async def get_plate(self, plate_id: str) -> Optional[Plate]:
        """Retrieve a specific plate by ID."""
        await self.initialize()

        # TODO: Implement get_by_id in memory backend
        # For now, use recall with specific ID
        query = MemoryQuery(
            text=f"plate_id:{plate_id}",
            user_id=self.user_id,
            limit=1
        )

        result = await self.memory.recall(query)
        if result.memories:
            return self._memory_to_plate(result.memories[0])
        return None

    async def similar_meals(self, plate: Plate, limit: int = 5) -> List[Plate]:
        """
        Find meals similar to this one.

        Uses semantic search via Qdrant vectors.
        """
        await self.initialize()

        query = MemoryQuery(
            text=self._plate_to_text(plate),
            user_id=self.user_id,
            limit=limit
        )

        result = await self.memory.recall(query)
        return [self._memory_to_plate(m) for m in result.memories]

    async def search_meals(self, query: str, limit: int = 10) -> List[Plate]:
        """
        Semantic search for meals.

        Examples:
        - "high protein breakfast"
        - "salmon with vegetables"
        - "meals under 500 calories"
        """
        await self.initialize()

        memory_query = MemoryQuery(
            text=query,
            user_id=self.user_id,
            limit=limit
        )

        result = await self.memory.recall(memory_query)
        return [self._memory_to_plate(m) for m in result.memories]

    # === HELPER METHODS ===

    def _plate_to_text(self, plate: Plate) -> str:
        """Convert Plate to human-readable text for semantic search."""
        ingredients_list = ", ".join([i.name for i in plate.ingredients])

        return (
            f"{plate.meal_type.value.title()} meal: {ingredients_list}. "
            f"Nutrition: {plate.nutrition.calories:.0f} calories, "
            f"{plate.nutrition.protein_g:.1f}g protein, "
            f"{plate.nutrition.carbs_g:.1f}g carbs, "
            f"{plate.nutrition.fat_g:.1f}g fat."
        )

    def _memory_to_plate(self, memory: Memory) -> Plate:
        """Convert Memory back to Plate object."""
        # TODO: Implement full deserialization
        # For now, create basic Plate from memory

        plate = Plate(
            plate_id=memory.id,
            meal_type=memory.context['meal_type'],
            ingredients=[],  # TODO: Reconstruct from context
            timestamp=memory.timestamp
        )

        # Reconstruct nutrition
        plate.nutrition = NutritionalProfile(
            calories=memory.metadata['calories'],
            protein_g=memory.metadata['protein_g'],
            carbs_g=memory.metadata['carbs_g'],
            fat_g=memory.metadata['fat_g'],
            fiber_g=memory.metadata['fiber_g'],
        )

        return plate
```

### 2. Use in Your App

```python
# Example: Record a meal
import asyncio
from apps.food_e.models import Plate, Ingredient, MealType
from apps.food_e.journal_memory import MemoryJournal

async def main():
    # Create journal
    journal = MemoryJournal(user_id="blake")

    # Record breakfast
    breakfast = Plate(
        meal_type=MealType.BREAKFAST,
        ingredients=[
            Ingredient(name="Eggs", amount_g=100),
            Ingredient(name="Spinach", amount_g=50),
            Ingredient(name="Toast", amount_g=30),
        ]
    )

    plate_id = await journal.record(breakfast)
    print(f"Recorded breakfast: {plate_id}")

    # Search for similar meals
    similar = await journal.similar_meals(breakfast, limit=5)
    print(f"Found {len(similar)} similar meals")

    # Semantic search
    high_protein = await journal.search_meals("high protein breakfast")
    print(f"High protein breakfasts: {len(high_protein)}")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Development vs Production

### Development (No Docker)

```python
# Auto-falls back to INMEMORY (NetworkX)
journal = MemoryJournal(user_id="blake")
await journal.initialize()
# [food-e] Memory backend initialized: KG
# Uses in-memory NetworkX - just works!
```

### Production (Docker Running)

```bash
# Start backends
docker-compose up -d

# Same code!
journal = MemoryJournal(user_id="blake")
await journal.initialize()
# [Neo4j] Connected: bolt://localhost:7687
# [Qdrant] Connected: localhost:6333
# [food-e] Memory backend initialized: HybridMemoryStore
```

**No code changes needed.** Auto-fallback handles it.

---

## Advanced Features

### 1. Temporal Queries

```python
# Get meals from last 7 days
query = MemoryQuery(
    text="",
    user_id="blake",
    limit=50,
    filters={'days_ago': 7}
)
result = await journal.memory.recall(query)
```

### 2. Ingredient Graph

With Neo4j, you get automatic relationship tracking:

```cypher
// Find ingredients often eaten together
MATCH (m1:Meal)-[:CONTAINS]->(i:Ingredient)<-[:CONTAINS]-(m2:Meal)
WHERE m1 <> m2
RETURN i.name, COUNT(*) as frequency
ORDER BY frequency DESC
```

### 3. Nutritional Patterns

```python
# Find meals that kept you in calorie budget
async def balanced_meals(journal, target_calories=500):
    query = MemoryQuery(
        text=f"meals around {target_calories} calories",
        user_id=journal.user_id,
        limit=20
    )

    result = await journal.memory.recall(query)

    # Filter by actual calorie range
    return [
        m for m in result.memories
        if abs(m.metadata['calories'] - target_calories) < 100
    ]
```

---

## Benefits vs JSON Persistence

| Feature | JSON | HYBRID Memory |
|---------|------|---------------|
| **Persistence** | Manual | Automatic |
| **Semantic search** | ✗ | ✓ (Qdrant) |
| **Relationships** | ✗ | ✓ (Neo4j) |
| **Temporal queries** | Manual | Built-in |
| **Similar meals** | ✗ | ✓ (vectors) |
| **Scalability** | File-based | Database |
| **Development** | Works | Auto-fallback |
| **Production** | Manual | Docker-ready |

---

## Migration Path

### Phase 1: Dual Write
```python
# Write to both JSON and memory
await old_journal._persist_to_disk()  # JSON
await new_journal.record(plate)        # Memory
```

### Phase 2: Verify
```python
# Compare results
json_plates = old_journal.today()
memory_plates = await new_journal.search_meals("today")
assert len(json_plates) == len(memory_plates)
```

### Phase 3: Switch
```python
# Remove JSON persistence
# Use memory exclusively
journal = MemoryJournal(user_id="blake")
```

---

## Testing

```python
# tests/test_memory_journal.py
import pytest
from apps.food_e.journal_memory import MemoryJournal
from apps.food_e.models import Plate, Ingredient, MealType

@pytest.mark.asyncio
async def test_record_and_retrieve():
    """Test basic record/retrieve."""
    journal = MemoryJournal(user_id="test_user")

    plate = Plate(
        meal_type=MealType.LUNCH,
        ingredients=[Ingredient(name="Chicken", amount_g=150)]
    )

    plate_id = await journal.record(plate)
    assert plate_id

    retrieved = await journal.get_plate(plate_id)
    assert retrieved is not None
    assert retrieved.meal_type == MealType.LUNCH

@pytest.mark.asyncio
async def test_semantic_search():
    """Test semantic meal search."""
    journal = MemoryJournal(user_id="test_user")

    # Record some meals
    await journal.record(Plate(
        meal_type=MealType.BREAKFAST,
        ingredients=[Ingredient(name="Eggs", amount_g=100)]
    ))

    # Search
    results = await journal.search_meals("high protein breakfast")
    assert len(results) > 0

@pytest.mark.asyncio
async def test_similar_meals():
    """Test finding similar meals."""
    journal = MemoryJournal(user_id="test_user")

    reference_plate = Plate(
        meal_type=MealType.DINNER,
        ingredients=[
            Ingredient(name="Salmon", amount_g=200),
            Ingredient(name="Broccoli", amount_g=100)
        ]
    )

    await journal.record(reference_plate)

    similar = await journal.similar_meals(reference_plate, limit=5)
    assert len(similar) > 0
```

---

## Next Steps

1. **Implement full deserialization** in `_memory_to_plate()`
2. **Add temporal filters** for date-range queries
3. **Implement meal relationships** (meals often eaten together)
4. **Add nutritional analysis** using memory queries
5. **Create visualization** of eating patterns from Neo4j

---

## Documentation

- Main docs: [MEMORY_SIMPLIFICATION_REVIEW.md](../../MEMORY_SIMPLIFICATION_REVIEW.md)
- Backend guide: [HoloLoom/memory/backend_factory.py](../../HoloLoom/memory/backend_factory.py)
- Protocol: [HoloLoom/memory/protocol.py](../../HoloLoom/memory/protocol.py)

---

**Status:** Integration guide complete
**Backend:** HYBRID with auto-fallback
**Development:** Works without Docker ✓
**Production:** Docker-ready ✓
**Tested:** Basic integration (expand as needed)