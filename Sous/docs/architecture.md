# SOUS Architecture

**Version:** 0.1.0 (MVP)
**Date:** November 2025
**Built With:** HoloLoom Concurrent Metaprompting

---

## Overview

SOUS is a data-first kitchen manager with clean separation between models, services, and data. The architecture follows these principles:

1. **Data-First**: JSON seed data drives everything
2. **Protocol-Based**: Clear interfaces between components
3. **Extensible**: Easy to add recipes, stores, features
4. **Testable**: Models and services are independently testable

---

## Layer Architecture

```
┌─────────────────────────────────────────────┐
│            CLI Interface (app.py)           │
│        (User interaction layer)             │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│            Services Layer                   │
│  - Scheduler (day-by-day planning)          │
│  - ShoppingListGenerator (store routing)    │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│            Models Layer                     │
│  - Event, Recipe, Store, Schedule           │
│  - Data validation and transformation       │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│            Data Layer (JSON)                │
│  - event_thanksgiving.json                  │
│  - recipes_thanksgiving.json                │
│  - stores.json, ingredients.json            │
└─────────────────────────────────────────────┘
```

---

## Core Models

### Event
Represents a cooking event (e.g., Thanksgiving).

**Key Attributes:**
- `date`: Event date
- `prep_days`: List of available prep days
- `appliances`: Kitchen appliances with capacity

**Key Methods:**
- `days_before_event(date)`: Calculate days before event
- `get_appliance(id)`: Lookup appliance by ID
- `is_prep_day(date)`: Check if date is a prep day

### Recipe
Represents a dish with steps, ingredients, and timing.

**Key Attributes:**
- `steps`: Ordered list of preparation steps
- `make_ahead_min_days/max_days`: Make-ahead window
- `required_appliances`: Appliances needed
- `is_optional`: Whether recipe is optional

**Key Methods:**
- `get_step(id)`: Lookup step by ID
- `total_time_minutes()`: Calculate total time
- `requires_appliance(type)`: Check appliance requirement

### Store
Represents a grocery store with routing metadata.

**Key Attributes:**
- `roles`: Store categories (anchor, specialty, etc.)
- `priority`: Lower = preferred (1 is highest)

**Key Methods:**
- `has_role(role)`: Check if store has role
- `__lt__(other)`: Enable sorting by priority

### DailyPlan
Represents tasks for a single day.

**Key Attributes:**
- `date`: Date for this plan
- `tasks`: Ordered list of tasks

**Key Methods:**
- `total_duration_minutes()`: Calculate total work
- `tasks_by_appliance(id)`: Get tasks using appliance
- `has_appliance_conflict()`: Check for conflicts

---

## Core Services

### Scheduler

Intelligent day-by-day scheduling engine.

**Algorithm:**
```
For each recipe:
  For each step:
    1. Determine eligible days based on:
       - make_ahead_min_days/max_days
       - must_be_same_day_as_event flag
       - must_be_day_before flag
       - can_be_any_prep_day flag

    2. Assign to days:
       - Event day (days_before = 0)
       - Day before (days_before = 1)
       - Earlier prep days (days_before > 1)

    3. Resolve appliance conflicts:
       - Serialize tasks using same appliance
       - Sort by optional status + duration
```

**Key Methods:**
- `generate_daily_plans(event, recipes)`: Main entry point
- `detect_conflicts(plans, event)`: Find appliance conflicts
- `suggest_optional_placement(plans)`: Suggest bonus tasks

### ShoppingListGenerator

Multi-store shopping list routing.

**Algorithm:**
```
1. Aggregate all ingredient IDs from recipes
2. For each ingredient:
   - If preferred_store set, use that
   - Else, choose from possible_stores by priority
3. Group by store_id
4. Return dict: store_id -> [ingredient_names]
```

**Key Methods:**
- `generate_shopping_lists(recipes, ingredients, stores)`: Main entry point
- `optimize_route(lists, stores)`: Suggest store visit order
- `estimate_shopping_time(lists)`: Estimate time per store

---

## Data Flow

### Thanksgiving Planning Flow

```
1. Load Data
   ├─ event_thanksgiving.json → Event object
   ├─ recipes_thanksgiving.json → List[Recipe]
   ├─ stores.json → List[Store]
   └─ ingredients.json → Dict[str, dict]

2. Generate Shopping Lists
   └─ ShoppingListGenerator.generate_shopping_lists()
      └─ Dict[store_id, List[ingredient_names]]

3. Generate Schedule
   └─ Scheduler.generate_daily_plans()
      ├─ For each prep day + event day:
      │  └─ Assign eligible recipe steps
      └─ List[DailyPlan]

4. Detect Conflicts
   └─ Scheduler.detect_conflicts()
      └─ List[conflict_descriptions]

5. Display Results
   └─ app.py prints formatted output
```

---

## Key Patterns

### 1. Multi-Store Routing

**Pattern:** Ingredients have `preferred_store` or `possible_stores` with priority-based selection.

**Example:**
```json
{
  "white_truffle_oil": {
    "name": "White Truffle Oil",
    "category": "specialty",
    "possible_stores": ["store_earthfare"],
    "preferred_store": "store_earthfare"
  }
}
```

**Routing Logic:**
1. Check `preferred_store` → Use if set
2. Else, filter `possible_stores` → Sort by priority → Select best
3. Fallback: Use highest priority store

### 2. Make-Ahead Windows

**Pattern:** Recipes specify `make_ahead_min_days` and `make_ahead_max_days`.

**Example:**
```json
{
  "make_ahead_min_days": 1,
  "make_ahead_max_days": 5
}
```

**Scheduling Logic:**
- If `days_before < min_days`: Not eligible
- If `days_before > max_days`: Not eligible
- Else: Eligible for this day

### 3. Appliance Constraints

**Pattern:** Steps specify `appliance_type` (oven, burner, microwave).

**Example:**
```json
{
  "appliance_type": "oven",
  "duration_minutes": 60
}
```

**Conflict Detection:**
- For each appliance:
  - Count tasks using it on same day
  - If count > capacity: Conflict detected

### 4. Decision Trees

**Pattern:** Recipes have `substitution_rules` for conditional logic.

**Example:**
```json
{
  "substitution_rules": [
    {
      "condition": "Haricots verts not available",
      "alternative_ingredient_id": "collards",
      "notes": "Use collard greens instead"
    }
  ]
}
```

**Usage:** Display to user or automate substitution logic.

---

## Extension Points

### Adding New Features

**New Recipe:**
1. Add to `sous/data/recipes_thanksgiving.json`
2. Follow Recipe schema
3. Run tests to verify

**New Store:**
1. Add to `sous/data/stores.json`
2. Update ingredient `possible_stores`
3. Adjust priorities

**New Service:**
1. Create in `sous/services/`
2. Implement service logic
3. Import in `app.py`
4. Add tests in `tests/`

**New Model:**
1. Create in `sous/models/`
2. Implement `from_dict()` and `to_dict()`
3. Update `__init__.py`
4. Add validation logic

---

## Performance Characteristics

| Operation | Complexity | Typical Time |
|-----------|------------|--------------|
| Load JSON data | O(n) | ~50ms |
| Generate shopping lists | O(r × i) | ~10ms |
| Generate schedule | O(r × s × d) | ~20ms |
| Detect conflicts | O(d × t × a) | ~5ms |
| Total (full plan) | O(r × s × d) | **~85ms** |

Where:
- r = recipes (9)
- i = ingredients (~40)
- s = steps per recipe (~2-3)
- d = days (4)
- t = tasks per day (~5-10)
- a = appliances (3)

---

## Testing Strategy

### Integration Tests

Located in `tests/test_thanksgiving.py`:

1. **test_data_loading** - Verify JSON data loads correctly
2. **test_shopping_list_generation** - Verify shopping lists generated
3. **test_schedule_generation** - Verify schedule created
4. **test_appliance_conflict_detection** - Verify conflicts detected
5. **test_multi_store_routing** - Verify store routing logic

**Run:** `python tests/test_thanksgiving.py`

### Manual Testing

**Run:** `python sous/app.py`

**Verify:**
- Shopping lists have items for all 6 stores
- Schedule covers Monday-Thursday
- Event day (Thursday) has finishing tasks
- Appliance conflicts are detected and reported

---

## Future Architecture Enhancements

### Phase 2: Time Windows

**Current:** Tasks are sequenced but don't have explicit start times.
**Future:** Add time window scheduling with conflict resolution.

```python
class Task:
    start_time: datetime  # Actual start time
    end_time: datetime    # Actual end time
```

### Phase 3: Recipe Scaling

**Current:** Recipes are fixed (e.g., "x2" in name).
**Future:** Dynamic scaling based on guest count.

```python
class Recipe:
    base_servings: int
    scale_factor: float  # 1.0, 2.0, 4.0, etc.
```

### Phase 4: Dietary Filters

**Current:** All recipes included.
**Future:** Filter by dietary restrictions.

```python
class Recipe:
    dietary_tags: List[str]  # vegan, gluten_free, etc.
    allergens: List[str]     # nuts, dairy, etc.
```

### Phase 5: Skill Level Adaptation

**Current:** Fixed recipe complexity.
**Future:** Adjust based on user skill level.

```python
class Step:
    skill_level: str  # beginner, intermediate, expert
    substitutable: bool
```

---

## Metaprompting Integration

SOUS was built using **concurrent metaprompting** - all 10 HoloLoom strategies applied in parallel:

1. **VERIFY** → Found gaps in original analysis
2. **CHALLENGE** → Identified failure modes
3. **REVERSE** → Defined ideal end state
4. **OPTIMIZE** → Refined patterns
5. **DEEP** → Added technical detail
6. **SCAFFOLD** → Organized architecture
7. **PRIME** → Compared to existing apps
8. **DEBATE** → Multi-perspective design
9. **TEACH** → Educational clarity
10. **TEMP-SIM** → Alternative approaches

**See:** `metaprompts/synthesis_guide.md` for complete analysis.

---

## Summary

SOUS demonstrates a **clean, data-driven architecture** for kitchen management:

- ✅ **Models** define core concepts (Event, Recipe, Store, Schedule)
- ✅ **Services** implement business logic (Scheduler, ShoppingListGenerator)
- ✅ **Data** drives behavior (JSON seed files)
- ✅ **CLI** provides user interface (app.py)
- ✅ **Tests** verify correctness (test_thanksgiving.py)

**All patterns from manual Thanksgiving planning are now automated!**
