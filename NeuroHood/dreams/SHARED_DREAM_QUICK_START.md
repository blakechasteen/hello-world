# Shared Dream Synchronization - Quick Start Guide

**Version**: 1.0.0
**Status**: Production Ready
**Test Results**: 21/21 PASSING

---

## Installation

The system is self-contained. Just ensure these imports work:

```python
from NeuroHood.dreams.shared_dream_sync import (
    SharedDreamSynchronizer,
    SharedDreamType,
    SymbolBlendingStrategy
)
```

---

## 5-Minute Quick Start

### Create a Basic Shared Dream

```python
from NeuroHood.dreams.shared_dream_sync import SharedDreamSynchronizer

sync = SharedDreamSynchronizer()

# Define participants with private facts
participants = [
    {'id': 'alice_001', 'name': 'Alice',
     'private_fact': 'I feel trapped at work'},
    {'id': 'bob_001', 'name': 'Bob',
     'private_fact': 'I struggle with anger'}
]

# Create dream
session = await sync.create_shared_dream(participants)

# Access results
print(session.shared_narrative)
print(session.participant_perspectives)
print(session.waking_effects)
```

---

## Key Concepts

### 1. Dyadic Dreams (2 residents)
Most intimate. Perfect for conflict resolution.

### 2. Triad Dreams (3 residents)
Balanced dynamics. Good for group bonding.

### 3. Larger Groups (4-5 residents)
Complex group consciousness.

---

## Blending Strategies

| Strategy | Best For |
|----------|----------|
| **INTERACTION** | Conflict resolution |
| **NARRATIVE** | Story-driven dreams |
| **LANDSCAPE** | Contemplation |
| **CONVERGENCE** | Deep bonding |

---

## Accessing Results

### Dream Narrative
```python
print(session.shared_narrative)
```

### Individual Perspectives
```python
for res_id, perspective in session.participant_perspectives.items():
    print(f"{res_id}: {perspective}")
```

### Relationship Effects
```python
for (res_a, res_b), effects in session.waking_effects.items():
    print(f"Relationship change: +{effects['relationship_strength']}")
```

---

## Integration with NeuroHood

### Update Relationships

```python
for (res_a, res_b), effects in session.waking_effects.items():
    relationship_scm.update(
        resident_a=res_a,
        resident_b=res_b,
        strength_change=effects['relationship_strength']
    )
```

---

## Privacy Features

**What's Hidden**: Private facts (never appear in dream)

**What's Shared**: Emotional essence (feel each other's state through symbols)

Example:
```
Private: "I cheated on my taxes" (HIDDEN)
Symbol: Stained hands (SHOWN in dream)
Effect: Other residents feel guilt/shame without knowing the cause
```

---

## Testing

### Run All Tests
```bash
pytest NeuroHood/dreams/test_shared_dream_sync.py -v
# Result: 21/21 PASSING
```

---

## Demo

### Run Interactive Demo
```bash
python demos/demo_shared_dream_sync.py
```

Shows all features including:
- Dyadic dreams
- Triad dreams
- Blending strategies
- Privacy preservation

---

## Common Use Cases

### Conflict Resolution
```python
session = await sync.create_shared_dream(
    [resident_a, resident_b],
    blending_strategy=SymbolBlendingStrategy.NARRATIVE
)
```

### Team Bonding
```python
session = await sync.create_shared_dream(
    team_residents,
    blending_strategy=SymbolBlendingStrategy.CONVERGENCE
)
```

---

## API Reference

### SharedDreamSynchronizer

```python
sync = SharedDreamSynchronizer()

session = await sync.create_shared_dream(
    participants: List[Dict],
    consciousness_level: float = 0.5,
    blending_strategy: SymbolBlendingStrategy = INTERACTION
) -> SharedDreamSession
```

### SharedDreamSession

```python
session.session_id              # Unique identifier
session.dream_type              # DYADIC/TRIAD/QUARTET/QUINTET
session.shared_narrative        # Blended dream narrative
session.participant_perspectives  # Per-resident POV
session.waking_effects          # Relationship changes
session.intensity               # 0.0-1.0 resonance
session.duration_minutes        # Dream length

session.to_dict() -> Dict       # Serialize
```

---

**For more details, see SHARED_DREAM_SYNC_IMPLEMENTATION.md**
