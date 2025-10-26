# Automotive Repair Domain

A comprehensive domain for automotive maintenance and repair expertise, demonstrating ExpertLoom's community extension framework.

## Overview

This domain captures knowledge about vehicle maintenance, including:
- Vehicle identification and tracking
- Fluid management (engine oil, coolant, etc.)
- Tire maintenance and pressure monitoring
- Component tracking (battery, brakes, filters)
- Diagnostic patterns (sounds, wear, conditions)

## Domain Statistics

- **10 entities** across 4 types (vehicle, fluid, tire, component)
- **44 aliases** for flexible mention recognition
- **7 numeric measurement patterns** (tire pressure, mileage, voltage, etc.)
- **4 categorical patterns** (oil condition, tire wear, sounds, fluid levels)
- **5 micropolicies** for safety alerts and maintenance reminders
- **5 reverse queries** for diagnostic assistance

## What This Domain Extracts

### Entities

**Vehicle:**
- 2015 Toyota Corolla (aliases: "corolla", "toyota", "the car", "my car")

**Fluids:**
- Engine Oil (aliases: "oil", "motor oil", "5W-30", "synthetic")
- Coolant (aliases: "coolant", "antifreeze", "radiator fluid")

**Tires:**
- All 4 tires individually tracked (FL, FR, RL, RR)
- Position-aware entity resolution

**Components:**
- Battery
- Front brake pads
- Air filter

### Measurements

**Numeric:**
- `tire_pressure_psi`: Tire pressure (PSI)
- `tread_depth_mm`: Tire tread depth (mm)
- `mileage`: Vehicle odometer reading
- `oil_quarts`: Oil volume used
- `temperature_f`: Temperature readings
- `battery_voltage`: Battery voltage (V)
- `brake_pad_thickness_mm`: Brake pad remaining thickness

**Categorical:**
- `oil_condition`: clean, dirty, black, contaminated, milky, amber
- `tire_wear`: even, uneven, cupping, feathering, bald, good
- `sound_type`: knocking, ticking, squealing, grinding, whining, rattling
- `fluid_level`: full, low, empty, adequate, topped off

## Example Usage

### Basic Entity Resolution and Measurement Extraction

```python
from pathlib import Path
from mythRL_core.entity_resolution import EntityRegistry, EntityResolver
from mythRL_core.entity_resolution.extractor import EntityExtractor

# Load domain
registry = EntityRegistry.load(Path("mythRL_core/domains/automotive/registry.json"))
resolver = EntityResolver(registry)
extractor = EntityExtractor(resolver, custom_patterns=registry.measurement_patterns)

# Process a maintenance note
note = "Checked the Corolla at 87,450 miles - oil looks dirty and front left tire at 28 PSI"
extracted = extractor.extract(note)

# Results:
# - Entities: Corolla -> vehicle-corolla-2015, oil -> fluid-engine-oil, front left tire -> component-tire-front-left
# - Measurements: mileage=87450.0, oil_condition='dirty', tire_pressure_psi=28.0
```

### Qdrant Storage with Semantic Search

```python
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

model = SentenceTransformer('all-MiniLM-L6-v2')
client = QdrantClient(host="localhost", port=6333)

# Extract and embed
extracted = extractor.extract(note)
embedding = model.encode(note)
payload = extracted.to_qdrant_payload()

# Store
point = PointStruct(
    id=str(uuid.uuid4()),
    vector=embedding.tolist(),
    payload=payload
)
client.upsert(collection_name="automotive_maintenance", points=[point])

# Query by semantic similarity
query_vector = model.encode("Which tires are low on pressure?")
results = client.query_points(
    collection_name="automotive_maintenance",
    query=query_vector.tolist(),
    limit=5
)

# Query by filter (specific component)
results = client.query_points(
    collection_name="automotive_maintenance",
    query=query_vector.tolist(),
    query_filter={"must": [{"key": "primary_entity_id", "match": {"value": "fluid-engine-oil"}}]},
    limit=5
)
```

## Micropolicies (Safety & Maintenance Rules)

### 1. Low Tire Pressure Alert (HIGH PRIORITY)
**Trigger:** `tire_pressure_psi < 28`
**Action:** "Tire pressure critically low (< 28 PSI) - check for leaks immediately and inflate to 32 PSI"

### 2. Oil Change Interval (MEDIUM PRIORITY)
**Trigger:** `miles_since_oil_change > 5000 OR oil_condition = black`
**Action:** "Oil change recommended - either due to mileage or condition. Use 5W-30 synthetic."

### 3. Brake Pad Replacement Threshold (HIGH PRIORITY)
**Trigger:** `brake_pad_thickness_mm < 3`
**Action:** "Brake pads below safe threshold (< 3mm) - schedule replacement immediately for safety"

### 4. Weak Battery Voltage (MEDIUM PRIORITY)
**Trigger:** `battery_voltage < 12.4`
**Action:** "Battery voltage low (< 12.4V) - test battery and charging system, may need replacement soon"

### 5. Coolant Condition Check (MEDIUM PRIORITY)
**Trigger:** `coolant_condition = dirty OR coolant_condition = rusty`
**Action:** "Coolant appears contaminated - flush cooling system and replace with fresh long-life coolant"

## Reverse Queries (Diagnostic Questions)

These are questions the system asks YOU to help diagnose problems:

### 1. Engine Noise Diagnosis
**Trigger:** Mentions of sounds (knocking, ticking, squealing, grinding)
**Question:** "You mentioned a {sound_type} sound - when exactly does it occur? (cold start, hot idle, acceleration, deceleration, braking)"
**Purpose:** Narrow down source based on occurrence timing

### 2. Recurring Tire Pressure Loss
**Trigger:** Same tire repeatedly drops below 28 PSI
**Question:** "{tire_name} pressure keeps dropping to {pressure} PSI - have you noticed any visible damage, punctures, or valve stem issues?"
**Purpose:** Identify cause of slow leak

### 3. Milky Oil (Head Gasket Failure)
**Trigger:** `oil_condition = milky`
**Question:** "Oil shows milky appearance - have you noticed white smoke from exhaust, overheating, or coolant level dropping? This could indicate head gasket failure."
**Purpose:** Diagnose coolant mixing with oil

### 4. High Mileage Period
**Trigger:** Mileage increased > 10,000 in < 3 months
**Question:** "Mileage jumped from {old_mileage} to {new_mileage} in {time_period} - are you doing more highway driving? May want to adjust maintenance schedule."
**Purpose:** Adjust maintenance for usage patterns

### 5. Uneven Tire Wear
**Trigger:** `tire_wear = uneven OR tire_wear = cupping`
**Question:** "{tire_name} showing {wear_pattern} wear - when was the last alignment check? This pattern often indicates alignment or suspension issues."
**Purpose:** Identify underlying cause of abnormal wear

## Test Notes

The domain includes 7 validation test phrases covering:
- Multiple entity types in one note
- Numeric measurements (pressure, voltage, mileage)
- Categorical assessments (oil condition, sounds)
- Safety-critical scenarios (low pressure, worn brakes)
- Diagnostic patterns (engine noises, fluid contamination)

## Run the Demo

```bash
python demo_automotive_domain.py
```

This will show:
- Entity resolution across all note examples
- Measurement extraction (numeric + categorical)
- Qdrant payload generation
- Domain statistics

## Contributing Improvements

This domain can be expanded with:
- More vehicle types (trucks, motorcycles, etc.)
- Additional fluids (brake fluid, transmission fluid, power steering)
- More components (spark plugs, alternator, serpentine belt)
- Seasonal considerations (winter prep, summer heat)
- More diagnostic patterns (vibrations, smells, leaks)
- Service history tracking
- Cost estimation micropolicies

## License

MIT - Feel free to adapt for your own vehicles or contribute improvements!

## Author

ExpertLoom Community - Example Domain

---

**This domain demonstrates the ExpertLoom framework's ability to capture and preserve automotive expertise in a structured, queryable format. Same infrastructure as beekeeping, completely different domain.**
