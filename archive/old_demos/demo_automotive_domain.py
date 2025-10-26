"""
Demo: Automotive Domain - Validate Community Contribution
===========================================================

Shows that the automotive domain works with the same infrastructure
as beekeeping, demonstrating domain-agnostic architecture.
"""

import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from mythRL_core.entity_resolution import EntityRegistry, EntityResolver
from mythRL_core.entity_resolution.extractor import EntityExtractor

print("="*80)
print("Automotive Domain Demo - Community Contribution Example")
print("="*80 + "\n")

# Load automotive domain
print("Loading automotive domain...")
registry_path = Path("mythRL_core/domains/automotive/registry.json")
registry = EntityRegistry.load(registry_path)

stats = registry.stats()
print(f"  Domain: {stats['domain']}")
print(f"  Total entities: {stats['total_entities']}")
print(f"  Total aliases: {stats['total_aliases']}")
print(f"  By type:")
for entity_type, count in stats['entities_by_type'].items():
    print(f"    - {entity_type}: {count}")
print()

# Create auto-tagger with domain-specific measurement patterns
resolver = EntityResolver(registry)
extractor = EntityExtractor(resolver, custom_patterns=registry.measurement_patterns)
print("Auto-tagger ready with automotive measurement patterns!\n")

# Test with realistic automotive maintenance notes
test_notes = [
    "2024-10-20: Checked the Corolla at 87,450 miles - oil looks dirty and front left tire at 28 PSI",
    "Changed engine oil and filter today, used 4.4 quarts of 5W-30 synthetic, reset maintenance light",
    "Front passenger tire low at 25 psi, found small nail in tread. Patched and inflated to 32 PSI.",
    "Battery voltage reading 12.2V this morning - cranking seems slow, may need replacement soon",
    "Front brake pads measured at 3mm remaining, squealing during braking. Schedule replacement.",
    "Heard knocking sound from engine during cold start. Goes away after warmup. Oil level OK.",
    "Topped off coolant - level was low but fluid still looks clean and pink",
]

print("Processing automotive maintenance notes...\n")
print("="*80 + "\n")

for i, note in enumerate(test_notes, 1):
    print(f"Note {i}:")
    print(f"  Raw: '{note}'")
    print()

    # AUTO-TAG!
    extracted = extractor.extract(note)

    # Show entities
    print(f"  Entities found: {len(extracted.entities)}")
    for entity in extracted.entities:
        print(f"    '{entity['matched_text']}' -> {entity['canonical_id']} ({entity['entity_type']})")

    # Show measurements
    if extracted.measurements:
        print(f"\n  Measurements extracted:")
        for key, value in extracted.measurements.items():
            print(f"    {key}: {value}")

    print(f"\n  Timestamp: {extracted.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")

    # Generate Qdrant payload
    payload = extracted.to_qdrant_payload()

    print(f"\n  Qdrant Payload Preview:")
    print(f"    primary_entity: {payload.get('primary_entity_id', 'none')}")
    print(f"    entity_ids: {payload['entity_ids']}")
    print(f"    has_measurements: {payload['has_measurements']}")

    # Show specific measurements if present
    interesting_keys = ['tire_pressure_psi', 'mileage', 'oil_condition',
                       'battery_voltage', 'brake_pad_thickness_mm', 'sound_type']
    for key in interesting_keys:
        if key in payload:
            print(f"    {key}: {payload[key]}")

    print("\n" + "-"*80 + "\n")

print("="*80)
print("Automotive Domain Validation Complete!")
print("="*80 + "\n")

print("What this demonstrates:")
print("  1. Domain-agnostic architecture - same code, different domain")
print("  2. Entity resolution: 'corolla' -> vehicle-corolla-2015, 'oil' -> fluid-engine-oil")
print("  3. Automotive-specific measurements: tire pressure, mileage, battery voltage")
print("  4. Categorical extraction: oil condition, sound types, fluid levels")
print("  5. Ready for Qdrant storage with semantic search + filtering")
print()

print("Next steps for this domain:")
print("  1. Test micropolicies (tire pressure alerts, oil change intervals)")
print("  2. Test reverse queries (noise diagnosis, pressure loss)")
print("  3. Full pipeline demo with Qdrant")
print("  4. Community contribution - submit as example domain")
print()

print("This is how the community extends ExpertLoom!")
print("Same infrastructure, new domains, preserved expertise.")
