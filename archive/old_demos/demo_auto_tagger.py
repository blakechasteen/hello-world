"""
Demo: Auto-Tagger Pipeline - Text -> Structured Qdrant Storage
===============================================================

The complete pipeline that takes raw text and produces
structured, entity-tagged payloads ready for Qdrant.
"""

import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from mythRL_core.entity_resolution import EntityRegistry, EntityResolver
from mythRL_core.entity_resolution.extractor import EntityExtractor

print("="*80)
print("Auto-Tagger Pipeline Demo")
print("="*80 + "\n")

# Load beekeeping domain
print("Loading beekeeping domain...")
registry_path = Path("mythRL_core/domains/beekeeping/registry.json")
registry = EntityRegistry.load(registry_path)
print(f"  Loaded {len(registry.entities)} entities\n")

# Create pipeline
resolver = EntityResolver(registry)
extractor = EntityExtractor(resolver)
print("Auto-tagger pipeline ready!\n")

# Test with realistic beekeeping notes
test_notes = [
    "2024-10-12: Checked jodi - 8 frames of brood, solid pattern, very calm",
    "Applied thymol treatment to dennis and the split. Dosage: 50g each.",
    "Half door hive weak at 2.5 frames. Temperament aggressive today. Temp 80F.",
    "Smallest colony questionable - only 2 frames, may not survive winter",
    "Dennis's hive super strong at 14.5 frames! Solid brood, gentle bees."
]

print("Processing notes through auto-tagger...\n")
print("="*80 + "\n")

for i, note in enumerate(test_notes, 1):
    print(f"Note {i}:")
    print(f"  Raw: '{note}'")
    print()

    # AUTO-TAG!
    extracted = extractor.extract(note)

    # Show what was extracted
    print(f"  Entities found: {len(extracted.entities)}")
    for entity in extracted.entities:
        print(f"    '{entity['matched_text']}' -> {entity['canonical_id']} ({entity['entity_type']})")

    if extracted.measurements:
        print(f"\n  Measurements extracted:")
        for key, value in extracted.measurements.items():
            print(f"    {key}: {value}")

    print(f"\n  Timestamp: {extracted.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")

    # Generate Qdrant payload
    payload = extracted.to_qdrant_payload()

    print(f"\n  Qdrant Payload:")
    print(f"    text: '{payload['text'][:60]}...'")
    print(f"    primary_entity_id: {payload.get('primary_entity_id', 'none')}")
    print(f"    entity_ids: {payload['entity_ids']}")
    print(f"    timestamp: {payload.get('timestamp')}")
    print(f"    has_measurements: {payload['has_measurements']}")

    # Show specific measurements if present
    interesting_keys = ['frames_of_brood', 'frames_of_bees', 'temperament',
                       'brood_pattern', 'dosage_grams', 'temperature_f']
    for key in interesting_keys:
        if key in payload:
            print(f"    {key}: {payload[key]}")

    print("\n" + "-"*80 + "\n")

print("="*80)
print("Pipeline Complete!")
print("="*80 + "\n")

print("What just happened:")
print("  1. Parsed plain English notes")
print("  2. Resolved 'jodi' -> hive-003, 'dennis' -> hive-002, etc.")
print("  3. Extracted measurements (frames, temp, dosage)")
print("  4. Extracted categorical data (temperament, brood pattern)")
print("  5. Parsed timestamps")
print("  6. Created structured Qdrant payloads")
print()

print("Next: Store these in Qdrant with embeddings!")
print("  - Vector search for semantic similarity")
print("  - Filter by hive_id, timestamp, measurements")
print("  - All notes about same hive automatically correlated")
