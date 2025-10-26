"""
Demo: Entity Resolution + Qdrant Integration
=============================================

Shows how entity resolver works with your beekeeping data
and automatically tags everything with canonical IDs.
"""

import sys
from pathlib import Path

# Add mythRL_core to path
sys.path.insert(0, str(Path(__file__).parent))

from mythRL_core.entity_resolution import EntityRegistry, EntityResolver

print("="*80)
print("Entity Resolver + Beekeeping Domain Demo")
print("="*80 + "\n")

# Step 1: Load beekeeping domain registry
print("Step 1: Loading beekeeping domain registry...")
registry_path = Path("mythRL_core/domains/beekeeping/registry.json")
registry = EntityRegistry.load(registry_path)

stats = registry.stats()
print(f"  Loaded domain: {stats['domain']}")
print(f"  Total entities: {stats['total_entities']}")
print(f"  Total aliases: {stats['total_aliases']}")
print(f"  By type:")
for entity_type, count in stats['entities_by_type'].items():
    print(f"    - {entity_type}: {count}")
print()

# Step 2: Create resolver
print("Step 2: Creating entity resolver...")
resolver = EntityResolver(registry)
print("  [OK] Resolver ready\n")

# Step 3: Test with real beekeeping notes
print("Step 3: Testing with realistic beekeeping notes...\n")

test_notes = [
    "Checked jodi today - saw 8 frames of brood, very active with goldenrod flow",
    "Applied thymol treatment round 2 to dennis and the split",
    "Half door hive is weak, only 2.5 frames. May need feeding.",
    "Smallest colony looks questionable for winter",
    "Jodi's hive and the double stack both looking strong"
]

for i, note in enumerate(test_notes, 1):
    print(f"Note {i}: {note}")
    print()

    # Resolve entities
    tagged = resolver.tag_text(note)

    if tagged['entity_count'] == 0:
        print("  No entities found\n")
        continue

    print(f"  Found {tagged['entity_count']} entities:")
    for entity in tagged['entities']:
        print(f"    '{entity['matched_text']}' -> {entity['canonical_id']}")
        print(f"      Type: {entity['entity_type']}, Confidence: {entity['confidence']}")

    print(f"\n  Canonical IDs for storage: {tagged['entity_ids']}")
    print()

# Step 4: Show how this would work with Qdrant
print("="*80)
print("How This Integrates with Qdrant")
print("="*80 + "\n")

print("""
When storing in Qdrant, you would:

1. Parse note text with entity resolver
2. Extract canonical IDs automatically
3. Store with structured metadata

Example:
  Note: "Checked jodi today - 8 frames of brood"

  Qdrant point:
    vector: [0.23, -0.45, ...]  # Embedding of text
    payload: {
      "text": "Checked jodi today - 8 frames of brood",
      "hive_id": "hive-003",              # Auto-resolved!
      "hive_name": "Jodi's Hive",         # Canonical name
      "timestamp": "2024-10-24",
      "frames_of_brood": 8.0,             # Extracted measurement
      "entity_ids": ["hive-003", "forage-goldenrod"]
    }

Now you can query by:
  - Semantic similarity (vector search)
  - Exact hive ID (filter: hive_id = "hive-003")
  - Time range (filter: timestamp > "2024-10-01")
  - Entity type (filter: entity_type = "hive")

ALL mentions of "jodi", "jodis", "jodi hive" are automatically
correlated to the same canonical ID!
""")

print("\n" + "="*80)
print("Next Steps")
print("="*80 + "\n")

print("1. Build auto-tagger that:")
print("   - Takes raw text (voice notes, manual logs)")
print("   - Resolves entities automatically")
print("   - Extracts measurements (regex or LLM)")
print("   - Stores in Qdrant with full metadata")
print()

print("2. Add temporal search:")
print("   - Time-weighted retrieval")
print("   - 'Recent vs historical' queries")
print("   - Trend detection over time")
print()

print("3. Build reverse query:")
print("   - System detects anomalies")
print("   - Asks YOU questions when uncertain")
print("   - 'Jodi dropped 2 frames - did you notice any issues?'")
print()

print("Ready to build the auto-tagger next!")
