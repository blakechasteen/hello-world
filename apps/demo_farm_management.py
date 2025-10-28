#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Farm Management Demo
====================
Demonstrates HoloLoom-based farm management apps:
- Beekeeping tracker with voice inspection notes
- Food-e tracker with grocery receipt OCR

Shows multimodal data ingestion via spinners and HoloLoom integration.
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from apps.farm_core.models import Asset, Log, AssetType, LogType
from apps.beekeeping.models import Hive, InspectionLog, QueenStatus, PopulationStrength
from apps.beekeeping.spinners.bee_inspection import process_bee_inspection


async def demo_beekeeping():
    """Demonstrate beekeeping app with voice notes."""
    print("=" * 70)
    print("BEEKEEPING DEMO - Voice Note Inspection Processing")
    print("=" * 70)

    # Simulate voice note transcript from beekeeper
    transcript = """
    Inspecting hive Alpha today, October 27th.
    Weather is sunny, about 68 degrees.
    Saw the queen on frame 3, she's laying well.
    Eggs visible in a nice solid pattern.
    Population looks strong, I'd say about 8 frames of bees.
    No signs of varroa mites or disease.
    Temperament is calm, very gentle hive.
    Honey stores look good, they have about 3 frames capped.
    Added a second super since they're building comb fast.
    Pollen coming in nicely.
    Will check again in two weeks.
    """

    print("\n--- RAW VOICE NOTE TRANSCRIPT ---")
    print(transcript)

    # Process with BeeInspectionAudioSpinner
    print("\n--- PROCESSING WITH SPINNER ---")
    shards = await process_bee_inspection(
        transcript=transcript,
        hive_id="Alpha",
        inspector="Blake",
        location="Backyard Apiary"
    )

    print(f"✓ Generated {len(shards)} memory shard(s)")

    # Display structured extraction
    if shards:
        shard = shards[0]
        print("\n--- STRUCTURED DATA EXTRACTION ---")
        print(f"Shard ID: {shard.id}")
        print(f"Episode: {shard.episode}")
        print(f"\nEntities: {', '.join(shard.entities)}")
        print(f"Motifs: {', '.join(shard.motifs)}")

        inspection_data = shard.metadata.get('inspection', {})
        print(f"\nExtracted Inspection Data:")
        print(f"  Hive ID: {inspection_data.get('hive_id')}")
        print(f"  Queen Status: {inspection_data.get('queen_status')}")
        print(f"  Population: {inspection_data.get('population_estimate')}")
        print(f"  Temperament: {inspection_data.get('temperament')}")
        print(f"  Health Issues: {inspection_data.get('health_issues')}")
        print(f"  Actions Taken: {inspection_data.get('actions_taken')}")
        print(f"  Honey Stores: {inspection_data.get('honey_stores')}")

        print("\n--- FORMATTED SUMMARY ---")
        print(shard.text)

    return shards


async def demo_models():
    """Demonstrate farm core data models."""
    print("\n\n" + "=" * 70)
    print("FARM CORE MODELS DEMO - Assets & Logs")
    print("=" * 70)

    # Create a hive asset
    print("\n--- CREATING HIVE ASSET ---")
    hive = Hive.create(
        name="Hive Alpha",
        location="Backyard Apiary",
        equipment_type="langstroth",
        num_boxes=2,
        queen_status=QueenStatus.PRESENT_LAYING,
        population=PopulationStrength.STRONG
    )

    print(f"Created: {hive.name}")
    print(f"  ID: {hive.asset_id}")
    print(f"  Type: {hive.asset_type.value}")
    print(f"  Location: {hive.location}")
    print(f"  Queen: {hive.queen_status.value}")
    print(f"  Population: {hive.population.value}")

    # Convert to memory shard
    print("\n--- CONVERTING TO MEMORY SHARD ---")
    shard = hive.to_shard()
    print(f"Shard ID: {shard.id}")
    print(f"Entities: {', '.join(shard.entities)}")
    print(f"Motifs: {', '.join(shard.motifs)}")
    print(f"\nShard Text:\n{shard.text}")

    # Create an inspection log
    print("\n--- CREATING INSPECTION LOG ---")
    inspection = InspectionLog.create(
        name="Weekly inspection",
        hive_id=hive.asset_id,
        description="Regular inspection showing strong colony",
        user="Blake",
        queen_status=QueenStatus.PRESENT_LAYING,
        population=PopulationStrength.STRONG,
        frames_of_bees=8,
        temperament="calm",
        honey_stores="good"
    )

    print(f"Created: {inspection.name}")
    print(f"  Log ID: {inspection.log_id}")
    print(f"  Type: {inspection.log_type.value}")
    print(f"  Timestamp: {inspection.timestamp}")
    print(f"  Hive: {inspection.asset_ids}")
    print(f"  User: {inspection.user}")

    # Convert log to shard
    log_shard = inspection.to_shard()
    print(f"\nLog Shard ID: {log_shard.id}")
    print(f"Episode: {log_shard.episode}")
    print(f"\nLog Text:\n{log_shard.text}")

    # Export to JSON
    print("\n--- EXPORTING TO JSON ---")
    hive_dict = hive.to_dict()
    log_dict = inspection.to_dict()

    import json
    print("Hive JSON:")
    print(json.dumps(hive_dict, indent=2, default=str))


async def demo_grocery_receipt():
    """Demonstrate food-e grocery receipt spinner."""
    print("\n\n" + "=" * 70)
    print("FOOD-E DEMO - Grocery Receipt OCR")
    print("=" * 70)

    # Simulate OCR text from receipt
    receipt_text = """
    WHOLE FOODS MARKET
    123 Main St, Boulder CO
    10/27/2025  3:45 PM

    ORGANIC BANANAS       $3.99
    MILK WHOLE GALLON     $5.49
    EGGS LARGE DOZEN      $4.99
    BREAD SOURDOUGH       $5.99
    CHICKEN BREAST        $12.99
    BROCCOLI ORGANIC      $3.49
    YOGURT GREEK          $6.99
    COFFEE BEANS          $14.99

    SUBTOTAL             $58.92
    TAX                   $3.54
    TOTAL                $62.46

    VISA ****1234
    """

    print("\n--- SIMULATED RECEIPT OCR ---")
    print(receipt_text)

    print("\n--- PROCESSING WITH GROCERY RECEIPT SPINNER ---")
    print("(Would use GroceryReceiptSpinner to extract structured data)")

    print("\nExpected extraction:")
    print("  Store: Whole Foods Market")
    print("  Date: 10/27/2025")
    print("  Items: 8")
    print("  Categories: produce (2), dairy (3), meat (1), bakery (1), beverages (1)")
    print("  Total: $62.46")


async def demo_integration():
    """Demonstrate how spinners integrate with HoloLoom."""
    print("\n\n" + "=" * 70)
    print("HOLOLOOM INTEGRATION - Spinners → Memory → Query")
    print("=" * 70)

    print("""
Workflow:
---------
1. RAW DATA (voice notes, receipts, photos)
   ↓
2. DOMAIN SPINNER (BeeInspectionAudioSpinner, GroceryReceiptSpinner)
   - Parses domain-specific structure
   - Extracts entities and motifs
   - Optional Ollama enrichment
   ↓
3. MEMORY SHARDS (standardized format)
   - id, text, entities, motifs, metadata
   ↓
4. HOLOLOOM STORAGE
   - Neo4j: Asset/log nodes + relationships (YarnGraph)
   - Qdrant: Text embeddings for semantic search
   ↓
5. QUERY INTERFACE (natural language)
   - "Which hives need inspection?"
   - "What did I buy last week?"
   - "Show health issues across all hives"
   ↓
6. WEAVING ORCHESTRATOR
   - Matryoshka embeddings (multi-scale)
   - Spectral graph features
   - Thompson Sampling decision engine
   ↓
7. SPACETIME OUTPUT
   - Natural language answer
   - Structured data
   - Full computational provenance

Example Queries:
----------------
Beekeeping:
  - "What's the queen status of hive Alpha?"
  - "Show all inspections with mite problems"
  - "Which hives are strong enough to split?"
  - "When was hive B last treated?"

Food-e:
  - "How much did I spend on groceries this month?"
  - "What produce did I buy last week?"
  - "Show all purchases from Whole Foods"
  - "Alert me when milk prices spike"

Cross-Domain:
  - "Show all activities this week across all farms"
  - "What patterns correlate with honey production?"
  - "Analyze spending vs seasonal bee forage"
    """)


async def main():
    """Run all demos."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 20 + "FARM MANAGEMENT SUITE" + " " * 27 + "║")
    print("║" + " " * 16 + "Powered by HoloLoom Neural Memory" + " " * 19 + "║")
    print("╚" + "=" * 68 + "╝")

    # Run demos
    await demo_beekeeping()
    await demo_models()
    await demo_grocery_receipt()
    await demo_integration()

    print("\n\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)

    print("""
Next Steps:
-----------
1. Initialize HoloLoom orchestrator with Neo4j + Qdrant backends
2. Implement FarmTracker.query() for natural language queries
3. Add more spinners (hive photos, soil tests, weather data)
4. Build CLI/API interface for mobile field entry
5. Create Grafana dashboards for farm metrics

Apps Created:
-------------
✓ apps/farm_core/       - Shared framework (models, tracker, protocols)
✓ apps/beekeeping/      - Apiary management with voice inspections
✓ apps/food-e/          - Food tracking with receipt OCR

To Use:
-------
from apps.beekeeping.spinners.bee_inspection import process_bee_inspection

shards = await process_bee_inspection(
    transcript="Saw queen laying, 8 frames of bees, calm temperament",
    hive_id="A",
    inspector="Blake"
)
# → Structured inspection data ready for HoloLoom storage
    """)


if __name__ == "__main__":
    asyncio.run(main())