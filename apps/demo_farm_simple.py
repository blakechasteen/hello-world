#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Farm Management Simple Demo
============================
Demonstrates farm management spinners and models WITHOUT full HoloLoom integration.
Shows data processing pipeline: raw data → spinners → structured shards.
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime
import json

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import only models and spinners (no orchestrator) - direct imports to avoid __init__ issues
import apps.farm_core.models as farm_models
import apps.beekeeping.models as bee_models
from apps.beekeeping.spinners.bee_inspection import process_bee_inspection

# Alias for convenience
Asset = farm_models.Asset
Log = farm_models.Log
AssetType = farm_models.AssetType
LogType = farm_models.LogType
AssetStatus = farm_models.AssetStatus

Hive = bee_models.Hive
InspectionLog = bee_models.InspectionLog
QueenStatus = bee_models.QueenStatus
PopulationStrength = bee_models.PopulationStrength
Temperament = bee_models.Temperament
HiveEquipmentType = bee_models.HiveEquipmentType


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

    print(f"OK Generated {len(shards)} memory shard(s)")

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
        equipment_type=HiveEquipmentType.LANGSTROTH,
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
    print(f"  Equipment: {hive.equipment_type.value}")

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
        log_type=LogType.OBSERVATION,  # Explicit since we're using factory
        description="Regular inspection showing strong colony",
        user="Blake",
        queen_status=QueenStatus.PRESENT_LAYING,
        population=PopulationStrength.STRONG,
        frames_of_bees=8,
        temperament=Temperament.CALM,
        honey_stores="good"
    )

    print(f"Created: {inspection.name}")
    print(f"  Log ID: {inspection.log_id}")
    print(f"  Type: {inspection.log_type.value}")
    print(f"  Timestamp: {inspection.timestamp}")
    print(f"  Hive: {inspection.asset_ids}")
    print(f"  User: {inspection.user}")
    print(f"  Queen: {inspection.queen_status.value}")
    print(f"  Population: {inspection.population.value}")

    # Convert log to shard
    log_shard = inspection.to_shard()
    print(f"\nLog Shard ID: {log_shard.id}")
    print(f"Episode: {log_shard.episode}")
    print(f"\nLog Text:\n{log_shard.text}")

    # Export to JSON
    print("\n--- EXPORTING TO JSON ---")
    hive_dict = hive.to_dict()
    log_dict = inspection.to_dict()

    print("Hive JSON (first 500 chars):")
    hive_json = json.dumps(hive_dict, indent=2, default=str)
    print(hive_json[:500] + "...")


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
5. QUERY INTERFACE (natural language via WeavingOrchestrator)
   - "Which hives need inspection?"
   - "What did I buy last week?"
   - "Show health issues across all hives"
   ↓
6. WEAVING CYCLE
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
    print("=" * 70)
    print(" " * 20 + "FARM MANAGEMENT SUITE")
    print(" " * 16 + "Powered by HoloLoom Neural Memory")
    print("=" * 70)

    # Run demos
    await demo_beekeeping()
    await demo_models()
    await demo_integration()

    print("\n\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)

    print("""
Architecture Summary:
--------------------
OK apps/farm_core/models.py       - Base Asset & Log classes (farmOS-inspired)
OK apps/farm_core/tracker.py      - FarmTracker base class (HoloLoom integration)
OK apps/beekeeping/models.py      - Hive, Queen, Apiary assets + specialized logs
OK apps/beekeeping/spinners/      - BeeInspectionAudioSpinner for voice notes
OK apps/food-e/spinners/          - GroceryReceiptSpinner for receipt OCR

Data Flow:
----------
Voice Note → BeeInspectionAudioSpinner → MemoryShard → HoloLoom (Neo4j + Qdrant)
          ↑                              ↑
  Automatic extraction:         Standardized format:
  - Hive ID                     - ID, text, entities
  - Queen status               - Motifs, metadata
  - Population                 - Episode linking
  - Health issues
  - Actions taken

Next Steps:
-----------
1. Complete FarmTracker integration with WeavingOrchestrator
2. Implement natural language query() method
3. Add more spinners (hive photos, soil tests, weather)
4. Build CLI/API for field data entry
5. Create Grafana dashboards

To Use Spinners:
----------------
from apps.beekeeping.spinners.bee_inspection import process_bee_inspection

shards = await process_bee_inspection(
    transcript="Saw queen laying, 8 frames of bees, calm temperament",
    hive_id="A",
    inspector="Blake"
)
# → Returns structured MemoryShards ready for HoloLoom storage
    """)


if __name__ == "__main__":
    asyncio.run(main())