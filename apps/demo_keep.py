#!/usr/bin/env python3
"""
Demo script for Keep beekeeping management application.

Demonstrates:
1. Creating an apiary with multiple hives
2. Adding colonies to hives
3. Recording inspections with various findings
4. Generating alerts based on inspection data
5. Using BeeKeeper assistant for recommendations
6. Tracking harvests
7. Generating reports

Run with: python apps/demo_keep.py
"""

import asyncio
from datetime import datetime, timedelta
from apps.keep import (
    Apiary,
    Hive,
    Colony,
    Inspection,
    HarvestRecord,
    BeeKeeper,
    HealthStatus,
    QueenStatus,
)
from apps.keep.types import HiveType, InspectionType, AlertLevel


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


async def main():
    """Run the Keep demo."""

    print_section("Keep - Beekeeping Management Demo")

    # 1. Create an apiary
    print("\n1. Creating Apiary...")
    apiary = Apiary(
        name="Sunny Meadows Apiary",
        location="Rural County, State",
        metadata={"owner": "Demo Beekeeper", "established": "2023"}
    )
    print(f"   Created: {apiary}")

    # 2. Add hives
    print("\n2. Adding Hives...")
    hive1 = Hive(
        name="Hive 001 - Alpha",
        hive_type=HiveType.LANGSTROTH,
        location="East field, near oak trees",
        installation_date=datetime(2024, 4, 1),
        notes="First hive, good location with morning sun"
    )
    apiary.add_hive(hive1)
    print(f"   Added: {hive1}")

    hive2 = Hive(
        name="Hive 002 - Beta",
        hive_type=HiveType.LANGSTROTH,
        location="West field, near pond",
        installation_date=datetime(2024, 4, 1),
    )
    apiary.add_hive(hive2)
    print(f"   Added: {hive2}")

    hive3 = Hive(
        name="Hive 003 - Gamma",
        hive_type=HiveType.TOP_BAR,
        location="Garden area",
        installation_date=datetime(2024, 5, 15),
    )
    apiary.add_hive(hive3)
    print(f"   Added: {hive3}")

    # 3. Add colonies
    print("\n3. Adding Colonies...")
    colony1 = Colony(
        hive_id=hive1.hive_id,
        origin="package",
        breed="Italian",
        established_date=datetime(2024, 4, 5),
        queen_age_months=6,
        population_estimate=50000,
        health_status=HealthStatus.EXCELLENT,
        queen_status=QueenStatus.PRESENT_LAYING,
        notes="Strong colony, good brood pattern"
    )
    apiary.add_colony(colony1)
    print(f"   Added: {colony1}")

    colony2 = Colony(
        hive_id=hive2.hive_id,
        origin="split from Hive 001",
        breed="Italian",
        established_date=datetime(2024, 5, 1),
        queen_age_months=2,
        population_estimate=30000,
        health_status=HealthStatus.GOOD,
        queen_status=QueenStatus.PRESENT_LAYING,
    )
    apiary.add_colony(colony2)
    print(f"   Added: {colony2}")

    colony3 = Colony(
        hive_id=hive3.hive_id,
        origin="swarm capture",
        breed="unknown",
        established_date=datetime(2024, 5, 20),
        population_estimate=25000,
        health_status=HealthStatus.FAIR,
        queen_status=QueenStatus.PRESENT_NOT_LAYING,
        notes="New swarm, queen not laying yet"
    )
    apiary.add_colony(colony3)
    print(f"   Added: {colony3}")

    # 4. Record inspections
    print("\n4. Recording Inspections...")

    # Hive 1 - Excellent inspection
    inspection1 = Inspection(
        hive_id=hive1.hive_id,
        colony_id=colony1.colony_id,
        timestamp=datetime.now() - timedelta(days=3),
        inspection_type=InspectionType.ROUTINE,
        weather="Sunny, light breeze",
        temperature=72.0,
        findings={
            "queen_seen": True,
            "eggs_seen": True,
            "larvae_seen": True,
            "capped_brood_seen": True,
            "frames_with_brood": 8,
            "frames_with_honey": 6,
            "frames_with_pollen": 3,
            "population_estimate": 50000,
            "swarm_cells": 0,
            "mites_observed": False,
            "beetles_observed": False,
            "disease_signs": [],
            "notes": "Excellent colony, strong brood pattern"
        },
        actions_taken=[
            "Checked all frames",
            "Added honey super",
            "Marked queen with blue dot"
        ],
        recommendations=["Continue routine inspections"],
        inspector="Demo Beekeeper",
        duration_minutes=25
    )
    apiary.record_inspection(inspection1)
    print(f"   Recorded: {inspection1}")

    # Hive 2 - Good inspection with minor pest observation
    inspection2 = Inspection(
        hive_id=hive2.hive_id,
        colony_id=colony2.colony_id,
        timestamp=datetime.now() - timedelta(days=5),
        inspection_type=InspectionType.ROUTINE,
        weather="Partly cloudy",
        temperature=68.0,
        findings={
            "queen_seen": False,
            "eggs_seen": True,
            "larvae_seen": True,
            "capped_brood_seen": True,
            "frames_with_brood": 5,
            "frames_with_honey": 4,
            "frames_with_pollen": 2,
            "population_estimate": 30000,
            "swarm_cells": 0,
            "mites_observed": True,
            "beetles_observed": False,
            "disease_signs": [],
            "notes": "Queen not seen but eggs present, some mites observed"
        },
        actions_taken=[
            "Thorough frame inspection",
            "Noted mite presence"
        ],
        recommendations=["Monitor mite levels", "Consider treatment if counts increase"],
        inspector="Demo Beekeeper",
        duration_minutes=30
    )
    apiary.record_inspection(inspection2)
    print(f"   Recorded: {inspection2}")

    # Hive 3 - Concern with queen
    inspection3 = Inspection(
        hive_id=hive3.hive_id,
        colony_id=colony3.colony_id,
        timestamp=datetime.now() - timedelta(days=2),
        inspection_type=InspectionType.HEALTH_CHECK,
        weather="Sunny",
        temperature=75.0,
        findings={
            "queen_seen": True,
            "eggs_seen": False,
            "larvae_seen": False,
            "capped_brood_seen": False,
            "frames_with_brood": 0,
            "frames_with_honey": 3,
            "frames_with_pollen": 1,
            "population_estimate": 25000,
            "swarm_cells": 0,
            "supersedure_cells": 2,
            "beetles_observed": True,
            "disease_signs": [],
            "notes": "Queen present but not laying, supersedure cells present, small hive beetles seen"
        },
        actions_taken=[
            "Thorough inspection",
            "Noted queen status",
            "Installed beetle traps"
        ],
        recommendations=["Monitor queen closely", "May need to requeen if no improvement"],
        inspector="Demo Beekeeper",
        duration_minutes=35
    )
    apiary.record_inspection(inspection3)
    print(f"   Recorded: {inspection3}")

    # 5. Record a harvest
    print("\n5. Recording Harvest...")
    harvest1 = HarvestRecord(
        hive_id=hive1.hive_id,
        timestamp=datetime.now() - timedelta(days=10),
        product_type="honey",
        quantity=45.0,
        unit="lbs",
        moisture_content=17.2,
        quality_notes="Light amber color, excellent quality",
        processing_notes="Extracted and bottled"
    )
    apiary.record_harvest(harvest1)
    print(f"   Recorded: {harvest1}")

    # 6. Show apiary summary
    print("\n6. Apiary Summary...")
    summary = apiary.get_apiary_summary()
    for key, value in summary.items():
        print(f"   {key.replace('_', ' ').title()}: {value}")

    # 7. Show active alerts
    print("\n7. Active Alerts...")
    alerts = apiary.get_active_alerts()
    if alerts:
        for alert in alerts:
            hive = apiary.hives.get(alert.hive_id)
            hive_name = hive.name if hive else alert.hive_id
            print(f"\n   [{alert.level.value.upper()}] {alert.title}")
            print(f"   Hive: {hive_name}")
            print(f"   Message: {alert.message}")
            print(f"   Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M')}")
    else:
        print("   No active alerts")

    # 8. Initialize BeeKeeper assistant
    print("\n8. Initializing BeeKeeper Assistant...")
    keeper = BeeKeeper(apiary, hololoom_enabled=False)
    print(f"   Initialized: {keeper}")

    # 9. Get recommendations
    print("\n9. BeeKeeper Recommendations...")
    recommendations = await keeper.get_recommendations(days_ahead=14)

    if recommendations:
        for i, rec in enumerate(recommendations[:5], 1):  # Show top 5
            print(f"\n   Recommendation #{i}")
            print(f"   Priority: {rec.priority.value.upper()}")
            print(f"   Title: {rec.title}")
            print(f"   Reasoning: {rec.reasoning}")
            print(f"   Actions:")
            for action in rec.actions:
                print(f"     - {action}")
            print(f"   Timeline: {rec.timeline}")
    else:
        print("   No recommendations at this time")

    # 10. Get hive reports
    print("\n10. Detailed Hive Reports...")
    for hive_id, hive in list(apiary.hives.items())[:2]:  # First 2 hives
        print(f"\n   Report for {hive.name}:")
        report = keeper.get_hive_report(hive_id)

        print(f"   Hive Type: {report['hive']['type']}")
        print(f"   Location: {report['hive']['location']}")

        if report['colony']:
            print(f"   Colony Health: {report['colony']['health_status']}")
            print(f"   Queen Status: {report['colony']['queen_status']}")
            print(f"   Population: ~{report['colony']['population_estimate']:,} bees")

        print(f"   Recent Inspections: {report['recent_activity']['inspections_last_30_days']}")
        print(f"   Active Alerts: {report['recent_activity']['active_alerts']}")

    # 11. Ask BeeKeeper questions
    print("\n11. Asking BeeKeeper Questions...")
    questions = [
        "How many hives do I have?",
        "What is the colony health status?",
        "How much honey was harvested this year?",
    ]

    for question in questions:
        answer = await keeper.ask_question(question)
        print(f"\n   Q: {question}")
        print(f"   A: {answer}")

    # 12. Show inspection history
    print("\n12. Recent Inspection History...")
    for hive_id, hive in apiary.hives.items():
        history = apiary.get_hive_history(hive_id, days=30)
        if history:
            print(f"\n   {hive.name}:")
            for insp in history:
                print(f"     - {insp.timestamp.strftime('%Y-%m-%d')}: {insp.inspection_type.value}")
                if insp.findings.get('queen_seen'):
                    print(f"       Queen: Seen")
                if insp.findings.get('mites_observed'):
                    print(f"       Alert: Mites observed")

    # Summary
    print_section("Demo Complete")
    print(f"\nSuccessfully demonstrated Keep beekeeping management:")
    print(f"  - Created apiary with {len(apiary.hives)} hives")
    print(f"  - Managed {len(apiary.colonies)} colonies")
    print(f"  - Recorded {len(apiary.inspections)} inspections")
    print(f"  - Tracked {len(apiary.harvests)} harvests")
    print(f"  - Generated {len(apiary.alerts)} alerts")
    print(f"  - Provided {len(recommendations)} intelligent recommendations")
    print("\nKeep is ready for production use!")
    print("\nNext steps:")
    print("  - See apps/keep/README.md for detailed documentation")
    print("  - Enable HoloLoom integration for enhanced reasoning")
    print("  - Add your own apiaries and start tracking!")


if __name__ == "__main__":
    # Run the async demo
    asyncio.run(main())
