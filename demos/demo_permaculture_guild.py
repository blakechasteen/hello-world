"""
Permaculture Guild Recommendation Demo

This demo shows how to use HoloLoom's RAG system for intelligent
companion planting recommendations.

**Created: 2025-11-21**

Usage:
    PYTHONPATH=. python demos/demo_permaculture_guild.py
"""

import asyncio
import os
from pathlib import Path

from HoloLoom.permaculture import (
    PlantDatabase,
    GuildRecommendationEngine,
    Layer,
    Function,
)
from HoloLoom.config import Config


def print_header(text: str) -> None:
    """Print formatted section header"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def print_plant(plant) -> None:
    """Pretty-print plant details"""
    print(f"\n🌱 {plant.common_name} ({plant.scientific_name})")
    print(f"   Family: {plant.family}")
    print(f"   Zones: {plant.hardiness_zones}")
    print(f"   Layers: {', '.join([l.value for l in plant.layers])}")
    print(f"   Functions: {', '.join([f.value for f in plant.functions])}")
    print(f"   Sun: {plant.sun_requirements} | Water: {plant.water_needs}")
    print(f"   Size: {plant.height_ft}ft tall × {plant.spread_ft}ft wide")
    if plant.description:
        print(f"   📝 {plant.description}")
    if plant.companions:
        print(f"   🤝 Companions: {', '.join(plant.companions[:5])}")


def print_guild_recommendation(rec) -> None:
    """Pretty-print guild recommendation"""
    print(f"\n🎯 Overall Score: {rec.score:.1%}")
    print(f"   Layer Diversity: {rec.layer_diversity_score:.1%}")
    print(f"   Function Diversity: {rec.function_diversity_score:.1%}")
    print(f"   Companion Compatibility: {rec.compatibility_score:.1%}")
    print(f"   Zone Compatible: {'✅' if rec.zone_compatibility else '⚠️ No'}")
    print(f"\n📊 Guild Analysis:")
    print("-" * 70)
    print(rec.reasoning)


async def demo_plant_database():
    """Demo 1: Load and explore plant database"""
    print_header("Demo 1: Plant Database")

    # Load plant database
    db_path = Path(__file__).parent.parent / "HoloLoom" / "permaculture" / "data" / "plants.json"
    print(f"\n📁 Loading plant database from: {db_path}")

    db = PlantDatabase.load_from_json(str(db_path))
    print(f"✅ Loaded {len(db)} plants")

    # Show a few example plants
    print("\n🌿 Sample Plants:")
    for name in ["Apple", "Comfrey", "White Clover", "Nasturtium"]:
        plant = db.get_plant(name)
        if plant:
            print_plant(plant)

    return db


async def demo_guild_recommendation(db: PlantDatabase):
    """Demo 2: Recommend a guild for an apple tree"""
    print_header("Demo 2: Guild Recommendation for Apple Tree")

    # Create guild engine with HoloLoom RAG
    print("\n🧠 Initializing HoloLoom RAG engine...")
    config = Config.fast()

    async with GuildRecommendationEngine(db, config) as engine:
        print("✅ RAG engine initialized (plant knowledge ingested)")

        # Recommend guild for apple tree in zone 6
        print("\n🍎 Generating guild recommendation for Apple tree (Zone 6)...")
        print("   Target: Diverse layers + key permaculture functions")

        recommendation = await engine.recommend_guild(
            central_plant_name="Apple",
            climate_zone=6,
            max_plants=7,
        )

        print_guild_recommendation(recommendation)

        return recommendation


async def demo_semantic_search(db: PlantDatabase):
    """Demo 3: Semantic plant search using RAG"""
    print_header("Demo 3: Semantic Plant Search (RAG)")

    config = Config.fast()

    async with GuildRecommendationEngine(db, config) as engine:
        # Example queries
        queries = [
            "nitrogen fixing ground covers for zone 5",
            "pest deterrent herbs that attract pollinators",
            "shade tolerant berry shrubs",
            "deep-rooted dynamic accumulators",
        ]

        for query in queries:
            print(f"\n🔍 Query: '{query}'")
            results = await engine.query_plants(query, limit=3)

            if results:
                print(f"   Found {len(results)} matches:")
                for plant in results:
                    print(f"   • {plant.common_name} - {plant.description[:60]}...")
            else:
                print("   No matches found")


async def demo_custom_guild():
    """Demo 4: Create guild with specific requirements"""
    print_header("Demo 4: Custom Guild Requirements")

    db_path = Path(__file__).parent.parent / "HoloLoom" / "permaculture" / "data" / "plants.json"
    db = PlantDatabase.load_from_json(str(db_path))

    config = Config.fast()

    async with GuildRecommendationEngine(db, config) as engine:
        print("\n🎯 Custom Requirements:")
        print("   Central Plant: Hazelnut")
        print("   Climate Zone: 7")
        print("   Focus: Food production + nitrogen fixation + pest control")

        # Custom target functions
        target_functions = {
            Function.FOOD,
            Function.NITROGEN_FIXER,
            Function.PEST_DETERRENT,
            Function.POLLINATOR_ATTRACTOR,
        }

        recommendation = await engine.recommend_guild(
            central_plant_name="Hazelnut",
            climate_zone=7,
            target_functions=target_functions,
            max_plants=6,
        )

        print_guild_recommendation(recommendation)


async def main():
    """Run all demos"""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║        🌳 HoloLoom Permaculture Guild Recommendation System 🌳       ║
║                                                                      ║
║  Intelligent companion planting using RAG and permaculture wisdom   ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
""")

    try:
        # Demo 1: Plant database
        db = await demo_plant_database()

        # Demo 2: Guild recommendation
        await demo_guild_recommendation(db)

        # Demo 3: Semantic search
        await demo_semantic_search(db)

        # Demo 4: Custom guild
        await demo_custom_guild()

        print_header("✅ All Demos Complete!")
        print("""
Next Steps:
  • Add more plants to HoloLoom/permaculture/data/plants.json
  • Customize guild requirements (layers, functions, max_plants)
  • Try semantic search with natural language queries
  • Integrate with site analysis tools
  • Build visual guild designer

Happy permaculture designing! 🌱
""")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
