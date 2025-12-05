"""
Zero-G MVP Demo: Simple Mission

This script demonstrates a complete Zero-G mission from Preflight through Orbit.

Mission Objective:
- Onboard local JSON files (users.json, products.json)
- Execute the full launch sequence
- Achieve stable orbit
- Query the Loom
- Land gracefully

This is the "Hello World" of Zero-G.
"""

import asyncio
import json
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "backend"))

from loom_core.simple_loom import create_simple_loom_core
from launch_system.simple_orchestrator import create_simple_launch_orchestrator
from g_series.g1_contact.simple_json_connector import create_json_connector
from g_series.protocols import DataSensitivity


async def create_demo_data():
    """Create demo JSON files for the mission"""
    print("📁 Creating demo data files...")

    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)

    # Create users.json
    users = [
        {"id": 1, "name": "Alice Johnson", "email": "alice@example.com", "role": "engineer"},
        {"id": 2, "name": "Bob Smith", "email": "bob@example.com", "role": "designer"},
        {"id": 3, "name": "Carol Williams", "email": "carol@example.com", "role": "manager"},
        {"id": 4, "name": "David Brown", "email": "david@example.com", "role": "engineer"},
        {"id": 5, "name": "Eve Davis", "email": "eve@example.com", "role": "analyst"}
    ]

    with open(data_dir / "users.json", "w") as f:
        json.dump(users, f, indent=2)

    # Create products.json
    products = [
        {"id": 101, "name": "Widget A", "price": 29.99, "category": "tools"},
        {"id": 102, "name": "Gadget B", "price": 49.99, "category": "electronics"},
        {"id": 103, "name": "Device C", "price": 99.99, "category": "electronics"},
        {"id": 104, "name": "Tool D", "price": 19.99, "category": "tools"}
    ]

    with open(data_dir / "products.json", "w") as f:
        json.dump(products, f, indent=2)

    print(f"✅ Demo data created in {data_dir}")
    return data_dir


async def run_mission():
    """Execute the complete Zero-G mission"""

    print("\n" + "="*60)
    print("🚀 ZERO-G MVP MISSION DEMO")
    print("="*60 + "\n")

    # Step 0: Create demo data
    data_dir = await create_demo_data()

    # Step 1: Initialize Loom Core
    print("\n🔧 Initializing Loom Core...")
    loom = await create_simple_loom_core()
    print("✅ Loom Core initialized\n")

    # Step 2: Create Launch Orchestrator
    print("🔧 Creating Launch Orchestrator...")
    orchestrator = await create_simple_launch_orchestrator(loom)
    print("✅ Launch Orchestrator ready\n")

    # Step 3: Create G1 Connector
    print("🔧 Creating G1 JSON Connector...")
    connector = await create_json_connector(str(data_dir))
    print("✅ G1 Connector ready\n")

    # Step 4: Connect to data sources (G1 Contact)
    print("📡 Connecting to data sources (G1 Zero-Move)...")

    health_users = await connector.connect(
        source_id="users",
        file_path="users.json",
        sensitivity=DataSensitivity.INTERNAL
    )
    print(f"   users.json: {health_users.status}")

    health_products = await connector.connect(
        source_id="products",
        file_path="products.json",
        sensitivity=DataSensitivity.INTERNAL
    )
    print(f"   products.json: {health_products.status}")

    # Get metadata (zero-move - no data read)
    print("\n📊 Retrieving metadata (zero-move)...")
    users_meta = await connector.get_metadata("users")
    products_meta = await connector.get_metadata("products")

    print(f"   users.json: {users_meta['file_size_human']}")
    print(f"   products.json: {products_meta['file_size_human']}")

    print("✅ Data sources connected (zero-move)\n")

    # Step 5: PREFLIGHT
    mission_params = {
        "astronaut_id": "demo_astronaut",
        "permissions": {"read": True, "write": False},
        "environment": {"network": "local", "storage": "disk"},
        "data_sources": ["users.json", "products.json"]
    }

    preflight_report = await orchestrator.execute_preflight(mission_params)

    if not preflight_report.can_proceed:
        print("❌ Preflight failed. Aborting mission.")
        return

    print(f"✅ Preflight Status: {preflight_report.overall_status.value.upper()}\n")

    # Step 6: COUNTDOWN
    input("\nPress ENTER to begin countdown... ")
    print()

    countdown_status = await orchestrator.execute_countdown()
    print(f"✅ Countdown complete\n")

    # Step 7: LIFTOFF
    liftoff_check = await orchestrator.execute_liftoff()
    print(f"✅ Lift-Off Status: {liftoff_check.status.value.upper()}\n")

    # Step 8: Schema Discovery (part of Boost)
    print("🔬 Discovering schemas...")
    users_schema = await connector.discover_schema("users", sample_size=100)
    products_schema = await connector.discover_schema("products", sample_size=100)

    print(f"   users schema: {len(users_schema.elements)} elements")
    print(f"   products schema: {len(products_schema.elements)} elements")
    print()

    # Step 9: BOOST
    boost_check = await orchestrator.execute_boost()
    print(f"✅ Boost Status: {boost_check.status.value.upper()}\n")

    # Step 10: ORBIT
    orbit_status = await orchestrator.achieve_orbit()
    print(f"✅ ORBIT ACHIEVED - Altitude: {orbit_status.orbital_altitude}\n")

    print("="*60)
    print("🌍 ZERO-G IS NOW IN STABLE ORBIT")
    print("="*60 + "\n")

    # Step 11: Query the Loom (while in orbit)
    print("🧠 Querying the Loom...\n")

    queries = [
        "What is Thompson Sampling?",
        "Explain the exploration-exploitation tradeoff",
        "How does Zero-G handle data onboarding?"
    ]

    for i, query in enumerate(queries, 1):
        print(f"Query {i}: {query}")
        result = await loom.weave(query)

        print(f"   Tool used: {result['action']}")
        print(f"   Duration: {result['duration_ms']:.1f}ms")
        print(f"   Threads consulted: {result['threads_consulted']}")
        print(f"   Response: {result['result'][:100]}...")
        print()

    # Step 12: Get Orbit Status
    final_status = await orchestrator.orbit.get_orbit_status()
    print(f"🛰️  Orbit Status:")
    print(f"   Altitude: {final_status.orbital_altitude}")
    print(f"   Apps docked: {len(final_status.apps_docked)}")
    print(f"   Uptime: {final_status.uptime_seconds:.1f}s")
    print(f"   Yarn stable: {final_status.yarn_stable}")
    print(f"   Warp aligned: {final_status.warp_aligned}")
    print()

    # Step 13: LANDING
    input("\nPress ENTER to begin landing sequence... ")
    print()

    landing_check = await orchestrator.land()
    print(f"✅ Landing Status: {landing_check.status.value.upper()}\n")

    # Step 14: Cleanup
    print("🧹 Cleaning up...")
    await connector.disconnect("users")
    await connector.disconnect("products")
    await loom.shutdown()
    print("✅ Cleanup complete\n")

    print("="*60)
    print("✅ MISSION COMPLETE - ZERO-G DEMO SUCCESSFUL")
    print("="*60 + "\n")

    print("📖 What just happened:")
    print("   1. Initialized Loom Core (reasoning engine)")
    print("   2. Created Launch Orchestrator (lifecycle manager)")
    print("   3. Connected to JSON files via G1 Zero-Move connector")
    print("   4. Executed Preflight checklist")
    print("   5. Counted down from T-10 to T-0")
    print("   6. Lifted off with metadata-only scanning")
    print("   7. Boosted with schema discovery and indexing")
    print("   8. Achieved stable orbit")
    print("   9. Queried the Loom successfully")
    print("   10. Landed gracefully")
    print()
    print("🎯 Zero-G successfully demonstrated:")
    print("   • Zero-move data access (metadata only)")
    print("   • Safe, staged lifecycle (Preflight → Orbit)")
    print("   • Rollback-capable operations")
    print("   • Complete provenance logging")
    print("   • Loom-based reasoning")
    print()
    print("Next steps:")
    print("   • Try the EVA demo: python demo_eva_mission.py")
    print("   • Explore the Cabin UI: cd ../../frontend/cabin_ui && npm start")
    print("   • Read the docs: ../../docs/ARCHITECTURE_OVERVIEW.md")
    print()


if __name__ == "__main__":
    # Run the mission
    asyncio.run(run_mission())
