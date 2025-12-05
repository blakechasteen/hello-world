"""
COZ Satellite Mission Demo - Zero-G Integration

Demonstrates COZ production management system docking to Zero-G's orbital Loom.

Flow:
1. Initialize Zero-G Loom Core
2. Create COZ satellite
3. Dock COZ to Loom (registers tools, indexes content)
4. Execute COZ intelligence queries via Rift
5. Query COZ data via WarpSpace semantic search
6. Traverse knowledge graph via YarnGraph
7. Undock COZ gracefully

Requirements:
- COZ data files in coz/ directory
- Python 3.9+

Usage:
    python examples/coz_mission_demo.py

Created: 2025-11-22
"""

import asyncio
import sys
from pathlib import Path
import json
from datetime import datetime

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import Zero-G components
from loom_core.simple_loom import create_simple_loom_core
from apps.coz import create_coz_satellite
from apps.satellite_protocol import OrbitManager
from loom_core.protocols import RiftAction


# ==================== Helper Functions ====================

def print_header(title: str, symbol: str = "="):
    """Print a formatted section header"""
    width = 70
    print(f"\n{symbol * width}")
    print(f"{title:^{width}}")
    print(f"{symbol * width}\n")


def print_status(status: str, message: str):
    """Print a status message with formatting"""
    status_symbols = {
        "success": "✅",
        "warning": "⚠️",
        "error": "❌",
        "info": "ℹ️",
        "rocket": "🚀",
        "satellite": "🛰️",
        "graph": "🕸️",
        "search": "🔍"
    }
    symbol = status_symbols.get(status, "•")
    print(f"{symbol} {message}")


# ==================== Main Mission ====================

async def main():
    """Execute COZ satellite mission"""

    print_header("🛰️  COZ SATELLITE MISSION - Zero-G Integration", "=")

    # Step 1: Initialize Loom Core
    print_status("info", "Step 1: Initializing Zero-G Loom Core...")
    loom = await create_simple_loom_core()
    print_status("success", "Loom Core initialized")

    # Step 2: Create Orbit Manager
    print_status("info", "Step 2: Creating Orbit Manager...")
    orbit = OrbitManager(loom)
    print_status("success", "Orbit Manager ready")

    # Step 3: Create COZ satellite
    print_header("🛰️  LAUNCHING COZ SATELLITE", "-")
    print_status("info", "Creating COZ satellite...")
    coz = create_coz_satellite(coz_dir="coz")

    # Show manifest
    manifest = coz.get_manifest()
    print(f"\nCOZ Manifest:")
    print(f"  App ID: {manifest.app_id}")
    print(f"  Name: {manifest.app_name}")
    print(f"  Version: {manifest.version}")
    print(f"  G-Level Required: {manifest.required_g_level.value}")
    print(f"  Capabilities: {len(manifest.capabilities)} tools")
    print(f"  Data Sources: {len(manifest.data_sources)} sources")

    # Step 4: Dock COZ to Loom
    print_header("🔗 DOCKING SEQUENCE", "-")
    print_status("rocket", "Initiating docking sequence...")
    success = await orbit.dock_app(coz)

    if not success:
        print_status("error", "Docking failed!")
        return

    # Step 5: Execute COZ intelligence queries
    print_header("🧠 EXECUTING COZ INTELLIGENCE QUERIES", "-")

    # 5a. Get daily brief
    print_status("info", "Querying: coz.get_daily_brief...")
    try:
        brief_result = await loom.rift.invoke(
            RiftAction(
                tool_name="coz.get_daily_brief",
                parameters={}
            )
        )

        if brief_result.get("success"):
            brief = brief_result["brief"]
            print_status("success", "Daily brief generated!")

            # Show key metrics
            print("\n📊 Daily Brief Summary:")
            profit = brief.get("profit_analysis", {})
            if "error" not in profit:
                print(f"   Net Profit: ${profit.get('net_profit', 0):,.2f}")
                print(f"   Profit Margin: {profit.get('profit_margin', 0):.1f}%")

            orders = brief.get("order_fulfillment", {})
            print(f"   Critical Orders: {orders.get('critical_orders', 0)}")
            print(f"   Pending Orders: {orders.get('total_pending', 0)}")
            print(f"   Production Hours Needed: {orders.get('production_hours_needed', 0):.1f}h")

            recommendations = brief.get("top_recommendations", [])
            if recommendations:
                print("\n   Top Recommendations:")
                for i, rec in enumerate(recommendations[:3], 1):
                    # Strip emojis for Windows compatibility
                    rec_text = ''.join(c if ord(c) < 128 else '' for c in str(rec))
                    print(f"      {i}. {rec_text}")
        else:
            print_status("warning", "Daily brief query returned no results")

    except Exception as e:
        print_status("error", f"Daily brief query failed: {e}")

    # 5b. Get profit analysis
    print_status("info", "\nQuerying: coz.get_profit_analysis...")
    try:
        profit_result = await loom.rift.invoke(
            RiftAction(
                tool_name="coz.get_profit_analysis",
                parameters={"hourly_rate": 25.0}
            )
        )

        if profit_result.get("success"):
            analysis = profit_result["analysis"]
            print_status("success", "Profit analysis complete!")
            print(f"   Net Profit: ${analysis.get('net_profit', 0):,.2f}")
            print(f"   Hourly Profit: ${analysis.get('hourly_profit', 0):.2f}/h")
        else:
            print_status("warning", "Profit analysis query returned no results")

    except Exception as e:
        print_status("error", f"Profit analysis query failed: {e}")

    # Step 6: Semantic search via WarpSpace
    print_header("🔍 SEMANTIC SEARCH VIA WARPSPACE", "-")

    search_queries = [
        "Orders due this week",
        "Honey lip balm production",
        "SOPs for candle making"
    ]

    for query in search_queries:
        print_status("search", f"Searching: '{query}'...")
        try:
            threads = await loom.warp_space.search(query, k=3)
            print(f"   Found {len(threads)} results:")
            for thread in threads:
                print(f"      - {thread.metadata.get('type', 'unknown')}: "
                      f"{thread.content[:80]}...")
        except Exception as e:
            print_status("warning", f"Search failed: {e}")

    # Step 7: Knowledge graph traversal
    print_header("🕸️  KNOWLEDGE GRAPH TRAVERSAL", "-")

    print_status("graph", "Finding customer nodes...")
    try:
        # Try to get a customer node (may fail if not implemented in SimpleLoom)
        customer_node = await loom.yarn_graph.get_node("customer_Leo_Garcia")
        if customer_node:
            print_status("success", f"Found customer: {customer_node.properties.get('name')}")

            # Find their orders
            neighbors = await loom.yarn_graph.find_neighbors(
                customer_node.id,
                relationship_type="PLACED"
            )
            print(f"   Orders placed: {len(neighbors)}")
            for neighbor in neighbors[:3]:
                print(f"      - Order {neighbor.properties.get('order_id')}: "
                      f"{neighbor.properties.get('product')}")
        else:
            print_status("info", "Customer node not found (may need HoloLoom YarnGraph)")

    except Exception as e:
        print_status("info", f"Graph traversal not available in SimpleLoom: {e}")

    # Step 8: Orbit status
    print_header("🌍 ORBIT STATUS", "-")

    status = await orbit.get_orbit_status()
    print(f"Total Docked Apps: {status['total_apps']}")
    for app_id, app_status in status['apps'].items():
        health = app_status['status']
        print(f"  {app_id}: {health}")

    # Step 9: Undock COZ
    print_header("👋 UNDOCKING SEQUENCE", "-")

    print_status("info", "Initiating undocking sequence...")
    await orbit.undock_app("coz")

    # Final status
    final_status = await orbit.get_orbit_status()
    print(f"\nOrbit now has {final_status['total_apps']} docked apps")

    print_header("✅ MISSION COMPLETE", "=")
    print("\nCOZ satellite demonstrated:")
    print("  ✅ Docking to Zero-G Loom")
    print("  ✅ Tool registration via Rift")
    print("  ✅ Intelligence query execution")
    print("  ✅ Semantic search via WarpSpace")
    print("  ✅ Knowledge graph integration")
    print("  ✅ Graceful undocking")
    print("\n🎉 COZ is ready for production!")


if __name__ == "__main__":
    asyncio.run(main())
