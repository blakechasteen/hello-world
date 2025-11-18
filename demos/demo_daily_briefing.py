"""
Demo: Daily Briefing System

Demonstrates the complete morning operational briefing with:
- Real SOPs (Kombucha, Kimchi, Sauerkraut, Hot Sauce)
- Seasonal focus from Coz schedule
- Inventory alerts
- Yesterday's profit summary
- Cash flow status
- Weather-adjusted recommendations

Created: 2025-11-16
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from elle.daily_briefing import DailyBriefingEngine


async def main():
    """Generate and display daily briefing"""

    print("=" * 70)
    print("ELLE DAILY BRIEFING SYSTEM - DEMO")
    print("Farm & Kitchen Cooperative")
    print("=" * 70)
    print()

    # Initialize briefing engine
    print("Initializing briefing engine...")
    engine = DailyBriefingEngine(
        coz_dir="coz",
        data_dir="elle/data",
        sop_dir="elle/sops"
    )

    # Show loaded SOPs
    print(f"\n✓ Loaded {len(engine.sops)} SOPs:")
    for product_name, sop_data in engine.sops.items():
        # Calculate metrics from dictionary
        cost = sop_data.get("cost_per_batch", 0)
        retail_price = sop_data.get("retail_price", 0)
        batch_size = sop_data.get("batch_size", 1)
        total_time = sop_data.get("total_time_minutes", 1)

        revenue = retail_price * batch_size
        profit = revenue - cost
        margin = (profit / revenue * 100) if revenue > 0 else 0
        hourly_roi = (profit / (total_time / 60)) if total_time > 0 else 0

        print(f"  • {product_name}: ${profit:.2f} profit, {margin:.0f}% margin, ${hourly_roi:.2f}/hr")

    print("\n" + "=" * 70)
    print("GENERATING BRIEFING...")
    print("=" * 70)
    print()

    # Generate briefing
    briefing = await engine.generate_briefing()

    # Display text version
    print(briefing.to_text())

    # Display voice version
    print("\n" + "=" * 70)
    print("VOICE VERSION (Text-to-Speech Ready)")
    print("=" * 70)
    print()
    print(briefing.to_voice())
    print()

    # Display detailed task breakdown
    print("\n" + "=" * 70)
    print("DETAILED TASK ANALYSIS")
    print("=" * 70)
    print()

    for priority in ["HIGH", "MEDIUM", "LOW"]:
        priority_tasks = [t for t in briefing.priority_tasks if t.priority == priority]
        if priority_tasks:
            print(f"\n{priority} PRIORITY ({len(priority_tasks)} tasks):")
            for task in priority_tasks:
                print(f"  • {task.task_name}")
                print(f"    Reason: {task.reason}")
                if task.estimated_duration_minutes:
                    hours = task.estimated_duration_minutes / 60
                    print(f"    Duration: {hours:.1f} hours")
                if task.estimated_profit:
                    print(f"    Profit: ${task.estimated_profit:.2f}")
                    if task.estimated_duration_minutes:
                        hourly = task.estimated_profit / (task.estimated_duration_minutes / 60)
                        print(f"    Hourly ROI: ${hourly:.2f}/hour")
                if task.weather_dependent:
                    print(f"    ⚠ Weather dependent")
                print()

    # Summary statistics
    print("\n" + "=" * 70)
    print("SESSION SUMMARY")
    print("=" * 70)
    print()

    total_estimated_profit = sum(
        t.estimated_profit for t in briefing.priority_tasks
        if t.estimated_profit
    )
    total_estimated_time = sum(
        t.estimated_duration_minutes for t in briefing.priority_tasks
        if t.estimated_duration_minutes
    )

    print(f"Total tasks recommended: {len(briefing.priority_tasks)}")
    if total_estimated_time > 0:
        print(f"Estimated total time: {total_estimated_time / 60:.1f} hours")
        print(f"Estimated total profit: ${total_estimated_profit:.2f}")
        print(f"Estimated hourly rate: ${total_estimated_profit / (total_estimated_time / 60):.2f}/hour")

    print(f"\nInventory alerts: {len(briefing.inventory_alerts)}")
    print(f"Weather conditions: {briefing.weather.condition if briefing.weather else 'Not available'}")

    print("\n" + "=" * 70)
    print()
    print("💡 TIP: Run this script every morning for your daily briefing!")
    print("💡 Add to cron: 0 6 * * * cd /path/to/mythRL && PYTHONPATH=. python demos/demo_daily_briefing.py")
    print()


if __name__ == "__main__":
    asyncio.run(main())
