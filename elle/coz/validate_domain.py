"""
COZ Domain Integration Test

Validates all parsers and intelligence engine integration.
Tests the complete data flow from parsing to daily brief generation.

Created: 2025-11-22
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from elle.coz.sync_manager import SyncManager
import re


def strip_emojis(text):
    """Remove emoji characters for Windows console compatibility"""
    # Simple approach: keep only ASCII printable characters
    return ''.join(c if ord(c) < 128 else '' for c in text)


def main():
    """Run comprehensive integration test"""

    print("=" * 70)
    print("COZ Domain Integration Test")
    print("=" * 70)

    # Initialize SyncManager (with correct path)
    print("\n1. Initializing SyncManager...")
    sync = SyncManager(coz_dir="coz")

    # Parse all files
    print("\n2. Parsing all COZ files...")
    result = sync.parse_all()
    print(f"   Status: {result.status.value}")
    print(f"   Files synced: {len(result.files_synced)}")
    print(f"   Parsed: {', '.join(result.files_synced)}")
    if result.errors:
        print(f"   Errors: {result.errors}")

    if len(result.files_synced) == 0:
        print("   [ERROR] No files parsed! Check file paths.")
        return

    print(f"\n   [SUCCESS] Successfully parsed {len(result.files_synced)} files")

    # Test individual parsers (access parsers directly)
    print("\n3. Testing Individual Parsers...")

    # Time tracking
    print("\n   Time Tracking:")
    time_summary = sync.time_tracking.get_time_summary()
    print(f"      Total hours: {time_summary.get('total_actual_hours', 0):.1f}h")
    print(f"      Efficiency: {time_summary.get('avg_efficiency_score', 0):.1%}")

    # Cost tracking
    print("\n   Cost Tracking:")
    cost_summary = sync.cost_tracking.get_cost_summary()
    print(f"      Total cost: ${cost_summary.get('total_cost', 0):,.2f}")
    print(f"      Avg per task: ${cost_summary.get('avg_cost_per_task', 0):.2f}")

    # Customer orders
    print("\n   Customer Orders:")
    revenue_pipeline = sync.customer_orders.get_revenue_pipeline()
    print(f"      Total revenue: ${revenue_pipeline.get('total', 0):,.2f}")
    print(f"      Pending: ${revenue_pipeline.get('total_pending', 0):,.2f}")

    # Production log
    print("\n   Production Log:")
    production_summary = sync.production_log.get_production_summary()
    print(f"      Total produced: {production_summary.get('total_produced', 0)} units")
    print(f"      Waste rate: {production_summary.get('avg_waste_percent', 0):.1f}%")

    # SOPs
    print("\n   SOPs:")
    sop_library = sync.sops.get_sop_library()
    print(f"      Total SOPs: {sop_library.get('total_sops', 0)}")
    print(f"      Avg time: {sop_library.get('avg_time', 0):.1f}h")

    # Test intelligence engine
    print("\n4. Testing Intelligence Engine...")

    # Profit analysis
    print("\n   Profit Analysis:")
    profit = sync.get_profit_analysis(hourly_rate=25.0)
    if 'error' in profit:
        print(f"      [WARNING] {profit['error']}")
    else:
        print(f"      Net profit: ${profit.get('net_profit', 0):,.2f}")
        print(f"      Profit margin: {profit.get('profit_margin', 0):.1f}%")
        print(f"      Hourly profit: ${profit.get('hourly_profit', 0):.2f}/h")

    # Efficiency insights
    print("\n   Efficiency Insights:")
    efficiency = sync.get_efficiency_insights()
    if 'error' in efficiency:
        print(f"      [WARNING] {efficiency['error']}")
    else:
        insights = efficiency.get('insights', [])
        for insight in insights[:3]:
            print(f"      - {strip_emojis(str(insight))}")

    # Cost insights
    print("\n   Cost Insights:")
    cost_insights = sync.get_cost_insights()
    if 'error' in cost_insights:
        print(f"      [WARNING] {cost_insights['error']}")
    else:
        insights = cost_insights.get('insights', [])
        for insight in insights[:3]:
            print(f"      - {strip_emojis(str(insight))}")

    # Production efficiency
    print("\n   Production Efficiency:")
    production_eff = sync.get_production_efficiency_analysis()
    if 'error' in production_eff:
        print(f"      [WARNING] {production_eff['error']}")
    else:
        insights = production_eff.get('insights', [])
        for insight in insights[:3]:
            print(f"      - {strip_emojis(str(insight))}")

    # Waste reduction
    print("\n   Waste Reduction:")
    waste = sync.get_waste_reduction_recommendations()
    if 'error' in waste:
        print(f"      [WARNING] {waste['error']}")
    else:
        insights = waste.get('insights', [])
        for insight in insights[:3]:
            print(f"      - {strip_emojis(str(insight))}")

    # Order fulfillment
    print("\n   Order Fulfillment:")
    orders = sync.get_order_fulfillment_optimization()
    if 'error' in orders:
        print(f"      [WARNING] {orders['error']}")
    else:
        insights = orders.get('insights', [])
        for insight in insights[:3]:
            print(f"      - {strip_emojis(str(insight))}")

    # Customer insights
    print("\n   Customer Insights:")
    customers = sync.get_customer_insights()
    if 'error' in customers:
        print(f"      [WARNING] {customers['error']}")
    else:
        insights = customers.get('insights', [])
        for insight in insights[:3]:
            print(f"      - {strip_emojis(str(insight))}")

    # Generate daily brief (without refinement for speed)
    print("\n5. Generating Daily Brief...")
    print("   (Using raw summary - refinement disabled for test speed)")

    try:
        brief = sync.get_daily_brief()

        print("\n   [CHART] Daily Brief Summary:")
        print("   " + "=" * 66)

        # Profit Analysis
        profit_brief = brief.get('profit_analysis', {})
        if 'error' not in profit_brief:
            print(f"\n   Profit Analysis:")
            print(f"      Net profit: ${profit_brief.get('net_profit', 0):,.2f}")
            print(f"      Profit margin: {profit_brief.get('profit_margin', 0):.1f}%")

        # Order Fulfillment
        orders = brief.get('order_fulfillment', {})
        print(f"\n   Order Fulfillment:")
        print(f"      Critical orders: {orders.get('critical_orders', 0)}")
        print(f"      Pending orders: {orders.get('total_pending', 0)}")
        print(f"      Production hours needed: {orders.get('production_hours_needed', 0):.1f}h")

        # Waste Alerts
        waste_alerts = brief.get('waste_alerts', {})
        print(f"\n   Waste Alerts:")
        print(f"      Total waste: {waste_alerts.get('total_waste', 0)} units")

        # Top Recommendations (efficiency insights)
        recommendations = brief.get('top_recommendations', [])
        if recommendations:
            print(f"\n   Top Recommendations:")
            for i, rec in enumerate(recommendations[:3], 1):
                rec_text = strip_emojis(str(rec))
                print(f"      {i}. {rec_text}")

        print(f"\n   [SUCCESS] Daily brief generated successfully")
        print(f"      Refinement used: {brief.get('refinement_used', False)}")
        print(f"      Date: {brief.get('date', 'N/A')}")

    except Exception as e:
        print(f"   [ERROR] Daily brief generation failed: {e}")
        import traceback
        traceback.print_exc()

    # Final validation
    print("\n" + "=" * 70)
    print("Integration Test Summary")
    print("=" * 70)

    # Count successes
    total_tests = 13  # 5 parsers + 7 intelligence modules + 1 daily brief
    successful_tests = result.files_synced

    # Intelligence module checks
    intelligence_modules = [
        ('Profit Analysis', 'error' not in profit),
        ('Efficiency Insights', 'error' not in efficiency),
        ('Cost Insights', 'error' not in cost_insights),
        ('Production Efficiency', 'error' not in production_eff),
        ('Waste Reduction', 'error' not in waste),
        ('Order Fulfillment', 'error' not in orders),
        ('Customer Insights', 'error' not in customers),
        ('Daily Brief', 'error' not in brief if 'brief' in locals() else False),
    ]

    print("\n[SUCCESS] Parser Status:")
    print(f"   {len(result.files_synced)}/{len(result.files_synced) + len(result.errors)} parsers successful")

    print("\n[SUCCESS] Intelligence Modules:")
    intelligence_success = sum(1 for _, success in intelligence_modules if success)
    for module_name, success in intelligence_modules:
        status = "[SUCCESS]" if success else "[WARNING]"
        print(f"   {status} {module_name}")

    print(f"\n   {intelligence_success}/{len(intelligence_modules)} modules successful")

    overall_success_rate = (len(result.files_synced) + intelligence_success) / (len(result.files_synced) + len(intelligence_modules))

    print("\n" + "=" * 70)
    print(f"Overall Success Rate: {overall_success_rate:.1%}")
    print("=" * 70)

    if overall_success_rate >= 0.9:
        print("\n[CELEBRATION] EXCELLENT! COZ domain is production-ready!")
    elif overall_success_rate >= 0.7:
        print("\n[SUCCESS] GOOD! Most features working, some minor issues.")
    else:
        print("\n[WARNING] NEEDS WORK. Several components failing.")

    print()


if __name__ == "__main__":
    main()
