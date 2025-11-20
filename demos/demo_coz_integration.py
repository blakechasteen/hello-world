#!/usr/bin/env python3
"""
Demo: Coz Integration with Elle Core

Demonstrates complete Coz parsing and sync system:
1. Parse all Coz files
2. Display current tasks from kanban
3. Show seasonal focus from schedule
4. List materials needing reorder
5. Daily task suggestions
6. Financial summary
7. Research knowledge base entries
8. Export data to JSON

Created: 2025-11-15
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# Add repo to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from elle.coz import (
    SyncManager,
    KanbanParser,
    FinancialsParser,
    ScheduleParser,
    ResearchParser,
    InventoryParser,
)


def print_header(title: str):
    """Print formatted header"""
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}\n")


def print_subheader(title: str):
    """Print formatted subheader"""
    print(f"\n--- {title} ---\n")


def demo_sync_manager():
    """Demo 1: Initialize and use SyncManager"""
    print_header("1. SYNC MANAGER - Initialize and Parse All Files")

    sync = SyncManager()

    print(f"Coz directory: {sync.coz_dir}")
    print(f"Elle data directory: {sync.elle_data_dir}")

    print_subheader("Parsing all Coz files...")

    result = sync.parse_all()

    print(f"Status: {result.status.value}")
    print(f"Timestamp: {result.timestamp}")
    print(f"Files synced: {', '.join(result.files_synced)}")
    print(f"\nItems synced:")
    for file_type, count in result.items_synced.items():
        print(f"  • {file_type}: {count}")

    if result.errors:
        print(f"\nErrors: {result.errors}")

    return sync


def demo_daily_tasks(sync: SyncManager):
    """Demo 2: Get daily task suggestions"""
    print_header("2. DAILY TASKS - Today's Focus and Suggestions")

    daily = sync.get_daily_tasks()

    print(f"Date: {daily['date']}")
    print(f"Seasonal Focus: {daily['focus_area']}")

    if daily['due_today']:
        print_subheader(f"Due Today ({len(daily['due_today'])} tasks)")
        for i, task in enumerate(daily['due_today'], 1):
            print(f"{i}. [{task['priority']:6s}] {task['task']}")
            if task['notes']:
                print(f"   Notes: {task['notes']}")

    if daily['in_progress']:
        print_subheader(f"In Progress ({len(daily['in_progress'])} tasks)")
        for i, task in enumerate(daily['in_progress'], 1):
            print(f"{i}. {task['task']}")

    if daily['overdue']:
        print_subheader(f"⚠️  OVERDUE ({len(daily['overdue'])} tasks)")
        for i, task in enumerate(daily['overdue'], 1):
            print(f"{i}. {task['task']} (Due: {task['due_date']})")

    if daily['upcoming']:
        print_subheader(f"Upcoming This Week ({len(daily['upcoming'])} tasks)")
        for i, task in enumerate(daily['upcoming'], 1):
            print(f"{i}. {task['task']} (Due: {task['due_date']})")

    print_subheader("Seasonal Tips")
    for tip in daily['seasonal_tips']:
        print(f"• {tip}")

    print_subheader("Recommended Tasks for This Season")
    for task in daily['recommended_tasks'][:5]:  # Show first 5
        print(f"• {task}")


def demo_schedule(sync: SyncManager):
    """Demo 3: Seasonal focus and recommendations"""
    print_header("3. SEASONAL FOCUS - Annual Calendar & Recommendations")

    season = sync.get_seasonal_context()

    print(f"Current Month: {season['current_month']}")
    print(f"Current Season: {season['current_season']}")
    print(f"Primary Focus: {season['recommendations']['primary_focus']}")

    print_subheader("Upcoming Monthly Focuses")
    for focus in season['upcoming_focuses'][:3]:
        print(f"• {focus['month']}: {focus['focus']}")

    print_subheader("Suggested Tasks for Current Season")
    for i, task in enumerate(season['recommendations']['suggested_tasks'][:5], 1):
        print(f"{i}. {task}")

    print_subheader("Seasonal Tips")
    for tip in season['recommendations']['seasonal_tips']:
        print(f"• {tip}")


def demo_financials(sync: SyncManager):
    """Demo 4: Financial summary"""
    print_header("4. FINANCIALS - Products, Pricing, and Reinvestment")

    financials = sync.get_financial_summary()

    print_subheader("Products and Pricing")
    print(f"{'Product':<20} {'Price':<10} {'Cost':<10} {'Profit':<10} {'Margin':<10}")
    print("-" * 60)

    for product in financials['products']:
        print(f"{product['name']:<20} "
              f"${product['unit_price']:>8.2f} "
              f"${product['cost_per_unit']:>8.2f} "
              f"${product['profit_per_unit']:>8.2f} "
              f"{product['gross_margin_percent']:>8.1f}%")

    if financials['reinvestment']:
        print_subheader("Reinvestment Plan (% of Monthly Profit)")
        reinv = financials['reinvestment']
        print(f"Infrastructure:    {reinv['infrastructure_percent']*100:.0f}%")
        print(f"Equipment:         {reinv['equipment_percent']*100:.0f}%")
        print(f"Emergency Buffer:  {reinv['emergency_buffer_percent']*100:.0f}%")
        print(f"Total Reinvested:  {reinv['total_reinvestment_percent']*100:.0f}%")


def demo_inventory(sync: SyncManager):
    """Demo 5: Inventory status and reorder alerts"""
    print_header("5. INVENTORY - Stock Levels and Reorder Alerts")

    inventory = sync.get_inventory_status()

    summary = inventory['summary']
    print(f"Total Items: {summary['total_items']}")
    print(f"Items by category:")
    for category, count in summary['by_category'].items():
        print(f"  • {category}: {count} items")

    print_subheader("Stock Level Summary")
    for level, count in summary['by_stock_level'].items():
        print(f"  • {level.title()}: {count} items")

    alerts = inventory['alerts']

    if alerts['critical']['count'] > 0:
        print_subheader("🚨 CRITICAL - Items Needing Immediate Reorder")
        for item in alerts['critical']['items']:
            print(f"  • {item['name']}: {item['quantity']} {item['unit']} "
                  f"(Reorder: {item['reorder_point']})")

    if alerts['low']['count'] > 0:
        print_subheader("⚠️  LOW STOCK - Items to Reorder Soon")
        for item in alerts['low']['items'][:5]:  # Show first 5
            print(f"  • {item['name']}: {item['quantity']} {item['unit']}")

    reorder = inventory['reorder_list']
    if reorder['by_supplier']:
        print_subheader("Reorder List by Supplier")
        for supplier, items in reorder['by_supplier'].items():
            print(f"\n{supplier}:")
            for item in items:
                print(f"  • {item['name']}")


def demo_research(sync: SyncManager):
    """Demo 6: Research notes and experiments"""
    print_header("6. RESEARCH NOTES - Experiments, Formulas & Insights")

    research = sync.research
    summary = research.get_summary()

    print(f"Research Projects: {summary['research_projects']}")
    print(f"Total Experiments: {summary['total_experiments']}")
    print(f"Total Formulas: {summary['total_formulas']}")
    print(f"Categories: {', '.join(summary['categories'])}")

    print_subheader("Research Projects")
    for i, project in enumerate(summary['projects'][:5], 1):
        print(f"\n{i}. {project['title']} ({project['category']})")
        print(f"   Experiments: {project['experiment_count']}")
        for exp in project['experiments'][:2]:  # First 2 experiments
            print(f"   • {exp['batch_number']}: {exp['description']}")
            print(f"     Result: {exp['result']}")

    if summary['total_formulas'] > 0:
        print_subheader("Key Formulas & Ratios")
        for formula in summary['projects'][0]['formulas'][:3]:
            if formula['type'] == 'ratio':
                print(f"  • Ratio: {formula['ratio']}")
            elif formula['type'] == 'yield':
                print(f"  • Yield: {formula['amount']} {formula['unit']}")

    # Show knowledge base entries
    kb_entries = sync.get_research_knowledge_base()
    print_subheader(f"Knowledge Base Entries ({len(kb_entries)} projects)")
    for i, entry in enumerate(kb_entries[:2], 1):
        print(f"\n{i}. {entry['title']}")
        print(f"   Category: {entry['category']}")
        print(f"   Experiments: {len(entry['experiments'])}")


def demo_kanban_details(sync: SyncManager):
    """Demo 7: Detailed kanban analysis"""
    print_header("7. KANBAN ANALYSIS - Tasks by Category & Priority")

    kanban = sync.kanban
    summary = kanban.get_summary()

    print(f"Total Tasks: {summary['total_tasks']}")
    print(f"Completed: {summary['completed']} ({summary['completion_rate']:.1%})")
    print(f"In Progress: {summary['in_progress']}")
    print(f"High Priority: {summary['high_priority']}")

    print_subheader("Tasks by Category")
    for category, count in summary['by_category'].items():
        print(f"  • {category}: {count} tasks")

    print_subheader("High Priority Tasks")
    high_priority = kanban.get_by_priority(kanban.tasks[0].priority if kanban.tasks else None)
    from elle.coz.kanban_parser import TaskPriority
    high_priority = kanban.get_by_priority(TaskPriority.HIGH)
    for task in high_priority[:5]:
        print(f"  • {task.task} (Due: {task.due_date.date() if task.due_date else 'No due date'})")


def demo_daily_brief(sync: SyncManager):
    """Demo 8: Comprehensive daily brief"""
    print_header("8. DAILY BRIEF - Comprehensive Operational Summary")

    brief = sync.get_daily_brief()

    print(f"Generated: {brief['timestamp']}")

    print_subheader("Daily Tasks Summary")
    daily = brief['daily_tasks']
    print(f"  • Due today: {len(daily['due_today'])} tasks")
    print(f"  • In progress: {len(daily['in_progress'])} tasks")
    print(f"  • Overdue: {len(daily['overdue'])} tasks")
    print(f"  • Upcoming (3 days): {len(daily['upcoming'])} tasks")

    print_subheader("Seasonal Focus")
    season = brief['seasonal_focus']
    print(f"  • Current focus: {season['recommendations']['primary_focus']}")
    print(f"  • Season: {season['current_season']}")

    print_subheader("Inventory Alerts")
    inv = brief['inventory_alerts']
    print(f"  • Critical items: {inv['critical']['count']}")
    print(f"  • Low stock items: {inv['low']['count']}")

    print_subheader("Products Available for Sale")
    for product in brief['financial_status']['products'][:3]:
        print(f"  • {product['name']}: ${product['unit_price']:.2f}")


def demo_export_data(sync: SyncManager):
    """Demo 9: Export data to JSON"""
    print_header("9. DATA EXPORT - Save All Data to JSON")

    export_path = sync.export_sync_data()
    print(f"Exported to: {export_path}")

    # Show file size
    file_size = Path(export_path).stat().st_size
    print(f"File size: {file_size:,} bytes")

    # Load and show structure
    with open(export_path, 'r') as f:
        data = json.load(f)

    print_subheader("Exported Data Structure")
    for key in data.keys():
        if key != 'timestamp':
            if isinstance(data[key], dict):
                sub_keys = list(data[key].keys())[:3]
                print(f"  • {key}: {sub_keys}...")
            else:
                print(f"  • {key}: {type(data[key]).__name__}")


def demo_sync_status(sync: SyncManager):
    """Demo 10: Sync statistics and integration status"""
    print_header("10. SYNC STATUS - Integration Health & Statistics")

    status = sync.get_integration_status()

    print_subheader("Managers Initialized")
    for manager, initialized in status['managers_initialized'].items():
        status_str = "✓ Ready" if initialized else "✗ Not ready"
        print(f"  • {manager}: {status_str}")

    print_subheader("Sync Statistics")
    stats = status['statistics']
    print(f"Total syncs: {stats['total_syncs']}")
    print(f"Successful: {stats['successful']}")
    print(f"Failed: {stats['failed']}")
    print(f"Success rate: {stats['success_rate']:.1%}")
    if stats['last_sync']:
        print(f"Last sync: {stats['last_sync']}")

    print_subheader("Current Sync Status")
    if status['sync_status']:
        print(f"Status: {status['sync_status']['status']}")
        print(f"Files: {status['sync_status']['files_synced']}")
        print(f"Items synced: {status['sync_status']['items_synced']}")


def main():
    """Run all demos"""
    print("\n" + "=" * 80)
    print(" " * 20 + "COZ INTEGRATION WITH ELLE CORE")
    print(" " * 25 + "Complete Demo System")
    print("=" * 80)

    try:
        # Demo 1: Initialize sync manager
        sync = demo_sync_manager()

        # Demo 2: Daily tasks
        demo_daily_tasks(sync)

        # Demo 3: Schedule and seasonal focus
        demo_schedule(sync)

        # Demo 4: Financials
        demo_financials(sync)

        # Demo 5: Inventory
        demo_inventory(sync)

        # Demo 6: Research notes
        demo_research(sync)

        # Demo 7: Kanban details
        demo_kanban_details(sync)

        # Demo 8: Daily brief
        demo_daily_brief(sync)

        # Demo 9: Export data
        demo_export_data(sync)

        # Demo 10: Sync status
        demo_sync_status(sync)

        # Summary
        print_header("SUMMARY")
        print("✓ All Coz files successfully parsed")
        print("✓ Daily tasks generated from kanban")
        print("✓ Seasonal recommendations from schedule")
        print("✓ Financial summary with reinvestment plan")
        print("✓ Inventory alerts and reorder list")
        print("✓ Research knowledge base entries")
        print("✓ Data exported to JSON for Elle Core integration")
        print("\nNext steps:")
        print("  1. Use SyncManager.get_daily_tasks() for Decision Engine")
        print("  2. Ingest research KB entries into HoloLoom")
        print("  3. Set up inventory alerts in Voice Interface")
        print("  4. Auto-populate SOP ingredient costs from financials")
        print("\n")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
